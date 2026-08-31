import httpx
import pytest
from openai import APIConnectionError
from pydantic import BaseModel

from arcus.adapters.arc_adapter import ArcAdapter
from arcus.routing.bandit import ContextualBandit, EpsilonGreedyBandit
from arcus.quality.gate import (
    check_empty,
    check_refusal,
    check_repetition,
    check_response,
    check_schema,
    check_truncation,
    call_with_quality_gate,
)


def _connection_error(message: str) -> APIConnectionError:
    request = httpx.Request("POST", "https://llm-api.arc.vt.edu/api/v1/chat/completions")
    return APIConnectionError(message=message, request=request)

LOOPING_TEXT = "the cat sat on the mat and " * 20
NORMAL_TEXT = (
    "Binary search works by repeatedly dividing a sorted array in half. "
    "You compare the target value to the middle element, then discard the "
    "half that can't contain it, until you either find the value or run "
    "out of elements to check."
)


def test_check_truncation():
    assert check_truncation("length") is not None
    assert check_truncation("stop") is None
    assert check_truncation(None) is None


@pytest.mark.parametrize("content", [None, "", "   "])
def test_check_empty_flags_empty_or_whitespace(content):
    assert check_empty(content) is not None


def test_check_empty_allows_short_but_real_answers():
    assert check_empty("42") is None
    assert check_empty("Yes.") is None


def test_check_repetition_flags_looping_text():
    assert check_repetition(LOOPING_TEXT) is not None


def test_check_repetition_allows_normal_text():
    assert check_repetition(NORMAL_TEXT) is None


def test_check_repetition_skips_short_content():
    assert check_repetition("the cat sat") is None


@pytest.mark.parametrize(
    "content",
    [
        "I cannot assist with that request.",
        "As an AI language model, I don't have opinions.",
        "I'm sorry, but I can't help with that.",
    ],
)
def test_check_refusal_flags_known_phrases(content):
    assert check_refusal(content) is not None


def test_check_refusal_allows_ontopic_answer():
    assert check_refusal(NORMAL_TEXT) is None


class _Point(BaseModel):
    x: int
    y: int


def test_check_schema_none_always_passes():
    assert check_schema("not even json", None) is None


def test_check_schema_valid_json_passes():
    assert check_schema('{"x": 1, "y": 2}', _Point) is None


def test_check_schema_invalid_json_fails():
    assert check_schema("not json at all", _Point) is not None


def test_check_schema_wrong_shape_fails():
    assert check_schema('{"x": "not an int"}', _Point) is not None


def test_check_response_aggregates_all_issues():
    result = check_response(content=None, finish_reason="length")
    assert not result.passed
    kinds = {issue.kind for issue in result.issues}
    assert "truncated" in kinds
    assert "empty" in kinds


def test_check_response_passes_clean_response():
    result = check_response(content=NORMAL_TEXT, finish_reason="stop")
    assert result.passed
    assert result.issues == []


def _make_adapter():
    return ArcAdapter(api_key="test-key")


def _mock_completion(content, finish_reason):
    class Message:
        pass

    class Choice:
        pass

    class Completion:
        pass

    message = Message()
    message.content = content
    choice = Choice()
    choice.message = message
    choice.finish_reason = finish_reason
    completion = Completion()
    completion.choices = [choice]
    return completion


ARMS = ["gpt-oss-120b", "GLM-5.3", "Kimi-K3"]


def test_first_arm_passes_returns_immediately():
    adapter = _make_adapter()
    adapter._client.chat.completions.create = lambda **kwargs: _mock_completion(NORMAL_TEXT, "stop")

    bandit = ContextualBandit(lambda: EpsilonGreedyBandit(ARMS[:2], epsilon=0.0), arms=ARMS[:2])

    outcome = call_with_quality_gate(adapter, bandit, "code:short", [{"role": "user", "content": "hi"}])

    assert outcome.passed
    assert len(outcome.attempts) == 1
    assert outcome.attempts[0].passed
    assert outcome.attempts[0].model == outcome.model_used


def test_first_arm_fails_second_arm_passes():
    adapter = _make_adapter()

    def fake_create(model, **kwargs):
        if model == "gpt-oss-120b":
            return _mock_completion(NORMAL_TEXT, "length")  # truncated
        return _mock_completion(NORMAL_TEXT, "stop")

    adapter._client.chat.completions.create = fake_create

    bandit = ContextualBandit(lambda: EpsilonGreedyBandit(ARMS[:2], epsilon=0.0), arms=ARMS[:2])
    # force "gpt-oss-120b" to be tried first, cold start would pick it
    # anyway since both start untried, but this keeps the test explicit
    bandit._get_bandit("code:short")._pulls["GLM-5.3"] = 0

    outcome = call_with_quality_gate(adapter, bandit, "code:short", [{"role": "user", "content": "hi"}])

    assert outcome.passed
    assert len(outcome.attempts) == 2
    assert outcome.attempts[0].model != outcome.attempts[1].model
    assert not outcome.attempts[0].passed
    assert outcome.attempts[1].passed
    assert outcome.model_used == outcome.attempts[1].model


def test_every_arm_fails_returns_last_attempt_without_looping_forever():
    adapter = _make_adapter()
    adapter._client.chat.completions.create = lambda **kwargs: _mock_completion(None, "length")

    bandit = ContextualBandit(lambda: EpsilonGreedyBandit(ARMS, epsilon=0.0), arms=ARMS)

    outcome = call_with_quality_gate(adapter, bandit, "code:short", [{"role": "user", "content": "hi"}])

    assert not outcome.passed
    assert len(outcome.attempts) == 3
    assert len({attempt.model for attempt in outcome.attempts}) == 3
    assert all(not attempt.passed for attempt in outcome.attempts)
    assert outcome.issues


def test_api_error_on_one_arm_falls_back_to_next_arm():
    adapter = _make_adapter()

    def fake_create(model, **kwargs):
        if model == "gpt-oss-120b":
            raise _connection_error("connection reset")
        return _mock_completion(NORMAL_TEXT, "stop")

    adapter._client.chat.completions.create = fake_create

    bandit = ContextualBandit(lambda: EpsilonGreedyBandit(ARMS[:2], epsilon=0.0), arms=ARMS[:2])
    bandit._get_bandit("code:short")._pulls["GLM-5.3"] = 0

    outcome = call_with_quality_gate(adapter, bandit, "code:short", [{"role": "user", "content": "hi"}])

    assert outcome.passed
    assert outcome.model_used == "GLM-5.3"
    assert len(outcome.attempts) == 2
    failed_attempt = outcome.attempts[0]
    assert not failed_attempt.passed
    assert failed_attempt.model == "gpt-oss-120b"
    assert failed_attempt.issues[0].kind == "api_error"


def test_every_arm_api_errors_returns_no_response_instead_of_crashing():
    adapter = _make_adapter()
    adapter._client.chat.completions.create = lambda **kwargs: (_ for _ in ()).throw(
        _connection_error("ARC is down")
    )

    bandit = ContextualBandit(lambda: EpsilonGreedyBandit(ARMS, epsilon=0.0), arms=ARMS)

    outcome = call_with_quality_gate(adapter, bandit, "code:short", [{"role": "user", "content": "hi"}])

    assert not outcome.passed
    assert outcome.response is None
    assert len(outcome.attempts) == 3
    assert all(attempt.issues[0].kind == "api_error" for attempt in outcome.attempts)
    assert outcome.issues[0].kind == "api_error"
