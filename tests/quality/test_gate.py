import httpx
import pytest
from openai import APIConnectionError, PermissionDeniedError, RateLimitError
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
    _call_with_backoff,
)


def _connection_error(message: str) -> APIConnectionError:
    request = httpx.Request("POST", "https://llm-api.arc.vt.edu/api/v1/chat/completions")
    return APIConnectionError(message=message, request=request)


def _rate_limit_error(message: str = "rate limited") -> RateLimitError:
    request = httpx.Request("POST", "https://llm-api.arc.vt.edu/api/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError(message=message, response=response, body=None)


def _permission_denied_error(message: str = "VPN required") -> PermissionDeniedError:
    request = httpx.Request("POST", "https://llm-api.arc.vt.edu/api/v1/chat/completions")
    response = httpx.Response(403, request=request)
    return PermissionDeniedError(message=message, response=response, body=None)

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

    bandit = ContextualBandit(lambda _context_key: EpsilonGreedyBandit(ARMS[:2], epsilon=0.0), arms=ARMS[:2])

    outcome = call_with_quality_gate(adapter, bandit, "code:short", [{"role": "user", "content": "hi"}])

    assert outcome.passed
    assert len(outcome.attempts) == 1
    assert outcome.attempts[0].passed
    assert outcome.attempts[0].model == outcome.model_used


def test_extra_kwargs_reach_the_underlying_api_call():
    adapter = _make_adapter()
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _mock_completion(NORMAL_TEXT, "stop")

    adapter._client.chat.completions.create = fake_create

    bandit = ContextualBandit(lambda _context_key: EpsilonGreedyBandit(ARMS[:1], epsilon=0.0), arms=ARMS[:1])

    call_with_quality_gate(
        adapter, bandit, "code:short", [{"role": "user", "content": "hi"}],
        extra_body={"files": [{"type": "file", "id": "abc123"}]},
    )

    assert captured["extra_body"] == {"files": [{"type": "file", "id": "abc123"}]}


def test_first_arm_fails_second_arm_passes():
    adapter = _make_adapter()

    def fake_create(model, **kwargs):
        if model == "gpt-oss-120b":
            return _mock_completion(NORMAL_TEXT, "length")  # truncated
        return _mock_completion(NORMAL_TEXT, "stop")

    adapter._client.chat.completions.create = fake_create

    bandit = ContextualBandit(lambda _context_key: EpsilonGreedyBandit(ARMS[:2], epsilon=0.0), arms=ARMS[:2])
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

    bandit = ContextualBandit(lambda _context_key: EpsilonGreedyBandit(ARMS, epsilon=0.0), arms=ARMS)

    outcome = call_with_quality_gate(adapter, bandit, "code:short", [{"role": "user", "content": "hi"}])

    assert not outcome.passed
    assert len(outcome.attempts) == 3
    assert len({attempt.model for attempt in outcome.attempts}) == 3
    assert all(not attempt.passed for attempt in outcome.attempts)
    assert outcome.issues


def test_max_attempts_respects_the_current_contexts_arm_count_not_a_global_one():
    # a bandit whose arm set differs by context (reasoning-effort
    # variants only on some task types, say), the retry budget for a
    # given call has to come from that specific context's own arm list,
    # not from whatever the bandit happened to be constructed with
    adapter = _make_adapter()
    adapter._client.chat.completions.create = lambda **kwargs: _mock_completion(None, "length")

    def factory(context_key):
        arms = ARMS if context_key == "code:short" else ARMS[:1]
        return EpsilonGreedyBandit(arms, epsilon=0.0)

    bandit = ContextualBandit(factory, arms=ARMS[:1])

    code_outcome = call_with_quality_gate(adapter, bandit, "code:short", [{"role": "user", "content": "hi"}])
    writing_outcome = call_with_quality_gate(adapter, bandit, "writing:short", [{"role": "user", "content": "hi"}])

    assert len(code_outcome.attempts) == len(ARMS)
    assert len(writing_outcome.attempts) == 1


def test_api_error_on_one_arm_falls_back_to_next_arm():
    adapter = _make_adapter()

    def fake_create(model, **kwargs):
        if model == "gpt-oss-120b":
            raise _connection_error("connection reset")
        return _mock_completion(NORMAL_TEXT, "stop")

    adapter._client.chat.completions.create = fake_create

    bandit = ContextualBandit(lambda _context_key: EpsilonGreedyBandit(ARMS[:2], epsilon=0.0), arms=ARMS[:2])
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

    bandit = ContextualBandit(lambda _context_key: EpsilonGreedyBandit(ARMS, epsilon=0.0), arms=ARMS)

    outcome = call_with_quality_gate(adapter, bandit, "code:short", [{"role": "user", "content": "hi"}])

    assert not outcome.passed
    assert outcome.response is None
    assert len(outcome.attempts) == 3
    assert all(attempt.issues[0].kind == "api_error" for attempt in outcome.attempts)
    assert outcome.issues[0].kind == "api_error"


def test_call_with_backoff_returns_immediately_on_success(monkeypatch):
    adapter = _make_adapter()
    adapter._client.chat.completions.create = lambda **kwargs: _mock_completion(NORMAL_TEXT, "stop")

    slept = []
    monkeypatch.setattr("arcus.quality.gate.time.sleep", lambda seconds: slept.append(seconds))

    result = _call_with_backoff(adapter, "gpt-oss-120b", [{"role": "user", "content": "hi"}])

    assert result.choices[0].message.content == NORMAL_TEXT
    assert slept == []


def test_call_with_backoff_forwards_extra_kwargs(monkeypatch):
    adapter = _make_adapter()
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _mock_completion(NORMAL_TEXT, "stop")

    adapter._client.chat.completions.create = fake_create

    _call_with_backoff(
        adapter, "gpt-oss-120b", [{"role": "user", "content": "hi"}],
        extra_body={"tool_ids": ["server:websearch"]},
    )

    assert captured["extra_body"] == {"tool_ids": ["server:websearch"]}


def test_call_with_backoff_retries_the_same_arm_and_recovers(monkeypatch):
    adapter = _make_adapter()
    calls = []

    def fake_create(model, **kwargs):
        calls.append(model)
        if len(calls) == 1:
            raise _rate_limit_error()
        return _mock_completion(NORMAL_TEXT, "stop")

    adapter._client.chat.completions.create = fake_create

    slept = []
    monkeypatch.setattr("arcus.quality.gate.time.sleep", lambda seconds: slept.append(seconds))

    result = _call_with_backoff(adapter, "gpt-oss-120b", [{"role": "user", "content": "hi"}])

    assert result.choices[0].message.content == NORMAL_TEXT
    assert calls == ["gpt-oss-120b", "gpt-oss-120b"]
    assert slept == [2.0]


def test_call_with_backoff_raises_after_exhausting_retries(monkeypatch):
    adapter = _make_adapter()
    adapter._client.chat.completions.create = lambda **kwargs: (_ for _ in ()).throw(_rate_limit_error())

    slept = []
    monkeypatch.setattr("arcus.quality.gate.time.sleep", lambda seconds: slept.append(seconds))

    with pytest.raises(RateLimitError):
        _call_with_backoff(adapter, "gpt-oss-120b", [{"role": "user", "content": "hi"}])

    # 3 retries after the first attempt, backoff doubling each time
    assert slept == [2.0, 4.0, 8.0]


def test_a_rate_limit_that_outlasts_every_retry_falls_through_like_any_other_api_error(monkeypatch):
    adapter = _make_adapter()
    adapter._client.chat.completions.create = lambda **kwargs: (_ for _ in ()).throw(_rate_limit_error())
    monkeypatch.setattr("arcus.quality.gate.time.sleep", lambda seconds: None)

    bandit = ContextualBandit(lambda _context_key: EpsilonGreedyBandit(ARMS[:1], epsilon=0.0), arms=ARMS[:1])

    outcome = call_with_quality_gate(
        adapter, bandit, "code:short", [{"role": "user", "content": "hi"}], max_attempts=1
    )

    assert not outcome.passed
    assert outcome.response is None
    assert outcome.attempts[0].issues[0].kind == "api_error"


def test_permission_denied_stops_immediately_without_trying_other_arms():
    adapter = _make_adapter()
    calls = []

    def fake_create(model, **kwargs):
        calls.append(model)
        raise _permission_denied_error()

    adapter._client.chat.completions.create = fake_create

    bandit = ContextualBandit(lambda _context_key: EpsilonGreedyBandit(ARMS, epsilon=0.0), arms=ARMS)

    outcome = call_with_quality_gate(adapter, bandit, "code:short", [{"role": "user", "content": "hi"}])

    assert not outcome.passed
    assert outcome.response is None
    assert len(calls) == 1
    assert len(outcome.attempts) == 1
    assert outcome.attempts[0].issues[0].kind == "permission_denied"
    assert outcome.issues[0].kind == "permission_denied"


def test_permission_denied_does_not_penalize_the_bandit():
    adapter = _make_adapter()
    adapter._client.chat.completions.create = lambda **kwargs: (_ for _ in ()).throw(_permission_denied_error())

    bandit = ContextualBandit(lambda _context_key: EpsilonGreedyBandit(ARMS, epsilon=0.0), arms=ARMS)
    underlying = bandit._get_bandit("code:short")

    outcome = call_with_quality_gate(adapter, bandit, "code:short", [{"role": "user", "content": "hi"}])

    assert outcome.attempts[0].reward is None
    assert all(pulls == 0 for pulls in underlying._pulls.values())
