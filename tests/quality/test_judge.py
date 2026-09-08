from arcus.adapters.arc_adapter import ArcModel
from arcus.quality.judge import (
    judge_model_for,
    judge_response,
    parse_score,
    should_judge,
)


class _FakeAdapter:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []

    def chat(self, model, messages, **kwargs):
        self.calls.append((model, messages, kwargs))
        if self.error:
            raise self.error

        class Message:
            pass

        class Choice:
            pass

        class Completion:
            pass

        message = Message()
        message.content = self.content
        choice = Choice()
        choice.message = message
        completion = Completion()
        completion.choices = [choice]
        return completion


def test_parse_score_reads_a_bare_number():
    assert parse_score("8") == 0.8


def test_parse_score_finds_a_number_buried_in_prose():
    # these are reasoning models, some residue around the number is
    # expected rather than exceptional
    assert parse_score("I'd say this deserves a 7 out of 10.") == 0.7


def test_parse_score_handles_a_decimal():
    assert parse_score("7.5") == 0.75


def test_parse_score_clamps_above_the_scale():
    # a judge ignoring the rubric shouldn't be able to drag an arm's
    # average past what the rubric allows
    assert parse_score("45") == 1.0


def test_parse_score_returns_none_with_no_number_to_find():
    assert parse_score("I'm not able to grade this") is None


def test_parse_score_returns_none_on_empty_content():
    assert parse_score(None) is None
    assert parse_score("") is None


def test_judge_model_swaps_so_nothing_grades_itself():
    assert judge_model_for(ArcModel.GPT_OSS_120B.value) != ArcModel.GPT_OSS_120B.value
    assert judge_model_for(ArcModel.KIMI_K3.value) == ArcModel.GPT_OSS_120B.value


def test_should_judge_respects_the_sample_rate():
    assert should_judge(sample_rate=1.0) is True
    assert should_judge(sample_rate=0.0) is False


def test_judge_response_returns_the_parsed_score():
    adapter = _FakeAdapter(content="9")
    assert judge_response(adapter, "what is 2+2", "4", responder=ArcModel.KIMI_K3.value) == 0.9


def test_judge_response_sends_both_question_and_answer_to_the_judge():
    adapter = _FakeAdapter(content="6")
    judge_response(adapter, "why is the sky blue", "rayleigh scattering", responder=ArcModel.KIMI_K3.value)

    sent = adapter.calls[0][1][0]["content"]
    assert "why is the sky blue" in sent
    assert "rayleigh scattering" in sent


def test_judge_response_does_not_cap_max_tokens():
    # a tight budget gets spent on hidden reasoning tokens before any
    # content appears, which would make every judgement come back empty
    adapter = _FakeAdapter(content="5")
    judge_response(adapter, "q", "a", responder=ArcModel.KIMI_K3.value)

    assert "max_tokens" not in adapter.calls[0][2]


def test_judge_response_returns_none_when_the_call_fails():
    adapter = _FakeAdapter(error=ConnectionError("arc unreachable"))
    assert judge_response(adapter, "q", "a", responder=ArcModel.KIMI_K3.value) is None


def test_judge_response_returns_none_on_an_unparseable_reply():
    adapter = _FakeAdapter(content="no idea, sorry")
    assert judge_response(adapter, "q", "a", responder=ArcModel.KIMI_K3.value) is None
