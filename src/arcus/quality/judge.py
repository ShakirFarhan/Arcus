import random
import re

from arcus.adapters.arc_adapter import ArcAdapter, ArcModel

# how often a passing response gets sent for grading. every response
# would double this tool's request volume against shared infra for a
# signal that doesn't need to be that dense, one in four still
# accumulates a real per-model picture over a few hundred questions.
JUDGE_SAMPLE_RATE = 0.25

# gpt-oss-120b is the fastest and cheapest of the four, which matters
# when this is pure overhead on top of the answer the user already got.
# the one case it can't grade is its own output: a model scoring its own
# answers is a known source of self-preference bias, and it's the exact
# kind of hole that makes a whole measurement dismissible, so those go
# to GLM-5.3 instead.
_DEFAULT_JUDGE = ArcModel.GPT_OSS_120B.value
_ALTERNATE_JUDGE = ArcModel.GLM_5_3.value

_RUBRIC = """\
You are grading how well an assistant answered a user's question.

Question:
{question}

Answer:
{answer}

Score the answer from 0 to 10:
0-2: wrong, off topic, or doesn't address the question
3-5: partly addresses it, with real gaps or errors
6-8: correct and useful, minor gaps at most
9-10: correct, complete, and well aimed at what was asked

Reply with the number only."""

# the models write to a hidden reasoning field before content, so the
# number can arrive with reasoning residue around it. pull the first
# number out rather than requiring the whole reply to be one.
_SCORE_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def judge_model_for(responder: str) -> str:
    return _ALTERNATE_JUDGE if responder == _DEFAULT_JUDGE else _DEFAULT_JUDGE


def should_judge(sample_rate: float = JUDGE_SAMPLE_RATE) -> bool:
    return random.random() < sample_rate


def parse_score(content: str | None) -> float | None:
    """Pulls a 0-10 score out of the judge's reply and normalizes it to
    the 0-1 range compute_reward expects. Returns None when there's no
    number to find at all, so the caller can tell "the judge said this
    was bad" apart from "the judge didn't answer usefully".
    """
    if not content:
        return None

    match = _SCORE_PATTERN.search(content)
    if match is None:
        return None

    score = float(match.group())
    # a judge that ignores the scale and replies 45 or -3 is telling us
    # nothing trustworthy, clamping keeps one malformed reply from
    # dragging an arm's average somewhere the rubric never allowed
    return max(0.0, min(1.0, score / 10))


def judge_response(
    adapter: ArcAdapter,
    question: str,
    answer: str,
    responder: str,
) -> float | None:
    """Grades one answer, 0-1. Returns None if the judge call failed or
    came back unparseable, which the caller should treat as "no opinion"
    rather than "scored zero", there's a real difference between a bad
    answer and a judge that didn't work.

    Deliberately doesn't set max_tokens: these are reasoning models and
    a tight budget gets spent on hidden reasoning tokens before any
    content comes out, which would turn every judgement into an empty
    reply. See the note in the README about this.
    """
    prompt = _RUBRIC.format(question=question, answer=answer)

    try:
        completion = adapter.chat(
            judge_model_for(responder),
            [{"role": "user", "content": prompt}],
        )
    except Exception:
        # the answer this is grading was already shown to the user a
        # while ago, a judging failure is not worth surfacing or
        # retrying, the row just stays unscored
        return None

    return parse_score(completion.choices[0].message.content)
