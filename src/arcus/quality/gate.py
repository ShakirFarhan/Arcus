import re
import time
from dataclasses import dataclass

from openai import APIError
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError

from arcus.adapters.arc_adapter import ArcAdapter
from arcus.routing.bandit import ContextualBandit
from arcus.routing.reward import compute_reward

_REFUSAL_MARKERS = re.compile(
    r"i cannot assist|i can't assist"
    r"|i cannot help|i can't help"
    r"|as an ai language model"
    r"|i'm sorry,? but i can'?t"
    r"|i am not able to|i'm not able to"
    r"|i won'?t be able to"
    r"|i must decline"
    r"|against my guidelines",
    re.IGNORECASE,
)

_REPETITION_THRESHOLD = 0.5
# below this many words there aren't enough trigrams for the duplication
# ratio to mean anything, a two-sentence answer isn't "looping" just
# because it reuses a word.
_MIN_WORDS_FOR_REPETITION_CHECK = 9


@dataclass(frozen=True)
class QualityIssue:
    kind: str
    detail: str


@dataclass(frozen=True)
class QualityCheckResult:
    passed: bool
    issues: list[QualityIssue]


def check_truncation(finish_reason: str | None) -> QualityIssue | None:
    if finish_reason == "length":
        return QualityIssue("truncated", "response was cut off before finishing (finish_reason=length)")
    return None


def check_empty(content: str | None) -> QualityIssue | None:
    # deliberately not a fuzzy "too short" length check, a correct answer
    # can legitimately be one word. this only catches actually-empty output.
    if not content or not content.strip():
        return QualityIssue("empty", "response was empty or whitespace only")
    return None


def check_repetition(content: str | None, threshold: float = _REPETITION_THRESHOLD) -> QualityIssue | None:
    if not content:
        return None

    words = content.split()
    if len(words) < _MIN_WORDS_FOR_REPETITION_CHECK:
        return None

    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    duplication_ratio = 1 - len(set(trigrams)) / len(trigrams)

    if duplication_ratio > threshold:
        return QualityIssue("repetitive", f"trigram duplication ratio {duplication_ratio:.2f} exceeds {threshold}")
    return None


def check_refusal(content: str | None) -> QualityIssue | None:
    if not content:
        return None
    if _REFUSAL_MARKERS.search(content):
        return QualityIssue("refusal", "response matched a known refusal phrase")
    return None


def check_schema(content: str | None, schema: type[BaseModel] | None) -> QualityIssue | None:
    if schema is None:
        return None
    if not content:
        return QualityIssue("schema_invalid", "no content to validate against schema")
    try:
        schema.model_validate_json(content)
    except ValidationError as e:
        return QualityIssue("schema_invalid", str(e))
    except ValueError as e:
        # model_validate_json raises a plain ValueError (from the underlying
        # json parse) when content isn't valid JSON at all, not a
        # ValidationError, both count as a schema failure here
        return QualityIssue("schema_invalid", str(e))
    return None


def check_response(
    content: str | None,
    finish_reason: str | None,
    schema: type[BaseModel] | None = None,
) -> QualityCheckResult:
    checks = [
        check_truncation(finish_reason),
        check_empty(content),
        check_repetition(content),
        check_refusal(content),
        check_schema(content, schema),
    ]
    issues = [issue for issue in checks if issue is not None]
    return QualityCheckResult(passed=len(issues) == 0, issues=issues)


@dataclass(frozen=True)
class AttemptDetail:
    model: str
    passed: bool
    reward: float
    latency_ms: float
    propensity: float
    issues: list[QualityIssue]


@dataclass(frozen=True)
class QualityGateOutcome:
    # None only when every arm errored out at the API level rather than
    # returning a bad answer, there was nothing to hand back at all
    response: ChatCompletion | None
    model_used: str
    passed: bool
    attempts: list[AttemptDetail]
    issues: list[QualityIssue]


def call_with_quality_gate(
    adapter: ArcAdapter,
    bandit: ContextualBandit,
    context_key: str,
    messages: list[dict],
    schema: type[BaseModel] | None = None,
    max_attempts: int | None = None,
) -> QualityGateOutcome:
    """Calls ARC, runs the response through the quality gate, and retries
    with a different arm on failure, feeding a real (not hardcoded)
    negative reward back to the bandit each time so it actually learns
    from the failure instead of just being told "not this one." Every
    attempt gets logged, not just the winning one, so a caller can later
    tell how often each model got caught by the gate, not only which
    model ended up serving the response.
    """
    max_attempts = min(max_attempts or len(bandit.arms), len(bandit.arms))

    already_tried: set[str] = set()
    attempts: list[AttemptDetail] = []
    completion = None
    arm = None
    last_issues: list[QualityIssue] = []

    for _ in range(max_attempts):
        arm = bandit.select_arm(context_key, exclude=already_tried)
        # propensity has to be read off the bandit's state right here,
        # before update() below changes the counts it's computed from
        propensity = bandit.propensity(context_key, arm, exclude=already_tried)

        start = time.monotonic()
        try:
            completion = adapter.chat(arm, messages)
        except APIError as e:
            # a network blip, rate limit, or ARC-side outage on this one
            # model isn't a reason to give up on the whole request, treat
            # it exactly like a failed quality check: log it, penalize
            # this arm for this context, and let the loop try the next
            # one instead of crashing the CLI with a raw traceback.
            latency_ms = (time.monotonic() - start) * 1000
            issue = QualityIssue("api_error", str(e))
            reward = compute_reward(latency_ms=latency_ms, model=arm, quality_score=0.0)
            bandit.update(context_key, arm, reward)

            attempts.append(
                AttemptDetail(
                    model=arm, passed=False, reward=reward, latency_ms=latency_ms,
                    propensity=propensity, issues=[issue],
                )
            )
            last_issues = [issue]
            completion = None
            already_tried.add(arm)
            continue

        latency_ms = (time.monotonic() - start) * 1000

        choice = completion.choices[0]
        result = check_response(choice.message.content, choice.finish_reason, schema)
        last_issues = result.issues

        quality_score = 1.0 if result.passed else 0.0
        reward = compute_reward(latency_ms=latency_ms, model=arm, quality_score=quality_score)
        bandit.update(context_key, arm, reward)

        attempts.append(
            AttemptDetail(
                model=arm,
                passed=result.passed,
                reward=reward,
                latency_ms=latency_ms,
                propensity=propensity,
                issues=result.issues,
            )
        )

        if result.passed:
            return QualityGateOutcome(
                response=completion, model_used=arm, passed=True, attempts=attempts, issues=[]
            )

        already_tried.add(arm)

    # every arm got tried and none passed (or errored out), hand back the
    # last attempt and let the caller decide what to do. response may be
    # None here if the very last arm failed with an API error rather than
    # a bad answer, callers need to handle that.
    return QualityGateOutcome(
        response=completion, model_used=arm, passed=False, attempts=attempts, issues=last_issues
    )
