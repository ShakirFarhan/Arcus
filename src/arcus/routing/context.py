import re
from dataclasses import dataclass
from enum import Enum


class TaskType(str, Enum):
    CODE = "code"
    REASONING_MATH = "reasoning_math"
    WRITING = "writing"
    LONG_DOCUMENT = "long_document"
    GENERAL = "general"


class LengthBucket(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    VERY_LONG = "very_long"


_CODE_MARKERS = re.compile(
    r"```"
    r"|traceback \(most recent call last\)"
    r"|\b(syntax|type|name|index|key|attribute|value|indentation)error\b"
    r"|\bexception in thread\b"
    r"|\bnpm err!"
    r"|^\s*(def|class|import|from|function|public|private|const|let)\s",
    re.IGNORECASE | re.MULTILINE,
)

_MATH_MARKERS = re.compile(
    r"\b(solve|prove|derivative|integral|equation|theorem|factorize|simplify)\b"
    r"|\btime complexity\b|\bbig[- ]o\b"
    r"|solve for [a-z]\b",
    re.IGNORECASE,
)

_DOC_MARKERS = re.compile(
    r"\bsummariz(e|ing)\b"
    r"|\btl;?dr\b"
    r"|\bthe following (document|article|text|paper)\b"
    r"|\bgiven the (text|document) below\b",
    re.IGNORECASE,
)

_WRITING_MARKERS = re.compile(
    r"\b(write|draft|compose)\s+(a|an|me a)\b"
    r"|\b(essay|blog post|cover letter|short story|poem)\b",
    re.IGNORECASE,
)

# anchor prompts for the embedding fallback below. GENERAL gets its own
# examples too, not just the other four, otherwise the fallback would
# always pick one of code/math/writing/doc even for plain small talk
# since there'd be nothing else in the running.
_SIMILARITY_THRESHOLD = 0.35

_TASK_TYPE_EXAMPLES: dict[TaskType, list[str]] = {
    TaskType.CODE: [
        "why is my for loop not terminating",
        "getting a null pointer exception in this function",
        "how do I fix this compiler error",
        "review this pull request for bugs",
        "my api call keeps returning a 500",
    ],
    TaskType.REASONING_MATH: [
        "walk me through the proof of this theorem",
        "what's the expected value of this dice roll",
        "help me figure out the recurrence relation for this algorithm",
        "is this argument logically valid",
        "explain why this inequality holds",
    ],
    TaskType.WRITING: [
        "help me phrase this paragraph better",
        "can you make this sound more persuasive",
        "give me a catchy title for this post",
        "rewrite this in a more formal tone",
        "I need a few opening lines for a speech",
    ],
    TaskType.LONG_DOCUMENT: [
        "here's a long report, pull out the key takeaways",
        "go through this contract and flag anything unusual",
        "condense this research paper into a few bullet points",
        "what are the main arguments made across these pages",
    ],
    TaskType.GENERAL: [
        "what's a good movie to watch tonight",
        "how's the weather looking this weekend",
        "tell me something interesting",
        "what time zone is Tokyo in",
        "hey, how are you",
    ],
}

_embedding_classifier_cache = None


def _get_embedding_classifier():
    # unlike the db engine, this one is worth caching as a real
    # singleton. loading the model is the expensive part (a few hundred
    # ms to a couple seconds), so paying it once per process is the
    # whole point. the model itself comes from arcus.embeddings, shared
    # with the semantic cache, so a request that needs both this fallback
    # and a cache lookup only pays that load cost once, not twice.
    global _embedding_classifier_cache
    if _embedding_classifier_cache is not None:
        return _embedding_classifier_cache

    from arcus.embeddings import embed

    labels: list[TaskType] = []
    examples: list[str] = []
    for task_type, prompts in _TASK_TYPE_EXAMPLES.items():
        for prompt in prompts:
            labels.append(task_type)
            examples.append(prompt)

    anchor_embeddings = embed(examples)

    _embedding_classifier_cache = (labels, anchor_embeddings)
    return _embedding_classifier_cache


def _classify_by_embedding(text: str) -> TaskType:
    try:
        labels, anchor_embeddings = _get_embedding_classifier()
        from arcus.embeddings import embed

        query_embedding = embed([text])[0]
    except Exception:
        # no internet on first run, weights not cached yet, whatever the
        # reason, the regex rules already did their best. don't let a
        # missing model take classification down entirely.
        return TaskType.GENERAL

    similarities = anchor_embeddings @ query_embedding
    best_idx = int(similarities.argmax())

    if similarities[best_idx] < _SIMILARITY_THRESHOLD:
        return TaskType.GENERAL

    return labels[best_idx]


def _looks_like_code(text: str) -> bool:
    return bool(_CODE_MARKERS.search(text))


def _looks_like_math(text: str) -> bool:
    return bool(_MATH_MARKERS.search(text))


def _looks_like_document_task(text: str) -> bool:
    return bool(_DOC_MARKERS.search(text))


def _looks_like_writing_request(text: str) -> bool:
    return bool(_WRITING_MARKERS.search(text))


def classify_task_type(text: str) -> TaskType:
    # order matters here. code and error output are the most confident
    # signal we ever get (and piping errors in is the single most common
    # thing this tool will see), so that gets checked first no matter
    # what else is in the prompt.
    if _looks_like_code(text):
        return TaskType.CODE
    if _looks_like_math(text):
        return TaskType.REASONING_MATH
    if _looks_like_document_task(text):
        return TaskType.LONG_DOCUMENT
    if _looks_like_writing_request(text):
        return TaskType.WRITING

    # nothing hit a keyword. rather than giving up and calling it
    # GENERAL, check how close it reads to labeled examples of each
    # category, catches phrasing the regex rules just don't cover.
    result = _classify_by_embedding(text)

    # a long wall of text with no other signal is still a document task
    # even if the embedding read came back wishy-washy about it, raw
    # length beats a low-confidence embedding guess here.
    if result == TaskType.GENERAL and bucket_length(text) in (LengthBucket.LONG, LengthBucket.VERY_LONG):
        return TaskType.LONG_DOCUMENT

    return result


def bucket_length(text: str) -> LengthBucket:
    # rough chars-per-token estimate (~4), not a real tokenizer. none of
    # ARC's four models use OpenAI's tokenizer anyway so an exact count
    # would just be fake precision. these thresholds are a first guess,
    # worth retuning once we have real query lengths to look at.
    approx_tokens = len(text) // 4

    if approx_tokens < 100:
        return LengthBucket.SHORT
    if approx_tokens < 500:
        return LengthBucket.MEDIUM
    if approx_tokens < 2000:
        return LengthBucket.LONG
    return LengthBucket.VERY_LONG


@dataclass(frozen=True)
class Context:
    task_type: TaskType
    length_bucket: LengthBucket

    @property
    def key(self) -> str:
        # this is the dict key the per-context bandits key off of, one
        # bandit instance per bucket.
        return f"{self.task_type.value}:{self.length_bucket.value}"


def classify(text: str) -> Context:
    return Context(task_type=classify_task_type(text), length_bucket=bucket_length(text))
