import pytest

from arcus.routing.context import Context, LengthBucket, TaskType, classify

PYTHON_TRACEBACK = """
Traceback (most recent call last):
  File "broken.py", line 12, in <module>
    result = 1 / 0
ZeroDivisionError: division by zero
"""

JS_SNIPPET = """
```js
function add(a, b) {
  return a + b;
}
```
"""


@pytest.mark.parametrize(
    "text, expected",
    [
        (PYTHON_TRACEBACK, TaskType.CODE),
        (JS_SNIPPET, TaskType.CODE),
        ("solve for x: 2x + 5 = 15", TaskType.REASONING_MATH),
        ("what's the time complexity of quicksort", TaskType.REASONING_MATH),
        ("summarize the following document: ...", TaskType.LONG_DOCUMENT),
        ("write a short story about a dragon", TaskType.WRITING),
        ("what's the capital of France", TaskType.GENERAL),
    ],
)
def test_task_type_classification(text, expected):
    assert classify(text).task_type == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # these are all deliberately keyword-free, none of the regex
        # rules should fire on them, so a correct result here only
        # happens if the embedding fallback is actually doing its job.
        ("my list keeps giving me an out of range error when I loop through it", TaskType.CODE),
        ("if I roll two dice what's the chance both come up as six", TaskType.REASONING_MATH),
        ("I need something punchy to open my speech with", TaskType.WRITING),
        ("here's a bunch of legal text, tell me what stands out", TaskType.LONG_DOCUMENT),
        ("what's a good name for a pet hamster", TaskType.GENERAL),
    ],
)
def test_embedding_fallback_classifies_keyword_free_prompts(text, expected):
    assert classify(text).task_type == expected


def test_code_wins_over_writing_when_both_present():
    text = "write an essay explaining what this code does:\n```python\ndef f(): pass\n```"
    assert classify(text).task_type == TaskType.CODE


def test_long_wall_of_text_with_no_other_signal_becomes_long_document():
    text = "the weather has been strange lately. " * 300
    result = classify(text)
    assert result.task_type == TaskType.LONG_DOCUMENT
    assert result.length_bucket in (LengthBucket.LONG, LengthBucket.VERY_LONG)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("hi", LengthBucket.SHORT),
        ("explain how binary search works " * 40, LengthBucket.MEDIUM),
        ("lorem ipsum dolor sit amet " * 200, LengthBucket.LONG),
        ("lorem ipsum dolor sit amet " * 1500, LengthBucket.VERY_LONG),
    ],
)
def test_length_buckets(text, expected):
    assert classify(text).length_bucket == expected


def test_context_key_format():
    ctx = Context(task_type=TaskType.CODE, length_bucket=LengthBucket.SHORT)
    assert ctx.key == "code:short"
