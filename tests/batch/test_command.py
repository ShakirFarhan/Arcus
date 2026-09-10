from pathlib import Path

import pytest

from arcus.batch.command import (
    estimate_runtime,
    infer_choices,
    pick_model,
    resolve_output_path,
)
from arcus.batch.io import InputRow

ALL_FOUR = ["gpt-oss-120b", "GLM-5.3", "Kimi-K3", "DeepSeek-V4-Flash"]


def _row(i, text):
    return InputRow(index=i, fields={"comment": text})


# --- reading the labels out of plain English --------------------------------


@pytest.mark.parametrize(
    "instruction, expected",
    [
        ("classify the sentiment as positive, negative, or neutral", ["positive", "negative", "neutral"]),
        ("label each as spam or not spam", ["spam", "not spam"]),
        ("sort into bug, feature, or question", ["bug", "feature", "question"]),
        ("classify as yes/no", ["yes", "no"]),
        ("mark it as either urgent or routine", ["urgent", "routine"]),
    ],
)
def test_labels_are_inferred_from_the_users_own_sentence(instruction, expected):
    assert infer_choices(instruction) == expected


@pytest.mark.parametrize(
    "instruction",
    [
        "summarize the main finding of this abstract",
        "extract the author's name",
        "write a one sentence summary",
        "what is this about",
    ],
)
def test_open_ended_instructions_do_not_invent_labels(instruction):
    # guessing wrong here would reject perfectly good free-text answers
    assert infer_choices(instruction) is None


def test_a_long_prose_instruction_with_commas_is_not_mistaken_for_a_label_list():
    instruction = (
        "read each response carefully and, considering tone as well as content, "
        "describe what the student found difficult, what they enjoyed, and anything "
        "they suggested changing, in one sentence"
    )
    assert infer_choices(instruction) is None


def test_duplicate_labels_are_rejected_rather_than_silently_deduped():
    assert infer_choices("classify as positive, positive, or negative") is None


# --- output naming ----------------------------------------------------------


def test_the_default_output_sits_next_to_the_input():
    assert resolve_output_path(Path("/data/survey.csv"), None) == Path("/data/survey.labeled.csv")


def test_jsonl_keeps_its_extension():
    assert resolve_output_path(Path("/data/rows.jsonl"), None) == Path("/data/rows.labeled.jsonl")


def test_an_explicit_output_path_wins():
    assert resolve_output_path(Path("/data/in.csv"), Path("/out/x.csv")) == Path("/out/x.csv")


def test_an_extensionless_input_still_produces_a_csv():
    assert resolve_output_path(Path("/data/export"), None) == Path("/data/export.labeled.csv")


# --- choosing the model -----------------------------------------------------


def test_a_normal_file_can_use_any_model():
    rows = [_row(i, "a short comment") for i in range(10)]
    assert pick_model(None, rows, "comment", ALL_FOUR) == "gpt-oss-120b"


def test_the_model_is_sized_against_the_biggest_row_not_the_average():
    # one enormous row among small ones still has to fit, or the job
    # discovers that at the end, after the waiting
    rows = [_row(i, "short") for i in range(500)]
    rows.append(_row(500, "x" * 700_000))  # ~175k tokens, past the 128k models

    assert pick_model(None, rows, "comment", ALL_FOUR) == "DeepSeek-V4-Flash"


def test_no_model_is_returned_when_even_the_largest_cannot_hold_a_row():
    rows = [_row(0, "y" * 3_000_000)]
    assert pick_model(None, rows, "comment", ALL_FOUR) is None


def test_headroom_is_left_for_the_instruction_and_the_answer():
    # a row that exactly fills the window leaves nothing for the prompt
    # wrapped around it
    rows = [_row(0, "z" * (131_072 * 4))]
    assert pick_model(None, rows, "comment", ["gpt-oss-120b"]) is None


def test_no_candidates_means_no_model():
    assert pick_model(None, [_row(0, "hi")], "comment", []) is None


# --- runtime estimation -----------------------------------------------------


def test_the_estimate_is_a_range_not_a_single_number():
    # five rows is far too small a sample to quote a precise figure from
    estimate = estimate_runtime([1000.0] * 5, remaining=800, concurrency=6)
    assert " to " in estimate


def test_a_bigger_job_estimates_longer():
    small = estimate_runtime([1000.0] * 5, remaining=50, concurrency=6)
    large = estimate_runtime([1000.0] * 5, remaining=50_000, concurrency=6)
    assert small != large


def test_nothing_to_estimate_from_says_so():
    assert estimate_runtime([], remaining=100, concurrency=6) == "unknown"
    assert estimate_runtime([1000.0], remaining=0, concurrency=6) == "unknown"


def test_more_workers_shortens_the_estimate():
    slow = estimate_runtime([2000.0] * 5, remaining=5_000, concurrency=1)
    fast = estimate_runtime([2000.0] * 5, remaining=5_000, concurrency=8)
    assert slow != fast


@pytest.mark.parametrize(
    "instruction",
    [
        # every one of these reads like a list to a naive comma split,
        # and inferring labels from any of them would reject every real
        # answer the model produced
        "describe what they found difficult, what they enjoyed, and what they would change",
        "extract the author, the year, and the journal name",
        "summarize the argument, the evidence, and the conclusion in a sentence",
        "rewrite this as a clearer, shorter, more direct sentence",
    ],
)
def test_prose_that_merely_contains_commas_never_becomes_a_label_list(instruction):
    assert infer_choices(instruction) is None


def test_a_label_list_of_more_than_six_is_treated_as_prose():
    instruction = "classify as a, b, c, d, e, f, g, h"
    assert infer_choices(instruction) is None
