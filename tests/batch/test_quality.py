from arcus.batch.quality import (
    compare_opinions,
    sample_indexes,
    score_against_gold,
    split_gold,
)


# --- gold set: the split has to stay honest ---------------------------------


def test_gold_rows_are_never_used_both_to_teach_and_to_test():
    labelled = [(f"text {i}", "positive" if i % 2 else "negative") for i in range(40)]

    split = split_gold(labelled, seed=1)

    example_texts = {t for t, _ in split.examples}
    held_texts = {t for t, _ in split.held_out}
    # overlap would mean scoring the model on the answers it was shown,
    # which reports a number far better than the truth
    assert example_texts & held_texts == set()
    assert len(example_texts) + len(held_texts) == 40


def test_the_split_is_shuffled_not_sliced_in_order():
    # a file sorted by label would otherwise put every positive in the
    # examples and every negative in the test set
    labelled = [(f"t{i}", "positive") for i in range(20)] + [(f"t{i}", "negative") for i in range(20, 40)]

    split = split_gold(labelled, seed=3)
    held_labels = {label for _, label in split.held_out}

    assert held_labels == {"positive", "negative"}


def test_the_split_is_reproducible_for_a_given_seed():
    labelled = [(f"t{i}", "x") for i in range(20)]
    assert split_gold(labelled, seed=7) == split_gold(labelled, seed=7)


def test_too_few_labels_are_spent_on_testing_rather_than_teaching():
    # a weak accuracy estimate beats a contaminated one
    split = split_gold([("only one", "positive")], seed=1)
    assert split.examples == []
    assert len(split.held_out) == 1


def test_no_labels_at_all_is_not_an_error():
    split = split_gold([], seed=1)
    assert split.examples == [] and split.held_out == []


# --- accuracy against ground truth ------------------------------------------


def test_accuracy_counts_matches_and_reports_the_misses():
    gold = [("a", "positive"), ("b", "negative"), ("c", "neutral")]
    predictions = {"a": "positive", "b": "positive", "c": "neutral"}

    result = score_against_gold(predictions, gold)

    assert result.correct == 2
    assert result.total == 3
    assert result.rate == 2 / 3
    assert result.misses == [("b", "negative", "positive")]


def test_accuracy_ignores_case_and_padding():
    result = score_against_gold({"a": "  Positive "}, [("a", "positive")])
    assert result.correct == 1


def test_a_row_with_no_prediction_counts_against_accuracy():
    # a row the model failed on is not a row it got right
    result = score_against_gold({}, [("a", "positive")])

    assert result.correct == 0
    assert result.misses[0][2] == "(no answer)"


def test_an_empty_gold_set_reports_zero_rather_than_dividing_by_zero():
    assert score_against_gold({}, []).rate == 0.0


# --- agreement between models -----------------------------------------------


def test_agreement_counts_only_rows_both_models_answered():
    primary = {0: "positive", 1: "negative", 2: "neutral"}
    second = {0: "positive", 1: "positive"}  # never got to row 2

    result = compare_opinions(primary, second)

    assert result.compared == 2
    assert result.agreed == 1
    assert result.disagreements == [(1, "negative", "positive")]


def test_agreement_ignores_case_differences():
    assert compare_opinions({0: "Positive"}, {0: "positive"}).agreed == 1


def test_total_agreement_and_total_disagreement():
    assert compare_opinions({0: "a", 1: "a"}, {0: "a", 1: "a"}).rate == 1.0
    assert compare_opinions({0: "a"}, {0: "b"}).rate == 0.0


def test_no_overlap_reports_nothing_rather_than_crashing():
    result = compare_opinions({0: "a"}, {5: "b"})
    assert result.compared == 0
    assert result.rate == 0.0


# --- sampling ---------------------------------------------------------------


def test_the_cross_check_sample_is_spread_across_the_file_not_the_front():
    # real exports arrive sorted, so checking the first slice would
    # describe one corner of the data and call it the whole thing
    picked = sample_indexes(1000, 0.2, seed=5)

    assert len(picked) == 200
    assert max(picked) > 800
    assert min(picked) < 200


def test_sampling_is_reproducible_and_ordered():
    a = sample_indexes(500, 0.1, seed=9)
    assert a == sample_indexes(500, 0.1, seed=9)
    assert a == sorted(a)


def test_a_tiny_file_still_gets_at_least_one_check():
    assert len(sample_indexes(3, 0.2, seed=1)) == 1


def test_asking_for_everything_returns_everything():
    assert sample_indexes(10, 1.0, seed=1) == list(range(10))


def test_no_rows_or_no_fraction_means_no_sample():
    assert sample_indexes(0, 0.5, seed=1) == []
    assert sample_indexes(100, 0.0, seed=1) == []
