from arcus.cache.benchmark import BENCHMARK_PAIRS, run_benchmark


def test_benchmark_set_has_a_meaningful_number_of_pairs():
    assert 50 <= len(BENCHMARK_PAIRS) <= 100


def test_param_diff_improves_precision_over_naive_cosine():
    results = run_benchmark()
    naive = results["naive_cosine"]
    full = results["with_param_diff"]

    # this is the whole point of building the parameter-diff check: it
    # should reject near-duplicate-but-actually-different pairs that
    # naive cosine similarity alone can't tell apart.
    assert full.precision > naive.precision

    # a real cost is expected (rejecting a few legitimate matches that
    # happen to differ in a number/entity), but it shouldn't gut the
    # cache's usefulness entirely.
    assert full.recall > 0.5
