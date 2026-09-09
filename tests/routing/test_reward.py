import pytest

from arcus.routing.reward import (
    RewardWeights,
    compute_reward,
    normalize_latency,
)


@pytest.mark.parametrize(
    "latency_ms, expected",
    [
        (0, 1.0),
        (20_000, 0.0),
        (40_000, 0.0),
    ],
)
def test_normalize_latency(latency_ms, expected):
    assert normalize_latency(latency_ms) == expected


def test_normalize_latency_stays_in_unit_range_for_mid_values():
    score = normalize_latency(10_000)
    assert 0.0 < score < 1.0


def test_compute_reward_rejects_weights_that_dont_sum_to_one():
    bad_weights = RewardWeights(quality=0.5, latency=0.9)
    with pytest.raises(ValueError):
        compute_reward(latency_ms=100, model="gpt-oss-120b", quality_score=1.0, weights=bad_weights)


def test_higher_quality_gives_higher_reward_all_else_equal():
    low = compute_reward(latency_ms=1000, model="GLM-5.3", quality_score=0.0)
    high = compute_reward(latency_ms=1000, model="GLM-5.3", quality_score=1.0)
    assert high > low


def test_lower_latency_gives_higher_reward_all_else_equal():
    slow = compute_reward(latency_ms=15_000, model="GLM-5.3", quality_score=1.0)
    fast = compute_reward(latency_ms=100, model="GLM-5.3", quality_score=1.0)
    assert fast > slow


def test_quality_outweighs_latency():
    # a fast wrong answer is worth about nothing, so it has to score
    # below a slow correct one
    fast_and_wrong = compute_reward(latency_ms=0, model="GLM-5.3", quality_score=0.0)
    slow_and_right = compute_reward(latency_ms=19_000, model="GLM-5.3", quality_score=1.0)
    assert slow_and_right > fast_and_wrong


def test_reward_no_longer_depends_on_which_model_answered():
    # the cost term was the only thing that varied by model, and it was
    # built from other providers' pricing for a service that's free.
    # with it gone, two identical outcomes score identically no matter
    # which arm produced them.
    a = compute_reward(latency_ms=1000, model="Kimi-K3", quality_score=0.8)
    b = compute_reward(latency_ms=1000, model="DeepSeek-V4-Flash", quality_score=0.8)
    assert a == b


def test_reward_stays_in_unit_range_for_any_model_name():
    # web search and reasoning variants route through model ids that no
    # lookup table was ever going to contain
    reward = compute_reward(
        latency_ms=500, model="gpt-oss-120b-thinking-high-legacy-tool-calling", quality_score=1.0
    )
    assert 0.0 <= reward <= 1.0


def test_perfect_outcome_scores_one_and_worst_scores_zero():
    assert compute_reward(latency_ms=0, model="GLM-5.3", quality_score=1.0) == pytest.approx(1.0)
    assert compute_reward(latency_ms=20_000, model="GLM-5.3", quality_score=0.0) == pytest.approx(0.0)
