import pytest

from arcus.routing.reward import (
    COST_SCORES,
    RewardWeights,
    compute_reward,
    normalize_latency,
)


def test_cost_scores_are_all_in_unit_range():
    for score in COST_SCORES.values():
        assert 0.0 <= score <= 1.0


def test_cost_scores_match_the_real_pricing_spread():
    # gpt-oss-120b and DeepSeek-V4-Flash are both cheap speed-tier
    # models, Kimi-K3 is priced as a premium agentic model, GLM-5.3
    # sits in between.
    assert COST_SCORES["gpt-oss-120b"] > COST_SCORES["GLM-5.3"]
    assert COST_SCORES["DeepSeek-V4-Flash"] > COST_SCORES["GLM-5.3"]
    assert COST_SCORES["GLM-5.3"] > COST_SCORES["Kimi-K3"]
    assert COST_SCORES["Kimi-K3"] == 0.0


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
    bad_weights = RewardWeights(quality=0.5, latency=0.5, cost=0.5)
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


def test_cheaper_model_gives_higher_reward_all_else_equal():
    expensive = compute_reward(latency_ms=1000, model="Kimi-K3", quality_score=1.0)
    cheap = compute_reward(latency_ms=1000, model="DeepSeek-V4-Flash", quality_score=1.0)
    assert cheap > expensive


def test_good_outcome_on_cheap_model_beats_bad_outcome_on_expensive_model():
    good = compute_reward(latency_ms=200, model="DeepSeek-V4-Flash", quality_score=1.0)
    bad = compute_reward(latency_ms=19_000, model="Kimi-K3", quality_score=0.0)
    assert good - bad > 0.5


def test_compute_reward_does_not_crash_for_a_model_with_no_published_rate():
    # web search routes through ARC's "legacy-tool-calling" model
    # variants, which aren't in MODEL_HOSTING_RATES, this shouldn't
    # raise a KeyError just because that name isn't in the cost table.
    reward = compute_reward(
        latency_ms=500, model="gpt-oss-120b-thinking-high-legacy-tool-calling", quality_score=1.0
    )
    assert 0.0 <= reward <= 1.0
