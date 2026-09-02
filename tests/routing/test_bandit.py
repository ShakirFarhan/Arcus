import math
import random

import numpy as np
import pytest

from arcus.routing.bandit import (
    ContextualBandit,
    EpsilonGreedyBandit,
    RandomBandit,
    ThompsonSamplingBandit,
    UCB1Bandit,
)

GOOD_REWARD_RANGE = (0.7, 0.9)
BAD_REWARD_RANGE = (0.1, 0.3)


def _reward_for(arm: str) -> float:
    lo, hi = GOOD_REWARD_RANGE if arm == "good" else BAD_REWARD_RANGE
    return random.uniform(lo, hi)


@pytest.mark.parametrize("bandit_cls", [EpsilonGreedyBandit, UCB1Bandit])
def test_count_based_bandits_try_every_arm_once_before_repeating(bandit_cls):
    bandit = bandit_cls(["a", "b", "c"])
    seen = []
    for _ in range(3):
        arm = bandit.select_arm()
        seen.append(arm)
        bandit.update(arm, reward=0.5)
    assert set(seen) == {"a", "b", "c"}


def test_epsilon_greedy_converges_to_better_arm():
    random.seed(42)
    bandit = EpsilonGreedyBandit(["good", "bad"], epsilon=0.1)

    counts = {"good": 0, "bad": 0}
    for _ in range(500):
        arm = bandit.select_arm()
        counts[arm] += 1
        bandit.update(arm, _reward_for(arm))

    assert counts["good"] > counts["bad"]
    assert counts["good"] / 500 > 0.8


def test_ucb1_converges_to_better_arm():
    random.seed(42)
    bandit = UCB1Bandit(["good", "bad"])

    counts = {"good": 0, "bad": 0}
    for _ in range(500):
        arm = bandit.select_arm()
        counts[arm] += 1
        bandit.update(arm, _reward_for(arm))

    assert counts["good"] > counts["bad"]
    assert counts["good"] / 500 > 0.8


def test_thompson_sampling_converges_to_better_arm():
    random.seed(42)
    np.random.seed(42)
    bandit = ThompsonSamplingBandit(["good", "bad"])

    counts = {"good": 0, "bad": 0}
    for _ in range(500):
        arm = bandit.select_arm()
        counts[arm] += 1
        bandit.update(arm, _reward_for(arm))

    assert counts["good"] > counts["bad"]
    assert counts["good"] / 500 > 0.8


def test_thompson_sampling_beta_update_favors_high_reward_arm():
    bandit = ThompsonSamplingBandit(["a", "b"])
    for _ in range(20):
        bandit.update("a", 0.9)

    assert bandit._alpha["a"] > bandit._beta["a"]
    # arm "b" was never touched, still at its uniform prior
    assert bandit._alpha["b"] == 1.0
    assert bandit._beta["b"] == 1.0


def test_contextual_bandit_keeps_independent_state_per_context():
    def factory(context_key):
        return EpsilonGreedyBandit(["a", "b"], epsilon=0.0)

    contextual = ContextualBandit(factory, arms=["a", "b"])

    for _ in range(10):
        arm = contextual.select_arm("code:short")
        contextual.update("code:short", arm, reward=1.0 if arm == "a" else 0.0)

    code_bandit = contextual._get_bandit("code:short")
    assert code_bandit._pulls["a"] + code_bandit._pulls["b"] == 10

    writing_bandit = contextual._get_bandit("writing:long")
    assert writing_bandit is not code_bandit
    assert writing_bandit._pulls == {"a": 0, "b": 0}


def test_random_bandit_ignores_reward_and_spreads_roughly_evenly():
    random.seed(42)
    bandit = RandomBandit(["good", "bad"])

    counts = {"good": 0, "bad": 0}
    for _ in range(1000):
        arm = bandit.select_arm()
        counts[arm] += 1
        bandit.update(arm, _reward_for(arm))

    # a real bandit would converge hard toward "good", random selection
    # shouldn't, both arms should land close to a 50/50 split
    assert 400 < counts["good"] < 600
    assert 400 < counts["bad"] < 600


def test_random_bandit_still_records_stats_it_doesnt_use_for_selection():
    bandit = RandomBandit(["a", "b"])
    bandit.update("a", 0.9)
    bandit.update("a", 0.7)
    bandit.update("b", 0.1)

    assert bandit._pulls == {"a": 2, "b": 1}
    assert bandit._reward_sums["a"] == pytest.approx(1.6)
    assert bandit._reward_sums["b"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "bandit_cls",
    [EpsilonGreedyBandit, UCB1Bandit, ThompsonSamplingBandit, RandomBandit],
)
def test_excluded_arm_is_never_selected(bandit_cls):
    bandit = bandit_cls(["a", "b", "c"])
    for _ in range(50):
        arm = bandit.select_arm(exclude={"a", "b"})
        assert arm == "c"
        bandit.update(arm, reward=0.5)


@pytest.mark.parametrize("bandit_cls", [EpsilonGreedyBandit, UCB1Bandit])
def test_cold_start_only_cycles_through_non_excluded_arms(bandit_cls):
    bandit = bandit_cls(["a", "b", "c"])
    seen = []
    for _ in range(2):
        arm = bandit.select_arm(exclude={"a"})
        seen.append(arm)
        bandit.update(arm, reward=0.5)

    assert "a" not in seen
    assert set(seen) == {"b", "c"}


@pytest.mark.parametrize("bandit_cls", [EpsilonGreedyBandit, UCB1Bandit, RandomBandit])
def test_propensities_sum_to_one_during_cold_start(bandit_cls):
    bandit = bandit_cls(["a", "b", "c"])
    total = sum(bandit.propensity(arm) for arm in ["a", "b", "c"])
    assert total == pytest.approx(1.0)


@pytest.mark.parametrize("bandit_cls", [EpsilonGreedyBandit, UCB1Bandit])
def test_propensity_of_untried_arm_is_one_during_cold_start(bandit_cls):
    bandit = bandit_cls(["a", "b", "c"])
    # nothing pulled yet, select_arm() always returns "a" first
    assert bandit.propensity("a") == 1.0
    assert bandit.propensity("b") == 0.0
    assert bandit.propensity("c") == 0.0


def test_epsilon_greedy_propensity_after_warmup():
    bandit = EpsilonGreedyBandit(["a", "b"], epsilon=0.2)
    bandit.update("a", 1.0)
    bandit.update("b", 0.0)

    # "a" is the clear best arm: epsilon/n from exploration, plus the
    # whole (1 - epsilon) share since it's the only best arm
    assert bandit.propensity("a") == pytest.approx(0.2 / 2 + 0.8)
    assert bandit.propensity("b") == pytest.approx(0.2 / 2)
    assert bandit.propensity("a") + bandit.propensity("b") == pytest.approx(1.0)


def test_ucb1_propensity_after_warmup_is_deterministic():
    bandit = UCB1Bandit(["a", "b"])
    bandit.update("a", 1.0)
    bandit.update("b", 0.0)

    total_pulls = 2
    score_a = 1.0 + math.sqrt(2 * math.log(total_pulls) / 1)
    score_b = 0.0 + math.sqrt(2 * math.log(total_pulls) / 1)
    expected_winner = "a" if score_a > score_b else "b"

    assert bandit.propensity(expected_winner) == 1.0
    loser = "b" if expected_winner == "a" else "a"
    assert bandit.propensity(loser) == 0.0


def test_random_bandit_propensity_is_uniform():
    bandit = RandomBandit(["a", "b", "c", "d"])
    bandit.update("a", 1.0)  # random selection ignores this, propensity shouldn't move

    for arm in ["a", "b", "c", "d"]:
        assert bandit.propensity(arm) == pytest.approx(0.25)


def test_random_bandit_propensity_respects_exclude():
    bandit = RandomBandit(["a", "b", "c"])
    assert bandit.propensity("c", exclude={"a", "b"}) == 1.0
    assert bandit.propensity("a", exclude={"a", "b"}) == 0.0


def test_thompson_sampling_propensity_sums_to_roughly_one():
    np.random.seed(0)
    bandit = ThompsonSamplingBandit(["a", "b", "c"])
    for _ in range(10):
        bandit.update("a", 0.9)
        bandit.update("b", 0.5)
        bandit.update("c", 0.1)

    total = sum(bandit.propensity(arm) for arm in ["a", "b", "c"])
    # Monte Carlo estimate, not exact, just has to land close to 1.0
    assert total == pytest.approx(1.0, abs=0.05)


def test_thompson_sampling_propensity_favors_stronger_arm():
    np.random.seed(0)
    bandit = ThompsonSamplingBandit(["good", "bad"])
    for _ in range(30):
        bandit.update("good", 0.95)
        bandit.update("bad", 0.05)

    assert bandit.propensity("good") > bandit.propensity("bad")


def test_contextual_bandit_propensity_forwards_to_underlying_bandit():
    contextual = ContextualBandit(lambda _context_key: EpsilonGreedyBandit(["a", "b"], epsilon=0.0), arms=["a", "b"])
    assert contextual.propensity("code:short", "a") == 1.0


def test_contextual_bandit_arms_for_returns_the_per_context_arm_list():
    def factory(context_key):
        arms = ["a", "b", "c"] if context_key == "code:short" else ["a", "b"]
        return EpsilonGreedyBandit(arms, epsilon=0.0)

    contextual = ContextualBandit(factory, arms=["a", "b"])

    assert contextual.arms_for("code:short") == ["a", "b", "c"]
    assert contextual.arms_for("writing:long") == ["a", "b"]
