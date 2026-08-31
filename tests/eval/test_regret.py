from arcus.eval.regret import (
    ArmDistribution,
    DEFAULT_ARM_DISTRIBUTIONS,
    run_regret_benchmark,
    simulate_regret,
)
from arcus.routing.bandit import EpsilonGreedyBandit, RandomBandit, ThompsonSamplingBandit, UCB1Bandit

ARMS = list(DEFAULT_ARM_DISTRIBUTIONS.keys())


def test_regret_curve_is_non_decreasing():
    curve = simulate_regret(lambda: EpsilonGreedyBandit(ARMS), DEFAULT_ARM_DISTRIBUTIONS, n_rounds=200, seed=1)

    for earlier, later in zip(curve, curve[1:]):
        assert later >= earlier


def test_regret_curve_length_matches_rounds():
    curve = simulate_regret(lambda: UCB1Bandit(ARMS), DEFAULT_ARM_DISTRIBUTIONS, n_rounds=150, seed=1)
    assert len(curve) == 150


def test_simulate_regret_is_reproducible_with_same_seed():
    first = simulate_regret(lambda: ThompsonSamplingBandit(ARMS), DEFAULT_ARM_DISTRIBUTIONS, n_rounds=100, seed=9)
    second = simulate_regret(lambda: ThompsonSamplingBandit(ARMS), DEFAULT_ARM_DISTRIBUTIONS, n_rounds=100, seed=9)
    assert first == second


def test_real_algorithms_beat_random_baseline_over_enough_rounds():
    results = run_regret_benchmark(n_rounds=3000, seed=42)

    random_final_regret = results["random"][-1]

    for name in ["epsilon_greedy", "ucb1", "thompson"]:
        assert results[name][-1] < random_final_regret


def test_arm_with_zero_spread_gives_zero_regret_for_any_policy():
    flat_distributions = {"a": ArmDistribution(mean=0.5, std=0.0), "b": ArmDistribution(mean=0.5, std=0.0)}
    curve = simulate_regret(lambda: RandomBandit(["a", "b"]), flat_distributions, n_rounds=50, seed=1)
    assert curve[-1] == 0.0
