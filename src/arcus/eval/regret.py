import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

from arcus.adapters.arc_adapter import ArcModel
from arcus.routing.bandit import Bandit, EpsilonGreedyBandit, RandomBandit, ThompsonSamplingBandit, UCB1Bandit

ARMS = [m.value for m in ArcModel]


@dataclass(frozen=True)
class ArmDistribution:
    mean: float
    std: float


# illustrative synthetic reward distributions, not measured. regret
# benchmarking needs a known ground-truth mean per arm to compute regret
# against, and that's exactly the thing live traffic can never give us:
# a real request only ever explores one arm per round, so we'd have no
# way to know what the *other* three would have scored that round. this
# is the standard way to study a bandit algorithm's exploration behavior
# in isolation from actual model quality, not a stand-in for missing
# data.
DEFAULT_ARM_DISTRIBUTIONS: dict[str, ArmDistribution] = {
    "gpt-oss-120b": ArmDistribution(mean=0.72, std=0.10),
    "GLM-5.3": ArmDistribution(mean=0.68, std=0.12),
    "Kimi-K3": ArmDistribution(mean=0.75, std=0.08),
    "DeepSeek-V4-Flash": ArmDistribution(mean=0.70, std=0.15),
}


def _sample_reward(dist: ArmDistribution, rng: random.Random) -> float:
    # clipped to [0, 1], real compute_reward() output is bounded there too
    return min(1.0, max(0.0, rng.gauss(dist.mean, dist.std)))


def simulate_regret(
    bandit_factory: Callable[[], Bandit],
    arm_distributions: dict[str, ArmDistribution],
    n_rounds: int,
    seed: int,
) -> list[float]:
    """Runs one bandit for n_rounds against a known synthetic reward
    environment and returns its cumulative regret curve. Regret per
    round is the gap between the best arm's true mean and the pulled
    arm's true mean, not the noisy realized reward, so the curve tracks
    the algorithm's actual exploration cost instead of being dominated
    by sampling noise.
    """
    # Thompson sampling draws from numpy's global RNG (see bandit.py),
    # the other algorithms only touch the stdlib random module, seeding
    # both keeps every algorithm's run reproducible regardless of which
    # one is being simulated.
    rng = random.Random(seed)
    np.random.seed(seed)

    bandit = bandit_factory()
    best_mean = max(dist.mean for dist in arm_distributions.values())

    cumulative_regret = 0.0
    curve = []
    for _ in range(n_rounds):
        arm = bandit.select_arm()
        reward = _sample_reward(arm_distributions[arm], rng)
        bandit.update(arm, reward)

        cumulative_regret += best_mean - arm_distributions[arm].mean
        curve.append(cumulative_regret)

    return curve


def run_regret_benchmark(
    arm_distributions: dict[str, ArmDistribution] = DEFAULT_ARM_DISTRIBUTIONS,
    n_rounds: int = 2000,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Cumulative regret curves for all three real algorithms plus the
    random baseline, demonstrating this is a real algorithm family and
    not just "a bandit works."
    """
    arms = list(arm_distributions.keys())
    factories: dict[str, Callable[[], Bandit]] = {
        "epsilon_greedy": lambda: EpsilonGreedyBandit(arms),
        "ucb1": lambda: UCB1Bandit(arms),
        "thompson": lambda: ThompsonSamplingBandit(arms),
        "random": lambda: RandomBandit(arms),
    }

    return {
        name: simulate_regret(factory, arm_distributions, n_rounds, seed)
        for name, factory in factories.items()
    }
