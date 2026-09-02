import math
import random
from typing import Callable, Protocol

import numpy as np


class Bandit(Protocol):
    arms: list[str]

    def select_arm(self, exclude: set[str] | None = None) -> str: ...
    def update(self, arm: str, reward: float) -> None: ...
    def propensity(self, arm: str, exclude: set[str] | None = None) -> float: ...


class _CountBasedBandit:
    """Shared bookkeeping for epsilon-greedy and UCB1. Both track a running
    per-arm pull count and reward sum, and both need every arm tried once
    before their selection formula is even defined (UCB1's confidence bound
    divides by pull count, epsilon-greedy has nothing to compare averages
    against on arm zero).
    """

    def __init__(self, arms: list[str]) -> None:
        self.arms = list(arms)
        self._pulls = {arm: 0 for arm in self.arms}
        self._reward_sums = {arm: 0.0 for arm in self.arms}

    def _candidates(self, exclude: set[str] | None) -> list[str]:
        if not exclude:
            return self.arms
        return [arm for arm in self.arms if arm not in exclude]

    def _untried_arm(self, exclude: set[str] | None) -> str | None:
        for arm in self._candidates(exclude):
            if self._pulls[arm] == 0:
                return arm
        return None

    def _mean_reward(self, arm: str) -> float:
        return self._reward_sums[arm] / self._pulls[arm]

    def update(self, arm: str, reward: float) -> None:
        self._pulls[arm] += 1
        self._reward_sums[arm] += reward

    def _cold_start_propensity(self, arm: str, exclude: set[str] | None) -> float | None:
        # returns None once every candidate arm has at least one pull, so
        # the caller knows to fall through to its own formula. while an
        # arm is still untried, select_arm() always returns the first
        # untried one it finds, so the propensity for that specific arm
        # is 1.0 and everything else is 0, not a probability distribution
        # spread across the untried arms.
        untried = self._untried_arm(exclude)
        if untried is None:
            return None
        return 1.0 if arm == untried else 0.0


class EpsilonGreedyBandit(_CountBasedBandit):
    def __init__(self, arms: list[str], epsilon: float = 0.1) -> None:
        super().__init__(arms)
        self.epsilon = epsilon

    def select_arm(self, exclude: set[str] | None = None) -> str:
        untried = self._untried_arm(exclude)
        if untried is not None:
            return untried

        candidates = self._candidates(exclude)

        if random.random() < self.epsilon:
            return random.choice(candidates)

        best_mean = max(self._mean_reward(arm) for arm in candidates)
        best_arms = [arm for arm in candidates if self._mean_reward(arm) == best_mean]
        return random.choice(best_arms)

    def propensity(self, arm: str, exclude: set[str] | None = None) -> float:
        cold_start = self._cold_start_propensity(arm, exclude)
        if cold_start is not None:
            return cold_start

        candidates = self._candidates(exclude)
        if arm not in candidates:
            return 0.0

        best_mean = max(self._mean_reward(a) for a in candidates)
        best_arms = [a for a in candidates if self._mean_reward(a) == best_mean]

        # epsilon/n chance of landing here through the random branch, plus
        # a share of the (1 - epsilon) greedy branch if this arm is one of
        # the (possibly tied) best ones.
        base = self.epsilon / len(candidates)
        if arm in best_arms:
            return base + (1 - self.epsilon) / len(best_arms)
        return base


class UCB1Bandit(_CountBasedBandit):
    def select_arm(self, exclude: set[str] | None = None) -> str:
        untried = self._untried_arm(exclude)
        if untried is not None:
            return untried

        candidates = self._candidates(exclude)

        # textbook UCB1, no tunable exploration constant on purpose, that's
        # the whole point of this algorithm over epsilon-greedy: provable
        # logarithmic regret without anything to hand-tune.
        total_pulls = sum(self._pulls.values())

        def ucb_score(arm: str) -> float:
            return self._mean_reward(arm) + math.sqrt(2 * math.log(total_pulls) / self._pulls[arm])

        best_score = max(ucb_score(arm) for arm in candidates)
        best_arms = [arm for arm in candidates if ucb_score(arm) == best_score]
        return random.choice(best_arms)

    def propensity(self, arm: str, exclude: set[str] | None = None) -> float:
        cold_start = self._cold_start_propensity(arm, exclude)
        if cold_start is not None:
            return cold_start

        candidates = self._candidates(exclude)
        if arm not in candidates:
            return 0.0

        total_pulls = sum(self._pulls.values())

        def ucb_score(a: str) -> float:
            return self._mean_reward(a) + math.sqrt(2 * math.log(total_pulls) / self._pulls[a])

        best_score = max(ucb_score(a) for a in candidates)
        best_arms = [a for a in candidates if ucb_score(a) == best_score]
        # UCB1 is deterministic except for ties, so the propensity is just
        # 1 over however many arms are tied for the top score.
        return 1 / len(best_arms) if arm in best_arms else 0.0


_THOMPSON_PROPENSITY_SAMPLES = 2000


class ThompsonSamplingBandit:
    def __init__(self, arms: list[str]) -> None:
        self.arms = list(arms)
        # Beta(1, 1) is a uniform prior. no cold-start branch needed here
        # like the other two, a fresh arm just has wide uncertainty and
        # naturally gets sampled a lot until its posterior narrows, that's
        # Thompson sampling's actual advantage over the count-based ones.
        self._alpha = {arm: 1.0 for arm in self.arms}
        self._beta = {arm: 1.0 for arm in self.arms}

    def select_arm(self, exclude: set[str] | None = None) -> str:
        candidates = [arm for arm in self.arms if not exclude or arm not in exclude]
        samples = {arm: np.random.beta(self._alpha[arm], self._beta[arm]) for arm in candidates}
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: float) -> None:
        # standard Beta-Bernoulli Thompson sampling assumes 0/1 rewards.
        # ours are continuous in [0, 1], so the reward itself gets treated
        # as a fractional pseudo-observation rather than a hard success or
        # failure. this is the usual way to stretch Beta-Bernoulli to
        # continuous rewards in that range.
        self._alpha[arm] += reward
        self._beta[arm] += 1 - reward

    def propensity(self, arm: str, exclude: set[str] | None = None) -> float:
        candidates = [a for a in self.arms if not exclude or a not in exclude]
        if arm not in candidates:
            return 0.0

        # there's no closed form for the probability that one independent
        # Beta draw beats a handful of others, so this estimates it the
        # same way select_arm() actually decides: draw a batch of samples
        # per arm and see how often this arm comes out on top. this is a
        # Monte Carlo estimate, not an exact value, that's expected and
        # documented, not a bug.
        draws = np.stack(
            [np.random.beta(self._alpha[a], self._beta[a], size=_THOMPSON_PROPENSITY_SAMPLES) for a in candidates]
        )
        winners = draws.argmax(axis=0)
        return float((winners == candidates.index(arm)).mean())


class RandomBandit(_CountBasedBandit):
    """Uniform random arm selection, ignores everything it's learned. This
    is the 'B' side of the A/B mode flag: hand this to a
    ContextualBandit instead of one of the real algorithms and you get a
    random-routing baseline to compare the real bandits against. update()
    still records pulls and reward sums like the other count-based
    bandits do, not because selection uses them, but so stats aggregation
    can report what random selection would have earned per arm.
    """

    def select_arm(self, exclude: set[str] | None = None) -> str:
        return random.choice(self._candidates(exclude))

    def propensity(self, arm: str, exclude: set[str] | None = None) -> float:
        candidates = self._candidates(exclude)
        return 1 / len(candidates) if arm in candidates else 0.0


class ContextualBandit:
    """One independent bandit instance per context bucket, the 'disjoint'
    contextual bandit approach, rather than one bandit shared across
    every kind of request. Instances are created lazily the first time a
    given context key shows up.

    algorithm_factory takes the context_key being resolved. Most callers
    build a bandit for one fixed arm list and just ignore the argument,
    but some contexts (code, math, long-document, when reasoning-effort
    variants are enabled) legitimately route across a bigger arm set
    than others, and the factory needs to know which context it's
    building for to decide that.
    """

    def __init__(self, algorithm_factory: Callable[[str], Bandit], arms: list[str]) -> None:
        self._algorithm_factory = algorithm_factory
        self.arms = list(arms)
        self._bandits: dict[str, Bandit] = {}

    def _get_bandit(self, context_key: str) -> Bandit:
        if context_key not in self._bandits:
            self._bandits[context_key] = self._algorithm_factory(context_key)
        return self._bandits[context_key]

    def arms_for(self, context_key: str) -> list[str]:
        """The arm list actually in play for this specific context, not
        just the nominal `arms` this instance was constructed with,
        those two can differ once a context-dependent factory is in
        use. Callers that need to know how many real fallback options
        exist for the context they're routing right now (the quality
        gate's retry budget, warm-start's stale-arm filter) should ask
        here, not read `.arms` directly.
        """
        return self._get_bandit(context_key).arms

    def select_arm(self, context_key: str, exclude: set[str] | None = None) -> str:
        return self._get_bandit(context_key).select_arm(exclude=exclude)

    def update(self, context_key: str, arm: str, reward: float) -> None:
        self._get_bandit(context_key).update(arm, reward)

    def propensity(self, context_key: str, arm: str, exclude: set[str] | None = None) -> float:
        return self._get_bandit(context_key).propensity(arm, exclude=exclude)
