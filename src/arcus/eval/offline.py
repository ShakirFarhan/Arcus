import random
from dataclasses import dataclass
from statistics import mean
from typing import Callable

from sqlmodel import Session, select

from arcus.storage.db import RequestLog

Policy = Callable[[str], str]


@dataclass(frozen=True)
class LoggedExample:
    context_key: str
    arm: str
    propensity: float
    reward: float


def load_logged_examples(engine, mode: str = "bandit") -> list[LoggedExample]:
    """Pulls rows out of the request log that are actually usable for
    offline evaluation: propensity has to be set (the whole point of
    logging it from day one) and reward has to be set (a row that errored
    out before a reward was computed can't tell an estimator anything).
    """
    with Session(engine) as session:
        rows = session.exec(
            select(RequestLog)
            .where(RequestLog.mode == mode)
            .where(RequestLog.propensity.is_not(None))
            .where(RequestLog.reward.is_not(None))
        ).all()

    return [
        LoggedExample(
            context_key=f"{row.task_type}:{row.length_bucket}",
            arm=row.model,
            propensity=row.propensity,
            reward=row.reward,
        )
        for row in rows
    ]


def policy_always(arm: str) -> Policy:
    return lambda context_key: arm


def greedy_policy_from_log(examples: list[LoggedExample]) -> Policy:
    """Builds a deterministic policy that, for each context bucket, picks
    whichever arm had the best average logged reward. Fit and evaluated
    on the same data (no held-out split), so its reported value will run
    a little optimistic, this is a known limitation of the simple
    version of this technique, not a bug. A proper cross-fitted version
    would split the log before fitting.
    """
    sums: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    for ex in examples:
        key = (ex.context_key, ex.arm)
        sums[key] = sums.get(key, 0.0) + ex.reward
        counts[key] = counts.get(key, 0) + 1

    best_arm_per_context: dict[str, str] = {}
    for (context_key, arm), total in sums.items():
        avg = total / counts[(context_key, arm)]
        current_best = best_arm_per_context.get(context_key)
        if current_best is None or avg > sums[(context_key, current_best)] / counts[(context_key, current_best)]:
            best_arm_per_context[context_key] = arm

    fallback_arm = max(counts, key=counts.get)[1] if counts else None

    def policy(context_key: str) -> str:
        return best_arm_per_context.get(context_key, fallback_arm)

    return policy


def ips_estimate(examples: list[LoggedExample], policy: Policy) -> float:
    """Inverse propensity scoring. For each logged row, if the target
    policy would have picked the same arm the logging policy actually
    picked, that row's reward counts, reweighted by how unlikely the
    logging policy was to pick it (rarer picks get boosted more, that's
    what corrects for the logging policy's own selection bias). Rows
    where the policies disagree contribute nothing, we simply never
    observed what would have happened.
    """
    if not examples:
        return 0.0

    weighted = [
        ex.reward / ex.propensity if policy(ex.context_key) == ex.arm else 0.0
        for ex in examples
    ]
    return mean(weighted)


def _fit_reward_model(examples: list[LoggedExample]) -> dict[tuple[str, str], float]:
    groups: dict[tuple[str, str], list[float]] = {}
    for ex in examples:
        groups.setdefault((ex.context_key, ex.arm), []).append(ex.reward)
    return {key: mean(rewards) for key, rewards in groups.items()}


def dr_estimate(
    examples: list[LoggedExample],
    policy: Policy,
    reward_model: dict[tuple[str, str], float] | None = None,
) -> float:
    """Doubly robust estimator: a direct-method estimate (the reward
    model's guess at what the target policy would earn) plus an IPS
    correction term that only kicks in on rows where the logged arm
    happens to match the target policy. Stays unbiased if either the
    reward model or the propensities are right, not both, that's the
    "doubly" part. Same same-data-fit caveat as greedy_policy_from_log
    applies to the default reward model here.
    """
    if not examples:
        return 0.0

    reward_model = reward_model if reward_model is not None else _fit_reward_model(examples)
    overall_mean = mean(ex.reward for ex in examples)

    total = 0.0
    for ex in examples:
        target_arm = policy(ex.context_key)
        q_target = reward_model.get((ex.context_key, target_arm), overall_mean)

        if target_arm == ex.arm:
            q_logged = reward_model.get((ex.context_key, ex.arm), overall_mean)
            correction = (ex.reward - q_logged) / ex.propensity
        else:
            correction = 0.0

        total += q_target + correction

    return total / len(examples)


def bootstrap_ci(
    examples: list[LoggedExample],
    estimator: Callable[[list[LoggedExample]], float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap: resample the log with replacement a bunch of
    times, run the estimator on each resample, and read the confidence
    interval off the sorted results. Plain stdlib random, no scipy, the
    log sizes here don't need anything fancier.
    """
    point = estimator(examples)
    if not examples:
        return point, point, point

    rng = random.Random(seed)
    n = len(examples)
    resampled = [estimator([examples[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples)]
    resampled.sort()

    lower_idx = int((1 - confidence) / 2 * n_resamples)
    upper_idx = min(int((1 + confidence) / 2 * n_resamples), n_resamples - 1)

    return point, resampled[lower_idx], resampled[upper_idx]


@dataclass(frozen=True)
class PolicyEvaluation:
    name: str
    ips_estimate: float
    ips_ci: tuple[float, float]
    dr_estimate: float
    dr_ci: tuple[float, float]


def evaluate_policies(
    engine,
    policies: dict[str, Policy],
    mode: str = "bandit",
    n_resamples: int = 1000,
    seed: int | None = None,
) -> list[PolicyEvaluation]:
    """Compares the logged policy against offline-estimated alternatives,
    each with a confidence interval.
    """
    examples = load_logged_examples(engine, mode=mode)

    logged_point, logged_low, logged_high = bootstrap_ci(
        examples, lambda ex: mean(e.reward for e in ex) if ex else 0.0, n_resamples, seed=seed
    )
    rows = [
        PolicyEvaluation(
            name="logged (as-run)",
            ips_estimate=logged_point,
            ips_ci=(logged_low, logged_high),
            dr_estimate=logged_point,
            dr_ci=(logged_low, logged_high),
        )
    ]

    for name, policy in policies.items():
        ips_point, ips_low, ips_high = bootstrap_ci(
            examples, lambda ex, p=policy: ips_estimate(ex, p), n_resamples, seed=seed
        )
        dr_point, dr_low, dr_high = bootstrap_ci(
            examples, lambda ex, p=policy: dr_estimate(ex, p), n_resamples, seed=seed
        )
        rows.append(
            PolicyEvaluation(
                name=name,
                ips_estimate=ips_point,
                ips_ci=(ips_low, ips_high),
                dr_estimate=dr_point,
                dr_ci=(dr_low, dr_high),
            )
        )

    return rows
