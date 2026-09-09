from dataclasses import dataclass


def normalize_latency(latency_ms: float, ceiling_ms: float = 20_000) -> float:
    # past the ceiling, slower is just uniformly bad, no reason to keep
    # penalizing harder. 20s is a reasonable worst-case wait for an
    # interactive CLI tool.
    return max(0.0, min(1.0, 1 - latency_ms / ceiling_ms))


@dataclass(frozen=True)
class RewardWeights:
    quality: float
    latency: float


# quality weighted higher than latency: a fast wrong answer is worth
# about nothing. latency still matters because it's the cost a user
# actually pays on shared, sometimes-contended infrastructure.
#
# an earlier version of this carried a third term, a cost score derived
# from other providers' published hosting rates for these same
# open-weight models. it was removed on purpose. ARC is free, nobody is
# billed those rates, and a fifth of every routing decision was being
# driven by a number that described a hypothetical deployment rather
# than this one. both terms here are measured from the request that
# actually happened.
DEFAULT_WEIGHTS = RewardWeights(quality=0.6, latency=0.4)


def compute_reward(
    latency_ms: float,
    model: str,
    quality_score: float,
    weights: RewardWeights = DEFAULT_WEIGHTS,
) -> float:
    """Scores one completed request in [0, 1].

    `model` is accepted but unused now that cost is gone. It stays in the
    signature because every call site passes it and because a future
    term keyed on the model (observed contention, say) would need it
    back, and a signature change would be the least interesting part of
    adding one.
    """
    weight_sum = weights.quality + weights.latency
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"reward weights must sum to 1.0, got {weight_sum}")

    return weights.quality * quality_score + weights.latency * normalize_latency(latency_ms)
