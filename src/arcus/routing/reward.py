from dataclasses import dataclass

# published commercial hosting rates for these open-weight models, one
# primary source per model, as of August 2026. this is a SIMULATED cost
# signal, ARC itself is free to the user, nobody's actually being
# billed these numbers. rates are dollars per 1M tokens.
#
# gpt-oss-120b: Groq
# GLM-5.3: Zhipu/Z.ai official pricing
# Kimi-K3: Moonshot AI official pricing (cache-miss rate, ignoring the
#   cache-hit discount, more precision than a simulated signal needs)
# DeepSeek-V4-Flash: DeepSeek official pricing (off-peak rate, ignoring
#   the peak-hours surcharge for the same reason)
MODEL_HOSTING_RATES: dict[str, tuple[float, float]] = {
    "gpt-oss-120b": (0.15, 0.60),
    "GLM-5.3": (1.40, 4.40),
    "Kimi-K3": (3.00, 15.00),
    "DeepSeek-V4-Flash": (0.22, 0.66),
}


def _blended_rate(model: str) -> float:
    input_rate, output_rate = MODEL_HOSTING_RATES[model]
    return (input_rate + output_rate) / 2


# computed from the table above rather than hardcoded, so the raw rates
# stay the actual source of truth. cheapest model scores 1.0, priciest
# scores 0.0, everything else falls in between.
def _compute_cost_scores() -> dict[str, float]:
    blended = {model: _blended_rate(model) for model in MODEL_HOSTING_RATES}
    cheapest = min(blended.values())
    priciest = max(blended.values())
    spread = priciest - cheapest

    return {model: 1 - (rate - cheapest) / spread for model, rate in blended.items()}


COST_SCORES = _compute_cost_scores()

# a neutral, middle-of-the-road cost score for a model with no published
# rate in the table above. Right now that's the web-search
# "legacy-tool-calling" model variants, those are an ARC-side serving
# mode rather than a separately priced product, so there's no real rate
# to look up, this keeps reward computation from crashing instead of
# asserting a pricing figure nobody's confirmed.
_UNKNOWN_MODEL_COST_SCORE = 0.5


def normalize_latency(latency_ms: float, ceiling_ms: float = 20_000) -> float:
    # past the ceiling, slower is just uniformly bad, no reason to keep
    # penalizing harder. 20s is a reasonable worst-case wait for an
    # interactive CLI tool.
    return max(0.0, min(1.0, 1 - latency_ms / ceiling_ms))


def basic_quality_score(finish_reason: str | None, content: str | None) -> float:
    """A minimal quality signal: finished cleanly (finish_reason == "stop")
    and non-empty content. The real request pipeline uses the fuller
    checks in quality/gate.py instead.
    """
    if finish_reason != "stop":
        return 0.0
    if not content or not content.strip():
        return 0.0
    return 1.0


@dataclass(frozen=True)
class RewardWeights:
    quality: float
    latency: float
    cost: float


# quality weighted highest: a fast, cheap, wrong answer is worth about
# nothing. latency next: it's the most immediately felt cost on shared,
# sometimes-contended infra, where quality can degrade during peak load.
# cost lowest: ARC is free to the user, this dimension exists to
# demonstrate routing that accounts for cost as an engineering practice,
# not because real budget pressure exists here.
DEFAULT_WEIGHTS = RewardWeights(quality=0.5, latency=0.3, cost=0.2)


def compute_reward(
    latency_ms: float,
    model: str,
    quality_score: float,
    weights: RewardWeights = DEFAULT_WEIGHTS,
) -> float:
    weight_sum = weights.quality + weights.latency + weights.cost
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"reward weights must sum to 1.0, got {weight_sum}")

    return (
        weights.quality * quality_score
        + weights.latency * normalize_latency(latency_ms)
        + weights.cost * COST_SCORES.get(model, _UNKNOWN_MODEL_COST_SCORE)
    )
