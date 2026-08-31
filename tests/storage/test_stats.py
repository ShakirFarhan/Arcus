from sqlmodel import SQLModel, create_engine

from arcus.routing.reward import COST_SCORES
from arcus.storage.db import log_request
from arcus.storage.stats import aggregate_by_arm_and_mode


def _in_memory_engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


def test_aggregates_correctly_across_models_and_modes():
    engine = _in_memory_engine()

    log_request(
        prompt="a", task_type="code", length_bucket="short", model="gpt-oss-120b",
        mode="bandit", reward=0.8, latency_ms=100, engine=engine,
    )
    log_request(
        prompt="b", task_type="code", length_bucket="short", model="gpt-oss-120b",
        mode="bandit", reward=0.6, latency_ms=200, engine=engine,
    )
    log_request(
        prompt="c", task_type="code", length_bucket="short", model="gpt-oss-120b",
        mode="random", reward=0.3, latency_ms=500, engine=engine,
    )
    log_request(
        prompt="d", task_type="writing", length_bucket="long", model="Kimi-K3",
        mode="bandit", reward=0.9, latency_ms=1000, engine=engine,
    )

    summaries = {(s.model, s.mode): s for s in aggregate_by_arm_and_mode(engine)}

    gpt_bandit = summaries[("gpt-oss-120b", "bandit")]
    assert gpt_bandit.request_count == 2
    assert gpt_bandit.avg_reward == 0.7
    assert gpt_bandit.avg_latency_ms == 150
    assert gpt_bandit.cost_score == COST_SCORES["gpt-oss-120b"]

    gpt_random = summaries[("gpt-oss-120b", "random")]
    assert gpt_random.request_count == 1
    assert gpt_random.avg_reward == 0.3

    kimi_bandit = summaries[("Kimi-K3", "bandit")]
    assert kimi_bandit.request_count == 1
    assert kimi_bandit.avg_reward == 0.9


def test_missing_reward_or_latency_excluded_from_average_not_treated_as_zero():
    engine = _in_memory_engine()

    log_request(
        prompt="a", task_type="general", length_bucket="short", model="GLM-5.3",
        mode="bandit", reward=1.0, latency_ms=100, engine=engine,
    )
    # no reward or latency logged for this one, e.g. the request errored
    # out before either could be computed
    log_request(
        prompt="b", task_type="general", length_bucket="short", model="GLM-5.3",
        mode="bandit", engine=engine,
    )

    summaries = {(s.model, s.mode): s for s in aggregate_by_arm_and_mode(engine)}
    glm = summaries[("GLM-5.3", "bandit")]

    assert glm.request_count == 2
    assert glm.avg_reward == 1.0
    assert glm.avg_latency_ms == 100


def test_missing_mode_grouped_as_unknown():
    engine = _in_memory_engine()

    log_request(
        prompt="a", task_type="general", length_bucket="short", model="DeepSeek-V4-Flash",
        engine=engine,
    )

    summaries = aggregate_by_arm_and_mode(engine)
    assert len(summaries) == 1
    assert summaries[0].mode == "unknown"
