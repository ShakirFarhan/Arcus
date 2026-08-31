from sqlmodel import SQLModel, create_engine

from arcus.routing.bandit import ContextualBandit, EpsilonGreedyBandit
from arcus.routing.warm_start import replay_history
from arcus.storage.db import log_request

ARMS = ["gpt-oss-120b", "GLM-5.3"]


def _in_memory_engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


def _factory():
    return EpsilonGreedyBandit(ARMS, epsilon=0.0)


def test_replay_history_reconstructs_pull_counts():
    engine = _in_memory_engine()

    log_request(
        prompt="p1", task_type="code", length_bucket="short", model="gpt-oss-120b",
        mode="bandit", reward=0.8, engine=engine,
    )
    log_request(
        prompt="p2", task_type="code", length_bucket="short", model="gpt-oss-120b",
        mode="bandit", reward=0.6, engine=engine,
    )
    log_request(
        prompt="p3", task_type="code", length_bucket="short", model="GLM-5.3",
        mode="bandit", reward=0.2, engine=engine,
    )

    bandit = ContextualBandit(_factory, arms=ARMS)
    replay_history(bandit, engine, mode="bandit")

    underlying = bandit._get_bandit("code:short")
    assert underlying._pulls == {"gpt-oss-120b": 2, "GLM-5.3": 1}
    assert underlying._reward_sums["gpt-oss-120b"] == 0.8 + 0.6
    assert underlying._reward_sums["GLM-5.3"] == 0.2


def test_replay_history_ignores_rows_with_no_reward():
    engine = _in_memory_engine()

    # a request that errored out before a reward could be computed
    log_request(
        prompt="p1", task_type="code", length_bucket="short", model="gpt-oss-120b",
        mode="bandit", reward=None, engine=engine,
    )

    bandit = ContextualBandit(_factory, arms=ARMS)
    replay_history(bandit, engine, mode="bandit")

    underlying = bandit._get_bandit("code:short")
    assert underlying._pulls == {"gpt-oss-120b": 0, "GLM-5.3": 0}


def test_replay_history_ignores_other_modes():
    engine = _in_memory_engine()

    log_request(
        prompt="p1", task_type="code", length_bucket="short", model="gpt-oss-120b",
        mode="random", reward=0.9, engine=engine,
    )

    bandit = ContextualBandit(_factory, arms=ARMS)
    replay_history(bandit, engine, mode="bandit")

    underlying = bandit._get_bandit("code:short")
    assert underlying._pulls == {"gpt-oss-120b": 0, "GLM-5.3": 0}


def test_replay_history_keeps_contexts_separate():
    engine = _in_memory_engine()

    log_request(
        prompt="p1", task_type="code", length_bucket="short", model="gpt-oss-120b",
        mode="bandit", reward=0.8, engine=engine,
    )
    log_request(
        prompt="p2", task_type="writing", length_bucket="long", model="GLM-5.3",
        mode="bandit", reward=0.5, engine=engine,
    )

    bandit = ContextualBandit(_factory, arms=ARMS)
    replay_history(bandit, engine, mode="bandit")

    code_bandit = bandit._get_bandit("code:short")
    writing_bandit = bandit._get_bandit("writing:long")

    assert code_bandit._pulls == {"gpt-oss-120b": 1, "GLM-5.3": 0}
    assert writing_bandit._pulls == {"gpt-oss-120b": 0, "GLM-5.3": 1}
