from sqlmodel import Session, SQLModel, create_engine, select

import pytest

from arcus.routing.bandit import (
    ContextualBandit,
    EpsilonGreedyBandit,
    RandomBandit,
    ThompsonSamplingBandit,
    UCB1Bandit,
)
from arcus.routing.warm_start import replay_history
from arcus.storage.db import RequestLog, log_request

ARMS = ["gpt-oss-120b", "GLM-5.3"]


def _in_memory_engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


def _factory(context_key):
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


def test_replay_history_ignores_rows_from_an_older_reward_generation():
    engine = _in_memory_engine()

    log_request(
        prompt="p1", task_type="code", length_bucket="short", model="gpt-oss-120b",
        mode="bandit", reward=1.0, engine=engine,
    )

    # rewrite it as a pre-grading row, the way an upgraded database
    # looks: quality was pass/fail then, so a surviving response scored
    # a flat 1.0 where a graded one rarely would
    with Session(engine) as session:
        row = session.exec(select(RequestLog)).one()
        row.reward_version = 1
        session.add(row)
        session.commit()

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


def test_replay_history_skips_rows_for_a_retired_model():
    engine = _in_memory_engine()

    # a model that was live when this row got logged but isn't one of
    # the bandit's current arms anymore (ARC renamed or dropped it)
    log_request(
        prompt="p1", task_type="code", length_bucket="short", model="some-old-retired-model",
        mode="bandit", reward=0.9, engine=engine,
    )
    log_request(
        prompt="p2", task_type="code", length_bucket="short", model="gpt-oss-120b",
        mode="bandit", reward=0.7, engine=engine,
    )

    bandit = ContextualBandit(_factory, arms=ARMS)
    replay_history(bandit, engine, mode="bandit")  # should not raise

    underlying = bandit._get_bandit("code:short")
    assert underlying._pulls == {"gpt-oss-120b": 1, "GLM-5.3": 0}
    assert underlying._reward_sums["gpt-oss-120b"] == 0.7


def test_replay_history_checks_the_stale_arm_filter_per_context():
    # an arm that's valid for one context but not another (a reasoning-
    # effort variant only offered for code/math contexts, say) should
    # replay into the context where it's valid and get skipped, not
    # crash, in the context where it isn't
    engine = _in_memory_engine()

    log_request(
        prompt="p1", task_type="code", length_bucket="short", model="only-valid-for-code",
        mode="bandit", reward=0.9, engine=engine,
    )
    log_request(
        prompt="p2", task_type="writing", length_bucket="long", model="only-valid-for-code",
        mode="bandit", reward=0.1, engine=engine,
    )

    def factory(context_key):
        arms = [*ARMS, "only-valid-for-code"] if context_key == "code:short" else ARMS
        return EpsilonGreedyBandit(arms, epsilon=0.0)

    bandit = ContextualBandit(factory, arms=ARMS)
    replay_history(bandit, engine, mode="bandit")  # should not raise

    code_bandit = bandit._get_bandit("code:short")
    writing_bandit = bandit._get_bandit("writing:long")

    assert code_bandit._pulls["only-valid-for-code"] == 1
    assert "only-valid-for-code" not in writing_bandit._pulls


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


def _replay_row_by_row(bandit, engine, mode):
    """The original implementation, kept as the reference the aggregated
    one is checked against.
    """
    from arcus.storage.db import REWARD_VERSION

    with Session(engine) as session:
        rows = session.exec(
            select(RequestLog)
            .where(RequestLog.mode == mode)
            .where(RequestLog.reward.is_not(None))
            .where(RequestLog.reward_version == REWARD_VERSION)
            .order_by(RequestLog.created_at)
        ).all()
    for row in rows:
        context_key = f"{row.task_type}:{row.length_bucket}"
        if row.model in bandit.arms_for(context_key):
            bandit.update(context_key, row.model, row.reward)


@pytest.mark.parametrize(
    "algorithm",
    [EpsilonGreedyBandit, UCB1Bandit, RandomBandit, ThompsonSamplingBandit],
)
def test_aggregated_replay_matches_row_by_row_exactly(algorithm):
    """Warm start sums the history in SQL instead of applying it one row
    at a time. That's only legitimate because every algorithm's state is
    a pure function of (pull count, reward total) per arm, so this pins
    the equivalence down rather than trusting the argument.
    """
    import random

    engine = _in_memory_engine()
    random.seed(11)
    tasks = ["code", "writing", "general"]
    lengths = ["short", "long"]

    for i in range(400):
        log_request(
            prompt=f"q{i}",
            task_type=random.choice(tasks),
            length_bucket=random.choice(lengths),
            model=random.choice(ARMS),
            mode="bandit",
            reward=round(random.random(), 6),
            engine=engine,
        )

    def build():
        return ContextualBandit(lambda _context_key: algorithm(ARMS), arms=ARMS)

    reference, aggregated = build(), build()
    _replay_row_by_row(reference, engine, "bandit")
    replay_history(aggregated, engine, "bandit")

    contexts = sorted(set(reference._bandits) | set(aggregated._bandits))
    assert contexts, "the fixture should have produced several contexts"

    for context_key in contexts:
        a = reference._get_bandit(context_key)
        b = aggregated._get_bandit(context_key)
        if isinstance(a, ThompsonSamplingBandit):
            assert a._alpha == pytest.approx(b._alpha)
            assert a._beta == pytest.approx(b._beta)
        else:
            assert a._pulls == b._pulls
            assert a._reward_sums == pytest.approx(b._reward_sums)


def test_restore_reproduces_what_repeated_updates_would_have_built():
    rewards = [0.2, 0.9, 0.55, 0.1]

    updated = EpsilonGreedyBandit(ARMS)
    for r in rewards:
        updated.update("gpt-oss-120b", r)

    restored = EpsilonGreedyBandit(ARMS)
    restored.restore("gpt-oss-120b", len(rewards), sum(rewards))

    assert updated._pulls == restored._pulls
    assert updated._reward_sums == pytest.approx(restored._reward_sums)


def test_thompson_restore_reproduces_its_posterior():
    rewards = [0.3, 0.8, 0.65]

    updated = ThompsonSamplingBandit(ARMS)
    for r in rewards:
        updated.update("GLM-5.3", r)

    restored = ThompsonSamplingBandit(ARMS)
    restored.restore("GLM-5.3", len(rewards), sum(rewards))

    assert updated._alpha == pytest.approx(restored._alpha)
    assert updated._beta == pytest.approx(restored._beta)
