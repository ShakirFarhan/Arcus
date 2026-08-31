from statistics import mean

import pytest
from sqlmodel import SQLModel, create_engine

from arcus.eval.offline import (
    LoggedExample,
    bootstrap_ci,
    dr_estimate,
    evaluate_policies,
    greedy_policy_from_log,
    ips_estimate,
    load_logged_examples,
    policy_always,
)
from arcus.storage.db import log_request


def _in_memory_engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


def test_ips_estimate_hand_computed():
    examples = [
        LoggedExample(context_key="ctx", arm="A", propensity=0.5, reward=1.0),
        LoggedExample(context_key="ctx", arm="B", propensity=0.5, reward=0.0),
    ]
    # row 1 matches the target policy, weight = 1.0 / 0.5 = 2.0
    # row 2 doesn't match, contributes 0
    # mean(2.0, 0.0) = 1.0
    assert ips_estimate(examples, policy_always("A")) == pytest.approx(1.0)


def test_dr_estimate_hand_computed_when_reward_model_fits_perfectly():
    examples = [
        LoggedExample(context_key="ctx", arm="A", propensity=0.5, reward=1.0),
        LoggedExample(context_key="ctx", arm="B", propensity=0.5, reward=0.0),
    ]
    # reward model exactly matches each observed reward here, so the IPS
    # correction term is zero on every row and DR collapses to the
    # direct-method estimate, 1.0 both rows
    assert dr_estimate(examples, policy_always("A")) == pytest.approx(1.0)


def test_dr_estimate_hand_computed_with_residual():
    examples = [
        LoggedExample(context_key="ctx", arm="A", propensity=0.5, reward=1.0),
        LoggedExample(context_key="ctx", arm="A", propensity=0.5, reward=0.0),
        LoggedExample(context_key="ctx", arm="B", propensity=0.5, reward=0.0),
    ]
    # reward model: q(ctx, A) = 0.5, q(ctx, B) = 0.0
    # row1: 0.5 + (1.0 - 0.5)/0.5 = 1.5
    # row2: 0.5 + (0.0 - 0.5)/0.5 = -0.5
    # row3: 0.5 + 0 (target policy A != logged arm B) = 0.5
    # mean(1.5, -0.5, 0.5) = 0.5
    assert dr_estimate(examples, policy_always("A")) == pytest.approx(0.5)


def test_dr_reduces_to_ips_when_reward_model_is_zero():
    examples = [
        LoggedExample(context_key="ctx", arm="A", propensity=0.4, reward=0.9),
        LoggedExample(context_key="ctx", arm="B", propensity=0.6, reward=0.2),
    ]
    # with q_hat == 0 everywhere, DR's direct-method term drops out and
    # the correction term becomes exactly reward/propensity on a match,
    # zero otherwise, that's the IPS formula itself. a real identity
    # between the two estimators, not just a coincidence on this data.
    zero_model = {("ctx", "A"): 0.0, ("ctx", "B"): 0.0}
    assert dr_estimate(examples, policy_always("A"), reward_model=zero_model) == pytest.approx(
        ips_estimate(examples, policy_always("A"))
    )


def test_ips_estimate_empty_log_is_zero():
    assert ips_estimate([], policy_always("A")) == 0.0


def test_greedy_policy_from_log_picks_best_average_arm_per_context():
    examples = [
        LoggedExample(context_key="code:short", arm="A", propensity=0.5, reward=0.9),
        LoggedExample(context_key="code:short", arm="B", propensity=0.5, reward=0.2),
        LoggedExample(context_key="writing:long", arm="A", propensity=0.5, reward=0.1),
        LoggedExample(context_key="writing:long", arm="B", propensity=0.5, reward=0.8),
    ]
    policy = greedy_policy_from_log(examples)

    assert policy("code:short") == "A"
    assert policy("writing:long") == "B"


def test_bootstrap_ci_is_exact_for_constant_values():
    examples = [LoggedExample(context_key="ctx", arm="A", propensity=1.0, reward=0.5) for _ in range(20)]
    estimator = lambda ex: mean(e.reward for e in ex)  # noqa: E731

    point, low, high = bootstrap_ci(examples, estimator, n_resamples=200, seed=1)

    assert point == pytest.approx(0.5)
    assert low == pytest.approx(0.5)
    assert high == pytest.approx(0.5)


def test_bootstrap_ci_brackets_the_point_estimate_for_varied_data():
    examples = [
        LoggedExample(context_key="ctx", arm="A", propensity=1.0, reward=r)
        for r in [0.1, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8, 0.4, 0.6, 1.0]
    ]
    estimator = lambda ex: mean(e.reward for e in ex)  # noqa: E731

    point, low, high = bootstrap_ci(examples, estimator, n_resamples=500, seed=7)

    assert low <= point <= high
    assert low < high


def test_bootstrap_ci_is_reproducible_with_same_seed():
    examples = [
        LoggedExample(context_key="ctx", arm="A", propensity=1.0, reward=r) for r in [0.1, 0.5, 0.9, 0.3]
    ]
    estimator = lambda ex: mean(e.reward for e in ex)  # noqa: E731

    first = bootstrap_ci(examples, estimator, n_resamples=100, seed=3)
    second = bootstrap_ci(examples, estimator, n_resamples=100, seed=3)

    assert first == second


def test_load_logged_examples_filters_missing_propensity_or_reward():
    engine = _in_memory_engine()

    log_request(
        prompt="a", task_type="code", length_bucket="short", model="gpt-oss-120b",
        mode="bandit", propensity=0.5, reward=0.8, engine=engine,
    )
    # no propensity, unusable for offline eval
    log_request(
        prompt="b", task_type="code", length_bucket="short", model="gpt-oss-120b",
        mode="bandit", propensity=None, reward=0.5, engine=engine,
    )
    # wrong mode, shouldn't be picked up when loading "bandit" rows
    log_request(
        prompt="c", task_type="code", length_bucket="short", model="gpt-oss-120b",
        mode="random", propensity=0.25, reward=0.5, engine=engine,
    )

    examples = load_logged_examples(engine, mode="bandit")

    assert len(examples) == 1
    assert examples[0].context_key == "code:short"
    assert examples[0].propensity == 0.5
    assert examples[0].reward == 0.8


def test_evaluate_policies_logged_row_matches_plain_mean_reward():
    engine = _in_memory_engine()

    for reward in [0.2, 0.6, 1.0]:
        log_request(
            prompt="p", task_type="code", length_bucket="short", model="gpt-oss-120b",
            mode="bandit", propensity=0.5, reward=reward, engine=engine,
        )

    results = evaluate_policies(
        engine, policies={"always gpt": policy_always("gpt-oss-120b")}, n_resamples=100, seed=5
    )

    logged_row = next(r for r in results if r.name == "logged (as-run)")
    assert logged_row.ips_estimate == pytest.approx(mean([0.2, 0.6, 1.0]))

    always_row = next(r for r in results if r.name == "always gpt")
    assert always_row.ips_estimate == pytest.approx(mean([r / 0.5 for r in [0.2, 0.6, 1.0]]))
