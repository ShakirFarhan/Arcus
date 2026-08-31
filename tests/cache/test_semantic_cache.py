from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import SQLModel, create_engine

from arcus.cache.semantic_cache import (
    CacheEntry,
    Volatility,
    classify_volatility,
    extract_params,
    lookup,
    params_conflict,
    store,
    ttl_seconds_for,
)


def _in_memory_engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.mark.parametrize(
    "query",
    [
        "what's the weather like today",
        "what is the current exchange rate for USD to EUR",
        "what's the latest version of python",
        "what's happening this week in the news",
    ],
)
def test_classify_volatility_flags_time_sensitive_queries(query):
    assert classify_volatility(query) == Volatility.VOLATILE


@pytest.mark.parametrize(
    "query",
    [
        "how does binary search work",
        "explain the difference between TCP and UDP",
        "what's the derivative of x^2",
    ],
)
def test_classify_volatility_allows_stable_queries(query):
    assert classify_volatility(query) == Volatility.STABLE


def test_ttl_seconds_for_volatile_is_zero():
    assert ttl_seconds_for(Volatility.VOLATILE) == 0


def test_ttl_seconds_for_stable_is_positive():
    assert ttl_seconds_for(Volatility.STABLE) > 0


def test_extract_params_finds_numbers_and_entities():
    params = extract_params("What's due for CS 3214 project 2 in France")
    assert "3214" in params
    assert "2" in params
    assert "France" in params


def test_params_conflict_true_for_different_numbers():
    assert params_conflict("when is project 2 due", "when is project 3 due")


def test_params_conflict_false_for_same_params():
    assert not params_conflict(
        "when is project 2 due for CS 3214", "what's the due date for CS 3214 project 2"
    )


def test_store_and_lookup_hit_on_near_identical_stable_query():
    engine = _in_memory_engine()
    store(
        "how does binary search work",
        "binary search splits a sorted array in half repeatedly",
        model="gpt-oss-120b",
        engine=engine,
    )

    result = lookup("explain how binary search works", engine=engine)

    assert result.hit
    assert "binary search" in result.response
    assert result.model == "gpt-oss-120b"


def test_lookup_misses_on_unrelated_query():
    engine = _in_memory_engine()
    store(
        "how does binary search work",
        "binary search splits a sorted array in half repeatedly",
        model="gpt-oss-120b",
        engine=engine,
    )

    result = lookup("what's a good recipe for banana bread", engine=engine)

    assert not result.hit


def test_lookup_misses_when_entry_expired_even_with_high_similarity():
    engine = _in_memory_engine()
    entry = store("how does binary search work", "some answer", model="gpt-oss-120b", engine=engine)

    # backdate it past its own TTL to simulate an expired entry
    from sqlmodel import Session

    with Session(engine) as session:
        db_entry = session.get(CacheEntry, entry.id)
        db_entry.created_at = datetime.now(UTC) - timedelta(seconds=db_entry.ttl_seconds + 10)
        session.add(db_entry)
        session.commit()

    result = lookup("explain how binary search works", engine=engine)
    assert not result.hit


def test_lookup_misses_volatile_entry_immediately():
    engine = _in_memory_engine()
    store("what's the weather today", "sunny, 75F", model="gpt-oss-120b", engine=engine)

    result = lookup("what's the weather today", engine=engine)
    assert not result.hit


def test_param_diff_rejects_high_similarity_conflicting_pair():
    engine = _in_memory_engine()
    store(
        "when is project 2 due for CS 3214",
        "project 2 is due next Friday",
        model="gpt-oss-120b",
        engine=engine,
    )

    with_diff = lookup("when is project 3 due for CS 3214", engine=engine, use_param_diff=True)
    without_diff = lookup("when is project 3 due for CS 3214", engine=engine, use_param_diff=False)

    assert not with_diff.hit
    assert without_diff.hit
