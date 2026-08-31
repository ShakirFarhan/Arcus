import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

import numpy as np
from sqlmodel import Field, Session, SQLModel, select

from arcus.embeddings import embed
from arcus.storage.db import get_engine

DEFAULT_SIMILARITY_THRESHOLD = 0.80


class Volatility(str, Enum):
    VOLATILE = "volatile"
    STABLE = "stable"


_VOLATILE_MARKERS = re.compile(
    r"\btoday\b|\bcurrent(ly)?\b|\blatest\b|\bnow\b"
    r"|\bthis (week|month|year)\b|\brecently\b|\bright now\b",
    re.IGNORECASE,
)

_SEVEN_DAYS_SECONDS = 7 * 24 * 60 * 60


def classify_volatility(query: str) -> Volatility:
    if _VOLATILE_MARKERS.search(query):
        return Volatility.VOLATILE
    return Volatility.STABLE


def ttl_seconds_for(volatility: Volatility) -> int:
    # VOLATILE gets a TTL of 0, so it expires the instant it's written
    # and can never actually be served back. that's simpler than a
    # separate "don't cache this" branch, the normal expiry check already
    # handles it. 7 days for STABLE is a first guess, easy to retune once
    # there's real usage data on how often people ask the same
    # conceptual question again.
    if volatility == Volatility.VOLATILE:
        return 0
    return _SEVEN_DAYS_SECONDS


_NUMBER_PATTERN = re.compile(r"\b\d[\d,]*\.?\d*\b")
# crude proper-noun/entity proxy: capitalized words not at the very start
# of the string, where every sentence's first word would otherwise cause
# false positives. not real NER, no NLP dependency for it, the benchmark
# in cache/benchmark.py is what honestly measures how well this actually
# performs rather than just asserting it does.
_ENTITY_PATTERN = re.compile(r"(?<!^)\b[A-Z][a-zA-Z]{2,}\b")


def extract_params(text: str) -> set[str]:
    numbers = set(_NUMBER_PATTERN.findall(text))
    entities = set(_ENTITY_PATTERN.findall(text.strip()))
    return numbers | entities


def params_conflict(query_a: str, query_b: str) -> bool:
    return extract_params(query_a) != extract_params(query_b)


class CacheEntry(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    query: str
    response: str
    # which ARC model actually produced this response. without this, a
    # cache hit would have no real arm to attribute stats to, "the cache"
    # isn't one of the four models catch/reward numbers get broken down
    # by.
    model: str
    ttl_seconds: int
    # raw float32 bytes, not JSON, cheaper to store and reload than a
    # list of floats for a 384-dim vector.
    embedding: bytes


def _serialize_embedding(vector: np.ndarray) -> bytes:
    return vector.astype(np.float32).tobytes()


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _is_expired(entry: CacheEntry) -> bool:
    # SQLite has no native timezone-aware datetime type, so a round-trip
    # through it comes back naive even though it was stored as UTC.
    # reattach the tzinfo rather than comparing naive against aware and
    # blowing up.
    created_at = entry.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)

    expires_at = created_at + timedelta(seconds=entry.ttl_seconds)
    return datetime.now(UTC) > expires_at


def store(query: str, response: str, model: str, engine=None) -> CacheEntry:
    engine = engine or get_engine()

    vector = embed([query])[0]
    ttl_seconds = ttl_seconds_for(classify_volatility(query))

    entry = CacheEntry(
        query=query,
        response=response,
        model=model,
        ttl_seconds=ttl_seconds,
        embedding=_serialize_embedding(vector),
    )

    with Session(engine) as session:
        session.add(entry)
        session.commit()
        session.refresh(entry)

    return entry


@dataclass(frozen=True)
class CacheResult:
    hit: bool
    response: str | None
    similarity: float | None
    matched_query: str | None
    model: str | None


def lookup(
    query: str,
    engine=None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    use_param_diff: bool = True,
) -> CacheResult:
    engine = engine or get_engine()

    with Session(engine) as session:
        entries = session.exec(select(CacheEntry)).all()

    live_entries = [e for e in entries if not _is_expired(e)]
    if not live_entries:
        return CacheResult(hit=False, response=None, similarity=None, matched_query=None, model=None)

    query_embedding = embed([query])[0]
    similarities = [
        (float(_deserialize_embedding(e.embedding) @ query_embedding), e) for e in live_entries
    ]
    best_similarity, best_entry = max(similarities, key=lambda pair: pair[0])

    if best_similarity < similarity_threshold:
        return CacheResult(hit=False, response=None, similarity=best_similarity, matched_query=None, model=None)

    if use_param_diff and params_conflict(query, best_entry.query):
        return CacheResult(hit=False, response=None, similarity=best_similarity, matched_query=None, model=None)

    return CacheResult(
        hit=True,
        response=best_entry.response,
        similarity=best_similarity,
        matched_query=best_entry.query,
        model=best_entry.model,
    )
