import os
from datetime import UTC, datetime
from pathlib import Path

from platformdirs import user_data_dir
from sqlmodel import Field, Session, SQLModel, create_engine


class RequestLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    prompt: str
    task_type: str
    length_bucket: str
    model: str

    # nullable since a cache hit skips the bandit and quality gate
    # entirely, but propensity in particular has to be a column from day
    # one: there's no way to go back and add propensity logging to rows
    # that already happened.
    propensity: float | None = Field(default=None)
    latency_ms: float | None = Field(default=None)
    finish_reason: str | None = Field(default=None)
    error: str | None = Field(default=None)

    # "bandit" or "random", whichever ContextualBandit actually drove
    # this request. mode plus model is the whole point of the A/B stats
    # comparison.
    mode: str | None = Field(default=None)
    # the actual compute_reward() output at request time, stored rather
    # than recomputed later. if reward.py's weights or cost table change
    # down the line, historical stats should still reflect what the
    # bandit was actually updated with, not whatever the weights happen
    # to be when someone runs `arcus stats`.
    reward: float | None = Field(default=None)

    # was this response served from the semantic cache instead of an
    # actual ARC call. needed for `arcus stats` to report a cache hit
    # rate, the current schema has no other way to tell a cache hit apart
    # from a normal request.
    cache_hit: bool = Field(default=False)
    # did this specific attempt pass the quality gate. combined with
    # `model`, this is what makes a per-model catch rate possible instead
    # of only ever seeing which model eventually succeeded.
    quality_passed: bool = Field(default=True)

    # groups every turn of one `arcus chat` session together. null for
    # one-shot `arcus "..."` calls, there's no session to group those into.
    conversation_id: str | None = Field(default=None, index=True)
    # position of this turn within its conversation, 0-indexed. null
    # alongside conversation_id for one-shot calls.
    turn_index: int | None = Field(default=None)


def _database_url() -> str:
    override = os.environ.get("ARCUS_DATABASE_URL")
    if override:
        return override

    data_dir = Path(user_data_dir("arcus"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{data_dir / 'arcus.db'}"


def get_engine():
    # deliberately not cached as a module-level singleton. sqlite engine
    # creation is cheap, and skipping the singleton means there's no
    # stale global state to worry about resetting between tests (or
    # between one CLI run and the next, if ARCUS_DATABASE_URL changes).
    engine = create_engine(_database_url())
    SQLModel.metadata.create_all(engine)
    return engine


def log_request(
    prompt: str,
    task_type: str,
    length_bucket: str,
    model: str,
    propensity: float | None = None,
    latency_ms: float | None = None,
    finish_reason: str | None = None,
    error: str | None = None,
    mode: str | None = None,
    reward: float | None = None,
    cache_hit: bool = False,
    quality_passed: bool = True,
    conversation_id: str | None = None,
    turn_index: int | None = None,
    engine=None,
) -> RequestLog:
    engine = engine or get_engine()

    entry = RequestLog(
        prompt=prompt,
        task_type=task_type,
        length_bucket=length_bucket,
        model=model,
        propensity=propensity,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
        error=error,
        mode=mode,
        reward=reward,
        cache_hit=cache_hit,
        quality_passed=quality_passed,
        conversation_id=conversation_id,
        turn_index=turn_index,
    )

    with Session(engine) as session:
        session.add(entry)
        session.commit()
        session.refresh(entry)

    return entry
