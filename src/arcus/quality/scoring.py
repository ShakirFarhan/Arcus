from sqlmodel import Session, select

from arcus.adapters.arc_adapter import ArcAdapter
from arcus.quality.judge import judge_response
from arcus.routing.reward import compute_reward
from arcus.storage.db import RequestLog

# a bounded sip rather than the whole queue: this runs in the background
# while the user's actual question is in flight, and the point is for it
# to disappear into that window rather than become its own wait. anything
# left over stays pending and gets picked up next run.
DEFAULT_BATCH = 3


def pending_rows(engine, limit: int = DEFAULT_BATCH) -> list[RequestLog]:
    with Session(engine) as session:
        return list(
            session.exec(
                select(RequestLog)
                .where(RequestLog.judge_pending == True)  # noqa: E712, SQLModel needs the comparison, not `is True`
                .where(RequestLog.response_text.is_not(None))
                .order_by(RequestLog.created_at)
                .limit(limit)
            ).all()
        )


def score_pending(adapter: ArcAdapter, engine, limit: int = DEFAULT_BATCH) -> int:
    """Grades up to `limit` rows that are waiting on a judgement and
    rewrites their reward with the graded quality term. Returns how many
    rows were actually scored.

    A row is cleared from the queue either way, judged or not. Leaving a
    permanently unparseable response pending would mean retrying it on
    every single invocation forever, and one lost judgement matters far
    less than that.
    """
    rows = pending_rows(engine, limit=limit)
    if not rows:
        return 0

    scored = 0
    with Session(engine) as session:
        for stale in rows:
            row = session.get(RequestLog, stale.id)
            if row is None:
                continue

            score = judge_response(
                adapter,
                question=row.prompt,
                answer=row.response_text or "",
                responder=row.model,
            )

            if score is not None:
                row.judge_score = score
                # recomputed from the same latency and model the request
                # actually saw, so the only thing that moves is the
                # quality term the judge just replaced
                row.reward = compute_reward(
                    latency_ms=row.latency_ms or 0.0,
                    model=row.model,
                    quality_score=score,
                )
                scored += 1

            row.judge_pending = False
            # the answer text was only ever held so the judge could read
            # it, no reason to keep every response on disk past that
            row.response_text = None
            session.add(row)

        session.commit()

    return scored
