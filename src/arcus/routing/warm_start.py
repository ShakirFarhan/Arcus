from sqlmodel import Session, select

from arcus.routing.bandit import ContextualBandit
from arcus.storage.db import RequestLog


def replay_history(bandit: ContextualBandit, engine, mode: str) -> None:
    """Rebuilds a bandit's learned state from past requests in the log.

    Every `arcus` invocation is a fresh process, there's no daemon keeping
    the bandit alive in memory between runs. Without this, each run would
    start from a blank slate and the router would never actually learn
    anything. It works because update() on all four algorithms is just an
    associative accumulation of pull counts and reward sums, so replaying
    the log in chronological order and feeding each row back through
    update() lands on the same state as if the process had been running
    continuously the whole time.
    """
    with Session(engine) as session:
        rows = session.exec(
            select(RequestLog)
            .where(RequestLog.mode == mode)
            .where(RequestLog.reward.is_not(None))
            .order_by(RequestLog.created_at)
        ).all()

    for row in rows:
        context_key = f"{row.task_type}:{row.length_bucket}"
        bandit.update(context_key, row.model, row.reward)
