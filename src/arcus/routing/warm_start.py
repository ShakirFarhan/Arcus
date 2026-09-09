from sqlmodel import Session, func, select

from arcus.routing.bandit import ContextualBandit
from arcus.storage.db import REWARD_VERSION, RequestLog


def replay_history(bandit: ContextualBandit, engine, mode: str) -> None:
    """Rebuilds a bandit's learned state from past requests in the log.

    Every `arcus` invocation is a fresh process, there's no daemon keeping
    the bandit alive in memory between runs. Without this, each run would
    start from a blank slate and the router would never actually learn
    anything.

    The work is pushed into a GROUP BY rather than replaying rows one at
    a time. Every algorithm here accumulates exactly two things per arm,
    a pull count and a reward total (Thompson's alpha and beta are just
    those two rearranged), and both are associative, so the totals fully
    describe the history that produced them. Summing 50k rows in SQL and
    restoring four numbers is identical in result to calling update()
    50k times, and doesn't get slower as the log grows: row-by-row replay
    cost about 0.7s at 50k rows, on every single invocation, before the
    request being asked about had even been sent.
    """
    with Session(engine) as session:
        totals = session.exec(
            select(
                RequestLog.task_type,
                RequestLog.length_bucket,
                RequestLog.model,
                func.count(RequestLog.id),
                func.sum(RequestLog.reward),
            )
            .where(RequestLog.mode == mode)
            .where(RequestLog.reward.is_not(None))
            # rewards from an older generation of the reward function
            # aren't on the same scale as current ones, so averaging
            # across generations would hand the bandit the mean of two
            # different measurements.
            .where(RequestLog.reward_version == REWARD_VERSION)
            .group_by(RequestLog.task_type, RequestLog.length_bucket, RequestLog.model)
        ).all()

    for task_type, length_bucket, model, pulls, reward_sum in totals:
        context_key = f"{task_type}:{length_bucket}"
        if model not in bandit.arms_for(context_key):
            # a model that was live and a valid arm for this context when
            # those rows were logged, but no longer is, either ARC
            # retired or renamed it, or it was only ever an arm here
            # under a config that's since been turned off. no arm to
            # credit the reward to, so skip rather than crash the replay.
            continue
        bandit.restore(context_key, model, pulls, float(reward_sum))
