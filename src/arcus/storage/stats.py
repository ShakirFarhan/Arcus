from dataclasses import dataclass
from statistics import mean

from sqlmodel import Session, select

from arcus.routing.reward import COST_SCORES
from arcus.storage.db import RequestLog


@dataclass(frozen=True)
class ArmModeSummary:
    model: str
    mode: str
    request_count: int
    avg_reward: float | None
    avg_latency_ms: float | None
    cost_score: float | None


def aggregate_by_arm_and_mode(engine) -> list[ArmModeSummary]:
    """Groups logged requests by (model, mode) and averages reward/latency
    per group. Done in plain Python rather than a SQL GROUP BY/AVG, this
    is a personal local CLI's log, realistically thousands of rows at
    most, not a scale where hand-rolled SQL aggregation earns its
    complexity over just reading the rows.
    """
    with Session(engine) as session:
        rows = session.exec(select(RequestLog)).all()

    groups: dict[tuple[str, str], list[RequestLog]] = {}
    for row in rows:
        key = (row.model, row.mode or "unknown")
        groups.setdefault(key, []).append(row)

    summaries = []
    for (model, mode), entries in groups.items():
        rewards = [e.reward for e in entries if e.reward is not None]
        latencies = [e.latency_ms for e in entries if e.latency_ms is not None]

        summaries.append(
            ArmModeSummary(
                model=model,
                mode=mode,
                request_count=len(entries),
                avg_reward=mean(rewards) if rewards else None,
                avg_latency_ms=mean(latencies) if latencies else None,
                cost_score=COST_SCORES.get(model),
            )
        )

    return sorted(summaries, key=lambda s: (s.model, s.mode))
