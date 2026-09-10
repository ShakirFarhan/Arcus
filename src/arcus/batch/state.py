"""Durable per-row job state, so an interrupted run resumes.

Kept in its own database rather than the request log. Batch rows are the
user's research data, sometimes thousands of them per job, and mixing
that into the log the router learns from would both distort routing and
make "delete my batch data" impossible to honour cleanly.
"""

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path

from platformdirs import user_data_dir
from sqlmodel import Field, Session, SQLModel, create_engine, select

PENDING = "pending"
DONE = "done"
FAILED = "failed"


class BatchJob(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    fingerprint: str = Field(index=True, unique=True)

    input_path: str
    output_path: str
    instruction: str
    column: str
    model: str
    total_rows: int

    # recorded per job rather than derived later, so the manifest can say
    # what actually produced these answers even after the config changes
    choices: str | None = Field(default=None)
    examples_used: int = Field(default=0)


class BatchItem(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    job_id: int = Field(index=True)
    row_index: int = Field(index=True)

    status: str = Field(default=PENDING, index=True)
    output: str | None = Field(default=None)
    error: str | None = Field(default=None)
    attempts: int = Field(default=0)
    # which model actually answered this row. constant for a normal job,
    # but recorded per row anyway: a manifest that merely asserts one
    # model was used is worth less than one that can show it.
    model: str | None = Field(default=None)


def _database_url() -> str:
    override = os.environ.get("ARCUS_BATCH_DATABASE_URL")
    if override:
        return override
    data_dir = Path(user_data_dir("arcus"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{data_dir / 'batches.db'}"


def get_engine():
    engine = create_engine(_database_url())
    SQLModel.metadata.create_all(engine)
    return engine


def fingerprint(input_path: Path, instruction: str, column: str) -> str:
    """Identifies a job, so re-running the same command resumes it and a
    changed command starts fresh.

    Hashes the head of the file plus its size rather than the whole
    thing: a batch input can be hundreds of megabytes and re-reading all
    of it just to decide whether to resume would be its own delay.
    Deliberately not mtime, which changes when a file is merely touched
    and would throw away a resumable job for no reason.
    """
    digest = hashlib.sha256()
    digest.update(str(input_path.resolve()).encode())
    digest.update(str(input_path.stat().st_size).encode())
    with input_path.open("rb") as fh:
        digest.update(fh.read(65_536))
    digest.update(instruction.encode())
    digest.update(column.encode())
    return digest.hexdigest()[:32]


def find_job(engine, fp: str) -> BatchJob | None:
    with Session(engine) as session:
        return session.exec(select(BatchJob).where(BatchJob.fingerprint == fp)).first()


def create_job(engine, **fields) -> BatchJob:
    job = BatchJob(**fields)
    with Session(engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)
    return job


def register_rows(engine, job_id: int, row_indexes: list[int]) -> None:
    """Records every row as pending, once, when the job is created."""
    with Session(engine) as session:
        for index in row_indexes:
            session.add(BatchItem(job_id=job_id, row_index=index))
        session.commit()


def pending_indexes(engine, job_id: int) -> set[int]:
    with Session(engine) as session:
        rows = session.exec(
            select(BatchItem.row_index)
            .where(BatchItem.job_id == job_id)
            .where(BatchItem.status == PENDING)
        ).all()
    return set(rows)


def counts(engine, job_id: int) -> dict[str, int]:
    with Session(engine) as session:
        items = session.exec(select(BatchItem).where(BatchItem.job_id == job_id)).all()
    result = {PENDING: 0, DONE: 0, FAILED: 0}
    for item in items:
        result[item.status] = result.get(item.status, 0) + 1
    return result


def record_result(
    engine,
    job_id: int,
    row_index: int,
    *,
    status: str,
    output: str | None = None,
    error: str | None = None,
    model: str | None = None,
    attempts: int = 1,
) -> None:
    with Session(engine) as session:
        item = session.exec(
            select(BatchItem)
            .where(BatchItem.job_id == job_id)
            .where(BatchItem.row_index == row_index)
        ).first()
        if item is None:
            item = BatchItem(job_id=job_id, row_index=row_index)
        item.status = status
        item.output = output
        item.error = error
        item.model = model
        item.attempts = attempts
        session.add(item)
        session.commit()


def failed_items(engine, job_id: int) -> list[BatchItem]:
    with Session(engine) as session:
        return list(
            session.exec(
                select(BatchItem)
                .where(BatchItem.job_id == job_id)
                .where(BatchItem.status == FAILED)
                .order_by(BatchItem.row_index)
            ).all()
        )


def forget_job(engine, job_id: int) -> None:
    """Drops a job and everything it recorded.

    Batch rows hold the user's own data, so being able to remove it
    completely matters more here than it would for a request log.
    """
    with Session(engine) as session:
        for item in session.exec(select(BatchItem).where(BatchItem.job_id == job_id)).all():
            session.delete(item)
        job = session.get(BatchJob, job_id)
        if job is not None:
            session.delete(job)
        session.commit()
