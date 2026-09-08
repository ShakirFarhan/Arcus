from sqlmodel import Session, SQLModel, create_engine, select

from arcus.quality import scoring
from arcus.quality.scoring import pending_rows, score_pending
from arcus.storage.db import RequestLog, log_request


def _in_memory_engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


def _log(engine, *, model="gpt-oss-120b", pending=True, response="an answer", latency=200.0):
    return log_request(
        prompt="why is the sky blue",
        task_type="general",
        length_bucket="short",
        model=model,
        mode="bandit",
        propensity=0.5,
        latency_ms=latency,
        reward=0.9,
        judge_pending=pending,
        response_text=response if pending else None,
        engine=engine,
    )


def test_pending_rows_only_returns_rows_awaiting_a_judgement():
    engine = _in_memory_engine()
    _log(engine, pending=True)
    _log(engine, pending=False)

    assert len(pending_rows(engine)) == 1


def test_pending_rows_skips_a_row_with_no_answer_text_to_grade():
    engine = _in_memory_engine()
    _log(engine, pending=True, response=None)

    assert pending_rows(engine) == []


def test_pending_rows_respects_the_limit():
    engine = _in_memory_engine()
    for _ in range(5):
        _log(engine)

    assert len(pending_rows(engine, limit=2)) == 2


def test_score_pending_writes_the_judgement_and_recomputes_the_reward(monkeypatch):
    engine = _in_memory_engine()
    _log(engine, model="GLM-5.3", latency=200.0)

    monkeypatch.setattr(scoring, "judge_response", lambda *a, **kw: 0.4)

    assert score_pending(object(), engine) == 1

    with Session(engine) as session:
        row = session.exec(select(RequestLog)).one()

    assert row.judge_score == 0.4
    assert row.judge_pending is False
    # the reward has to fall now that quality is 0.4 rather than the
    # flat 1.0 a passing response used to get
    assert row.reward < 0.9


def test_score_pending_recomputes_from_the_rows_own_latency_and_model(monkeypatch):
    from arcus.routing.reward import compute_reward

    engine = _in_memory_engine()
    _log(engine, model="Kimi-K3", latency=1500.0)

    monkeypatch.setattr(scoring, "judge_response", lambda *a, **kw: 0.6)
    score_pending(object(), engine)

    with Session(engine) as session:
        row = session.exec(select(RequestLog)).one()

    assert row.reward == compute_reward(latency_ms=1500.0, model="Kimi-K3", quality_score=0.6)


def test_score_pending_clears_the_stored_answer_once_it_has_been_graded(monkeypatch):
    engine = _in_memory_engine()
    _log(engine)

    monkeypatch.setattr(scoring, "judge_response", lambda *a, **kw: 0.7)
    score_pending(object(), engine)

    with Session(engine) as session:
        row = session.exec(select(RequestLog)).one()

    # no reason to keep every response this tool ever produced on disk
    # past the moment the judge needed to read it
    assert row.response_text is None


def test_score_pending_drops_a_row_the_judge_could_not_score(monkeypatch):
    engine = _in_memory_engine()
    _log(engine)

    monkeypatch.setattr(scoring, "judge_response", lambda *a, **kw: None)

    assert score_pending(object(), engine) == 0

    with Session(engine) as session:
        row = session.exec(select(RequestLog)).one()

    # cleared from the queue anyway, otherwise an unparseable response
    # gets retried on every single invocation forever
    assert row.judge_pending is False
    assert row.judge_score is None
    assert row.reward == 0.9


def test_score_pending_is_a_noop_with_an_empty_queue():
    engine = _in_memory_engine()
    assert score_pending(object(), engine) == 0
