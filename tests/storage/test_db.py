from sqlmodel import Session, SQLModel, create_engine, select, text

from arcus.storage.db import REWARD_VERSION, RequestLog, _database_url, get_engine, log_request


def _in_memory_engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


def test_log_request_roundtrip():
    engine = _in_memory_engine()

    entry = log_request(
        prompt="why is my loop infinite",
        task_type="code",
        length_bucket="short",
        model="gpt-oss-120b",
        engine=engine,
    )

    assert entry.id is not None

    with Session(engine) as session:
        rows = session.exec(select(RequestLog)).all()

    assert len(rows) == 1
    assert rows[0].prompt == "why is my loop infinite"
    assert rows[0].task_type == "code"
    assert rows[0].model == "gpt-oss-120b"


def test_optional_fields_default_to_none():
    engine = _in_memory_engine()

    entry = log_request(
        prompt="hi",
        task_type="general",
        length_bucket="short",
        model="Kimi-K3",
        engine=engine,
    )

    assert entry.propensity is None
    assert entry.latency_ms is None
    assert entry.finish_reason is None
    assert entry.error is None
    assert entry.mode is None
    assert entry.reward is None
    assert entry.cache_hit is False
    assert entry.quality_passed is True


def test_optional_fields_get_stored_when_given():
    engine = _in_memory_engine()

    entry = log_request(
        prompt="hi",
        task_type="general",
        length_bucket="short",
        model="Kimi-K3",
        propensity=0.4,
        latency_ms=812.5,
        finish_reason="stop",
        mode="bandit",
        reward=0.82,
        cache_hit=True,
        quality_passed=False,
        engine=engine,
    )

    assert entry.propensity == 0.4
    assert entry.latency_ms == 812.5
    assert entry.finish_reason == "stop"
    assert entry.mode == "bandit"
    assert entry.reward == 0.82
    assert entry.cache_hit is True
    assert entry.quality_passed is False


def test_conversation_fields_roundtrip():
    engine = _in_memory_engine()

    entry = log_request(
        prompt="what about in python?",
        task_type="code",
        length_bucket="short",
        model="gpt-oss-120b",
        conversation_id="abc-123",
        turn_index=1,
        engine=engine,
    )

    assert entry.conversation_id == "abc-123"
    assert entry.turn_index == 1

    with Session(engine) as session:
        rows = session.exec(select(RequestLog)).all()

    assert rows[0].conversation_id == "abc-123"
    assert rows[0].turn_index == 1


def test_conversation_fields_default_to_none():
    engine = _in_memory_engine()

    entry = log_request(
        prompt="hi",
        task_type="general",
        length_bucket="short",
        model="Kimi-K3",
        engine=engine,
    )

    assert entry.conversation_id is None
    assert entry.turn_index is None


def test_database_url_honors_env_override(monkeypatch):
    monkeypatch.setenv("ARCUS_DATABASE_URL", "postgresql://user:pass@localhost/arcus_dev")
    assert _database_url() == "postgresql://user:pass@localhost/arcus_dev"


def test_database_url_defaults_to_platformdirs_sqlite_path(monkeypatch, tmp_path):
    monkeypatch.delenv("ARCUS_DATABASE_URL", raising=False)
    monkeypatch.setattr("arcus.storage.db.user_data_dir", lambda name: str(tmp_path))

    url = _database_url()

    assert url == f"sqlite:///{tmp_path / 'arcus.db'}"


def test_get_engine_creates_sqlite_file_on_disk(monkeypatch, tmp_path):
    monkeypatch.delenv("ARCUS_DATABASE_URL", raising=False)
    monkeypatch.setattr("arcus.storage.db.user_data_dir", lambda name: str(tmp_path))

    get_engine()

    assert (tmp_path / "arcus.db").exists()


def _legacy_schema_engine(path):
    """A database shaped the way it was before grading existed: no
    reward_version, judge, or response_text columns.
    """
    engine = create_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        session.exec(
            text(
                "CREATE TABLE requestlog ("
                "id INTEGER PRIMARY KEY, created_at DATETIME, prompt VARCHAR, "
                "task_type VARCHAR, length_bucket VARCHAR, model VARCHAR, "
                "propensity FLOAT, latency_ms FLOAT, finish_reason VARCHAR, "
                "error VARCHAR, mode VARCHAR, reward FLOAT, cache_hit BOOLEAN, "
                "quality_passed BOOLEAN, conversation_id VARCHAR, turn_index INTEGER)"
            )
        )
        session.exec(
            text(
                "INSERT INTO requestlog (prompt, task_type, length_bucket, model, mode, reward) "
                "VALUES ('old question', 'code', 'short', 'gpt-oss-120b', 'bandit', 1.0)"
            )
        )
        session.commit()
    return engine


def test_migrate_adds_the_new_columns_to_an_older_database(monkeypatch, tmp_path):
    db_path = tmp_path / "arcus.db"
    _legacy_schema_engine(db_path)

    monkeypatch.setenv("ARCUS_DATABASE_URL", f"sqlite:///{db_path}")
    engine = get_engine()  # should not raise

    with Session(engine) as session:
        columns = {row[1] for row in session.exec(text("PRAGMA table_info(requestlog)")).all()}

    assert {"reward_version", "judge_pending", "judge_score", "response_text"} <= columns


def test_migrate_marks_pre_existing_rows_as_the_older_reward_generation(monkeypatch, tmp_path):
    db_path = tmp_path / "arcus.db"
    _legacy_schema_engine(db_path)

    monkeypatch.setenv("ARCUS_DATABASE_URL", f"sqlite:///{db_path}")
    engine = get_engine()

    with Session(engine) as session:
        row = session.exec(select(RequestLog)).one()

    # a row written before grading existed was scored pass/fail, calling
    # it current would quietly mix two different measurements
    assert row.reward_version == 1


def test_new_rows_carry_the_current_reward_generation():
    engine = _in_memory_engine()
    entry = log_request(
        prompt="q", task_type="code", length_bucket="short", model="gpt-oss-120b", engine=engine
    )
    assert entry.reward_version == REWARD_VERSION
