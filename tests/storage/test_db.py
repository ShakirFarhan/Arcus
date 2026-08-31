from sqlmodel import Session, SQLModel, create_engine, select

from arcus.storage.db import RequestLog, _database_url, get_engine, log_request


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
