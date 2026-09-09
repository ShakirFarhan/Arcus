"""What happens on a base install, where sentence-transformers isn't there.

The whole point of putting it behind an extra is that everything keeps
working without it, so these exercise the degraded paths rather than the
happy ones.
"""

import pytest
from sqlmodel import SQLModel, create_engine

from arcus import embeddings
from arcus.cache import semantic_cache
from arcus.routing.context import TaskType, classify_task_type


def _in_memory_engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def no_embeddings(monkeypatch):
    monkeypatch.setattr(embeddings, "_model_cache", None)
    monkeypatch.setattr(embeddings, "embeddings_available", lambda: False)
    monkeypatch.setattr(semantic_cache, "embeddings_available", lambda: False)

    def _boom(*a, **kw):
        raise AssertionError("embeddings should never be touched without the extra installed")

    monkeypatch.setattr(semantic_cache, "embed", _boom)


def test_lookup_reports_a_miss_without_embeddings(no_embeddings):
    result = semantic_cache.lookup("anything at all", engine=_in_memory_engine())

    assert result.hit is False
    assert result.response is None


def test_store_is_a_noop_without_embeddings(no_embeddings):
    engine = _in_memory_engine()

    assert semantic_cache.store("q", "an answer", model="gpt-oss-120b", engine=engine) is None

    # and nothing was written, so a later lookup can't half-match on it
    from sqlmodel import Session, select

    with Session(engine) as session:
        assert session.exec(select(semantic_cache.CacheEntry)).all() == []


def test_classification_still_works_from_the_regex_rules(no_embeddings, monkeypatch):
    def _boom(*a, **kw):
        raise embeddings.EmbeddingsUnavailable("not installed")

    monkeypatch.setattr("arcus.routing.context._get_embedding_classifier", _boom)

    # the rules carry these on their own, no embedding needed
    assert classify_task_type("Traceback (most recent call last):") == TaskType.CODE
    assert classify_task_type("write me a poem about spring") == TaskType.WRITING
    # and an unmatched prompt falls back rather than raising
    assert classify_task_type("mm hmm sure whatever") == TaskType.GENERAL


def test_get_embedding_model_raises_a_actionable_error(monkeypatch):
    monkeypatch.setattr(embeddings, "_model_cache", None)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(embeddings.EmbeddingsUnavailable) as excinfo:
        embeddings.get_embedding_model()

    # the message has to tell someone how to fix it, not just what broke
    assert "arcus-cli[cache]" in str(excinfo.value)


def test_embeddings_available_reflects_the_installed_state():
    # sentence-transformers *is* installed in the dev environment, so
    # this is the positive case; the fixture above covers the negative
    assert embeddings.embeddings_available() is True
