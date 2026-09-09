import pytest

from arcus.embeddings import embeddings_available


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_embeddings: needs sentence-transformers, which lives behind the optional `cache` extra",
    )


def pytest_collection_modifyitems(config, items):
    """Skips the embedding-dependent tests on a base install.

    The `cache` extra is optional on purpose (it pulls in torch, which is
    most of a gigabyte), so `pip install arcus-cli` alone has to leave a
    runnable test suite behind. CI installs the extra so these actually
    execute somewhere; this only matters for someone working from the
    lighter install.
    """
    if embeddings_available():
        return

    skip = pytest.mark.skip(reason="needs the `cache` extra: pip install 'arcus-cli[cache]'")
    for item in items:
        if "requires_embeddings" in item.keywords:
            item.add_marker(skip)
