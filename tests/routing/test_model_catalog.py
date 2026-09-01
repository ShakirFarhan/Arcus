import time

from arcus.adapters.arc_adapter import ArcModel
from arcus.routing.model_catalog import filter_to_live, known_arms


class _FakeAdapter:
    def __init__(self, model_ids, calls=None):
        self._model_ids = model_ids
        self._calls = calls if calls is not None else []

    def list_models(self):
        self._calls.append(1)
        return self._model_ids


def _all_configured():
    return [m.value for m in ArcModel]


def test_known_arms_returns_full_list_when_everything_is_live(monkeypatch, tmp_path):
    monkeypatch.setattr("arcus.routing.model_catalog.user_cache_dir", lambda name: str(tmp_path))

    adapter = _FakeAdapter(_all_configured() + ["gpt-oss-120b-thinking-high"])
    assert sorted(known_arms(adapter)) == sorted(_all_configured())


def test_known_arms_drops_a_model_missing_from_the_live_catalog(monkeypatch, tmp_path):
    monkeypatch.setattr("arcus.routing.model_catalog.user_cache_dir", lambda name: str(tmp_path))

    live = [m for m in _all_configured() if m != ArcModel.GLM_5_3.value]
    adapter = _FakeAdapter(live)

    result = known_arms(adapter)
    assert ArcModel.GLM_5_3.value not in result
    assert set(result) == set(live)


def test_known_arms_falls_back_when_the_live_check_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("arcus.routing.model_catalog.user_cache_dir", lambda name: str(tmp_path))

    class _Broken:
        def list_models(self):
            raise ConnectionError("arc is unreachable")

    assert sorted(known_arms(_Broken())) == sorted(_all_configured())


def test_known_arms_falls_back_when_none_of_the_configured_models_are_live(monkeypatch, tmp_path):
    monkeypatch.setattr("arcus.routing.model_catalog.user_cache_dir", lambda name: str(tmp_path))

    adapter = _FakeAdapter(["some-completely-different-catalog"])
    assert sorted(known_arms(adapter)) == sorted(_all_configured())


def test_known_arms_does_not_hit_the_adapter_twice_within_the_cache_window(monkeypatch, tmp_path):
    monkeypatch.setattr("arcus.routing.model_catalog.user_cache_dir", lambda name: str(tmp_path))

    calls = []
    adapter = _FakeAdapter(_all_configured(), calls=calls)

    known_arms(adapter)
    known_arms(adapter)

    assert len(calls) == 1


def test_known_arms_rechecks_once_the_cache_has_expired(monkeypatch, tmp_path):
    monkeypatch.setattr("arcus.routing.model_catalog.user_cache_dir", lambda name: str(tmp_path))
    monkeypatch.setattr("arcus.routing.model_catalog._CACHE_TTL_SECONDS", 0)

    calls = []
    adapter = _FakeAdapter(_all_configured(), calls=calls)

    known_arms(adapter)
    time.sleep(0.01)
    known_arms(adapter)

    assert len(calls) == 2


def test_filter_to_live_drops_candidates_missing_from_the_catalog(monkeypatch, tmp_path):
    monkeypatch.setattr("arcus.routing.model_catalog.user_cache_dir", lambda name: str(tmp_path))

    adapter = _FakeAdapter(["gpt-oss-120b-thinking-high-legacy-tool-calling"])
    candidates = [
        "gpt-oss-120b-thinking-high-legacy-tool-calling",
        "Kimi-K3-thinking-max-legacy-tool-calling",
    ]

    result = filter_to_live(adapter, candidates)

    assert result == ["gpt-oss-120b-thinking-high-legacy-tool-calling"]


def test_filter_to_live_falls_back_to_all_candidates_when_the_catalog_check_fails(monkeypatch, tmp_path):
    monkeypatch.setattr("arcus.routing.model_catalog.user_cache_dir", lambda name: str(tmp_path))

    class _Broken:
        def list_models(self):
            raise ConnectionError("arc is unreachable")

    candidates = ["a-legacy-model", "another-legacy-model"]
    assert filter_to_live(_Broken(), candidates) == candidates


def test_filter_to_live_shares_the_cache_with_known_arms(monkeypatch, tmp_path):
    monkeypatch.setattr("arcus.routing.model_catalog.user_cache_dir", lambda name: str(tmp_path))

    calls = []
    adapter = _FakeAdapter(_all_configured() + ["a-legacy-model"], calls=calls)

    known_arms(adapter)
    filter_to_live(adapter, ["a-legacy-model"])

    assert len(calls) == 1
