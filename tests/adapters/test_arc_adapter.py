import pytest

from arcus.adapters.arc_adapter import ArcAdapter, ArcModel


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("ARC_API_KEY", raising=False)
    with pytest.raises(ValueError):
        ArcAdapter()


def test_explicit_key_wins_over_env(monkeypatch):
    monkeypatch.setenv("ARC_API_KEY", "env-key")
    adapter = ArcAdapter(api_key="explicit-key")
    assert adapter._client.api_key == "explicit-key"


def test_falls_back_to_env_key(monkeypatch):
    monkeypatch.setenv("ARC_API_KEY", "env-key")
    adapter = ArcAdapter()
    assert adapter._client.api_key == "env-key"


def test_uses_arc_base_url(monkeypatch):
    monkeypatch.setenv("ARC_API_KEY", "env-key")
    adapter = ArcAdapter()
    assert str(adapter._client.base_url) == "https://llm-api.arc.vt.edu/api/v1/"


@pytest.mark.parametrize(
    "model",
    [
        ArcModel.GPT_OSS_120B,
        ArcModel.GLM_5_3,
        ArcModel.KIMI_K3,
        ArcModel.DEEPSEEK_V4_FLASH,
        "some-future-model-not-in-the-enum-yet",
    ],
)
def test_chat_forwards_model_and_messages(monkeypatch, model):
    monkeypatch.setenv("ARC_API_KEY", "env-key")
    adapter = ArcAdapter()

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return "fake completion"

    adapter._client.chat.completions.create = fake_create

    messages = [{"role": "user", "content": "hi"}]
    result = adapter.chat(model, messages)

    expected_model = model.value if isinstance(model, ArcModel) else model
    assert captured["model"] == expected_model
    assert captured["messages"] == messages
    assert result == "fake completion"


def test_list_models_returns_ids(monkeypatch):
    monkeypatch.setenv("ARC_API_KEY", "env-key")
    adapter = ArcAdapter()

    class _Model:
        def __init__(self, id_):
            self.id = id_

    adapter._client.models.list = lambda: [_Model("gpt-oss-120b"), _Model("GLM-5.3-thinking-high")]

    assert adapter.list_models() == ["gpt-oss-120b", "GLM-5.3-thinking-high"]


def test_chat_passes_through_extra_kwargs(monkeypatch):
    monkeypatch.setenv("ARC_API_KEY", "env-key")
    adapter = ArcAdapter()

    captured = {}
    adapter._client.chat.completions.create = lambda **kwargs: captured.update(kwargs)

    adapter.chat(ArcModel.KIMI_K3, [{"role": "user", "content": "hi"}], temperature=0.2)

    assert captured["temperature"] == 0.2
