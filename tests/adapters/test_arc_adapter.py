import httpx
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


def test_upload_file_posts_multipart_and_returns_the_id(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_API_KEY", "env-key")
    adapter = ArcAdapter()

    doc = tmp_path / "notes.txt"
    doc.write_text("hello world")

    captured = {}

    def fake_post(url, headers, files, timeout):
        captured["url"] = url
        captured["headers"] = headers
        request = httpx.Request("POST", url)
        return httpx.Response(200, request=request, json={"id": "file-abc123"})

    monkeypatch.setattr("arcus.adapters.arc_adapter.httpx.post", fake_post)

    file_id = adapter.upload_file(str(doc))

    assert file_id == "file-abc123"
    assert captured["url"] == f"{ArcAdapter.BASE_URL}/files/"
    assert captured["headers"]["Authorization"] == "Bearer env-key"


def test_upload_file_raises_for_a_missing_file(monkeypatch):
    monkeypatch.setenv("ARC_API_KEY", "env-key")
    adapter = ArcAdapter()

    with pytest.raises(OSError):
        adapter.upload_file("/no/such/file.txt")


def test_upload_file_raises_on_a_bad_status(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_API_KEY", "env-key")
    adapter = ArcAdapter()

    doc = tmp_path / "notes.txt"
    doc.write_text("hello")

    def fake_post(url, headers, files, timeout):
        request = httpx.Request("POST", url)
        return httpx.Response(500, request=request)

    monkeypatch.setattr("arcus.adapters.arc_adapter.httpx.post", fake_post)

    with pytest.raises(httpx.HTTPStatusError):
        adapter.upload_file(str(doc))


def test_delete_file_sends_a_delete_request_with_the_file_id_in_the_url(monkeypatch):
    monkeypatch.setenv("ARC_API_KEY", "env-key")
    adapter = ArcAdapter()

    captured = {}

    def fake_delete(url, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        request = httpx.Request("DELETE", url)
        return httpx.Response(200, request=request)

    monkeypatch.setattr("arcus.adapters.arc_adapter.httpx.delete", fake_delete)

    adapter.delete_file("file-abc123")

    assert captured["url"] == f"{ArcAdapter.BASE_URL}/files/file-abc123"
    assert captured["headers"]["Authorization"] == "Bearer env-key"


def test_delete_file_raises_on_a_bad_status(monkeypatch):
    monkeypatch.setenv("ARC_API_KEY", "env-key")
    adapter = ArcAdapter()

    def fake_delete(url, headers, timeout):
        request = httpx.Request("DELETE", url)
        return httpx.Response(404, request=request)

    monkeypatch.setattr("arcus.adapters.arc_adapter.httpx.delete", fake_delete)

    with pytest.raises(httpx.HTTPStatusError):
        adapter.delete_file("file-abc123")


def test_chat_passes_through_extra_kwargs(monkeypatch):
    monkeypatch.setenv("ARC_API_KEY", "env-key")
    adapter = ArcAdapter()

    captured = {}
    adapter._client.chat.completions.create = lambda **kwargs: captured.update(kwargs)

    adapter.chat(ArcModel.KIMI_K3, [{"role": "user", "content": "hi"}], temperature=0.2)

    assert captured["temperature"] == 0.2
