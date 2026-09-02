import io

import httpx
import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from arcus import cli
from arcus.config import ArcusConfig
from arcus.routing.context import Context, LengthBucket, TaskType
from arcus.storage.db import RequestLog


def _in_memory_engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture(autouse=True)
def _no_real_embedding_load(monkeypatch):
    # run_ask kicks off a background thread that loads the real
    # sentence-transformers model, no reason to pay that cost (or need
    # network/disk access to it) in a unit test.
    monkeypatch.setattr(cli, "get_embedding_model", lambda: None)


@pytest.fixture(autouse=True)
def _fake_known_arms(monkeypatch):
    # known_arms()/filter_to_live() normally check ARC's live model
    # catalog against disk cache, neither of which a unit test should be
    # touching. tests that care about routing behavior mock
    # call_with_quality_gate directly anyway, so the exact arm list here
    # doesn't matter beyond being non-empty and consistent with what
    # those mocks expect.
    monkeypatch.setattr(
        cli, "known_arms", lambda adapter: ["gpt-oss-120b", "GLM-5.3", "Kimi-K3", "DeepSeek-V4-Flash"]
    )
    monkeypatch.setattr(cli, "filter_to_live", lambda adapter, candidates: candidates)


def _mock_completion(content, finish_reason="stop"):
    class Message:
        pass

    class Choice:
        pass

    class Completion:
        pass

    message = Message()
    message.content = content
    choice = Choice()
    choice.message = message
    choice.finish_reason = finish_reason
    completion = Completion()
    completion.choices = [choice]
    return completion


def test_build_prompt_uses_positional_args_when_no_pipe(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert cli._build_prompt(["explain", "recursion"]) == "explain recursion"


def test_build_prompt_uses_piped_content_alone(monkeypatch):
    stdin = io.StringIO("Traceback (most recent call last):\nValueError: bad")
    monkeypatch.setattr(stdin, "isatty", lambda: False)
    monkeypatch.setattr("sys.stdin", stdin)

    assert cli._build_prompt([]) == "Traceback (most recent call last):\nValueError: bad"


def test_build_prompt_combines_piped_content_with_positional_args(monkeypatch):
    stdin = io.StringIO("ValueError: bad input")
    monkeypatch.setattr(stdin, "isatty", lambda: False)
    monkeypatch.setattr("sys.stdin", stdin)

    result = cli._build_prompt(["why", "is", "this", "failing"])
    assert "ValueError: bad input" in result
    assert "why is this failing" in result


def test_run_setup_wizard_exits_cleanly_when_the_prompt_is_aborted(monkeypatch, capsys):
    import typer

    def _abort(*a, **kw):
        raise typer.Abort()

    monkeypatch.setattr(cli.typer, "prompt", _abort)
    monkeypatch.setattr(
        cli, "ArcAdapter", lambda **kw: (_ for _ in ()).throw(AssertionError("shouldn't reach the key check"))
    )
    monkeypatch.setattr(
        cli, "save_config", lambda config: (_ for _ in ()).throw(AssertionError("shouldn't save anything"))
    )

    with pytest.raises(SystemExit):
        cli._run_setup_wizard()

    # covers both an actual ctrl-c and a non-interactive run (piped
    # input, a script, CI) with no terminal to prompt against at all
    assert "interactive terminal" in capsys.readouterr().out


def test_run_setup_wizard_saves_config_on_a_working_key(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **kw: "sk-a-real-key")

    class _StubAdapter:
        def __init__(self, api_key):
            self.api_key = api_key

        def chat(self, model, messages, **kwargs):
            return _mock_completion("ok")

    monkeypatch.setattr(cli, "ArcAdapter", _StubAdapter)

    saved = {}
    monkeypatch.setattr(cli, "save_config", lambda config: saved.update(key=config.arc_api_key))
    monkeypatch.setattr(cli, "config_path", lambda: tmp_path / "config.toml")

    config = cli._run_setup_wizard()

    assert config.arc_api_key == "sk-a-real-key"
    assert saved["key"] == "sk-a-real-key"


def test_run_ask_cache_hit_skips_bandit_and_logs_it(monkeypatch):
    engine = _in_memory_engine()
    monkeypatch.setattr(cli, "get_engine", lambda: engine)
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: object())
    monkeypatch.setattr(
        cli, "classify", lambda text: Context(task_type=TaskType.CODE, length_bucket=LengthBucket.SHORT)
    )

    class _Hit:
        hit = True
        response = "binary search splits the array in half each step"
        model = "gpt-oss-120b"

    monkeypatch.setattr(cli, "cache_lookup", lambda query, engine: _Hit())

    def _boom(*args, **kwargs):
        raise AssertionError("bandit path should not run on a cache hit")

    monkeypatch.setattr(cli, "call_with_quality_gate", _boom)

    cli.run_ask("how does binary search work")

    with Session(engine) as session:
        rows = session.exec(select(RequestLog)).all()

    assert len(rows) == 1
    assert rows[0].cache_hit is True
    assert rows[0].model == "gpt-oss-120b"
    assert rows[0].quality_passed is True


def test_run_ask_cache_miss_logs_each_attempt_and_stores_on_success(monkeypatch):
    from arcus.quality.gate import AttemptDetail

    engine = _in_memory_engine()
    monkeypatch.setattr(cli, "get_engine", lambda: engine)
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: object())
    monkeypatch.setattr(
        cli, "classify", lambda text: Context(task_type=TaskType.CODE, length_bucket=LengthBucket.SHORT)
    )

    class _Miss:
        hit = False
        response = None
        model = None

    monkeypatch.setattr(cli, "cache_lookup", lambda query, engine: _Miss())
    monkeypatch.setattr(cli, "replay_history", lambda bandit, engine, mode: None)

    stored = {}
    monkeypatch.setattr(
        cli, "cache_store", lambda query, response, model, engine: stored.update(
            query=query, response=response, model=model
        )
    )

    class _Outcome:
        response = _mock_completion("binary search splits the array in half each step")
        model_used = "gpt-oss-120b"
        passed = True
        attempts = [
            AttemptDetail(
                model="GLM-5.3", passed=False, reward=0.1, latency_ms=500.0,
                propensity=1.0, issues=[],
            ),
            AttemptDetail(
                model="gpt-oss-120b", passed=True, reward=0.9, latency_ms=300.0,
                propensity=0.5, issues=[],
            ),
        ]
        issues = []

    monkeypatch.setattr(cli, "call_with_quality_gate", lambda *a, **kw: _Outcome())

    cli.run_ask("how does binary search work")

    with Session(engine) as session:
        rows = session.exec(select(RequestLog)).all()

    assert len(rows) == 2
    assert {r.model for r in rows} == {"GLM-5.3", "gpt-oss-120b"}
    failed_row = next(r for r in rows if r.model == "GLM-5.3")
    assert failed_row.quality_passed is False
    assert failed_row.cache_hit is False

    assert stored["model"] == "gpt-oss-120b"
    assert "binary search" in stored["response"]


def test_run_ask_does_not_cache_a_failed_outcome(monkeypatch):
    engine = _in_memory_engine()
    monkeypatch.setattr(cli, "get_engine", lambda: engine)
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: object())
    monkeypatch.setattr(
        cli, "classify", lambda text: Context(task_type=TaskType.CODE, length_bucket=LengthBucket.SHORT)
    )

    class _Miss:
        hit = False
        response = None
        model = None

    monkeypatch.setattr(cli, "cache_lookup", lambda query, engine: _Miss())
    monkeypatch.setattr(cli, "replay_history", lambda bandit, engine, mode: None)

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("a failed outcome should never get cached")

    monkeypatch.setattr(cli, "cache_store", _should_not_be_called)

    class _Outcome:
        response = _mock_completion(None, finish_reason="length")
        model_used = "gpt-oss-120b"
        passed = False
        attempts = []
        issues = []

    monkeypatch.setattr(cli, "call_with_quality_gate", lambda *a, **kw: _Outcome())

    cli.run_ask("how does binary search work")


def test_run_ask_handles_every_arm_api_erroring_without_crashing(monkeypatch, capsys):
    # call_with_quality_gate() returns response=None when every arm
    # errored out at the API level rather than returning a bad answer.
    # run_ask has to print a clean message instead of crashing trying to
    # read .choices off of None.
    engine = _in_memory_engine()
    monkeypatch.setattr(cli, "get_engine", lambda: engine)
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: object())
    monkeypatch.setattr(
        cli, "classify", lambda text: Context(task_type=TaskType.CODE, length_bucket=LengthBucket.SHORT)
    )

    class _Miss:
        hit = False
        response = None
        model = None

    monkeypatch.setattr(cli, "cache_lookup", lambda query, engine: _Miss())
    monkeypatch.setattr(cli, "replay_history", lambda bandit, engine, mode: None)
    monkeypatch.setattr(
        cli, "cache_store", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("shouldn't cache a None response"))
    )

    class _Outcome:
        response = None
        model_used = "gpt-oss-120b"
        passed = False
        attempts = []
        issues = []

    monkeypatch.setattr(cli, "call_with_quality_gate", lambda *a, **kw: _Outcome())

    cli.run_ask("how does binary search work")

    assert "no usable response" in capsys.readouterr().out


def test_run_ask_shows_a_vpn_message_when_arc_denies_access(monkeypatch, capsys):
    from arcus.quality.gate import QualityIssue

    engine = _in_memory_engine()
    monkeypatch.setattr(cli, "get_engine", lambda: engine)
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: object())
    monkeypatch.setattr(
        cli, "classify", lambda text: Context(task_type=TaskType.CODE, length_bucket=LengthBucket.SHORT)
    )

    class _Miss:
        hit = False
        response = None
        model = None

    monkeypatch.setattr(cli, "cache_lookup", lambda query, engine: _Miss())
    monkeypatch.setattr(cli, "replay_history", lambda bandit, engine, mode: None)

    class _Outcome:
        response = None
        model_used = "gpt-oss-120b"
        passed = False
        attempts = []
        issues = [QualityIssue("permission_denied", "VPN required")]

    monkeypatch.setattr(cli, "call_with_quality_gate", lambda *a, **kw: _Outcome())

    cli.run_ask("how does binary search work")

    assert "VPN" in capsys.readouterr().out


def test_build_image_content_encodes_the_file_and_keeps_the_prompt(tmp_path):
    image_path = tmp_path / "screenshot.png"
    image_path.write_bytes(b"not a real png, just test bytes")

    content = cli._build_image_content("what's in this image?", str(image_path))

    assert content[0] == {"type": "text", "text": "what's in this image?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_image_content_rejects_a_non_image_file(tmp_path):
    text_path = tmp_path / "notes.txt"
    text_path.write_text("just some text")

    with pytest.raises(ValueError):
        cli._build_image_content("describe this", str(text_path))


def test_run_image_ask_sends_image_content_to_the_vision_model_only(monkeypatch, tmp_path, capsys):
    image_path = tmp_path / "screenshot.png"
    image_path.write_bytes(b"fake png bytes")

    monkeypatch.setattr(cli, "get_engine", lambda: _in_memory_engine())
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: object())
    monkeypatch.setattr(cli, "replay_history", lambda bandit, engine, mode: None)
    monkeypatch.setattr(
        cli, "classify", lambda text: Context(task_type=TaskType.GENERAL, length_bucket=LengthBucket.SHORT)
    )

    captured = {}

    def _fake_gate(adapter, bandit, context_key, messages):
        captured["messages"] = messages
        captured["arms"] = bandit.arms

        class _Outcome:
            response = _mock_completion("that's a stack trace")
            model_used = "Kimi-K3"
            passed = True
            attempts = []
            issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)

    cli.run_image_ask("what's wrong here?", str(image_path))

    assert captured["arms"] == ["Kimi-K3"]
    sent_content = captured["messages"][0]["content"]
    assert sent_content[0]["text"] == "what's wrong here?"
    assert sent_content[1]["type"] == "image_url"
    assert "that's a stack trace" in capsys.readouterr().out


def test_run_image_ask_refuses_when_the_vision_model_is_unavailable(monkeypatch, tmp_path, capsys):
    image_path = tmp_path / "screenshot.png"
    image_path.write_bytes(b"fake png bytes")

    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "get_engine", lambda: _in_memory_engine())
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: object())
    monkeypatch.setattr(cli, "known_arms", lambda adapter: ["gpt-oss-120b"])

    with pytest.raises(SystemExit):
        cli.run_image_ask("what's wrong here?", str(image_path))

    assert "isn't currently available" in capsys.readouterr().out


def test_run_image_ask_errors_cleanly_on_a_missing_file(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "get_engine", lambda: _in_memory_engine())
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: object())

    with pytest.raises(SystemExit):
        cli.run_image_ask("what's this?", "/no/such/file.png")

    assert "couldn't read" in capsys.readouterr().out


class _FakeDocAdapter:
    def __init__(self, file_id="file-abc123", upload_error=None, delete_error=None):
        self.file_id = file_id
        self.upload_error = upload_error
        self.delete_error = delete_error
        self.deleted_ids = []

    def upload_file(self, path):
        if self.upload_error:
            raise self.upload_error
        return self.file_id

    def delete_file(self, file_id):
        if self.delete_error:
            raise self.delete_error
        self.deleted_ids.append(file_id)


def test_run_doc_ask_attaches_the_uploaded_file_id_to_the_request(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "get_engine", lambda: _in_memory_engine())
    fake_adapter = _FakeDocAdapter()
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: fake_adapter)
    monkeypatch.setattr(cli, "replay_history", lambda bandit, engine, mode: None)
    monkeypatch.setattr(
        cli, "classify", lambda text: Context(task_type=TaskType.GENERAL, length_bucket=LengthBucket.SHORT)
    )

    captured = {}

    def _fake_gate(adapter, bandit, context_key, messages, **kwargs):
        captured["extra_body"] = kwargs.get("extra_body")
        captured["arms"] = bandit.arms

        class _Outcome:
            response = _mock_completion("the summary is X")
            model_used = "gpt-oss-120b"
            passed = True
            attempts = []
            issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)

    cli.run_doc_ask("summarize this", "/some/doc.pdf")

    assert captured["extra_body"] == {"files": [{"type": "file", "id": "file-abc123"}]}
    assert set(captured["arms"]) == {"gpt-oss-120b", "GLM-5.3", "Kimi-K3", "DeepSeek-V4-Flash"}
    assert "the summary is X" in capsys.readouterr().out
    assert fake_adapter.deleted_ids == ["file-abc123"]


def test_run_doc_ask_still_cleans_up_the_file_when_the_answer_fails(monkeypatch):
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "get_engine", lambda: _in_memory_engine())
    fake_adapter = _FakeDocAdapter()
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: fake_adapter)
    monkeypatch.setattr(cli, "replay_history", lambda bandit, engine, mode: None)
    monkeypatch.setattr(
        cli, "classify", lambda text: Context(task_type=TaskType.GENERAL, length_bucket=LengthBucket.SHORT)
    )

    class _Outcome:
        response = None
        model_used = "gpt-oss-120b"
        passed = False
        attempts = []
        issues = []

    monkeypatch.setattr(cli, "call_with_quality_gate", lambda *a, **kw: _Outcome())

    cli.run_doc_ask("summarize this", "/some/doc.pdf")

    assert fake_adapter.deleted_ids == ["file-abc123"]


def test_run_doc_ask_survives_a_cleanup_failure(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "get_engine", lambda: _in_memory_engine())
    fake_adapter = _FakeDocAdapter(delete_error=httpx.HTTPError("delete failed"))
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: fake_adapter)
    monkeypatch.setattr(cli, "replay_history", lambda bandit, engine, mode: None)
    monkeypatch.setattr(
        cli, "classify", lambda text: Context(task_type=TaskType.GENERAL, length_bucket=LengthBucket.SHORT)
    )

    class _Outcome:
        response = _mock_completion("the answer")
        model_used = "gpt-oss-120b"
        passed = True
        attempts = []
        issues = []

    monkeypatch.setattr(cli, "call_with_quality_gate", lambda *a, **kw: _Outcome())

    cli.run_doc_ask("summarize this", "/some/doc.pdf")  # should not raise

    assert "the answer" in capsys.readouterr().out


def test_run_doc_ask_errors_cleanly_when_the_file_cant_be_read(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "get_engine", lambda: _in_memory_engine())
    fake_adapter = _FakeDocAdapter(upload_error=FileNotFoundError("no such file"))
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: fake_adapter)

    with pytest.raises(SystemExit):
        cli.run_doc_ask("summarize this", "/no/such/doc.pdf")

    assert "couldn't read" in capsys.readouterr().out


def test_run_doc_ask_errors_cleanly_on_an_upload_failure(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "get_engine", lambda: _in_memory_engine())
    fake_adapter = _FakeDocAdapter(upload_error=httpx.HTTPError("upload failed"))
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: fake_adapter)

    with pytest.raises(SystemExit):
        cli.run_doc_ask("summarize this", "/some/doc.pdf")

    assert "couldn't upload" in capsys.readouterr().out


def test_run_web_ask_attaches_the_websearch_tool_id_and_uses_legacy_arms(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "get_engine", lambda: _in_memory_engine())
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: object())
    monkeypatch.setattr(cli, "replay_history", lambda bandit, engine, mode: None)
    monkeypatch.setattr(
        cli, "classify", lambda text: Context(task_type=TaskType.GENERAL, length_bucket=LengthBucket.SHORT)
    )

    captured = {}

    def _fake_gate(adapter, bandit, context_key, messages, **kwargs):
        captured["extra_body"] = kwargs.get("extra_body")
        captured["arms"] = bandit.arms

        class _Outcome:
            response = _mock_completion("current info here [1]")
            model_used = cli._WEB_SEARCH_MODELS[0]
            passed = True
            attempts = []
            issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)

    cli.run_web_ask("what's the latest on X")

    assert captured["extra_body"] == {"tool_ids": ["server:websearch"]}
    assert captured["arms"] == cli._WEB_SEARCH_MODELS
    assert "current info here" in capsys.readouterr().out


def test_main_dispatches_to_doc_ask(monkeypatch):
    called = []
    monkeypatch.setattr(
        cli,
        "run_doc_ask",
        lambda prompt, doc_path, random_mode, model_override=None: called.append(
            (prompt, doc_path, random_mode, model_override)
        ),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    cli.main(["--doc", "paper.pdf", "summarize", "this"])

    assert called == [("summarize this", "paper.pdf", False, None)]


def test_main_dispatches_to_web_ask(monkeypatch):
    called = []
    monkeypatch.setattr(
        cli,
        "run_web_ask",
        lambda prompt, random_mode, model_override=None: called.append((prompt, random_mode, model_override)),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    cli.main(["--web", "what's", "new"])

    assert called == [("what's new", False, None)]


def test_main_rejects_combining_image_and_doc_flags(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "run_image_ask", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("shouldn't be called"))
    )
    monkeypatch.setattr(
        cli, "run_doc_ask", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("shouldn't be called"))
    )

    cli.main(["--image", "a.png", "--doc", "b.pdf", "question"])

    assert "only one of" in capsys.readouterr().out


def _canned_input(responses):
    # feeds builtins.input() one canned reply per call, then raises
    # EOFError once exhausted, same as a real ctrl-d would.
    it = iter(responses)

    def _input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    return _input


def _chat_setup(monkeypatch, engine):
    monkeypatch.setattr(cli, "get_engine", lambda: engine)
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: object())
    monkeypatch.setattr(cli, "replay_history", lambda bandit, engine, mode: None)
    monkeypatch.setattr(
        cli, "classify", lambda text: Context(task_type=TaskType.CODE, length_bucket=LengthBucket.SHORT)
    )


def test_run_chat_remembers_prior_turns(monkeypatch):
    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)

    calls = []

    def _fake_gate(adapter, bandit, context_key, messages):
        calls.append([dict(m) for m in messages])

        class _Outcome:
            response = _mock_completion(f"reply {len(calls)}")
            model_used = "gpt-oss-120b"
            passed = True
            attempts = []
            issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)
    monkeypatch.setattr("builtins.input", _canned_input(["first question", "what about in python?", "exit"]))

    cli.run_chat()

    assert len(calls) == 2
    assert calls[0] == [{"role": "user", "content": "first question"}]
    assert calls[1] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "what about in python?"},
    ]


def test_run_chat_drops_unanswered_turn_from_history(monkeypatch):
    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)

    calls = []

    def _fake_gate(adapter, bandit, context_key, messages):
        calls.append([dict(m) for m in messages])

        if len(calls) == 1:
            class _Outcome:
                response = None
                model_used = "gpt-oss-120b"
                passed = False
                attempts = []
                issues = []
        else:
            class _Outcome:
                response = _mock_completion("second reply")
                model_used = "gpt-oss-120b"
                passed = True
                attempts = []
                issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)
    monkeypatch.setattr("builtins.input", _canned_input(["broken question", "second question", "exit"]))

    cli.run_chat()

    assert len(calls) == 2
    assert calls[1] == [{"role": "user", "content": "second question"}]


def test_run_chat_shows_a_vpn_message_when_arc_denies_access(monkeypatch, capsys):
    from arcus.quality.gate import QualityIssue

    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)

    class _Outcome:
        response = None
        model_used = "gpt-oss-120b"
        passed = False
        attempts = []
        issues = [QualityIssue("permission_denied", "VPN required")]

    monkeypatch.setattr(cli, "call_with_quality_gate", lambda *a, **kw: _Outcome())
    monkeypatch.setattr("builtins.input", _canned_input(["a question", "exit"]))

    cli.run_chat()

    assert "VPN" in capsys.readouterr().out


def test_run_chat_exits_on_exit_keyword(monkeypatch):
    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("gate should never be called before exit")

    monkeypatch.setattr(cli, "call_with_quality_gate", _should_not_be_called)
    monkeypatch.setattr("builtins.input", _canned_input(["exit"]))

    cli.run_chat()


def test_run_chat_exits_cleanly_on_eof(monkeypatch):
    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("gate should never be called before eof")

    monkeypatch.setattr(cli, "call_with_quality_gate", _should_not_be_called)
    monkeypatch.setattr("builtins.input", _canned_input([]))

    cli.run_chat()


def test_run_chat_skips_blank_input(monkeypatch):
    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)

    calls = []

    def _fake_gate(adapter, bandit, context_key, messages):
        calls.append(messages)

        class _Outcome:
            response = _mock_completion("ok")
            model_used = "gpt-oss-120b"
            passed = True
            attempts = []
            issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)
    monkeypatch.setattr("builtins.input", _canned_input(["", "   ", "real question", "exit"]))

    cli.run_chat()

    assert len(calls) == 1


def test_run_chat_logs_shared_conversation_id_and_incrementing_turn_index(monkeypatch):
    from arcus.quality.gate import AttemptDetail

    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)

    def _fake_gate(adapter, bandit, context_key, messages):
        class _Outcome:
            response = _mock_completion("ok")
            model_used = "gpt-oss-120b"
            passed = True
            attempts = [
                AttemptDetail(model="gpt-oss-120b", passed=True, reward=0.9, latency_ms=100.0, propensity=1.0, issues=[]),
            ]
            issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)
    monkeypatch.setattr("builtins.input", _canned_input(["q1", "q2", "exit"]))

    cli.run_chat()

    with Session(engine) as session:
        rows = session.exec(select(RequestLog)).all()

    assert len(rows) == 2
    assert rows[0].conversation_id is not None
    assert rows[0].conversation_id == rows[1].conversation_id
    assert {r.turn_index for r in rows} == {0, 1}


def test_run_chat_saves_a_transcript_when_requested(monkeypatch, tmp_path):
    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)

    def _fake_gate(adapter, bandit, context_key, messages):
        class _Outcome:
            response = _mock_completion("the answer")
            model_used = "gpt-oss-120b"
            passed = True
            attempts = []
            issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)
    monkeypatch.setattr("builtins.input", _canned_input(["hello there", "exit"]))

    save_path = tmp_path / "transcript.md"
    cli.run_chat(save_path=str(save_path))

    saved = save_path.read_text()
    assert "hello there" in saved
    assert "the answer" in saved
    assert "gpt-oss-120b" in saved


def test_run_chat_skips_writing_a_transcript_with_zero_turns(monkeypatch, tmp_path):
    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)
    monkeypatch.setattr("builtins.input", _canned_input(["exit"]))

    save_path = tmp_path / "transcript.md"
    cli.run_chat(save_path=str(save_path))

    assert not save_path.exists()


def test_trim_history_drops_oldest_turns_in_pairs():
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(23)]

    trimmed = cli._trim_history(messages, max_messages=20)

    assert len(trimmed) == 19
    assert trimmed[0]["role"] == "user"


def test_trim_history_leaves_short_history_untouched():
    messages = [{"role": "user", "content": "hi"}]
    assert cli._trim_history(messages, max_messages=20) == messages


def test_trim_history_handles_multimodal_content_the_same_as_text():
    # an --image turn's content is a list of content parts, not a plain
    # string, trimming shouldn't care, it only ever looks at position
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(23)]
    messages[-1] = {"role": "user", "content": [{"type": "text", "text": "what's wrong here"}]}

    trimmed = cli._trim_history(messages, max_messages=20)

    assert len(trimmed) == 19
    assert trimmed[0]["role"] == "user"
    assert trimmed[-1]["content"] == [{"type": "text", "text": "what's wrong here"}]


def test_run_models_lists_the_live_catalog_and_flags_routed_ones(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(
        cli,
        "ArcAdapter",
        lambda api_key: type(
            "_A", (), {"list_models": lambda self: ["gpt-oss-120b", "gpt-oss-120b-thinking-high", "Kimi-K3"]}
        )(),
    )

    cli.run_models()

    output = capsys.readouterr().out
    assert "gpt-oss-120b-thinking-high" in output
    assert "docs.arc.vt.edu" in output


def test_run_models_exits_cleanly_when_arc_is_unreachable(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))

    class _Broken:
        def list_models(self):
            raise ConnectionError("arc is unreachable")

    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: _Broken())

    with pytest.raises(SystemExit):
        cli.run_models()

    assert "couldn't reach" in capsys.readouterr().out


def test_main_dispatches_to_models(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "run_models", lambda: called.append(True))
    cli.main(["models"])
    assert called == [True]


def test_run_config_shows_masked_key_and_algorithm(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="sk-verysecret1234", bandit_algorithm="ucb1")
    )

    cli.run_config([])

    out = capsys.readouterr().out
    assert "sk-verysecret1234" not in out
    assert "1234" in out
    assert "ucb1" in out


def test_run_config_set_updates_and_saves(monkeypatch):
    saved = {}
    monkeypatch.setattr(
        cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k", bandit_algorithm="thompson")
    )
    monkeypatch.setattr(cli, "save_config", lambda config: saved.update(algorithm=config.bandit_algorithm))

    cli.run_config(["set", "bandit_algorithm", "ucb1"])

    assert saved["algorithm"] == "ucb1"


def test_run_config_set_rejects_an_invalid_algorithm(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))

    with pytest.raises(SystemExit):
        cli.run_config(["set", "bandit_algorithm", "not-a-real-one"])

    assert "isn't a valid" in capsys.readouterr().out


def test_run_config_set_enables_reasoning_variants(monkeypatch, capsys):
    saved = {}
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(
        cli, "save_config", lambda config: saved.update(enabled=config.enable_reasoning_variants)
    )

    cli.run_config(["set", "enable_reasoning_variants", "true"])

    assert saved["enabled"] is True
    assert "hasn't been confirmed against a live ARC key" in capsys.readouterr().out


def test_run_config_set_rejects_a_non_boolean_reasoning_variants_value(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))

    with pytest.raises(SystemExit):
        cli.run_config(["set", "enable_reasoning_variants", "yes"])

    assert "isn't valid" in capsys.readouterr().out


def test_run_config_unrecognized_args_print_usage(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))

    with pytest.raises(SystemExit):
        cli.run_config(["blah"])

    assert "usage" in capsys.readouterr().out


def test_main_dispatches_to_config(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "run_config", lambda args: called.append(args))
    cli.main(["config", "set", "bandit_algorithm", "ucb1"])
    assert called == [["set", "bandit_algorithm", "ucb1"]]


def test_run_stats_reports_cache_hit_rate(monkeypatch, capsys):
    engine = _in_memory_engine()
    monkeypatch.setattr(cli, "get_engine", lambda: engine)

    from arcus.storage.db import log_request

    log_request(
        prompt="a", task_type="code", length_bucket="short", model="gpt-oss-120b",
        mode="bandit", reward=0.8, latency_ms=200, cache_hit=False, quality_passed=True, engine=engine,
    )
    log_request(
        prompt="b", task_type="code", length_bucket="short", model="gpt-oss-120b",
        mode="bandit", reward=None, cache_hit=True, quality_passed=True, engine=engine,
    )

    cli.run_stats()

    output = capsys.readouterr().out
    assert "cache hit rate" in output
    assert "1/2" in output


def test_run_stats_handles_empty_log(monkeypatch, capsys):
    engine = _in_memory_engine()
    monkeypatch.setattr(cli, "get_engine", lambda: engine)

    cli.run_stats()

    assert "no requests logged" in capsys.readouterr().out


# --- run_eval ---------------------------------------------------------------


def _log_eval_rows(engine, n, model="gpt-oss-120b", reward=0.8, propensity=0.5):
    from arcus.storage.db import log_request

    for i in range(n):
        log_request(
            prompt=f"question {i}",
            task_type="code",
            length_bucket="short",
            model=model,
            mode="bandit",
            reward=reward,
            propensity=propensity,
            latency_ms=200,
            engine=engine,
        )


def test_run_eval_reports_no_data_when_log_is_empty(monkeypatch, capsys):
    engine = _in_memory_engine()
    monkeypatch.setattr(cli, "get_engine", lambda: engine)

    cli.run_eval()

    assert "no eligible logged requests" in capsys.readouterr().out


def test_run_eval_flags_a_thin_sample_but_still_prints_the_table(monkeypatch, capsys):
    engine = _in_memory_engine()
    monkeypatch.setattr(cli, "get_engine", lambda: engine)
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: object())
    _log_eval_rows(engine, 3)

    cli.run_eval()

    output = capsys.readouterr().out
    assert "illustrative" in output
    assert "offline policy evaluation" in output
    assert "greedy" in output


def test_run_eval_warns_at_one_below_the_floor_but_not_at_the_floor(monkeypatch, capsys):
    engine = _in_memory_engine()
    monkeypatch.setattr(cli, "get_engine", lambda: engine)
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: object())
    _log_eval_rows(engine, cli._MIN_EVAL_EXAMPLES - 1)

    cli.run_eval()

    assert "illustrative" in capsys.readouterr().out


def test_run_eval_still_works_when_arcs_catalog_is_unreachable(monkeypatch, capsys, tmp_path):
    # arcus eval is fundamentally a local operation, reading your own
    # logged history, it shouldn't be dead in the water just because
    # you're off-VPN when you run it. undoes the autouse known_arms
    # fake for this one test, this is specifically about proving the
    # real model_catalog fallback (already unit tested on its own in
    # tests/routing/test_model_catalog.py) actually gets exercised by
    # run_eval, not just that a fake never gets called.
    from arcus.routing.model_catalog import known_arms as real_known_arms

    monkeypatch.setattr(cli, "known_arms", real_known_arms)
    monkeypatch.setattr("arcus.routing.model_catalog.user_cache_dir", lambda name: str(tmp_path))

    engine = _in_memory_engine()
    monkeypatch.setattr(cli, "get_engine", lambda: engine)
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))

    class _Unreachable:
        def list_models(self):
            raise ConnectionError("offline, no VPN")

    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: _Unreachable())
    _log_eval_rows(engine, 5)

    cli.run_eval()

    output = capsys.readouterr().out
    assert "offline policy evaluation" in output
    flattened = "".join(output.split())
    assert "alwaysgpt-oss-120b" in flattened


def test_run_eval_omits_the_thin_sample_warning_above_the_floor(monkeypatch, capsys):
    engine = _in_memory_engine()
    monkeypatch.setattr(cli, "get_engine", lambda: engine)
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: object())
    _log_eval_rows(engine, cli._MIN_EVAL_EXAMPLES)

    cli.run_eval()

    output = capsys.readouterr().out
    # a table cell can legitimately wrap onto its own line depending on
    # console width, strip whitespace so the assertion doesn't depend on
    # exactly where Rich decided to break a long policy name
    flattened = "".join(output.split())
    assert "illustrative" not in output
    assert "alwaysgpt-oss-120b" in flattened
    assert "alwaysGLM-5.3" in flattened


def test_main_prints_version(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_version", lambda: "1.2.3")
    cli.main(["--version"])
    assert "1.2.3" in capsys.readouterr().out


def test_main_prints_version_with_short_flag(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_version", lambda: "1.2.3")
    cli.main(["-V"])
    assert "1.2.3" in capsys.readouterr().out


def test_version_falls_back_when_not_installed(monkeypatch):
    def _raise(name):
        raise cli.PackageNotFoundError(name)

    monkeypatch.setattr(cli, "version", _raise)
    assert "source" in cli._version()


def test_main_dispatches_to_completion(monkeypatch, capsys):
    cli.main(["--completion", "zsh"])
    out = capsys.readouterr().out
    assert "compdef" in out


def test_completion_supports_bash(capsys):
    cli.main(["--completion", "bash"])
    out = capsys.readouterr().out
    assert "complete -F" in out


def test_completion_rejects_an_unknown_shell(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--completion", "fish"])
    assert "no completion script" in capsys.readouterr().out


def test_main_dispatches_to_stats(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "run_stats", lambda: called.append(True))
    cli.main(["stats"])
    assert called == [True]


def test_main_dispatches_to_eval(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "run_eval", lambda: called.append(True))
    cli.main(["eval"])
    assert called == [True]


def test_main_dispatches_to_chat(monkeypatch):
    called = []
    monkeypatch.setattr(
        cli, "run_chat", lambda random_mode, save_path=None: called.append((random_mode, save_path))
    )
    cli.main(["chat", "--random"])
    assert called == [(True, None)]


def test_main_dispatches_to_chat_with_save_path(monkeypatch):
    called = []
    monkeypatch.setattr(
        cli, "run_chat", lambda random_mode, save_path=None: called.append((random_mode, save_path))
    )
    cli.main(["chat", "--save", "transcript.md"])
    assert called == [(False, "transcript.md")]


def test_main_strips_random_flag_and_passes_it_through(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "_build_prompt", lambda args: "explain recursion")
    monkeypatch.setattr(
        cli,
        "run_ask",
        lambda prompt, random_mode, model_override=None: seen.update(
            prompt=prompt, random_mode=random_mode, model_override=model_override
        ),
    )

    cli.main(["--random", "explain", "recursion"])

    assert seen == {"prompt": "explain recursion", "random_mode": True, "model_override": None}


def test_main_prints_usage_for_empty_input(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_build_prompt", lambda args: "")
    cli.main([])
    assert "usage" in capsys.readouterr().out


def test_extract_flag_value_returns_none_when_absent():
    assert cli._extract_flag_value(["a", "b"], "--image") is None


def test_extract_flag_value_returns_none_when_flag_is_the_last_arg():
    assert cli._extract_flag_value(["a", "--image"], "--image") is None


def test_extract_flag_value_returns_the_following_arg():
    assert cli._extract_flag_value(["ask", "--image", "shot.png", "more"], "--image") == "shot.png"


def test_strip_flag_and_value_removes_both():
    assert cli._strip_flag_and_value(["a", "--image", "shot.png", "b"], "--image") == ["a", "b"]


def test_strip_flag_and_value_is_a_noop_when_absent():
    assert cli._strip_flag_and_value(["a", "b"], "--image") == ["a", "b"]


def test_main_strips_image_flag_and_dispatches_to_image_ask(monkeypatch):
    called = []
    monkeypatch.setattr(
        cli,
        "run_image_ask",
        lambda prompt, image_path, model_override=None: called.append((prompt, image_path, model_override)),
    )
    monkeypatch.setattr(
        cli, "run_ask", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("run_ask shouldn't fire here"))
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    cli.main(["explain", "--image", "shot.png", "this", "error"])

    assert called == [("explain this error", "shot.png", None)]


def test_main_strips_model_flag_and_dispatches_to_run_ask(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        cli,
        "run_ask",
        lambda prompt, random_mode, model_override=None: seen.update(
            prompt=prompt, random_mode=random_mode, model_override=model_override
        ),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    cli.main(["--model", "GLM-5.3", "explain", "recursion"])

    assert seen == {"prompt": "explain recursion", "random_mode": False, "model_override": "GLM-5.3"}


def test_main_rejects_model_combined_with_random_before_dispatching_anywhere(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "run_ask", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("shouldn't be called"))
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    cli.main(["--model", "GLM-5.3", "--random", "explain recursion"])

    assert "contradict" in capsys.readouterr().out


# --- _parse_request_flags / _validate_mode_flags -------------------------


def test_parse_request_flags_pulls_out_every_flag_and_leaves_the_question():
    flags = cli._parse_request_flags(
        ["--model", "GLM-5.3", "--doc", "paper.pdf", "--random", "summarize", "this"]
    )
    assert flags.model_override == "GLM-5.3"
    assert flags.doc_path == "paper.pdf"
    assert flags.image_path is None
    assert flags.web_mode is False
    assert flags.random_mode is True
    assert flags.remaining == ["summarize", "this"]


def test_parse_request_flags_defaults_when_nothing_present():
    flags = cli._parse_request_flags(["just", "a", "question"])
    assert flags.model_override is None
    assert flags.doc_path is None
    assert flags.image_path is None
    assert flags.web_mode is False
    assert flags.random_mode is False
    assert flags.remaining == ["just", "a", "question"]


def test_validate_mode_flags_rejects_more_than_one_attachment_mode():
    flags = cli._parse_request_flags(["--doc", "a.pdf", "--web", "question"])
    error = cli._validate_mode_flags(flags)
    assert error is not None
    assert "only one of" in error


def test_validate_mode_flags_rejects_model_combined_with_random():
    flags = cli._parse_request_flags(["--model", "GLM-5.3", "--random", "question"])
    error = cli._validate_mode_flags(flags)
    assert error is not None
    assert "contradict" in error


def test_validate_mode_flags_allows_model_alone():
    flags = cli._parse_request_flags(["--model", "GLM-5.3", "question"])
    assert cli._validate_mode_flags(flags) is None


# --- _resolve_model_override -----------------------------------------------


def test_resolve_model_override_accepts_a_live_model():
    assert cli._resolve_model_override(object(), "GLM-5.3") is None


def test_resolve_model_override_rejects_a_model_arc_isnt_serving():
    error = cli._resolve_model_override(object(), "gpt-5-nonexistent")
    assert error is not None
    assert "gpt-5-nonexistent" in error


def test_resolve_model_override_for_web_mode_requires_a_legacy_tool_calling_arm():
    valid = cli._WEB_SEARCH_MODELS[0]
    assert cli._resolve_model_override(object(), valid, web_mode=True) is None

    error = cli._resolve_model_override(object(), "GLM-5.3", web_mode=True)
    assert error is not None
    assert "doesn't do --web" in error


def test_resolve_model_override_for_image_mode_requires_the_vision_model():
    assert cli._resolve_model_override(object(), cli._VISION_MODEL, image_mode=True) is None

    error = cli._resolve_model_override(object(), "GLM-5.3", image_mode=True)
    assert error is not None
    assert cli._VISION_MODEL in error


# --- _arms_for_reasoning_context / _build_bandit ---------------------------


_BASE_ARMS = ["gpt-oss-120b", "GLM-5.3", "Kimi-K3", "DeepSeek-V4-Flash"]


def test_arms_for_reasoning_context_adds_variants_for_a_code_context():
    result = cli._arms_for_reasoning_context(_BASE_ARMS, "code:short", object())
    assert set(_BASE_ARMS).issubset(set(result))
    assert set(cli._REASONING_VARIANTS).issubset(set(result))


def test_arms_for_reasoning_context_leaves_non_reasoning_contexts_alone():
    result = cli._arms_for_reasoning_context(_BASE_ARMS, "writing:short", object())
    assert result == _BASE_ARMS


def test_build_bandit_ignores_context_when_reasoning_variants_are_disabled():
    bandit = cli._build_bandit(_BASE_ARMS, cli.EpsilonGreedyBandit, object(), enable_reasoning_variants=False)
    assert bandit.arms_for("code:short") == _BASE_ARMS
    assert bandit.arms_for("writing:short") == _BASE_ARMS


def test_build_bandit_adds_variants_only_for_reasoning_contexts_when_enabled():
    bandit = cli._build_bandit(_BASE_ARMS, cli.EpsilonGreedyBandit, object(), enable_reasoning_variants=True)

    code_arms = bandit.arms_for("code:short")
    assert set(_BASE_ARMS).issubset(set(code_arms))
    assert len(code_arms) > len(_BASE_ARMS)
    assert bandit.arms_for("writing:short") == _BASE_ARMS


# --- run_ask(model_override=...) -------------------------------------------


def test_run_ask_with_model_override_forces_a_single_arm_and_skips_the_cache(monkeypatch):
    from arcus.quality.gate import AttemptDetail

    engine = _in_memory_engine()
    monkeypatch.setattr(cli, "get_engine", lambda: engine)
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: object())
    monkeypatch.setattr(cli, "replay_history", lambda bandit, engine, mode: None)
    monkeypatch.setattr(
        cli, "classify", lambda text: Context(task_type=TaskType.CODE, length_bucket=LengthBucket.SHORT)
    )
    monkeypatch.setattr(
        cli, "cache_lookup", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("cache shouldn't be checked"))
    )
    monkeypatch.setattr(
        cli, "cache_store", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("cache shouldn't be written"))
    )

    captured = {}

    def _fake_gate(adapter, bandit, context_key, messages):
        captured["arms"] = bandit.arms

        class _Outcome:
            response = _mock_completion("a specific answer")
            model_used = "GLM-5.3"
            passed = True
            attempts = [
                AttemptDetail(
                    model="GLM-5.3", passed=True, reward=0.9, latency_ms=120.0, propensity=1.0, issues=[]
                )
            ]
            issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)

    cli.run_ask("explain recursion", model_override="GLM-5.3")

    assert captured["arms"] == ["GLM-5.3"]

    with Session(engine) as session:
        rows = session.exec(select(RequestLog)).all()
    assert len(rows) == 1
    assert rows[0].model == "GLM-5.3"
    assert rows[0].mode == "manual"


def test_run_ask_with_model_override_rejects_an_unknown_model(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k"))
    monkeypatch.setattr(cli, "get_engine", lambda: _in_memory_engine())
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: object())
    monkeypatch.setattr(
        cli, "call_with_quality_gate", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("shouldn't be called"))
    )

    with pytest.raises(SystemExit):
        cli.run_ask("explain recursion", model_override="not-a-real-model")

    assert "not-a-real-model" in capsys.readouterr().out


def test_run_ask_adds_reasoning_variants_for_a_code_prompt_when_enabled(monkeypatch):
    engine = _in_memory_engine()
    monkeypatch.setattr(cli, "get_engine", lambda: engine)
    monkeypatch.setattr(
        cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k", enable_reasoning_variants=True)
    )
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: object())
    monkeypatch.setattr(cli, "replay_history", lambda bandit, engine, mode: None)
    monkeypatch.setattr(
        cli, "classify", lambda text: Context(task_type=TaskType.CODE, length_bucket=LengthBucket.SHORT)
    )

    class _Miss:
        hit = False
        response = None
        model = None

    monkeypatch.setattr(cli, "cache_lookup", lambda query, engine: _Miss())

    captured = {}

    def _fake_gate(adapter, bandit, context_key, messages):
        captured["arms"] = bandit.arms_for(context_key)

        class _Outcome:
            response = _mock_completion("here's a function")
            model_used = "gpt-oss-120b"
            passed = True
            attempts = []
            issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)

    cli.run_ask("write a function to reverse a linked list")

    assert set(cli._REASONING_VARIANTS).issubset(set(captured["arms"]))


def test_run_ask_leaves_a_writing_prompt_on_the_base_four_even_when_enabled(monkeypatch):
    engine = _in_memory_engine()
    monkeypatch.setattr(cli, "get_engine", lambda: engine)
    monkeypatch.setattr(
        cli, "_ensure_config", lambda: ArcusConfig(arc_api_key="k", enable_reasoning_variants=True)
    )
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: object())
    monkeypatch.setattr(cli, "replay_history", lambda bandit, engine, mode: None)
    monkeypatch.setattr(
        cli, "classify", lambda text: Context(task_type=TaskType.WRITING, length_bucket=LengthBucket.SHORT)
    )

    class _Miss:
        hit = False
        response = None
        model = None

    monkeypatch.setattr(cli, "cache_lookup", lambda query, engine: _Miss())

    captured = {}

    def _fake_gate(adapter, bandit, context_key, messages):
        captured["arms"] = bandit.arms_for(context_key)

        class _Outcome:
            response = _mock_completion("here's a draft")
            model_used = "gpt-oss-120b"
            passed = True
            attempts = []
            issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)

    cli.run_ask("write me a short poem")

    assert captured["arms"] == ["gpt-oss-120b", "GLM-5.3", "Kimi-K3", "DeepSeek-V4-Flash"]


# --- run_chat inline --doc / --web / --image / --model ---------------------


def test_run_chat_doc_flag_attaches_a_file_for_that_turn_only(monkeypatch):
    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)
    fake_adapter = _FakeDocAdapter()
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: fake_adapter)

    captured = []

    def _fake_gate(adapter, bandit, context_key, messages, **kwargs):
        captured.append(kwargs.get("extra_body"))

        class _Outcome:
            response = _mock_completion("the doc says X")
            model_used = "gpt-oss-120b"
            passed = True
            attempts = []
            issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)
    monkeypatch.setattr(
        "builtins.input", _canned_input(["--doc paper.pdf summarize this", "a plain follow-up", "exit"])
    )

    cli.run_chat()

    assert captured[0] == {"files": [{"type": "file", "id": "file-abc123"}]}
    assert captured[1] is None
    assert fake_adapter.deleted_ids == ["file-abc123"]


def test_run_chat_doc_flag_cleans_up_the_file_even_when_the_answer_fails(monkeypatch):
    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)
    fake_adapter = _FakeDocAdapter()
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: fake_adapter)

    class _Outcome:
        response = None
        model_used = "gpt-oss-120b"
        passed = False
        attempts = []
        issues = []

    monkeypatch.setattr(cli, "call_with_quality_gate", lambda *a, **kw: _Outcome())
    monkeypatch.setattr(
        "builtins.input", _canned_input(["--doc paper.pdf summarize this", "a follow-up", "exit"])
    )

    cli.run_chat()

    # the failed turn's file still gets cleaned up, and the unanswered
    # turn doesn't poison history for the next one
    assert fake_adapter.deleted_ids == ["file-abc123"]


def test_run_chat_recovers_from_an_unclosed_quote_and_keeps_going(monkeypatch):
    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)

    calls = []

    def _fake_gate(adapter, bandit, context_key, messages):
        calls.append(1)

        class _Outcome:
            response = _mock_completion("fine")
            model_used = "gpt-oss-120b"
            passed = True
            attempts = []
            issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)
    monkeypatch.setattr(
        "builtins.input",
        _canned_input(['--doc "unclosed.pdf summarize this', "a normal question", "exit"]),
    )

    cli.run_chat()  # should not raise

    # the malformed line never reaches the gate, the next one still does
    assert len(calls) == 1


def test_run_chat_web_flag_routes_that_turn_through_legacy_search_arms(monkeypatch):
    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)

    captured = {}

    def _fake_gate(adapter, bandit, context_key, messages, **kwargs):
        captured["extra_body"] = kwargs.get("extra_body")
        captured["arms"] = bandit.arms

        class _Outcome:
            response = _mock_completion("here's what's current")
            model_used = cli._WEB_SEARCH_MODELS[0]
            passed = True
            attempts = []
            issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)
    monkeypatch.setattr("builtins.input", _canned_input(["--web what's new in numpy", "exit"]))

    cli.run_chat()

    assert captured["extra_body"] == {"tool_ids": ["server:websearch"]}
    assert set(captured["arms"]) == set(cli._WEB_SEARCH_MODELS)


def test_run_chat_image_flag_forces_the_vision_model_for_that_turn(monkeypatch, tmp_path):
    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)

    image_path = tmp_path / "shot.png"
    image_path.write_bytes(b"fake png bytes")

    captured = {}

    def _fake_gate(adapter, bandit, context_key, messages):
        captured["arms"] = bandit.arms
        captured["content"] = messages[-1]["content"]

        class _Outcome:
            response = _mock_completion("that's a null pointer")
            model_used = cli._VISION_MODEL
            passed = True
            attempts = []
            issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)
    monkeypatch.setattr("builtins.input", _canned_input([f"--image {image_path} what's wrong", "exit"]))

    cli.run_chat()

    assert captured["arms"] == [cli._VISION_MODEL]
    assert isinstance(captured["content"], list)
    assert captured["content"][0]["text"] == "what's wrong"


def test_run_chat_model_flag_forces_a_single_arm_for_that_turn(monkeypatch):
    from arcus.quality.gate import AttemptDetail

    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)

    captured = {}

    def _fake_gate(adapter, bandit, context_key, messages):
        captured["arms"] = bandit.arms

        class _Outcome:
            response = _mock_completion("a forced answer")
            model_used = "Kimi-K3"
            passed = True
            attempts = [
                AttemptDetail(
                    model="Kimi-K3", passed=True, reward=0.9, latency_ms=90.0, propensity=1.0, issues=[]
                )
            ]
            issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)
    monkeypatch.setattr("builtins.input", _canned_input(["--model Kimi-K3 pick this one", "exit"]))

    cli.run_chat()

    assert captured["arms"] == ["Kimi-K3"]

    with Session(engine) as session:
        rows = session.exec(select(RequestLog)).all()
    assert rows[0].mode == "manual"


def test_run_chat_doc_flag_combined_with_model_override_uses_both(monkeypatch):
    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)
    fake_adapter = _FakeDocAdapter()
    monkeypatch.setattr(cli, "ArcAdapter", lambda api_key: fake_adapter)

    captured = {}

    def _fake_gate(adapter, bandit, context_key, messages, **kwargs):
        captured["arms"] = bandit.arms
        captured["extra_body"] = kwargs.get("extra_body")

        class _Outcome:
            response = _mock_completion("the doc, from GLM specifically")
            model_used = "GLM-5.3"
            passed = True
            attempts = []
            issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)
    monkeypatch.setattr(
        "builtins.input",
        _canned_input(["--doc paper.pdf --model GLM-5.3 summarize this", "exit"]),
    )

    cli.run_chat()

    # both the forced arm and the file attachment have to survive
    # together, neither flag should silently win over the other
    assert captured["arms"] == ["GLM-5.3"]
    assert captured["extra_body"] == {"files": [{"type": "file", "id": "file-abc123"}]}
    assert fake_adapter.deleted_ids == ["file-abc123"]


def test_run_chat_image_flag_with_a_matching_model_override_is_accepted(monkeypatch, tmp_path):
    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)

    image_path = tmp_path / "shot.png"
    image_path.write_bytes(b"fake png bytes")

    captured = {}

    def _fake_gate(adapter, bandit, context_key, messages):
        captured["arms"] = bandit.arms

        class _Outcome:
            response = _mock_completion("that's a null pointer")
            model_used = cli._VISION_MODEL
            passed = True
            attempts = []
            issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)
    monkeypatch.setattr(
        "builtins.input",
        _canned_input([f"--image {image_path} --model {cli._VISION_MODEL} what's wrong", "exit"]),
    )

    cli.run_chat()

    assert captured["arms"] == [cli._VISION_MODEL]


def test_run_chat_image_flag_with_a_mismatched_model_override_is_rejected(monkeypatch, tmp_path):
    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)

    image_path = tmp_path / "shot.png"
    image_path.write_bytes(b"fake png bytes")

    monkeypatch.setattr(
        cli, "call_with_quality_gate", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("shouldn't be called"))
    )
    monkeypatch.setattr(
        "builtins.input",
        _canned_input([f"--image {image_path} --model GLM-5.3 what's wrong", "exit"]),
    )

    cli.run_chat()  # should not raise, just reject the turn and move on


def test_run_chat_rejects_combining_attachment_flags_in_one_turn(monkeypatch, capsys):
    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)

    monkeypatch.setattr(
        cli, "call_with_quality_gate", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("shouldn't be called"))
    )
    monkeypatch.setattr(
        "builtins.input", _canned_input(["--doc a.pdf --web conflicting question", "exit"])
    )

    cli.run_chat()

    assert "only one of" in capsys.readouterr().out


def test_run_chat_rejects_an_unknown_model_override_and_keeps_going(monkeypatch):
    engine = _in_memory_engine()
    _chat_setup(monkeypatch, engine)

    calls = []

    def _fake_gate(adapter, bandit, context_key, messages):
        calls.append(1)

        class _Outcome:
            response = _mock_completion("fine")
            model_used = "gpt-oss-120b"
            passed = True
            attempts = []
            issues = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)
    monkeypatch.setattr(
        "builtins.input",
        _canned_input(["--model not-a-real-model bad override", "a fine question", "exit"]),
    )

    cli.run_chat()

    # the bad turn never reaches the gate, the next turn still does
    assert len(calls) == 1
