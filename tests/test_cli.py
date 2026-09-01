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
    monkeypatch.setattr(cli, "run_doc_ask", lambda prompt, doc_path, random_mode: called.append((prompt, doc_path, random_mode)))
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    cli.main(["--doc", "paper.pdf", "summarize", "this"])

    assert called == [("summarize this", "paper.pdf", False)]


def test_main_dispatches_to_web_ask(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "run_web_ask", lambda prompt, random_mode: called.append((prompt, random_mode)))
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    cli.main(["--web", "what's", "new"])

    assert called == [("what's new", False)]


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
    monkeypatch.setattr(cli, "run_ask", lambda prompt, random_mode: seen.update(prompt=prompt, random_mode=random_mode))

    cli.main(["--random", "explain", "recursion"])

    assert seen == {"prompt": "explain recursion", "random_mode": True}


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
    monkeypatch.setattr(cli, "run_image_ask", lambda prompt, image_path: called.append((prompt, image_path)))
    monkeypatch.setattr(
        cli, "run_ask", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("run_ask shouldn't fire here"))
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    cli.main(["explain", "--image", "shot.png", "this", "error"])

    assert called == [("explain this error", "shot.png")]
