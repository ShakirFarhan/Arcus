import io

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
    # known_arms() normally checks ARC's live model catalog against
    # disk cache, neither of which a unit test should be touching. tests
    # that care about routing behavior mock call_with_quality_gate
    # directly anyway, so the exact arm list here doesn't matter beyond
    # being non-empty and consistent with what those mocks expect.
    monkeypatch.setattr(
        cli, "known_arms", lambda adapter: ["gpt-oss-120b", "GLM-5.3", "Kimi-K3", "DeepSeek-V4-Flash"]
    )


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

    monkeypatch.setattr(cli, "call_with_quality_gate", lambda *a, **kw: _Outcome())

    cli.run_ask("how does binary search work")

    assert "no usable response" in capsys.readouterr().out


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
        else:
            class _Outcome:
                response = _mock_completion("second reply")
                model_used = "gpt-oss-120b"
                passed = True
                attempts = []

        return _Outcome()

    monkeypatch.setattr(cli, "call_with_quality_gate", _fake_gate)
    monkeypatch.setattr("builtins.input", _canned_input(["broken question", "second question", "exit"]))

    cli.run_chat()

    assert len(calls) == 2
    assert calls[1] == [{"role": "user", "content": "second question"}]


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


def test_trim_history_drops_oldest_turns_in_pairs():
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(23)]

    trimmed = cli._trim_history(messages, max_messages=20)

    assert len(trimmed) == 19
    assert trimmed[0]["role"] == "user"


def test_trim_history_leaves_short_history_untouched():
    messages = [{"role": "user", "content": "hi"}]
    assert cli._trim_history(messages, max_messages=20) == messages


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
    monkeypatch.setattr(cli, "run_chat", lambda random_mode: called.append(random_mode))
    cli.main(["chat", "--random"])
    assert called == [True]


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
