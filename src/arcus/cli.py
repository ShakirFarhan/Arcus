import base64
import mimetypes
import sys
import threading
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import get_args
from uuid import uuid4

import httpx
import typer
from rich.console import Console
from rich.table import Table
from sqlmodel import Session, select

from arcus.adapters.arc_adapter import ArcAdapter, ArcModel
from arcus.cache.semantic_cache import lookup as cache_lookup
from arcus.cache.semantic_cache import store as cache_store
from arcus.config import ArcusConfig, BanditAlgorithm, config_path, load_config, save_config
from arcus.embeddings import get_embedding_model
from arcus.quality.gate import call_with_quality_gate
from arcus.routing.bandit import (
    Bandit,
    ContextualBandit,
    EpsilonGreedyBandit,
    RandomBandit,
    ThompsonSamplingBandit,
    UCB1Bandit,
)
from arcus.routing.context import Context, classify
from arcus.routing.model_catalog import filter_to_live, known_arms
from arcus.routing.warm_start import replay_history
from arcus.storage.db import RequestLog, get_engine, log_request
from arcus.storage.stats import aggregate_by_arm_and_mode

_ARC_DOCS_URL = "https://www.docs.arc.vt.edu/ai/011_llm_api_arc_vt_edu.html"

# Kimi-K3 is the one model ARC's own docs describe as vision-capable
# ("native multimodal understanding"), the other three aren't
# documented either way, so image requests go straight to it rather
# than through the usual multi-model bandit comparison. Confirmed
# directly against the API: GLM-5.3 and DeepSeek-V4-Flash both reject
# image content outright, gpt-oss-120b accepts the request but reports
# it can't actually see the image.
_VISION_MODEL = ArcModel.KIMI_K3.value

# web search needs one of ARC's "legacy-tool-calling" model variants,
# not the regular arms. DeepSeek's variant accepts the search tool_id
# without erroring but doesn't reliably act on it (tested: answered a
# time-sensitive question wrong, with no source citation, while the
# other three got it right and cited sources), so it's left out here
# until that's confirmed fixed on ARC's side.
_WEB_SEARCH_MODELS = [
    "gpt-oss-120b-thinking-high-legacy-tool-calling",
    "Kimi-K3-thinking-max-legacy-tool-calling",
    "glm-52-thinking-high-legacy-tool-calling",
]

_VALID_BANDIT_ALGORITHMS = get_args(BanditAlgorithm)

_ALGORITHM_FACTORIES: dict[BanditAlgorithm, "type[Bandit]"] = {
    "epsilon_greedy": EpsilonGreedyBandit,
    "ucb1": UCB1Bandit,
    "thompson": ThompsonSamplingBandit,
}

_COMPLETION_SCRIPTS = {
    "bash": """\
_arcus_completions() {
    local cur=${COMP_WORDS[COMP_CWORD]}
    if [ "$COMP_CWORD" -eq 1 ]; then
        COMPREPLY=($(compgen -W "chat stats models config --random --version --image --doc --web" -- "$cur"))
    fi
}
complete -F _arcus_completions arcus
""",
    "zsh": """\
#compdef arcus
_arcus() {
    if [ "$CURRENT" -eq 2 ]; then
        compadd chat stats models config --random --version --image --doc --web
    fi
}
_arcus
""",
}


def main(argv: list[str] | None = None) -> None:
    """Entry point for the `arcus` command. Dispatches by hand on argv[0]
    rather than using typer's subcommand machinery. the whole point is for
    `arcus "some question"` to just work with no subcommand at all, and a
    real subcommand parser (click underneath typer) fights that: it wants
    to treat the prompt text itself as an unrecognized command. `stats`,
    `chat`, `models`, and `config` are the reserved words, everything
    else is prompt text.
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in ("--version", "-V"):
        Console().print(f"arcus {_version()}")
        return

    if argv and argv[0] == "--completion":
        shell = argv[1] if len(argv) > 1 else ""
        _print_completion(shell)
        return

    if argv and argv[0] == "stats":
        run_stats()
        return

    if argv and argv[0] == "models":
        run_models()
        return

    if argv and argv[0] == "config":
        run_config(argv[1:])
        return

    if argv and argv[0] == "chat":
        random_mode = "--random" in argv
        save_path = _extract_flag_value(argv, "--save")
        run_chat(random_mode=random_mode, save_path=save_path)
        return

    random_mode = "--random" in argv
    image_path = _extract_flag_value(argv, "--image")
    doc_path = _extract_flag_value(argv, "--doc")
    web_mode = "--web" in argv
    argv = [arg for arg in argv if arg not in ("--random", "--web")]
    argv = _strip_flag_and_value(argv, "--image")
    argv = _strip_flag_and_value(argv, "--doc")

    if sum(bool(x) for x in (image_path, doc_path, web_mode)) > 1:
        Console().print("[red]use only one of --image, --doc, or --web at a time.[/red]")
        return

    prompt = _build_prompt(argv)
    if not prompt:
        _print_usage()
        return

    if image_path:
        run_image_ask(prompt, image_path)
        return

    if doc_path:
        run_doc_ask(prompt, doc_path, random_mode=random_mode)
        return

    if web_mode:
        run_web_ask(prompt, random_mode=random_mode)
        return

    run_ask(prompt, random_mode=random_mode)


def _extract_flag_value(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    idx = argv.index(flag)
    if idx + 1 >= len(argv):
        return None
    return argv[idx + 1]


def _strip_flag_and_value(argv: list[str], flag: str) -> list[str]:
    if flag not in argv:
        return argv
    result = list(argv)
    idx = result.index(flag)
    del result[idx : idx + 2]
    return result


def _build_prompt(args: list[str]) -> str:
    # if stdin isn't a tty, something got piped in (a traceback, a test
    # failure, whatever), treat that as the main
    # content and let any positional args add an explicit instruction
    # on top of it, e.g. `python broken.py 2>&1 | arcus "why is this
    # failing"`.
    piped = sys.stdin.read().strip() if not sys.stdin.isatty() else ""
    positional = " ".join(args).strip()

    if piped and positional:
        return f"{piped}\n\n{positional}"
    return piped or positional


def _print_usage() -> None:
    Console().print(
        'usage: arcus "<question>"   or   arcus chat   or   arcus stats   or   '
        "arcus models   or   arcus config"
    )


def _version() -> str:
    try:
        return version("arcus-cli")
    except PackageNotFoundError:
        # running straight from a checkout rather than an installed
        # package, there's no distribution metadata to read
        return "unknown (running from source)"


def _print_completion(shell: str) -> None:
    script = _COMPLETION_SCRIPTS.get(shell)
    if script is None:
        Console().print(
            f"[red]no completion script for '{shell}', supported shells: "
            f"{', '.join(_COMPLETION_SCRIPTS)}[/red]"
        )
        raise SystemExit(1)
    # plain print rather than console.print, this output gets piped
    # straight into eval by the shell and needs to stay free of rich's
    # formatting and color codes
    print(script)


def _ensure_config() -> ArcusConfig:
    try:
        return load_config()
    except FileNotFoundError:
        return _run_setup_wizard()


def _run_setup_wizard() -> ArcusConfig:
    console = Console()
    console.print("[bold]no ARC key found, let's get you set up.[/bold]")
    console.print(
        "grab one from llm.arc.vt.edu under "
        "User profile > Settings > Account > API keys.\n"
        f"full docs on the service: {_ARC_DOCS_URL}\n"
    )

    api_key = typer.prompt("ARC API key", hide_input=True)

    console.print("checking that key works...")
    try:
        ArcAdapter(api_key=api_key).chat(
            ArcModel.GPT_OSS_120B,
            [{"role": "user", "content": "reply with just the word ok"}],
            max_tokens=5,
        )
    except Exception as e:
        console.print(f"[red]that key didn't work:[/red] {e}")
        raise SystemExit(1) from e

    config = ArcusConfig(arc_api_key=api_key)
    save_config(config)
    console.print(f"[green]saved to {config_path()}[/green]\n")
    return config


def run_ask(prompt: str, random_mode: bool = False) -> None:
    console = Console()

    # the embedding model backs both the context classifier's fallback
    # path and the semantic cache lookup below, and loading it costs a
    # few seconds the first time any process touches it. kick that load
    # off now, in the background, so it's warm (or at least warming) by
    # the time either one actually calls embed(), instead of eating that
    # cost inline and in serial with everything else.
    #
    # this has to start AFTER building the ArcAdapter, not before.
    # constructing the openai client is the first time this process
    # touches httpx internals, and openai does that lazily, on client
    # construction, not on import. starting the embedding thread first
    # let it race that first-time httpx touch against sentence-transformers'
    # own (torch/huggingface_hub) import chain, which also reaches into
    # httpx, and that produced a real, reliably reproducible crash:
    # "partially initialized module 'httpx' ... circular import". building
    # the adapter first means the main thread finishes touching httpx
    # before any second thread gets a chance to.
    config = _ensure_config()
    engine = get_engine()
    adapter = ArcAdapter(api_key=config.arc_api_key)

    threading.Thread(target=get_embedding_model, daemon=True).start()

    context = classify(prompt)
    mode = "random" if random_mode else "bandit"

    cached = cache_lookup(prompt, engine=engine)
    if cached.hit:
        console.print(cached.response)
        log_request(
            prompt=prompt,
            task_type=context.task_type.value,
            length_bucket=context.length_bucket.value,
            model=cached.model,
            cache_hit=True,
            quality_passed=True,
            engine=engine,
        )
        return

    arms = known_arms(adapter)
    algorithm_factory = RandomBandit if random_mode else _ALGORITHM_FACTORIES[config.bandit_algorithm]
    bandit = ContextualBandit(lambda: algorithm_factory(arms), arms=arms)
    # rebuild what this bandit already learned from past requests in this
    # same mode, otherwise every invocation starts back at zero since
    # there's no daemon holding it in memory between runs.
    replay_history(bandit, engine, mode=mode)

    content, model_used, passed, issues = _route_and_answer(
        adapter, bandit, context, prompt, [{"role": "user", "content": prompt}], mode, engine
    )

    console.print(content if content else _failure_message(issues))

    if passed and content:
        cache_store(prompt, content, model=model_used, engine=engine)


def _build_image_content(prompt: str, image_path: str) -> list[dict]:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None or not mime_type.startswith("image/"):
        raise ValueError(f"'{image_path}' doesn't look like an image file")

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
    ]


def run_image_ask(prompt: str, image_path: str) -> None:
    console = Console()

    config = _ensure_config()
    engine = get_engine()
    adapter = ArcAdapter(api_key=config.arc_api_key)

    threading.Thread(target=get_embedding_model, daemon=True).start()

    try:
        image_content = _build_image_content(prompt, image_path)
    except (OSError, ValueError) as e:
        console.print(f"[red]couldn't read '{image_path}':[/red] {e}")
        raise SystemExit(1) from e

    live_arms = known_arms(adapter)
    if _VISION_MODEL not in live_arms:
        console.print(f"[red]{_VISION_MODEL} isn't currently available for image requests.[/red]")
        raise SystemExit(1)

    # the semantic cache has no concept of image content, matching only
    # on the text portion would risk serving back an answer about a
    # completely different image, so this skips the cache entirely
    # rather than mislead the way arcus chat's follow-up turns would.
    context = classify(prompt)
    bandit = ContextualBandit(lambda: EpsilonGreedyBandit([_VISION_MODEL], epsilon=0.0), arms=[_VISION_MODEL])
    replay_history(bandit, engine, mode="bandit")

    content, model_used, passed, issues = _route_and_answer(
        adapter, bandit, context, prompt, [{"role": "user", "content": image_content}], "bandit", engine
    )

    console.print(content if content else _failure_message(issues))


def run_doc_ask(prompt: str, doc_path: str, random_mode: bool = False) -> None:
    console = Console()

    config = _ensure_config()
    engine = get_engine()
    adapter = ArcAdapter(api_key=config.arc_api_key)

    threading.Thread(target=get_embedding_model, daemon=True).start()

    try:
        file_id = adapter.upload_file(doc_path)
    except OSError as e:
        console.print(f"[red]couldn't read '{doc_path}':[/red] {e}")
        raise SystemExit(1) from e
    except httpx.HTTPError as e:
        console.print(f"[red]couldn't upload '{doc_path}' to ARC:[/red] {e}")
        raise SystemExit(1) from e

    # unlike vision, RAG works across all four regular models (confirmed
    # directly against the API), so this goes through the normal bandit
    # comparison rather than a single forced arm. the semantic cache
    # still gets skipped though, a cached answer keyed on question text
    # alone would risk answering about a completely different document.
    context = classify(prompt)
    mode = "random" if random_mode else "bandit"
    arms = known_arms(adapter)
    algorithm_factory = RandomBandit if random_mode else _ALGORITHM_FACTORIES[config.bandit_algorithm]
    bandit = ContextualBandit(lambda: algorithm_factory(arms), arms=arms)
    replay_history(bandit, engine, mode=mode)

    content, model_used, passed, issues = _route_and_answer(
        adapter,
        bandit,
        context,
        prompt,
        [{"role": "user", "content": prompt}],
        mode,
        engine,
        extra_body={"files": [{"type": "file", "id": file_id}]},
    )

    console.print(content if content else _failure_message(issues))

    try:
        adapter.delete_file(file_id)
    except httpx.HTTPError:
        # the answer's already shown, not worth failing the command over
        # cleanup, the file just sits in the user's ARC account until
        # they remove it there themselves
        pass


def run_web_ask(prompt: str, random_mode: bool = False) -> None:
    console = Console()

    config = _ensure_config()
    engine = get_engine()
    adapter = ArcAdapter(api_key=config.arc_api_key)

    threading.Thread(target=get_embedding_model, daemon=True).start()

    # a web-search answer reflects a moment in time the same way a
    # volatile query does elsewhere in the cache, caching it risks
    # serving something stale later, so this skips the cache entirely
    # rather than try to guess which searched answers are safe to keep.
    context = classify(prompt)
    mode = "random" if random_mode else "bandit"
    arms = filter_to_live(adapter, _WEB_SEARCH_MODELS)
    algorithm_factory = RandomBandit if random_mode else _ALGORITHM_FACTORIES[config.bandit_algorithm]
    bandit = ContextualBandit(lambda: algorithm_factory(arms), arms=arms)
    replay_history(bandit, engine, mode=mode)

    content, model_used, passed, issues = _route_and_answer(
        adapter,
        bandit,
        context,
        prompt,
        [{"role": "user", "content": prompt}],
        mode,
        engine,
        extra_body={"tool_ids": ["server:websearch"]},
    )

    console.print(content if content else _failure_message(issues))


def _failure_message(issues: list) -> str:
    if any(issue.kind == "permission_denied" for issue in issues):
        return "[red]ARC restricts API access to VT's campus network, connect to the VPN and try again.[/red]"
    return "[red]no usable response from any model.[/red]"


def _route_and_answer(
    adapter: ArcAdapter,
    bandit: ContextualBandit,
    context: Context,
    prompt: str,
    messages: list[dict],
    mode: str,
    engine,
    conversation_id: str | None = None,
    turn_index: int | None = None,
    **extra_kwargs,
) -> tuple[str | None, str, bool, list]:
    """Runs one turn through the quality gate, logs every attempt, and
    returns (content, model_used, passed, issues). content can be
    non-None even when passed is False (the last attempt still produced
    text, it just didn't clear the gate); content is None only when
    every arm errored out with nothing to show at all, in which case
    issues carries the reason from the last attempt.

    extra_kwargs passes straight through to call_with_quality_gate, this
    is how RAG's `files` parameter and web search's `tool_ids` reach the
    actual API call.
    """
    outcome = call_with_quality_gate(adapter, bandit, context.key, messages, **extra_kwargs)

    for attempt in outcome.attempts:
        log_request(
            prompt=prompt,
            task_type=context.task_type.value,
            length_bucket=context.length_bucket.value,
            model=attempt.model,
            propensity=attempt.propensity,
            latency_ms=attempt.latency_ms,
            mode=mode,
            reward=attempt.reward,
            quality_passed=attempt.passed,
            conversation_id=conversation_id,
            turn_index=turn_index,
            engine=engine,
        )

    # response is None when every arm errored out at the API level (see
    # call_with_quality_gate), not just returned a bad answer
    content = outcome.response.choices[0].message.content if outcome.response else None
    return content, outcome.model_used, outcome.passed, outcome.issues


_MAX_CHAT_MESSAGES = 20  # ~10 exchanges, a first-guess cap like the
# reward weights, not tuned against anything yet


def _trim_history(messages: list[dict], max_messages: int = _MAX_CHAT_MESSAGES) -> list[dict]:
    if len(messages) <= max_messages:
        return messages
    # drop whole oldest turns (user+assistant pairs), not a single
    # message, so the transcript never gets left with an orphaned
    # question and no answer in front of it
    excess = len(messages) - max_messages
    excess += excess % 2
    return messages[excess:]


def _write_transcript(path: str, turns: list[str]) -> None:
    header = f"# arcus chat transcript\n\nsaved {datetime.now(UTC).isoformat()}\n\n"
    Path(path).write_text(header + "\n".join(turns))


def run_chat(random_mode: bool = False, save_path: str | None = None) -> None:
    console = Console()

    config = _ensure_config()
    engine = get_engine()
    adapter = ArcAdapter(api_key=config.arc_api_key)  # before the thread, same reason as run_ask

    threading.Thread(target=get_embedding_model, daemon=True).start()

    mode = "random" if random_mode else "bandit"
    arms = known_arms(adapter)
    algorithm_factory = RandomBandit if random_mode else _ALGORITHM_FACTORIES[config.bandit_algorithm]
    bandit = ContextualBandit(lambda: algorithm_factory(arms), arms=arms)
    replay_history(bandit, engine, mode=mode)

    conversation_id = str(uuid4())
    messages: list[dict] = []
    turn_index = 0
    # kept separately from messages, which gets trimmed by _trim_history
    # as the conversation grows, a saved transcript should have the
    # whole conversation, not just whatever's still in the active window
    transcript: list[str] = []

    console.print("[bold]chatting with arcus, type 'exit' or ctrl-d to leave.[/bold]\n")

    while True:
        try:
            user_input = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})
        transcript.append(f"**You:** {user_input}\n")
        context = classify(user_input)

        content, model_used, passed, issues = _route_and_answer(
            adapter,
            bandit,
            context,
            user_input,
            messages,
            mode,
            engine,
            conversation_id=conversation_id,
            turn_index=turn_index,
        )

        if content:
            messages.append({"role": "assistant", "content": content})
            transcript.append(f"**Arcus ({model_used}):** {content}\n")
            console.print(f"[green]arcus ({model_used}):[/green] {content}\n")
        elif any(issue.kind == "permission_denied" for issue in issues):
            # nothing usable came back, don't leave an unanswered
            # question sitting in history for the next turn to trip over
            messages.pop()
            transcript.append("*(no response: VPN required)*\n")
            console.print(
                "[red]ARC restricts API access to VT's campus network, "
                "connect to the VPN and try again.[/red]\n"
            )
        else:
            messages.pop()
            transcript.append("*(no usable response)*\n")
            console.print("[red]no usable response from any model, try rephrasing.[/red]\n")

        messages = _trim_history(messages)
        turn_index += 1

    if save_path and transcript:
        _write_transcript(save_path, transcript)
        console.print(f"[dim]transcript saved to {save_path}[/dim]")


def run_models() -> None:
    console = Console()
    config = _ensure_config()
    adapter = ArcAdapter(api_key=config.arc_api_key)

    try:
        live_ids = adapter.list_models()
    except Exception as e:
        console.print(f"[red]couldn't reach ARC's model catalog:[/red] {e}")
        raise SystemExit(1) from e

    routed = {m.value for m in ArcModel}

    table = Table(title="models ARC is currently serving")
    table.add_column("model")
    table.add_column("routed by arcus", justify="center")

    for model_id in sorted(live_ids):
        table.add_row(model_id, "yes" if model_id in routed else "")

    console.print(table)
    console.print(f"\nfull docs: {_ARC_DOCS_URL}")


def run_config(args: list[str]) -> None:
    console = Console()
    config = _ensure_config()

    if not args:
        masked_key = f"...{config.arc_api_key[-4:]}" if len(config.arc_api_key) > 4 else "****"
        console.print(f"arc_api_key: {masked_key}")
        console.print(f"bandit_algorithm: {config.bandit_algorithm}")
        console.print(f"\nconfig file: {config_path()}")
        return

    if len(args) == 3 and args[0] == "set" and args[1] == "bandit_algorithm":
        value = args[2]
        if value not in _VALID_BANDIT_ALGORITHMS:
            console.print(
                f"[red]'{value}' isn't a valid bandit_algorithm, choose one of: "
                f"{', '.join(_VALID_BANDIT_ALGORITHMS)}[/red]"
            )
            raise SystemExit(1)
        updated = config.model_copy(update={"bandit_algorithm": value})
        save_config(updated)
        console.print(f"[green]bandit_algorithm set to {value}[/green]")
        return

    console.print(
        "[red]usage: arcus config   or   arcus config set bandit_algorithm "
        f"<{'|'.join(_VALID_BANDIT_ALGORITHMS)}>[/red]"
    )
    raise SystemExit(1)


def run_stats() -> None:
    console = Console()
    engine = get_engine()

    with Session(engine) as session:
        rows = session.exec(select(RequestLog)).all()

    if not rows:
        console.print("no requests logged yet, go ask arcus something.")
        return

    table = Table(title="arcus stats")
    table.add_column("model")
    table.add_column("mode")
    table.add_column("requests", justify="right")
    table.add_column("avg reward", justify="right")
    table.add_column("avg latency (ms)", justify="right")
    table.add_column("cost score", justify="right")

    for summary in aggregate_by_arm_and_mode(engine):
        table.add_row(
            summary.model,
            summary.mode,
            str(summary.request_count),
            f"{summary.avg_reward:.3f}" if summary.avg_reward is not None else "-",
            f"{summary.avg_latency_ms:.0f}" if summary.avg_latency_ms is not None else "-",
            f"{summary.cost_score:.2f}" if summary.cost_score is not None else "-",
        )

    console.print(table)

    cache_hits = sum(1 for row in rows if row.cache_hit)
    gate_catches = sum(1 for row in rows if not row.quality_passed)

    console.print(f"\ncache hit rate: {cache_hits / len(rows):.1%} ({cache_hits}/{len(rows)})")
    console.print(f"quality gate catches: {gate_catches} attempt(s) failed and got retried")


if __name__ == "__main__":
    main()
