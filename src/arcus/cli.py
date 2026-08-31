import sys
import threading
from importlib.metadata import PackageNotFoundError, version
from uuid import uuid4

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
from arcus.routing.model_catalog import known_arms
from arcus.routing.warm_start import replay_history
from arcus.storage.db import RequestLog, get_engine, log_request
from arcus.storage.stats import aggregate_by_arm_and_mode

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
        COMPREPLY=($(compgen -W "chat stats --random --version" -- "$cur"))
    fi
}
complete -F _arcus_completions arcus
""",
    "zsh": """\
#compdef arcus
_arcus() {
    if [ "$CURRENT" -eq 2 ]; then
        compadd chat stats --random --version
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
    to treat the prompt text itself as an unrecognized command. `stats` is
    the one reserved word, everything else is prompt text.
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

    if argv and argv[0] == "chat":
        random_mode = "--random" in argv
        run_chat(random_mode=random_mode)
        return

    random_mode = "--random" in argv
    argv = [arg for arg in argv if arg != "--random"]

    prompt = _build_prompt(argv)
    if not prompt:
        _print_usage()
        return

    run_ask(prompt, random_mode=random_mode)


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
    Console().print('usage: arcus "<question>"   or   arcus chat   or   arcus stats')


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

    content, model_used, passed = _route_and_answer(
        adapter, bandit, context, prompt, [{"role": "user", "content": prompt}], mode, engine
    )

    console.print(content if content else "[red]no usable response from any model.[/red]")

    if passed and content:
        cache_store(prompt, content, model=model_used, engine=engine)


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
) -> tuple[str | None, str, bool]:
    """Runs one turn through the quality gate, logs every attempt, and
    returns (content, model_used, passed). content can be non-None even
    when passed is False (the last attempt still produced text, it just
    didn't clear the gate); content is None only when every arm errored
    out with nothing to show at all.
    """
    outcome = call_with_quality_gate(adapter, bandit, context.key, messages)

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
    return content, outcome.model_used, outcome.passed


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


def run_chat(random_mode: bool = False) -> None:
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
        context = classify(user_input)

        content, model_used, passed = _route_and_answer(
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
            console.print(f"[green]arcus ({model_used}):[/green] {content}\n")
        else:
            # nothing usable came back, don't leave an unanswered
            # question sitting in history for the next turn to trip over
            messages.pop()
            console.print("[red]no usable response from any model, try rephrasing.[/red]\n")

        messages = _trim_history(messages)
        turn_index += 1


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
