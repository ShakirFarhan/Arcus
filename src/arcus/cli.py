import base64
import mimetypes
import shlex
import sys
import threading
from dataclasses import dataclass
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
from arcus.eval.offline import evaluate_policies, greedy_policy_from_log, load_logged_examples, policy_always
from arcus.quality.gate import QualityIssue, call_with_quality_gate
from arcus.routing.bandit import (
    Bandit,
    ContextualBandit,
    EpsilonGreedyBandit,
    RandomBandit,
    ThompsonSamplingBandit,
    UCB1Bandit,
)
from arcus.routing.context import Context, TaskType, classify
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
        COMPREPLY=($(compgen -W "chat stats eval models config --random --model --version --image --doc --web" -- "$cur"))
    fi
}
complete -F _arcus_completions arcus
""",
    "zsh": """\
#compdef arcus
_arcus() {
    if [ "$CURRENT" -eq 2 ]; then
        compadd chat stats eval models config --random --model --version --image --doc --web
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
    `chat`, `models`, `config`, and `eval` are the reserved words,
    everything else is prompt text.
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

    if argv and argv[0] == "eval":
        run_eval()
        return

    if argv and argv[0] == "chat":
        random_mode = "--random" in argv
        save_path = _extract_flag_value(argv, "--save")
        run_chat(random_mode=random_mode, save_path=save_path)
        return

    flags = _parse_request_flags(argv)
    error = _validate_mode_flags(flags)
    if error:
        Console().print(f"[red]{error}[/red]")
        return

    prompt = _build_prompt(flags.remaining)
    if not prompt:
        _print_usage()
        return

    if flags.image_path:
        run_image_ask(prompt, flags.image_path, model_override=flags.model_override)
        return

    if flags.doc_path:
        run_doc_ask(prompt, flags.doc_path, random_mode=flags.random_mode, model_override=flags.model_override)
        return

    if flags.web_mode:
        run_web_ask(prompt, random_mode=flags.random_mode, model_override=flags.model_override)
        return

    run_ask(prompt, random_mode=flags.random_mode, model_override=flags.model_override)


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


def _tokenize_chat_line(line: str) -> list[str]:
    """Splits a raw chat line into flag-parsing tokens. Plain
    shlex.split() treats a single quote as the start of a quoted
    string, which makes it choke on perfectly ordinary English, "what's
    new" is an unclosed quote as far as shlex is concerned. Only
    double quotes count as quoting here (for a --doc path with a space
    in it), everything else, apostrophes included, is left alone.
    Escape processing is off too, otherwise a Windows path like
    C:\\Users\\foo.pdf gets mangled by backslash-escape handling.
    """
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.quotes = '"'
    lexer.escape = ""
    return list(lexer)


@dataclass(frozen=True)
class _RequestFlags:
    remaining: list[str]
    image_path: str | None
    doc_path: str | None
    web_mode: bool
    model_override: str | None
    random_mode: bool


def _parse_request_flags(tokens: list[str]) -> _RequestFlags:
    """Pulls --image/--doc/--web/--model/--random out of a token list and
    hands back whatever's left over as the actual question text. Used
    for both the top-level argv and, per turn, a chat line split with
    shlex, so a flag works exactly the same way in both places instead
    of chat needing its own separate syntax.
    """
    random_mode = "--random" in tokens
    image_path = _extract_flag_value(tokens, "--image")
    doc_path = _extract_flag_value(tokens, "--doc")
    web_mode = "--web" in tokens
    model_override = _extract_flag_value(tokens, "--model")

    remaining = [t for t in tokens if t not in ("--random", "--web")]
    remaining = _strip_flag_and_value(remaining, "--image")
    remaining = _strip_flag_and_value(remaining, "--doc")
    remaining = _strip_flag_and_value(remaining, "--model")

    return _RequestFlags(
        remaining=remaining,
        image_path=image_path,
        doc_path=doc_path,
        web_mode=web_mode,
        model_override=model_override,
        random_mode=random_mode,
    )


def _validate_mode_flags(flags: _RequestFlags) -> str | None:
    """Catches flag combinations that can't mean anything before a
    request ever goes out, rather than letting them silently pick
    whichever branch happens to run first. Returns the message to show
    the user, or None if the combination is fine.
    """
    if sum(bool(x) for x in (flags.image_path, flags.doc_path, flags.web_mode)) > 1:
        return "use only one of --image, --doc, or --web at a time."
    if flags.model_override and flags.random_mode:
        return "--model and --random contradict each other, pick one."
    return None


def _resolve_model_override(
    adapter: ArcAdapter,
    model: str,
    *,
    web_mode: bool = False,
    image_mode: bool = False,
) -> str | None:
    """Checks a --model override against what ARC is actually serving
    right now and against the mode it's paired with. Returns an error
    message to show the user, or None if the override is usable as-is.
    """
    if image_mode:
        if model != _VISION_MODEL:
            return f"--image only works with {_VISION_MODEL} right now, that's the one model confirmed to actually see an attached image."
        return None

    if web_mode:
        live_web_models = filter_to_live(adapter, _WEB_SEARCH_MODELS)
        if model not in live_web_models:
            return f"'{model}' doesn't do --web, valid choices: {', '.join(live_web_models)}"
        return None

    live_arms = known_arms(adapter)
    if model not in live_arms:
        return f"'{model}' isn't a model ARC is currently serving, valid choices: {', '.join(sorted(live_arms))}"
    return None


def _forced_arm_bandit(model: str, engine, mode: str = "manual") -> ContextualBandit:
    """A single-arm bandit that always picks `model`, no exploration.
    Used for both the vision-only path and any explicit --model
    override, replay_history still runs so stats/eval can see these
    calls, they're just tagged under their own mode rather than mixed
    into the real bandit's learning history.
    """
    bandit = ContextualBandit(lambda _context_key: EpsilonGreedyBandit([model], epsilon=0.0), arms=[model])
    replay_history(bandit, engine, mode=mode)
    return bandit


# ARC's docs list these as their own catalog model ids (the same pattern
# already used above for the web-search legacy-tool-calling variants),
# not a reasoning_effort parameter tacked onto the base model id. NOT
# verified against a live key from here, there's no ARC account in this
# environment to confirm it. enable_reasoning_variants defaults off in
# config.py for exactly this reason, don't flip it on without running
# a real question through one of these contexts first and checking it
# actually answers instead of erroring.
_REASONING_VARIANTS = [
    "gpt-oss-120b-thinking-high",
    "GLM-5.3-thinking-high",
    "Kimi-K3-thinking-high",
    "DeepSeek-V4-Flash-thinking-max",
]

# a fast/cheap model doing a good job on small talk isn't leaving
# quality on the table, these are the contexts where trading latency
# and cost for a reasoning boost is plausibly worth comparing
_REASONING_TASK_TYPE_VALUES = {TaskType.CODE.value, TaskType.REASONING_MATH.value, TaskType.LONG_DOCUMENT.value}


def _arms_for_reasoning_context(base_arms: list[str], context_key: str, adapter: ArcAdapter) -> list[str]:
    task_type_value = context_key.split(":", 1)[0]
    if task_type_value not in _REASONING_TASK_TYPE_VALUES:
        return base_arms

    # same live-catalog cross-check web search's arm list already uses,
    # falls back to trusting the hardcoded guess above if the live
    # catalog can't be read at all, rather than losing routing options
    # over a transient problem
    live_variants = filter_to_live(adapter, _REASONING_VARIANTS)
    return base_arms + [v for v in live_variants if v not in base_arms]


def _build_bandit(
    arms: list[str],
    algorithm_factory,
    adapter: ArcAdapter,
    enable_reasoning_variants: bool,
) -> ContextualBandit:
    """Builds the bandit for the normal multi-model routing path (plain
    questions, --doc, chat). With reasoning variants off (the default),
    every context shares the same fixed arm list, same as before this
    existed. With them on, code/math/long-document contexts additionally
    get ARC's -thinking-* ids in the running, everything else is
    unaffected.
    """
    if not enable_reasoning_variants:
        return ContextualBandit(lambda _context_key: algorithm_factory(arms), arms=arms)

    def _factory(context_key: str):
        return algorithm_factory(_arms_for_reasoning_context(arms, context_key, adapter))

    return ContextualBandit(_factory, arms=arms)


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
        'usage: arcus "<question>" [--random | --model NAME] [--doc PATH | --web | --image PATH]\n'
        "       arcus chat   or   arcus stats   or   arcus eval   or   "
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


def _setup():
    """Shared bootstrap for every entry point that talks to ARC: load
    config, open the db engine, build the adapter, and kick off the
    embedding-model warm-up thread in the background so it's ready (or
    at least warming) by the time the context classifier's fallback path
    or the semantic cache actually needs it, instead of paying that cost
    inline and in serial with everything else.

    The thread has to start AFTER the adapter is built, not before.
    Constructing the openai client is the first time a process touches
    httpx internals, and openai does that lazily, on client
    construction, not on import. Starting the embedding thread first let
    it race that first-time httpx touch against sentence-transformers'
    own (torch/huggingface_hub) import chain, which also reaches into
    httpx, and that produced a real, reliably reproducible crash:
    "partially initialized module 'httpx' ... circular import". Building
    the adapter first means the main thread finishes touching httpx
    before any second thread gets a chance to.
    """
    config = _ensure_config()
    engine = get_engine()
    adapter = ArcAdapter(api_key=config.arc_api_key)
    threading.Thread(target=get_embedding_model, daemon=True).start()
    return config, engine, adapter


def run_ask(prompt: str, random_mode: bool = False, model_override: str | None = None) -> None:
    console = Console()
    config, engine, adapter = _setup()

    if model_override:
        error = _resolve_model_override(adapter, model_override)
        if error:
            console.print(f"[red]{error}[/red]")
            raise SystemExit(1)

    context = classify(prompt)
    mode = "random" if random_mode else "bandit"

    # a forced model is an explicit ask for that model's own answer, not
    # whatever answered last time, serving (or writing) a cache entry
    # here would quietly ignore the override
    if not model_override:
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

    if model_override:
        bandit = _forced_arm_bandit(model_override, engine)
        mode = "manual"
    else:
        arms = known_arms(adapter)
        algorithm_factory = RandomBandit if random_mode else _ALGORITHM_FACTORIES[config.bandit_algorithm]
        bandit = _build_bandit(arms, algorithm_factory, adapter, config.enable_reasoning_variants)
        # rebuild what this bandit already learned from past requests in
        # this same mode, otherwise every invocation starts back at zero
        # since there's no daemon holding it in memory between runs.
        replay_history(bandit, engine, mode=mode)

    content, model_used, passed, issues = _route_and_answer(
        adapter, bandit, context, prompt, [{"role": "user", "content": prompt}], mode, engine, console
    )

    console.print(content if content else _failure_message(issues))

    if passed and content and not model_override:
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


def run_image_ask(prompt: str, image_path: str, model_override: str | None = None) -> None:
    console = Console()
    _config, engine, adapter = _setup()

    if model_override:
        error = _resolve_model_override(adapter, model_override, image_mode=True)
        if error:
            console.print(f"[red]{error}[/red]")
            raise SystemExit(1)

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
    bandit = _forced_arm_bandit(_VISION_MODEL, engine, mode="bandit")

    content, _, _, issues = _route_and_answer(
        adapter, bandit, context, prompt, [{"role": "user", "content": image_content}], "bandit", engine, console
    )

    console.print(content if content else _failure_message(issues))


def run_doc_ask(
    prompt: str, doc_path: str, random_mode: bool = False, model_override: str | None = None
) -> None:
    console = Console()
    config, engine, adapter = _setup()

    if model_override:
        error = _resolve_model_override(adapter, model_override)
        if error:
            console.print(f"[red]{error}[/red]")
            raise SystemExit(1)

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
    if model_override:
        bandit = _forced_arm_bandit(model_override, engine)
        mode = "manual"
    else:
        arms = known_arms(adapter)
        algorithm_factory = RandomBandit if random_mode else _ALGORITHM_FACTORIES[config.bandit_algorithm]
        bandit = _build_bandit(arms, algorithm_factory, adapter, config.enable_reasoning_variants)
        replay_history(bandit, engine, mode=mode)

    content, _, _, issues = _route_and_answer(
        adapter,
        bandit,
        context,
        prompt,
        [{"role": "user", "content": prompt}],
        mode,
        engine,
        console,
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


def run_web_ask(prompt: str, random_mode: bool = False, model_override: str | None = None) -> None:
    console = Console()
    config, engine, adapter = _setup()

    if model_override:
        error = _resolve_model_override(adapter, model_override, web_mode=True)
        if error:
            console.print(f"[red]{error}[/red]")
            raise SystemExit(1)

    # a web-search answer reflects a moment in time the same way a
    # volatile query does elsewhere in the cache, caching it risks
    # serving something stale later, so this skips the cache entirely
    # rather than try to guess which searched answers are safe to keep.
    context = classify(prompt)
    mode = "random" if random_mode else "bandit"
    if model_override:
        bandit = _forced_arm_bandit(model_override, engine)
        mode = "manual"
    else:
        arms = filter_to_live(adapter, _WEB_SEARCH_MODELS)
        algorithm_factory = RandomBandit if random_mode else _ALGORITHM_FACTORIES[config.bandit_algorithm]
        bandit = ContextualBandit(lambda _context_key: algorithm_factory(arms), arms=arms)
        replay_history(bandit, engine, mode=mode)

    content, _, _, issues = _route_and_answer(
        adapter,
        bandit,
        context,
        prompt,
        [{"role": "user", "content": prompt}],
        mode,
        engine,
        console,
        extra_body={"tool_ids": ["server:websearch"]},
    )

    console.print(content if content else _failure_message(issues))


def _failure_message(issues: list[QualityIssue]) -> str:
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
    console: Console,
    conversation_id: str | None = None,
    turn_index: int | None = None,
    **extra_kwargs,
) -> tuple[str | None, str, bool, list[QualityIssue]]:
    """Runs one turn through the quality gate, logs every attempt, and
    returns (content, model_used, passed, issues). content can be
    non-None even when passed is False (the last attempt still produced
    text, it just didn't clear the gate); content is None only when
    every arm errored out with nothing to show at all, in which case
    issues carries the reason from the last attempt.

    extra_kwargs passes straight through to call_with_quality_gate, this
    is how RAG's `files` parameter and web search's `tool_ids` reach the
    actual API call.

    The quality gate needs the whole response back, content and
    finish_reason both, before it can tell whether to retry on a
    different arm, so there's nothing to stream here: showing partial
    text that might get thrown away a second later is worse than no
    text at all. A status spinner is the honest way to say "still
    working" without pretending otherwise.
    """
    with console.status("[dim]thinking...[/dim]"):
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
    config, engine, adapter = _setup()

    mode = "random" if random_mode else "bandit"
    arms = known_arms(adapter)
    algorithm_factory = RandomBandit if random_mode else _ALGORITHM_FACTORIES[config.bandit_algorithm]
    bandit = _build_bandit(arms, algorithm_factory, adapter, config.enable_reasoning_variants)
    replay_history(bandit, engine, mode=mode)

    # built lazily, not here, most chat sessions never touch --doc/--web/
    # --image, and each one costs a replay_history() scan of the log,
    # no reason to pay that on every `arcus chat` invocation
    image_bandit: ContextualBandit | None = None
    web_bandit: ContextualBandit | None = None

    conversation_id = str(uuid4())
    messages: list[dict] = []
    turn_index = 0
    # kept separately from messages, which gets trimmed by _trim_history
    # as the conversation grows, a saved transcript should have the
    # whole conversation, not just whatever's still in the active window
    transcript: list[str] = []

    console.print(
        "[bold]chatting with arcus, type 'exit' or ctrl-d to leave.[/bold]\n"
        "[dim]--doc PATH, --web, --image PATH, and --model NAME work inline, "
        "one attachment per turn.[/dim]\n"
    )

    while True:
        try:
            raw_input = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not raw_input:
            continue
        if raw_input.lower() in ("exit", "quit"):
            break

        try:
            tokens = _tokenize_chat_line(raw_input)
        except ValueError as e:
            console.print(f"[red]couldn't parse that line: {e}[/red]")
            continue

        flags = _parse_request_flags(tokens)
        error = _validate_mode_flags(flags)
        if error:
            console.print(f"[red]{error}[/red]")
            continue

        user_input = " ".join(flags.remaining).strip()
        if not user_input:
            console.print("[red]that's just flags, no question.[/red]")
            continue

        if flags.model_override:
            override_error = _resolve_model_override(
                adapter, flags.model_override, web_mode=flags.web_mode, image_mode=bool(flags.image_path)
            )
            if override_error:
                console.print(f"[red]{override_error}[/red]")
                continue

        turn_bandit = bandit
        turn_mode = mode
        turn_kwargs: dict = {}
        label = ""
        history_content: str | list[dict] = user_input
        file_id: str | None = None

        if flags.image_path:
            try:
                history_content = _build_image_content(user_input, flags.image_path)
            except (OSError, ValueError) as e:
                console.print(f"[red]couldn't read '{flags.image_path}':[/red] {e}")
                continue
            if image_bandit is None:
                image_bandit = _forced_arm_bandit(_VISION_MODEL, engine, mode="bandit")
            turn_bandit = image_bandit
            turn_mode = "bandit"
            label = f" (image: {flags.image_path})"

        elif flags.doc_path:
            try:
                file_id = adapter.upload_file(flags.doc_path)
            except OSError as e:
                console.print(f"[red]couldn't read '{flags.doc_path}':[/red] {e}")
                continue
            except httpx.HTTPError as e:
                console.print(f"[red]couldn't upload '{flags.doc_path}' to ARC:[/red] {e}")
                continue
            turn_kwargs = {"extra_body": {"files": [{"type": "file", "id": file_id}]}}
            label = f" (doc: {flags.doc_path})"

        elif flags.web_mode:
            if web_bandit is None:
                web_arms = filter_to_live(adapter, _WEB_SEARCH_MODELS)
                web_factory = RandomBandit if random_mode else _ALGORITHM_FACTORIES[config.bandit_algorithm]
                web_bandit = ContextualBandit(lambda _context_key: web_factory(web_arms), arms=web_arms)
                replay_history(web_bandit, engine, mode=mode)
            turn_bandit = web_bandit
            turn_kwargs = {"extra_body": {"tool_ids": ["server:websearch"]}}
            label = " (web)"

        if flags.model_override:
            turn_bandit = _forced_arm_bandit(flags.model_override, engine)
            turn_mode = "manual"
            label += f" (model: {flags.model_override})"

        messages.append({"role": "user", "content": history_content})
        transcript.append(f"**You{label}:** {user_input}\n")
        context = classify(user_input)

        content, model_used, _, issues = _route_and_answer(
            adapter,
            turn_bandit,
            context,
            user_input,
            messages,
            turn_mode,
            engine,
            console,
            conversation_id=conversation_id,
            turn_index=turn_index,
            **turn_kwargs,
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

        if file_id:
            try:
                adapter.delete_file(file_id)
            except httpx.HTTPError:
                # the answer's already shown, not worth interrupting the
                # chat over cleanup, the file just sits in the user's ARC
                # account until they remove it there themselves
                pass

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
        console.print(f"enable_reasoning_variants: {str(config.enable_reasoning_variants).lower()}")
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

    if len(args) == 3 and args[0] == "set" and args[1] == "enable_reasoning_variants":
        value = args[2].lower()
        if value not in ("true", "false"):
            console.print(f"[red]'{args[2]}' isn't valid, use 'true' or 'false'.[/red]")
            raise SystemExit(1)
        updated = config.model_copy(update={"enable_reasoning_variants": value == "true"})
        save_config(updated)
        console.print(f"[green]enable_reasoning_variants set to {value}[/green]")
        if value == "true":
            console.print(
                "[yellow]this hasn't been confirmed against a live ARC key from this build, "
                "ask a code/math question and check it actually answers before trusting it.[/yellow]"
            )
        return

    console.print(
        "[red]usage: arcus config   or   arcus config set bandit_algorithm "
        f"<{'|'.join(_VALID_BANDIT_ALGORITHMS)}>   or   arcus config set "
        "enable_reasoning_variants <true|false>[/red]"
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


# a bootstrap confidence interval on a handful of rows is technically a
# number but not a meaningful one, this is a plain rule-of-thumb floor,
# not derived from anything, below it the table still prints but gets a
# banner saying not to trust it yet
_MIN_EVAL_EXAMPLES = 30


def run_eval() -> None:
    # wider than the default auto-detected width: "always <model>" policy
    # names are long enough that a normal terminal width truncates them
    # with an ellipsis, and this table is meant to be read carefully
    # (or copied out), not skimmed, so the full names matter more here
    # than they do in the shorter arcus stats table.
    console = Console(width=120)
    engine = get_engine()

    examples = load_logged_examples(engine, mode="bandit")
    if not examples:
        console.print("no eligible logged requests yet, go use arcus a bit first (bandit mode, not --random).")
        return

    if len(examples) < _MIN_EVAL_EXAMPLES:
        console.print(
            f"[yellow]only {len(examples)} logged bandit-mode request(s) so far, fewer than "
            f"{_MIN_EVAL_EXAMPLES}. the numbers below are illustrative, not a reliable comparison yet.[/yellow]\n"
        )

    config = _ensure_config()
    adapter = ArcAdapter(api_key=config.arc_api_key)
    arms = known_arms(adapter)

    policies = {"greedy (best logged avg per context)": greedy_policy_from_log(examples)}
    for arm in arms:
        policies[f"always {arm}"] = policy_always(arm)

    results = evaluate_policies(engine, policies, mode="bandit")

    table = Table(title="offline policy evaluation (IPS / doubly-robust vs. logged history)")
    table.add_column("policy")
    table.add_column("IPS estimate", justify="right")
    table.add_column("IPS 95% CI", justify="right")
    table.add_column("DR estimate", justify="right")
    table.add_column("DR 95% CI", justify="right")

    for row in results:
        table.add_row(
            row.name,
            f"{row.ips_estimate:.3f}",
            f"[{row.ips_ci[0]:.3f}, {row.ips_ci[1]:.3f}]",
            f"{row.dr_estimate:.3f}",
            f"[{row.dr_ci[0]:.3f}, {row.dr_ci[1]:.3f}]",
        )

    console.print(table)
    console.print(f"\nbased on {len(examples)} logged bandit-mode request(s).")


if __name__ == "__main__":
    main()
