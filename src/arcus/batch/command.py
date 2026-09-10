"""The `arcus batch` command.

Shaped around one idea: the user types a file and a sentence, and the
tool does the deciding. Every flag is an override of something that
already has a sensible default, and the expensive part never starts
until they've seen real output from their own data.
"""

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

from arcus.adapters.arc_adapter import ArcAdapter, context_limit
from arcus.batch import state
from arcus.batch.io import (
    InputRow,
)
from arcus.batch.quality import compare_opinions
from arcus.batch.runner import (
    DEFAULT_CONCURRENCY,
    RowResult,
)
from arcus.routing.context import estimate_tokens

PREVIEW_ROWS = 5
CROSS_CHECK_FRACTION = 0.2


@dataclass
class BatchOptions:
    input_path: Path
    instruction: str
    column: str | None = None
    output_path: Path | None = None
    choices: list[str] | None = None
    model: str | None = None
    concurrency: int = DEFAULT_CONCURRENCY
    cross_check: bool = True
    assume_yes: bool = False
    limit: int | None = None


# a real label list is short and sits at the end of the sentence. these
# bounds exist because the failure is asymmetric: inferring labels that
# aren't there makes every valid answer look invalid and fails the whole
# job, while missing a list the user did write just means they pass
# --choices. So the rules err heavily towards not guessing.
_MAX_TAIL_CHARS = 60
_MAX_WORDS_PER_LABEL = 2
_MAX_LABELS = 6


def infer_choices(instruction: str) -> list[str] | None:
    """Pulls the allowed labels out of a plain-English instruction.

    "classify the sentiment as positive, negative, or neutral" visibly
    names three categories, and a user who wrote that sentence should
    not have to restate them as a flag.

    Prose is the thing to protect against. An instruction like "describe
    what they found difficult, what they enjoyed, and anything they
    suggested changing" is full of commas and reads exactly like a list
    to a naive split, and treating those fragments as the only permitted
    answers would reject every real response the model produced.
    """
    match = re.search(
        r"\b(?:as|into|either)\b[:\s]+(.+?)(?:\.|$)",
        instruction.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None

    tail = match.group(1).strip()
    # "mark it as either urgent or routine" matches on `as`, leaving the
    # `either` behind at the front of the list
    tail = re.sub(r"^either\s+", "", tail, flags=re.IGNORECASE)

    # a genuine list of labels is short. a sentence that merely contains
    # commas is not.
    if len(tail) > _MAX_TAIL_CHARS:
        return None

    parts = re.split(r",|\bor\b|\band\b|/", tail, flags=re.IGNORECASE)
    labels = [p.strip().strip("\"'`.").lower() for p in parts]
    labels = [label for label in labels if label]

    if not labels or any(len(label.split()) > _MAX_WORDS_PER_LABEL for label in labels):
        return None

    # two is a real choice; one is a misread
    if 2 <= len(labels) <= _MAX_LABELS and len(labels) == len(set(labels)):
        return labels
    return None


def resolve_output_path(input_path: Path, given: Path | None) -> Path:
    if given is not None:
        return given
    return input_path.with_name(f"{input_path.stem}.labeled{input_path.suffix or '.csv'}")


def pick_model(adapter: ArcAdapter, rows: list[InputRow], column: str, candidates: list[str]) -> str | None:
    """Chooses the one model the whole job will use.

    Sized against the *largest* row rather than a typical one: a job that
    commits to a model which can't hold its biggest input discovers that
    at the end, after the waiting.
    """
    if not candidates:
        return None
    biggest = max((estimate_tokens(row.text_for(column)) for row in rows), default=0)
    # leave room for the instruction, any examples, and the answer
    needed = biggest + 2_000
    fitting = [c for c in candidates if (limit := context_limit(c)) is None or needed <= limit]
    return fitting[0] if fitting else None


def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f}m"
    return f"{minutes / 60:.1f}h"


def estimate_runtime(sample_latencies_ms: list[float], remaining: int, concurrency: int) -> str:
    """Turns preview timings into a range, never a single number.

    A five-row sample is far too small to extrapolate confidently from:
    measured latency on this service varies about twofold from the
    model's own nondeterminism alone, before load is considered. Quoting
    "6 minutes" from five rows would be false precision.
    """
    if not sample_latencies_ms or remaining <= 0:
        return "unknown"
    average = sum(sample_latencies_ms) / len(sample_latencies_ms) / 1000
    per_row = average / max(concurrency, 1)
    low = per_row * remaining * 0.6
    high = per_row * remaining * 2.0
    return f"{_fmt_duration(low)} to {_fmt_duration(high)}"


def write_manifest(path: Path, job: state.BatchJob, summary: dict) -> None:
    """Records what produced these answers.

    Anyone publishing findings from machine-labelled data should be able
    to say exactly what did the labelling, and increasingly gets asked.
    """
    payload = {
        "tool": "arcus",
        "finished_at": datetime.now(UTC).isoformat(),
        "input": job.input_path,
        "output": job.output_path,
        "instruction": job.instruction,
        "column": job.column,
        "model": job.model,
        "allowed_answers": json.loads(job.choices) if job.choices else None,
        "examples_used": job.examples_used,
        "rows": summary,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_preview(console: Console, results: list[RowResult], column: str) -> None:
    for result in results:
        text = result.row.text_for(column)
        shown = text if len(text) <= 68 else text[:65] + "..."
        answer = result.output if result.ok else f"[red]failed: {result.error}[/red]"
        console.print(f"    [dim]row {result.row.index + 1:<5}[/dim] {shown}")
        console.print(f"    {' ' * 10} [green]→ {answer}[/green]\n")


def cross_check_report(
    console: Console,
    primary: dict[int, str],
    second: dict[int, str],
    second_model: str,
) -> list[tuple[int, str, str]]:
    """Reports how often a second model reached the same answer.

    Framed carefully. Agreement between two language models is not
    accuracy: their training overlaps, so their mistakes correlate, and
    they can agree confidently while both being wrong. What it does tell
    you is where the data is ambiguous, which is the half of the story
    worth acting on.
    """
    result = compare_opinions(primary, second)
    if not result.compared:
        return []

    console.print(
        f"  cross-checked {result.compared} rows against {second_model}: "
        f"[bold]{result.rate:.0%}[/bold] matched"
    )
    if result.disagreements:
        console.print(
            f"  [yellow]{len(result.disagreements)} disagreed[/yellow] "
            "[dim](where two models differ the row is genuinely ambiguous; "
            "agreement is not proof the rest are right)[/dim]"
        )
    return result.disagreements
