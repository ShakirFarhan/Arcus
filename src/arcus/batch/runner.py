"""Running one instruction across many rows.

The routing this project does everywhere else is deliberately switched
off here. A dataset labelled half by one model and half by another has a
confound baked into it that no amount of downstream analysis removes, so
a batch job picks one model and stays with it, and a row that model
can't answer is recorded as failed rather than quietly handed to a
different labeller.
"""

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from openai import APIError, PermissionDeniedError, RateLimitError

from arcus.adapters.arc_adapter import ArcAdapter
from arcus.batch.io import InputRow
from arcus.quality.gate import is_concurrency_limit

# ARC caps concurrent requests at 10 per account, verified live: 24
# requests spread across four models returned exactly 10 completions and
# 14 rejections, so spreading a job over models buys no extra
# parallelism. Staying under the cap leaves room for the user's own
# interactive arcus in another terminal, which would otherwise start
# failing the moment a batch job ran.
DEFAULT_CONCURRENCY = 6
MAX_CONCURRENCY = 9

# retries against the same model, never a different one
MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 2.0


@dataclass
class RowResult:
    row: InputRow
    output: str | None
    error: str | None
    attempts: int
    model: str
    latency_ms: float

    @property
    def ok(self) -> bool:
        return self.output is not None


@dataclass
class Progress:
    done: int = 0
    failed: int = 0
    total: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def completed(self) -> int:
        return self.done + self.failed

    @property
    def eta_seconds(self) -> float | None:
        if self.completed < 3:
            return None
        rate = self.completed / max(time.monotonic() - self.started_at, 1e-9)
        remaining = self.total - self.completed
        return remaining / rate if rate > 0 else None


class Aborted(Exception):
    """Raised when the whole job can't continue, as opposed to one row."""


def build_prompt(instruction: str, row: InputRow, column: str, examples: list[tuple[str, str]] | None = None) -> str:
    """Assembles the request for one row.

    The instruction is written in plain language by the user; the row's
    text is kept clearly separated from it so a comment that happens to
    contain something imperative reads as data rather than as a second
    instruction.
    """
    parts = [instruction.strip()]

    if examples:
        parts.append("\nExamples:")
        for text, answer in examples:
            parts.append(f"\nInput: {text}\nAnswer: {answer}")

    parts.append(f"\nInput: {row.text_for(column)}\nAnswer:")
    return "\n".join(parts)


def normalize(answer: str) -> str:
    return answer.strip().strip(".").strip()


def match_choice(answer: str, choices: list[str]) -> str | None:
    """Maps a model's answer onto one of the allowed labels.

    Models drift into `Positive.` or `Sentiment: positive` or
    `**positive**` even when told not to, and at five thousand rows
    nobody notices until the analysis is wrong. Exact match first, then a
    contained-word match, then give up rather than guess: a wrong label
    recorded confidently is worse than a row flagged for review.
    """
    cleaned = normalize(answer).lower().strip("*_`")
    lowered = {c.lower(): c for c in choices}

    if cleaned in lowered:
        return lowered[cleaned]

    # `Sentiment: positive` and `**positive**` and `positive!`
    words = set(re.findall(r"[a-z0-9_-]+", cleaned))
    hits = [original for lower, original in lowered.items() if lower in words]
    if len(hits) == 1:
        return hits[0]

    return None


def run_batch(
    adapter: ArcAdapter,
    model: str,
    rows: list[InputRow],
    instruction: str,
    column: str,
    *,
    choices: list[str] | None = None,
    examples: list[tuple[str, str]] | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    on_result: Callable[[RowResult, Progress], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Progress:
    """Runs `instruction` over `rows`, one model throughout.

    Results are handed to `on_result` as they finish rather than
    collected and returned, so the caller can write each one to disk
    immediately. A job killed halfway then leaves half a usable file
    behind instead of nothing.
    """
    concurrency = max(1, min(concurrency, MAX_CONCURRENCY))
    progress = Progress(total=len(rows))
    lock = threading.Lock()
    stop = threading.Event()
    aborted: list[Aborted] = []

    def handle(row: InputRow) -> None:
        # every queued task checks this before doing anything, so an
        # abort halfway through a 10,000 row job skips the remaining
        # 9,000-odd instead of firing them all at a connection that has
        # already refused. The handful already in flight still finish;
        # an HTTP request in progress can't be cleanly recalled.
        if stop.is_set():
            return

        try:
            result = _run_row(adapter, model, row, instruction, column, choices, examples)
        except Aborted as e:
            stop.set()
            with lock:
                aborted.append(e)
            return

        with lock:
            if result.ok:
                progress.done += 1
            else:
                progress.failed += 1
            if on_result is not None:
                on_result(result, progress)

        if should_stop is not None and should_stop():
            stop.set()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(handle, row) for row in rows]
        for future in futures:
            future.result()

    if aborted:
        raise aborted[0]

    return progress


def _run_row(
    adapter: ArcAdapter,
    model: str,
    row: InputRow,
    instruction: str,
    column: str,
    choices: list[str] | None,
    examples: list[tuple[str, str]] | None,
) -> RowResult:
    prompt = build_prompt(instruction, row, column, examples)
    started = time.monotonic()
    last_error = "no attempt made"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            completion = adapter.chat(model, [{"role": "user", "content": prompt}])
            content = completion.choices[0].message.content or ""

            if not content.strip():
                last_error = "empty response"
            elif choices:
                matched = match_choice(content, choices)
                if matched is None:
                    last_error = f"answer not one of the allowed choices: {content.strip()[:80]!r}"
                else:
                    return _ok(row, matched, attempt, model, started)
            else:
                return _ok(row, normalize(content), attempt, model, started)

        except PermissionDeniedError as e:
            # the VPN dropped or access was revoked. that is true of
            # every remaining row, so failing them one at a time would
            # just be a slow way of failing the job.
            raise Aborted(f"ARC refused the request: {e}") from e

        except (RateLimitError, APIError) as e:
            if is_concurrency_limit(e):
                # the account's own connection budget, not this row's
                # fault and not this model's, so it doesn't count as an
                # attempt
                time.sleep(_BACKOFF_SECONDS * attempt)
                continue
            last_error = str(e)[:200]

        if attempt < MAX_ATTEMPTS:
            time.sleep(_BACKOFF_SECONDS * (attempt - 1) if attempt > 1 else 0)

    return RowResult(
        row=row,
        output=None,
        error=last_error,
        attempts=MAX_ATTEMPTS,
        model=model,
        latency_ms=(time.monotonic() - started) * 1000,
    )


def _ok(row: InputRow, output: str, attempt: int, model: str, started: float) -> RowResult:
    return RowResult(
        row=row,
        output=output,
        error=None,
        attempts=attempt,
        model=model,
        latency_ms=(time.monotonic() - started) * 1000,
    )
