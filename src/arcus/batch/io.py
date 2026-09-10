"""Reading rows in and writing results out.

Everything here is about not destroying someone's data. A batch job is
usually operating on the only copy of a dataset somebody spent weeks
collecting, and the failure modes are silent: a stray comma in a model's
answer shifts every later column, a Windows-exported CSV carries a BOM
that turns the first header into something nothing matches, an unquoted
newline splits one row into two. None of those raise an error. They just
produce a file that looks fine and is wrong.
"""

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Excel writes UTF-8 with a byte-order mark, and it is by a wide margin
# the most common way a spreadsheet reaches a tool like this. Reading it
# as plain utf-8 leaves a zero-width ﻿ glued to the first header,
# so `id` silently becomes `﻿id` and every lookup against it misses.
# utf-8-sig strips it when present and behaves as utf-8 when not.
_PRIMARY_ENCODING = "utf-8-sig"

# Anything that isn't valid UTF-8 is almost always a Windows-1252 export
# (curly quotes, en dashes). latin-1 decodes every byte sequence without
# raising, so it's the honest last resort: better a slightly wrong
# character than refusing to read the file at all.
_FALLBACK_ENCODING = "latin-1"

# csv caps field size at 128k by default, which a column holding a whole
# document will exceed. The limit exists to bound memory on hostile
# input; these are files the user chose to hand us.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


class BatchInputError(Exception):
    """The input file can't be used, with a reason worth showing a user."""


@dataclass(frozen=True)
class InputRow:
    index: int
    fields: dict[str, str]

    def text_for(self, column: str) -> str:
        return (self.fields.get(column) or "").strip()


def _decode(path: Path) -> tuple[list[str], str]:
    """Reads the file, returning its lines and the encoding that worked."""
    raw = path.read_bytes()
    for encoding in (_PRIMARY_ENCODING, _FALLBACK_ENCODING):
        try:
            return raw.decode(encoding).splitlines(keepends=True), encoding
        except UnicodeDecodeError:
            continue
    raise BatchInputError(f"couldn't decode '{path.name}' as text")


def sniff_format(path: Path) -> str:
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        return "jsonl"
    if path.suffix.lower() in (".csv", ".tsv", ".txt"):
        return "csv"
    # fall back to looking at the first non-blank line rather than
    # trusting an extension that might just be missing
    lines, _ = _decode(path)
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        return "jsonl" if stripped.startswith("{") else "csv"
    raise BatchInputError(f"'{path.name}' is empty")


def read_rows(path: Path) -> tuple[list[str], Iterator[InputRow]]:
    """Returns the column names and a lazy iterator over the rows.

    Deliberately streamed: a batch input can be hundreds of megabytes and
    there's no reason to hold it all in memory when every row is handled
    independently.
    """
    if not path.exists():
        raise BatchInputError(f"no such file: {path}")

    fmt = sniff_format(path)
    lines, encoding = _decode(path)

    if fmt == "jsonl":
        return _read_jsonl(lines, path)
    return _read_csv(lines, path, encoding)


def _read_csv(lines: list[str], path: Path, encoding: str):
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    reader = csv.reader(lines, delimiter=delimiter)

    try:
        header = next(reader)
    except StopIteration:
        raise BatchInputError(f"'{path.name}' is empty") from None

    columns = _dedupe(header)
    if not any(c.strip() for c in columns):
        raise BatchInputError(f"'{path.name}' has no usable header row")

    def rows() -> Iterator[InputRow]:
        for i, values in enumerate(reader):
            if not any(v.strip() for v in values):
                continue  # blank line, not a row
            # short rows pad, long rows keep the overflow rather than
            # dropping data the user can see in their own file
            fields = {c: (values[j] if j < len(values) else "") for j, c in enumerate(columns)}
            for extra in range(len(columns), len(values)):
                fields[f"column_{extra + 1}"] = values[extra]
            yield InputRow(index=i, fields=fields)

    return columns, rows()


def _read_jsonl(lines: list[str], path: Path):
    columns: list[str] = []
    parsed: list[dict] = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise BatchInputError(f"{path.name} line {lineno} isn't valid JSON: {e.msg}") from e
        if not isinstance(obj, dict):
            raise BatchInputError(f"{path.name} line {lineno} is a {type(obj).__name__}, expected an object")
        parsed.append(obj)
        for key in obj:
            if key not in columns:
                columns.append(key)

    if not parsed:
        raise BatchInputError(f"'{path.name}' is empty")

    def rows() -> Iterator[InputRow]:
        for i, obj in enumerate(parsed):
            yield InputRow(index=i, fields={k: _stringify(v) for k, v in obj.items()})

    return columns, rows()


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _dedupe(header: list[str]) -> list[str]:
    """Makes column names unique.

    Duplicate headers are common in exported spreadsheets, and a plain
    dict would silently keep only the last one, so a prompt referring to
    the first column would quietly read the wrong data.
    """
    seen: dict[str, int] = {}
    result = []
    for raw in header:
        name = raw.strip() or "column"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        result.append(name)
    return result


def guess_text_column(columns: list[str], sample: list[InputRow]) -> str | None:
    """Picks the column most likely to hold the text being asked about.

    Longest average content wins, which is a crude rule that happens to
    be right nearly every time: the free-text answer in a survey export
    dwarfs the ids, timestamps and codes around it. Returns None when
    there's nothing to go on, and the caller asks instead of guessing.
    """
    if not columns:
        return None
    if len(columns) == 1:
        return columns[0]
    if not sample:
        return None

    def average_length(column: str) -> float:
        lengths = [len(row.text_for(column)) for row in sample]
        return sum(lengths) / len(lengths) if lengths else 0.0

    best = max(columns, key=average_length)
    return best if average_length(best) > 0 else None


class ResultWriter:
    """Writes results as they finish, with the input columns preserved.

    Appends rather than buffering, so a job killed at row 4,000 leaves
    4,000 usable rows on disk instead of an empty file. Uses the csv
    module's quoting rather than joining strings, because a model answer
    containing a comma, a quote or a newline is routine and hand-rolled
    writing corrupts the file without ever raising.
    """

    def __init__(self, path: Path, columns: list[str], output_columns: list[str], resuming: bool = False):
        self.path = path
        self.columns = columns
        self.output_columns = output_columns
        exists = path.exists() and path.stat().st_size > 0
        self._handle = path.open("a" if (resuming and exists) else "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle, quoting=csv.QUOTE_MINIMAL)
        if not (resuming and exists):
            self._writer.writerow([*columns, *output_columns])
            self._handle.flush()

    def write(self, row: InputRow, outputs: dict[str, str]) -> None:
        self._writer.writerow(
            [row.fields.get(c, "") for c in self.columns]
            + [outputs.get(c, "") for c in self.output_columns]
        )
        # flushed per row on purpose: the whole point is that an
        # interrupted job leaves finished work behind
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
