"""Hostile-input tests for batch file handling.

Every case here is one that produces a wrong file rather than an error,
which is the only kind that actually costs someone their data.
"""

import csv
import json

import pytest

from arcus.batch.io import (
    BatchInputError,
    ResultWriter,
    guess_text_column,
    read_rows,
    sniff_format,
)


def _write(tmp_path, name, content, encoding="utf-8"):
    p = tmp_path / name
    if isinstance(content, bytes):
        p.write_bytes(content)
    else:
        p.write_text(content, encoding=encoding)
    return p


# --- encoding traps ---------------------------------------------------------


def test_excel_style_utf8_bom_does_not_corrupt_the_first_header(tmp_path):
    # Excel writes this by default. Read as plain utf-8 the first column
    # becomes '﻿id' and every lookup against 'id' silently misses.
    raw = "﻿id,comment\n1,it was fine\n".encode("utf-8")
    path = _write(tmp_path, "excel.csv", raw)

    columns, rows = read_rows(path)

    assert columns == ["id", "comment"]
    assert next(iter(rows)).fields["id"] == "1"


def test_windows_1252_file_still_reads(tmp_path):
    # curly quotes and en dashes from Word, not valid utf-8
    raw = "id,comment\n1,it was \x93fine\x94 \x96 mostly\n".encode("latin-1")
    path = _write(tmp_path, "cp1252.csv", raw)

    columns, rows = read_rows(path)
    row = next(iter(rows))

    assert columns == ["id", "comment"]
    assert "fine" in row.fields["comment"]


def test_emoji_and_non_latin_text_survive(tmp_path):
    path = _write(tmp_path, "unicode.csv", "id,comment\n1,\"课程很好 🎉 מצוין\"\n")

    _, rows = read_rows(path)
    assert "🎉" in next(iter(rows)).fields["comment"]


# --- csv structure traps ----------------------------------------------------


def test_quoted_newlines_do_not_split_one_row_into_two(tmp_path):
    path = _write(
        tmp_path,
        "multiline.csv",
        'id,comment\n1,"first line\nsecond line\nthird line"\n2,short\n',
    )

    _, rows = read_rows(path)
    collected = list(rows)

    assert len(collected) == 2
    assert collected[0].fields["comment"].count("\n") == 2
    assert collected[1].fields["comment"] == "short"


def test_commas_inside_quoted_values_stay_in_one_field(tmp_path):
    path = _write(tmp_path, "commas.csv", 'id,comment\n1,"well organized, but brutal, honestly"\n')

    _, rows = read_rows(path)
    row = next(iter(rows))

    assert row.fields["comment"] == "well organized, but brutal, honestly"


def test_escaped_quotes_inside_values(tmp_path):
    path = _write(tmp_path, "quotes.csv", 'id,comment\n1,"she said ""it was fine"" and left"\n')

    _, rows = read_rows(path)
    assert next(iter(rows)).fields["comment"] == 'she said "it was fine" and left'


def test_duplicate_headers_are_kept_distinct(tmp_path):
    # a plain dict would keep only the last, so a prompt naming the first
    # column would silently read the wrong data
    path = _write(tmp_path, "dupes.csv", "note,note,note\na,b,c\n")

    columns, rows = read_rows(path)
    row = next(iter(rows))

    assert len(set(columns)) == 3
    assert [row.fields[c] for c in columns] == ["a", "b", "c"]


def test_short_rows_pad_instead_of_raising(tmp_path):
    path = _write(tmp_path, "ragged.csv", "id,comment,extra\n1,only two fields\n")

    columns, rows = read_rows(path)
    row = next(iter(rows))

    assert row.fields["extra"] == ""


def test_overlong_rows_keep_their_extra_values(tmp_path):
    # dropping data the user can see in their own file would be worse
    # than surfacing it under a generated name
    path = _write(tmp_path, "long.csv", "id,comment\n1,hello,surprise\n")

    _, rows = read_rows(path)
    row = next(iter(rows))

    assert "surprise" in row.fields.values()


def test_blank_lines_are_not_treated_as_rows(tmp_path):
    path = _write(tmp_path, "blanks.csv", "id,comment\n1,a\n\n\n2,b\n\n")

    _, rows = read_rows(path)
    assert len(list(rows)) == 2


def test_a_cell_larger_than_the_csv_default_limit(tmp_path):
    # csv caps fields at 128k by default; a column holding a document
    # blows straight through it
    big = "x" * 200_000
    path = tmp_path / "huge.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "doc"])
        w.writerow(["1", big])

    _, rows = read_rows(path)
    assert len(next(iter(rows)).fields["doc"]) == 200_000


def test_crlf_line_endings(tmp_path):
    path = _write(tmp_path, "crlf.csv", "id,comment\r\n1,fine\r\n2,good\r\n")

    _, rows = read_rows(path)
    collected = list(rows)

    assert len(collected) == 2
    # the \r must not survive into the value
    assert collected[0].fields["comment"] == "fine"


def test_tsv_uses_tabs(tmp_path):
    path = _write(tmp_path, "data.tsv", "id\tcomment\n1\tfine, really\n")

    columns, rows = read_rows(path)
    assert columns == ["id", "comment"]
    assert next(iter(rows)).fields["comment"] == "fine, really"


# --- jsonl ------------------------------------------------------------------


def test_jsonl_reads_and_unions_keys(tmp_path):
    path = _write(
        tmp_path,
        "data.jsonl",
        '{"id": 1, "comment": "a"}\n{"id": 2, "note": "b"}\n',
    )

    columns, rows = read_rows(path)
    collected = list(rows)

    assert columns == ["id", "comment", "note"]
    assert collected[1].fields["note"] == "b"


def test_jsonl_nested_values_become_json_text(tmp_path):
    path = _write(tmp_path, "nested.jsonl", '{"id": 1, "tags": ["a", "b"]}\n')

    _, rows = read_rows(path)
    assert json.loads(next(iter(rows)).fields["tags"]) == ["a", "b"]


def test_jsonl_reports_the_bad_line_number(tmp_path):
    path = _write(tmp_path, "bad.jsonl", '{"id": 1}\nnot json at all\n')

    with pytest.raises(BatchInputError) as e:
        read_rows(path)

    assert "line 2" in str(e.value)


def test_format_detected_without_a_helpful_extension(tmp_path):
    jsonl = _write(tmp_path, "mystery", '{"a": 1}\n')
    assert sniff_format(jsonl) == "jsonl"

    csvish = _write(tmp_path, "mystery2", "a,b\n1,2\n")
    assert sniff_format(csvish) == "csv"


# --- empty / missing --------------------------------------------------------


def test_empty_file_is_rejected_clearly(tmp_path):
    with pytest.raises(BatchInputError, match="empty"):
        read_rows(_write(tmp_path, "empty.csv", ""))


def test_header_only_file_yields_no_rows(tmp_path):
    _, rows = read_rows(_write(tmp_path, "headeronly.csv", "id,comment\n"))
    assert list(rows) == []


def test_missing_file_is_rejected_clearly(tmp_path):
    with pytest.raises(BatchInputError, match="no such file"):
        read_rows(tmp_path / "nope.csv")


# --- column guessing --------------------------------------------------------


def test_guesses_the_long_free_text_column_over_ids_and_codes(tmp_path):
    path = _write(
        tmp_path,
        "survey.csv",
        "student_id,term,response\n"
        "1041,F26,\"The course was well organized but the pace was genuinely brutal\"\n"
        "1042,F26,\"Honestly I struggled the entire semester and never caught up\"\n",
    )

    columns, rows = read_rows(path)
    assert guess_text_column(columns, list(rows)) == "response"


def test_declines_to_guess_when_every_column_is_empty(tmp_path):
    path = _write(tmp_path, "blank.csv", "a,b\n,\n")
    columns, rows = read_rows(path)
    assert guess_text_column(columns, list(rows)) is None


def test_single_column_needs_no_guessing(tmp_path):
    path = _write(tmp_path, "one.csv", "comment\nhello\n")
    columns, rows = read_rows(path)
    assert guess_text_column(columns, list(rows)) == "comment"


# --- writing back out -------------------------------------------------------


def _roundtrip(tmp_path, answer):
    """Writes one result and reads the output back as a real CSV."""
    src = _write(tmp_path, "in.csv", "id,comment\n1,hello\n")
    columns, rows = read_rows(src)
    row = next(iter(rows))

    out = tmp_path / "out.csv"
    with ResultWriter(out, columns, ["result"]) as w:
        w.write(row, {"result": answer})

    with out.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_a_model_answer_containing_commas_does_not_shift_columns(tmp_path):
    parsed = _roundtrip(tmp_path, "positive, but with reservations, mostly")

    assert len(parsed) == 1
    assert parsed[0]["result"] == "positive, but with reservations, mostly"
    assert parsed[0]["id"] == "1"


def test_a_model_answer_containing_newlines_does_not_split_the_row(tmp_path):
    parsed = _roundtrip(tmp_path, "line one\nline two\nline three")

    assert len(parsed) == 1
    assert parsed[0]["result"].count("\n") == 2


def test_a_model_answer_containing_quotes_survives(tmp_path):
    parsed = _roundtrip(tmp_path, 'he said "it was fine", then left')

    assert parsed[0]["result"] == 'he said "it was fine", then left'


def test_results_are_on_disk_before_the_writer_is_closed(tmp_path):
    # a job killed mid-run has to leave its finished rows behind
    src = _write(tmp_path, "in.csv", "id,c\n1,a\n2,b\n")
    columns, rows = read_rows(src)
    collected = list(rows)

    out = tmp_path / "out.csv"
    writer = ResultWriter(out, columns, ["result"])
    writer.write(collected[0], {"result": "first"})

    # deliberately not closed
    assert "first" in out.read_text(encoding="utf-8")


def test_resuming_appends_instead_of_starting_over(tmp_path):
    src = _write(tmp_path, "in.csv", "id,c\n1,a\n2,b\n")
    columns, rows = read_rows(src)
    collected = list(rows)
    out = tmp_path / "out.csv"

    with ResultWriter(out, columns, ["result"]) as w:
        w.write(collected[0], {"result": "first"})

    with ResultWriter(out, columns, ["result"], resuming=True) as w:
        w.write(collected[1], {"result": "second"})

    with out.open(newline="", encoding="utf-8") as fh:
        parsed = list(csv.DictReader(fh))

    assert [p["result"] for p in parsed] == ["first", "second"]


def test_a_fresh_run_overwrites_rather_than_appending(tmp_path):
    src = _write(tmp_path, "in.csv", "id,c\n1,a\n")
    columns, rows = read_rows(src)
    row = next(iter(rows))
    out = tmp_path / "out.csv"

    with ResultWriter(out, columns, ["result"]) as w:
        w.write(row, {"result": "old"})
    with ResultWriter(out, columns, ["result"]) as w:
        w.write(row, {"result": "new"})

    with out.open(newline="", encoding="utf-8") as fh:
        parsed = list(csv.DictReader(fh))

    assert [p["result"] for p in parsed] == ["new"]


def test_multiple_output_columns(tmp_path):
    src = _write(tmp_path, "in.csv", "id,abstract\n1,some paper\n")
    columns, rows = read_rows(src)
    row = next(iter(rows))
    out = tmp_path / "out.csv"

    with ResultWriter(out, columns, ["sample_size", "method"]) as w:
        w.write(row, {"sample_size": "42", "method": "randomized trial"})

    with out.open(newline="", encoding="utf-8") as fh:
        parsed = list(csv.DictReader(fh))

    assert parsed[0]["sample_size"] == "42"
    assert parsed[0]["method"] == "randomized trial"
