import pytest
from sqlmodel import SQLModel, create_engine

from arcus.batch import state


@pytest.fixture
def engine():
    e = create_engine("sqlite://")
    SQLModel.metadata.create_all(e)
    return e


def _job(engine, **overrides):
    fields = dict(
        fingerprint="fp-1",
        input_path="/tmp/in.csv",
        output_path="/tmp/out.csv",
        instruction="classify the sentiment",
        column="comment",
        model="GLM-5.3",
        total_rows=100,
    )
    fields.update(overrides)
    return state.create_job(engine, **fields)


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# --- identifying a job ------------------------------------------------------


def test_the_same_command_on_the_same_file_resumes(tmp_path):
    f = _write(tmp_path, "in.csv", "id,c\n1,a\n2,b\n")
    a = state.fingerprint(f, "classify the sentiment", "c")
    b = state.fingerprint(f, "classify the sentiment", "c")
    assert a == b


def test_changing_the_instruction_starts_a_new_job(tmp_path):
    f = _write(tmp_path, "in.csv", "id,c\n1,a\n")
    assert state.fingerprint(f, "classify sentiment", "c") != state.fingerprint(f, "classify topic", "c")


def test_changing_the_column_starts_a_new_job(tmp_path):
    f = _write(tmp_path, "in.csv", "id,c,d\n1,a,b\n")
    assert state.fingerprint(f, "classify", "c") != state.fingerprint(f, "classify", "d")


def test_editing_the_file_starts_a_new_job(tmp_path):
    f = _write(tmp_path, "in.csv", "id,c\n1,a\n")
    before = state.fingerprint(f, "classify", "c")
    _write(tmp_path, "in.csv", "id,c\n1,a\n2,COMPLETELY DIFFERENT\n")
    assert state.fingerprint(f, "classify", "c") != before


def test_merely_touching_a_file_does_not_throw_away_a_resumable_job(tmp_path):
    # mtime changes for all sorts of innocent reasons; discarding hours
    # of finished work over one would be its own bug
    import os
    import time

    f = _write(tmp_path, "in.csv", "id,c\n1,a\n")
    before = state.fingerprint(f, "classify", "c")
    time.sleep(0.01)
    os.utime(f, None)
    assert state.fingerprint(f, "classify", "c") == before


def test_the_same_content_at_a_different_path_is_a_different_job(tmp_path):
    a = _write(tmp_path, "a.csv", "id,c\n1,x\n")
    b = _write(tmp_path, "b.csv", "id,c\n1,x\n")
    assert state.fingerprint(a, "classify", "c") != state.fingerprint(b, "classify", "c")


# --- resume -----------------------------------------------------------------


def test_a_fresh_job_has_every_row_pending(engine):
    job = _job(engine)
    state.register_rows(engine, job.id, list(range(50)))

    assert state.pending_indexes(engine, job.id) == set(range(50))
    assert state.counts(engine, job.id)[state.PENDING] == 50


def test_finished_rows_drop_out_of_the_pending_set(engine):
    job = _job(engine)
    state.register_rows(engine, job.id, list(range(10)))

    for i in range(4):
        state.record_result(engine, job.id, i, status=state.DONE, output="positive", model="GLM-5.3")

    assert state.pending_indexes(engine, job.id) == set(range(4, 10))


def test_a_crash_at_row_four_thousand_resumes_at_four_thousand(engine):
    job = _job(engine, total_rows=10_000)
    state.register_rows(engine, job.id, list(range(10_000)))
    for i in range(4_000):
        state.record_result(engine, job.id, i, status=state.DONE, output="x", model="GLM-5.3")

    remaining = state.pending_indexes(engine, job.id)

    assert len(remaining) == 6_000
    assert min(remaining) == 4_000


def test_failed_rows_are_not_retried_on_resume_but_are_reportable(engine):
    job = _job(engine)
    state.register_rows(engine, job.id, list(range(5)))
    state.record_result(engine, job.id, 2, status=state.FAILED, error="unmappable answer", model="GLM-5.3")

    assert 2 not in state.pending_indexes(engine, job.id)
    failed = state.failed_items(engine, job.id)
    assert [f.row_index for f in failed] == [2]
    assert failed[0].error == "unmappable answer"


def test_recording_the_same_row_twice_updates_rather_than_duplicating(engine):
    job = _job(engine)
    state.register_rows(engine, job.id, [0])
    state.record_result(engine, job.id, 0, status=state.FAILED, error="first try", model="GLM-5.3")
    state.record_result(engine, job.id, 0, status=state.DONE, output="positive", model="GLM-5.3")

    counts = state.counts(engine, job.id)
    assert counts[state.DONE] == 1
    assert counts[state.FAILED] == 0


def test_which_model_answered_each_row_is_recorded(engine):
    # a manifest that merely asserts one model was used is worth less
    # than one that can show it row by row
    job = _job(engine)
    state.register_rows(engine, job.id, [0, 1])
    state.record_result(engine, job.id, 0, status=state.DONE, output="a", model="GLM-5.3")
    state.record_result(engine, job.id, 1, status=state.FAILED, error="x", model="GLM-5.3")

    assert {f.model for f in state.failed_items(engine, job.id)} == {"GLM-5.3"}


def test_finding_an_existing_job_by_fingerprint(engine):
    job = _job(engine, fingerprint="abc123")
    found = state.find_job(engine, "abc123")

    assert found is not None and found.id == job.id
    assert state.find_job(engine, "nothing-like-it") is None


def test_forgetting_a_job_removes_its_rows_too(engine):
    # batch rows hold the user's own research data, so removing it has
    # to actually remove it
    job = _job(engine)
    state.register_rows(engine, job.id, list(range(20)))
    state.record_result(engine, job.id, 0, status=state.DONE, output="x", model="GLM-5.3")

    state.forget_job(engine, job.id)

    assert state.find_job(engine, job.fingerprint) is None
    assert state.counts(engine, job.id) == {state.PENDING: 0, state.DONE: 0, state.FAILED: 0}
