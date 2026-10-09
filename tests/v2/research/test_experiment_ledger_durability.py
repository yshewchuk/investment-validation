"""Synthetic CSV durability probes; every ledger and fault belongs to tmp_path."""
from __future__ import annotations

import csv
import fcntl
import io
import multiprocessing
import os
import stat
from pathlib import Path

import pytest

from experiments import lib

KEY = ("id", "spec_hash", "stage")
CTX = multiprocessing.get_context("spawn")


def _row(number=901, **changes):
    return {
        "id": f"EXP-{number}", "spec_hash": f"synthetic-{number}",
        "date": "2026-01-01", "stage": "ran", "oos_mean_mid": "0.1",
        "sharpe_trade": "1.2", "promoted": "False", **changes,
    }


def _csv(rows=(), fields=None, **options):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields or lib.LEDGER_COLUMNS, **options)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, strict=True))


def _worker(path, task, start, results):
    results.put(("ready",))
    try:
        if not start.wait(30):
            raise TimeoutError("parent did not release synthetic start barrier")
        action, rows, unique_by = task
        if action == "ensure":
            lib.ledger_ensure(path)
            count = 0
        elif action == "outer":
            # #555's refusal identity lock encloses append and must stay distinct.
            with Path(f"{path}.lock").open("a") as outer:
                fcntl.flock(outer, fcntl.LOCK_EX)
                count = lib.ledger_append(rows, path, unique_by=unique_by)
        else:
            count = lib.ledger_append(rows, path, unique_by=unique_by)
        results.put(("ok", count))
    except Exception as exc:
        results.put((type(exc).__name__, str(exc)))


def _stop_process(process):
    if process.pid is None:
        return
    if process.is_alive():
        process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)


def _race(path, tasks):
    start, results = CTX.Event(), CTX.Queue()
    processes = [CTX.Process(target=_worker, args=(path, task, start, results))
                 for task in tasks]
    try:
        for process in processes:
            process.start()
        assert [results.get(timeout=20) for _ in processes] == [("ready",)] * len(processes)
        start.set()
        outcomes = [results.get(timeout=20) for _ in processes]
        for process in processes:
            process.join(timeout=20)
            assert process.exitcode == 0, "synthetic ledger worker stalled or crashed"
        return outcomes
    finally:
        start.set()
        for process in processes:
            _stop_process(process)
        results.close()
        results.join_thread()


def test_first_creation_races_ensure_and_appends_without_losing_rows(tmp_path):
    path = tmp_path / "nested" / "ledger.csv"
    tasks = [("ensure", [], None)] * 3 + [
        ("append", [_row(n)], None) for n in range(901, 908)
    ]
    outcomes = _race(path, tasks)
    assert sorted(outcomes) == [("ok", 0)] * 3 + [("ok", 1)] * 7
    assert sorted(row["id"] for row in _read(path)) == [f"EXP-{n}" for n in range(901, 908)]
    assert path.read_bytes().count(_csv()) == 1
    lock = Path(f"{path}.append.lock")
    inode = lock.stat().st_ino
    before = path.read_bytes()
    assert lib.ledger_ensure(path) == path
    assert lib.ledger_append([_row(908)], path) == 1
    assert path.read_bytes().startswith(before)
    assert lock.stat().st_ino == inode


def test_outer_refusal_lock_and_ordinary_append_do_not_deadlock(tmp_path):
    path = tmp_path / "ledger.csv"
    outcomes = _race(path, [
        ("outer", [_row(901)], KEY), ("append", [_row(902)], None),
        ("outer", [_row(903)], KEY), ("append", [_row(904)], None),
    ])
    assert outcomes == [("ok", 1)] * 4
    assert len(_read(path)) == 4
    assert Path(f"{path}.lock").stat().st_ino != Path(f"{path}.append.lock").stat().st_ino


@pytest.mark.parametrize("conflict", [False, True])
def test_concurrent_keyed_replays_have_one_winner(tmp_path, conflict):
    path = tmp_path / "ledger.csv"
    rows = [_row(date=f"2026-01-0{i + 1}" if conflict else "2026-01-01") for i in range(5)]
    outcomes = _race(path, [("append", [row], KEY) for row in rows])
    assert outcomes.count(("ok", 1)) == 1
    if conflict:
        assert sum(outcome[0] == "LedgerError" for outcome in outcomes) == 4
    else:
        assert outcomes.count(("ok", 0)) == 4
    assert len(_read(path)) == 1
    assert _read(path)[0] in rows


def test_legacy_duplicates_and_keyed_batch_counts(tmp_path):
    path = tmp_path / "ledger.csv"
    assert lib.ledger_append([_row(), _row()], path) == 2
    before = path.read_bytes()
    assert lib.ledger_append([_row()], path) == 1
    assert path.read_bytes().startswith(before)
    before = path.read_bytes()
    with pytest.raises(lib.LedgerError):
        lib.ledger_append([_row()], path, unique_by=KEY)
    assert path.read_bytes() == before
    clean = tmp_path / "keyed.csv"
    assert lib.ledger_append([_row(), _row(), _row(902)], clean, unique_by=KEY) == 2
    before = clean.read_bytes()
    assert lib.ledger_append([_row(902), _row(903), _row(903)], clean, unique_by=KEY) == 1
    assert clean.read_bytes().startswith(before)
    assert [row["id"] for row in _read(clean)] == ["EXP-901", "EXP-902", "EXP-903"]


@pytest.mark.parametrize("existing", [False, True])
def test_conflicting_keyed_batch_is_all_or_nothing(tmp_path, existing):
    path = tmp_path / "ledger.csv"
    if existing:
        lib.ledger_append([_row()], path)
    before = path.read_bytes() if existing else None
    rows = [_row(902), _row(), _row(date="2026-02-01")]
    with pytest.raises(lib.LedgerError):
        lib.ledger_append(rows, path, unique_by=KEY)
    assert (path.read_bytes() if path.exists() else None) == before


def test_complete_serialized_row_controls_replay_and_preserves_header_bytes(tmp_path):
    path = tmp_path / "ledger.csv"
    fields = ["note", *reversed(lib.LEDGER_COLUMNS)]
    stored = _row(note='a comma, a "quote"\nand a newline', oos_mean_mid="", sharpe_trade="1")
    before = _csv([stored], fields, quoting=csv.QUOTE_ALL, lineterminator="\n")
    path.write_bytes(before)
    replay = {**stored, "oos_mean_mid": None, "sharpe_trade": 1, "ignored": "not a column"}
    assert lib.ledger_append([replay], path, unique_by=KEY) == 0
    assert path.read_bytes() == before
    with pytest.raises(lib.LedgerError):
        lib.ledger_append([{**replay, "note": "contradictory extra column"}], path, unique_by=KEY)
    assert path.read_bytes() == before
    assert lib.ledger_append([_row(902)], path, unique_by=KEY) == 1
    assert path.read_bytes().startswith(before)
    assert _read(path) == [stored, {"note": "", **_row(902)}]


@pytest.mark.parametrize("unique_by", [(), "id", ("unknown",), ("id", "id"), (1,)])
def test_invalid_replay_definition_never_publishes(tmp_path, unique_by):
    path = tmp_path / "ledger.csv"
    with pytest.raises(lib.LedgerError):
        lib.ledger_append([_row()], path, unique_by=unique_by)
    assert not path.exists()


@pytest.mark.parametrize("key_value", ["", None])
def test_empty_replay_cells_normalize_without_native_id_validation(tmp_path, key_value):
    path = tmp_path / "ledger.csv"
    assert lib.ledger_append([_row(id=key_value)], path, unique_by=KEY) == 1
    before = path.read_bytes()
    assert lib.ledger_append([_row(id="")], path, unique_by=KEY) == 0
    assert path.read_bytes() == before
    assert _read(path) == [_row(id="")]


def test_replay_can_use_an_existing_extended_header_column(tmp_path):
    path = tmp_path / "ledger.csv"
    path.write_bytes(_csv(fields=[*lib.LEDGER_COLUMNS, "run_id"]))
    rows = [_row(run_id="run-a"), _row(run_id="run-b")]
    assert lib.ledger_append(rows, path, unique_by=("run_id",)) == 2
    assert lib.ledger_append(rows, path, unique_by=("run_id",)) == 0
    assert _read(path) == rows


@pytest.mark.parametrize("existing", [False, True])
def test_missing_required_incoming_column_never_partially_publishes(tmp_path, existing):
    path = tmp_path / "ledger.csv"
    if existing:
        lib.ledger_append([_row()], path)
    before = path.read_bytes() if existing else None
    incomplete = _row(903)
    del incomplete["promoted"]
    with pytest.raises(lib.LedgerError):
        lib.ledger_append([_row(902), incomplete], path)
    assert (path.read_bytes() if path.exists() else None) == before


@pytest.mark.parametrize("payload", [
    b"", b"id,spec_hash\r\n", _csv().rstrip(b"\r\n"),
    _csv() + b'"unterminated\r\n', _csv() + b"short,row\r\n",
    _csv() + b"1,2,3,4,5,6,7,8\r\n", _csv([_row()]).rstrip(b"\r\n"),
    _csv().rstrip(b"\r\n") + b",id\r\n", _csv() + b"\xff\r\n",
])
@pytest.mark.parametrize("operation", ["ensure", "append"])
def test_malformed_existing_csv_is_refused_unchanged(tmp_path, payload, operation):
    path = tmp_path / "ledger.csv"
    path.write_bytes(payload)
    with pytest.raises(lib.LedgerError):
        if operation == "ensure":
            lib.ledger_ensure(path)
        else:
            lib.ledger_append([_row(902)], path)
    assert path.read_bytes() == payload


def test_empty_batch_and_default_path_remain_inside_test_root(tmp_path):
    assert lib.LEDGER_PATH.is_relative_to(tmp_path)
    assert lib.ledger_append([]) == 0
    path = lib.LEDGER_PATH
    assert path.read_bytes() == _csv()
    assert lib.ledger_append([_row()], unique_by=KEY) == 1
    before = path.read_bytes()
    inode = path.stat().st_ino
    assert lib.ledger_append([_row()], unique_by=KEY) == 0
    assert lib.ledger_ensure() == path
    assert path.read_bytes() == before
    assert path.stat().st_ino == inode
    assert Path(f"{path}.append.lock").is_file()


def test_publication_fsyncs_same_directory_temp_before_replace_then_directory(tmp_path, monkeypatch):
    path = tmp_path / "ledger.csv"
    lib.ledger_append([_row()], path)
    before = path.read_bytes()
    events = []
    real_fsync, real_replace, real_directory = os.fsync, os.replace, lib.fsync_directory

    def fsync(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            events.append("file-fsync")
            assert path.read_bytes() == before
        return real_fsync(fd)

    def replace(source, destination):
        assert Path(source).parent == path.parent
        assert Path(destination) == path
        assert Path(source).read_bytes().startswith(before)
        assert _read(Path(source)) == [_row(), _row(902)]
        events.append("replace")
        return real_replace(source, destination)

    def directory(directory_path):
        assert _read(path) == [_row(), _row(902)]
        events.append(("directory-fsync", directory_path))
        return real_directory(directory_path)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(lib, "fsync_directory", directory)
    assert lib.ledger_append([_row(902)], path) == 1
    assert events == ["file-fsync", "replace", *[
        ("directory-fsync", directory) for directory in path.parents
    ]]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("fault", ["write", "file-fsync", "replace"])
def test_failures_before_replacement_preserve_prior_state(tmp_path, monkeypatch, existing, fault):
    path = tmp_path / "ledger.csv"
    if existing:
        lib.ledger_append([_row()], path)
    before = path.read_bytes() if existing else None

    def fail(*args, **kwargs):
        raise OSError("synthetic publication failure")

    real_temp = lib.tempfile.NamedTemporaryFile

    class PartialWrite:
        def __init__(self, *args, **kwargs):
            self.stream = real_temp(*args, **kwargs)
            self.name = self.stream.name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, payload):
            self.stream.write(payload[:max(1, len(payload) // 2)])
            self.stream.flush()
            fail()

    with monkeypatch.context() as patch:
        if fault == "write":
            patch.setattr(lib.tempfile, "NamedTemporaryFile", PartialWrite)
        else:
            patch.setattr(os, "fsync" if fault == "file-fsync" else "replace", fail)
        with pytest.raises(OSError, match="synthetic publication failure"):
            lib.ledger_append([_row(902)], path, unique_by=KEY)
    assert (path.read_bytes() if path.exists() else None) == before
    assert lib.ledger_append([_row(902)], path, unique_by=KEY) == 1
    assert _read(path) == ([_row()] if existing else []) + [_row(902)]


def test_failed_directory_sync_keeps_new_bytes_and_keyed_replay_finishes_sync(tmp_path, monkeypatch):
    path = tmp_path / "ledger.csv"
    lib.ledger_append([_row()], path)
    real_directory = lib.fsync_directory
    synced = []

    def fail(directory):
        raise OSError("synthetic directory sync failure")

    with monkeypatch.context() as patch:
        patch.setattr(lib, "fsync_directory", fail)
        with pytest.raises(OSError, match="synthetic directory sync failure"):
            lib.ledger_append([_row(902)], path, unique_by=KEY)
    after = path.read_bytes()
    assert _read(path) == [_row(), _row(902)]

    def sync(directory):
        synced.append(directory)
        return real_directory(directory)

    monkeypatch.setattr(lib, "fsync_directory", sync)
    assert lib.ledger_append([_row(902)], path, unique_by=KEY) == 0
    assert path.read_bytes() == after
    assert synced == list(path.parents)


@pytest.mark.parametrize("existing", [False, True])
def test_observed_out_of_band_edit_is_preserved_without_rollback(tmp_path, monkeypatch, existing):
    path = tmp_path / "ledger.csv"
    if existing:
        lib.ledger_append([_row()], path)
    external = _csv([_row(950)])
    real_fsync = os.fsync
    edited = []

    def fsync(fd):
        result = real_fsync(fd)
        if stat.S_ISREG(os.fstat(fd).st_mode):
            path.write_bytes(external)
            edited.append(True)
        return result

    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(lib.LedgerError):
        lib.ledger_append([_row(902)], path)
    assert edited == [True]
    assert path.read_bytes() == external


def _crash_worker(path, fault):
    if fault == "before-replace":
        real_fsync = os.fsync

        def fsync(fd):
            real_fsync(fd)
            if stat.S_ISREG(os.fstat(fd).st_mode):
                os._exit(73)

        os.fsync = fsync
    else:
        def directory(_path):
            os._exit(73)

        lib.fsync_directory = directory
    lib.ledger_append([_row(902)], path, unique_by=KEY)
    os._exit(74)  # The intended test-owned fault seam was not reached.


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("fault", ["before-replace", "after-replace"])
def test_crashed_process_leaves_complete_old_or_new_csv_and_retry_is_exact(tmp_path, fault, existing):
    path = tmp_path / "ledger.csv"
    if existing:
        lib.ledger_append([_row()], path)
    before = path.read_bytes() if existing else None
    expected = ([_row()] if existing else []) + [_row(902)]
    process = CTX.Process(target=_crash_worker, args=(path, fault))
    try:
        process.start()
        process.join(timeout=20)
        assert process.exitcode == 73, "child did not stop at the intended crash seam"
    finally:
        _stop_process(process)
    if fault == "before-replace":
        assert (path.read_bytes() if path.exists() else None) == before
    else:
        assert path.read_bytes().startswith(before or b"")
        assert _read(path) == expected
    assert lib.ledger_append([_row(902)], path, unique_by=KEY) == (fault == "before-replace")
    assert _read(path) == expected
    assert lib.ledger_append([_row(902)], path, unique_by=KEY) == 0


@pytest.mark.parametrize("mode", [0o600, 0o640, 0o664])
def test_atomic_replacement_preserves_existing_access_bits(tmp_path, mode):
    path = tmp_path / "ledger.csv"
    lib.ledger_append([_row()], path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    path.chmod(mode)
    assert lib.ledger_append([_row(902)], path) == 1
    assert stat.S_IMODE(path.stat().st_mode) == mode


@pytest.mark.parametrize("existing", [False, True])
def test_unreadable_oversize_candidate_does_not_poison_the_ledger(tmp_path, existing):
    path = tmp_path / "ledger.csv"
    if existing:
        lib.ledger_append([_row()], path)
    before = path.read_bytes() if existing else None
    with pytest.raises(lib.LedgerError):
        lib.ledger_append([_row(902, spec_hash="x" * (csv.field_size_limit() + 1))], path)
    assert (path.read_bytes() if path.exists() else None) == before
    assert lib.ledger_append([_row(903)], path) == 1
    assert _read(path) == ([_row()] if existing else []) + [_row(903)]


def test_retry_finishes_parent_directory_sync_after_mkdir_failure(tmp_path, monkeypatch):
    from engine.v2.foundation import artifacts

    path = tmp_path / "created" / "nested" / "ledger.csv"
    failed_parents = []

    def fail(directory):
        failed_parents.append(directory)
        raise OSError("synthetic mkdir parent sync failure")

    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "fsync_directory", fail)
        with pytest.raises(OSError, match="synthetic mkdir parent sync failure"):
            lib.ledger_append([_row()], path, unique_by=KEY)
    assert failed_parents == [tmp_path]
    assert (tmp_path / "created").is_dir()
    assert not path.exists()
    synced = []
    real_sync = lib.fsync_directory

    def sync(directory):
        synced.append(directory)
        return real_sync(directory)

    monkeypatch.setattr(lib, "fsync_directory", sync)
    assert lib.ledger_append([_row()], path, unique_by=KEY) == 1
    assert synced == list(path.parents)
    assert _read(path) == [_row()]
    synced.clear()
    assert lib.ledger_append([_row()], path, unique_by=KEY) == 0
    assert synced == list(path.parents)


@pytest.mark.parametrize("invalid", [None, "not a mapping", 1, []])
def test_non_mapping_rows_are_refused_before_publication(tmp_path, invalid):
    path = tmp_path / "ledger.csv"
    with pytest.raises(lib.LedgerError):
        lib.ledger_append([_row(), invalid], path)
    assert not path.exists()


def test_a_replay_key_can_span_success_and_refusal_stages(tmp_path):
    """Native callers can reserve one terminal outcome independently of its stage."""
    path = tmp_path / "ledger.csv"
    key = ("id", "spec_hash")
    assert lib.ledger_append([_row()], path, unique_by=key) == 1
    before = path.read_bytes()
    with pytest.raises(lib.LedgerError):
        lib.ledger_append([_row(stage="refused")], path, unique_by=key)
    assert path.read_bytes() == before


def test_destination_symlink_refuses_without_touching_its_target(tmp_path):
    target = tmp_path / "target.csv"
    target.write_bytes(_csv([_row()]))
    path = tmp_path / "alias.csv"
    path.symlink_to(target)
    before = target.read_bytes()
    with pytest.raises(lib.LedgerError):
        lib.ledger_append([_row(902)], path)
    assert path.is_symlink()
    assert target.read_bytes() == before


def test_parent_directory_alias_uses_the_same_permanent_lock(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)
    path = parent / "ledger.csv"
    lib.ledger_append([_row()], path)
    inode = Path(f"{path}.append.lock").stat().st_ino
    assert lib.ledger_append([_row(902)], alias / "ledger.csv") == 1
    assert Path(f"{path}.append.lock").stat().st_ino == inode
    assert _read(path) == [_row(), _row(902)]
