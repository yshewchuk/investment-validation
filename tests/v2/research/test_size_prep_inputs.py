"""Pinned Tier-1 manifest, write-root bounds generation and reconstructed provenance (synthetic data only)."""
import gzip
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path

import pytest

from engine.v2.contracts import DataQuery, KeyPredicate, TableContractRef
from engine.v2.data.computed_move_bounds import COLUMNS, CONTRACT, TABLE
from engine.v2.data import errors as data_errors
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, canonical_json, content_hash
from engine.v2.ops import computed_moves_store
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.errors import OpsError
from engine.v2.ops.workspace.catalog_rows import copy_closure_rows
from engine.v2.ops.workspace.head import commit_scope_head
from engine.v2.ops.workspace.objects import admit_closure_objects
from engine.v2.ops.workspace.overlay import create_workspace
from engine.v2.ops.workspace.scope import SCOPE_HEAD_VERSION
from experiments import size_prep_inputs as prep
from tests.ops_support import FakeClock
from tests.data_scan_support import catalog_and_store, commit_tables, contract_for, contract_ref_for, publish_and_inspect
from tests.v2.ops.test_v2_ops_computed_moves_store import (
    _AS_OF, _BDAYS, _EVENT_DAYS, _build_parent, _closes_csv, _event_row, _head_row, _series_csv)

REF = TableContractRef(contract_id=CONTRACT.contract_id, definition_hash=CONTRACT.definition_hash)
_EVENTS = contract_for("earnings_events")
_EVENTS_REF = contract_ref_for(_EVENTS)


def _bound_rows(repository, snapshot):
    records = repository.fragment_records(snapshot, TABLE)
    count = sum(r.row_count for r in records)
    request = DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=REF,
        columns=COLUMNS, key_filter=(KeyPredicate(column="ticker", operator="in",
            values=tuple(sorted({r.partition_key for r in records}))),), order_by=CONTRACT.primary_key,
        max_batch_rows=max(1, count), max_result_rows=count)
    return [row for batch in repository.scan(request, table_name=TABLE) for row in batch.to_pylist()]


@pytest.fixture(autouse=True)
def _small_bar(monkeypatch):
    monkeypatch.setattr(computed_moves_store, "MIN_SCOREABLE", 5)


_PRODUCTION = (Path("/stand-in-production-root"),)


def _put(source_root, ticker, key, fetched_at, body):
    directory = source_root / "data" / "raw" / "fetch" / "yfinance" / key[:2]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (key + ".body.gz")).write_bytes(gzip.compress(body, mtime=0))
    (directory / (key + ".meta.json")).write_text(json.dumps({
        "source": "yfinance", "endpoint": "history", "key": key, "status": 200, "fetched_at": fetched_at,
        "params": {"ticker": ticker, "period": "max"}}))


@pytest.fixture
def source_root(tmp_path):
    root = tmp_path / "source"
    _put(root, "AAAA", "aa01", "2025-01-01T00:00:00+00:00", _series_csv(50.0))
    _put(root, "AAAA", "aa02", "2026-01-01T00:00:00+00:00", _closes_csv())
    _put(root, "BBBB", "bb01", "2024-02-01T00:00:00+00:00", _closes_csv())
    return root


def _refusal(call, code):
    with pytest.raises(OpsError) as raised:
        call()
    assert raised.value.code == code


def test_manifest_pins_the_newest_entry_per_ticker_with_body_hash(source_root):
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["BBBB", "AAAA"])
    assert [(e["ticker"], e["key"], e["fetched_at"]) for e in manifest["entries"]] == [
        ("AAAA", "aa02", "2026-01-01T00:00:00+00:00"), ("BBBB", "bb01", "2024-02-01T00:00:00+00:00")]
    assert all(len(e["body_sha256"]) == 64 for e in manifest["entries"])
    assert prep.validate_manifest(deepcopy(manifest)) == manifest


def test_manifest_refusals(source_root):
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    _refusal(lambda: prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["ZZZZ"]), "SOURCE_NOT_FOUND")
    _refusal(lambda: prep.build_manifest(source_root, session_ceiling="2024-2-5", tickers=["AAAA"]), "INVALID_REQUEST")
    _refusal(lambda: prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA", "AAAA"]), "INVALID_REQUEST")
    _refusal(lambda: prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=[]), "INVALID_REQUEST")
    tampered = deepcopy(manifest)
    tampered["entries"][0]["fetched_at"] = "2020-01-01T00:00:00+00:00"
    _refusal(lambda: prep.validate_manifest(tampered), "INVALID_REQUEST")
    extra = deepcopy(manifest)
    extra["extra"] = 1
    _refusal(lambda: prep.validate_manifest(extra), "INVALID_REQUEST")


def test_pinned_fetcher_is_not_newest_wins_and_detects_every_drift(source_root):
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    _put(source_root, "AAAA", "aa03", "2026-06-01T00:00:00+00:00", _series_csv(70.0))
    cache, fetch = prep.pinned_history_fetcher(source_root, manifest)
    assert cache["AAAA"].key == "aa02"
    assert fetch("AAAA") == (_closes_csv(), "complete", {}, [])
    assert fetch("BBBB") == (b"", "legitimate_empty", {}, [])
    meta = source_root / "data/raw/fetch/yfinance/aa/aa02.meta.json"
    body = source_root / "data/raw/fetch/yfinance/aa/aa02.body.gz"
    original_meta, original_body = meta.read_text(), body.read_bytes()
    body.write_bytes(gzip.compress(_series_csv(1.0), mtime=0))
    _refusal(lambda: prep.pinned_history_fetcher(source_root, manifest), "INPUT_CHANGED")
    body.write_bytes(original_body)
    meta.write_text(original_meta.replace("2026-01-01", "2026-01-02"))
    _refusal(lambda: prep.pinned_history_fetcher(source_root, manifest), "INPUT_CHANGED")
    meta.write_text(original_meta.replace('"status": 200', '"status": 500'))
    _refusal(lambda: prep.pinned_history_fetcher(source_root, manifest), "INPUT_CHANGED")
    meta.unlink()
    _refusal(lambda: prep.pinned_history_fetcher(source_root, manifest), "INPUT_CHANGED")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "work"
    root.mkdir()
    clock = FakeClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    store = ArtifactStore(root)
    _build_parent(conn, clock, store, events_rows=[_event_row("AAAA", day) for day in _EVENT_DAYS])
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot", lambda *a, **k: (["AAAA"], {}))
    before = _head_row(conn)["snapshot_id"]
    conn.close()
    return root, before


def _tree(root):
    return sorted((str(path.relative_to(root)), path.stat().st_size) for path in root.rglob("*") if path.is_file())


def test_generate_bounds_writes_only_the_write_root_and_labels_provenance(source_root, workspace):
    root, parent = workspace
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    source_before = _tree(source_root)
    record = prep.generate_bounds(root, source_root=source_root, manifest=manifest, scope="shadow", as_of=_AS_OF,
                                  production_roots=_PRODUCTION)
    assert record["record_id"] == content_hash({k: v for k, v in record.items() if k != "record_id"})
    assert _tree(source_root) == source_before
    assert record["label"] == "reconstructed"
    assert record["refresh_noop"] is False
    assert record["entry_count"] == 1 and record["manifest_entry_count"] == 1
    assert record["session_ceiling"] == _AS_OF and record["latest_fetched_at"] == "2026-01-01T00:00:00+00:00"
    assert record["entries_fetched_after_ceiling"] == 1
    assert record["source_snapshot_id"] == parent and record["bounds_snapshot_id"] != parent
    stored = json.loads((root / "size_prep" / ("provenance-" + record["record_id"].split(":")[1] + ".json")).read_text())
    assert stored == record
    conn = open_catalog(root / "catalog.sqlite", clock=FakeClock())
    try:
        snapshot = Repository(conn, ArtifactStore(root)).resolve_full(record["bounds_snapshot_id"])
        assert TABLE in snapshot.table_manifests
    finally:
        conn.close()


def test_generate_bounds_refusals_happen_before_any_write(source_root, workspace, tmp_path):
    root, _ = workspace
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    _refusal(lambda: prep.generate_bounds(root, source_root=source_root, manifest=manifest, scope="shadow",
                                          as_of="2024-02-06", production_roots=_PRODUCTION), "INVALID_REQUEST")
    empty = tmp_path / "empty"
    empty.mkdir()
    _refusal(lambda: prep.generate_bounds(empty, source_root=source_root, manifest=manifest, scope="shadow",
                                          as_of=_AS_OF, production_roots=_PRODUCTION), "INVALID_REQUEST")
    (source_root / "data/raw/fetch/yfinance/aa/aa02.body.gz").write_bytes(gzip.compress(b"x", mtime=0))
    _refusal(lambda: prep.generate_bounds(root, source_root=source_root, manifest=manifest, scope="shadow",
                                          as_of=_AS_OF, production_roots=_PRODUCTION), "INPUT_CHANGED")
    assert not (root / "size_prep").exists() and list(empty.iterdir()) == []


def test_fetch_rehashes_served_bytes_and_manifest_rejects_padded_ticker(source_root):
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    _, fetch = prep.pinned_history_fetcher(source_root, manifest)
    (source_root / "data/raw/fetch/yfinance/aa/aa02.body.gz").write_bytes(gzip.compress(_series_csv(1.0), mtime=0))
    _refusal(lambda: fetch("AAAA"), "INPUT_CHANGED")
    padded = deepcopy(manifest)
    padded["entries"][0]["ticker"] = "AAAA "
    padded["manifest_id"] = prep._identity(padded)
    _refusal(lambda: prep.validate_manifest(padded), "INVALID_REQUEST")


def test_oversized_corrupt_and_symlinked_bodies_are_refused(source_root, monkeypatch):
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["BBBB"])
    body = source_root / "data/raw/fetch/yfinance/bb/bb01.body.gz"
    monkeypatch.setattr(prep, "MAX_BODY_BYTES", 10)
    _refusal(lambda: prep.pinned_history_fetcher(source_root, manifest), "RESOURCE_LIMIT_EXCEEDED")
    monkeypatch.undo()
    original = body.read_bytes()
    body.write_bytes(original[:20])
    _refusal(lambda: prep.pinned_history_fetcher(source_root, manifest), "SOURCE_INVALID")
    body.unlink()
    target = source_root / "elsewhere.gz"
    target.write_bytes(original)
    body.symlink_to(target)
    _refusal(lambda: prep.pinned_history_fetcher(source_root, manifest), "SOURCE_INVALID")


def test_generate_bounds_refuses_unpinned_targets_without_committing(source_root, workspace, monkeypatch):
    root, parent = workspace
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAAA", "CCCC"], {}))
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    _refusal(lambda: prep.generate_bounds(root, source_root=source_root, manifest=manifest, scope="shadow",
                                          as_of=_AS_OF, production_roots=_PRODUCTION), "SOURCE_NOT_FOUND")
    conn = open_catalog(root / "catalog.sqlite", clock=FakeClock())
    try:
        assert _head_row(conn)["snapshot_id"] == parent
    finally:
        conn.close()
    assert not (root / "size_prep").exists()


def test_missing_source_root_and_oversized_sidecar_are_typed_refusals(source_root, tmp_path, monkeypatch):
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    missing = tmp_path / "nowhere"
    _refusal(lambda: prep.build_manifest(missing, session_ceiling=_AS_OF, tickers=["AAAA"]), "INVALID_REQUEST")
    _refusal(lambda: prep.pinned_history_fetcher(missing, manifest), "INVALID_REQUEST")
    monkeypatch.setattr(prep, "MAX_META_BYTES", 10)
    _refusal(lambda: prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"]), "SOURCE_INVALID")
    _refusal(lambda: prep.pinned_history_fetcher(source_root, manifest), "SOURCE_INVALID")


def test_rerun_after_commit_is_marked_noop_and_still_proves_bounds(source_root, workspace):
    root, _ = workspace
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    first = prep.generate_bounds(root, source_root=source_root, manifest=manifest, scope="shadow", as_of=_AS_OF,
                                 production_roots=_PRODUCTION)
    second = prep.generate_bounds(root, source_root=source_root, manifest=manifest, scope="shadow", as_of=_AS_OF,
                                  production_roots=_PRODUCTION)
    assert second["refresh_noop"] is True
    assert second["entry_count"] == 1 and second["manifest_entry_count"] == 1
    assert second["source_snapshot_id"] == second["bounds_snapshot_id"] == first["bounds_snapshot_id"]


def _digest(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name, _ in _tree(root)}


def test_a_stand_in_production_root_is_never_opened_or_changed(source_root, workspace, tmp_path):
    """A second, fully built catalog plays production; generation reads/writes only the given write root."""
    production = tmp_path / "production"
    production.mkdir()
    clock = FakeClock()
    conn = open_catalog(production / "catalog.sqlite", clock=clock)
    _build_parent(conn, clock, ArtifactStore(production), events_rows=[_event_row("AAAA", day) for day in _EVENT_DAYS])
    conn.close()
    before = _digest(production)
    assert before
    root, _ = workspace
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    record = prep.generate_bounds(root, source_root=source_root, manifest=manifest, scope="shadow", as_of=_AS_OF,
                                  production_roots=[production])
    assert _digest(production) == before
    assert record["bounds_snapshot_id"] != record["source_snapshot_id"]
    # the strict-confinement guard: a write root that is not the catalog's own directory is refused
    other = tmp_path / "other"
    other.mkdir()
    (other / "catalog.sqlite").symlink_to(production / "catalog.sqlite")
    _refusal(lambda: prep.generate_bounds(other, source_root=source_root, manifest=manifest, scope="shadow",
                                          as_of=_AS_OF, production_roots=[production]), "INVALID_REQUEST")
    assert _digest(production) == before


def test_write_root_related_to_a_production_root_is_refused(source_root, workspace, tmp_path):
    root, _ = workspace
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    base = dict(source_root=source_root, manifest=manifest, scope="shadow", as_of=_AS_OF)
    for production in ([root], [root.parent], [root / "sub"], []):
        _refusal(lambda: prep.generate_bounds(root, production_roots=production, **base), "INVALID_REQUEST")
    link = tmp_path / "link"
    link.symlink_to(root)
    _refusal(lambda: prep.generate_bounds(link, production_roots=_PRODUCTION, **base), "INVALID_REQUEST")
    assert not (root / "size_prep").exists()


def test_unhashable_ticker_entries_are_a_typed_refusal(source_root):
    _refusal(lambda: prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=[["AAAA"]]), "INVALID_REQUEST")
    _refusal(lambda: prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=[{"a": 1}]), "INVALID_REQUEST")


def test_sidecar_scan_limit_is_a_typed_refusal(source_root, monkeypatch):
    monkeypatch.setattr(prep, "MAX_SIDECARS", 1)
    _refusal(lambda: prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"]), "RESOURCE_LIMIT_EXCEEDED")


def test_generation_consumes_the_pinned_bytes_and_commits_checkable_bounds(source_root, workspace):
    root, _ = workspace
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    _put(source_root, "AAAA", "aa09", "2027-01-01T00:00:00+00:00", _series_csv(10.0))
    source_digest_before = _digest(source_root)
    record = prep.generate_bounds(root, source_root=source_root, manifest=manifest, scope="shadow", as_of=_AS_OF,
                                  production_roots=_PRODUCTION)
    assert _digest(source_root) == source_digest_before
    conn = open_catalog(root / "catalog.sqlite", clock=FakeClock())
    try:
        repo = Repository(conn, ArtifactStore(root))
        snapshot = repo.resolve_full(record["bounds_snapshot_id"])
        rows = _bound_rows(repo, snapshot.snapshot)
        assert {row["event_date"] for row in rows} == {d.date().isoformat() for d in _EVENT_DAYS}
        _bday_isos = {d.date().isoformat() for d in _BDAYS}
        for row in rows:
            assert row["ticker"] == "AAAA"
            assert row["pre_close_date"] in _bday_isos and row["post_close_date"] in _bday_isos
            assert row["pre_close_date"] <= row["event_date"] <= row["post_close_date"]
        pinned = "sha256:" + hashlib.sha256(_closes_csv()).hexdigest()
        newer = "sha256:" + hashlib.sha256(_series_csv(10.0)).hexdigest()
        values = {str(v) for r in conn.execute("SELECT * FROM data_raw_receipts").fetchall() for v in tuple(r)}
        assert pinned in values and newer not in values
    finally:
        conn.close()


def test_noop_refresh_without_bounds_in_the_head_is_refused(source_root, workspace, monkeypatch):
    root, parent = workspace
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])

    def fake(args, root, conn, clock, **kw):
        return {"parent_snapshot_id": parent, "result": {"candidate_snapshot_id": None}}

    monkeypatch.setattr("engine.v2.ops.cli.computed_moves_command", fake)
    _refusal(lambda: prep.generate_bounds(root, source_root=source_root, manifest=manifest, scope="shadow",
                                          as_of=_AS_OF, production_roots=_PRODUCTION), "PUBLICATION_REFUSED")
    assert not (root / "size_prep").exists()


def test_pinned_entry_with_other_ticker_endpoint_period_or_status_is_refused(source_root):
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    meta = source_root / "data/raw/fetch/yfinance/aa/aa02.meta.json"
    original_meta = meta.read_text()
    for old, new in [
        ('"ticker": "AAAA"', '"ticker": "ZZZZ"'),
        ('"endpoint": "history"', '"endpoint": "quote"'),
        ('"period": "max"', '"period": "1y"'),
    ]:
        meta.write_text(original_meta.replace(old, new))
        _refusal(lambda: prep.pinned_history_fetcher(source_root, manifest), "INPUT_CHANGED")
        meta.write_text(original_meta)


def test_manifest_ignores_non_history_non_max_and_failed_entries(source_root):
    _put(source_root, "CCCC", "cc01", "2024-03-01T00:00:00+00:00", _closes_csv())
    _put(source_root, "CCCC", "cc02", "2025-03-01T00:00:00+00:00", _closes_csv())
    cc02_meta = source_root / "data/raw/fetch/yfinance/cc/cc02.meta.json"
    cc02_meta.write_text(cc02_meta.read_text().replace('"status": 200', '"status": 500'))
    _put(source_root, "CCCC", "cc03", "2026-03-01T00:00:00+00:00", _closes_csv())
    cc03_meta = source_root / "data/raw/fetch/yfinance/cc/cc03.meta.json"
    cc03_meta.write_text(cc03_meta.read_text().replace('"period": "max"', '"period": "1y"'))
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["CCCC"])
    assert manifest["entries"][0]["key"] == "cc01"


def test_bounds_option_is_validated_at_the_cli_boundary_even_for_dry_run(tmp_path):
    from types import SimpleNamespace

    from engine.v2.ops.cli import computed_moves_command

    args = SimpleNamespace(source_root=tmp_path, scope="shadow", as_of=_AS_OF, dry_run=True)
    for invalid in ("yes", 1, None):
        _refusal(lambda: computed_moves_command(args, tmp_path, None, None, include_source_bounds=invalid),
                 "INVALID_REQUEST")
    assert not (tmp_path / computed_moves_store.INPUT_PATH).exists()


def test_history_pair_is_validated_at_the_cli_boundary(tmp_path):
    from types import SimpleNamespace

    from engine.v2.ops.cli import computed_moves_command

    args = SimpleNamespace(source_root=tmp_path, scope="shadow", as_of=_AS_OF, dry_run=True)
    for invalid in ((), [], (None, None), (1, 2, 3), ({}, None), "ab", ("ab", len), ([], len)):
        _refusal(lambda: computed_moves_command(args, tmp_path, None, None, history=invalid),
                 "INVALID_REQUEST")

    from engine.v2.ops.recovery import SupervisorLock

    holder = SupervisorLock(tmp_path / "supervisor.lock")
    assert holder.acquire()
    try:
        _refusal(lambda: computed_moves_command(args, tmp_path, None, None, history=[]), "INVALID_REQUEST")
    finally:
        holder.release()


def test_sidecar_whose_meta_key_differs_from_its_file_name_is_ignored(source_root):
    meta = source_root / "data/raw/fetch/yfinance/bb/bb01.meta.json"
    meta.write_text(meta.read_text().replace('"key": "bb01"', '"key": "other"'))
    _refusal(lambda: prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["BBBB"]), "SOURCE_NOT_FOUND")


def test_symlinked_provenance_directory_is_refused_before_any_write(source_root, workspace, tmp_path):
    root, _ = workspace
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "size_prep").symlink_to(outside)
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    _refusal(lambda: prep.generate_bounds(root, source_root=source_root, manifest=manifest, scope="shadow",
                                          as_of=_AS_OF, production_roots=_PRODUCTION), "INVALID_REQUEST")
    assert list(outside.iterdir()) == []


def test_fifo_and_symlink_cache_files_are_refused_without_blocking(source_root):
    import os

    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["BBBB"])
    body = source_root / "data/raw/fetch/yfinance/bb/bb01.body.gz"
    body.unlink()
    os.mkfifo(body)
    _refusal(lambda: prep.pinned_history_fetcher(source_root, manifest), "SOURCE_INVALID")
    body.unlink()
    meta = source_root / "data/raw/fetch/yfinance/bb/bb01.meta.json"
    meta.rename(meta.with_name("real.json"))
    meta.symlink_to(meta.with_name("real.json"))
    _refusal(lambda: prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["BBBB"]), "SOURCE_INVALID")


def test_scan_logs_bounded_progress_to_stderr(source_root, monkeypatch, capsys):
    monkeypatch.setattr(prep, "PROGRESS_EVERY", 1)
    prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    captured = capsys.readouterr()
    err = captured.err
    assert "[size-prep] scan 1 of at most" in err and "ETA<=" in err and "rss_kib=" in err
    assert "manifest end" in err and captured.out == ""


def test_production_roots_must_be_a_collection_of_paths(source_root, workspace):
    root, _ = workspace
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    base = dict(source_root=source_root, manifest=manifest, scope="shadow", as_of=_AS_OF)
    for bad in (str(root), root, None, [1], b"x"):
        _refusal(lambda: prep.generate_bounds(root, production_roots=bad, **base), "INVALID_REQUEST")
    assert not (root / "size_prep").exists()


def _digest_tree(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.fixture
def pinned_ws(tmp_path):
    """A real research workspace over a production snapshot whose earnings_events carry only EEEE rows."""
    prod = tmp_path / "wsprod"
    prod.mkdir()
    conn, clock, store = catalog_and_store(prod)
    record = publish_and_inspect(store, _EVENTS, _EVENTS_REF,
                                 [_event_row("EEEE", day) for day in _EVENT_DAYS], "2024")
    snap = commit_tables(conn, clock, {"earnings_events": [record]}, {"earnings_events": _EVENTS},
                         store=store)
    conn.close()
    workspace = create_workspace(tmp_path / "ws", production_catalog=prod / "catalog.sqlite",
                                 production_store=prod / "store", snapshot_id=snap.snapshot_id,
                                 manifest_hash=snap.manifest_hash, clock=clock)
    return workspace, clock, prod


@pytest.fixture
def no_events_ws(tmp_path):
    """A real research workspace over a production snapshot that carries no earnings_events table."""
    prod = tmp_path / "nsprod"
    prod.mkdir()
    conn, clock, store = catalog_and_store(prod)
    securities = contract_for("securities")
    row = dict(ticker="EEEE", year=2024, first_date=None, last_date=None, mcap_usd=1.5e9,
               mcap_log=21.1, mcap_raw=1.5, mcap_unit_era="billions", mcap_quantized=False,
               n_obs=250, src="orats")
    record = publish_and_inspect(store, securities, contract_ref_for(securities), [row], "2024")
    snap = commit_tables(conn, clock, {"securities": [record]}, {"securities": securities}, store=store)
    conn.close()
    workspace = create_workspace(tmp_path / "nsws", production_catalog=prod / "catalog.sqlite",
                                 production_store=prod / "store", snapshot_id=snap.snapshot_id,
                                 manifest_hash=snap.manifest_hash, clock=clock)
    return workspace, clock, prod


def _ws_state(workspace):
    files = _digest_tree(workspace.paths.store)
    artifacts = workspace.conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
    return files, artifacts


def test_v1_manifest_shape_is_unchanged_when_no_exclusions_are_requested(source_root):
    v1 = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    assert v1["schema_version"] == "tier1_history_manifest.v1" == prep.MANIFEST_VERSION
    assert set(v1) == {"schema_version", "session_ceiling", "entries", "manifest_id"}
    assert "excluded_tickers" not in canonical_json(v1)
    assert v1["manifest_id"] == content_hash({k: v for k, v in v1.items() if k != "manifest_id"})
    assert prep.validate_manifest(deepcopy(v1)) == v1


def test_build_manifest_requires_boolean_exclusion_opt_in(source_root):
    _refusal(lambda: prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"], allow_exclusions=1), "INVALID_REQUEST")


def test_exclusion_manifest_is_distinctly_identified_and_canonical(source_root):
    plain = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    with_empty = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"],
                                     allow_exclusions=True)
    assert with_empty["schema_version"] == "tier1_history_manifest.v2" == prep.EXCLUSION_MANIFEST_VERSION
    assert plain["schema_version"] == prep.MANIFEST_VERSION
    assert with_empty["excluded_tickers"] == [] and with_empty["entries"] == plain["entries"]
    assert with_empty["manifest_id"] != plain["manifest_id"]
    v2 = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["QQQQ", "AAAA", "ZZZZ"],
                             allow_exclusions=True)
    assert v2["excluded_tickers"] == ["QQQQ", "ZZZZ"] and prep.validate_manifest(deepcopy(v2)) == v2
    assert v2["manifest_id"] == content_hash({k: v for k, v in v2.items() if k != "manifest_id"})
    all_excluded = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["ZZZZ"],
                                       allow_exclusions=True)
    assert all_excluded["entries"] == [] and all_excluded["excluded_tickers"] == ["ZZZZ"]
    assert prep.validate_manifest(deepcopy(all_excluded)) == all_excluded
    for mutate in (lambda m: m.update(excluded_tickers=["ZZZZ", "QQQQ"]),
                   lambda m: m.update(excluded_tickers=None),
                   lambda m: m.update(excluded_tickers=["QQQQ", "QQQQ"]),
                   lambda m: m.update(excluded_tickers=[" ZZZZ"]),
                   lambda m: m.update(excluded_tickers=["AAAA"]),
                   lambda m: (m.update(excluded_tickers=[]), m["entries"].clear())):
        tampered = deepcopy(v2)
        mutate(tampered)
        tampered["manifest_id"] = prep._identity(tampered)
        _refusal(lambda: prep.validate_manifest(tampered), "INVALID_REQUEST")
    stripped = deepcopy(v2)
    del stripped["excluded_tickers"]
    _refusal(lambda: prep.validate_manifest(stripped), "INVALID_REQUEST")
    for pair in (lambda m: m.update(schema_version=prep.MANIFEST_VERSION),
                 lambda m: m.update(schema_version="tier1_history_manifest.v3")):
        tampered = deepcopy(v2)
        pair(tampered)
        tampered["manifest_id"] = prep._identity(tampered)
        _refusal(lambda: prep.validate_manifest(tampered), "INVALID_REQUEST")
    promoted = deepcopy(plain)
    promoted["excluded_tickers"] = []
    promoted["manifest_id"] = prep._identity(promoted)
    _refusal(lambda: prep.validate_manifest(promoted), "INVALID_REQUEST")


def test_legacy_generate_bounds_refuses_v2_manifests_before_opening_the_catalog(source_root, workspace):
    root, parent = workspace
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA", "ZZZZ"],
                                   allow_exclusions=True)
    with pytest.raises(OpsError) as raised:
        prep.generate_bounds(root, source_root=source_root, manifest=manifest, scope="shadow",
                             as_of=_AS_OF, production_roots=_PRODUCTION)
    assert raised.value.code == "WORKSPACE_REFUSED"
    assert raised.value.problem.details["reason"] == "workspace_admission_required"
    assert not (root / "size_prep").exists()
    conn = open_catalog(root / "catalog.sqlite", clock=FakeClock())
    try:
        assert _head_row(conn)["snapshot_id"] == parent
    finally:
        conn.close()


def test_listed_exclusions_fetch_empty_even_under_strict(source_root):
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA", "ZZZZ"],
                                   allow_exclusions=True)
    _, lenient = prep.pinned_history_fetcher(source_root, manifest)
    assert lenient("ZZZZ") == lenient("QQQQ") == (b"", "legitimate_empty", {}, [])
    cache, strict = prep.pinned_history_fetcher(source_root, manifest, strict=True)
    assert cache["AAAA"].key == "aa02"
    assert strict("AAAA")[0] == prep._body(cache["AAAA"])
    assert strict("AAAA")[1] == "complete" and strict("ZZZZ") == (b"", "legitimate_empty", {}, [])
    _refusal(lambda: strict("QQQQ"), "SOURCE_NOT_FOUND")


def test_admission_validates_scope_before_workspace_source_read(source_root):
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"],
                                   allow_exclusions=True)

    def unexpected(*args, **kwargs):
        raise AssertionError("workspace was touched before scope validation")

    workspace = SimpleNamespace(source_snapshot=unexpected, record_read_pin=unexpected,
        source=SimpleNamespace(resolve=lambda *_: None), snapshot_id="snapshot", manifest_hash="manifest")
    with pytest.raises(OpsError) as raised:
        prep.admit_manifest(workspace, " bad ", manifest, clock=FakeClock())
    assert raised.value.code == "WORKSPACE_REFUSED"
    assert raised.value.problem.details["reason"] == "scope_invalid"


def test_admission_reject_missing_workspace_before_source_read(source_root):
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    with pytest.raises(OpsError) as raised:
        prep.admit_manifest(None, "size-prep", manifest, clock=FakeClock())
    assert raised.value.code == "WORKSPACE_REFUSED"
    assert raised.value.problem.details["reason"] == "workspace_invalid"


def test_admission_maps_workspace_snapshot_read_failure(source_root):
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])

    def broken_snapshot():
        raise OSError("synthetic snapshot read failure")

    workspace = SimpleNamespace(source_snapshot=broken_snapshot,
        record_read_pin=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected write")),
        source=SimpleNamespace(resolve=lambda *_: None), snapshot_id="snapshot", manifest_hash="manifest")
    with pytest.raises(OpsError) as raised:
        prep.admit_manifest(workspace, "size-prep", manifest, clock=FakeClock())
    assert raised.value.code == "WORKSPACE_REFUSED"
    assert raised.value.problem.details == {"reason": "storage_failure", "cause": "OSError"}


def test_admission_refuses_event_bearing_exclusions_before_any_workspace_write(pinned_ws, source_root,
                                                                               capsys, monkeypatch):
    workspace, clock, prod = pinned_ws
    scan_caps = []
    scan_queries = []
    original_scan = prep.Repository.scan

    def recording_scan(self, query, *, table_name, max_object_bytes=None):
        scan_caps.append(max_object_bytes)
        scan_queries.append(query)
        yield from original_scan(self, query, table_name=table_name, max_object_bytes=max_object_bytes)

    monkeypatch.setattr(prep.Repository, "scan", recording_scan)
    before, prod_before = _ws_state(workspace), _digest_tree(prod)
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA", "EEEE", "ZZZZ"],
                                   allow_exclusions=True)
    monkeypatch.setattr(prep, "MAX_LOGGED_NAMES", 1)
    with pytest.raises(OpsError) as raised:
        prep.admit_manifest(workspace, "size-prep", manifest, clock=clock)
    assert raised.value.code == "PUBLICATION_REFUSED"
    assert raised.value.problem.details["reason"] == "excluded_ticker_has_events"
    assert raised.value.problem.details["tickers"] == "EEEE"
    assert _ws_state(workspace) == before and _digest_tree(prod) == prod_before
    err = capsys.readouterr().err
    assert "excluded=2" in err and "EEEE (+1 more)" in err
    assert scan_caps == [prep.MAX_EVENT_OBJECT_BYTES]
    assert scan_queries[0].key_filter == (
        KeyPredicate(column="ticker", operator="in", values=("EEEE", "ZZZZ")),)


def test_admission_refuses_oversized_event_fragment(pinned_ws, source_root, monkeypatch):
    workspace, clock, prod = pinned_ws
    before, prod_before = _ws_state(workspace), _digest_tree(prod)
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA", "EEEE", "ZZZZ"],
                                   allow_exclusions=True)
    monkeypatch.setattr(prep, "MAX_EVENT_OBJECT_BYTES", 1)
    with pytest.raises(data_errors.DataError) as raised:
        prep.admit_manifest(workspace, "size-prep", manifest, clock=clock)
    assert raised.value.code == "RESULT_LIMIT_EXCEEDED"
    assert _ws_state(workspace) == before and _digest_tree(prod) == prod_before


def test_admission_records_manifest_cas_and_read_pin_in_workspace_only(pinned_ws, source_root):
    workspace, clock, prod = pinned_ws
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA", "ZZZZ"],
                                   allow_exclusions=True)
    prod_before = _digest_tree(prod)
    admission = prep.admit_manifest(workspace, "size-prep", manifest, clock=clock)
    fields = {k: v for k, v in admission.items() if k not in ("admission_id", "manifest_ref", "read_pin_ref")}
    assert admission["admission_id"] == content_hash(fields)
    assert fields == {"schema_version": prep.ADMISSION_VERSION, "scope": "size-prep",
        "scope_binding_id": content_hash({"schema_version": SCOPE_HEAD_VERSION, "scope": "size-prep",
            "snapshot_id": workspace.snapshot_id, "manifest_hash": workspace.manifest_hash}),
        "snapshot_id": workspace.snapshot_id, "manifest_hash": workspace.manifest_hash,
        "manifest_id": manifest["manifest_id"], "manifest_object": admission["manifest_ref"].content_hash,
        "session_ceiling": _AS_OF, "entry_count": 1, "excluded_count": 1}
    assert admission["manifest_ref"].schema_ref == prep.EXCLUSION_MANIFEST_VERSION
    assert workspace.store.read_verified(admission["manifest_ref"]) == canonical_json(manifest).encode("utf-8")
    pin = json.loads(workspace.store.read_verified(admission["read_pin_ref"]))
    assert pin["pin"] == fields and pin["name"].startswith("size-prep-manifest:")
    for ref in (admission["manifest_ref"], admission["read_pin_ref"]):
        row = workspace.conn.execute("SELECT ref_json FROM artifacts WHERE artifact_id = ?",
                                     (ref.artifact_id,)).fetchone()
        assert row is not None
    assert _digest_tree(prod) == prod_before


def test_admission_refuses_when_the_snapshot_lacks_the_events_table(no_events_ws, source_root):
    workspace, clock, prod = no_events_ws
    before, prod_before = _ws_state(workspace), _digest_tree(prod)
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA", "ZZZZ"],
                                   allow_exclusions=True)
    with pytest.raises(OpsError) as raised:
        prep.admit_manifest(workspace, "size-prep", manifest, clock=clock)
    assert raised.value.code == "PUBLICATION_REFUSED"
    assert raised.value.problem.details["reason"] == "event_coverage_unavailable"
    assert raised.value.problem.details["excluded_count"] == 1
    assert _ws_state(workspace) == before and _digest_tree(prod) == prod_before


def test_admission_needs_no_event_coverage_without_exclusions(no_events_ws, source_root):
    workspace, clock, _ = no_events_ws
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    admission = prep.admit_manifest(workspace, "size-prep", manifest, clock=clock)
    assert admission["excluded_count"] == 0
    assert admission["manifest_ref"].schema_ref == prep.MANIFEST_VERSION


@pytest.fixture
def bounds_ws(tmp_path, monkeypatch):
    """A prepared workspace: a production shadow snapshot with AAAA events whose closure rows and
    objects are admitted and whose ``shadow`` scope head is committed, so bounds can publish into it."""
    prod = tmp_path / "bprod"
    prod.mkdir()
    conn, clock, store = catalog_and_store(prod)
    head = _build_parent(conn, clock, store, events_rows=[_event_row("AAAA", day) for day in _EVENT_DAYS])
    manifest_hash = conn.execute(
        "SELECT manifest_hash FROM data_snapshots WHERE snapshot_id = ?", (head["snapshot_id"],)).fetchone()[0]
    conn.close()
    ws = create_workspace(tmp_path / "bws", production_catalog=prod / "catalog.sqlite",
                          production_store=prod / "store", snapshot_id=head["snapshot_id"],
                          manifest_hash=manifest_hash, clock=clock)
    copy_closure_rows(ws)
    admit_closure_objects(ws)
    commit_scope_head(ws, "shadow", clock=clock)
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot", lambda *a, **k: (["AAAA"], {}))
    return ws, clock, prod


def _heads(conn):
    return {row["scope"]: row["snapshot_id"]
            for row in conn.execute("SELECT scope, snapshot_id FROM data_snapshot_heads")}


def _pin_names(workspace):
    """Read-pin names present in the workspace store; fragment bytes are skipped, not refused."""
    names = set()
    for path in (workspace.paths.store / "objects").rglob("*"):
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_bytes())
        except ValueError:
            continue
        if isinstance(document, dict) and "pin" in document and "name" in document:
            names.add(document["name"])
    return names


def test_generate_workspace_bounds_publishes_into_the_workspace_only(bounds_ws, source_root):
    ws, clock, prod = bounds_ws
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    prod_before, heads_before = _digest_tree(prod), _heads(ws.conn)
    _, artifacts_before = _ws_state(ws)
    record = prep.generate_workspace_bounds(ws, "shadow", manifest, source_root=source_root,
                                            as_of=_AS_OF, clock=clock)
    assert record["label"] == "reconstructed" and record["refresh_noop"] is False
    assert record["source_snapshot_id"] == ws.snapshot_id and record["bounds_snapshot_id"] != ws.snapshot_id
    assert record["entry_count"] == 1 and record["manifest_entry_count"] == 1
    suffix = record["record_id"].split(":", 1)[1]
    stored = json.loads((ws.paths.root / "size_prep" / ("provenance-" + suffix + ".json")).read_text())
    assert stored == record
    assert _heads(ws.conn) == {**heads_before, "shadow": record["bounds_snapshot_id"]}
    snapshot = Repository(ws.conn, ws.store, pin_snapshots=True).resolve_full(record["bounds_snapshot_id"])
    assert TABLE in snapshot.table_manifests
    rows = _bound_rows(Repository(ws.conn, ws.store), snapshot.snapshot)
    assert {row["ticker"] for row in rows} == {"AAAA"}
    assert ws.conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] > artifacts_before
    assert "size-prep-bounds:" + suffix in _pin_names(ws)
    assert "size-prep-manifest:" + manifest["manifest_id"].split(":", 1)[1] in _pin_names(ws)
    assert _digest_tree(prod) == prod_before
    assert (ws.paths.root / "supervisor.lock").is_file() and not (prod / "supervisor.lock").exists()


def test_generate_workspace_bounds_refuses_a_moved_or_absent_head(bounds_ws, source_root):
    ws, clock, prod = bounds_ws
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    first = prep.generate_workspace_bounds(ws, "shadow", manifest, source_root=source_root,
                                           as_of=_AS_OF, clock=clock)
    record_name = "provenance-" + first["record_id"].split(":", 1)[1] + ".json"
    state, heads, prod_before = _ws_state(ws), _heads(ws.conn), _digest_tree(prod)
    with pytest.raises(OpsError) as raised:
        prep.generate_workspace_bounds(ws, "shadow", manifest, source_root=source_root,
                                       as_of=_AS_OF, clock=clock)
    assert raised.value.code == "WORKSPACE_REFUSED"
    assert raised.value.problem.details == {"reason": "head_not_pinned", "scope": "shadow"}
    assert _ws_state(ws) == state and _heads(ws.conn) == heads
    with pytest.raises(OpsError) as raised:
        prep.generate_workspace_bounds(ws, "smoke", manifest, source_root=source_root,
                                       as_of=_AS_OF, clock=clock)
    assert raised.value.code == "WORKSPACE_REFUSED"
    assert raised.value.problem.details == {"reason": "head_not_pinned", "scope": "smoke"}
    assert _heads(ws.conn) == heads and "smoke" not in heads
    assert sorted(p.name for p in (ws.paths.root / "size_prep").iterdir()) == [record_name]
    assert _digest_tree(prod) == prod_before


def test_generate_workspace_bounds_refuses_an_event_bearing_exclusion_before_refresh(pinned_ws, source_root):
    ws, clock, prod = pinned_ws
    state, prod_before = _ws_state(ws), _digest_tree(prod)
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA", "EEEE", "ZZZZ"],
                                   allow_exclusions=True)
    with pytest.raises(OpsError) as raised:
        prep.generate_workspace_bounds(ws, "shadow", manifest, source_root=source_root,
                                       as_of=_AS_OF, clock=clock)
    assert raised.value.code == "PUBLICATION_REFUSED"
    assert raised.value.problem.details["reason"] == "excluded_ticker_has_events"
    assert _ws_state(ws) == state and _digest_tree(prod) == prod_before
    assert _heads(ws.conn) == {} and not (ws.paths.root / "size_prep").exists()


def test_generate_workspace_bounds_refuses_a_non_string_scope_as_a_typed_refusal(bounds_ws, source_root):
    ws, clock, prod = bounds_ws
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    state, prod_before = _ws_state(ws), _digest_tree(prod)
    with pytest.raises(OpsError) as raised:
        prep.generate_workspace_bounds(ws, 123, manifest, source_root=source_root, as_of=_AS_OF, clock=clock)
    assert raised.value.code == "WORKSPACE_REFUSED"
    assert raised.value.problem.details["reason"] == "scope_invalid"
    assert _ws_state(ws) == state and _digest_tree(prod) == prod_before


def test_generate_workspace_bounds_refuses_as_of_below_the_ceiling_without_writing(bounds_ws, source_root):
    ws, clock, prod = bounds_ws
    manifest = prep.build_manifest(source_root, session_ceiling=_AS_OF, tickers=["AAAA"])
    state, prod_before = _ws_state(ws), _digest_tree(prod)
    _refusal(lambda: prep.generate_workspace_bounds(ws, "shadow", manifest, source_root=source_root,
                                                    as_of="2024-02-06", clock=clock), "INVALID_REQUEST")
    assert _ws_state(ws) == state and _digest_tree(prod) == prod_before
    assert _heads(ws.conn) == {"shadow": ws.snapshot_id}
    assert not (ws.paths.root / "size_prep").exists()


def test_objects_root_is_validated_at_the_cli_boundary(tmp_path):
    from types import SimpleNamespace

    from engine.v2.ops.cli import computed_moves_command

    args = SimpleNamespace(source_root=tmp_path, scope="shadow", as_of=_AS_OF, dry_run=True)
    for invalid in (123, 1.5, True, [], {}, b"objects"):
        _refusal(lambda: computed_moves_command(args, tmp_path, None, None, objects_root=invalid),
                 "INVALID_REQUEST")
    assert not (tmp_path / computed_moves_store.INPUT_PATH).exists()
    assert not (tmp_path / "supervisor.lock").exists()
