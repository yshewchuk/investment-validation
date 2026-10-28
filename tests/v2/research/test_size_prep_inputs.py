"""Pinned Tier-1 manifest, write-root bounds generation and reconstructed provenance (synthetic data only)."""
import gzip
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from engine.v2.contracts import DataQuery, KeyPredicate, TableContractRef
from engine.v2.data.computed_move_bounds import COLUMNS, CONTRACT, TABLE
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, content_hash
from engine.v2.ops import computed_moves_store
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.errors import OpsError
from experiments import size_prep_inputs as prep
from tests.ops_support import FakeClock
from tests.v2.ops.test_v2_ops_computed_moves_store import (
    _AS_OF, _BDAYS, _EVENT_DAYS, _build_parent, _closes_csv, _event_row, _head_row, _series_csv)

REF = TableContractRef(contract_id=CONTRACT.contract_id, definition_hash=CONTRACT.definition_hash)


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
    err = capsys.readouterr().err
    assert "[size-prep] scan 1 of at most" in err and "ETA<=" in err and "rss_kib=" in err
    assert "manifest end" in err and capsys.readouterr().out == ""
