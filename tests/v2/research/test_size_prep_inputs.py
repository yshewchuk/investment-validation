"""Pinned Tier-1 manifest, write-root bounds generation and reconstructed provenance (synthetic data only)."""
import gzip
import json
from copy import deepcopy

import pytest

from engine.v2.data.computed_move_bounds import TABLE
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore
from engine.v2.ops import computed_moves_store
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.errors import OpsError
from experiments import size_prep_inputs as prep
from tests.ops_support import FakeClock
from tests.v2.ops.test_v2_ops_computed_moves_store import (
    _AS_OF, _EVENT_DAYS, _build_parent, _closes_csv, _event_row, _head_row, _series_csv)


@pytest.fixture(autouse=True)
def _small_bar(monkeypatch):
    monkeypatch.setattr(computed_moves_store, "MIN_SCOREABLE", 5)


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
    record = prep.generate_bounds(root, source_root=source_root, manifest=manifest, scope="shadow", as_of=_AS_OF)
    assert _tree(source_root) == source_before
    assert record["label"] == "reconstructed"
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
                                          as_of="2024-02-06"), "INVALID_REQUEST")
    empty = tmp_path / "empty"
    empty.mkdir()
    _refusal(lambda: prep.generate_bounds(empty, source_root=source_root, manifest=manifest, scope="shadow",
                                          as_of=_AS_OF), "INVALID_REQUEST")
    (source_root / "data/raw/fetch/yfinance/aa/aa02.body.gz").write_bytes(gzip.compress(b"x", mtime=0))
    _refusal(lambda: prep.generate_bounds(root, source_root=source_root, manifest=manifest, scope="shadow",
                                          as_of=_AS_OF), "INPUT_CHANGED")
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
