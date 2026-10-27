"""Pinned Tier-1 history manifest, write-root bound-source generation and reconstructed provenance."""
import hashlib
import gzip
import os
import stat
import zlib
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from engine.v2.foundation import SystemClock, canonical_json, content_hash
from engine.v2.ops.catalog import connect
from engine.v2.ops.computed_moves_store import tier1_yfinance_history_fetcher
from engine.v2.ops.errors import fail
from engine.v2.ops.legacy_adapter import iter_raw_fetch_cache

MANIFEST_VERSION = "tier1_history_manifest.v1"
PROVENANCE_VERSION = "size_prep_provenance.v1"
_ENTRY_KEYS = {"ticker", "key", "fetched_at", "body_sha256"}
MAX_BODY_BYTES = 64 * 1024 * 1024


def _body(entry):
    """Decompressed cache body, read from a regular non-symlink file and refused above MAX_BODY_BYTES."""
    try:
        if not stat.S_ISREG(os.lstat(entry.path).st_mode):
            raise fail("SOURCE_INVALID", "a Tier-1 cache body must be a regular file")
        with gzip.open(entry.path, "rb") as handle:
            body = handle.read(MAX_BODY_BYTES + 1)
    except (OSError, EOFError, zlib.error):
        raise fail("SOURCE_INVALID", "a Tier-1 cache body is unreadable or corrupt") from None
    if len(body) > MAX_BODY_BYTES:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "a Tier-1 cache body exceeds the decompressed size limit")
    return body


def _day(value):
    try:
        ok = type(value) is str and date.fromisoformat(value).isoformat() == value
    except ValueError:
        ok = False
    if not ok:
        raise fail("INVALID_REQUEST", "a canonical YYYY-MM-DD session ceiling is required")
    return value


def _retrieved(value):
    """UTC instant of a fetched_at string; naive values are read as UTC."""
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        raise fail("INVALID_REQUEST", "fetched_at must be an ISO timestamp") from None
    return when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when.astimezone(timezone.utc)


def _identity(manifest):
    return content_hash({k: v for k, v in manifest.items() if k != "manifest_id"})


def build_manifest(source_root, *, session_ceiling, tickers):
    """Pin the entry the existing Tier-1 rule would pick for each explicit ticker."""
    _day(session_ceiling)
    if (type(tickers) is not list or not tickers or len(set(tickers)) != len(tickers)
            or any(type(t) is not str or not t or t != t.strip() for t in tickers)):
        raise fail("INVALID_REQUEST", "explicit unique ticker list required")
    cache, _ = tier1_yfinance_history_fetcher(source_root)
    entries = []
    for ticker in sorted(tickers):
        entry = cache.get(ticker)
        if entry is None:
            raise fail("SOURCE_NOT_FOUND", "a requested ticker has no eligible Tier-1 history entry")
        fetched_at = entry.meta.get("fetched_at")
        _retrieved(fetched_at)
        entries.append({"ticker": ticker, "key": entry.key, "fetched_at": fetched_at,
                        "body_sha256": hashlib.sha256(_body(entry)).hexdigest()})
    manifest = {"schema_version": MANIFEST_VERSION, "session_ceiling": session_ceiling, "entries": entries}
    return {**manifest, "manifest_id": _identity(manifest)}


def validate_manifest(manifest):
    """Closed shape, canonical ordering and identity; no source access."""
    if (type(manifest) is not dict or set(manifest) != {"schema_version", "session_ceiling", "entries", "manifest_id"}
            or manifest["schema_version"] != MANIFEST_VERSION or type(manifest["entries"]) is not list
            or not manifest["entries"]):
        raise fail("INVALID_REQUEST", "invalid Tier-1 history manifest")
    _day(manifest["session_ceiling"])
    for entry in manifest["entries"]:
        if (type(entry) is not dict or set(entry) != _ENTRY_KEYS
                or any(type(entry[k]) is not str or not entry[k] for k in _ENTRY_KEYS)):
            raise fail("INVALID_REQUEST", "invalid Tier-1 history manifest entry")
        _retrieved(entry["fetched_at"])
    tickers = [e["ticker"] for e in manifest["entries"]]
    if any(t != t.strip() for t in tickers):
        raise fail("INVALID_REQUEST", "invalid Tier-1 history manifest entry")
    if tickers != sorted(set(tickers)) or manifest["manifest_id"] != _identity(manifest):
        raise fail("INVALID_REQUEST", "Tier-1 history manifest ordering or identity mismatch")
    return manifest


def pinned_history_fetcher(source_root, manifest, *, strict=False):
    """Return ``(cache, fetch)`` shaped like ``tier1_yfinance_history_fetcher``, serving only pinned entries."""
    validate_manifest(manifest)
    available = {entry.key: entry for entry in iter_raw_fetch_cache(source_root, "yfinance")}
    cache = {}
    for pinned in manifest["entries"]:
        entry = available.get(pinned["key"])
        if (entry is None or entry.endpoint != "history" or entry.params.get("period") != "max"
                or str(entry.params.get("ticker")) != pinned["ticker"]
                or str(entry.meta.get("status")) != "200"
                or entry.meta.get("fetched_at") != pinned["fetched_at"]
                or hashlib.sha256(_body(entry)).hexdigest() != pinned["body_sha256"]):
            raise fail("INPUT_CHANGED", "a pinned Tier-1 history entry is missing or changed")
        cache[pinned["ticker"]] = entry

    digests = {p["ticker"]: p["body_sha256"] for p in manifest["entries"]}

    def fetch(ticker):
        entry = cache.get(ticker)
        if entry is None:
            if strict:
                raise fail("SOURCE_NOT_FOUND", "a refresh target is not pinned in the Tier-1 manifest")
            return b"", "legitimate_empty", {}, []
        body = _body(entry)
        if hashlib.sha256(body).hexdigest() != digests[ticker]:
            raise fail("INPUT_CHANGED", "a pinned Tier-1 history entry changed while being served")
        return body, "complete", {}, []

    return cache, fetch


def _head(conn, scope):
    row = conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope = ?", (scope,)).fetchone()
    return row["snapshot_id"] if row else None


def _provenance(manifest, source_snapshot_id, bounds_snapshot_id):
    entries = manifest["entries"]
    ceiling = date.fromisoformat(manifest["session_ceiling"])
    by_time = sorted(entries, key=lambda e: _retrieved(e["fetched_at"]))
    record = {"schema_version": PROVENANCE_VERSION, "label": "reconstructed",
        "session_ceiling": manifest["session_ceiling"], "manifest_id": manifest["manifest_id"],
        "entry_count": len(entries), "earliest_fetched_at": by_time[0]["fetched_at"],
        "latest_fetched_at": by_time[-1]["fetched_at"],
        "entries_fetched_after_ceiling": sum(_retrieved(e["fetched_at"]).date() > ceiling for e in entries),
        "source_snapshot_id": source_snapshot_id, "bounds_snapshot_id": bounds_snapshot_id}
    return {**record, "record_id": content_hash(record)}


def _write_record(write_root, record):
    directory = Path(write_root) / "size_prep"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("provenance-" + record["record_id"].split(":", 1)[1] + ".json")
    staging = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    staging.write_text(canonical_json(record))
    os.replace(staging, path)


def generate_bounds(write_root, *, source_root, manifest, scope, as_of):
    """Refresh the catalog in ``write_root`` with source bounds from pinned inputs; return the provenance record."""
    from engine.v2.ops.cli import computed_moves_command

    validate_manifest(manifest)
    if as_of != manifest["session_ceiling"]:
        raise fail("INVALID_REQUEST", "as_of must equal the manifest session ceiling")
    root = Path(write_root).resolve()
    catalog = root / "catalog.sqlite"
    if not catalog.is_file() or catalog.resolve().parent != root:
        raise fail("INVALID_REQUEST", "catalog.sqlite must exist directly inside the write root")
    started = time.monotonic()
    print("[size-prep] bounds start entries=" + str(len(manifest["entries"])), file=sys.stderr)
    history = pinned_history_fetcher(source_root, manifest, strict=True)
    conn = connect(str(catalog), must_exist=True)
    try:
        before = _head(conn, scope)
        args = SimpleNamespace(source_root=Path(source_root), scope=scope, as_of=as_of, dry_run=False)
        computed_moves_command(args, root, conn, SystemClock(), history=history, include_source_bounds=True)
        after = _head(conn, scope)
    finally:
        conn.close()
    record = _provenance(manifest, before, after)
    _write_record(root, record)
    print("[size-prep] bounds end elapsed=" + format(time.monotonic() - started, ".1f") + "s", file=sys.stderr)
    return record
