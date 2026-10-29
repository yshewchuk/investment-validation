"""Pinned Tier-1 history manifest, write-root bound-source generation, reconstructed provenance and workspace admission."""
import hashlib
import gzip
import json
import os
import resource
import stat
import zlib
import sys
import tempfile
import time
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from engine.data.fetch import CachedEntry
from engine.v2.contracts import DataQuery, KeyPredicate
from engine.v2.data.repository import Repository
from engine.v2.data.computed_move_bounds import TABLE
from engine.v2.data.events import TABLE_NAME as EVENTS_TABLE
from engine.v2.foundation import ArtifactStore, SystemClock, canonical_json, content_hash
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import register_artifact, transaction
from engine.v2.ops.errors import fail
from engine.v2.ops.workspace.scope import resolve_size_scope_snapshot, size_scope_repository

MANIFEST_VERSION = "tier1_history_manifest.v1"
EXCLUSION_MANIFEST_VERSION = "tier1_history_manifest.v2"
PROVENANCE_VERSION = "size_prep_provenance.v1"
ADMISSION_VERSION = "size_prep_manifest_admission.v1"
_ENTRY_KEYS = {"ticker", "key", "fetched_at", "body_sha256"}
_MANIFEST_KEYS = {"schema_version", "session_ceiling", "entries", "manifest_id"}
_EXCLUDED_KEY = "excluded_tickers"
MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_LOGGED_NAMES = 20
MAX_EVENT_OBJECT_BYTES = 64 * 1024 * 1024


def _read_regular(path, limit, *, gz=False):
    """Read at most ``limit + 1`` bytes from ONE open of a regular, non-symlink file; None if absent."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError:
        raise fail("SOURCE_INVALID", "a Tier-1 cache file cannot be opened as a regular file") from None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise fail("SOURCE_INVALID", "a Tier-1 cache file must be a regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            return (gzip.GzipFile(fileobj=handle) if gz else handle).read(limit + 1)
    except (OSError, EOFError, zlib.error):
        raise fail("SOURCE_INVALID", "a Tier-1 cache file is unreadable or corrupt") from None


def _body(entry):
    """Decompressed cache body from a single capped regular-file open, refused above MAX_BODY_BYTES."""
    body = _read_regular(entry.path, MAX_BODY_BYTES, gz=True)
    if body is None:
        raise fail("SOURCE_INVALID", "a Tier-1 cache body disappeared")
    if len(body) > MAX_BODY_BYTES:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "a Tier-1 cache body exceeds the decompressed size limit")
    return body


def _log(message, started):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("[size-prep] " + message + " elapsed=" + format(time.monotonic() - started, ".1f") + "s rss_kib=" + str(rss), file=sys.stderr)


MAX_META_BYTES = 1024 * 1024
MAX_SIDECARS = 2_000_000
PROGRESS_EVERY = 100_000


def _entries(source_root, keys=None):
    """Stream Tier-1 yfinance sidecars as CachedEntry objects; at most MAX_SIDECARS scanned, capped regular-file reads only."""
    base = Path(source_root) / "data" / "raw" / "fetch" / "yfinance"
    if not Path(source_root).is_dir():
        raise fail("INVALID_REQUEST", "source root must be an existing directory")
    scanned, started = 0, time.monotonic()
    for meta_path in base.glob("*/*.meta.json"):
        scanned += 1
        if scanned > MAX_SIDECARS:
            raise fail("RESOURCE_LIMIT_EXCEEDED", "the Tier-1 cache holds more sidecars than the scan limit")
        if scanned % PROGRESS_EVERY == 0:
            rate = scanned / max(time.monotonic() - started, 1e-9)
            _log("scan " + str(scanned) + " of at most " + str(MAX_SIDECARS) + " sidecars, " + format(rate, ".0f")
                 + "/s, ETA<=" + format((MAX_SIDECARS - scanned) / max(rate, 1e-9), ".0f") + "s", started)
        key = meta_path.name[: -len(".meta.json")]
        if keys is not None and key not in keys:
            continue
        raw = _read_regular(meta_path, MAX_META_BYTES)
        if raw is None:
            continue
        if len(raw) > MAX_META_BYTES:
            raise fail("SOURCE_INVALID", "a Tier-1 sidecar must be within the size limit")
        try:
            meta = json.loads(raw)
        except ValueError:
            continue
        body_path = meta_path.with_name(key + ".body.gz")
        if isinstance(meta, dict) and meta.get("key") == key and body_path.exists():
            yield CachedEntry(meta=meta, path=body_path)
    _log("scan end " + str(scanned) + " sidecars", started)


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


def build_manifest(source_root, *, session_ceiling, tickers, allow_exclusions=False):
    """Pin the entry the existing Tier-1 rule would pick for each explicit ticker.

    ``allow_exclusions`` (workspace-mode admission, slice 2a of #1614): instead of
    refusing, tickers with no eligible Tier-1 entry are retained in a canonical
    ``excluded_tickers`` list and the manifest is published as
    ``tier1_history_manifest.v2``; admission still refuses if an excluded ticker
    has any event. ``allow_exclusions=False`` keeps ``tier1_history_manifest.v1``
    byte-for-byte.
    """
    _day(session_ceiling)
    if (type(tickers) is not list or not tickers
            or any(type(t) is not str or not t or t != t.strip() for t in tickers)
            or len(set(tickers)) != len(tickers)):
        raise fail("INVALID_REQUEST", "explicit unique ticker list required")
    started = time.monotonic()
    _log("manifest start tickers=" + str(len(tickers)), started)
    cache = {}
    for entry in _entries(source_root):
        if entry.endpoint != "history" or entry.params.get("period") != "max" or str(entry.meta.get("status")) != "200":
            continue
        name = entry.params.get("ticker")
        order = (str(entry.meta.get("fetched_at") or ""), str(entry.path))
        if name and (str(name) not in cache or order > cache[str(name)][0]):
            cache[str(name)] = (order, entry)
    entries, excluded = [], []
    for ticker in sorted(tickers):
        entry = cache.get(ticker, (None, None))[1]
        if entry is None:
            if not allow_exclusions:
                raise fail("SOURCE_NOT_FOUND", "a requested ticker has no eligible Tier-1 history entry")
            excluded.append(ticker)
            continue
        fetched_at = entry.meta.get("fetched_at")
        _retrieved(fetched_at)
        entries.append({"ticker": ticker, "key": entry.key, "fetched_at": fetched_at,
                        "body_sha256": hashlib.sha256(_body(entry)).hexdigest()})
    _log("manifest end entries=" + str(len(entries)) + " excluded=" + str(len(excluded)), started)
    version = EXCLUSION_MANIFEST_VERSION if allow_exclusions else MANIFEST_VERSION
    manifest = {"schema_version": version, "session_ceiling": session_ceiling, "entries": entries}
    if allow_exclusions:
        manifest[_EXCLUDED_KEY] = sorted(set(excluded))
    return {**manifest, "manifest_id": _identity(manifest)}


def validate_manifest(manifest):
    """Closed shape, canonical ordering and identity; no source access.

    The shape and version are one exact pair: v1 carries no ``excluded_tickers``,
    v2 requires the canonical list, and entries plus exclusions are never both empty.
    """
    if type(manifest) is not dict:
        raise fail("INVALID_REQUEST", "invalid Tier-1 history manifest")
    version = manifest.get("schema_version")
    if version == MANIFEST_VERSION:
        expected_keys = _MANIFEST_KEYS
    elif version == EXCLUSION_MANIFEST_VERSION:
        expected_keys = _MANIFEST_KEYS | {_EXCLUDED_KEY}
    else:
        raise fail("INVALID_REQUEST", "invalid Tier-1 history manifest")
    if set(manifest) != expected_keys or type(manifest["entries"]) is not list:
        raise fail("INVALID_REQUEST", "invalid Tier-1 history manifest")
    excluded = manifest.get(_EXCLUDED_KEY)
    if version == EXCLUSION_MANIFEST_VERSION and type(excluded) is not list:
        raise fail("INVALID_REQUEST", "invalid Tier-1 history manifest exclusion list")
    if excluded is not None and (type(excluded) is not list
            or any(type(t) is not str or not t or t != t.strip() for t in excluded)
            or excluded != sorted(set(excluded))):
        raise fail("INVALID_REQUEST", "invalid Tier-1 history manifest exclusion list")
    if not manifest["entries"] and not (excluded or []):
        raise fail("INVALID_REQUEST", "empty bounds require an explicit exclusion admission")
    _day(manifest["session_ceiling"])
    for entry in manifest["entries"]:
        if (type(entry) is not dict or set(entry) != _ENTRY_KEYS
                or any(type(entry[k]) is not str or not entry[k] for k in _ENTRY_KEYS)):
            raise fail("INVALID_REQUEST", "invalid Tier-1 history manifest entry")
        _retrieved(entry["fetched_at"])
    tickers = [e["ticker"] for e in manifest["entries"]]
    if any(t != t.strip() for t in tickers):
        raise fail("INVALID_REQUEST", "invalid Tier-1 history manifest entry")
    if excluded is not None and set(excluded) & set(tickers):
        raise fail("INVALID_REQUEST", "a ticker is both pinned and excluded")
    if tickers != sorted(set(tickers)) or manifest["manifest_id"] != _identity(manifest):
        raise fail("INVALID_REQUEST", "Tier-1 history manifest ordering or identity mismatch")
    return manifest


def pinned_history_fetcher(source_root, manifest, *, strict=False):
    """Return ``(cache, fetch)`` shaped like ``tier1_yfinance_history_fetcher``, serving only pinned entries."""
    validate_manifest(manifest)
    available = {entry.key: entry for entry in _entries(source_root, {p["key"] for p in manifest["entries"]})}
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
    excluded = (set(manifest[_EXCLUDED_KEY]) if manifest["schema_version"] == EXCLUSION_MANIFEST_VERSION
                else set())

    def fetch(ticker):
        entry = cache.get(ticker)
        if entry is None:
            if strict and ticker not in excluded:
                raise fail("SOURCE_NOT_FOUND", "a refresh target is not pinned in the Tier-1 manifest")
            return b"", "legitimate_empty", {}, []
        body = _body(entry)
        if hashlib.sha256(body).hexdigest() != digests[ticker]:
            raise fail("INPUT_CHANGED", "a pinned Tier-1 history entry changed while being served")
        return body, "complete", {}, []

    return cache, fetch


def _provenance(manifest, served, source_snapshot_id, bounds_snapshot_id, *, refresh_noop):
    """``served`` are the manifest entries the refresh actually fetched (none on a no-op)."""
    ceiling = date.fromisoformat(manifest["session_ceiling"])
    by_time = sorted(served, key=lambda e: _retrieved(e["fetched_at"]))
    record = {"schema_version": PROVENANCE_VERSION, "label": "reconstructed",
        "session_ceiling": manifest["session_ceiling"], "manifest_id": manifest["manifest_id"],
        "manifest_entry_count": len(manifest["entries"]), "entry_count": len(served),
        "earliest_fetched_at": by_time[0]["fetched_at"] if by_time else None,
        "latest_fetched_at": by_time[-1]["fetched_at"] if by_time else None,
        "entries_fetched_after_ceiling": sum(_retrieved(e["fetched_at"]).date() > ceiling for e in served),
        "source_snapshot_id": source_snapshot_id, "bounds_snapshot_id": bounds_snapshot_id,
        "refresh_noop": refresh_noop}
    return {**record, "record_id": content_hash(record)}


def _write_record(write_root, record):
    directory = Path(write_root) / "size_prep"
    if directory.is_symlink():
        raise fail("INVALID_REQUEST", "the provenance directory must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("provenance-" + record["record_id"].split(":", 1)[1] + ".json")
    descriptor, name = tempfile.mkstemp(dir=directory, prefix=path.name + ".tmp-")
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(canonical_json(record))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _related(left, right):
    return left == right or left in right.parents or right in left.parents


def _checked_write_root(write_root, production_roots):
    """Literal path equals its realpath and is unrelated to every named production root.

    TODO: consolidate with the research workspace's root check (PR #1506 ``_checked_root``) once merged.
    """
    literal = Path(os.path.abspath(write_root))
    if Path(os.path.realpath(literal)) != literal:
        raise fail("INVALID_REQUEST", "the write root must not traverse a symlink")
    if (type(production_roots) not in (list, tuple)
            or any(not isinstance(p, (str, os.PathLike)) for p in production_roots)):
        raise fail("INVALID_REQUEST", "production_roots must be a list or tuple of paths")
    production = [Path(os.path.realpath(p)) for p in production_roots]
    if not production or any(_related(literal, p) for p in production):
        raise fail("INVALID_REQUEST", "the write root must name production roots and be unrelated to each")
    return literal


def generate_bounds(write_root, *, source_root, manifest, scope, as_of, production_roots):
    """Refresh the catalog in ``write_root`` (unrelated to every ``production_roots`` entry) with source bounds from pinned inputs; return the provenance record."""
    from engine.v2.ops.cli import computed_moves_command

    validate_manifest(manifest)
    if manifest["schema_version"] == EXCLUSION_MANIFEST_VERSION:
        raise fail("WORKSPACE_REFUSED", "exclusion manifests are admitted through workspace mode only",
                   details={"reason": "workspace_admission_required"})
    if as_of != manifest["session_ceiling"]:
        raise fail("INVALID_REQUEST", "as_of must equal the manifest session ceiling")
    root = _checked_write_root(write_root, production_roots)
    catalog = root / "catalog.sqlite"
    if not catalog.is_file() or catalog.resolve().parent != root:
        raise fail("INVALID_REQUEST", "catalog.sqlite must exist directly inside the write root")
    started = time.monotonic()
    _log("bounds start entries=" + str(len(manifest["entries"]))
         + " excluded=" + str(len(manifest.get(_EXCLUDED_KEY) or ())), started)
    cache, fetch = pinned_history_fetcher(source_root, manifest, strict=True)
    served = set()

    def tracked(ticker):
        result = fetch(ticker)
        served.add(ticker)
        return result

    history = (cache, tracked)
    conn = open_catalog(catalog, clock=SystemClock())
    try:
        args = SimpleNamespace(source_root=Path(source_root), scope=scope, as_of=as_of, dry_run=False)
        report = computed_moves_command(args, root, conn, SystemClock(), history=history, include_source_bounds=True)
        source = report["parent_snapshot_id"]
        candidate = report["result"].get("candidate_snapshot_id")
        bounds = candidate or source
        if candidate is None:
            parent = Repository(
                conn, ArtifactStore(root), pin_snapshots=True
            ).resolve_full(source)
            if TABLE not in parent.table_manifests:
                raise fail("PUBLICATION_REFUSED", "the refresh committed nothing and the head carries no source bounds")
    finally:
        conn.close()
    pinned = [e for e in manifest["entries"] if e["ticker"] in served]
    record = _provenance(manifest, pinned, source, bounds, refresh_noop=candidate is None)
    _write_record(root, record)
    _log("bounds end", started)
    return record


def _names(values):
    """A bounded rendering of ticker names for stderr logs: at most MAX_LOGGED_NAMES, then a remainder count."""
    shown = ", ".join(values[:MAX_LOGGED_NAMES])
    return shown + (" (+%d more)" % (len(values) - MAX_LOGGED_NAMES) if len(values) > MAX_LOGGED_NAMES else "")


def _excluded_event_tickers(source, snapshot, excluded, started):
    """Excluded tickers that carry at least one earnings event in the pinned snapshot; read-only.

    A nonempty exclusion list over a snapshot without the events table cannot be verified and
    refuses ``PUBLICATION_REFUSED`` (reason ``event_coverage_unavailable``); empty exclusions need no scan.
    """
    if not excluded:
        return []
    if EVENTS_TABLE not in snapshot.table_versions:
        raise fail("PUBLICATION_REFUSED", "the pinned snapshot does not carry the earnings events table",
                   details={"reason": "event_coverage_unavailable", "excluded_count": len(excluded)})
    contract = source.table_contract(snapshot, EVENTS_TABLE)
    ref = snapshot.table_versions[EVENTS_TABLE].table_contract_ref
    predicate = KeyPredicate(column="ticker", operator="in", values=tuple(excluded))
    bound = source.scan_population_bound(snapshot.snapshot_id, table_name=EVENTS_TABLE,
                                         table_contract_ref=ref, key_filter=(predicate,))
    _log("event check " + str(bound) + " candidate rows for " + str(len(excluded)) + " exclusions", started)
    if bound < 1:
        return []
    query = DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=ref, columns=("ticker",),
                      key_filter=(predicate,), order_by=tuple(contract.primary_key),
                      max_batch_rows=min(bound, contract.maximum_batch_rows), max_result_rows=bound)
    return sorted({row["ticker"] for batch in source.scan(query, table_name=EVENTS_TABLE,
                                                          max_object_bytes=MAX_EVENT_OBJECT_BYTES)
                   for row in batch.to_pylist()})


def admit_manifest(workspace, scope, manifest, *, clock):
    """Admit a manifest into a research workspace: bind it to the pinned snapshot, refuse
    event-bearing exclusions over the read-only source, and record manifest CAS/read-pin
    metadata in ``workspace.store``/``workspace.conn`` only — never a production path."""
    validate_manifest(manifest)
    started = time.monotonic()
    excluded = sorted(set(manifest.get(_EXCLUDED_KEY) or ()))
    snapshot = resolve_size_scope_snapshot(workspace, scope)
    _log("admission start snapshot=" + snapshot.snapshot_id + " excluded=" + str(len(excluded))
         + " names=" + (_names(excluded) if excluded else "none"), started)
    offenders = _excluded_event_tickers(workspace.source, snapshot, excluded, started)
    if offenders:
        raise fail("PUBLICATION_REFUSED", "an excluded ticker has earnings events in the pinned snapshot",
                   details={"reason": "excluded_ticker_has_events", "tickers": _names(offenders)})
    view = size_scope_repository(workspace, scope, clock=clock)
    view.resolve_pinned(scope)
    ref = workspace.store.publish_bytes(canonical_json(manifest).encode("utf-8"),
                                       schema_ref=manifest["schema_version"])
    with transaction(workspace.conn):
        register_artifact(workspace.conn, ref, None, clock)
    fields = {"schema_version": ADMISSION_VERSION, "scope": scope, "scope_binding_id": view.binding_id,
              "snapshot_id": snapshot.snapshot_id, "manifest_hash": snapshot.manifest_hash,
              "manifest_id": manifest["manifest_id"], "manifest_object": ref.content_hash,
              "session_ceiling": manifest["session_ceiling"], "entry_count": len(manifest["entries"]),
              "excluded_count": len(excluded)}
    pin = workspace.record_read_pin("size-prep-manifest:" + manifest["manifest_id"].split(":", 1)[1],
                                    fields, clock=clock)
    _log("admission end excluded=" + str(len(excluded)) + " manifest_object=" + ref.content_hash, started)
    return {**fields, "admission_id": content_hash(fields), "manifest_ref": ref, "read_pin_ref": pin}
