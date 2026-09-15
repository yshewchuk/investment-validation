"""``price_history`` capture — advancing the Tier-2 ``price_history`` catalog
table (design confirmed 2026-09-14; SEND-BACK 2026-09-14: "it isn't a Tier-2
table... no private ``_insert_*`` bypass and no separate ops-root SQLite
registry"). See ``engine.v2.data.price_history``'s module docstring for the
pure diff/as-of logic this orchestrates, and ``engine.v2.data.price_history_table``
for the registered ``TableContract``.

**Storage.** Every price_history fragment is a whole-ticker rewrite,
published through the shared operations ``ArtifactStore`` (``ops_root``, the
same store every other catalog object lives in) and committed through the
public ``catalog.commit_snapshot`` -- no private ``_insert_*`` call, no
separate sqlite file. See ``engine.v2.data.price_history_table``'s module
docstring for why one fragment always covers a whole ticker's history rather
than being appended to byte-for-byte: the catalog's non-overlapping-fragment
invariant cannot accept a later fragment that corrects an already-past date.

**Cadence (SEND-BACK requirement 1).** :func:`capture` always resolves
price_history's OWN newest dataset version first
(``Repository.latest_dataset_version`` -- a version chain kept independent of
any snapshot, since a capture run's cadence has nothing to do with the
nightly snapshot cadence), rewrites only the tickers this run's diff
touched, and then commits ONE NEW SNAPSHOT under ``scope`` that carries
every OTHER table's dataset version forward UNCHANGED (``Repository.
resolve_full`` on ``scope``'s current head -- proof, not a re-scan: every
byte of a committed snapshot is reconstructable from catalog SQL rows alone,
and ``catalog.py``'s ``_insert_contract``/``_insert_object``/
``_insert_fragment``/``_insert_dataset_version`` are idempotent no-ops on an
identical payload) alongside the fresh price_history version. A run that
changes nothing therefore commits a snapshot identical to the current head,
which ``commit_snapshot`` itself resolves as a no-op (no head CAS). The
derived-file reference pins (``engine.v2.data.reference_catalog``) are
copied forward from the base generation's own receipt into the new one, in
the same transaction, so a plan resolving the new head still finds them.

**Generation-pin compatibility, documented gap.** ``engine.v2.ops.
generation_binding.accepted_generation_refs`` reads a barrier stage's
accepted legacy read-set off ``attempt_input_bindings`` for the snapshot's
own committing ``attempt_id``. This module's commit reuses the BASE
generation's ``attempt_id``/``fence`` verbatim in its own receipt (a
deliberate choice, not an oversight: no legacy file is read here, so the
accepted legacy manifest is genuinely unchanged, and reusing the same
attempt_id means the existing ``legacy_manifest.json`` binding is found
without minting a new one -- which would need a real ``attempts`` row this
CLI-driven, non-supervised commit has no reason to create). ``fence_check``
is a caller-injected no-op for the same reason (``catalog.commit_snapshot``'s
own module docstring: this is one of the two judgement calls its design
explicitly leaves to a caller outside the supervised job system); real
concurrency safety here is ``commit_snapshot``'s own head compare-and-swap,
not the job fence.

**Multi-retrieval capture (2026-09-14 addition).** The planned
``live=True`` re-downloader (``engine/data/fetch.py:104-113``, the same
dated-cache-key pattern ``engine/data/pulls/computed_moves.py:140-152``
already uses) makes each day's Tier-1 ``yfinance``/``history`` fetch its own
cache entry -- ``params`` stays ``{"ticker", "period": "max"}`` on every one
(the ``date.today()`` component lives only in the opaque cache key, per
``engine/data/fetch.py``'s ``cache_key``/``CachedEntry.meta``, which never
records it), so :func:`_tier1_retrievals` collects EVERY matching entry per
ticker rather than the single last-iterated one, and :func:`capture` walks
them in ascending ``fetched_at`` order (ties on the retrieved body's own
hash) -- an already-captured ``source_hash`` is a no-op, and one older than
the ticker's latest already-observed ``retrieved_at`` is refused exactly as
a legitimate backdated capture would be, whether that ordering violation
comes from a single run or across two.
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from engine.v2.contracts import ObjectRef, TableContractRef
from engine.v2.data import catalog as data_catalog
from engine.v2.data import manifests, objects, price_history, reference_catalog
from engine.v2.data import price_download_sources as sources
from engine.v2.data.errors import DataError
from engine.v2.data.price_history_table import PRICE_HISTORY_CONTRACT, PRICE_HISTORY_TABLE_NAME
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, Clock, SystemClock, content_hash
from engine.v2.ops.errors import fail
from engine.v2.ops.legacy_adapter import iter_raw_fetch_cache

__all__ = ["FRAGMENT_COLUMNS", "capture"]

FRAGMENT_COLUMNS = ("ticker", "date", "close_adj", "close_raw", "high_raw", "retrieved_at",
                   "deleted", "source_kind", "source_hash", "capture_id")

_ARROW_SCHEMA = pa.schema([
    ("ticker", pa.string()), ("date", pa.string()), ("close_adj", pa.float64()),
    ("close_raw", pa.float64()), ("high_raw", pa.float64()), ("retrieved_at", pa.string()),
    ("deleted", pa.bool_()), ("source_kind", pa.string()), ("source_hash", pa.string()),
    ("capture_id", pa.string()),
])

_SUCCESS_OUTCOMES = ("added", "no_change", "duplicate_source_hash")
_PRICE_HISTORY_CONTRACT_REF = TableContractRef(contract_id=PRICE_HISTORY_CONTRACT.contract_id,
                                               definition_hash=PRICE_HISTORY_CONTRACT.definition_hash)


# --------------------------------------------------------------------------
# scanning the two legacy sources (read-only)
# --------------------------------------------------------------------------


def _px_retrievals(source_root: Path) -> dict[str, dict]:
    return {entry["ticker"]: entry for entry in _scan_legacy_px_tree(source_root)}


def _scan_legacy_px_tree(source_root: Path) -> list[dict]:
    directory = Path(source_root) / "earnings_predictions" / "data" / "raw" / "yfinance"
    if not directory.is_dir() or directory.is_symlink():
        return []
    out = []
    for path in sorted(directory.glob("px_*.csv")):
        if not path.is_file() or path.is_symlink():
            continue
        info = path.stat()
        out.append({"ticker": path.stem[len("px_"):], "path": path, "byte_size": info.st_size,
                    "mtime": info.st_mtime})
    return out


def _iso_from_epoch(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _tier1_retrievals(source_root: Path) -> dict[str, list[dict]]:
    """Every not-yet-superseded Tier-1 ``history``/``period=max`` retrieval
    per ticker, ascending by ``fetched_at`` (ties on the retrieved body's own
    sha256) -- see the module docstring's "multi-retrieval capture" note.
    Undated and ``live=True``-dated cache entries are indistinguishable by
    ``params`` alone (``engine/data/fetch.py``'s ``cache_key`` never stores
    the day component in ``meta``), which is exactly why every matching
    entry is collected here rather than the single last-iterated one.
    """
    by_ticker: dict[str, list[dict]] = {}
    for entry in iter_raw_fetch_cache(source_root, "yfinance"):
        if entry.endpoint != "history" or entry.params.get("period") != "max":
            continue
        ticker = entry.params.get("ticker")
        if not ticker:
            continue
        fetched_at = entry.meta.get("fetched_at") or _iso_from_epoch(entry.path.stat().st_mtime)
        by_ticker.setdefault(ticker, []).append(
            {"ticker": ticker, "entry": entry, "fetched_at": fetched_at})
    for items in by_ticker.values():
        items.sort(key=lambda item: (item["fetched_at"],
                                     hashlib.sha256(item["entry"].body()).hexdigest()))
    return by_ticker


def _stringify_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """``date`` as a plain ``YYYY-MM-DD`` string -- the one shape every
    downstream piece (``price_history.diff_retrieval``/``as_of_view``, the
    parquet fragment's ``pa.string()`` column) assumes."""
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out


def _parse_px(entry: dict) -> tuple[pd.DataFrame, bytes, str]:
    raw = entry["path"].read_bytes()
    frame = _stringify_dates(sources.read_legacy_px_csv(entry["path"]))
    retrieved_at = _iso_from_epoch(entry["mtime"])
    return frame, raw, retrieved_at


def _tier1_meta_bytes(entry) -> bytes:
    import json
    return json.dumps(entry.meta, sort_keys=True).encode("utf-8")


def _parse_tier1(entry: dict) -> tuple[pd.DataFrame, bytes, str]:
    body = entry["entry"].body()
    meta_bytes = _tier1_meta_bytes(entry["entry"])
    frame = sources.read_tier1_body(body)
    frame = frame.assign(close_raw=float("nan"), high_raw=float("nan"))
    frame = _stringify_dates(frame)
    retrieved_at = entry["fetched_at"]
    combined = body + b"\x00" + meta_bytes
    return frame, combined, retrieved_at


def _overlap_disagreement(px_entry: dict, tier1_entry: dict) -> int:
    """Count of dates where the px and Tier-1 series disagree on
    ``close_adj`` (only column both sources carry), for the dry-run report.
    """
    px_frame = sources.read_legacy_px_csv(px_entry["path"])
    tier1_frame = sources.read_tier1_body(tier1_entry["entry"].body())
    left = px_frame[["date", "close_adj"]].dropna()
    right = tier1_frame[["date", "close_adj"]].dropna()
    merged = left.merge(right, on="date", suffixes=("_px", "_tier1"))
    if merged.empty:
        return 0
    differs = (merged["close_adj_px"] - merged["close_adj_tier1"]).abs() > 1e-6
    return int(differs.sum())


# --------------------------------------------------------------------------
# reading a ticker's current stored rows straight off its fragment object
# (no snapshot resolution needed: price_history's own version chain is
# independent of any snapshot -- see the module docstring)
# --------------------------------------------------------------------------


def _read_fragment_rows(store: ArtifactStore, record) -> pd.DataFrame:
    columns = [c for c in FRAGMENT_COLUMNS if c != "ticker"]
    if record is None:
        return pd.DataFrame(columns=columns)
    path = objects.verify_object_path(store, record.object_ref)
    table = pq.read_table(path, columns=list(FRAGMENT_COLUMNS))
    if table.num_rows == 0:
        return pd.DataFrame(columns=columns)
    return table.to_pandas().drop(columns=["ticker"])


def _write_ticker_fragment(store: ArtifactStore, ticker: str, rows: pd.DataFrame, *,
                           request_hash: str):
    ordered = rows.sort_values(["date", "retrieved_at"]).reset_index(drop=True).copy()
    ordered.insert(0, "ticker", ticker)
    table = pa.Table.from_pandas(ordered[list(FRAGMENT_COLUMNS)], schema=_ARROW_SCHEMA,
                                 preserve_index=False)
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    raw = buf.getvalue().to_pybytes()
    ref = store.publish_bytes(raw, schema_ref=objects.PARQUET_FRAGMENT_SCHEMA_REF)
    object_ref = ObjectRef(kind="parquet_fragment", object_id=ref.artifact_id,
                           content_hash=ref.content_hash, byte_size=ref.byte_size)
    inspection = objects.inspect_fragment(store, object_ref, PRICE_HISTORY_CONTRACT,
                                          _PRICE_HISTORY_CONTRACT_REF, ticker)
    return manifests.fragment_record(inspection, _PRICE_HISTORY_CONTRACT_REF, input_receipt_refs=(),
                                     import_request_hash=request_hash)


# --------------------------------------------------------------------------
# capture log (data_price_captures, schema v7) -- read prior attempts across
# every past receipt; new attempts are staged in memory and inserted only
# inside the generation commit's own transaction (record_references), since
# schema v7's own trigger requires the referencing receipt to already be
# 'committed'.
# --------------------------------------------------------------------------


def _prior_attempts(conn: sqlite3.Connection, ticker: str) -> list[dict]:
    rows = conn.execute(
        "SELECT source_hash, retrieved_at, outcome FROM data_price_captures WHERE ticker = ? "
        "ORDER BY created_at", (ticker,)).fetchall()
    return [{"source_hash": r["source_hash"], "retrieved_at": r["retrieved_at"],
            "outcome": r["outcome"]} for r in rows]


def _capture_id(ticker: str, source_hash: str, retrieved_at: str) -> str:
    """Deterministic, stable ``capture_id`` embedded in a price_history row on
    an ``added`` outcome -- a function of the retrieval's own identity only
    (never ``created_at``/wall-clock), so a row's provenance stamp never
    depends on which run happened to add it.
    """
    return "capture_" + content_hash(
        {"ticker": ticker, "source_hash": source_hash, "retrieved_at": retrieved_at}
    ).removeprefix("sha256:")[:32]


def _audit_capture_id(receipt_id: str, capture_id: str, outcome: str) -> str:
    """The ``data_price_captures`` primary key -- distinct from the row-level
    ``capture_id`` above so that a *repeated* attempt at the same retrieval
    (a later run's ``duplicate_source_hash``/``refused_backdate`` no-op) never
    collides with an earlier run's audit row for the same retrieval: it is
    scoped to the actually-assigned ``receipt_id`` of the commit that logs it,
    which is unique per commit (``_commit_generation``'s ``request_hash``
    always embeds the base snapshot the run started from, which advances
    every time the scope's head moves).
    """
    return "capture_" + content_hash(
        {"receipt_id": receipt_id, "capture_id": capture_id, "outcome": outcome}
    ).removeprefix("sha256:")[:32]


def _attempt(capture_id: str, ticker: str, source_kind: str, source_hash: str, retrieved_at: str,
            outcome: str, rows_added: int, rows_tombstoned: int, created_at: str) -> dict:
    return {"capture_id": capture_id, "ticker": ticker, "source_kind": source_kind,
           "source_hash": source_hash, "retrieved_at": retrieved_at, "outcome": outcome,
           "rows_added": rows_added, "rows_tombstoned": rows_tombstoned, "created_at": created_at}


# --------------------------------------------------------------------------
# one ticker's whole capture run: every not-yet-captured entry, in order
# --------------------------------------------------------------------------


@dataclass
class _TickerOutcome:
    results: list
    attempts: list
    stored: pd.DataFrame
    changed: bool


def _capture_ticker(conn: sqlite3.Connection, ticker: str, entries: list[tuple[str, dict]], *,
                    stored: pd.DataFrame, created_at: str) -> _TickerOutcome:
    run_log = _prior_attempts(conn, ticker)
    results: list[dict] = []
    attempts: list[dict] = []
    changed = False

    def _record(attempt: dict) -> None:
        run_log.append(attempt)
        attempts.append(attempt)

    for source_kind, entry in entries:
        try:
            frame, raw, retrieved_at = (_parse_px(entry) if source_kind == "legacy_px_csv"
                                        else _parse_tier1(entry))
        except Exception as exc:  # noqa: BLE001 -- a malformed source (e.g. a zero-byte px
                                  # csv, real-data dry-run 2026-09-14) is this entry's own
                                  # problem, never a reason to abort the ticker or the run.
            results.append({"ticker": ticker, "outcome": "error",
                           "error": getattr(exc, "code", type(exc).__name__)})
            continue
        source_hash = hashlib.sha256(raw).hexdigest()
        capture_id = _capture_id(ticker, source_hash, retrieved_at)
        if any(a["source_hash"] == source_hash and a["outcome"] in _SUCCESS_OUTCOMES
               for a in run_log):
            _record(_attempt(capture_id, ticker, source_kind, source_hash, retrieved_at,
                             "duplicate_source_hash", 0, 0, created_at))
            results.append({"ticker": ticker, "outcome": "duplicate_source_hash", "rows_added": 0,
                           "rows_tombstoned": 0})
            continue
        successful_ats = [a["retrieved_at"] for a in run_log if a["outcome"] in _SUCCESS_OUTCOMES]
        try:
            price_history.check_not_backdated(successful_ats, retrieved_at)
        except DataError:
            _record(_attempt(capture_id, ticker, source_kind, source_hash, retrieved_at,
                             "refused_backdate", 0, 0, created_at))
            results.append({"ticker": ticker, "outcome": "error", "error": "INPUT_CHANGED"})
            continue
        try:
            new_rows = price_history.diff_retrieval(
                stored, frame, retrieved_at=retrieved_at, source_kind=source_kind,
                source_hash=source_hash, capture_id=capture_id)
        except DataError:
            _record(_attempt(capture_id, ticker, source_kind, source_hash, retrieved_at,
                             "refused_partial", 0, 0, created_at))
            results.append({"ticker": ticker, "outcome": "error", "error": "VALIDATION_FAILED"})
            continue
        if new_rows.empty:
            _record(_attempt(capture_id, ticker, source_kind, source_hash, retrieved_at,
                             "no_change", 0, 0, created_at))
            results.append({"ticker": ticker, "outcome": "no_change", "rows_added": 0,
                           "rows_tombstoned": 0})
            continue
        added = int((~new_rows["deleted"]).sum())
        tombstoned = int(new_rows["deleted"].sum())
        stored = pd.concat([stored, new_rows], ignore_index=True)
        changed = True
        _record(_attempt(capture_id, ticker, source_kind, source_hash, retrieved_at, "added",
                         added, tombstoned, created_at))
        results.append({"ticker": ticker, "outcome": "added", "rows_added": added,
                        "rows_tombstoned": tombstoned,
                        "estimated_compressed_bytes": len(raw)})
    return _TickerOutcome(results=results, attempts=attempts, stored=stored, changed=changed)


# --------------------------------------------------------------------------
# generation commit: reuse the other tables' dataset versions unchanged,
# add price_history's fresh one, copy the reference-input pins forward
# --------------------------------------------------------------------------


def _copy_reference_inputs(conn: sqlite3.Connection, *, old_receipt_id: str, new_receipt_id: str) -> None:
    inputs = reference_catalog.reference_inputs_for_receipt(conn, receipt_id=old_receipt_id)
    reference_catalog.insert_reference_inputs(conn, new_receipt_id, inputs)


def _insert_price_captures(conn: sqlite3.Connection, receipt_id: str, attempts: list[dict]) -> None:
    for a in attempts:
        audit_id = _audit_capture_id(receipt_id, a["capture_id"], a["outcome"])
        conn.execute(
            "INSERT INTO data_price_captures (capture_id, receipt_id, ticker, source_kind, "
            "source_hash, retrieved_at, outcome, rows_added, rows_tombstoned, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (audit_id, receipt_id, a["ticker"], a["source_kind"], a["source_hash"],
             a["retrieved_at"], a["outcome"], a["rows_added"], a["rows_tombstoned"],
             a["created_at"]))


def _commit_generation(conn: sqlite3.Connection, store: ArtifactStore, scope: str, *,
                       prior_manifest, updated_records: dict, attempts: list[dict],
                       clock: Clock):
    head = conn.execute("SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = ?",
                        (scope,)).fetchone()
    if head is None:
        # DEPENDENCY_FAILED, not SNAPSHOT_NOT_READY: that code is registered
        # in DATA_FAILURE_CODES (engine.v2.data.errors), not ops's own
        # FAILURE_CODES (engine.v2.contracts.operations) that engine.v2.ops.
        # errors.fail validates against -- see engine/v2/ops/snapshot_promotion.py:351
        # for a pre-existing (unrelated, out of this task's scope) instance of
        # the same mismatch.
        raise fail("DEPENDENCY_FAILED",
                  "scope has no existing head snapshot to add price_history to",
                  details={"scope": scope})
    old_snapshot_id, old_generation = head["snapshot_id"], head["generation"]
    old_receipt_id = reference_catalog.committed_receipt_for_snapshot(
        conn, scope=scope, snapshot_id=old_snapshot_id)
    if old_receipt_id is None:
        raise fail("DEPENDENCY_FAILED", "scope's head snapshot has no committed import receipt",
                  details={"scope": scope, "snapshot_id": old_snapshot_id})
    old_receipt = conn.execute(
        "SELECT attempt_id, fence FROM data_import_receipts WHERE receipt_id = ?",
        (old_receipt_id,)).fetchone()

    repository = Repository(conn, store)
    resolved = repository.resolve_full(old_snapshot_id)

    ph_records = tuple(sorted(updated_records.values(), key=lambda r: r.partition_key))
    ph_parent = prior_manifest.dataset_version_ref.dataset_version_id if prior_manifest else None
    ph_manifest = manifests.dataset_manifest(
        _PRICE_HISTORY_CONTRACT_REF, ph_records, knowledge_mode="reconstructed",
        coverage_receipt_refs=(), availability_evidence_refs=(),
        parent_dataset_version_id=ph_parent)

    table_manifests = {name: m for name, m in resolved.table_manifests.items()
                       if name != PRICE_HISTORY_TABLE_NAME}
    table_manifests[PRICE_HISTORY_TABLE_NAME] = ph_manifest
    new_snapshot = manifests.snapshot_ref(
        table_manifests, calendar_version=resolved.snapshot.calendar_version,
        source_priority_version=resolved.snapshot.source_priority_version,
        finality_receipt_refs=resolved.snapshot.finality_receipt_refs,
        parent_snapshot_id=old_snapshot_id)

    other_records = tuple(r for r in resolved.records
                          if r.table_contract_ref.contract_id != PRICE_HISTORY_CONTRACT.contract_id)
    all_records = other_records + ph_records
    contracts = {c.contract_id: c for c in resolved.contracts}
    contracts[PRICE_HISTORY_CONTRACT.contract_id] = PRICE_HISTORY_CONTRACT
    all_objects = tuple({r.object_ref.object_id: r.object_ref for r in all_records}.values())

    request_hash = content_hash(
        {"kind": "price_history_generation", "scope": scope, "base_snapshot_id": old_snapshot_id,
         "price_history_dataset_version_id": ph_manifest.dataset_version_ref.dataset_version_id,
         "result_manifest_hash": new_snapshot.manifest_hash})
    receipt_id = "receipt_ph_" + request_hash.removeprefix("sha256:")[:32]

    def _record_references(c: sqlite3.Connection, rid: str) -> None:
        _copy_reference_inputs(c, old_receipt_id=old_receipt_id, new_receipt_id=rid)
        _insert_price_captures(c, rid, attempts)

    return data_catalog.commit_snapshot(
        conn, scope=scope, request_hash=request_hash, contracts=tuple(contracts.values()),
        objects=all_objects, records=all_records, manifests=tuple(table_manifests.values()),
        snapshot=new_snapshot, expected_head_snapshot_id=old_snapshot_id,
        expected_head_generation=old_generation, receipt_id=receipt_id,
        attempt_id=old_receipt["attempt_id"], fence=old_receipt["fence"],
        fence_check=lambda c: None, clock=clock, store=store,
        record_references=_record_references, audit_partitions=False)


# --------------------------------------------------------------------------
# capture — the ops-CLI-facing orchestrator
# --------------------------------------------------------------------------


def capture(conn: sqlite3.Connection, store: ArtifactStore, source_root: Path, *, scope: str,
           dry_run: bool = False, clock: Clock | None = None) -> dict:
    """Read both legacy sources (read-only), capture every not-yet-captured
    retrieval per ticker (px file takes precedence over Tier-1; every
    dated/undated Tier-1 ``history`` entry a ticker has, oldest first), and
    -- unless ``dry_run`` -- commit one new snapshot generation under
    ``scope`` carrying every other table's dataset version forward unchanged
    alongside price_history's fresh one. Never touches ``source_root``'s bytes.
    """
    clock = clock or SystemClock()
    source_root = Path(source_root)
    px = _px_retrievals(source_root)
    tier1 = _tier1_retrievals(source_root)
    tickers = sorted(set(px) | set(tier1))
    created_at = clock.now().isoformat()

    repository = Repository(conn, store)
    prior_manifest, prior_records = repository.latest_dataset_version(PRICE_HISTORY_CONTRACT.contract_id)
    prior_by_ticker = {r.partition_key: r for r in prior_records}
    updated_records = dict(prior_by_ticker)

    results: list[dict] = []
    all_attempts: list[dict] = []
    disagreements: dict[str, int] = {}

    for ticker in tickers:
        entries = ([("legacy_px_csv", px[ticker])] if ticker in px
                  else [("tier1_fetch", e) for e in tier1.get(ticker, [])])
        stored = _read_fragment_rows(store, prior_by_ticker.get(ticker))
        outcome = _capture_ticker(conn, ticker, entries, stored=stored, created_at=created_at)
        results.extend(outcome.results)
        all_attempts.extend(outcome.attempts)
        if outcome.changed and not dry_run:
            request_hash = content_hash(
                {"kind": "price_history_fragment", "ticker": ticker, "created_at": created_at})
            updated_records[ticker] = _write_ticker_fragment(
                store, ticker, outcome.stored, request_hash=request_hash)
        if ticker in px and ticker in tier1:
            try:
                disagreements[ticker] = _overlap_disagreement(px[ticker], tier1[ticker][-1])
            except Exception:  # noqa: BLE001 -- best-effort reporting only.
                pass

    report = _summarize(results, disagreements, dry_run=dry_run)
    if dry_run:
        return report
    receipt = _commit_generation(conn, store, scope, prior_manifest=prior_manifest,
                                 updated_records=updated_records, attempts=all_attempts, clock=clock)
    report["receipt_id"] = receipt.receipt_id
    report["result_snapshot_id"] = receipt.resulting_head_snapshot_id
    return report


def _summarize(results: list[dict], disagreements: dict, *, dry_run: bool) -> dict:
    by_outcome: dict[str, int] = {}
    tickers_seen = set()
    rows_added = rows_tombstoned = estimated_bytes = 0
    for r in results:
        by_outcome[r["outcome"]] = by_outcome.get(r["outcome"], 0) + 1
        tickers_seen.add(r["ticker"])
        rows_added += r.get("rows_added", 0)
        rows_tombstoned += r.get("rows_tombstoned", 0)
        estimated_bytes += r.get("estimated_compressed_bytes", 0)
    return {"schema_version": "price_history_capture_report.v1.0", "dry_run": dry_run,
           "tickers_seen": len(tickers_seen), "by_outcome": by_outcome, "rows_added": rows_added,
           "rows_tombstoned": rows_tombstoned, "estimated_compressed_bytes": estimated_bytes,
           "overlapping_tickers": len(disagreements),
           "overlapping_tickers_with_disagreement": sum(1 for v in disagreements.values() if v > 0),
           "disagreeing_dates_total": sum(disagreements.values())}
