"""Pure legacy event mapping: ``earnings_events`` rows -> ``EarningsEvent``,
versioned ``legacy_event_mapping.v1`` — phase-2 guide §5.4, §8.3, task brief
decision 3.

``get_event`` is the one impure entry point (it drives bounded scans through
a caller-supplied ``Repository``); every value transform below it
(:func:`map_row`, :func:`security_id_for_ticker`) is pure and independently
testable.

Column semantics were read from ``engine.data.schemas.EARNINGS_EVENTS``'s
``Column.doc`` text (task brief decision 3, final bullet) before writing this
mapping:

* ``session``/``session_src`` map straight across, as decision 3 directs.
  **Stop-and-report**: ``Column.doc`` declares ``session`` nullable ("null
  when no source in ``engine.calendar.SESSION_PRIORITY`` supplied a
  session"), but ``EarningsEvent.session`` is a required, non-``None`` ``str``
  — the two contradict. Rather than leave ``get_event`` unable to map any row
  with a null session, this module treats a null legacy ``session``/
  ``session_src`` as the empty string ``""`` (a "no session determined"
  sentinel), and this contradiction is called out in the task report as
  instructed rather than silently resolved.
* ``date_conflict``'s doc ("Forward sources disagree ... both rows kept")
  confirms a conflict is visible as *two rows*, not a rewritten duplicate
  primary key — consistent with checking ``event_cluster_id`` across sibling
  rows for conflict_status (decision 3) rather than expecting duplicate
  ``event_id``s.
* no other read column's doc contradicts decision 3.

Layer 1 of ``system_rearchitecture.md`` §4.1: imports only
``engine.v2.contracts``, ``engine.v2.foundation``, and this package's own
``errors`` — never ``engine.v2.ops`` or legacy ``engine.*``. ``get_event``
takes a ``Repository``-shaped object (duck-typed: anything with a ``.scan``
method) purely as a parameter, so this module never imports
``repository.py`` and creates no import cycle.
"""
from __future__ import annotations

from engine.v2.contracts.data import DataQuery, EarningsEvent, EventRef, KeyPredicate, SnapshotRef
from engine.v2.foundation import content_hash

from .errors import fail

__all__ = ["TABLE_NAME", "get_event", "map_row", "security_id_for_ticker"]

TABLE_NAME = "earnings_events"

#: Columns this module ever reads off an ``earnings_events`` row.
_EVENT_COLUMNS = ("event_id", "ticker", "event_date", "session", "session_src",
                  "date_agree", "date_conflict", "event_cluster_id")

#: Small, fixed bounds for the two lookups below — event/cluster lookups
#: return at most a handful of rows; these are far under every table's
#: (50000, 2000000) contract cap (task brief Facts).
_BATCH_CAP = 1000
_RESULT_CAP = 1000


def security_id_for_ticker(ticker: str) -> str:
    """``security_id = sha256({scheme: legacy_ticker.v1, ticker: exact_ticker})``
    (phase-2 guide §5.4)."""
    return content_hash({"scheme": "legacy_ticker.v1", "ticker": ticker})


def get_event(repository, event_ref: EventRef, snapshot_ref: SnapshotRef) -> EarningsEvent:
    """§8.3: one bounded query by exact ``event_id``, verified against the
    requested calendar revision. Zero rows is refused as not found; more than
    one row for one ``event_id`` is refused as ``MANIFEST_CORRUPT`` — a
    duplicate primary key is exactly the kind of catalog/fragment integrity
    defect that code already names elsewhere in this package (repository.py's
    own note: the guide's ``INTEGRITY_FAILED`` maps to ``MANIFEST_CORRUPT``
    here). ``IDENTITY_CONFLICT`` is reserved for an ambiguous *mapping*
    (task brief decision 4), not a duplicate-row defect, so it is not used
    for this case.
    """
    if TABLE_NAME not in snapshot_ref.table_versions:
        raise fail("CONTRACT_MISMATCH", "snapshot has no earnings_events table")
    dvr = snapshot_ref.table_versions[TABLE_NAME]
    if event_ref.calendar_revision != dvr.dataset_version_id:
        raise fail("CONTRACT_MISMATCH",
                  "event calendar_revision does not match the pinned dataset version",
                  details={"event_id": event_ref.event_id})
    rows = _rows_by_event_id(repository, snapshot_ref, dvr.table_contract_ref, event_ref.event_id)
    if not rows:
        raise fail("CONTRACT_MISMATCH", "event_id is not present under this calendar revision",
                  details={"event_id": event_ref.event_id})
    if len(rows) > 1:
        raise fail("MANIFEST_CORRUPT", "more than one row for one event_id is an integrity failure",
                  details={"event_id": event_ref.event_id})
    row = rows[0]
    conflict = bool(row["date_conflict"]) or _has_cluster_conflict(
        repository, snapshot_ref, dvr.table_contract_ref, row)
    return map_row(row, event_ref=event_ref, conflict=conflict)


def map_row(row: dict, *, event_ref: EventRef, conflict: bool) -> EarningsEvent:
    """One decoded ``earnings_events`` row -> ``EarningsEvent`` (decision 3)."""
    event_date = row["event_date"]
    scheduled_date = event_date.date().isoformat() if hasattr(event_date, "date") else str(event_date)[:10]
    return EarningsEvent(
        event_ref=event_ref,
        security_id=security_id_for_ticker(row["ticker"]),
        ticker_at_event=row["ticker"],
        scheduled_event_date=scheduled_date,
        session=row["session"] or "",
        actual_announcement_at=None,
        session_source=row["session_src"] or "",
        confidence=1.0 if row["date_agree"] else 0.0,
        conflict_status="conflict" if (row["date_conflict"] or conflict) else "none",
        known_from=None,
        supersedes_revision=None,
    )


def _rows_by_event_id(repository, snapshot_ref: SnapshotRef, contract_ref, event_id: str) -> list[dict]:
    query = DataQuery(
        snapshot_id=snapshot_ref.snapshot_id, table_contract_ref=contract_ref,
        columns=_EVENT_COLUMNS,
        key_filter=(KeyPredicate(column="event_id", operator="eq", values=(event_id,)),),
        order_by=("event_id",), max_batch_rows=_BATCH_CAP, max_result_rows=_RESULT_CAP)
    return _collect(repository, query)


def _has_cluster_conflict(repository, snapshot_ref: SnapshotRef, contract_ref, row: dict) -> bool:
    """True iff another row in this same pinned dataset version shares
    ``event_cluster_id`` with a different ``event_id`` (decision 3)."""
    cluster_id = row.get("event_cluster_id")
    if cluster_id is None:
        return False
    query = DataQuery(
        snapshot_id=snapshot_ref.snapshot_id, table_contract_ref=contract_ref,
        columns=("event_id", "ticker", "event_cluster_id"),
        key_filter=(KeyPredicate(column="ticker", operator="eq", values=(row["ticker"],)),),
        order_by=("event_id",), max_batch_rows=_BATCH_CAP, max_result_rows=_RESULT_CAP)
    for sibling in _collect(repository, query):
        if sibling["event_cluster_id"] == cluster_id and sibling["event_id"] != row["event_id"]:
            return True
    return False


def _collect(repository, query: DataQuery) -> list[dict]:
    rows: list[dict] = []
    for batch in repository.scan(query, table_name=TABLE_NAME):
        rows.extend(batch.to_pylist())
    return rows
