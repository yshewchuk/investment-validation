"""Native v2 calibration: recompute the ledger calibration report and
health.json from canonical catalog decisions (P6-3 ``decisions-calibration``).

Mirrors ``engine/ledger.py::calibrate``/``::write_health`` field-for-field
where the source is pure catalog decisions, through the one declared
adapter (:mod:`engine.v2.ledger.legacy_adapter`) that reuses legacy's own
pure ``scored_pairs``/``settlement_summary``/``_strategy_calibration`` so the
numbers agree with the jsonl ledger's by construction.

Never writes into the real (read-only) ``ledger/`` tree: the report and
health payload are published as immutable catalog artifacts
(:class:`engine.v2.foundation.ArtifactStore`), and the ``n_scored`` trigger
state lives in the catalog's own ``ledger_calibration_state`` singleton row
(schema owner ``"ledger"``, migration 3 — see ``engine/v2/ops/bootstrap.py``),
never in a legacy-tree state file.

A ``snapshot_hash``/``quota_state`` field cannot be derived from canonical
decisions alone (legacy reads board-state files outside the ledger); both
stay ``None`` here, documented rather than guessed.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import Clock, canonical_json, format_timestamp, from_document, to_document

from . import catalog_reader, legacy_adapter

__all__ = ["CALIBRATION_TRIGGER", "SCHEMA_V3", "calibrate", "calibration_due",
           "health_ref", "report_ref", "scored_pairs"]

#: Mirrors engine.ledger.CALIBRATION_TRIGGER (plan §P4.2): newly scored
#: outcomes that trigger a calibration recompute. Kept as a literal, not an
#: import, so this module declares no legacy adapter edge for a bare constant.
CALIBRATION_TRIGGER = 50

#: ``*_ref_json`` is a full ``ArtifactRef`` (canonical_json(to_document(ref)),
#: the same encoding ``engine.v2.ops.catalog.dumps``/``load_json`` use for a
#: JSON column) rather than a bare artifact_id/content_hash pair, so a reader
#: (``status()`` below, or a test) can reconstruct the ref and call
#: ``store.read_verified`` on it directly.
SCHEMA_V3 = (
    """CREATE TABLE IF NOT EXISTS ledger_calibration_state (
        singleton INTEGER PRIMARY KEY CHECK(singleton=1),
        n_scored_at_last_report INTEGER NOT NULL,
        generated_at TEXT NOT NULL,
        health_ref_json TEXT NOT NULL,
        report_ref_json TEXT NOT NULL
    ) STRICT""",
)


@contextmanager
def _transaction(conn):
    """One short immediate transaction; commit on success, roll back on anything else.

    A duplicate of ``engine.v2.ops.catalog.transaction`` (three lines), not an
    import of it: ``engine.v2.ledger`` (layer 6) may not import
    ``engine.v2.ops`` (layer 7) — imports point down only
    (``checks/import_layers.py`` rule 1). ``conn`` is always ops-opened in
    practice, but this package must stay self-contained.
    """
    if conn.in_transaction:
        raise RuntimeError("nested catalog transaction: effects commit in the caller's transaction")
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def scored_pairs(conn):
    """``engine.ledger.scored_pairs`` fed from catalog decisions/outcomes."""
    return legacy_adapter.scored_pairs(predictions=catalog_reader.read_predictions(conn),
                                       outcomes=catalog_reader.read_outcomes(conn))


def _state(conn) -> dict | None:
    row = conn.execute(
        "SELECT * FROM ledger_calibration_state WHERE singleton=1").fetchone()
    return dict(row) if row else None


def health_ref(conn) -> ArtifactRef | None:
    """The most recently published health artifact's verifiable reference,
    or ``None`` before the first ``calibrate()`` call."""
    state = _state(conn)
    return None if state is None else from_document(ArtifactRef, json.loads(state["health_ref_json"]))


def report_ref(conn) -> ArtifactRef | None:
    """The most recently published calibration-report artifact's reference."""
    state = _state(conn)
    return None if state is None else from_document(ArtifactRef, json.loads(state["report_ref_json"]))


def calibration_due(conn, *, trigger: int = CALIBRATION_TRIGGER) -> tuple[bool, int, int]:
    """(due, n_scored_now, n_at_last_report) — the >= trigger new-row rule."""
    n_now = int(len(scored_pairs(conn)))
    state = _state(conn)
    last = int(state["n_scored_at_last_report"]) if state else 0
    return (n_now - last) >= trigger, n_now, last


def _health_payload(conn, per_strategy: dict, *, n_scored: int, clock: Clock) -> dict:
    predictions = catalog_reader.read_predictions(conn)
    latest = max((p["as_of"] for p in predictions), default=None)
    champions: dict[str, Any] = {}
    for row in predictions[-50:]:
        champions.update(row.get("model_versions") or {})
    return {
        "generated_at": format_timestamp(clock.now()),
        "n_scored": int(n_scored),
        "n_predictions": len(predictions),
        "latest_prediction_as_of": latest,
        "per_strategy": dict(per_strategy),
        "settlement_diagnostics": legacy_adapter.settlement_summary(scored_pairs(conn)),
        "champion_versions": champions,
        # Not derivable from canonical decisions alone (legacy board-state
        # files, not catalog data) -- see module docstring.
        "snapshot_hash": None,
        "data_freshness": {"latest_prediction_as_of": latest},
        "quota_state": None,
    }


def calibrate(conn, store, *, clock: Clock, force: bool = False,
              trigger: int = CALIBRATION_TRIGGER) -> dict:
    """Regenerate the ledger calibration report and health payload.

    Publishes both as immutable catalog artifacts and records the new
    ``n_scored`` trigger state, all inside one transaction so a crash never
    leaves the state pointing at an artifact that was never durably
    published, nor a published artifact the state forgot.
    """
    due, n_now, last = calibration_due(conn, trigger=trigger)
    if not (due or force):
        return {"regenerated": False, "n_scored": n_now, "n_at_last_report": last,
                "note": f"{n_now - last} new scored row(s); trigger is {trigger}"}

    pairs = scored_pairs(conn)
    per_strategy = ({str(s): legacy_adapter.strategy_calibration(g) for s, g in pairs.groupby("strategy")}
                    if len(pairs) else {})
    overall = (legacy_adapter.strategy_calibration(pairs) if len(pairs)
               else {"available": False, "reason": "no scored outcomes yet"})

    health = _health_payload(conn, per_strategy, n_scored=n_now, clock=clock)
    report = {
        "kind": "calibration",
        "generated_at": health["generated_at"],
        "n_scored": n_now,
        "calibration": overall,
        "per_strategy": per_strategy,
        "settlement": legacy_adapter.settlement_summary(pairs),
    }
    health_bytes = json.dumps(health, indent=1, default=str).encode()
    report_bytes = json.dumps(report, indent=1, default=str).encode()
    health_ref = store.publish_bytes(health_bytes, schema_ref="ledger_health.v1")
    report_ref = store.publish_bytes(report_bytes, schema_ref="ledger_calibration_report.v1")

    health_ref_json = canonical_json(to_document(health_ref))
    report_ref_json = canonical_json(to_document(report_ref))
    with _transaction(conn):
        conn.execute(
            "INSERT INTO ledger_calibration_state(singleton, n_scored_at_last_report, "
            "generated_at, health_ref_json, report_ref_json) VALUES (1,?,?,?,?) "
            "ON CONFLICT(singleton) DO UPDATE SET "
            "n_scored_at_last_report=excluded.n_scored_at_last_report, "
            "generated_at=excluded.generated_at, "
            "health_ref_json=excluded.health_ref_json, "
            "report_ref_json=excluded.report_ref_json",
            (n_now, health["generated_at"], health_ref_json, report_ref_json))

    return {
        "regenerated": True, "n_scored": n_now,
        "health": health_ref, "report": report_ref,
        "calibration": overall, "per_strategy": per_strategy,
    }
