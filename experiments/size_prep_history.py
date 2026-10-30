"""Workspace eligible-history and size-panel preparation over a pinned bounds record. Library-only: no production caller yet; the supported size_search prep entry point (#1614 slice 4) wires it."""
import time
from pathlib import Path

from engine.v2.data._bounded_catalog import probe_catalog
from engine.v2.data.repository import Repository
from engine.v2.foundation import canonical_json, content_hash
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.workspace.scope import validate_size_scope_request
from engine.v2.research.eligibility_policy import eligibility_policy
from experiments import eligible_history, size_panel
from experiments.size_prep_inputs import PROVENANCE_VERSION, _log, _read_regular

HISTORY_TABLES = ("earnings_events", "daily_market", "computed_moves", "computed_move_bounds", "price_history")
PANEL_VERSION = "size_prep_panel.v1"
# uncompressed/compressed allowance; the native reader bounds uncompressed row-group bytes by max_object_bytes
EXPANSION_FACTOR = 16


def _limits_refusal(table=None):
    details = {"reason": "limits_unavailable"}
    if table is not None:
        details["table"] = table
    return fail("PUBLICATION_REFUSED", "history limits cannot be derived from the bounds snapshot", details=details)


def _derive_history_limits(repository, snapshot, event_count, *, margin=2):
    """Private: callers must bound the catalog first (prepare_workspace_panel runs the bounded probe)."""
    if type(event_count) is not int or event_count < 1 or type(margin) is not int or margin < 1:
        raise _limits_refusal()
    rows = fragments = total_bytes = biggest = 0
    for table in HISTORY_TABLES:
        if table not in snapshot.table_versions:
            raise _limits_refusal(table)
        records = repository.fragment_records(snapshot, table)
        rows += sum(record.row_count for record in records)
        fragments += len(records)
        total_bytes += sum(record.object_ref.byte_size for record in records)
        biggest = max([biggest] + [record.object_ref.byte_size for record in records])
    limits = {"max_source_rows": rows * margin, "max_fragments": fragments * margin,
              "max_object_bytes": biggest * margin * EXPANSION_FACTOR, "max_total_bytes": total_bytes * margin,
              "max_output_rows": event_count * 3 * margin}
    if any(value <= 0 for value in limits.values()):
        raise _limits_refusal()
    return limits


def _record_invalid():
    return fail("PUBLICATION_REFUSED", "workspace bounds record is not the published provenance",
                details={"reason": "record_invalid"})


def _record_matches(workspace, record):
    if type(record) is not dict:
        return False
    try:
        record_id = record.get("record_id")
        body = {key: value for key, value in record.items() if key != "record_id"}
        if (type(record_id) is not str or "bounds_snapshot_id" not in record
                or record.get("schema_version") != PROVENANCE_VERSION
                or record.get("label") != "reconstructed"
                or content_hash(body) != record_id
                or record.get("source_snapshot_id") != workspace.snapshot_id):
            return False
        path = (Path(workspace.paths.root) / "size_prep"
                / ("provenance-" + record_id.split(":", 1)[1] + ".json"))
        expected = canonical_json(record).encode("utf-8")
        return _read_regular(path, len(expected)) == expected
    except (AttributeError, IndexError, KeyError, OSError, OpsError, TypeError, ValueError):
        return False


def prepare_workspace_panel(workspace, scope, record, *, event_ids, as_of_month, history_start, source_start,
                            metadata_limits, clock):
    ids_ok = (type(event_ids) is list and bool(event_ids)
              and all(type(key) is str and key for key in event_ids)
              and len(set(event_ids)) == len(event_ids))
    if (not ids_ok or type(scope) is not str or type(as_of_month) is not str
            or type(history_start) is not str or type(source_start) is not str
            or type(metadata_limits) is not dict):
        raise fail("INVALID_REQUEST", "invalid workspace panel request")
    validate_size_scope_request(workspace, scope)
    size_panel.validate_panel_inputs({"source_start": source_start, "metadata_limits": metadata_limits})
    if not _record_matches(workspace, record):
        raise _record_invalid()
    row = workspace.conn.execute(
        "SELECT snapshot_id FROM data_snapshot_heads WHERE scope = ?", (scope,)).fetchone()
    if row is None or row["snapshot_id"] != record["bounds_snapshot_id"]:
        raise fail("WORKSPACE_REFUSED", "the scope head is not the bounds snapshot",
                   details={"reason": "head_not_bounds", "scope": scope})
    probe_limits = {k: metadata_limits[k] for k in (
        "max_catalog_tables", "max_catalog_fragments", "max_catalog_bytes", "max_fragments", "max_partitions")}
    workspace.conn.execute("BEGIN")
    try:
        for table in HISTORY_TABLES:
            probe_catalog(workspace.conn, record["bounds_snapshot_id"], table, probe_limits)
    finally:
        if workspace.conn.in_transaction:
            workspace.conn.rollback()
    started = time.monotonic()
    repository = Repository(workspace.conn, workspace.store, pin_snapshots=True)
    bounds = repository.resolve(record["bounds_snapshot_id"])
    limits = _derive_history_limits(repository, bounds, len(event_ids))
    _log("workspace panel start events=" + str(len(event_ids)) + " limits=" + canonical_json(limits), started)
    policy = eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30)
    shared = dict(as_of_month=as_of_month, policy=policy, event_ids=list(event_ids), horizons=[1, 3, 5],
                  history_start=history_start, limits=limits)
    history = eligible_history.materialize(repository, bounds, **shared)
    panel = size_panel.materialize(repository, bounds, **shared,
                                   history_snapshot=history["feature_snapshot"],
                                   price_history_snapshot=bounds, source_start=source_start,
                                   metadata_limits=metadata_limits)
    fields = {"schema_version": PANEL_VERSION, "scope": scope, "record_id": record["record_id"],
              "bounds_snapshot_id": bounds.snapshot_id,
              "history_snapshot_id": history["feature_snapshot"].snapshot_id,
              "feature_snapshot_id": panel["feature_snapshot"].snapshot_id,
              "eligibility_plan_id": panel["context"]["eligibility_plan"]["plan_id"],
              "events_sha": content_hash(sorted(event_ids)), "limits": limits}
    doc = {**fields, "panel_id": content_hash(fields)}
    ref = workspace.record_read_pin("size-prep-panel:" + doc["panel_id"].split(":", 1)[1], doc, clock=clock)
    _log("workspace panel end", started)
    return {**doc, "read_pin_ref": ref}
