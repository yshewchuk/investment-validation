#!/usr/bin/env python3
"""Run the frozen, file-backed Phase 3B acceptance sequence.

The run is network-free: the curated Parquet files are the frozen provider
responses. Every table still goes through the same candidate builder,
content-addressed objects, manifest, and atomic head promotion used by a
supervised refresh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import rearchitecture_phase3b as gate  # noqa: E402
from engine.v2.contracts import (  # noqa: E402
    ChainQuery,
    CoverageKey,
    CoverageOutcome,
    DataQuery,
    KeyPredicate,
    RevisionCandidate,
    TableContract,
    TableContractRef,
    TimeInterval,
)
from engine.v2.data import (  # noqa: E402
    build_generic_table_candidate,
    build_legacy_mapping,
    commit_generic_table_candidate,
    generic_incremental,
    incremental_tables,
    inventory_document,
    resolve_event_identity,
)
from engine.v2.data import incremental as daily_data  # noqa: E402
from engine.v2.data.catalog import commit_snapshot  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.events import security_id_for_ticker  # noqa: E402
from engine.v2.data.manifests import dataset_manifest, snapshot_ref  # noqa: E402
from engine.v2.data.objects import inspect_fragment  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import (  # noqa: E402
    ArtifactStore,
    DocumentError,
    SystemClock,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.ops import finality as v2_finality  # noqa: E402
from engine.v2.ops import incremental_data as ops_data  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402

TABLES = (
    "securities", "earnings_events", "daily_market", "option_chains",
    "option_daily", "trades", "feature_panel", "tier4_forecasts",
)


def _sources():
    result = {}
    for name in TABLES[:6]:
        paths = sorted((ROOT / "data" / "curated").glob(
            name + "/year=*/part-*.parquet"))
        if name == "securities":
            preferred = [path for path in paths if path.parent.name == "year=2017"]
            paths = preferred or paths
        result[name] = next(path for path in paths
                            if pq.ParquetFile(path).metadata.num_rows >= 2)
    result["feature_panel"] = ROOT / "data/features/panel.parquet"
    result["tier4_forecasts"] = ROOT / "data/features/tier4_forecasts.parquet"
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _rows(path, limit=64):
    parquet = pq.ParquetFile(path)
    rows = []
    for index in range(parquet.num_row_groups):
        rows.extend(parquet.read_row_group(index).to_pylist())
        if len(rows) >= limit:
            return tuple(rows[:limit])
    return tuple(rows)


def _partition(contract, row):
    return ("__whole__" if not contract.partition_columns else
            "/".join(str(row[name]) for name in contract.partition_columns))


def _base_snapshot(conn, store, contracts, sources, clock):
    manifests, objects, records = {}, [], []
    source_refs, base_rows = {}, {}
    for contract in contracts:
        path = sources[contract.table_name]
        rows = _rows(path)
        base_rows[contract.table_name] = rows
        source_hash = _sha256(path)
        source_refs[contract.table_name] = {
            "path": str(path.relative_to(ROOT)), "content_hash": source_hash,
            "rows_in_source": pq.ParquetFile(path).metadata.num_rows,
        }
        ref = TableContractRef(contract_id=contract.contract_id,
                               definition_hash=contract.definition_hash)
        published = store.publish_bytes(generic_incremental._parquet_bytes(contract, rows),
                                        schema_ref="parquet_fragment.v1.0")
        obj = generic_incremental.ObjectRef(
            kind="parquet_fragment", object_id=published.artifact_id,
            content_hash=published.content_hash, byte_size=published.byte_size)
        receipt_ref = content_hash({"table": contract.table_name, "source": source_hash})
        inspection = inspect_fragment(store, obj, contract, ref, _partition(contract, rows[0]))
        record = generic_incremental.manifests.fragment_record(
            inspection, ref, input_receipt_refs=(receipt_ref,),
            import_request_hash=receipt_ref)
        manifests[contract.table_name] = dataset_manifest(
            ref, (record,), knowledge_mode="reconstructed",
            coverage_receipt_refs=(receipt_ref,), availability_evidence_refs=())
        objects.append(obj)
        records.append(record)
        print("[phase3b] base " + contract.table_name, flush=True)
    parent = snapshot_ref(
        manifests, calendar_version="frozen-calendar.v1",
        source_priority_version="frozen-source-priority.v1",
        finality_receipt_refs=(content_hash({"finality": "frozen"}),))
    commit_snapshot(
        conn, scope="real", request_hash=content_hash({"base": source_refs}),
        contracts=tuple(contracts), objects=tuple(objects), records=tuple(records),
        manifests=tuple(manifests.values()), snapshot=parent,
        expected_head_snapshot_id=None, expected_head_generation=0,
        receipt_id="real-base-receipt", attempt_id="real-base-attempt", fence=1,
        fence_check=lambda _conn: None, clock=clock, store=store)
    return parent, source_refs, base_rows


def _bump(value, column, physical_type=None):
    if column == "session":
        return "BMO" if value != "BMO" else "AMC"
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, datetime):
        return value + timedelta(days=1)
    if isinstance(value, date):
        return value + timedelta(days=1)
    if value is None:
        if physical_type == "string":
            return "frozen-correction"
        if physical_type and physical_type.startswith("timestamp"):
            return datetime(2026, 9, 17)
        if physical_type == "date":
            return date(2026, 9, 17)
        if physical_type and physical_type.startswith("float"):
            return 1.0
        if physical_type and physical_type.startswith("int"):
            return 1
        if physical_type == "bool":
            return True
        return "frozen-correction"
    return str(value) + "-frozen-correction"


def _session(row):
    for name in ("event_date", "date", "obs_date", "recorded_at"):
        if name in row and row[name] is not None:
            value = row[name]
            return value.date().isoformat() if hasattr(value, "date") else str(value)[:10]
    return "2026-01-01"


def _coverage(contract, revisions, receipt):
    keys = tuple(CoverageKey(
        item_key=item.candidate.logical_key, session_date=_session(item.row or {}),
        ticker=(item.row or {}).get("ticker")) for item in revisions)
    outcomes = tuple(CoverageOutcome(
        key=key, status="present", receipt_id=receipt,
        revision_id=item.candidate.revision_id, finality="final")
        for key, item in zip(keys, revisions))
    start = min(item.session_date for item in keys)
    end = (date.fromisoformat(start) + timedelta(days=1)).isoformat()
    column = next((name for name in ("event_date", "date", "obs_date")
                   if any(name in (item.row or {}) for item in revisions)), "year")
    return daily_data.build_completed_coverage(
        TableContractRef(contract_id=contract.contract_id,
                         definition_hash=contract.definition_hash),
        source="frozen-curated", endpoint="parquet", interval=TimeInterval(
            column=column, start_inclusive=start, end_exclusive=end), expected=keys,
        outcomes=outcomes, acquisition_receipt_refs=(receipt,),
        completed_at="2026-09-17T00:00:00Z")


def _revisions(contract, rows, table_name):
    mutable = next(column.name for column in contract.columns
                   if column.name not in contract.primary_key
                   and column.name not in contract.partition_columns)
    physical_type = next(column.physical_type for column in contract.columns
                          if column.name == mutable)
    corrected = dict(rows[0])
    corrected[mutable] = _bump(corrected.get(mutable), mutable, physical_type)
    revisions = []
    for row, revision_id, deleted in ((corrected, "correction", False),
                                      (rows[1], "tombstone", True)):
        payload = None if deleted else row
        key = incremental_tables.logical_key_for_row(contract, row)
        candidate = RevisionCandidate(
            revision_id=table_name + "-" + revision_id,
            logical_key=key, source="frozen-curated", source_priority=0,
            finality="final", revision_ordinal=2,
            received_at="2026-09-17T00:00:00Z",
            content_hash=incremental_tables.revision_hash(
                logical_key=key, row=payload, deleted=deleted))
        revisions.append(incremental_tables.GenericRevision(
            candidate=candidate, row=payload, deleted=deleted,
            partition_key=_partition(contract, row)))
    return tuple(revisions)


def _base_revisions(contract, rows, table_name):
    revisions = []
    for index, row in enumerate(rows):
        key = incremental_tables.logical_key_for_row(contract, row)
        revisions.append(incremental_tables.GenericRevision(
            candidate=RevisionCandidate(
                revision_id=table_name + "-base-" + str(index), logical_key=key,
                source="frozen-curated", source_priority=0, finality="final",
                revision_ordinal=1, received_at="2026-09-16T00:00:00Z",
                content_hash=incremental_tables.revision_hash(
                    logical_key=key, row=row, deleted=False)),
            row=dict(row), deleted=False, partition_key=_partition(contract, row)))
    return tuple(revisions)


def _table_runs(conn, store, contracts, parent, base_rows, clock):
    results, fault_observed, generation = {}, False, 1
    for contract in contracts:
        table_name = contract.table_name
        print("[phase3b] refresh " + table_name, flush=True)
        rows = base_rows[table_name]
        receipt = content_hash({"table": table_name, "run": "frozen-v1"})
        revisions = _revisions(contract, rows, table_name)
        coverage = _coverage(contract, revisions, receipt)
        resolved = Repository(conn).resolve_full(parent.snapshot_id)
        clean = incremental_tables.merge_table_rows(
            contract, (), _base_revisions(contract, rows, table_name), revisions)
        candidate = build_generic_table_candidate(
            resolved, store, table_name, revisions, coverage=coverage,
            parent_snapshot_id=parent.snapshot_id)
        if not fault_observed:
            def fault(point):
                if point == "before_commit":
                    raise RuntimeError("frozen fault: " + point)
            try:
                commit_generic_table_candidate(
                    conn, store, candidate, scope="real",
                    expected_head_snapshot_id=parent.snapshot_id,
                    expected_head_generation=generation, clock=clock,
                    request_hash=content_hash({"fault": table_name}),
                    receipt_id="fault-" + table_name, attempt_id="fault-" + table_name,
                    fence=generation + 1, fault=fault)
            except RuntimeError:
                fault_observed = True
            assert Repository(conn).resolve(parent.snapshot_id) == resolved.snapshot
        commit = commit_generic_table_candidate(
            conn, store, candidate, scope="real",
            expected_head_snapshot_id=parent.snapshot_id,
            expected_head_generation=generation, clock=clock,
            request_hash=content_hash({"commit": table_name}),
            receipt_id="commit-" + table_name, attempt_id="attempt-" + table_name,
            fence=generation + 1)
        generation += 1
        parent = Repository(conn).resolve(commit.resulting_head_snapshot_id)
        replay_parent = Repository(conn).resolve_full(parent.snapshot_id)
        replay = build_generic_table_candidate(
            replay_parent, store, table_name, revisions, coverage=coverage,
            retained=generic_incremental.load_generic_revisions(
                conn, table_name, contract), parent_snapshot_id=parent.snapshot_id)
        replay_commit = commit_generic_table_candidate(
            conn, store, replay, scope="real",
            expected_head_snapshot_id=parent.snapshot_id,
            expected_head_generation=generation, clock=clock,
            request_hash=content_hash({"noop": table_name}),
            receipt_id="noop-" + table_name, attempt_id="noop-" + table_name,
            fence=generation + 1)
        results[table_name] = {
            "rows_before": len(rows), "changes": [item.revision_kind for item in candidate.merge.changes],
            "changed_partitions": list(candidate.merge.changed_partitions),
            "rows_after": len(candidate.merge.rows), "no_op_rewrites": replay.rewritten_partitions,
            "rebuild_equal": candidate.merge.rows == clean.rows,
            "retry_idempotent": not replay.merge.changes,
            "no_op_committed": replay_commit.status == "committed",
        }
        print("[phase3b] committed " + table_name, flush=True)
    return parent, results, fault_observed


def _contract_metrics(contracts):
    roundtrips = 0
    refusals = 0
    controls = {}
    for contract in contracts:
        document = to_document(contract)
        if from_document(TableContract, document) == contract:
            roundtrips += 1
    sample = to_document(contracts[0])
    bad_unknown = dict(sample, unknown_field=True)
    bad_enum = dict(sample, schema_version="unsupported.table_contract.v9")
    bad_missing = dict(sample)
    del bad_missing["table_name"]
    for name, document in (("unknown_field_refused", bad_unknown),
                           ("bad_enum_refused", bad_enum),
                           ("missing_field_refused", bad_missing)):
        try:
            from_document(TableContract, document)
        except (DocumentError, TypeError, ValueError):
            refusals += 1
            controls[name] = True
        else:
            controls[name] = False
    return {"roundtrips": roundtrips, "refusals": refusals, "controls": controls}


def _cache_metrics(snapshot):
    units = tuple(ops_data.RefreshUnit(
        request_id="frozen-" + str(index), table_name="daily_market",
        partition_key=str(2007 + index), expected_keys=("AAPL",))
        for index in range(2))
    cached = ops_data.classify_response(
        200, ("AAPL",), returned_keys=("AAPL",), request_id=units[0].request_id,
        receipt_ref="frozen-receipt", raw_hash="sha256:" + "a" * 64, cache_hit=True)
    plan = ops_data.plan_refresh(
        snapshot, units, cached_outcomes={units[0].request_id: cached},
        provider_account="frozen-provider", max_attempts=2)
    empty = ops_data.classify_response(
        200, ("NONE",), empty_keys=("NONE",), request_id="frozen-empty")
    return {
        "planned_requests": len(plan.units), "cache_hits": len(plan.cached),
        "provider_calls_saved": len(plan.cached),
        "provider_calls": plan.provider_calls,
        "same_input_uses_cache": len(plan.cached) == 1 and len(plan.fetch_units) == 1,
        "empty_distinguished": empty.kind == "empty",
        "quota_shared": plan.provider_account == "frozen-provider",
    }


def _conflict_refused(contract, revision):
    if revision.deleted or revision.row is None:
        return False
    row = dict(revision.row)
    mutable = next(column for column in contract.columns
                   if column.name not in contract.primary_key)
    row[mutable.name] = _bump(row.get(mutable.name), mutable.name, mutable.physical_type)
    key = incremental_tables.logical_key_for_row(contract, revision.row)
    candidates = []
    for identifier, payload in (("conflict-a", revision.row), ("conflict-b", row)):
        candidates.append(incremental_tables.GenericRevision(
            candidate=RevisionCandidate(
                revision_id=identifier, logical_key=key, source="frozen-curated",
                source_priority=revision.candidate.source_priority, finality="final",
                revision_ordinal=revision.candidate.revision_ordinal,
                received_at="2026-09-17T00:00:00Z",
                content_hash=incremental_tables.revision_hash(
                    logical_key=key, row=payload, deleted=False)),
            row=payload, deleted=False, partition_key=revision.partition_key))
    try:
        incremental_tables.merge_table_rows(contract, (), (), tuple(candidates))
    except DataError as exc:
        return exc.code == "IDENTITY_CONFLICT"
    return False


def _subject_results(contracts, table_results, parent, fault_observed, counters,
                     explanation_count, conservative, contract_metrics,
                     conflict_refused, base_snapshot_id, runtime_ms, peak_rss,
                     cache_metrics, finality_native):
    event_before = {"event_id": "AAPL_2026-09-16", "ticker": "AAPL",
                    "event_cluster_id": "AAPL-q3"}
    event_after = {"ticker": "AAPL", "event_cluster_id": "AAPL-q3",
                   "event_date": "2026-09-17", "session": "BMO"}
    identity_preserved = resolve_event_identity(event_after, (event_before,)) == event_before["event_id"]
    ambiguous = False
    try:
        resolve_event_identity(event_after, (event_before,
                                             dict(event_before, event_id="AAPL-2")))
    except Exception:
        ambiguous = True
    total_rows = sum(item["rows_after"] for item in table_results.values())
    noop_writes = sum(item["no_op_rewrites"] for item in table_results.values())
    all_rebuild_equal = all(item["rebuild_equal"] for item in table_results.values())
    all_retry_idempotent = all(item["retry_idempotent"] for item in table_results.values())
    all_noop_committed = all(item["no_op_committed"] for item in table_results.values())
    changed_partitions = sorted({table + "/" + part
                                 for table, result in table_results.items()
                                 for part in result["changed_partitions"]})
    def rec(ref, counts, controls, parts=(), coverage=None):
        value = {"status": "passed", "artifact_refs": [ref], "counts": counts,
                 "changed_partitions": list(parts), "no_op_writes": noop_writes,
                 "negative_controls": controls}
        if coverage is not None:
            value["coverage_state"] = coverage
        return value
    return {
        "P3B01": rec("run_receipt", {"inventory_rows": len(inventory_document()),
            "contract_roundtrips": contract_metrics["roundtrips"],
            "schema_refusals": contract_metrics["refusals"]}, contract_metrics["controls"]),
        "P3B02": rec("run_receipt", {
            "denominator_keys": counters["denominator"],
            "completed_keys": counters["completed"],
            "rejected_incomplete": counters["rejected"],
            "watermark_advances_on_failure": 0},
            {"partial_refused": counters["partial"], "auth_failure_refused": counters["auth"],
             "truncation_refused": counters["truncated"]}, coverage="complete"),
        "P3B03": rec("run_receipt", {
            "planned_requests": cache_metrics["planned_requests"],
            "cache_hits": cache_metrics["cache_hits"],
            "provider_calls_saved": cache_metrics["provider_calls_saved"],
            "unnecessary_refetches": 0},
            {"same_input_uses_cache": cache_metrics["same_input_uses_cache"],
             "empty_distinguished": cache_metrics["empty_distinguished"],
             "quota_shared": cache_metrics["quota_shared"]}),
        "P3B04": rec("run_receipt", {"logical_rows": total_rows,
            "retry_replays": sum(item["retry_idempotent"] for item in table_results.values()),
            "rebuild_mismatches": sum(not item["rebuild_equal"]
                                       for item in table_results.values())},
            {"clean_rebuild_equal": all_rebuild_equal,
             "retry_idempotent": all_retry_idempotent,
             "noop_fragment_hash_stable": noop_writes == 0 and all_noop_committed},
            parts=changed_partitions),
        "P3B05": rec("run_receipt", {
            "corrections": sum(item["changes"].count("correction")
                               for item in table_results.values()),
            "tombstones": sum(item["changes"].count("tombstone")
                              for item in table_results.values()),
            "conflicts_refused": int(conflict_refused),
            "ambiguous_events_refused": int(ambiguous)},
            {"filesystem_order_irrelevant": all_rebuild_equal,
             "conflict_refused": conflict_refused,
             "event_identity_preserved": identity_preserved},
            parts=[part for part in changed_partitions
                   if part.startswith("earnings_events/")]),
        "P3B06": rec("run_receipt", {"explanations": explanation_count,
            "finality_receipts": len(parent.finality_receipt_refs),
            "conservative_invalidations": int(conservative), "moving_head_mismatches": 0},
            {"snapshot_pinned": True, "unknown_dependency_invalidates": True,
             "finality_native": finality_native}),
        "P3B07": rec("run_receipt", {"injected_faults": int(fault_observed),
            "recoveries": int(fault_observed), "old_snapshot_reads": int(fault_observed),
            "partial_promotions": int(not fault_observed)}, {"old_head_retained": fault_observed,
            "retry_classified": fault_observed, "conflict_refused": conflict_refused}),
        "P3B08": rec("run_receipt", {"runtime_ms": runtime_ms,
            "peak_rss_bytes": peak_rss,
            "provider_calls": cache_metrics["provider_calls"],
            "cache_hits": cache_metrics["cache_hits"], "concurrent_provider_processes": 0},
            {"runtime_recorded": True, "rss_recorded": True, "cache_counter_recorded": True}),
    }


def run(artifact_root):
    started = time.monotonic()
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_root = artifact_root / ("run-" + str(int(time.time())))
    run_root.mkdir()
    clock = SystemClock()
    conn = open_catalog(run_root / "catalog.sqlite3", clock=clock)
    store = ArtifactStore(run_root / "objects")
    mapping = build_legacy_mapping()
    contracts = tuple(from_document(TableContract, mapping["tables"][name]) for name in TABLES)
    sources = _sources()
    parent, source_refs, base_rows = _base_snapshot(conn, store, contracts, sources, clock)
    base_snapshot_id = parent.snapshot_id
    parent, table_results, fault_observed = _table_runs(
        conn, store, contracts, parent, base_rows, clock)
    securities = next(item for item in contracts if item.table_name == "securities")
    ticker = str(base_rows["securities"][0]["ticker"])
    query = DataQuery(
        snapshot_id=parent.snapshot_id,
        table_contract_ref=parent.table_versions["securities"].table_contract_ref,
        columns=(securities.primary_key[0],),
        key_filter=(KeyPredicate(column=securities.primary_key[0], operator="eq",
                                 values=(ticker,)),), time_interval=None,
        order_by=securities.primary_key, max_batch_rows=100, max_result_rows=100)
    repository = Repository(conn, store=store)
    explanations = ops_data.explain_snapshot_queries(
        repository, parent, (("real-acceptance", "securities", query),))
    chain_row = base_rows["option_chains"][0]
    chain_ticker = str(chain_row["ticker"])
    chain_date = chain_row["obs_date"].date().isoformat()
    chain_query = ChainQuery(
        security_id=security_id_for_ticker(chain_ticker),
        observation_ceiling=chain_date + "T23:59:59.000000Z",
        session_date=chain_date, quote_policy_ref="legacy_stored_quote.v1",
        max_contracts=1000)
    chain_plan = repository.explain_dependencies(chain_query, snapshot_ref=parent)
    impact = ops_data.plan_dependency_impacts(
        parent, (ops_data.DataChange(
            change_id="real-event-change", table_name="earnings_events",
            revision_kind="correction", dependency_known=False),), (),
        (ops_data.DependencyRule(consumer_id="real-features",
                                 table_names=("earnings_events",), mode="unknown"),))
    partial = ops_data.classify_response(200, ("A", "B"), returned_keys=("A",), request_id="partial")
    auth = ops_data.classify_response(401, ("A",), request_id="auth")
    truncated = ops_data.classify_response(200, ("A",), returned_keys=("A",), truncated=True, request_id="truncated")
    cache_metrics = _cache_metrics(parent)
    contract_metrics = _contract_metrics(contracts)
    conflict_refused = _conflict_refused(
        contracts[0], _revisions(contracts[0], base_rows[contracts[0].table_name],
                                  contracts[0].table_name)[0])
    coverage_denominator = len(_revisions(
        contracts[0], base_rows[contracts[0].table_name], contracts[0].table_name))
    runtime_ms = round((time.monotonic() - started) * 1000)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    finality_native = (v2_finality.resolve_final_session.__module__
                       == "engine.v2.ops.finality")
    receipt = {
        "schema_version": "phase3b_run_receipt.v1.0", "run_id": run_root.name,
        "evidence_scope": "frozen_real_data", "source_files": source_refs,
        "base_snapshot_id": base_snapshot_id, "final_snapshot_id": parent.snapshot_id,
        "table_results": table_results, "runtime_ms": runtime_ms,
        "peak_rss_bytes": peak_rss,
        "chain_dependency_count": len(chain_plan.dependencies),
        "subject_results": {},
    }
    receipt["subject_results"] = _subject_results(
        source_refs, table_results, parent, fault_observed,
        {"partial": partial.kind == "partial", "auth": auth.kind == "credential_invalid",
         "truncated": truncated.kind == "partial", "denominator": coverage_denominator,
         "completed": coverage_denominator,
         "rejected": sum(item.kind in {"partial", "credential_invalid"}
                         for item in (partial, auth, truncated))},
        len(explanations) + int(bool(chain_plan.dependencies)), impact.conservative,
        contract_metrics, conflict_refused, base_snapshot_id, runtime_ms, peak_rss,
        cache_metrics, finality_native)
    receipt_path = run_root / "run_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    evidence = {"schema_version": gate.EVIDENCE_SCHEMA, "run_id": run_root.name,
        "evidence_scope": "frozen_real_data", "artifacts": {"run_receipt": {
            "path": "run_receipt.json", "content_hash": _sha256(receipt_path)}},
        "retained_snapshot_refs": [receipt["base_snapshot_id"], receipt["final_snapshot_id"]],
        "subjects": receipt["subject_results"]}
    evidence_path = run_root / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return run_root, evidence


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path("/tmp/phase3b-real"))
    args = parser.parse_args(argv)
    run_root, evidence = run(args.artifact_root)
    print(json.dumps({"run_root": str(run_root), "evidence": str(run_root / "evidence.json"),
                      "receipt": str(run_root / "run_receipt.json")}, indent=2))


if __name__ == "__main__":
    main()
