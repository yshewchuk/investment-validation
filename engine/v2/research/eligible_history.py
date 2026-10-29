"""Internal history implementation; experiments.eligible_history owns public admission.

Private build/read helpers trust orchestration's installed producer identity; they
are not independent admitted APIs or a security boundary against Python callers.
"""
from datetime import date, timedelta

import pyarrow as pa
import pyarrow.parquet as pq

from engine.v2.contracts.data import DataQuery, KeyPredicate, ObjectRef, TableContractRef
from engine.v2.data import catalog, manifests, objects, query
from engine.v2.data.computed_move_bounds import CONTRACT as BOUNDS_CONTRACT
from engine.v2.data.computed_move_bounds import TABLE as BOUNDS_TABLE
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT, COMPUTED_MOVES_V5_CONTRACT
from engine.v2.data.errors import fail
from engine.v2.foundation import SystemClock, canonical_json, content_hash, to_document
from engine.v2.research import eligibility_policy, event_history
from engine.v2.research._snapshot import artifact_store, catalog_connection
from engine.v2.research.eligible_history_table import (
    CONTRACT,
    METADATA,
    TABLE,
    admit_fields,
    recipe,
)
from engine.v2.research.experiment_population import load_population
from engine.v2.research.magnitude_features import _feature_recipe, calendar_input
from engine.v2.research.magnitude_inputs import forecast_cutoff
from engine.v2.research.progress import Progress, stage
from engine.v2.research.snapshot_inputs import admit_snapshot

_LIMITS = {"max_source_rows", "max_fragments", "max_object_bytes", "max_total_bytes", "max_output_rows"}


def _limits(limits):
    if (type(limits) is not dict or set(limits) != _LIMITS
            or any(type(n) is not int or n <= 0 for n in limits.values())):
        raise fail("CONTRACT_MISMATCH", "explicit positive history limits required")


def admit_source(repository, snapshot, limits):
    """Bound decoded metadata/physical objects after native snapshot resolution."""
    _limits(limits)
    admit_snapshot(repository, snapshot)
    all_records = []
    for table in ("earnings_events", "daily_market", "computed_moves", BOUNDS_TABLE):
        if table not in snapshot.table_versions:
            raise fail("CONTRACT_MISMATCH", "history source requires native source bounds")
        all_records.extend(repository.fragment_records(snapshot, table))
    if (len(all_records) > limits["max_fragments"]
            or sum(r.row_count for r in all_records) > limits["max_source_rows"]
            or sum(r.object_ref.byte_size for r in all_records) > limits["max_total_bytes"]
            or any(r.object_ref.byte_size > limits["max_object_bytes"] for r in all_records)):
        raise fail("RESULT_LIMIT_EXCEEDED", "history source exceeds explicit bounds")
    if repository.table_contract(snapshot, BOUNDS_TABLE) != BOUNDS_CONTRACT:
        raise fail("CONTRACT_MISMATCH", "unsupported source-bound contract")


def _anchors(repository, snapshot, limits, required_tickers=()):
    records = repository.fragment_records(snapshot, BOUNDS_TABLE)
    if set(required_tickers) - {r.partition_key for r in records}:
        raise fail("CONTRACT_MISMATCH", "required ticker is missing native source bounds")
    count = sum(r.row_count for r in records)
    if not count:
        return {}
    request = DataQuery(snapshot_id=snapshot.snapshot_id,
        table_contract_ref=snapshot.table_versions[BOUNDS_TABLE].table_contract_ref,
        columns=tuple(c.name for c in BOUNDS_CONTRACT.columns), order_by=BOUNDS_CONTRACT.primary_key,
        key_filter=(KeyPredicate(column="ticker", operator="in",
            values=tuple(sorted({r.partition_key for r in records}))),),
        max_batch_rows=min(count, BOUNDS_CONTRACT.maximum_batch_rows), max_result_rows=count)
    result = {}
    for batch in repository.scan(request, table_name=BOUNDS_TABLE, max_object_bytes=limits["max_object_bytes"]):
        for row in batch.to_pylist():
            key = row["ticker"], row["event_date"]
            if key in result:
                raise fail("CONTRACT_MISMATCH", "duplicate source bounds")
            result[key] = row
    if len(result) != count:
        raise fail("CONTRACT_MISMATCH", "incomplete source bounds")
    return result


def _bound(event, anchors, records):
    """Authenticate metadata to exact singleton object without opening that object."""
    key = event.ticker, event.event_date.date().isoformat()
    row, record = anchors.get(key), records.get(key)
    if row is None:
        raise fail("CONTRACT_MISMATCH", "required event is missing native source bounds")
    if (record is None or record.row_count != 1
            or record.primary_key_min != key or record.primary_key_max != key
            or record.partition_key != key[0] or record.time_min != key[1] or record.time_max != key[1]
            or row["outcome_object_id"] != record.object_ref.object_id
            or row["outcome_content_hash"] != record.object_ref.content_hash):
        raise fail("HOLDOUT_ACCESS_DENIED", "source bounds do not bind the exact isolated outcome")
    return _bound_dates(row, event, key)


def _bound_dates(row, event, key):
    """Null means unavailable; non-null dates must be the valid source bracket."""
    lo, hi = row["pre_close_date"], row["post_close_date"]
    if lo is None and hi is None:
        return None
    try:
        left, right = date.fromisoformat(lo), date.fromisoformat(hi)
        if (left.isoformat() != lo or right.isoformat() != hi or not lo <= key[1] <= hi
                or not 0 < (right - left).days <= 5 or row["session"] != event.session):
            raise ValueError
    except (TypeError, ValueError):
        raise fail("CONTRACT_MISMATCH", "source label bounds are invalid") from None
    return row


def _request(repository, snapshot, policy, event_ids, horizons, limits):
    try:
        expected = eligibility_policy.eligibility_policy(
            version=policy["schema_version"], embargo_days=policy["embargo_days"])
        if policy != expected:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise fail("CONTRACT_MISMATCH", "invalid history eligibility policy") from None
    admit_source(repository, snapshot, limits)
    if (type(event_ids) is not list or not event_ids
            or any(type(key) is not str for key in event_ids) or len(set(event_ids)) != len(event_ids)
            or type(horizons) is not list or not horizons
            or any(type(h) is not int or h not in (1, 3, 5) for h in horizons)
            or horizons != [1, 3, 5]):
        raise fail("CONTRACT_MISMATCH", "explicit unique targets and ordered native horizons required")
    if len(event_ids) * len(horizons) > limits["max_output_rows"]:
        raise fail("RESULT_LIMIT_EXCEEDED", "history output exceeds explicit bounds")


def _forecasts(event, horizons, calendar, clocks):
    """Reuse identical validated clocks within one preparation's fixed calendar."""
    forecasts = []
    for horizon in horizons:
        key = (event.event_date.date().isoformat(), event.session, horizon)
        if key not in clocks:
            clocks[key] = forecast_cutoff(event.event_date, event.session, horizon, calendar)
        forecasts.append(dict(clocks[key]))
    return forecasts


def _prepare(repository, snapshot, *, as_of_month, policy, event_ids, horizons, history_start, limits,
             producer):
    _request(repository, snapshot, policy, event_ids, horizons, limits)
    targets = load_population(repository, snapshot, as_of_month=as_of_month, event_ids=event_ids)
    population = load_population(repository, snapshot, as_of_month=as_of_month)
    calendar, calendar_source = calendar_input(repository, snapshot)
    clocks = {}
    forecasts = {event.event_id: _forecasts(event, horizons, calendar, clocks)
        for event in targets.itertuples(index=False)}
    start, _ = event_history._window(history_start,
        min(c["source_session_ceiling"] for event_clocks in forecasts.values() for c in event_clocks))
    ceilings = {}
    for target in targets.itertuples(index=False):
        ceilings[target.ticker] = max(ceilings.get(target.ticker, ""),
            *(c["source_session_ceiling"] for c in forecasts[target.event_id]))
    days = population.event_date.dt.strftime("%Y-%m-%d")
    selected = population[(population.ticker.isin(targets.ticker)) & (days >= start.isoformat())
        & ((days < population.ticker.map(ceilings)) | population.event_id.isin(event_ids))]
    groups, unavailable, qualified = _groups(repository, snapshot, selected, forecasts,
        event_ids, horizons, calendar, clocks, limits)
    metadata = {"schema_version": "event_group_bounds.v1", "source_snapshot": to_document(snapshot),
        "holdout_as_of_month": as_of_month, "policy_id": policy["policy_id"], "horizons": horizons,
        "groups": groups}
    metadata["metadata_id"] = content_hash(metadata)
    eligibility = eligibility_policy.plan_eligibility(repository, snapshot, as_of_month=as_of_month,
        policy=policy, metadata=metadata, evaluation_event_ids=[])
    # Full canonical metadata is reread by plan_eligibility; unavailable contributors
    # above never remove a held event's window or confer a new read capability.
    retained = set(eligibility["retained_event_ids"])
    if set(event_ids) - retained:
        raise fail("HOLDOUT_ACCESS_DENIED", "development targets are excluded by grouped eligibility")
    candidates = selected[selected.event_id.isin(retained)].sort_values(["event_date", "event_id"])
    candidates = candidates[candidates.event_id.map(lambda key:
        qualified[key]["post_close_date"] < ceilings[qualified[key]["ticker"]]
        and qualified[key]["event_date"] < ceilings[qualified[key]["ticker"]])]
    contract = repository.table_contract(snapshot, "computed_moves")
    if contract not in (COMPUTED_MOVES_CONTRACT, COMPUTED_MOVES_V5_CONTRACT):
        raise fail("CONTRACT_MISMATCH", "history requires native computed moves")
    plans, reader = event_history.plan_with_reader(repository, snapshot, candidates, contract)
    target_rows = [{"event_id": e.event_id, "ticker": e.ticker,
        "event_date": e.event_date.date().isoformat(), "forecasts": forecasts[e.event_id]}
        for e in targets.sort_values("event_id").itertuples(index=False)]
    identity = {"schema_version": "eligible_history_context.v1", "source_snapshot": to_document(snapshot),
        "eligibility": eligibility, "producer": producer, "targets": target_rows,
        "history_start": start.isoformat(),
        "calendar_source": calendar_source, "feature_recipe": recipe(), "limits": dict(limits),
        "unavailable_contributors": sorted(unavailable),
        "contributors": [{"event_id": e.event_id, **qualified[e.event_id]} for e, _, _ in plans],
        "availability": {"basis": "reconstructed_date_bounds", "source_publication": "unverified",
                         "event_schedule_as_known": "unverified"}}
    context = {**identity, "context_id": content_hash(identity)}
    return context, targets.set_index("event_id", drop=False), population, plans, reader


def _groups(repository, snapshot, selected, forecasts, event_ids, horizons, calendar, clocks, limits):
    """Source-derived label windows; missing evidence differs from null labels."""
    anchors = _anchors(repository, snapshot, limits, required_tickers=selected.ticker.unique())
    records = {}
    for record in repository.fragment_records(snapshot, "computed_moves"):
        key = record.primary_key_min
        if key in records:
            raise fail("HOLDOUT_ACCESS_DENIED", "ambiguous outcome object identity")
        records[key] = record
    groups, unavailable, qualified = [], [], {}
    for event in selected.itertuples(index=False):
        bound = _bound(event, anchors, records)
        if bound is None:
            unavailable.append(event.event_id)
            if event.event_id in event_ids:
                raise fail("CONTRACT_MISMATCH", "target label bounds are unavailable")
            continue
        qualified[event.event_id] = bound
        event_clocks = forecasts.get(event.event_id) or _forecasts(event, horizons, calendar, clocks)
        groups.append({"event_id": event.event_id, "ticker": event.ticker,
            "event_date": event.event_date.date().isoformat(), "label_start": bound["pre_close_date"],
            "label_end": bound["post_close_date"], "forecasts": [{"horizon_sessions": c["horizon_sessions"],
                "decision_date": c["source_session_ceiling"], "source_session_ceiling": c["source_session_ceiling"]}
                for c in event_clocks]})
    return groups, unavailable, qualified


def _slot(context):
    return "eligible-history-" + context["context_id"].removeprefix("sha256:")


def _existing(repository, context):
    slot = _slot(context)
    row = catalog_connection(repository).execute(
        """SELECT CASE WHEN scope=? AND source_manifest_hash=? AND attempt_id=?
        AND fence=1 AND status='committed' AND length(result_snapshot_id)=37
        AND substr(result_snapshot_id,1,5)='snap_'
        AND substr(result_snapshot_id,6) NOT GLOB '*[^0-9a-f]*'
        THEN result_snapshot_id END AS snapshot_id FROM data_import_receipts WHERE receipt_id=?""",
        (slot, context["context_id"], slot, slot)).fetchone()
    if row is None:
        return None
    if row["snapshot_id"] is None:
        raise fail("IDENTITY_CONFLICT", "eligible history receipt differs")
    return repository.resolve(row["snapshot_id"])


def _rows(snapshot, prepared):
    context, targets, population, plans, reader = prepared
    bounds = {row["event_id"]: row for row in context["contributors"]}
    by_ticker = {}
    for plan in plans:
        by_ticker.setdefault(plan[0].ticker, []).append(plan)

    def checked(event, request):
        rows = reader(event, request)
        expected = (date.fromisoformat(bounds[event.event_id]["post_close_date"]) + timedelta(days=1)).isoformat()
        if len(rows) != 1 or rows[0]["available_as_of_date"] != expected or rows[0]["skipped"] is not False:
            raise fail("CONTRACT_MISMATCH", "outcome availability differs from qualified source bounds")
        return rows

    meter = Progress("eligible history targets", len(context["targets"]))
    for target in context["targets"]:
        rows = []
        for cutoff in target["forecasts"]:
            start, ceiling = event_history._window(context["history_start"], cutoff["source_session_ceiling"])
            prior = [plan for plan in by_ticker.get(target["ticker"], ())
                if plan[0].event_date.date() < ceiling
                and bounds[plan[0].event_id]["post_close_date"] < ceiling.isoformat()]
            history = event_history._assemble_history(snapshot, targets.loc[target["event_id"]],
                population, prior, start=start, decision=ceiling, purpose="selection", read_rows=checked)
            _feature_recipe(history["features"])
            rows.append({"event_id": target["event_id"], "context_id": context["context_id"],
                "horizon_sessions": cutoff["horizon_sessions"], "event_date": target["event_date"],
                "cutoff_id": cutoff["cutoff_id"], "source_session_ceiling": cutoff["source_session_ceiling"],
                "history_status": history["history_status"], "history_json": canonical_json(history),
                **history["features"]})
        meter.tick()
        yield rows


def _materialize(repository, snapshot, **request):
    """Reconstruct admission, then publish/reuse one immutable bounded context."""
    with stage("eligible history admission", report_refusals=True):
        prepared = _prepare(repository, snapshot, **request)
    context = prepared[0]
    existing = _existing(repository, context)
    if existing is not None:
        return {"context": context, "feature_snapshot": existing}
    store, records = artifact_store(repository), []
    ref = TableContractRef(contract_id=CONTRACT.contract_id, definition_hash=CONTRACT.definition_hash)
    schema = pa.schema([pa.field(c.name, query.arrow_type_for(c.physical_type), nullable=c.nullable)
        for c in CONTRACT.columns])
    total_bytes = 0
    with stage("eligible history materialization"):
        for rows in _rows(snapshot, prepared):
            sink = pa.BufferOutputStream()
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), sink)
            payload = sink.getvalue().to_pybytes()
            total_bytes += len(payload)
            if len(payload) > request["limits"]["max_object_bytes"] or total_bytes > request["limits"]["max_total_bytes"]:
                raise fail("RESULT_LIMIT_EXCEEDED", "history output exceeds physical byte limits")
            cap = request["limits"]["max_object_bytes"]
            with pq.ParquetFile(pa.BufferReader(payload), thrift_string_size_limit=min(cap, 2**31 - 1),
                    thrift_container_size_limit=min(cap, 2**31 - 1)) as parquet:
                expanded = sum(parquet.metadata.row_group(i).total_byte_size for i in range(parquet.num_row_groups))
            if expanded > cap:
                raise fail("RESULT_LIMIT_EXCEEDED", "history output exceeds native uncompressed read limit")
            published = store.publish_bytes(payload, schema_ref="parquet_fragment.v1")
            obj = ObjectRef(kind="parquet_fragment", object_id=published.artifact_id,
                content_hash=published.content_hash, byte_size=published.byte_size)
            inspected = objects.inspect_fragment(store, obj, CONTRACT, ref, context["context_id"])
            records.append(manifests.fragment_record(inspected, ref, input_receipt_refs=(),
                import_request_hash=context["context_id"]))
    logical = objects.partition_logical_hash(store, [r.object_ref for r in records], CONTRACT, ref,
        context["context_id"])
    manifest = manifests.dataset_manifest(ref, records, knowledge_mode="reconstructed",
        coverage_receipt_refs=(), availability_evidence_refs=(),
        partition_logical_hashes={context["context_id"]: logical})
    output = manifests.snapshot_ref({TABLE: manifest}, calendar_version=snapshot.calendar_version,
        source_priority_version=snapshot.source_priority_version, finality_receipt_refs=())
    slot = _slot(context)
    committed = catalog.commit_snapshot(catalog_connection(repository), scope=slot,
        request_hash=context["context_id"], contracts=(CONTRACT,), objects=tuple(r.object_ref for r in records),
        records=records, manifests=(manifest,), snapshot=output, expected_head_snapshot_id=None,
        expected_head_generation=0, receipt_id=slot, attempt_id=slot, fence=1,
        fence_check=lambda conn: None, clock=SystemClock(), store=store)
    return {"context": context, "feature_snapshot": committed.snapshot_ref}


def _stored_query(repository, feature_snapshot, context, columns, limits):
    """Bound the exact immutable feature read before opening any payload."""
    count = sum(len(t["forecasts"]) for t in context["targets"])
    records = repository.fragment_records(feature_snapshot, TABLE)
    if (sum(r.row_count for r in records) != count
            or any(r.object_ref.byte_size > limits["max_object_bytes"] for r in records)
            or sum(r.object_ref.byte_size for r in records) > limits["max_total_bytes"]):
        raise fail("RESULT_LIMIT_EXCEEDED", "stored history exceeds explicit bounds")
    query_ = DataQuery(snapshot_id=feature_snapshot.snapshot_id,
        table_contract_ref=feature_snapshot.table_versions[TABLE].table_contract_ref,
        columns=(*METADATA, *columns), order_by=CONTRACT.primary_key,
        key_filter=(KeyPredicate(column="context_id", operator="eq", values=(context["context_id"],)),),
        max_batch_rows=min(count, CONTRACT.maximum_batch_rows), max_result_rows=count)
    return query_


def _iter_read(repository, source, feature_snapshot, *, columns, **request):
    """Validate committed history rows incrementally for bounded consumers."""
    context = _prepare(repository, source, **request)[0]
    if (_existing(repository, context) != feature_snapshot or set(feature_snapshot.table_versions) != {TABLE}):
        raise fail("HOLDOUT_ACCESS_DENIED", "experiment input is not the exact eligible-history artifact")
    admit_fields(repository.table_contract(feature_snapshot, TABLE), context,
        policy_id=request["policy"]["policy_id"], columns=columns)
    query_ = _stored_query(repository, feature_snapshot, context, columns, request["limits"])
    expected = [(t["event_id"], c["horizon_sessions"], c["cutoff_id"], c["source_session_ceiling"])
        for t in context["targets"] for c in t["forecasts"]]
    max_object_bytes = request["limits"]["max_object_bytes"]

    def rows():
        position = 0
        for batch in repository.scan(query_, table_name=TABLE,
                max_object_bytes=max_object_bytes):
            for row in batch.to_pylist():
                if (position >= len(expected)
                        or (row["event_id"], row["horizon_sessions"], row["cutoff_id"],
                            row["source_session_ceiling"]) != expected[position]):
                    raise fail("CONTRACT_MISMATCH", "stored history target/cutoff coverage differs")
                position += 1
                yield row
        if position != len(expected):
            raise fail("CONTRACT_MISMATCH", "stored history target/cutoff coverage differs")

    class RowStream:
        def __iter__(self):
            return rows()

    return {"context": context, "feature_snapshot": to_document(feature_snapshot), "rows": RowStream()}


def _read(repository, source, feature_snapshot, *, columns, **request):
    """Preserve the detached list API for callers that require a complete result."""
    result = _iter_read(repository, source, feature_snapshot, columns=columns, **request)
    return {**result, "rows": list(result["rows"])}
