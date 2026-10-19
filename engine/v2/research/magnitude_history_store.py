"""Protected build-time history and stored-only reads through native snapshots."""
from __future__ import annotations

import json
import math

import pyarrow as pa
import pyarrow.parquet as pq

from engine.v2.contracts.data import DataQuery, KeyPredicate, ObjectRef, TableContractRef
from engine.v2.data import catalog, errors, manifests, objects, query
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT
from engine.v2.foundation import SystemClock, canonical_json, content_hash, to_document
from engine.v2.research import event_history
from engine.v2.research._progress import Progress, stage
from engine.v2.research._snapshot import artifact_store, catalog_connection
from engine.v2.research.experiment_population import load_population
from engine.v2.research.magnitude_features import _feature_recipe
from engine.v2.research.magnitude_folds import plan_magnitude_folds
from engine.v2.research.magnitude_history_table import (
    CONTRACT,
    FEATURES,
    METADATA,
    TABLE,
    feature_recipe,
)


def build_context(repository, snapshot, *, producer, history_start, **request):
    """Derive source, population, calendars and clocks before accessing outcomes."""
    if repository.resolve(snapshot.snapshot_id) != snapshot:
        raise errors.fail("SNAPSHOT_UNRESOLVED", "history source must be an exact committed snapshot")
    plan = plan_magnitude_folds(repository, snapshot, **request)
    ceiling = min(c["source_session_ceiling"] for e in plan["events"] for c in e["forecasts"])
    start, _ = event_history._window(history_start, ceiling)
    identity = {"schema_version": "magnitude_materialization.v1", "source_snapshot": to_document(snapshot),
        "temporal_plan": plan, "history_start": start.isoformat(), "feature_recipe": feature_recipe(),
        "producer": producer}
    return {**identity, "context_id": content_hash(identity)}


def _names(context):
    """One immutable catalog success slot per complete build context."""
    if (not isinstance(context, dict)
            or not {"source_snapshot", "temporal_plan", "history_start"} <= context.keys()
            or not isinstance(context["temporal_plan"], dict)
            or context.get("context_id") != content_hash({k: v for k, v in context.items() if k != "context_id"})):
        raise errors.fail("IDENTITY_CONFLICT", "history context digest differs")
    key = context["context_id"].removeprefix("sha256:")
    return "magnitude-history-" + key, "magnitude-history-build-" + key


def _existing(repository, context):
    """Resolve only the exact successful receipt, without following its head."""
    scope, receipt = _names(context)
    row = catalog_connection(repository).execute(
        "SELECT * FROM data_import_receipts WHERE receipt_id=?", (receipt,)).fetchone()
    if row is None:
        return None
    if ((row["scope"], row["source_manifest_hash"], row["attempt_id"], row["fence"], row["status"])
            != (scope, context["context_id"], receipt, 1, "committed")):
        raise errors.fail("IDENTITY_CONFLICT", "history build receipt differs from its context")
    return repository.resolve(row["result_snapshot_id"])


def _event_ceiling(event, start, event_day):
    """Validate exact horizon coverage and causal windows for one canonical target."""
    forecasts = event["forecasts"]
    if (any(type(c["horizon_sessions"]) is not int for c in forecasts)
            or sorted(c["horizon_sessions"] for c in forecasts) != [1, 3, 5]):
        raise ValueError
    for cutoff in forecasts:
        _, decision = event_history._window(start, cutoff["source_session_ceiling"])
        if (decision >= event_day
                or any(cutoff[k] != event[k] for k in ("event_date", "session"))):
            raise ValueError
    return max(c["source_session_ceiling"] for c in forecasts)


def _target_ceilings(indexed, context):
    """Guard every derived target/window before planning or opening any outcome."""
    ceilings = {}
    try:
        for event in context["temporal_plan"]["events"]:
            if event["event_id"] not in indexed.index:
                raise errors.fail("HOLDOUT_ACCESS_DENIED", "history target is not admitted")
            target = indexed.loc[event["event_id"]]
            if (event["event_date"] != target.event_date.date().isoformat()
                    or any(event[k] != target[k] for k in ("ticker", "session", "population_use"))):
                raise ValueError
            ceiling = _event_ceiling(event, context["history_start"], target.event_date.date())
            ceilings[event["ticker"]] = max(ceilings.get(event["ticker"], ""), ceiling)
    except (KeyError, TypeError, ValueError):
        raise errors.fail("CONTRACT_MISMATCH", "history target or forecast window differs") from None
    return ceilings


def _preflight_source(repository, snapshot, context):
    """Prove the complete candidate union before any per-forecast history reads."""
    plan = context["temporal_plan"]
    population = load_population(repository, snapshot, as_of_month=plan["holdout_as_of_month"],
                                 purpose="selection")
    indexed = population.set_index("event_id", drop=False)
    ceilings = _target_ceilings(indexed, context)
    days = population.event_date.dt.strftime("%Y-%m-%d")
    candidates = population[(days >= context["history_start"])
        & (days < population.ticker.map(ceilings).fillna(""))]
    contract = repository.table_contract(snapshot, "computed_moves")
    if contract != COMPUTED_MOVES_CONTRACT:
        raise errors.fail("CONTRACT_MISMATCH", "history requires native computed moves")
    plans = event_history._plans(repository, snapshot,
        candidates.sort_values(["event_date", "event_id"]), contract)
    return indexed, plans


def _prepare_history(repository, snapshot, context):
    """Lazily decode at most one row per admitted candidate; no cross-build cache."""
    with stage("history preflight"):
        population, plans = _preflight_source(repository, snapshot, context)
    decoded, by_ticker = {}, {}
    for event, request, record in plans:
        by_ticker.setdefault(event.ticker, []).append((event, request, record))
    def read_rows(event, request):
        """Read only when a forecast first needs this preflighted candidate."""
        if event.event_id not in decoded:
            decoded[event.event_id] = [row for batch in repository.scan(request, table_name="computed_moves")
                                       for row in batch.to_pylist()]
        return decoded[event.event_id]
    return population, by_ticker, read_rows


def _event_rows(repository, snapshot, context, event, prepared):
    """Build all horizons and retain every consumed or omitted input identity."""
    population, by_ticker, read_rows = prepared
    target = population.loc[event["event_id"]]
    rows = []
    for cutoff in sorted(event["forecasts"], key=lambda c: c["horizon_sessions"]):
        start, decision = event_history._window(context["history_start"], cutoff["source_session_ceiling"])
        plans = [p for p in by_ticker.get(event["ticker"], ()) if p[0].event_date.date() < decision]
        history = event_history._assemble_history(snapshot, target, population, plans,
            start=start, decision=decision, purpose="selection", read_rows=read_rows)
        _feature_recipe(history["features"])  # Validate names/numbers; the stored recipe is already pinned.
        lineage = {k: v for k, v in history.items() if k not in ("features", "holdout_exclusions")}
        rows.append({"event_id": event["event_id"], "context_id": context["context_id"],
            "horizon_sessions": cutoff["horizon_sessions"], "event_date": event["event_date"],
            "cutoff_id": cutoff["cutoff_id"], "source_session_ceiling": cutoff["source_session_ceiling"],
            "history_status": lineage["history_status"], "history_json": canonical_json(lineage),
            **history["features"]})
    return rows


def materialize(repository, snapshot, context):
    """Publish one complete context; replay returns its original stored artifact."""
    if (not isinstance(context, dict) or context.get("source_snapshot") != to_document(snapshot)
            or repository.resolve(snapshot.snapshot_id) != snapshot):
        raise errors.fail("IDENTITY_CONFLICT", "history source differs from its build context")
    existing = _existing(repository, context)
    if existing is not None:
        read_stored(repository, existing, context)
        return existing
    with stage("history prepare"):
        prepared = _prepare_history(repository, snapshot, context)
    store, records = artifact_store(repository), []
    ref = TableContractRef(contract_id=CONTRACT.contract_id, definition_hash=CONTRACT.definition_hash)
    schema = pa.schema([pa.field(c.name, query.arrow_type_for(c.physical_type), nullable=c.nullable)
                        for c in CONTRACT.columns])
    ordered = sorted(context["temporal_plan"]["events"], key=lambda e: e["event_id"])
    progress = Progress("history events", len(ordered))
    with stage("history materialize"):
        for event in ordered:
            sink = pa.BufferOutputStream()
            rows = _event_rows(repository, snapshot, context, event, prepared)
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), sink)
            published = store.publish_bytes(sink.getvalue().to_pybytes(), schema_ref="parquet_fragment.v1")
            obj = ObjectRef(kind="parquet_fragment", object_id=published.artifact_id,
                            content_hash=published.content_hash, byte_size=published.byte_size)
            inspection = objects.inspect_fragment(store, obj, CONTRACT, ref, context["context_id"])
            records.append(manifests.fragment_record(inspection, ref, input_receipt_refs=(),
                                                    import_request_hash=context["context_id"]))
            progress.tick()
    partition_hash = objects.partition_logical_hash(store, [r.object_ref for r in records],
                                                    CONTRACT, ref, context["context_id"])
    manifest = manifests.dataset_manifest(ref, records, knowledge_mode="reconstructed",
        coverage_receipt_refs=(), availability_evidence_refs=(),
        partition_logical_hashes={context["context_id"]: partition_hash})
    output = manifests.snapshot_ref({TABLE: manifest}, calendar_version=snapshot.calendar_version,
        source_priority_version=snapshot.source_priority_version, finality_receipt_refs=())
    scope, receipt = _names(context)
    result = catalog.commit_snapshot(catalog_connection(repository), scope=scope,
        request_hash=context["context_id"], contracts=(CONTRACT,), objects=tuple(r.object_ref for r in records),
        records=records, manifests=(manifest,), snapshot=output, expected_head_snapshot_id=None,
        expected_head_generation=0, receipt_id=receipt, attempt_id=receipt, fence=1,
        fence_check=lambda conn: None, clock=SystemClock(), store=store)
    return result.snapshot_ref


def _scan(repository, snapshot, context, columns, count):
    """One explicit context predicate and manifest-derived complete row bound."""
    request = DataQuery(snapshot_id=snapshot.snapshot_id,
        table_contract_ref=snapshot.table_versions[TABLE].table_contract_ref, columns=tuple(columns),
        key_filter=(KeyPredicate(column="context_id", operator="eq", values=(context["context_id"],)),),
        order_by=CONTRACT.primary_key,
        max_batch_rows=min(CONTRACT.maximum_batch_rows, count), max_result_rows=count)
    return [row for batch in repository.scan(request, table_name=TABLE) for row in batch.to_pylist()]


def _lineage(history, context, event, cutoff):
    """Validate the closed stored history envelope against independently derived pins."""
    plan = context["temporal_plan"]
    pins = {name: plan[name] for name in event_history._CONTEXT if name != "population_use"}
    pins.update(schema_version="research_event_history.v2", event_id=event["event_id"],
        ticker=event["ticker"], event_date=event["event_date"], population_use=event["population_use"],
        purpose="selection", history_start=context["history_start"],
        decision_session=cutoff["source_session_ceiling"], computed_moves_version=
        context["source_snapshot"]["table_versions"]["computed_moves"]["dataset_version_id"])
    lists = ("history", "omitted_history", "read_inputs")
    keys = set(pins) | {*lists, "history_id", "history_status"}
    if (set(history) != keys or any(history[name] != value for name, value in pins.items())
            or any(type(history[name]) is not list for name in lists)):
        raise ValueError
    identity = {k: v for k, v in history.items()
                if k not in ("history_id", "history_status")}
    if content_hash(identity) != history["history_id"]:
        raise ValueError


def _metadata(row, context, event, cutoff):
    """Complete row identity and lineage checks before any feature-column scan."""
    expected = {"event_id": event["event_id"], "context_id": context["context_id"],
        "horizon_sessions": cutoff["horizon_sessions"], "event_date": event["event_date"],
        "cutoff_id": cutoff["cutoff_id"], "source_session_ceiling": cutoff["source_session_ceiling"]}
    try:
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError
        history = json.loads(row["history_json"])
        _lineage(history, context, event, cutoff)
        if (canonical_json(history) != row["history_json"]
                or row["history_status"] != ("available" if history["history"] else "empty")
                or history["history_status"] != row["history_status"]):
            raise ValueError
        return len(history["history"])
    except (ValueError, TypeError, KeyError, AttributeError):
        raise errors.fail("CONTRACT_MISMATCH", "stored history metadata differs from its context") from None


def _preflight_features(repository, snapshot, context):
    """Prove complete event isolation from native inspected object metadata."""
    events = sorted(context["temporal_plan"]["events"], key=lambda e: e["event_id"])
    records = repository.fragment_records(snapshot, TABLE)
    if len(records) != len(events):
        raise errors.fail("HOLDOUT_ACCESS_DENIED", "stored history object coverage differs")
    expected = []
    for record, event in zip(records, events):
        prefix = (event["event_id"], context["context_id"])
        if (record.row_count != 3 or record.partition_key != context["context_id"]
                or record.primary_key_min != (*prefix, 1) or record.primary_key_max != (*prefix, 5)
                or record.time_min != event["event_date"] or record.time_max != event["event_date"]):
            raise errors.fail("HOLDOUT_ACCESS_DENIED", "stored history event isolation cannot be proven")
        expected.extend((event, cutoff) for cutoff in sorted(event["forecasts"], key=lambda c: c["horizon_sessions"]))
    return expected


def _validate_features(row, count):
    """Stored numbers must obey the declared units, count and missingness recipe."""
    if type(row["n_prior"]) is not int or row["n_prior"] != count:
        raise errors.fail("CONTRACT_MISMATCH", "stored history count differs")
    for name in FEATURES[1:]:
        minimum = int(name.split("_", 1)[0][3:]) if name.startswith("ema") else 1
        value = row[name]
        if ((count < minimum and value is not None) or (count >= minimum and
                (type(value) not in (int, float) or not math.isfinite(value)))
                or (value is not None and "abs_move" in name and value < 0)):
            raise errors.fail("CONTRACT_MISMATCH", "stored history feature or missingness is invalid")


def read_stored(repository, snapshot, context):
    """Globally admit metadata, then decode stored columns without outcome reads."""
    if _existing(repository, context) != snapshot or set(snapshot.table_versions) != {TABLE}:
        raise errors.fail("IDENTITY_CONFLICT", "feature snapshot is not the committed context output")
    if repository.table_contract(snapshot, TABLE) != CONTRACT:
        raise errors.fail("CONTRACT_MISMATCH", "stored history contract differs")
    expected = _preflight_features(repository, snapshot, context)
    metadata = _scan(repository, snapshot, context, METADATA, len(expected))
    if len(metadata) != len(expected):
        raise errors.fail("CONTRACT_MISMATCH", "stored history row coverage differs")
    counts = [_metadata(row, context, event, cutoff) for row, (event, cutoff) in zip(metadata, expected)]
    rows = _scan(repository, snapshot, context, (*METADATA, *FEATURES), len(expected))
    if len(rows) != len(metadata):
        raise errors.fail("CONTRACT_MISMATCH", "stored history row count changed")
    for row, admitted, count in zip(rows, metadata, counts):
        if {key: row[key] for key in METADATA} != admitted:
            raise errors.fail("CONTRACT_MISMATCH", "stored history count or metadata changed")
        _validate_features(row, count)
    identity = {"context": context, "feature_snapshot": to_document(snapshot), "rows": rows}
    return {**identity, "inputs_id": content_hash(identity)}
