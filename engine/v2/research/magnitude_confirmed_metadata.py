"""Bounded complete pinned earnings metadata, not coverage or numerical authority."""
import sys

import pandas as pd

from engine.v2.contracts.data import DataQuery, KeyPredicate, TableContractRef
from engine.v2.data import bounded_snapshot, manifests, query
from engine.v2.data.errors import fail
from engine.v2.foundation import content_hash, iter_canonical_json, to_document
from engine.v2.research import confirmed_metadata as reducer
from engine.v2.research import progress
from engine.v2.research.snapshot_inputs import admit_snapshot

_TABLE = "earnings_events"
_COLUMNS = ("event_id", "ticker", "event_date", "session", "date_conflict",
            "event_cluster_id", "src_orats")
_REF = TableContractRef(contract_id="legacy.earnings_events.v2",
    definition_hash="sha256:3cdea8601b2ec005dd9f079639998450c63502a41472fe0cdc6b6987ec47aeaf")
_CATALOG_LIMITS = {"max_catalog_tables", "max_catalog_fragments", "max_catalog_bytes",
                   "max_fragments", "max_partitions"}
_LIMITS = _CATALOG_LIMITS | {"max_tickers", "max_scan_rows", "max_object_bytes",
    "max_total_object_bytes", "max_batch_rows", "max_retained_rows",
    "max_retained_metadata_bytes", "max_proof_bytes"}


def _cap(value, limits, name):
    if value > limits[name]:
        raise fail("RESULT_LIMIT_EXCEEDED", f"confirmed metadata exceeds {name}")


def _request(tickers, limits):
    if (type(limits) is not dict or limits.keys() != _LIMITS
            or any(type(v) is not int or not 0 < v < sys.maxsize for v in limits.values())):
        raise fail("CONTRACT_MISMATCH", "confirmed metadata requires exact positive limits")
    if type(tickers) is not list:
        raise fail("CONTRACT_MISMATCH", "tickers must be a nonempty plain list")
    _cap(len(tickers), limits, "max_tickers")
    return reducer.validate_tickers(tickers), dict(limits)


def _bytes(document, limit):
    total = 0
    for chunk in iter_canonical_json(document):
        total += len(chunk.encode("utf-8"))
        if total > limit:
            raise fail("RESULT_LIMIT_EXCEEDED", "confirmed metadata byte limit exceeded")
    return total


def _years(records):
    years = set()
    for record in records:
        year = record.partition_key
        if (type(year) is not str or len(year) != 4 or not year.isascii()
                or not year.isdigit() or year == "0000"):
            raise fail("CONTRACT_MISMATCH", "earnings partition must be a canonical year")
        years.add(year)
    return sorted(years)


def _inventory(session, snapshot, tickers, limits):
    snapshot = admit_snapshot(session, snapshot)
    resolved = session.resolve_full(snapshot.snapshot_id)
    manifest = resolved.table_manifests[_TABLE]
    contract = session.table_contract(snapshot, _TABLE)
    ref = manifest.dataset_version_ref.table_contract_ref
    if ref != _REF or manifests.table_contract_hash(contract) != _REF.definition_hash:
        raise fail("UNSUPPORTED_CONTRACT", "unsupported exact earnings contract")
    records = session.fragment_records(snapshot, _TABLE)
    years = _years(records)
    plans, candidates, total_rows, total_bytes = [], [], 0, 0
    meter = progress.Progress("confirmed metadata preflight", len(years))
    for year in years:
        predicates = (KeyPredicate(column="year", operator="eq", values=(int(year),)),
                      KeyPredicate(column="ticker", operator="in", values=tuple(tickers)))
        population = query.plan_scan_population(contract, records, key_filter=predicates)
        bound = session.scan_population_bound(snapshot.snapshot_id, table_name=_TABLE,
            table_contract_ref=ref, key_filter=predicates)
        if bound != population.row_count:
            raise fail("CONTRACT_MISMATCH", "inconsistent earnings candidate bound")
        candidates.extend(population.records)
        total_rows += bound
        _cap(total_rows, limits, "max_scan_rows")
        for record in population.records:
            _cap(record.object_ref.byte_size, limits, "max_object_bytes")
            total_bytes += record.object_ref.byte_size
            _cap(total_bytes, limits, "max_total_object_bytes")
        plans.append(DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=ref,
            columns=_COLUMNS, key_filter=predicates, order_by=("event_id",),
            max_result_rows=bound,
            max_batch_rows=min(limits["max_batch_rows"], contract.maximum_batch_rows,
                               max(1, bound))))
        meter.tick()
    if (len(candidates) != len(records)
            or sorted(r.fragment_id for r in candidates) != sorted(r.fragment_id for r in records)):
        raise fail("CONTRACT_MISMATCH", "earnings candidates do not exhaust membership")
    source = {"snapshot": to_document(snapshot), "table": _TABLE,
        "contract_ref": to_document(ref), "dataset_manifest": to_document(manifest),
        "fragment_records": [to_document(r) for r in records], "columns": list(_COLUMNS)}
    return source, plans, {"partitions": len(years), "fragments": len(records),
                           "candidate_rows": total_rows}


def _scan(session, plans, tickers, limits, counts):
    frames, retained_rows, retained_bytes = [], 0, 0
    meter = progress.Progress("confirmed metadata partitions", len(plans), min_items=0)
    for plan in plans:
        for batch in session.scan(plan, table_name=_TABLE,
                                  max_object_bytes=limits["max_object_bytes"]):
            _cap(retained_rows + batch.num_rows, limits, "max_retained_rows")
            frame = batch.to_pandas()
            rows, _ = reducer.metadata_rows(frame, tickers=tickers)
            for row in rows:
                retained_bytes += _bytes(row, limits["max_retained_metadata_bytes"] - retained_bytes)
            retained_rows += len(rows)
            frames.append(frame)
            meter.tick(0)  # Bounded heartbeat while the partition is still provisional.
        meter.tick()
    empty = pd.DataFrame(columns=_COLUMNS)
    empty["event_date"] = pd.Series(dtype="datetime64[ns]")
    events = pd.concat(frames, ignore_index=True) if frames else empty
    sequences, ambiguity = reducer.reduce_with_counts(events, tickers=tickers)
    counts.update(selected_rows=retained_rows,
                  confirmed_rows=sum(s["confirmed_rows"] for s in sequences))
    counts.update(ambiguity)
    return sequences


def build_index(repository, snapshot, *, tickers, limits):
    """Authenticate and exhaust a bounded pinned view; publish no partial proof.

    Synthetic-only boundary. Receipt references remain unopened; no coverage,
    observed-PIT, protected-value or numerical-contributor admission is implied.
    """
    tickers, limits = _request(tickers, limits)
    with progress.stage("confirmed metadata index", report_refusals=True):
        with bounded_snapshot.open_bounded_snapshot(repository, snapshot, table_name=_TABLE,
                limits={k: limits[k] for k in _CATALOG_LIMITS}) as session:
            source, plans, counts = _inventory(session, snapshot, tickers, limits)
            progress.decision("confirmed metadata index",
                f"{len(plans)} partitions; {counts['candidate_rows']} candidate rows; "
                f"effective batch rows {min((p.max_batch_rows for p in plans), default=0)}"
                f"..{max((p.max_batch_rows for p in plans), default=0)}")
            sequences = _scan(session, plans, tickers, limits, counts)
        document = {"schema_version": "magnitude_confirmed_metadata.v1",
            "policy": {"confirmation": "src_orats_and_session.v1", "warmup_events": 4,
                "conflicts": "unconfirm_date_and_cluster_conflicts.v2",
                "coverage": "complete_pinned_view_only.v1"},
            "source": source, "tickers": tickers, "sequences": sequences, "counts": counts}
        document["index_id"] = content_hash(document)
        _bytes(document, limits["max_proof_bytes"])
        return document
