"""Native daily OHLCV reconstruction; callers supply verified acquisition pins."""
import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from engine.v2.contracts import ColumnContract, ObjectRef, TableContract, TableContractRef
from engine.v2.data import errors, manifests, objects, query
from engine.v2.foundation import format_timestamp, parse_timestamp

TABLE = "massive_daily"
NORMALIZER = "massive_daily_normalizer.v1"
_VALUES = dict(open="o", high="h", low="l", close="c", volume="v")
_TEXT = ("ticker", "adjustment_basis", "date", "occurrence_id", "raw_receipt_id",
         "raw_hash", "first_received_at", "observed_at")


def _contract():
    columns = tuple(ColumnContract(name=n, physical_type="string", nullable=False) for n in _TEXT)
    columns += (ColumnContract(name="source_timestamp_ms", physical_type="int64", nullable=False,
                              unit="milliseconds_since_unix_epoch", timezone="UTC"),)
    columns += tuple(ColumnContract(name=n, physical_type="float64", nullable=False,
        unit="provider_reported_shares" if n == "volume" else "price_per_share",
        adjustment_basis="explicit row adjustment_basis; provider values unchanged") for n in _VALUES)
    contract = TableContract(contract_id=TABLE + ".v1", table_name=TABLE, semantic_version="1.0.0",
        definition_hash="sha256:" + "0" * 64, columns=columns,
        primary_key=("ticker", "adjustment_basis", "date"), duplicate_policy="duplicate session dates refuse",
        foreign_keys=(), partition_columns=("ticker",), filterable_columns=("ticker", "adjustment_basis"),
        orderable_columns=("ticker", "adjustment_basis", "date"), observation_time_column="date",
        publication_time_column=None, receipt_time_column="first_received_at",
        finality_semantics="immutable reconstruction; later corrections require explicit new inputs",
        provenance_semantics="original page receipt/hash/clocks and completion occurrence; artifact pinned by dataset",
        coverage_semantics="saved clipped acquisition intervals only; no calendar or missing-session inference",
        schema_evolution_policy="new contract identity for semantic changes", maximum_batch_rows=1000)
    return replace(contract, definition_hash=manifests.table_contract_hash(contract))


CONTRACT = _contract()


def _values(row):
    values = {name: row[key] for name, key in _VALUES.items()}
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values.values())
            or any(float(v) != v for v in values.values())
            or values["volume"] < 0 or not 0 < values["low"] <= min(values["open"], values["close"])
            <= max(values["open"], values["close"]) <= values["high"]):
        raise ValueError
    return values


def normalize_rows(rows, request, binding, evidence):
    """Preserve numbers and original page clocks; clip only to the saved binding."""
    result = []
    try:
        if request["adjustment_basis"] not in ("raw", "split_adjusted"):
            raise ValueError
        if any(format_timestamp(parse_timestamp(evidence[k])) != evidence[k]
               for k in ("first_received_at", "observed_at")):
            raise ValueError
        for row in rows:
            if type(row["t"]) is not int:
                raise ValueError
            instant = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=row["t"])
            day = instant.astimezone(ZoneInfo("America/New_York")).date().isoformat()
            if not binding["start"] <= day <= binding["end"]:
                continue
            values = _values(row)
            result.append({"ticker": request["ticker"], "date": day,
                "adjustment_basis": request["adjustment_basis"], "source_timestamp_ms": row["t"],
                "occurrence_id": binding["occurrence_id"], **values,
                **{k: evidence[k] for k in ("raw_receipt_id", "raw_hash", "first_received_at", "observed_at")}})
    except (ValueError, TypeError, KeyError, OverflowError):
        raise errors.fail("CONTRACT_MISMATCH", "Massive daily row is invalid") from None
    return result


def build_candidate(store, rows, acquisition_id, request_hash, *, contract=CONTRACT):
    """Publish immutable bytes; return one inspected native dataset and snapshot."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    ordered = sorted(rows, key=lambda r: tuple(r[k] for k in contract.primary_key))
    if (not ordered or len(ordered) > 1000
            or len({tuple(r[k] for k in contract.primary_key) for r in ordered}) != len(ordered)
            or len({r["ticker"] for r in ordered}) != 1):
        raise errors.fail("CONTRACT_MISMATCH", "Massive daily candidate identity is invalid")
    schema = pa.schema([pa.field(c.name, query.arrow_type_for(c.physical_type), nullable=c.nullable)
                        for c in contract.columns])
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(ordered, schema=schema), sink)
    artifact = store.publish_bytes(sink.getvalue().to_pybytes(), schema_ref="parquet_fragment.v1")
    obj = ObjectRef(kind="parquet_fragment", object_id=artifact.artifact_id,
                    content_hash=artifact.content_hash, byte_size=artifact.byte_size)
    ref = TableContractRef(contract_id=contract.contract_id, definition_hash=contract.definition_hash)
    inspection = objects.inspect_fragment(store, obj, contract, ref, ordered[0]["ticker"])
    record = manifests.fragment_record(inspection, ref,
        input_receipt_refs=sorted({r["raw_receipt_id"] for r in ordered}), import_request_hash=request_hash)
    manifest = manifests.dataset_manifest(ref, (record,), knowledge_mode="reconstructed",
        coverage_receipt_refs=(acquisition_id,), availability_evidence_refs=())
    snapshot = manifests.snapshot_ref({contract.table_name: manifest}, calendar_version="massive_eastern_session_date.v1",
        source_priority_version="massive_only.v1", finality_receipt_refs=())
    return record, manifest, snapshot
