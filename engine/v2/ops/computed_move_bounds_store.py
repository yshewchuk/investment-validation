"""Opt-in metadata-only sibling of native outcomes; no research raw-source reader."""
from datetime import date, timedelta

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from engine.v2.contracts.data import ObjectRef, TableContractRef
from engine.v2.data import errors, manifests, objects
from engine.v2.data.computed_move_bounds import COLUMNS, CONTRACT, TABLE, source_bounds
from engine.v2.data.computed_moves import _collapse_and_order
from engine.v2.foundation import content_hash

REF = TableContractRef(contract_id=CONTRACT.contract_id, definition_hash=CONTRACT.definition_hash)


def write_bounds(store, rows, events, sd, outcome_records):
    """Bind finalized v5 classifications to their actual source dates and object bytes."""
    sessions = {event.event_date.date().isoformat(): event.session
                for event in _collapse_and_order(events).itertuples()}
    records = {record.primary_key_min: record for record in outcome_records}
    bounds = []
    for row in rows:
        record = records[(row["ticker"], row["event_date"])]
        session = sessions[row["event_date"]]
        session = None if pd.isna(session) else session
        pre, post = (None, None)
        if row["skipped"] is False and row["available_as_of_date"] is not None:
            pre, post = source_bounds(sd, row["event_date"], session)
            if post is None or (date.fromisoformat(post) + timedelta(days=1)).isoformat() != row["available_as_of_date"]:
                raise errors.fail("CONTRACT_MISMATCH", "native outcome and source bounds differ")
        bounds.append({**{name: row[name] for name in ("ticker", "event_date", "capture_id", "source_hash")},
            "session": session, "pre_close_date": pre, "post_close_date": post,
            "outcome_object_id": record.object_ref.object_id,
            "outcome_content_hash": record.object_ref.content_hash})
    schema = pa.schema([pa.field(c.name, pa.string(), nullable=c.nullable) for c in CONTRACT.columns])
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(bounds, schema=schema).select(COLUMNS), sink)
    published = store.publish_bytes(sink.getvalue().to_pybytes(), schema_ref="parquet_fragment.v1")
    obj = ObjectRef(kind="parquet_fragment", object_id=published.artifact_id,
                    content_hash=published.content_hash, byte_size=published.byte_size)
    inspection = objects.inspect_fragment(store, obj, CONTRACT, REF, rows[0]["ticker"])
    return (manifests.fragment_record(inspection, REF, input_receipt_refs=(),
                                     import_request_hash=published.content_hash),)


def compose(parent, rewritten, bounds_by_ticker, table_manifests):
    """Drop rewritten stale sidecars even on default-off refreshes; keep untouched ones."""
    previous = parent.table_manifests.get(TABLE)
    if previous is None and not bounds_by_ticker:
        return ()
    if previous is not None and previous.dataset_version_ref.table_contract_ref != REF:
        raise errors.fail("CONTRACT_MISMATCH", "unsupported native source-bound contract")
    kept = [r for r in parent.records if r.table_contract_ref == REF and r.partition_key not in rewritten]
    fresh = [r for records in (bounds_by_ticker or {}).values() for r in records]
    records = tuple(sorted((*kept, *fresh), key=lambda r: (r.partition_key, r.primary_key_min)))
    manifest = manifests.dataset_manifest(REF, records, knowledge_mode="reconstructed",
        coverage_receipt_refs=(), availability_evidence_refs=(),
        parent_dataset_version_id=previous.dataset_version_ref.dataset_version_id if previous else None)
    table_manifests[TABLE] = manifest
    return records


def request_identity(outcomes_request, bounds_by_ticker):
    """Keep default request identity; explicitly bind every opt-in metadata fragment."""
    if bounds_by_ticker is None:
        return outcomes_request
    return content_hash({"outcomes_request": outcomes_request, "source_bounds":
        {ticker: [r.fragment_id for r in records] for ticker, records in sorted(bounds_by_ticker.items())}})
