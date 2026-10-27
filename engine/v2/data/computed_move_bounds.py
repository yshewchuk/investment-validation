"""Native source dates, without outcomes, for policy admission before value reads."""
from dataclasses import replace

import numpy as np

from engine.v2.contracts.data import ColumnContract, TableContract
from engine.v2.data.computed_moves import source_session_indices
from engine.v2.data.manifests import table_contract_hash

TABLE = "computed_move_bounds"
COLUMNS = ("ticker", "event_date", "session", "pre_close_date", "post_close_date",
           "capture_id", "source_hash", "outcome_object_id", "outcome_content_hash")


def source_bounds(sd, event_day, session):
    """Return the actual native price-date bracket, never a guessed market calendar."""
    indices = source_session_indices(sd, np.datetime64(event_day, "ns"), session)
    return ((None, None) if indices is None else
            tuple(str(sd[index].astype("datetime64[D]")) for index in indices))


def _contract():
    columns = tuple(ColumnContract(name=name, physical_type="string",
        nullable=name in ("session", "pre_close_date", "post_close_date")) for name in COLUMNS)
    contract = TableContract(contract_id=TABLE + ".v1", table_name=TABLE,
        semantic_version="1.0.0", definition_hash="sha256:" + "0" * 64, columns=columns,
        primary_key=("ticker", "event_date"), duplicate_policy="none_by_construction",
        foreign_keys=(), partition_columns=("ticker",), filterable_columns=("ticker", "event_date"),
        orderable_columns=("ticker", "event_date"), observation_time_column="event_date",
        publication_time_column=None, receipt_time_column=None,
        finality_semantics="reconstructed actual source dates; no publication claim",
        provenance_semantics="native source bracket bound to exact computed outcome object and capture",
        coverage_semantics="one row per produced outcome; null bounds for every unavailable label",
        schema_evolution_policy="new contract identity for changed semantics", maximum_batch_rows=65536,
        legacy_mapping_ref=None)
    return replace(contract, definition_hash=table_contract_hash(contract))


CONTRACT = _contract()
