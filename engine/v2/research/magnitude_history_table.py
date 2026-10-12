"""Immutable native storage contract for research-qualified history columns."""
from dataclasses import replace

from engine.v2.contracts.data import ColumnContract, TableContract
from engine.v2.data.manifests import table_contract_hash
from engine.v2.research.magnitude_features import _FEATURES

TABLE = "research_magnitude_history"
FEATURES = _FEATURES
METADATA = ("event_id", "context_id", "horizon_sessions", "event_date", "cutoff_id",
            "source_session_ceiling", "history_status", "history_json")


def feature_recipe():
    """Closed names, original percentage-point units and explicit missingness."""
    return {"schema_version": "stored_magnitude_history.v1", "arithmetic": "panel_math.history_features",
            "features": [{"name": name, "on_missing": "zero_count" if name == "n_prior" else "null"}
                         for name in FEATURES], "units": "percentage_points", "imputation": "none"}


def _contract():
    """Version all columns, event isolation and reconstructed knowledge semantics."""
    columns = tuple(ColumnContract(name=name, physical_type="int64" if name == "horizon_sessions"
                                   else "string", nullable=False) for name in METADATA)
    columns += tuple(ColumnContract(name=name, physical_type="int64" if name == "n_prior" else "float64",
        nullable=name != "n_prior", unit="count" if name == "n_prior" else "percent",
        null_policy="zero count" if name == "n_prior" else "null when history is insufficient")
        for name in FEATURES)
    placeholder = TableContract(contract_id=TABLE + ".v1", table_name=TABLE, semantic_version="1.0.0",
        definition_hash="sha256:" + "0" * 64, columns=columns,
        primary_key=("event_id", "context_id", "horizon_sessions"), duplicate_policy="none_by_construction",
        foreign_keys=(), partition_columns=("context_id",), filterable_columns=("context_id", "event_id"),
        orderable_columns=("event_id", "context_id", "horizon_sessions"), observation_time_column="event_date",
        publication_time_column=None, receipt_time_column=None,
        finality_semantics="immutable reconstruction; no observed publication claim",
        provenance_semantics="physical context digest and complete protected history input identity",
        coverage_semantics="exact admitted target set, with horizons 1, 3 and 5 for every event",
        schema_evolution_policy="new contract identity for any semantic or column change",
        maximum_batch_rows=65536, legacy_mapping_ref=None)
    return replace(placeholder, definition_hash=table_contract_hash(placeholder))


CONTRACT = _contract()
