"""Native signed IV-crush target definition, without a publisher or source claim."""
from dataclasses import replace

from engine.v2.contracts.data import ColumnContract, TableContract

from . import manifests

IV_CRUSH_TABLE_NAME = "iv_crush_targets"


def _contract():
    columns = tuple(ColumnContract(name=n, physical_type="string", nullable=n == "session")
                    for n in ("ticker", "event_date", "session"))
    columns += tuple(ColumnContract(name=n, physical_type="string", nullable=True,
        observation_time_semantics="selected quoted close date, YYYY-MM-DD; no publication claim")
        for n in ("pre_date", "post_date"))
    columns += tuple(ColumnContract(name=n, physical_type="float64", nullable=True, unit="percent",
        null_policy="missing/nonfinite anchor; finite values retained even on skipped pairs")
        for n in ("pre_iv30", "post_iv30"))
    columns += (
        ColumnContract(name="crush_pct_iv30", physical_type="float64", nullable=True,
            unit="percent", null_policy="null exactly when skipped; never imputed or floored",
            observation_time_semantics="signed relative IV30 change: 100*(post_iv30/pre_iv30-1); "
                                       "adjacent daily rows span at most five calendar days; "
                                       "finite positive pre-IV and finite post-IV, no post-positive filter"),
        ColumnContract(name="available_as_of_date", physical_type="string", nullable=True,
            observation_time_semantics="post_date plus one calendar day; modeled day-clock floor, "
                                       "not evidence of source publication or finality"),
        ColumnContract(name="skipped", physical_type="bool", nullable=False),
        ColumnContract(name="skip_reason", physical_type="string", nullable=True),
        ColumnContract(name="source_id", physical_type="string", nullable=False),
        ColumnContract(name="input_hash", physical_type="string", nullable=False),
    )
    contract = TableContract(contract_id=IV_CRUSH_TABLE_NAME + ".v1", table_name=IV_CRUSH_TABLE_NAME,
        semantic_version="1.0.0", definition_hash="sha256:" + "0" * 64, columns=columns,
        primary_key=("ticker", "event_date"), duplicate_policy="duplicate event dates refuse",
        foreign_keys=(), partition_columns=("ticker",), filterable_columns=("ticker",),
        orderable_columns=("ticker", "event_date"), observation_time_column="event_date",
        publication_time_column=None, receipt_time_column=None,
        finality_semantics="no finality or actual publication evidence; reconstructed targets only",
        provenance_semantics="input_hash is the canonical hash of the supplied typed event/daily projections; "
                             "source_id is caller-declared identity, unverified by the pure builder",
        coverage_semantics="one row per supplied event, skipped rows included; no completeness inference; "
                           "crush_pct_iv30 and available_as_of_date are null exactly when skipped; "
                           "skip_reason is null exactly when not skipped",
        schema_evolution_policy="new contract identity for any semantic change", maximum_batch_rows=10000)
    return replace(contract, definition_hash=manifests.table_contract_hash(contract))


IV_CRUSH_CONTRACT = _contract()
