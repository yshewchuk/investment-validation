"""Separate experiment-qualified realized history; live contracts are not aliases."""
from dataclasses import replace

from engine.v2.contracts.data import ColumnContract
from engine.v2.data.errors import fail
from engine.v2.data.manifests import table_contract_hash
from engine.v2.foundation import to_document
from engine.v2.research.magnitude_history_table import CONTRACT as _V1
from engine.v2.research.magnitude_history_table import FEATURES as HISTORY_FEATURES
from engine.v2.research.magnitude_history_table import METADATA as METADATA

TABLE = "research_eligible_history"
SEASON_HEAT_FEATURES = ("season_heat_mean_abs_5", "season_heat_mean_abs_20",
    "season_heat_mean_abs_60", "season_heat_count_20", "season_heat_ratio_20_252")
FEATURES = (*HISTORY_FEATURES, *SEASON_HEAT_FEATURES)
_SEASON_HEAT_COLUMNS = (
    ColumnContract(name="season_heat_mean_abs_5", physical_type="float64", nullable=True,
        unit="percent", null_policy="null until 20 observed labels in the 5-session window"),
    ColumnContract(name="season_heat_mean_abs_20", physical_type="float64", nullable=True,
        unit="percent", null_policy="null until 20 observed labels in the 20-session window"),
    ColumnContract(name="season_heat_mean_abs_60", physical_type="float64", nullable=True,
        unit="percent", null_policy="null until 20 observed labels in the 60-session window"),
    ColumnContract(name="season_heat_count_20", physical_type="int64", nullable=False,
        unit="count", null_policy="always present; zero before any observed label"),
    ColumnContract(name="season_heat_ratio_20_252", physical_type="float64", nullable=True,
        unit="ratio", null_policy="null below the event minimum or when the 252-session mean is zero"),
)
_base = replace(_V1, table_name=TABLE, contract_id=TABLE + ".v2", semantic_version="2.0.0",
    columns=(*_V1.columns, *_SEASON_HEAT_COLUMNS),
    provenance_semantics="exact source anchors, grouped eligibility and contributor identity",
    coverage_semantics="exact requested event/horizon set; policy-qualified reconstructed history")
CONTRACT = replace(_base, definition_hash=table_contract_hash(_base))


def recipe():
    """Closed provenance, not a caller-controlled research-safe flag."""
    return {"schema_version": "eligible_history_features.v2", "usage": "experiment",
        "contract": to_document(CONTRACT), "fields": list(FEATURES),
        "arithmetic": "panel_math.history_features",
        "season_heat": {"fields": list(SEASON_HEAT_FEATURES),
            "arithmetic": "research.season_heat.compute_season_heat.v1"},
        "units": "percentage_points", "missing": "native_null_and_warmup",
        "implied_quote_history": "unsupported"}


def admit_fields(contract, context, *, policy_id, columns):
    """Refuse live/unknown provenance and unsupported aliases or derivations."""
    if (contract != CONTRACT or context.get("feature_recipe") != recipe()
            or context.get("eligibility", {}).get("policy", {}).get("policy_id") != policy_id
            or type(columns) not in (list, tuple) or not columns
            or any(type(name) is not str or name not in FEATURES for name in columns)
            or len(set(columns)) != len(columns)):
        raise fail("HOLDOUT_ACCESS_DENIED", "experiment fields require exact eligible-history provenance")
