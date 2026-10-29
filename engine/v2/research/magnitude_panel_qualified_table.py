"""Additive eligible-history binding; existing v1/v2 descriptors stay frozen."""
from dataclasses import replace

from engine.v2.contracts.data import ColumnContract
from engine.v2.data.errors import fail
from engine.v2.data.manifests import table_contract_hash
from engine.v2.foundation import content_hash
from engine.v2.research import magnitude_panel_table as v1
from engine.v2.research import magnitude_panel_v2_table as v2

TABLE = v1.TABLE
METADATA = (*v1.METADATA, "current_quote_status")

_DAILY_HASH = "sha256:ae94d7433748268b7cc7269e368221c983eb63e42547105dba3e89acd58b4e8b"
# Frozen native/legacy ORATS normalization limits, not comparator masks.
_DAILY_RANGES = {"implied_move": (0., 100.), "iv10": (0., 500.), "iv30": (0., 500.),
    "exern_iv10": (0., 500.), "exern_iv30": (0., 500.), "rvol30": (0., 500.),
    "skew": (-50., 50.), "contango": (-100., 100.), "fwd90_30": (0., 500.),
    "fexern90_30": (0., 500.), "iee": (-50., 50.), "spot": (0., 1e6)}


def feature_dictionary(subset):
    """Keep numeric order/units, explicitly replace incompatible history policy."""
    fields = v2.feature_dictionary(subset)
    for field in fields:
        if field["name"] == "mean_prior_or_implied":
            field["derivation_policy"] = {**field["derivation_policy"],
                "raw_source_access": "selected_daily_columns_yearly_decode_permitted",
                "noncontributors": "mask_to_null_no_event_quote_lookup_or_contribution"}
        for dependency in field["source_dependencies"]:
            if dependency["role"] == "stored_history":
                dependency["policy"] = {"schema_version": "magnitude_eligible_history_source.v1",
                    "recipe_schema": "eligible_history_features.v1",
                    "compatibility": "exact_installed_context_source_policy_month_cutoffs",
                    "rebuild": "explicit_separate_materialization_only"}
            elif dependency["role"] == "daily_market":
                dependency["policy"] = {**dependency["policy"],
                    "supported_contract_hash": _DAILY_HASH,
                    "normalization_ranges": {k: list(v) for k, v in _DAILY_RANGES.items()},
                    "observable_violation": "refuse", "stored_zero": "preserve_status_only_current",
                    "raw_source_access": "selected_columns_yearly_decode_then_causal_selection"}
    return fields


def recipe(subset):
    """New source identity, not a new feature/search axis or fitting authority."""
    result = v2.recipe(subset)
    fields = feature_dictionary(subset)
    return {**result, "schema_version": "research_magnitude_panel.v3", "dictionary": fields,
            "dictionary_hash": content_hash(fields)}


FEATURES = tuple(f["name"] for f in feature_dictionary("HMO"))
_base = replace(v2.CONTRACT, contract_id=TABLE + ".v3", semantic_version="3.0.0",
    columns=(*v2.CONTRACT.columns,
        ColumnContract(name="current_quote_status", physical_type="string", nullable=False),
        ColumnContract(name="has_implied_quote", physical_type="int64", nullable=False)))
CONTRACT = replace(_base, definition_hash=table_contract_hash(_base))


def admit_columns(columns):
    if (type(columns) not in (list, tuple) or not columns
            or any(type(name) is not str or name not in (*FEATURES, "has_implied_quote") for name in columns)
            or len(columns) != len(set(columns))):
        raise fail("HOLDOUT_ACCESS_DENIED", "qualified panel requires explicit closed feature projection")
