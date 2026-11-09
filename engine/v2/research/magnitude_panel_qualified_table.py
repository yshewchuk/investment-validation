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
SEASON = ("season_heat_mean_abs_5", "season_heat_mean_abs_20", "season_heat_mean_abs_60",
          "season_heat_count_20", "season_heat_ratio_20_252")


def _season_fields():
    """Detached season descriptors; declared only, appended last, nothing computed."""
    fields = []
    for name in SEASON:
        count = name == "season_heat_count_20"
        fields.append({"name": name, "block": "season", "source_alias": name,
            "native_output_alias": name, "physical_type": "int64" if count else "float64",
            "unit": "count" if count else ("ratio" if name.endswith("_ratio_20_252") else "percent"),
            "nullable": not count, "on_missing": "zero_count" if count else "null",
            "source_dependencies": [{"role": "season_population", "policy": {
                "schema_version": "season_heat_population.v1", "reader": "season_population",
                "causal": "available_as_of_date <= decision_date", "own_ticker": "excluded"}}],
            "derivation_policy": {"schema_version": "season_heat.v1",
                "function": "compute_season_heat", "output": name}})
    return fields

_DAILY_HASH = "sha256:ae94d7433748268b7cc7269e368221c983eb63e42547105dba3e89acd58b4e8b"
# Frozen native/legacy ORATS normalization limits, not comparator masks.
_DAILY_RANGES = {"implied_move": (0., 100.), "iv10": (0., 500.), "iv30": (0., 500.),
    "exern_iv10": (0., 500.), "exern_iv30": (0., 500.), "rvol30": (0., 500.),
    "skew": (-50., 50.), "contango": (-100., 100.), "fwd90_30": (0., 500.),
    "fexern90_30": (0., 500.), "iee": (-50., 50.), "spot": (0., 1e6)}


def _base_subset(subset):
    return "HMO" if subset in ("MO-", "M", "HMOS") else subset


def _dictionary(subset):
    history_free = subset in ("MO-", "M")
    fields = v2.feature_dictionary(_base_subset(subset))
    if history_free:
        # Preserve frozen HMO descriptors, then remove all features whose registered derivation uses earnings history.
        fields = fields[12:28] if subset == "M" else fields[12:]
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
    if history_free:
        fields = [field for field in fields
            if field["name"] != "mean_prior_or_implied"
            and not ({dependency["role"] for dependency in field["source_dependencies"]}
                     & {"stored_history", "prior_outcomes"})]
    return fields


def feature_dictionary(subset):
    """Keep numeric order/units, explicitly replace incompatible history policy."""
    return _dictionary(subset) + (_season_fields() if subset == "HMOS" else [])


def recipe(subset):
    """New source identity, not a new feature/search axis or fitting authority."""
    history_free = subset in ("MO-", "M")
    result = v2.recipe(_base_subset(subset))
    fields = feature_dictionary(subset)
    quote = None if history_free else result["quote_indicator"]
    return {**result, "schema_version": "research_magnitude_panel.v3", "subset": subset,
            "dictionary": fields, "dictionary_hash": content_hash(fields),
            "quote_indicator": quote,
            "model_columns": [field["name"] for field in fields] + ([quote["name"]] if quote else [])}


FEATURES = tuple(f["name"] for f in feature_dictionary("HMO"))
FEATURES_HMOS = (*FEATURES, *SEASON)
_base = replace(v2.CONTRACT, contract_id=TABLE + ".v3", semantic_version="3.0.0",
    columns=(*v2.CONTRACT.columns,
        ColumnContract(name="current_quote_status", physical_type="string", nullable=False),
        ColumnContract(name="has_implied_quote", physical_type="int64", nullable=False)))
_contract_v4 = replace(_base, contract_id=TABLE + ".v4", semantic_version="4.0.0",
    columns=(*_base.columns, *(ColumnContract(name=field["name"], physical_type=field["physical_type"],
        nullable=field["nullable"], unit=field["unit"], null_policy=field["on_missing"])
        for field in _season_fields())))
CONTRACT_V4 = replace(_contract_v4, definition_hash=table_contract_hash(_contract_v4))
CONTRACT = CONTRACT_V4  # Slice 5 installs v4; no compatibility reader for sealed v3 panels.


def admit_columns(columns, *, season=False):
    allowed = (*FEATURES, "has_implied_quote", *(SEASON if season else ()))
    if (type(columns) not in (list, tuple) or not columns
            or any(type(name) is not str or name not in allowed for name in columns)
            or len(columns) != len(set(columns))):
        raise fail("HOLDOUT_ACCESS_DENIED", "qualified panel requires explicit closed feature projection")
