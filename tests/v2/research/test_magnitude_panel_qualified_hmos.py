"""HMOS season declaration pins: frozen v3 hashes, appended fields, v4 contract, admission."""
import pytest

from engine.v2.contracts.data import ColumnContract
from engine.v2.data.errors import DataError
from engine.v2.research import magnitude_panel_qualified_table as table

_PINNED_DICTIONARY_HASHES = {
    "H": "sha256:65851a015c584c3670fb035080fef6dd5652900297f7fbcb80b87ee537bd7147",
    "HM": "sha256:357e272622d79d8d4335881627b5821dd14958e5f7d108f661a08f731a5f0b08",
    "HMO": "sha256:a4d8f7f5c97c671fd27bf91f94e4845d98079072f2bd7e7cabf4369890485133",
    "MO-": "sha256:1c465790de5bf578fadb5cf2ab8e5859f6782e216804007362a9823fb882411e",
    "M": "sha256:15fe24643ba0013501e5dd0bc5807d1459db83f75e9451cc2ec4d09a34c2d726",
}
_PINNED_V3_DEFINITION_HASH = "sha256:db79adb50b3efd4e9eea61407e433b98a12d2a3262e0526ff91db97afdc6dacb"

# Literal season assignments; not implementation constants.
_SEASON = [
    ("season_heat_mean_abs_5", "float64", "percent", True, "null"),
    ("season_heat_mean_abs_20", "float64", "percent", True, "null"),
    ("season_heat_mean_abs_60", "float64", "percent", True, "null"),
    ("season_heat_count_20", "int64", "count", False, "zero_count"),
    ("season_heat_ratio_20_252", "float64", "ratio", True, "null"),
]


def _expected_season_field(name, physical_type, unit, nullable, on_missing):
    return {"name": name, "block": "season", "source_alias": name, "native_output_alias": name,
        "physical_type": physical_type, "unit": unit, "nullable": nullable, "on_missing": on_missing,
        "source_dependencies": [{"role": "season_population", "policy": {
            "schema_version": "season_heat_population.v1", "reader": "season_population",
            "causal": "available_as_of_date <= decision_date", "own_ticker": "excluded"}}],
        "derivation_policy": {"schema_version": "season_heat.v1",
            "function": "compute_season_heat", "output": name}}


def _expected_season_fields():
    return [_expected_season_field(*assignment) for assignment in _SEASON]


@pytest.mark.parametrize("subset", sorted(_PINNED_DICTIONARY_HASHES))
def test_existing_subset_hashes_are_frozen(subset):
    assert table.recipe(subset)["dictionary_hash"] == _PINNED_DICTIONARY_HASHES[subset]
    assert table.recipe(subset)["schema_version"] == "research_magnitude_panel.v3"


def test_v3_contract_definition_hash_is_frozen():
    assert table.CONTRACT.definition_hash == _PINNED_V3_DEFINITION_HASH
    assert table.CONTRACT.contract_id.endswith(".v3")


def test_hmos_dictionary_is_hmo_plus_season_appended_last():
    hmos = table.feature_dictionary("HMOS")
    assert hmos == table.feature_dictionary("HMO") + _expected_season_fields()
    assert [field["name"] for field in hmos[-5:]] == list(table.SEASON)
    for field in hmos:
        assert set(field) == {"name", "block", "source_alias", "native_output_alias", "physical_type",
            "unit", "nullable", "on_missing", "source_dependencies", "derivation_policy"}
    assert hmos is not table.feature_dictionary("HMO")


def test_hmos_hash_differs_from_every_pinned_subset():
    hmos_hash = table.recipe("HMOS")["dictionary_hash"]
    assert hmos_hash not in set(_PINNED_DICTIONARY_HASHES.values())
    assert table.feature_dictionary("HMO") == table.recipe("HMO")["dictionary"]


def test_hmos_recipe_model_columns_and_quote():
    hmos = table.recipe("HMOS")
    names = [field["name"] for field in hmos["dictionary"]]
    assert hmos["subset"] == "HMOS"
    assert hmos["model_columns"] == names + ["has_implied_quote"]
    assert hmos["model_columns"] == table.recipe("HMO")["model_columns"][:-1] \
        + list(table.SEASON) + ["has_implied_quote"]
    assert hmos["quote_indicator"] == table.recipe("HMO")["quote_indicator"]
    assert hmos["quote_indicator"]["name"] == "has_implied_quote"
    hmo = table.recipe("HMO")
    assert hmo["model_columns"] == [field["name"] for field in hmo["dictionary"]] + ["has_implied_quote"]
    assert len(hmo["model_columns"]) == 43


def test_features_hmos_tuple():
    assert table.FEATURES_HMOS == (*table.FEATURES, *table.SEASON)


def test_season_field_types_and_nullability():
    fields = {field["name"]: field for field in table.feature_dictionary("HMOS")[-5:]}
    assert fields["season_heat_count_20"]["physical_type"] == "int64"
    assert fields["season_heat_count_20"]["nullable"] is False
    for name in ("season_heat_mean_abs_5", "season_heat_mean_abs_20", "season_heat_mean_abs_60",
                 "season_heat_ratio_20_252"):
        assert fields[name]["physical_type"] == "float64"
        assert fields[name]["nullable"] is True


def test_contract_v4_extends_v3_with_season_columns_in_order():
    season_columns = [ColumnContract(name=name, physical_type=physical_type, nullable=nullable,
        unit=unit, null_policy=on_missing)
        for name, physical_type, unit, nullable, on_missing in _SEASON]
    assert table.CONTRACT_V4.columns == (*table.CONTRACT.columns, *season_columns)
    assert table.CONTRACT_V4.contract_id == table.TABLE + ".v4"
    assert table.CONTRACT_V4.semantic_version == "4.0.0"
    assert table.CONTRACT_V4.definition_hash != table.CONTRACT.definition_hash
    assert table.CONTRACT.contract_id == table.TABLE + ".v3"


@pytest.mark.parametrize("columns", [
    ["season_heat_count_20"],
    [*table.FEATURES, "has_implied_quote", "season_heat_mean_abs_5"],
    ["season_heat_mean_abs_5", "season_heat_mean_abs_20", "season_heat_mean_abs_60",
     "season_heat_count_20", "season_heat_ratio_20_252"]])
def test_admit_columns_rejects_season_by_default(columns):
    with pytest.raises(DataError) as raised:
        table.admit_columns(columns)
    assert raised.value.code == "HOLDOUT_ACCESS_DENIED"


def test_admit_columns_admits_season_only_with_flag():
    columns = [*table.FEATURES_HMOS, "has_implied_quote"]
    assert table.admit_columns(columns, season=True) is None
    assert table.admit_columns(list(table.SEASON), season=True) is None


@pytest.mark.parametrize("columns", [
    [], "n_prior", ["n_prior", "n_prior"], ["not_a_feature"], [None], ["n_prior", ""]])
def test_admit_columns_still_refuses_bad_projections(columns):
    for kwargs in ({}, {"season": True}):
        with pytest.raises(DataError) as raised:
            table.admit_columns(columns, **kwargs)
        assert raised.value.code == "HOLDOUT_ACCESS_DENIED"
