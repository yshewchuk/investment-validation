"""HMOSXL/HMOSXLP derived-recipe declarations and stored-panel separation."""
import hashlib

import pytest

from engine.v2.data.errors import DataError
from engine.v2.foundation import canonical_json
from engine.v2.research import magnitude_panel_qualified_table as table
from engine.v2.research.derived_features import SET_NAMES, derived_names

_FROZEN_RECIPE = "c6767cbdc4b7f4ba335e76bd5167a7837a51fd362bcb5a83231c560374211f8d"
_FROZEN_DICTIONARIES = {
    "H": "sha256:65851a015c584c3670fb035080fef6dd5652900297f7fbcb80b87ee537bd7147",
    "HM": "sha256:357e272622d79d8d4335881627b5821dd14958e5f7d108f661a08f731a5f0b08",
    "HMO": "sha256:a4d8f7f5c97c671fd27bf91f94e4845d98079072f2bd7e7cabf4369890485133",
    "MO-": "sha256:1c465790de5bf578fadb5cf2ab8e5859f6782e216804007362a9823fb882411e",
    "M": "sha256:15fe24643ba0013501e5dd0bc5807d1459db83f75e9451cc2ec4d09a34c2d726",
    "HMOS": "sha256:ab953ea4b9370f5d848871aac7bf827040c36e021515f7cf85af63c267f94b2e",
    "HMOSX": "sha256:d65ffe29ef656ae56fee5817dccf0464391b926af35e4186369dd9e3b2c49606",
}


def _sha256(document):
    encoded = canonical_json(document)
    return hashlib.sha256(encoded.encode() if isinstance(encoded, str) else encoded).hexdigest()


def _base():
    return table.feature_dictionary("HMOSX")


def test_frozen_hmosx_recipe_and_subset_dictionaries():
    assert _sha256(table.recipe("HMOSX")) == _FROZEN_RECIPE
    assert "base_subset" not in table.recipe("HMOSX")
    assert "derivation" not in table.recipe("HMOSX")
    for subset, digest in _FROZEN_DICTIONARIES.items():
        assert table.recipe(subset)["dictionary_hash"] == digest


def test_hmosxl_adds_only_log_descriptors():
    base, xl = _base(), table.feature_dictionary("HMOSXL")
    derived = xl[len(base):]
    assert xl[:len(base)] == base
    assert len(derived) == 39
    assert [field["name"] for field in derived] == list(derived_names("HMOSXL"))
    recipe = table.recipe("HMOSXL")
    assert recipe["base_subset"] == "HMOSX"
    assert recipe["subset"] == "HMOSXL"
    assert recipe["model_columns"] == (table.recipe("HMOSX")["model_columns"][:-1]
        + list(derived_names("HMOSXL")) + ["has_implied_quote"])
    assert recipe["dictionary_hash"] not in {
        table.recipe("HMOSX")["dictionary_hash"],
        table.recipe("HMOSXLP")["dictionary_hash"]}


def test_hmosxlp_appends_products_after_the_log_descriptors():
    base = _base()
    xl_derived = table.feature_dictionary("HMOSXL")[len(base):]
    derived = table.feature_dictionary("HMOSXLP")[len(base):]
    assert len(derived) == 67
    assert derived[:39] == xl_derived
    assert [field["name"] for field in derived] == list(derived_names("HMOSXLP"))
    units = {field["name"]: field["unit"] for field in base}
    product = next(field for field in derived
        if field["name"] == "implied_move_x_mean_prior_abs_move")
    assert [dep["name"] for dep in product["source_dependencies"]] == [
        "implied_move", "mean_prior_abs_move"]
    assert all(dep["role"] == "panel_column" for dep in product["source_dependencies"])
    assert product["derivation_policy"]["function"] == "product"
    assert product["unit"] == f"{units['implied_move']}*{units['mean_prior_abs_move']}"
    log = next(field for field in derived if field["name"] == "log1p_implied_move")
    assert log["derivation_policy"]["function"] == "log1p"
    assert log["unit"] == f"log1p({units['implied_move']})"
    for field in derived:
        assert field["physical_type"] == "float64"
        assert field["nullable"] is True
        assert field["on_missing"] == "null"


def test_hmosxlp_descriptors_are_exact_literals():
    base = _base()
    units = {field["name"]: field["unit"] for field in base}
    derived = table.feature_dictionary("HMOSXLP")[len(base):]
    by_name = {field["name"]: field for field in derived}
    assert by_name["log1p_implied_move"] == {
        "name": "log1p_implied_move", "block": "derived",
        "source_alias": "log1p_implied_move", "native_output_alias": "log1p_implied_move",
        "physical_type": "float64", "unit": f"log1p({units['implied_move']})",
        "nullable": True, "on_missing": "null",
        "source_dependencies": [{"role": "panel_column", "name": "implied_move"}],
        "derivation_policy": {"schema_version": "derived_feature.v1", "function": "log1p",
            "sources": ["implied_move"], "output": "log1p_implied_move"}}
    assert by_name["implied_move_x_mean_prior_abs_move"] == {
        "name": "implied_move_x_mean_prior_abs_move", "block": "derived",
        "source_alias": "implied_move_x_mean_prior_abs_move",
        "native_output_alias": "implied_move_x_mean_prior_abs_move",
        "physical_type": "float64",
        "unit": f"{units['implied_move']}*{units['mean_prior_abs_move']}",
        "nullable": True, "on_missing": "null",
        "source_dependencies": [{"role": "panel_column", "name": "implied_move"},
            {"role": "panel_column", "name": "mean_prior_abs_move"}],
        "derivation_policy": {"schema_version": "derived_feature.v1", "function": "product",
            "sources": ["implied_move", "mean_prior_abs_move"],
            "output": "implied_move_x_mean_prior_abs_move"}}
    for field in derived:
        assert field["derivation_policy"]["sources"]
        assert field["derivation_policy"]["sources"] == [dep["name"]
            for dep in field["source_dependencies"]]
    with pytest.raises(DataError) as raised:
        table.panel_recipe("HMOSXL")
    assert "HMOSXL" in str(raised.value)


def test_derived_sources_are_hmosx_model_columns():
    columns = set(table.recipe("HMOSX")["model_columns"])
    for subset in SET_NAMES:
        for field in table.feature_dictionary(subset)[len(_base()):]:
            assert {dep["name"] for dep in field["source_dependencies"]} <= columns
            assert set(field["derivation_policy"]["sources"]) <= columns


def test_derived_dictionary_hash_is_deterministic():
    for subset in SET_NAMES:
        assert table.recipe(subset)["dictionary_hash"] == table.recipe(subset)["dictionary_hash"]


def test_panel_recipe_is_stored_panel_only():
    assert table.panel_recipe("HMOSX") == table.recipe("HMOSX")
    for subset in SET_NAMES:
        with pytest.raises(DataError) as raised:
            table.panel_recipe(subset)
        assert raised.value.code == "CONTRACT_MISMATCH"
