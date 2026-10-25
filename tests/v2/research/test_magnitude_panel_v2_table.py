"""Independent ordered v2 descriptor/projection and immutable-v1 oracles."""
import copy
import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.manifests import table_contract_hash
from engine.v2.foundation import content_hash, to_document
from engine.v2.research import magnitude_panel_table as v1
from engine.v2.research import magnitude_panel_v2_table as v2
from tests.v2.research.magnitude_panel_policy_support import (
    EXPECTED_SOURCES,
    EXPECTED_SPECIAL,
    forbid_policy_io,
)

# Literal order/block/unit/source-role/derivation assignment; not implementation constants.
_FIELDS = [
    ("n_prior", "history", "count", "stored_history", None),
    ("mean_prior_move", "history", "percent", "stored_history", None),
    ("mean_prior_abs_move", "history", "percent", "stored_history", None),
    ("ema2_prior_move", "history", "percent", "stored_history", None),
    ("ema2_prior_abs_move", "history", "percent", "stored_history", None),
    ("ema4_prior_move", "history", "percent", "stored_history", None),
    ("ema4_prior_abs_move", "history", "percent", "stored_history", None),
    ("ema8_prior_move", "history", "percent", "stored_history", None),
    ("ema8_prior_abs_move", "history", "percent", "stored_history", None),
    ("ema12_prior_move", "history", "percent", "stored_history", None),
    ("ema12_prior_abs_move", "history", "percent", "stored_history", None),
    ("signed_streak", "prior_events", "count", "prior_outcomes", "runup"),
    ("ema12r_abs", "prior_events", "percent", "stored_history", "special"),
    ("dist_high", "runup", "percent", "runup_prices", "runup"),
    ("dist_ema", "runup", "percent", "runup_prices", "runup"),
    ("abs_dist_ema", "runup", "percent", "runup_prices", "special"),
    ("ret5", "runup", "percent", "runup_prices", "runup"),
    ("ret10", "runup", "percent", "runup_prices", "runup"),
    ("ret20", "runup", "percent", "runup_prices", "runup"),
    ("spy_ret21", "spy", "percent", "market_context", "regime"),
    ("spy_ret63", "spy", "percent", "market_context", "regime"),
    ("spy_ret252", "spy", "percent", "market_context", "regime"),
    ("spy_dd252", "spy", "percent", "market_context", "regime"),
    ("spy_vol5", "spy", "percent", "market_context", "regime"),
    ("spy_vol20", "spy", "percent", "market_context", "regime"),
    ("spy_vol60", "spy", "percent", "market_context", "regime"),
    ("spy_vol252", "spy", "percent", "market_context", "regime"),
    ("spy_vol20_rel252", "spy", "ratio", "market_context", "regime"),
    ("implied_move", "daily_market", "percent", "daily_market", "special"),
    ("iv10", "daily_market", "percent", "daily_market", None),
    ("iv30", "daily_market", "percent", "daily_market", None),
    ("exern_iv10", "daily_market", "percent", "daily_market", None),
    ("exern_iv30", "daily_market", "percent", "daily_market", None),
    ("iee", "daily_market", "unknown", "daily_market", None),
    ("skew", "daily_market", "unknown", "daily_market", None),
    ("contango", "daily_market", "unknown", "daily_market", None),
    ("fwd90_30", "daily_market", "percent", "daily_market", None),
    ("fexern90_30", "daily_market", "percent", "daily_market", None),
    ("rvol30", "daily_market", "percent", "daily_market", None),
    ("spot", "daily_market", "usd", "daily_market", None),
    ("mcap_log", "daily_market", "log_usd", "daily_market", "special"),
    ("mean_prior_or_implied", "implied_history", "percent", "daily_market event_metadata", "special"),
]


def _expected_dictionary(subset):
    result = []
    for name, block, unit, roles, operation in _FIELDS[:{"H": 11, "HM": 28, "HMO": 42}[subset]]:
        derived = None
        if operation == "special":
            derived = copy.deepcopy(EXPECTED_SPECIAL[name])
        elif operation:
            function = "runup_math.add_runup_features" if operation == "runup" else "regime.add_regime_features"
            derived = {"schema_version": "magnitude_existing_math.v1",
                "function": "engine.v2.features." + function, "output": name}
        result.append({"name": name, "block": block, "source_alias": name,
            "native_output_alias": "im" if name == "implied_move" else name,
            "physical_type": "int64" if name == "n_prior" else "float64", "unit": unit,
            "nullable": name != "n_prior", "on_missing": "zero_count" if name == "n_prior" else "null",
            "source_dependencies": [{"role": role, "policy": copy.deepcopy(EXPECTED_SOURCES[role])}
                for role in roles.split()], "derivation_policy": derived})
    return result


def _stdlib_hash(document):
    return "sha256:" + hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.mark.parametrize("subset,width", [("H", 11), ("HM", 28), ("HMO", 42)])
def test_complete_ordered_descriptors_and_preserved_v1_fields(subset, width):
    dictionary = v2.feature_dictionary(subset)
    assert dictionary == _expected_dictionary(subset)
    assert len(dictionary) == width
    base = {item["name"]: item for item in v1.feature_dictionary(subset)}
    for item in dictionary:
        assert len(item) == 10
        if item["name"] in base:
            assert {key: item[key] for key in base[item["name"]]} == base[item["name"]]
        roles = [entry["role"] for entry in item["source_dependencies"]]
        assert roles == sorted(set(roles))


@pytest.mark.parametrize("subset,width", [("H", 11), ("HM", 28), ("HMO", 43)])
def test_closed_recipe_and_raw_projection(subset, width):
    dictionary = _expected_dictionary(subset)
    quote = {"name": "has_implied_quote", "source": "daily_market.quote_status", "true_status": "present",
        "false_statuses": ["zero", "absent", "unavailable"], "position": "after_numeric_dictionary"}
    names = [item["name"] for item in dictionary]
    expected = {"schema_version": "research_magnitude_panel.v2", "subset": subset,
        "dictionary": dictionary, "dictionary_hash": _stdlib_hash(dictionary), "imputation": "none",
        "missing_indicators": "family_recipe_train_only_selected_columns",
        "quote_indicator": quote if subset == "HMO" else None,
        "model_columns": names + (["has_implied_quote"] if subset == "HMO" else [])}
    assert v2.recipe(subset) == expected
    assert len(expected["model_columns"]) == width
    assert "has_implied_quote" not in names
    assert v2.recipe(subset)["dictionary_hash"] != v1.recipe(subset)["dictionary_hash"]


def test_exact_native_contract_inheritance_and_numeric_columns():
    assert v2.CONTRACT.contract_id == "research_magnitude_panel.v2"
    assert v2.CONTRACT.semantic_version == "2.0.0"
    assert v2.CONTRACT.table_name == "research_magnitude_panel"
    assert v2.CONTRACT.primary_key == ("event_id", "context_id", "horizon_sessions")
    for field in fields(v1.CONTRACT):
        if field.name not in {"contract_id", "semantic_version", "columns", "definition_hash"}:
            assert getattr(v2.CONTRACT, field.name) == getattr(v1.CONTRACT, field.name)
    base = {column.name: column for column in v1.CONTRACT.columns}
    expected_names = list(v1.METADATA) + [field[0] for field in _FIELDS]
    assert [column.name for column in v2.CONTRACT.columns] == expected_names
    for column in v2.CONTRACT.columns:
        original = {"abs_dist_ema": "dist_ema", "mean_prior_or_implied": "implied_move"}.get(column.name, column.name)
        assert column == replace(base[original], name=column.name)
        if original == column.name:
            assert column is base[original]
    assert v2.CONTRACT.definition_hash == table_contract_hash(v2.CONTRACT)
    assert v2.CONTRACT.definition_hash != v1.CONTRACT.definition_hash
    assert v2.CONTRACT.definition_hash != v2.recipe("HMO")["dictionary_hash"]


class _StringSubclass(str):
    pass


@pytest.mark.parametrize("bad", [None, True, 1, 1.0, {}, [], (), b"H", "", "h", "HM ", "latest", _StringSubclass("H")])
@pytest.mark.parametrize("factory", [v2.feature_dictionary, v2.recipe])
def test_invalid_subsets_refuse_before_policy_access(bad, factory, monkeypatch):
    def forbidden():
        raise AssertionError("invalid subset reached policy access")

    monkeypatch.setattr(v2, "policy_definitions", forbidden)
    with pytest.raises(DataError) as error:
        factory(bad)
    assert error.value.code == "CONTRACT_MISMATCH"
    assert error.value.problem.message == "unknown v2 panel feature subset"


@pytest.mark.parametrize("factory", [v2.feature_dictionary, v2.recipe])
def test_explicit_subset_required(factory):
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        factory()


def test_detached_nested_containers_and_no_v1_mutation():
    before = {subset: copy.deepcopy(v1.recipe(subset)) for subset in ("H", "HM", "HMO")}
    first = v2.recipe("HMO")
    original = copy.deepcopy(first)
    first["dictionary"][0]["source_dependencies"][0]["policy"]["rebuild"] = "poison"
    assert first["dictionary"][1] == original["dictionary"][1]
    first["dictionary"][-1]["derivation_policy"]["eligibility"]["embargo_days"] = 0
    first["quote_indicator"]["false_statuses"].clear()
    first["model_columns"].clear()
    assert v2.recipe("HMO") == original
    dictionary = v2.feature_dictionary("HMO")
    dictionary[0]["source_dependencies"][0]["policy"]["rebuild"] = "poison"
    dictionary[-1]["derivation_policy"]["eligibility"]["embargo_days"] = 0
    assert dictionary[1] == original["dictionary"][1]
    assert v2.feature_dictionary("HMO") == original["dictionary"]
    assert {subset: v1.recipe(subset) for subset in before} == before


@pytest.mark.parametrize("subset", ["H", "HM", "HMO"])
@pytest.mark.parametrize("factory", [v2.feature_dictionary, v2.recipe])
def test_factories_do_not_read_sources_or_admit_rows(subset, factory, monkeypatch):
    with monkeypatch.context() as guarded:
        forbid_policy_io(guarded)
        with pytest.raises(AssertionError, match="file/database I/O"):
            Path("synthetic-unopened-file").read_text()
        actual = factory(subset)
    assert (actual if isinstance(actual, list) else actual["dictionary"]) == _expected_dictionary(subset)
    assert v2.__all__ == ["CONTRACT", "feature_dictionary", "recipe"]
    assert not hasattr(v2, "validate_rows") and not hasattr(v2, "build_context")


@pytest.mark.parametrize("subset,expected", [
    ("H", "sha256:baafd78e86ae6583c561e0ecabf9735ad612d2cd2fe4915e1ef36096ca90075f"), ("HM", "sha256:cf021a739db8007593e7401f2bfe3744ab69795401ff43908605c95de9684df9"), ("HMO", "sha256:21a88956f12177613742de95834c7f310216825e01da1f2d813dba3589991505")])
def test_literal_dictionary_ids_and_policy_order_sensitivity(subset, expected):
    dictionary = v2.feature_dictionary(subset)
    assert _stdlib_hash(dictionary) == expected
    assert v2.recipe(subset)["dictionary_hash"] == expected
    dictionary[0]["source_dependencies"][0]["policy"]["rebuild"] = "allowed"
    assert content_hash(dictionary) != expected
    dictionary = v2.feature_dictionary(subset)
    dictionary.reverse()
    assert content_hash(dictionary) != expected


def test_literal_contract_id_and_native_definition_hash_exclusion():
    base = {column.name: column for column in v1.CONTRACT.columns}
    aliases = {"abs_dist_ema": "dist_ema", "mean_prior_or_implied": "implied_move"}
    names = list(v1.METADATA) + [field[0] for field in _FIELDS]
    expected = replace(v1.CONTRACT, contract_id="research_magnitude_panel.v2", semantic_version="2.0.0",
        columns=tuple(replace(base[aliases.get(name, name)], name=name) for name in names))
    document = to_document(expected)
    document.pop("definition_hash")
    assert _stdlib_hash(document) == "sha256:e174bb1ec49cea0c214fadda7197ab43aa56b47e07529cd303126c0439565194"
    assert v2.CONTRACT.definition_hash == "sha256:e174bb1ec49cea0c214fadda7197ab43aa56b47e07529cd303126c0439565194"
    assert table_contract_hash(replace(v2.CONTRACT, definition_hash="ignored")) == v2.CONTRACT.definition_hash


@pytest.mark.parametrize("role,field,value,changed", [
    ("market_context", "instrument", "SPY", {"HM", "HMO"}),
    ("stored_history", "rebuild", "allowed", {"H", "HM", "HMO"}),
    ("event_metadata", "prefix_start", "history_start", {"HMO"}),
])
def test_inline_source_content_changes_exactly_dependent_identities(role, field, value, changed, monkeypatch):
    before = {subset: v2.recipe(subset) for subset in ("H", "HM", "HMO")}
    policies = v2.policy_definitions()
    policies["source_policies"][role][field] = value
    monkeypatch.setattr(v2, "policy_definitions", lambda: copy.deepcopy(policies))
    for subset, original in before.items():
        actual = v2.recipe(subset)
        assert (actual["dictionary_hash"] != original["dictionary_hash"]) == (subset in changed)
        assert (content_hash(actual) != content_hash(original)) == (subset in changed)


def test_unchanged_v1_historical_literal_identities():
    assert v1.CONTRACT.definition_hash == "sha256:7de4dc1290e0fd6b5cddfc35927885adb0f380d67cd8bfb8c4475911c45b6259"
    assert [content_hash(v1.recipe(subset)) for subset in ("H", "HM", "HMO")] == [
        "sha256:f6ddf6fb0b7c579449345154e078d0215cf0017be021dbb049949ec4e710de7c",
        "sha256:7cb05c7449d6b68c72b86bb0493c494b2f5812a7a1839fb2d9e926f1b03797de",
        "sha256:f8dc8e8472436a46f6c8d8255c80cbfe28bf13291fb5e790a38f7dc8406ff4b2"]


def test_contract_constructor_replays_through_native_hash(monkeypatch):
    hashed = []

    def native(contract):
        hashed.append(contract)
        return table_contract_hash(contract)

    monkeypatch.setattr(v2, "table_contract_hash", native)
    replayed = v2._contract()
    assert replayed == v2.CONTRACT
    document = to_document(replayed)
    document.pop("definition_hash")
    assert _stdlib_hash(document) == "sha256:e174bb1ec49cea0c214fadda7197ab43aa56b47e07529cd303126c0439565194"
    assert hashed == [replace(v2.CONTRACT, definition_hash="sha256:" + "0" * 64)]
