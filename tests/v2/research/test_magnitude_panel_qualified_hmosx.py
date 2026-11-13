"""HMOSX declaration pins: appended x fields, v5 contract, extended admission."""
import copy
import json
from datetime import date

import pytest

from engine.v2.contracts.data import ColumnContract
from engine.v2.data.errors import DataError
from engine.v2.data.manifests import table_contract_hash
from engine.v2.foundation import content_hash
from engine.v2.research import magnitude_panel_qualified as builder
from engine.v2.research import magnitude_panel_qualified_table as table
from engine.v2.research import magnitude_panel_qualified_x as x
from engine.v2.research.tail_history import compute_tail_history
from experiments import size_panel
from tests.v2.research.test_magnitude_panel_qualified import panel_data

_PINNED_DICTIONARY_HASHES = {
    "H": "sha256:65851a015c584c3670fb035080fef6dd5652900297f7fbcb80b87ee537bd7147",
    "HM": "sha256:357e272622d79d8d4335881627b5821dd14958e5f7d108f661a08f731a5f0b08",
    "HMO": "sha256:a4d8f7f5c97c671fd27bf91f94e4845d98079072f2bd7e7cabf4369890485133",
    "MO-": "sha256:1c465790de5bf578fadb5cf2ab8e5859f6782e216804007362a9823fb882411e",
    "M": "sha256:15fe24643ba0013501e5dd0bc5807d1459db83f75e9451cc2ec4d09a34c2d726",
}

# Literal assignments; not implementation constants.
_X_TABLE = [
    ("tail_max_abs", "tail", "float64", "percent", True, "null"),
    ("tail_std_abs", "tail", "float64", "percent", True, "null"),
    ("tail_p75_abs", "tail", "float64", "percent", True, "null"),
    ("tail_p90_abs", "tail", "float64", "percent", True, "null"),
    ("tail_share_gt10", "tail", "float64", "fraction", True, "null"),
    ("tail_max4_abs", "tail", "float64", "percent", True, "null"),
    ("tail_std_over_mean", "tail", "float64", "ratio", True, "null"),
    ("tail_max_over_mean", "tail", "float64", "ratio", True, "null"),
    ("iv10_30", "ivdyn", "float64", "ratio", True, "null"),
    ("iv_rv", "ivdyn", "float64", "ratio", True, "null"),
    ("ex_ratio", "ivdyn", "float64", "ratio", True, "null"),
    ("imp_chg5", "ivdyn", "float64", "ratio", True, "null"),
    ("imp_chg10", "ivdyn", "float64", "ratio", True, "null"),
    ("iv30_chg5", "ivdyn", "float64", "ratio", True, "null"),
    ("iv30_chg20", "ivdyn", "float64", "ratio", True, "null"),
    ("iv10_chg5", "ivdyn", "float64", "ratio", True, "null"),
    ("own_rvol5", "ownvol", "float64", "percent", True, "null"),
    ("own_rvol10", "ownvol", "float64", "percent", True, "null"),
    ("own_rvol60", "ownvol", "float64", "percent", True, "null"),
    ("own_absret1", "ownvol", "float64", "percent", True, "null"),
    ("own_rvol5_60", "ownvol", "float64", "ratio", True, "null"),
    ("cal_amc", "calendar", "int64", "flag", False, "refuse"),
    ("cal_weekday", "calendar", "int64", "weekday", False, "refuse"),
    ("cal_month", "calendar", "int64", "month", False, "refuse"),
    ("cal_days_since_prior", "calendar", "int64", "days", True, "null"),
]

_SOURCE_DEPS = {
    "tail": ("stored_history", {
        "schema_version": "tail_history_source.v1", "reader": "history_json_contributors",
        "causal": "available_as_of_date <= decision_date", "own_ticker": "only"}),
    "ivdyn": ("daily_market", {
        "schema_version": "iv_dynamics_source.v1", "reader": "daily_market_surface_rows",
        "causal": "date <= source_session_ceiling",
        "stale_reference": "missing_if_reference_older_than_2k_plus_4_calendar_days"}),
    "ownvol": ("price_history", {
        "schema_version": "own_vol_source.v1", "reader": "price_history_close_adj",
        "causal": "date <= source_session_ceiling",
        "stale_last_close": "missing_if_older_than_5_calendar_days"}),
    "calendar": ("event_calendar", {
        "schema_version": "calendar_features_source.v1",
        "reader": "stored_event_session_and_date",
        "causal": "event_date_and_session_define_forecast_clock"}),
}
_DERIVATIONS = {
    "tail": ("tail_history.v1", "compute_tail_history"),
    "ivdyn": ("iv_dynamics.v1", "compute_iv_dynamics"),
    "ownvol": ("own_vol.v1", "compute_own_vol"),
    "calendar": ("calendar_features.v1", "calendar_features"),
}


def _expected_field(name, block, physical_type, unit, nullable, on_missing):
    role, policy = _SOURCE_DEPS[block]
    dependencies = [{"role": role, "policy": dict(policy)}]
    if name == "cal_days_since_prior":
        tail_role, tail_policy = _SOURCE_DEPS["tail"]
        dependencies.append({"role": tail_role, "policy": dict(tail_policy)})
    schema_version, function = _DERIVATIONS[block]
    return {"name": name, "block": block, "source_alias": name,
        "native_output_alias": name, "physical_type": physical_type, "unit": unit,
        "nullable": nullable, "on_missing": on_missing,
        "source_dependencies": dependencies,
        "derivation_policy": {"schema_version": schema_version,
            "function": function, "output": name}}


@pytest.mark.parametrize("subset", sorted(_PINNED_DICTIONARY_HASHES))
def test_existing_subset_hashes_are_frozen(subset):
    assert table.recipe(subset)["dictionary_hash"] == _PINNED_DICTIONARY_HASHES[subset]


def test_hmos_dictionary_is_hmo_plus_season_unchanged():
    hmos = table.feature_dictionary("HMOS")
    assert hmos == table.feature_dictionary("HMO") + table._season_fields()
    assert table.recipe("HMOS")["dictionary_hash"] == content_hash(
        table.feature_dictionary("HMO") + table._season_fields())


def test_hmosx_dictionary_appends_x_fields_in_order():
    hmosx = table.feature_dictionary("HMOSX")
    assert hmosx == table.feature_dictionary("HMOS") + x.x_fields()
    assert [field["name"] for field in hmosx[-25:]] == list(x.X_COLUMNS)
    assert x.X_COLUMNS == (*x.TAIL, *x.IVDYN, *x.OWNVOL, *x.CALENDAR)
    recipe = table.recipe("HMOSX")
    assert recipe["subset"] == "HMOSX"
    assert recipe["model_columns"] == [field["name"] for field in recipe["dictionary"]] \
        + ["has_implied_quote"]
    assert recipe["model_columns"][-1] == "has_implied_quote"
    assert len(recipe["dictionary"]) == 72
    assert all(field["physical_type"] in ("float64", "int64")
        for field in recipe["dictionary"])
    assert len(recipe["model_columns"]) == 73
    assert recipe["dictionary_hash"] not in {
        *_PINNED_DICTIONARY_HASHES.values(),
        table.recipe("HMOS")["dictionary_hash"]}


def test_each_block_has_distinct_primary_role_and_schema():
    fields = x.x_fields()
    roles = {block: {field["source_dependencies"][0]["role"]
        for field in fields if field["block"] == block}
        for block in ("tail", "ivdyn", "ownvol", "calendar")}
    schemas = {block: {field["source_dependencies"][0]["policy"]["schema_version"]
        for field in fields if field["block"] == block}
        for block in ("tail", "ivdyn", "ownvol", "calendar")}
    assert len({frozenset(value) for value in roles.values()}) == 4
    assert len({frozenset(value) for value in schemas.values()}) == 4


@pytest.mark.parametrize("block", ["tail", "ivdyn", "ownvol", "calendar"])
def test_changing_one_block_policy_changes_hash_and_leaves_others(block):
    fields = x.x_fields()
    mutated = copy.deepcopy(fields)
    for field in mutated:
        if field["block"] == block:
            field["source_dependencies"][0]["policy"]["schema_version"] += ".changed"
    assert content_hash(mutated) != content_hash(fields)
    for original, changed in zip(fields, mutated):
        if original["block"] != block:
            assert changed == original


def test_x_field_types_units_nullability_match_the_spec():
    assert x.x_fields() == [_expected_field(*row) for row in _X_TABLE]
    for field in x.x_fields():
        assert set(field) == {"name", "block", "source_alias", "native_output_alias",
            "physical_type", "unit", "nullable", "on_missing", "source_dependencies",
            "derivation_policy"}
    assert x.x_fields() is not x.x_fields()


def test_contract_v5_extends_v4_with_x_columns_in_order():
    x_columns = [ColumnContract(name=field["name"], physical_type=field["physical_type"],
        nullable=field["nullable"], unit=field["unit"], null_policy=field["on_missing"])
        for field in x.x_fields()]
    assert table.CONTRACT_V5.contract_id == table.TABLE + ".v5"
    assert table.CONTRACT_V5.contract_id.endswith(".v5")
    assert table.CONTRACT_V5.semantic_version == "5.0.0"
    assert table.CONTRACT_V5.columns == (*table._contract_v4.columns, *x_columns)
    assert table.CONTRACT_V5.definition_hash != table.CONTRACT_V4.definition_hash
    assert table.CONTRACT_V5.definition_hash == table_contract_hash(table._contract_v5)
    assert table.CONTRACT is table.CONTRACT_V5


@pytest.mark.parametrize("columns", [
    list(x.X_COLUMNS),
    [*table.FEATURES_HMOSX, "has_implied_quote"],
])
def test_admit_columns_rejects_x_groups_by_default(columns):
    for kwargs in ({}, {"season": True}):
        with pytest.raises(DataError) as raised:
            table.admit_columns(columns, **kwargs)
        assert raised.value.code == "HOLDOUT_ACCESS_DENIED"


def test_admit_columns_admits_x_groups_only_with_flag():
    assert table.admit_columns([*table.FEATURES_HMOSX, "has_implied_quote"],
        season=True, x_groups=True) is None
    assert table.admit_columns(list(x.X_COLUMNS), x_groups=True) is None


@pytest.mark.parametrize("columns", [
    [], "n_prior", ["n_prior", "n_prior"], ["not_a_feature"], [None], ["n_prior", ""]])
def test_admit_columns_still_refuses_bad_projections(columns):
    for kwargs in ({}, {"season": True}, {"x_groups": True}):
        with pytest.raises(DataError) as raised:
            table.admit_columns(columns, **kwargs)
        assert raised.value.code == "HOLDOUT_ACCESS_DENIED"


def test_request_feature_set_order_still_excludes_hmosx():
    request = pytest.importorskip("engine.v2.experiments.magnitude_search_request")
    ordered = getattr(request, "FEATURE_SET_ORDER", None)
    if ordered is None:
        pytest.skip("no exported feature-set order constant")
    assert tuple(ordered) == ("H", "HM", "HMO")
    assert "HMOSX" not in ordered


def _prepared_rows(data):
    prepared = builder._prepare(data.repository, data.panel_source,
        producer=size_panel._producer({}),
        history_admitted=size_panel._history(data.repository, data.panel_source, data.panel_request),
        **data.panel_request)
    return list(builder._rows(data.repository, data.panel_source, data.prices, prepared))


def test_lineage_records_x_missingness(panel_data):
    rows = _prepared_rows(panel_data)
    assert rows
    expected = set((*x.IVDYN, *x.OWNVOL))
    for row in rows:
        blocks = json.loads(row["lineage"])
        assert set(blocks["x_pending"]["missing"]) == expected
        assert all(reason == "not_implemented_until_slice_6b"
            for reason in blocks["x_pending"]["missing"].values())
        if row["cal_days_since_prior"] is None:
            assert "cal_days_since_prior" in blocks["history"]["missing"]
        if row["tail_max_abs"] is None:
            assert "tail_max_abs" in blocks["history"]["missing"]


def test_history_x_empty_history_leaves_tail_and_days_since_prior_null():
    result = builder._history_x({"history": []}, {},
        {"session": "AMC", "event_date": "2024-03-08"})
    assert {name: result[name] for name in x.TAIL} == dict.fromkeys(x.TAIL)
    assert result["cal_days_since_prior"] is None
    assert result["cal_amc"] == 1
    assert result["cal_weekday"] == date(2024, 3, 8).weekday()
    assert result["cal_month"] == 3


def test_history_x_uses_absolute_signed_moves_and_last_contributor_date():
    history = {"history": [{"event_id": "a", "event_date": "2023-06-01"},
                           {"event_id": "b", "event_date": "2023-12-15"}]}
    result = builder._history_x(history, {"a": -4.0, "b": 7.5},
        {"session": "BMO", "event_date": "2024-03-08"})
    assert {name: result[name] for name in x.TAIL} == compute_tail_history([4.0, 7.5])
    assert result["cal_days_since_prior"] == (date(2024, 3, 8) - date(2023, 12, 15)).days
    assert result["cal_amc"] == 0
    assert result["cal_weekday"] == date(2024, 3, 8).weekday()
    assert result["cal_month"] == 3


def test_history_x_hand_computed_oracle():
    history = {"history": [{"event_id": "a", "event_date": "2023-01-10"},
                           {"event_id": "b", "event_date": "2023-04-11"},
                           {"event_id": "c", "event_date": "2023-07-12"},
                           {"event_id": "d", "event_date": "2023-10-10"}]}
    moves = {"a": 2.0, "b": -4.0, "c": 12.0, "d": -6.0}
    result = builder._history_x(history, moves,
        {"session": "AMC", "event_date": "2024-01-09"})
    assert sorted(abs(value) for value in moves.values()) == [2.0, 4.0, 6.0, 12.0]
    assert result["tail_max_abs"] == 12.0
    assert result["tail_max4_abs"] == 12.0
    assert result["tail_share_gt10"] == 0.25
    assert result["tail_p75_abs"] == 7.5
    assert result["tail_p90_abs"] == pytest.approx(10.2)
    assert result["tail_std_abs"] == pytest.approx(14 ** 0.5)
    assert result["tail_std_over_mean"] == pytest.approx((14 ** 0.5) / 6.0)
    assert result["tail_max_over_mean"] == 2.0
    assert result["cal_amc"] == 1
    assert result["cal_weekday"] == 1
    assert result["cal_month"] == 1
    assert result["cal_days_since_prior"] == 91


def test_history_x_refuses_an_unknown_session():
    with pytest.raises(DataError) as raised:
        builder._history_x({"history": []}, {},
            {"session": "XYZ", "event_date": "2024-03-08"})
    assert raised.value.code == "CONTRACT_MISMATCH"
    assert "session" in str(raised.value)


def test_built_rows_g1_g4_match_independent_identities(panel_data):
    rows = _prepared_rows(panel_data)
    assert rows
    histories = set()
    for row in rows:
        event = date.fromisoformat(row["event_date"])
        assert row["cal_weekday"] == event.weekday()
        assert row["cal_month"] == event.month
        assert row["cal_amc"] in (0, 1)
        assert row["tail_max_abs"] >= row["mean_prior_abs_move"] - 1e-9
        assert row["tail_max4_abs"] <= row["tail_max_abs"] + 1e-9
        assert row["tail_p90_abs"] <= row["tail_max_abs"] + 1e-9
        if row["n_prior"] == 1:
            assert row["tail_max_abs"] == pytest.approx(row["mean_prior_abs_move"])
            assert row["tail_std_abs"] == 0
            assert row["tail_max_over_mean"] == pytest.approx(1.0)
        histories.add(json.dumps(json.loads(row["lineage"])["history"]["sources"],
            sort_keys=True))
    assert any(row["cal_days_since_prior"] is not None
        and row["cal_days_since_prior"] > 0 for row in rows)
    if len(histories) > 1:
        assert len({row["tail_max_abs"] for row in rows}) > 1


def test_built_rows_match_v5_contract_and_carry_g1_g4(panel_data):
    rows = _prepared_rows(panel_data)
    assert rows
    columns = {column.name for column in table.CONTRACT.columns}
    assert len((*x.TAIL, *x.CALENDAR)) == 12
    for row in rows:
        assert set(row) == columns
        assert set((*x.TAIL, *x.CALENDAR)) <= set(row)
        assert all(row[name] is not None for name in (*x.TAIL, *x.CALENDAR))
        assert all(row[name] is None for name in (*x.IVDYN, *x.OWNVOL))
        assert row["cal_amc"] in (0, 1)
        assert row["cal_weekday"] == date.fromisoformat(row["event_date"]).weekday()
        assert row["cal_month"] == int(row["event_date"][5:7])


def test_x_null_counts_are_logged_once_with_every_column(panel_data, capsys):
    rows = _prepared_rows(panel_data)
    lines = [line for line in capsys.readouterr().err.splitlines()
             if "[panel x columns]" in line]
    assert len(lines) == 1
    assert f"rows={len(rows)}" in lines[0]
    assert all(f"{name}=" in lines[0] for name in x.X_COLUMNS)
    assert all(f"{name}={len(rows)}" in lines[0] for name in (*x.IVDYN, *x.OWNVOL))
