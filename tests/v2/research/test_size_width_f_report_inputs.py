"""Tests for the F report-input producers (experiments/size_width_f_report_inputs.py): counts, merge and the writer.

Synthetic frames only. Expected counts are hand-counted in this file from the fixture rows, never from the module.
"""
import hashlib

import numpy as np
import pandas as pd
import pytest

from experiments import size_pnl_receipts as receipts
from experiments import size_width_f_quotes as wf
from experiments import size_width_f_report_inputs as ri
from experiments.fair_value_inputs import FairValueRefusal
from experiments.size_pnl_shortvol import Refusal
from experiments.size_pnl_straddle import EXIT_UNQUOTED
from tests.v2.research.test_size_width_f_quotes import DEFAULT_PLAN, _frames, _resolved, _setup


def _legs():
    """Four rows of F's leg table (no chain_kind, annIdiv or annActDiv: see _inputs): three in 2024, one in 2025."""
    return pd.DataFrame({
        "obs_date": ["2024-03-01", "2024-03-01", "2024-03-02", "2025-01-05"],
        "status": ["ok", "no_fair_value", "ok", "rate_unknown"],
        "source": ["smvvol", None, "smvvol", "smvvol"],
        "clamp_side": ["none", "low", None, "none"],
        "repaired": [False, True, False, False],
        "unknown_bid_size": [False, False, False, False],
        "unknown_ask_size": [True, False, False, False],
        "unknown_oi": [False, True, False, False],
        "carry_fallback": [False, True, False, False],
        "dividend_fallback": [False, False, False, False],
        "dividend_applied": [True, False, False, False],
        "eff_bid": [1.0, 0.5, 2.0, 1.0],
        "eff_ask": [1.2, 0.7, 2.2, 1.0],
        "fill_buy": [1.05, 0.7, np.nan, 0.9],   # mids 1.1, 0.6, 2.1, 1.0
        "fill_sell": [1.15, np.nan, 2.05, 1.2],
    })


def _inputs():
    """The priced frame's input columns, same index as _legs: chain A (one null chain_kind) in 2024, chain B in 2025."""
    return pd.DataFrame({
        "chain_kind": ["A", "A", None, "B"],
        "annIdiv": [0.02, "n/a", 0.03, None],
        "annActDiv": [0.01, np.nan, 0.02, 0.04],
    })


def _rows():
    """Four F taken rows: two in arm 1.0 in 2024, one in arm 1.0 in 2025, one in arm E in 2024."""
    return pd.DataFrame({
        "arm": ["1.0", "1.0", "1.0", "E"], "structure": ["S", "S", "S", "S"],
        "event_date": ["2024-02-01", "2024-06-01", "2025-01-02", "2024-03-03"],
        "rung": ["rung1", EXIT_UNQUOTED, "rung1", EXIT_UNQUOTED],
        "exit_fair_mid": [False, False, True, False], "exit_negative": [False, False, True, False],
        "excluded_rung3": [False, True, False, False], "excluded_unpriceable": [False, False, False, True],
    })


EXIT_WANT = {
    ("1.0", "S", 2024): {"trades": 2, "exit_fair_mid": 0, "exit_unquoted_worst_case": 1, "exit_negative": 0,
                         "excluded_rung3": 1, "excluded_unpriceable": 0},
    ("1.0", "S", 2025): {"trades": 1, "exit_fair_mid": 1, "exit_unquoted_worst_case": 0, "exit_negative": 1,
                         "excluded_rung3": 0, "excluded_unpriceable": 0},
    ("E", "S", 2024): {"trades": 1, "exit_fair_mid": 0, "exit_unquoted_worst_case": 1, "exit_negative": 0,
                       "excluded_rung3": 0, "excluded_unpriceable": 1},
}
CATEGORIES = {
    (2024, "A", "status", "ok"): 1, (2024, "A", "status", "no_fair_value"): 1,
    (2024, "null", "status", "ok"): 1, (2025, "B", "status", "rate_unknown"): 1,
    (2024, "A", "source", "smvvol"): 1, (2024, "A", "source", "none"): 1,
    (2024, "null", "source", "smvvol"): 1, (2025, "B", "source", "smvvol"): 1,
    (2024, "A", "clamp_side", "none"): 1, (2024, "A", "clamp_side", "low"): 1,
    (2024, "null", "clamp_side", "null"): 1, (2025, "B", "clamp_side", "none"): 1,
}
COUNTS = {
    (2024, "A"): dict(rows=2, repaired=1, unknown_bid_size=0, unknown_ask_size=1, unknown_oi=1, carry_fallback=1,
                      dividend_fallback=0, dividend_applied=1, annIdiv_known=1, annActDiv_known=1,
                      better_than_mid_buy=1, better_than_mid_sell=1),
    (2024, "null"): dict(rows=1, repaired=0, unknown_bid_size=0, unknown_ask_size=0, unknown_oi=0, carry_fallback=0,
                         dividend_fallback=0, dividend_applied=0, annIdiv_known=1, annActDiv_known=1,
                         better_than_mid_buy=0, better_than_mid_sell=0),
    (2025, "B"): dict(rows=1, repaired=0, unknown_bid_size=0, unknown_ask_size=0, unknown_oi=0, carry_fallback=0,
                      dividend_fallback=0, dividend_applied=0, annIdiv_known=0, annActDiv_known=1,
                      better_than_mid_buy=1, better_than_mid_sell=1),
}


def _leg_want():
    counts = {(y, c, m, ""): n for (y, c), ms in COUNTS.items() for m, n in ms.items()}
    return {**CATEGORIES, **counts}


def _as_dict(frame):
    return {(int(y), c, m, v): n for y, c, m, v, n in
            zip(frame["year"], frame["chain_kind"], frame["metric"], frame["value"], frame["count"])}


def _tables():
    return {"legs": ri.leg_counts(_legs(), _inputs(), point="entry"), "exits": ri.exit_flag_counts(_rows())}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exit_dict(frame):
    got = {}
    for arm, structure, year, metric, count in zip(frame["arm"], frame["structure"], frame["year"], frame["metric"],
                                                   frame["count"]):
        got.setdefault((arm, structure, int(year)), {})[metric] = count
    return got


def test_exit_flag_counts_emits_every_metric_for_every_group_zeros_included():
    out = ri.exit_flag_counts(_rows())
    assert list(out.columns) == ["arm", "structure", "year", "metric", "count"]
    assert pd.api.types.is_integer_dtype(out["count"])
    assert _exit_dict(out) == EXIT_WANT
    assert len(out) == 18


def test_exit_flag_counts_of_empty_input_keeps_the_columns():
    out = ri.exit_flag_counts(_rows().iloc[0:0])
    assert out.empty and list(out.columns) == ["arm", "structure", "year", "metric", "count"]


def test_leg_counts_hand_counts_with_null_keys_nan_and_better_than_mid():
    out = ri.leg_counts(_legs(), _inputs(), point="entry")
    assert set(out["point"]) == {"entry"}
    assert len(out) == len(_leg_want()) == 48
    assert pd.api.types.is_integer_dtype(out["count"])
    assert _as_dict(out) == _leg_want()


def test_leg_counts_with_point_exit_gives_the_same_hand_counts_under_exit():
    out = ri.leg_counts(_legs(), _inputs(), point="exit")
    assert set(out["point"]) == {"exit"}
    assert len(out) == len(_leg_want()) == 48
    assert pd.api.types.is_integer_dtype(out["count"])
    assert _as_dict(out) == _leg_want()


def test_leg_counts_refuses_a_missing_leg_column():
    with pytest.raises(FairValueRefusal) as caught:
        ri.leg_counts(_legs().drop(columns=["eff_bid"]), _inputs(), point="entry")
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "column_missing"
    assert caught.value.details["table"] == "legs"
    assert caught.value.details["columns"] == ["eff_bid"]


def test_leg_counts_refuses_a_missing_input_column():
    with pytest.raises(FairValueRefusal) as caught:
        ri.leg_counts(_legs(), _inputs().drop(columns=["annIdiv"]), point="entry")
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "column_missing"
    assert caught.value.details["table"] == "inputs"
    assert caught.value.details["columns"] == ["annIdiv"]


def test_leg_counts_refuses_inputs_whose_index_differs_from_the_legs():
    inputs = _inputs().set_axis([10, 11, 12, 13], axis=0)
    with pytest.raises(FairValueRefusal) as caught:
        ri.leg_counts(_legs(), inputs, point="entry")
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "leg_inputs_misaligned"


def test_leg_counts_refuses_a_point_other_than_entry_or_exit():
    with pytest.raises(ValueError, match="point"):
        ri.leg_counts(_legs(), _inputs(), point="entry_leg")


def test_leg_counts_of_empty_input_keeps_the_columns():
    out = ri.leg_counts(_legs().iloc[0:0], _inputs().iloc[0:0], point="exit")
    assert out.empty and list(out.columns) == ["point", "year", "chain_kind", "metric", "value", "count"]


def test_merging_two_halves_equals_counting_the_whole_leg_table():
    legs, inputs = _legs(), _inputs()
    merged = ri.merge_counts([ri.leg_counts(legs.iloc[:2], inputs.iloc[:2], point="entry"),
                              ri.leg_counts(legs.iloc[2:], inputs.iloc[2:], point="entry")])
    whole = ri.leg_counts(legs, inputs, point="entry")
    pd.testing.assert_frame_equal(merged.reset_index(drop=True), whole.reset_index(drop=True), check_dtype=False)
    assert pd.api.types.is_integer_dtype(merged["count"])


def test_merging_two_halves_equals_counting_the_whole_exit_table():
    rows = _rows()
    merged = ri.merge_counts([ri.exit_flag_counts(rows.iloc[:2]), ri.exit_flag_counts(rows.iloc[2:])])
    pd.testing.assert_frame_equal(merged.reset_index(drop=True), ri.exit_flag_counts(rows).reset_index(drop=True),
                                  check_dtype=False)
    assert pd.api.types.is_integer_dtype(merged["count"])


def test_merging_halves_that_share_group_keys_sums_them_to_the_hand_counts():
    legs, inputs, rows = _legs(), _inputs(), _rows()
    # Row 0 vs rows 1-3: legs 0 and 1 are both (2024, "A"); rows 0 and 1 are both arm 1.0, structure S, 2024.
    merged_legs = ri.merge_counts([ri.leg_counts(legs.iloc[:1], inputs.iloc[:1], point="entry"),
                                   ri.leg_counts(legs.iloc[1:], inputs.iloc[1:], point="entry")])
    whole_legs = ri.leg_counts(legs, inputs, point="entry")
    pd.testing.assert_frame_equal(merged_legs.reset_index(drop=True), whole_legs.reset_index(drop=True), check_dtype=False)
    assert _as_dict(merged_legs) == _leg_want()
    assert _as_dict(merged_legs)[(2024, "A", "rows", "")] == 2  # hand count: legs 0 and 1 are both (2024, "A")

    merged_exits = ri.merge_counts([ri.exit_flag_counts(rows.iloc[:1]), ri.exit_flag_counts(rows.iloc[1:])])
    whole_exits = ri.exit_flag_counts(rows)
    pd.testing.assert_frame_equal(merged_exits.reset_index(drop=True), whole_exits.reset_index(drop=True), check_dtype=False)
    assert pd.api.types.is_integer_dtype(merged_exits["count"])
    assert _exit_dict(merged_exits) == EXIT_WANT
    assert _exit_dict(merged_exits)[("1.0", "S", 2024)]["trades"] == 2  # hand count: rows 0 and 1 are both arm 1.0, 2024


def test_merging_empty_frames_keeps_the_columns_and_an_empty_list_is_refused():
    empty = ri.leg_counts(_legs().iloc[0:0], _inputs().iloc[0:0], point="exit")
    merged = ri.merge_counts([empty, empty])
    assert merged.empty and list(merged.columns) == list(empty.columns)
    with pytest.raises(ValueError, match="at least one frame"):
        ri.merge_counts([])


def test_write_publishes_each_table_then_one_receipt_bound_by_hashes(tmp_path):
    out, tables = tmp_path / "out", _tables()
    receipt = ri.write(out, "inner", "H1", "W", tables)
    stage = out / "report_inputs" / "inner"
    assert sorted(p.name for p in stage.iterdir()) == ["exits.parquet", "legs.parquet"]
    assert receipt["stage"] == "inner" and receipt["inputs_hash"] == "H1" and receipt["scope"] == "W"
    assert receipt["tables"] == {name: {"sha256": _sha(stage / f"{name}.parquet"), "rows": len(frame),
                                        "bytes": (stage / f"{name}.parquet").stat().st_size} for name, frame in tables.items()}
    assert receipt["rows"] == sum(len(f) for f in tables.values()) and receipt["leg_rows"] == 0
    assert receipt["bytes"] == sum(e["bytes"] for e in receipt["tables"].values())
    for name, frame in tables.items():
        pd.testing.assert_frame_equal(pd.read_parquet(stage / f"{name}.parquet"), frame)
    assert receipts.load_receipt(out / "receipts" / "report_inputs_inner.json") == receipt


def test_an_exact_replay_completes_and_returns_the_same_receipt(tmp_path):
    out = tmp_path / "out"
    first = ri.write(out, "outer", "H", "W", _tables())
    assert ri.write(out, "outer", "H", "W", _tables()) == first
    assert sorted(p.name for p in (out / "report_inputs" / "outer").iterdir()) == ["exits.parquet", "legs.parquet"]


def test_changed_table_bytes_refuse_and_leave_the_old_bytes_and_receipt(tmp_path):
    out = tmp_path / "out"
    first = ri.write(out, "inner", "H", "W", _tables())
    path = out / "report_inputs" / "inner" / "legs.parquet"
    before = path.read_bytes()
    changed = {"legs": ri.leg_counts(_legs().iloc[:1], _inputs().iloc[:1], point="entry"), "exits": ri.exit_flag_counts(_rows())}
    with pytest.raises(Refusal) as caught:
        ri.write(out, "inner", "H", "W", changed)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "report_input_changed"
    assert path.read_bytes() == before
    assert sorted(p.name for p in (out / "report_inputs" / "inner").iterdir()) == ["exits.parquet", "legs.parquet"]
    assert receipts.load_receipt(out / "receipts" / "report_inputs_inner.json") == first


def test_a_changed_inputs_hash_refuses_the_receipt(tmp_path):
    out = tmp_path / "out"
    ri.write(out, "inner", "H", "W", _tables())
    with pytest.raises(Refusal) as caught:
        ri.write(out, "inner", "H2", "W", _tables())
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "tuning_receipt_changed"


@pytest.mark.parametrize("field", ["fabricated", "sha256", "rows", "bytes", "correct"])
def test_a_carried_part_is_bound_only_if_its_published_file_holds_exactly_its_metadata(tmp_path, field):
    out = tmp_path / "out"
    if field == "fabricated":
        part = {"sha256": "0" * 64, "rows": 1, "bytes": 1}  # no file was published under this name
    else:
        published = ri.publish(out, "inner", "carried", _rows())
        wrong = {"sha256": "0" * 64, "rows": published["rows"] + 1, "bytes": published["bytes"] + 1}
        part = published if field == "correct" else {**published, field: wrong[field]}
    tables, receipt_path = {"exits": ri.exit_flag_counts(_rows())}, out / "receipts" / "report_inputs_inner.json"
    if field == "correct":
        receipt = ri.write(out, "inner", "H", "W", tables, parts={"carried": part})
        assert receipt["tables"]["carried"] == published
        assert receipts.load_receipt(receipt_path) == receipt
        return
    with pytest.raises(Refusal) as caught:
        ri.write(out, "inner", "H", "W", tables, parts={"carried": part})
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "report_input_changed"
    assert not receipt_path.exists()


def test_a_bad_stage_or_no_tables_is_refused_before_anything_is_written(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="stage"):
        ri.write(out, "middle", "H", "W", _tables())
    with pytest.raises(ValueError, match="tables"):
        ri.write(out, "inner", "H", "W", {})
    assert not out.exists()


def test_a_name_in_both_parts_and_tables_is_refused_before_anything_is_written(tmp_path):
    out = tmp_path / "out"
    part = {"sha256": "0" * 64, "rows": 1, "bytes": 1}
    with pytest.raises(ValueError, match="both parts and tables: \\['exits'\\]"):
        ri.write(out, "inner", "H", "W", _tables(), parts={"exits": part})
    assert not out.exists()


def test_leg_counts_of_a_real_priced_table_counts_each_quote_row_once(tmp_path, monkeypatch):
    priced_in, real = [], wf._price

    def spy(rows, vols, side):
        priced_in.append(rows.copy())
        return real(rows, vols, side)

    monkeypatch.setattr(wf, "_price", spy)
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    priced = fq.table(chunk).reset_index(drop=True)  # F's real entry leg table: two expiries of 82 listed rows
    (rows,) = priced_in  # the frame F priced: quote rows joined with lookup_inputs, same index
    assert len(priced) == len(rows) == 164
    out = ri.leg_counts(priced, rows, point="entry")

    def total(metric):
        return out.loc[out["metric"] == metric, "count"].sum()

    assert pd.api.types.is_integer_dtype(out["count"])

    assert total("rows") == len(priced)
    assert set(rows["chain_kind"]) == {"fetch"}  # hand count: every synthetic quote row is chain_kind "fetch"
    assert out.loc[out["metric"] == "rows"].groupby("chain_kind")["count"].sum().to_dict() == {"fetch": 164}
    for metric, col in (("annIdiv_known", "annIdiv"), ("annActDiv_known", "annActDiv")):
        assert total(metric) == int(np.isfinite(pd.to_numeric(rows[col], errors="coerce")).sum()), metric


def test_the_real_leg_table_has_the_leg_columns_and_none_of_the_input_columns(tmp_path):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    legs = fq.legs(chunk, 0, _resolved(chunk, 0))
    for col in ri.LEG_COLS:
        assert col in legs.columns, col
    for col in ri.INPUT_COLS:
        assert col not in legs.columns, col
