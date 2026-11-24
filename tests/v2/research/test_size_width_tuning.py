import json

import numpy as np
import pandas as pd
import pytest

from experiments import size_pnl_receipts as receipts
from experiments import size_width_tuning as tuning
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_width_sweep import _TAKEN, E_ARM, INNER, Sweep

TWIN = ("TWIN-P", "TWIN-P5")


def trade(eid, structure, entry, exit_, *, cost=1.0, exit_value=1.5, mid=2.0, secured=1000.0):
    entry, exit_ = pd.Timestamp(entry), pd.Timestamp(exit_)
    return {"event_id": eid, "event_date": entry, "structure": structure, "k": 1.0, "entry_date": entry,
            "exit_date": exit_, "entry_cost": cost, "exit_value": exit_value, "exit_value_mid": mid,
            "pnl_dollars": 0.0, "pnl_dollars_mid": 0.0, "secured_per_contract": secured}


def frame(*rows):
    return pd.DataFrame(list(rows), columns=_TAKEN)


def sc(ret, trades=300, se=0.0):
    return {"ret": ret, "se": se, "trades": trades}


def scored(eid, k, value, date="2024-01-15", name="CND-PS"):
    return {"event_id": eid, "event_date": pd.Timestamp(date), "structure": name, "width": 0.04, "entry_cost": 3.0,
            "rel_spread": 0.1, "mcap_usd": 20e9, "exp_pnl_sim": value, "status": "scored", "reason": None,
            "alpha": 0.5, "k": k}


def _cnd_trades(n=300):
    start = pd.Timestamp("2024-01-02")
    return [trade(f"t{i}", "CND-PS", start + pd.Timedelta(days=i), start + pd.Timedelta(days=i + 5)) for i in range(n)]


def _sweep_cnd_at_15():
    sw = Sweep()
    sw.taken[("CND-PS", 1.5)] = [frame(*_cnd_trades())]
    return sw


def _no_constants(name):
    raise ValueError(f"non-strict JSON constant {name}")


# ---- unit score -----------------------------------------------------------

def test_unit_score_matches_the_hand_computed_one_trade_return():
    # By hand (start 200,000, cap 0.66, target share 0.25; cost 1.0, mid 1.5, secured 5,000 per contract):
    #   headroom = 0.66 * 200,000 = 132,000; want = 0.25 * 132,000 / 5,000 = 6.6; floor -> 6 contracts.
    #   pnl = 6 * (1.5 - 1.0) * 100 = 300; final = 200,300; ret = 300 / 200,000 = 0.0015.
    rows = frame(trade("a", "CND-PS", "2024-01-02", "2024-01-20", cost=1.0, exit_value=1.5, mid=1.5, secured=5000.0))
    score = tuning.unit_score(rows, INNER)
    assert score["ret"] == pytest.approx(0.0015)
    assert score["trades"] == 1


def test_unit_score_se_is_the_sample_sd_over_the_six_window_quarters():
    # INNER quarters are 2023Q3..2024Q4 (n = 6). By hand, with cost 1.0, mid 1.5, secured 5,000 per contract:
    #   trade a: entry 2023-08-01, exit 2023-08-20 -> 6 contracts, gain1 = 6 * 0.5 * 100 = 300.
    #   trade b: entry 2024-02-01, exit 2024-02-20, equity 200,000 + 300 = 200,300 -> headroom 132,198,
    #            want 6.6099 -> 6 contracts, gain2 = 300.
    #   quarter returns: 2023Q3 = gain1 / 200,000; 2023Q4 = 0; 2024Q1 = gain2 / 200,300; 2024Q2..Q4 = 0.
    rows = frame(trade("a", "CND-PS", "2023-08-01", "2023-08-20", cost=1.0, exit_value=1.5, mid=1.5, secured=5000.0),
                 trade("b", "CND-PS", "2024-02-01", "2024-02-20", cost=1.0, exit_value=1.5, mid=1.5, secured=5000.0))
    q = [300 / 200000, 0.0, 0.0, 300 / 200300, 0.0, 0.0]
    score = tuning.unit_score(rows, INNER)
    assert score["se"] == pytest.approx(np.std(q, ddof=1) / np.sqrt(6))
    assert score["ret"] == pytest.approx(600 / 200000)
    assert score["trades"] == 2


# ---- choose ---------------------------------------------------------------

def test_choose_picks_a_clear_interior_winner():
    assert tuning.choose({0.5: sc(0.01), 1.0: sc(0.02), 1.5: sc(0.09), 2.0: sc(0.03)}) == {"k": 1.5, "flags": []}


def test_choose_ties_within_one_se_go_nearest_one_then_smaller():
    assert tuning.choose({0.75: sc(0.10, se=0.05), 1.25: sc(0.10, se=0.05)}) == {"k": 0.75, "flags": []}
    assert tuning.choose({1.0: sc(0.09, se=0.05), 1.5: sc(0.10, se=0.05)}) == {"k": 1.0, "flags": []}


def test_choose_cutoff_uses_the_best_candidates_se_not_the_high_variance_one():
    # The cutoff is set by the best candidate (k 1.5, ret 0.10, se 0.01); the high-variance k 1.0 (ret 0.05, se 0.20)
    # falls below it and cannot displace k 1.5.
    assert tuning.choose({1.0: sc(0.05, se=0.20), 1.5: sc(0.10, se=0.01)}) == {"k": 1.5, "flags": []}


def test_choose_excludes_ks_under_300_trades():
    assert tuning.choose({1.0: sc(0.05, trades=300), 2.0: sc(0.50, trades=299)}) == {"k": 1.0, "flags": []}
    assert tuning.choose({1.0: sc(0.05, trades=300), 2.0: sc(0.50, trades=300)}) == {"k": 2.0, "flags": []}


def test_choose_with_no_eligible_k_defaults_to_one():
    assert tuning.choose({1.0: sc(0.05, trades=299), 1.5: sc(0.2, trades=10)}) == {"k": 1.0, "flags": ["k_defaulted"]}


def test_choose_at_an_edge_takes_the_best_interior_k():
    assert tuning.choose({3.0: sc(0.5), 2.0: sc(0.3), 0.5: sc(0.1)}) == {"k": 2.0, "flags": ["k_at_edge"]}
    assert tuning.choose({0.5: sc(0.5), 1.0: sc(0.2), 2.0: sc(0.1)}) == {"k": 1.0, "flags": ["k_at_edge"]}


def test_choose_with_only_edge_ks_scored_defaults_to_one():
    assert tuning.choose({0.5: sc(0.5), 3.0: sc(0.3)}) == {"k": 1.0, "flags": ["k_at_edge", "k_defaulted"]}


def test_choose_raw_edge_argmax_is_not_pulled_inside_by_the_tie_rule():
    # The raw argmax is 3.0 (an edge), so the tie rule must not run on the full grid: 1.0 is within 3.0's SE and would win.
    assert tuning.choose({3.0: sc(0.10, se=0.03), 2.0: sc(0.09, se=0.001), 1.0: sc(0.075, se=0.0)}) == {"k": 2.0, "flags": ["k_at_edge"]}
    # A non-edge argmax whose tie-rule pick lands on an edge also takes the interior branch.
    assert tuning.choose({1.5: sc(0.10, se=0.05), 0.5: sc(0.09, se=0.05)}) == {"k": 1.5, "flags": ["k_at_edge"]}


# ---- units and factors ----------------------------------------------------

def test_units_partition_the_structures_per_arm():
    assert tuning.units("A") == {"all": NAMES}
    assert set(tuning.units("B")) == set(NAMES) and all(len(v) == 1 for v in tuning.units("B").values())
    d = tuning.units("D")
    assert d["T"] == TWIN
    assert set(d["C"]) == set(NAMES) - set(TWIN)


def test_factors_map_every_structure_and_arm_e_maps_to_e():
    body = {"all": {"k": 1.5, "members": list(NAMES)}}
    assert tuning.factors("A", body) == {n: 1.5 for n in NAMES}
    assert tuning.factors(E_ARM, {}) == {n: E_ARM for n in NAMES}


# ---- tune -----------------------------------------------------------------

def test_tune_b_picks_the_scored_k_and_defaults_the_rest():
    body = tuning.tune(_sweep_cnd_at_15(), "B")
    assert body["CND-PS"]["k"] == 1.5 and body["CND-PS"]["flags"] == []
    assert body["CND-PS"]["members"] == ["CND-PS"]
    assert body["CND-PS"]["scores"]["1.5"]["not_scored"] is False
    assert body["CND-PS"]["scores"]["1.0"]["not_scored"] is True
    for n in NAMES:
        if n != "CND-PS":
            assert body[n]["k"] == 1.0 and body[n]["flags"] == ["k_defaulted"]


def test_tune_a_and_d_follow_the_same_rows():
    sw = _sweep_cnd_at_15()
    a = tuning.tune(sw, "A")
    assert a["all"]["k"] == 1.5 and a["all"]["members"] == list(NAMES)
    assert a["all"]["scores"]["1.5"]["trades"] == 300
    d = tuning.tune(sw, "D")
    assert d["T"]["k"] == 1.0 and d["T"]["flags"] == ["k_defaulted"]
    assert d["T"]["members"] == list(TWIN)
    assert d["C"]["k"] == 1.5 and d["C"]["scores"]["1.5"]["not_scored"] is False
    assert len(d["C"]["members"]) == 5


# ---- e summary ------------------------------------------------------------

def test_e_summary_same_k_at_ninety_percent_and_not_at_fifty():
    sw = Sweep()
    sw.scored[("CND-PS", E_ARM)] = [pd.DataFrame([scored(f"e{i}", 1.0 if i < 9 else 0.5, 1.0) for i in range(10)])]
    summary = tuning.e_summary(sw)["CND-PS"]
    assert summary["same_k"] is True
    assert summary["modal_share"] == pytest.approx(0.9)
    sw.scored[("CND-PS", E_ARM)] = [pd.DataFrame([scored(f"f{i}", 1.0 if i < 5 else 0.5, 1.0) for i in range(10)])]
    assert tuning.e_summary(sw)["CND-PS"]["same_k"] is False


def test_e_summary_of_no_candidates_has_zero_modal_share():
    summary = tuning.e_summary(Sweep())["CND-PS"]
    assert summary["modal_share"] == 0.0 and summary["same_k"] is False and summary["candidates"] == 0


def _infeasible(eid):
    return {**scored(eid, 1.0, np.nan), "status": "undetermined", "reason": "thin_history"}


def test_e_summary_share_counts_events_with_no_feasible_k_in_the_denominator():
    sw = Sweep()
    sw.scored[("CND-PS", E_ARM)] = [pd.DataFrame([scored(f"s{i}", 1.0, 1.0) for i in range(6)]
                                                 + [_infeasible(f"u{i}") for i in range(4)])]
    summary = tuning.e_summary(sw)["CND-PS"]
    assert summary["modal_share"] == pytest.approx(0.6)
    assert summary["same_k"] is False
    assert summary["no_feasible_k"] == 4 and summary["candidates"] == 10


def test_e_summary_same_k_when_nine_of_ten_candidates_share_one_k():
    sw = Sweep()
    sw.scored[("CND-PS", E_ARM)] = [pd.DataFrame([scored(f"s{i}", 1.0, 1.0) for i in range(9)] + [_infeasible("u")])]
    summary = tuning.e_summary(sw)["CND-PS"]
    assert summary["modal_share"] == pytest.approx(0.9)
    assert summary["same_k"] is True


def test_e_summary_same_k_at_the_eight_of_ten_boundary():
    sw = Sweep()
    sw.scored[("CND-PS", E_ARM)] = [pd.DataFrame([scored(f"s{i}", 1.0, 1.0) for i in range(8)]
                                                 + [scored("x", 0.5, 1.0), scored("y", 1.5, 1.0)])]
    summary = tuning.e_summary(sw)["CND-PS"]
    assert summary["modal_share"] == pytest.approx(0.8)
    assert summary["same_k"] is True


# ---- receipts -------------------------------------------------------------

def test_run_inner_writes_receipts_replays_and_refuses_a_changed_body(tmp_path):
    tuning.run_inner(Sweep(), tmp_path, "h1")
    for arm in "ABDE":
        assert (tmp_path / "receipts" / f"tuning_{arm}.json").exists()
    tuning.run_inner(Sweep(), tmp_path, "h1")  # an identical body replays
    with pytest.raises(Refusal) as exc:
        tuning.run_inner(_sweep_cnd_at_15(), tmp_path, "h1")
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.details["reason"] == "tuning_receipt_changed"


def test_run_inner_bodies_are_strict_json_even_with_empty_curves(tmp_path):
    tuning.run_inner(Sweep(), tmp_path, "h1")
    for arm in "ABDE":
        text = (tmp_path / "receipts" / f"tuning_{arm}.json").read_text()
        json.loads(text, parse_constant=_no_constants)
    e_body = json.loads((tmp_path / "receipts" / "tuning_E.json").read_text())["body"]
    assert e_body["CND-PS"]["curve"]["1.0"] is None


def _body(arm, k=1.0):
    return {unit: {"k": k, "flags": [], "members": list(m), "scores": {}} for unit, m in tuning.units(arm).items()}


def _write_at(out, arm, body, *, arm_field=None, window="2023H2-2024", inputs_hash="h1"):
    receipts.write_receipt(out / "receipts" / f"tuning_{arm}.json",
                           {"experiment": "EXP-189", "arm": arm_field or arm, "inputs_hash": inputs_hash,
                            "window": window, "body": body})


def test_frozen_accepts_a_valid_receipt(tmp_path):
    _write_at(tmp_path, "A", _body("A", k=1.5))
    assert tuning.frozen(tmp_path, "A", "h1") == {n: 1.5 for n in NAMES}


def test_frozen_refuses_a_receipt_for_b_copied_to_the_a_path(tmp_path):
    _write_at(tmp_path, "A", _body("B", k=1.5), arm_field="B")
    with pytest.raises(Refusal) as exc:
        tuning.frozen(tmp_path, "A", "h1")
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.details["reason"] == "tuning_receipt_changed"


def test_frozen_refuses_a_receipt_with_the_wrong_window(tmp_path):
    _write_at(tmp_path, "A", _body("A"), window="2024")
    with pytest.raises(Refusal) as exc:
        tuning.frozen(tmp_path, "A", "h1")
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.details["reason"] == "tuning_receipt_changed"


def test_frozen_refuses_a_unit_k_off_the_grid(tmp_path):
    body = _body("A")
    body["all"]["k"] = 99.0  # not on GRID
    _write_at(tmp_path, "A", body)
    with pytest.raises(Refusal) as exc:
        tuning.frozen(tmp_path, "A", "h1")
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.details["reason"] == "tuning_receipt_changed"


def test_run_outer_refuses_a_missing_receipt_before_any_evaluation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(tuning, "evaluate", lambda *a, **k: calls.append(a) or {})
    with pytest.raises(Refusal) as exc:
        tuning.run_outer(Sweep(), tmp_path, ["A", "B"], "h1")
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.details["reason"] == "tuning_receipt_missing"
    assert calls == []


def test_run_outer_refuses_receipts_bound_to_other_inputs(tmp_path):
    tuning.run_inner(Sweep(), tmp_path, "h1")
    with pytest.raises(Refusal) as exc:
        tuning.run_outer(Sweep(), tmp_path, ["A"], "h2")
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.details["reason"] == "tuning_receipt_changed"


def test_run_outer_evaluates_production_and_each_arm(tmp_path):
    sw = Sweep()
    tuning.run_inner(sw, tmp_path, "h1")
    out = tuning.run_outer(sw, tmp_path, ["A", "B", "D", "E"], "h1")
    assert set(out) == {"production", "A", "B", "D", "E"}
    for result in out.values():
        assert {"account", "structures"} <= set(result)
        assert set(result["structures"]) == set(NAMES)


def test_run_outer_looks_up_arm_e_rows_under_e_arm(tmp_path):
    sw = Sweep()
    tuning.run_inner(sw, tmp_path, "h1")
    d25 = pd.Timestamp("2025-03-03")
    sw.taken[("CND-PS", E_ARM)] = [frame(trade("x", "CND-PS", d25, d25 + pd.Timedelta(days=5)))]
    out = tuning.run_outer(sw, tmp_path, ["E"], "h1")
    assert out["E"]["structures"]["CND-PS"]["trades"] == 1
    assert out["production"]["structures"]["CND-PS"]["trades"] == 0


# ---- evaluate -------------------------------------------------------------

def test_evaluate_combines_the_structures_into_one_account():
    sw = Sweep()
    for n in ("CND-PS", "TWIN-P"):
        sw.taken[(n, 1.0)] = [frame(trade(f"{n}-1", n, "2024-02-01", "2024-02-10"),
                                    trade(f"{n}-2", n, "2024-03-01", "2024-03-10"))]
    # By hand (cost 1.0, mid 2.0 -> 200 back per contract, secured 1,000 per contract, cap 0.66, target share 0.25, start 200,000).
    # Each structure trades 2024-02-01 -> 02-10 and 2024-03-01 -> 03-10.
    #   02-01: headroom 132,000 -> want 33 -> 33 contracts (cash 196,700); the other structure: headroom 99,000 -> want 24.75 -> 24
    #          contracts (cash 194,300).
    #   02-10 exits: +33 * 200 + 24 * 200 = +11,400 -> cash 205,700.
    #   03-01: headroom 135,762 -> 33 contracts (cash 202,400); then headroom 102,762 -> want 25.69 -> 25 contracts (cash 199,900).
    #   03-10 exits: +33 * 200 + 25 * 200 = +11,600 -> final 211,500; pnl 11,500; total_return 0.0575; 4 funded of 4.
    # CND-PS alone: 33 contracts on 02-01 (cash 196,700); exit +6,600 -> 203,300; 33 on 03-01 (cash 200,000);
    #   exit +6,600 -> final 206,600; 2 funded of 2.
    out = tuning.evaluate(sw, {n: 1.0 for n in NAMES}, INNER)
    assert out["account"]["trades"] == 4
    assert out["account"]["funded"] == 4
    assert out["account"]["final_equity"] == pytest.approx(211500.0)
    assert out["account"]["total_return"] == pytest.approx(0.0575)
    assert out["structures"]["CND-PS"]["final_equity"] == pytest.approx(206600.0)
    assert out["structures"]["CND-PS"]["funded"] == 2 and out["structures"]["CND-PS"]["trades"] == 2
    assert out["structures"]["BFLY-P"]["trades"] == 0
