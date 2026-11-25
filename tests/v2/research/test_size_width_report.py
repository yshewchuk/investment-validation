"""EXP-189 report: synthetic receipts only, no market data."""
import json

import pytest

from experiments import size_pnl_receipts as receipts
from experiments import size_width_report as report
from experiments import size_width_tuning as tuning
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_width_sizing import GRID
from experiments.size_width_success import PREFERENCE
from experiments.size_width_tuning import TWIN

PERIODS = ("2025", "2023H2", "2024")
CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
IDENTITY = {"plan_id": "PLAN", "registration_id": "REG", "seal_id": "SEAL", "panel_id": "P"}
HALVES = ("2023H2", "2024")
BASE, PASS, FAIL = 0.01, 0.02, 0.005
FLAT_RETS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.045, 0.04, 0.03, 0.02, 0.01]
HEADINGS = ["### T1 error model", "### T2 chosen factors", "### T2b arm E", "### T3 curves", "### T4 anchor",
            "### T5 production versus arms", "### T6 per-structure", "### T7 success table"]
MODEL = {"L": -3.0, "b": 0.8, "s": 0.4, "n": 400, "mean_log_y": -3.2,
         "dropped": {"nonpositive_forecast": 0, "nonpositive_label": 1}, "skew": 0.1, "excess_kurtosis": 0.2}
DECLINED = {n: {**{str(k): 0.1 for k in GRID}, "E": 0.2} for n in NAMES}
ANCHOR = {n: {str(k): {"ret": 0.01, "se": 0.002, "trades": 400} for k in GRID} for n in NAMES}


def _values(label, passing):
    """(total return, breakeven x) of one label: production is the base, passing arms beat it, the rest fail."""
    if label == "production":
        return BASE, 0.45
    if label in passing:
        return PASS, 0.40
    return FAIL, 0.60


def _eval(ret):
    """One period's evaluate: the account and every structure at ``ret``."""
    return {"account": {"total_return": ret, "trades": 400}, "structures": {n: {"total_return": ret, "trades": 50} for n in NAMES}}


def _scores(rets, trades=1000):
    """Score cells for every grid k, one return per k."""
    return {str(k): {"ret": r, "se": 0.01, "trades": trades, "not_scored": False} for k, r in zip(GRID, rets)}


def _unit(members, k, rets):
    """A tuning unit with factor ``k`` and the scores of ``rets``."""
    return {"k": k, "flags": [], "members": list(members), "scores": _scores(rets)}


FAKE = {"calibration": {"ok": True}, "N": tuning.FAKE_DRAWS, "sim_draws": tuning.FAKE_SIM_DRAWS, "seed_tag": tuning.FAKE_SEED_TAG,
        "model": MODEL}


def _write(out, *, passing=(), c_status="ok", smoke=False, model=True, b_rets=None, c_rets=None, e_peak=None,
           outer_over=None, inputs_over=None, tuning_over=None):
    """A full synthetic receipt set under ``out/receipts``; arms in ``passing`` beat production on every test.
    ``c_rets`` gives C's score curves their own returns (default: B's). ``outer_over`` and ``inputs_over`` (inputs_outer) are merged
    into those bodies, and ``tuning_over`` maps an arm to fields merged into its envelope, for the identity-conflict tests."""
    rec = out / "receipts"
    b_rets = b_rets or [BASE] * len(GRID)
    c_rets = c_rets or b_rets
    labels = ["production", "A", "B", *(["C"] if c_status == "ok" else []), "D", "E"]
    arms = [a for a in ("A", "B", "C", "D", "E") if a != "C" or c_status == "ok"]
    inner_r, inner_s, outer_r, outer_s = {}, {}, {}, {}
    for label in labels:
        ret, x = _values(label, passing)
        st = {p: {"trades": 400, "x": x, "mid": ret, "alpha0": ret - 0.03} for p in PERIODS}
        inner_r[label] = {p: _eval(ret) for p in HALVES}
        inner_s[label] = {p: st[p] for p in HALVES}
        outer_r[label] = {"2025": _eval(ret)}
        outer_s[label] = {"2025": st["2025"]}
    cal = None
    if model:
        cal = {"ok": c_status == "ok", "reason": None if c_status == "ok" else "fake_not_calibrated",
               "rmse_m0": 0.1, "rmse_fake": 0.1, "mae_m0": 0.05, "mae_fake": 0.05, "rmse_rel": 0.01, "mae_rel": -0.01}
    receipts.write_receipt(rec / "inner_summary.json", {
        "stage": "inner", "inputs_hash": "H", "scope": None, "smoke": smoke, "c_status": c_status,
        "model": MODEL if model else None, "calibration": cal, "anchor": ANCHOR,
        "results": inner_r, "stats": inner_s, "declined": DECLINED})
    receipts.write_receipt(rec / "outer_summary.json", {
        "stage": "outer", "inputs_hash": "H", "scope": None, "smoke": smoke, "arms": arms,
        "results": outer_r, "stats": outer_s, "declined": DECLINED, **(outer_over or {})})
    receipts.write_receipt(rec / "inputs_inner.json", {"stage": "inputs", "inputs_hash": "H", "scope": None, "identity": IDENTITY})
    receipts.write_receipt(rec / "inputs_outer.json", {"stage": "inputs", "inputs_hash": "H", "scope": None, "identity": IDENTITY,
                                                       **(inputs_over or {})})

    def tune_receipt(arm, body):
        envelope = {"experiment": "EXP-189", "arm": arm, "inputs_hash": "H", "window": "2023H2-2024", "body": body}
        if arm == "C":
            envelope["fake"] = FAKE
        receipts.write_receipt(rec / f"tuning_{arm}.json", {**envelope, **(tuning_over or {}).get(arm, {})})

    rest = tuple(n for n in NAMES if n not in TWIN)
    curve = {str(k): (0.02 if k == e_peak else BASE) for k in GRID}
    curve["E"] = BASE
    tune_receipt("A", {"all": _unit(NAMES, 0.75, b_rets)})
    tune_receipt("B", {n: _unit((n,), 1.0, b_rets) for n in NAMES})
    tune_receipt("D", {"T": _unit(TWIN, 1.25, b_rets), "C": _unit(rest, 1.0, b_rets)})
    tune_receipt("E", {n: {"k_share": {"0.75": 0.5, "1.25": 0.5}, "modal_share": 0.5, "same_k": False,
                           "candidates": 400, "no_feasible_k": 3, "curve": curve} for n in NAMES})
    if c_status == "ok":
        tune_receipt("C", {n: _unit((n,), 1.5, c_rets) for n in NAMES})


def _text(out):
    return (out / "REPORT.md").read_text(encoding="utf-8")


def _line(text, prefix):
    return next(line for line in text.splitlines() if line.startswith(prefix))


def test_passing_arm_a_is_recommended(tmp_path):
    _write(tmp_path, passing=("A",))
    assert report.assemble(tmp_path)["recommendation"] == "A"
    rule = json.loads((tmp_path / "proposed_rule.json").read_text(encoding="utf-8"))
    assert rule["recommendation"] == "A" and len(rule["rows"]) == 7
    assert all(r["divisor"] == f"divisor / {r['k']}" for r in rule["rows"])
    text = _text(tmp_path)
    for heading in HEADINGS:
        assert heading in text
    assert "2025 NOT blind for widths" in text
    assert "### Proposed board width rule (arm A)" in text


def test_no_passing_arm_gives_no_change(tmp_path):
    _write(tmp_path)
    assert report.assemble(tmp_path)["recommendation"] == "no_change"
    assert json.loads((tmp_path / "proposed_rule.json").read_text(encoding="utf-8"))["recommendation"] == "no_change"
    assert "No change recommended." in _text(tmp_path)


@pytest.mark.parametrize("passing, expected", [(("A", "D"), "A"), (("D", "B"), "D"), (("C", "B"), "C"), (("B", "E"), "B")])
def test_two_passing_arms_follow_preference(tmp_path, passing, expected):
    _write(tmp_path, passing=passing)
    assert report.assemble(tmp_path)["recommendation"] == expected
    assert [a for a in PREFERENCE if a in passing][0] == expected


def test_c_omitted_when_fake_not_calibrated(tmp_path):
    _write(tmp_path, c_status="fake_not_calibrated")
    assert not (tmp_path / "receipts" / "tuning_C.json").exists()
    report.assemble(tmp_path)
    text = _text(tmp_path)
    assert "| omitted |" in _line(text, f"| {NAMES[0]} |")
    assert "omitted" in _line(text, "| 0.5 |")
    t7 = text.split("### T7 success table")[1].split("### T8-T9")[0]
    assert "| C |" not in t7


def test_flatness_and_agree_arithmetic_on_b(tmp_path):
    _write(tmp_path, c_status="fake_not_calibrated", b_rets=FLAT_RETS, e_peak=1.5)
    report.assemble(tmp_path)
    text = _text(tmp_path)
    assert "flatness (best - score)/se at k=0.75, 1.0, 1.25: 2.000, 1.000, 0.000" in text
    assert "argmax k: B=1.25 C=omitted exp_pnl_sim=1.5; agree: yes" in text


def test_agree_is_no_beyond_one_grid_step(tmp_path):
    _write(tmp_path, c_status="fake_not_calibrated", b_rets=FLAT_RETS, e_peak=2.0)
    report.assemble(tmp_path)
    assert "argmax k: B=1.25 C=omitted exp_pnl_sim=2.0; agree: no" in _text(tmp_path)


def test_b_flatness_at_k_one_is_the_rule_flatness(tmp_path):
    _write(tmp_path, passing=("B",), c_status="fake_not_calibrated", b_rets=FLAT_RETS)
    assert report.assemble(tmp_path)["recommendation"] == "B"
    rule = json.loads((tmp_path / "proposed_rule.json").read_text(encoding="utf-8"))
    assert all(r["flatness_se"] == pytest.approx((0.05 - 0.04) / 0.01) for r in rule["rows"])


def test_flat_note_takes_flatness_from_primary_curve_c_not_recommended_b(tmp_path):
    # C's k = 1.0 flatness is 0.0 (constant curve); B's is 1.0 (FLAT_RETS). B is recommended, so the rule must print C's 0.0.
    _write(tmp_path, passing=("B",), b_rets=FLAT_RETS, c_rets=[BASE] * len(GRID))
    assert report.assemble(tmp_path)["recommendation"] == "B"
    rule = json.loads((tmp_path / "proposed_rule.json").read_text(encoding="utf-8"))
    assert all(r["flatness_se"] == pytest.approx(0.0) for r in rule["rows"])
    text = _text(tmp_path)
    assert "on arm C's per-structure curve" in text
    assert "it is not arm B's own tuning curve" in text


def test_flat_note_names_b_as_primary_curve_when_c_omitted(tmp_path):
    _write(tmp_path, passing=("A",), c_status="fake_not_calibrated")
    assert report.assemble(tmp_path)["recommendation"] == "A"
    text = _text(tmp_path)
    assert "on arm B's per-structure curve" in text
    assert "it is not arm A's own tuning curve" in text


def test_no_flat_note_for_no_change(tmp_path):
    _write(tmp_path)
    assert report.assemble(tmp_path)["recommendation"] == "no_change"
    assert "flatness_se: T3's flatness" not in _text(tmp_path)


def test_e_recommended_renders_the_e_sentence(tmp_path):
    _write(tmp_path, passing=("E",))
    assert report.assemble(tmp_path)["recommendation"] == "E"
    assert json.loads((tmp_path / "proposed_rule.json").read_text(encoding="utf-8"))["recommendation"] == "E"
    assert "Recommendation: E, pick k maximising exp_pnl_sim per event." in _text(tmp_path)


def test_w5_placeholder_names_slice_w6b3_by_default(tmp_path):
    _write(tmp_path)
    report.assemble(tmp_path)
    assert "W6b-3" in _text(tmp_path)
    assert "not available: the driver does not yet run W5's tolerance bands and fallback arm (EXP-189 slice W6b-3)" in _text(tmp_path)


def test_w5_tables_are_embedded_verbatim(tmp_path):
    _write(tmp_path)
    custom = "### T8-T9 tolerance bands and fallback\n\n| band | value |\n|---|---|\n| phi | 0.9 |"
    report.assemble(tmp_path, w5_tables=custom)
    assert custom in _text(tmp_path)


def test_smoke_run_line_and_t1_not_available_without_model(tmp_path):
    _write(tmp_path, smoke=True, model=False, c_status="smoke_no_fake")
    report.assemble(tmp_path)
    text = _text(tmp_path)
    assert "SMOKE RUN: a scope-bound subset, not a result." in text
    assert "not available (c_status=smoke_no_fake)." in text
    assert not (tmp_path / "receipts" / "tuning_C.json").exists()


def test_missing_outer_summary_refuses(tmp_path):
    _write(tmp_path)
    (tmp_path / "receipts" / "outer_summary.json").unlink()
    with pytest.raises(Refusal) as caught:
        report.assemble(tmp_path)
    assert caught.value.details["reason"] == "tuning_receipt_missing"


def test_missing_tuning_b_refuses(tmp_path):
    _write(tmp_path)
    (tmp_path / "receipts" / "tuning_B.json").unlink()
    with pytest.raises(Refusal) as caught:
        report.assemble(tmp_path)
    assert caught.value.details["reason"] == "tuning_receipt_missing"


def _refused(out, reason, expected=None):
    """assemble refuses as an identity conflict under ``reason`` and writes neither REPORT.md nor proposed_rule.json."""
    with pytest.raises(Refusal) as caught:
        report.assemble(out, expected=expected)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == reason
    assert not (out / "REPORT.md").exists() and not (out / "proposed_rule.json").exists()


def test_outer_summary_with_another_inputs_hash_refuses(tmp_path):
    _write(tmp_path, outer_over={"inputs_hash": "OTHER"})
    _refused(tmp_path, "tuning_receipt_changed")


def test_outer_summary_with_another_scope_refuses(tmp_path):
    _write(tmp_path, outer_over={"scope": "OTHER"})
    _refused(tmp_path, "tuning_receipt_changed")


def test_smoke_flag_differing_between_inner_and_outer_refuses(tmp_path):
    _write(tmp_path, outer_over={"smoke": True})
    _refused(tmp_path, "tuning_receipt_changed")


def test_outer_arms_differing_from_c_status_refuse(tmp_path):
    _write(tmp_path, outer_over={"arms": ["A", "B", "D", "E"]})  # c_status is ok, so C must be listed
    _refused(tmp_path, "tuning_receipt_changed")


def test_tuning_receipt_bound_to_another_inputs_hash_refuses(tmp_path):
    _write(tmp_path, tuning_over={"D": {"inputs_hash": "OTHER"}})
    _refused(tmp_path, "tuning_receipt_changed")


def test_tuning_receipt_naming_another_arm_refuses(tmp_path):
    _write(tmp_path, tuning_over={"B": {"arm": "A"}})
    _refused(tmp_path, "tuning_receipt_changed")


def test_inputs_outer_identity_differing_refuses(tmp_path):
    _write(tmp_path, inputs_over={"identity": {**IDENTITY, "seal_id": "OTHER"}})
    _refused(tmp_path, "tuning_receipt_changed")


def test_expected_with_another_plan_id_refuses(tmp_path):
    _write(tmp_path)
    _refused(tmp_path, "expected_mismatch", expected={**IDENTITY, "plan_id": "OTHER"})


def test_matching_expected_passes(tmp_path):
    _write(tmp_path, passing=("A",))
    assert report.assemble(tmp_path, expected=dict(IDENTITY))["recommendation"] == "A"
