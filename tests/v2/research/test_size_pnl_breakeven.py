"""EXP-188 Amendment 6b: the breakeven alpha of the report, and the repriced short-vol endpoints. Synthetic fixtures only; no store."""
import numpy as np
import pandas as pd
import pytest

from engine.fills import breakeven_alpha
from experiments import size_pnl_assemble as assemble
from experiments import size_pnl_breakeven as breakeven
from experiments import size_pnl_driver as driver
from experiments import size_pnl_shortvol as sv
from tests.v2.research.test_size_pnl_assemble import (  # noqa: F401 (_no_plan_admission is the autouse check_plan no-op)
    HARSH, _negate_median_mid, _no_plan_admission, _section_lines, build, plan_frame)
from tests.v2.research.test_size_pnl_shortvol import chunk_of, events_for, pool, taken  # noqa: F401 (pool is a fixture)

NOT_TESTABLE = "### Breakeven alpha per model and book\n\nnot_testable: breakeven_endpoints_not_in_decisions"


def _frame(rows):
    """Decision-style rows: (event_id, model_id, book, arm_or_rule, event_date, taken, lots)."""
    return pd.DataFrame(rows, columns=["event_id", "model_id", "book", "arm_or_rule", "event_date", "taken", "lots"])


def _endpoints(a0, a1):
    """score()-shaped endpoint frames: {event_id: pnl_per_share} at alpha 0 and alpha 1."""
    return (pd.DataFrame({"event_id": list(a0), "pnl_per_share": list(a0.values())}),
            pd.DataFrame({"event_id": list(a1), "pnl_per_share": list(a1.values())}))


def test_known_answer_totals_breakeven_and_table():
    # Two taken rows in one (model, book, rule, year). Per-share P&L at alpha 0 / alpha 1: (-0.30, +0.10) with 2 lots, (-0.10, +0.30) with 1 lot.
    # Dollars: pnl_a0 = -0.30 x 100 x 2 + -0.10 x 100 x 1 = -70.0; pnl_a1 = 0.10 x 100 x 2 + 0.30 x 100 x 1 = 50.0.
    # Breakeven = 70 / 120 = 0.58333, so the cell reads 0.583. The untaken row carries NaN endpoints and drops out of the table.
    frame = _frame([("e1", "M0", "short_vol", "as_is", "2024-06-10", True, 2),
                    ("e2", "M0", "short_vol", "as_is", "2024-07-08", True, 1),
                    ("e3", "M0", "short_vol", "as_is", "2024-07-15", False, 1)])
    a0, a1 = _endpoints({"e1": -0.30, "e2": -0.10}, {"e1": 0.10, "e2": 0.30})
    out = breakeven.with_endpoints(frame, a0, a1)
    assert out.loc[out["event_id"] == "e1", "pnl_dollars_a0"].iloc[0] == pytest.approx(-60.0)
    assert out.loc[out["event_id"] == "e2", "pnl_dollars_a0"].iloc[0] == pytest.approx(-10.0)
    assert out.loc[out["event_id"] == "e1", "pnl_dollars_a1"].iloc[0] == pytest.approx(20.0)
    assert out.loc[out["event_id"] == "e2", "pnl_dollars_a1"].iloc[0] == pytest.approx(30.0)
    assert np.isnan(out.loc[out["event_id"] == "e3", "pnl_dollars_a0"].iloc[0]) and np.isnan(out.loc[out["event_id"] == "e3", "pnl_dollars_a1"].iloc[0])
    assert breakeven.read(-70.0, 50.0) == "0.583"
    assert breakeven_alpha(-70.0, 50.0) == pytest.approx(7 / 12)
    text = breakeven.section(out)
    assert text.startswith("### Breakeven alpha per model and book (fraction of the bid-ask spread to capture to break even)")
    assert "0.5 = break even at mid; below 0.5 profitable at mid" in text
    assert "| model | book | rule | year | trades | pnl_a0 | pnl_a1 | breakeven_alpha |" in text
    assert "| M0 | short_vol | as_is | 2024 | 2 | -70.00 | 50.00 | 0.583 |" in text


@pytest.mark.parametrize("a0, a1, expected", [
    (10.0, 30.0, "profitable at alpha 0 (breakeven 0)"),  # profitable at the worst fills: no crossing inside [0, 1]
    (-30.0, -10.0, "loses even at alpha 1 (no breakeven)"),  # still losing at the best fills
    (float("nan"), 5.0, "n/a"),  # a non-finite total has no breakeven
    (0.0, 0.0, "n/a"),  # a flat line at exactly 0 has no crossing to report
])
def test_read_edge_cases(a0, a1, expected):
    assert breakeven.read(a0, a1) == expected


def test_two_years_give_two_table_rows():
    # 2024: lots 1, pnl (-30, +10) -> breakeven 30 / 40 = 0.750. 2025: lots 1, pnl (+10, +30) -> profitable at alpha 0.
    frame = _frame([("e1", "M0", "short_vol", "as_is", "2024-06-10", True, 1),
                    ("e2", "M0", "short_vol", "as_is", "2025-03-03", True, 1)])
    a0, a1 = _endpoints({"e1": -0.30, "e2": 0.10}, {"e1": 0.10, "e2": 0.30})
    text = breakeven.section(breakeven.with_endpoints(frame, a0, a1))
    assert "| M0 | short_vol | as_is | 2024 | 1 | -30.00 | 10.00 | 0.750 |" in text
    assert "| M0 | short_vol | as_is | 2025 | 1 | 10.00 | 30.00 | profitable at alpha 0 (breakeven 0) |" in text


def test_missing_endpoint_columns_give_the_not_testable_string():
    frame = _frame([("e1", "M0", "short_vol", "as_is", "2024-06-10", True, 1)])
    assert breakeven.section(frame) == NOT_TESTABLE
    assert breakeven.section(frame.assign(pnl_dollars_a0=np.nan, pnl_dollars_a1=np.nan)) == NOT_TESTABLE


def test_repriced_short_vol_scores_at_the_endpoint_alpha(pool):
    # Event A: BUY 94 is rung 2 (bid 1.0, ask 2.0); B has no exit map (rung 1). Decisions are priced at alpha 0.5 (entry cost 0.54 for A).
    # At alpha 0 the entry buys at the ask and sells at the bid: cost 1.46 - 0.99 - 0.99 + 1.46 = 0.94 for both events, and the
    # exit sells the rung-2 leg at its bid (1.0). So pnl_per_share at alpha 0 is at most the pnl_per_share at alpha 0.5 for every row.
    chunk = chunk_of(("A", {94: (1.0, 2.0, 2)}), ("B", {}))
    events, decisions = events_for("A", "B"), taken(chunk, "A", "B")
    at_half = sv.score(chunk, events, decisions, "CND-PS", 0.5, mid_marks=True)
    with pytest.raises(ValueError, match="decision was not priced at this alpha"):
        sv.score(chunk, events, decisions, "CND-PS", 0.0, mid_marks=True)  # the default path still refuses a re-priced alpha
    at_zero = sv.score(chunk, events, decisions, "CND-PS", 0.0, mid_marks=True, repriced=True)
    assert list(at_zero["event_id"]) == list(at_half["event_id"]) == ["A", "B"]
    assert (at_zero["entry_cost"].to_numpy() >= at_half["entry_cost"].to_numpy() - 1e-12).all()  # buying at the ask costs more at 0
    assert (at_zero["pnl_per_share"].to_numpy() <= at_half["pnl_per_share"].to_numpy() + 1e-12).all()
    assert at_zero.set_index("event_id").loc["A", "entry_cost"] == pytest.approx(0.94)
    # Repriced at the original alpha, the endpoint call equals the mid-marks score of the same decisions.
    pd.testing.assert_frame_equal(sv.score(chunk, events, decisions, "CND-PS", 0.5, mid_marks=True, repriced=True), at_half)


def test_ends_is_none_when_the_flag_is_off():
    calls = []
    assert driver._ends(False, lambda a: calls.append(a)) is None
    assert calls == []  # no endpoint is scored when none are stored


def test_ends_scores_both_endpoints_when_the_flag_is_on():
    calls = []

    def score_at(a):
        calls.append(a)
        return f"pnl@{a}"

    assert driver._ends(True, score_at) == ("pnl@0.0", "pnl@1.0")
    assert calls == [0.0, 1.0]


def test_t10_notes_name_the_marks_each_verdict_reads(tmp_path):
    # The headline T10 reads the mid marks; the harsh T10 reads the stored primary fills and says it is not a verdict basis.
    out = build(tmp_path, rows_fn=_negate_median_mid)
    assemble.assemble(out, plan_frame(), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    headline = "\n".join(_section_lines(text, "### T10 Verdicts"))
    harsh = "\n".join(_section_lines(text, HARSH + "T10 Verdicts"))
    assert "T10 reads the mid marks" in headline and "T10 here reads the harsh fills" not in headline
    assert "T10 here reads the harsh fills (rung-2 zero-bid legs at alpha 0): reported only, not a verdict basis." in harsh
    assert "T10 reads the mid marks" not in harsh


def test_month_straddle_rows_carry_breakeven_endpoints_for_g_prod_and_r_l(monkeypatch):
    """Pins the Amendment 6b endpoints on the straddle rows of ``_month``: the G_PROD gate arm and both R_L rules carry
    ``ends``, so their pnl_dollars_a0 / pnl_dollars_a1 are finite. Stubbed scorers: mid-mark pnl_per_share -0.2 at alpha 0
    and 0.4 at alpha 1, one lot, so the dollars are -20.0 and 40.0 (x CONTRACT_MULTIPLIER 100 x 1 lot)."""
    events = pd.DataFrame({"event_id": ["E3"], "event_date": pd.to_datetime(["2025-01-15"]), "pred_move": [5.0], "pred_crush": [-12.0],
                           "pred_abs_move": [5.0], "implied_move": [4.0]})
    shortvol_cols = ["event_id", "event_date", "structure", "status", "reason"]
    scored = {("as_is", name): pd.DataFrame(columns=shortvol_cols) for name in driver.shortvol.NAMES}
    endpoint = {0.0: -0.2, 1.0: 0.4}

    def marks(per_share):
        return pd.DataFrame({"event_id": ["E3"], "entry_cost": [10.0], "exit_value": [101.5], "rung": [2], "lots": [1],
                             "pnl_per_share": [per_share], "pnl_dollars": [per_share * 100]})

    def straddle_score(chunk_, events_, dec, alpha, *, drop_unquoted=False, mid_marks=False):
        return marks(endpoint[alpha]) if mid_marks and alpha in endpoint else marks(0.1)

    def taken(reason):
        return pd.DataFrame({"event_id": ["E3"], "event_date": pd.to_datetime(["2025-01-15"]), "edge": [1.0], "cutoff": [0.0],
                             "rel_spread": [0.1], "status": ["taken"], "reason": [reason]})

    monkeypatch.setattr(driver.shortvol, "decide", lambda scored_: pd.DataFrame(columns=shortvol_cols))
    monkeypatch.setattr(driver.shortvol, "score", lambda *a, **kw: pd.DataFrame(columns=["event_id", "entry_cost", "exit_value", "rung", "lots", "pnl_per_share", "pnl_dollars"]))
    monkeypatch.setattr(driver.shortvol, "secured", lambda *a, **kw: pd.DataFrame(columns=["event_id", "secured_per_contract"]))
    monkeypatch.setattr(driver.straddle, "decide", lambda events_, history, rule, chunk_, gate=None: taken("taken"))
    monkeypatch.setattr(driver.straddle, "score", straddle_score)
    monkeypatch.setattr(driver.straddle, "history_rows", lambda first, events_: pd.DataFrame(columns=["event_date", "edge"]))

    frames = driver._month("prod", events, scored, set(), {}, pd.Timestamp(driver.OUTER[0]), driver._model_state(),
                           sessions={}, gate=lambda events_, chunk_: taken("taken"))
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[rows["book"] == "straddle"]
    for arm in (driver.straddle.GATE_RULE, *driver.straddle.RULES):
        row = rows[rows["arm_or_rule"] == arm]
        assert len(row) == 1, arm
        a0, a1 = float(row["pnl_dollars_a0"].iloc[0]), float(row["pnl_dollars_a1"].iloc[0])
        assert np.isfinite(a0) and np.isfinite(a1), arm
        assert a0 == pytest.approx(-20.0) and a1 == pytest.approx(40.0), arm
