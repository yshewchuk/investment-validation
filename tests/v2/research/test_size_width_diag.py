"""EXP-189 report diagnostics D1 and D2 (synthetic receipts and rows only, no market data).

Every expected value below is written out by hand from the literal rows; none is computed with size_width_diag."""
import math

import numpy as np
import pandas as pd
import pytest

from experiments import size_pnl_shortvol as sv
from experiments import size_width_diag as diag
from experiments import size_width_f_fills as ff
from experiments import size_width_f_report_inputs as ri
from experiments import size_width_pricing as seam
from experiments import size_width_report as report
from experiments.size_pnl_pool import build_pool
from experiments.size_pnl_shortvol import NAMES
from test_size_width_a_report import _a_run
from test_size_width_f_report import _body, _diag_rows, _f_run, _refused, _rewrite, _text, redo_diag
from test_size_width_report import _write
from tests.v2.research.test_size_pnl_shortvol import _tables, chunk_of, events_for
from tests.v2.research.test_size_width_f_fills import EVENTS as F_EVENTS, _tight_frames
from tests.v2.research.test_size_width_f_quotes import DEFAULT_PLAN, _setup
from tests.v2.research.test_size_width_sweep import DRAW, NAME, _two_months

FACTORS = {"production": {n: 1.0 for n in NAMES}}


def _row(eid, event, exit_date, name, exp, xv, xvm, pnl, pnlm, arm="1.0"):
    """A taken row with entry_cost and entry_cost_mid both 1.0, so the realised fill return is exit_value - 1."""
    return {"event_id": eid, "event_date": pd.Timestamp(event), "exit_date": pd.Timestamp(exit_date), "structure": name, "arm": arm,
            "exp_pnl_sim": exp, "entry_cost": 1.0, "entry_cost_mid": 1.0, "exit_value": xv, "exit_value_mid": xvm,
            "pnl_dollars": pnl, "pnl_dollars_mid": pnlm}


# Inner stage (2023H2 and 2024) and outer stage (2025). The rows a label must not take are in: arm "0.75", a fallback arm, and row a4,
# which exits on 2024-01-05, after the 2023H2 window (so it is in no period).
INNER = [
    _row("a1", "2023-08-01", "2023-08-20", "BFLY-P", 0.01, 1.01, 1.02, 1.0, 2.0),
    _row("a2", "2023-09-01", "2023-09-20", "BFLY-P", 0.02, 1.02, 1.03, 2.0, 3.0),
    _row("a3", "2023-10-01", "2023-10-20", "BFLY-P", 0.03, 1.03, 0.98, 3.0, -2.0),
    _row("b1", "2023-08-05", "2023-08-25", "CTR5", 0.01, 1.05, 1.00, 5.0, 0.0),
    _row("b2", "2023-11-05", "2023-11-25", "CTR5", 0.04, 0.97, 1.10, -3.0, 10.0),
    _row("a4", "2023-12-20", "2024-01-05", "BFLY-P", 0.05, 1.00, 1.00, 0.0, 0.0),
    _row("c1", "2024-02-10", "2024-02-28", "CTR5", 0.05, 1.02, 0.99, 2.0, -1.0),
    _row("a1", "2023-08-01", "2023-08-20", "BFLY-P", 0.02, 1.09, 1.09, 9.0, 9.0, arm="0.75"),
    _row("a1", "2023-08-01", "2023-08-20", "BFLY-P", 0.02, 1.09, 1.09, 9.0, 9.0, arm="fallback phi=0.1 factors=BFLY-P=1.0"),
]
OUTER = [_row("d1", "2025-05-01", "2025-05-20", "BFLY-P", 0.04, 1.00, 1.00, 0.0, 0.0)]

# D1, production. Trades: 2023H2 has a1, a2, a3, b1, b2 (a4 is out); 2024 has c1; 2025 has d1.
# Fill wins (pnl > 0): 2023H2 4 of 5, 2024 1 of 1, 2025 0 of 1. Mid wins (pnl_mid > 0): 2023H2 3 of 5, 2024 0 of 1, 2025 0 of 1.
D1_HEADER = "| arm | period | structure | trades | win % at F fills | win % at mid |"
D1_LINES = [
    "| production | 2025 | all | 1 | 0.00 | 0.00 |",
    "| production | 2025 | BFLY-P | 1 | 0.00 | 0.00 |",
    "| production | 2025 | BFLY-P5 | 0 | n/a | n/a |",
    "| production | 2025 | CND-PS | 0 | n/a | n/a |",
    "| production | 2025 | CTR5 | 0 | n/a | n/a |",
    "| production | 2025 | RAMP7 | 0 | n/a | n/a |",
    "| production | 2025 | TWIN-P | 0 | n/a | n/a |",
    "| production | 2025 | TWIN-P5 | 0 | n/a | n/a |",
    "| production | 2023H2 | all | 5 | 80.00 | 60.00 |",
    "| production | 2023H2 | BFLY-P | 3 | 100.00 | 66.67 |",
    "| production | 2023H2 | BFLY-P5 | 0 | n/a | n/a |",
    "| production | 2023H2 | CND-PS | 0 | n/a | n/a |",
    "| production | 2023H2 | CTR5 | 2 | 50.00 | 50.00 |",
    "| production | 2023H2 | RAMP7 | 0 | n/a | n/a |",
    "| production | 2023H2 | TWIN-P | 0 | n/a | n/a |",
    "| production | 2023H2 | TWIN-P5 | 0 | n/a | n/a |",
    "| production | 2024 | all | 1 | 100.00 | 0.00 |",
    "| production | 2024 | BFLY-P | 0 | n/a | n/a |",
    "| production | 2024 | BFLY-P5 | 0 | n/a | n/a |",
    "| production | 2024 | CND-PS | 0 | n/a | n/a |",
    "| production | 2024 | CTR5 | 1 | 100.00 | 0.00 |",
    "| production | 2024 | RAMP7 | 0 | n/a | n/a |",
    "| production | 2024 | TWIN-P | 0 | n/a | n/a |",
    "| production | 2024 | TWIN-P5 | 0 | n/a | n/a |",
]

# D2 over the seven pooled production rows (a1-a3, b1, b2, c1, d1). Fill returns: .01 .02 .03 .05 -.03 .02 0.00; mid returns:
# .02 .03 -.02 0.00 .10 -.01 0.00; predictions .01 .02 .03 .01 .04 .05 .04 (predicted mid = predicted fill, both costs 1.0).
# Pooled Spearman: F -11.5 / sqrt(27 x 27.5) = -0.422; mid -5.5 / sqrt(27 x 27.5) = -0.202.
# Within structure x year: (BFLY-P, 2023) has 3 rows, rho +1 at fill and -0.5 at mid; (CTR5, 2023) has 2 rows, rho -1 at fill and +1 at mid;
# the one-row cells are undefined and drop out. Fill: (3 x 1 + 2 x -1) / 5 = 0.200; mid: (3 x -0.5 + 2 x 1) / 5 = 0.100.
# 2023 (5 rows): fill -4.5 / sqrt(95) = -0.462, mid 3.5 / sqrt(95) = 0.359. 2024 and 2025 have one row each: n/a.
D2_HEADER = "| arm | predicted vs realised | trades | pooled | within structure x year (event-weighted) | 2023 | 2024 | 2025 |"
D2_LINES = [
    "| production | mid vs mid | 7 | -0.202 | 0.100 | 0.359 | n/a | n/a |",
    "| production | mid vs F | 7 | -0.422 | 0.200 | -0.462 | n/a | n/a |",
    "| production | F vs F | 7 | -0.422 | 0.200 | -0.462 | n/a | n/a |",
]

# Quintiles by predicted return: bins {a1, b1} (.01), {a2} (.02), {a3} (.03), {b2, d1} (.04), {c1} (.05); realised means in %.
QUINTILE_HEADER = "| arm | predicted vs realised | quintile | trades | mean predicted % | mean realised % |"
QUINTILE_LINES = [
    "| production | mid vs mid | 1 | 2 | 1.00 | 1.00 |",
    "| production | mid vs mid | 2 | 1 | 2.00 | 3.00 |",
    "| production | mid vs mid | 3 | 1 | 3.00 | -2.00 |",
    "| production | mid vs mid | 4 | 2 | 4.00 | 5.00 |",
    "| production | mid vs mid | 5 | 1 | 5.00 | -1.00 |",
    "| production | mid vs F | 1 | 2 | 1.00 | 3.00 |",
    "| production | mid vs F | 2 | 1 | 2.00 | 2.00 |",
    "| production | mid vs F | 3 | 1 | 3.00 | 3.00 |",
    "| production | mid vs F | 4 | 2 | 4.00 | -1.50 |",
    "| production | mid vs F | 5 | 1 | 5.00 | 2.00 |",
    "| production | F vs F | 1 | 2 | 1.00 | 3.00 |",
    "| production | F vs F | 2 | 1 | 2.00 | 2.00 |",
    "| production | F vs F | 3 | 1 | 3.00 | 3.00 |",
    "| production | F vs F | 4 | 2 | 4.00 | -1.50 |",
    "| production | F vs F | 5 | 1 | 5.00 | 2.00 |",
]


def _write_parts(out, inner, outer):
    """Taken parts of both stages under ``out``, bound to inputs_hash H and scope None."""
    ri.write(out, "inner", "H", None, {"taken_0000": pd.DataFrame(inner)})
    ri.write(out, "outer", "H", None, {"taken_0000": pd.DataFrame(outer)})


def _section(out):
    return diag.section(out, {"inputs_hash": "H", "scope": None}, FACTORS)


def test_d1_table_is_the_literal_win_rates(tmp_path):
    _write_parts(tmp_path, INNER, OUTER)
    lines = _section(tmp_path).splitlines()
    assert D1_HEADER in lines
    for line in D1_LINES:
        assert line in lines


def test_d2_table_is_the_literal_rank_checks(tmp_path):
    _write_parts(tmp_path, INNER, OUTER)
    lines = _section(tmp_path).splitlines()
    assert D2_HEADER in lines
    for line in D2_LINES:
        assert line in lines


def test_quintile_table_is_the_literal_quintile_means(tmp_path):
    _write_parts(tmp_path, INNER, OUTER)
    lines = _section(tmp_path).splitlines()
    assert QUINTILE_HEADER in lines
    for line in QUINTILE_LINES:
        assert line in lines


def test_fewer_than_five_pooled_rows_emit_no_quintiles(tmp_path):
    _write_parts(tmp_path, INNER[:3], OUTER)  # four pair rows: a1, a2, a3 and d1
    assert _section(tmp_path).splitlines()[-1] == "(no rows)"


def test_no_taken_parts_give_empty_tables_with_n_a(tmp_path):
    ri.write(tmp_path, "inner", "H", None, {"exit_flags": pd.DataFrame({"x": [1]})})
    ri.write(tmp_path, "outer", "H", None, {"exit_flags": pd.DataFrame({"x": [1]})})
    lines = _section(tmp_path).splitlines()
    assert "| production | 2025 | all | 0 | n/a | n/a |" in lines
    assert "| production | mid vs mid | 0 | n/a | n/a | n/a | n/a | n/a |" in lines
    assert lines[-1] == "(no rows)"


def test_predicted_mid_is_the_exit_value_over_the_mid_debit():
    frame = pd.DataFrame([
        {"structure": "BFLY-P", "event_date": pd.Timestamp("2024-05-01"), "exp_pnl_sim": 0.25, "entry_cost": 1.0, "entry_cost_mid": 0.8,
         "exit_value": 1.1, "exit_value_mid": 0.9, "exit_date": pd.Timestamp("2024-05-20")},
        {"structure": "BFLY-P", "event_date": pd.Timestamp("2024-05-01"), "exp_pnl_sim": 0.25, "entry_cost": 1.0, "entry_cost_mid": 0.0,
         "exit_value": 1.1, "exit_value_mid": 0.9, "exit_date": pd.Timestamp("2024-05-20")}])
    got = diag._pairs(frame)
    assert got["pred_fill"].iloc[0] == pytest.approx(0.25)
    assert got["pred_mid"].iloc[0] == pytest.approx(0.5625)  # (0.25 + 1) * 1.0 / 0.8 - 1
    assert got["real_fill"].iloc[0] == pytest.approx(0.1)
    assert got["real_mid"].iloc[0] == pytest.approx(0.125)  # 0.9 / 0.8 - 1
    assert math.isnan(got["pred_mid"].iloc[1]) and math.isnan(got["real_mid"].iloc[1])  # no mid debit: undefined


# ---- refusals, each through assemble: nothing is written -------------------------------------------------------------

def _part(out, stage):
    return next((out / "report_inputs" / stage).glob("taken_*.parquet"))


def _missing_exp(out):
    redo_diag(out, {"inner": _diag_rows("inner").drop(columns="exp_pnl_sim")})


def _changed_bytes(out):
    part = _part(out, "inner")
    part.write_bytes(part.read_bytes() + b"0")


def _missing_part(out):
    _part(out, "inner").unlink()


def _no_receipt(out):
    (out / "receipts" / "report_inputs_inner.json").unlink()


def _other_inputs_hash(out):
    path = out / "receipts" / "report_inputs_inner.json"
    body = _body(path)
    body["inputs_hash"] = "OTHER"
    _rewrite(path, body)


def _duplicate_row(out):
    inner = _diag_rows("inner")
    redo_diag(out, {"inner": pd.concat([inner, inner.iloc[[0]]], ignore_index=True)})


def _nan_exp(out):
    inner = _diag_rows("inner")
    inner.loc[0, "exp_pnl_sim"] = float("nan")
    redo_diag(out, {"inner": inner})


REFUSALS = [
    ("missing exp_pnl_sim", _missing_exp, "diag_columns_missing"),
    ("part bytes changed", _changed_bytes, "report_input_changed"),
    ("part missing", _missing_part, "report_input_changed"),
    ("receipt missing", _no_receipt, "tuning_receipt_missing"),
    ("receipt bound to another inputs_hash", _other_inputs_hash, "tuning_receipt_changed"),
    ("duplicate row", _duplicate_row, "diag_rows_invalid"),
    ("NaN exp_pnl_sim", _nan_exp, "diag_rows_invalid"),
]


@pytest.mark.parametrize("mutate, reason", [(c[1], c[2]) for c in REFUSALS], ids=[c[0] for c in REFUSALS])
def test_diagnostic_refusals_write_nothing(tmp_path, mutate, reason):
    _f_run(tmp_path)
    mutate(tmp_path)
    _refused(tmp_path, reason)


# ---- the report -----------------------------------------------------------------------------------------------------

def test_an_f_run_report_carries_d1_and_d2(tmp_path):
    _f_run(tmp_path)
    report.assemble(tmp_path)
    text = _text(tmp_path)
    assert "### D1 beside T5: win rate by arm, period and structure" in text
    assert "| arm | period | structure | trades | win % at F fills | win % at mid |" in text
    assert "### D2 beside T5: prediction rank check at fixed width" in text
    assert "### D2 beside T5: predicted quintiles" in text


def test_a_model_a_report_renames_f_in_the_diagnostics(tmp_path):
    _a_run(tmp_path)
    report.assemble(tmp_path)
    text = _text(tmp_path)
    assert "| arm | period | structure | trades | win % at model A fills | win % at mid |" in text
    assert "win % at F fills" not in text and "mid vs model A" in text


def test_a_current_run_report_has_no_diagnostics(tmp_path):
    _write(tmp_path, passing=("A",))
    report.assemble(tmp_path)
    text = _text(tmp_path)
    assert "### D1" not in text and "### D2" not in text


# ---- identity: the simulated exit value does not depend on the fill ----------------------------------------------------

def test_simulated_exit_value_does_not_depend_on_the_fill():
    pool = build_pool(*_tables(), floor=None)
    chunk, prior, cur, _sessions = _two_months()
    sims = sv.simulate(chunk, cur, pool, draws=DRAW)
    half = sv.score_events(chunk, cur, sims, NAME, alpha=0.5).set_index("event_id")
    zero = sv.score_events(chunk, cur, sims, NAME, alpha=0.0).set_index("event_id")
    both = half.index[(half["status"] == "scored") & (zero["status"] == "scored")]
    assert len(both) > 0
    h, z = half.loc[both], zero.loc[both]
    assert (h["entry_cost"] != z["entry_cost"]).all()  # the fills differ
    np.testing.assert_allclose(((h["exp_pnl_sim"] + 1) * h["entry_cost"]).to_numpy(dtype=float),
                               ((z["exp_pnl_sim"] + 1) * z["entry_cost"]).to_numpy(dtype=float), rtol=1e-12)


def test_hand_built_exit_value_is_fill_independent():
    # Hand derivation, CND-PS at PRED 6 on chunk_of's ladder: spot 100, mid(K) = 1 + 0.01 (K-100)^2, bid/ask = mid -/+ 0.1.
    # Width is 6 / 100 / 2 = 0.03 of spot, so the legs (one contract each) are up1 SELL 103, dn1 SELL 97, up2 BUY 106,
    # dn2 BUY 94; the atm reference leg has quantity 0 and is not traded.
    # Degenerate draw: terminal spot 100 (zero move) and no time value left, so each put is its intrinsic max(K - 100, 0):
    # 94 -> 0, 97 -> 0, 103 -> 3, 106 -> 6.
    # Exit value: the long 106 is sold for +6, the long 94 for +0, the short 103 is bought back for -3, the short 97 for -0:
    # 6 + 0 - 3 - 0 = 3.0.
    # Entry at alpha 0.5 (mids): 2 x 1.36 - 2 x 1.09 = 0.54. At alpha 0.0 (buy at ask, sell at bid): 2 x 1.46 - 2 x 0.99 = 0.94.
    chunk, events = chunk_of(("A", {})), events_for("A")
    grid = np.arange(80.0, 121.0)
    sims = {"A": (grid, np.maximum(grid - 100.0, 0.0))}
    for alpha, cost in ((0.5, 0.54), (0.0, 0.94)):
        row = sv.score_events(chunk, events, sims, NAME, alpha=alpha).iloc[0]
        assert row["status"] == "scored"
        assert row["entry_cost"] == pytest.approx(cost)
        assert (row["exp_pnl_sim"] + 1) * row["entry_cost"] == pytest.approx(3.0, rel=1e-12)


def test_hand_built_exit_value_under_the_f_copy(tmp_path):
    # The same degenerate draw on the F pricing copy: the F fixtures' E0 is CND-PS at PRED 6 on the same strikes (80..120)
    # and spot 100, so its legs and exit value are the ones derived in test_hand_built_exit_value_is_fill_independent.
    # Only the entry debit is the F one (fair-value fills, the same at every alpha); the exit value stays 3.0.
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    events = F_EVENTS.iloc[[0]]
    grid = np.arange(60.0, 141.0)
    sims = {"E0": (grid, np.maximum(grid - 100.0, 0.0))}
    copy = seam.load(fills=ff.FFills(fq))
    for alpha in (0.5, 0.0):
        row = copy.score_events(chunk, events, sims, NAME, alpha=alpha).iloc[0]
        assert row["status"] == "scored"
        assert (row["exp_pnl_sim"] + 1) * row["entry_cost"] == pytest.approx(3.0, rel=1e-12)


# ---- D1 with two labels: A (BFLY-P at arm "0.75", CTR5 at "1.0") and E (every structure at arm "E") -----------------------

A_FACTORS = {**{n: 1.0 for n in NAMES}, "BFLY-P": 0.75}
E_FACTORS = {n: "E" for n in NAMES}
E_ROWS = [  # arm "E" rows, added to a local copy of INNER
    _row("e1", "2023-09-05", "2023-09-25", "BFLY-P", 0.04, 1.04, 1.05, 4.0, -1.0, arm="E"),
    _row("e2", "2024-01-10", "2024-01-25", "CTR5", 0.02, 0.98, 1.01, -2.0, 1.0, arm="E"),
    _row("e3", "2025-06-01", "2025-06-20", "BFLY-P", 0.03, 1.03, 1.02, 1.0, 1.0, arm="E"),
]
# Label A, by hand: BFLY-P takes only a1's arm "0.75" row (fill 9.0, mid 9.0); CTR5 takes b1 (5.0, 0.0), b2 (-3.0, 10.0) in 2023H2
# and c1 (2.0, -1.0) in 2024; d1 (BFLY-P, arm 1.0) and a2, a3, a4 (BFLY-P, arm 1.0) are not taken.
A_LINES = [
    "| A | 2025 | all | 0 | n/a | n/a |",
    "| A | 2025 | BFLY-P | 0 | n/a | n/a |",
    "| A | 2023H2 | all | 3 | 66.67 | 66.67 |",
    "| A | 2023H2 | BFLY-P | 1 | 100.00 | 100.00 |",
    "| A | 2023H2 | CTR5 | 2 | 50.00 | 50.00 |",
    "| A | 2024 | all | 1 | 100.00 | 0.00 |",
    "| A | 2024 | BFLY-P | 0 | n/a | n/a |",
    "| A | 2024 | CTR5 | 1 | 100.00 | 0.00 |",
]
# Label E, by hand: only the three arm "E" rows. e1 (4.0, -1.0) in 2023H2 for BFLY-P; e2 (-2.0, 1.0) in 2024 for CTR5;
# e3 (1.0, 1.0) in 2025 for BFLY-P. d1 (arm 1.0) and a1's arm "0.75" row are not taken.
E_LINES = [
    "| E | 2023H2 | all | 1 | 100.00 | 0.00 |",
    "| E | 2023H2 | BFLY-P | 1 | 100.00 | 0.00 |",
    "| E | 2024 | all | 1 | 0.00 | 100.00 |",
    "| E | 2024 | CTR5 | 1 | 0.00 | 100.00 |",
    "| E | 2025 | all | 1 | 100.00 | 100.00 |",
    "| E | 2025 | BFLY-P | 1 | 100.00 | 100.00 |",
]


def test_d1_labels_take_their_own_arm_and_e_takes_only_e_rows(tmp_path):
    _write_parts(tmp_path, INNER + E_ROWS, OUTER)
    text = diag.section(tmp_path, {"inputs_hash": "H", "scope": None}, {"A": A_FACTORS, "E": E_FACTORS})
    lines = text.splitlines()
    for line in A_LINES + E_LINES:
        assert line in lines
    assert "| A | 2023H2 | BFLY-P | 3 | 100.00 | 66.67 |" not in lines  # what the arm 1.0 BFLY-P rows would give
    after_d1 = text.split(diag.D2_HEAD, 1)[1]  # the D2 and quintile tables
    assert [line for line in after_d1.splitlines() if line.startswith("| E |")] == []
