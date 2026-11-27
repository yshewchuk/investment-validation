"""EXP-190 Q6b part 2: the reported-only helpers (implied_table, s_frames, headline_flags) on hand-made synthetic
frames. Oracles are written in this file, in plain numpy and python, never by calling the code under test."""
import math

import numpy as np
import pandas as pd
import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from engine.v2.research.magnitude_panel_qualified_table import recipe
from experiments import size_quantile_reported as rep
from experiments.size_quantile_calibration import QCOLS

LEVELS = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
DATES = {"a": "2024-01-02", "b": "2024-01-03", "c": "2024-01-04"}
EVENTS = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]}, index=["p", "q", "r", "s"])


def _sealed(rows, columns=("f_mu", "implied_move")):
    targets = [{"event_id": e, "event_date": d} for e, d in DATES.items()]
    return {"model_columns": list(columns), "rows": rows, "context": {"target_events": targets}}


# ---------- 1. implied_table ----------

def test_implied_table_values_dates_and_the_hash_of_the_consumed_values():
    rows = [{"event_id": "b", "implied_move": None}, {"event_id": "a", "implied_move": 2.5},
            {"event_id": "c", "implied_move": 0.0}]
    frame, digest = rep.implied_table(_sealed(rows))
    assert list(frame.index) == ["b", "a", "c"] and frame.index.name == "event_id"
    assert list(frame.columns) == ["event_date", "implied_move"]
    assert frame.loc["a", "event_date"] == "2024-01-02"
    assert np.isnan(frame.loc["b", "implied_move"])
    assert frame.loc["a", "implied_move"] == 2.5 and frame.loc["c", "implied_move"] == 0.0
    assert digest == content_hash([["a", DATES["a"], 2.5], ["b", DATES["b"], None], ["c", DATES["c"], 0.0]])
    moved = _sealed(rows)
    moved["context"]["target_events"][0]["event_date"] = "2024-01-09"  # event "a" only
    assert rep.implied_table(moved)[1] != digest


def test_infinite_implied_moves_are_nan_in_the_frame_and_none_in_the_digest():
    rows = [{"event_id": "a", "implied_move": math.inf}, {"event_id": "b", "implied_move": -math.inf},
            {"event_id": "c", "implied_move": 1.5}]
    frame, digest = rep.implied_table(_sealed(rows))
    assert np.isnan(frame.loc["a", "implied_move"]) and np.isnan(frame.loc["b", "implied_move"])
    assert frame.loc["c", "implied_move"] == 1.5
    assert digest == content_hash([["a", DATES["a"], None], ["b", DATES["b"], None], ["c", DATES["c"], 1.5]])


@pytest.mark.parametrize("rows, columns", [
    ([{"event_id": "a", "implied_move": 1.0}], ("f_mu",)),                     # not bound in the model
    ([{"event_id": "a", "f_mu": 1.0}], ("f_mu", "implied_move")),              # a row lacks the key
    ([{"event_id": "a", "implied_move": 1.0}, {"event_id": "a", "implied_move": 2.0}], ("implied_move",)),
    ([{"event_id": "a", "implied_move": "1.0"}], ("implied_move",)),           # a str is not a number
    ([{"event_id": "a", "implied_move": True}], ("implied_move",)),            # a bool is not a number
])
def test_implied_table_refuses_unbound_inputs(rows, columns):
    with pytest.raises(OpsError) as exc:
        rep.implied_table(_sealed(rows, columns))
    assert exc.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.problem.details["reason"] == "implied_move_unbound"


def test_implied_table_accepts_the_real_v14_feature_columns():
    # Loader contract: "implied_move" is in recipe("HMOSXLP")["model_columns"] (what load_sealed_panel returns as
    # sealed["model_columns"]) and in recipe("HMOSX")["model_columns"] (the columns experiments/size_quantile_sources.py
    # `_panel_rows` and `_load` read the stored panel rows with).
    assert "implied_move" in recipe("HMOSXLP")["model_columns"]
    assert "implied_move" in recipe("HMOSX")["model_columns"]
    columns = list(recipe("HMOSXLP")["model_columns"])
    row_a = {column: None for column in recipe("HMOSX")["model_columns"]}
    row_a.update({"event_id": "a", "implied_move": 2.5})
    row_b = {column: None for column in recipe("HMOSX")["model_columns"]}
    row_b.update({"event_id": "b", "implied_move": None})
    frame, _ = rep.implied_table(_sealed([row_a, row_b], columns))
    assert list(frame.index) == ["a", "b"]
    assert frame.loc["a", "implied_move"] == 2.5
    assert np.isnan(frame.loc["b", "implied_move"])


# ---------- 2. s_frames ----------

FIN = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
NAN7 = [float("nan")] * 7
EMPTY_FLOOR = {"events": 0, "events_bound": 0, "donors": 0, "donors_bound": 0}


def _outcome(status, ids, scored=None, unscored=None, floor=None):
    return {"status": status, "receipt": {"event_ids": ids, "content_hash": "sha256:x"}, "scored": scored or {},
            "unscored": unscored or {}, "floor": floor or dict(EMPTY_FLOOR)}


def _s_results():
    qa = [v + 0.5 for v in FIN]
    return {
        "2023": _outcome("ok", ["a", "b", "c", "d"],
                         scored={"a": {"raw": FIN, "q": qa}, "b": {"raw": FIN, "q": NAN7},
                                 "c": {"raw": [1.0, 2.0, 3.0, math.inf, 5.0, 6.0, 7.0], "q": FIN}},
                         unscored={"d": "scale_burn_in"},
                         floor={"events": 3, "events_bound": 1, "donors": 0, "donors_bound": 0}),
        "2024": _outcome("fit_failed", ["e"]),
        "2025": _outcome("ok", ["f", "g"], scored={"f": {"raw": FIN, "q": FIN}, "g": {"raw": FIN, "q": FIN}},
                         floor={"events": 2, "events_bound": 0, "donors": 4, "donors_bound": 1}),
    }


def test_s_frames_reasons_values_and_floor_shares_with_zero_denominators():
    frame, raw, years = rep.s_frames(_s_results())
    assert list(frame.index) == list("abcdefg") and list(raw.index) == list("abcdefg")
    assert list(frame.columns) == QCOLS + ["reason"] and list(raw.columns) == QCOLS
    assert [None if pd.isna(r) else r for r in frame["reason"]] == [
        None, "non_finite_output", "non_finite_output", "scale_burn_in", "fit_failed", None, None]
    np.testing.assert_allclose(frame.loc[["a", "f", "g"], QCOLS].to_numpy(float),
                               [[v + 0.5 for v in FIN], FIN, FIN])
    assert np.isnan(frame.loc[["b", "c", "d", "e"], QCOLS].to_numpy(float)).all()
    np.testing.assert_allclose(raw.loc[["a", "f", "g"]].to_numpy(float), [FIN, FIN, FIN])
    assert np.isnan(raw.loc[["b", "c", "d", "e"]].to_numpy(float)).all()
    assert years["2023"]["status"] == "ok" and years["2024"]["status"] == "fit_failed"
    assert years["2023"]["floor"]["events_share"] == pytest.approx(1 / 3)
    assert math.isnan(years["2023"]["floor"]["donors_share"])  # no donors: a zero denominator is NaN
    assert math.isnan(years["2024"]["floor"]["events_share"]) and math.isnan(years["2024"]["floor"]["donors_share"])
    assert years["2025"]["floor"] == {"events": 2, "events_bound": 0, "donors": 4, "donors_bound": 1,
                                      "events_share": 0.0, "donors_share": 0.25}


# ---------- 3. headline_flags ----------

def _arm(ids, value):
    rows = pd.DataFrame([[value] * 7] * len(ids), index=ids, columns=QCOLS, dtype=float)
    return {"paired_ids": pd.Index(ids), "conditional": {"rows": rows}}


def _frame(reasons, value):
    """A baseline frame: {event id: reason}, every quantile equal to value."""
    ids = list(reasons)
    frame = pd.DataFrame([[value] * 7] * len(ids), index=ids, columns=QCOLS, dtype=float)
    frame["reason"] = [reasons[i] for i in ids]
    return frame


def _mean_pinball(y, value):
    """Mean over events and levels of max(a * d, (a - 1) * d) with d = y - value (every quantile is value)."""
    return float(np.mean([max(a * (yy - value), (a - 1) * (yy - value)) for yy in y for a in LEVELS]))


def _flag_setup():
    results = {"Q": _arm(list("pqrs"), 10.0), "Q_raw": _arm(list("pqrs"), 3.0)}
    frames = {
        "B1": _frame({"p": None, "q": None, "r": None, "s": "pool_too_small"}, 2.0),  # scored: p, q, r
        "B2": _frame({"p": None, "q": None, "s": None}, 5.0),                          # r absent: p, q, s
        "B3": _frame({"p": None, "q": "not_testable", "r": None, "s": None}, 3.0),      # scored: p, r, s
    }
    return results, frames


def test_headline_flags_use_each_baselines_own_subset_and_the_strict_comparison():
    results, frames = _flag_setup()
    flags = {(f["arm"], f["baseline"]): f for f in rep.headline_flags(results, frames, EVENTS)}
    arm_value, base_value = {"Q": 10.0, "Q_raw": 3.0}, {"B1": 2.0, "B2": 5.0, "B3": 3.0}
    subsets = {"B1": "pqr", "B2": "pqs", "B3": "prs"}
    expected_lost = {("Q", "B1"): True, ("Q", "B2"): True, ("Q", "B3"): True,
                     ("Q_raw", "B1"): True, ("Q_raw", "B2"): False,
                     ("Q_raw", "B3"): False}  # Q_raw equals B3 exactly on its subset: a tie is not lost
    assert sorted(flags) == sorted(expected_lost)
    for (arm, base), flag in flags.items():
        ids = list(subsets[base])
        y = [float(EVENTS.loc[i, "y"]) for i in ids]
        assert flag["n"] == len(ids)
        assert flag["arm_mean_pinball"] == pytest.approx(_mean_pinball(y, arm_value[arm]), rel=1e-12)
        assert flag["baseline_mean_pinball"] == pytest.approx(_mean_pinball(y, base_value[base]), rel=1e-12)
        assert flag["lost"] is expected_lost[(arm, base)], (arm, base)


def test_no_scored_pair_for_a_baseline_gives_nan_means_and_no_lost_verdict():
    results, _ = _flag_setup()
    frames = {base: _frame({i: "pool_too_small" for i in "pqrs"}, 2.0) for base in ("B1", "B2", "B3")}
    rows = rep.headline_flags(results, frames, EVENTS)
    assert len(rows) == 6 and [r["n"] for r in rows] == [0] * 6
    assert all(math.isnan(r["arm_mean_pinball"]) and math.isnan(r["baseline_mean_pinball"]) and r["lost"] is None
               for r in rows)
