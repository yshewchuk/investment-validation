from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from engine.pnl_sim import MIN_POOL, ResidualPool, expected_pnl


def _history(rows: int) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame({
        "event_date": pd.date_range("2020-01-01", periods=rows, freq="D"),
        "pred_abs_move": 2.0 + index / rows * 8.0,
        "err_move": index / 100.0 - 1.5,
        "err_crush": 5.0 - index / 50.0,
    })


def test_draw_evidence_preserves_samples_and_records_exact_indices() -> None:
    history = _history(600)
    pool = ResidualPool(history, buckets=2)
    cutoff = pd.Timestamp("2022-01-01")

    default = pool.draw(cutoff, 3.0, 40, np.random.default_rng(912))
    evidence: dict = {}
    observed = pool.draw(
        cutoff,
        3.0,
        40,
        np.random.default_rng(912),
        evidence=evidence,
    )

    np.testing.assert_array_equal(observed[0], default[0])
    np.testing.assert_array_equal(observed[1], default[1])
    assert evidence["status"] == "selected"
    assert evidence["cutoff_index"] == 600
    assert evidence["causal_indices"] == list(range(600))
    assert evidence["bucket_index"] == 0
    assert len(evidence["eligible_indices"]) == 300
    assert evidence["fallback_used"] is False
    assert evidence["fallback_indices"] == []
    selected = np.asarray(evidence["selected_indices"])
    np.testing.assert_array_equal(observed[0], history.loc[selected, "err_move"])
    np.testing.assert_array_equal(observed[1], history.loc[selected, "err_crush"])


def test_evidence_rows_bind_selected_indices_to_exact_residual_values() -> None:
    history = _history(600)
    pool = ResidualPool(history, buckets=2)
    evidence: dict = {}
    pool.draw("2022-01-01", 3.0, 4, np.random.default_rng(17), evidence=evidence)

    rows = pool.evidence_rows(evidence["selected_indices"])

    assert len(rows) == 4
    assert rows[0]["event_date"].startswith("2020-")
    assert rows[0]["err_move"] == history.loc[evidence["selected_indices"][0], "err_move"]


def test_evidence_rows_reject_unknown_selection_index() -> None:
    with np.testing.assert_raises(IndexError):
        ResidualPool(_history(300)).evidence_rows([300])


def test_documented_population_shares_row_objects_not_the_wrapping_list() -> None:
    """The retention fix: each call returns a FRESH list (never cached or
    retained per cutoff — see `documented_population`'s docstring for why),
    but its elements are the SAME dict objects every time, sliced out of the
    one-time `_documented_rows()` build rather than rebuilt.
    """
    pool = ResidualPool(_history(600))

    first = pool.documented_population(300)
    second = pool.documented_population(300)
    other = pool.documented_population(299)

    assert first is not second  # not cached — nothing is retained per cutoff
    assert first == second  # but the content, and the row objects, agree
    assert all(a is b for a, b in zip(first, second))
    assert first[:299] == other
    assert all(a is b for a, b in zip(first, other))
    assert len(first) == 300
    assert first[0] == {
        "event_date": pd.Timestamp("2020-01-01").isoformat(),
        "pred_abs_move": 2.0,
        "err_move": -1.5,
        "err_crush": 5.0,
    }
    # Plain dicts (not the read-only row cache's mapping view): this is what
    # the checkpoint sink's JSON writer requires.
    assert type(first[0]) is dict


def _canonical_hash(rows: list[dict]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_documented_population_matches_the_old_per_cutoff_rebuild() -> None:
    """Byte-identical to the previous ``[dict(row) for row in
    _all_rows()[:cutoff]]`` shape, at several cutoffs including both edges
    (0 and the full length). The fix changes HOW the list is built (slice a
    once-built full copy instead of rebuilding one per cutoff), never WHAT it
    contains.
    """
    pool = ResidualPool(_history(400))

    for cutoff in (0, 1, 150, 399, 400):
        old = [dict(row) for row in pool._all_rows()[:cutoff]]
        new = pool.documented_population(cutoff)
        assert new == old
        assert _canonical_hash(new) == _canonical_hash(old)


def test_documented_population_marginal_cost_is_a_slice_not_a_copy() -> None:
    """The retention fix, measured: after the pool's rows are documented
    ONCE, a new DISTINCT cutoff costs a list of references (~8 bytes/row),
    not a fresh ``[dict(row) for row in ...]`` copy.

    This is the regression for the runaway: a 40-forward-event capture run
    hits close to 40 distinct cutoffs (one per distinct forward event_date,
    not shared the way a boundary/pinned/coarse rescore's cutoff is), and the
    OLD shape retained one full per-cutoff dict-copy PER distinct cutoff,
    unbounded -- see ``engine.pnl_sim.ResidualPool._documented_rows``.
    """
    import tracemalloc

    n = 20_000
    pool = ResidualPool(_history(n))

    tracemalloc.start()
    before_first = tracemalloc.take_snapshot()
    pool.documented_population(n)  # forces the one-time full build
    after_first = tracemalloc.take_snapshot()
    first_call_growth = sum(
        s.size_diff for s in after_first.compare_to(before_first, "lineno")
        if s.size_diff > 0
    )
    assert first_call_growth > 0

    # 39 more DISTINCT cutoffs, near the end of the pool (the shape a 40
    # forward-event pass actually produces).
    baseline = tracemalloc.take_snapshot()
    cutoffs = np.unique(np.linspace(n - 500, n, 40, dtype=int))
    for cutoff in cutoffs:
        pool.documented_population(int(cutoff))
    after = tracemalloc.take_snapshot()
    marginal_growth = sum(
        s.size_diff for s in after.compare_to(baseline, "lineno") if s.size_diff > 0
    )
    tracemalloc.stop()

    assert marginal_growth < 0.10 * first_call_growth, (
        marginal_growth, first_call_growth,
    )


def test_evidence_rows_are_read_only_and_shared_across_calls() -> None:
    """Rows are cached and reused by reference (the OOM fix); a caller must
    not be able to mutate them, because that would corrupt every OTHER
    request's checkpoint sharing the same pool."""
    pool = ResidualPool(_history(50))

    first = pool.evidence_rows([3])
    try:
        first[0]["err_move"] = 999.0
    except TypeError:
        pass
    else:
        raise AssertionError("expected a read-only row to reject mutation")

    # The cache is unaffected by the attempted (and rejected) mutation, and
    # the SAME row objects are handed back on a later call.
    second = pool.evidence_rows([3])
    assert second[0]["err_move"] == first[0]["err_move"]
    assert second[0] is first[0]


def test_draw_evidence_records_bucket_fallback_population() -> None:
    pool = ResidualPool(_history(300), buckets=10)
    evidence: dict = {}
    pool.draw(
        pd.Timestamp("2021-01-01"),
        3.0,
        20,
        np.random.default_rng(23),
        evidence=evidence,
    )

    assert 0 < len(evidence["eligible_indices"]) < MIN_POOL
    assert evidence["fallback_used"] is True
    assert evidence["fallback_indices"] == list(range(300))
    assert set(evidence["selected_indices"]) <= set(evidence["fallback_indices"])


def test_expected_pnl_evidence_preserves_exact_result_and_seed() -> None:
    pool = ResidualPool(_history(600), buckets=2)
    arguments = {
        "exit_legs": [{"strike": 100.0, "qty": 1.0, "side": "sell"}],
        "spot": 100.0,
        "entry_cost": 4.25,
        "pre_iv30": 42.0,
        "pred_abs_move": 3.0,
        "pred_iv_crush": -18.0,
        "dte_exit": 8.0,
        "event_date": pd.Timestamp("2022-01-01"),
        "pool": pool,
        "key": "STR-THRU|event-7",
        "draws": 500,
    }

    default = expected_pnl(**arguments)
    evidence: dict = {}
    observed = expected_pnl(**arguments, evidence=evidence)

    assert observed == default
    seed_material = "STR-THRU|event-7|2022-01-01 00:00:00"
    expected_seed = int.from_bytes(
        hashlib.sha256(seed_material.encode()).digest()[:8],
        "big",
    )
    assert evidence["status"] == "completed"
    assert evidence["seed"] == expected_seed
    assert evidence["seed_material"] == seed_material
    assert evidence["residual_draw"]["selected_indices"]
    assert evidence["residual_draw"]["cutoff_index"] == 600


def test_thin_pool_refusal_reports_causal_population_without_drawing() -> None:
    history = _history(MIN_POOL - 1)
    pool = ResidualPool(history)
    evidence: dict = {}

    result = expected_pnl(
        exit_legs=[{"strike": 100.0, "qty": 1.0, "side": "sell"}],
        spot=100.0,
        entry_cost=4.25,
        pre_iv30=42.0,
        pred_abs_move=5.0,
        pred_iv_crush=-18.0,
        dte_exit=8.0,
        event_date=pd.Timestamp("2022-01-01"),
        pool=pool,
        key="thin",
        evidence=evidence,
    )

    assert result is None
    assert evidence["status"] == "refused"
    assert evidence["refusal"] == "THIN_RESIDUAL_POOL"
    draw = evidence["residual_draw"]
    assert draw["status"] == "refused"
    assert draw["refusal"] == "THIN_RESIDUAL_POOL"
    assert draw["causal_indices"] == list(range(MIN_POOL - 1))
    assert draw["eligible_indices"] == []
    assert draw["fallback_indices"] == []
    assert draw["selected_indices"] == []


# ---------------------------------------------------------------------------
# mutation-pilot triage: behaviour of the live gate no test above pinned.
# Expected values are derived by hand from the documented legacy semantics.
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from engine import pnl_sim  # noqa: E402
from engine.pnl_sim import black_scholes_put, load_history, trailing_cutoff  # noqa: E402


def _flat_pool(rows: int = 300) -> ResidualPool:
    """Zero move and crush errors, all before 2022: every draw is the forecast."""
    return ResidualPool(pd.DataFrame({
        "event_date": pd.date_range("2020-01-01", periods=rows, freq="D"),
        "pred_abs_move": np.linspace(1.0, 9.0, rows),
        "err_move": 0.0,
        "err_crush": 0.0,
    }))


def _at_expiry(exit_legs, **overrides):
    arguments = dict(
        exit_legs=exit_legs, spot=100.0, entry_cost=10.0, pre_iv30=40.0,
        pred_abs_move=10.0, pred_iv_crush=-20.0, dte_exit=0.0,
        event_date=pd.Timestamp("2022-01-03"), pool=_flat_pool(), key="STR-THRU",
        draws=400,
    )
    arguments.update(overrides)
    return expected_pnl(**arguments)


def test_black_scholes_put_at_expiry_is_intrinsic_value():
    np.testing.assert_array_equal(
        black_scholes_put([90.0, 100.0, 110.0], 100.0, 0.0, 0.3), [10.0, 0.0, 0.0])
    assert float(black_scholes_put(100.0, 100.0, -1.0, 0.3)) == 0.0


def test_leg_sides_quantities_and_skipped_legs_at_expiry():
    """|move| = 10 on every draw, so spot exits at 90 or 110. Long two 105
    puts (exit side "sell") and short one 95 put (exit side "buy"): the down
    path is worth 2*15 - 5 = 25 and the up path 0, so on entry cost 10 the
    return is +1.5 or -1 and the mean is 2.5*win - 1. The legs listed first
    must be skipped, not stop the loop: no strike, a non-finite strike, a
    zero quantity and a missing quantity (which defaults to zero)."""
    skipped = [
        {"strike": None, "qty": 1.0, "side": "sell"},
        {"strike": float("nan"), "qty": 1.0, "side": "sell"},
        {"strike": 99.0, "qty": 0.0, "side": "sell"},
        {"strike": 99.0, "side": "sell"},
    ]
    legs = skipped + [
        {"strike": 105.0, "qty": 2.0, "side": "SELL"},
        {"strike": 95.0, "qty": 1.0, "side": "buy"},
    ]
    result = _at_expiry(legs)
    win = result["win_sim"]
    assert 0.3 < win < 0.7
    assert result["exp_pnl_sim"] == pytest.approx(2.5 * win - 1.0)
    assert (result["sim_p10"], result["sim_p90"]) == (-1.0, 1.5)
    assert result["pool_n"] == 300
    no_side = _at_expiry([{"strike": 105.0, "qty": 1.0}])  # no side: a short leg
    assert no_side["sim_p10"] == pytest.approx(-2.5)


@pytest.mark.parametrize("change, refusal", [
    ({"exit_legs": []}, "MISSING_EXIT_LEGS"),
    ({"spot": float("nan")}, "NONFINITE_INPUT"),
    ({"pred_iv_crush": float("inf")}, "NONFINITE_INPUT"),
    ({"pre_iv30": None}, "NONFINITE_INPUT"),
    ({"pre_iv30": 0.0}, "INVALID_INPUT"),
    ({"entry_cost": 0.0}, "INVALID_INPUT"),
    ({"dte_exit": -1.0}, "INVALID_INPUT"),
])
def test_unsimulable_inputs_are_undetermined_with_and_without_evidence(change, refusal):
    change = dict(change)
    legs = change.pop("exit_legs", [{"strike": 100.0, "qty": 1.0, "side": "sell"}])
    assert _at_expiry(legs, **change) is None
    evidence: dict = {}
    assert _at_expiry(legs, evidence=evidence, **change) is None
    assert (evidence["status"], evidence["refusal"]) == ("refused", refusal)
    assert evidence["schema_version"] == "expected_pnl_evidence.v1"


@pytest.mark.parametrize("change", [
    {"pre_iv30": 0.5}, {"entry_cost": 0.5}, {"dte_exit": 0.0}, {"dte_exit": 0.5},
])
def test_small_positive_inputs_still_simulate(change):
    legs = [{"strike": 100.0, "qty": 1.0, "side": "sell"}]
    assert _at_expiry(legs, **change) is not None


def test_pool_is_causal_and_exactly_min_pool_rows_simulate():
    dates = pd.date_range("2020-01-01", periods=MIN_POOL + 50, freq="D")
    history = pd.DataFrame({"event_date": dates, "pred_abs_move": 5.0,
                            "err_move": np.where(np.arange(len(dates)) < MIN_POOL, 0.0, 50.0),
                            "err_crush": 0.0})
    pool = ResidualPool(history)
    cutoff = dates[MIN_POOL]
    assert pool.before(cutoff) == MIN_POOL
    move, _ = pool.draw(cutoff, 5.0, 2000, np.random.default_rng(1))
    assert move.size == 2000 and (move == 0.0).all()  # nothing on/after the cutoff
    assert pool.draw(dates[MIN_POOL - 1], 5.0, 5, np.random.default_rng(1))[0].size == 0


def test_pool_rejects_missing_columns_and_keeps_rows_nan_elsewhere():
    with pytest.raises(ValueError):
        ResidualPool(pd.DataFrame({"event_date": [], "pred_abs_move": [], "err_move": []}))
    history = _history(300)
    history["ticker"] = None  # an unrelated, all-missing column
    assert len(ResidualPool(history)) == 300


def test_default_pool_is_deciles_and_selected_evidence_names_the_cutoff():
    pool = ResidualPool(_history(3000))
    evidence: dict = {}
    pool.draw("2030-01-01", 5.0, 10, np.random.default_rng(3), evidence=evidence)
    assert evidence["bucket_count"] == 10
    assert len(evidence["bucket_edges"]) == 9
    assert len(evidence["eligible_indices"]) == 300
    assert evidence["cutoff"] == "2030-01-01T00:00:00"


def test_bucket_membership_is_right_closed_at_a_tied_edge():
    """Legacy buckets with ``searchsorted(side="right")``: a prediction equal
    to a decile edge belongs to the bucket above it, and so do tied rows."""
    history = pd.DataFrame({
        "event_date": pd.date_range("2020-01-01", periods=601, freq="D"),
        "pred_abs_move": [0.0] * 300 + [1.0] * 301,
        "err_move": 0.0, "err_crush": 0.0,
    })
    evidence: dict = {}
    ResidualPool(history, buckets=2).draw("2030-01-01", 1.0, 5, np.random.default_rng(0),
                                          evidence=evidence)
    assert evidence["bucket_edges"] == [1.0]
    assert evidence["bucket_index"] == 1
    assert evidence["eligible_indices"] == list(range(300, 601))
    assert evidence["fallback_used"] is False


def test_a_bucket_of_exactly_min_pool_rows_does_not_fall_back():
    history = pd.DataFrame({
        "event_date": pd.date_range("2020-01-01", periods=2 * MIN_POOL, freq="D"),
        "pred_abs_move": [0.0] * MIN_POOL + [1.0] * MIN_POOL,
        "err_move": 0.0, "err_crush": 0.0,
    })
    evidence: dict = {}
    ResidualPool(history, buckets=2).draw("2030-01-01", 1.0, 5, np.random.default_rng(0),
                                          evidence=evidence)
    assert len(evidence["eligible_indices"]) == MIN_POOL
    assert evidence["fallback_used"] is False


def test_evidence_rows_accept_index_zero_and_reject_negative_indices():
    pool = ResidualPool(_history(300))
    assert pool.evidence_rows([0])[0]["event_date"].startswith("2020-01-01")
    with pytest.raises(IndexError):
        pool.evidence_rows([-1])


# -- trailing gate bar and its stored history ------------------------------


def _gate_history():
    """Monthly-window fixture: as_of 2024-07-15 -> window [2024-01-01, 2024-07-01)."""
    inside = pd.date_range("2024-01-01", "2024-06-30", periods=120)
    values = np.linspace(-0.5, 0.5, 120)
    frame = pd.DataFrame({"event_date": inside, "exp_pnl_sim": values})
    outside = pd.DataFrame({
        "event_date": pd.to_datetime(["2023-12-31", "2024-07-01", "2024-07-10"]),
        "exp_pnl_sim": [9.0, 9.0, 9.0],
    })
    missing = pd.DataFrame({"event_date": pd.to_datetime(["2024-03-01"]),
                            "exp_pnl_sim": [np.nan]})
    return pd.concat([frame, outside, missing], ignore_index=True), values


def test_trailing_cutoff_is_the_top_fifth_of_the_prior_six_calendar_months():
    history, inside = _gate_history()
    bar = trailing_cutoff(history, "2024-07-15")
    assert bar == pytest.approx(float(np.quantile(inside, 0.80)))
    assert trailing_cutoff(history, "2024-07-15", quantile=0.5) == pytest.approx(
        float(np.quantile(inside, 0.5)))


def test_trailing_cutoff_window_boundaries_and_minimum():
    day = pd.DataFrame({"event_date": pd.to_datetime(["2024-01-01"] * 100),
                        "exp_pnl_sim": np.linspace(0.0, 1.0, 100)})
    assert trailing_cutoff(day, "2024-07-15") == pytest.approx(0.8)  # start is inclusive
    assert trailing_cutoff(day.iloc[:99], "2024-07-15") is None
    assert trailing_cutoff(day, "2024-07-15", min_window=101) is None
    assert trailing_cutoff(day, "2024-07-15", window_months=5) is None
    assert trailing_cutoff(None, "2024-07-15") is None
    assert trailing_cutoff(pd.DataFrame(), "2024-07-15") is None


def test_load_history_reads_the_given_or_default_path(tmp_path, monkeypatch):
    frame = pd.DataFrame({"event_date": ["2024-01-02", "2024-01-03"],
                          "exp_pnl_sim": [0.1, -0.2]})
    given = tmp_path / "given.parquet"
    frame.to_parquet(given)
    loaded = load_history(str(given))
    assert list(loaded["event_date"]) == list(pd.to_datetime(frame["event_date"]))
    assert list(loaded["exp_pnl_sim"]) == [0.1, -0.2]
    assert load_history(str(tmp_path / "absent.parquet")) is None

    from engine import paths

    monkeypatch.setattr(paths, "ROOT", tmp_path)
    assert load_history() is None
    default = tmp_path / pnl_sim.HISTORY_PATH
    default.parent.mkdir(parents=True)
    frame.iloc[:1].to_parquet(default)
    assert len(load_history()) == 1
