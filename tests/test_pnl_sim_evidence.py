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
