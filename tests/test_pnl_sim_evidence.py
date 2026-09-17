from __future__ import annotations

import hashlib

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
