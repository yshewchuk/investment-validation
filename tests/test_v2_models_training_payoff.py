"""``engine.v2.models.training.payoff`` -- fitting causal payoff artifacts.

guides/rearchitecture_phase5_models.md P5-4 acceptance for the training
layer: "bit-identical to today's inline fit on the same rows"; "a planted
future label/upstream in-sample leakage fails" (here: causality -- a
post-cutoff row can't enter a fold's artifact); "reusing native_payoff's
fitting math" (checked directly: the builder calls the SAME function, so
this is a parity/wiring test, not a re-implementation of the arithmetic
native_payoff's own tests already pin against legacy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.v2.models.payoff_artifact import PayoffLineArtifact, PayoffSurfaceArtifact
from engine.v2.models.training.payoff import (
    build_payoff_line_artifact,
    build_payoff_surface_artifact,
)
from engine.v2.scoring import native_payoff


def _synthetic_trades(n: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    driver = rng.uniform(0.0, 8.0, size=n)
    spot = np.full(n, 100.0)
    noise = rng.normal(0.0, 0.01, size=n)
    exit_value = (0.02 + 0.004 * driver + noise) * spot
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "driver": driver, "spot_entry": spot, "exit_value": exit_value,
        "exit_date": dates.astype(str),
    })


def _rows(trades: pd.DataFrame) -> list[dict]:
    return trades.to_dict("records")


def _synthetic_runup_rows(n: int, *, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    implied = rng.uniform(0.0, 8.0, size=n)
    spot_entry = np.full(n, 100.0)
    signed_move = rng.normal(0.0, 3.0, size=n)
    spot_exit = spot_entry * np.exp(signed_move / 100.0)
    strike = np.full(n, 100.0)
    moneyness = 100.0 * np.log(spot_exit / strike)
    design = native_payoff.runup_payoff_design(implied, moneyness)
    coeffs = np.array([0.02, 0.004, 0.001, 0.0005, 0.0007, 0.0003])
    noise = rng.normal(0.0, 0.002, size=n)
    exit_value = (design @ coeffs + noise) * spot_entry
    dates = pd.date_range("2020-01-01", periods=n, freq="D").astype(str)
    return [
        {"driver": float(implied[i]), "spot_entry": float(spot_entry[i]),
         "spot_exit": float(spot_exit[i]), "strike": float(strike[i]),
         "exit_value": float(exit_value[i]), "exit_date": dates[i]}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# bit-identical to the inline fit on the same rows
# ---------------------------------------------------------------------------


def test_build_payoff_line_artifact_bit_identical_to_inline_fit():
    trades = _synthetic_trades(300, seed=7)
    rows = _rows(trades)
    cutoff = str(trades["exit_date"].iloc[250])

    inline = native_payoff.fit_payoff_line(rows, before=cutoff)
    artifact = build_payoff_line_artifact(
        rows, strategy="STR-THRU", driver="abs_move", alpha=0.5, before=cutoff,
    )

    assert isinstance(artifact, PayoffLineArtifact)
    assert artifact.n == inline["n"]
    assert artifact.intercept == inline["intercept"]
    assert artifact.slope == inline["slope"]
    assert artifact.r == inline["r"]
    if np.isnan(inline["resid_sd"]):
        assert np.isnan(artifact.resid_sd)
    else:
        assert artifact.resid_sd == inline["resid_sd"]
    assert artifact.residuals == tuple(float(v) for v in inline["residuals"])


def test_build_payoff_surface_artifact_bit_identical_to_inline_fit():
    rows = _synthetic_runup_rows(300, seed=3)
    cutoff = str(rows[250]["exit_date"])

    inline = native_payoff.fit_runup_payoff_surface(rows, before=cutoff)
    artifact = build_payoff_surface_artifact(rows, alpha=0.5, before=cutoff)

    assert isinstance(artifact, PayoffSurfaceArtifact)
    assert artifact.n == inline["n"]
    assert artifact.coefficients == tuple(float(v) for v in inline["coefficients"])
    assert artifact.r == inline["r"]
    assert artifact.resid_sd == inline["resid_sd"]
    assert artifact.residuals == tuple(float(v) for v in inline["residuals"])


def test_build_payoff_line_artifact_returns_none_below_min_trades():
    rows = _rows(_synthetic_trades(10, seed=1))
    assert build_payoff_line_artifact(
        rows, strategy="STR-THRU", driver="abs_move", alpha=0.5,
    ) is None


def test_build_payoff_surface_artifact_returns_none_below_min_trades():
    rows = _synthetic_runup_rows(10, seed=1)
    assert build_payoff_surface_artifact(rows, alpha=0.5) is None


# ---------------------------------------------------------------------------
# causality: a post-cutoff row cannot enter a fold's artifact
# ---------------------------------------------------------------------------


def test_causal_cutoff_excludes_a_post_cutoff_row_from_the_line_artifact():
    good_rows = [
        {"driver": 0.0, "spot_entry": 100.0, "exit_value": 2.0, "exit_date": "2026-09-01"},
        {"driver": 10.0, "spot_entry": 100.0, "exit_value": 6.0, "exit_date": "2026-09-01"},
    ]
    future_row = {"driver": 5.0, "spot_entry": 100.0, "exit_value": 999.0,
                  "exit_date": "2026-09-20"}

    without_future = build_payoff_line_artifact(
        good_rows, strategy="STR-THRU", driver="abs_move", alpha=0.5,
        before="2026-09-16", min_trades=2,
    )
    with_future = build_payoff_line_artifact(
        good_rows + [future_row], strategy="STR-THRU", driver="abs_move",
        alpha=0.5, before="2026-09-16", min_trades=2,
    )
    assert with_future.n == without_future.n == 2
    assert with_future.content_hash == without_future.content_hash
    assert with_future.window_end == "2026-09-01"

    # And the future row DOES change the fit once it is no longer excluded --
    # proving the artifact is causal by exclusion, not by coincidence.
    unbounded = build_payoff_line_artifact(
        good_rows + [future_row], strategy="STR-THRU", driver="abs_move",
        alpha=0.5, before=None, min_trades=2,
    )
    assert unbounded.n == 3
    assert unbounded.content_hash != with_future.content_hash


def test_causal_cutoff_excludes_a_post_cutoff_row_from_the_surface_artifact():
    rows = _synthetic_runup_rows(300, seed=5)
    cutoff = str(rows[250]["exit_date"])
    future_row = dict(rows[0])
    future_row["exit_date"] = "2099-01-01"
    future_row["exit_value"] = 1e9

    without_future = build_payoff_surface_artifact(rows, alpha=0.5, before=cutoff)
    with_future = build_payoff_surface_artifact(
        rows + [future_row], alpha=0.5, before=cutoff,
    )
    assert with_future.n == without_future.n
    assert with_future.content_hash == without_future.content_hash


def test_causal_window_never_includes_a_row_on_or_after_the_cutoff():
    rows = [
        {"driver": 0.0, "spot_entry": 100.0, "exit_value": 2.0, "exit_date": "2026-09-01"},
        {"driver": 1.0, "spot_entry": 100.0, "exit_value": 3.0, "exit_date": "2026-09-16"},
        {"driver": 2.0, "spot_entry": 100.0, "exit_value": 4.0, "exit_date": "2026-09-02"},
    ]
    artifact = build_payoff_line_artifact(
        rows, strategy="STR-THRU", driver="abs_move", alpha=0.5,
        before="2026-09-16", min_trades=2,
    )
    assert artifact.n == 2
    assert artifact.window_start == "2026-09-01"
    assert artifact.window_end == "2026-09-02"


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def test_artifact_provenance_matches_the_call():
    rows = _rows(_synthetic_trades(250, seed=2))
    artifact = build_payoff_line_artifact(
        rows, strategy="CTR5", driver="abs_move", alpha=0.75, before="2020-09-01",
    )
    assert artifact.strategy == "CTR5"
    assert artifact.driver == "abs_move"
    assert artifact.alpha == 0.75
    assert artifact.cutoff == "2020-09-01"
    assert artifact.key == ("CTR5", 0.75, "2020-09-01")
