"""Shared synthetic fixtures for the D19 render-parity comparator tests.

Importable from BOTH the parent test process and the bounded subprocess
worker (via ``PHASE2_RENDER_PARITY_TEST_PATCH=tests.render_parity_support``)
so the two sides build the exact same panel/trades/registry from the exact
same code -- the only way an "identical" scenario can legitimately produce a
byte-identical bundle on both sides.
"""
from __future__ import annotations

import json
import sqlite3
import tarfile
from pathlib import Path

import pandas as pd

from engine.score import ScoreResult

TICKER = "ZZTEST"
AS_OF = pd.Timestamp("2026-08-10")
EVENT = pd.Timestamp("2026-08-12")
ROW_ID = "2026-08-10|ZZTEST|STR-THRU|100.0000|2026-08-13"


def result(strike=100.0, strike_offset=None, **kwargs) -> dict:
    base = dict(
        as_of=AS_OF, event_date=EVENT, session="AMC", exp_pnl_model=0.031,
        win_model=0.56, model_p10=-0.22, model_p90=0.41, exp_pnl_analog=0.028,
        win_analog=0.54, ci_low=-0.01, ci_high=0.07, n_analogs=120,
        gate_score=0.71, gate_threshold=0.6, gate_pass=True, entry_date=EVENT,
        exit_date=EVENT + pd.Timedelta(days=1), strike=strike, spot=100.0,
        expiry=EVENT + pd.Timedelta(days=2), entry_cost=6.0, dte_entry=2,
        payoff={"intercept": 0.01, "slope": 0.006, "n": 400},
        driver_name="abs_move", driver_prediction=7.5,
        model_versions={"size": "size_v13"}, snapshot_hash="snap-test",
    )
    base.update(kwargs)
    row = ScoreResult(ticker=TICKER, strategy="STR-THRU", **base).as_dict()
    row["strike_offset"] = strike_offset
    return row


def score_document() -> dict:
    board = [result()]
    ladder = [result(strike=95.0, strike_offset=-5.0)]
    for row in board + ladder:
        row["row_id"] = "|".join(str(row.get(k, "")) for k in
                                 ("ticker", "strategy", "event_date", "strike", "expiry"))
    return {"rows": board, "ladder": ladder, "analog_entry_coverage": 1.0,
            "expected_population": [f"{TICKER}|STR-THRU|{EVENT.date()}"],
            "observed_population": [f"{TICKER}|STR-THRU|{EVENT.date()}"],
            "tickers": [TICKER]}


def finality_doc() -> dict:
    return {"date": str(AS_OF.date()), "market_wide": True, "daily_share": 1.0,
            "chain_share": 1.0, "is_final": True, "detail": "synthetic", "tickers": 1,
            "covered": 1}


def model_evidence_doc() -> dict:
    return {"generated_at": "2026-08-01T00:00:00+00:00", "elapsed_s": 4.2,
            "models": {"size": {"inputs": {}}}}


def panel() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": [TICKER, TICKER], "date": pd.to_datetime(["2026-05-11", "2026-08-12"]),
        "k": [30, 31], "implied_move": [6.0, 7.2], "or_implied": [6.1, 7.2],
        "move": [-4.0, 7.9], "abs_move": [4.0, 7.9]})


def trades() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": [TICKER], "strategy": ["STR-THRU"], "event_date": [EVENT],
        "entry_date": [EVENT], "exit_date": [EVENT + pd.Timedelta(days=1)],
        "fill_alpha": [0.5], "entry_cost": [6.0], "exit_value": [6.6], "ret": [0.10]})


def prediction_row() -> dict:
    score = result()
    return {
        "schema_version": 3, "row_id": ROW_ID, "written_at": "2026-08-10T21:30:00+00:00",
        "as_of": str(AS_OF.date()), "decision_ts": "2026-08-10T21:30:00+00:00",
        "ticker": TICKER, "event_id": None, "event_date": str(EVENT.date()), "session": "AMC",
        "strategy": "STR-THRU",
        "settlement": {"policy": "close", "spec_version": 1, "structure_spec": None,
                       "structure_params": None, "variant": None},
        "structure": {"strike": 100.0, "expiry": str((EVENT + pd.Timedelta(days=2)).date()),
                      "legs": [], "decision_date": str(AS_OF.date()),
                      "entry_date": str(EVENT.date()),
                      "exit_date": str((EVENT + pd.Timedelta(days=1)).date()), "dte_entry": 2},
        "intended_prices": {"alpha": 0.5, "quote_date": str(AS_OF.date()), "quoted_cost": 6.0,
                            "entry_cost": 6.0, "spot": 100.0},
        "finality": {}, "score": score, "model_versions": {}, "snapshot_hash": "snap-test",
        "audit_receipt": None, "supersedes": None,
    }


def outcome_row() -> dict:
    return {"row_id": ROW_ID, "status": "resolved", "event_date": str(EVENT.date()),
            "resolved_at": "2026-08-15T21:00:00+00:00", "realized_pnl": 0.02,
            "realized_entry_cost": 6.1, "realized_exit_value": 8.0, "reason": None,
            "exit_finality": {"is_final": True}}


def build_ledger_generation_tar(tmp_path: Path) -> bytes:
    """A tiny catalog -> ``export_generation`` -> one generation -> tar bytes."""
    from engine.v2.ledger.export import export_generation

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE decisions (sequence INTEGER, kind TEXT, payload_json TEXT)")
    conn.execute("INSERT INTO decisions VALUES (1, 'prediction', ?)",
                (json.dumps(prediction_row()),))
    conn.execute("INSERT INTO decisions VALUES (2, 'outcome', ?)",
                (json.dumps(outcome_row()),))
    conn.commit()
    ledger_root = tmp_path / "ledger_root"
    generation = export_generation(conn, ledger_root, generation="gen1")
    tar_path = tmp_path / "ledger_generation.tar"
    with tarfile.open(tar_path, "w") as archive:
        for child in sorted(generation.iterdir()):
            archive.add(child, arcname=child.name)
    conn.close()
    return tar_path.read_bytes()


class FakeRegistry:
    def champion(self, role, strategy=None):
        raise KeyError(role)


class FakeScorer:
    """Stands in for ``engine.score.Scorer``: holds context/trades/registry only."""

    def __init__(self, *, context):
        self.context = context
        self.registry = FakeRegistry()
        self.trades = trades()


def patch_scorer(monkeypatch, features_module, score_module) -> None:
    """In-process monkeypatch, for the PARENT test that builds the v2 side."""
    context = features_module.FeatureContext(panel=panel(), daily=None, calendar=None)
    monkeypatch.setattr(features_module.FeatureContext, "load",
                        staticmethod(lambda *a, **k: context))
    monkeypatch.setattr(score_module, "Scorer", FakeScorer)


def patch() -> None:
    """Fresh-process patch hook for the bounded subprocess worker (no
    ``monkeypatch`` fixture available there -- the process exits right
    after, so a bare attribute assignment needs no cleanup).

    ``PHASE2_RENDER_PARITY_TEST_SELFCHECK_OK`` (``"1"``/``"0"``, default
    ``"1"``) additionally stubs ``engine.dashboard.selfcheck.selfcheck``
    itself: re-deriving a matching digest for a HAND-BUILT ``ScoreResult``
    row (never actually produced by real scoring) is its own, separately
    tested concern (AGAINST real boards; see AGENTS.md "Self-check parity"),
    not something D19's comparator plumbing needs to re-prove here. This
    keeps the synthetic test focused on the comparator's OWN subprocess/
    receipt machinery.
    """
    import os

    import engine.dashboard.selfcheck as selfcheck_module
    import engine.features as features_module
    import engine.score as score_module

    context = features_module.FeatureContext(panel=panel(), daily=None, calendar=None)
    features_module.FeatureContext.load = staticmethod(lambda *a, **k: context)
    score_module.Scorer = FakeScorer

    ok = os.environ.get("PHASE2_RENDER_PARITY_TEST_SELFCHECK_OK", "1") == "1"

    class _FakeSelfcheckResult:
        def as_dict(self):
            return {"ok": ok, "known": True, "detail": "stubbed for the D19 comparator test"}

    selfcheck_module.selfcheck = lambda *a, **k: _FakeSelfcheckResult()
