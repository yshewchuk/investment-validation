"""EXP-188: the inner stage's decisions pass (``_inner_decisions``): decisions_inner.parquet, its receipt and its replay.
Real readers and driver; pool, simulation, scorers and tuning stubbed through ``outer_stubs``."""
import hashlib
import json

import pandas as pd

from experiments import size_pnl_driver as driver, size_pnl_tuning as tuning
from tests.v2.research.test_size_pnl_driver import (  # noqa: F401  (bounded, stubs, structure_gate_passes, equivalence_passes are fixtures)
    OK, bounded, chunk, sources, stubs, structure_gate_passes, equivalence_passes)
from tests.v2.research.test_size_pnl_driver_outer import outer_stubs, use_chunks  # noqa: F401

FRONT = ["model_id", "book", "arm_or_rule", "structure", "event_id", "event_date", "status", "reason"]


def _inner(tmp_path, monkeypatch, out=None, src=None):
    """Inner run over E1 (2023 Q2, burn-in) and E2 (2024 Q3, scored)."""
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    return driver.run(src or sources(), "inner", out or tmp_path, prereg=OK)


def _frame(out):
    return pd.read_parquet(out / "decisions_inner.parquet")


def test_inner_publishes_decisions_for_the_scored_window_only(tmp_path, outer_stubs, monkeypatch):
    summary = _inner(tmp_path, monkeypatch)
    assert (tmp_path / "decisions_inner.parquet").exists()
    frame = _frame(tmp_path)
    assert set(frame["event_id"].astype(str)) == {"E2"}  # E1 is 2023-06: burn-in history only
    dates = pd.to_datetime(frame["event_date"])
    assert (dates >= pd.Timestamp("2023-07-01")).all() and (dates < pd.Timestamp("2025-01-01")).all()
    assert list(frame.columns[:len(FRONT)]) == FRONT
    assert summary["inner_decisions_rows"] == len(frame)
    assert set(frame["book"]) == {"short_vol", "straddle"}
    assert set(frame.loc[frame["book"] == "short_vol", "arm_or_rule"]) == {"as_is", "retuned"}
    assert set(frame.loc[frame["book"] == "straddle", "arm_or_rule"]) == set(driver.straddle.RULES)


def test_inner_decisions_label_only_the_retuned_arm_in_sample(tmp_path, outer_stubs, monkeypatch):
    _inner(tmp_path, monkeypatch)
    frame = _frame(tmp_path)
    retuned = frame["arm_or_rule"] == "retuned"
    assert retuned.any()
    assert frame.loc[retuned, "in_sample"].all()
    assert not frame.loc[~retuned, "in_sample"].any()


def test_inner_retuned_rows_use_the_frozen_c_and_the_receipt_binds_it(tmp_path, outer_stubs, monkeypatch):
    monkeypatch.setattr(tuning, "tune", lambda items, mid, **kw: {"stage": "tuning", "model_id": mid, "c": 1.25, "skipped": False})
    seen, scored = set(), driver.shortvol.score_events

    def recording(chunk_, events, sims, name, *, alpha=0.5, c=1.0):
        seen.add(c)
        return scored(chunk_, events, sims, name, alpha=alpha, c=c)

    monkeypatch.setattr(driver.shortvol, "score_events", recording)
    summary = _inner(tmp_path, monkeypatch)
    assert {1.0, 1.25} <= seen  # as_is scores at unity, retuned at the frozen c
    rec = tuning.load_receipt(tmp_path / "receipts" / "inner_decisions.json")
    inner = tuning.load_receipt(tmp_path / "receipts" / "inner.json")
    assert rec["stage"] == "inner_decisions"
    assert rec["c"] == {mid: 1.25 for mid in summary["models"]}
    assert rec["tuning"] == inner["models"]
    assert rec["decisions_sha256"] == hashlib.sha256((tmp_path / "decisions_inner.parquet").read_bytes()).hexdigest()
    assert rec["rows"] > 0 and rec["rows"] == summary["inner_decisions_rows"]
    assert rec["emit_from"] == "2023-07-01"


def test_inner_json_and_tuning_receipts_do_not_change_content(tmp_path, outer_stubs, monkeypatch):
    def receipts(out):
        return {p.name: p.read_bytes() for p in (out / "receipts").iterdir() if p.name == "inner.json" or p.name.startswith("tuning_")}

    real = driver._inner_decisions
    monkeypatch.setattr(driver, "_inner_decisions", lambda *a, **k: 0)
    _inner(tmp_path, monkeypatch, out=tmp_path / "stubbed")
    monkeypatch.setattr(driver, "_inner_decisions", real)
    _inner(tmp_path, monkeypatch, out=tmp_path / "real")
    stubbed, actual = receipts(tmp_path / "stubbed"), receipts(tmp_path / "real")
    names = set(stubbed) & set(actual)
    assert "inner.json" in names and any(n.startswith("tuning_") for n in names)
    assert {n: stubbed[n] for n in names} == {n: actual[n] for n in names}
    assert "decisions_sha256" not in json.loads(actual["inner.json"])


def test_exact_inner_replay_keeps_decisions_inner(tmp_path, outer_stubs, monkeypatch):
    first = _inner(tmp_path, monkeypatch)
    before = (tmp_path / "decisions_inner.parquet").read_bytes()
    second = _inner(tmp_path, monkeypatch)
    assert second == first
    assert (tmp_path / "decisions_inner.parquet").read_bytes() == before
