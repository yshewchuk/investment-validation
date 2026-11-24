"""EXP-188: the inner stage's decisions pass (``_inner_decisions``): decisions_inner.parquet, its receipt and its replay.
Real readers and driver; pool, simulation and scorers stubbed through ``outer_stubs``. Short vol is as-is only (Amendment 6)."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from experiments import size_pnl_driver as driver, size_pnl_receipts as tuning
from experiments.size_pnl_shortvol import Refusal
from tests.v2.research.test_size_pnl_driver import (  # noqa: F401  (bounded, stubs, structure_gate_passes, equivalence_passes are fixtures)
    CONFLICT, OK, bounded, chunk, sources, stubs, structure_gate_passes, equivalence_passes)
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
    assert set(frame["book"]) == {"short_vol"}  # no gate bound: the inner stage emits no straddle rows (Amendment 6)
    assert set(frame.loc[frame["book"] == "short_vol", "arm_or_rule"]) == {"as_is"}


def test_inner_decisions_carry_no_in_sample_or_c_columns(tmp_path, outer_stubs, monkeypatch):
    _inner(tmp_path, monkeypatch)
    frame = _frame(tmp_path)
    assert "in_sample" not in frame.columns and "c" not in frame.columns


def test_inner_decisions_receipt_binds_models_and_has_no_tuning_or_c(tmp_path, outer_stubs, monkeypatch):
    summary = _inner(tmp_path, monkeypatch)
    rec = tuning.load_receipt(tmp_path / "receipts" / "inner_decisions.json")
    assert rec["stage"] == "inner_decisions"
    assert sorted(rec["models"]) == sorted(summary["models"])
    assert "tuning" not in rec and "c" not in rec
    assert rec["decisions_sha256"] == hashlib.sha256((tmp_path / "decisions_inner.parquet").read_bytes()).hexdigest()
    assert rec["rows"] > 0 and rec["rows"] == summary["inner_decisions_rows"]
    assert rec["emit_from"] == "2023-07-01"


def test_inner_json_does_not_change_content(tmp_path, outer_stubs, monkeypatch):
    def receipts(out):
        return {p.name: p.read_bytes() for p in (out / "receipts").iterdir() if p.name == "inner.json"}

    real = driver._inner_decisions
    monkeypatch.setattr(driver, "_inner_decisions", lambda *a, **k: 0)
    _inner(tmp_path, monkeypatch, out=tmp_path / "stubbed")
    monkeypatch.setattr(driver, "_inner_decisions", real)
    _inner(tmp_path, monkeypatch, out=tmp_path / "real")
    stubbed, actual = receipts(tmp_path / "stubbed"), receipts(tmp_path / "real")
    names = set(stubbed) & set(actual)
    assert "inner.json" in names and not any(p.name.startswith("tuning_") for p in (tmp_path / "real" / "receipts").iterdir())
    assert {n: stubbed[n] for n in names} == {n: actual[n] for n in names}
    assert "decisions_sha256" not in json.loads(actual["inner.json"])


def test_outer_refuses_when_inner_decisions_receipt_is_missing(tmp_path, outer_stubs, monkeypatch):
    _inner(tmp_path, monkeypatch)
    (tmp_path / "receipts" / "inner_decisions.json").unlink()
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "outer", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inner_decisions_missing"


def test_outer_refuses_when_inner_decisions_is_bound_to_other_inputs(tmp_path, outer_stubs, monkeypatch):
    _inner(tmp_path, monkeypatch)
    rec = tuning.load_receipt(tmp_path / "receipts" / "inner_decisions.json")
    tampered = {k: v for k, v in rec.items() if k != "content_hash"}
    tampered["inputs"] = "0" * 64
    (tmp_path / "receipts" / "inner_decisions.json").unlink()
    tuning.write_receipt(tmp_path / "receipts" / "inner_decisions.json", tampered)
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "outer", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inner_decisions_mismatch"


def test_outer_refuses_when_inner_decisions_models_is_null(tmp_path, outer_stubs, monkeypatch):
    # A hash-valid receipt with models=null must refuse as a mismatch, not raise a builtin exception.
    _inner(tmp_path, monkeypatch)
    rec = tuning.load_receipt(tmp_path / "receipts" / "inner_decisions.json")
    tampered = {k: v for k, v in rec.items() if k != "content_hash"}
    tampered["models"] = None
    (tmp_path / "receipts" / "inner_decisions.json").unlink()
    tuning.write_receipt(tmp_path / "receipts" / "inner_decisions.json", tampered)
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "outer", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inner_decisions_mismatch"


def test_outer_refuses_when_decisions_inner_bytes_changed_after_publish(tmp_path, outer_stubs, monkeypatch):
    _inner(tmp_path, monkeypatch)
    with open(tmp_path / "decisions_inner.parquet", "ab") as handle:
        handle.write(b"X")
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "outer", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inner_decisions_mismatch"


def test_exact_inner_replay_keeps_decisions_inner(tmp_path, outer_stubs, monkeypatch):
    first = _inner(tmp_path, monkeypatch)
    before = (tmp_path / "decisions_inner.parquet").read_bytes()
    second = _inner(tmp_path, monkeypatch)
    assert second == first
    assert (tmp_path / "decisions_inner.parquet").read_bytes() == before


def _marks(ids, pnl, mid_marks):
    """Literal marks of a taken row: one lot at rung 2, exit 2.0 (3.0 under mid_marks); ``pnl`` is (primary, mid) P&L."""
    primary, mid = pnl
    return pd.DataFrame({"event_id": ids, "entry_cost": 1.0, "exit_value": 3.0 if mid_marks else 2.0, "rung": 2, "lots": 1,
                         "pnl_per_share": 1.0, "pnl_dollars": mid if mid_marks else primary})


def test_inner_decisions_carry_the_scored_pnl_and_taken_rows(tmp_path, stubs, monkeypatch):
    def scored_rows(chunk_, events, sims, name, *, alpha=0.5):
        return pd.DataFrame({"event_id": events["event_id"].astype(str).to_numpy(), "event_date": events["event_date"].to_numpy(),
                             "structure": name, "status": "scored", "reason": "scored"})

    monkeypatch.setattr(driver.shortvol, "score_events", scored_rows)
    monkeypatch.setattr(driver.shortvol, "decide", lambda scored: scored.assign(cutoff=np.nan, status="taken", reason="taken"))
    monkeypatch.setattr(driver.shortvol, "score", lambda chunk_, events, mine, name, alpha, *, drop_unquoted=False, mid_marks=False:
                        _marks(mine["event_id"].astype(str).to_numpy(), (7.0, 9.0), mid_marks))
    _inner(tmp_path, monkeypatch)
    frame = _frame(tmp_path)
    models = frame["model_id"].nunique()
    assert models >= 1
    assert set(frame["event_id"].astype(str)) == {"E2"}  # E1 is burn-in: no rows
    short = frame[frame["book"] == "short_vol"]
    assert not short.empty and (frame["book"] == "short_vol").all()  # the inner stage runs without a gate: no straddle rows
    assert (short["status"] == "taken").all() and (short["pnl_dollars"] == 7.0).all()
    assert (short["pnl_dollars_mid"] == 9.0).all() and (short["rung"] == 2).all()
    assert len(frame) == models * len(driver.shortvol.NAMES)
