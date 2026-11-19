"""EXP-188 S9: driver orchestration with small fakes. Real inputs and readers; pool, ladder, scorers and tuning stubbed."""
import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError, fail
from experiments import size_pnl_driver as driver
from experiments import size_pnl_readers as readers
from experiments import size_pnl_tuning as tuning
from experiments.size_pnl_shortvol import Refusal

RUN = Path(__file__).resolve().parents[3] / "experiments" / "EXP-188_size_decision_pnl_rescore" / "run.py"
CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
DENIED = "HOLDOUT_ACCESS_DENIED"
SEAL = {"seal": "s"}
EVENTS = {"E1": "2023-06-10", "E2": "2024-08-07", "E3": "2025-01-15", "E4": "2025-02-12"}
PLAN = {"plan_id": "PLAN", "anchor": {"run_id": "REG"}, "horizons": [1, 5], "candidates": [],
        "selection": {"extra_refits": []}}
EXPECTED = {"plan_id": "PLAN", "registration_id": "REG", "seal_id": content_hash(SEAL), "panel_id": "P"}
OK = lambda: {"spec_hash": "SPEC", "spec_hash_checked": True}  # noqa: E731


def _base():
    """One row per event: production columns and label columns, for the fake fetch."""
    rows = [{"event_id": e, "pred_abs_move": 5.0, "pred_iv_crush_30": -12.0, "horizon_sessions": 1,
             "abs_move": 4.0, "crush_pct_iv30": -10.0} for e in EVENTS]
    return pd.DataFrame(rows)


BASE = _base()
ADMITTED = readers.Admitted({"events": pd.DataFrame({"event_id": list(EVENTS),
                                                     "event_date": pd.to_datetime(list(EVENTS.values()))}),
                             "excluded": {}})
DATES = pd.to_datetime(list(EVENTS.values()))
PLAN_FRAME = pd.DataFrame({"event_id": list(EVENTS), "ticker": "T", "event_date": DATES,
                           "session": "AMC", "entry_date": DATES - pd.Timedelta(days=1),
                           "exit_date": DATES + pd.Timedelta(days=1)})
FACTS = pd.DataFrame({"pre_iv30": 40.0, "mcap_usd": 2e10, "implied_move": 5.0},
                     index=pd.Index(list(EVENTS), name="event_id"))


def sources(**over):
    fetch_log = over.pop("fetch_log", None)

    def fetch(columns, ids):
        if fetch_log is not None:
            fetch_log.append(("labels" if "abs_move" in columns else "production", tuple(ids)))
        return BASE[BASE["event_id"].isin(list(ids))]

    base = dict(plan=PLAN, read=lambda slot: SEAL if slot == "driver-seal" else None,
                selection={"selections": []}, pair_candidates={}, blend_ids={}, expected=EXPECTED,
                panel_id="P", admit=lambda ids: {"admitted": sorted(ids), "dropped": []},
                admitted=ADMITTED, fetch=fetch, production_hash="H", production_table_hash="H",
                plan_frame=PLAN_FRAME, load_index=lambda keys: None,
                facts=lambda ids: FACTS.reindex(list(ids)))
    base.update(over)
    return driver.Sources(**base)


def chunk(*eids):
    """A minimal ladder chunk: the event ids and dte_exit are all the driver reads outside the scorers."""
    return {"event_id": np.array(eids, dtype=object), "dte_exit": np.full(len(eids), 9, dtype=np.int32)}


@pytest.fixture
def stubs(monkeypatch):
    """Pool, scorers and tuning are stubbed so the tests exercise the driver's own orchestration.

    Returns the call log that the stubs append to (tune calls and straddle decides)."""
    log = []

    def fake_tune(items, mid, **kwargs):
        log.append(("tune", mid, tuple(sorted(str(e) for _c, ev, _s in items for e in ev["event_id"]))))
        return {"stage": "tuning", "model_id": mid, "c": 1.0, "events": sum(len(ev) for _c, ev, _s in items)}

    def fake_scored(chunk_, events, sims, name, *, alpha=0.5, c=1.0):
        return pd.DataFrame({"event_id": events["event_id"].astype(str), "event_date": events["event_date"],
                             "structure": name, "status": "undetermined", "reason": "thin_history",
                             "exp_pnl_sim": np.nan, "entry_cost": np.nan, "rel_spread": np.nan,
                             "mcap_usd": events["mcap_usd"], "alpha": alpha, "c": c})

    def fake_short_score(chunk_, events, decisions, name, alpha, *, c=1.0):
        return pd.DataFrame(columns=["event_id", "structure", "entry_cost", "exit_value", "rung", "lots",
                                     "pnl_per_share", "pnl_dollars"])

    def fake_straddle(events, history, rule, chunk_):
        log.append(("decide", tuple(sorted(events["event_id"].astype(str)))))
        return pd.DataFrame({"event_id": events["event_id"].astype(str), "event_date": events["event_date"],
                             "edge": events["pred_abs_move"] - events["implied_move"], "cutoff": np.nan,
                             "rel_spread": np.nan, "status": "undetermined", "reason": "thin_history"})

    def fake_straddle_score(chunk_, events, dec, alpha, **kwargs):
        return pd.DataFrame(columns=["event_id", "entry_cost", "exit_value", "rung", "lots", "pnl_per_share",
                                     "pnl_dollars"])

    monkeypatch.setattr(driver, "build_pool", lambda table, labels: None)
    monkeypatch.setattr(tuning, "tune", fake_tune)
    monkeypatch.setattr(driver.shortvol, "simulate", lambda chunk_, events, pool, **kw: {})
    monkeypatch.setattr(driver.shortvol, "score_events", fake_scored)
    monkeypatch.setattr(driver.shortvol, "decide", lambda scored: scored.assign(cutoff=np.nan))
    monkeypatch.setattr(driver.shortvol, "score", fake_short_score)
    monkeypatch.setattr(driver.straddle, "decide", fake_straddle)
    monkeypatch.setattr(driver.straddle, "score", fake_straddle_score)
    return log


def use_chunks(monkeypatch, pairs, seen=None):
    """Replace the ladder with a generator over fixed (quarter, chunk) pairs; ``seen`` records the plan."""
    def fake(admitted, plan, load_index):
        if seen is not None:
            seen.append(plan)
        yield from pairs
    monkeypatch.setattr(driver.ladder, "ladder_chunks", fake)


# ---- refusals and ordering ----------------------------------------------------

def test_prereg_refusal_runs_before_any_source_call(tmp_path):
    calls = []

    def refuse():
        raise fail(CONFLICT, "no planned row", details={"reason": "planned_row_missing"})

    src = sources(read=lambda slot: calls.append(slot))
    with pytest.raises(OpsError):
        driver.run(src, "inner", tmp_path, prereg=refuse)
    assert calls == []


def test_bad_stage_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        driver.run(sources(), "2025", tmp_path, prereg=OK)


def test_outer_refuses_before_building_any_chunk(tmp_path, stubs):
    loads = []
    with pytest.raises(Refusal) as caught:
        driver.run(sources(load_index=lambda keys: loads.append(keys)), "outer", tmp_path, prereg=OK)
    assert caught.value.details["reason"] == "tuning_receipt_missing"
    assert loads == []


# ---- inner stage, replay and identity -----------------------------------------

def test_inner_writes_receipts_replays_read_only_and_refuses_a_changed_binding(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    src, out = sources(), tmp_path / "out"
    summary = driver.run(src, "inner", out, prereg=OK)
    mid = summary["models"][0]
    assert sorted(p.name for p in (out / "receipts").iterdir()) == sorted(
        ["inputs.json", f"tuning_{mid}.json", "inner.json"])
    before = {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in (out / "receipts").iterdir()}
    assert driver.run(src, "inner", out, prereg=OK) == summary  # exact replay succeeds
    assert {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in (out / "receipts").iterdir()} == before
    changed = dataclasses.replace(src, expected={**EXPECTED, "panel_id": "OTHER"})
    with pytest.raises(OpsError) as caught:
        driver.run(changed, "inner", out, prereg=OK)
    assert caught.value.code == CONFLICT
    assert {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in (out / "receipts").iterdir()} == before


def test_event_ids_must_be_admitted_and_restrict_the_plan_and_tables(tmp_path, stubs, monkeypatch):
    seen = []
    use_chunks(monkeypatch, [("2023Q2", chunk("E1"))], seen)
    with pytest.raises(OpsError) as caught:
        driver.run(sources(), "inner", tmp_path / "bad", event_ids={"E1", "ZZ"}, prereg=OK)
    assert caught.value.code == DENIED
    driver.run(sources(), "inner", tmp_path / "ok", event_ids={"E1"}, prereg=OK)
    assert set(seen[0]["event_id"]) == {"E1"}
    tunes = [e for e in stubs if e[0] == "tune"]
    assert tunes and all(entry[2] == ("E1",) for entry in tunes)  # the tables were restricted too


# ---- outer stage: R6 ordering and the warm-up -----------------------------------

def test_outer_reads_a_month_s_labels_only_after_its_rows_are_complete(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)  # writes inputs, tuning and inner receipts
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    summary = driver.run(sources(fetch_log=stubs), "outer", tmp_path, prereg=OK)
    # Two 2025 months, each with one row per (arm, short-vol name) and one per straddle rule.
    assert summary["rows"] == 2 * (2 * len(driver.shortvol.NAMES) + len(driver.straddle.RULES))
    assert (tmp_path / "decisions.parquet").exists()
    assert not (tmp_path / "decisions.parquet.tmp").exists()
    for event in ("E3", "E4"):
        decided = max(i for i, e in enumerate(stubs) if e[0] == "decide" and event in e[1])
        labelled = next(i for i, e in enumerate(stubs) if e[0] == "labels" and event in e[1])
        assert labelled > decided


def test_warm_up_month_adds_history_only(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    summary = driver.run(sources(fetch_log=stubs), "outer", tmp_path, prereg=OK)
    assert summary["rows"] == 0  # 2024 H2 is warm-up: history is built, but no decision row is written
    assert not any(e[0] == "labels" and e[1] == ("E2",) for e in stubs)  # and no month's labels are read


def test_changed_outer_receipt_leaves_decisions_untouched_and_no_tmp(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    (tmp_path / "decisions.parquet").write_bytes(b"EXISTING")
    tuning.write_receipt(tmp_path / "receipts" / "outer.json",
                         {"stage": "outer", "decisions_sha256": "OTHER", "rows": 99, "models": []})
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "outer", tmp_path, prereg=OK)
    assert caught.value.details["reason"] == "tuning_receipt_changed"
    assert (tmp_path / "decisions.parquet").read_bytes() == b"EXISTING"
    assert not (tmp_path / "decisions.parquet.tmp").exists()


# ---- run.py gate ---------------------------------------------------------------

def _run_main():
    spec = importlib.util.spec_from_file_location("exp188_run", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argv(tmp_path, *extra):
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps(EXPECTED), encoding="utf-8")
    return ["--stage", "inner", "--out-dir", str(tmp_path / "out"), "--expected", str(expected),
            "--sources", "no_such_module_zz:make", *extra]


def test_run_py_requires_no_ledger(tmp_path):
    with pytest.raises(SystemExit):
        _run_main().main(_argv(tmp_path))


def test_run_py_refuses_on_prereg_before_importing_sources(tmp_path, monkeypatch, capsys):
    run_py = _run_main()

    def refuse():
        raise fail(CONFLICT, "no planned row", details={"reason": "planned_row_missing"})

    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", refuse)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    assert run_py.main(_argv(tmp_path, "--no-ledger")) == 2
    assert json.loads(capsys.readouterr().out)["refused"] == CONFLICT


def test_run_py_refuses_not_bounded_before_importing_sources(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.delenv("BOUNDED_RUN_NESTED", raising=False)
    assert run_py.main(_argv(tmp_path, "--no-ledger")) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == "RESOURCE_UNAVAILABLE"
