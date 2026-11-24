"""EXP-188 S9a: inner-stage driver orchestration with small fakes. Real inputs and readers; pool and simulation
stubbed. Short vol is as-is only (Amendment 6): there is no divisor re-tune and no tuning receipt."""
import dataclasses
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.v2.data import errors as data_errors
from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError, fail
from experiments import size_pnl_driver as driver
from experiments import size_pnl_readers as readers
from experiments import size_pnl_receipts as tuning
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
EMPTY_MARKS = ["event_id", "entry_cost", "exit_value", "rung", "lots", "pnl_per_share", "pnl_dollars"]


def _avail(event_iso):
    """A label's native available_as_of_date by default: one calendar day after its event date (ISO text)."""
    return (pd.Timestamp(event_iso) + pd.Timedelta(days=1)).date().isoformat()


def _base():
    """One row per event: production columns and label columns (with availability), for the fake fetch."""
    rows = [{"event_id": e, "pred_abs_move": 5.0, "pred_iv_crush_30": -12.0, "horizon_sessions": 1,
             "abs_move": 4.0, "crush_pct_iv30": -10.0, "available_as_of_date": _avail(EVENTS[e])} for e in EVENTS]
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
    def fetch(columns, ids):
        return BASE[BASE["event_id"].isin(list(ids))]

    base = dict(plan=PLAN, read=lambda slot: SEAL if slot == "driver-seal" else None,
                selection={"selections": []}, pair_candidates={}, blend_ids={}, expected=EXPECTED,
                panel_id="P", admit=lambda ids: {"admitted": sorted(ids), "dropped": []},
                admitted=ADMITTED, fetch=fetch, production_hash="H", production_table_hash="H",
                plan_frame=PLAN_FRAME, load_index=lambda keys: None,
                facts=lambda ids: FACTS.reindex(list(ids)), snapshot_id="snap-1", crush_table_hash="CTH")
    base.update(over)
    return driver.Sources(**base)


SIMS = lambda chunk_, events, pool, **kw: {e: "sim" for e in events["event_id"]}  # noqa: E731


def chunk(*eids):
    """A minimal ladder chunk: the event ids and dte_exit are all the driver reads outside the simulator."""
    return {"event_id": np.array(eids, dtype=object), "dte_exit": np.full(len(eids), 9, dtype=np.int32)}


@pytest.fixture(autouse=True)
def bounded(monkeypatch):
    """Every run sits under the supervisor; the not-bounded tests unset the variable themselves."""
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")


@pytest.fixture(autouse=True)
def structure_gate_passes(monkeypatch):
    """The structure gate passes every event (no structure undetermined) unless a test replaces these."""
    monkeypatch.setattr(driver.shortvol, "score_events", lambda chunk_, events, sims, name, **kw: pd.DataFrame(
        {"event_id": events["event_id"].to_numpy(), "status": "scored", "reason": None}))


@pytest.fixture(autouse=True)
def equivalence_passes(monkeypatch):
    """The fast-path preflight passes on every chunk unless a test replaces it, so the stubbed pools still hold."""
    monkeypatch.setattr(driver, "_equivalence", lambda *args, **kwargs: {"checked": 1, "skipped": 0, "max_abs_diff": 0.0})


@pytest.fixture
def stubs(monkeypatch):
    """Pool and simulation are stubbed so the tests exercise the driver's own orchestration.

    Returns an empty call log (the divisor re-tune and its tune stub were removed by Amendment 6).
    """
    log = []

    def fake_scored(chunk_, events, sims, name, *, alpha=0.5):
        return pd.DataFrame({"event_id": events["event_id"].astype(str).to_numpy(), "event_date": events["event_date"].to_numpy(),
                             "structure": name, "status": "scored", "reason": "scored"})

    def fake_straddle(events, history, rule, chunk_):
        return pd.DataFrame({"event_id": events["event_id"].astype(str), "event_date": events["event_date"],
                             "edge": events["pred_abs_move"] - events["implied_move"], "cutoff": np.nan, "rel_spread": np.nan,
                             "status": "undetermined", "reason": "thin_history"})

    monkeypatch.setattr(driver, "build_pool", lambda table, labels: None)
    monkeypatch.setattr(driver.shortvol, "simulate", SIMS)
    monkeypatch.setattr(driver.shortvol, "score_events", fake_scored)
    monkeypatch.setattr(driver.shortvol, "decide", lambda scored: scored.assign(cutoff=np.nan))
    monkeypatch.setattr(driver.shortvol, "score", lambda chunk_, events, decisions, name, alpha, *, drop_unquoted=False, mid_marks=False: pd.DataFrame(columns=EMPTY_MARKS))
    monkeypatch.setattr(driver.straddle, "decide", fake_straddle)
    monkeypatch.setattr(driver.straddle, "score", lambda chunk_, events, dec, alpha, **kw: pd.DataFrame(columns=EMPTY_MARKS))
    return log


def use_chunks(monkeypatch, pairs):
    """Replace the ladder with a generator over fixed (quarter, chunk) pairs."""
    def fake(admitted, plan, load_index):
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


def test_prereg_refusal_wins_over_invalid_sources(tmp_path):
    def refuse():
        raise fail(CONFLICT, "no planned row", details={"reason": "planned_row_missing"})

    with pytest.raises(OpsError) as caught:
        driver.run(sources(plan={}), "inner", tmp_path, prereg=refuse)
    assert caught.value.code == CONFLICT
    with pytest.raises(OpsError) as caught:
        driver.run(None, "inner", tmp_path, prereg=refuse)
    assert caught.value.code == CONFLICT


@pytest.mark.parametrize("over", [{"plan": {}}, {"selection": None}, {"expected": {}}])
def test_malformed_sources_refuse_before_any_read(tmp_path, over):
    calls = []
    src = sources(read=lambda slot: calls.append(slot), **over)
    with pytest.raises(OpsError) as caught:
        driver.run(src, "inner", tmp_path, prereg=OK)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.details["reason"] == "sources_invalid"
    assert calls == []


@pytest.mark.parametrize("stage", ["typo", "2025"])  # "outer" is a valid stage since S9b
def test_a_typo_stage_is_refused(tmp_path, stage):
    with pytest.raises(ValueError, match="stage must be one of"):
        driver.run(sources(), stage, tmp_path, prereg=OK)


def test__prepare_rejects_a_typo_stage_before_any_source_call():
    calls = []
    with pytest.raises(ValueError, match="stage must be one of"):
        driver._prepare(sources(read=lambda slot: calls.append(slot)), "typo")
    assert calls == []


# ---- inner stage, replay and identity -----------------------------------------

def test_inner_writes_receipts_replays_read_only_and_refuses_a_changed_binding(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    src, out = sources(), tmp_path / "out"
    summary = driver.run(src, "inner", out, prereg=OK)
    assert sorted(p.name for p in (out / "receipts").iterdir()) == sorted(
        ["inputs_inner.json", "inner.json", "inner_decisions.json"])
    before = {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in (out / "receipts").iterdir()}
    assert driver.run(src, "inner", out, prereg=OK) == summary  # exact replay succeeds
    assert {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in (out / "receipts").iterdir()} == before
    changed = dataclasses.replace(src, expected={**EXPECTED, "panel_id": "OTHER"})
    with pytest.raises(OpsError) as caught:
        driver.run(changed, "inner", out, prereg=OK)
    assert caught.value.code == CONFLICT
    assert {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in (out / "receipts").iterdir()} == before


def test_inner_receipts_bind_scope_none_and_the_snapshot(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1"))])
    summary = driver.run(sources(), "inner", tmp_path, prereg=OK)
    mid = summary["models"][0]
    inputs = tuning.load_receipt(tmp_path / "receipts" / "inputs_inner.json")
    inner = tuning.load_receipt(tmp_path / "receipts" / "inner.json")
    assert inputs["scope"] is None and inner["scope"] is None
    assert inputs["snapshot"] == "snap-1"
    assert inputs["crush_table_hash"] == "CTH"
    production = next(m["model_id"] for m in inputs["roster"]["models"] if m["kind"] == "production")
    assert inputs["models"][production] == {"status": "ready"}
    assert inner["stage"] == "inner"
    assert inner["models"] == [mid]  # the sorted ready model ids; no tuning hash
    assert inner["inputs"] == inputs["content_hash"]
    assert inner["ladder_skipped"] == {}
    assert inner["events_by_model"] == {mid: 1}
    assert inputs["stage"] == "inputs"
    for digest in (inputs["plan"], inputs["facts"]):
        assert digest.startswith("sha256:")
        remainder = digest[len("sha256:"):]
        assert len(remainder) == 64 and set(remainder) <= set("0123456789abcdef")


def test_model_events_aligns_every_column_by_event_id():
    table = pd.DataFrame({"event_date": pd.to_datetime(["2023-06-10", "2024-08-07", "2025-01-15"]),
                          "pred_move": [1.5, 2.5, 3.5], "pred_crush": [-10.0, -20.0, -30.0]},
                         index=pd.Index(["E1", "E2", "E3"]))
    # Facts deliberately in a different row order than the table.
    facts = pd.DataFrame({"pre_iv30": [33.0, 31.0, 32.0], "mcap_usd": [3e9, 1e9, 2e9],
                          "implied_move": [6.0, 4.0, 5.0]},
                         index=pd.Index(["E3", "E1", "E2"], name="event_id"))
    # The chunk lists events in yet another order, with a distinct dte_exit for each.
    ladder = {"event_id": np.array(["E2", "E1", "E3"], dtype=object),
              "dte_exit": np.array([7, 8, 9], dtype=np.int32)}
    out = driver.model_events(table, facts, ladder, {}).set_index("event_id")
    expected = {
        "E1": dict(event_date=pd.Timestamp("2023-06-10"), pred_move=1.5, pred_abs_move=1.5, pred_crush=-10.0,
                   pre_iv30=31.0, mcap_usd=1e9, implied_move=4.0, dte_exit=8),
        "E2": dict(event_date=pd.Timestamp("2024-08-07"), pred_move=2.5, pred_abs_move=2.5, pred_crush=-20.0,
                   pre_iv30=32.0, mcap_usd=2e9, implied_move=5.0, dte_exit=7),
        "E3": dict(event_date=pd.Timestamp("2025-01-15"), pred_move=3.5, pred_abs_move=3.5, pred_crush=-30.0,
                   pre_iv30=33.0, mcap_usd=3e9, implied_move=6.0, dte_exit=9),
    }
    assert sorted(out.index) == ["E1", "E2", "E3"]
    for eid, fields in expected.items():
        for column, value in fields.items():
            assert out.loc[eid, column] == value, (eid, column)


def test_model_events_derives_dte_when_exit_quotes_are_missing():
    table = pd.DataFrame({"event_date": pd.to_datetime(["2023-06-10"]), "pred_move": [1.5], "pred_crush": [-10.0]},
                         index=pd.Index(["E1"]))
    facts = pd.DataFrame({"pre_iv30": [33.0], "mcap_usd": [3e9], "implied_move": [6.0]},
                         index=pd.Index(["E1"], name="event_id"))
    gap = {"event_id": np.array(["E1"], dtype=object), "dte_exit": np.array([-1], dtype=np.int32),
           "expiry": np.array(["2023-07-21"], dtype="datetime64[ns]")}
    out = driver.model_events(table, facts, gap, {"E1": "2023-07-14"})
    assert out["dte_exit"].tolist() == [7]  # 2023-07-21 minus 2023-07-14


def test_an_event_unsimulable_under_one_model_leaves_for_every_model(tmp_path, stubs, monkeypatch):
    real, seen = readers.model_tables, {}

    def twinned(bound, roster, admitted, production, **kw):
        tables = real(bound, roster, admitted, production, **kw)
        mid = next(iter(tables))
        seen["twin"] = tables[mid]["table"].copy()
        seen["twin"].attrs["twin"] = True
        return {mid: tables[mid], "twin": {**tables[mid], "table": seen["twin"]}}

    def sims_of(chunk_, events, table, labels):
        return {e: (None if table.attrs.get("twin") and e == "E2" else "sim") for e in events["event_id"]}

    monkeypatch.setattr(readers, "model_tables", twinned)
    monkeypatch.setattr(driver, "_sims", sims_of)
    use_chunks(monkeypatch, [("2023Q2", chunk("E1", "E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    inner = tuning.load_receipt(tmp_path / "receipts" / "inner.json")
    assert inner["common_dropped"] == {"2023Q2": 1}  # E2 leaves for every model
    assert inner["events"] == 1  # the taken population: the common-dropped E2 is not counted
    assert set(inner["events_by_model"].values()) == {inner["events"]}


def test_an_event_missing_from_one_table_leaves_for_every_model(tmp_path, stubs, monkeypatch):
    real, seen = readers.model_tables, {}

    def twinned(bound, roster, admitted, production, **kw):
        tables = real(bound, roster, admitted, production, **kw)
        mid = next(iter(tables))
        seen["twin"] = tables[mid]["table"].drop(index="E3", errors="ignore")  # model B lacks E3
        return {mid: tables[mid], "twin": {**tables[mid], "table": seen["twin"]}}

    monkeypatch.setattr(readers, "model_tables", twinned)
    use_chunks(monkeypatch, [("2023Q2", chunk("E1", "E2", "E3"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    assert tuning.load_receipt(tmp_path / "receipts" / "inner.json")["common_dropped"] == {"2023Q2": 1}  # E3 is in neither model's items


def test_an_event_declined_at_unity_stays_in_every_models_items_as_a_non_trade(tmp_path, stubs, monkeypatch):
    """The twin's E2 is non-"scored" (width_declined). A non-trade is P&L 0 for that model and structure: E2 stays in both
    models' items, and the receipt counts the twin's declines (one per structure) under its own model."""
    real, seen = readers.model_tables, {}

    def twinned(bound, roster, admitted, production, **kw):
        tables = real(bound, roster, admitted, production, **kw)
        mid = next(iter(tables))
        seen["twin"] = tables[mid]["table"].copy()
        seen["twin"].attrs["twin"] = True
        return {mid: tables[mid], "twin": {**tables[mid], "table": seen["twin"]}}

    def sims_of(chunk_, events, table, labels):  # the twin's E2 is marked; a marked sim is still a simulation
        return {e: ("undetermined" if table.attrs.get("twin") and e == "E2" else "sim") for e in events["event_id"]}

    def scored_of(chunk_, events, sims, name, *, alpha=0.5, **kw):  # the twin's E2 is declined
        declined = [sims.get(e) == "undetermined" for e in events["event_id"]]
        return pd.DataFrame({"event_id": events["event_id"].to_numpy(),
                             "status": ["undetermined" if d else "scored" for d in declined],
                             "reason": ["width_declined" if d else None for d in declined]})

    monkeypatch.setattr(readers, "model_tables", twinned)
    monkeypatch.setattr(driver, "_sims", sims_of)
    monkeypatch.setattr(driver.shortvol, "score_events", scored_of)
    use_chunks(monkeypatch, [("2023Q2", chunk("E1", "E2"))])
    summary = driver.run(sources(), "inner", tmp_path, prereg=OK)
    assert sorted(summary["models"]) == sorted([next(mid for mid in summary["models"] if mid != "twin"), "twin"])
    inner = tuning.load_receipt(tmp_path / "receipts" / "inner.json")
    primary = next(mid for mid in summary["models"] if mid != "twin")
    assert inner["non_trades"] == {primary: {}, "twin": {"width_declined": len(driver.shortvol.NAMES)}}  # 7 == len(NAMES)
    assert inner["missing_facts"] == {primary: {}, "twin": {}}  # a width decline is a non-trade, never a missing fact
    assert inner["common_dropped"] == {"2023Q2": 0}  # the quarter-level count is unchanged; no model-level drop here


def test_an_entry_quote_unusable_event_is_a_missing_fact_and_stays_in_the_items(tmp_path, stubs, monkeypatch):
    """The twin's E2 is non-"scored" with reason entry_quote_unusable. That is a missing fact, not a non-trade:
    it is counted under missing_facts and not under non_trades, and the driver keeps E2 in every model's items."""
    real, seen = readers.model_tables, {}

    def twinned(bound, roster, admitted, production, **kw):
        tables = real(bound, roster, admitted, production, **kw)
        mid = next(iter(tables))
        seen["twin"] = tables[mid]["table"].copy()
        seen["twin"].attrs["twin"] = True
        return {mid: tables[mid], "twin": {**tables[mid], "table": seen["twin"]}}

    def sims_of(chunk_, events, table, labels):  # the twin's E2 is marked; a marked sim is still a simulation
        return {e: ("undetermined" if table.attrs.get("twin") and e == "E2" else "sim") for e in events["event_id"]}

    def scored_of(chunk_, events, sims, name, *, alpha=0.5, **kw):  # the twin's E2 has no usable entry quote
        missing = [sims.get(e) == "undetermined" for e in events["event_id"]]
        return pd.DataFrame({"event_id": events["event_id"].to_numpy(),
                             "status": ["undetermined" if m else "scored" for m in missing],
                             "reason": ["entry_quote_unusable" if m else None for m in missing]})

    monkeypatch.setattr(readers, "model_tables", twinned)
    monkeypatch.setattr(driver, "_sims", sims_of)
    monkeypatch.setattr(driver.shortvol, "score_events", scored_of)
    use_chunks(monkeypatch, [("2023Q2", chunk("E1", "E2"))])
    summary = driver.run(sources(), "inner", tmp_path, prereg=OK)
    assert sorted(summary["models"]) == sorted([next(mid for mid in summary["models"] if mid != "twin"), "twin"])
    inner = tuning.load_receipt(tmp_path / "receipts" / "inner.json")
    primary = next(mid for mid in summary["models"] if mid != "twin")
    assert inner["missing_facts"] == {primary: {}, "twin": {"entry_quote_unusable": len(driver.shortvol.NAMES)}}  # 7 == len(NAMES)
    assert inner["non_trades"] == {primary: {}, "twin": {}}  # the missing fact is not counted as a non-trade


def test_inner_refuses_a_changed_plan_window_on_replay(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    changed = PLAN_FRAME.copy()
    changed.loc[changed["event_id"] == "E2", "entry_date"] = pd.Timestamp("2024-08-05")  # same ids, one entry_date
    with pytest.raises(Refusal) as caught:
        driver.run(sources(plan_frame=changed), "inner", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inputs_receipt_changed"


def _cutoff_pools(tmp_path, monkeypatch, e1_available):
    """Inner run over 2023 Q2 with E1 and E2 (E2 in 2024-08). E1's exit is 2024-07-20, so exit_date + 1 is before the 2024-08
    month start; only E1's native ``available_as_of_date`` differs between calls. Returns (table ids, label ids) per pool."""
    plan = PLAN_FRAME.copy()
    plan.loc[plan["event_id"] == "E1", "exit_date"] = pd.Timestamp("2024-07-20")

    def fetch(columns, ids):
        rows = BASE[BASE["event_id"].isin(list(ids))].copy()
        rows.loc[rows["event_id"] == "E1", "available_as_of_date"] = e1_available
        return rows

    pools = []
    monkeypatch.setattr(driver, "build_pool",
                        lambda table, labels: pools.append((set(table.index.astype(str)), set(labels.index.astype(str)))))
    use_chunks(monkeypatch, [("2023Q2", chunk("E1", "E2"))])
    driver.run(sources(plan_frame=plan, fetch=fetch), "inner", tmp_path, prereg=OK)
    return pools


def test_label_printed_after_the_month_start_is_not_admitted_though_exit_plus_one_is_before(tmp_path, stubs, monkeypatch):
    pools = _cutoff_pools(tmp_path, monkeypatch, "2024-08-05")  # printed after 2024-08-01, the 2024-08 month start
    labelled = [ids for tabled, ids in pools if "E1" in tabled]  # the 2024-08 month: E1 is in its table
    assert labelled and all(ids == set() for ids in labelled)


def test_label_printed_before_the_month_start_is_admitted(tmp_path, stubs, monkeypatch):
    pools = _cutoff_pools(tmp_path, monkeypatch, "2024-07-25")  # the same label, printed before 2024-08-01
    labelled = [ids for tabled, ids in pools if "E1" in tabled]
    assert labelled and all(ids == {"E1"} for ids in labelled)


def test_label_available_on_the_month_start_is_not_admitted(tmp_path, stubs, monkeypatch):
    """The cutoff is strictly before the month start: a label available on 2024-08-01 itself is not in the pool."""
    pools = _cutoff_pools(tmp_path, monkeypatch, "2024-08-01")
    labelled = [ids for tabled, ids in pools if "E1" in tabled]
    assert labelled and all(ids == set() for ids in labelled)


def test_label_available_one_day_before_the_month_start_is_admitted(tmp_path, stubs, monkeypatch):
    pools = _cutoff_pools(tmp_path, monkeypatch, "2024-07-31")
    labelled = [ids for tabled, ids in pools if "E1" in tabled]
    assert labelled and all(ids == {"E1"} for ids in labelled)


def test_residual_pool_excludes_history_missing_from_any_ready_model(tmp_path, stubs, monkeypatch):
    """H1 is an old production event in model A's table only; no residual pool of either model may hold it."""
    real, pools = readers.model_tables, []

    def twinned(bound, roster, admitted, production, **kw):
        tables = real(bound, roster, admitted, production, **kw)
        mid = next(iter(tables))
        base = tables[mid]["table"]
        old = base.iloc[[0]].copy()
        old.index = ["H1"]
        old["event_date"] = pd.Timestamp("2022-06-01")
        return {mid: {**tables[mid], "table": pd.concat([base, old])}, "twin": {**tables[mid], "table": base.copy()}}

    monkeypatch.setattr(readers, "model_tables", twinned)
    monkeypatch.setattr(driver, "build_pool", lambda table, labels: pools.append(set(table.index.astype(str))))
    use_chunks(monkeypatch, [("2023Q2", chunk("E1", "E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    assert len(pools) >= 2 and not any("H1" in pool for pool in pools)


def test_plan_row_with_a_non_admitted_id_refuses_before_facts(tmp_path, stubs, monkeypatch):
    calls = []

    def counting(ids):
        calls.append(list(ids))
        return FACTS.reindex(list(ids))

    plan = PLAN_FRAME.copy()
    plan.loc[plan["event_id"] == "E1", "event_id"] = "ZZ"  # same window and date, id not admitted
    use_chunks(monkeypatch, [("2023Q2", chunk("E1"))])
    with pytest.raises(OpsError) as caught:
        driver.run(sources(plan_frame=plan, facts=counting), "inner", tmp_path, prereg=OK)
    assert caught.value.code == DENIED
    assert caught.value.problem.details["reason"] == "key_not_admitted"
    assert calls == []


def test_plan_row_with_a_shifted_event_date_refuses_before_facts(tmp_path, stubs, monkeypatch):
    calls = []

    def counting(ids):
        calls.append(list(ids))
        return FACTS.reindex(list(ids))

    plan = PLAN_FRAME.copy()
    plan.loc[plan["event_id"] == "E2", "event_date"] = pd.Timestamp("2024-08-08")  # same id, shifted date
    use_chunks(monkeypatch, [("2023Q2", chunk("E1"))])
    with pytest.raises(OpsError) as caught:
        driver.run(sources(plan_frame=plan, facts=counting), "inner", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.problem.details["reason"] == "event_date_mismatch"
    assert calls == []


def test_duplicate_plan_id_refuses_before_facts_labels_or_receipts(tmp_path, stubs, monkeypatch):
    calls = []

    def counting(ids):
        calls.append(list(ids))
        return FACTS.reindex(list(ids))

    plan = pd.concat([PLAN_FRAME, PLAN_FRAME[PLAN_FRAME["event_id"] == "E1"]], ignore_index=True)  # E1 twice
    use_chunks(monkeypatch, [("2023Q2", chunk("E1"))])
    with pytest.raises(OpsError) as caught:
        driver.run(sources(plan_frame=plan, facts=counting), "inner", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.problem.details["reason"] == "duplicate_event_id"
    assert calls == []
    assert not (tmp_path / "receipts").exists()


def test_duplicate_id_outside_the_inner_window_refuses_before_any_receipt(tmp_path, stubs, monkeypatch):
    plan = pd.concat([PLAN_FRAME, PLAN_FRAME[PLAN_FRAME["event_id"] == "E3"]], ignore_index=True)  # E3 is a 2025 row
    use_chunks(monkeypatch, [("2023Q2", chunk("E1"))])
    with pytest.raises(OpsError) as caught:
        driver.run(sources(plan_frame=plan), "inner", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.problem.details["reason"] == "duplicate_event_id"
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize("column", ["event_date", "entry_date", "exit_date"])
def test_null_plan_date_refuses_before_any_receipt(tmp_path, stubs, monkeypatch, column):
    plan = PLAN_FRAME.copy()
    plan.loc[plan["event_id"] == "E1", column] = pd.NaT
    use_chunks(monkeypatch, [("2023Q2", chunk("E1"))])
    with pytest.raises(OpsError) as caught:
        driver.run(sources(plan_frame=plan), "inner", tmp_path, prereg=OK)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.details["reason"] == "plan_dates_null"
    assert not (tmp_path / "receipts").exists()


def test_plan_without_exit_date_refuses_through_check_plan(tmp_path, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1"))])
    with pytest.raises(OpsError) as caught:
        driver.run(sources(plan_frame=PLAN_FRAME.drop(columns="exit_date")), "inner", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.problem.details["reason"] == "plan_malformed"
    assert not (tmp_path / "receipts").exists()


def test_facts_missing_a_requested_id_refuses_before_any_receipt(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    with pytest.raises(OpsError) as caught:
        driver.run(sources(facts=lambda ids: FACTS.reindex(["E1"])), "inner", tmp_path, prereg=OK)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.details["reason"] == "facts_incomplete"
    assert not (tmp_path / "receipts").exists()


def test_duplicate_facts_ids_refuse_before_any_receipt(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    # Same id set as the plan, but E1 appears twice.
    facts = lambda ids: pd.concat([FACTS.reindex(list(ids)), FACTS.reindex(["E1"])])  # noqa: E731
    with pytest.raises(OpsError) as caught:
        driver.run(sources(facts=facts), "inner", tmp_path, prereg=OK)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.details["reason"] == "facts_incomplete"
    assert not (tmp_path / "receipts").exists()


def test_duplicate_facts_columns_refuse_before_any_receipt(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    facts = lambda ids: FACTS.reindex(list(ids)).rename(columns={"mcap_usd": "pre_iv30"})  # two pre_iv30 columns  # noqa: E731
    with pytest.raises(OpsError) as caught:
        driver.run(sources(facts=facts), "inner", tmp_path, prereg=OK)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.details["reason"] == "facts_incomplete"
    assert not (tmp_path / "receipts").exists()


def test_infinite_facts_refuse_before_any_receipt(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    facts = lambda ids: FACTS.reindex(list(ids)).assign(mcap_usd=float("inf"))  # noqa: E731
    with pytest.raises(OpsError) as caught:
        driver.run(sources(facts=facts), "inner", tmp_path, prereg=OK)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.details["reason"] == "facts_incomplete"
    assert not (tmp_path / "receipts").exists()


def test_production_fetch_never_reads_an_outer_year_event(tmp_path, stubs, monkeypatch):
    calls = []

    def recording(columns, ids):
        ids = list(ids)
        calls.append([str(e) for e in ids])
        return BASE[BASE["event_id"].isin(ids)]

    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    driver.run(sources(fetch=recording), "inner", tmp_path, prereg=OK)
    assert calls
    assert not any(e in ids for ids in calls for e in ("E3", "E4"))


def test_facts_are_fetched_once_per_run(tmp_path, stubs, monkeypatch):
    calls = []

    def counting(ids):
        calls.append(list(ids))
        return FACTS.reindex(list(ids))

    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    driver.run(sources(facts=counting), "inner", tmp_path, prereg=OK)
    assert calls == [["E1", "E2"]]


def test_changed_snapshot_refuses_an_inner_replay(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    with pytest.raises(Refusal) as caught:
        driver.run(sources(snapshot_id="snap-2"), "inner", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inputs_receipt_changed"


def test_changed_fact_value_refuses_an_inner_replay(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    moved = FACTS.copy()
    moved.loc["E2", "pre_iv30"] = 41.0
    with pytest.raises(Refusal) as caught:
        driver.run(sources(facts=lambda ids: moved.reindex(list(ids))), "inner", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inputs_receipt_changed"


def test_changed_label_value_refuses_an_inner_replay(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)

    def relabelled(columns, ids):
        rows = BASE[BASE["event_id"].isin(list(ids))].copy()
        rows.loc[rows["event_id"] == "E2", "abs_move"] = 9.0  # one label value moves
        return rows

    with pytest.raises(Refusal) as caught:
        driver.run(sources(fetch=relabelled), "inner", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inputs_receipt_changed"


def test_label_rows_carry_availability_so_a_changed_date_changes_the_row():
    frame = pd.DataFrame({"abs_move": [4.0], "crush_pct_iv30": [-10.0], "available": ["2024-08-08"]},
                         index=pd.Index(["E2"], name="event_id"))
    moved = frame.assign(available=["2024-09-01"])
    assert driver._label_rows([frame]) == [["E2", 4.0, -10.0, "2024-08-08"]]
    assert driver._label_rows([frame]) != driver._label_rows([moved])
    assert content_hash(driver._label_rows([frame])) != content_hash(driver._label_rows([moved]))


def _inner_labels_receipt(out, monkeypatch, e2_available):
    """Inner run over E1 (2023 Q2) and E2 (2024 Q3); only E2's native ``available_as_of_date`` differs between calls."""
    base = BASE.copy()
    base.loc[base["event_id"] == "E2", "available_as_of_date"] = e2_available

    def fetch(columns, ids):
        return base[base["event_id"].isin(list(ids))].copy()

    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    driver.run(sources(fetch=fetch), "inner", out, prereg=OK)
    return tuning.load_receipt(out / "receipts" / "inputs_inner.json")["labels"]


def test_changed_label_availability_changes_the_inputs_labels_receipt(tmp_path, stubs, monkeypatch):
    first = _inner_labels_receipt(tmp_path / "a", monkeypatch, "2024-08-08")
    second = _inner_labels_receipt(tmp_path / "b", monkeypatch, "2024-09-01")
    assert first != second


def test_changed_ladder_quote_refuses_an_inner_replay(tmp_path, stubs, monkeypatch):
    """The ladder digest binds the consumed quotes: one quote value changed under the same snapshot and plan is refused."""
    def quoted(bid):
        return {**chunk("E1"), "bid": np.array([1.5, bid])}

    out = tmp_path / "out"
    use_chunks(monkeypatch, [("2023Q2", quoted(2.0))])
    driver.run(sources(), "inner", out, prereg=OK)
    before = {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in (out / "receipts").iterdir()}
    assert driver.run(sources(), "inner", out, prereg=OK)  # identical rerun replays
    assert {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in (out / "receipts").iterdir()} == before
    use_chunks(monkeypatch, [("2023Q2", quoted(2.5))])
    driver.run(sources(), "inner", tmp_path / "other", prereg=OK)
    assert (tuning.load_receipt(tmp_path / "other" / "receipts" / "inner.json")["ladder"]
            != tuning.load_receipt(out / "receipts" / "inner.json")["ladder"])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "inner", out, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inner_receipt_changed"
    assert {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in (out / "receipts").iterdir()} == before


def test_zero_ready_models_refuses_after_the_inputs_receipt(tmp_path, stubs, monkeypatch):
    monkeypatch.setattr(readers, "model_tables", lambda bound, roster, admitted, production, **kw: {
        m["model_id"]: {"status": "excluded", "reason": "forced"} for m in roster["models"]})
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "inner", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "no_ready_models"
    assert [p.name for p in (tmp_path / "receipts").iterdir()] == ["inputs_inner.json"]


def test_ladder_skipped_counts_reach_the_inner_receipt_and_summary(tmp_path, stubs, monkeypatch):
    first, second = chunk("E1"), {**chunk("E2"), "skipped": {"no_chain": 1, "thin": 3}}
    first["skipped"] = {"no_chain": 2}
    use_chunks(monkeypatch, [("2023Q2", first), ("2024Q3", second)])
    summary = driver.run(sources(), "inner", tmp_path, prereg=OK)
    expected = {"no_chain": 3, "thin": 3}
    assert tuning.load_receipt(tmp_path / "receipts" / "inner.json")["ladder_skipped"] == expected
    assert summary["ladder_skipped"] == expected


# ---- run.py gate ---------------------------------------------------------------

def test_equivalence_runs_before_scoring_and_is_receipted(tmp_path, stubs, monkeypatch):
    def logged(*args, **kwargs):
        stubs.append(("equivalence",))
        return {"checked": 1, "skipped": 0, "max_abs_diff": 0.0}

    monkeypatch.setattr(driver, "_equivalence", logged)
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    out = tmp_path / "out"
    summary = driver.run(sources(), "inner", out, prereg=OK)
    assert stubs and {entry[0] for entry in stubs} == {"equivalence"}
    receipt = json.loads((out / "receipts" / "inner.json").read_text())
    assert receipt["equivalence"] == {summary["models"][0]: {"checked": 1, "skipped": 0, "max_abs_diff": 0.0}}


def test_a_fast_path_mismatch_aborts_before_any_inner_receipt(tmp_path, stubs, monkeypatch):
    def mismatch(*args, **kwargs):
        raise fail(CONFLICT, "the fast path disagrees with expected_pnl", details={"reason": "fast_path_mismatch"})

    monkeypatch.setattr(driver, "_equivalence", mismatch)
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    with pytest.raises(OpsError) as caught:
        driver.run(sources(), "inner", tmp_path / "out", prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.problem.details["reason"] == "fast_path_mismatch"
    assert not (tmp_path / "out" / "receipts" / "inner.json").exists()


def test_no_simulable_sample_refuses_before_any_inner_receipt(tmp_path, stubs, monkeypatch):
    monkeypatch.setattr(driver, "_equivalence", lambda *args, **kwargs: None)
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    with pytest.raises(OpsError) as caught:
        driver.run(sources(), "inner", tmp_path / "out", prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.problem.details["reason"] == "equivalence_empty"
    assert caught.value.problem.details["models"]
    assert not (tmp_path / "out" / "receipts" / "inner.json").exists()


def _run_main():
    spec = importlib.util.spec_from_file_location("exp188_run", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NO_SOURCES = "no_such_module_zz:make"
FAKE = "exp188_fake_sources:make"


def _argv(tmp_path, *extra, sources=NO_SOURCES):
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps(EXPECTED), encoding="utf-8")
    return ["--stage", "inner", "--out-dir", str(tmp_path / "out"), "--expected", str(expected),
            "--sources", sources, *extra]


def _install_sources(monkeypatch, src):
    """Make ``src`` the factory behind the ``--sources`` spec FAKE."""
    module = types.ModuleType("exp188_fake_sources")
    module.make = lambda args: src
    monkeypatch.setitem(sys.modules, "exp188_fake_sources", module)


def test_run_refuses_sources_with_a_non_callable_source(tmp_path):
    with pytest.raises(OpsError) as caught:
        driver.run(sources(facts=None), "inner", tmp_path, prereg=OK)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.details["reason"] == "sources_invalid"
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize("snapshot_id", ["", "   "])
def test_run_refuses_a_blank_snapshot_id(tmp_path, snapshot_id):
    with pytest.raises(OpsError) as caught:
        driver.run(sources(snapshot_id=snapshot_id), "inner", tmp_path, prereg=OK)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.details["reason"] == "sources_invalid"
    assert not (tmp_path / "receipts").exists()


def test_run_refuses_facts_that_are_not_a_frame(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    with pytest.raises(OpsError) as caught:
        driver.run(sources(facts=lambda ids: None), "inner", tmp_path, prereg=OK)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.details["reason"] == "facts_incomplete"
    assert not (tmp_path / "receipts").exists()


def test_run_py_requires_no_ledger(tmp_path):
    with pytest.raises(SystemExit):
        _run_main().main(_argv(tmp_path))


def test_run_py_refuses_a_typo_stage(tmp_path):
    run_py = _run_main()
    with pytest.raises(SystemExit):  # a typo stage is refused by argparse
        run_py.main(["--stage", "outr", "--out-dir", str(tmp_path), "--expected", "e.json",
                     "--sources", NO_SOURCES, "--no-ledger"])
    assert run_py._parser().parse_args(["--stage", "outer", "--out-dir", "o", "--expected", "e",
                                        "--sources", NO_SOURCES, "--no-ledger"]).stage == "outer"


def test_run_py_refuses_on_prereg_before_importing_sources(tmp_path, monkeypatch, capsys):
    run_py = _run_main()

    def refuse():
        raise fail(CONFLICT, "no planned row", details={"reason": "planned_row_missing"})

    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", refuse)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    assert run_py.main(_argv(tmp_path, "--no-ledger")) == 2
    assert json.loads(capsys.readouterr().out)["refused"] == CONFLICT


def test_run_py_refuses_not_bounded_before_the_sources_factory(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.delenv("BOUNDED_RUN_NESTED", raising=False)
    calls = []
    module = types.ModuleType("exp188_fake_sources")

    def make(args):  # spy: records any call to the factory
        calls.append(args)
        return sources()

    module.make = make
    monkeypatch.setitem(sys.modules, "exp188_fake_sources", module)
    assert run_py.main(_argv(tmp_path, "--no-ledger", sources=FAKE)) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "not_bounded"
    assert calls == []


def test_run_py_refuses_a_sources_spec_without_a_colon(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    assert run_py.main(_argv(tmp_path, "--no-ledger", sources="nocolon")) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "sources_invalid"


def test_run_py_refuses_not_bounded_through_the_driver(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.delenv("BOUNDED_RUN_NESTED", raising=False)
    _install_sources(monkeypatch, sources())
    assert run_py.main(_argv(tmp_path, "--no-ledger", sources=FAKE)) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == "RESOURCE_UNAVAILABLE"
    assert refused["reason"] == "not_bounded"


def test_driver_run_refuses_not_bounded_after_prereg_before_any_read(tmp_path, monkeypatch):
    monkeypatch.delenv("BOUNDED_RUN_NESTED", raising=False)
    calls = []
    src = sources(read=lambda slot: calls.append(slot))

    def prereg():
        calls.append("prereg")
        return OK()

    with pytest.raises(OpsError) as caught:
        driver.run(src, "inner", tmp_path, prereg=prereg)
    assert caught.value.code == "RESOURCE_UNAVAILABLE"
    assert caught.value.problem.details["reason"] == "not_bounded"
    assert calls == ["prereg"]  # prereg first; no source read


def test_run_py_propagates_a_type_error_raised_inside_the_factory(tmp_path, monkeypatch):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    module = types.ModuleType("exp188_fake_sources")

    def make(args):
        raise TypeError("boom")

    module.make = make
    monkeypatch.setitem(sys.modules, "exp188_fake_sources", module)
    with pytest.raises(TypeError, match="boom"):  # a real error in the body, not a sources_invalid refusal
        run_py.main(_argv(tmp_path, "--no-ledger", sources=FAKE))


def test_run_refuses_sources_without_a_dict_expected(tmp_path):
    with pytest.raises(OpsError) as caught:
        driver.run(sources(expected=None), "inner", tmp_path, prereg=OK)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.details["reason"] == "sources_invalid"
    assert not (tmp_path / "receipts").exists()


def test_run_py_refuses_a_factory_result_that_is_not_sources(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    _install_sources(monkeypatch, object())
    assert run_py.main(_argv(tmp_path, "--no-ledger", sources=FAKE)) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == "INVALID_EXPERIMENT_SPEC"
    assert refused["reason"] == "sources_invalid"


def test_run_py_refuses_a_factory_with_the_wrong_signature(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    module = types.ModuleType("exp188_fake_sources")
    module.make = lambda: sources()  # takes no args, so calling it with the parsed args is a TypeError
    monkeypatch.setitem(sys.modules, "exp188_fake_sources", module)
    assert run_py.main(_argv(tmp_path, "--no-ledger", sources=FAKE)) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == "INVALID_EXPERIMENT_SPEC"
    assert refused["reason"] == "sources_invalid"


def test_run_py_refusal_message_carries_the_import_error_cause(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")

    def broken_import(name):
        raise ImportError("adapter dep missing")

    monkeypatch.setattr("importlib.import_module", broken_import)
    assert run_py.main(_argv(tmp_path, "--no-ledger", sources=FAKE)) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["reason"] == "sources_invalid"
    assert "adapter dep missing" in refused["message"]


def test_run_py_refuses_an_expected_that_differs_from_the_sources(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    _install_sources(monkeypatch, sources(expected={**EXPECTED, "panel_id": "OTHER"}))
    assert run_py.main(_argv(tmp_path, "--no-ledger", sources=FAKE)) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == CONFLICT
    assert refused["reason"] == "expected_mismatch"
    assert not (tmp_path / "out").exists()  # refused before the driver wrote anything


def test_run_py_refuses_an_expected_that_is_not_a_json_object(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    argv = _argv(tmp_path, "--no-ledger")
    (tmp_path / "expected.json").write_text("[1]", encoding="utf-8")
    assert run_py.main(argv) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == "INVALID_EXPERIMENT_SPEC"
    assert refused["reason"] == "input_unreadable"


def test_run_py_refuses_an_expected_missing_identity_keys_before_the_factory(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    calls = []
    module = types.ModuleType("exp188_fake_sources")

    def make(args):  # spy: records any call to the factory
        calls.append(args)
        return sources()

    module.make = make
    monkeypatch.setitem(sys.modules, "exp188_fake_sources", module)
    argv = _argv(tmp_path, "--no-ledger", sources=FAKE)
    (tmp_path / "expected.json").write_text("{}", encoding="utf-8")
    assert run_py.main(argv) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == "INVALID_EXPERIMENT_SPEC"
    assert refused["reason"] == "input_unreadable"
    assert calls == []


def test_run_py_refuses_an_expected_that_is_a_symlink(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    argv = _argv(tmp_path, "--no-ledger")
    target = tmp_path / "target.json"
    target.write_text(json.dumps(EXPECTED), encoding="utf-8")
    (tmp_path / "expected.json").unlink()
    os.symlink(target, tmp_path / "expected.json")
    assert run_py.main(argv) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == "INVALID_EXPERIMENT_SPEC"
    assert refused["reason"] == "input_unreadable"


def test_run_py_main_returns_zero_and_prints_the_summary(tmp_path, stubs, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    _install_sources(monkeypatch, sources())
    assert run_py.main(_argv(tmp_path, "--no-ledger", sources=FAKE)) == 0
    summary = json.loads(capsys.readouterr().out)
    inner = tuning.load_receipt(tmp_path / "out" / "receipts" / "inner.json")
    assert summary["stage"] == "inner"
    assert summary["models"] == sorted(inner["models"]) and summary["models"]


def test_run_py_main_refuses_a_plan_without_exit_date(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    _install_sources(monkeypatch, sources(plan_frame=PLAN_FRAME.drop(columns="exit_date")))
    assert run_py.main(_argv(tmp_path, "--no-ledger", sources=FAKE)) == 2
    assert json.loads(capsys.readouterr().out)["refused"] == CONFLICT


def test_run_py_stdout_is_one_json_document_when_a_dependency_prints(tmp_path, stubs, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])

    def noisy(ids):  # a library print during run
        print("noise")
        return FACTS.reindex(list(ids))

    _install_sources(monkeypatch, sources(facts=noisy))
    assert run_py.main(_argv(tmp_path, "--no-ledger", sources=FAKE)) == 0
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["stage"] == "inner"
    assert "noise" in captured.err


def test_run_py_refuses_a_native_data_error_as_json(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    module = types.ModuleType("exp188_fake_sources")

    def make(args):
        raise data_errors.fail("CONTRACT_MISMATCH", "x")

    module.make = make
    monkeypatch.setitem(sys.modules, "exp188_fake_sources", module)
    assert run_py.main(_argv(tmp_path, "--no-ledger", sources=FAKE)) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == "CONTRACT_MISMATCH"
    assert refused["reason"] is None


def test_run_py_refusal_message_has_no_absolute_path(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    argv = _argv(tmp_path, "--no-ledger")
    argv[argv.index("--expected") + 1] = "/nonexistent/abs/path.json"
    assert run_py.main(argv) == 2
    out = capsys.readouterr().out
    assert "/nonexistent" not in out and "<path>" in json.loads(out)["message"]


def test_run_py_synthetic_stage_dispatches_to_run_synthetic_with_smoke_ids_and_replay(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    _install_sources(monkeypatch, sources())
    (tmp_path / "ids.json").write_text('["E1"]', encoding="utf-8")
    calls = []

    def fake_synthetic(src, out_dir, *args, **kwargs):
        calls.append(kwargs)
        return {"stage": "synthetic", "models": [], "rows": 0, "stop": []}

    def no_run(*args, **kwargs):
        raise AssertionError("the synthetic stage must not dispatch to run")

    monkeypatch.setattr(driver, "run_synthetic", fake_synthetic)
    monkeypatch.setattr(driver, "run", no_run)
    argv = _argv(tmp_path, "--no-ledger", "--replay", "--event-ids", str(tmp_path / "ids.json"), sources=FAKE)
    argv[argv.index("--stage") + 1] = "synthetic"
    assert run_py.main(argv) == 0
    assert len(calls) == 1 and calls[0]["event_ids"] == ["E1"] and calls[0]["replay"] is True
    assert json.loads(capsys.readouterr().out)["stage"] == "synthetic"


def test_run_py_replay_with_another_stage_is_ignored(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    _install_sources(monkeypatch, sources())
    stages = []

    def fake_run(src, stage, out_dir, **kwargs):
        stages.append(stage)
        return {"stage": stage, "models": [], "excluded": {}, "rows": 0, "ladder_skipped": {}}

    def no_synthetic(*args, **kwargs):
        raise AssertionError("the inner stage must not dispatch to run_synthetic")

    monkeypatch.setattr(driver, "run", fake_run)
    monkeypatch.setattr(driver, "run_synthetic", no_synthetic)
    assert run_py.main(_argv(tmp_path, "--no-ledger", "--replay", sources=FAKE)) == 0
    assert stages == ["inner"]
    assert json.loads(capsys.readouterr().out)["stage"] == "inner"


def test_frames_digest_binds_column_names_and_dtypes():
    base = pd.DataFrame({"event_id": ["E1", "E2"], "pnl_dollars": [1.0, 2.0]})
    renamed = base.rename(columns={"pnl_dollars": "pnl"})
    retyped = base.assign(pnl_dollars=base["pnl_dollars"].astype("float32"))
    assert driver._frames_digest([base]) == driver._frames_digest([base.copy()])
    assert driver._frames_digest([base]) != driver._frames_digest([renamed])
    assert driver._frames_digest([base]) != driver._frames_digest([retyped])


# ---- facts exclusion, the population requested from facts, and the funnel -----------

def test_facts_missing_event_leaves_every_table_and_the_facts_receipt(tmp_path, stubs, monkeypatch, capsys):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    requested = []

    def facts(ids):
        requested.append(sorted(ids))
        return FACTS.reindex(list(ids)).assign(pre_iv30=lambda f: f["pre_iv30"].where(f.index != "E2"))

    src = sources(facts=facts)
    summary = driver.run(src, "inner", tmp_path, prereg=OK)
    assert requested == [["E1", "E2"]]
    assert src.exclusions["facts_missing"] == {"E2"}
    inputs = tuning.load_receipt(tmp_path / "receipts" / "inputs_inner.json")
    inner = tuning.load_receipt(tmp_path / "receipts" / "inner.json")
    assert inner["events_by_model"] == {summary["models"][0]: 1}  # E2 is in no model's events
    assert inputs["facts"] == content_hash(driver._facts_rows(FACTS.loc[["E1"]], {"E1"}))
    err = capsys.readouterr().err
    assert "[pnl_driver] facts_missing=1" in err
    assert "[pnl_driver] funnel admitted=4 paired=1 excluded: facts_missing=1 common_dropped=0" in err


def test_event_excluded_before_facts_is_never_requested(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    requested = []

    def fetch(columns, ids):  # E2 has no production row, so it leaves the population before any facts request
        return BASE[BASE["event_id"].isin(list(ids)) & (BASE["event_id"] != "E2")]

    def facts(ids):
        requested.append(sorted(ids))
        return FACTS.reindex(list(ids))

    summary = driver.run(sources(fetch=fetch, facts=facts), "inner", tmp_path, prereg=OK)
    assert requested == [["E1"]]
    assert summary["models"]


def test_every_event_dropped_by_facts_refuses_empty_population(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    facts = lambda ids: FACTS.reindex(list(ids)).assign(pre_iv30=np.nan)  # noqa: E731
    with pytest.raises(Refusal) as caught:
        driver.run(sources(facts=facts), "inner", tmp_path, prereg=OK)
    assert caught.value.details["reason"] == "empty_population"
    assert not (tmp_path / "receipts").exists()


# ---- pre-2025 events without a label leave the paired population -----------------

def _labels_blanked(*eids):
    """A fetch that serves BASE with the label NaN for ``eids``: each keeps its forecast and facts but has no label row."""
    def fetch(columns, ids):
        out = BASE[BASE["event_id"].isin(list(ids))].copy()
        out.loc[out["event_id"].isin(list(eids)), "abs_move"] = np.nan
        return out
    return fetch


def test_inner_event_without_a_label_leaves_every_model_table_and_the_receipts(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    summary = driver.run(sources(fetch=_labels_blanked("E2")), "inner", tmp_path, prereg=OK)
    inner = tuning.load_receipt(tmp_path / "receipts" / "inner.json")
    assert inner["events_by_model"] == {mid: 1 for mid in summary["models"]}
    assert inner["events"] == 1


def test_every_inner_event_without_a_label_refuses_empty_population(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(fetch=_labels_blanked("E1", "E2")), "inner", tmp_path, prereg=OK)
    assert caught.value.details["reason"] == "empty_population"
    assert not (tmp_path / "receipts").exists()
