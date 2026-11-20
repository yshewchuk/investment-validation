"""EXP-188 S9a: inner-stage driver orchestration with small fakes. Real inputs and readers; pool, simulation and tuning
stubbed."""
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
    def fetch(columns, ids):
        return BASE[BASE["event_id"].isin(list(ids))]

    base = dict(plan=PLAN, read=lambda slot: SEAL if slot == "driver-seal" else None,
                selection={"selections": []}, pair_candidates={}, blend_ids={}, expected=EXPECTED,
                panel_id="P", admit=lambda ids: {"admitted": sorted(ids), "dropped": []},
                admitted=ADMITTED, fetch=fetch, production_hash="H", production_table_hash="H",
                plan_frame=PLAN_FRAME, load_index=lambda keys: None,
                facts=lambda ids: FACTS.reindex(list(ids)), snapshot_id="snap-1")
    base.update(over)
    return driver.Sources(**base)


def chunk(*eids):
    """A minimal ladder chunk: the event ids and dte_exit are all the driver reads outside the simulator."""
    return {"event_id": np.array(eids, dtype=object), "dte_exit": np.full(len(eids), 9, dtype=np.int32)}


@pytest.fixture
def stubs(monkeypatch):
    """Pool, simulation and tuning are stubbed so the tests exercise the driver's own orchestration.

    Returns the call log that the tune stub appends to.
    """
    log = []

    def fake_tune(items, mid, **kwargs):
        log.append(("tune", mid, tuple(sorted(str(e) for _c, ev, _s in items for e in ev["event_id"]))))
        return {"stage": "tuning", "model_id": mid, "c": 1.0, "skipped": False,
                "events": sum(len(ev) for _c, ev, _s in items)}

    monkeypatch.setattr(driver, "build_pool", lambda table, labels: None)
    monkeypatch.setattr(tuning, "tune", fake_tune)
    monkeypatch.setattr(driver.shortvol, "simulate", lambda chunk_, events, pool, **kw: {})
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


@pytest.mark.parametrize("stage", ["outer", "2025"])
def test_only_the_inner_stage_runs(tmp_path, stage):
    with pytest.raises(ValueError, match="stage must be one of"):
        driver.run(sources(), stage, tmp_path, prereg=OK)


def test_prepare_rejects_a_typo_stage_before_any_source_call():
    calls = []
    with pytest.raises(ValueError, match="stage must be one of"):
        driver.prepare(sources(read=lambda slot: calls.append(slot)), "typo")
    assert calls == []


# ---- inner stage, replay and identity -----------------------------------------

def test_inner_writes_receipts_replays_read_only_and_refuses_a_changed_binding(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    src, out = sources(), tmp_path / "out"
    summary = driver.run(src, "inner", out, prereg=OK)
    mid = summary["models"][0]
    assert sorted(p.name for p in (out / "receipts").iterdir()) == sorted(
        ["inputs_inner.json", f"tuning_{mid}.json", "inner.json"])
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
    tuned = tuning.load_receipt(tmp_path / "receipts" / f"tuning_{mid}.json")
    assert inputs["scope"] is None and inner["scope"] is None
    assert inputs["snapshot"] == "snap-1"
    production = next(m["model_id"] for m in inputs["roster"]["models"] if m["kind"] == "production")
    assert inputs["models"][production] == {"status": "ready"}
    # Literal content of each receipt (the tune stub's fields, and the bindings between receipts).
    assert tuned["stage"] == "tuning" and tuned["model_id"] == mid and tuned["c"] == 1.0
    assert inner["stage"] == "inner"
    assert inner["models"] == {mid: tuned["content_hash"]}
    assert inner["inputs"] == inputs["content_hash"]
    assert inner["ladder_skipped"] == {}
    assert inner["events_by_model"] == {mid: 1}
    assert inputs["stage"] == "inputs"
    for digest in (inputs["plan"], inputs["facts"]):
        assert digest.startswith("sha256:")
        remainder = digest[len("sha256:"):]
        assert len(remainder) == 64 and set(remainder) <= set("0123456789abcdef")


def test_real_tune_skips_with_no_trades_and_binds_the_receipt(tmp_path, monkeypatch):
    """The real ``tune`` (not stubbed) on a fixture where nothing is taken: the S6 fallback is persisted and bound."""
    monkeypatch.setattr(driver, "build_pool", lambda table, labels: None)
    monkeypatch.setattr(driver.shortvol, "simulate", lambda chunk_, events, pool, **kw: {})
    scored_ids = lambda chunk_, events, sims, name, **kw: pd.DataFrame({"event_id": events["event_id"].to_numpy()})  # noqa: E731
    monkeypatch.setattr(tuning, "score_events", scored_ids)
    monkeypatch.setattr(tuning, "decide", lambda scored: scored.iloc[:0])  # every row undetermined, none taken
    monkeypatch.setattr(tuning, "score", lambda chunk_, events, mine, name, alpha, c: pd.DataFrame(
        {"pnl_dollars": pd.Series(dtype=float)}))
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    summary = driver.run(sources(), "inner", tmp_path, prereg=OK)
    mid = summary["models"][0]
    tuned = tuning.load_receipt(tmp_path / "receipts" / f"tuning_{mid}.json")
    inner = tuning.load_receipt(tmp_path / "receipts" / "inner.json")
    _c, reason = tuning.choose_c({1.0: 0.0}, 0)  # the literal skip reason for fewer than MIN_TRADES
    assert tuned["skipped"] is True and tuned["c"] == 1.0 and tuned["trades"] == 0
    assert tuned["skip_reason"] == reason == "fewer_than_300_inner_trades"
    assert inner["models"][mid] == tuned["content_hash"]


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
    out = driver.model_events(table, facts, ladder).set_index("event_id")
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


def test_inner_refuses_a_changed_plan_window_on_replay(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    changed = PLAN_FRAME.copy()
    changed.loc[changed["event_id"] == "E2", "entry_date"] = pd.Timestamp("2024-08-05")  # same ids, one entry_date
    with pytest.raises(Refusal) as caught:
        driver.run(sources(plan_frame=changed), "inner", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inputs_receipt_changed"


def test_inner_month_pool_drops_a_label_printed_after_a_later_month_starts(tmp_path, stubs, monkeypatch):
    """E1's exit (2024-08-03) falls after E2's month starts (2024-08-01): E1 is in E2's month table but not its labels."""
    plan = PLAN_FRAME.copy()
    plan.loc[plan["event_id"] == "E1", "exit_date"] = pd.Timestamp("2024-08-03")
    pools = []
    monkeypatch.setattr(driver, "build_pool",
                        lambda table, labels: pools.append((set(table.index.astype(str)), set(labels.index.astype(str)))))
    use_chunks(monkeypatch, [("2023Q2", chunk("E1", "E2"))])
    driver.run(sources(plan_frame=plan), "inner", tmp_path, prereg=OK)
    assert any("E1" in tabled and "E1" not in labelled for tabled, labelled in pools)


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


def test_facts_missing_a_requested_id_refuses_before_any_receipt(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    with pytest.raises(OpsError) as caught:
        driver.run(sources(facts=lambda ids: FACTS.reindex(["E1"])), "inner", tmp_path, prereg=OK)
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


def test_run_py_requires_no_ledger(tmp_path):
    with pytest.raises(SystemExit):
        _run_main().main(_argv(tmp_path))


def test_run_py_accepts_only_the_inner_stage(tmp_path):
    with pytest.raises(SystemExit):
        _run_main().main(["--stage", "outer", "--out-dir", str(tmp_path), "--expected", "e.json",
                          "--sources", NO_SOURCES, "--no-ledger"])


def test_run_py_refuses_on_prereg_before_importing_sources(tmp_path, monkeypatch, capsys):
    run_py = _run_main()

    def refuse():
        raise fail(CONFLICT, "no planned row", details={"reason": "planned_row_missing"})

    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", refuse)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    assert run_py.main(_argv(tmp_path, "--no-ledger")) == 2
    assert json.loads(capsys.readouterr().out)["refused"] == CONFLICT


def test_run_py_refuses_a_sources_spec_without_a_colon(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    assert run_py.main(_argv(tmp_path, "--no-ledger", sources="nocolon")) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "sources_invalid"


def test_run_py_refuses_not_bounded_before_importing_sources(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.delenv("BOUNDED_RUN_NESTED", raising=False)
    assert run_py.main(_argv(tmp_path, "--no-ledger")) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == "RESOURCE_UNAVAILABLE"
    assert refused["reason"] == "not_bounded"


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
