"""EXP-188 S9b: the outer stage and the smoke ids (``--event-ids``) with small fakes. Real inputs and readers; pool,
simulation, scorers and tuning stubbed."""
import json
import shutil
import sys
import types

import numpy as np
import pandas as pd
import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import size_pnl_driver as driver
from experiments import size_pnl_readers as readers
from experiments import size_pnl_tuning as tuning
from experiments.size_pnl_shortvol import Refusal
from tests.v2.research.test_size_pnl_driver import (  # noqa: F401  (bounded, stubs, structure_gate_passes are fixtures)
    BASE, CONFLICT, DENIED, EVENTS, EXPECTED, FACTS, FAKE, OK, PLAN_FRAME, _install_sources, _run_main, bounded, chunk, sources,
    stubs, structure_gate_passes)

FRONT = ["model_id", "book", "arm_or_rule", "structure", "event_id", "event_date", "status", "reason"]
EMPTY_MARKS = ["event_id", "entry_cost", "exit_value", "rung", "lots", "pnl_per_share", "pnl_dollars"]


def use_chunks(monkeypatch, pairs, seen=None):
    """Replace the ladder with a generator over fixed (quarter, chunk) pairs; ``seen`` records each plan it is given."""
    def fake(admitted, plan, load_index):
        if seen is not None: seen.append(plan)
        yield from pairs
    monkeypatch.setattr(driver.ladder, "ladder_chunks", fake)


def recording(log):
    """A fetch that logs (``labels`` or ``production``, ids) per call, then serves the BASE rows."""
    def fetch(columns, ids):
        ids = [str(e) for e in ids]
        log.append(("labels" if "abs_move" in columns else "production", tuple(ids)))
        return BASE[BASE["event_id"].isin(ids)]
    return fetch


def _run_both(tmp_path, monkeypatch, src=None):
    """Inner over 2024 Q3, then outer over 2024 Q3 (warm-up) and 2025 Q1. Returns the outer summary."""
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(src or sources(), "inner", tmp_path, prereg=OK)
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    return driver.run(src or sources(), "outer", tmp_path, prereg=OK)


def _argv_for(tmp_path, stage, *extra):
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps(EXPECTED), encoding="utf-8")
    return ["--stage", stage, "--out-dir", str(tmp_path / "out"), "--expected", str(expected), "--sources", FAKE, "--no-ledger", *extra]


@pytest.fixture
def outer_stubs(stubs, monkeypatch):
    """Scorers and straddle stubs for the outer stage: every row is scored and none is taken. The straddle decisions
    append their ids to the shared ``stubs`` log, so the tests can order decisions against label reads."""
    def fake_scored(chunk_, events, sims, name, *, alpha=0.5, c=1.0):
        return pd.DataFrame({"event_id": events["event_id"].astype(str).to_numpy(), "event_date": events["event_date"].to_numpy(),
                             "structure": name, "status": "scored", "reason": "scored"})

    def fake_straddle(events, history, rule, chunk_):
        stubs.append(("decide", tuple(sorted(events["event_id"].astype(str)))))
        return pd.DataFrame({"event_id": events["event_id"].astype(str), "event_date": events["event_date"],
                             "edge": events["pred_abs_move"] - events["implied_move"], "cutoff": np.nan, "rel_spread": np.nan,
                             "status": "undetermined", "reason": "thin_history"})

    monkeypatch.setattr(driver.shortvol, "score_events", fake_scored)
    monkeypatch.setattr(driver.shortvol, "decide", lambda scored: scored.assign(cutoff=np.nan))
    monkeypatch.setattr(driver.shortvol, "score", lambda chunk_, events, decisions, name, alpha, *, c=1.0, drop_unquoted=False: pd.DataFrame(columns=EMPTY_MARKS))
    monkeypatch.setattr(driver.straddle, "decide", fake_straddle)
    monkeypatch.setattr(driver.straddle, "score", lambda chunk_, events, dec, alpha, **kw: pd.DataFrame(columns=EMPTY_MARKS))
    return stubs


# ---- refusals before any chunk --------------------------------------------------

def test_outer_refuses_before_building_any_chunk(tmp_path, outer_stubs):
    loads = []
    with pytest.raises(Refusal) as caught:
        driver.run(sources(load_index=lambda keys: loads.append(keys)), "outer", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "tuning_receipt_missing"
    assert loads == []
    assert not (tmp_path / "receipts" / "inputs_outer.json").exists()


def _stub_tables(monkeypatch, rows, moves=None):
    """Roster of one production and one candidate model; each model's table holds its own ``rows`` (event id -> date),
    clipped to the stage's window. ``moves`` (model id -> pred_move) tells the models apart inside a stub; default 5.0."""
    roster = {"models": [{"model_id": "prod", "kind": "production"}, {"model_id": "cand", "kind": "v14"}]}
    monkeypatch.setattr(driver.inputs, "resolve_roster", lambda *a, **kw: roster)
    monkeypatch.setattr(driver.inputs, "bind_models", lambda *a, **kw: {})

    def model_tables(bound, roster_, admitted, production, *, start, end, **kw):
        out = {}
        for mid, events in rows.items():
            keep = [(e, d) for e, d in events.items() if pd.Timestamp(start) <= pd.Timestamp(d) < pd.Timestamp(end)]
            table = pd.DataFrame({"pred_move": (moves or {}).get(mid, 5.0), "pred_crush": -12.0, "event_date": pd.to_datetime([d for _e, d in keep])},
                                 index=pd.Index([e for e, _d in keep], name="event_id"))
            out[mid] = {"status": "ready", "table": table, "binding": {"rows": len(keep)}}
        return out

    monkeypatch.setattr(readers, "model_tables", model_tables)


@pytest.mark.parametrize("short", ["prod", "cand"])
def test_outer_refuses_a_ready_model_without_2025_rows_before_any_chunk(tmp_path, stubs, monkeypatch, short):
    """A ready model with only 2024 rows refuses, production or candidate: its 2025 predictions are not bound yet."""
    rows = {"prod": EVENTS, "cand": EVENTS}
    rows[short] = {"E2": EVENTS["E2"]}
    _stub_tables(monkeypatch, rows)
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    loads = []
    with pytest.raises(Refusal) as caught:
        driver.run(sources(load_index=lambda keys: loads.append(keys)), "outer", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "outer_predictions_unbound"
    assert loads == []
    assert not (tmp_path / "receipts" / "inputs_outer.json").exists()  # the refusal leaves no outer input receipt


def test_outer_receipt_binds_what_it_consumed(tmp_path, outer_stubs, monkeypatch):
    _run_both(tmp_path, monkeypatch)
    outer = tuning.load_receipt(tmp_path / "receipts" / "outer.json")
    tuned = {mid: tuning.load_receipt(tmp_path / "receipts" / f"tuning_{mid}.json") for mid in outer["models"]}
    assert outer["tuning"] == {mid: t["content_hash"] for mid, t in tuned.items()}
    assert outer["c"] == {mid: t["c"] for mid, t in tuned.items()}
    assert outer["labels"] == content_hash([["E3", 4.0, -10.0], ["E4", 4.0, -10.0]])  # the 2025 month reads, in order
    assert outer["common_dropped"] == 0


def test_partial_2025_coverage_is_a_recorded_common_drop(tmp_path, outer_stubs, monkeypatch):
    """Candidate has no E4 row: not refused, E4 leaves the common pool and is counted in outer.json."""
    _stub_tables(monkeypatch, {"prod": EVENTS, "cand": {k: v for k, v in EVENTS.items() if k != "E4"}})
    outer = _run_both(tmp_path, monkeypatch)
    assert outer["rows"] > 0
    assert tuning.load_receipt(tmp_path / "receipts" / "outer.json")["common_dropped"] == 1


def test_changed_pre_2025_label_between_stages_refuses_inner_inputs(tmp_path, outer_stubs, monkeypatch):
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)

    def changed(columns, ids):
        rows = BASE[BASE["event_id"].isin([str(e) for e in ids])].copy()
        if "abs_move" in columns: rows["abs_move"] = rows["abs_move"] + 1.0
        return rows

    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(fetch=changed), "outer", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inner_inputs_mismatch"
    assert not (tmp_path / "receipts" / "inputs_outer.json").exists()


def test_changed_inner_window_prediction_between_stages_refuses_inner_tables(tmp_path, outer_stubs, monkeypatch):
    """Labels are unchanged, but one inner-window production value moved: the re-bound inner tables and production differ."""
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)

    def moved(columns, ids):
        rows = BASE[BASE["event_id"].isin([str(e) for e in ids])].copy()
        if "pred_abs_move" in columns: rows.loc[rows["event_id"] == "E2", "pred_abs_move"] = 6.0
        return rows

    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(fetch=moved), "outer", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inner_tables_mismatch"
    assert not (tmp_path / "receipts" / "inputs_outer.json").exists()  # preflight: no outer input receipt


def test_changed_inner_window_plan_row_between_stages_refuses_inner_plan(tmp_path, outer_stubs, monkeypatch):
    """Tables, production and labels are unchanged; E2 (inner window) has a different entry_date in the outer run's plan."""
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    changed = PLAN_FRAME.copy()
    changed.loc[changed["event_id"] == "E2", "entry_date"] = pd.Timestamp("2024-08-05")  # same ids, one entry_date, still valid
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(plan_frame=changed), "outer", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inner_plan_mismatch"
    assert not (tmp_path / "receipts" / "inputs_outer.json").exists()  # preflight: no outer input receipt


def test_changed_inner_window_fact_between_stages_refuses_inner_facts(tmp_path, outer_stubs, monkeypatch):
    """Labels, tables and plan are unchanged; E2 (inner window) has a different pre_iv30 in the outer run's facts."""
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    moved = FACTS.copy()
    moved.loc["E2", "pre_iv30"] = 41.0
    calls = []

    def counting(ids):
        calls.append(list(ids))
        return moved.reindex(list(ids))

    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(facts=counting), "outer", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inner_facts_mismatch"
    assert not (tmp_path / "receipts" / "inputs_outer.json").exists()  # preflight: no outer input receipt
    assert len(calls) == 1  # the outer run still makes exactly one facts call


def test_outer_makes_one_facts_call_for_the_stage_and_inner_window(tmp_path, outer_stubs, monkeypatch):
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    calls = []

    def counting(ids):
        calls.append(sorted(ids))
        return FACTS.reindex(list(ids))

    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    driver.run(sources(facts=counting), "outer", tmp_path, prereg=OK)
    assert calls == [["E1", "E2", "E3", "E4"]]  # one call: the stage plan's ids plus the inner window's


def test_oracle_one_taken_2025_event_carries_its_literal_marks(tmp_path, outer_stubs, monkeypatch):
    """Only E3 is taken; its scorer marks are literal, so its decision rows must carry exactly those values."""
    monkeypatch.setattr(driver.shortvol, "decide", lambda scored: scored[scored["event_id"].astype(str) == "E3"].assign(status="taken", reason="taken"))
    marks = {"event_id": ["E3"], "exit_value": [101.5], "rung": [2], "lots": [3], "pnl_dollars": [-42.25], "entry_cost": [10.0], "pnl_per_share": [-0.5]}
    monkeypatch.setattr(driver.shortvol, "score", lambda chunk_, events, decisions, name, alpha, *, c=1.0, drop_unquoted=False: pd.DataFrame(marks))
    _run_both(tmp_path, monkeypatch)
    frame = pd.read_parquet(tmp_path / "decisions.parquet")
    taken = frame[(frame["book"] == "short_vol") & (frame["event_id"] == "E3")]
    assert len(taken) == 2 * len(driver.shortvol.NAMES)  # both arms, every short-vol structure
    assert (taken["pnl_dollars"] == -42.25).all() and (taken["rung"] == 2).all()


def test_outer_refuses_when_inner_json_names_another_model_hash(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path / "inner", prereg=OK)
    inner = tuning.load_receipt(tmp_path / "inner" / "receipts" / "inner.json")
    tampered = {k: v for k, v in inner.items() if k != "content_hash"}
    tampered["models"] = {mid: "0" * 64 for mid in tampered["models"]}
    (tmp_path / "inner" / "receipts" / "inner.json").unlink()
    tuning.write_receipt(tmp_path / "inner" / "receipts" / "inner.json", tampered)
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "outer", tmp_path / "inner", prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inner_receipt_mismatch"


def test_outer_refuses_when_inner_json_names_other_inner_tables(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path / "inner", prereg=OK)
    copy = tmp_path / "copy"
    shutil.copytree(tmp_path / "inner", copy)
    inner = tuning.load_receipt(copy / "receipts" / "inner.json")
    tampered = {k: v for k, v in inner.items() if k != "content_hash"}
    tampered["inputs"] = "0" * 64
    (copy / "receipts" / "inner.json").unlink()
    tuning.write_receipt(copy / "receipts" / "inner.json", tampered)
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "outer", copy, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inner_tables_mismatch"


def test_outer_refuses_an_inner_receipt_tuned_on_smoke_ids(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1"))])
    driver.run(sources(), "inner", tmp_path / "inner", event_ids={"E1"}, prereg=OK)
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "outer", tmp_path / "inner", prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "scope_mismatch"


# ---- smoke ids -------------------------------------------------------------------

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


def test_non_collection_smoke_event_ids_is_an_invalid_spec(tmp_path, stubs):
    with pytest.raises(OpsError) as caught:
        driver.run(sources(), "inner", tmp_path, event_ids=0, prereg=OK)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.details["reason"] == "event_ids_invalid"


def test_empty_smoke_set_refuses_before_any_receipt(tmp_path, stubs):
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "inner", tmp_path, event_ids=set(), prereg=OK)
    assert caught.value.details["reason"] == "empty_population"
    assert not (tmp_path / "receipts").exists()


def test_smoke_ids_outside_the_window_refuse_before_the_inputs_receipt(tmp_path, stubs, monkeypatch):
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "inner", tmp_path, event_ids={"E3"}, prereg=OK)  # E3 is outside the inner window
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "empty_population"
    assert not (tmp_path / "receipts" / "inputs_inner.json").exists()
    summary = driver.run(sources(), "inner", tmp_path, event_ids={"E2"}, prereg=OK)  # the corrected rerun succeeds
    assert summary["stage"] == "inner"
    assert (tmp_path / "receipts" / "inputs_inner.json").exists()


def test_run_py_forwards_event_ids_to_the_outer_stage(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps(["E1", "E2"]), encoding="utf-8")
    seen = {}

    def spy(src, stage, out_dir, *, event_ids=None, prereg=None):
        seen.update(stage=stage, event_ids=event_ids)
        return {"stage": stage, "models": [], "excluded": {}, "rows": 0, "ladder_skipped": {}}

    monkeypatch.setattr(driver, "run", spy)
    _install_sources(monkeypatch, sources())
    assert run_py.main(_argv_for(tmp_path, "outer", "--event-ids", str(ids_file))) == 0
    assert seen == {"stage": "outer", "event_ids": ["E1", "E2"]}
    assert json.loads(capsys.readouterr().out)["stage"] == "outer"


def test_run_py_refuses_an_event_ids_file_holding_null(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    ids_file = tmp_path / "ids.json"
    ids_file.write_text("null", encoding="utf-8")
    monkeypatch.setattr(driver, "run", lambda *a, **kw: pytest.fail("the driver must not run"))
    _install_sources(monkeypatch, sources())
    assert run_py.main(_argv_for(tmp_path, "inner", "--event-ids", str(ids_file))) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == "INVALID_EXPERIMENT_SPEC"
    assert refused["reason"] == "event_ids_invalid"


def test_smoke_tables_binding_describes_the_rows_consumed(tmp_path, stubs, monkeypatch):
    real, captured = readers.model_tables, {}

    def spy(*args, **kw):
        out = real(*args, **kw)
        captured.update(out)
        return out

    monkeypatch.setattr(readers, "model_tables", spy)
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    full = driver.run(sources(), "inner", tmp_path / "full", prereg=OK)
    bindings = {mid: captured[mid]["binding"] for mid in full["models"]}
    assert tuning.load_receipt(tmp_path / "full" / "receipts" / "inputs_inner.json")["tables"] == bindings
    use_chunks(monkeypatch, [("2023Q2", chunk("E1"))])
    smoke = driver.run(sources(), "inner", tmp_path / "one", event_ids={"E1"}, prereg=OK)
    one = tuning.load_receipt(tmp_path / "one" / "receipts" / "inputs_inner.json")["tables"]
    assert set(one) == set(smoke["models"]) and one != bindings  # the smoke entry describes its rows, not the model binding
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path / "two", event_ids={"E1", "E2"}, prereg=OK)
    two = tuning.load_receipt(tmp_path / "two" / "receipts" / "inputs_inner.json")["tables"]
    assert two != one  # a changed smoke id set changes the entry


def test_outer_smoke_of_only_warm_up_events_refuses_unbound_2025(tmp_path, outer_stubs, monkeypatch):
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, event_ids={"E2"}, prereg=OK)  # E2 is 2024-H2: warm-up only
    loads = []
    with pytest.raises(Refusal) as caught:
        driver.run(sources(load_index=lambda keys: loads.append(keys)), "outer", tmp_path, event_ids={"E2"}, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "outer_predictions_unbound"
    assert loads == []
    assert not (tmp_path / "receipts" / "inputs_outer.json").exists()  # refused before the outer input receipt


# ---- outer stage: R6 ordering, the warm-up and the pool ---------------------------

def test_outer_reads_a_month_s_labels_only_after_its_rows_are_complete(tmp_path, outer_stubs, monkeypatch):
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)  # writes inputs, tuning and inner receipts
    outer_stubs.clear()
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    summary = driver.run(sources(fetch=recording(outer_stubs)), "outer", tmp_path, prereg=OK)
    # Two 2025 months, each with one row per (arm, short-vol name) and one per straddle rule.
    assert summary["rows"] == 2 * (2 * len(driver.shortvol.NAMES) + len(driver.straddle.RULES))
    assert (tmp_path / "decisions.parquet").exists()
    assert not list(tmp_path.glob("*.tmp"))
    for event in ("E3", "E4"):
        decided = max(i for i, e in enumerate(outer_stubs) if e[0] == "decide" and event in e[1])
        labelled = next(i for i, e in enumerate(outer_stubs) if e[0] == "labels" and event in e[1])
        assert labelled > decided


def test_warm_up_month_adds_history_only(tmp_path, outer_stubs, monkeypatch):
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    outer_stubs.clear()
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    summary = driver.run(sources(fetch=recording(outer_stubs)), "outer", tmp_path, prereg=OK)
    assert summary["rows"] == 0  # 2024 H2 is warm-up: history is built, but no decision row is written
    # Only the initial pre-2025 label read covers E2; no month's labels are read.
    assert sum(1 for e in outer_stubs if e[0] == "labels" and "E2" in e[1]) == 1


def test_outer_pool_sees_the_2023_history(tmp_path, outer_stubs, monkeypatch):
    """The outer tables span INNER[0]..OUTER[1], so a 2025 month's pool includes the 2023 residual row."""
    def model_tables(bound, roster, admitted, production, *, start, end, **kw):
        keep = [(e, d) for e, d in EVENTS.items() if pd.Timestamp(start) <= pd.Timestamp(d) < pd.Timestamp(end)]
        table = pd.DataFrame({"pred_move": 5.0, "pred_crush": -12.0, "event_date": pd.to_datetime([d for _e, d in keep])},
                             index=pd.Index([e for e, _d in keep], name="event_id"))
        return {m["model_id"]: {"status": "ready", "table": table, "binding": {"rows": len(keep)}} for m in roster["models"]}

    pools = []
    monkeypatch.setattr(readers, "model_tables", model_tables)
    monkeypatch.setattr(driver, "build_pool", lambda table, labels: pools.append(table))
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    pools.clear()
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    driver.run(sources(), "outer", tmp_path, prereg=OK)
    e1_date = pd.Timestamp(EVENTS["E1"])
    assert pools and all(t["event_date"].min() == e1_date for t in pools)  # the 2023 row is in every pool
    assert any(len(t) == 2 for t in pools)  # the 2025-01 month's pool holds the 2023 and 2024 rows


def test_decisions_parquet_has_the_front_columns_first(tmp_path, outer_stubs, monkeypatch):
    _run_both(tmp_path, monkeypatch)
    frame = pd.read_parquet(tmp_path / "decisions.parquet")
    assert list(frame.columns[:len(FRONT)]) == FRONT
    assert len(frame) > 0


# ---- outer receipts and the decisions file ----------------------------------------

def test_exact_outer_replay_does_not_rewrite_decisions(tmp_path, outer_stubs, monkeypatch):
    first = _run_both(tmp_path, monkeypatch)
    before = (tmp_path / "decisions.parquet").stat()
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    second = driver.run(sources(), "outer", tmp_path, prereg=OK)
    after = (tmp_path / "decisions.parquet").stat()
    assert second == first
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert not list(tmp_path.glob("*.tmp"))


def test_symlinked_decisions_refuses_an_outer_rerun(tmp_path, outer_stubs, monkeypatch):
    _run_both(tmp_path, monkeypatch)
    real = tmp_path / "real.parquet"
    (tmp_path / "decisions.parquet").rename(real)
    (tmp_path / "decisions.parquet").symlink_to(real)
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "outer", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "decisions_artifact_invalid"


def test_dangling_decisions_symlink_refuses_an_outer_rerun(tmp_path, outer_stubs, monkeypatch):
    _run_both(tmp_path, monkeypatch)
    final = tmp_path / "decisions.parquet"
    final.unlink()
    final.symlink_to(tmp_path / "missing.parquet")
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "outer", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "decisions_artifact_invalid"
    assert final.is_symlink() and not final.exists()


def test_changed_outer_receipt_leaves_decisions_untouched_and_no_tmp(tmp_path, outer_stubs, monkeypatch):
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    (tmp_path / "decisions.parquet").write_bytes(b"EXISTING")
    tuning.write_receipt(tmp_path / "receipts" / "outer.json", {"stage": "outer", "decisions_sha256": "OTHER", "rows": 99, "models": []})
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "outer", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "outer_receipt_changed"
    assert (tmp_path / "decisions.parquet").read_bytes() == b"EXISTING"
    assert not list(tmp_path.glob("*.tmp"))


def test_changed_ladder_quote_refuses_an_outer_replay(tmp_path, outer_stubs, monkeypatch):
    """The outer receipt binds the consumed quotes: one 2025 quote changed under identical decisions is refused, decisions untouched."""
    def quoted(bid):
        return {**chunk("E3", "E4"), "bid": np.array([1.5, bid])}

    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", quoted(2.0))])
    driver.run(sources(), "outer", tmp_path, prereg=OK)
    before = (tmp_path / "decisions.parquet").stat()
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", quoted(2.5))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "outer", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "outer_receipt_changed"
    after = (tmp_path / "decisions.parquet").stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert not list(tmp_path.glob("*.tmp"))


# ---- shared scoreability (design section 4) and the decisions cap ---------------------

def _shared_drop_run(tmp_path, monkeypatch):
    """Inner over 2024 Q3, then outer over 2024 Q3 and 2025 Q1 with the candidate's pred_move at 6.0 (the production's 5.0).
    Returns the decide inputs, recorded in call order."""
    _stub_tables(monkeypatch, {"prod": EVENTS, "cand": EVENTS}, moves={"cand": 6.0})
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    decide_inputs, passthrough = [], driver.shortvol.decide

    def recording_decide(scored):
        decide_inputs.append(scored.copy())
        return passthrough(scored)

    monkeypatch.setattr(driver.shortvol, "decide", recording_decide)
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    return decide_inputs


def _assert_e3_shared_dropped(tmp_path, decide_inputs):
    frame = pd.read_parquet(tmp_path / "decisions.parquet")
    assert set(frame["model_id"]) == {"prod", "cand"}  # both models still publish rows
    assert "E3" not in set(frame["event_id"].astype(str))
    assert not any("E3" in set(d["event_id"].astype(str)) for d in decide_inputs)  # neither model's history
    assert any("E4" in set(d["event_id"].astype(str)) for d in decide_inputs)
    assert tuning.load_receipt(tmp_path / "receipts" / "outer.json")["scoreability_dropped"] == 1


def test_one_model_simulation_failure_drops_the_event_for_every_model(tmp_path, outer_stubs, monkeypatch):
    decide_inputs = _shared_drop_run(tmp_path, monkeypatch)
    simulate = driver.shortvol.simulate

    def no_e3_for_cand(chunk_, events, pool, **kw):
        sims = simulate(chunk_, events, pool, **kw)
        if set(events["pred_move"]) == {6.0}: sims.pop("E3", None)  # the candidate's 2025-01 simulation is missing
        return sims

    monkeypatch.setattr(driver.shortvol, "simulate", no_e3_for_cand)
    driver.run(sources(), "outer", tmp_path, prereg=OK)
    _assert_e3_shared_dropped(tmp_path, decide_inputs)


def test_one_model_structure_refusal_drops_the_event_for_every_model(tmp_path, outer_stubs, monkeypatch):
    decide_inputs = _shared_drop_run(tmp_path, monkeypatch)
    first = driver.shortvol.NAMES[0]

    def refuse_e3_for_cand(chunk_, events, sims, name, *, alpha=0.5, c=1.0):
        ids = events["event_id"].astype(str).to_numpy()
        cand = set(events["pred_move"]) == {6.0} and name == first
        status = ["undetermined" if cand and e == "E3" else "scored" for e in ids]
        return pd.DataFrame({"event_id": ids, "event_date": events["event_date"].to_numpy(), "structure": name,
                             "status": status, "reason": "scored"})

    monkeypatch.setattr(driver.shortvol, "score_events", refuse_e3_for_cand)
    driver.run(sources(), "outer", tmp_path, prereg=OK)
    _assert_e3_shared_dropped(tmp_path, decide_inputs)


def test_oversized_decisions_tmp_refuses_before_outer_json(tmp_path, outer_stubs, monkeypatch):
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    monkeypatch.setattr(driver.decisions, "MAX_DECISIONS_BYTES", 10)
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "outer", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "decisions_artifact_invalid"
    assert not (tmp_path / "receipts" / "outer.json").exists()
    assert not (tmp_path / "decisions.parquet").exists()
    assert not list(tmp_path.glob("*.tmp"))


# ---- run.py: --event-ids shape, and the cross-month oracle ------------------------

@pytest.mark.parametrize("payload", [{"a": 1}, [1, 2]])
def test_run_py_refuses_event_ids_not_an_array_of_strings_before_the_factory(tmp_path, monkeypatch, capsys, payload):
    run_py = _run_main()
    monkeypatch.setattr(run_py.size_pnl_prereg, "require_preregistration", OK)
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps(payload), encoding="utf-8")
    calls = []
    module = types.ModuleType("exp188_fake_sources")

    def make(args):  # spy: records any call to the factory
        calls.append(args)
        return sources()

    module.make = make
    monkeypatch.setitem(sys.modules, "exp188_fake_sources", module)
    assert run_py.main(_argv_for(tmp_path, "inner", "--event-ids", str(ids_file))) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == "INVALID_EXPERIMENT_SPEC"
    assert refused["reason"] == "event_ids_invalid"
    assert calls == []


def test_cross_month_history_oracle_through_the_driver(tmp_path, outer_stubs, monkeypatch):
    """Warm-up 2024-H2 history reaches the 2025 decisions; each 2025 month's pool and labels obey R6 and the cutoff."""
    _stub_tables(monkeypatch, {"prod": EVENTS, "cand": EVENTS})
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)
    edges = {"E2": 0.25, "E3": 0.5, "E4": 0.75}
    statuses = {"E2": "scored", "E3": "scored", "E4": "scored"}

    def scored_literal(chunk_, events, sims, name, *, alpha=0.5, c=1.0):
        ids = events["event_id"].astype(str).to_numpy()
        return pd.DataFrame({"event_id": ids, "event_date": events["event_date"].to_numpy(), "structure": name,
                             "edge": [edges[e] for e in ids], "status": [statuses[e] for e in ids], "reason": "scored"})

    decide_inputs, passthrough = [], driver.shortvol.decide

    def recording_decide(scored):
        decide_inputs.append(scored.copy())
        return passthrough(scored)

    pools = []
    monkeypatch.setattr(driver.shortvol, "score_events", scored_literal)
    monkeypatch.setattr(driver.shortvol, "decide", recording_decide)
    monkeypatch.setattr(driver, "build_pool",
                        lambda table, labels: pools.append((table["event_date"].max(), set(labels.index.astype(str)))))
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    summary = driver.run(sources(), "outer", tmp_path, prereg=OK)
    assert summary["rows"] > 0

    # Decide ran only for the two 2025 months, never for the warm-up month: 2 ready models x 2 months x 2 arms x every short-vol name.
    assert len(decide_inputs) == 2 * 2 * 2 * len(driver.shortvol.NAMES)
    assert {"E2", "E3"} <= set(decide_inputs[0]["event_id"].astype(str))  # January: warm-up history plus its own event
    last = decide_inputs[-1]  # February: history through January plus February's event
    assert {"E2", "E3", "E4"} <= set(last["event_id"].astype(str))
    assert dict(zip(last["event_id"].astype(str), last["edge"])) == edges
    frame = pd.read_parquet(tmp_path / "decisions.parquet")
    assert (pd.to_datetime(frame["event_date"]) >= pd.Timestamp("2025-01-01")).all()

    # Pools: January's table is the rows before 2025-01-01 (max E2); February's adds E3 (max E3). Warm-up's max is E1.
    by_max = {}
    for max_date, labelled in pools:
        by_max.setdefault(max_date, []).append(labelled)
    january, february = pd.Timestamp("2024-08-07"), pd.Timestamp("2025-01-15")
    assert set(by_max) == {pd.Timestamp("2023-06-10"), january, february}
    # January: E3's exit (2025-01-16) is not printed before 2025-01-01, so its label is absent; E2's is present.
    assert by_max[january] and all("E3" not in labelled and "E2" in labelled for labelled in by_max[january])
    # February: E3's label, read after January's rows were complete, is present; E4's exit (2025-02-13) is not printed.
    assert by_max[february] and all("E3" in labelled and "E4" not in labelled for labelled in by_max[february])


def test_outer_arms_use_their_own_c_and_rows_carry_marks_and_forecasts(tmp_path, outer_stubs, monkeypatch):
    """The tuning receipts carry c=0.7. The as_is arm must score at c=1.0 and the retuned arm at c=0.7; every straddle rule
    carries literal marks; each model's rows carry that model's literal pred_move and pred_crush. The scorers are stubbed:
    the shortvol scorer records the c it receives and returns marks derived from it."""
    literals = {"prod": (5.25, -12.5), "cand": (6.75, -17.0)}  # (pred_move, pred_crush): distinct per model and per column
    marks = {"exit_value": 101.5, "rung": 2, "lots": 3, "pnl_dollars": -42.25}

    def tune_c07(items, mid, **kwargs):  # the inner run's tuning receipt: c=0.7 for every model
        return {"stage": "tuning", "model_id": mid, "c": 0.7, "skipped": False, "events": sum(len(ev) for _c, ev, _s in items)}

    monkeypatch.setattr(tuning, "tune", tune_c07)
    _stub_tables(monkeypatch, {"prod": EVENTS, "cand": EVENTS})  # roster and bindings; the tables are replaced below

    def model_tables(bound, roster, admitted, production, *, start, end, **kw):
        out = {}
        for mid, (move, crush) in literals.items():
            keep = [(e, d) for e, d in EVENTS.items() if pd.Timestamp(start) <= pd.Timestamp(d) < pd.Timestamp(end)]
            table = pd.DataFrame({"pred_move": move, "pred_crush": crush, "event_date": pd.to_datetime([d for _e, d in keep])},
                                 index=pd.Index([e for e, _d in keep], name="event_id"))
            out[mid] = {"status": "ready", "table": table, "binding": {"rows": len(keep)}}
        return out

    monkeypatch.setattr(readers, "model_tables", model_tables)
    use_chunks(monkeypatch, [("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, prereg=OK)  # tunes c=0.7; the inner run's own scoring is not recorded

    calls = []

    def recording_score_events(chunk_, events, sims, name, *, alpha=0.5, c=1.0):
        calls.append(c)
        return pd.DataFrame({"event_id": events["event_id"].astype(str).to_numpy(), "event_date": events["event_date"].to_numpy(),
                             "structure": name, "status": "scored", "reason": "scored"})

    def c_marks(chunk_, events, decisions, name, alpha, *, c=1.0, drop_unquoted=False):
        ids = decisions["event_id"].astype(str).to_numpy()
        return pd.DataFrame({"event_id": ids, "exit_value": marks["exit_value"], "rung": marks["rung"], "lots": marks["lots"],
                             "entry_cost": 10.0, "pnl_per_share": -0.5, "pnl_dollars": 100.0 * c})

    def taken_straddle(events, history, rule, chunk_):
        return pd.DataFrame({"event_id": events["event_id"].astype(str).to_numpy(), "event_date": events["event_date"].to_numpy(),
                             "edge": 0.5, "cutoff": np.nan, "rel_spread": np.nan, "status": "taken", "reason": "taken"})

    def straddle_marks(chunk_, events, dec, alpha, **kw):
        ids = dec["event_id"].astype(str).to_numpy()
        return pd.DataFrame({"event_id": ids, "exit_value": marks["exit_value"], "rung": marks["rung"], "lots": marks["lots"],
                             "pnl_dollars": marks["pnl_dollars"], "entry_cost": 10.0, "pnl_per_share": -0.5})

    monkeypatch.setattr(driver.shortvol, "score_events", recording_score_events)
    monkeypatch.setattr(driver.shortvol, "score", c_marks)
    monkeypatch.setattr(driver.straddle, "decide", taken_straddle)
    monkeypatch.setattr(driver.straddle, "score", straddle_marks)
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    driver.run(sources(), "outer", tmp_path, prereg=OK)

    outer = tuning.load_receipt(tmp_path / "receipts" / "outer.json")
    assert outer["c"] == {"prod": 0.7, "cand": 0.7}  # the receipt holds the c the test relies on
    assert set(calls) == {1.0, 0.7}  # every call received one of the two c values, and both were used

    frame = pd.read_parquet(tmp_path / "decisions.parquet")
    short = frame[frame["book"] == "short_vol"]
    for arm, expected in (("as_is", 100.0), ("retuned", 70.0)):
        rows = short[short["arm_or_rule"] == arm]
        assert len(rows) > 0
        assert np.isclose(rows["pnl_dollars"].to_numpy(dtype=float), expected).all()
    straddle_rows = frame[frame["book"] == "straddle"]
    assert set(straddle_rows["arm_or_rule"]) == set(driver.straddle.RULES)
    for rule in driver.straddle.RULES:
        rows = straddle_rows[straddle_rows["arm_or_rule"] == rule]
        assert len(rows) > 0
        for col, value in marks.items():
            assert rows[col].notna().all() and (rows[col].to_numpy(dtype=float) == value).all()
    for mid, (move, crush) in literals.items():
        rows = frame[frame["model_id"] == mid]
        assert len(rows) > 0
        assert (rows["pred_move"].to_numpy(dtype=float) == move).all()
        assert (rows["pred_crush"].to_numpy(dtype=float) == crush).all()


def test_ladder_content_digest_sees_a_mid_array_change_in_a_large_array():
    """_content hashes ndarrays by value (pd.util.hash_array), not by repr: one mid-array change in a large array changes the digest."""
    a = np.linspace(0.0, 1.0, 5000)
    b = a.copy()
    b[2500] += 1e-12
    assert driver._content(a) != driver._content(b)
    assert driver._content(a) == driver._content(a.copy())

    s = np.array([f"s{i}" for i in range(3000)], dtype=object)
    t = s.copy()
    t[1500] = "changed"
    assert driver._content(s) != driver._content(t)
    assert driver._content(s) == driver._content(s.copy())
