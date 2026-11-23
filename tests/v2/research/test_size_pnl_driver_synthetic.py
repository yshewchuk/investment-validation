"""EXP-188 SY2 driver: run_synthetic on a tiny inner population. The SY1 generator and the synthetic scoring are stood in
for (the scoring needs the full simulator); the receipts, refusals, evaluation, publish and T11 are the real code."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import size_pnl_decisions as decisions, size_pnl_driver as driver, size_pnl_readers as readers
from experiments import size_pnl_synthetic as synthetic, size_pnl_tuning as tuning
from experiments.size_pnl_shortvol import Refusal

CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
SEAL = {"seal": "s"}
EVENTS = {"E1": "2023-06-10", "E2": "2024-08-07", "E3": "2025-01-15", "E4": "2025-02-12"}
PLAN = {"plan_id": "PLAN", "anchor": {"run_id": "REG"}, "horizons": [1, 5], "candidates": [], "selection": {"extra_refits": []}}
EXPECTED = {"plan_id": "PLAN", "registration_id": "REG", "seal_id": content_hash(SEAL), "panel_id": "P"}
OK = lambda: {"spec_hash": "SPEC", "spec_hash_checked": True}  # noqa: E731
SPEC = {"synthetic_arm": {"seed_tag": "sy2-test", "inner_years": [2023, 2024], "families": ["oracle", "median_noise", "shrinkage"],
                          "spread_ladder": [0.6], "shrinkage_L": [1.0], "level_gap_min_s": 0.4}}
def _avail(day):
    """A label's native available_as_of_date by default: one calendar day after its event date (ISO text)."""
    return (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()


BASE = pd.DataFrame([{"event_id": e, "pred_abs_move": 5.0, "pred_iv_crush_30": -12.0, "horizon_sessions": 1,
                      "abs_move": 4.0, "crush_pct_iv30": -10.0, "available_as_of_date": _avail(EVENTS[e])} for e in EVENTS])
ADMITTED = readers.Admitted({"events": pd.DataFrame({"event_id": list(EVENTS), "event_date": pd.to_datetime(list(EVENTS.values()))}),
                             "excluded": {}})
DATES = pd.to_datetime(list(EVENTS.values()))
PLAN_FRAME = pd.DataFrame({"event_id": list(EVENTS), "ticker": ["T1", "T2", "T1", "T2"], "event_date": DATES, "session": "AMC",
                           "entry_date": DATES - pd.Timedelta(days=1), "exit_date": DATES + pd.Timedelta(days=1)})
FACTS = pd.DataFrame({"pre_iv30": 40.0, "mcap_usd": 2e10, "implied_move": 5.0}, index=pd.Index(list(EVENTS), name="event_id"))


def sources(**over):
    def fetch(columns, ids):
        return BASE[BASE["event_id"].isin(list(ids))]

    base = dict(plan=PLAN, read=lambda slot: SEAL if slot == "driver-seal" else None, selection={"selections": []},
                pair_candidates={}, blend_ids={}, expected=EXPECTED, panel_id="P",
                admit=lambda ids: {"admitted": sorted(ids), "dropped": []}, admitted=ADMITTED, fetch=fetch,
                production_hash="H", production_table_hash="H", plan_frame=PLAN_FRAME, load_index=lambda keys: None,
                facts=lambda ids: FACTS.reindex(list(ids)), snapshot_id="snap-1")
    base.update(over)
    return driver.Sources(**base)


def chunk(*eids):
    return {"event_id": np.array(eids, dtype=object), "dte_exit": np.full(len(eids), 9, dtype=np.int32)}


def use_chunks(monkeypatch, pairs):
    def fake(admitted, plan, load_index):
        yield from pairs
    monkeypatch.setattr(driver.ladder, "ladder_chunks", fake)


def snapshot(out):
    """Every file under ``out`` with its bytes."""
    return {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


@pytest.fixture(autouse=True)
def bounded(monkeypatch):
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")


@pytest.fixture(autouse=True)
def preregistered_spec(monkeypatch):
    monkeypatch.setattr(driver.size_pnl_prereg, "load_spec", lambda spec_path=None: (SPEC, "SPEC"))


@pytest.fixture(autouse=True)
def structure_gate_passes(monkeypatch):
    monkeypatch.setattr(driver.shortvol, "score_events", lambda chunk_, events, sims, name, **kw: pd.DataFrame(
        {"event_id": events["event_id"].to_numpy(), "status": "scored", "reason": None}))


@pytest.fixture(autouse=True)
def equivalence_passes(monkeypatch):
    monkeypatch.setattr(driver, "_equivalence", lambda *args, **kwargs: {"checked": 1, "skipped": 0, "max_abs_diff": 0.0})


@pytest.fixture(autouse=True)
def secured_stubbed(monkeypatch):
    """The Amendment 2 secured-cash lookup needs a real ladder chunk; these tests stub the chunk."""
    monkeypatch.setattr(driver.shortvol, "secured", lambda chunk_, events, decisions, name, c=1.0: pd.DataFrame({"event_id": pd.Series([], dtype=object), "secured_per_contract": pd.Series([], dtype=float)}))


@pytest.fixture
def stubs(monkeypatch):
    """Pool, simulation and tuning stubbed, as in the inner-stage driver tests."""
    def fake_tune(items, mid, **kwargs):
        return {"stage": "tuning", "model_id": mid, "c": 1.0, "skipped": False, "events": sum(len(ev) for _c, ev, _s in items)}

    monkeypatch.setattr(driver, "build_pool", lambda table, labels: None)
    monkeypatch.setattr(tuning, "tune", fake_tune)
    monkeypatch.setattr(driver.shortvol, "simulate", lambda chunk_, events, pool, **kw: {e: "sim" for e in events["event_id"]})


def fake_entries(labels, arm):
    """Ready entries for each predictor of the arm, built on the labelled events (the SY1 shape)."""
    table = pd.DataFrame({"event_date": pd.to_datetime(labels["event_date"]).to_numpy(),
                          "pred_move": labels["abs_move"].to_numpy(dtype=float), "pred_crush": -10.0},
                         index=pd.Index(labels.index.astype(str), name="event_id"))
    out = {}
    for n, pred in enumerate(synthetic.predictors(arm), start=1):
        mid = synthetic.model_id(arm, pred)
        out[mid] = {"status": "ready", "source": "synthetic", "family": pred["family"], "params": pred["params"], "table": table,
                    "binding": content_hash({"model": mid}), "dropped": {}, "metrics": {"n": len(table), "rmse": float(n), "mae": 0.5 * n}}
    return out


def fake_scores(src, ready, plan, facts_all, labels):
    """One straddle row per (model, event), P&L offset by the model's order so the evaluation has a spread. Each row carries the
    model's pred_move: constant within each calendar quarter for the shrinkage model, arbitrary finite values for the others."""
    ids, dates = plan["event_id"].astype(str).to_numpy(), pd.to_datetime(plan["event_date"]).to_numpy()
    quarter_key = pd.DatetimeIndex(dates).year * 4 + pd.DatetimeIndex(dates).quarter
    shrink_move = 5.0 + 0.1 * (quarter_key - quarter_key.min()).to_numpy(dtype=float)
    frames = []
    for k, mid in enumerate(sorted(ready)):
        move = shrink_move if ready[mid]["family"] == "shrinkage" else 5.0 + np.arange(len(ids)) * 0.25
        frames.append(pd.DataFrame({"model_id": mid, "book": "straddle", "arm_or_rule": "R_L", "structure": None, "event_id": ids,
                                    "event_date": dates, "status": "taken", "reason": None,
                                    "pnl_dollars": 1.0 + k + np.arange(len(ids)) * 0.5, "pred_move": move}))
    return frames, {}, "sha256:fake", 0, {}


@pytest.fixture
def fakes(monkeypatch):
    """Stand-ins for the SY1 generator and the synthetic scoring. Returns the labelled ids the generator was given."""
    seen = []

    def generate(admitted, labels, production, arm):
        seen.append(sorted(labels.index.astype(str)))
        return fake_entries(labels, arm)

    monkeypatch.setattr(synthetic, "generate", generate)
    monkeypatch.setattr(driver, "_synthetic_scores", fake_scores)
    return seen


def test_refuses_without_the_bounded_run_before_any_write(tmp_path, monkeypatch, fakes):
    monkeypatch.delenv("BOUNDED_RUN_NESTED", raising=False)
    with pytest.raises(OpsError) as caught:
        driver.run_synthetic(sources(), tmp_path, prereg=OK)
    assert caught.value.code == "RESOURCE_UNAVAILABLE"
    assert caught.value.problem.details["reason"] == "not_bounded"
    assert not (tmp_path / "receipts").exists()


def test_a_non_bool_replay_refuses_before_any_read(tmp_path, monkeypatch, fakes):
    def no_read(*args, **kwargs):
        raise AssertionError("an invalid replay flag must refuse before any read")

    monkeypatch.setattr(readers, "read_production", no_read)
    with pytest.raises(OpsError) as caught:
        driver.run_synthetic(sources(), tmp_path, prereg=OK, replay="false")
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.details["reason"] == "replay_invalid"
    assert not (tmp_path / "receipts").exists()


def test_a_plan_event_that_is_not_admitted_refuses_before_any_read(tmp_path, monkeypatch, fakes):
    def no_read(*args, **kwargs):
        raise AssertionError("a plan event outside the admission must refuse before any read")

    monkeypatch.setattr(readers, "read_production", no_read)
    admitted = readers.Admitted({"events": pd.DataFrame({"event_id": ["E2"], "event_date": pd.to_datetime(["2024-08-07"])}), "excluded": {}})
    with pytest.raises(OpsError) as caught:
        driver.run_synthetic(sources(admitted=admitted), tmp_path, prereg=OK)
    assert caught.value.code == "HOLDOUT_ACCESS_DENIED"
    assert caught.value.problem.details["reason"] == "key_not_admitted"
    assert not (tmp_path / "receipts").exists()


def test_writes_the_synthetic_artifacts_and_no_tuning_or_outer_files(tmp_path, monkeypatch, fakes):
    def no_tune(*args, **kwargs):
        raise AssertionError("the synthetic stage must never tune")

    monkeypatch.setattr(tuning, "tune", no_tune)
    summary = driver.run_synthetic(sources(), tmp_path, prereg=OK)
    assert sorted(p.name for p in (tmp_path / "receipts").iterdir()) == ["inputs_synthetic.json", "synthetic.json"]
    assert (tmp_path / "decisions_synthetic.parquet").is_file() and not (tmp_path / "decisions.parquet").exists()
    text = (tmp_path / "t11_synthetic.md").read_text(encoding="utf-8")
    assert "T11 SYNTHETIC calibration, X-table" in text and "T11 SYNTHETIC slopes" in text
    assert "T11 SYNTHETIC means" in text and "T11 SYNTHETIC contrasts" in text
    assert "| X7 | replay | not_run |" in text
    assert summary["stage"] == "synthetic" and summary["models"] == sorted(summary["models"]) and len(summary["models"]) == 3


def test_a_failed_t11_publication_leaves_no_report_and_no_tmp_file(tmp_path, monkeypatch, fakes):
    real_replace = driver.os.replace

    def fail_t11(src, dst, *args, **kwargs):  # the T11 publication's rename fails; every other rename (the decisions publish) is real
        if str(dst).endswith("t11_synthetic.md"):
            raise OSError("rename refused")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(driver.os, "replace", fail_t11)
    with pytest.raises(OSError, match="rename refused"):
        driver.run_synthetic(sources(), tmp_path, prereg=OK)
    assert (tmp_path / "decisions_synthetic.parquet").is_file()  # the failure is at the T11 step, after the decisions publish
    assert not (tmp_path / "t11_synthetic.md").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".t11")]


def test_replay_is_byte_identical_and_passes_x7(tmp_path, fakes):
    driver.run_synthetic(sources(), tmp_path, prereg=OK, replay=True)
    first = snapshot(tmp_path)
    assert "| X7 | replay | pass |" in first["t11_synthetic.md"].decode("utf-8")
    driver.run_synthetic(sources(), tmp_path, prereg=OK, replay=True)
    assert snapshot(tmp_path) == first


def test_replay_then_plain_run_keeps_the_receipt_and_decisions_byte_identical(tmp_path, fakes):
    driver.run_synthetic(sources(), tmp_path, prereg=OK, replay=True)
    keep = ("receipts/synthetic.json", "decisions_synthetic.parquet")
    first = {name: (tmp_path / name).read_bytes() for name in keep}
    driver.run_synthetic(sources(), tmp_path, prereg=OK)
    assert {name: (tmp_path / name).read_bytes() for name in keep} == first


def test_replay_then_plain_run_keeps_t11_byte_identical_and_x7_pass(tmp_path, fakes):
    driver.run_synthetic(sources(), tmp_path, prereg=OK, replay=True)
    first = (tmp_path / "t11_synthetic.md").read_bytes()
    assert "| X7 | replay | pass |" in first.decode("utf-8")
    driver.run_synthetic(sources(), tmp_path, prereg=OK)
    assert (tmp_path / "t11_synthetic.md").read_bytes() == first


def test_a_replay_receipt_for_other_inputs_is_ignored(tmp_path, fakes):
    driver.run_synthetic(sources(), tmp_path, prereg=OK)
    tuning.write_receipt(tmp_path / "receipts" / "synthetic_replay.json",
                         {"stage": "synthetic_replay", "scope": None, "inputs": "sha256:other", "ladder": "sha256:other", "equal": True})
    driver.run_synthetic(sources(), tmp_path, prereg=OK)
    assert "| X7 | replay | not_run |" in (tmp_path / "t11_synthetic.md").read_text(encoding="utf-8")


def test_a_replay_receipt_with_other_frames_or_a_non_bool_verdict_is_ignored(tmp_path):
    (tmp_path / "receipts").mkdir()
    path = tmp_path / "receipts" / "synthetic_replay.json"
    body = {"stage": "synthetic_replay", "scope": None, "inputs": "sha256:in", "ladder": "sha256:l", "frames": "sha256:f", "declined": "sha256:d", "equal": True}
    tuning.write_receipt(path, body)
    assert driver._replay_equal(tmp_path, "sha256:in", "sha256:l", "sha256:f", "sha256:d") is True
    assert driver._replay_equal(tmp_path, "sha256:in", "sha256:l", "sha256:other", "sha256:d") is None  # frames differ
    path.unlink()
    tuning.write_receipt(path, {**body, "equal": "false"})
    assert driver._replay_equal(tmp_path, "sha256:in", "sha256:l", "sha256:f", "sha256:d") is None  # a string verdict is not a bool


def test_a_replay_whose_second_scoring_differs_fails_x7_and_stops(tmp_path, monkeypatch, fakes):
    calls = []

    def scores(*args):
        frames, skipped, ladder, dropped, declined = fake_scores(*args)
        calls.append(1)
        if len(calls) == 2:  # the replay's second scoring returns altered P&L
            frames = [f.assign(pnl_dollars=f["pnl_dollars"] + 1.0) for f in frames]
        return frames, skipped, ladder, dropped, declined

    monkeypatch.setattr(driver, "_synthetic_scores", scores)
    driver.run_synthetic(sources(), tmp_path, prereg=OK, replay=True)
    text = (tmp_path / "t11_synthetic.md").read_text(encoding="utf-8")
    assert "| X7 | replay | fail |" in text and "X7 replay" in text  # the stop list carries the X7 entry


def test_a_replay_whose_declined_counts_differ_fails_x7(tmp_path, monkeypatch, fakes):
    calls = []

    def scores(*args):
        frames, skipped, ladder, dropped, _declined = fake_scores(*args)
        declined = {mid: {name: [0, 1] for name in driver.shortvol.NAMES} for mid in args[1]}  # per model and unit: [declined, candidates]
        calls.append(1)
        if len(calls) == 2:  # the replay's second scoring returns the same frames with one more decline
            declined = copy.deepcopy(declined)
            declined[sorted(args[1])[0]][sorted(driver.shortvol.NAMES)[0]][0] += 1
        return frames, skipped, ladder, dropped, declined

    monkeypatch.setattr(driver, "_synthetic_scores", scores)
    driver.run_synthetic(sources(), tmp_path, prereg=OK, replay=True)
    assert "| X7 | replay | fail |" in (tmp_path / "t11_synthetic.md").read_text(encoding="utf-8")


def test_a_spec_that_differs_from_the_preregistered_one_refuses_before_any_write(tmp_path, fakes):
    with pytest.raises(Refusal) as caught:
        driver.run_synthetic(sources(), tmp_path, prereg=lambda: {"spec_hash": "OTHER", "spec_hash_checked": True})
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "spec_not_preregistered"
    assert not any(tmp_path.iterdir())


def test_changed_inputs_refuse_and_leave_the_receipts_as_they_were(tmp_path, fakes):
    driver.run_synthetic(sources(), tmp_path, prereg=OK)
    before = snapshot(tmp_path)
    with pytest.raises(Refusal) as caught:
        driver.run_synthetic(sources(snapshot_id="snap-2"), tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inputs_receipt_changed"
    assert snapshot(tmp_path) == before


def test_a_changed_crush_table_hash_changes_the_synthetic_inputs_receipt_and_refuses_a_replay(tmp_path, fakes):
    driver.run_synthetic(sources(), tmp_path / "a", prereg=OK)
    driver.run_synthetic(sources(crush_table_hash="OTHER-CRUSH"), tmp_path / "b", prereg=OK)
    first = (tmp_path / "a" / "receipts" / "inputs_synthetic.json").read_bytes()
    assert first != (tmp_path / "b" / "receipts" / "inputs_synthetic.json").read_bytes()
    with pytest.raises(Refusal) as caught:
        driver.run_synthetic(sources(crush_table_hash="OTHER-CRUSH"), tmp_path / "a", prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inputs_receipt_changed"
    assert (tmp_path / "a" / "receipts" / "inputs_synthetic.json").read_bytes() == first


def test_the_stage_reads_no_2025_event(tmp_path, monkeypatch, fakes):
    ends, real_prod, real_labels = [], readers.read_production, readers.read_labels

    def spy(real):
        def wrapped(*args, **kwargs):
            ends.append(kwargs.get("end"))
            return real(*args, **kwargs)
        return wrapped

    monkeypatch.setattr(readers, "read_production", spy(real_prod))
    monkeypatch.setattr(readers, "read_labels", spy(real_labels))
    driver.run_synthetic(sources(), tmp_path, prereg=OK)
    assert ends and all(pd.Timestamp(e) <= pd.Timestamp("2025-01-01") for e in ends)
    assert fakes and all(set(ids) <= {"E1", "E2"} for ids in fakes)


def test_deleting_the_synthetic_files_leaves_the_inner_receipts_byte_identical(tmp_path, monkeypatch, stubs, fakes):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    out = tmp_path / "out"
    driver.run(sources(), "inner", out, prereg=OK)
    inner = {p.name: p.read_bytes() for p in (out / "receipts").iterdir()}
    driver.run_synthetic(sources(), out, prereg=OK)
    for path in (out / "decisions_synthetic.parquet", out / "t11_synthetic.md",
                 out / "receipts" / "synthetic.json", out / "receipts" / "inputs_synthetic.json"):
        path.unlink()
    driver.run(sources(), "inner", out, prereg=OK)
    assert {p.name: p.read_bytes() for p in (out / "receipts").iterdir()} == inner


def test_month_scores_and_month_take_arms_and_emit_from(monkeypatch):
    """The arms= and emit_from= keywords of _month_scores and _month: only the as-is arm is scored, and no rows emit before emit_from."""
    monkeypatch.setattr(driver, "build_pool", lambda table, labels: None)
    monkeypatch.setattr(driver, "_printed", lambda labels, cutoff: labels)
    monkeypatch.setattr(driver.shortvol, "simulate", lambda chunk_, events, pool, **kw: {e: "sim" for e in events["event_id"]})
    monkeypatch.setattr(driver.shortvol, "score_events", lambda chunk_, events, sims, name, **kw: pd.DataFrame(
        {"event_id": events["event_id"].to_numpy(), "status": "scored", "reason": None}))
    monkeypatch.setattr(driver.straddle, "decide", lambda events, edges, rule, chunk_: pd.DataFrame(
        {"event_date": events["event_date"].to_numpy(), "edge": np.ones(len(events))}))
    events = pd.DataFrame({"event_id": ["E1", "E2"], "event_date": pd.to_datetime(["2023-06-10", "2023-06-20"])})
    table = pd.DataFrame({"event_date": pd.to_datetime(["2023-01-05"])}, index=pd.Index(["H0"]))
    start, st = pd.Timestamp("2023-06-01"), driver._model_state(1.0)
    scored, missing = driver._month_scores("m", table, events, chunk("E1", "E2"), None, start, st, arms=(("as_is", 1.0),))
    assert set(scored) == {("as_is", name) for name in driver.shortvol.NAMES} and missing == set()
    frames = driver._month("m", events, scored, set(), chunk("E1", "E2"), start, st, arms=(("as_is", 1.0),),
                           emit_from=pd.Timestamp("2099-01-01"))
    assert frames == [] and st["scored"]["retuned"] == []
    assert len(st["scored"]["as_is"]) == len(driver.shortvol.NAMES) and len(st["edges"]) == 2


def test_publish_defaults_keep_the_outer_names_and_the_stem_renames_both(tmp_path):
    frame = pd.DataFrame({"model_id": ["m"], "book": ["straddle"], "arm_or_rule": ["R_L"], "structure": [None], "event_id": ["E1"],
                          "event_date": pd.to_datetime(["2023-06-10"]), "status": ["taken"], "reason": [None], "pnl_dollars": [1.0]})
    (tmp_path / "a").mkdir()
    decisions.publish(tmp_path / "a", [frame], {"stage": "outer"})
    assert (tmp_path / "a" / "decisions.parquet").is_file() and (tmp_path / "a" / "receipts" / "outer.json").is_file()
    (tmp_path / "b").mkdir()
    decisions.publish(tmp_path / "b", [frame], {"stage": "x"}, stem="decisions_synthetic", receipt="synthetic")
    assert (tmp_path / "b" / "decisions_synthetic.parquet").is_file() and (tmp_path / "b" / "receipts" / "synthetic.json").is_file()
    with pytest.raises(Refusal) as caught:
        decisions.publish(tmp_path / "b", [frame], {"stage": "y"}, stem="decisions_synthetic", receipt="synthetic")
    assert caught.value.details["reason"] == "synthetic_receipt_changed"


def test_run_synthetic_refuses_an_expected_that_differs_from_the_plan_identity(tmp_path, fakes):
    with pytest.raises(OpsError) as caught:
        driver.run_synthetic(sources(expected={**EXPECTED, "panel_id": "OTHER"}), tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert not (tmp_path / "receipts").exists()


def test_an_empty_scored_population_refuses_before_any_decision(tmp_path, monkeypatch, fakes):
    def all_undetermined(src, ready, plan, facts_all, labels):
        frames, skipped, ladder, dropped, declined = fake_scores(src, ready, plan, facts_all, labels)
        return [f.assign(status="undetermined") for f in frames], skipped, ladder, dropped, declined

    monkeypatch.setattr(driver, "_synthetic_scores", all_undetermined)
    with pytest.raises(Refusal) as caught:
        driver.run_synthetic(sources(), tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT and caught.value.details["reason"] == "empty_population"
    assert not (tmp_path / "receipts" / "inputs_synthetic.json").exists()
    assert not (tmp_path / "decisions_synthetic.parquet").exists()


def test_a_refused_smoke_run_pins_nothing_and_a_broader_run_in_the_same_dir_succeeds(tmp_path, monkeypatch, fakes):
    def all_undetermined(src, ready, plan, facts_all, labels):
        frames, skipped, ladder, dropped, declined = fake_scores(src, ready, plan, facts_all, labels)
        return [f.assign(status="undetermined") for f in frames], skipped, ladder, dropped, declined

    monkeypatch.setattr(driver, "_synthetic_scores", all_undetermined)
    with pytest.raises(Refusal) as caught:
        driver.run_synthetic(sources(), tmp_path, prereg=OK, event_ids=["E1"])
    assert caught.value.details["reason"] == "empty_population"
    assert not (tmp_path / "receipts" / "inputs_synthetic.json").exists() and not (tmp_path / "decisions_synthetic.parquet").exists()
    monkeypatch.setattr(driver, "_synthetic_scores", fake_scores)
    summary = driver.run_synthetic(sources(), tmp_path, prereg=OK)
    assert (tmp_path / "receipts" / "inputs_synthetic.json").is_file() and (tmp_path / "decisions_synthetic.parquet").is_file()
    assert summary["stage"] == "synthetic"


def test_one_decided_quarter_refuses_as_underclustered(tmp_path, monkeypatch, fakes):
    def one_event_decided(src, ready, plan, facts_all, labels):
        frames, skipped, ladder, dropped, declined = fake_scores(src, ready, plan, facts_all, labels)
        return [f.assign(status=["taken" if e == "E1" else "undetermined" for e in f["event_id"]]) for f in frames], skipped, ladder, dropped, declined

    monkeypatch.setattr(driver, "_synthetic_scores", one_event_decided)
    with pytest.raises(Refusal) as caught:
        driver.run_synthetic(sources(), tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT and caught.value.details["reason"] == "underclustered_population"
    assert not (tmp_path / "receipts" / "inputs_synthetic.json").exists() and not (tmp_path / "decisions_synthetic.parquet").exists()


def test_one_ticker_refuses_as_underclustered(tmp_path, fakes):
    with pytest.raises(Refusal) as caught:
        driver.run_synthetic(sources(plan_frame=PLAN_FRAME.assign(ticker="T1")), tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT and caught.value.details["reason"] == "underclustered_population"
    assert not (tmp_path / "receipts" / "inputs_synthetic.json").exists() and not (tmp_path / "decisions_synthetic.parquet").exists()


# Six inner events in five quarters and two tickers. E1 (the shrinkage model's first-quarter gap) and E3 (unscoreable for one model)
# leave for every model, so E2, E4, E5 and E6 survive: four events in four quarters and both tickers for the evaluation.
EVENTS3 = {"E1": "2023-06-10", "E2": "2023-09-12", "E3": "2024-08-07", "E4": "2023-12-06", "E5": "2024-02-14", "E6": "2024-11-13"}
TICKERS3 = {"E1": "T1", "E2": "T2", "E3": "T1", "E4": "T1", "E5": "T2", "E6": "T1"}
DATES3 = pd.to_datetime(list(EVENTS3.values()))
ADMITTED3 = readers.Admitted({"events": pd.DataFrame({"event_id": list(EVENTS3), "event_date": DATES3}), "excluded": {}})
PLAN3 = pd.DataFrame({"event_id": list(EVENTS3), "ticker": [TICKERS3[e] for e in EVENTS3], "event_date": DATES3, "session": "AMC",
                      "entry_date": DATES3 - pd.Timedelta(days=1), "exit_date": DATES3 + pd.Timedelta(days=1)})
BASE3 = pd.DataFrame([{"event_id": e, "pred_abs_move": 5.0, "pred_iv_crush_30": -12.0, "horizon_sessions": 1,
                       "abs_move": 4.0, "crush_pct_iv30": -10.0, "available_as_of_date": _avail(EVENTS3[e])} for e in EVENTS3])
FACTS3 = pd.DataFrame({"pre_iv30": 40.0, "mcap_usd": 2e10, "implied_move": 5.0}, index=pd.Index(list(EVENTS3), name="event_id"))


def _scored(chunk_, events, sims, name, **kwargs):
    """Every event scored for the structure; the real decide then sets the status (thin history stays undetermined)."""
    n = len(events)
    return pd.DataFrame({"event_id": events["event_id"].astype(str).to_numpy(), "event_date": events["event_date"].to_numpy(),
                         "structure": name, "status": "scored", "reason": None, "exp_pnl_sim": 1.0 + np.arange(n),
                         "mcap_usd": 2e10, "rel_spread": 0.1, "width": 1.0})


def _marks(chunk_, events, decisions, *args, **kwargs):
    """The P&L marks of the taken rows (the real scorers need the chunk's quotes)."""
    taken = decisions[decisions["status"] == "taken"]
    n = len(taken)
    return pd.DataFrame({"event_id": taken["event_id"].astype(str).to_numpy(), "exit_value": np.ones(n), "rung": np.ones(n, dtype=int),
                         "lots": np.ones(n, dtype=int), "entry_cost": np.ones(n), "pnl_per_share": np.ones(n),
                         "pnl_dollars": 10.0 * np.array([int(e[1:]) for e in taken["event_id"].astype(str)], dtype=float)})  # 10 x the event number


def _straddle_decide(events, edges, rule, chunk_):
    """Every straddle event taken (the real rules need the chunk's quotes)."""
    return pd.DataFrame({"event_id": events["event_id"].astype(str).to_numpy(), "event_date": events["event_date"].to_numpy(),
                         "edge": events["pred_abs_move"].to_numpy(dtype=float), "cutoff": np.nan, "rel_spread": np.nan,
                         "status": "taken", "reason": None})


def test_the_real_scoring_loop_drops_an_event_unscoreable_for_one_model_from_every_model(tmp_path, monkeypatch):
    """No _synthetic_scores stub: the real month loop runs. The shrinkage model has no E1 (its first quarter), so E1 leaves the
    common population; one non-shrinkage model has no simulation for E3, so E3 leaves for every model. MIN_WINDOW is 1 here, since
    the production 100 leaves every short-vol row thin on six events. Then E2 (no prior history) and E6 (no prior row in its six-month
    window) are undetermined under every short-vol unit: published as evidence, but outside E. E4 and E5 are determined and form E.
    Predictions are 5.0 except E3 (104.0) and the realised abs_move is 4.0, so on E = {E4, E5} rmse = mae = 1.0.
    Each straddle P&L is 10 x its event number (E2 20, E4 40, E5 50, E6 60)."""
    monkeypatch.setattr(driver.shortvol, "MIN_WINDOW", 1)
    def generate(admitted, labels, production, arm):
        entries = fake_entries(labels, arm)
        shrink = {mid for mid in entries if entries[mid]["family"] == "shrinkage"}
        odd = sorted(mid for mid in entries if mid not in shrink)[0]
        out = {}
        for mid, entry in entries.items():
            table = entry["table"].drop(index=["E1"]) if mid in shrink else entry["table"]
            table = table.assign(pred_move=np.where(table.index == "E3", 104.0, 5.0))
            if mid == odd: table.attrs["odd"] = True  # the pool keeps the table's attrs through the month slices
            out[mid] = {**entry, "table": table}
        return out

    monkeypatch.setattr(synthetic, "generate", generate)
    monkeypatch.setattr(driver, "build_pool", lambda table, labels: table)
    monkeypatch.setattr(driver.shortvol, "simulate", lambda chunk_, events, pool, **kw: {
        e: (None if pool.attrs.get("odd") and e == "E3" else "sim") for e in events["event_id"]})
    monkeypatch.setattr(driver.shortvol, "score_events", _scored)
    monkeypatch.setattr(driver.shortvol, "score", _marks)
    monkeypatch.setattr(driver.straddle, "decide", _straddle_decide)
    monkeypatch.setattr(driver.straddle, "score", _marks)
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2023Q3", chunk("E2")), ("2023Q4", chunk("E4")), ("2024Q1", chunk("E5")),
                             ("2024Q3", chunk("E3")), ("2024Q4", chunk("E6"))])
    src = sources(plan_frame=PLAN3, admitted=ADMITTED3, facts=lambda ids: FACTS3.reindex(list(ids)),
                  fetch=lambda columns, ids: BASE3[BASE3["event_id"].isin(list(ids))])
    out = tmp_path / "out"
    summary = driver.run_synthetic(src, out, prereg=OK)
    frame = pd.read_parquet(out / "decisions_synthetic.parquet")
    units = len(driver.shortvol.NAMES) + len(driver.straddle.RULES)
    assert sorted(frame["event_id"].astype(str).unique()) == ["E2", "E4", "E5", "E6"]  # E1 leaves with the shrinkage gap, E3 for every model
    assert len(set(frame["model_id"])) == 3
    assert len(frame) == 3 * 4 * units  # one row per model per surviving event per unit
    assert not frame.duplicated(["model_id", "book", "arm_or_rule", "structure", "event_id"]).any()
    taken = frame["status"] == "taken"  # the P&L oracle: a taken row carries 10 x its event's number, every other row carries no P&L
    assert taken.any()
    expected_pnl = {"E2": 20.0, "E4": 40.0, "E5": 50.0, "E6": 60.0}
    assert (frame.loc[taken, "pnl_dollars"] == frame.loc[taken, "event_id"].astype(str).map(expected_pnl)).all()
    assert frame.loc[~taken, "pnl_dollars"].isna().all()
    receipt = json.loads((out / "receipts" / "synthetic.json").read_text(encoding="utf-8"))
    assert receipt["scoreability_dropped"] == 1
    assert receipt["common_dropped"] == 1 and receipt["events"] == 2 and receipt["events_undetermined"] == 2  # E2, E6 leave E
    assert sorted(receipt["events_by_model"].values()) == [2, 2, 2]  # E4 and E5 in every model's table
    assert receipt["metrics"] and all(m["n"] == 2 and m["rmse"] == 1.0 and m["mae"] == 1.0 for m in receipt["metrics"].values())
    inputs = json.loads((out / "receipts" / "inputs_synthetic.json").read_text(encoding="utf-8"))
    assert receipt["models"] == sorted(receipt["models"]) and len(receipt["models"]) == 3
    assert receipt["scope"] == inputs["scope"] and receipt["inputs"] == inputs["content_hash"]
    assert receipt["arm"] == content_hash(SPEC["synthetic_arm"]) and receipt["planned"] == len(PLAN3)
    assert receipt["events_no_prediction"] == 0 and receipt["evaluation"].startswith("sha256:")
    assert receipt["stop"] == summary["stop"]
    first = snapshot(out)
    driver.run_synthetic(src, out, prereg=OK)
    assert snapshot(out) == first  # a second identical run is byte-identical


def test_a_planned_event_in_no_model_table_is_counted_as_no_prediction(tmp_path, fakes):
    """E9 is planned in the inner window but no model table holds it: it is counted in events_no_prediction, and planned is the plan size."""
    plan = pd.concat([PLAN_FRAME, pd.DataFrame({"event_id": ["E9"], "ticker": ["T2"], "event_date": pd.to_datetime(["2024-03-05"]), "session": ["AMC"],
                                               "entry_date": pd.to_datetime(["2024-03-04"]), "exit_date": pd.to_datetime(["2024-03-06"])})], ignore_index=True)
    admitted = readers.Admitted({"events": pd.DataFrame({"event_id": [*EVENTS, "E9"], "event_date": pd.to_datetime([*EVENTS.values(), "2024-03-05"])}), "excluded": {}})
    driver.run_synthetic(sources(plan_frame=plan, admitted=admitted), tmp_path, prereg=OK)
    receipt = json.loads((tmp_path / "receipts" / "synthetic.json").read_text(encoding="utf-8"))
    assert receipt["events_no_prediction"] == 1 and receipt["planned"] == 3  # E1, E2 and E9 are planned in the inner window


def test_reads_take_only_the_planned_events_and_an_unplanned_admitted_event_is_never_requested(tmp_path, monkeypatch, fakes):
    """E5 is admitted and dated inside the inner window but is not in the plan. Neither reader is asked for it, and the generator
    sees only the planned labels, so E5's label cannot reach any other event's shrinkage forecast."""
    admitted = readers.Admitted({"events": pd.DataFrame({"event_id": [*EVENTS, "E5"], "event_date": pd.to_datetime([*EVENTS.values(), "2024-05-01"])}), "excluded": {}})
    asked = []
    real_prod, real_labels = readers.read_production, readers.read_labels

    def spy(real):
        def wrapped(*args, **kwargs):
            asked.append(kwargs.get("event_ids"))
            return real(*args, **kwargs)
        return wrapped

    monkeypatch.setattr(readers, "read_production", spy(real_prod))
    monkeypatch.setattr(readers, "read_labels", spy(real_labels))
    driver.run_synthetic(sources(admitted=admitted), tmp_path, prereg=OK)
    assert asked == [{"E1", "E2"}, {"E1", "E2"}]  # production, then labels: the plan's ids only
    assert fakes == [["E1", "E2"]]
    receipt = json.loads((tmp_path / "receipts" / "synthetic.json").read_text(encoding="utf-8"))
    assert receipt["declined"] == {} and receipt["planned"] == 2


def test_the_real_loop_drops_a_planned_event_one_model_lacks_from_every_model(monkeypatch):
    """_pool_tables is replaced so model m2's table has no E3 while m1's does. The real month loop must still complete, E3 must be
    absent from every model's frames, and scoreability_dropped must count it (E3 is the only dropped id)."""
    monkeypatch.setattr(driver.shortvol, "MIN_WINDOW", 1)
    full = pd.DataFrame({"event_date": DATES3, "pred_move": 5.0, "pred_crush": -10.0}, index=pd.Index(list(EVENTS3), name="event_id"))
    short = full.drop(index=["E3"])
    ready = {mid: {"status": "ready", "family": "oracle", "params": {}, "table": full} for mid in ("m1", "m2")}
    monkeypatch.setattr(driver, "_pool_tables", lambda ready_: {mid: (short if mid == "m2" else full) for mid in ready_})
    labels = pd.DataFrame(
        {"abs_move": 4.0, "crush_pct_iv30": -10.0, "available": [_avail(EVENTS3[e]) for e in EVENTS3]},
        index=pd.Index(list(EVENTS3), name="event_id"),
    )
    monkeypatch.setattr(driver, "build_pool", lambda table, labels: table)
    monkeypatch.setattr(driver.shortvol, "simulate", lambda chunk_, events, pool, **kw: {e: "sim" for e in events["event_id"]})
    monkeypatch.setattr(driver.shortvol, "score_events", _scored)
    monkeypatch.setattr(driver.shortvol, "score", _marks)
    monkeypatch.setattr(driver.straddle, "decide", _straddle_decide)
    monkeypatch.setattr(driver.straddle, "score", _marks)
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2023Q3", chunk("E2")), ("2023Q4", chunk("E4")), ("2024Q1", chunk("E5")),
                             ("2024Q3", chunk("E3")), ("2024Q4", chunk("E6"))])
    src = sources(plan_frame=PLAN3, admitted=ADMITTED3)
    frames, _skipped, _digest, dropped, _declined = driver._synthetic_scores(src, ready, PLAN3, FACTS3, labels)
    assert dropped == 1
    seen = pd.concat(frames, ignore_index=True)
    assert set(seen["model_id"]) == {"m1", "m2"}
    assert "E3" not in set(seen["event_id"].astype(str))  # absent for every model, m1 included


def test_the_real_loop_counts_width_declined_rows_per_model_and_unit(monkeypatch):
    """Same setup as the test above, but shortvol.score_events declines rows: for NAMES[0] E4 and E6 are declined with reason
    width_declined; for NAMES[1] E5 is declined with reason other_reason, which must not be counted. Hand count: m1 sees E1..E6 (6
    candidates), m2 sees E1, E2, E4, E5, E6 (5 candidates, E3 absent). So NAMES[0] is [2, 6] for m1 and [2, 5] for m2, NAMES[1] and
    every other unit is [0, 6] for m1 and [0, 5] for m2."""
    monkeypatch.setattr(driver.shortvol, "MIN_WINDOW", 1)
    full = pd.DataFrame({"event_date": DATES3, "pred_move": 5.0, "pred_crush": -10.0}, index=pd.Index(list(EVENTS3), name="event_id"))
    short = full.drop(index=["E3"])
    ready = {mid: {"status": "ready", "family": "oracle", "params": {}, "table": full} for mid in ("m1", "m2")}
    monkeypatch.setattr(driver, "_pool_tables", lambda ready_: {mid: (short if mid == "m2" else full) for mid in ready_})
    labels = pd.DataFrame(
        {"abs_move": 4.0, "crush_pct_iv30": -10.0, "available": [_avail(EVENTS3[e]) for e in EVENTS3]},
        index=pd.Index(list(EVENTS3), name="event_id"),
    )
    monkeypatch.setattr(driver, "build_pool", lambda table, labels: table)
    monkeypatch.setattr(driver.shortvol, "simulate", lambda chunk_, events, pool, **kw: {e: "sim" for e in events["event_id"]})

    def declining(chunk_, events, sims, name, **kwargs):
        out = _scored(chunk_, events, sims, name, **kwargs)
        if name == driver.shortvol.NAMES[0]:
            hit = out["event_id"].astype(str).isin(["E4", "E6"]).to_numpy()
            out.loc[hit, "status"] = "declined"
            out.loc[hit, "reason"] = "width_declined"
        elif name == driver.shortvol.NAMES[1]:
            hit = (out["event_id"].astype(str) == "E5").to_numpy()
            out.loc[hit, "status"] = "declined"
            out.loc[hit, "reason"] = "other_reason"
        return out

    monkeypatch.setattr(driver.shortvol, "score_events", declining)
    monkeypatch.setattr(driver.shortvol, "score", _marks)
    monkeypatch.setattr(driver.straddle, "decide", _straddle_decide)
    monkeypatch.setattr(driver.straddle, "score", _marks)
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2023Q3", chunk("E2")), ("2023Q4", chunk("E4")), ("2024Q1", chunk("E5")),
                             ("2024Q3", chunk("E3")), ("2024Q4", chunk("E6"))])
    src = sources(plan_frame=PLAN3, admitted=ADMITTED3)
    *_, declined = driver._synthetic_scores(src, ready, PLAN3, FACTS3, labels)
    assert declined["m1"][driver.shortvol.NAMES[0]] == [2, 6]
    assert declined["m2"][driver.shortvol.NAMES[0]] == [2, 5]
    assert declined["m1"][driver.shortvol.NAMES[1]] == [0, 6]
    assert declined["m2"][driver.shortvol.NAMES[1]] == [0, 5]
    for name in driver.shortvol.NAMES[2:]:
        assert declined["m1"][name] == [0, 6] and declined["m2"][name] == [0, 5]


def _real_month_loop(monkeypatch):
    """The real synthetic month loop on the six inner events, with the shortvol and straddle scorers stood in for as in the tests above."""
    monkeypatch.setattr(driver.shortvol, "MIN_WINDOW", 1)
    monkeypatch.setattr(synthetic, "generate", lambda admitted, labels, production, arm: fake_entries(labels, arm))
    monkeypatch.setattr(driver, "build_pool", lambda table, labels: table)
    monkeypatch.setattr(driver.shortvol, "simulate", lambda chunk_, events, pool, **kw: {e: "sim" for e in events["event_id"]})
    monkeypatch.setattr(driver.shortvol, "score_events", _scored)
    monkeypatch.setattr(driver.shortvol, "score", _marks)
    monkeypatch.setattr(driver.straddle, "decide", _straddle_decide)
    monkeypatch.setattr(driver.straddle, "score", _marks)
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2023Q3", chunk("E2")), ("2023Q4", chunk("E4")), ("2024Q1", chunk("E5")),
                             ("2024Q3", chunk("E3")), ("2024Q4", chunk("E6"))])


def test_a_synthetic_event_with_a_missing_implied_move_leaves_every_model_table_and_the_decisions(tmp_path, monkeypatch):
    """E2 has finite pre_iv30 and mcap_usd but a NaN implied_move. It is dropped before scoring: absent from every decisions_synthetic
    row, from every table binding (the receipt's binding is recomputed for each model, so none is the unrestricted one), and recorded
    as the only id under facts_missing. Per the receipt, E2 is in no model table: events_no_prediction is 1 and common_dropped is 0."""
    _real_month_loop(monkeypatch)
    facts = FACTS3.assign(implied_move=[np.nan if e == "E2" else 5.0 for e in FACTS3.index])
    src = sources(plan_frame=PLAN3, admitted=ADMITTED3, facts=lambda ids: facts.reindex(list(ids)),
                  fetch=lambda columns, ids: BASE3[BASE3["event_id"].isin(list(ids))])
    out = tmp_path / "out"
    driver.run_synthetic(src, out, prereg=OK)
    frame = pd.read_parquet(out / "decisions_synthetic.parquet")
    assert "E2" not in set(frame["event_id"].astype(str))
    assert src.exclusions["facts_missing"] == {"E2"}
    inputs = json.loads((out / "receipts" / "inputs_synthetic.json").read_text(encoding="utf-8"))
    assert len(inputs["tables"]) == 3
    assert all(binding != content_hash({"model": mid}) for mid, binding in inputs["tables"].items())
    receipt = json.loads((out / "receipts" / "synthetic.json").read_text(encoding="utf-8"))
    assert receipt["planned"] == 6 and receipt["events_no_prediction"] == 1 and receipt["common_dropped"] == 0


def test_a_synthetic_run_whose_every_event_has_a_missing_fact_refuses_empty_population(tmp_path, monkeypatch):
    """Every event has a NaN implied_move, so every model table is empty after the facts exclusion. The run refuses empty_population
    before any inputs receipt or decision is written, and nothing earlier in the run crashes on the empty tables."""
    _real_month_loop(monkeypatch)
    facts = FACTS3.assign(implied_move=np.nan)
    src = sources(plan_frame=PLAN3, admitted=ADMITTED3, facts=lambda ids: facts.reindex(list(ids)),
                  fetch=lambda columns, ids: BASE3[BASE3["event_id"].isin(list(ids))])
    with pytest.raises(Refusal) as caught:
        driver.run_synthetic(src, tmp_path / "out", prereg=OK)
    assert caught.value.code == CONFLICT and caught.value.details["reason"] == "empty_population"
    assert src.exclusions["facts_missing"] == set(EVENTS3)
    assert not (tmp_path / "out" / "receipts" / "inputs_synthetic.json").exists()
    assert not (tmp_path / "out" / "decisions_synthetic.parquet").exists()


def test_a_synthetic_event_without_a_forecast_is_never_requested_from_facts(tmp_path, monkeypatch):
    """E3 has no production forecast (its fetch rows are withheld) and an invalid mcap_usd of 5000. Facts are requested only for the
    events that have a forecast, so the run completes, E3 is in no table and no decision, and facts never receive its id."""
    _real_month_loop(monkeypatch)
    requested = []
    facts = FACTS3.assign(mcap_usd=[5000.0 if e == "E3" else 2e10 for e in FACTS3.index])

    def fact_source(ids):
        requested.append(sorted(ids))
        return facts.reindex(list(ids))

    src = sources(plan_frame=PLAN3, admitted=ADMITTED3, facts=fact_source,
                  fetch=lambda columns, ids: BASE3[BASE3["event_id"].isin(list(ids)) & (BASE3["event_id"] != "E3")])
    result = driver.run_synthetic(src, tmp_path / "out", prereg=OK)
    assert result["stage"] == "synthetic"
    assert requested == [sorted(set(EVENTS3) - {"E3"})]
    assert "E3" not in set(pd.read_parquet(tmp_path / "out" / "decisions_synthetic.parquet")["event_id"].astype(str))
    receipt = json.loads((tmp_path / "out" / "receipts" / "synthetic.json").read_text(encoding="utf-8"))
    assert receipt["planned"] == 6 and receipt["events_no_prediction"] == 1
