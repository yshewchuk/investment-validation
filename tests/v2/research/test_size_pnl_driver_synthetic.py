"""EXP-188 SY2 driver: run_synthetic on a tiny inner population. The SY1 generator and the synthetic scoring are stood in
for (the scoring needs the full simulator); the receipts, refusals, evaluation, publish and T11 are the real code."""
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
BASE = pd.DataFrame([{"event_id": e, "pred_abs_move": 5.0, "pred_iv_crush_30": -12.0, "horizon_sessions": 1,
                      "abs_move": 4.0, "crush_pct_iv30": -10.0} for e in EVENTS])
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
def structure_gate_passes(monkeypatch):
    monkeypatch.setattr(driver.shortvol, "score_events", lambda chunk_, events, sims, name, **kw: pd.DataFrame(
        {"event_id": events["event_id"].to_numpy(), "status": "scored"}))


@pytest.fixture(autouse=True)
def equivalence_passes(monkeypatch):
    monkeypatch.setattr(driver, "_equivalence", lambda *args, **kwargs: {"checked": 1, "skipped": 0, "max_abs_diff": 0.0})


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
    """One straddle row per (model, event), P&L offset by the model's order so the evaluation has a spread."""
    ids, dates = plan["event_id"].astype(str).to_numpy(), pd.to_datetime(plan["event_date"]).to_numpy()
    frames = [pd.DataFrame({"model_id": mid, "book": "straddle", "arm_or_rule": "R_L", "structure": None, "event_id": ids,
                            "event_date": dates, "status": "taken", "reason": None,
                            "pnl_dollars": 1.0 + k + np.arange(len(ids)) * 0.5}) for k, mid in enumerate(sorted(ready))]
    return frames, {}, "sha256:fake", 0


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
        driver.run_synthetic(sources(), tmp_path, spec=SPEC, prereg=OK)
    assert caught.value.code == "RESOURCE_UNAVAILABLE"
    assert caught.value.problem.details["reason"] == "not_bounded"
    assert not (tmp_path / "receipts").exists()


def test_writes_the_synthetic_artifacts_and_no_tuning_or_outer_files(tmp_path, monkeypatch, fakes):
    def no_tune(*args, **kwargs):
        raise AssertionError("the synthetic stage must never tune")

    monkeypatch.setattr(tuning, "tune", no_tune)
    summary = driver.run_synthetic(sources(), tmp_path, spec=SPEC, prereg=OK)
    assert sorted(p.name for p in (tmp_path / "receipts").iterdir()) == ["inputs_synthetic.json", "synthetic.json"]
    assert (tmp_path / "decisions_synthetic.parquet").is_file() and not (tmp_path / "decisions.parquet").exists()
    text = (tmp_path / "t11_synthetic.md").read_text(encoding="utf-8")
    assert "T11 SYNTHETIC calibration, X-table" in text and "T11 SYNTHETIC slopes" in text
    assert "| X7 | replay | not_run |" in text
    assert summary["stage"] == "synthetic" and summary["models"] == sorted(summary["models"]) and len(summary["models"]) == 3


def test_replay_is_byte_identical_and_passes_x7(tmp_path, fakes):
    driver.run_synthetic(sources(), tmp_path, spec=SPEC, prereg=OK, replay=True)
    first = snapshot(tmp_path)
    assert "| X7 | replay | pass |" in first["t11_synthetic.md"].decode("utf-8")
    driver.run_synthetic(sources(), tmp_path, spec=SPEC, prereg=OK, replay=True)
    assert snapshot(tmp_path) == first


def test_replay_then_plain_run_keeps_the_receipt_and_decisions_byte_identical(tmp_path, fakes):
    driver.run_synthetic(sources(), tmp_path, spec=SPEC, prereg=OK, replay=True)
    keep = ("receipts/synthetic.json", "decisions_synthetic.parquet")
    first = {name: (tmp_path / name).read_bytes() for name in keep}
    driver.run_synthetic(sources(), tmp_path, spec=SPEC, prereg=OK)
    assert {name: (tmp_path / name).read_bytes() for name in keep} == first


def test_changed_inputs_refuse_and_leave_the_receipts_as_they_were(tmp_path, fakes):
    driver.run_synthetic(sources(), tmp_path, spec=SPEC, prereg=OK)
    before = snapshot(tmp_path)
    with pytest.raises(Refusal) as caught:
        driver.run_synthetic(sources(snapshot_id="snap-2"), tmp_path, spec=SPEC, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "inputs_receipt_changed"
    assert snapshot(tmp_path) == before


def test_the_stage_reads_no_2025_event(tmp_path, monkeypatch, fakes):
    ends, real_prod, real_labels = [], readers.read_production, readers.read_labels

    def spy(real):
        def wrapped(*args, **kwargs):
            ends.append(kwargs.get("end"))
            return real(*args, **kwargs)
        return wrapped

    monkeypatch.setattr(readers, "read_production", spy(real_prod))
    monkeypatch.setattr(readers, "read_labels", spy(real_labels))
    driver.run_synthetic(sources(), tmp_path, spec=SPEC, prereg=OK)
    assert ends and all(pd.Timestamp(e) <= pd.Timestamp("2025-01-01") for e in ends)
    assert fakes and all(set(ids) <= {"E1", "E2"} for ids in fakes)


def test_deleting_the_synthetic_files_leaves_the_inner_receipts_byte_identical(tmp_path, monkeypatch, stubs, fakes):
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    out = tmp_path / "out"
    driver.run(sources(), "inner", out, prereg=OK)
    inner = {p.name: p.read_bytes() for p in (out / "receipts").iterdir()}
    driver.run_synthetic(sources(), out, spec=SPEC, prereg=OK)
    for path in (out / "decisions_synthetic.parquet", out / "t11_synthetic.md",
                 out / "receipts" / "synthetic.json", out / "receipts" / "inputs_synthetic.json"):
        path.unlink()
    driver.run(sources(), "inner", out, prereg=OK)
    assert {p.name: p.read_bytes() for p in (out / "receipts").iterdir()} == inner


def test_month_scores_and_month_take_arms_and_emit_from(monkeypatch):
    """The arms= and emit_from= keywords of _month_scores and _month: only the as-is arm is scored, and no rows emit before emit_from."""
    monkeypatch.setattr(driver, "build_pool", lambda table, labels: None)
    monkeypatch.setattr(driver, "_printed", lambda labels, exits, cutoff: labels)
    monkeypatch.setattr(driver.shortvol, "simulate", lambda chunk_, events, pool, **kw: {e: "sim" for e in events["event_id"]})
    monkeypatch.setattr(driver.shortvol, "score_events", lambda chunk_, events, sims, name, **kw: pd.DataFrame(
        {"event_id": events["event_id"].to_numpy(), "status": "scored"}))
    monkeypatch.setattr(driver.straddle, "decide", lambda events, edges, rule, chunk_: pd.DataFrame(
        {"event_date": events["event_date"].to_numpy(), "edge": np.ones(len(events))}))
    events = pd.DataFrame({"event_id": ["E1", "E2"], "event_date": pd.to_datetime(["2023-06-10", "2023-06-20"])})
    table = pd.DataFrame({"event_date": pd.to_datetime(["2023-01-05"])}, index=pd.Index(["H0"]))
    start, st = pd.Timestamp("2023-06-01"), driver._model_state(1.0)
    scored, missing = driver._month_scores("m", table, events, chunk("E1", "E2"), None, {}, start, st, arms=(("as_is", 1.0),))
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
        driver.run_synthetic(sources(expected={**EXPECTED, "panel_id": "OTHER"}), tmp_path, spec=SPEC, prereg=OK)
    assert caught.value.code == CONFLICT
    assert not (tmp_path / "receipts").exists()


# Three inner events in two quarters and two tickers, so one drop leaves two quarters and two tickers for the evaluation.
EVENTS3 = {"E1": "2023-06-10", "E2": "2023-09-12", "E3": "2024-08-07"}
TICKERS3 = {"E1": "T1", "E2": "T2", "E3": "T1"}
DATES3 = pd.to_datetime(list(EVENTS3.values()))
ADMITTED3 = readers.Admitted({"events": pd.DataFrame({"event_id": list(EVENTS3), "event_date": DATES3}), "excluded": {}})
PLAN3 = pd.DataFrame({"event_id": list(EVENTS3), "ticker": [TICKERS3[e] for e in EVENTS3], "event_date": DATES3, "session": "AMC",
                      "entry_date": DATES3 - pd.Timedelta(days=1), "exit_date": DATES3 + pd.Timedelta(days=1)})
BASE3 = pd.DataFrame([{"event_id": e, "pred_abs_move": 5.0, "pred_iv_crush_30": -12.0, "horizon_sessions": 1,
                       "abs_move": 4.0, "crush_pct_iv30": -10.0} for e in EVENTS3])
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
                         "lots": np.ones(n, dtype=int), "entry_cost": np.ones(n), "pnl_per_share": np.ones(n), "pnl_dollars": np.ones(n)})


def _straddle_decide(events, edges, rule, chunk_):
    """Every straddle event taken (the real rules need the chunk's quotes)."""
    return pd.DataFrame({"event_id": events["event_id"].astype(str).to_numpy(), "event_date": events["event_date"].to_numpy(),
                         "edge": events["pred_abs_move"].to_numpy(dtype=float), "cutoff": np.nan, "rel_spread": np.nan,
                         "status": "taken", "reason": None})


def test_the_real_scoring_loop_drops_an_event_unscoreable_for_one_model_from_every_model(tmp_path, monkeypatch):
    """No _synthetic_scores stub: the real month loop runs. One model has no simulation for E3, so E3 leaves for every model."""
    def generate(admitted, labels, production, arm):
        entries = fake_entries(labels, arm)
        odd = sorted(entries)[0]
        table = entries[odd]["table"].copy()
        table.attrs["odd"] = True  # the pool keeps the table's attrs through the month slices
        return {**entries, odd: {**entries[odd], "table": table}}

    monkeypatch.setattr(synthetic, "generate", generate)
    monkeypatch.setattr(driver, "build_pool", lambda table, labels: table)
    monkeypatch.setattr(driver.shortvol, "simulate", lambda chunk_, events, pool, **kw: {
        e: (None if pool.attrs.get("odd") and e == "E3" else "sim") for e in events["event_id"]})
    monkeypatch.setattr(driver.shortvol, "score_events", _scored)
    monkeypatch.setattr(driver.shortvol, "score", _marks)
    monkeypatch.setattr(driver.straddle, "decide", _straddle_decide)
    monkeypatch.setattr(driver.straddle, "score", _marks)
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2023Q3", chunk("E2")), ("2024Q3", chunk("E3"))])
    src = sources(plan_frame=PLAN3, admitted=ADMITTED3, facts=lambda ids: FACTS3.reindex(list(ids)),
                  fetch=lambda columns, ids: BASE3[BASE3["event_id"].isin(list(ids))])
    out = tmp_path / "out"
    driver.run_synthetic(src, out, spec=SPEC, prereg=OK)
    frame = pd.read_parquet(out / "decisions_synthetic.parquet")
    units = len(driver.shortvol.NAMES) + len(driver.straddle.RULES)
    assert sorted(frame["event_id"].astype(str).unique()) == ["E1", "E2"]  # E3 is gone for every model
    assert len(set(frame["model_id"])) == 3
    assert len(frame) == 3 * 2 * units  # one row per model per surviving event per unit
    assert not frame.duplicated(["model_id", "book", "arm_or_rule", "structure", "event_id"]).any()
    receipt = json.loads((out / "receipts" / "synthetic.json").read_text(encoding="utf-8"))
    assert receipt["scoreability_dropped"] == 1
    first = snapshot(out)
    driver.run_synthetic(src, out, spec=SPEC, prereg=OK)
    assert snapshot(out) == first  # a second identical run is byte-identical
