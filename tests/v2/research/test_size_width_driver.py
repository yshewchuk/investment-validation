"""EXP-189 driver: synthetic runs only. The pricing seam is stubbed on size_pnl_shortvol (every event is declined, so no account
is walked) and the pool is stubbed as the EXP-188 driver tests stub it; the sources and chunks are the EXP-188 fixtures."""
import pandas as pd
import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError, fail
from experiments import size_pnl_driver as base
from experiments import size_pnl_readers as readers
from experiments import size_pnl_receipts as receipts
from experiments import size_pnl_shortvol as sv
from experiments import size_width_driver as driver
from experiments import size_width_fake as fake
from experiments import size_width_tuning as tuning
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_width_sizing import GRID
from experiments.size_width_success import LADDER, breakeven_alpha
from experiments.size_width_sweep import _TAKEN, RET_COLS, WINDOWS, Sweep
from tests.v2.research.test_size_pnl_driver import BASE, EVENTS, OK, PLAN_FRAME, chunk, sources, use_chunks

CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
NAME = "CND-PS"


@pytest.fixture(autouse=True)
def bounded(monkeypatch):
    monkeypatch.setenv("BOUNDED_RUN_NESTED", "1")


def _m0_table():
    ids = list(EVENTS)
    return pd.DataFrame({"event_date": pd.to_datetime([EVENTS[e] for e in ids]), "pred_move": 5.0, "pred_crush": -12.0},
                        index=pd.Index(ids, name="event_id"))


def _m0_key(roster):
    return next(m["model_id"] for m in roster["models"] if "M0" in m["roles"])


@pytest.fixture
def m0(monkeypatch):
    """The M0 table is the only ready model, keyed by the roster's M0 model id."""
    def tables(bound, roster, admitted, production, **kw):
        return {_m0_key(roster): {"status": "ready", "table": _m0_table(), "binding": {"model": "M0"}}}
    monkeypatch.setattr(readers, "model_tables", tables)


@pytest.fixture
def pricing(monkeypatch):
    """Every event is simulated, scored and declined (thin_history): the sweep runs, no account is walked."""
    def simulate(chunk_, events, pool, **kw):
        return {str(e): "sim" for e in events["event_id"]}

    def score_events(chunk_, events, sims, name, *, alpha=0.5):
        n = len(events)
        return pd.DataFrame({"event_id": events["event_id"].astype(str).to_numpy(), "event_date": events["event_date"].to_numpy(),
                             "structure": [name] * n, "status": ["scored"] * n, "reason": [None] * n,
                             "exp_pnl_sim": [0.01] * n, "entry_cost": [2.0] * n})

    def decide(scored):
        return scored.assign(status="undetermined", reason="thin_history")

    def score(chunk_, events, decisions, name, alpha, *, drop_unquoted=False, mid_marks=False, repriced=False):
        ids = [str(e) for e in decisions["event_id"]]
        n = len(ids)
        return pd.DataFrame({"event_id": pd.Series(ids, dtype=object), "structure": [name] * n, "entry_cost": [2.0] * n, "exit_value": [1.0] * n,
                             "pnl_dollars": [0.0] * n, "pnl_per_share": [0.0] * n})

    def secured(chunk_, events, decisions, name):
        ids = [str(e) for e in decisions["event_id"]]
        return pd.DataFrame({"event_id": pd.Series(ids, dtype=object), "secured_per_contract": [5.0] * len(ids)})

    monkeypatch.setattr(sv, "simulate", simulate)
    monkeypatch.setattr(sv, "score_events", score_events)
    monkeypatch.setattr(sv, "decide", decide)
    monkeypatch.setattr(sv, "score", score)
    monkeypatch.setattr(sv, "secured", secured)
    monkeypatch.setattr(driver, "build_pool", lambda table, labels: None)
    monkeypatch.setattr(fake, "build_pool", lambda table, labels: None)


def _loaded(src, ids=None):
    """The loaded run of the outer stage, built by hand (the inputs are the EXP-188 fixtures)."""
    return {"ids": ids, "plan": base._plan_window(PLAN_FRAME, base._window("outer"), ids), "table": _m0_table(),
            "labels": readers.read_labels(src.fetch, src.admitted, start=base.INNER[0], end=base.OUTER[0], event_ids=ids),
            "facts_all": base._load_facts(src, set(EVENTS)), "sessions": base._sessions(src), "exits": base._exits(src),
            "inputs_hash": "H", "inputs": {"stage": "inputs", "inputs_hash": "H"}}


def _inner_summary(out, inputs_hash="H", c_status="smoke_no_fake"):
    receipts.write_receipt(out / "receipts" / "inner_summary.json", {
        "stage": "inner", "inputs_hash": inputs_hash, "scope": None, "smoke": False, "c_status": c_status, "model": None,
        "calibration": None, "anchor": {}, "results": {}, "stats": {}, "declined": {}})


def _tuning_receipts(out, inputs_hash="H"):
    def one(members):
        return {"k": 1.0, "flags": [], "members": list(members), "scores": {}}
    bodies = {"A": {"all": one(NAMES)}, "B": {n: one((n,)) for n in NAMES},
              "D": {"T": one(tuning.TWIN), "C": one([n for n in NAMES if n not in tuning.TWIN])},
              "E": {n: {"k_share": {}, "modal_share": 0.0, "same_k": False, "candidates": 0, "no_feasible_k": 0, "curve": {}}
                    for n in NAMES}}
    for arm, body in bodies.items():
        receipts.write_receipt(out / "receipts" / f"tuning_{arm}.json", {"experiment": "EXP-189", "arm": arm,
                                                                         "inputs_hash": inputs_hash, "window": "2023H2-2024", "body": body})


# ---- refusals and stage -----------------------------------------------------

def test_not_bounded_refuses_resource_unavailable(tmp_path, monkeypatch):
    monkeypatch.delenv("BOUNDED_RUN_NESTED", raising=False)
    with pytest.raises(OpsError) as caught:
        driver.run(sources(), "inner", tmp_path, prereg=OK)
    assert caught.value.code == "RESOURCE_UNAVAILABLE"
    assert caught.value.problem.details["reason"] == "not_bounded"


def test_a_typo_stage_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="stage must be one of"):
        driver.run(sources(), "typo", tmp_path, prereg=OK)


def test_missing_m0_inputs_refuse_before_any_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(readers, "model_tables", lambda bound, roster, admitted, production, **kw: {})
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "inner", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "m0_inputs_missing"
    assert not (tmp_path / "receipts").exists()


def test_pricing_is_the_shortvol_seam(monkeypatch):
    assert driver.PRICING is sv


def test_the_sweep_is_built_on_the_pricing_seam(tmp_path, m0, pricing, monkeypatch):
    seen = []

    class Recorder(Sweep):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("pricing"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(driver, "Sweep", Recorder)
    use_chunks(monkeypatch, [("2023Q3", chunk("E1")), ("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path, event_ids=["E1", "E2"], prereg=OK)
    assert seen and all(p is sv for p in seen)


# ---- months ---------------------------------------------------------------------

def test_months_yield_output_false_before_the_emit_date_and_true_from_it(monkeypatch):
    src = sources()
    use_chunks(monkeypatch, [("2023Q2", chunk("E1")), ("2024Q3", chunk("E2"))])
    facts = base._load_facts(src, {"E1", "E2"})
    got = list(driver._months(src, PLAN_FRAME, _m0_table(), facts, {}, pd.Timestamp("2024-08-01")))
    assert [sorted(events["event_id"]) for _c, events, _o in got] == [["E1"], ["E2"]]
    assert [output for _c, _events, output in got] == [False, True]


# ---- inner stage ----------------------------------------------------------------

def test_smoke_inner_runs_end_to_end_without_a_fake_and_writes_no_tuning_c(tmp_path, m0, pricing, monkeypatch):
    use_chunks(monkeypatch, [("2023Q3", chunk("E1")), ("2024Q3", chunk("E2"))])
    out = tmp_path / "out"
    summary = driver.run(sources(), "inner", out, event_ids=["E1", "E2"], prereg=OK)
    assert summary["c_status"] == "smoke_no_fake" and summary["months"] == 2
    assert (out / "receipts" / "inputs_inner.json").exists()
    inner = receipts.load_receipt(out / "receipts" / "inner_summary.json")
    assert inner["c_status"] == "smoke_no_fake" and inner["smoke"] is True
    assert not (out / "receipts" / "tuning_C.json").exists()
    assert set(inner["results"]) == {"production", "A", "B", "D", "E"}
    assert all(set(periods) == {"2023H2", "2024"} for periods in inner["results"].values())
    assert set(inner["stats"]["production"]["2024"]) == {"trades", "x", "mid", "alpha0"}
    assert set(inner["declined"]) == set(NAMES) and all("E" in inner["declined"][n] for n in NAMES)


def test_the_full_inner_run_with_too_few_events_refuses_m0_inputs_insufficient(tmp_path, m0, pricing, monkeypatch):
    use_chunks(monkeypatch, [("2023Q3", chunk("E1")), ("2024Q3", chunk("E2"))])
    with pytest.raises(OpsError) as caught:
        driver.run(sources(), "inner", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.problem.details["reason"] == "m0_inputs_insufficient"


# ---- outer preflight -------------------------------------------------------------

def test_outer_without_an_inner_summary_refuses_tuning_receipt_missing(tmp_path, m0):
    src = sources()
    with pytest.raises(Refusal) as caught:
        driver._outer(src, tmp_path / "out", _loaded(src))
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "tuning_receipt_missing"


def test_outer_with_an_inner_summary_for_other_inputs_refuses_tuning_receipt_changed(tmp_path, m0):
    src = sources()
    _inner_summary(tmp_path / "out", inputs_hash="OTHER")
    with pytest.raises(Refusal) as caught:
        driver._outer(src, tmp_path / "out", _loaded(src))
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "tuning_receipt_changed"


def test_outer_with_a_missing_tuning_receipt_refuses_before_any_month(tmp_path, m0):
    src, out = sources(), tmp_path / "out"
    _inner_summary(out)
    with pytest.raises(Refusal) as caught:
        driver._outer(src, out, _loaded(src))
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "tuning_receipt_missing"
    assert not (out / "months").exists()


def test_outer_freezes_each_2025_month_before_its_labels_are_read(tmp_path, m0, pricing, monkeypatch):
    src, out = sources(), tmp_path / "out"
    _inner_summary(out)
    _tuning_receipts(out)
    loaded = _loaded(src)  # the pre-2025 labels are read before the recorder is installed
    seen, real = [], readers.read_labels

    def recording(fetch, admitted, **kw):
        month = pd.Timestamp(kw["start"]).to_period("M")
        if month.year == 2025:
            seen.append((str(month), (out / "months" / f"{month}.parquet").exists()))
        return real(fetch, admitted, **kw)

    monkeypatch.setattr(readers, "read_labels", recording)
    use_chunks(monkeypatch, [("2024Q3", chunk("E2")), ("2025Q1", chunk("E3", "E4"))])
    summary = driver._outer(src, out, loaded)
    assert seen == [("2025-01", True), ("2025-02", True)]
    assert summary["months"] == 3 and summary["arms"] == ["A", "B", "D", "E"]
    assert (out / "receipts" / "outer_summary.json").exists()
    outer = receipts.load_receipt(out / "receipts" / "outer_summary.json")
    assert outer["inputs_outer"].startswith("sha256:") and outer["labels_2025"].startswith("sha256:")
    assert set(outer["results"]) == {"production", "A", "B", "D", "E"}
    assert all(set(periods) == {"2025"} for periods in outer["results"].values())
    assert outer["arms"] == ["A", "B", "D", "E"]
    jan, feb = (real(src.fetch, src.admitted, start=pd.Timestamp(s), end=pd.Timestamp(e), event_ids=None)
                for s, e in (("2025-01-01", "2025-02-01"), ("2025-02-01", "2025-03-01")))
    assert outer["labels_2025"] == content_hash(base._label_rows([jan, feb]))
    # the Jan 2025 freeze holds E3's declined rows: every (structure, arm) pair, all thin_history, none taken
    frozen = pd.read_parquet(out / "months" / "2025-01.parquet")
    assert set(frozen["event_id"]) == {"E3"} and (frozen["status"] == "undetermined").all()
    assert (frozen["reason"] == "thin_history").all() and (frozen["book"] == "short_vol").all() and (frozen["model_id"] == "M0").all()
    assert len(frozen) == len(NAMES) * (len(GRID) + 1)
    assert set(frozen["arm_or_rule"]) == {str(k) for k in GRID} | {"E"}


# ---- stats and declined share -----------------------------------------------------

def _taken(eid, exit_date, rets):
    row = dict.fromkeys(_TAKEN + RET_COLS)
    row.update({"event_id": eid, "event_date": pd.Timestamp("2024-03-01"), "structure": NAME, "k": 1.0,
                "entry_date": pd.Timestamp("2024-03-01"), "exit_date": pd.Timestamp(exit_date), "entry_cost": 2.0,
                "exit_value": 1.0, "exit_value_mid": 1.0, "pnl_dollars": 0.0, "pnl_dollars_mid": 0.0,
                "secured_per_contract": 5.0})
    row.update(dict(zip(RET_COLS, rets)))
    return row


def _window_sweep(rows):
    sweep = Sweep()
    sweep.taken[(NAME, 1.0)] = [pd.DataFrame([_taken(f"e{i}", "2024-03-10", r) for i, r in enumerate(rows)], columns=_TAKEN + RET_COLS)]
    return sweep


def test_stats_are_the_breakeven_of_the_window_means_and_none_without_rows():
    by = {n: 1.0 for n in NAMES}
    # two rows per ladder column, averaging to 0.20, 0.10, -0.10, -0.20, -0.30 for alpha 0, 0.25, 0.5, 0.75, 1.0
    sweep = _window_sweep([[0.30, 0.20, 0.00, -0.10, -0.20], [0.10, 0.00, -0.20, -0.30, -0.40]])
    got = driver._stats(sweep, by, WINDOWS["2024"])
    # the mean crosses zero between alpha 0.25 and 0.5: 0.25 + 0.10 * 0.25 / 0.20 = 0.375
    assert got["x"] == pytest.approx(0.375)
    assert got["mid"] == pytest.approx(-0.10)
    assert got["alpha0"] == pytest.approx(0.20)
    assert got["trades"] == 2
    # always positive means: no crossing, breakeven 0.0
    assert driver._stats(_window_sweep([[0.3, 0.2, 0.1, 0.05, 0.02]] * 2), by, WINDOWS["2024"])["x"] == 0.0
    # always negative means: no crossing, breakeven 1.0
    assert driver._stats(_window_sweep([[-0.02, -0.1, -0.2, -0.25, -0.3]] * 2), by, WINDOWS["2024"])["x"] == 1.0
    empty = driver._stats(Sweep(), by, WINDOWS["2025"])
    assert empty == {"trades": 0, "x": None, "mid": None, "alpha0": None}


def test_declined_share_is_declined_over_declined_plus_taken():
    sweep = Sweep()
    sweep.declined[(NAME, 1.0)] = {"thin_history": 3, "below_cutoff": 1}
    sweep.taken[(NAME, 1.0)] = [pd.DataFrame(index=range(4), columns=_TAKEN)]
    sweep.declined[(NAME, "E")] = {"thin_history": 2}
    sweep.declined[(NAME, 2.0)] = {}
    assert driver._declined(sweep) == {NAME: {"1.0": 0.5, "E": 1.0, "2.0": 0.0}}


# ---- load: input binding and refusals ---------------------------------------------

def test_inputs_hash_is_stable_on_rerun_and_binds_the_inner_m0_table(m0, monkeypatch):
    src = sources()
    first = driver._load(src, "inner", None, OK())["inputs_hash"]
    assert driver._load(src, "inner", None, OK())["inputs_hash"] == first

    def moved(bound, roster, admitted, production, **kw):
        table = _m0_table()
        table.loc["E1", "pred_move"] = 6.0  # one inner event's prediction moves
        return {_m0_key(roster): {"status": "ready", "table": table, "binding": {"model": "M0"}}}

    monkeypatch.setattr(readers, "model_tables", moved)
    assert driver._load(src, "inner", None, OK())["inputs_hash"] != first


def test_inner_and_outer_loads_derive_the_same_inputs_hash(m0, monkeypatch):
    src = sources()
    monkeypatch.setattr(base.outer_inputs, "bind_outer", lambda plan, read, bound, roster, admit, **kw: bound)  # outer-only binder, pass-through
    inner = driver._load(src, "inner", None, OK())
    outer = driver._load(src, "outer", None, OK())
    assert outer["inputs_hash"] == inner["inputs_hash"]
    for key in ("inner_table", "inner_plan", "inner_facts"):
        assert outer["inputs"][key] == inner["inputs"][key]


def test_null_plan_date_refuses_plan_dates_null_in_load(m0):
    plan = PLAN_FRAME.copy()
    plan.loc[plan["event_id"] == "E1", "entry_date"] = pd.NaT
    with pytest.raises(OpsError) as caught:
        driver._load(sources(plan_frame=plan), "inner", None, OK())
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.details["reason"] == "plan_dates_null"


def test_an_inner_window_with_no_event_refuses_empty_population_in_load(m0, monkeypatch):
    monkeypatch.setattr(base, "_plan_window", lambda *a, **k: PLAN_FRAME.iloc[0:0])
    with pytest.raises(Refusal) as caught:
        driver._load(sources(), "inner", None, OK())
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "empty_population"


# ---- outer predictions, preflight and the fit population -----------------------

def test_outer_refuses_when_the_m0_table_has_no_2025_rows(tmp_path, monkeypatch):
    table = _m0_table().loc[["E1", "E2"]]
    monkeypatch.setattr(readers, "model_tables", lambda bound, roster, admitted, production, **kw: {
        _m0_key(roster): {"status": "ready", "table": table, "binding": {"model": "M0"}}})
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "outer", tmp_path, prereg=OK)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "outer_predictions_unbound"
    assert not (tmp_path / "receipts").exists()


def test_outer_preflight_failure_writes_no_inputs_receipt(tmp_path, m0):
    out = tmp_path / "out"
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "outer", out, prereg=OK)
    assert caught.value.details["reason"] == "tuning_receipt_missing"
    assert not list(tmp_path.rglob("inputs_outer.json"))


def test_fit_population_is_the_2023_24_rows_with_labels_printed_before_2025(tmp_path, monkeypatch, pricing):
    table = _m0_table().assign(pred_move=[1.0, 2.0, 3.0, 4.0])  # E1 is 2023H1, E2 is 2024, E3 and E4 are 2025

    def fetch(columns, ids):
        out = BASE[BASE["event_id"].isin(list(ids))].copy()
        out.loc[out["event_id"] == "E2", "available_as_of_date"] = "2025-01-02"  # E2's label prints after 2024-12-31
        return out

    seen = []

    def fit(pred, y):
        seen.append((pred.tolist(), y.tolist()))
        raise fail(CONFLICT, "too few events", details={"reason": "m0_inputs_insufficient"})

    monkeypatch.setattr(readers, "model_tables", lambda bound, roster, admitted, production, **kw: {
        _m0_key(roster): {"status": "ready", "table": table, "binding": {"model": "M0"}}})
    monkeypatch.setattr(driver.errmodel, "fit", fit)
    use_chunks(monkeypatch, [("2023Q3", chunk("E1")), ("2024Q3", chunk("E2"))])
    summary = driver.run(sources(fetch=fetch), "inner", tmp_path, event_ids=["E1", "E2"], prereg=OK)
    assert summary["c_status"] == "smoke_no_fake"
    assert len(seen) == 1
    pred, y = seen[0]
    assert pred == [1.0, 2.0]  # 2023H1 row (E1) is included
    assert y[0] == 4.0
    assert pd.isna(y[1])  # E2's label printed after 2024-12-31 is NaN, not 4.0


def test_fake_and_anchor_are_run_on_the_pricing_seam(tmp_path, m0, pricing, monkeypatch):
    seen = {}

    def fit(pred, y):
        return {"L": 1.0, "b": 0.8, "s": 0.2, "mean_log_y": 1.0, "n": 300}

    def run_fake(months, table, labels, model, sessions, out, inputs_hash, **kw):
        seen["fake"] = kw["pricing"]
        return {"status": "fake_not_calibrated", "calibration": {"ok": False}}

    def run_anchor(months, table, labels, sessions, **kw):
        seen["anchor"] = kw["pricing"]
        return {}

    monkeypatch.setattr(driver.errmodel, "fit", fit)
    monkeypatch.setattr(driver.fake, "run_fake", run_fake)
    monkeypatch.setattr(driver.fake, "run_anchor", run_anchor)
    use_chunks(monkeypatch, [("2023Q3", chunk("E1")), ("2024Q3", chunk("E2"))])
    summary = driver.run(sources(), "inner", tmp_path, event_ids=["E1", "E2"], prereg=OK)
    assert summary["c_status"] == "fake_not_calibrated"
    assert seen == {"fake": driver.PRICING, "anchor": driver.PRICING}
