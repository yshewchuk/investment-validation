"""EXP-190 Q6a: the Stage A driver end to end on synthetic data (fake labels and fits; no data/, no catalog).
Results are checked against an oracle written in this file, not against the production helpers."""
import contextlib
import io
import json
import math
import statistics
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import yaml

from engine.pnl_sim import MIN_POOL
from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError, fail
from experiments import lib
from experiments import size_pnl_readers as readers
from experiments import size_quantile_arm_s as arm_s
from experiments import size_quantile_fits as fits
from experiments import size_quantile_prereg as prereg
from experiments import size_quantile_stage_a as stage_a
from experiments import size_quantile_walk as walk

Q_KEYS = ["q10", "q25", "q40", "q50", "q60", "q75", "q90"]  # the arm's quantile columns, as the report names them

REPORT = "stage_a_report.json"
REAL_SPEC = prereg.SPEC_PATH
SPEC, _ = prereg.load_spec()
FIXED = SPEC["grid"]["fixed"]
STAMP = "2026-11-25T12:00:00+00:00"
LEVELS = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
Z = [statistics.NormalDist().inv_cdf(level) for level in LEVELS]
YEARS = ["2022", "2023", "2024", "2025"]
BINDING, DICT = "sha256:" + "b" * 64, "sha256:" + "a" * 64
TABLE_HASH, ALT_TABLE = "sha256:" + "e" * 64, "sha256:" + "f" * 64


def _candidates():
    return [{"candidate_id": f"h{leaves}_{leaf}", "feature_recipe": {"subset": "HMOSXLP"},
             "recipe": {"schema_version": "magnitude_ridge_hgb_recipe.v1", "model": "history_hgb",
                        "estimator": {"params": {**FIXED, "max_leaf_nodes": leaves, "min_samples_leaf": leaf,
                                                 "loss": "quantile", "quantile": 0.5}}}}
            for leaves in (7, 15) for leaf in (300, 1000)]


def _binding(table_hash, frame):
    """The production binding from its definition: the table hash and the rows sorted by event id, each as
    [event_id, event_date.isoformat(), pred_move, pred_crush]."""
    rows = [[event, day.isoformat(), float(move), float(crush)]
            for event, day, move, crush in sorted(zip(frame.index, frame["event_date"], frame["pred_move"],
                                                      frame["pred_crush"]), key=lambda row: row[0])]
    return content_hash({"table": table_hash, "rows": rows})


def _world():
    """Training 2020-2021 (20 events a month), then 2022-2025 (150 a month, days 1-28). mu ~ U(4, 10), s in
    {0.2, 1, 3} independent of mu, y = max(0, mu + s * normal). label_end is the next day, so all are before 2026."""
    rng = np.random.default_rng(20260829)
    events = []
    for year in range(2020, 2026):
        for month in range(1, 13):
            for k in range(20 if year < 2022 else 150):
                day = (k % 20 if year < 2022 else k % 28) + 1
                text = f"{year}-{month:02d}-{day:02d}"
                mu, s = float(rng.uniform(4, 10)), float(rng.choice([0.2, 1.0, 3.0]))
                events.append({"id": f"{text}-{k:03d}", "date": text, "ticker": f"T{k % 40:02d}", "mu": mu, "s": s,
                               "y": max(0.0, mu + s * float(rng.normal())),
                               "label_end": (date.fromisoformat(text) + timedelta(days=1)).isoformat()})
    events.sort(key=lambda e: e["id"])
    ids = [e["id"] for e in events]
    # implied_move (B2's option-implied move): None on every 97th event, 0.0 on every 89th, else mu.
    implied = {e["id"]: None if k % 97 == 0 else 0.0 if k % 89 == 0 else e["mu"] for k, e in enumerate(events)}
    # Outer training lists are the annual rule's: every event dated before the year starts, in event order.
    folds = [{"outer": {"year": y, "training_candidate_event_ids": [
        e["id"] for e in events if e["date"] < f"{y}-01-01"]}} for y in ("2023", "2024", "2025")]
    sealed = {
        "binding": BINDING, "dictionary_hash": DICT, "model_columns": ["f_mu", "f_s", "f_empty", "implied_move"],
        "evaluation_labels_before": "2026-02-01", "events": [{"event_id": i} for i in ids],
        "context": {"target_events": [{"event_id": e["id"], "ticker": e["ticker"], "event_date": e["date"],
                                       "label_end": e["label_end"]} for e in events],
                    "eligibility_plan": {"random_windows": []}},
        "fold_events": [{"event_id": e["id"], "ticker": e["ticker"], "event_date": e["date"],
                         "forecasts": [{"source_session_ceiling": e["date"]}]} for e in events],
        "plan": {"plan_id": "P1", "candidates": _candidates(), "folds": folds},
        "rows": [{"event_id": e["id"], "f_mu": e["mu"], "f_s": e["s"], "f_empty": None,
                  "implied_move": implied[e["id"]]} for e in events],
    }
    # Production forecast drifts biased from 2023-07-01: its pooled earlier errors hide the bias, so its bands under/over-cover in the judged periods.
    frame = pd.DataFrame({"pred_move": [e["mu"] + (3.0 if e["date"] >= "2023-07-01" else 0.0) for e in events],
                          "pred_crush": 0.0,
                          "event_date": [pd.Timestamp(e["date"]) for e in events]}, index=ids)
    labels = pd.DataFrame({"abs_move": [e["y"] for e in events], "crush_pct_iv30": 0.0}, index=ids)
    admitted = readers.Admitted({"events": pd.DataFrame({"event_id": ids, "event_date": [
        pd.Timestamp(e["date"]) for e in events]}), "excluded": {}})
    return {"events": events, "mu": {e["id"]: e["mu"] for e in events}, "s": {e["id"]: e["s"] for e in events},
            "y": {e["id"]: e["y"] for e in events}, "sealed": sealed, "frame": frame, "labels": labels,
            "dates": {e["id"]: e["date"] for e in events}, "admitted": admitted, "implied": implied}


PRODUCTION_BINDING = _binding(TABLE_HASH, _world()["frame"])
B3_BINDING = "sha256:" + "c" * 64
B3_YEARS = {"2023": "bound", "2024": "bound", "2025": "bound"}


def _b3_read(_slot):
    return "SELECTION"


B3_READ, B3_ADMIT = _b3_read, object()  # dummy binder inputs: the binders are faked in every test


def _digest(series):
    """The B3 series hash, written in the test: sorted [id, repr(float(value))] pairs under content_hash."""
    return content_hash(sorted([i, repr(float(v))] for i, v in series.items()))


def _b3(world, years=None):
    """A B3 binding: mu plus small noise for the events of each bound year (a year without a series is not bound)."""
    years = B3_YEARS if years is None else years
    rng = np.random.default_rng(190)
    kept = [e for e in world["events"] if years.get(e["date"][:4]) == "bound"]
    series = pd.Series([e["mu"] + 0.1 * float(rng.normal()) for e in kept], dtype=float,
                       index=pd.Index([e["id"] for e in kept], name="event_id"))
    return {"b3_candidate_id": "B3", "years": years, "series": series, "rule": "v14_mae_winner",
            "m3_candidate_id": "M3", "binding": B3_BINDING, "series_hash": _digest(series)}


def _fake_fit(boom=False):
    def regress(recipe, train_x, y, eval_x, *, columns):
        if boom:
            raise ValueError("boom")
        quantile = recipe["estimator"]["params"].get("quantile")  # arm S's absolute-error recipe has none: median
        z = 0.0 if quantile is None else statistics.NormalDist().inv_cdf(quantile)
        values = np.asarray(eval_x, dtype=float)
        return {"predictions": (values[:, 0] + values[:, 1] * z).tolist()}
    return regress


def _patch_b3(monkeypatch, b3, calls=None):
    """Fake both binders: bind_b3 names b3's rule and M3, and bind_series returns b3 as the bound series. When calls is
    given, each binder's arguments are appended to it, and bind_b3 also records the binding it returned."""
    binding = {"rule": b3["rule"], "m3_candidate_id": b3["m3_candidate_id"],
               "b3_candidate_id": b3["b3_candidate_id"]}

    def fake_bind_b3(plan, selection):
        if calls is not None:
            calls.append(("bind_b3", plan, selection, binding))
        return binding

    def fake_bind_series(plan, read, bound, admit):
        if calls is not None:
            calls.append(("bind_series", plan, read, bound, admit))
        return b3
    monkeypatch.setattr(stage_a, "bind_b3", fake_bind_b3)
    monkeypatch.setattr(stage_a, "bind_series", fake_bind_series)


def _next_day(world, event_id):
    """The event's label availability as the readers return it: event date + 1 day, ISO text."""
    return (date.fromisoformat(world["dates"][event_id]) + timedelta(days=1)).isoformat()


def _install(monkeypatch, world, *, omit=(), regress=None, read=None, s_regress=None):
    """Fake the label reader (dropping `omit` on the evaluation read only), the fit regressions of the walk and of
    arm S (`s_regress` for arm S only), and both B3 binders (returning _b3(world))."""
    def fake_read(sealed, event_ids, labels_before):
        drop = set(omit) if labels_before == "2026-02-01" else set()
        # As the real reader: a label counts only if it is available (event date + 1 day) before labels_before.
        kept = [e for e in event_ids if e not in drop
                and (date.fromisoformat(world["dates"][e]) + timedelta(days=1)).isoformat() < labels_before]
        return pd.DataFrame({"event_id": kept, "absolute_move_pct": [world["y"][e] for e in kept],
                             "target_available_on": [_next_day(world, e) for e in kept]})
    monkeypatch.setattr(walk, "training_labels", read or fake_read)
    monkeypatch.setattr(stage_a, "training_labels", read or fake_read)
    monkeypatch.setattr(arm_s, "training_labels", read or fake_read)
    monkeypatch.setattr(fits, "fit_regression_predictions", regress or _fake_fit())
    monkeypatch.setattr(arm_s, "fit_regression_predictions", s_regress or _fake_fit())
    _patch_b3(monkeypatch, _b3(world))


def _spy(monkeypatch, world):
    """Any label read or fit call is recorded and fails the test."""
    calls = []

    def no_read(*_args):
        calls.append("labels")
        raise AssertionError("no label may be read")

    def no_fit(*_args, **_kwargs):
        calls.append("fit")
        raise AssertionError("no fit may run")
    _install(monkeypatch, world, regress=no_fit, read=no_read, s_regress=no_fit)
    return calls


def _planned_row(digest):
    return {"id": "EXP-190", "spec_hash": digest, "date": "2026-11-25", "stage": "planned", "oos_mean_mid": "",
            "sharpe_trade": "", "promoted": "False"}


def _guard(case, *, settled=True, min_donors=30):
    """A tmp copy of the real spec (settled unless told otherwise, tail minimum set to min_donors), its PLANNED ledger
    row and one EXP-190 directory."""
    spec = yaml.safe_load(REAL_SPEC.read_text(encoding="utf-8"))
    spec["distribution_tails"]["min_donors"] = min_donors
    if settled:
        spec["stage_b"] = {**spec["stage_b"], "status": "settled"}
        spec["preregistered_at"] = STAMP
    spec_path = case / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    ledger_path = case / "LEDGER.csv"
    lib.ledger_append([_planned_row(prereg.load_spec(spec_path)[1])], path=ledger_path)
    (case / "exps" / "EXP-190_x").mkdir(parents=True)
    return {"spec_path": spec_path, "ledger_path": ledger_path, "run_log_path": case / "results" / "run_log.jsonl",
            "experiments_dir": case / "exps"}


def _fake_fetch(world, out, calls, *, frame=None, avail=None, extra=None, nan_crush=frozenset()):
    """The shared store as fetch(columns, ids), with literal column names and event_id plus the requested columns.
    Each call is recorded with whether every fit_<year>.json exists yet. frame replaces the production forecasts,
    avail overrides label availability by id, extra adds one unrequested row to the answer, nan_crush lists the ids
    whose crush_pct_iv30 is NaN (read_labels drops those labels)."""
    def fetch(columns, ids):
        calls.append({"columns": tuple(columns), "ids": list(ids),
                      "fits": all((out / "fits" / f"fit_{year}.json").exists() for year in YEARS)})
        if "pred_abs_move" in columns:
            src = world["frame"] if frame is None else frame
            got = {"pred_abs_move": src.loc[ids, "pred_move"].to_numpy(float),
                   "pred_iv_crush_30": src.loc[ids, "pred_crush"].to_numpy(float), "horizon_sessions": [1] * len(ids)}
        else:
            crush = [np.nan if e in nan_crush else float(world["labels"].at[e, "crush_pct_iv30"]) for e in ids]
            got = {"abs_move": world["labels"].loc[ids, "abs_move"].to_numpy(float),
                   "crush_pct_iv30": np.array(crush, dtype=float),
                   "available_as_of_date": [(avail or {}).get(e) or (date.fromisoformat(world["dates"][e])
                                                                     + timedelta(days=1)).isoformat() for e in ids]}
        answer = pd.DataFrame({"event_id": ids, **{column: got[column] for column in columns}})
        if extra is not None:
            answer = pd.concat([answer, pd.DataFrame({"event_id": [extra], **{column: [0.0] for column in columns}})])
        return answer
    return fetch


def _label_fetches(calls):
    return [call for call in calls if "abs_move" in call["columns"]]


def _run(world, out, guard, *, calls=None, years=YEARS, production_hash=TABLE_HASH, table_hash=TABLE_HASH,
         b3=None, monkeypatch=None, bound_b3_binding=None, **store):
    """Stage A on the fake shared store (calls records every fetch). b3, when given, replaces the faked binders' B3
    (needs monkeypatch); bound_b3_binding is the binding run_inputs bound, if any."""
    if b3 is not None:
        _patch_b3(monkeypatch, b3)
    fetch = _fake_fetch(world, out, [] if calls is None else calls, **store)
    return stage_a.run_stage_a(world["sealed"], years, world["admitted"], fetch, out,
                               production_hash=production_hash, table_hash=table_hash, b3_read=B3_READ,
                               b3_admit=B3_ADMIT, spec_path=guard["spec_path"], ledger_path=guard["ledger_path"],
                               run_log_path=guard["run_log_path"], experiments_dir=guard["experiments_dir"],
                               bound_b3_binding=bound_b3_binding)


def _refusal(exc):
    return exc.value.problem.code, exc.value.problem.details.get("reason")


def _why(doc):
    """Failure detail for the full-run status assert: top-level verdict fields and, per arm, the checked rules."""
    lines = [f"status={doc.get('status')!r}", f"reasons={doc.get('reasons')!r}",
             f"arm_verdicts={doc.get('arm_verdicts')!r}"]
    for name, arm in (doc.get("arms") or {}).items():
        cond = arm.get("conditional") or {}
        lines.append(f"arm {name}: status={arm.get('status')!r}")
        lines.append(f"  accuracy={arm.get('accuracy')!r}")
        lines.append(f"  calibration={arm.get('calibration')!r}")
        lines.append(f"  conditional.passed={cond.get('passed')!r}")
        lines.append(f"  conditional.periods={cond.get('periods')!r}")
        lines.append(f"  conditional.W_interval={cond.get('W_interval')!r}")
        lines.append(f"  conditional.diff_interval={cond.get('diff_interval')!r}")
        lines.append(f"  conditional.skips={cond.get('skips')!r}")
        lines.append(f"  share={arm.get('share')!r}")
    return "\n".join(lines)


def _true_quantiles(world, event_id):
    """The event's true quantiles at LEVELS, from its mu and s, floored at zero."""
    return np.maximum(world["mu"][event_id] + world["s"][event_id] * np.array(Z), 0.0)


def _donor_pools(world):
    """Returns donor_pools(month) -> (hi, lo), from the design rule alone. Donors are events dated in
    [M - 12 months, M), dated >= 2022-01-01, with label_end + 1 day < M. hi holds the donors with y > true q90
    and lo those with y < true q10, each scaled by its true band width (at least 0.5)."""
    dated = [(date.fromisoformat(e["date"]), date.fromisoformat(e["label_end"]) + timedelta(days=1), e["id"])
             for e in world["events"]]
    pools = {}

    def donor_pools(month):
        if month not in pools:
            start = month.replace(year=month.year - 1)
            hi, lo = [], []
            for day, ready, event_id in dated:
                if not (start <= day < month and day >= date(2022, 1, 1) and ready < month):
                    continue
                q, y = _true_quantiles(world, event_id), world["y"][event_id]
                if y > q[6]:
                    hi.append((y - q[6]) / max(q[6] - q[3], 0.5))
                if y < q[0]:
                    lo.append((y - q[0]) / max(q[3] - q[0], 0.5))
            pools[month] = (np.array(hi), np.array(lo))
        return pools[month]

    return donor_pools


def _oracle_mean_crps(world, quarter):
    """Mean CRPS over the quarter's events, from the design rule alone (no production helper).
    quantiles[0], [3], [6] are the event's q10, q50, q90."""
    grid = (np.arange(20000) + 0.5) / 20000
    donor_pools = _donor_pools(world)
    scores = []
    for event in quarter:
        month = date.fromisoformat(event["date"]).replace(day=1)
        hi, lo = donor_pools(month)
        q = _true_quantiles(world, event["id"])
        q10, q50, q90 = q[0], q[3], q[6]
        upper = q90 + max(q90 - q50, 0.5) * np.quantile(hi, np.clip((grid - 0.9) / 0.1, 0.0, 1.0))
        lower = np.maximum(0.0, q10 + max(q50 - q10, 0.5) * np.quantile(lo, np.clip(grid / 0.1, 0.0, 1.0)))
        Q = np.where(grid > 0.9, upper, np.where(grid < 0.1, lower, np.interp(grid, LEVELS, q)))
        y = world["y"][event["id"]]
        scores.append(2 * np.mean(np.where(y >= Q, grid * (y - Q), (1 - grid) * (Q - y))))
    return float(np.mean(scores))


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    case = tmp_path_factory.mktemp("stage_a_full")
    world, guard, out = _world(), _guard(case), case / "out"
    stdout, stderr = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as patch:
        _install(patch, world)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            doc = _run(world, out, guard)
    return {"world": world, "out": out, "doc": doc, "guard": guard, "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue()}


def test_full_run_passes_and_every_checked_number_matches_the_oracle(full_run):
    doc, out, world = full_run["doc"], full_run["out"], full_run["world"]
    assert doc["status"] == "passed" and doc["arm_verdicts"]["Q_raw"] == "pass", _why(doc)
    assert json.loads((out / REPORT).read_text(encoding="utf-8")) == doc
    assert content_hash({k: v for k, v in doc.items() if k != "content_hash"}) == doc["content_hash"]
    assert full_run["stdout"] == ""
    assert "[quantile stage A] start" in full_run["stderr"] and "done in" in full_run["stderr"]
    assert "stage_b_arms" not in doc and "stage_b_arms" not in (out / REPORT).read_text(encoding="utf-8")
    assert doc["not_included"] == ["y0_check", "b0_conditional_value", "unconditional_width_terciles"]
    assert sorted(p.name for p in out.iterdir()) == ["fits", REPORT]
    inputs = doc["inputs"]
    assert inputs["production_binding"] == PRODUCTION_BINDING and inputs["production_table_hash"] == TABLE_HASH
    for year in YEARS:
        receipt = json.loads((out / "fits" / f"fit_{year}.json").read_text(encoding="utf-8"))
        assert inputs["fit_receipt_hashes"][year] == receipt["content_hash"]
    labels = world["labels"]
    b0 = sorted([e, float(labels.at[e, "abs_move"]), float(labels.at[e, "crush_pct_iv30"]),
                 (date.fromisoformat(world["dates"][e]) + timedelta(days=1)).isoformat()] for e in labels.index)
    assert inputs["b0_labels_hash"] == content_hash(b0)
    evaluated = [e["id"] for e in world["events"] if e["date"] >= "2022-01-01"]
    assert inputs["evaluation_labels_hash"] == content_hash(sorted([e, world["y"][e]] for e in evaluated))
    assert inputs["unlabelled_outside_periods"] == []
    assert inputs["sealed_binding"] == BINDING
    assert inputs["spec_hash"] == prereg.load_spec(full_run["guard"]["spec_path"])[1]
    # require_preregistration returns preregistered_at as str(datetime), not the raw stamp.
    assert inputs["preregistered_at"] == "2026-11-25 12:00:00+00:00" and inputs["years"] == YEARS
    assert inputs["evaluation_labels_before"] == "2026-02-01"
    assert inputs["bootstrap"] == {"resamples": 10000, "seed": 188}
    assert doc["selection_period_disclosure"] == SPEC["stage_a"]["selection_period_disclosure"]

    rows = doc["arms"]["Q_raw"]["conditional"]["rows"]
    losses = []
    for row in rows:
        event = row["event_id"]
        assert row["y"] == pytest.approx(world["y"][event])
        for level, z in zip(LEVELS, Z):
            miss = world["y"][event] - max(world["mu"][event] + world["s"][event] * z, 0.0)
            losses.append(max(level * miss, (level - 1) * miss))
    pooled = next(t for t in doc["arms"]["Q_raw"]["tables"] if t["slice"] == "pooled")
    assert pooled["n"] == len(rows)
    assert pooled["mean_pinball"] == pytest.approx(sum(losses) / len(losses), rel=1e-9)
    assert pooled["crps"] > 0 and math.isfinite(pooled["crps"])

    eval_2025 = [e["id"] for e in world["events"] if e["date"].startswith("2025")]
    z90, z50 = Z[LEVELS.index(0.9)], Z[LEVELS.index(0.5)]
    hi = sum(max(world["mu"][e] + world["s"][e] * z90, 0.0) - max(world["mu"][e] + world["s"][e] * z50, 0.0) < 0.5
             for e in eval_2025)
    floor = next(r for r in doc["floor_report"] if r["arm"] == "Q_raw" and r["year"] == 2025)
    assert floor["events"] == len(eval_2025)
    assert floor["event_hi_bound"] == hi
    assert floor["event_hi_share"] == pytest.approx(hi / len(eval_2025))

    admitted = [e["id"] for e in world["events"] if "2023-07-01" <= e["date"] <= "2025-12-31"]
    share = next(r for r in doc["arms"]["Q_raw"]["share"] if r["period"] == "pooled")
    assert share["admitted"] == len(admitted) == 4500
    assert share["share"] == pytest.approx(share["paired"] / len(admitted))


def test_q_raw_quarter_crps_matches_an_independent_oracle(full_run):
    world, doc = full_run["world"], full_run["doc"]
    quarter = [e for e in world["events"] if "2024-04-01" <= e["date"] <= "2024-06-30"]
    row = next(t for t in doc["arms"]["Q_raw"]["tables"] if t["slice"] == "quarter" and t["label"] == "2024Q2")
    assert row["n"] == len(quarter)
    assert row["crps"] == pytest.approx(_oracle_mean_crps(world, quarter), rel=1e-3)


def test_history_is_the_arms_own_scored_out_of_sample_rows(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    _install(monkeypatch, world)
    judged, real_judge = {}, stage_a.judge_arm

    def recording(*args, **kwargs):
        judged[args[0]] = args[4].copy()
        return real_judge(*args, **kwargs)
    monkeypatch.setattr(stage_a, "judge_arm", recording)
    _run(world, out, guard)

    # Q3: a thin tail keeps its quantiles, so every window event is a scored Q_raw row and the history is all of them.
    window = [e for e in world["events"] if "2022-01-01" <= e["date"] <= "2025-12-31"]
    history = judged["Q_raw"]
    assert len(history) == len(window)
    got_pairs = sorted(zip(history["q50"].tolist(), history["y"].tolist()))
    want_pairs = sorted((max(world["mu"][e["id"]], 0.0), world["y"][e["id"]]) for e in window)
    np.testing.assert_allclose(np.array(got_pairs), np.array(want_pairs))
    got_available = sorted(pd.to_datetime(history["label_available"]))
    want_available = sorted(pd.Timestamp(e["date"]) + pd.Timedelta(days=2) for e in window)
    assert got_available == want_available


def test_replay_is_identical_and_a_changed_binding_refuses_without_touching_the_report(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    _install(monkeypatch, world)
    first = _run(world, out, guard)
    before = (out / REPORT).read_bytes()
    assert _run(world, out, guard) == first
    assert (out / REPORT).read_bytes() == before
    with pytest.raises(OpsError) as exc:
        _run(world, out, guard, production_hash=ALT_TABLE, table_hash=ALT_TABLE)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "stage_a_report_changed")
    assert (out / REPORT).read_bytes() == before
    assert not list(out.glob("*.tmp"))


def test_production_hash_mismatch_refuses_before_any_label_read_or_fit(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    reads, fetched = _spy(monkeypatch, world), []
    with pytest.raises(OpsError) as exc:
        _run(world, out, guard, calls=fetched, table_hash=ALT_TABLE)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "production_hash_mismatch")
    assert reads == [] and not _label_fetches(fetched) and not out.exists()


def test_non_finite_production_prediction_refuses_before_any_fit_or_label_read(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    reads, fetched = _spy(monkeypatch, world), []
    frame = world["frame"].copy()
    frame.loc[frame.index[0], "pred_move"] = np.nan
    with pytest.raises(OpsError) as exc:
        _run(world, out, guard, calls=fetched, frame=frame)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "production_series_failed")
    assert reads == [] and not _label_fetches(fetched) and not out.exists()


def test_unrequested_production_key_refuses_as_holdout_access_denied(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    reads, fetched = _spy(monkeypatch, world), []
    with pytest.raises(OpsError) as exc:
        _run(world, out, guard, calls=fetched, extra="not-admitted-event")
    assert _refusal(exc) == ("HOLDOUT_ACCESS_DENIED", "unrequested_key_returned")
    assert reads == [] and not _label_fetches(fetched) and not out.exists()


def test_the_binders_receive_the_sealed_plan_the_reader_the_binding_and_the_admit(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    _install(monkeypatch, world)
    calls = []
    _patch_b3(monkeypatch, _b3(world), calls)
    _run(world, out, guard)
    assert [call[0] for call in calls] == ["bind_b3", "bind_series"]
    _, plan, selection, binding = calls[0]
    assert plan is world["sealed"]["plan"] and selection == B3_READ("selection") == "SELECTION"
    _, bound_plan, read, bound, admit = calls[1]
    assert bound_plan is world["sealed"]["plan"]
    assert read is B3_READ and admit is B3_ADMIT
    assert bound is binding


@pytest.mark.parametrize("which", ["bind_b3", "bind_series"])
def test_a_binder_refusal_propagates_with_no_label_read_fit_or_report(monkeypatch, tmp_path, which):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    reads, fetched = _spy(monkeypatch, world), []
    error = fail("EXPERIMENT_IDENTITY_CONFLICT", f"{which} refused", details={"reason": f"{which}_refused"})

    def refuse(*_args):
        raise error
    monkeypatch.setattr(stage_a, which, refuse)
    with pytest.raises(OpsError) as exc:
        _run(world, out, guard, calls=fetched)
    assert exc.value is error
    assert reads == [] and not _label_fetches(fetched) and not out.exists()
    assert not (out / REPORT).exists()


def test_a_bound_b3_binding_other_than_the_bound_one_refuses_before_any_fit(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    reads, fetched = _spy(monkeypatch, world), []
    with pytest.raises(OpsError) as exc:
        _run(world, out, guard, calls=fetched, bound_b3_binding="sha256:" + "d" * 64)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "stage_a_b3_changed")
    assert reads == [] and not _label_fetches(fetched) and not out.exists()


def test_b0_label_not_available_after_its_event_date_refuses_and_publishes_nothing(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    _install(monkeypatch, world)
    event = next(e["id"] for e in world["events"] if e["date"] == "2024-03-05")
    with pytest.raises(OpsError) as exc:
        _run(world, out, guard, avail={event: "2024-03-05"})
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "label_availability_missing")
    assert not (out / REPORT).exists() and not list(out.glob("*.tmp"))


def test_production_is_read_before_fits_and_labels_only_after_every_fit(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    _install(monkeypatch, world)
    fetched = []
    _run(world, out, guard, calls=fetched)
    production = [call for call in fetched if "pred_abs_move" in call["columns"]]
    assert len(production) == 1 and not production[0]["fits"]
    labels = _label_fetches(fetched)
    assert labels and all(call["fits"] for call in labels)  # every walk fit is written before any outcome is read
    assert fetched.index(production[0]) < fetched.index(labels[0])


def test_unsettled_stage_b_refuses_before_any_label_read_or_fit(monkeypatch, tmp_path):
    world, guard = _world(), _guard(tmp_path, settled=False)
    calls = _spy(monkeypatch, world)
    with pytest.raises(OpsError) as exc:
        _run(world, tmp_path / "out", guard)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "stage_b_pending")
    assert calls == [] and not (tmp_path / "out").exists()


def test_spec_changed_after_the_check_refuses_before_any_label_read_or_fit(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    calls = _spy(monkeypatch, world)
    monkeypatch.setattr(stage_a, "load_spec", lambda _path: (SPEC, "sha256:" + "0" * 64))
    with pytest.raises(OpsError) as exc:
        _run(world, out, guard)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "spec_changed")
    assert calls == [] and not out.exists()


@pytest.mark.parametrize("min_donors", [50, 29])
def test_tail_minimum_other_than_the_code_refuses_before_any_fetch_label_read_or_fit(monkeypatch, tmp_path,
                                                                                     min_donors):
    world, guard, out = _world(), _guard(tmp_path, min_donors=min_donors), tmp_path / "out"
    reads, fetched = _spy(monkeypatch, world), []
    with pytest.raises(OpsError) as exc:
        _run(world, out, guard, calls=fetched)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "stage_a_tails_mismatch")
    assert reads == [] and fetched == [] and not out.exists()


@pytest.mark.parametrize("years", [
    ["2023", "2024"], ["2022", "2024", "2025"], ["2023", "2024", "2025", "2026"], ["2023", "2024", 2025],
    "2023,2024,2025"])
def test_years_must_cover_2023_to_2025_and_end_in_2025(monkeypatch, tmp_path, years):
    world, guard = _world(), _guard(tmp_path)
    calls = _spy(monkeypatch, world)
    with pytest.raises(OpsError) as exc:
        _run(world, tmp_path / "out", guard, years=years)
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "stage_a_years_invalid")
    assert calls == [] and not (tmp_path / "out").exists()


def test_late_failure_leaves_no_report_and_no_temp_file(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    _install(monkeypatch, world)

    def late(*_args):
        raise RuntimeError("late failure")
    monkeypatch.setattr(stage_a, "stage_a_verdict", late)
    with pytest.raises(RuntimeError):
        _run(world, out, guard)
    assert not (out / REPORT).exists() and not list(out.glob("*.tmp"))
    assert (out / "fits" / "fit_2025.json").exists()


def test_in_period_event_without_a_label_is_counted_unpaired_not_refused(monkeypatch, tmp_path, full_run):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    picks = [next(e["id"] for e in world["events"] if e["date"].startswith(prefix)) for prefix in
             ("2023-08", "2024-05", "2025-11")]  # Q4 option B: three in-period events lose their evaluation label
    _install(monkeypatch, world, omit=picks)
    doc = _run(world, out, guard)
    assert json.loads((out / REPORT).read_text(encoding="utf-8")) == doc
    assert doc["unlabelled_in_period"] == {"2023": 1, "2024": 1, "2025": 1}
    assert doc["inputs"]["unlabelled_outside_periods"] == []
    pooled = next(r for r in full_run["doc"]["arms"]["Q_raw"]["share"] if r["period"] == "pooled")
    now = next(r for r in doc["arms"]["Q_raw"]["share"] if r["period"] == "pooled")
    assert now["admitted"] == pooled["admitted"] == 4500  # the denominator still counts them
    assert now["paired"] == pooled["paired"] - 3  # and only they leave the paired count
    assert not {r["event_id"] for r in doc["arms"]["Q_raw"]["conditional"]["rows"]} & set(picks)
    drops = [(d["year"], d["arm_reason"], d["b0_reason"], d["n"]) for d in doc["arms"]["Q_raw"]["drops"]
             if d["arm_reason"] == "label_missing"]
    assert sorted(drops) == [(2023, "label_missing", "label_missing", 1), (2024, "label_missing", "label_missing", 1),
                             (2025, "label_missing", "label_missing", 1)]


def test_unlabelled_event_outside_the_periods_is_recorded_not_refused(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    omitted = next(e["id"] for e in world["events"] if e["date"].startswith("2022"))
    _install(monkeypatch, world, omit=(omitted,))
    doc = _run(world, out, guard)
    assert doc["inputs"]["unlabelled_outside_periods"] == [omitted]
    assert json.loads((out / REPORT).read_text(encoding="utf-8")) == doc


def test_fit_failure_publishes_a_not_testable_report(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    _install(monkeypatch, world, regress=_fake_fit(boom=True))
    fetched = []
    doc = _run(world, out, guard, calls=fetched)
    assert doc["status"] == "not_testable"
    assert doc["reasons"] == ["Q:fit_failed", "Q_raw:fit_failed"]
    assert doc["arm_verdicts"] == {"Q": "not_testable", "Q_raw": "not_testable"}
    assert doc["inputs"]["unlabelled_outside_periods"] is None
    assert doc["inputs"]["b0_labels_hash"] is None and not _label_fetches(fetched)
    assert doc["arms"] == {} and doc["fit_status"] == {"2022": "fit_failed"} and doc["floor_report"] == []
    assert doc["unlabelled_in_period"] == {} and doc["crps_exclusions"] == []
    assert doc["inputs"]["evaluation_labels_hash"] is None and doc["inputs"]["b1b2_labels_hash"] is None
    assert doc["reported"] == {} and doc["arm_s"] == {} and doc["headline_flags"] == [] and doc["crps_rows"] == []
    assert doc["inputs"]["arm_s_receipt_hashes"] == {} and isinstance(doc["inputs"]["b2_implied_hash"], str)
    assert doc["inputs"]["b3"]["rule"] == "v14_mae_winner"
    assert json.loads((out / REPORT).read_text(encoding="utf-8")) == doc


def test_memory_error_in_calibration_refuses_resource_limit(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    _install(monkeypatch, world)

    def out_of_memory(*_args):
        raise MemoryError
    monkeypatch.setattr(stage_a, "calibrate", out_of_memory)
    with pytest.raises(OpsError) as exc:
        _run(world, out, guard)
    assert exc.value.problem.code == "RESOURCE_LIMIT_EXCEEDED"
    assert not (out / REPORT).exists()


B0_COLS = [f"b0_q{round(a * 100)}" for a in LEVELS]


def _b0_oracle(world, frame=None):
    """B0's quantiles and bucket for each event, from the design rule in plain numpy (no ResidualPool, build_pool or
    pool_quantiles; only the MIN_POOL constant is taken from engine.pnl_sim).
    Donors: every fixture event dated strictly before the event, by event date, as the native pool is. Label
    availability is not a B0 filter (the open question in issue #3047). err_move = donor abs_move - donor pred_move.
    An event with zero earlier donors is unscored: the oracle returns (None, None), as production's ResidualPool
    refuses the draw (reason refused_draw) when the earlier pool is empty.
    Bucket edges are the earlier donors' pred_move quantiles on production's grid, np.linspace(0, 1, 11)[1:-1]
    (the same float values, not a re-derived 0.1..0.9 grid, which can differ by an ulp and move a donor sitting exactly
    on an edge into another bucket). Membership rule, as in ResidualPool.draw: a value v belongs to bucket
    searchsorted(edges, v, side="right"), so an edge value falls in the bucket ABOVE it, for donors and for the event's
    forecast alike. The event's bucket uses its fixture pred_move (with the +3 drift from 2023-07-01). The draw is the
    donors in that bucket, or every earlier donor when the bucket holds fewer than MIN_POOL."""
    # The forecast table: its index is the event set (donors and scored events) and its pred_move the forecast.
    # B3 passes its own series; the default is the production frame with every event.
    frame = world["frame"] if frame is None else frame
    members = set(frame.index)
    events = [e for e in world["events"] if e["id"] in members]
    ids = [e["id"] for e in events]
    dates = np.array([e["date"] for e in events])  # ISO dates compare correctly as strings
    pred = frame.loc[ids, "pred_move"].to_numpy(float)
    err = np.array([world["y"][i] for i in ids], dtype=float) - pred
    position = {i: k for k, i in enumerate(ids)}
    grid = np.linspace(0, 1, 11)[1:-1]  # production's bucket grid (buckets=10), bit-for-bit

    def oracle(event_id):
        k = position[event_id]
        earlier = dates < dates[k]
        donor_pred, donor_err = pred[earlier], err[earlier]
        if donor_pred.size < MIN_POOL:  # fewer than MIN_POOL earlier donors: unscored, no quantiles and no bucket
            return None, None
        edges = np.quantile(donor_pred, grid)
        f = pred[k]
        bucket = int(np.searchsorted(edges, f, side="right"))
        in_bucket = np.searchsorted(edges, donor_pred, side="right") == bucket
        draw = donor_err[in_bucket] if in_bucket.sum() >= MIN_POOL else donor_err
        return np.quantile(np.maximum(f + draw, 0.0), LEVELS), float(bucket)

    return oracle


def test_b0_rows_match_an_independent_design_rule_oracle(full_run):
    world, doc = full_run["world"], full_run["doc"]
    oracle = _b0_oracle(world)
    rows = pd.DataFrame(doc["arms"]["Q_raw"]["conditional"]["rows"])
    checked = rows[rows["quarter"].isin(["2023Q3", "2025Q4"])]
    assert len(checked) >= 60
    want = [oracle(event_id) for event_id in checked["event_id"]]
    np.testing.assert_allclose(checked["bucket"].to_numpy(float), [bucket for _, bucket in want], rtol=1e-9)
    np.testing.assert_allclose(checked[B0_COLS].to_numpy(float), np.array([q for q, _ in want]), rtol=1e-9)

    quarter = [e for e in world["events"] if "2025-10-01" <= e["date"] <= "2025-12-31"]
    table = next(t for t in doc["arms"]["Q_raw"]["tables"] if t["slice"] == "quarter" and t["label"] == "2025Q4")
    assert table["n"] == len(quarter)
    q = np.array([oracle(e["id"])[0] for e in quarter])
    y = np.array([world["y"][e["id"]] for e in quarter], dtype=float)
    levels = np.array(LEVELS)
    miss = y[:, None] - q
    pinball = np.maximum(levels * miss, (levels - 1) * miss)
    assert table["b0_mean_pinball"] == pytest.approx(pinball.mean(), rel=1e-9)


def test_arms_schema_and_not_included_list(full_run):
    doc = full_run["doc"]
    assert doc["schema_version"] == "size_quantile_stage_a.v4"
    assert doc["inputs"]["run_inputs_hash"] is None  # library call: no entry point, so the y0 check stays listed
    assert set(doc["arms"]) == {"Q", "Q_raw"} and set(doc["arm_verdicts"]) == {"Q", "Q_raw"}
    assert doc["not_included"] == ["y0_check", "b0_conditional_value", "unconditional_width_terciles"]
    assert {"arm_s", "reported", "headline_flags", "crps_rows"} <= set(doc)
    assert {"arm_s_receipt_hashes", "b2_implied_hash", "b3"} <= set(doc["inputs"])
    assert {"crps_exclusions", "unlabelled_in_period", "floor_report"} <= set(doc)


def _b0_pool_oracle(world):
    """B0 quantiles and month pools from the design rule, in plain numpy. Donors are the 2022+ events (the B0 frame's
    rows), dated in [M - 12 months, M) with label_end + 1 day < M, scored by their own B0 quantiles from _b0_oracle.
    A tail with under 30 donors makes the month thin. Returns (quantiles_of, pool_of); pool_of(month) is (d_lo, d_hi)
    or None."""
    oracle = _b0_oracle(world)
    quantiles = {e["id"]: oracle(e["id"])[0] for e in world["events"] if e["date"] >= "2022-01-01"}
    return oracle, _tail_pool_of(world, quantiles)


def test_b0_crps_in_the_quarter_table_matches_an_independent_oracle(full_run):
    world, doc = full_run["world"], full_run["doc"]
    oracle, pool_of = _b0_pool_oracle(world)
    quarter = [e for e in world["events"] if "2024-04-01" <= e["date"] <= "2024-06-30"]
    row = next(t for t in doc["arms"]["Q_raw"]["tables"] if t["slice"] == "quarter" and t["label"] == "2024Q2")
    grid = (np.arange(20000) + 0.5) / 20000
    scores = []
    for event in quarter:
        pool = pool_of(date.fromisoformat(event["date"]).replace(day=1))
        if pool is None:
            continue
        d_lo, d_hi = pool
        q = oracle(event["id"])[0]
        s_lo, s_hi = max(q[3] - q[0], 0.5), max(q[6] - q[3], 0.5)
        lower = np.maximum(0.0, q[0] + s_lo * np.quantile(d_lo, np.clip(grid / 0.1, 0.0, 1.0)))
        upper = q[6] + s_hi * np.quantile(d_hi, np.clip((grid - 0.9) / 0.1, 0.0, 1.0))
        Q = np.where(grid > 0.9, upper, np.where(grid < 0.1, lower, np.interp(grid, LEVELS, q)))
        y = world["y"][event["id"]]
        scores.append(2 * np.mean(np.where(y >= Q, grid * (y - Q), (1 - grid) * (Q - y))))
    assert row["b0_n_crps"] == len(scores) > 0
    assert row["b0_crps"] == pytest.approx(float(np.mean(scores)), rel=1e-3)


def test_crps_exclusions_and_pooled_crps_counts_match_the_oracles(full_run):
    world, doc = full_run["world"], full_run["doc"]
    _, pool_of = _b0_pool_oracle(world)
    excl = {(r["arm"], r["year"]): r for r in doc["crps_exclusions"]}
    # The report is JSON: row order is not part of the contract, so compare by (arm, year) and reject duplicate keys.
    arms = ("Q", "Q_raw", "B0", "S", "B1", "B2", "B3")
    assert len(excl) == len(doc["crps_exclusions"]) == 3 * len(arms)
    assert sorted(excl) == sorted((arm, year) for arm in arms for year in (2023, 2024, 2025))
    in_period = [e for e in world["events"] if "2023-07-01" <= e["date"] <= "2025-12-31"]
    for year in (2023, 2024, 2025):
        events_of_year = [e for e in in_period if e["date"].startswith(str(year))]
        for arm in ("Q", "Q_raw", "B0"):
            assert excl[(arm, year)]["scored"] == len(events_of_year)  # every in-period event is scored here
        missing = sum(pool_of(date.fromisoformat(e["date"]).replace(day=1)) is None for e in events_of_year)
        assert excl[("B0", year)]["crps_excluded"] == missing
    pooled = {arm: next(t for t in doc["arms"][arm]["tables"] if t["slice"] == "pooled") for arm in ("Q", "Q_raw")}
    for arm in ("Q", "Q_raw"):  # the pooled CRPS count is the scored events that keep a pool
        kept = sum(excl[(arm, year)]["scored"] - excl[(arm, year)]["crps_excluded"] for year in (2023, 2024, 2025))
        assert pooled[arm]["n_crps"] == kept
    b0_kept = sum(excl[("B0", year)]["scored"] - excl[("B0", year)]["crps_excluded"] for year in (2023, 2024, 2025))
    assert pooled["Q_raw"]["b0_n_crps"] == b0_kept


FLOOR, MIN_TAIL = 0.5, 30  # the design's scale floor and minimum tail donors


def _in_period(text):
    return "2023-07-01" <= text <= "2025-12-31"


def _month(text):
    return date.fromisoformat(text).replace(day=1)


def _oracle_b2(world):
    """B2 quantiles of every event dated 2022+ that it scores, in plain numpy: donors dated in the 12 months before the
    event's month, available (date + 1 day) before that month, with a positive implied move; at least 500 of them.
    An event without a positive implied move has no quantiles."""
    events, implied = world["events"], world["implied"]
    day = {e["id"]: date.fromisoformat(e["date"]) for e in events}
    usable = {e["id"]: implied[e["id"]] is not None and implied[e["id"]] > 0 for e in events}
    ratio_of, out = {}, {}
    for event in events:
        if event["date"] < "2022-01-01" or not usable[event["id"]]:
            continue
        month = _month(event["date"])
        if month not in ratio_of:
            start = month.replace(year=month.year - 1)
            donors = [d for d in day if usable[d] and start <= day[d] < month and day[d] + timedelta(days=1) < month]
            ratio_of[month] = (np.quantile([world["y"][d] / implied[d] for d in donors], LEVELS)
                               if len(donors) >= 500 else None)
        if ratio_of[month] is not None:
            out[event["id"]] = implied[event["id"]] * ratio_of[month]
    return out


def _s_quantiles(world):
    """Arm S's quantiles for the events dated 2023+ (2022 has no donors: it is the first year), in plain numpy. Both arm
    S fits predict mu, so the median and the scale are mu and the 0.5 floor never binds. Month M's donors are the events
    dated 2022+ in [M - 12 months, M) with label available before M (date + 1 day < M); under 500 donors leaves the month
    unscored. The quantiles are max(sort(mu + mu * z), 0) with z = quantile((y - mu_d) / mu_d, LEVELS) over the donors."""
    dated = [(date.fromisoformat(e["date"]), e["id"]) for e in world["events"] if e["date"] >= "2022-01-01"]
    mu, y = world["mu"], world["y"]
    z_of = {}

    def z(month):
        if month not in z_of:
            start = month.replace(year=month.year - 1)
            ids = [i for day, i in dated if start <= day < month and day + timedelta(days=1) < month]
            z_of[month] = np.quantile([(y[i] - mu[i]) / mu[i] for i in ids], LEVELS) if len(ids) >= 500 else None
        return z_of[month]

    out = {}
    for e in world["events"]:
        if e["date"] < "2023-01-01":
            continue
        zz = z(_month(e["date"]))
        if zz is not None:
            m = mu[e["id"]]
            out[e["id"]] = np.maximum(np.sort(m + m * zz), 0.0)
    return out


def _b1_quantiles(world):
    """Arm B1's quantiles for the events dated 2022+, in plain numpy. Donors are every event with label available before
    the month (date + 1 day < M), full history, x = the production pred_move and y = abs_move. The bucket edges are the
    donors' quantiles at 0.2, 0.4, 0.6 and 0.8; a month is unscored when a bucket holds fewer than 20 donors. Quantiles
    are max(f, 0.05) times the bucket's quantiles of ratio = y / max(x, 0.05)."""
    dated = [(date.fromisoformat(e["date"]), e["id"]) for e in world["events"]]
    x_of, y_of = world["frame"]["pred_move"].to_dict(), world["y"]
    pools = {}

    def pool(month):
        if month not in pools:
            ids = [i for day, i in dated if day + timedelta(days=1) < month]
            x = np.array([x_of[i] for i in ids], dtype=float)
            y = np.array([y_of[i] for i in ids], dtype=float)
            edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
            bucket = np.digitize(x, edges)
            if np.bincount(bucket, minlength=5).min() < 20:
                pools[month] = None
            else:
                ratio = y / np.maximum(x, 0.05)
                pools[month] = (edges, [np.quantile(ratio[bucket == b], LEVELS) for b in range(5)])
        return pools[month]

    out = {}
    for e in world["events"]:
        if e["date"] < "2022-01-01":
            continue
        found = pool(_month(e["date"]))
        if found is not None:
            edges, ratio_q = found
            f = x_of[e["id"]]
            out[e["id"]] = max(f, 0.05) * ratio_q[int(np.digitize(f, edges))]
    return out


def _tail_pool_of(world, quantiles):
    """pool_of(month) -> (d_lo, d_hi) or None: the design's tail rule from one arm's own oracle quantiles (a dict of event
    id -> seven quantiles, for its scored events). Donors are dated in [M - 12 months, M) with label_end + 1 day < M (the
    availability donor_pool uses). d_hi is (y - q90) / max(q90 - q50, 0.5) over y > q90 and d_lo is (y - q10) /
    max(q50 - q10, 0.5) over y < q10; a month with under 30 donors on either tail is thin (None)."""
    donors = [(date.fromisoformat(e["date"]), date.fromisoformat(e["label_end"]) + timedelta(days=1),
               world["y"][e["id"]], quantiles[e["id"]]) for e in world["events"] if e["id"] in quantiles]
    pools = {}

    def pool_of(month):
        if month not in pools:
            start = month.replace(year=month.year - 1)
            hi, lo = [], []
            for day, ready, y, q in donors:
                if not (start <= day < month and ready < month):
                    continue
                if y > q[6]:
                    hi.append((y - q[6]) / max(q[6] - q[3], FLOOR))
                if y < q[0]:
                    lo.append((y - q[0]) / max(q[3] - q[0], FLOOR))
            pools[month] = (np.array(lo), np.array(hi)) if min(len(lo), len(hi)) >= MIN_TAIL else None
        return pools[month]

    return pool_of


def _exact_crps(y, q, d_lo, d_hi):
    """CRPS of one event's tailed quantile function, integrated exactly with no grid. The function is piecewise linear
    through the lower-tail knots, the seven quantiles and the upper-tail knots, floored at zero. Each segment is cut where
    the linear piece crosses zero or y, so on every piece the function is linear and the indicator 1{y < Q} is constant;
    the integrand is then quadratic and Simpson's rule is exact."""
    q = np.asarray(q, dtype=float)
    s_lo, s_hi = max(q[3] - q[0], FLOOR), max(q[6] - q[3], FLOOR)
    lo, hi = np.sort(np.asarray(d_lo, dtype=float)), np.sort(np.asarray(d_hi, dtype=float))
    u = np.concatenate([np.linspace(0.0, 0.1, lo.size), LEVELS, np.linspace(0.9, 1.0, hi.size)])
    x = np.concatenate([q[0] + s_lo * lo, q, q[6] + s_hi * hi])
    keep = u[1:] > u[:-1]  # a zero-width step is a jump, with no area
    a, b, xa, xb = u[:-1][keep], u[1:][keep], x[:-1][keep], x[1:][keep]
    span = np.where(xb == xa, 1.0, xb - xa)
    cuts = [np.zeros_like(a), np.ones_like(a)]
    for level in (0.0, float(y)):
        crossing = (xa - level) * (xb - level) < 0
        cuts.append(np.where(crossing, (level - xa) / span, 0.0))
    f = np.sort(np.stack(cuts, axis=1), axis=1)  # fractions along each segment, sorted, with zero-width pieces
    p, r = f[:, :-1], f[:, 1:]
    m = (p + r) / 2
    width = (b - a)[:, None] * (r - p)

    def at(t):
        return a[:, None] + (b - a)[:, None] * t

    def floored(t):
        return np.maximum(xa[:, None] + (xb - xa)[:, None] * t, 0.0)

    qp, qm, qr = floored(p), floored(m), floored(r)
    below = (y < qm).astype(float)  # 1{y < Q}, constant on each piece

    def loss(u_, q_):
        return (y - q_) * (u_ - below)

    total = width / 6 * (loss(at(p), qp) + 4 * loss(at(m), qm) + loss(at(r), qr))
    return 2.0 * float(total.sum())


def _arm_crps(world, quantiles, in_period):
    """CRPS of each in-period event for one arm, from its oracle quantiles and the tail rule. None where the arm has no
    quantile for the event or its month's tail pool is thin."""
    pool_of = _tail_pool_of(world, quantiles)
    out = {}
    for event_id in in_period:
        q = quantiles.get(event_id)
        pool = None if q is None else pool_of(_month(world["dates"][event_id]))
        out[event_id] = None if pool is None else _exact_crps(world["y"][event_id], q, *pool)
    return out


def _mean_pinball(y, q):
    """Mean over events and the seven levels of max(a * d, (a - 1) * d) with d = y - q, from the definition."""
    d = np.asarray(y, float)[:, None] - np.asarray(q, float)
    a = np.array(LEVELS)
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def test_b2_table_pooled_n_and_mean_pinball_match_the_plain_numpy_oracle(full_run):
    world, doc = full_run["world"], full_run["doc"]
    oracle = {i: q for i, q in _oracle_b2(world).items() if _in_period(world["dates"][i])}
    row = next(t for t in doc["reported"]["B2"]["tables"] if t["slice"] == "pooled")
    assert row["n"] == len(oracle) > 0
    assert row["mean_pinball"] == pytest.approx(_mean_pinball([world["y"][i] for i in oracle], list(oracle.values())),
                                                rel=1e-9)


def test_reported_tables_have_no_pass_flags_and_b1_b3_count_every_in_period_event(full_run):
    doc = full_run["doc"]
    assert set(doc["reported"]) == {"S", "B1", "B2", "B3"}
    for table in doc["reported"].values():
        assert set(table) == {"status", "not_testable_reasons", "share", "drops", "tables"}
    in_period = sum("2023-07-01" <= e["date"] <= "2025-12-31" for e in full_run["world"]["events"])
    assert in_period == 4500
    for name in ("B1", "B3"):  # every in-period event has a band and a B3 forecast in this world
        pooled = next(t for t in doc["reported"][name]["tables"] if t["slice"] == "pooled")
        assert pooled["n"] == in_period, name


def test_headline_flags_match_the_oracle(full_run):
    doc, world = full_run["doc"], full_run["world"]
    b2 = _oracle_b2(world)
    flags = {(f["arm"], f["baseline"]): f for f in doc["headline_flags"]}
    assert sorted(flags) == sorted((arm, base) for arm in ("Q", "Q_raw") for base in ("B1", "B2", "B3"))
    for arm in ("Q", "Q_raw"):
        rows = doc["arms"][arm]["conditional"]["rows"]
        for base in ("B1", "B2", "B3"):  # B1 and B3 score every paired event here; B2 only the oracle's events
            kept = [r for r in rows if base != "B2" or r["event_id"] in b2]
            y = [r["y"] for r in kept]
            arm_mean = _mean_pinball(y, [[r[k] for k in Q_KEYS] for r in kept])
            flag = flags[(arm, base)]
            assert flag["n"] == len(kept), (arm, base)
            assert flag["arm_mean_pinball"] == pytest.approx(arm_mean, rel=1e-9)
            if base == "B2":
                base_mean = _mean_pinball(y, [b2[r["event_id"]] for r in kept])
                assert flag["baseline_mean_pinball"] == pytest.approx(base_mean, rel=1e-9)
            assert flag["lost"] == (arm_mean > flag["baseline_mean_pinball"]), (arm, base)


def test_arm_s_publishes_status_and_scale_floor_shares_per_year(full_run):
    doc = full_run["doc"]
    assert sorted(doc["arm_s"]) == sorted(YEARS)
    assert {year: doc["arm_s"][year]["status"] for year in YEARS} == {year: "ok" for year in YEARS}
    # 2022 is the first year: no donors, nothing scored, so both shares are NaN (null in the JSON).
    assert doc["arm_s"]["2022"]["floor"] == {"events": 0, "events_bound": 0, "donors": 0, "donors_bound": 0,
                                             "events_share": None, "donors_share": None}
    for year in YEARS:
        floor = doc["arm_s"][year]["floor"]
        assert floor["events_bound"] == 0 and floor["donors_bound"] == 0  # the scale is the forecast move, >= 4
        assert floor["events_share"] == (0.0 if floor["events"] else None)
        assert floor["donors_share"] == (0.0 if floor["donors"] else None)


def test_inputs_record_the_arm_s_receipts_the_implied_table_and_the_b3_binding(full_run):
    doc, out, world = full_run["doc"], full_run["out"], full_run["world"]
    for year in YEARS:
        receipt = json.loads((out / "fits" / f"arm_s_fit_{year}.json").read_text(encoding="utf-8"))
        assert doc["inputs"]["arm_s_receipt_hashes"][year] == receipt["content_hash"]
    implied = [[e, world["dates"][e], world["implied"][e]] for e in sorted(world["implied"])]
    assert doc["inputs"]["b2_implied_hash"] == content_hash(implied)
    assert doc["inputs"]["b3"] == {"binding": B3_BINDING, "b3_candidate_id": "B3", "m3_candidate_id": "M3",
                                   "rule": "v14_mae_winner", "years": B3_YEARS,
                                   "series_hash": _digest(_b3(world)["series"])}


def test_b0_labels_hash_covers_the_available_date(monkeypatch, tmp_path, full_run):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    _install(monkeypatch, world)
    event = world["events"][-1]["id"]
    changed = (date.fromisoformat(world["dates"][event]) + timedelta(days=5)).isoformat()  # still after the event
    doc = _run(world, out, guard, avail={event: changed})
    labels = world["labels"]
    b0 = sorted([e, float(labels.at[e, "abs_move"]), float(labels.at[e, "crush_pct_iv30"]),
                 changed if e == event else (date.fromisoformat(world["dates"][e]) + timedelta(days=1)).isoformat()]
                for e in labels.index)
    assert doc["inputs"]["b0_labels_hash"] == content_hash(b0)
    assert doc["inputs"]["b0_labels_hash"] != full_run["doc"]["inputs"]["b0_labels_hash"]


def test_b1_and_b2_do_not_depend_on_crush_or_its_availability(monkeypatch, tmp_path, full_run):
    """B1 and B2 donors carry only the move label. Crush is missing for all of 2021 (read_labels drops those rows) and
    delayed for 2020 H2; B1 and B2 pooled n and mean pinball stay those of the full run. B0 may change, so it is not
    compared."""
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    _install(monkeypatch, world)
    no_crush = {e["id"] for e in world["events"] if e["date"].startswith("2021")}
    late = {e["id"]: (date.fromisoformat(e["date"]) + timedelta(days=5)).isoformat() for e in world["events"]
            if "2020-07-01" <= e["date"] <= "2020-12-31"}
    doc = _run(world, out, guard, nan_crush=no_crush, avail=late)
    for name in ("B1", "B2"):
        want = next(t for t in full_run["doc"]["reported"][name]["tables"] if t["slice"] == "pooled")
        got = next(t for t in doc["reported"][name]["tables"] if t["slice"] == "pooled")
        assert want["n"] > 0 and got["n"] == want["n"], name
        assert got["mean_pinball"] == pytest.approx(want["mean_pinball"], rel=1e-12), name


def test_b1b2_labels_hash_binds_every_sealed_event_move_and_availability(full_run):
    world, doc = full_run["world"], full_run["doc"]
    want = content_hash(sorted([e["id"], world["y"][e["id"]], _next_day(world, e["id"])] for e in world["events"]))
    assert doc["inputs"]["b1b2_labels_hash"] == want


def test_every_arm_s_fit_failing_still_publishes_with_s_not_testable(monkeypatch, tmp_path, full_run):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    _install(monkeypatch, world, s_regress=_fake_fit(boom=True))
    doc = _run(world, out, guard)
    assert json.loads((out / REPORT).read_text(encoding="utf-8")) == doc
    assert doc["status"] == full_run["doc"]["status"] and doc["arm_verdicts"] == full_run["doc"]["arm_verdicts"]
    assert doc["reported"]["S"]["status"] == "not_testable"
    pooled = next(t for t in doc["reported"]["S"]["tables"] if t["slice"] == "pooled")
    assert pooled["n"] == 0
    assert {year: doc["arm_s"][year]["status"] for year in doc["arm_s"]} == {year: "fit_failed" for year in YEARS}


def test_arm_s_fits_before_any_evaluation_label_read_or_b0_label_fetch(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    order, fetched, marks = [], [], []

    def logged_read(sealed, event_ids, labels_before):
        if labels_before == "2026-02-01":
            order.append("evaluation_labels")
        return pd.DataFrame({"event_id": list(event_ids), "absolute_move_pct": [world["y"][e] for e in event_ids],
                             "target_available_on": [_next_day(world, e) for e in event_ids]})
    _install(monkeypatch, world, read=logged_read)
    real = stage_a.fit_arm_s

    def logged_fit(*args):
        order.append("arm_s")
        marks.append(len(fetched))
        return real(*args)
    monkeypatch.setattr(stage_a, "fit_arm_s", logged_fit)
    _run(world, out, guard, calls=fetched)
    assert order.index("arm_s") < order.index("evaluation_labels")
    labels = _label_fetches(fetched)
    assert labels and fetched.index(labels[0]) >= marks[0]


def test_arm_s_refusal_propagates_and_publishes_no_report(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    _install(monkeypatch, world)
    error = fail("EXPERIMENT_IDENTITY_CONFLICT", "raw_receipt_mismatch: refused",
                 details={"reason": "raw_receipt_mismatch"})

    def refuse(*_args):
        raise error
    monkeypatch.setattr(stage_a, "fit_arm_s", refuse)
    with pytest.raises(OpsError) as exc:
        _run(world, out, guard)
    assert exc.value is error
    assert not (out / REPORT).exists() and not list(out.glob("*.tmp"))


def test_q_walk_fit_failure_never_calls_arm_s(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    _install(monkeypatch, world, regress=_fake_fit(boom=True))

    def never(*_args):
        raise AssertionError("arm S must not run when the Q walk failed")
    monkeypatch.setattr(stage_a, "fit_arm_s", never)
    doc = _run(world, out, guard)
    assert doc["status"] == "not_testable" and doc["arm_s"] == {} and doc["reported"] == {}
    assert doc["headline_flags"] == [] and doc["crps_rows"] == []
    assert doc["inputs"]["arm_s_receipt_hashes"] == {} and doc["inputs"]["b3"]["rule"] == "v14_mae_winner"


def test_unbound_implied_move_refuses_before_any_walk_fit(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    world["sealed"]["model_columns"] = ["f_mu", "f_s", "f_empty"]
    reads = _spy(monkeypatch, world)
    with pytest.raises(OpsError) as exc:
        _run(world, out, guard)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "implied_move_unbound")
    assert reads == [] and not out.exists()


def test_unbound_b3_year_is_not_testable_and_the_report_still_publishes(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    _install(monkeypatch, world)
    doc = _run(world, out, guard, monkeypatch=monkeypatch,
               b3=_b3(world, years={"2023": "bound", "2024": "missing", "2025": "bound"}))
    assert json.loads((out / REPORT).read_text(encoding="utf-8")) == doc
    table = doc["reported"]["B3"]
    assert table["status"] == "not_testable"
    assert [d["n"] for d in table["drops"] if d["year"] == 2024 and d["arm_reason"] == "not_testable"] == [1800]
    pooled = next(t for t in table["tables"] if t["slice"] == "pooled")
    assert pooled["n"] == 2700  # 2023 H2 and 2025 only
    bound = [r for r in doc["arms"]["Q"]["conditional"]["rows"] if r["quarter"][:4] in ("2023", "2025")]
    flag = next(f for f in doc["headline_flags"] if f["arm"] == "Q" and f["baseline"] == "B3")
    assert flag["n"] == len(bound)
    assert doc["inputs"]["b3"]["years"] == {"2023": "bound", "2024": "missing", "2025": "bound"}


def test_b3_with_every_year_unbound_and_an_empty_series_still_publishes(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    _install(monkeypatch, world)
    empty = pd.Series([], dtype=float, index=pd.Index([], name="event_id"))
    b3 = {**_b3(world), "years": {"2023": "missing", "2024": "missing", "2025": "missing"}, "series": empty,
          "series_hash": content_hash([])}
    doc = _run(world, out, guard, b3=b3, monkeypatch=monkeypatch)
    assert json.loads((out / REPORT).read_text(encoding="utf-8")) == doc
    assert doc["reported"]["B3"]["status"] == "not_testable"
    pooled = next(t for t in doc["reported"]["B3"]["tables"] if t["slice"] == "pooled")
    assert pooled["n"] == 0
    flag = next(f for f in doc["headline_flags"] if f["arm"] == "Q" and f["baseline"] == "B3")
    assert flag["n"] == 0 and flag["lost"] is None and flag["arm_mean_pinball"] is None  # NaN is null in JSON


def _nonnull_mean(rows, name):
    """Mean of the non-null values of one column over the rows (null is None or NaN)."""
    values = [r[name] for r in rows if r[name] is not None and not math.isnan(r[name])]
    return float(np.mean(values))


def test_crps_rows_hold_one_record_per_in_period_event_with_all_seven_frames(full_run, expected):
    doc, world = full_run["doc"], full_run["world"]
    in_period = sorted(e["id"] for e in world["events"] if "2023-07-01" <= e["date"] <= "2025-12-31")
    assert sorted(r["event_id"] for r in doc["crps_rows"]) == in_period
    assert all(set(r) == {"event_id", "Q", "Q_raw", "B0", "S", "B1", "B2", "B3"} for r in doc["crps_rows"])
    # Q and B0 are scored on the Q arm's paired ids: their crps_rows means equal Q's pooled crps and b0_crps.
    paired = {r["event_id"] for r in doc["arms"]["Q"]["conditional"]["rows"]}
    paired_rows = [r for r in doc["crps_rows"] if r["event_id"] in paired]
    q_pooled = next(t for t in doc["arms"]["Q"]["tables"] if t["slice"] == "pooled")
    assert _nonnull_mean(paired_rows, "Q") == pytest.approx(q_pooled["crps"], rel=1e-9)
    assert _nonnull_mean(paired_rows, "B0") == pytest.approx(q_pooled["b0_crps"], rel=1e-9)
    # B1 and B3 score every in-period event, so their crps_rows means equal their reported pooled crps.
    for name in ("B1", "B3"):
        pooled = next(t for t in doc["reported"][name]["tables"] if t["slice"] == "pooled")
        assert _nonnull_mean(doc["crps_rows"], name) == pytest.approx(pooled["crps"], rel=1e-9), name
    # B0 per event: each row's B0 CRPS is the oracle's for that event (Q has no per-event oracle here).
    for row in doc["crps_rows"]:
        want = expected["crps"]["B0"][row["event_id"]]
        if want is None:
            assert row["B0"] is None, row["event_id"]
        else:
            assert row["B0"] == pytest.approx(want, rel=1e-8, abs=1e-9), row["event_id"]


def test_q_table_m3_column_is_the_b3_series_under_the_v14_rule(full_run):
    world, doc = full_run["world"], full_run["doc"]
    series = _b3(world)["series"]
    m3 = {event: value for event, value in zip(series.index, series.to_numpy(float))}
    ids = [r["event_id"] for r in doc["arms"]["Q"]["conditional"]["rows"]]
    assert all(i in m3 for i in ids)
    row = next(t for t in doc["arms"]["Q"]["tables"] if t["slice"] == "pooled")
    assert row["n_m3"] == row["n"] == len(ids) > 0
    assert row["mae_m3"] == pytest.approx(float(np.mean([abs(world["y"][i] - m3[i]) for i in ids])), rel=1e-9)


def test_lowest_inner_rule_leaves_the_q_m3_column_empty(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    _install(monkeypatch, world)
    doc = _run(world, out, guard, b3={**_b3(world), "rule": "lowest_inner_mae_hmosxlp"}, monkeypatch=monkeypatch)
    row = next(t for t in doc["arms"]["Q"]["tables"] if t["slice"] == "pooled")
    assert row["n_m3"] == 0 and row["mae_m3"] is None


@pytest.fixture(scope="module")
def expected(full_run):
    """Oracle quantiles and CRPS of B0, S, B1, B2 and B3 per event, from the design rules in plain numpy (no production
    code). The quantiles cover the events dated 2022+ that each arm scores; the CRPS covers the in-period events, None
    where an arm has no quantile or its month's tail pool is thin."""
    world = full_run["world"]
    in_period = [e["id"] for e in world["events"] if _in_period(e["date"])]
    b0 = _b0_oracle(world)
    series = _b3(world)["series"]
    b3 = _b0_oracle(world, pd.DataFrame({"pred_move": series.to_numpy(float)}, index=series.index))
    quantiles = {"B0": {e["id"]: b0(e["id"])[0] for e in world["events"] if e["date"] >= "2022-01-01"},
                 "S": _s_quantiles(world), "B1": _b1_quantiles(world), "B2": _oracle_b2(world),
                 # an event with no earlier series donors is unscored (refused_draw): left out, so not a tail donor either
                 "B3": {i: q for i in series.index if (q := b3(i)[0]) is not None}}
    crps = {name: _arm_crps(world, q, in_period) for name, q in quantiles.items()}
    return {"in_period": in_period, "quantiles": quantiles, "crps": crps}


def _scored(expected, name):
    """The in-period events an arm scores (each has an oracle quantile), in event order."""
    return [i for i in expected["in_period"] if i in expected["quantiles"][name]]


ARMS_NEW = ["S", "B1", "B2", "B3"]


@pytest.mark.parametrize("name", ARMS_NEW)
def test_crps_rows_match_the_oracle_per_event(full_run, expected, name):
    got = {r["event_id"]: r[name] for r in full_run["doc"]["crps_rows"]}
    want = expected["crps"][name]
    assert sorted(got) == sorted(expected["in_period"])
    for event_id in expected["in_period"]:
        if want[event_id] is None:
            assert got[event_id] is None, (name, event_id)
        else:
            assert got[event_id] is not None, (name, event_id)
            assert got[event_id] == pytest.approx(want[event_id], rel=1e-8, abs=1e-9), (name, event_id)
    values = [v for v in got.values() if v is not None]
    assert len(values) > 1000 and len(set(values)) > 1, name


@pytest.mark.parametrize("name", ARMS_NEW)
def test_reported_pooled_row_matches_the_oracle(full_run, expected, name):
    world, doc = full_run["world"], full_run["doc"]
    ids = _scored(expected, name)
    assert all(i in expected["quantiles"]["B0"] for i in ids)  # every event it scores is scored by B0 too
    row = next(t for t in doc["reported"][name]["tables"] if t["slice"] == "pooled")
    assert row["n"] == len(ids) > 0
    q = [expected["quantiles"][name][i] for i in ids]
    assert row["mean_pinball"] == pytest.approx(_mean_pinball([world["y"][i] for i in ids], q), rel=1e-9)
    crps = [expected["crps"][name][i] for i in ids if expected["crps"][name][i] is not None]
    assert row["n_crps"] == len(crps)
    assert row["crps"] == pytest.approx(float(np.mean(crps)), rel=1e-8)


@pytest.mark.parametrize("name", ARMS_NEW)
def test_reported_quarter_row_matches_the_oracle(full_run, expected, name):
    world = full_run["world"]
    ids = [i for i in _scored(expected, name) if "2024-04-01" <= world["dates"][i] <= "2024-06-30"]
    row = next(t for t in full_run["doc"]["reported"][name]["tables"]
               if t["slice"] == "quarter" and t["label"] == "2024Q2")
    assert row["n"] == len(ids) > 0
    q = [expected["quantiles"][name][i] for i in ids]
    assert row["mean_pinball"] == pytest.approx(_mean_pinball([world["y"][i] for i in ids], q), rel=1e-9)


