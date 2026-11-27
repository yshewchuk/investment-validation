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
from engine.v2.ops.errors import OpsError
from experiments import lib
from experiments import size_pnl_readers as readers
from experiments import size_quantile_fits as fits
from experiments import size_quantile_prereg as prereg
from experiments import size_quantile_stage_a as stage_a
from experiments import size_quantile_walk as walk

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
    # Outer training lists are the annual rule's: every event dated before the year starts, in event order.
    folds = [{"outer": {"year": y, "training_candidate_event_ids": [
        e["id"] for e in events if e["date"] < f"{y}-01-01"]}} for y in ("2023", "2024", "2025")]
    sealed = {
        "binding": BINDING, "dictionary_hash": DICT, "model_columns": ["f_mu", "f_s", "f_empty"],
        "evaluation_labels_before": "2026-02-01", "events": [{"event_id": i} for i in ids],
        "context": {"target_events": [{"event_id": e["id"], "ticker": e["ticker"], "event_date": e["date"],
                                       "label_end": e["label_end"]} for e in events],
                    "eligibility_plan": {"random_windows": []}},
        "fold_events": [{"event_id": e["id"], "ticker": e["ticker"], "event_date": e["date"],
                         "forecasts": [{"source_session_ceiling": e["date"]}]} for e in events],
        "plan": {"plan_id": "P1", "candidates": _candidates(), "folds": folds},
        "rows": [{"event_id": e["id"], "f_mu": e["mu"], "f_s": e["s"], "f_empty": None} for e in events],
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
            "dates": {e["id"]: e["date"] for e in events}, "admitted": admitted}


PRODUCTION_BINDING = _binding(TABLE_HASH, _world()["frame"])


def _fake_fit(boom=False):
    def regress(recipe, train_x, y, eval_x, *, columns):
        if boom:
            raise ValueError("boom")
        z = statistics.NormalDist().inv_cdf(recipe["estimator"]["params"]["quantile"])
        values = np.asarray(eval_x, dtype=float)
        return {"predictions": (values[:, 0] + values[:, 1] * z).tolist()}
    return regress


def _install(monkeypatch, world, *, omit=(), regress=None, read=None):
    """Fake the label reader (dropping `omit` on the evaluation read only) and the walk's fit regression."""
    def fake_read(sealed, event_ids, labels_before):
        drop = set(omit) if labels_before == "2026-02-01" else set()
        kept = [e for e in event_ids if e not in drop]
        return pd.DataFrame({"event_id": kept, "absolute_move_pct": [world["y"][e] for e in kept]})
    monkeypatch.setattr(walk, "training_labels", read or fake_read)
    monkeypatch.setattr(stage_a, "training_labels", read or fake_read)
    monkeypatch.setattr(fits, "fit_regression_predictions", regress or _fake_fit())


def _spy(monkeypatch, world):
    """Any label read or fit call is recorded and fails the test."""
    calls = []

    def no_read(*_args):
        calls.append("labels")
        raise AssertionError("no label may be read")

    def no_fit(*_args, **_kwargs):
        calls.append("fit")
        raise AssertionError("no fit may run")
    _install(monkeypatch, world, regress=no_fit, read=no_read)
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


def _fake_fetch(world, out, calls, *, frame=None, avail=None, extra=None):
    """The shared store as fetch(columns, ids), with literal column names and event_id plus the requested columns.
    Each call is recorded with whether every fit_<year>.json exists yet. frame replaces the production forecasts,
    avail overrides label availability by id, extra adds one unrequested row to the answer."""
    def fetch(columns, ids):
        calls.append({"columns": tuple(columns), "ids": list(ids),
                      "fits": all((out / "fits" / f"fit_{year}.json").exists() for year in YEARS)})
        if "pred_abs_move" in columns:
            src = world["frame"] if frame is None else frame
            got = {"pred_abs_move": src.loc[ids, "pred_move"].to_numpy(float),
                   "pred_iv_crush_30": src.loc[ids, "pred_crush"].to_numpy(float), "horizon_sessions": [1] * len(ids)}
        else:
            got = {"abs_move": world["labels"].loc[ids, "abs_move"].to_numpy(float),
                   "crush_pct_iv30": world["labels"].loc[ids, "crush_pct_iv30"].to_numpy(float),
                   "available_as_of_date": [(avail or {}).get(e) or (date.fromisoformat(world["dates"][e])
                                                                     + timedelta(days=1)).isoformat() for e in ids]}
        answer = pd.DataFrame({"event_id": ids, **{column: got[column] for column in columns}})
        if extra is not None:
            answer = pd.concat([answer, pd.DataFrame({"event_id": [extra], **{column: [0.0] for column in columns}})])
        return answer
    return fetch


def _label_fetches(calls):
    return [call for call in calls if "abs_move" in call["columns"]]


def _run(world, out, guard, *, calls=None, years=YEARS, production_hash=TABLE_HASH, table_hash=TABLE_HASH, **store):
    """Stage A on the fake shared store; calls records every fetch and store goes to _fake_fetch."""
    fetch = _fake_fetch(world, out, [] if calls is None else calls, **store)
    return stage_a.run_stage_a(world["sealed"], years, world["admitted"], fetch, out, production_hash=production_hash,
                               table_hash=table_hash, spec_path=guard["spec_path"], ledger_path=guard["ledger_path"],
                               run_log_path=guard["run_log_path"], experiments_dir=guard["experiments_dir"])


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
    assert doc["not_included"] == ["y0_check", "S", "B1", "B2", "B3", "headline_loss_flags", "b0_conditional_value",
                                   "unconditional_width_terciles"]
    assert sorted(p.name for p in out.iterdir()) == ["fits", REPORT]
    inputs = doc["inputs"]
    assert inputs["production_binding"] == PRODUCTION_BINDING and inputs["production_table_hash"] == TABLE_HASH
    for year in YEARS:
        receipt = json.loads((out / "fits" / f"fit_{year}.json").read_text(encoding="utf-8"))
        assert inputs["fit_receipt_hashes"][year] == receipt["content_hash"]
    labels = world["labels"]
    b0 = sorted([e, float(labels.at[e, "abs_move"]), float(labels.at[e, "crush_pct_iv30"])] for e in labels.index)
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
    assert doc["inputs"]["evaluation_labels_hash"] is None
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


def _b0_oracle(world):
    """B0's quantiles and bucket for each event, from the design rule in plain numpy (no ResidualPool, build_pool or
    pool_quantiles; only the MIN_POOL constant is taken from engine.pnl_sim).
    Donors: every fixture event dated strictly before the event, by event date, as the native pool is. Label
    availability is not a B0 filter (the open question in issue #3047). err_move = donor abs_move - donor pred_move.
    Bucket edges are the earlier donors' pred_move quantiles on production's grid, np.linspace(0, 1, 11)[1:-1]
    (the same float values, not a re-derived 0.1..0.9 grid, which can differ by an ulp and move a donor sitting exactly
    on an edge into another bucket). Membership rule, as in ResidualPool.draw: a value v belongs to bucket
    searchsorted(edges, v, side="right"), so an edge value falls in the bucket ABOVE it, for donors and for the event's
    forecast alike. The event's bucket uses its fixture pred_move (with the +3 drift from 2023-07-01). The draw is the
    donors in that bucket, or every earlier donor when the bucket holds fewer than MIN_POOL."""
    events = world["events"]
    ids = [e["id"] for e in events]
    dates = np.array([e["date"] for e in events])  # ISO dates compare correctly as strings
    pred = world["frame"].loc[ids, "pred_move"].to_numpy(float)
    err = np.array([world["y"][i] for i in ids], dtype=float) - pred
    position = {i: k for k, i in enumerate(ids)}
    grid = np.linspace(0, 1, 11)[1:-1]  # production's bucket grid (buckets=10), bit-for-bit

    def oracle(event_id):
        k = position[event_id]
        earlier = dates < dates[k]
        donor_pred, donor_err = pred[earlier], err[earlier]
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
    assert doc["schema_version"] == "size_quantile_stage_a.v2"
    assert set(doc["arms"]) == {"Q", "Q_raw"} and set(doc["arm_verdicts"]) == {"Q", "Q_raw"}
    assert doc["not_included"] == ["y0_check", "S", "B1", "B2", "B3", "headline_loss_flags", "b0_conditional_value",
                                   "unconditional_width_terciles"]
    assert not {"arm_s", "headline_loss_flags"} & set(doc)
    assert not {"arm_s_receipt_hashes", "b2_implied_hash"} & set(doc["inputs"])
    assert {"crps_exclusions", "unlabelled_in_period", "floor_report"} <= set(doc)


def _b0_pool_oracle(world):
    """B0 quantiles and month pools from the design rule, in plain numpy. Donors are the 2022+ events (the B0 frame's
    rows), dated in [M - 12 months, M) with label_end + 1 day < M, scored by their own B0 quantiles from _b0_oracle.
    A tail with under 30 donors makes the month thin. Returns (quantiles_of, pool_of); pool_of(month) is (d_lo, d_hi)
    or None."""
    oracle = _b0_oracle(world)
    donors = [(date.fromisoformat(e["date"]), date.fromisoformat(e["label_end"]) + timedelta(days=1),
               world["y"][e["id"]], oracle(e["id"])[0]) for e in world["events"] if e["date"] >= "2022-01-01"]
    pools = {}

    def pool_of(month):
        if month not in pools:
            start = month.replace(year=month.year - 1)
            hi, lo = [], []
            for day, ready, y, q in donors:
                if not (start <= day < month and ready < month):
                    continue
                if y > q[6]:
                    hi.append((y - q[6]) / max(q[6] - q[3], 0.5))
                if y < q[0]:
                    lo.append((y - q[0]) / max(q[3] - q[0], 0.5))
            pools[month] = (np.array(lo), np.array(hi)) if min(len(lo), len(hi)) >= 30 else None
        return pools[month]

    return oracle, pool_of


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
    assert len(excl) == len(doc["crps_exclusions"]) == 9
    assert sorted(excl) == sorted((arm, year) for arm in ("Q", "Q_raw", "B0") for year in (2023, 2024, 2025))
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


