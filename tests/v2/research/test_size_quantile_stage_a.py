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

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import lib
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
PRODUCTION_BINDING, TABLE_HASH = "sha256:" + "d" * 64, "sha256:" + "e" * 64
ALT_BINDING = "sha256:" + "f" * 64


def _candidates():
    return [{"candidate_id": f"h{leaves}_{leaf}", "feature_recipe": {"subset": "HMOSXLP"},
             "recipe": {"schema_version": "magnitude_ridge_hgb_recipe.v1", "model": "history_hgb",
                        "estimator": {"params": {**FIXED, "max_leaf_nodes": leaves, "min_samples_leaf": leaf,
                                                 "loss": "quantile", "quantile": 0.5}}}}
            for leaves in (7, 15) for leaf in (300, 1000)]


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
    production = {"frame": pd.DataFrame({"pred_move": [e["mu"] + (3.0 if e["date"] >= "2023-07-01" else 0.0) for e in events],
                                         "pred_crush": 0.0,
                                         "event_date": [e["date"] for e in events]}, index=ids),
                  "binding": PRODUCTION_BINDING, "table_hash": TABLE_HASH}
    labels = pd.DataFrame({"abs_move": [e["y"] for e in events], "crush_pct_iv30": 0.0}, index=ids)
    return {"events": events, "mu": {e["id"]: e["mu"] for e in events}, "s": {e["id"]: e["s"] for e in events},
            "y": {e["id"]: e["y"] for e in events}, "sealed": sealed, "production": production, "labels": labels}


def _fake_fit(boom=False):
    def regress(recipe, train_x, y, eval_x, *, columns):
        if boom:
            raise ValueError("boom")
        z = statistics.NormalDist().inv_cdf(recipe["estimator"]["params"]["quantile"])
        values = np.asarray(eval_x, dtype=float)
        return {"predictions": (values[:, 0] + values[:, 1] * z).tolist()}
    return regress


def _install(monkeypatch, world, *, omit=(), regress=None, read=None):
    """Fake the label reader (dropping `omit` on the evaluation read only) and the fit regression."""
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


def _guard(case, *, settled=True):
    """A tmp copy of the real spec (settled unless told otherwise), its PLANNED ledger row and one EXP-190 directory."""
    spec = yaml.safe_load(REAL_SPEC.read_text(encoding="utf-8"))
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


def _run(world, out, guard, *, years=YEARS, production=None):
    return stage_a.run_stage_a(world["sealed"], years, production or world["production"], world["labels"], out,
                               spec_path=guard["spec_path"], ledger_path=guard["ledger_path"],
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


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    case = tmp_path_factory.mktemp("stage_a_full")
    world, guard, out = _world(), _guard(case), case / "out"
    stdout, stderr = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as patch:
        _install(patch, world)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            doc = _run(world, out, guard)
    return {"world": world, "out": out, "doc": doc, "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}


def test_full_run_passes_and_every_checked_number_matches_the_oracle(full_run):
    doc, out, world = full_run["doc"], full_run["out"], full_run["world"]
    assert doc["status"] == "passed" and doc["arm_verdicts"]["Q_raw"] == "pass", _why(doc)
    assert json.loads((out / REPORT).read_text(encoding="utf-8")) == doc
    assert content_hash({k: v for k, v in doc.items() if k != "content_hash"}) == doc["content_hash"]
    assert full_run["stdout"] == ""
    assert "[quantile stage A] start" in full_run["stderr"] and "done in" in full_run["stderr"]
    assert "stage_b_arms" not in doc and "stage_b_arms" not in (out / REPORT).read_text(encoding="utf-8")
    assert doc["not_included"] == ["y0_check", "S", "B1", "B2", "B3", "baseline_crps", "headline_loss_flags",
                                   "b0_conditional_value", "unconditional_width_terciles"]
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


def test_replay_is_identical_and_a_changed_binding_refuses_without_touching_the_report(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    _install(monkeypatch, world)
    first = _run(world, out, guard)
    before = (out / REPORT).read_bytes()
    assert _run(world, out, guard) == first
    assert (out / REPORT).read_bytes() == before
    with pytest.raises(OpsError) as exc:
        _run(world, out, guard, production={**world["production"], "binding": ALT_BINDING})
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "stage_a_report_changed")
    assert (out / REPORT).read_bytes() == before
    assert not list(out.glob("*.tmp"))


def test_unsettled_stage_b_refuses_before_any_label_read_or_fit(monkeypatch, tmp_path):
    world, guard = _world(), _guard(tmp_path, settled=False)
    calls = _spy(monkeypatch, world)
    with pytest.raises(OpsError) as exc:
        _run(world, tmp_path / "out", guard)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "stage_b_pending")
    assert calls == [] and not (tmp_path / "out").exists()


def test_years_must_cover_2023_to_2025_and_end_in_2025(monkeypatch, tmp_path):
    world, guard = _world(), _guard(tmp_path)
    calls = _spy(monkeypatch, world)
    with pytest.raises(OpsError) as exc:
        _run(world, tmp_path / "out", guard, years=["2023", "2024"])
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


def test_in_period_admitted_event_without_a_label_refuses_and_drops_nothing(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    missing = next(e["id"] for e in world["events"] if "2023-07-01" <= e["date"] <= "2025-12-31")
    _install(monkeypatch, world, omit=(missing,))
    with pytest.raises(OpsError) as exc:
        _run(world, out, guard)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "admitted_label_missing")
    assert not (out / REPORT).exists()


def test_fit_failure_publishes_a_not_testable_report(monkeypatch, tmp_path):
    world, guard, out = _world(), _guard(tmp_path), tmp_path / "out"
    _install(monkeypatch, world, regress=_fake_fit(boom=True))
    doc = _run(world, out, guard)
    assert doc["status"] == "not_testable" and "Q_raw:fit_failed" in doc["reasons"]
    assert doc["arms"] == {} and doc["fit_status"] == {"2022": "fit_failed"} and doc["floor_report"] == []
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
