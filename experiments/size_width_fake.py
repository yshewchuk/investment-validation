"""EXP-189 W3: seeded fake-forecast draws, tuning C and its receipt, the perfect-forecast anchor (design section 4)."""
import hashlib
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine.pnl_sim import DRAWS
from experiments import size_pnl_receipts as receipts
from experiments import size_width_errmodel as errmodel
from experiments.size_pnl_pool import build_pool
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_width_sweep import INNER, Sweep, in_window
from experiments import size_width_tuning as tuning
from experiments.size_width_tuning import MIN_TRADES, choose, unit_score

N_DRAWS, SEED_TAG = tuning.FAKE_DRAWS, tuning.FAKE_SEED_TAG


def _say(msg):
    print(f"[width_fake] {msg}", file=sys.stderr)


def _rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def draw_seed(event_id, d):
    return int.from_bytes(hashlib.sha256(f"{event_id}|{SEED_TAG}|{d}".encode()).digest()[:8], "big")


def draw_z(event_ids, d):
    """One standard normal per event for draw ``d``; an event's z depends only on (event, draw)."""
    return np.array([np.random.default_rng(draw_seed(str(e), d)).standard_normal() for e in event_ids], dtype=float)


def z_matrix(event_ids, n=N_DRAWS):
    return np.array([draw_z(event_ids, d) for d in range(n)], dtype=float)


def fake_pred(model, event_ids, labels, d):
    """Fake forecasts of draw ``d`` indexed by str event id; NaN where the label is not finite and > 0 (never imputed)."""
    y = labels["abs_move"].reindex(event_ids).to_numpy(dtype=float)
    ok = np.isfinite(y) & (y > 0)
    fake = errmodel.fake_forecast(model, np.where(ok, y, 1.0), draw_z(event_ids, d))
    return pd.Series(np.where(ok, fake, np.nan), index=[str(e) for e in event_ids])


def _run(months, table, labels, pred, sessions, *, draws, names):
    """One full sweep on the forecast Series ``pred``: the pool of a model whose forecasts are ``pred`` (crush stays M0's),
    then the unit score of every (structure, k) on the inner window."""
    pool = build_pool(table.assign(pred_move=table.index.astype(str).map(pred)), labels)
    sweep = Sweep(names=names)
    for chunk, events, output in months:
        sweep.add_month(chunk, events.assign(pred_move=events["event_id"].astype(str).map(pred)), pool,
                        output=output, sessions=sessions, draws=draws)
    return {name: {k: unit_score(sweep.rows([(name, k)], INNER), INNER) for k in sweep.ks} for name in names}


def run_draw(months, table, labels, model, d, sessions, *, draws=DRAWS, names=NAMES):
    """Draw ``d`` of the seeded fake model: the sweep scores of that model's forecasts."""
    return _run(months, table, labels, fake_pred(model, table.index, labels, d), sessions, draws=draws, names=names)


def run_anchor(months, table, labels, sessions, *, draws=DRAWS, names=NAMES):
    """Perfect-forecast anchor: the forecast is the actual move (same gate, marks and exit). Descriptive only."""
    actual = labels["abs_move"].reindex(table.index).to_numpy(dtype=float)
    ok = np.isfinite(actual) & (actual > 0)
    pred = pd.Series(np.where(ok, actual, np.nan), index=[str(e) for e in table.index])
    return _run(months, table, labels, pred, sessions, draws=draws, names=names)


def pool_draws(per_draw):
    """{name: {k: {"ret": mean over draws, "se": sample sd / sqrt(N) (0.0 when N == 1), "trades": sum over draws}}}."""
    n = len(per_draw)
    out = {}
    for name in per_draw[0]:
        out[name] = {}
        for k in per_draw[0][name]:
            rets = np.array([d[name][k]["ret"] for d in per_draw], dtype=float)
            out[name][k] = {"ret": float(np.mean(rets)),
                            "se": float(np.std(rets, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
                            "trades": int(sum(d[name][k]["trades"] for d in per_draw))}
    return out


def tune_c(per_draw):
    """Tuning C: the argmax of the pooled draws per structure (W4's choose, MIN_TRADES), in arm B's body shape."""
    return {name: {**choose(scores), "members": [name],
                   "scores": {str(k): {**s, "not_scored": s["trades"] < MIN_TRADES} for k, s in scores.items()}}
            for name, scores in pool_draws(per_draw).items()}


def calibration(model, table, labels, n=N_DRAWS):
    """M0's inner forecasts against their labels, checked with the seeded fake draws on the same rows."""
    inner = in_window(table, INNER)
    ids = inner.index
    label = labels["abs_move"].reindex(ids).to_numpy(dtype=float)
    return errmodel.calibrate(model, inner["pred_move"].to_numpy(dtype=float), label, z_matrix(ids, n))


def write_tuning_c(out, body, inputs_hash, model, cal, n):
    if cal.get("ok") is not True:
        raise Refusal("EXPERIMENT_IDENTITY_CONFLICT", "tuning C refused: the fake forecast is not calibrated to M0",
                      {"reason": "fake_not_calibrated"})
    n = tuning.admit_draws(n)
    if not tuning.fake_model_ok(model):
        raise Refusal("EXPERIMENT_IDENTITY_CONFLICT", "tuning C refused: the fake error model is malformed",
                      {"reason": "fake_model_invalid"})
    return receipts.write_receipt(Path(out) / "receipts" / "tuning_C.json", {
        "experiment": "EXP-189", "arm": "C", "inputs_hash": inputs_hash, "window": "2023H2-2024", "body": body,
        "fake": {"N": n, "seed_tag": SEED_TAG, "model": model, "calibration": cal}})


def run_fake(months, table, labels, model, sessions, out, inputs_hash, *, n=N_DRAWS, draws=DRAWS):
    """Calibrate the fake, else run draws 0..n-1 and write the tuning_C receipt. Returns {"status": "ok", "body", "calibration", "pooled"}
    or {"status": "fake_not_calibrated", "calibration"} (no draws, no receipt)."""
    n = tuning.admit_draws(n)
    if not tuning.fake_model_ok(model):
        raise Refusal("EXPERIMENT_IDENTITY_CONFLICT", "tuning C refused: the fake error model is malformed",
                      {"reason": "fake_model_invalid"})
    started = time.time()
    _say(f"stage start n={n} draws={draws} rss={_rss():.0f}MiB")
    cal = calibration(model, table, labels, n)
    _say(f"calibration ok={cal['ok']} rmse_rel={cal.get('rmse_rel', float('nan')):+.3f} mae_rel={cal.get('mae_rel', float('nan')):+.3f}")
    if not cal["ok"]:
        _say(f"stage end elapsed={time.time() - started:.1f}s status=fake_not_calibrated rss={_rss():.0f}MiB")
        return {"status": "fake_not_calibrated", "calibration": cal}
    per_draw = []
    for d in range(n):
        per_draw.append(run_draw(months, table, labels, model, d, sessions, draws=draws))
        elapsed = time.time() - started
        _say(f"draw {d + 1}/{n} elapsed={elapsed:.1f}s eta={elapsed / (d + 1) * (n - d - 1):.1f}s "
             f"rate={(d + 1) / max(elapsed, 1e-9) * 60:.1f}draws/min rss={_rss():.0f}MiB")
    body = tune_c(per_draw)
    write_tuning_c(out, body, inputs_hash, model, cal, n)
    _say(f"stage end elapsed={time.time() - started:.1f}s status=ok rss={_rss():.0f}MiB")
    return {"status": "ok", "body": body, "calibration": cal, "pooled": pool_draws(per_draw)}


def t4_markdown(scores):
    """T4 table of the anchor's account return in percent by structure and k; n/a where a cell is under MIN_TRADES."""
    ks = sorted({k for row in scores.values() for k in row})
    lines = ["### T4 anchor (perfect forecast, 2023-24, descriptive only)", "",
             "| structure | " + " | ".join(f"k={k:g}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
    for name, row in scores.items():
        cells = []
        for k in ks:
            s = row.get(k)
            cells.append("n/a" if s is None or s["trades"] < MIN_TRADES else f"{100 * s['ret']:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
