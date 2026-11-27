"""EXP-190 Q6b part 2: reported-only Stage A side (arm S, B1-B3, headline flags). Caller size_quantile_stage_a."""
import numbers

import numpy as np
import pandas as pd

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail
from engine.v2.research.progress import decision
from experiments.size_quantile_arm_tables import evaluate_arm
from experiments.size_quantile_calibration import QCOLS
from experiments.size_quantile_metrics import Rules, pinball

LEVELS, NAN = Rules().levels, float("nan")


def implied_table(sealed):
    """(frame, digest) for B2: event_date and implied_move (None -> NaN) by event id. Refuses unbound inputs."""
    def refuse(why):
        return fail("EXPERIMENT_IDENTITY_CONFLICT", f"implied_move_unbound: {why}",
                    details={"reason": "implied_move_unbound"})
    rows, real = sealed["rows"], lambda v: v is None or (isinstance(v, numbers.Real) and not isinstance(v, bool))
    if "implied_move" not in sealed["model_columns"] or any("implied_move" not in r or not real(r["implied_move"])
                                                             for r in rows):
        raise refuse("implied_move is not bound to every row as None or a real number")
    ids = [r["event_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise refuse("an event id repeats")
    values = [np.nan if r["implied_move"] is None else float(r["implied_move"]) for r in rows]
    values = [v if np.isfinite(v) else np.nan for v in values]  # None, nan, +inf and -inf all become NaN, once
    dates = {e["event_id"]: e["event_date"] for e in sealed["context"]["target_events"]}
    frame = pd.DataFrame({"event_date": [dates[i] for i in ids], "implied_move": values}, ids).rename_axis("event_id")
    consumed = [[i, dates[i], v if np.isfinite(v) else None] for i, v in zip(ids, values)]
    return frame, content_hash(sorted(consumed, key=lambda r: r[0]))


def s_frames(results):
    """(frame, raw, years) of arm S from fit_arm_s's {year: outcome}; years: status and floor with the SCALE shares."""
    ids, rows, raws, years, nan = [], [], [], {}, [NAN] * len(QCOLS)
    for year, out in results.items():
        for e in out["receipt"]["event_ids"]:
            s = out["scored"].get(e)
            if out["status"] != "ok":
                reason = "fit_failed"
            elif s is None:
                reason = out["unscored"][e]
            else:
                reason = None if np.isfinite(s["raw"]).all() and np.isfinite(s["q"]).all() else "non_finite_output"
            ids.append(e)
            rows.append([*(s["q"] if reason is None else nan), reason])
            raws.append(s["raw"] if reason is None else nan)
        fl = out["floor"]
        share = {"events_share": fl["events_bound"] / fl["events"] if fl["events"] else NAN,
                 "donors_share": fl["donors_bound"] / fl["donors"] if fl["donors"] else NAN}
        years[year] = {"status": out["status"], "floor": {**fl, **share}}
    return pd.DataFrame(rows, ids, [*QCOLS, "reason"]), pd.DataFrame(raws, ids, QCOLS), years


def reported_tables(frames, b0, events, s_raw):
    """Per name in S, B1, B2, B3: status, reasons, share, drops and tables from evaluate_arm (no pass flags)."""
    out = {}
    for name in ("S", "B1", "B2", "B3"):
        res = evaluate_arm(name, frames[name], b0, events, raw=s_raw if name == "S" else frames[name])
        out[name] = {k: res[k] for k in ("status", "not_testable_reasons", "share", "drops", "tables")}
    return out


def headline_flags(results, frames, events):
    """Mean pinball of arm and baseline over the paired events the baseline scored; lost: arm strictly worse."""
    rows = []
    for arm in ("Q", "Q_raw"):
        paired, arm_rows = results[arm]["paired_ids"], results[arm]["conditional"]["rows"]
        for bn in ("B1", "B2", "B3"):
            pid = paired[paired.isin(frames[bn].index[frames[bn]["reason"].isna()])]
            y = events.loc[pid, "y"].to_numpy(float)
            mean = lambda q: float(pinball(y, q, LEVELS).mean()) if len(pid) else NAN
            a, b = mean(arm_rows.loc[pid, QCOLS].to_numpy(float)), mean(frames[bn].loc[pid, QCOLS].to_numpy(float))
            lost = bool(a > b) if len(pid) else None
            rows.append(dict(arm=arm, baseline=bn, n=len(pid), arm_mean_pinball=a, baseline_mean_pinball=b, lost=lost))
            decision("quantile headline", f"arm={arm} baseline={bn} n={len(pid)} lost={lost}")
    return rows
