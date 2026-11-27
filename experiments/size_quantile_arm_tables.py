"""EXP-190 Q5b part 1: arm tables, scored share, pass rules 1 (accuracy) and 3 (calibration).

Pure over caller-loaded frames: no file I/O, no stored artifacts, no production caller yet (Q6 first).
Part 1 only: per-rule `passed` flags for pass rules 1 (accuracy) and 3 (calibration), no combined pass verdict.
Rule 2, conditional tests and Stage A verdict are part 2; the 8-of-10 and 5-of-10 quarter counts are design section 5.
"""
import numpy as np
import pandas as pd

from engine.v2.ops.errors import fail
from engine.v2.research.progress import decision, stage
from experiments.size_pnl_contrast import RESAMPLES, SEED, bootstrap_interval
from experiments.size_quantile_calibration import QCOLS
from experiments.size_quantile_metrics import BANDS, Rules, coverage, crossing_rate, pinball

LEVELS = Rules().levels
NAMES = [f"p{round(a * 100)}" for a in LEVELS]
PERIODS = {"2023H2": ("2023Q3", "2023Q4"), "2024": tuple(f"2024Q{k}" for k in range(1, 5)),
           "2025": tuple(f"2025Q{k}" for k in range(1, 5))}
QUARTERS = [q for qs in PERIODS.values() for q in qs]
QUARTER_PERIOD = {q: p for p, qs in PERIODS.items() for q in qs}
MIN_SCORED_SHARE, MIN_ACCURACY_QUARTERS, MIN_CALIBRATION_QUARTERS = 0.80, 8, 5
NAN = float("nan")


def _mean(x):
    return float(np.mean(x)) if len(x) else NAN


def _pair(arm, b0, events, raw):
    """Input checks, then the admitted in-period events and each side's reason per event."""
    frames = [events, arm, b0] + ([] if raw is None else [raw])
    if not {"event_date", "ticker", "y", "m0", "m3"} <= set(events.columns) or not all(
            set(QCOLS + ["reason"]) <= set(f.columns) for f in (arm, b0)) or (
            raw is not None and not set(QCOLS) <= set(raw.columns)):
        raise ValueError("events, arm, b0 and raw need their columns")
    if any(f.index.has_duplicates for f in frames):
        raise ValueError("indexes must be unique")
    if not (set(arm.index) | set(b0.index)) <= set(events.index):
        raise ValueError("arm and b0 ids must all be in events")
    dates = pd.to_datetime(events["event_date"])
    if dates.isna().any():
        raise ValueError("event_date must not be null")
    quarter = dates.dt.to_period("Q").astype(str)
    inper = quarter.isin(QUARTERS)
    ev = events[inper].assign(quarter=quarter[inper], period=quarter[inper].map(QUARTER_PERIOD),
                              year=dates[inper].dt.year)
    y = ev["y"].to_numpy(dtype=float)
    if ev["ticker"].isna().any() or np.isinf(y).any() or (y < 0).any():
        raise ValueError("in-period events need a ticker and non-negative, non-infinite y")
    for frame in (arm, b0):
        ok = frame["reason"].isna().to_numpy()
        if not np.isfinite(frame.loc[ok, QCOLS].to_numpy(dtype=float)).all():
            raise ValueError("scored rows need seven finite quantiles")
    labels = []
    for frame in (arm, b0):
        reason = frame["reason"].reindex(ev.index)
        text = np.where(reason.isna(), "scored", reason.astype(str))
        labels.append(np.where(np.isnan(y), "label_missing", np.where(ev.index.isin(frame.index), text, "no_output")))
    return ev, labels[0], labels[1]


def _row(slice_name, label, m, a):
    """One table row over the paired events picked by mask m (NaN metrics and zero counts when empty)."""
    cov = {k: v[m] for k, v in a["cov"].items()}
    covb = {k: v[m] for k, v in a["covb"].items()}
    y, q, pa, pb = a["y"][m], a["q"][m], a["pa"][m], a["pb"][m]
    crps = a["crps"][m][np.isfinite(a["crps"][m])]
    bc = a["b0crps"][m]
    sel0, sel3 = m & np.isfinite(a["m0"]), m & np.isfinite(a["m3"])
    lv = pa.mean(axis=0) if len(pa) else np.full(len(LEVELS), NAN)
    return {"slice": slice_name, "label": label, "n": int(m.sum()),
            **{f"pinball_{p}": x for p, x in zip(NAMES, lv)},
            "mean_pinball": _mean(pa), "cover_p25_p75": _mean(cov["p25_p75"]),
            "cover_p10_p90": _mean(cov["p10_p90"]), **{f"le_{p}": _mean(cov[f"le_{p}"]) for p in NAMES},
            "median_width": float(np.median(q[:, 5] - q[:, 1])) if len(q) else NAN,
            "mae_p50": _mean(np.abs(y - q[:, 3])),
            "mae_m0": _mean(np.abs(a["y"][sel0] - a["m0"][sel0])), "n_m0": int(sel0.sum()),
            "mae_m3": _mean(np.abs(a["y"][sel3] - a["m3"][sel3])), "n_m3": int(sel3.sum()),
            "crps": _mean(crps), "n_crps": int(crps.size),
            "crossing_rate": crossing_rate(a["raw"][m]) if a["raw"] is not None else NAN,
            "b0_mean_pinball": _mean(pb), "b0_cover_p25_p75": _mean(covb["p25_p75"]),
            "b0_cover_p10_p90": _mean(covb["p10_p90"]),
            "b0_crps": _mean(bc[np.isfinite(bc)]), "b0_n_crps": int(np.isfinite(bc).sum())}


def evaluate_arm(name, arm, b0, events, *, raw=None, resamples=RESAMPLES, seed=SEED):
    """Tables, scored share and pass rules 1 and 3 for one arm against B0; no verdict for the arm."""
    try:
        with stage(f"quantile arm tables {name}"):
            if isinstance(resamples, bool) or not isinstance(resamples, (int, np.integer)) or resamples < 1:
                raise ValueError("resamples must be an integer >= 1")
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
                raise ValueError("seed must be a non-negative integer")
            ev, arm_lab, b0_lab = _pair(arm, b0, events, raw)
            paired = (arm_lab == "scored") & (b0_lab == "scored")
            pid, sub = ev.index[paired], ev[paired]
            rawq = None if raw is None else raw.reindex(pid)[QCOLS].to_numpy(dtype=float)
            if rawq is not None and not np.isfinite(rawq).all():
                raise ValueError("raw needs seven finite values for every paired event")
            y = sub["y"].to_numpy(dtype=float)
            q, qb = (f.loc[pid, QCOLS].to_numpy(dtype=float) for f in (arm, b0))
            crps = arm["crps"].reindex(pid).to_numpy(dtype=float) if "crps" in arm else np.full(len(pid), NAN)
            b0crps = b0["crps"].reindex(pid).to_numpy(dtype=float) if "crps" in b0 else np.full(len(pid), NAN)
            m0, m3 = sub["m0"].to_numpy(dtype=float), sub["m3"].to_numpy(dtype=float)
            pa, pb = pinball(y, q, LEVELS), pinball(y, qb, LEVELS)
            cov, covb = ({k: v.to_numpy() for k, v in coverage(y, x, LEVELS).items()} for x in (q, qb))
            a = {"y": y, "q": q, "pa": pa, "pb": pb, "cov": cov, "covb": covb, "raw": rawq, "crps": crps,
                 "b0crps": b0crps, "m0": m0, "m3": m3,
                 "quarter": sub["quarter"].to_numpy(), "period": sub["period"].to_numpy(),
                 "ticker": sub["ticker"].to_numpy()}
            share = []
            for p in [*PERIODS, "pooled"]:
                pm = np.ones(len(ev), bool) if p == "pooled" else (ev["period"] == p).to_numpy()
                adm, pai = int(pm.sum()), int((paired & pm).sum())
                s = pai / adm if adm else NAN
                share.append([p, adm, pai, s, bool(s < MIN_SCORED_SHARE)])
            reasons = [f"{p}:{r}" for p, _, pai, _, low in share[:-1]
                       for r, hit in (("period_no_paired_events", pai == 0), ("scored_share_below_floor", low)) if hit]
            if len(set(a["quarter"])) < 2 or len(set(a["ticker"])) < 2:
                reasons.append("bootstrap_too_few_groups")
            rows = [_row("pooled", "pooled", np.ones(len(y), bool), a)]
            rows += [_row("period", p, a["period"] == p, a) for p in PERIODS]
            rows += [_row("quarter", qq, a["quarter"] == qq, a) for qq in QUARTERS]
            qmask = [a["quarter"] == qq for qq in QUARTERS]
            d = pa.mean(axis=1) - pb.mean(axis=1)
            periods_below = {p: _mean(a["pa"][a["period"] == p]) < _mean(a["pb"][a["period"] == p]) for p in PERIODS}
            quarters_below = sum(_mean(a["pa"][m]) < _mean(a["pb"][m]) for m in qmask)
            iv = {k: bootstrap_interval(d, a[k], resamples=resamples, seed=seed)[1:] if len(set(a[k])) > 1
                  else (NAN, NAN) for k in ("quarter", "ticker")}
            acc = bool(all(periods_below.values()) and quarters_below >= MIN_ACCURACY_QUARTERS
                       and iv["quarter"][1] < 0 and iv["ticker"][1] < 0) if not reasons else None
            no_worse = {b: sum(abs(_mean(a["cov"][b][m]) - nom) <= abs(_mean(a["covb"][b][m]) - nom) + 1e-12
                               for m in qmask) for b, (_, _, nom) in BANDS.items()}
            cal = bool(all(c >= MIN_CALIBRATION_QUARTERS for c in no_worse.values())) if not reasons else None
            drops = pd.DataFrame({"year": ev["year"].to_numpy(dtype=int)[~paired], "arm_reason": arm_lab[~paired],
                                  "b0_reason": b0_lab[~paired]})
            drops = drops.groupby(["year", "arm_reason", "b0_reason"]).size().reset_index(name="n")
            status = "not_testable" if reasons else "testable"
            decision(f"quantile arm tables {name}",
                     f"admitted={len(ev)} paired={int(paired.sum())} status={status} reasons={reasons}")
            decision(f"quantile arm tables {name}",
                     f"accuracy passed={acc} quarters_below={quarters_below} calibration passed={cal}")
            return {"arm": name, "status": status, "not_testable_reasons": reasons,
                    "share": pd.DataFrame(share, columns=["period", "admitted", "paired", "share", "below_floor"]),
                    "drops": drops, "paired_ids": pid, "tables": pd.DataFrame(rows),
                    "accuracy": {"periods_below": periods_below, "quarters_below": quarters_below,
                                 "quarter_interval": iv["quarter"], "ticker_interval": iv["ticker"],
                                 "passed": acc},
                    "calibration": {"quarters_no_worse": no_worse, "passed": cal}}
    except MemoryError:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "quantile arm tables ran out of memory") from None
