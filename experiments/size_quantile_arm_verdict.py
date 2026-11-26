"""EXP-190 Q5b part 2: conditional tests, pass rule 2, arm and Stage A verdicts (design section 5).

Pure over caller-loaded frames: no file I/O, nothing stored (Q6 stores and receipts), no caller yet (Q6 first).
Uses size_quantile_arm_tables (evaluate_arm, PERIODS, QUARTER_PERIOD, LEVELS), size_quantile_metrics (level_width,
cell_scores, coverage) and size_pnl_contrast.bootstrap_interval. Not computed here: headline loss flags to B1-B3 and
B0's own conditional value (follow-up issue); unconditional width terciles (Q6 computes them from the raw rows).
Raw rows (judge_arm "rows", indexed by paired event id): quarter, period, ticker, bucket, y, QCOLS, b0_ + QCOLS, W, L,
e, level_reason, zero_median, used, W_contribution, W_tercile, L_contribution, L_tercile.
"""
import numpy as np
import pandas as pd

from engine.v2.ops.errors import fail
from engine.v2.research.progress import decision, stage
from experiments.size_pnl_contrast import RESAMPLES, SEED, bootstrap_interval
from experiments.size_quantile_arm_tables import LEVELS, PERIODS, QUARTER_PERIOD, evaluate_arm
from experiments.size_quantile_calibration import QCOLS
from experiments.size_quantile_metrics import cell_scores, coverage, level_width

NAN = float("nan")
TERC = ["period", "tercile", "n", "mean_e", "iqr_y", "cover_p25_p75", "cover_p10_p90"]


def _ci(d, groups, resamples, seed):
    """Ticker bootstrap of the event mean as (lo, hi); (NaN, NaN) under two groups."""
    return (NAN, NAN) if len(set(groups)) < 2 else bootstrap_interval(d, groups, resamples=resamples, seed=seed)[1:]


def _one(score, mask, ids, cells, e, q50, period):
    """One score on the rows picked by mask: contributions, tercile labels, per-cell table, per-period (rho, diff)."""
    idx = np.flatnonzero(mask)
    pe, pc = cell_scores(ids[idx], [cells[i] for i in idx], e[idx], q50[idx], score[idx])
    n = len(mask)
    contrib, terc, used = np.full(n, NAN), np.full(n, NAN), np.zeros(n, bool)
    contrib[idx], terc[idx], used[idx] = pe["contribution"], pe["tercile"], pe["used"]
    pc = pc.assign(period=[QUARTER_PERIOD[c[0]] for c in pc["cell"]])
    per = {}
    for p in PERIODS:
        sel, cp = used & (period == p), pc[(pc["skip"] == "") & (pc["period"] == p)]
        per[p] = (float(contrib[sel].mean()) if sel.any() else NAN,
                  float(np.average(cp["tercile_diff"], weights=cp["n"])) if len(cp) else NAN)
    return contrib, terc, used, pc, per


def _terciles(period, y, e, q, mask, tercile):
    """Per period and tercile of the masked events: n, mean e, realised IQR of y, both coverages."""
    out = []
    for p in PERIODS:
        for k in range(3):
            m = mask & (period == p) & (tercile == k)
            if m.any():
                cov, (q75, q25) = coverage(y[m], q[m], LEVELS), np.quantile(y[m], [0.75, 0.25])
                out.append({"period": p, "tercile": k, "n": int(m.sum()), "mean_e": float(e[m].mean()),
                            "iqr_y": float(q75 - q25), "cover_p25_p75": float(cov["p25_p75"].mean()),
                            "cover_p10_p90": float(cov["p10_p90"].mean())})
    return pd.DataFrame(out, columns=TERC)


def judge_arm(name, arm, b0, events, history, *, raw=None, resamples=RESAMPLES, seed=SEED):
    """Conditional tests and pass rule 2 for one arm, then its verdict (Q and Q_raw only; others None)."""
    try:
        with stage(f"quantile arm verdict {name}"):
            if not isinstance(history, pd.DataFrame) or not {"q50", "y", "label_available"} <= set(history.columns):
                raise ValueError("history must be a DataFrame with q50, y and label_available columns")
            if not isinstance(b0, pd.DataFrame) or "bucket" not in b0.columns:
                raise ValueError("b0 must be a DataFrame with a bucket column")
            res = evaluate_arm(name, arm, b0, events, raw=raw, resamples=resamples, seed=seed)
            pid = res["paired_ids"]
            ev, ap, bp = events.loc[pid], arm.loc[pid], b0.loc[pid]
            bucket = bp["bucket"].to_numpy(float)
            if not np.isfinite(bucket).all():
                raise ValueError("paired events need a finite B0 bucket")
            dates = pd.to_datetime(ev["event_date"])
            quarter = dates.dt.to_period("Q").astype(str).to_numpy()
            period = np.array([QUARTER_PERIOD[x] for x in quarter])
            q, qb = ap[QCOLS].to_numpy(float), bp[QCOLS].to_numpy(float)
            y, ids, tk = ev["y"].to_numpy(float), np.asarray(pid), ev["ticker"].to_numpy()
            e, W = np.abs(y - q[:, 3]), q[:, 5] - q[:, 1]
            month = dates.dt.to_period("M").dt.to_timestamp().to_numpy()
            L, lr = level_width(month, q[:, 3], history["q50"], history["y"], history["label_available"])
            keep, zero = lr == "", q[:, 3] <= 1e-9
            cells = [(x, int(b)) for x, b in zip(quarter, bucket)]
            wc, wt, wu, wpc, wp = _one(W, keep, ids, cells, e, q[:, 3], period)
            lc, lt, _, _, lp = _one(L, keep, ids, cells, e, q[:, 3], period)
            ratio = np.divide(W, q[:, 3], out=np.full(len(W), NAN), where=~zero)
            rc, rt, ru, _, rp = _one(ratio, keep & ~zero, ids, cells, e, q[:, 3], period)
            w_iv, d_iv = _ci(wc[wu], tk[wu], resamples, seed), _ci((wc - lc)[wu], tk[wu], resamples, seed)
            r_iv = _ci(rc[ru], tk[ru], resamples, seed)
            creasons = [f"{p}:no_usable_cell" for p in PERIODS if not (wu & (period == p)).any()]
            if len(set(tk[wu])) < 2:
                creasons.append("conditional_bootstrap_too_few_groups")
            reasons = res["not_testable_reasons"] + creasons
            status = "not_testable" if reasons else "testable"
            ok = all(wp[p][0] > 0 and wp[p][0] > lp[p][0] and wp[p][1] > 0 and wp[p][1] > lp[p][1] for p in PERIODS)
            passed2 = None if status == "not_testable" else bool(ok and w_iv[0] > 0 and d_iv[0] > 0)
            skips = {"level_width_burn_in": int((lr == "level_width_burn_in").sum()),
                     "level_width_empty_bin": int((lr == "level_width_empty_bin").sum()),
                     "zero_median": int((keep & zero).sum())}
            for s in ("cell_small", "constant_e", "zero_e_residual"):
                hit = wpc["skip"] == s
                skips[f"cells_{s}"], skips[f"events_{s}"] = int(hit.sum()), int(wpc.loc[hit, "n"].sum())
            if status == "not_testable":
                res["accuracy"]["passed"] = res["calibration"]["passed"] = None
            verdict = None
            if name in ("Q", "Q_raw"):
                ok_all = res["accuracy"]["passed"] and passed2 and res["calibration"]["passed"]
                verdict = "not_testable" if status == "not_testable" else ("pass" if ok_all else "fail")
            rows = pd.DataFrame({"quarter": quarter, "period": period, "ticker": tk, "bucket": bucket, "y": y,
                                 **{c: q[:, k] for k, c in enumerate(QCOLS)},
                                 **{f"b0_{c}": qb[:, k] for k, c in enumerate(QCOLS)}, "W": W, "L": L, "e": e,
                                 "level_reason": lr, "zero_median": zero, "used": wu, "W_contribution": wc,
                                 "W_tercile": wt, "L_contribution": lc, "L_tercile": lt}, index=pid)
            periods = pd.DataFrame([{"period": p, "W_spearman": wp[p][0], "L_spearman": lp[p][0],
                                     "W_tercile_diff": wp[p][1], "L_tercile_diff": lp[p][1]} for p in PERIODS])
            ratio_out = {"periods": pd.DataFrame([{"period": p, "spearman": rp[p][0], "tercile_diff": rp[p][1]}
                                                  for p in PERIODS]),
                         "interval": r_iv, "terciles": _terciles(period, y, e, q, ru, rt)}
            res.update(status=status, not_testable_reasons=reasons, verdict=verdict, conditional={
                "periods": periods, "W_interval": w_iv, "diff_interval": d_iv, "passed": passed2, "skips": skips,
                "terciles": _terciles(period, y, e, q, wu, wt), "ratio": ratio_out, "rows": rows})
            decision(f"quantile arm verdict {name}", f"skips={skips} conditional_reasons={creasons} status={status}")
            decision(f"quantile arm verdict {name}", f"rule2 passed={passed2} verdict={verdict}")
            return res
    except MemoryError:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "quantile arm verdict ran out of memory") from None


def stage_a_verdict(q, q_raw):
    """Stage A from the Q and Q_raw judge_arm results; None means that arm's own step failed."""
    for res, arm in ((q, "Q"), (q_raw, "Q_raw")):
        if res is not None and (not isinstance(res, dict) or res.get("arm") != arm
                                or res.get("verdict") not in ("pass", "fail", "not_testable")):
            raise ValueError(f"{arm} must be a judge_arm result for {arm} with verdict pass, fail or not_testable")
    arms, reasons = {}, []
    for label, res, missing in (("Q", q, "Q:calibration_failed"), ("Q_raw", q_raw, "Q_raw:fit_failed")):
        arms[label] = "not_testable" if res is None else res["verdict"]
        if res is None:
            reasons.append(missing)
        elif res["verdict"] == "not_testable":
            reasons += [f"{label}:{r}" for r in res["not_testable_reasons"]]
    status = "not_testable" if q_raw is None else "passed" if "pass" in arms.values() else "not_passed"
    stage_b = [a for a in ("Q", "Q_raw") if arms[a] == "pass"] if status == "passed" else []
    decision("quantile stage A", f"status={status} arms={arms} reasons={reasons} stage_b_arms={stage_b}")
    return {"status": status, "reasons": reasons, "arms": arms, "stage_b_arms": stage_b}
