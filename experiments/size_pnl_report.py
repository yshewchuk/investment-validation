"""Report tables T1-T11 for EXP-188 slice S8b. Research-only: reached from the manual EXP-188 runner (the T11 tables through `size_pnl_driver.run_synthetic`); no production entrypoint calls it.

Pure functions: each takes plain frames, lists or dicts and returns a Markdown
string (a ``### Tn`` heading, an optional note line, then a pipe table) (t3_forecast returns two sections, T3 and T3b).
No I/O, no logging, no engine imports. ``build_report`` joins the sections.

Input contract: the frames are the S5/S7 scored outputs concatenated by the S9 driver, which adds the
`model_id`, `structure`/`rule`, `arm` and `taken` columns the producers do not carry themselves
(`size_pnl_shortvol.score`, `size_pnl_straddle.score`/`decide`), plus `quarter`/`ticker` labels.

Not wired to a production caller yet: the S9 driver (guides/size_decision_pnl_design.md section 9)
builds these inputs, calls each `t*` function and writes `REPORT.md` via `build_report`.
"""

import math

import numpy as np
import pandas as pd

from experiments.size_pnl_contrast import classify, CLEAR_SHARE, DIRECTIONAL_SHARE

ARMS = ("as_is", "retuned")
POOLS = ("common_2023", "native_full_history")
STATUSES = ("taken", "not_taken", "undetermined")
RUNG_ALWAYS = "exit_unquoted_worst_case"


def _require(frame, cols, who):
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"{who}: missing column(s) {missing}")


def _flag(series, who, col):
    if len(series) == 0:
        return series.astype(bool)
    if pd.api.types.is_bool_dtype(series):
        ok = series.notna()
    else:
        ok = series.isin([0, 1, True, False])
    if not ok.all():
        bad = series[~ok].iloc[0]
        raise ValueError(f"{who}: column {col!r} must be bool or 0/1, got {bad!r}")
    return series.astype(bool)


def _finite(series, who, col):
    if len(series) == 0:
        return
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"{who}: column {col!r} must be numeric, got dtype {series.dtype}")
    bad = ~np.isfinite(series.to_numpy(dtype=float))
    if bad.any():
        raise ValueError(f"{who}: column {col!r} has non-finite value {series.iloc[int(np.argmax(bad))]!r}")


def _cell(x):
    if x is None or (isinstance(x, (float, np.floating)) and math.isnan(x)):
        return "-"
    if isinstance(x, (float, np.floating)):
        return f"{x:.4g}"
    return str(x).replace("|", "\\|")


def _md_table(headers, rows):
    if not rows:
        return "(no rows)"
    head = [_cell(h) for h in headers]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _section(title, headers, rows, note=None):
    parts = [f"### {title}"]
    if note:
        parts.append(note)
    parts.append(_md_table(headers, rows))
    return "\n\n".join(parts)


def _per_event(total, model_id, n_events):
    if model_id not in n_events:
        raise ValueError(f"no candidate-event count for model {model_id!r}")
    n = n_events[model_id]
    if (isinstance(n, bool) or not isinstance(n, (int, float, np.integer, np.floating))
            or not math.isfinite(n) or n < 0 or n != int(n)):
        raise ValueError(f"_per_event: candidate-event count for model {model_id!r} "
                         f"must be a non-negative integer, got {n!r}")
    return total / n if n else None


def _years(year_signs):
    marks = {1: "+", -1: "-", 0: "0"}
    return " ".join(f"{y}:{marks[int(np.sign(v))]}" for y, v in sorted(year_signs.items()))


def _interval(iv):
    return f"[{iv[0]:.4g}, {iv[1]:.4g}]"


def t1_roster(roster):
    rows = [[r["id"], r["model"], r["type"], "+".join(r["roles"]),
             f"excluded: {r['excluded_reason']}" if r["excluded_reason"] else "included"]
            for r in roster]
    return _section("T1 Roster", ["id", "model", "type", "roles", "status"], rows)


def t2_population(events):
    _require(events, ["quarter", "in_e"], "t2_population")
    events = events.assign(in_e=_flag(events["in_e"], "t2_population", "in_e"))
    rows = []
    for quarter, g in events.groupby("quarter"):
        kept = int(g["in_e"].sum())
        rows.append([quarter, len(g), kept, len(g) - kept])
    if rows:
        rows.append(["total"] + [sum(r[i] for r in rows) for i in (1, 2, 3)])
    return _section("T2 Population", ["quarter", "candidates", "in_E", "dropped"], rows,
                    note="dropped = candidates not in E (undetermined under some model).")


def t3_forecast(frame, pairs):
    _require(frame, ["model_id", "pred", "actual"], "t3_forecast")
    _finite(frame["pred"], "t3_forecast", "pred")
    _finite(frame["actual"], "t3_forecast", "actual")
    g = frame.assign(above=frame["actual"] > frame["pred"]).groupby("model_id")
    per = pd.DataFrame({"n": g.size(), "mean_forecast": g["pred"].mean(), "share": g["above"].mean()})
    rows = [[m, int(r["n"]), r["mean_forecast"], r["share"]] for m, r in per.iterrows()]
    means = per["mean_forecast"].to_dict()
    ratios = []
    for pair_id, mean_id, median_id in pairs:
        if mean_id in means and median_id in means:
            base = means[mean_id]
            ratio = means[median_id] / base if base else None
        else:
            ratio = "not_testable"
        ratios.append([pair_id, mean_id, median_id, ratio])
    first = _section("T3 Forecast calibration",
                     ["model_id", "n", "mean_forecast", "actual_above_forecast_share"], rows,
                     note="actual_above_forecast_share = mean of (realised abs move > forecast).")
    second = _section("T3b Mean forecast ratio, median-type / mean-type",
                      ["pair_id", "mean_type", "median_type", "ratio"], ratios)
    return first + "\n\n" + second


def t4_t5_shortvol(scored, n_events, tuning, *, arm):
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    who = "t4_t5_shortvol"
    _require(scored, ["model_id", "arm", "taken", "pnl_dollars"], who)
    sub = scored[scored["arm"] == arm]
    sub = sub.assign(taken=_flag(sub["taken"], who, "taken"))
    taken = sub[sub["taken"]]
    _finite(taken["pnl_dollars"], who, "pnl_dollars")
    rows = []
    # score() returns taken trades only; models with zero scored rows still get a row.
    for m in sorted(set(sub["model_id"]) | set(n_events)):
        pnl = taken[taken["model_id"] == m]["pnl_dollars"]
        total = float(pnl.sum())
        total_cell = f"{total:.2f}"
        per_event = _per_event(total, m, n_events)
        hit = float((pnl > 0).mean()) if len(pnl) else None
        if arm == "as_is":
            rows.append([m, 1.0, len(pnl), total_cell, per_event, hit])
            continue
        body = tuning.get(m)
        if body is None:
            raise ValueError(f"no tuning receipt for model {m!r}")
        skip = body["skip_reason"] if body["skipped"] else "-"
        rows.append([m, body["c"], len(pnl), total_cell, per_event, hit, skip])
    headers = ["model_id", "c", "trades", "total_pnl", "pnl_per_event", "hit_rate"]
    if arm == "retuned":
        headers.append("skip_note")
    title = "T4 Short-vol, as is" if arm == "as_is" else "T5 Short-vol, retuned"
    return _section(title, headers, rows,
                    note="pnl_per_event = total pnl / candidate events in E (untaken rows count 0). "
                         "hit_rate = share of taken trades with pnl > 0.")


def t6_straddle(scored, decisions, n_events):
    _require(scored, ["model_id", "rule", "pnl_dollars"], "t6_straddle scored")
    _require(decisions, ["model_id", "rule", "status"], "t6_straddle decisions")
    _finite(scored["pnl_dollars"], "t6_straddle", "pnl_dollars")
    bad = set(decisions["status"]) - set(STATUSES)
    if bad:
        raise ValueError(f"t6_straddle: unknown decision status {sorted(bad, key=str)} in column 'status'")
    rules = set(decisions["rule"])
    decided = set(decisions["model_id"])
    pairs = set(zip(decisions["model_id"], decisions["rule"]))
    pairs |= set(zip(scored["model_id"], scored["rule"]))
    # models with candidate events but no decisions get a zero row for every decided rule
    pairs |= {(m, r) for m in n_events if m not in decided for r in rules}
    rows = []
    for m, rule in sorted(pairs):
        g = decisions[(decisions["model_id"] == m) & (decisions["rule"] == rule)]
        counts = [int((g["status"] == s).sum()) for s in STATUSES]
        mine = scored[(scored["model_id"] == m) & (scored["rule"] == rule)]
        pnl = float(mine["pnl_dollars"].sum())
        rows.append([m, rule, *counts, _per_event(pnl, m, n_events)])
    return _section("T6 Straddle buy rules",
                    ["model_id", "rule", "taken", "not_taken", "undetermined", "pnl_per_event"], rows,
                    note="undetermined is reported separately and never folded into not_taken.")


def t7_contrasts(contrasts):
    headers = ["hypothesis", "pair_id", "arm", "status", "n", "mean", "quarters +/-/0",
               "share_positive", "year_signs", "quarter_90%", "ticker_90%"]
    rows = []
    for key in sorted(contrasts):
        hyp, pair_id, arm = key
        s = contrasts[key]
        if s is None:
            rows.append([hyp, pair_id, arm, "not_testable"] + [None] * 7)
            continue
        qc = s["quarter_counts"]
        total = sum(qc.values())
        share = qc["positive"] / total if total else None
        rows.append([hyp, pair_id, arm, "ok", s["n_events"], s["mean"],
                     f"{qc['positive']}/{qc['negative']}/{qc['zero']}", share,
                     _years(s["year_signs"]), _interval(s["quarter_interval"]),
                     _interval(s["ticker_interval"])])
    return _section("T7 Contrasts", headers, rows,
                    note="Intervals are 90% bootstrap intervals; share_positive = positive quarters / all quarters.")


def t8_rungs(shares, *, models=None):
    rungs = sorted({RUNG_ALWAYS}.union(*(d.keys() for d in shares.values())))
    ids = sorted(set(shares) | (set() if models is None else set(models)))
    rows = [[m] + [shares.get(m, {}).get(r) for r in rungs] for m in ids]
    return _section("T8 Exit rungs", ["model_id", *rungs], rows,
                    note=f"Share per exit rung; {RUNG_ALWAYS} is always shown.")


def t9_pool_sensitivity(rows):
    table = {}
    for r in rows:
        if r["pool"] not in POOLS:
            raise ValueError(f"t9_pool_sensitivity: unknown pool {r['pool']!r}")
        if r["pool"] in table.get(r["model_id"], {}):
            raise ValueError(f"t9_pool_sensitivity: duplicate (model_id, pool) pair "
                             f"{(r['model_id'], r['pool'])!r}")
        v = r["pnl_per_event"]
        if v is not None and not (isinstance(v, (int, float, np.integer, np.floating))
                                  and math.isfinite(v)):
            raise ValueError(f"t9_pool_sensitivity: column 'pnl_per_event' must be finite or None, got {v!r}")
        table.setdefault(r["model_id"], {})[r["pool"]] = v
    out = []
    for m in sorted(table):
        common = table[m].get("common_2023")
        native = table[m].get("native_full_history")
        diff = None if common is None or native is None else native - common
        out.append([m, common, native, diff])
    return _section("T9 Pool sensitivity",
                    ["model_id", "common_2023", "native_full_history", "native_minus_common"], out,
                    note="Production-only labelled sensitivity: pnl per candidate event under each pool; "
                         "not a basis for choosing a SIZE winner.")


def t10_verdicts(hypotheses):
    rows = [[name, f"{sign:+d}", classify(pairs, sign)]
            for name, (sign, pairs) in hypotheses.items()]
    note = (f"The {CLEAR_SHARE:.0%} and {DIRECTIONAL_SHARE:.0%} quarter shares are the author's "
            "judgement calls, not derived. Categories use signs and interval exclusion only "
            "(consistency), never a P&L magnitude; results do not change any SIZE winner.")
    return _section("T10 Verdicts", ["hypothesis", "predicted_sign", "category"], rows, note=note)


def t11_synthetic(result):
    stop = ", ".join(result["stop"]) or "none"
    first = _section("T11 SYNTHETIC calibration, X-table", ["id", "scope", "result", "detail"], result["x"],
                     note=f"SYNTHETIC: controls only, never a roster or SIZE result (power read-across deferred to #2685); stop on a failed X1, X2, X6 or X7: {stop}")
    second = _section("T11 SYNTHETIC slopes", ["type", "unit", "metric", "slope", "lo_90", "hi_90"], result["slopes"],
                      note="dollars per event per error point; quarter bootstrap 90% interval")
    means = _section("T11 SYNTHETIC means", ["model_id", "family", "params", "unit", "n_events", "taken", "not_taken",
                                             "mean_pnl_per_event", "rmse", "mae", "declined_share"], result["means"],
                     note="mean_pnl_per_event = mean P&L per event of E (untaken rows count 0). The quarterly series behind each mean is "
                          "the published decisions_synthetic.parquet.")
    contrasts = _section("T11 SYNTHETIC contrasts", ["unit", "rung_s", "mean", "quarters +/-/0", "quarter_lo_90", "quarter_hi_90",
                                                     "ticker_lo_90", "ticker_hi_90", "year_signs"], result["contrasts"],
                         note="median-type minus mean-type P&L per event at each rung s >= level_gap_min_s; 90% quarter and ticker bootstrap intervals.")
    return [first, second, means, contrasts]


def build_report(sections):
    return "# EXP-188 REPORT\n\n" + "\n\n".join(sections) + "\n"
