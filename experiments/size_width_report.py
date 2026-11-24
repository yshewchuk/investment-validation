"""EXP-189 report (W6b): REPORT.md and proposed_rule.json assembled read-only from the receipts (design sections 6-7)."""
import json
from pathlib import Path

from experiments import size_pnl_receipts as receipts
from experiments import size_width_errmodel as errmodel
from experiments import size_width_fake as fake
from experiments import size_width_success as success
from experiments import size_width_tuning as tuning
from experiments.size_pnl_report import md_table
from experiments.size_pnl_shortvol import NAMES
from experiments.size_width_sizing import GRID

PERIODS = ("2025", "2023H2", "2024")
TUNED = ("A", "B", "C", "D")
FLAT_KS = (0.75, 1.0, 1.25)
TITLE = "# EXP-189 report: production short-vol width tuning (M0)"
NOT_BLIND = "2025 NOT blind for widths: every 2025 number below was seen while the design was written."
SMOKE = "SMOKE RUN: a scope-bound subset, not a result."
EVIDENCE = ("Evidence, not a change: the board decision is the user's, with the DYN-SV chooser review (EXP-188 Decision 3); "
            "after a change EXP-188 is rerun on the new widths.")
PLACEHOLDER = ("### T8-T9 tolerance bands and fallback\n\nnot available: the driver does not yet run W5's tolerance bands and fallback arm "
               "(EXP-189 slice W6b-3); its tables enter through the `w5_tables` parameter of `assemble`.")
T2_HEADERS = ["structure", "A", "B", "C", "D", "E modal share", "E same_k", "flags"]
T3_HEADERS = ["k", "B score", "B se", "B trades", "C score", "C se", "C trades", "exp_pnl_sim", "declined share 2025"]
T5_HEADERS = ["arm", "period", "trades", "account return %", "return per trade %", "breakeven alpha",
              "harsh alpha-0 return per trade %"]


def _pct(x):
    """``x`` as a percent to 2 decimals, or n/a for None."""
    return "n/a" if x is None else f"{100 * x:.2f}"


def _f3(x):
    """``x`` to 3 decimals, or n/a for None."""
    return "n/a" if x is None else f"{x:.3f}"


def _k(k):
    """A grid k as text, or n/a for None."""
    return "n/a" if k is None else str(k)


def _load(out, name):
    """The verified receipt ``name`` under ``out/receipts``; a missing or tampered one refuses."""
    return receipts.load_receipt(out / "receipts" / name)


def _unit(body, name):
    """The tuning unit of ``body`` that holds structure ``name``."""
    return next(u for u in body.values() if name in u["members"])


def _scored(cell):
    """True when a score cell exists and is scored (not under MIN_TRADES)."""
    return cell is not None and not cell.get("not_scored")


def _scored_pairs(scores):
    """(k, ret) for each scored grid k of one structure's scores."""
    return [(k, scores[str(k)]["ret"]) for k in GRID if _scored(scores.get(str(k)))]


def _argmax(pairs):
    """The k with the highest value among (k, value) pairs whose value is not None, or None."""
    got = [(k, v) for k, v in pairs if v is not None]
    return max(got, key=lambda kv: kv[1])[0] if got else None


def _agree(ks):
    """yes when every available argmax k lies within one grid step, no otherwise, n/a when fewer than two are available."""
    idx = [GRID.index(k) for k in ks if k is not None]
    if len(idx) < 2:
        return "n/a"
    return "yes" if max(idx) - min(idx) <= 1 else "no"


def _flatness(scores):
    """(a, b, c): (best - ret_k) / se at FLAT_KS for the best scored k's se; None where a cell is not scored or se <= 0."""
    scored = {k: scores[str(k)] for k in GRID if _scored(scores.get(str(k)))}
    if not scored:
        return [None, None, None]
    top = scored[max(scored, key=lambda k: scored[k]["ret"])]
    out = []
    for k in FLAT_KS:
        c = scored.get(k)
        out.append(None if c is None or not top["se"] > 0 else (top["ret"] - c["ret"]) / top["se"])
    return out


def _arm_cells(body, name, k):
    """score, se and trades cells of one arm at grid k: omitted when the arm is absent, n/a when not scored."""
    if body is None:
        return ["omitted"] * 3
    c = body[name]["scores"].get(str(k))
    if c is None:
        return ["n/a"] * 3
    if c.get("not_scored"):
        return ["n/a", "n/a", str(c["trades"])]
    return [f"{100 * c['ret']:.2f}", f"{100 * c['se']:.2f}", str(c["trades"])]


def _t1(inner):
    """T1 error model, or a not-available line when the model or calibration is absent."""
    if isinstance(inner["model"], dict) and isinstance(inner["calibration"], dict):
        return errmodel.t1_markdown(inner["model"], inner["calibration"]).rstrip()
    return f"### T1 error model\n\nnot available (c_status={inner['c_status']})."


def _t2(tuned, body_E):
    """T2 chosen factors and T2b arm E's k shares, as two sections."""
    fac = {arm: tuning.factors(arm, body) for arm, body in tuned.items()}
    rows = []
    for n in NAMES:
        ks = [str(fac[arm][n]) if arm in fac else "omitted" for arm in TUNED]
        flags = ";".join(f"{arm}:{flag}" for arm in TUNED if arm in tuned for flag in _unit(tuned[arm], n)["flags"]) or "-"
        rows.append([n, *ks, _f3(body_E[n]["modal_share"]), str(body_E[n]["same_k"]), flags])
    shares = [[n, "; ".join(f"{k}={v:.3f}" for k, v in sorted(body_E[n]["k_share"].items(), key=lambda kv: float(kv[0]))),
               body_E[n]["candidates"], body_E[n]["no_feasible_k"]] for n in NAMES]
    return ["### T2 chosen factors (2023-24 tuning)\n\n" + md_table(T2_HEADERS, rows),
            "### T2b arm E: share of candidate events at each k\n\n"
            + md_table(["structure", "shares", "candidates", "no_feasible_k"], shares)]


def _t3(body_B, body_C, body_E, outer):
    """T3 curves per structure; returns (markdown, flatness at k = 1.0 per structure from the primary curve)."""
    primary = body_C if body_C is not None else body_B
    parts = ["### T3 curves per structure (2023-24, account return % at marks_mid; 2025 NOT blind for widths)"]
    flat1 = {}
    for n in NAMES:
        rows = [[str(k), *_arm_cells(body_B, n, k), *_arm_cells(body_C, n, k), _pct(body_E[n]["curve"].get(str(k))),
                 _f3(outer["declined"].get(n, {}).get(str(k)))] for k in (*GRID, "E")]
        fl = _flatness(primary[n]["scores"])
        flat1[n] = fl[1]
        kB = _argmax(_scored_pairs(body_B[n]["scores"]))
        kC = None if body_C is None else _argmax(_scored_pairs(body_C[n]["scores"]))
        kE = _argmax([(k, body_E[n]["curve"].get(str(k))) for k in GRID])
        parts += [f"#### {n}", md_table(T3_HEADERS, rows),
                  "flatness (best - score)/se at k=0.75, 1.0, 1.25: " + ", ".join(_f3(v) for v in fl),
                  f"argmax k: B={_k(kB)} C={'omitted' if body_C is None else _k(kC)} exp_pnl_sim={_k(kE)}; "
                  f"agree: {_agree([kB, kC, kE])}"]
    return "\n\n".join(parts), flat1


def _t5(labels, merged, stats):
    """T5 production versus the arms, per period."""
    rows = [[label, p, merged[label][p]["account"]["trades"], _pct(merged[label][p]["account"]["total_return"]),
             _pct(stats[label][p]["mid"]), _f3(stats[label][p]["x"]), _pct(stats[label][p]["alpha0"])]
            for label in labels for p in PERIODS]
    return ("### T5 production versus arms ($200,000 account at marks_mid; 2025 NOT blind for widths)\n\n"
            + md_table(T5_HEADERS, rows))


def _t6(labels, merged):
    """T6 per-structure stand-alone accounts in 2025."""
    rows = [[n, *(_pct(merged[label]["2025"]["structures"][n]["total_return"]) for label in labels)] for n in NAMES]
    return ("### T6 per-structure stand-alone accounts (2025, return % at marks_mid; 2025 NOT blind for widths)\n\n"
            + md_table(["structure", *labels], rows))


def assemble(out_dir, *, w5_tables=None):
    """Read the receipts under ``out_dir`` (a missing or tampered one refuses), write REPORT.md and proposed_rule.json,
    and return {"report": path, "recommendation": arm or "no_change"}."""
    out = Path(out_dir)
    inner, outer = _load(out, "inner_summary.json"), _load(out, "outer_summary.json")
    bodies = {arm: _load(out, f"tuning_{arm}.json")["body"] for arm in ("A", "B", "D", "E")}
    if inner["c_status"] == "ok":
        bodies["C"] = _load(out, "tuning_C.json")["body"]
    labels = [label for label in ("production", "A", "B", "C", "D", "E") if label == "production" or label in bodies]
    merged = {label: {**inner["results"][label], **outer["results"][label]} for label in labels}
    stats = {label: {**inner["stats"][label], **outer["stats"][label]} for label in labels}
    body_E = bodies["E"]
    breakevens = {label: {p: stats[label].get(p, {}).get("x") for p in PERIODS} for label in labels}
    table = success.success_table(merged["production"], {label: merged[label] for label in labels if label != "production"},
                                  breakevens, body_E)
    t3, flat1 = _t3(bodies["B"], bodies.get("C"), body_E, outer)
    rec = success.recommend(table)
    if rec in TUNED:
        fac = tuning.factors(rec, bodies[rec])
        rule = success.board_rule(rec, fac, {n: flat1[n] for n in NAMES},
                                  {n: outer["declined"].get(n, {}).get(str(fac[n])) for n in NAMES})
    else:
        rule = success.board_rule(rec, None, {}, {})
    head = "\n\n".join([TITLE, NOT_BLIND, *([SMOKE] if inner["smoke"] else [])])
    rule_text = success.render_rule(rule) + "\n\n" + EVIDENCE
    w5 = PLACEHOLDER if w5_tables is None else w5_tables
    sections = [head, _t1(inner), *_t2({arm: bodies[arm] for arm in TUNED if arm in bodies}, body_E), t3,
                fake.t4_markdown({n: {float(k): cell for k, cell in row.items()} for n, row in inner["anchor"].items()}).rstrip(),
                _t5(labels, merged, stats), _t6(labels, merged), success.render_t7(table), rule_text, w5]
    report = out / "REPORT.md"
    report.write_text("\n\n".join(s.rstrip() for s in sections) + "\n", encoding="utf-8")
    (out / "proposed_rule.json").write_text(json.dumps(rule, indent=2, sort_keys=True), encoding="utf-8")
    return {"report": str(report), "recommendation": rec or "no_change"}
