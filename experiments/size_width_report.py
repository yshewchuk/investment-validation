"""EXP-189 report (W6b): REPORT.md and proposed_rule.json assembled read-only from the receipts (design sections 6-7)."""
import json
import os
from pathlib import Path

from engine.v2.foundation import content_hash
from experiments import size_pnl_receipts as receipts
from experiments import size_width_band as band
from experiments import size_width_choices as choices
from experiments import size_width_errmodel as errmodel
from experiments import size_width_fake as fake
from experiments import size_width_f_report_counts as f_report_counts
from experiments import size_width_f_views as f_views
from experiments import size_width_flex as flex
from experiments import size_width_success as success
from experiments import size_width_tuning as tuning
from experiments.size_pnl_report import md_table
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_width_sizing import GRID

_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
IDENTITY_KEYS = ("plan_id", "registration_id", "seal_id", "panel_id")
ARMS = ("A", "B", "C", "D", "E")
PERIODS = ("2025", "2023H2", "2024")
TUNED = ("A", "B", "C", "D")
FLAT_KS = (0.75, 1.0, 1.25)
TITLE = "# EXP-189 report: production short-vol width tuning (M0)"
NOT_BLIND = "2025 NOT blind for widths: every 2025 number below was seen while the design was written."
SMOKE = "SMOKE RUN: a scope-bound subset, not a result."
EVIDENCE = ("Evidence, not a change: the board decision is the user's, with the DYN-SV chooser review (EXP-188 Decision 3); "
            "after a change EXP-188 is rerun on the new widths.")
F_HEAD = "POST-HOC (Amendment 1, F fills): chosen after the full run under mid fills had been seen; not a pre-registered test and nothing in it is blind. The original run stays EXP-189's only pre-registered result."
PLACEHOLDER = "### T8-T9 tolerance bands and fallback\n\nnot available: these stage summaries carry no tolerance-band or fallback results."
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


def _digest(receipt):
    """content_hash of a loaded receipt's body, without its own content_hash key (the digest the summaries bind)."""
    return content_hash({k: v for k, v in receipt.items() if k != "content_hash"})


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
    """T3 curves per structure; returns (markdown, flat1), where flat1 is the flatness at k = 1.0 per structure from the
    primary curve (C when calibrated, else B). It is what the proposed rule prints as flatness_se for any recommended arm."""
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


def _flat_note(bodies, rec):
    """The proposed rule's flatness_se paragraph for recommended arm ``rec``: T3's k = 1.0 flatness of the primary curve,
    which is arm C's per-structure curve when C is calibrated and present, else arm B's. It is not arm ``rec``'s own
    tuning curve unless ``rec`` is the primary arm, and it is measured at k = 1.0, not at the proposed k."""
    source = "C" if "C" in bodies else "B"
    text = (f"flatness_se: T3's flatness at k = 1.0 on arm {source}'s per-structure curve "
            f"(the primary curve: C when calibrated, else B), whichever arm is recommended")
    return text + "." if source == rec else text + f"; it is not arm {rec}'s own tuning curve."


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


def _conflict(what, reason="tuning_receipt_changed"):
    """The refusal for receipts that do not belong to one driver run; ``what`` names the mismatch."""
    return Refusal(_CONFLICT, f"report receipts do not belong to one run: {what}", {"reason": reason})


def _bound(out, expected):
    """(inner, outer, bodies, f_run): the summaries and inputs receipts checked to belong to one driver run, every arm's tuning receipt
    validated and bound (tuning.frozen), the bodies by arm, and whether the inputs bind F (a fill_model key). Any mismatch refuses before anything is built or written."""
    inner, outer = _load(out, "inner_summary.json"), _load(out, "outer_summary.json")
    inputs_inner, inputs_outer = _load(out, "inputs_inner.json"), _load(out, "inputs_outer.json")
    if inner["stage"] != "inner": raise _conflict("inner_summary.json is not the inner stage")
    if outer["stage"] != "outer": raise _conflict("outer_summary.json is not the outer stage")
    if any(r["inputs_hash"] != inner["inputs_hash"] for r in (outer, inputs_inner, inputs_outer)): raise _conflict("inputs_hash differs")
    if any(r["scope"] != inner["scope"] for r in (outer, inputs_inner, inputs_outer)): raise _conflict("scope differs")
    if inner["smoke"] != outer["smoke"]: raise _conflict("smoke flag differs between the inner and outer summaries")
    if outer.get("inputs_outer") != _digest(inputs_outer): raise _conflict("inputs_outer differs from the inner/outer summary's input digest")
    if inner.get("inputs_inner") != _digest(inputs_inner): raise _conflict("inputs_inner differs from the inner/outer summary's input digest")
    arms = [a for a in ARMS if a != "C" or inner["c_status"] == "ok"]
    if outer["arms"] != arms: raise _conflict(f"outer arms {outer['arms']} are not {arms} for c_status {inner['c_status']}")
    if inputs_outer["identity"] != inputs_inner["identity"]: raise _conflict("inputs_outer identity differs from inputs_inner")
    has_in, has_out = "fill_model" in inputs_inner, "fill_model" in inputs_outer
    if has_in != has_out or inputs_inner.get("fill_model") != inputs_outer.get("fill_model"): raise _conflict("fill_model differs between the inputs receipts")
    f_run = has_in
    if f_run and not (isinstance(inputs_inner["fill_model"], dict) and inputs_inner["fill_model"].get("fill_model_id") == "F"): raise _conflict("fill_model does not bind F")
    bodies = {}
    for arm in outer["arms"]:
        tuning.frozen(out, arm, inner["inputs_hash"])
        bodies[arm] = _load(out, f"tuning_{arm}.json")["body"]
    if expected is not None:
        for key in IDENTITY_KEYS:
            if inputs_inner["identity"].get(key) != expected.get(key):
                raise _conflict(f"--expected {key} differs from the identity bound in the inputs receipts", "expected_mismatch")
    return inner, outer, bodies, f_run


def _choices(out, inner):
    """(inner, outer) arm E choice receipts, None where absent; a present one must be its stage, bound to the summaries' inputs and scope."""
    found = []
    for stage in ("inner", "outer"):
        path = out / "receipts" / f"e_choices_{stage}.json"
        receipt = receipts.load_receipt(path) if path.exists() else None
        if receipt is not None and (receipt.get("stage") != stage or receipt.get("inputs_hash") != inner["inputs_hash"]
                                    or receipt.get("scope") != inner["scope"]):
            raise _conflict(f"e_choices_{stage} receipt differs")
        found.append(receipt)
    return found


def _check_args(out_dir, expected, w5_tables):
    """Refuses a malformed argument before anything is read: out_dir a path, expected None or a dict of str identity keys, w5_tables None or a str."""
    def bad(what):
        return Refusal("INVALID_EXPERIMENT_SPEC", f"report argument {what}", {"reason": "report_input_invalid"})
    if not isinstance(out_dir, (str, os.PathLike)): raise bad(f"out_dir must be a str or os.PathLike, got {type(out_dir).__name__}")
    if expected is not None and (not isinstance(expected, dict) or any(not isinstance(expected.get(k), str) for k in IDENTITY_KEYS)):
        raise bad(f"expected must be None or a dict with a str for each of {IDENTITY_KEYS}, got {type(expected).__name__}")
    if w5_tables is not None and not isinstance(w5_tables, str): raise bad(f"w5_tables must be None or a str, got {type(w5_tables).__name__}")


def assemble(out_dir, *, expected=None, w5_tables=None):
    """Read the receipts under ``out_dir`` (a missing, tampered, or mismatched one refuses before anything is written; ``expected``
    is the identity dict compared with the inputs receipts), write REPORT.md and proposed_rule.json, and return
    {"report": path, "recommendation": arm or "no_change"}."""
    _check_args(out_dir, expected, w5_tables)
    out = Path(out_dir)
    inner, outer, bodies, f_run = _bound(out, expected)
    echoes = _choices(out, inner)
    has_flex = "flex" in inner or "flex" in outer
    if has_flex:
        flex.admit(inner, band.frozen_bands(out, inner["inputs_hash"]), {"production": {n: 1.0 for n in NAMES},
                   **{arm: tuning.frozen(out, arm, inner["inputs_hash"]) for arm in outer["arms"]}}, outer)
    labels = [label for label in ("production", "A", "B", "C", "D", "E") if label == "production" or label in bodies]
    if f_run:
        if not (f_views.admit(inner["results"], labels, ("2023H2", "2024")) and f_views.admit(outer["results"], labels, ("2025",))):
            raise _conflict("the F views in results are missing, malformed, extra or differ from the deciding values")
        if not (f_report_counts.admit(inner["results"], labels, ("2023H2", "2024")) and f_report_counts.admit(outer["results"], labels, ("2025",))):
            raise _conflict("the F counts in results are missing, malformed or inconsistent")
    elif f_views.has_views(inner["results"]) or f_views.has_views(outer["results"]):
        raise _conflict("F views in a run that does not bind F")
    merged = {label: {**inner["results"][label], **outer["results"][label]} for label in labels}
    stats = {label: {**inner["stats"][label], **outer["stats"][label]} for label in labels}
    body_E = bodies["E"]
    t2c, t5c = [], []
    if f_run:
        t2c = [f_report_counts.counts_markdown("T2c F entry non-trades by reason, arm, year and structure "
                                               "(2023-24 tuning years; shares are % of candidates)", labels, ("2023H2", "2024"), merged)]
        t5c = [f_report_counts.counts_markdown("T5c F entry non-trades by reason, arm, year and structure "
                                               "(2025 NOT blind for widths; shares are % of candidates)", labels, ("2025",), merged),
               f_report_counts.nonpositive_markdown(labels, PERIODS, merged)]
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
    if f_run: rule["label"] = f_views.LABEL
    head = "\n\n".join([TITLE, NOT_BLIND, *([SMOKE] if inner["smoke"] else []), *([F_HEAD] if f_run else [])])
    t5 = f_views.t5_markdown(labels, PERIODS, merged, stats) if f_run else _t5(labels, merged, stats)
    t6 = _t6(labels, merged) + (f"\n\n{f_views.t6_markdown(labels, merged)}" if f_run else "")
    flat_note = [_flat_note(bodies, rec)] if rec in TUNED else []
    rule_text = "\n\n".join([success.render_rule(rule), *flat_note, EVIDENCE])
    w5 = w5_tables if w5_tables is not None else flex.render(inner, outer) if has_flex else PLACEHOLDER
    sections = [head, _t1(inner), *_t2({arm: bodies[arm] for arm in TUNED if arm in bodies}, body_E), *t2c, t3,
                fake.t4_markdown({n: {float(k): cell for k, cell in row.items()} for n, row in inner["anchor"].items()}).rstrip(),
                t5, *t5c, *([choices.report_line(*echoes)] if any(e is not None for e in echoes) else []),
                t6, success.render_t7(table), rule_text, w5]
    report = out / "REPORT.md"
    text = "\n\n".join(s.rstrip() for s in sections) + "\n"
    report.write_text(f_views.label_headings(text.replace("at marks_mid", "at F fills")) if f_run else text, encoding="utf-8")
    (out / "proposed_rule.json").write_text(json.dumps(rule, indent=2, sort_keys=True), encoding="utf-8")
    return {"report": str(report), "recommendation": rec or "no_change"}
