"""EXP-190 Amendment 1: report-only re-read of the stored Stage A run; no fit, panel, label or ledger read."""
import argparse
import json
import sys
from pathlib import Path

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError, fail
from engine.v2.research.progress import stage
from experiments.size_pnl_prereg import _read_bounded
from experiments.size_quantile_arm_tables import MIN_SCORED_SHARE, PERIODS, accuracy_rule, calibration_rule
from experiments.size_quantile_arm_verdict import conditional_rule, stage_a_verdict
from experiments.size_quantile_prereg import SPEC_PATH, load_spec
from experiments.size_quantile_stage_a import MAX_REPORT_BYTES, REPORT, SCHEMA as REPORT_SCHEMA, _publish, _refuse

SCHEMA, FLOOR, ARMS = "size_quantile_stage_a_amendment.v1", "scored_share_below_floor", ("Q", "Q_raw")
MALFORMED = (KeyError, TypeError, IndexError, ValueError, AttributeError)


def _entry(spec):
    try:
        (entry,) = [a for a in spec["amendments"] if a.get("id") == "A1"
                    and {"applies_to", "scored_share_floor"} <= set(a)]
        floors = (spec["stage_a"]["min_scored_share"], entry["scored_share_floor"]["min_scored_share"])
        if all(isinstance(entry["applies_to"][k], str)
               for k in ("spec_hash", "run_inputs_hash", "report_content_hash")) and floors == (MIN_SCORED_SHARE,) * 2:
            return entry
    except MALFORMED:
        pass
    raise _refuse("amendment_missing", "spec has no single A1 entry with its floor and pins")


def _load(path, schema, pinned, unbound):
    try:
        doc = json.loads(_read_bounded(path, MAX_REPORT_BYTES))
    except (OSError, ValueError):
        raise _refuse("amended_input_unreadable", f"{path.name} is missing, unreadable or not JSON") from None
    try:
        own = content_hash({k: v for k, v in doc.items() if k != "content_hash"}) if isinstance(doc, dict) else None
    except (TypeError, ValueError):
        own = None
    if own is None or doc.get("schema_version") != schema or own != doc.get("content_hash") or own != pinned:
        raise _refuse(unbound, f"{path.name} is not bound to its pinned content hash")
    return doc


def _amend(arm, stored):
    drops, years = stored["drops"], {int(p[:4]) for p in PERIODS}
    if [r["period"] for r in stored["share"]] != [*PERIODS, "pooled"] or any(d["year"] not in years for d in drops):
        raise _refuse("amended_shares_inconsistent", f"{arm} share rows or drop years are not the periods")
    rows = []
    for row in stored["share"]:
        p = row["period"]
        sel = [d for d in drops if p == "pooled" or d["year"] == int(p[:4])]
        if row["admitted"] - row["paired"] != sum(d["n"] for d in sel) or any(
                d["arm_reason"] == d["b0_reason"] == "scored" for d in sel):
            raise _refuse("amended_shares_inconsistent", f"{arm} {p}: drops do not account for unpaired events")
        gap = {r: sum(d["n"] for d in sel if d["b0_reason"] == r) for r in {d["b0_reason"] for d in sel} - {"scored"}}
        forecast = row["admitted"] - sum(gap.values())
        share = row["paired"] / forecast if forecast else None
        rows.append({"period": p, "admitted": row["admitted"], "paired": row["paired"], "original_share": row["share"],
            "original_below_floor": row["below_floor"], "b0_gap": sum(gap.values()), "with_forecast": forecast,
            "share": share, "b0_gap_by_reason": gap, "below_floor": share is not None and share < MIN_SCORED_SHARE})
    reasons = [r for r in stored["not_testable_reasons"] if not r.endswith(":" + FLOOR)]
    reasons += [f"{r['period']}:{FLOOR}" for r in rows[:-1] if r["below_floor"]]
    acc = cond = cal = None
    verdict = "not_testable"
    if not reasons:
        a, c, k = stored["accuracy"], stored["conditional"], stored["calibration"]
        acc = accuracy_rule(a["periods_below"], a["quarters_below"], a["quarter_interval"], a["ticker_interval"])
        cond = conditional_rule(c["periods"], c["W_interval"], c["diff_interval"])
        cal = calibration_rule(k["quarters_no_worse"])
        verdict = "pass" if acc and cond and cal else "fail"
    return {"arm": arm, "verdict": verdict, "not_testable_reasons": reasons, "share": rows,
            "accuracy_passed": acc, "conditional_passed": cond, "calibration_passed": cal}


def run(out_dir, *, spec_path=SPEC_PATH) -> dict:
    try:
        with stage("quantile amendment"):
            spec, digest = load_spec(spec_path)
            entry = _entry(spec)
            pins, out = entry["applies_to"], Path(out_dir)
            run_inputs = _load(out / "run_inputs.json", "size_quantile_run_inputs.v2", pins["run_inputs_hash"],
                               "amended_run_inputs_unbound")
            if run_inputs.get("mode") != "full" or run_inputs.get("spec_hash") != pins["spec_hash"]:
                raise _refuse("amended_run_inputs_unbound", "run inputs are not the full run of the original spec")
            report = _load(out / REPORT, REPORT_SCHEMA, pins["report_content_hash"], "amended_report_unbound")
            inputs = report.get("inputs") if isinstance(report.get("inputs"), dict) else {}
            if (inputs.get("run_inputs_hash") != run_inputs["content_hash"]
                    or inputs.get("spec_hash") != pins["spec_hash"]):
                raise _refuse("amended_report_unbound", "report is not bound to these run inputs and spec")
            try:
                drops = [d for arm in ARMS for d in report["arms"][arm]["drops"]]
                if any(report["unlabelled_in_period"].values()) or any(
                        "label_missing" in (d["arm_reason"], d["b0_reason"]) for d in drops):
                    raise _refuse("amended_share_unlabelled", "a drop has an unlabelled event: B0 output is unknown")
                results = {arm: _amend(arm, report["arms"][arm]) for arm in ARMS}
                verdict = stage_a_verdict(results["Q"], results["Q_raw"])
                body = {"schema_version": SCHEMA, "label": entry["label"], "post_hoc": True,
                        "preregistration_checked": False, "arms": results, "min_scored_share": MIN_SCORED_SHARE,
                        "registered_result": {k: report[k] for k in ("status", "reasons", "arm_verdicts")},
                        "status": verdict["status"], "reasons": verdict["reasons"], "arm_verdicts": verdict["arms"],
                        "inputs": {"report_content_hash": report["content_hash"],
                                   "run_inputs_hash": run_inputs["content_hash"],
                                   "original_spec_hash": pins["spec_hash"], "amendment_spec_hash": digest}}
            except MALFORMED:
                raise _refuse("amended_report_malformed", "the stored report lacks what the re-read reads") from None
            return _publish(out, body, f"stage_a_report_amendment1_{digest[:12]}.json", "amended_report_changed")
    except MemoryError:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "amendment re-read ran out of memory") from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="EXP-190 Amendment 1 report-only re-read.")
    parser.add_argument("--out-dir", required=True)
    try:
        doc = run(parser.parse_args(argv).out_dir, spec_path=SPEC_PATH)
    except OpsError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    shares = {arm: res["share"] for arm, res in doc["arms"].items()}
    print(json.dumps({**{k: doc[k] for k in ("status", "reasons", "arm_verdicts", "content_hash")}, "shares": shares}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
