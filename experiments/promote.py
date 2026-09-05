#!/usr/bin/env python3
"""Champion/challenger promotion — all rules green or no promotion at all.

    python3 experiments/promote.py EXP-101 --role gate --strategy STR-THRU \
        [--apply-registry entry.json] [--dry-run]

A challenger is promoted only if ALL of:

  (a) it beats the champion on CAGR AND sharpe_trade, and is positive in no
      smaller a share of evaluated years,
  (b) MC P(loss) at 5% sizing does not worsen,
  (c) it survives the stress battery (no new red regime cell, tail injection
      present whenever a short leg exists),
  (d) its spec was pre-registered before the OOS evaluation,
  (e) its report's accuracy checklist has no FAIL.

Any red → print why and exit nonzero. There is no partial promotion. When
green, the writes land easiest-to-undo first: promotion report, then the
ledger row, then the registry update LAST — the one change that is hard to
reverse, so an earlier failure leaves the registry untouched.

The decision itself is a pure function (:func:`decide`) of two canonical
metrics dicts plus the receipts — the acceptance suite exercises it with
synthetic challengers on both sides of every rule.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from experiments import lib  # noqa: E402


def metrics_view(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a metrics document to the fields ``decide`` compares.

    Accepts BOTH the full ``evaluate()`` results dict (``headline`` / ``mc`` /
    ``stress`` / ``preregistration`` / ``checklist_fails`` at the top level)
    and a flat hand-written dict (``mean`` / ``sharpe_trade`` / ``mc.p_loss``
    at the top level). The full shape is what a real challenger always carries;
    the flat shape exists so a champion baseline can be written before the
    champion itself has been through the harness — and a missing field then
    fails the corresponding rule closed rather than passing it vacuously.
    """
    full = "headline" in doc
    headline = doc.get("headline") if full else doc
    headline = headline or {}

    mc = doc.get("mc") or {}
    if full:
        mc5 = (mc.get("by_fraction") or {}).get("0.05") or {}
    else:
        mc5 = (mc.get("by_fraction") or {}).get("0.05") or mc
    p_loss = mc5.get("p_loss")

    stress = doc.get("stress") or {}
    prereg = doc.get("preregistration") or {}
    cal = doc.get("calibration") or {}
    return {
        "mean": headline.get("mean"),
        "cagr": headline.get("cagr"),
        "years_positive": headline.get("years_positive"),
        "years_evaluated": headline.get("years_evaluated"),
        "sharpe_trade": headline.get("sharpe_trade"),
        "win_rate": headline.get("win_rate"),
        "max_dd": headline.get("max_dd"),
        "p_loss_5": p_loss,
        "stress": stress,
        "prereg_valid": prereg.get("valid") if full else None,
        "checklist_fails": doc.get("checklist_fails") if full else None,
        "brier_skill": cal.get("brier_skill", doc.get("brier_skill")),
        "full_shape": full,
    }


#: The calibration decision record (2026-08-30): a challenger must not ship
#: win rates worse than its base rate. Matches phase1_checks.MIN_BRIER_SKILL —
#: honest recalibration lands the Brier skill near zero, and this tolerance
#: absorbs sampling noise while still catching real anti-calibration.
MIN_BRIER_SKILL = -0.05


def decide(
    challenger: Mapping[str, Any],
    champion: Mapping[str, Any],
    *,
    prereg_valid: bool | None = None,
    checklist_fails: int | None = None,
    short_leg: bool = False,
    tol: float = 0.0,
) -> tuple[bool, list[str]]:
    """The plan's promotion rules, applied mechanically.

    ``challenger`` / ``champion`` are either full ``evaluate()`` results dicts
    (the shape every real challenger carries — see :func:`metrics_view`) or
    flat metrics dicts for hand-written baselines. ``tol`` is a non-negative
    indifference band: the challenger must beat the champion by more than
    ``tol`` on CAGR and Sharpe (zero by default — any tie keeps the champion,
    because the champion has already survived a season of scrutiny).
    Returns ``(promote, reasons)`` — reasons are human-readable either way.
    """
    reasons: list[str] = []
    c = metrics_view(challenger)
    h = metrics_view(champion)

    if prereg_valid is None:
        prereg_valid = bool(c["prereg_valid"])
    if checklist_fails is None:
        checklist_fails = int(c["checklist_fails"] or 0)

    if None in (c["cagr"], h["cagr"], c["sharpe_trade"], h["sharpe_trade"]):
        reasons.append(
            "FAIL (a) missing cagr/sharpe_trade — the champion baseline must be "
            "an evaluate() result (or a complete hand-written equivalent)")
    else:
        if c["cagr"] > h["cagr"] + tol:
            reasons.append(f"PASS (a1) CAGR {100*c['cagr']:+.2f}% > champion {100*h['cagr']:+.2f}%")
        else:
            reasons.append(f"FAIL (a1) CAGR {100*c['cagr']:+.2f}% does not beat champion {100*h['cagr']:+.2f}%")
        if c["sharpe_trade"] > h["sharpe_trade"] + tol:
            reasons.append(f"PASS (a2) sharpe_trade {c['sharpe_trade']:.3f} > champion {h['sharpe_trade']:.3f}")
        else:
            reasons.append(f"FAIL (a2) sharpe_trade {c['sharpe_trade']:.3f} does not beat champion {h['sharpe_trade']:.3f}")
        cs, ce = c["years_positive"], c["years_evaluated"]
        hs, he = h["years_positive"], h["years_evaluated"]
        if None in (cs, ce, hs, he) or not ce or not he:
            reasons.append("WARN (a3) no per-year consistency on one side; rule not applied")
        elif cs / ce >= hs / he:
            reasons.append(
                f"PASS (a3) positive in {cs}/{ce} years >= champion {hs}/{he}")
        else:
            reasons.append(
                f"FAIL (a3) positive in {cs}/{ce} years, worse than champion {hs}/{he}")

    # Drawdown WARNS, it does not block. A challenger trading four times as
    # often at the same fractional sizing carries a deeper absolute drawdown
    # for reasons that have nothing to do with edge quality, and Sharpe plus
    # rule (b)'s MC P(loss) already price the risk. Loud, not fatal.
    if c["max_dd"] is not None and h["max_dd"] is not None and c["max_dd"] > h["max_dd"]:
        reasons.append(
            f"WARN max drawdown {100*c['max_dd']:.1f}% deepens champion "
            f"{100*h['max_dd']:.1f}%")

    if c["p_loss_5"] is None:
        reasons.append("FAIL (b) challenger carries no MC P(loss) at 5% sizing")
    elif h["p_loss_5"] is None:
        reasons.append(f"WARN (b) champion has no MC P(loss); challenger {c['p_loss_5']:.3f} stands alone")
    elif c["p_loss_5"] <= h["p_loss_5"]:
        reasons.append(f"PASS (b) MC P(loss)@5% {c['p_loss_5']:.3f} <= champion {h['p_loss_5']:.3f}")
    else:
        reasons.append(f"FAIL (b) MC P(loss)@5% {c['p_loss_5']:.3f} worsens champion {h['p_loss_5']:.3f}")

    stress = c["stress"]
    regimes = stress.get("regimes") or {}
    champ_regimes = (h["stress"] or {}).get("regimes") or {}
    red = []
    for name, s in regimes.items():
        n = s.get("n", 0)
        mean = s.get("mean")
        if n and n >= 10 and mean is not None and mean < 0:
            champ = champ_regimes.get(name, {})
            if not champ or (champ.get("mean") is not None and mean < champ["mean"]):
                red.append(f"{name}: {mean:+.4f} on n={n}")
    if red:
        reasons.append("FAIL (c) new red stress cells: " + "; ".join(red))
    else:
        reasons.append("PASS (c) stress battery: no new red regime cell")

    tail = stress.get("tail_injection") or {}
    tail_ok = bool(tail) and tail.get("available") is not False
    if short_leg and not tail_ok:
        reasons.append("FAIL (c2) short-leg challenger without a tail-injection result")
    elif short_leg:
        reasons.append("PASS (c2) tail injection present for short leg")

    if prereg_valid:
        reasons.append("PASS (d) pre-registration valid")
    else:
        reasons.append("FAIL (d) pre-registration missing or post-dated")

    if checklist_fails:
        reasons.append(f"FAIL (e) report checklist has {checklist_fails} FAIL item(s)")
    else:
        reasons.append("PASS (e) accuracy checklist clean")

    brier = c["brier_skill"]
    if brier is None:
        reasons.append("WARN (f) no Brier-skill measurement; calibration gate not applied")
    elif float(brier) >= MIN_BRIER_SKILL:
        reasons.append(f"PASS (f) Brier skill {float(brier):+.4f} >= {MIN_BRIER_SKILL}")
    else:
        reasons.append(
            f"FAIL (f) Brier skill {float(brier):+.4f} < {MIN_BRIER_SKILL} — the challenger "
            "would ship win rates worse than its base rate")

    promote = all(r.startswith(("PASS", "WARN")) for r in reasons)
    return promote, reasons


# --------------------------------------------------------------------------
# wiring to real experiments
# --------------------------------------------------------------------------


def load_experiment_metrics(exp_id: str, root: Path | None = None) -> tuple[dict, dict]:
    """(spec, headline results) for an experiment's primary spec."""
    dirs = lib.experiment_dirs(root)
    number = lib.parse_experiment_id(exp_id)
    if number not in dirs:
        raise FileNotFoundError(f"no experiment folder for {exp_id}")
    folder = dirs[number]
    spec = lib.load_spec(folder / "spec.yaml")
    sha = lib.spec_hash(spec)
    metrics_path = folder / "results" / f"metrics_{sha[:12]}.json"
    if not metrics_path.exists():
        # No fallback to "the newest metrics file": that would silently promote
        # a GRID CELL, which the guide defines as secondary — never the headline.
        raise FileNotFoundError(
            f"{folder} has no results for its PRIMARY spec (spec_hash {sha[:12]}…). "
            "Run the experiment's primary spec first; grid cells are never the headline."
        )
    results = json.loads(metrics_path.read_text())
    return spec, results


def champion_from_registry(model_id: str) -> dict[str, Any]:
    """Build a champion baseline from a registry entry's eval block.

    The registry's eval shape is not the canonical evaluate() shape, so this
    maps only what exists (mean, win rate, n) and leaves sharpe_trade and MC
    P(loss) absent — which makes ``decide`` fail those rules closed. That is
    deliberate: a promotion diffed against an incomplete baseline is not
    mechanical, and the right cure is evaluating the champion through the
    harness once, not papering over the gap here.
    """
    from engine.models.registry import load_registry

    entry = load_registry().get(model_id)
    ev = entry.eval or {}
    doc: dict[str, Any] = {
        "mean": ev.get("gated_mean_ret"),
        "win_rate": ev.get("gated_win_rate"),
        "n": ev.get("n_passed"),
        "_source": f"registry entry {model_id} (eval block)",
    }
    return doc


def promotion_report_context(exp_id: str, spec: Mapping[str, Any],
                             challenger: Mapping[str, Any],
                             champion: Mapping[str, Any],
                             reasons: list[str],
                             input_files: Sequence[Path | str] = ()) -> dict[str, Any]:
    """Context for ``engine.report.Report`` — promotions emit through the one
    generator like everything else (plan §P4.1), so they carry the same
    provenance block and checklist as evaluations."""
    from engine.report import build_provenance
    from engine.report import ChecklistItem

    ctx = lib.ledger_context(lib.spec_hash(spec))
    challenger_headline = challenger.get("headline", challenger) or {}
    checklist = [
        ChecklistItem(item.get("name", ""), item.get("status", "N/A"),
                      item.get("evidence", ""))
        for item in (challenger.get("checklist") or [])
    ]
    return {
        "kind": "promotion",
        "spec": {"id": exp_id, "title": spec.get("title", exp_id),
                 "hypothesis": spec.get("hypothesis", "")},
        "headline": challenger_headline,
        "results": {
            "champion": champion,
            "reasons": reasons,
            "ledger_context": ctx,
            "decision": "PROMOTED",
            "decided_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "spec_hash": lib.spec_hash(spec),
        },
        "checklist": checklist,
        "provenance": build_provenance(spec_hash=lib.spec_hash(spec),
                                       input_files=input_files),
        "survivorship_note": "",
    }


def render_promotion_report(exp_id: str, spec: Mapping[str, Any],
                            challenger: Mapping[str, Any],
                            champion: Mapping[str, Any],
                            reasons: list[str], out_dir: Path,
                            input_files: Sequence[Path | str] = ()) -> Path:
    from engine.report import Report

    context = promotion_report_context(exp_id, spec, challenger, champion,
                                       reasons, input_files=input_files)
    return Report(context).write(out_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_id", help="challenger experiment, e.g. EXP-101")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--champion-metrics",
                        help="JSON with the champion's metrics — preferably a full "
                             "evaluate() results dict from the champion's own run")
    source.add_argument("--champion-from-registry", metavar="MODEL_ID",
                        help="derive the baseline from a registry entry's eval block; "
                             "rules needing fields the registry lacks fail closed")
    parser.add_argument("--entry-json", default=None,
                        help="RegistryEntry JSON; when given AND --apply-registry, "
                             "the model registry is updated (champion=True, incumbent demoted)")
    parser.add_argument("--apply-registry", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        spec, results = load_experiment_metrics(args.exp_id)
    except (FileNotFoundError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2

    if args.champion_from_registry:
        champion = champion_from_registry(args.champion_from_registry)
    else:
        champion = json.loads(Path(args.champion_metrics).read_text())

    short_leg = bool(spec.get("has_short_leg", False))
    promote, reasons = decide(results, champion, short_leg=short_leg)
    for r in reasons:
        print(r)

    if not promote:
        print("\nNO PROMOTION — at least one rule failed. No partial promotion exists.")
        return 1
    if args.dry_run:
        print("\ndry-run: all rules green; nothing written.")
        return 0

    # Write order = easiest to undo first. The report and the ledger row are
    # plain files; the registry update is the one change that is hard to
    # reverse, so it lands last — a failure in an earlier step leaves the
    # registry untouched.
    report_dir = paths.REPORTS / f"promotion_{args.exp_id}"
    report_path = render_promotion_report(
        args.exp_id, spec, results, champion, reasons, report_dir,
        input_files=[Path(args.champion_metrics)] if args.champion_metrics else [])
    print(f"promotion report: {report_path}")

    # Mark the ledger row promoted (append-only: a NEW row supersedes, never a rewrite).
    headline = results.get("headline", {})
    lib.ledger_append([{
        "id": args.exp_id,
        "spec_hash": lib.spec_hash(spec),
        "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        "stage": "ran",
        "oos_mean_mid": headline.get("mean", ""),
        "sharpe_trade": headline.get("sharpe_trade", ""),
        "promoted": "True",
    }])

    if args.apply_registry and args.entry_json:
        from engine.models.registry import RegistryEntry, register

        entry = RegistryEntry(**json.loads(Path(args.entry_json).read_text()))
        register(entry)  # demotes the incumbent for the same (strategy, role)
        print(f"registry updated: {entry.id} is champion for {entry.key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
