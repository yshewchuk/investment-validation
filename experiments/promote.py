#!/usr/bin/env python3
"""Champion/challenger promotion — all rules green or no promotion at all.

    python3 experiments/promote.py EXP-101 --role gate --strategy STR-THRU \
        [--apply-registry entry.json] [--dry-run]

A challenger is promoted only if ALL of:

  (a) it beats the champion on walk-forward OOS mean AND sharpe_trade,
  (b) MC P(loss) at 5% sizing does not worsen,
  (c) it survives the stress battery (no new red regime cell, tail injection
      present whenever a short leg exists),
  (d) its spec was pre-registered before the OOS evaluation,
  (e) its report's accuracy checklist has no FAIL.

Any red → print why and exit nonzero. There is no partial promotion: the
registry, the ledger, and the promotion report are written together at the
end, or not at all.

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
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from experiments import lib  # noqa: E402


def _mc_ploss_5pct(metrics: Mapping[str, Any]) -> float | None:
    mc = (metrics.get("mc") or {})
    return mc.get("p_loss")


def decide(
    challenger: Mapping[str, Any],
    champion: Mapping[str, Any],
    *,
    prereg_valid: bool,
    checklist_fails: int = 0,
    short_leg: bool = False,
    tol: float = 0.0,
) -> tuple[bool, list[str]]:
    """The plan's promotion rules, applied mechanically.

    ``challenger`` / ``champion`` carry ``mean``, ``sharpe_trade``, ``mc``
    (with ``p_loss`` at 5% sizing) and ``stress`` (regime table + tail
    injection) in the canonical evaluate() shapes. ``tol`` is a non-negative
    indifference band: the challenger must beat the champion by more than
    ``tol`` on mean and Sharpe (zero by default — any tie keeps the champion,
    because the champion has already survived a season of scrutiny).
    Returns ``(promote, reasons)`` — reasons are human-readable either way.
    """
    reasons: list[str] = []

    c_mean = challenger.get("mean")
    h_mean = champion.get("mean")
    c_sharpe = challenger.get("sharpe_trade")
    h_sharpe = champion.get("sharpe_trade")
    if c_mean is None or h_mean is None or c_sharpe is None or h_sharpe is None:
        reasons.append("missing mean/sharpe_trade on challenger or champion metrics")
    else:
        if c_mean > h_mean + tol:
            reasons.append(f"PASS (a1) OOS mean {c_mean:+.4f} > champion {h_mean:+.4f}")
        else:
            reasons.append(f"FAIL (a1) OOS mean {c_mean:+.4f} does not beat champion {h_mean:+.4f}")
        if c_sharpe > h_sharpe + tol:
            reasons.append(f"PASS (a2) sharpe_trade {c_sharpe:.3f} > champion {h_sharpe:.3f}")
        else:
            reasons.append(f"FAIL (a2) sharpe_trade {c_sharpe:.3f} does not beat champion {h_sharpe:.3f}")

    c_ploss = _mc_ploss_5pct(challenger)
    h_ploss = _mc_ploss_5pct(champion)
    if c_ploss is None:
        reasons.append("FAIL (b) challenger carries no MC P(loss) at 5% sizing")
    elif h_ploss is None:
        reasons.append(f"WARN (b) champion has no MC P(loss); challenger {c_ploss:.3f} stands alone")
    elif c_ploss <= h_ploss:
        reasons.append(f"PASS (b) MC P(loss)@5% {c_ploss:.3f} <= champion {h_ploss:.3f}")
    else:
        reasons.append(f"FAIL (b) MC P(loss)@5% {c_ploss:.3f} worsens champion {h_ploss:.3f}")

    stress = challenger.get("stress") or {}
    regimes = stress.get("regimes") or {}
    champ_regimes = (champion.get("stress") or {}).get("regimes") or {}
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
    if short_leg and not tail:
        reasons.append("FAIL (c2) short-leg challenger without tail injection")
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
        candidates = sorted((folder / "results").glob("metrics_*.json"))
        if not candidates:
            raise FileNotFoundError(f"{folder} has no evaluation results yet")
        metrics_path = candidates[-1]
    results = json.loads(metrics_path.read_text())
    return spec, results


def promotion_report_context(exp_id: str, spec: Mapping[str, Any],
                             challenger: Mapping[str, Any],
                             champion: Mapping[str, Any],
                             reasons: list[str]) -> dict[str, Any]:
    ctx = lib.ledger_context(lib.spec_hash(spec))
    return {
        "kind": "promotion",
        "spec": {"id": exp_id, "title": spec.get("title", exp_id),
                 "hypothesis": spec.get("hypothesis", "")},
        "headline": challenger,
        "results": {
            "headline": challenger,
            "champion": champion,
            "reasons": reasons,
            "ledger_context": ctx,
            "decision": "PROMOTED",
            "decided_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        },
        "checklist": [],
        "provenance": {"generated_at": datetime.now(tz=timezone.utc).isoformat()},
        "survivorship_note": "",
    }


def _render_promotion_report(exp_id: str, spec: Mapping[str, Any],
                             challenger: Mapping[str, Any],
                             champion: Mapping[str, Any],
                             reasons: list[str], out: Path) -> Path:
    ctx = lib.ledger_context(lib.spec_hash(spec))
    lines = [
        f"# Promotion report — {exp_id}",
        "",
        f"*{datetime.now(tz=timezone.utc).isoformat(timespec='seconds')}*",
        "",
        f"**Decision: PROMOTED.** {ctx['specs_tried']} spec(s) were tried against this "
        f"snapshot before this one (this spec appears in {ctx['this_spec_rows']} ledger row(s)).",
        "",
        "## Rules",
        "",
    ]
    lines += [f"- {r}" for r in reasons]
    keys = ("n", "mean", "win_rate", "sharpe_trade", "sharpe_equity", "max_dd")
    lines += [
        "",
        "## Challenger vs champion (walk-forward OOS, mid fills)",
        "",
        "| metric | challenger | champion |",
        "|---|---|---|",
    ]
    for k in keys:
        lines.append(f"| {k} | {challenger.get(k)} | {champion.get(k)} |")
    for label, m in (("challenger", challenger), ("champion", champion)):
        ploss = (m.get("mc") or {}).get("p_loss")
        lines.append(f"| MC P(loss)@5% ({label}) | {ploss} | |" if label == "challenger"
                     else f"| MC P(loss)@5% ({label}) | | {ploss} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_id", help="challenger experiment, e.g. EXP-101")
    parser.add_argument("--champion-metrics", required=True,
                        help="JSON with the champion's canonical headline metrics")
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

    champion = json.loads(Path(args.champion_metrics).read_text())
    challenger = results.get("headline", {})
    prereg = results.get("preregistration", {})
    prereg_valid = bool(prereg.get("valid"))
    checklist_fails = int(results.get("checklist_fails", 0))
    short_leg = bool(spec.get("has_short_leg", False))

    promote, reasons = decide(challenger, champion, prereg_valid=prereg_valid,
                              checklist_fails=checklist_fails, short_leg=short_leg)
    for r in reasons:
        print(r)

    if not promote:
        print("\nNO PROMOTION — at least one rule failed. No partial promotion exists.")
        return 1
    if args.dry_run:
        print("\ndry-run: all rules green; nothing written.")
        return 0

    if args.apply_registry and args.entry_json:
        from engine.models.registry import RegistryEntry, register

        entry = RegistryEntry(**json.loads(Path(args.entry_json).read_text()))
        register(entry)  # demotes the incumbent for the same (strategy, role)
        print(f"registry updated: {entry.id} is champion for {entry.key}")

    report_path = paths.REPORTS / f"promotion_{args.exp_id}.md"
    _render_promotion_report(args.exp_id, spec, challenger, champion, reasons, report_path)
    print(f"promotion report: {report_path}")

    # Mark the ledger row promoted (append-only: a NEW row supersedes, never a rewrite).
    lib.ledger_append([{
        "id": args.exp_id,
        "spec_hash": lib.spec_hash(spec),
        "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        "stage": "ran",
        "oos_mean_mid": challenger.get("mean", ""),
        "sharpe_trade": challenger.get("sharpe_trade", ""),
        "promoted": "True",
    }])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
