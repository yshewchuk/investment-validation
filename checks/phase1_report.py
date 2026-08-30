#!/usr/bin/env python3
"""Render the Phase 1 evidence report.

    python3 checks/phase1_report.py

Assembles what the phase actually produced — the replayed trade set, the trained
champions, the acceptance results, the calibration — into
``reports/phase1_scoring.md``, with a provenance block that makes the whole
thing regenerable.

Phase 4 owns the general report generator. This is the Phase-1-shaped version of
the same contract, and it emits the same provenance block, so the numbers here
are auditable now rather than after Phase 4 lands.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.data import manifest, store  # noqa: E402
from engine.models.registry import load_registry  # noqa: E402


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _fmt(value, spec=".4f"):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


def trade_set_section() -> str:
    trades = store.read_table("trades")
    engine_rows = trades[trades["provenance"].astype(str) == "engine.replay"]
    if engine_rows.empty:
        return "_No engine-replayed trades. Run `python3 -m engine.build_trades`._\n"

    lines = [
        "| Strategy | Events | Rows | Years | Mean ret @ worst | @ mid | @ best |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for strategy, group in engine_rows.groupby("strategy"):
        years = pd.to_datetime(group["event_date"]).dt.year
        means = {}
        for alpha in (0.0, 0.5, 1.0):
            at = group[np.isclose(group["fill_alpha"].astype(float), alpha)]
            means[alpha] = float(at["ret"].mean()) if len(at) else float("nan")
        lines.append(
            f"| {strategy} | {group['event_id'].nunique():,} | {len(group):,} | "
            f"{years.min()}–{years.max()} | {_fmt(means[0.0], '+.4f')} | "
            f"{_fmt(means[0.5], '+.4f')} | {_fmt(means[1.0], '+.4f')} |"
        )

    body = "\n".join(lines)
    by_year = []
    thru = engine_rows[
        (engine_rows["strategy"] == "STR-THRU")
        & np.isclose(engine_rows["fill_alpha"].astype(float), 0.5)
    ].copy()
    if len(thru):
        thru["year"] = pd.to_datetime(thru["event_date"]).dt.year
        by_year.append("\n**STR-THRU at mid fills, by year**\n")
        by_year.append("| Year | n | Mean | Median | Win rate |")
        by_year.append("|---|---:|---:|---:|---:|")
        for year, group in thru.groupby("year"):
            by_year.append(
                f"| {year} | {len(group):,} | {_fmt(float(group['ret'].mean()), '+.4f')} | "
                f"{_fmt(float(group['ret'].median()), '+.4f')} | "
                f"{_fmt(float((group['ret'] > 0).mean()), '.3f')} |"
            )
    return body + "\n" + "\n".join(by_year) + "\n"


def registry_section() -> str:
    registry = load_registry()
    if not registry.entries:
        return "_Registry is empty. Run `python3 -m engine.models.training.train_all`._\n"
    lines = [
        "| id | role | strategy | target | OOS years | n | r | MAE | champion |",
        "|---|---|---|---|---|---:|---:|---:|:--:|",
    ]
    for entry in sorted(registry.entries, key=lambda e: (e.role, e.id)):
        ev = entry.eval or {}
        years = entry.train_years
        span = f"{min(years)}–{max(years)}" if years else "—"
        n = ev.get("n")
        lines.append(
            f"| `{entry.id}` | {entry.role} | {entry.strategy} | {entry.target} | "
            f"{span} | {n:,} | {_fmt(ev.get('r'))} | {_fmt(ev.get('mae'))} | "
            f"{'✓' if entry.champion else ''} |"
            if isinstance(n, int)
            else
            f"| `{entry.id}` | {entry.role} | {entry.strategy} | {entry.target} | "
            f"{span} | — | {_fmt(ev.get('r'))} | {_fmt(ev.get('mae'))} | "
            f"{'✓' if entry.champion else ''} |"
        )

    extra = ["\n**Against the published research numbers**\n"]
    extra.append("| Model | This program | Published | Source |")
    extra.append("|---|---|---|---|")
    for entry in registry.entries:
        ref = (entry.eval or {}).get("reference") or {}
        if not ref:
            continue
        if entry.role == "size":
            extra.append(
                f"| `{entry.id}` | OOS r = {_fmt(entry.eval.get('r'))} | "
                f"r = {ref.get('oos_r')} | {ref.get('source')} |"
            )
        elif entry.role == "implied_t1":
            extra.append(
                f"| `{entry.id}` | MAE = {_fmt(entry.eval.get('mae'), '.2f')}pp, "
                f"r = {_fmt(entry.eval.get('r'))} | MAE {ref.get('mae_pp')}pp, "
                f"r {ref.get('r')} | {ref.get('source')} |"
            )
        elif entry.role == "gate":
            extra.append(
                f"| `{entry.id}` | lift = {_fmt(entry.eval.get('gate_lift'), '+.4f')}/trade | "
                f"{ref.get('lift_per_trade'):+.3f}/trade | {ref.get('source')} |"
            )

    swap = ""
    for entry in registry.entries:
        comparison = (entry.eval or {}).get("feature_set_comparison")
        if comparison:
            servable = comparison.get("servable", {})
            legacy = comparison.get("legacy", {})
            swap = (
                "\n**What live-servability costs** — the research feature list contains "
                "`implied_move`, which no upcoming event can supply. Both lists "
                f"evaluated on the same {servable.get('n', 0):,} rows:\n\n"
                "| Feature list | OOS r | MAE |\n|---|---:|---:|\n"
                f"| servable (`or_implied`) | {_fmt(servable.get('r'))} | "
                f"{_fmt(servable.get('mae'))} |\n"
                f"| legacy (`implied_move`) | {_fmt(legacy.get('r'))} | "
                f"{_fmt(legacy.get('mae'))} |\n"
            )
    return "\n".join(lines) + "\n" + "\n".join(extra) + "\n" + swap


def checks_section(results) -> str:
    if not results:
        return "_No acceptance run recorded._\n"
    lines = ["| Check | Result | Detail |", "|---|:--:|---|"]
    for row in results:
        status = "SKIP" if row.get("skipped") else ("PASS" if row.get("passed") else "**FAIL**")
        detail = str(row.get("detail", "")).replace("|", "\\|")
        lines.append(f"| `{row['name']}` | {status} | {detail} |")
    return "\n".join(lines) + "\n"


def calibration_section(reports) -> str:
    if not reports:
        return "_No calibration run recorded._\n"
    out = []
    for strategy, doc in reports.items():
        if "skipped" in doc:
            out.append(f"**{strategy}** — skipped: {doc['skipped']}\n")
            continue
        model = doc.get("model_layer", {})
        out.append(
            f"**{strategy} — model layer** (n={model.get('n')}, "
            f"years {model.get('years')})\n\n"
            f"- base rate {_fmt(model.get('base_rate'), '.3f')}, "
            f"Brier {_fmt(model.get('brier'))} vs base-rate predictor "
            f"{_fmt(model.get('brier_base_rate'))} "
            f"(skill {_fmt(model.get('brier_skill'), '+.4f')})\n"
            f"- beats the base rate: **{model.get('beats_base_rate')}**\n"
            f"- reliability monotonicity {_fmt(model.get('reliability_monotonicity'), '+.2f')}, "
            f"P&L decile monotonicity {_fmt(model.get('pnl_monotonicity'), '+.2f')}\n"
            f"- mean predicted P&L {_fmt(model.get('mean_predicted_pnl'), '+.4f')} vs "
            f"realized {_fmt(model.get('mean_realized_pnl'), '+.4f')}\n"
        )
        reliability = model.get("reliability") or []
        if reliability:
            out.append("\n| Bin | n | Predicted win | Realized win | Gap |")
            out.append("|---|---:|---:|---:|---:|")
            for row in reliability:
                out.append(
                    f"| {row['bin']} | {row['n']} | {_fmt(row['predicted'], '.3f')} | "
                    f"{_fmt(row['realized'], '.3f')} | {_fmt(row['gap'], '+.3f')} |"
                )
            out.append("")
    return "\n".join(out) + "\n"


def provenance_section(replay_report) -> str:
    snapshot = manifest.read_snapshot() or {}
    tables = snapshot.get("tables", {})
    lines = [
        f"- **Snapshot hash:** `{snapshot.get('snapshot', 'unknown')}`",
        f"- **Panel sha256:** `{(snapshot.get('panel_sha256') or 'unknown')[:16]}…`",
        f"- **Code:** git `{_git_head()}`",
        f"- **Store format:** {snapshot.get('format', 'unknown')}",
        f"- **Generated:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "| Tier-2 table | Rows | Years | Content hash |",
        "|---|---:|---|---|",
    ]
    for name in sorted(tables):
        row = tables[name]
        lines.append(
            f"| {name} | {row.get('rows', 0):,} | {row.get('years', '—')} | "
            f"`{str(row.get('content_hash', ''))[:16]}…` |"
        )
    if replay_report:
        lines.append("")
        lines.append("| Replay | Planned | With chains | Priced | Coverage |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in replay_report.get("results", []):
            lines.append(
                f"| {row['strategy']} | {row['planned']:,} | {row.get('replayable', 0):,} | "
                f"{row['priced']:,} | {row.get('coverage', 0):.1%} |"
            )
    registry = load_registry()
    if registry.entries:
        lines.append("")
        lines.append("| Model artifact | sha256 | seed |")
        lines.append("|---|---|---:|")
        for entry in sorted(registry.entries, key=lambda e: e.id):
            lines.append(f"| `{entry.id}` | `{entry.artifact_sha256[:16]}…` | {entry.seed} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def sections(
    *,
    checks_json: Path | None = None,
    calibration_json: Path | None = None,
    replay_json: Path | None = None,
) -> list[dict]:
    """Phase 1's evidence as generator sections.

    Phase 1 predates the report generator and shipped its own markdown
    template; the content was always the point, the format never was. The
    generator now owns the frame — provenance, checklist, glossary, units —
    and this supplies what only Phase 1 knows.
    """
    checks = _read_json(checks_json) if checks_json else None
    calibration = _read_json(calibration_json) if calibration_json else None
    replay_report = _read_json(replay_json) if replay_json else None

    return [
        {"title": "The replayed trade set",
         "note": "`engine.replay` prices each structure against real ORATS chains at "
                 "five fill alphas. This is the analog layer's evidence base, the "
                 "gate's training target, and the substrate Phase 2's backtests use — "
                 "one code path, so the live scorer and the research code cannot drift "
                 "apart.",
         "body": trade_set_section().splitlines() + [
             "",
             "The worst→mid→best spread across those columns is the program's headline "
             "risk made visible: it is the same trades, on the same days, differing "
             "only in what was assumed about execution.",
         ]},
        {"title": "Champions", "body": registry_section().splitlines()},
        {"title": "Acceptance checks", "body": checks_section(checks).splitlines()},
        {"title": "Calibration detail",
         "note": "Predicted win rates graded against realized outcomes, out of sample. "
                 "The bar is the base-rate predictor: a model that always forecasts the "
                 "unconditional win rate is perfectly calibrated and perfectly useless, "
                 "so beating its Brier score is what shows the predictions carry "
                 "information about *which* trades win.",
         "body": calibration_section(calibration).splitlines()},
        {"title": "Phase-1 accuracy standards",
         "note": "The Phase 4 standards, answered for this phase.",
         "columns": ["#", "Standard", "Status"],
         "align": ["---", "---", "---"],
         "rows": [
             ["1", "Real traded/quoted prices only",
              "**PASS** — ORATS chain bid/ask via `engine.structures.price_structure`; "
              "oquants fitted marks refused at the adapter"],
             ["2", "Leak audit passed",
              "**PASS** — `assert_causal` runs on every scoring call; the `poison` "
              "check confirms it raises rather than warns"],
             ["3", "Headline numbers walk-forward OOS only",
              "**PASS** — every registry metric comes from `common.walk_forward`; the "
              "served model is refit on everything but its published numbers are not"],
             ["4", "Fill sensitivity shown",
              "**PASS** — every trade priced at α ∈ {0, .25, .5, .75, 1}; §1 reports "
              "worst/mid/best"],
             ["5", "Multiple-testing ledger cited",
              "**DEFERRED** — Phase 2 owns the ledger. Phase 1 registered one champion "
              "per role with no spec search"],
             ["6", "Survivorship caveat quantified",
              "**PARTIAL** — the event universe is the ORATS calendar, which includes "
              "delisted names; chain coverage is the binding limit and is reported "
              "against the full planned universe"],
             ["7", "Prediction ledger",
              "**LIVE** — `engine.ledger` writes frozen predictions and scores them "
              "after the event; cumulative calibration lands in "
              "`ledger/calibration/REPORT.md`"],
         ]},
        {"title": "Phase-1 provenance detail", "body": provenance_section(replay_report).splitlines()},
    ]


def build(**kwargs) -> str:
    """Render the Phase 1 report through the Phase 4 generator."""
    import tempfile

    from engine.report import Report

    with tempfile.TemporaryDirectory() as tmp:
        return Report(_context(**kwargs)).write(Path(tmp)).read_text()


def _context(*, checks_json=None, calibration_json=None, replay_json=None) -> dict:
    from engine.report import build_provenance

    calibration = _read_json(calibration_json) if calibration_json else None
    block = None
    if isinstance(calibration, dict):
        # The acceptance suite writes one calibration block per strategy; the
        # report's own section renders whichever one carries deciles.
        for value in calibration.values():
            if isinstance(value, dict) and value.get("deciles"):
                block = value
                break
    return {
        "kind": "calibration",
        "spec": {"id": "PHASE-1", "title": "Scoring engine: evidence report",
                 "type": "descriptive",
                 "hypothesis": "The scoring engine reproduces the research code's "
                               "trades exactly, respects the leak boundary, and its "
                               "champions' predictions are calibrated well enough to "
                               "ship as rankings."},
        "results": {"headline": {}, "stress": {}, "mc": {},
                    "calibration": block or {"available": False,
                                             "reason": "no calibration artifact supplied"}},
        "headline": {}, "backtest": {}, "checklist": [],
        "provenance": build_provenance(seeds={}, input_files=[
            p for p in (checks_json, calibration_json, replay_json) if p
        ]),
        "survivorship_note": "",
        "calibration": block,
        "calibration_raw": block or {"available": False,
                                     "reason": "no calibration artifact supplied"},
        "funnel": [],
        "extra_sections": sections(checks_json=checks_json,
                                   calibration_json=calibration_json,
                                   replay_json=replay_json),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checks-json", default=None)
    ap.add_argument("--calibration-json", default=None)
    ap.add_argument("--replay-json", default=None)
    ap.add_argument("--out", default=str(paths.REPORTS / "phase1_scoring.md"))
    args = ap.parse_args(argv)

    from engine.report import Report  # noqa: E402

    out = paths.assert_writable(Path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    context = _context(
        checks_json=Path(args.checks_json) if args.checks_json else None,
        calibration_json=Path(args.calibration_json) if args.calibration_json else None,
        replay_json=Path(args.replay_json) if args.replay_json else None,
    )
    Report(context).write(out.parent, filename=out.name)
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
