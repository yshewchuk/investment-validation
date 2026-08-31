#!/usr/bin/env python3
"""Render the Phase 3 evidence report.

    python3 checks/phase3_checks.py --json reports/phase3_checks.json
    python3 checks/phase3_report.py --checks-json reports/phase3_checks.json

Assembles what the phase actually produced — the rendered bundle, the nightly
job's shape, the acceptance results, and the operational state (cron, quota,
publish target) — into ``reports/phase3_dashboard.md`` through the Phase 4
generator, so it carries the same provenance block as every other report.

The one thing this file cannot measure is the manual Access check (guide test
7: a phone loads the published URL through the Cloudflare Access login, and an
unauthenticated curl gets a login page rather than data). It is recorded as
OPEN with no date until someone runs it and passes ``--access-check-date``.
Recording it as anything else would be a fabricated verification of the
control that stands between licensed data and the open internet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.report import Report, build_provenance  # noqa: E402

DEFAULT_BUNDLE = paths.ROOT / "dashboard" / "earnings"


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


def bundle_section(bundle: Path) -> list[str]:
    """What the rendered bundle contains, measured off disk."""
    meta = _read_json(bundle / "data" / "meta.json")
    board = _read_json(bundle / "data" / "board.json")
    if not meta or not board:
        return [
            f"_No bundle at `{bundle.relative_to(paths.ROOT)}`. Run "
            "`python3 -m engine.dashboard.nightly --no-refresh --no-publish` first._"
        ]

    rows = board.get("rows", [])
    gate_pass = sum(1 for r in rows if r.get("gate_pass") is True)
    scored = sum(1 for r in rows if r.get("scored"))
    no_chain = sum(1 for r in rows if "NO_CHAIN" in (r.get("flags") or []))
    extrapolated = sum(1 for r in rows if r.get("extrapolated"))
    tickers = sorted((bundle / "data" / "tickers").glob("*.json"))
    board_bytes = (bundle / "data" / "board.json").stat().st_size

    lines = [
        "| Item | Value |",
        "|---|---|",
        f"| as of | {meta.get('as_of')} |",
        f"| rendered at | {str(meta.get('generated_at'))[:19].replace('T', ' ')} UTC |",
        f"| horizon | {meta.get('horizon_days')} days |",
        f"| events × strategies | {meta.get('n_events')} events, {len(rows)} board rows |",
        f"| rows with an estimate | {scored} ({scored / len(rows):.0%} of the board) |"
        if rows else "| rows with an estimate | 0 |",
        f"| rows the gate passed | {gate_pass} |",
        f"| rows with no chain | {no_chain} |",
        f"| non-ATM rows (EXTRAPOLATED) | {extrapolated} |",
        f"| fill alpha | {meta.get('fill_alpha')} (mid) |",
        f"| snapshot hash | `{str(meta.get('snapshot_hash'))[:24]}` |",
        f"| board.json | {board_bytes:,} bytes (budget 1,200,000) |",
        f"| per-ticker files | {len(tickers)}, loaded lazily |",
        "",
        "**Champion models behind the numbers**",
        "",
        "| Role | Registry id |",
        "|---|---|",
    ]
    for role, version in sorted((meta.get("model_versions") or {}).items()):
        lines.append(f"| {role} | `{version}` |")
    if not meta.get("model_versions"):
        lines.append("| — | no champions registered |")

    strategies = meta.get("strategies") or {}
    lines += [
        "",
        "**Strategies on the board**",
        "",
        "| Strategy | Enabled | Note |",
        "|---|:--:|---|",
    ]
    for name, block in sorted(strategies.items()):
        note = block.get("detail", "") if not block.get("enabled") else "scored"
        lines.append(f"| {name} | {'yes' if block.get('enabled') else 'no'} | {note} |")

    if not rows:
        lines += [
            "",
            "> **The board is empty, and that is the data's state rather than the "
            "renderer's.** The cached ORATS calendar ends before today, so nothing is "
            "confirmed inside the horizon and the night raises `no_upcoming_events` "
            "instead of inventing one. The Sep-1 quota reset is what closes the gap; "
            "the acceptance suite below scores a real window to prove the pipeline "
            "produces rows when the calendar carries them.",
        ]
    return lines


def freshness_section(bundle: Path) -> list[str]:
    meta = _read_json(bundle / "data" / "meta.json") or {}
    fresh = meta.get("freshness") or {}
    quota = meta.get("quota") or {}
    remaining = quota.get("remaining")
    lines = [
        "| Signal | Value |",
        "|---|---|",
        f"| daily_market through | {fresh.get('daily_market_last_date') or '—: no daily rows read'} |",
        f"| age at render | {fresh.get('daily_market_age_days')} days |"
        if fresh.get("daily_market_age_days") is not None
        else "| age at render | —: no daily rows read |",
        f"| ORATS quota remaining | {remaining:,} |" if isinstance(remaining, int)
        else "| ORATS quota remaining | —: quota ledger has no entry yet |",
        f"| reserve floor | {quota.get('reserve_floor')} calls |",
        f"| late (backfilled) nights | {', '.join(meta.get('late_as_ofs') or []) or 'none'} |",
        "",
        "Staleness is displayed, never silenced: when the refresh step cannot spend "
        "quota the board still renders, and the age of its data is on the model-health "
        "view rather than in a log nobody reads.",
    ]

    flags = (_read_json(bundle / "data" / "flags.json") or {}).get("flags") or []
    lines += ["", "**Flags on this snapshot**", ""]
    if not flags:
        lines.append("None raised.")
    else:
        lines += ["| Kind | Detail |", "|---|---|"]
        for flag in flags:
            detail = flag.get("detail") or json.dumps(
                {k: v for k, v in flag.items() if k != "kind"}, default=str
            )
            lines.append(f"| `{flag.get('kind')}` | {str(detail).replace('|', '/')[:200]} |")
    return lines


def checks_section(checks) -> list[str]:
    if not checks:
        return ["_No acceptance results supplied. Run `checks/phase3_checks.py --json ...`._"]
    lines = ["| # | Check | Result | Detail | Seconds |", "|---:|---|:--:|---|---:|"]
    for i, row in enumerate(checks):
        status = "SKIP" if row.get("skipped") else ("**PASS**" if row["passed"] else "**FAIL**")
        detail = str(row.get("detail", "")).replace("|", "/").replace("\n", " ")[:180]
        lines.append(
            f"| {i} | `{row['name']}` | {status} | {detail} | {row.get('elapsed_s', 0):.0f} |"
        )
    return lines


def ops_section(bundle: Path, *, access_check_date: str | None) -> list[str]:
    meta = _read_json(bundle / "data" / "meta.json") or {}
    cron = meta.get("cron") or {}
    published = paths.ROOT / "dashboard" / "published" / "current"
    published_meta = _read_json(published / "data" / "meta.json")
    lines = [
        "| Concern | State |",
        "|---|---|",
        f"| cron entry | `{cron.get('entry', 'not recorded in the bundle')}` |",
        f"| idempotent re-run | {'yes' if cron.get('idempotent', True) else 'no'} |",
        f"| local publish target | "
        f"{'serving as_of ' + str(published_meta.get('as_of')) if published_meta else 'nothing published locally yet'} |",
        f"| remote target | {'configured' if _remote_configured() else 'not configured — see dashboard/README.md'} |",
        f"| manual Access check (guide test 7) | "
        f"{'passed ' + access_check_date if access_check_date else '**OPEN** — not yet performed'} |",
        "",
        "The remote channel needs steps only the account holder can take (create the "
        "Cloudflare project, put Access in front of it, drop the publish token in "
        "`.env`); the checklist is in `dashboard/README.md`. Until that is done the "
        "bundle publishes locally and `publish_bundle` refuses any target whose "
        "unauthenticated probe returns 200 — the automated half of test 7, which "
        "`access_rule` exercises above.",
    ]
    if not access_check_date:
        lines += [
            "",
            "> **The manual Access check is open.** Nothing in this repository can "
            "prove that a stranger hitting the published URL gets a login page rather "
            "than the board — that takes a phone and a browser. Until someone runs it "
            "and records the date here, the remote channel is unverified, and the "
            "board carries position intent and licensed ORATS-derived quotes.",
        ]
    return lines


def _remote_configured() -> bool:
    if not paths.ENV_FILE.exists():
        return False
    try:
        text = paths.ENV_FILE.read_text()
    except OSError:
        return False
    return any(
        key in text
        for key in ("DASHBOARD_PUBLISH_CMD", "CF_PAGES_TOKEN", "R2_ACCESS_KEY_ID")
    )


def sections(checks, bundle: Path, *, access_check_date: str | None) -> list[dict]:
    return [
        {"title": "The rendered bundle",
         "note": "One renderer produces the bundle; the desk server serves it and the "
                 "publisher ships it. There is exactly one rendering path, so the phone "
                 "view, the desk view and the engine cannot drift — and the nightly "
                 "self-check re-scores board rows through `engine.score` to prove it "
                 "rather than assert it.",
         "body": bundle_section(bundle)},
        {"title": "Data freshness and quota at render time", "body": freshness_section(bundle)},
        {"title": "Acceptance checks",
         "note": "`checks/phase3_checks.py`. The guide's seven tests plus the "
                 "constraints it states as prose, which are only real if something "
                 "enforces them.",
         "body": checks_section(checks)},
        {"title": "Operations and remote access",
         "body": ops_section(bundle, access_check_date=access_check_date)},
        {"title": "Phase-3 accuracy standards",
         "note": "The Phase 4 standards, answered for this phase.",
         "columns": ["#", "Standard", "Status"],
         "align": ["---", "---", "---"],
         "rows": [
             ["1", "Real traded/quoted prices only",
              "**PASS** — the board displays `ScoreResult` fields; every price under "
              "them comes from ORATS chain bid/ask through `engine.structures`"],
             ["2", "Leak audit passed",
              "**PASS** — inherited: every board row is a scoring call, and "
              "`assert_causal` runs inside it"],
             ["3", "Headline numbers walk-forward OOS only",
              "**N/A** — the board forecasts; it reports no backtest headline. The "
              "models behind it carry their walk-forward metrics in the registry, "
              "shown per row as `model_versions`"],
             ["4", "Fill sensitivity shown",
              "**PARTIAL** — every row states the fill alpha it was priced at "
              "(mid, α=0.5). A worst/best toggle on the board is Phase 5 work, when "
              "the measured α̂ replaces the assumption"],
             ["5", "Multiple-testing ledger cited",
              "**N/A** — no spec search happens here; the board runs the promoted "
              "champions"],
             ["6", "Survivorship caveat quantified",
              "**N/A** — the board is forward-looking: its universe is the confirmed "
              "calendar, not a historical sample"],
             ["7", "Prediction ledger",
              "**LIVE** — step 4 of the nightly freezes the ATM board into "
              "`ledger/predictions/` BEFORE rendering, and refuses duplicate row ids. "
              "The strike ladder is not frozen: it is an EXTRAPOLATED view of the same "
              "decision, and freezing it would pad the calibration sample with rows "
              "nobody would trade"],
         ]},
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checks-json", default=None)
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--access-check-date", default=None,
                    help="YYYY-MM-DD the manual Access check was performed and passed")
    ap.add_argument("--out", default=str(paths.REPORTS / "phase3_dashboard.md"))
    args = ap.parse_args(argv)

    checks = _read_json(args.checks_json) if args.checks_json else None
    bundle = Path(args.bundle)
    if args.access_check_date:
        pd.Timestamp(args.access_check_date)  # raises on a malformed date

    passed = sum(1 for c in (checks or []) if c["passed"] and not c.get("skipped"))
    out = paths.assert_writable(Path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)

    context = {
        "kind": "audit",
        "spec": {
            "id": "PHASE-3", "title": "Continuous monitoring dashboard", "type": "descriptive",
            "hypothesis": "Every upcoming print carries, on one page refreshed nightly "
                          "and reachable from a phone, what each strategy expects to "
                          "make and the evidence behind it — with no number on the page "
                          "that the engine would not reproduce.",
        },
        "results": {"headline": {}, "stress": {}, "mc": {}},
        "headline": {}, "backtest": {}, "checklist": [],
        "provenance": build_provenance(
            seeds={},
            input_files=[
                p for p in [
                    Path(args.checks_json) if args.checks_json else None,
                    bundle / "data" / "board.json",
                    bundle / "data" / "meta.json",
                ] if p and p.exists()
            ],
        ),
        "survivorship_note": "",
        "calibration": None,
        "funnel": [
            {"stage": "acceptance checks defined", "events": len(checks or []),
             "note": "checks/phase3_checks.py"},
            {"stage": "checks passing", "events": passed,
             "note": "run with --json to record the detail", "headline": True},
        ],
        "extra_sections": sections(checks, bundle, access_check_date=args.access_check_date),
    }
    Report(context).write(out.parent, filename=out.name)
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
