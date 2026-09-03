#!/usr/bin/env python3
"""Scaffold a new experiment: numbered folder, pre-registered spec, ledger row.

    python3 experiments/new_experiment.py --title "CAL-P exact-spec backtest" \
        --hypothesis "Short ~1DTE put + long back put opened together pre-print
        is positive at mid fills." [--id EXP-101] [--dir experiments/]

Creates::

    experiments/EXP-101_calp_exact_spec/
      spec.yaml      # pre-registered: stamped with preregistered_at NOW (UTC)
      run.py         # template wired to engine.evaluate
      results/       # run artifacts land here (run_log.jsonl, metrics, cache)
      figures/       # report figures
      REPORT.md      # written by the evaluation, never by hand

and appends the PLANNED row to ``experiments/LEDGER.csv``.

The ``preregistered_at`` stamp is the enforcement point: ``engine.evaluate``
refuses the OOS stage if the stamp is missing or later than the first recorded
run, so a spec edited after results are seen can never silently become
"pre-registered". To change a hypothesis after running, scaffold a NEW
experiment — that is the point of the multiple-testing ledger.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import lib  # noqa: E402

SPEC_TEMPLATE = """\
id: {id}
title: {title}
hypothesis: {hypothesis}
strategy: {strategy}
price_source: ORATS chains (bid/ask, validated +/-2-3%) + engine.replay pricing
primary_spec:
  # The ONE configuration whose OOS result is the headline. Everything else
  # in the grid is secondary and is reported as such — this is the
  # anti-post-hoc-selection mechanism, keep it strict.
  front_dte: 1
  back_dte: 20
  entry: T-1
  strike: ATM-same
grid:
  # Optional. Every cell is evaluated and logged, none of them is the headline.
  {{}}
data_snapshot: {snapshot}
walk_forward:
  unit: year
  min_train_years: 2
equity_mode: cashflow
has_short_leg: false
promotion_target: null
preregistered_at: "{preregistered_at}"
"""

RUN_TEMPLATE = '''#!/usr/bin/env python3
"""{id} — {title}.

Run:  python3 experiments/{folder}/run.py

Pre-registration lives in spec.yaml; engine.evaluate enforces it. The primary
spec's OOS result is the headline; grid cells are secondary.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.evaluate import evaluate  # noqa: E402
from experiments import lib  # noqa: E402

HERE = Path(__file__).resolve().parent


def build_trades():
    """Return the priced trade frame (engine.replay output shape).

    TODO(experimenter): generate the candidate trades via engine.replay over
    the event universe, or load a pre-built priced dataset. One row per
    (event x fill_alpha) with at least the columns engine.evaluate requires:
    event_id, ticker, event_date, entry_date, exit_date, fill_alpha,
    entry_cost, exit_value, ret.
    """
    raise NotImplementedError("wire the candidate trade generator here")


def main() -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    trades = build_trades()

    # Grid: the primary spec runs first and is the headline; each grid cell
    # then runs as a secondary spec (its own ledger row, labeled in the
    # report appendix).
    result = evaluate(spec, trades, run_dir=HERE)
    lib.record_evaluation(HERE, spec, result.results)

    grid = spec.get("grid") or {{}}
    for key, values in grid.items():
        for value in values:
            cell = dict(spec)
            cell["primary_spec"] = dict(spec["primary_spec"])
            cell["primary_spec"][key] = value
            # Grid cells legitimately differ from the registered primary spec;
            # the label exempts them from the spec-hash continuity check —
            # they are secondary results, never the headline.
            cell["grid_cell"] = True
            cell_result = evaluate(cell, trades, run_dir=HERE)
            lib.record_evaluation(HERE, cell, cell_result.results)

    print(f"report: {{result.report_path}}")


if __name__ == "__main__":
    main()
'''


def scaffold(title: str, hypothesis: str, *, exp_id: str | None = None,
             strategy: str = "UNSPECIFIED", root: Path | None = None,
             ledger_path: Path | None = None) -> Path:
    root = Path(root or lib.EXPERIMENTS_DIR)
    number = lib.parse_experiment_id(exp_id) if exp_id else lib.next_experiment_number(root)
    exp_id = f"EXP-{number}"
    folder = root / f"{exp_id}_{lib.slugify(title)}"
    if folder.exists():
        raise FileExistsError(f"{folder} already exists — pick a new id or title")
    existing_ids = {p.name.split("_")[0] for p in root.glob("EXP-*")}
    if exp_id in existing_ids:
        raise FileExistsError(f"{exp_id} already exists under {root}")

    import json

    snapshot = None
    from engine import paths

    if paths.SNAPSHOT_FILE.exists():
        try:
            snapshot = json.loads(paths.SNAPSHOT_FILE.read_text()).get("snapshot")
        except (ValueError, OSError):
            snapshot = None

    now = datetime.now(tz=timezone.utc)
    stamp = now.isoformat(timespec="seconds")
    folder.mkdir(parents=True)
    (folder / "results").mkdir()
    (folder / "figures").mkdir()

    # Titles and hypotheses are free prose and routinely contain YAML-breaking
    # characters (colons, percents). Emit them as JSON-quoted scalars, which
    # are valid YAML double-quoted strings, so the spec always parses.
    spec_text = SPEC_TEMPLATE.format(
        id=exp_id,
        title=json.dumps(title),
        hypothesis=json.dumps(hypothesis.strip().replace("\n", " ")),
        # `*` is a YAML alias marker and `CAL-P` is fine unquoted; quoting every
        # strategy the same way means a programme-wide spec (strategy: "*")
        # scaffolds instead of writing a spec.yaml that will not parse.
        strategy=json.dumps(strategy),
        snapshot=snapshot or "UNKNOWN", preregistered_at=stamp,
    )
    (folder / "spec.yaml").write_text(spec_text)
    (folder / "run.py").write_text(
        RUN_TEMPLATE.format(id=exp_id, title=title, folder=folder.name))
    (folder / "REPORT.md").write_text(
        f"# {exp_id} — {title}\n\n*No evaluation has run yet. An experiment "
        "without a REPORT.md generated by engine.report does not exist.*\n")

    lib.ledger_append([{
        "id": exp_id,
        "spec_hash": lib.spec_hash(lib.load_spec(folder / "spec.yaml")),
        "date": now.strftime("%Y-%m-%d"),
        "stage": "planned",
        "oos_mean_mid": "",
        "sharpe_trade": "",
        "promoted": "False",
    }], path=ledger_path)
    return folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--id", default=None, help="EXP-NNN (default: next free >= 101)")
    parser.add_argument("--strategy", default="UNSPECIFIED",
                        help="CAL-P | STR-THRU | STR-RUNUP | ...")
    parser.add_argument("--dir", default=None, help="experiments root (default: repo)")
    args = parser.parse_args(argv)
    try:
        folder = scaffold(args.title, args.hypothesis, exp_id=args.id,
                          strategy=args.strategy,
                          root=Path(args.dir) if args.dir else None)
    except (FileExistsError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(f"scaffolded {folder}")
    print(f"spec: {folder / 'spec.yaml'}  (preregistered_at stamped, PLANNED row in LEDGER.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
