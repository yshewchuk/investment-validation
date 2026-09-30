#!/usr/bin/env python3
"""EXP-147 — STR-THRU gate promotion: confirmatory validation of the
forecast+analog gate (EXP-145's arm 7, re-run as its own pre-registered spec).

Run:  python3 experiments/EXP-147_str_thru_gate_promotion_confirmatory_val/run.py

Confirmatory, not discovery: EXP-145 found this feature set the best of seven
compared arms. This re-runs ONLY that configuration through the harness under
its own pre-registration, so a promotion decision cites a clean single
hypothesis test rather than the winner of a 7-way search. Pre-registration
lives in spec.yaml; engine.evaluate enforces it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.evaluate import evaluate  # noqa: E402
from engine.models.training import gate_forecast_analog as ga  # noqa: E402
from experiments import common, common_v2, lib  # noqa: E402

HERE = Path(__file__).resolve().parent
STRATEGY = "STR-THRU"
V2_CATALOG = paths.ROOT / "private" / "ops" / "catalog.sqlite"
V2_STORE_ROOT = paths.ROOT / "private" / "ops" / "objects"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    spec = lib.load_spec(HERE / "spec.yaml")
    v2_snapshot_id = spec.get("v2_snapshot_id")
    if not v2_snapshot_id:
        raise SystemExit(
            f"[{spec['id']}] spec.yaml is missing v2_snapshot_id — refusing to resolve "
            "the v2 trades snapshot as \"latest\"; set it explicitly once the pinned "
            "snapshot exists."
        )
    print(f"[{spec['id']}] loading engine trades …", flush=True)
    trades = common_v2.load_v2_trades(
        STRATEGY, catalog=V2_CATALOG, store_root=V2_STORE_ROOT, snapshot_id=v2_snapshot_id,
    )
    print(f"[{spec['id']}] {len(trades):,} rows / "
          f"{trades['event_id'].nunique():,} events", flush=True)

    dataset = ga.build_dataset(trades)
    print(f"[{spec['id']}] dataset: {len(dataset):,} rows, "
          f"{len(ga.FEATURES)} features", flush=True)

    gate, gate_state = common.make_trained_gate(
        ga.STRATEGY + "_forecast_analog", dataset, list(ga.FEATURES),
        top_fraction=ga.TOP_FRACTION,
    )
    spy = common.load_spy_daily()
    # same pinned snapshot as trades — never a second resolve
    repricer = common_v2.make_v2_repricer(
        STRATEGY, catalog=V2_CATALOG, store_root=V2_STORE_ROOT, snapshot_id=v2_snapshot_id,
    )

    def required_outputs(result):
        return [
            {
                "title": "Gate construction (promotion candidate)",
                "body": [
                    f"engine.models.training.gate_forecast_analog: {len(ga.FEATURES)} "
                    f"features — the incumbent's registered 41 plus "
                    f"{', '.join(ga.EXTRA_FEATURES)}.",
                    "No stored threshold: each walk-forward fold chose its own "
                    f"top-{ga.TOP_FRACTION:.0%} quantile on that fold's training "
                    "predictions (experiments.common.make_trained_gate).",
                    f"Fold interactions recorded: {len(gate_state.stats)}.",
                    "Live serving verified byte-identical to this training data's "
                    "forecast column and to the board's own analog layer before "
                    "this experiment ran — see engine/score.py "
                    "Scorer._forecast_for_gate / _gate_feature_frame.",
                ],
            },
            {
                "title": "Data provenance (v2)",
                "body": [
                    f"Trades and T±1 repricing both read v2 snapshot {v2_snapshot_id!r} "
                    "(experiments.common_v2.load_v2_trades / make_v2_repricer) — never the legacy "
                    "curated trades store this experiment previously fingerprinted via input_files.",
                ],
            },
        ]

    result = evaluate(
        spec, trades, gate=gate, run_dir=HERE,
        repricer=repricer, spy_daily=spy,
        extra_sections=required_outputs,
    )
    if not args.no_ledger:
        lib.record_evaluation(HERE, spec, result.results)
    print(f"[{spec['id']}] report: {result.report_path}", flush=True)
    print(f"[{spec['id']}] headline: mean={result.results['headline'].get('mean')} "
          f"cagr={result.results['headline'].get('cagr')} "
          f"sharpe_trade={result.results['headline'].get('sharpe_trade')}",
          flush=True)


if __name__ == "__main__":
    main()
