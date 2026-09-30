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

import copy
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

    # Champion re-evaluation: a promotion decision needs gate_midfill_str_thru
    # scored on the SAME v2 snapshot, trades, repricer, walk-forward and MC
    # settings as the candidate above -- not the stored EXP-145 metrics, which
    # were computed on the pre-v2 legacy trades universe and are not a valid
    # comparison once the candidate runs on a different trades universe.
    # common.make_registered_gate applies the registry's own stored threshold,
    # refit per fold, exactly like EXP-145's arm1_incumbent_model.
    champion_gate, champion_state = common.make_registered_gate(
        STRATEGY, dataset, gate_id="gate_midfill_str_thru",
    )

    def champion_extra_sections(result):
        return [{
            "title": "Champion re-evaluation (same v2 snapshot)",
            "body": [
                "engine.models.training.gate (registered gate_midfill_str_thru), "
                "refit per fold with the registry's stored threshold "
                "(experiments.common.make_registered_gate), evaluated on the "
                f"identical v2 snapshot {v2_snapshot_id!r}, trades, repricer, "
                "walk-forward and MC settings as the candidate above.",
                f"Fold interactions recorded: {len(champion_state.stats)}.",
            ],
        }]

    champion_spec = copy.deepcopy(spec)
    champion_spec["title"] = f"{spec['title']} — champion re-evaluation (gate_midfill_str_thru)"
    # Legitimately differs from the PLANNED row's hash (different gate, same
    # trades/harness/settings) -- grid_cell=True is the harness's own exemption
    # for exactly this, the same pattern EXP-145's run_arm() uses per arm.
    champion_spec["grid_cell"] = True
    champion_result = evaluate(
        champion_spec, trades, gate=champion_gate, run_dir=HERE,
        repricer=repricer, spy_daily=spy,
        extra_sections=champion_extra_sections,
    )
    if not args.no_ledger:
        lib.record_evaluation(HERE, champion_spec, champion_result.results)
    print(f"[{spec['id']}] champion report: {champion_result.report_path}", flush=True)
    print(f"[{spec['id']}] champion headline: mean={champion_result.results['headline'].get('mean')} "
          f"cagr={champion_result.results['headline'].get('cagr')} "
          f"sharpe_trade={champion_result.results['headline'].get('sharpe_trade')}",
          flush=True)


if __name__ == "__main__":
    main()
