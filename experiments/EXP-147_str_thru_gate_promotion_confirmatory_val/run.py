#!/usr/bin/env python3
"""EXP-147 — STR-THRU gate promotion: confirmatory validation of the
forecast+analog gate (EXP-145's arm 7, re-run as its own pre-registered spec).

Run:  python3 experiments/EXP-147_str_thru_gate_promotion_confirmatory_val/run.py

Confirmatory, not discovery: EXP-145 found this feature set the best of seven
compared arms. This re-runs ONLY that configuration through the harness under
its own pre-registration, so a promotion decision cites a clean single
hypothesis test rather than the winner of a 7-way search. Pre-registration
lives in spec.yaml; engine.evaluate enforces it.

``main()`` takes optional keyword overrides so other experiments (e.g.
EXP-187) can reuse this runner with a different candidate feature set and/or
a different registered champion to compare against, without copying the
file. Called with no arguments (as EXP-184's wrapper does), behaviour is
BYTE-FOR-BYTE unchanged: the candidate is gate_forecast_analog's full
feature set and the champion comparator is gate_midfill_str_thru.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.evaluate import evaluate  # noqa: E402
from engine.models.training import gate_forecast_analog as ga  # noqa: E402
from engine.v2.research import experiment_trades  # noqa: E402
from experiments import common, common_v2, lib  # noqa: E402

HERE = Path(__file__).resolve().parent
STRATEGY = "STR-THRU"
V2_CATALOG = paths.ROOT / "private" / "ops" / "catalog.sqlite"
V2_STORE_ROOT = paths.ROOT / "private" / "ops"


def main(
    *,
    candidate_features: Sequence[str] | None = None,
    candidate_gate_name: str | None = None,
    candidate_feature_line: str | None = None,
    candidate_provenance_line: str | None = None,
    champion_gate_id: str = "gate_midfill_str_thru",
) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--holdout-as-of-month", required=True)
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
        as_of_month=args.holdout_as_of_month,
    )
    print(f"[{spec['id']}] {len(trades):,} rows / "
          f"{trades['event_id'].nunique():,} events", flush=True)

    dataset = ga.build_dataset(trades, trade_provenance=experiment_trades.PROVENANCE)
    print(f"[{spec['id']}] dataset: {len(dataset):,} rows, "
          f"{len(ga.FEATURES)} features", flush=True)

    features = list(candidate_features) if candidate_features is not None else list(ga.FEATURES)
    gate_name = candidate_gate_name or (ga.STRATEGY + "_forecast_analog")
    feature_line = candidate_feature_line or (
        f"engine.models.training.gate_forecast_analog: {len(ga.FEATURES)} "
        f"features — the incumbent's registered 41 plus "
        f"{', '.join(ga.EXTRA_FEATURES)}."
    )
    provenance_line = candidate_provenance_line or (
        "Live serving verified byte-identical to this training data's "
        "forecast column and to the board's own analog layer before "
        "this experiment ran — see engine/score.py "
        "Scorer._forecast_for_gate / _gate_feature_frame."
    )
    gate, gate_state = common.make_trained_gate(
        gate_name, dataset, features,
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
                    feature_line,
                    "No stored threshold: each walk-forward fold chose its own "
                    f"top-{ga.TOP_FRACTION:.0%} quantile on that fold's training "
                    "predictions (experiments.common.make_trained_gate).",
                    f"Fold interactions recorded: {len(gate_state.stats)}.",
                    provenance_line,
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

    # Champion re-evaluation: a promotion decision needs the named champion
    # comparator scored on the SAME v2 snapshot, trades, repricer, walk-forward
    # and MC settings as the candidate above -- not the stored EXP-145 metrics,
    # which were computed on the pre-v2 legacy trades universe and are not a
    # valid comparison once the candidate runs on a different trades universe.
    # common.make_registered_gate applies the registry's own stored threshold,
    # refit per fold, exactly like EXP-145's arm1_incumbent_model.
    champion_gate, champion_state = common.make_registered_gate(
        STRATEGY, dataset, gate_id=champion_gate_id,
    )

    def champion_extra_sections(result):
        return [{
            "title": "Champion re-evaluation (same v2 snapshot)",
            "body": [
                f"engine.models.training.gate (registered {champion_gate_id}), "
                "refit per fold with the registry's stored threshold "
                "(experiments.common.make_registered_gate), evaluated on the "
                f"identical v2 snapshot {v2_snapshot_id!r}, trades, repricer, "
                "walk-forward and MC settings as the candidate above.",
                f"Fold interactions recorded: {len(champion_state.stats)}.",
            ],
        }]

    champion_spec = copy.deepcopy(spec)
    champion_spec["title"] = f"{spec['title']} — champion re-evaluation ({champion_gate_id})"
    # Legitimately differs from the PLANNED row's hash (different gate, same
    # trades/harness/settings) -- grid_cell=True is the harness's own exemption
    # for exactly this, the same pattern EXP-145's run_arm() uses per arm.
    champion_spec["grid_cell"] = True
    champion_run_dir = HERE / "champion"
    if not args.no_ledger:
        # Register the grid cell's exact spec_hash BEFORE evaluate() runs, not
        # only after: if the run dies partway, results/metrics_<hash>.json and
        # REPORT.md must never exist with zero ledger trace of the attempt.
        from datetime import datetime, timezone

        lib.ledger_append([{
            "id": champion_spec.get("id", ""),
            "spec_hash": lib.spec_hash(champion_spec),
            "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
            "stage": "planned",
            "oos_mean_mid": "",
            "sharpe_trade": "",
            "promoted": "False",
        }])
    champion_result = evaluate(
        champion_spec, trades, gate=champion_gate, run_dir=champion_run_dir,
        repricer=repricer, spy_daily=spy,
        extra_sections=champion_extra_sections,
    )
    if not args.no_ledger:
        lib.record_evaluation(champion_run_dir, champion_spec, champion_result.results)
    print(f"[{spec['id']}] champion report: {champion_result.report_path}", flush=True)
    print(f"[{spec['id']}] champion headline: mean={champion_result.results['headline'].get('mean')} "
          f"cagr={champion_result.results['headline'].get('cagr')} "
          f"sharpe_trade={champion_result.results['headline'].get('sharpe_trade')}",
          flush=True)


if __name__ == "__main__":
    main()
