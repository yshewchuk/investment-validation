"""Smoke-price one native gate variant; no fit, report or recording authority."""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import (
    ExperimentSpec,
    ResolvedExperimentPlan,
    experiment_spec_from_document,
    resolve_experiment_plan,
)
from engine.v2.research._pricing import STRUCTURES
from engine.v2.research.experiment_population import load_population
from engine.v2.research.replay import ReplayResult, replay


@dataclass(frozen=True)
class GateReplayInputs:
    snapshot_id: str
    strategy: str
    fill_alpha: float
    event_ids: tuple[str, ...]
    holdout_as_of_month: str
    random_membership_version: str
    rolling_membership_version: str


@dataclass(frozen=True)
class GateVariantPricing:
    execution_plan: ResolvedExperimentPlan
    stage_inputs: GateReplayInputs
    replay: ReplayResult


def _plan(spec, strategy):
    """Resolve only declarations this native pricing stage actually consumes."""
    plan = resolve_experiment_plan(spec)
    alpha = plan.economic_params.get("fill")
    try:
        valid_alpha = type(alpha) in (int, float) and isfinite(alpha) and 0 <= alpha <= 1
    except OverflowError:
        valid_alpha = False
    if (plan.runner != "native-gate-replay" or plan.price_source != "option_chains"
            or not isinstance(strategy, str) or strategy not in STRUCTURES
            or spec.input_files
            or set(plan.economic_params) != {"fill"} or not valid_alpha):
        raise fail("INVALID_EXPERIMENT_SPEC", "unsupported native gate pricing declaration")
    return plan


def price_variant(repository, snapshot, spec: ExperimentSpec, *, strategy: str,
                  as_of_month: str, event_ids=None) -> GateVariantPricing:
    """Fresh per-arm replay, never evaluation of a caller's prepriced frame.

    The caller pins the snapshot once for the run; every arm resolves its own
    plan and consumes its own fill input. Other economics remain unsupported.
    Gate fitting, identity reservation and durable reports belong to later
    orchestration, so this function has no recording or promotion authority.
    """
    plan = _plan(spec, strategy)
    events = load_population(repository, snapshot, as_of_month=as_of_month,
                             purpose="sweep", event_ids=event_ids)
    if events.empty:
        raise fail("EXPERIMENT_VARIANT_FAILED", "native gate pricing population is empty")
    inputs = GateReplayInputs(
        snapshot.snapshot_id, strategy, float(plan.economic_params["fill"]),
        tuple(events["event_id"]), as_of_month,
        events.iloc[0]["random_membership_version"],
        events.iloc[0]["rolling_membership_version"],
    )
    try:
        result = replay(repository, snapshot, inputs.strategy, events,
                        alphas=(inputs.fill_alpha,), progress_every=0)
    except (ValueError, TypeError, KeyError, OverflowError):
        raise fail("EXPERIMENT_VARIANT_FAILED", "native gate pricing failed") from None
    actual = tuple(result.trades["event_id"])
    if len(actual) != len(inputs.event_ids) or set(actual) != set(inputs.event_ids):
        raise fail("EXPERIMENT_VARIANT_FAILED", "native gate pricing population is incomplete")
    return GateVariantPricing(plan, inputs, result)


def main(argv=None) -> int:
    """Print smoke evidence from the current pin without recording any result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--scope", default="shadow")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--as-of-month", required=True)
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args(argv)
    if not args.no_ledger:
        parser.error("--no-ledger is required; recorded execution is unavailable")
    try:
        spec = experiment_spec_from_document(json.loads(args.spec.read_text()))
        _plan(spec, args.strategy)
        connection = sqlite3.connect(args.catalog.resolve().as_uri() + "?mode=ro",
                                     uri=True, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            repository = Repository(connection, ArtifactStore(args.store_root))
            snapshot = repository.resolve_pinned(args.scope)
            result = price_variant(repository, snapshot, spec, strategy=args.strategy,
                                   as_of_month=args.as_of_month)
        finally:
            connection.close()
    except (DataError, OpsError) as exc:
        print(json.dumps({"refused": exc.code}, sort_keys=True))
        return 2
    except (TypeError, ValueError):
        print(json.dumps({"refused": "INVALID_EXPERIMENT_SPEC"}, sort_keys=True))
        return 2
    except (OSError, sqlite3.Error):
        print(json.dumps({"refused": "INPUT_CHANGED"}, sort_keys=True))
        return 2
    print(json.dumps({"execution_plan": result.execution_plan.as_document(),
                      "stage_inputs": asdict(result.stage_inputs),
                      "priced_events": len(result.replay.trades),
                      "recording_mode": "no-ledger"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
