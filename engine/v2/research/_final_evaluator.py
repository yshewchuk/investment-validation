"""Private frozen single-winner final evaluator (#490); no fit, arm or callback."""
from __future__ import annotations

import math
from dataclasses import dataclass

from engine.v2.data import errors
from engine.v2.research._final_population import FinalPopulation, load_final_population
from engine.v2.research._trades_publish import read_existing_trades
from engine.v2.research._trades_revisions import PROVENANCE


@dataclass(frozen=True)
class SetMetrics:
    n_events: int
    n_trades: int
    n_without_trade: int
    total_ret: float
    mean_ret: float | None      # None when n_trades == 0
    win_rate: float | None      # fraction of trades with ret > 0; None when n_trades == 0

    def __post_init__(self) -> None:
        if self.n_trades == 0 and (self.mean_ret is not None or self.win_rate is not None):
            raise ValueError("mean and win rate must be None when a set has no trade")


@dataclass(frozen=True)
class FinalMetrics:
    snapshot_id: str
    holdout_as_of_month: str
    variant_id: str
    random: SetMetrics
    rolling: SetMetrics
    overlap_count: int
    ambiguous_count: int
    # NEVER add a combined/average/pooled field.


def evaluate_final_holdout(repository, request, population) -> FinalMetrics:
    """Deny a non- or mismatched population, then score each held-out set separately."""
    if not isinstance(population, FinalPopulation):
        raise errors.fail("HOLDOUT_ACCESS_DENIED", "a validated final population is required")
    expected = load_final_population(repository, request)
    if population != expected:
        raise errors.fail("HOLDOUT_ACCESS_DENIED", "the population does not match the request")
    snapshot = repository.resolve(request.snapshot_id)
    try:
        trades = read_existing_trades(repository, snapshot)
    except ValueError:
        if repository.fragment_records(snapshot, "trades"):
            raise
        raise errors.fail(
            "POPULATION_COLLAPSED",
            "the trades table has no fragments in this snapshot",
            details={"snapshot_id": snapshot.snapshot_id},
        ) from None
    winner = trades[
        (trades["provenance"] == PROVENANCE) & (trades["variant"] == request.variant_id)
    ]
    held_out = set(population.random_event_ids) | set(population.rolling_event_ids)
    winner = winner[winner["event_id"].isin(held_out)]
    if winner.empty:
        raise errors.fail(
            "POPULATION_COLLAPSED",
            "no winner trade in either held-out set",
            details={"snapshot_id": snapshot.snapshot_id, "variant_id": request.variant_id},
        )
    rets = [float(value) for value in winner["ret"]]
    if any(not math.isfinite(value) for value in rets):
        raise errors.fail("CONTRACT_MISMATCH", "a winner trade has a non-finite ret")
    metrics = {variant: _set_metrics(winner, population, variant)
               for variant in ("random", "rolling")}
    return FinalMetrics(
        snapshot_id=population.snapshot_id,
        holdout_as_of_month=population.holdout_as_of_month,
        variant_id=request.variant_id,
        random=metrics["random"],
        rolling=metrics["rolling"],
        overlap_count=population.overlap_count,
        ambiguous_count=population.ambiguous_count,
    )


def _set_metrics(winner, population, variant):
    """One held-out set's scores; overlap counts in both, sets never merge."""
    event_ids = getattr(population, f"{variant}_event_ids")
    held = winner[winner["event_id"].isin(event_ids)]
    rets = [float(value) for value in held["ret"]]
    n_trades = len(rets)
    return SetMetrics(
        n_events=len(event_ids),
        n_trades=n_trades,
        n_without_trade=len(set(event_ids) - set(held["event_id"])),
        total_ret=sum(rets),
        mean_ret=(sum(rets) / n_trades) if n_trades else None,
        win_rate=(sum(1 for value in rets if value > 0) / n_trades) if n_trades else None,
    )