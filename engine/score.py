"""``score(ticker, strategy, strike, expiry, as_of)`` — the Phase 1 public API.

Two estimation layers, always both, never averaged:

**Model layer.** The champion models predict a quantity — the size of the move,
the implied move at T−1 — and that prediction is turned into a P&L distribution
by drawing from the model's own held-out residuals and pushing each draw through
an empirically-calibrated payoff (:mod:`engine.payoff`), against the *real*
entry cost from the *real* chain. ``exp_pnl_model`` is that distribution's mean;
``win_model`` is the share of it above zero. No normality is assumed anywhere:
earnings residuals are skewed and fat-tailed, and the tails are the part that
decides whether a long-vol structure pays.

**Analog layer.** The empirical distribution of matched historical trades from
:mod:`engine.analogs`, closed before the decision date, priced by the same
replay code at the same fill alpha.

When they disagree, ``LAYER_DISAGREE`` is raised as a flag and both numbers are
carried forward. Averaging them would hide the one situation the two layers
exist to detect: a model extrapolating past its evidence.

Three hard rules the guide sets, enforced here rather than documented:

* **CAL-P is not scored.** It returns ``UNVALIDATED_STRUCTURE`` and no numbers.
  The +2.0%/trade figure people remember belongs to EXP-046b, which tested
  straddle legs entered at T−14 and unwound *before* the print — a different
  trade. Until Phase 2 backlog items 1–2 run, there is no evidence for this
  structure, and a plausible number here would be worse than a refusal.
* **Non-ATM strikes are labelled EXTRAPOLATED**, until the moneyness edge-decay
  experiment promotes them.
* **Determinism.** ``(request, snapshot)`` → identical result. The snapshot hash
  is embedded in the output, so a number can always be traced to the data that
  produced it.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine import paths
from engine.analogs import AnalogMatcher, AnalogSet, bucket_frame
from engine.audit import FeatureVector, assert_causal, assert_decision_causal
from engine.calendar import trading_calendar
from engine.data import manifest, store
from engine.data.features import tier4
from engine.entry_rules import rule_for
from engine.structure_registry import live_strategies, superseded_by
from engine.features import (
    DAILY_STATE_COLUMNS,
    EVENT_HISTORY_FEATURES,
    FeatureContext,
    add_absolute_features,
    add_quote_indicators,
    daily_state_frame,
    entry_feature_frame,
    live_features,
    session_for,
)
from engine.fills import BAD_QUOTE_COST_PCT, MID, FillModel
from engine.forecast_sizing import (
    FORECAST_SIZED,
    describe as describe_sizing,
    forecast_params,
)
from engine.models.registry import Registry, RegistryError, load_registry
from engine.payoff import (
    PayoffError,
    PayoffMap,
    RunupPayoffSurface,
    fit_payoff,
    fit_runup_payoff,
    scale_runup_move,
    simulate_returns,
    simulate_runup_returns,
)
from engine.replay import (
    ChainIndex,
    legs_exit_spot,
    legs_spot_dte,
    load_chain_index,
    plan_events,
)
from engine.structures import (
    STRUCTURES,
    ExpirySelector,
    LegSpec,
    StrikeSelector,
    Structure,
    LadderTooCoarse,
    StructureError,
)

__all__ = [
    "ScoreRequest",
    "ScoreResult",
    "Scorer",
    "score",
    "score_calendar",
    "DISABLED_STRATEGIES",
    "FLAGS",
]

#: Structures the scorer refuses to put a number on, and why.
DISABLED_STRATEGIES = {
    "CAL-P": (
        "The exact spec (put legs, entry shortly before the print with a ~1 DTE "
        "front, held THROUGH the print, both legs closed together after) has "
        "never been backtested. EXP-046b tested straddle legs at T-14 unwound "
        "pre-print — a different structure. Phase 2 backlog 1-2 must run first."
    ),
    "CND-P": (
        "EXP-121 registered and ran the risk-mechanics validation (defined-risk "
        "falsification, assignment exposure, the oracle ceiling) but nothing has "
        "reviewed its result or decided to promote the structure. There is no "
        "gate for it — the STR-THRU-shaped one the mechanics call for is not yet "
        "registered — and no fill/execution evidence beyond the replay. Added to "
        "STRUCTURES for engine.replay/build_trades so EXP-121 could price it; "
        "that registry is shared with the live board's default strategy list, "
        "which is what put it here before a single trade should ever be "
        "recommended off it."
    ),
}

FLAGS = (
    "UNVALIDATED_STRUCTURE",
    "EXTRAPOLATED",
    "LAYER_DISAGREE",
    "THIN_HISTORY",
    "THIN_ANALOGS",
    "NO_CHAIN",
    "WIDE_MARKET",
    "BAD_QUOTE",
    "NO_MODEL",
    "MISSING_FEATURES",
    "NO_PAYOFF_MAP",
    "QUOTE_REPAIRED",
    "PROJECTED_CALENDAR",
    "OUT_OF_DOMAIN",
    # A validated structure that simply lost to a better one in its family.
    # Deliberately NOT UNVALIDATED_STRUCTURE: that flag means "nobody has shown
    # this works", and stamping it on a shape with three completed experiments
    # behind it would misreport the evidence rather than the decision. See
    # engine/structure_registry.py.
    "SUPERSEDED",
    # A structure whose shape comes from a forecast, on an event that has none.
    # Distinct from MISSING_FEATURES: the model ran, the row simply cannot be
    # given a shape, and pricing a default-width one instead would put a number
    # on the board for a trade nobody chose.
    "NO_FORECAST",
    # Two legs resolved onto one contract, so the structure was refused. Kept
    # apart from NO_CHAIN deliberately: the chain is present and correct, and a
    # reader who sees NO_CHAIN goes and re-pulls quotes that are already fine.
    # What is too coarse is the ticker's listed strike ladder relative to the
    # width asked for — see guides/coarse_ladder_collision.md.
    "COARSE_LADDER",
)

#: A ticker with fewer prior events than this is the regime where the size model
#: measurably degrades (EXP-024 addendum 2).
THIN_HISTORY_EVENTS = 4

#: Market-cap floor of the gates' training universe. The `<1B` slice joined the
#: target set 2026-09-01 (engine/data/pulls/sep2026_plan.py) with near-zero
#: prior coverage, and EXP-118 showed retraining on the expanded universe does
#: not clear the champions. Until a gate trained on the expanded universe
#: promotes, the decision is withheld for names the champion never saw: a gate
#: call on a name outside its evidence is an undetermined result, not a trade.
#:
#: NOT re-examined when the computed-moves clause was removed on 2026-09-06,
#: and it should be. That clause was justified by the same "promoted before the
#: universe expanded" argument, and the registry now records both champion
#: gates as promoted 2026-09-05 rather than 2026-08-30 — while the panel they
#: trained on carries 22,416 pre-2026 rows below this floor across 1,429
#: tickers. Whether those rows reached the replay trade set is the open
#: question; EXP-118's conclusion is what holds the floor up, so retiring it
#: needs an experiment rather than a reading of the dates.
GATE_MCAP_FLOOR = 1e9


#: |strike/spot − 1| within this is "at the money" — the only region the current
#: evidence covers.
ATM_TOLERANCE_PCT = 2.0

#: Draws used to turn a point prediction into a P&L distribution.
MODEL_DRAWS = 4000

#: Below this, a quoted implied move is treated as ABSENT rather than displayed.
#:
#: EXP-110 established that `or_implied` uses exactly 0 for "no quote" on 25.6%
#: of `daily_market`. It is not only exact zeros: a further 8.3% of rows fall in
#: (0, 1), and that band is degenerate rather than a population of low-vol
#: names — its median is 0.00065% and even its 99th percentile is 0.88%. Real
#: quotes are the 66.1% at or above 1%, whose median is 8.25%.
#:
#: A first version of this guard tested `> 0` and let 1e-06 through, which
#: produced a model/market ratio of 10,258,753 on the board.
#:
#: The 1.0 threshold is a judgement call — mine, not the data's. It is set where
#: the degenerate band ends rather than where a genuine quote is implausible,
#: and it fails toward silence: a real sub-1% quote would be shown as "no quote"
#: rather than as a number that makes a ratio meaningless.
MIN_QUOTED_IMPLIED_MOVE = 1.0

#: Panel columns read at the last pre-print close. Supplied to a model only when
#: the decision is taken at that close; see ``Scorer._features``.
_PANEL_MARKET_BLOCK = (
    "or_implied", "or_skewing", "or_contango", "or_rvol30", "or_exern30",
    # `or_exern_z252` was here and is deliberately gone: it is quarantined
    # (engine.features.QUARANTINED_FEATURES) because the stored values leak the
    # future on 507 panel rows. It was never consumed — `model_inputs` is built
    # from the champion's own feature list, not from this one — so copying it
    # into the feature frame only made it available to be adopted by mistake.
    # TODO(2026-Q4): no action here when the column is deleted.
    "or_iv30", "or_iee", "or_fwd90_30", "or_fexern90_30",
    "mcap_log", "mcap_usd", "spy_ret21", "spy_ret63", "spy_ret252",
    "spy_dd252", "spy_vol20", "dist_high", "dist_ema", "ret5", "ret10", "ret20",
    "abs_dist_high", "abs_dist_ema", "has_implied_quote",
)


# --------------------------------------------------------------------------
# request / result
# --------------------------------------------------------------------------


#: Ladder strikes step this fraction of spot either side of ATM.
LADDER_STEP = 0.025

#: Decimals a ladder strike is quantized to. `ScoreRequest.key()` renders the
#: strike at four places and the dashboard stores floats at six, so a strike
#: carrying more precision than that is a strike the round trip cannot return:
#: `spot * 0.975` on a $369.59 name is 360.35024999999996, the bundle keeps
#: 360.35025, and the two format to DIFFERENT four-place keys — a different
#: bootstrap seed, and a self-check mismatch on a row nothing is wrong with.
#: An option strike is quoted in cents, so four places is already generous.
LADDER_STRIKE_DP = 4


def ladder_strike(spot: float, offset: float) -> float:
    """A strike `offset` away from `spot`, quantized so it round-trips."""
    return round(float(spot) * (1.0 + float(offset)), LADDER_STRIKE_DP)


@dataclass(frozen=True)
class ScoreRequest:
    ticker: str
    strategy: str
    as_of: pd.Timestamp | None = None
    event_date: pd.Timestamp | None = None
    strike: float | None = None
    expiry: pd.Timestamp | None = None
    fill: FillModel = MID
    variant: str | None = None
    session: str | None = None
    #: Decide this many sessions before the entry close. ``None`` keeps the
    #: structure's own setting, which for every shipped structure is "decide at
    #: the entry close". Opt-in per request so the nightly's champion path is
    #: untouched while the T−2 variant is still unvalidated.
    decision_offset: int | None = None
    #: Price a not-yet-tradeable entry on the newest chain within this many
    #: SESSIONS of the decision, instead of returning NO_CHAIN. This is what
    #: lets the board show a trade several days out. `None` keeps the strict
    #: behaviour: no chain for the entry date, no price.
    quote_max_age_sessions: int | None = None
    #: Ceiling on the stale-quote fallback: never substitute a chain dated
    #: after this. ``None`` means no ceiling.
    #:
    #: The fallback above anchors on the ENTRY date, which on a forward board
    #: is in the future. Live that is harmless — no chain exists between today
    #: and the entry — but replaying a past night against a store that has
    #: since been refreshed, it reaches forward past the night being replayed.
    #: :func:`score_calendar` pre-loads the fallback key anchored on ``as_of``
    #: instead, so the two disagreed: the board reported NO_CHAIN for a key its
    #: index lacked while a re-score fetched the same key on demand and priced
    #: it. Eight of twenty rows in the historical dry-run diverged that way.
    #: Setting this makes both paths refuse the same chains.
    chain_as_of: pd.Timestamp | None = None
    #: Parameters handed to the structure factory, for a structure whose SHAPE
    #: varies per event rather than being fixed by its strategy code — a
    #: per-event tent width, a different back DTE, a shifted anchor. Without
    #: this the scorer can only ever price a strategy's default parameterisation,
    #: which is why every variant so far has had to be a separate replay rather
    #: than something the live board could show.
    #:
    #: The factory validates its own arguments, so an unknown key raises rather
    #: than being silently ignored — a structure quietly priced at its default
    #: while the row claims otherwise is the failure this must not have.
    structure_params: Mapping[str, Any] | None = None

    def key(self) -> str:
        """Stable identity, for the deterministic bootstrap seed."""
        parts = [
            self.ticker,
            self.strategy,
            str(pd.Timestamp(self.as_of).date()) if self.as_of is not None else "",
            str(pd.Timestamp(self.event_date).date()) if self.event_date is not None else "",
            f"{self.strike:.4f}" if self.strike is not None else "",
            str(pd.Timestamp(self.expiry).date()) if self.expiry is not None else "",
            f"{self.fill.alpha:.4f}",
            self.variant or "",
            "" if self.decision_offset is None else f"d{self.decision_offset:+d}",
            "" if self.quote_max_age_sessions is None else f"q{self.quote_max_age_sessions}",
            "" if self.chain_as_of is None
            else f"c{pd.Timestamp(self.chain_as_of).date()}",
            # Two events priced at different structure parameters are different
            # TRADES, so they must not share an identity or a bootstrap seed.
            # Sorted, so an equal dict always renders equal.
            "" if not self.structure_params else ",".join(
                f"{k}={self.structure_params[k]!r}" for k in sorted(self.structure_params)
            ),
        ]
        return "|".join(parts)


@dataclass
class ScoreResult:
    ticker: str
    strategy: str
    as_of: pd.Timestamp
    event_date: pd.Timestamp | None = None
    session: str | None = None

    # model layer
    exp_pnl_model: float | None = None
    win_model: float | None = None
    #: The pre-recalibration probability, kept for diagnostics. ``win_model`` is
    #: the shipped (recalibrated) value whenever a recalibration map is available.
    win_model_raw: float | None = None
    model_p10: float | None = None
    model_p90: float | None = None

    # analog layer
    exp_pnl_analog: float | None = None
    win_analog: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_analogs: int = 0
    analog_widened: int = 0

    #: The simulated expectation the TWIN-P5 gate turns on, and the share of
    #: draws that finished profitable. Recorded whether or not the row trades:
    #: a reader judging a decline needs the number it was declined on.
    exp_pnl_sim: float | None = None
    win_sim: float | None = None

    #: The resolved legs, in the order the structure declares them — what to
    #: BUY and SELL, at which strike, in what quantity, and the quote each
    #: price came from.
    #:
    #: A row that recommends a five-strike structure and does not say which five
    #: strikes is not actionable: the reader has to rebuild the geometry from
    #: `structure_params` and a ladder they cannot see. The snapped strikes are
    #: also not the ones the forecast asked for — median 2.1% off, p90 15.1% —
    #: so the shape on the board is the only authority on what to trade.
    legs: list = field(default_factory=list)

    # gate
    gate_score: float | None = None
    gate_threshold: float | None = None
    gate_pass: bool | None = None

    # structure geometry, read off the priced legs
    #: Mean relative spread across the entry legs. One leg's spread says little;
    #: eight legs is sixteen crossings round trip, and this is what an
    #: arithmetic entry rule tests before it lets that happen.
    rel_spread: float | None = None
    #: The structure's own width parameter in dollars, for a structure that
    #: declares which two legs define it (`params["width_legs"]`). `w` for
    #: TWIN-P, and the term `cost < w` is tested against.
    structure_width: float | None = None
    #: Terminal payoff at the structure's peak, in dollars — `structure_width`
    #: times the shape's declared `peak_multiple`. The reward term is
    #: `cost < peak / 2`, which is `cost < w` only for a shape whose peak is
    #: `2w`; carrying the peak explicitly is what keeps that rule honest when a
    #: shape peaks somewhere else (TWIN-P5's tight wing peaks at `a`, not `2a`).
    structure_peak: float | None = None

    # forecast sizing (Tier 4)
    #: The forecast that set this structure's shape, in percent of spot.
    forecast_abs_move: float | None = None
    #: Its 80% band and the SD of the held-out errors behind it, from the SAME
    #: fold model and the same residual pool the stored table used. A width
    #: borrowed from a different fit than the centre would be two answers to
    #: one question.
    forecast_p10: float | None = None
    forecast_p90: float | None = None
    forecast_sd: float | None = None
    #: Which feature model produced it, and which fold's fit. A row sized by a
    #: forecast has to say which model sized it — otherwise a champion
    #: promotion silently changes what the board recommended and nothing
    #: records that it did.
    forecast_model: str | None = None
    forecast_fold: pd.Timestamp | None = None
    #: The parameters actually handed to the structure factory.
    structure_params: dict | None = None
    #: Complete selection rule, including defaults and explicit ladder overrides.
    structure_spec: dict | None = None
    variant: str | None = None

    # the trade being scored
    entry_date: pd.Timestamp | None = None
    exit_date: pd.Timestamp | None = None
    #: Earlier of the decision date and the entry date — the cutoff every
    #: piece of evidence behind this score respects.
    evidence_cutoff: pd.Timestamp | None = None
    strike: float | None = None
    expiry: pd.Timestamp | None = None
    entry_cost: float | None = None
    #: The close the entry was PRICED on. Equal to ``entry_date`` for every
    #: structure decided at its entry, and one or more sessions earlier for one
    #: decided early — in which case `entry_cost` is an estimate for the entry,
    #: not the fill. Say so on the board; see guides/str_thru_t2_decision.md §6.3
    #: for the measured drift between the two.
    quote_date: pd.Timestamp | None = None
    #: Sessions between `quote_date` and `entry_date`. 0 is a contemporaneous
    #: quote; anything above it is an estimate whose staleness the board must
    #: show, because the fill will be at the entry close, not this one.
    quote_age_sessions: int | None = None
    #: The bound the quote fallback ran under. Recorded because it is part of
    #: HOW this row was produced: without it the row cannot be re-scored, and a
    #: digest over a request you cannot reconstruct proves nothing. The nightly
    #: selfcheck reads it straight back out.
    quote_max_age_sessions: int | None = None
    #: The strike the CALLER asked for, as distinct from `strike`, which is what
    #: resolved against the chain. They differ exactly when resolution failed —
    #: and that is the case the record has to survive: without it a failed
    #: strike-ladder row is indistinguishable from an ATM row, and re-scoring it
    #: silently produces a different trade.
    requested_strike: float | None = None
    spot: float | None = None
    dte_entry: int | None = None
    fill_alpha: float = 0.5

    extrapolated: bool = False
    flags: list[str] = field(default_factory=list)
    model_versions: dict = field(default_factory=dict)
    payoff: dict = field(default_factory=dict)
    analog_buckets: dict = field(default_factory=dict)
    snapshot_hash: str = ""
    detail: str = ""
    #: What the option market is quoting for this print AS OF THE DECISION
    #: DATE — today, for a live board. This is the number a reader compares the
    #: model call against, because they are deciding now and may enter a day
    #: early or late; a quote pinned to the day the strategy happens to execute
    #: would make their read of the trade an artefact of the data assembly.
    #:
    #: Same for every strategy on a print, because it is a property of the
    #: event and the date, not of the structure.
    implied_move: float | None = None

    #: The quote at THIS TRADE'S entry date — what the model actually consumed
    #: (``live_features(..., as_of=result.entry_date)``). Per-TRADE, so two
    #: structures on one print differ: STR-RUNUP enters weeks before STR-THRU
    #: and implied move rises into a print, which is the STR-RUNUP thesis. Kept
    #: because it is the honest provenance of the model's input, and shown in
    #: the detail rather than on the board.
    implied_move_at_entry: float | None = None

    driver_name: str | None = None
    driver_prediction: float | None = None
    #: Predicted absolute stock movement over the actual STR-RUNUP holding
    #: window. The registered model predicts fourteen sessions; other entry
    #: timings scale that distribution linearly by days_before_print / 14.
    runup_move_prediction: float | None = None
    runup_move_p10: float | None = None
    runup_move_p90: float | None = None
    runup_move_days: float | None = None
    runup_move_scale: float | None = None
    #: The interval around ``driver_prediction`` itself — the "± 2" on an
    #: expected move of 6 — as opposed to ``model_p10``/``model_p90``, which are
    #: percentiles of the resulting TRADE RETURN. Both are real and they answer
    #: different questions; only the second was reachable from the board before.
    #:
    #: Nominally 10th/90th, so nominally an 80% band. Measured coverage is
    #: 72.8% (EXP-112, confirmed EXP-114): the pool these are drawn from is
    #: unconditional, so the band is one width for every event and too narrow
    #: overall. The board labels it with the measured number rather than the
    #: nominal one. EXP-114 adopted per-decile conditioning to fix this; that
    #: is a separate change, gated on a recalibration refit.
    driver_p10: float | None = None
    driver_p90: float | None = None
    #: For a row we could not price: the newest chain the store holds for this
    #: ticker, and its age in days. It turns a bare NO_CHAIN into the actionable
    #: fact — whether a refresh would fix it, or the name was never covered.
    chain_last_obs: str | None = None
    chain_age_days: int | None = None
    #: The exact feature values the champion consumed, in its own feature order,
    #: with the date each was observed as of. A prediction nobody can take apart
    #: is not evidence, and this is what makes the number on the board
    #: reconstructable by hand.
    model_inputs: dict = field(default_factory=dict)
    model_input_as_of: str | None = None

    def flag(self, name: str) -> None:
        if name not in self.flags:
            self.flags.append(name)

    @property
    def scored(self) -> bool:
        return self.exp_pnl_model is not None or self.exp_pnl_analog is not None

    def as_dict(self) -> dict:
        out = asdict(self)
        for key in (
            "as_of", "event_date", "entry_date", "exit_date", "expiry",
            "evidence_cutoff", "quote_date", "forecast_fold",
        ):
            value = out.get(key)
            out[key] = str(pd.Timestamp(value).date()) if value is not None else None
        out["fill"] = out.pop("fill_alpha", None)
        return out

    def digest(self) -> str:
        """Hash of the serialized result — the determinism test's comparison."""
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True, default=str).encode()
        ).hexdigest()


# --------------------------------------------------------------------------
# the scorer
# --------------------------------------------------------------------------


#: Distinguishes "not looked up yet" from "looked up, and the answer is None".
_UNSET = object()


class Scorer:
    """Holds the loaded models, trades, and data context for a scoring run.

    Constructing one loads the panel, the engine-replayed trades, the champions
    and the calendar. That is a few seconds; doing it per call would make the
    guide's five-minute budget for a three-week calendar unreachable, so the
    dashboard builds one and reuses it.
    """

    def __init__(
        self,
        *,
        registry: Registry | None = None,
        trades: pd.DataFrame | None = None,
        context: FeatureContext | None = None,
        snapshot: str | None = None,
        verify_artifacts: bool = True,
    ):
        # Lazily-built inputs for the expected-P&L gate, per INSTANCE so a
        # differently-configured environment cannot inherit them. _UNSET rather
        # than None because None is a real, cacheable answer here: "the pool
        # could not be built", which must not be retried on every row.
        self._pool = _UNSET
        self._pre_iv = _UNSET
        self._crush = _UNSET
        self._crush_frame = _UNSET
        self._latest_iv = _UNSET

        self.registry = registry if registry is not None else load_registry()
        self.snapshot = snapshot if snapshot is not None else _snapshot_hash()
        self.context = context or FeatureContext.load()
        self.calendar = self.context.calendar or trading_calendar()

        if trades is None:
            trades = store.read_table("trades")
        engine_rows = trades[trades["provenance"].astype(str) == "engine.replay"]
        self.trades = self._enrich(engine_rows)
        self.matcher = AnalogMatcher(self.trades, snapshot=self.snapshot)

        self._models: dict[tuple[str, str], object] = {}
        self._payoffs: dict[tuple[str, float, object], PayoffMap] = {}
        self._runup_payoffs: dict[
            tuple[float, object], RunupPayoffSurface | None
        ] = {}
        self._recalibrations: dict[tuple[str, float, object], object] = {}
        self._recal_pairs = None
        self._quotes: dict[tuple[str, object], float | None] = {}
        #: Tier-4 fold models, one fit per fold per process. A board spans at
        #: most two folds, so this bounds an otherwise per-row cost.
        self._serving_models: dict[pd.Timestamp, object] = {}
        self._verify = verify_artifacts
        #: live_features() results, keyed by (ticker, event_date, entry_date,
        #: session) — every score() call recomputes _live_values from scratch
        #: (its own docstring: "one call per score"), and score_calendar()
        #: calls score() once per (event, strategy, offset). Most strategies
        #: share the same decision date for one event (only STR-RUNUP enters
        #: early), so without this a board with N strategies pays the full
        #: live_features cost — panel history slice, regime/runup/orats
        #: feature blocks, pre-print vol — up to N times per event instead of
        #: once. Grew from a non-issue into the board's dominant cost as
        #: strategies were added; found while diagnosing a nightly rebuild
        #: that used to fit in the "five-minute budget" score_calendar's own
        #: docstring quotes and now runs much longer.
        self._live_features_cache: dict[tuple, dict[str, float]] = {}

    # -- setup -------------------------------------------------------------

    def _enrich(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Attach the panel columns the analog buckets and payoff fits need.

        The Tier-2 ``trades`` schema is deliberately narrow — it holds the trade,
        not the world around it. Market cap, the quoted implied move, and the
        realized move all live in the panel, and joining them here once is what
        lets the matcher bucket and the payoff calibrate.
        """
        if trades.empty:
            return trades
        out = trades.copy()
        out["event_date"] = pd.to_datetime(out["event_date"])
        panel = self.context.panel
        columns = [
            "ticker", "date", "mcap_usd", "or_implied", "mean_prior_or_implied",
            "abs_move", "n_prior",
        ]
        available = [c for c in columns if c in panel.columns]
        out = out.merge(
            panel[available].rename(columns={"date": "event_date"}),
            on=["ticker", "event_date"],
            how="left",
        )
        # Reconstructed from the stored legs rather than recomputed, so the
        # analog buckets describe the trade that was actually priced.
        out["spot_entry"], out["dte_entry"] = legs_spot_dte(out)
        out["spot_exit"] = legs_exit_spot(out)
        out["im_t1"] = out["or_implied"]

        # The implied-move bucket has to be measured at the *entry* date on both
        # sides of the match, or STR-RUNUP is mis-bucketed: its entry sits
        # fourteen trading days before the print, and the implied move at that
        # point is a materially different number from the one quoted at the
        # close before the event. Using the panel's event-level `or_implied` for
        # the trades and an entry-date reading for the request would compare two
        # different quantities and call them the same bucket.
        out["entry_date"] = pd.to_datetime(out["entry_date"])
        state = daily_state_frame(
            out[["ticker", "entry_date"]].copy(),
            daily=self.context.daily,
            as_of_column="entry_date",
        )
        out["implied_at_entry"] = state["im"].to_numpy()
        return bucket_frame(out)

    def model(self, role: str, strategy: str = "*"):
        key = (strategy, role)
        if key not in self._models:
            try:
                entry, artifact = self.registry.load_champion(
                    role, strategy, verify=self._verify
                )
                self._models[key] = (entry, artifact)
            except RegistryError:
                self._models[key] = None
        return self._models[key]

    def payoff(self, strategy: str, alpha: float, before) -> PayoffMap | None:
        """Payoff map fitted only on trades closed before ``before``."""
        stamp = pd.Timestamp(before).normalize() if before is not None else None
        key = (strategy, round(float(alpha), 4), stamp)
        if key not in self._payoffs:
            try:
                self._payoffs[key] = fit_payoff(
                    self.trades, strategy, alpha=alpha, before=stamp
                )
            except (PayoffError, KeyError):
                self._payoffs[key] = None
        return self._payoffs[key]

    def runup_payoff(self, alpha: float, before) -> RunupPayoffSurface | None:
        """Two-driver surface fitted only on STR-RUNUP trades closed earlier."""
        stamp = pd.Timestamp(before).normalize() if before is not None else None
        key = (round(float(alpha), 4), stamp)
        if key not in self._runup_payoffs:
            try:
                self._runup_payoffs[key] = fit_runup_payoff(
                    self.trades,
                    alpha=alpha,
                    before=stamp,
                )
            except (PayoffError, KeyError):
                self._runup_payoffs[key] = None
        return self._runup_payoffs[key]

    def recalibration(self, strategy: str, alpha: float, before):
        """Monotone win-rate recalibration fitted on pairs closed before ``before``.

        Returns ``None`` (and the raw probability ships unchanged) when no pairs
        table exists yet or too few pairs had closed by the cutoff. The pairs are
        loaded once per Scorer.
        """
        from engine import recalibrate

        stamp = pd.Timestamp(before).normalize() if before is not None else None
        key = (strategy, round(float(alpha), 4), stamp)
        if key not in self._recalibrations:
            if self._recal_pairs is None:
                self._recal_pairs = recalibrate.load_pairs()
            self._recalibrations[key] = (
                recalibrate.fit_recalibration(
                    strategy, alpha, before=stamp, pairs=self._recal_pairs
                )
                if not self._recal_pairs.empty
                else None
            )
        return self._recalibrations[key]

    # -- the call ----------------------------------------------------------

    def score(self, request: ScoreRequest, *, chain_index: ChainIndex | None = None) -> ScoreResult:
        result = ScoreResult(
            ticker=request.ticker,
            strategy=request.strategy,
            as_of=pd.Timestamp(request.as_of).normalize() if request.as_of is not None else None,
            fill_alpha=request.fill.alpha,
            snapshot_hash=self.snapshot,
        )

        beaten = superseded_by(request.strategy)
        if beaten is not None:
            # Live on the board as a row that names its successor, not as a
            # silent omission: an operator who cannot see that TWIN-P stopped
            # trading, and what replaced it, will assume a data outage.
            if request.event_date is not None:
                result.event_date = pd.Timestamp(request.event_date).normalize()
                result.session = request.session
            result.flag("SUPERSEDED")
            result.detail = (
                f"{request.strategy} is superseded by {beaten.strategy} "
                f"({beaten.evidence})"
                + (f" — {beaten.notes}" if beaten.notes else "")
            )
            return result
        if request.strategy in DISABLED_STRATEGIES:
            # Carry the event identity even though nothing is scored. The row
            # still appears on the board, and one with a blank date reads as a
            # rendering bug rather than as the deliberate refusal it is.
            if request.event_date is not None:
                result.event_date = pd.Timestamp(request.event_date).normalize()
                result.session = request.session
            result.flag("UNVALIDATED_STRUCTURE")
            result.detail = DISABLED_STRATEGIES[request.strategy]
            return result
        if request.strategy not in STRUCTURES:
            raise KeyError(f"unknown strategy {request.strategy!r}")

        event_date, session = self._resolve_event(request)
        result.event_date, result.session = event_date, session
        result.requested_strike = (
            float(request.strike) if request.strike is not None else None
        )
        structure = self._structure(request)

        window = self.calendar.resolve_offsets(
            event_date, session, structure.entry_offset, structure.exit_offset,
            decision_offset=structure.decision_offset,
        )
        result.entry_date, result.exit_date = window.entry_date, window.exit_date
        # The decision date, not the entry date. They are the same close for
        # every structure that has not set a `decision_offset`, so nothing moves
        # yet; what changes is that the default is now the date the score is
        # *taken*, which is the date the rest of the pipeline already audits
        # against.
        if result.as_of is None:
            result.as_of = window.decision_date
        assert_decision_causal(result.as_of, event_date, session, calendar=self.calendar)

        # The cutoff for *evidence* — which analogs are eligible, which trades
        # the payoff map may be fitted on — is the earlier of the declared
        # decision date and the date the position is actually opened, never the
        # later. They differ for STR-RUNUP, whose entry is fourteen trading days
        # before the last information-free close: a caller passing `as_of` at
        # that close for a position opened a fortnight earlier would otherwise
        # admit two weeks of trades that had not yet closed when the decision
        # was taken.
        result.evidence_cutoff = min(
            d for d in (result.as_of, result.entry_date) if d is not None
        )
        if self.calendar.is_projected(window.exit_date):
            result.flag("PROJECTED_CALENDAR")

        # -- the shape, for a structure whose shape comes from a forecast ---
        # Before pricing, which is the whole point: the strikes cannot be
        # chosen until the width is known, and the width comes from a model
        # whose features are entirely pricing-free. See engine/forecast_sizing.
        if request.strategy in FORECAST_SIZED and not request.structure_params:
            request, structure = self._size_from_forecast(request, result, structure)
            if structure is None:
                return result

        # -- the live chain: entry cost, strike, and the moneyness label ----
        self._price_entry(request, structure, result, chain_index)

        # -- features ------------------------------------------------------
        features = self._features(request, result)

        # Carried before the layers run: every layer reads this frame, and two
        # of them already used `im` internally without ever surfacing it.
        # TWO quotes, and they answer different questions.
        #
        # `implied_move_at_entry` is what the model consumed: the quote at this
        # trade's own entry date. It is per-TRADE, so STR-RUNUP and STR-THRU on
        # the same print legitimately differ — one enters weeks before the other
        # and implied move rises into a print.
        #
        # `implied_move` is what the market is quoting NOW, at the decision
        # date. That is the one a reader needs, because they are deciding today
        # and may pull the trigger a day early or late: showing them a quote
        # from the day the strategy happens to execute makes their read of the
        # trade an artefact of how the data is assembled.
        if "im" in features.columns:
            at_entry = features["im"].iloc[0]
            if pd.notna(at_entry) and float(at_entry) >= MIN_QUOTED_IMPLIED_MOVE:
                result.implied_move_at_entry = float(at_entry)
        today = self._quote_today(request.ticker, result.as_of)
        if today is not None:
            result.implied_move = today

        # -- layers --------------------------------------------------------
        # EXP-117: a quote that fails the cost-of-spot sanity ceiling gets no
        # model, analog, or gate numbers — the entry cost shown is the
        # diagnosis, and a confident P&L built on a junk quote is worse than
        # none. The row stays visible with the flag.
        if "BAD_QUOTE" in result.flags:
            result.detail = (
                f"entry cost {result.entry_cost:.2f} is "
                f"{result.entry_cost / result.spot * 100.0:.0f}% of spot "
                f"{result.spot:.2f} — above the {BAD_QUOTE_COST_PCT:.0f}% "
                "bad-quote ceiling; not scored"
            )
            return result
        self._score_model(request, result, features)
        self._score_analogs(request, result, features)
        self._score_gate(request, result, features)
        self._compare_layers(result)
        return result

    # -- pieces ------------------------------------------------------------

    def _resolve_event(self, request: ScoreRequest) -> tuple[pd.Timestamp, str]:
        if request.event_date is not None:
            event_date = pd.Timestamp(request.event_date).normalize()
        else:
            event_date = self._next_event(request.ticker, request.as_of)
        session = request.session or session_for(request.ticker, event_date)
        return event_date, session

    def _next_event(self, ticker: str, as_of) -> pd.Timestamp:
        events = store.read_table(
            "earnings_events", columns=["ticker", "event_date", "session"]
        )
        rows = events[events["ticker"] == ticker]
        if as_of is not None:
            rows = rows[rows["event_date"] >= pd.Timestamp(as_of).normalize()]
        if rows.empty:
            raise KeyError(f"no upcoming earnings event for {ticker}")
        return pd.Timestamp(rows.sort_values("event_date")["event_date"].iloc[0]).normalize()

    def _structure(self, request: ScoreRequest) -> Structure:
        """The strategy's structure, with any caller-named strike or expiry bound.

        The guide's signature is ``score(ticker, strategy, strike, expiry, as_of)``,
        so those arguments have to reach the legs. A strike that only reached the
        moneyness label would price an ATM structure and report it as though it
        were the requested one — the numbers would be real and about a different
        trade.

        Only *independent* selectors are overridden — those that name no other
        leg. A leg declared ``same_as`` another follows it by design: overriding
        both would let a straddle's call and put drift onto different strikes
        and stop being a straddle.

        The test is ``leg.strike.refs``, not ``kind != "same_as"``. That
        distinction was invisible while ``same_as`` was the only cross-leg
        selector and became a real defect the moment it was not: CND-P and
        TWIN-P resolve their strikes through ``offset_from``, ``mirror`` and
        ``grid_step``, and naming a strike used to replace EVERY leg with a
        fixed one. A laddered TWIN-P collapsed all seven legs onto a single
        strike — four long and four short at the same price, a position worth
        exactly zero at every settlement, costing four spreads. It priced, it
        scored, and it was labelled TWIN-P.

        Latent rather than live when it was found: ``strike_ladder`` only
        ladders rows the gate passed, TWIN-P passes almost none, CND-P is
        disabled, and ladder rows never enter the ledger. The ATM board was
        never affected. Fixed here because "almost none" is not none.
        """
        try:
            structure = STRUCTURES[request.strategy](**dict(request.structure_params or {}))
        except TypeError as exc:
            raise ValueError(
                f"{request.strategy}: structure_params {dict(request.structure_params or {})} "
                f"not accepted by its factory — {exc}"
            ) from exc
        if request.decision_offset is not None:
            structure = Structure(
                name=structure.name, legs=structure.legs,
                entry_offset=structure.entry_offset,
                exit_offset=structure.exit_offset,
                decision_offset=int(request.decision_offset),
                description=structure.description,
                params=dict(structure.params),
            )
        if request.strike is None and request.expiry is None:
            return structure

        legs = []
        for leg in structure.legs:
            strike_sel = leg.strike
            if request.strike is not None and not leg.strike.refs:
                strike_sel = StrikeSelector(kind="fixed", strike=float(request.strike))
            expiry_sel = leg.expiry
            if request.expiry is not None:
                expiry_sel = ExpirySelector(
                    kind="fixed", expiry=pd.Timestamp(request.expiry)
                )
            legs.append(
                LegSpec(
                    name=leg.name, right=leg.right, side=leg.side,
                    expiry=expiry_sel, strike=strike_sel, qty=leg.qty,
                )
            )
        return Structure(
            name=structure.name,
            legs=tuple(legs),
            entry_offset=structure.entry_offset,
            exit_offset=structure.exit_offset,
            decision_offset=structure.decision_offset,
            description=structure.description,
            params=dict(structure.params) | {
                "requested_strike": request.strike,
                "requested_expiry": (
                    str(pd.Timestamp(request.expiry).date())
                    if request.expiry is not None
                    else None
                ),
            },
        )

    def _price_entry(self, request, structure, result, chain_index) -> None:
        """Resolve and price the structure on the chain the DECISION can see.

        For every structure decided at its entry close that is the entry chain,
        unchanged. For one decided early it is the decision-date chain — which
        is the only chain that exists when the call has to be made. The entry
        chain for a session is not published until that session is over, so
        pricing against it is what makes today's prediction unactionable.

        `quote_date` records which it was. Everything filled from here —
        `entry_cost`, `strike`, `expiry`, `dte_entry` — is then an ESTIMATE for
        the entry rather than the fill, and must be labelled that way wherever
        it is shown.
        """
        from engine.jsonio import json_safe
        result.structure_spec = json_safe(asdict(structure))
        result.structure_params = dict(request.structure_params or {})
        result.variant = request.variant
        result.quote_max_age_sessions = request.quote_max_age_sessions
        result.quote_date = result.quote_date or result.entry_date
        if structure.decided_early:
            window = self.calendar.resolve_offsets(
                result.event_date, result.session,
                structure.entry_offset, structure.exit_offset,
                decision_offset=structure.decision_offset,
            )
            result.quote_date = window.decision_date
        index = chain_index
        if index is None:
            index = load_chain_index(
                [(request.ticker, result.quote_date), (request.ticker, result.exit_date)],
                progress_every=0,
            )
        rows = index.get(request.ticker, result.quote_date)
        if (rows is None or rows.empty) and request.quote_max_age_sessions is not None:
            # An UPCOMING entry has no chain and never will until that session
            # closes, which is what leaves a forward board unpriced. The newest
            # chain we hold is strictly OLDER information, so it cannot leak.
            #
            # Substituting one was previously rejected outright, and that was
            # right at the time: the cache was event-centric, so a name's newest
            # chain came from its LAST print — a 93-day median, a different
            # quarter's surface, expiries that do not span the coming event. The
            # nightly now pulls a fresh chain for every board name, so the
            # substitute is usually one session old. The AGE BOUND is what makes
            # the difference, and it is not optional: past it, we go back to
            # refusing rather than quoting fiction.
            fallback, age = self._fresh_quote_date(
                request, result, int(request.quote_max_age_sessions)
            )
            if fallback is not None:
                candidate = index.get(request.ticker, fallback)
                if (candidate is None or candidate.empty) and chain_index is None:
                    # We own this index, and it was built before the fallback
                    # date was known — so the key it needs is simply not in it.
                    # A caller-supplied index (the board) pre-loads these; a
                    # one-off `score()` has to fetch on demand, and without this
                    # the same row prices on the board and returns NO_CHAIN when
                    # the nightly selfcheck re-scores it.
                    extra = load_chain_index(
                        [(request.ticker, fallback)], progress_every=0
                    )
                    candidate = extra.get(request.ticker, fallback)
                if candidate is not None and not candidate.empty:
                    rows = candidate
                    result.quote_date = fallback
                    result.quote_age_sessions = age
                    result.flag("STALE_QUOTE")
        if rows is None or rows.empty:
            result.flag("NO_CHAIN")
            self._note_chain_age(request, result)
            return

        from engine.replay import _clean
        from engine.structures import ChainSnapshot, price_structure

        clean = _clean(rows)
        if clean.empty:
            result.flag("NO_CHAIN")
            self._note_chain_age(request, result)
            return
        snapshot = ChainSnapshot(
            ticker=request.ticker,
            obs_date=result.quote_date,
            event_date=result.event_date,
            rows=clean,
            session=result.session,
        )
        try:
            priced = price_structure(structure, snapshot, request.fill)
        except LadderTooCoarse as exc:
            # The chain is fine; this ticker's strikes are too far apart to
            # carry the shape. The message names the legs that collided, and it
            # is the only thing on the row that says so — so it is written to
            # `detail` here rather than left to a caller.
            result.flag("COARSE_LADDER")
            result.detail = str(exc)
            self._note_chain_age(request, result)
            return
        except (StructureError, ValueError):
            result.flag("NO_CHAIN")
            self._note_chain_age(request, result)
            return

        result.entry_cost = float(priced.cost)
        result.spot = float(priced.spot)
        # Kept for the expected-P&L gate, which has to reprice these exact legs
        # at simulated outcomes. Stashed on the instance rather than added to
        # the dataclass: it is working state for one scoring pass, and putting
        # ResolvedLeg objects in a serialised result would put chain quotes into
        # the ledger and the dashboard payload.
        result._priced_legs = tuple(priced.legs)
        result.legs = [
            {
                "name": leg.name,
                "side": leg.side,
                "right": leg.right,
                "qty": float(leg.qty),
                "strike": float(leg.strike),
                "expiry": str(pd.Timestamp(leg.expiry).date()),
                "dte": int(leg.dte),
                "bid": None if leg.bid is None else float(leg.bid),
                "ask": None if leg.ask is None else float(leg.ask),
                "price": None if leg.price is None else float(leg.price),
                "cash_flow": None if leg.cash_flow is None else float(leg.cash_flow),
                "wide_market": bool(leg.wide_market),
            }
            for leg in priced.legs
        ]
        result.strike = float(priced.legs[0].strike)
        result.expiry = priced.legs[0].expiry
        result.dte_entry = int(priced.legs[0].dte)
        result.rel_spread = _mean_relative_spread(priced)
        result.structure_width = _structure_width(structure, priced)
        result.structure_peak = _structure_peak(structure, result.structure_width)
        if priced.any_wide_market:
            result.flag("WIDE_MARKET")
        # EXP-117: a straddle costing more than BAD_QUOTE_COST_PCT of spot is
        # not a tradeable quote. WIDE_MARKET sees the spread; this sees the
        # level. The row stays on the board with its cost and the flag, but
        # the scoring layers refuse to produce confident numbers from it.
        if result.spot and result.entry_cost is not None:
            cost_pct = result.entry_cost / result.spot * 100.0
            if cost_pct > BAD_QUOTE_COST_PCT:
                result.flag("BAD_QUOTE")
        if bool(clean.get("quote_repaired", pd.Series(dtype=bool)).any()):
            result.flag("QUOTE_REPAIRED")

        moneyness = abs(result.strike / result.spot - 1.0) * 100.0
        if request.strike is not None:
            moneyness = abs(float(request.strike) / result.spot - 1.0) * 100.0
        result.extrapolated = moneyness > ATM_TOLERANCE_PCT
        if result.extrapolated:
            result.flag("EXTRAPOLATED")

    def _fresh_quote_date(self, request, result, max_age: int):
        """Newest chain for this name within ``max_age`` SESSIONS of the entry.

        Sessions, not calendar days: a Monday entry quoted on the preceding
        Friday is one session stale, and counting it as three would refuse a
        board every weekend.
        """
        from engine.replay import latest_chain_date

        anchor = result.quote_date or result.entry_date
        if anchor is None:
            return None, None
        try:
            newest = latest_chain_date(request.ticker, anchor)
        except Exception:  # a board must not die on a diagnostic
            return None, None
        if newest is None:
            return None, None
        newest = pd.Timestamp(newest).normalize()
        if request.chain_as_of is not None and newest > pd.Timestamp(
            request.chain_as_of
        ).normalize():
            # Dated after the night being scored: information the board did not
            # have. Refuse it here so the caller's pre-loaded index and an
            # on-demand re-score reach the same verdict.
            return None, None
        try:
            age = self.calendar.index_of(anchor, side="prev") - self.calendar.index_of(
                newest, side="prev"
            )
        except (KeyError, ValueError):
            return None, None
        if age < 0 or age > max_age:
            return None, None
        return newest, int(age)

    def _note_chain_age(self, request, result) -> None:
        """Record how old the newest chain for this ticker is. Never prices off it.

        Substituting an older chain was considered and rejected: the cache is
        event-centric, so a name's newest chain is from its LAST print — measured
        at a 93-day median across a live board. That is a different quarter's
        surface, a different spot, and expiries that do not span the upcoming
        print. An entry cost derived from it would look like a quote and be
        fiction, and it would then be frozen into the prediction ledger. The age
        is reported instead, because it is what tells a reader whether a refresh
        fixes this row or the name was never covered at all.
        """
        from engine.replay import latest_chain_date

        try:
            newest = latest_chain_date(request.ticker, result.entry_date)
        except Exception:  # a diagnostic must never take the score down
            return
        if newest is None:
            result.detail = result.detail or "no chain in the store for this ticker"
            return
        result.chain_last_obs = str(pd.Timestamp(newest).date())
        if result.entry_date is not None:
            result.chain_age_days = int(
                (pd.Timestamp(result.entry_date).normalize() - pd.Timestamp(newest)).days
            )

    def _panel_row(self, request, result):
        """The panel's row for this event, or None when it has not happened."""
        panel = self.context.panel
        rows = panel[
            (panel["ticker"] == request.ticker) & (panel["date"] == result.event_date)
        ]
        return rows.iloc[0] if len(rows) else None

    def _live_values(self, request, result) -> dict[str, float]:
        """``live_features`` for an event the panel does not carry.

        Shared by the market block and the event-history block below — they
        are two slices of the same vector, and computing it twice would
        double the cost of every forward row on the board. Also cached ACROSS
        different ``score()`` calls, keyed on (ticker, event_date, entry_date,
        session): a board scores every strategy for one event, most of which
        share the same decision date (only STR-RUNUP enters early), so without
        this the same expensive recomputation ran once per strategy instead of
        once per event.
        """
        key = (
            request.ticker,
            pd.Timestamp(result.event_date).normalize() if result.event_date is not None else None,
            pd.Timestamp(result.entry_date).normalize() if result.entry_date is not None else None,
            result.session,
        )
        if key in self._live_features_cache:
            return self._live_features_cache[key]
        try:
            vector = live_features(
                request.ticker,
                result.event_date,
                as_of=result.entry_date,
                session=result.session,
                context=self.context,
            )
            values = dict(vector.values)
        except (KeyError, ValueError, FileNotFoundError):
            # No prior events, or no price history to build the run-up block
            # from. The model layer will report MISSING_FEATURES, which is the
            # honest answer for a name we know nothing about. Cached too — a
            # ticker with no history stays without it for every strategy that
            # asks, not just the first.
            values = {}
        self._live_features_cache[key] = values
        return values

    def _market_block(self, request, result) -> dict[str, float]:
        """Market state at the last pre-print close, from whichever path has it.

        A historical event has a panel row and the block is read straight off it —
        the same numbers the models were trained on, no recomputation.

        **An upcoming event has no panel row**, because the panel is built from
        realized events. Without the live path the model layer would be dark for
        exactly the prints the dashboard exists to score, which would make the
        whole phase backtest-only. ``features.live_features`` extends the panel's
        own recursions one event forward to fill the gap, and
        ``checks/phase1_replay.py`` asserts the two paths agree to 1e-9 on
        historical events, so the fallback is not a second implementation with a
        second set of answers.
        """
        row = self._panel_row(request, result)
        if row is not None:
            return {
                column: (float(row[column]) if pd.notna(row[column]) else np.nan)
                for column in _PANEL_MARKET_BLOCK
                if column in row.index
            }
        live = self._live_values(request, result)
        return {c: live[c] for c in _PANEL_MARKET_BLOCK if c in live}

    def _features(self, request, result) -> pd.DataFrame:
        """The one leak-audited feature row every layer reads."""
        frame = pd.DataFrame(
            [
                {
                    "ticker": request.ticker,
                    "event_date": result.event_date,
                    "entry_date": result.entry_date,
                    "entry_cost": result.entry_cost,
                    "spot_entry": result.spot,
                    "dte_entry": result.dte_entry,
                }
            ]
        )
        built = entry_feature_frame(
            frame, panel=self.context.panel, daily=self.context.daily,
            as_of_column="entry_date",
        )
        built["entry_cost_pct"] = (
            pd.to_numeric(built["entry_cost"], errors="coerce")
            / pd.to_numeric(built["spot_entry"], errors="coerce") * 100.0
        )
        built["dte_entry"] = pd.to_numeric(built["dte_entry"], errors="coerce")
        # `days_before_print` counts TRADING days from the entry to the last
        # pre-print close, which is how `implied_t1.build_dataset` defines it.
        # Calendar days would be the same feature name carrying a different
        # quantity — a training/serving skew that produces no error and quietly
        # shifts every run-up prediction.
        built["days_before_print"] = _trading_days_before(
            self.calendar, result.entry_date, result.event_date, result.session
        )

        # The panel's market-state block — `or_implied`, `dist_high`,
        # `spy_vol20`, the market cap — is read at the last pre-print close. The
        # size model needs it, and for a structure that *enters* at that close
        # (STR-THRU) it is exactly contemporaneous and entirely legitimate.
        #
        # For one that enters fourteen trading days earlier (STR-RUNUP) the same
        # values would be a fortnight of hindsight, so they are withheld rather
        # than passed and audited: a model that needs them then reports
        # MISSING_FEATURES and declines to score, which is the correct outcome —
        # it is the wrong model for that decision date. STR-RUNUP is driven by
        # `implied_t1`, whose features are as-of the entry by construction.
        window = self.calendar.resolve_offsets(
            result.event_date, result.session, 0, 1
        )
        if result.entry_date is not None and result.entry_date >= window.last_pre_print:
            block = self._market_block(request, result)
            for column, value in block.items():
                built[column] = value

        # `entry_feature_frame` reads the event-history recursions — n_prior,
        # the prior-move means, the EMAs — off the panel, and an UPCOMING event
        # has no panel row, so every one of them lands NaN and every model
        # declines with MISSING_FEATURES. That is the whole forward board.
        #
        # These are not stale substitutes: they are recursions over the ticker's
        # PRIOR events, which is exactly what the panel would hold, computed one
        # event forward by `live_features` and cut off at the entry date.
        # `checks/phase1_replay.py` asserts the two paths agree to 1e-9 on
        # historical events, so this is not a second set of answers.
        if self._panel_row(request, result) is None:
            live = self._live_values(request, result)
            for column in EVENT_HISTORY_FEATURES:
                if column not in live:
                    continue
                if column not in built.columns or pd.isna(built[column].iloc[0]):
                    built[column] = live[column]

        # Derive the absolute-valued inputs on the ASSEMBLED frame, after the
        # market block has landed. `load_panel` applies the same derivation on
        # read and `live_features` applies it to its own frame, but the scorer
        # builds a third frame from `entry_feature_frame` plus that block — and
        # a feature the model lists but no path supplies is a silent blackout:
        # promoting size_v1_4 without this made every row MISSING_FEATURES.
        built = add_absolute_features(built)
        built = add_quote_indicators(built)

        n_prior = built.get("n_prior")
        if n_prior is not None and pd.notna(n_prior.iloc[0]) and n_prior.iloc[0] < THIN_HISTORY_EVENTS:
            result.flag("THIN_HISTORY")

        # The audit that the guide requires on EVERY call, not just in tests.
        stamped = {
            c: result.entry_date
            for c in built.columns
            if c in set(EVENT_HISTORY_FEATURES) | set(DAILY_STATE_COLUMNS)
        }
        vector = FeatureVector(
            ticker=request.ticker,
            as_of=result.entry_date,
            values={
                c: (float(built[c].iloc[0]) if pd.notna(built[c].iloc[0]) else float("nan"))
                for c in stamped
            },
            feature_as_of=stamped,
            event_date=result.event_date,
            session=result.session,
        )
        assert_causal(vector)
        return built

    def _score_model(self, request, result, features) -> None:
        strategy = request.strategy
        if strategy == "STR-RUNUP":
            self._score_runup_model(request, result, features)
            return
        loaded_size = self.model("size")
        loaded_implied = self.model("implied_t1")

        from engine.payoff import PAYOFF_DRIVER

        driver = PAYOFF_DRIVER.get(strategy)
        if driver is None:
            result.flag("NO_PAYOFF_MAP")
            return
        loaded = loaded_size if driver == "abs_move" else loaded_implied
        if loaded is None:
            result.flag("NO_MODEL")
            return
        entry, artifact = loaded
        result.model_versions[f"{driver}"] = entry.id

        # Record the inputs BEFORE any early return: a row that declined to
        # score is exactly the one where someone needs to see what went in.
        result.model_input_as_of = (
            str(pd.Timestamp(result.entry_date).date())
            if result.entry_date is not None
            else None
        )
        result.model_inputs = {
            name: (
                float(features[name].iloc[0])
                if name in features.columns and pd.notna(features[name].iloc[0])
                else None
            )
            for name in artifact.features
        }

        missing = [f for f in artifact.features if f not in features.columns]
        if missing:
            result.flag("MISSING_FEATURES")
            result.detail = f"model {entry.id} needs {missing}"
            return
        X = features[list(artifact.features)].to_numpy(dtype=float)
        if not np.isfinite(X).all():
            absent = [
                f for f, ok in zip(artifact.features, np.isfinite(X[0])) if not ok
            ]
            result.flag("MISSING_FEATURES")
            result.detail = f"model {entry.id}: non-finite {absent}"
            return

        # The champion's prediction needs no chain, so it is taken FIRST and
        # kept even when the row cannot be priced. It is the program's actual
        # signal — predicted |move| against the market's quoted implied
        # (EXP-040), predicted T-1 implied move (EXP-043) — and a monitoring
        # board that hides it until an option quote arrives is withholding the
        # one number it already knows.
        point = float(artifact.predict(X)[0])
        result.driver_name = driver
        result.driver_prediction = point

        # The band on that prediction, for the same reason and on the same
        # terms: it is the model's own held-out residuals, so it needs no chain
        # either. Computing it down in the P&L branch would have withheld it
        # from exactly the rows the driver column exists to serve — a board more
        # than a day or two ahead of its prints has a prediction on nearly every
        # row and an entry cost on almost none.
        #
        # The draw ORDER is deliberate: this rng and these draws then feed the
        # P&L simulation below unchanged, so exp_pnl_model, win_model and the
        # return percentiles are bit-identical to before this band existed.
        rng = np.random.default_rng(
            int.from_bytes(
                hashlib.sha256(f"{self.snapshot}|{request.key()}".encode()).digest()[:8],
                "big",
            )
        )
        # `prediction=` is inert unless the artifact carries buckets, so this is
        # bit-identical for every champion saved before EXP-115.
        draws = point + artifact.residual_draws(MODEL_DRAWS, rng, prediction=point)
        result.driver_p10 = float(np.quantile(draws, 0.10))
        result.driver_p90 = float(np.quantile(draws, 0.90))

        # Expected PnL, though, is a return ON THE PREMIUM. Without an entry
        # cost there is no denominator — this is arithmetic, not a policy, and
        # it is why a current chain is the binding requirement for the P&L
        # columns rather than something a fallback could paper over.
        if result.entry_cost is None or result.entry_cost <= 0 or result.spot is None:
            # A row that was already refused keeps the reason it was refused
            # for. Without this guard the coarse-ladder rows pick NO_CHAIN up
            # here too — pricing failed, so of course there is no cost — and
            # reach the board carrying both flags, which is exactly the
            # mislabel COARSE_LADDER exists to remove.
            if "COARSE_LADDER" not in result.flags:
                result.flag("NO_CHAIN")
            return
        payoff = self.payoff(strategy, request.fill.alpha, result.evidence_cutoff)
        if payoff is None:
            result.flag("NO_PAYOFF_MAP")
            return
        result.payoff = payoff.as_dict()
        # Two independent uncertainties, both real: how wrong the prediction of
        # the driver may be, and how much the payoff line fails to explain even
        # given the driver. Folding in only the first would produce intervals
        # that are too narrow in exactly the reassuring direction. The first is
        # `draws`, already taken above with the driver band.

        # Both draws are empirical. An earlier version drew this one from a
        # Gaussian of the right standard deviation, which is wrong in a way that
        # shows up exactly where it matters: a long-vol payoff's residuals are
        # right-skewed — many small losses against a few large gains — so a
        # symmetric draw puts too much mass above the line and overstates
        # P(profit) on every event. It made the win-rate forecast lose to its own
        # base rate in the calibration check.
        noise = payoff.residual_draws(MODEL_DRAWS, rng)
        returns = simulate_returns(draws, payoff, result.spot, result.entry_cost, noise)

        result.exp_pnl_model = float(returns.mean())
        raw_win = float((returns > 0).mean())
        result.win_model_raw = raw_win
        recal = self.recalibration(strategy, request.fill.alpha, result.evidence_cutoff)
        result.win_model = (
            float(np.ravel(recal.transform(raw_win))[0]) if recal is not None else raw_win
        )
        result.model_p10 = float(np.quantile(returns, 0.10))
        result.model_p90 = float(np.quantile(returns, 0.90))

    def _score_runup_model(self, request, result, features) -> None:
        """Forecast STR-RUNUP PnL from implied move and pre-print spot movement."""
        loaded_implied = self.model("implied_t1")
        loaded_move = self.model("runup_move")
        if loaded_implied is None or loaded_move is None:
            result.flag("NO_MODEL")
            missing_roles = [
                role
                for role, loaded in (
                    ("implied_t1", loaded_implied),
                    ("runup_move", loaded_move),
                )
                if loaded is None
            ]
            result.detail = f"missing champion model roles {missing_roles}"
            return

        implied_entry, implied_artifact = loaded_implied
        move_entry, move_artifact = loaded_move
        result.model_versions["im_t1"] = implied_entry.id
        result.model_versions["runup_move"] = move_entry.id
        result.model_input_as_of = (
            str(pd.Timestamp(result.entry_date).date())
            if result.entry_date is not None
            else None
        )
        all_features = tuple(
            dict.fromkeys((*implied_artifact.features, *move_artifact.features))
        )
        result.model_inputs = {
            name: (
                float(features[name].iloc[0])
                if name in features.columns and pd.notna(features[name].iloc[0])
                else None
            )
            for name in all_features
        }
        missing = [name for name in all_features if name not in features.columns]
        if missing:
            result.flag("MISSING_FEATURES")
            result.detail = f"runup models need {missing}"
            return
        absent = [
            name
            for name in all_features
            if not np.isfinite(float(features[name].iloc[0]))
        ]
        if absent:
            result.flag("MISSING_FEATURES")
            result.detail = f"runup models have non-finite {absent}"
            return

        implied_x = features[list(implied_artifact.features)].to_numpy(float)
        move_x = features[list(move_artifact.features)].to_numpy(float)
        point_implied = float(implied_artifact.predict(implied_x)[0])
        point_move_d14 = float(move_artifact.predict(move_x)[0])
        days = float(features["days_before_print"].iloc[0])
        if not np.isfinite(days) or days < 0:
            result.flag("MISSING_FEATURES")
            result.detail = f"invalid days_before_print {days}"
            return
        scale = days / 14.0

        result.driver_name = "im_t1"
        result.driver_prediction = point_implied
        result.runup_move_days = days
        result.runup_move_scale = scale
        result.runup_move_prediction = float(
            scale_runup_move(max(point_move_d14, 0.0), days)
        )

        rng = np.random.default_rng(
            int.from_bytes(
                hashlib.sha256(
                    f"{self.snapshot}|{request.key()}".encode()
                ).digest()[:8],
                "big",
            )
        )
        implied_draws = point_implied + implied_artifact.residual_draws(
            MODEL_DRAWS,
            rng,
            prediction=point_implied,
        )
        implied_draws = np.maximum(implied_draws, 0.0)
        move_draws_d14 = point_move_d14 + move_artifact.residual_draws(
            MODEL_DRAWS,
            rng,
            prediction=point_move_d14,
        )
        move_draws = scale_runup_move(
            np.maximum(move_draws_d14, 0.0),
            days,
        )
        result.driver_p10 = float(np.quantile(implied_draws, 0.10))
        result.driver_p90 = float(np.quantile(implied_draws, 0.90))
        result.runup_move_p10 = float(np.quantile(move_draws, 0.10))
        result.runup_move_p90 = float(np.quantile(move_draws, 0.90))

        if (
            result.entry_cost is None
            or result.entry_cost <= 0
            or result.spot is None
            or result.strike is None
        ):
            if "COARSE_LADDER" not in result.flags:
                result.flag("NO_CHAIN")
            return
        payoff = self.runup_payoff(request.fill.alpha, result.evidence_cutoff)
        if payoff is None:
            result.flag("NO_PAYOFF_MAP")
            return
        result.payoff = payoff.as_dict()

        signed_moves = rng.choice((-1.0, 1.0), size=MODEL_DRAWS) * move_draws
        noise = payoff.residual_draws(MODEL_DRAWS, rng)
        returns = simulate_runup_returns(
            implied_draws,
            signed_moves,
            payoff,
            spot=result.spot,
            strike=result.strike,
            cost=result.entry_cost,
            payoff_noise=noise,
        )
        result.exp_pnl_model = float(returns.mean())
        raw_win = float((returns > 0).mean())
        result.win_model_raw = raw_win
        # Existing STR-RUNUP isotonic pairs were generated by the retired
        # one-driver simulator. Reusing that map would calibrate a probability
        # with observations from different forecast semantics. Keep the new
        # Monte Carlo win frequency raw until its own causal pairs accumulate.
        result.win_model = raw_win
        result.model_p10 = float(np.quantile(returns, 0.10))
        result.model_p90 = float(np.quantile(returns, 0.90))

    def _quote_today(self, ticker: str, as_of) -> float | None:
        """The implied move quoted at ``as_of``, regardless of when a trade enters.

        Read from the last ``daily_market`` row on or before the decision date,
        which is the same rule ``daily_state_frame`` applies everywhere else —
        on-or-before, because that close's own quotes are known to us at it.

        Cached per (ticker, date): every strategy on a print asks the same
        question, and this must return the same answer to all of them or the
        board is back to showing a number that depends on the structure.
        """
        if as_of is None:
            return None
        stamp = pd.Timestamp(as_of).normalize()
        key = (str(ticker), stamp)
        if key in self._quotes:
            return self._quotes[key]

        value = None
        try:
            state = daily_state_frame(
                pd.DataFrame({"ticker": [str(ticker)], "as_of": [stamp]}),
                daily=self.context.daily,
            )
            quoted = state["im"].iloc[0] if "im" in state.columns else None
            if quoted is not None and pd.notna(quoted) and float(quoted) >= MIN_QUOTED_IMPLIED_MOVE:
                value = float(quoted)
        except (KeyError, IndexError, ValueError):
            value = None
        self._quotes[key] = value
        return value

    def _score_analogs(self, request, result, features) -> None:
        implied = features.get("im")
        prior = features.get("mean_prior_or_implied")
        ratio = None
        if implied is not None and prior is not None:
            a, b = implied.iloc[0], prior.iloc[0]
            if pd.notna(a) and pd.notna(b) and b:
                ratio = float(a) / float(b)

        mcap = None
        if "mcap_log" in features.columns and pd.notna(features["mcap_log"].iloc[0]):
            mcap = float(np.exp(features["mcap_log"].iloc[0]))

        moneyness = None
        if result.strike is not None and result.spot:
            moneyness = abs(result.strike / result.spot - 1.0) * 100.0

        buckets = self.matcher.buckets_for(
            mcap_usd=mcap,
            dte=result.dte_entry,
            moneyness_pct=moneyness,
            implied_ratio=ratio,
        )
        analogs: AnalogSet = self.matcher.match(
            request.strategy,
            buckets,
            alpha=request.fill.alpha,
            as_of=result.evidence_cutoff,
            request_key=request.key(),
        )
        result.exp_pnl_analog = analogs.mean
        result.win_analog = analogs.win_rate
        result.ci_low, result.ci_high = analogs.ci_low, analogs.ci_high
        result.n_analogs = analogs.n
        result.analog_widened = analogs.widened
        result.analog_buckets = analogs.as_dict()
        if analogs.thin:
            result.flag("THIN_ANALOGS")
        if analogs.unavailable:
            # The match still HAPPENED, on the dimensions that had values — it
            # is simply coarser than the header implies, and `analog_widened`
            # now says so. Recorded in the detail rather than as its own flag:
            # a thin match and a coarse one are both "weak evidence", and the
            # number that distinguishes them is n_analogs.
            result.detail = (result.detail or "") + (
                f" analogs matched without {', '.join(analogs.unavailable)}"
            ).strip()

    def _gate_in_domain(self, request, features) -> bool:
        """Is this name inside the champion gate's training universe?

        A gate call on a name the champion never saw is an undetermined result,
        not a trade — so this exists to withhold rather than to decide. What it
        must NOT do is withhold on names the champion did see.

        **The computed-moves clause was removed 2026-09-06.** It refused any
        ticker with a file in `paths.COMPUTED_MOVES`, on the reasoning that
        those were the EXP-117 names the oquants panel does not carry and that
        no gate had trained on. Both halves stopped being true:

        * Moves are now computed for every ticker rather than only the ones
          oquants misses, so the directory no longer identifies that
          subpopulation. Of its 2,852 names, **2,814 (98.7%) are names oquants
          also carries** — ADBE and CASY among them. The clause had inverted
          into refusing the very universe the gates were trained on: 130 of the
          live board's 188 tickers, 266 of its 289 OUT_OF_DOMAIN rows, and 70%
          of every STR-THRU and STR-RUNUP row.
        * `engine/data/features/panel.py` builds the panel with
          `extra_moves_dirs=(paths.COMPUTED_MOVES,)`, so those names are IN the
          training data. Both champion gates were promoted 2026-09-05, the same
          day the computed moves landed. Even the 15 genuinely computed-only
          board names carry pre-2026 panel history — KEN 11 rows, CRESY 36,
          IBEX 7 — so "absent by construction" was false for them too.

        The market-cap floor below is a separate rule and still stands: EXP-118
        showed retraining on the expanded `<1B` universe does not clear the
        champions, so the champions stand only if they decline the sizes they
        were never validated on.
        """
        if "mcap_log" in features.columns:
            mcap_log = pd.to_numeric(features["mcap_log"], errors="coerce")
            if len(mcap_log) and np.isfinite(mcap_log.iloc[0]):
                if float(mcap_log.iloc[0]) < np.log(GATE_MCAP_FLOOR):
                    return False
        return True

    # -- forecast sizing (Tier 4) -----------------------------------------

    def _serving(self, fold, *, produces: str = "pred_abs_move"):
        """The Tier-4 fold model, fit or cached once per fold per process.

        Fitting is ~6 seconds and a three-week board spans at most two folds,
        so this is the difference between a board that costs seconds and one
        that refits per row.
        """
        key = (pd.Timestamp(fold), produces)
        if key not in self._serving_models:
            model = tier4.feature_model(produces) if produces != "pred_abs_move" else None
            self._serving_models[key] = tier4.serving_model(
                key[0], panel=self.context.panel, model=model
            )
        return self._serving_models[key]

    def _size_from_forecast(self, request, result, structure):
        """Set the structure's shape from the feature model's forecast.

        Returns ``(request, structure)``, with ``structure`` ``None`` when the
        event cannot be sized. Declining is the correct outcome and not a
        degraded one: pricing the factory's DEFAULT width instead would put a
        real number on the board for a trade nobody chose, and the row would
        claim to be forecast-sized while being nothing of the kind.

        The request is replaced rather than mutated so that everything
        downstream — the bootstrap seed, the recorded parameters, the structure
        rebuild — sees the same parameterisation.
        """
        try:
            served = self._serving(
                tier4.serving_fold(result.event_date, result.as_of)
            )
        except Exception as exc:  # a board must not die on one unfit fold
            result.flag("NO_FORECAST")
            result.detail = f"{request.strategy}: no fold model — {exc}"
            return request, None

        # The pricing columns are still empty here, and that is the point: the
        # size model reads none of them. `_features` runs again after pricing
        # for the layers that DO need them; it deduplicates its own flags, and
        # the second call is what the gate and analog layers see.
        features = self._features(request, result)
        missing = [f for f in served.features if f not in features.columns]
        if missing:
            result.flag("NO_FORECAST")
            result.detail = f"{request.strategy}: forecast needs {missing}"
            return request, None

        forecast = float(served.predict(features)[0])
        params = forecast_params(request.strategy, forecast)
        result.forecast_model = served.model_id
        result.forecast_fold = served.fold_start
        if pd.notna(forecast):
            result.forecast_abs_move = forecast
            p10, p90, sd, _ = served.interval([forecast])
            # NaN when Tier 4 has not been built or the fold's pool is too thin.
            # No band is the right answer there; a fabricated one is not.
            if np.isfinite(sd[0]):
                result.forecast_p10 = float(p10[0])
                result.forecast_p90 = float(p90[0])
                result.forecast_sd = float(sd[0])
        if params is None:
            result.flag("NO_FORECAST")
            result.detail = describe_sizing(request.strategy, None)
            return request, None

        result.structure_params = dict(params)
        sized = replace(request, structure_params=dict(params))
        return sized, self._structure(sized)

    # -- arithmetic entry rules -------------------------------------------

    def _apply_entry_rule(self, request, result, features) -> None:
        """Gate a strategy that has an arithmetic rule instead of a model.

        Reached only when the registry has no gate for the strategy, so a
        promoted model always wins: the rule is the floor, never an override.
        """
        rule = rule_for(request.strategy)
        if rule is None:
            return
        facts = {
            "cost": result.entry_cost,
            "w": result.structure_width,
            "peak": result.structure_peak,
            "rel_spread": result.rel_spread,
            "mcap_usd": _feature_value(features, "mcap_usd"),
        }
        # The simulated expectation, for the strategies whose rule asks for it.
        # Computed only when a term needs it: TWIN-P still gates on arithmetic
        # and must not pay for a simulation it does not read.
        if any(need in ("exp_pnl_sim", "pnl_cutoff")
               for term in rule.terms for need in term.needs):
            facts.update(self._simulated_pnl(request, result, features))
        verdict = rule.evaluate(facts)
        result.model_versions["gate"] = f"entry-rule:{rule.strategy}"
        result.gate_pass = verdict.passed
        if verdict.passed is None:
            result.flag("MISSING_FEATURES")
        result.detail = (
            f"{result.detail}; {verdict.detail}" if result.detail else verdict.detail
        )

    def _crush_table(self):
        """The realized-crush frame, read ONCE per scorer.

        ``iv_crush.crush_frame()`` reads every ``daily_market`` row — ~9M — to
        pair each event's pre- and post-print vol. Two callers needed it and
        each was calling it, so a nightly paid that read twice before this
        existed. Cached as an empty frame on failure, so a missing table costs
        one attempt rather than one per row.
        """
        if self._crush_frame is _UNSET:
            from engine.models.training import iv_crush

            try:
                self._crush_frame = iv_crush.crush_frame()
            except Exception:
                self._crush_frame = pd.DataFrame(
                    columns=["ticker", "event_date", "pre_iv30", "crush_pct_iv30"])
        return self._crush_frame

    def _residual_pool(self):
        """The paired ``(move, crush)`` error pool, built once per process.

        From Tier 4's stored forecasts against the realized outcomes: the move
        from the panel, the crush from Tier 2's pre/post print vol pair. Both
        errors for an event come from that event, which is what carries the
        dependence between them without anyone estimating it.
        """
        from engine import pnl_sim

        # Cached on the INSTANCE, never on the class. A class-level cache
        # survives into a differently-configured environment — the nightly's
        # replay builds a synthetic panel, and a pool retained from real data
        # made some rows score against one universe and some against another,
        # which surfaced as four unexplained selfcheck mismatches.
        if self._pool is not _UNSET:
            return self._pool
        try:
            from engine.data.features import tier4
            from engine.features import load_panel
            from engine.models.training import iv_crush

            panel = load_panel()[["ticker", "date", "abs_move"]].rename(
                columns={"date": "event_date"})
            forecasts = tier4.load_forecasts()[
                ["ticker", "event_date", "pred_abs_move", "pred_iv_crush_30"]]
            crush = self._crush_table()[["ticker", "event_date", "crush_pct_iv30"]]
            h = forecasts.merge(panel, on=["ticker", "event_date"], how="inner")
            h = h.merge(crush, on=["ticker", "event_date"], how="inner")
            h["err_move"] = h["abs_move"] - h["pred_abs_move"]
            h["err_crush"] = h["crush_pct_iv30"] - h["pred_iv_crush_30"]
            pool = pnl_sim.ResidualPool(h.dropna(subset=["err_move", "err_crush"]))
        except Exception:
            pool = None
        self._pool = pool
        return pool

    def _pre_print_iv(self, request, result, features=None) -> float | None:
        """iv30 at the last pre-print close — the level the crush multiplies.

        For a FORWARD event that close has not happened, so there is nothing to
        look up. The backtest never surfaced this because every event it scored
        had already printed. The live answer is the most recent iv30 the panel
        holds for the name, which is what a trader would use and which the crush
        forecast is itself conditioned on — recorded as a substitution rather
        than silently equated with the historical anchor.
        """
        # `pre_iv30` is a Tier-3 column as of 2026-09-05, so the live feature
        # frame already carries it — for a FORWARD event too, which is the whole
        # reason it was moved there. The Tier-2 pairing below is the fallback
        # for a panel built before that.
        if features is not None and "pre_iv30" in getattr(features, "columns", ()):
            value = _feature_value(features, "pre_iv30")
            if value is not None and value == value:
                return float(value)
        if self._pre_iv is _UNSET:
            try:
                frame = self._crush_table()[["ticker", "event_date", "pre_iv30"]]
                self._pre_iv = {(t, pd.Timestamp(d)): v for t, d, v
                                in zip(frame.ticker, frame.event_date, frame.pre_iv30)}
            except Exception:
                self._pre_iv = {}
        stored = self._pre_iv.get((request.ticker, pd.Timestamp(result.event_date)))
        if stored is not None and stored == stored:
            return float(stored)
        return self._latest_iv30(request.ticker, result.event_date)

    def _latest_iv30(self, ticker: str, before) -> float | None:
        """The newest iv30 for ``ticker`` strictly before ``before``.

        Strictly before, because a forward event's own date is in the future and
        anything at or after it would not exist yet.
        """
        if self._latest_iv is _UNSET:
            try:
                frame = self._crush_table()[["ticker", "event_date", "pre_iv30"]].dropna()
                frame = frame.sort_values("event_date")
                self._latest_iv = {t: g for t, g in frame.groupby("ticker")}
            except Exception:
                self._latest_iv = {}
        rows = self._latest_iv.get(ticker)
        if rows is None or rows.empty:
            return None
        earlier = rows[rows["event_date"] < pd.Timestamp(before)]
        return float(earlier["pre_iv30"].iloc[-1]) if len(earlier) else None

    def _crush_forecast(self, request, result, features=None) -> float | None:
        """``pred_iv_crush_30`` — SERVED for a forward event, stored for a past one.

        The stored Tier-4 table only covers events that have printed, so reading
        it alone made every FORWARD row undetermined and took TWIN-P5 off the
        board entirely. The size forecast never had this problem because it goes
        through ``tier4.serving_model``; this now does the same, and falls back
        to the stored value for a historical row where the two agree by
        construction (§5 of the Tier-4 guide).
        """
        from engine.data.features import tier4

        if self._crush is _UNSET:
            try:
                frame = tier4.load_forecasts()[["ticker", "event_date", "pred_iv_crush_30"]]
                self._crush = {(t, pd.Timestamp(d)): v for t, d, v
                               in zip(frame.ticker, frame.event_date, frame.pred_iv_crush_30)}
            except Exception:
                self._crush = {}
        stored = self._crush.get((request.ticker, pd.Timestamp(result.event_date)))
        if stored is not None and stored == stored:
            return float(stored)
        if features is None:
            return None
        try:
            served = self._serving(
                tier4.serving_fold(result.event_date, result.as_of),
                produces="pred_iv_crush_30",
            )
            value = float(served.predict(features)[0])
            return value if value == value else None
        except Exception:
            return None

    def _expectation(self, request, result, features=None) -> dict | None:
        """Simulate this event's return distribution, or ``None`` if it cannot.

        The exit legs are the entry legs with their sides reversed — the
        position is closed, so what was bought is sold. Their strikes and
        expiry are fixed at entry, and only the DTE remaining differs, which is
        exactly what the exit date supplies.
        """
        from engine import pnl_sim

        legs = getattr(result, "_priced_legs", None)
        if not legs or result.event_date is None:
            return None
        pool = self._residual_pool()
        if pool is None:
            return None
        exit_legs = [
            {"strike": float(leg.strike), "qty": float(leg.qty),
             "side": "sell" if str(leg.side).lower() == "buy" else "buy"}
            for leg in legs
        ]
        dte_exit = None
        if result.expiry is not None and result.exit_date is not None:
            dte_exit = float((pd.Timestamp(result.expiry) - pd.Timestamp(result.exit_date)).days)
        return pnl_sim.expected_pnl(
            exit_legs=exit_legs,
            spot=result.spot,
            entry_cost=result.entry_cost,
            pre_iv30=self._pre_print_iv(request, result, features),
            pred_abs_move=result.forecast_abs_move,
            pred_iv_crush=self._crush_forecast(request, result, features),
            dte_exit=dte_exit,
            event_date=result.event_date,
            pool=pool,
            key=request.strategy,
        )

    def _simulated_pnl(self, request, result, features=None) -> dict:
        """``exp_pnl_sim`` and the trailing bar it must clear, or an empty dict.

        Empty rather than zeros: every missing input has to reach the rule as a
        MISSING FACT so the verdict is undetermined. A simulated expectation
        defaulted to zero would sit exactly on a gate whose bar is a small
        positive number, and the row would read as a considered rejection.
        """
        from engine import pnl_sim

        out: dict = {}
        try:
            history = pnl_sim.load_history()
            if history is not None:
                bar = pnl_sim.trailing_cutoff(history, result.event_date)
                if bar is not None:
                    out["pnl_cutoff"] = bar
            sim = self._expectation(request, result, features)
            if sim is not None:
                out["exp_pnl_sim"] = sim["exp_pnl_sim"]
                result.exp_pnl_sim = sim["exp_pnl_sim"]
                result.win_sim = sim["win_sim"]
        except Exception as exc:  # a board must not die on one unsimulable row
            result.detail = (f"{result.detail}; expected-P&L unavailable: {exc}"
                             if result.detail else f"expected-P&L unavailable: {exc}")
        return out

    #: Columns a gate may name that are not part of the base ``_features()``
    #: frame — computed on demand in :meth:`_gate_feature_frame` from
    #: signals that already exist elsewhere on ``result`` or via Tier 4,
    #: never re-derived a second way. Kept in one place so a gate's feature
    #: list is the only thing that decides whether the extra computation
    #: runs at all — an old gate whose features never name these pays
    #: nothing for them.
    _GATE_FORECAST_COLUMNS = (
        "pred_abs_move", "pred_abs_move_p10", "pred_abs_move_p90",
        "pred_abs_move_sd", "forecast_edge",
    )
    _GATE_ANALOG_COLUMNS = ("analog_mean", "analog_win_rate", "analog_n")

    def _forecast_for_gate(self, request, result, features) -> dict[str, float]:
        """``pred_abs_move`` (+p10/p90/sd), served leak-safe via Tier 4.

        The SAME machinery :meth:`_size_from_forecast` uses for
        forecast-sized structures — factored out here so a gate that merely
        CONSUMES the forecast as a feature (never sizes its structure from
        it, which is STR-THRU's case) reuses the identical leak-safe serving
        path rather than a second implementation that could silently drift
        from the stored Tier-4 table a gate was trained against.
        """
        out = {c: float("nan") for c in self._GATE_FORECAST_COLUMNS}
        try:
            served = self._serving(tier4.serving_fold(result.event_date, result.as_of))
        except Exception:  # a board must not die on one unfit fold
            return out
        if any(f not in features.columns for f in served.features):
            return out
        try:
            forecast = float(served.predict(features)[0])
        except Exception:
            return out
        if forecast != forecast:  # NaN
            return out
        out["pred_abs_move"] = forecast
        try:
            p10, p90, sd, _ = served.interval([forecast])
            if np.isfinite(sd[0]):
                out["pred_abs_move_p10"] = float(p10[0])
                out["pred_abs_move_p90"] = float(p90[0])
                out["pred_abs_move_sd"] = float(sd[0])
        except Exception:
            pass
        implied = _feature_value(features, "im")
        if implied is not None:
            out["forecast_edge"] = forecast - float(implied)
        return out

    def _gate_feature_frame(self, request, result, features, artifact_features):
        """``features``, extended with forecast/analog columns a gate names.

        A no-op copy for every gate that does not name these columns — old
        gates (the STR-THRU/STR-RUNUP incumbents, any arithmetic-rule
        strategy) are unaffected. ``analog_mean``/``analog_win_rate``/
        ``analog_n`` are read straight off ``result`` rather than
        recomputed: :meth:`_score_analogs` already ran earlier in
        :meth:`score` and produced exactly this row's matched analog set, at
        the SAME request fill alpha the gate's other features (e.g.
        ``entry_cost_pct``) already carry — recomputing would be a second,
        possibly-inconsistent answer to a question already answered.
        """
        wanted = set(artifact_features)
        needed_forecast = [c for c in self._GATE_FORECAST_COLUMNS
                          if c in wanted and c not in features.columns]
        needed_analog = [c for c in self._GATE_ANALOG_COLUMNS
                         if c in wanted and c not in features.columns]
        if not needed_forecast and not needed_analog:
            return features
        out = features.copy()
        if needed_forecast:
            for column, value in self._forecast_for_gate(request, result, features).items():
                out[column] = value
        if needed_analog:
            out["analog_mean"] = (
                float(result.exp_pnl_analog) if result.exp_pnl_analog is not None else float("nan")
            )
            out["analog_win_rate"] = (
                float(result.win_analog) if result.win_analog is not None else float("nan")
            )
            out["analog_n"] = float(result.n_analogs)
        return out

    def _score_gate(self, request, result, features) -> None:
        loaded = self.model("gate", request.strategy)
        if loaded is None:
            self._apply_entry_rule(request, result, features)
            return
        entry, artifact = loaded
        result.model_versions["gate"] = entry.id
        if not self._gate_in_domain(request, features):
            result.flag("OUT_OF_DOMAIN")
            return
        features = self._gate_feature_frame(request, result, features, artifact.features)
        # A gate that declines says WHY. Both branches used to `return` in
        # silence, which put an unexplained `n/a` on the board — indistinguishable
        # from a name the gate had never been asked about. It matters more since
        # the implied-move sentinel was nulled (EXP-110): a name with no quoted
        # implied move now has a non-finite `im`, and withholding on it is the
        # right call, but only if the row says so.
        missing = [f for f in artifact.features if f not in features.columns]
        if missing:
            result.flag("MISSING_FEATURES")
            result.detail = f"gate {entry.id} needs {missing}"
            return
        X = features[list(artifact.features)].to_numpy(dtype=float)
        if not np.isfinite(X).all():
            absent = [f for f, ok in zip(artifact.features, np.isfinite(X[0])) if not ok]
            result.flag("MISSING_FEATURES")
            # `.strip()` on the fragment ate the separating space and ran the
            # analog detail straight into this one ("…dte_bandgate gate_…").
            # The join belongs between the parts, not inside one of them.
            note = f"gate {entry.id}: non-finite {absent}"
            result.detail = f"{result.detail}; {note}" if result.detail else note
            return
        result.gate_score = float(artifact.predict(X)[0])
        result.gate_threshold = entry.threshold
        if entry.threshold is not None:
            result.gate_pass = bool(result.gate_score >= entry.threshold)

    def _compare_layers(self, result) -> None:
        model, analog = result.exp_pnl_model, result.exp_pnl_analog
        if model is None or analog is None:
            return
        if np.sign(model) != np.sign(analog):
            result.flag("LAYER_DISAGREE")
            return
        if result.ci_low is not None and not (result.ci_low <= model <= result.ci_high):
            result.flag("LAYER_DISAGREE")


# --------------------------------------------------------------------------
# module-level convenience
# --------------------------------------------------------------------------

_DEFAULT: Scorer | None = None


def _scorer() -> Scorer:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Scorer()
    return _DEFAULT


def _snapshot_hash() -> str:
    snap = manifest.read_snapshot()
    return snap.get("snapshot", "") if snap else ""


def score(
    ticker: str,
    strategy: str,
    *,
    as_of=None,
    event_date=None,
    strike: float | None = None,
    expiry=None,
    fill: FillModel = MID,
    scorer: Scorer | None = None,
    **kwargs,
) -> ScoreResult:
    """Score one (ticker, strategy) at a decision date."""
    request = ScoreRequest(
        ticker=ticker,
        strategy=strategy,
        as_of=as_of,
        event_date=event_date,
        strike=strike,
        expiry=expiry,
        fill=fill,
        **kwargs,
    )
    return (scorer or _scorer()).score(request)


#: The exceptions that mean "this row cannot be priced", as opposed to "the
#: engine is broken". A calendar row that hits one becomes a NO_CHAIN placeholder
#: rather than taking the whole board down — most often an event whose chains
#: were never pulled.
UNSCORABLE = (KeyError, StructureError)


def unscorable_result(
    request: ScoreRequest, *, as_of, snapshot: str, exc: Exception
) -> ScoreResult:
    """The placeholder for a row the engine cannot price.

    Shared rather than inlined because the dashboard self-check re-scores board
    rows through a second path: if it built its own placeholder, the two could
    drift and every unpriceable row would read as a self-check mismatch. That
    is also why the flag is chosen from the exception TYPE here — both paths
    must reach the same one from the same failure.
    """
    result = ScoreResult(
        ticker=request.ticker,
        strategy=request.strategy,
        as_of=pd.Timestamp(as_of),
        event_date=None if request.event_date is None else pd.Timestamp(request.event_date),
        snapshot_hash=snapshot,
        detail=str(exc),
    )
    result.flag("COARSE_LADDER" if isinstance(exc, LadderTooCoarse) else "NO_CHAIN")
    return result


def score_calendar(
    as_of=None,
    *,
    horizon_days: int = 21,
    strategies: Sequence[str] | None = None,
    fill: FillModel = MID,
    scorer: Scorer | None = None,
    tickers: Iterable[str] | None = None,
    progress_every: int = 50,
    alt_strikes: int = 0,
    quote_max_age_sessions: int | None = 5,
) -> pd.DataFrame:
    """Score every confirmed event in the next ``horizon_days`` × every strategy.

    The dashboard's input. One chain index is loaded for the whole board rather
    than one per event, which is the difference between seconds and the wrong
    side of the guide's five-minute budget.

    ``alt_strikes`` adds *n* strikes either side of ATM per row, expressed as
    ±2.5% steps of spot. It defaults to **0**, which is a deliberate choice
    rather than an omission: every non-ATM score is `EXTRAPOLATED` until the
    moneyness edge-decay experiment is promoted (Phase 2 backlog 4), so
    generating five times the rows would multiply the board's cost to produce
    labelled guesses. The Phase 3 strike explorer calls :func:`score` per strike
    on demand, which is the right shape for a view nobody looks at until they
    pick a ticker.
    """
    engine = scorer or _scorer()
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    horizon = as_of + pd.Timedelta(days=horizon_days)
    strategies = list(strategies) if strategies is not None else live_strategies(sorted(STRUCTURES))

    events = store.read_table(
        "earnings_events", columns=["event_id", "ticker", "event_date", "session"]
    )
    events = events[
        (events["event_date"] >= as_of)
        & (events["event_date"] <= horizon)
        & events["session"].notna()
    ]
    if tickers is not None:
        events = events[events["ticker"].isin(set(tickers))]
    events = events.sort_values(["event_date", "ticker"]).reset_index(drop=True)
    print(f"  [score] {len(events):,} events in {as_of.date()} → {horizon.date()}", flush=True)

    # One pass to learn which chains the whole board needs.
    keys: set[tuple[str, pd.Timestamp]] = set()
    for strategy in strategies:
        if strategy in DISABLED_STRATEGIES:
            continue
        plan = plan_events(STRUCTURES[strategy](), events, calendar=engine.calendar)
        keys |= plan.chain_keys
    if quote_max_age_sessions is not None:
        # Every row whose entry has not happened is priced off the newest chain
        # we hold, so those have to be in the index too — otherwise the board
        # falls back to a chain it never loaded and reports NO_CHAIN anyway.
        from engine.replay import latest_chain_date

        for ticker in events["ticker"].astype(str).unique():
            try:
                newest = latest_chain_date(ticker, as_of)
            except Exception:
                continue
            if newest is not None:
                keys.add((ticker, pd.Timestamp(newest).normalize()))
    index = load_chain_index(keys, progress_every=0) if keys else ChainIndex({})

    #: Alternative strikes are offered as fractions of spot either side of ATM.
    offsets = [None] + [
        step * sign
        for step in (LADDER_STEP * k for k in range(1, alt_strikes + 1))
        for sign in (-1.0, 1.0)
    ]

    rows: list[dict] = []
    started = time.time()
    for i, event in enumerate(events.itertuples(index=False)):
        for strategy in strategies:
            atm_spot: float | None = None
            for offset in offsets:
                strike = (
                    None if offset is None or atm_spot is None
                    else ladder_strike(atm_spot, offset)
                )
                if offset is not None and strike is None:
                    continue  # the ATM pass found no chain; nothing to step off
                request = ScoreRequest(
                    ticker=str(event.ticker),
                    strategy=strategy,
                    as_of=None,
                    event_date=pd.Timestamp(event.event_date),
                    session=str(event.session),
                    strike=strike,
                    fill=fill,
                    quote_max_age_sessions=quote_max_age_sessions,
                    chain_as_of=as_of,
                )
                try:
                    result = engine.score(request, chain_index=index)
                except UNSCORABLE as exc:
                    result = unscorable_result(
                        request, as_of=as_of, snapshot=engine.snapshot, exc=exc
                    )
                if offset is None:
                    atm_spot = result.spot
                rows.append(result.as_dict() | {"strike_offset": offset})
        if progress_every and i and i % progress_every == 0:
            elapsed = time.time() - started
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining_events = len(events) - i
            eta_s = remaining_events / rate if rate > 0 else float("nan")
            print(
                f"  [score] {i:,}/{len(events):,} events, {elapsed:.0f}s elapsed, "
                f"{rate:.2f} events/s, ETA {eta_s:.0f}s "
                f"(live_features cache: {len(engine._live_features_cache):,} entries)",
                flush=True,
            )
    print(f"  [score] {len(rows):,} scores in {time.time()-started:.0f}s "
          f"(live_features cache: {len(engine._live_features_cache):,} entries)", flush=True)
    frame = pd.DataFrame(rows)
    return pd.concat([frame, dynamic_short_vol(frame)], ignore_index=True) \
        if len(frame) else frame


#: The families the dynamic chooser may offer, and why these five.
#:
#: EXP-141 chose them out of sample — ranked on 2018-2022 by how often each was
#: the realized best, evaluated on 2023-2026 — and the training menu came back
#: identical to the full-sample one. The three excluded families are coin flips
#: inside an argmax: precision 12.6%, 12.3% and 16.4% against a 12.5% chance
#: baseline, and dropping them was worth +0.37 of Sharpe and +35% of profit per
#: dollar of collateral on the holdout, which is more than reshaping the move
#: distribution (EXP-138) and recalibrating the expectation (EXP-139) produced
#: combined.
#:
#: Read `guides/exp141_smaller_menu.md` before changing this list. The gain is
#: economic, not skill — lift over chance FALLS as the menu shrinks, 1.98x to
#: 1.59x — so adding a family back does not dilute an edge, it reintroduces a
#: coin flip.
DYNAMIC_MENU: tuple[str, ...] = ("TWIN-P", "TWIN-P5", "CND-PS", "BFLY-P", "BFLY-P5")

#: What the chooser is called on the board.
DYNAMIC_STRATEGY = "DYN-SV"

#: What identifies one event among scored rows.
#:
#: NOT ``event_id``. That column is the Tier-2 ``earnings_events`` key and it
#: does not survive into a board row: :meth:`ScoreResult.as_dict` carries
#: ``ticker`` and ``event_date`` and nothing else that names the event. The
#: first version of this grouped on ``event_id`` and passed its unit tests,
#: because the fixture invented the column the real caller never supplies.
_EVENT_KEY: tuple[str, ...] = ("ticker", "event_date")


def dynamic_short_vol(frame: pd.DataFrame,
                      menu: tuple[str, ...] = DYNAMIC_MENU) -> pd.DataFrame:
    """One row per event: the menu structure with the highest expected P&L.

    A meta-strategy over rows that have already been scored, rather than a
    sixth structure. That is not a shortcut — a chooser has no leg list of its
    own, and expressing it as a ``STRUCTURES`` entry would mean inventing one.
    What it emits is the winner's own row, relabelled, with ``chosen_strategy``
    and the runner-up recorded so the board shows WHICH structure it picked and
    by how much.

    **It does not re-gate.** The winner keeps its own entry-rule verdict, so a
    chooser row that says NO says it for the same reason the underlying
    structure did. Picking a structure and deciding to trade it are separate
    decisions and the board should not merge them.

    Ties and missing simulations decline rather than default: an event where no
    menu member produced an ``exp_pnl_sim`` yields no chooser row at all, which
    reads as "no opinion" instead of an arbitrary pick.
    """
    if frame.empty or "exp_pnl_sim" not in frame.columns:
        return frame.iloc[0:0]
    live = frame[
        frame["strategy"].isin(menu)
        & frame["exp_pnl_sim"].notna()
        # The ATM pass only: a ladder row is an alternative strike for a
        # structure, not a different structure, and ranking across them would
        # let one family enter the contest five times.
        & (frame.get("strike_offset").isna() if "strike_offset" in frame else True)
    ]
    if live.empty:
        return frame.iloc[0:0]

    missing = [c for c in _EVENT_KEY if c not in live.columns]
    if missing:
        raise KeyError(
            f"dynamic_short_vol needs {_EVENT_KEY} to identify an event; "
            f"missing {missing}. Board rows come from ScoreResult.as_dict(), "
            "which carries no event_id — that column exists in Tier-2 tables, "
            "not on the board.")

    picks = []
    for _, block in live.groupby(list(_EVENT_KEY), sort=False, dropna=False):
        ordered = block.sort_values("exp_pnl_sim", ascending=False)
        best = ordered.iloc[0].to_dict()
        runner = ordered.iloc[1] if len(ordered) > 1 else None
        best["chosen_strategy"] = best["strategy"]
        best["strategy"] = DYNAMIC_STRATEGY
        best["chosen_margin"] = (
            None if runner is None
            else float(best["exp_pnl_sim"]) - float(runner["exp_pnl_sim"]))
        best["menu_size"] = int(len(ordered))
        detail = (f"chose {best['chosen_strategy']} of {len(ordered)} "
                  f"(exp P&L {100*float(best['exp_pnl_sim']):+.1f}%")
        if runner is not None:
            detail += (f", next {runner['strategy']} "
                       f"{100*float(runner['exp_pnl_sim']):+.1f}%")
        best["detail"] = f"{detail})" + (
            f"; {best['detail']}" if best.get("detail") else "")
        picks.append(best)
    return pd.DataFrame(picks)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _trading_days_before(calendar, entry_date, event_date, session) -> float:
    """Trading days from ``entry_date`` to the last pre-print close.

    Zero for a structure entering at that close (STR-THRU); 14 for one entering
    fourteen trading days earlier (STR-RUNUP). Matches the ``j`` grid the
    ``implied_t1`` dataset is built on.
    """
    if entry_date is None or event_date is None or session is None:
        return float("nan")
    try:
        pre = calendar.last_pre_print(event_date, session)
        return float(
            calendar.index_of(pre, side="prev") - calendar.index_of(entry_date, side="prev")
        )
    except (KeyError, ValueError):
        return float("nan")


def _mean_relative_spread(priced) -> float | None:
    """Mean ``(ask − bid) / mid`` over the entry legs, or ``None``.

    The mean rather than the max: one wide wing on an eight-leg structure is a
    different problem from eight uniformly wide legs, and the cost that decides
    whether the trade is crossable is the total, which the mean tracks.
    """
    values = []
    for leg in priced.legs:
        mid = 0.5 * (float(leg.bid) + float(leg.ask))
        if mid > 0:
            values.append((float(leg.ask) - float(leg.bid)) / mid)
    return float(np.mean(values)) if values else None


def _structure_peak(structure, width: float | None) -> float | None:
    """The shape's terminal payoff at its peak, in dollars.

    ``None`` when the structure does not declare a ``peak_multiple`` — the
    reward term then reports UNDETERMINED rather than silently testing a
    number it made up.
    """
    multiple = structure.params.get("peak_multiple")
    if multiple is None or width is None:
        return None
    try:
        return float(multiple) * float(width)
    except (TypeError, ValueError):
        return None


def _structure_width(structure, priced) -> float | None:
    """The structure's own ``w``, for one that declares which legs define it.

    Read off the PRICED legs rather than recomputed from the width parameter,
    because the parameter is a target and the strike is what the ladder
    actually listed. An entry rule comparing cost against a target width the
    market never offered would be testing a trade that does not exist.
    """
    names = structure.params.get("width_legs")
    if not names:
        return None
    try:
        lo, hi = (float(priced.leg(name).strike) for name in names)
    except (KeyError, ValueError):
        return None
    return abs(hi - lo)


def _feature_value(features, column: str) -> float | None:
    if column not in features.columns or features.empty:
        return None
    value = features[column].iloc[0]
    return float(value) if pd.notna(value) else None


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
