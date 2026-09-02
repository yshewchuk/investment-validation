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
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from engine import paths
from engine.analogs import AnalogMatcher, AnalogSet, bucket_frame
from engine.audit import FeatureVector, assert_causal, assert_decision_causal
from engine.calendar import trading_calendar
from engine.data import manifest, store
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
from engine.models.registry import Registry, RegistryError, load_registry
from engine.payoff import PayoffError, PayoffMap, fit_payoff, simulate_returns
from engine.replay import ChainIndex, legs_spot_dte, load_chain_index, plan_events
from engine.structures import (
    STRUCTURES,
    ExpirySelector,
    LegSpec,
    StrikeSelector,
    Structure,
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
    )
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
)

#: A ticker with fewer prior events than this is the regime where the size model
#: measurably degrades (EXP-024 addendum 2).
THIN_HISTORY_EVENTS = 4

#: Market-cap floor of the gates' training universe. The `<1B` slice joined the
#: target set 2026-09-01 (engine/data/pulls/sep2026_plan.py) with near-zero
#: prior coverage, one day after the gates were promoted on 2026-08-30, and
#: EXP-118 showed retraining on the expanded universe does not clear the
#: champions. Until a gate trained on the expanded universe promotes, the
#: decision is withheld for names the champion never saw: a gate call on a
#: name outside its evidence is an undetermined result, not a trade.
GATE_MCAP_FLOOR = 1e9


def _computed_moves_names() -> frozenset:
    """The EXP-117 universe: names the oquants panel does not carry, scored
    from synthesized moves — absent from every promoted gate's training data
    by construction."""
    return frozenset(
        p.stem[len("moves_"):] for p in paths.COMPUTED_MOVES.glob("moves_*.json")
    )

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

    # gate
    gate_score: float | None = None
    gate_threshold: float | None = None
    gate_pass: bool | None = None

    # the trade being scored
    entry_date: pd.Timestamp | None = None
    exit_date: pd.Timestamp | None = None
    #: Earlier of the decision date and the entry date — the cutoff every
    #: piece of evidence behind this score respects.
    evidence_cutoff: pd.Timestamp | None = None
    strike: float | None = None
    expiry: pd.Timestamp | None = None
    entry_cost: float | None = None
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
            "evidence_cutoff",
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
        self._recalibrations: dict[tuple[str, float, object], object] = {}
        self._recal_pairs = None
        self._quotes: dict[tuple[str, object], float | None] = {}
        self._verify = verify_artifacts

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
            "ticker", "date", "mcap_usd", "or_implied", "mean_prior_implied_move",
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
        structure = self._structure(request)

        window = self.calendar.resolve_offsets(
            event_date, session, structure.entry_offset, structure.exit_offset
        )
        result.entry_date, result.exit_date = window.entry_date, window.exit_date
        if result.as_of is None:
            result.as_of = window.entry_date
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

        Only *independent* selectors are overridden. A leg declared ``same_as``
        another follows it by design: overriding both would let a straddle's call
        and put drift onto different strikes and stop being a straddle.
        """
        structure = STRUCTURES[request.strategy]()
        if request.strike is None and request.expiry is None:
            return structure

        legs = []
        for leg in structure.legs:
            strike_sel = leg.strike
            if request.strike is not None and leg.strike.kind != "same_as":
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
        """Resolve and price the structure on the real entry chain."""
        index = chain_index
        if index is None:
            index = load_chain_index(
                [(request.ticker, result.entry_date), (request.ticker, result.exit_date)],
                progress_every=0,
            )
        rows = index.get(request.ticker, result.entry_date)
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
            obs_date=result.entry_date,
            event_date=result.event_date,
            rows=clean,
            session=result.session,
        )
        try:
            priced = price_structure(structure, snapshot, request.fill)
        except (StructureError, ValueError):
            result.flag("NO_CHAIN")
            self._note_chain_age(request, result)
            return

        result.entry_cost = float(priced.cost)
        result.spot = float(priced.spot)
        result.strike = float(priced.legs[0].strike)
        result.expiry = priced.legs[0].expiry
        result.dte_entry = int(priced.legs[0].dte)
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

        One call per score, shared by the market block and the event-history
        block below — they are two slices of the same vector, and computing it
        twice would double the cost of every forward row on the board.
        """
        try:
            vector = live_features(
                request.ticker,
                result.event_date,
                as_of=result.entry_date,
                session=result.session,
                context=self.context,
            )
        except (KeyError, ValueError, FileNotFoundError):
            # No prior events, or no price history to build the run-up block
            # from. The model layer will report MISSING_FEATURES, which is the
            # honest answer for a name we know nothing about.
            return {}
        return dict(vector.values)

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
        prior = features.get("mean_prior_implied_move")
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

        The gates were promoted 2026-08-30 on a universe with near-zero `<1B`
        coverage and no computed-moves names; both joined the board 2026-09-01.
        EXP-118 showed retraining on the expanded universe does not clear the
        champions, so the champions stand — which is only coherent if the gate
        also refuses to decide the names it was never validated on. A gate call
        on an out-of-domain name is an undetermined result, not a trade.
        """
        if request.ticker in _computed_moves_names():
            return False
        if "mcap_log" in features.columns:
            mcap_log = pd.to_numeric(features["mcap_log"], errors="coerce")
            if len(mcap_log) and np.isfinite(mcap_log.iloc[0]):
                if float(mcap_log.iloc[0]) < np.log(GATE_MCAP_FLOOR):
                    return False
        return True

    def _score_gate(self, request, result, features) -> None:
        loaded = self.model("gate", request.strategy)
        if loaded is None:
            return
        entry, artifact = loaded
        result.model_versions["gate"] = entry.id
        if not self._gate_in_domain(request, features):
            result.flag("OUT_OF_DOMAIN")
            return
        missing = [f for f in artifact.features if f not in features.columns]
        if missing:
            return
        X = features[list(artifact.features)].to_numpy(dtype=float)
        if not np.isfinite(X).all():
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
    """The NO_CHAIN placeholder for a row the engine cannot price.

    Shared rather than inlined because the dashboard self-check re-scores board
    rows through a second path: if it built its own placeholder, the two could
    drift and every unpriceable row would read as a self-check mismatch.
    """
    result = ScoreResult(
        ticker=request.ticker,
        strategy=request.strategy,
        as_of=pd.Timestamp(as_of),
        event_date=None if request.event_date is None else pd.Timestamp(request.event_date),
        snapshot_hash=snapshot,
        detail=str(exc),
    )
    result.flag("NO_CHAIN")
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
    strategies = list(strategies) if strategies is not None else sorted(STRUCTURES)

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
    index = load_chain_index(keys, progress_every=0) if keys else ChainIndex({})

    #: Alternative strikes are offered as fractions of spot either side of ATM.
    offsets = [None] + [
        step * sign
        for step in (0.025 * k for k in range(1, alt_strikes + 1))
        for sign in (-1.0, 1.0)
    ]

    rows: list[dict] = []
    started = time.time()
    for i, event in enumerate(events.itertuples(index=False)):
        for strategy in strategies:
            atm_spot: float | None = None
            for offset in offsets:
                strike = (
                    None if offset is None or atm_spot is None else atm_spot * (1 + offset)
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
            print(
                f"  [score] {i:,}/{len(events):,} events, {time.time()-started:.0f}s",
                flush=True,
            )
    print(f"  [score] {len(rows):,} scores in {time.time()-started:.0f}s", flush=True)
    return pd.DataFrame(rows)


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


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
