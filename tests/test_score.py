"""The scoring API.

Built against a synthetic world — a hand-made panel, chain, trade set and
registry — so the behaviour under test is the scorer's logic rather than the
state of the real store. The integration properties (that these numbers match
the replayed trades, that the models are the registered champions) are the
acceptance layer's job, in ``checks/phase1_*``.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from engine import score as score_mod
from engine.audit import LeakError
from engine.calendar import TradingCalendar
from engine.features import FeatureContext
from engine.fills import BEST, MID, WORST
from engine.models.registry import ModelArtifact, Registry, RegistryEntry
from engine.replay import ChainIndex
from engine.score import ScoreRequest, ScoreResult, Scorer

TICKER = "TEST"
EVENT = pd.Timestamp("2024-05-02")
ENTRY = pd.Timestamp("2024-05-02")  # AMC → last pre-print close is the event date
EXIT = pd.Timestamp("2024-05-03")


class Linear:
    """Predicts a fixed value, so payoff arithmetic is checkable by hand."""

    def __init__(self, value=5.0):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def calendar():
    """Spans the whole synthetic history, so early decision dates resolve too."""
    return TradingCalendar(pd.bdate_range("2015-01-01", "2024-12-31"))


#: Enough names and history for the analog buckets (n >= 30) and the payoff fit
#: (n >= 200) to be satisfied without widening, so the tests exercise the
#: intended path rather than the fallback.
PEERS = [TICKER] + [f"PEER{i}" for i in range(9)]
HISTORY = pd.date_range("2016-05-02", periods=32, freq="91D")


@pytest.fixture
def panel():
    """Ten names with quarterly events; TICKER's last event is the one scored."""
    rng = np.random.default_rng(11)
    rows = []
    for ticker in PEERS:
        for k, date in enumerate(HISTORY):
            # The realized move has to vary: it is the payoff map's driver, and
            # a constant driver has no line to fit.
            abs_move = float(rng.uniform(1.0, 12.0))
            rows.append(
                {
                    "ticker": ticker, "k": k + 4, "date": date, "year": date.year,
                    "quarter": "Q1", "move": abs_move, "abs_move": abs_move,
                    "implied_move": 6.0,
                    "n_prior": k + 4,
                    "mean_prior_move": 1.0, "mean_prior_abs_move": 5.0,
                    "mean_prior_implied_move": 6.0,
                    "ema2_prior_move": 1.0, "ema4_prior_move": 1.0,
                    "ema8_prior_move": 1.0, "ema12_prior_move": 1.0,
                    "ema2_prior_abs_move": 5.0, "ema4_prior_abs_move": 5.0,
                    "ema8_prior_abs_move": 5.0, "ema12_prior_abs_move": 5.0,
                    "ema12r_abs": 5.0, "signed_streak": 2.0,
                    "or_implied": 6.0, "or_rvol30": 40.0, "mcap_log": np.log(5e9),
                    "mcap_usd": 5e9, "spy_vol20": 15.0, "spy_dd252": -5.0,
                    "dist_high": -10.0, "dist_ema": 3.0,
                }
            )
    frame = pd.DataFrame(rows)
    # Move TICKER's final event onto the date under test.
    last = frame[(frame["ticker"] == TICKER)].index[-1]
    frame.loc[last, "date"] = EVENT
    frame.loc[last, "year"] = EVENT.year
    return frame


@pytest.fixture
def daily():
    dates = pd.bdate_range("2015-06-01", periods=2400)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "ticker": ticker, "date": dates, "src_iv": "orats",
                    "implied_move": 6.0, "iv10": 40.0, "iv30": 35.0,
                    "exern_iv10": 30.0, "exern_iv30": 28.0, "iee": 1.2,
                    "skew": 0.3, "contango": 0.1, "fwd90_30": 32.0,
                    "fexern90_30": 30.0, "rvol30": 38.0, "spot": 100.0,
                    "mcap_log": np.log(5e9),
                }
            )
            for ticker in PEERS
        ],
        ignore_index=True,
    )


@pytest.fixture
def chain_index():
    def chain(obs, call, put):
        rows = []
        for expiry, dte in ((pd.Timestamp("2024-05-03"), 1), (pd.Timestamp("2024-05-24"), 22)):
            for strike in (95.0, 100.0, 105.0):
                for right, (bid, ask) in (("C", call), ("P", put)):
                    scale = 1.0 if dte < 10 else 2.0
                    rows.append(
                        {
                            "ticker": TICKER, "obs_date": obs, "expiry": expiry,
                            "dte": dte, "strike": strike, "right": right,
                            "bid": bid * scale, "ask": ask * scale, "spot": 100.0,
                            "quote_repaired": False,
                        }
                    )
        return pd.DataFrame(rows)

    return ChainIndex(
        {
            (TICKER, ENTRY): chain(ENTRY, (2.0, 2.4), (1.0, 1.4)),
            (TICKER, EXIT): chain(EXIT, (3.0, 3.4), (2.0, 2.4)),
        }
    )


@pytest.fixture
def trades(panel):
    """A Tier-2-shaped trade set, so the scorer's own enrichment is exercised.

    Known mid-fill mean of 0.08 and a payoff whose exit value is exactly
    ``(0.01 + 0.6·|move|)·spot``, so both layers have a checkable answer.
    """
    spread = np.random.default_rng(5).normal(0, 0.05, len(panel))
    rows = []
    for i, event in enumerate(panel.itertuples(index=False)):
        if event.date >= EVENT:
            continue  # the scored event itself is not its own analog
        move = float(event.abs_move)
        # Returns need spread around their mean, or the bootstrap CI collapses
        # onto the mean and the interval tests would assert nothing.
        jitter = float(spread[i])
        for alpha, ret in (
            (0.0, -0.25 + jitter), (0.5, 0.08 + jitter), (1.0, 0.3 + jitter)
        ):
            rows.append(
                {
                    "trade_id": f"{event.ticker}:{event.date.date()}:a{int(alpha*100)}",
                    "kind": "sim", "strategy": "STR-THRU", "variant": "e+0_x+1",
                    "ticker": event.ticker,
                    "event_id": f"{event.ticker}_{event.date.date()}",
                    "event_date": event.date, "year": event.date.year,
                    "legs": json.dumps({"spot_entry": 100.0, "dte_entry": 2, "entry": [], "exit": []}),
                    "entry_date": event.date,
                    "exit_date": event.date + pd.Timedelta(days=1),
                    "strike": 100.0, "expiry": event.date + pd.Timedelta(days=1),
                    "fill_alpha": alpha, "entry_cost": 3.4,
                    "exit_value": (0.01 + 0.6 * move) * 100.0,
                    "ret": ret, "provenance": "engine.replay",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def registry(tmp_path):
    from engine.models.training import size_model

    art = ModelArtifact(
        model=Linear(5.0),
        role="size",
        features=size_model.FEATURES,
        residuals=np.array([-1.0, 0.0, 1.0]),
        target="abs_move",
    )
    path = tmp_path / "size.joblib"
    digest = art.save(path)
    entry = RegistryEntry(
        id="size_test", role="size", strategy="*",
        artifact=str(path), artifact_sha256=digest,
        features=list(size_model.FEATURES), target="abs_move",
        train_window="test", champion=True,
    )
    return Registry(entries=[entry])


@pytest.fixture
def scorer(registry, trades, panel, daily, calendar):
    context = FeatureContext(panel=panel, daily=daily, calendar=calendar)
    return Scorer(
        registry=registry, trades=trades, context=context, snapshot="snap-test"
    )


def request(**kwargs) -> ScoreRequest:
    base = dict(
        ticker=TICKER, strategy="STR-THRU", event_date=EVENT, session="AMC", fill=MID
    )
    base.update(kwargs)
    return ScoreRequest(**base)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


class TestDisabledStrategies:
    def test_cal_p_returns_no_numbers(self, scorer):
        result = scorer.score(request(strategy="CAL-P"))
        assert result.flags == ["UNVALIDATED_STRUCTURE"]
        assert result.exp_pnl_model is None
        assert result.exp_pnl_analog is None
        assert not result.scored

    def test_cal_p_explains_why(self, scorer):
        result = scorer.score(request(strategy="CAL-P"))
        assert "EXP-046b" in result.detail

    def test_unknown_strategy_raises(self, scorer):
        with pytest.raises(KeyError, match="unknown strategy"):
            scorer.score(request(strategy="NOPE"))

    def test_cnd_p_is_disabled_too(self, scorer):
        """CND-P shipped in engine.structures for EXP-121's replay/backtest use.

        STRUCTURES is shared with score_calendar's default strategy list — the
        live board — so a structure landing in that dict reaches production
        scoring unless something opts it back out. It reached the board once,
        233 spurious rows in one render, before this guard existed.
        """
        result = scorer.score(request(strategy="CND-P"))
        assert result.flags == ["UNVALIDATED_STRUCTURE"]
        assert not result.scored

    def test_every_registered_structure_is_either_promoted_or_disabled(self):
        """The general form of the CND-P leak, so the NEXT new structure trips
        this instead of shipping silently onto the live board.

        A structure only belongs in `score_calendar`'s default universe once
        SOMETHING has decided it is ready — a promoted gate, a pre-registered
        arithmetic entry rule, or the CAL-P/CND-P precedent of an explicit
        disable-with-reason. `STRUCTURES` has no fourth state; being merely
        defined is not being ready.

        The entry-rule branch is not a loophole. An arithmetic rule is a
        universe definition registered in an experiment's spec before it ran
        (engine/entry_rules.py), and it decides every row rather than the rows
        a training window happens to reach — which is more coverage than a gate
        gives, not less.
        """
        from engine.entry_rules import rule_for
        from engine.models.registry import load_registry
        from engine.score import DISABLED_STRATEGIES
        from engine.structures import STRUCTURES

        registry = load_registry(missing_ok=True)
        unaccounted = [
            name for name in STRUCTURES
            if name not in DISABLED_STRATEGIES
            and rule_for(name) is None
            and not (registry and registry.has_champion("gate", name))
        ]
        assert not unaccounted, (
            f"{unaccounted} are in STRUCTURES with no promoted gate, no entry "
            "rule and no DISABLED_STRATEGIES entry — they will be scored on "
            "the live board by default. Add a DISABLED_STRATEGIES reason until "
            "one of those exists."
        )


class TestEntryPricing:
    def test_prices_the_real_chain(self, scorer, chain_index):
        result = scorer.score(request(), chain_index=chain_index)
        # ATM straddle at the 1-DTE expiry, mid: (2.2 + 1.2) = 3.4.
        assert result.entry_cost == pytest.approx(3.4)
        assert result.strike == 100.0
        assert result.spot == 100.0
        assert result.dte_entry == 1

    def test_fill_alpha_moves_the_cost_the_right_way(self, scorer, chain_index):
        worst = scorer.score(request(fill=WORST), chain_index=chain_index)
        best = scorer.score(request(fill=BEST), chain_index=chain_index)
        assert worst.entry_cost == pytest.approx(3.8)
        assert best.entry_cost == pytest.approx(3.0)

    def test_no_chain_is_flagged_not_interpolated(self, scorer):
        result = scorer.score(request(), chain_index=ChainIndex({}))
        assert "NO_CHAIN" in result.flags
        assert result.entry_cost is None

    def test_analog_layer_still_answers_without_a_chain(self, scorer):
        result = scorer.score(request(), chain_index=ChainIndex({}))
        assert result.exp_pnl_analog is not None
        assert result.n_analogs > 0

    def test_session_sets_the_entry_date(self, scorer, chain_index):
        amc = scorer.score(request(session="AMC"), chain_index=chain_index)
        assert amc.entry_date == EVENT
        bmo = scorer.score(request(session="BMO"), chain_index=chain_index)
        assert bmo.entry_date < EVENT


class TestDecisionDate:
    """`as_of` is the date the score is *taken*, which every audit keys on.

    It now defaults to the structure's decision close rather than its entry
    close. Those are the same close for every structure that ships today, so
    these tests pin the current behaviour and the propagation that will make
    the T−2 variant work.
    """

    def test_as_of_defaults_to_the_decision_close(self, scorer, chain_index):
        result = scorer.score(request(), chain_index=chain_index)
        assert result.as_of == result.entry_date

    def test_an_explicit_as_of_is_still_respected(self, scorer, chain_index):
        asked = EVENT - pd.Timedelta(days=7)
        result = scorer.score(request(as_of=asked), chain_index=chain_index)
        assert result.as_of == asked

    def test_a_pinned_contract_keeps_the_decision_offset(self, scorer, monkeypatch):
        """`_structure` rebuilds the spec to pin the caller's strike/expiry. If
        it dropped the decision offset there, asking for a specific contract
        would silently revert the trade to deciding at its entry close."""
        from engine.structures import STRUCTURES, straddle_through

        monkeypatch.setitem(
            STRUCTURES, "STR-THRU", lambda: straddle_through(decision_offset=-1)
        )
        rebuilt = scorer._structure(request(strike=105.0))
        assert rebuilt.decision_offset == -1

    def test_an_early_decision_moves_as_of_off_the_entry_date(
        self, scorer, chain_index, monkeypatch
    ):
        """The whole point of the offset: the score is taken a session before
        the trade is placed, so there is time to act on it."""
        from engine.structures import STRUCTURES, straddle_through

        monkeypatch.setitem(
            STRUCTURES, "STR-THRU", lambda: straddle_through(decision_offset=-1)
        )
        result = scorer.score(request(), chain_index=chain_index)
        assert result.as_of < result.entry_date
        assert result.evidence_cutoff == result.as_of


class TestModelLayer:
    def test_produces_a_distribution_not_a_point(self, scorer, chain_index):
        result = scorer.score(request(), chain_index=chain_index)
        assert result.exp_pnl_model is not None
        assert result.model_p10 < result.exp_pnl_model < result.model_p90

    def test_win_rate_is_a_probability(self, scorer, chain_index):
        result = scorer.score(request(), chain_index=chain_index)
        assert 0.0 <= result.win_model <= 1.0

    def test_records_the_model_version(self, scorer, chain_index):
        result = scorer.score(request(), chain_index=chain_index)
        assert result.model_versions["abs_move"] == "size_test"

    def test_records_the_payoff_map_it_used(self, scorer, chain_index):
        result = scorer.score(request(), chain_index=chain_index)
        assert result.payoff["driver"] == "abs_move"
        assert result.payoff["n"] > 0

    def test_no_champion_is_flagged_not_faked(self, trades, panel, daily, calendar, chain_index):
        context = FeatureContext(panel=panel, daily=daily, calendar=calendar)
        engine = Scorer(
            registry=Registry(entries=[]), trades=trades, context=context, snapshot="s"
        )
        result = engine.score(request(), chain_index=chain_index)
        assert "NO_MODEL" in result.flags
        assert result.exp_pnl_model is None


class TestAnalogLayer:
    def test_reports_the_matched_population(self, scorer, chain_index):
        result = scorer.score(request(), chain_index=chain_index)
        assert result.n_analogs > 0
        assert result.exp_pnl_analog == pytest.approx(0.08, abs=0.02)
        assert result.ci_low < result.exp_pnl_analog < result.ci_high

    def test_alpha_selects_a_different_population(self, scorer, chain_index):
        worst = scorer.score(request(fill=WORST), chain_index=chain_index)
        mid = scorer.score(request(fill=MID), chain_index=chain_index)
        assert worst.exp_pnl_analog == pytest.approx(-0.25, abs=0.02)
        assert mid.exp_pnl_analog == pytest.approx(0.08, abs=0.02)

    def test_future_trades_are_never_analogs(self, scorer, chain_index):
        """Decide before any trade has closed and there is nothing to learn from."""
        first_event = HISTORY[0]
        result = scorer.score(
            request(event_date=first_event), chain_index=ChainIndex({})
        )
        assert result.n_analogs == 0
        assert "THIN_ANALOGS" in result.flags


class TestFlags:
    def test_atm_is_not_extrapolated(self, scorer, chain_index):
        result = scorer.score(request(), chain_index=chain_index)
        assert result.extrapolated is False
        assert "EXTRAPOLATED" not in result.flags

    def test_a_far_strike_is_labelled_extrapolated(self, scorer, chain_index):
        result = scorer.score(request(strike=105.0), chain_index=chain_index)
        assert result.strike == 105.0
        assert result.extrapolated is True
        assert "EXTRAPOLATED" in result.flags

    def test_a_strike_absent_from_the_chain_is_refused(self, scorer, chain_index):
        """Not silently priced at the money and reported as the requested one."""
        result = scorer.score(request(strike=120.0), chain_index=chain_index)
        assert "NO_CHAIN" in result.flags
        assert result.entry_cost is None

    def test_layer_disagreement_is_surfaced_not_averaged(
        self, scorer, chain_index, monkeypatch
    ):
        result = scorer.score(request(), chain_index=chain_index)
        # Force opposite signs and re-run the comparison.
        result.exp_pnl_model, result.exp_pnl_analog = 0.5, -0.5
        result.flags.clear()
        scorer._compare_layers(result)
        assert "LAYER_DISAGREE" in result.flags
        # Both numbers survive; nothing was averaged away.
        assert result.exp_pnl_model == 0.5 and result.exp_pnl_analog == -0.5

    def test_agreement_raises_no_flag(self, scorer):
        result = ScoreResult(ticker=TICKER, strategy="STR-THRU", as_of=EVENT)
        result.exp_pnl_model, result.exp_pnl_analog = 0.10, 0.08
        result.ci_low, result.ci_high = 0.05, 0.15
        scorer._compare_layers(result)
        assert "LAYER_DISAGREE" not in result.flags

    def test_a_model_outside_the_analog_ci_disagrees(self, scorer):
        result = ScoreResult(ticker=TICKER, strategy="STR-THRU", as_of=EVENT)
        result.exp_pnl_model, result.exp_pnl_analog = 0.90, 0.08
        result.ci_low, result.ci_high = 0.05, 0.15
        scorer._compare_layers(result)
        assert "LAYER_DISAGREE" in result.flags

    def test_wide_markets_are_flagged(self, scorer, panel, daily, calendar):
        rows = []
        for strike in (95.0, 100.0, 105.0):
            for right in ("C", "P"):
                for obs in (ENTRY, EXIT):
                    rows.append(
                        {
                            "ticker": TICKER, "obs_date": obs,
                            "expiry": pd.Timestamp("2024-05-03"), "dte": 1,
                            "strike": strike, "right": right,
                            "bid": 0.1, "ask": 5.0, "spot": 100.0,
                            "quote_repaired": False,
                        }
                    )
        frame = pd.DataFrame(rows)
        index = ChainIndex(
            {
                (TICKER, ENTRY): frame[frame["obs_date"] == ENTRY],
                (TICKER, EXIT): frame[frame["obs_date"] == EXIT],
            }
        )
        result = scorer.score(request(), chain_index=index)
        assert "WIDE_MARKET" in result.flags


class TestGateDomain:
    """The gate decides only inside the universe it was validated on (EXP-118)."""

    def _stub_gate(self, scorer):
        artifact = ModelArtifact(
            model=Linear(0.10),
            role="gate",
            features=("mcap_log",),
            residuals=np.array([-0.01, 0.0, 0.01]),
            target="ret",
        )
        entry = RegistryEntry(
            id="gate_test", role="gate", strategy="STR-THRU",
            artifact="x", artifact_sha256="",
            features=["mcap_log"], target="ret",
            train_window="test", champion=True, threshold=0.05,
        )
        scorer._models[("STR-THRU", "gate")] = (entry, artifact)

    def _result(self):
        return ScoreResult(ticker=TICKER, strategy="STR-THRU", as_of=EVENT)

    def test_an_in_domain_name_gets_a_decision(self, scorer):
        from engine.score import GATE_MCAP_FLOOR

        self._stub_gate(scorer)
        result = self._result()
        features = pd.DataFrame({"mcap_log": [np.log(2 * GATE_MCAP_FLOOR)]})
        scorer._score_gate(request(), result, features)
        assert "OUT_OF_DOMAIN" not in result.flags
        assert result.gate_score == pytest.approx(0.10)
        assert result.gate_pass is True

    def test_a_sub_1b_name_gets_no_decision(self, scorer):
        from engine.score import GATE_MCAP_FLOOR

        self._stub_gate(scorer)
        result = self._result()
        features = pd.DataFrame({"mcap_log": [np.log(GATE_MCAP_FLOOR / 2)]})
        scorer._score_gate(request(), result, features)
        assert "OUT_OF_DOMAIN" in result.flags
        assert result.gate_score is None
        assert result.gate_pass is None

    def test_a_computed_moves_name_gets_no_decision(self, scorer, monkeypatch):
        from engine import score as sm

        self._stub_gate(scorer)
        monkeypatch.setattr(sm, "_computed_moves_names", lambda: frozenset({TICKER}))
        result = self._result()
        features = pd.DataFrame({"mcap_log": [np.log(2e10)]})
        scorer._score_gate(request(), result, features)
        assert "OUT_OF_DOMAIN" in result.flags
        assert result.gate_pass is None


class TestCausality:
    def test_a_bmo_decision_on_the_event_date_is_refused(self, scorer, chain_index):
        with pytest.raises(LeakError):
            scorer.score(request(session="BMO", as_of=EVENT), chain_index=chain_index)


class TestRequestedContract:
    """`strike=` and `expiry=` must reach the legs, not just the label."""

    def test_a_requested_strike_is_actually_priced(self, scorer, chain_index):
        atm = scorer.score(request(), chain_index=chain_index)
        far = scorer.score(request(strike=105.0), chain_index=chain_index)
        assert atm.strike == 100.0
        assert far.strike == 105.0
        # Both are priced — the request selected a contract rather than a label.
        assert far.entry_cost is not None and atm.entry_cost is not None

    def test_a_requested_expiry_is_actually_priced(self, scorer, chain_index):
        near = scorer.score(request(), chain_index=chain_index)
        far = scorer.score(
            request(expiry=pd.Timestamp("2024-05-24")), chain_index=chain_index
        )
        assert near.dte_entry == 1
        assert far.dte_entry == 22
        # The 22-DTE quotes are scaled 2x in the fixture, so it must cost more.
        assert far.entry_cost > near.entry_cost

    def test_a_straddle_keeps_both_legs_on_one_strike(self, scorer, chain_index):
        """The put follows the call by `same_as`; overriding both would break it."""
        structure = scorer._structure(request(strike=105.0))
        kinds = {leg.name: leg.strike.kind for leg in structure.legs}
        assert kinds["call"] == "fixed"
        assert kinds["put"] == "same_as"

    def test_an_unrequested_structure_is_returned_unchanged(self, scorer):
        from engine.structures import straddle_through

        assert scorer._structure(request()).to_dict() == straddle_through().to_dict()

    def test_the_requested_contract_is_recorded_in_the_params(self, scorer):
        params = scorer._structure(request(strike=105.0)).params
        assert params["requested_strike"] == 105.0


class TestMarketBlock:
    """Where the last-pre-print-close market state comes from, and when."""

    def test_a_historical_event_reads_it_off_the_panel(self, scorer, chain_index):
        result = scorer.score(request(), chain_index=chain_index)
        block = scorer._market_block(request(), result)
        assert block["or_implied"] == 6.0
        assert block["mcap_usd"] == pytest.approx(5e9)

    def test_an_upcoming_event_falls_back_to_the_live_path(
        self, scorer, chain_index, monkeypatch
    ):
        """No panel row exists for a print that has not happened.

        Without this fallback the model layer would be dark for exactly the
        events the dashboard exists to score.
        """
        from engine.audit import FeatureVector

        upcoming = pd.Timestamp("2024-08-01")
        called = {}

        def fake_live(ticker, event_date, **kwargs):
            called["ticker"] = ticker
            return FeatureVector(
                ticker=ticker,
                as_of=kwargs.get("as_of") or event_date,
                values={"or_implied": 7.5, "mcap_usd": 6e9},
                feature_as_of={},
                event_date=event_date,
            )

        monkeypatch.setattr(score_mod, "live_features", fake_live)
        result = ScoreResult(
            ticker=TICKER, strategy="STR-THRU", as_of=upcoming,
            event_date=upcoming, session="AMC", entry_date=upcoming,
        )
        block = scorer._market_block(request(event_date=upcoming), result)
        assert called["ticker"] == TICKER
        assert block["or_implied"] == 7.5

    def test_an_unknown_name_yields_an_empty_block_not_a_crash(
        self, scorer, monkeypatch
    ):
        def raise_key_error(*args, **kwargs):
            raise KeyError("no prior panel events")

        monkeypatch.setattr(score_mod, "live_features", raise_key_error)
        result = ScoreResult(
            ticker="UNKNOWN", strategy="STR-THRU", as_of=EVENT,
            event_date=EVENT, session="AMC", entry_date=EVENT,
        )
        assert scorer._market_block(request(ticker="UNKNOWN"), result) == {}

    def test_an_early_entry_gets_no_market_block(self, scorer, panel, daily, calendar):
        """STR-RUNUP enters 14 trading days out; the block would be hindsight."""
        result = ScoreResult(
            ticker=TICKER, strategy="STR-RUNUP", as_of=EVENT,
            event_date=EVENT, session="AMC",
            entry_date=calendar.shift(EVENT, -14),
        )
        features = scorer._features(request(strategy="STR-RUNUP"), result)
        assert "or_implied" not in features.columns
        assert features["days_before_print"].iloc[0] == 14.0


class TestTrainingServingAgreement:
    """Features must mean at serving time what they meant at training time."""

    def test_days_before_print_counts_trading_days(self, scorer, chain_index):
        from engine.score import _trading_days_before

        cal = scorer.calendar
        # STR-THRU enters at the last pre-print close: zero trading days before.
        assert _trading_days_before(cal, EVENT, EVENT, "AMC") == 0.0
        # Fourteen trading days earlier is 14, not the ~20 calendar days it spans.
        entry = cal.shift(EVENT, -14)
        assert _trading_days_before(cal, entry, EVENT, "AMC") == 14.0
        assert (EVENT - entry).days > 14  # calendar days would have differed

    def test_str_thru_scores_days_before_print_as_zero(self, scorer, chain_index):
        features = scorer._features(
            request(), scorer.score(request(), chain_index=chain_index)
        )
        assert features["days_before_print"].iloc[0] == 0.0

    def test_a_missing_date_or_session_yields_nan_not_a_wrong_number(self, scorer):
        from engine.score import _trading_days_before

        assert np.isnan(_trading_days_before(scorer.calendar, None, EVENT, "AMC"))
        assert np.isnan(_trading_days_before(scorer.calendar, EVENT, None, "AMC"))
        assert np.isnan(_trading_days_before(scorer.calendar, EVENT, EVENT, None))

    def test_an_entry_before_the_calendar_starts_raises_upstream(self, scorer):
        """`resolve_offsets` is where an out-of-range window is caught."""
        with pytest.raises(KeyError):
            scorer.calendar.shift(scorer.calendar.first, -5)


class TestEvidenceCutoff:
    """Evidence follows the earlier of the decision and the entry, never the later."""

    def test_defaults_to_the_entry_date(self, scorer, chain_index):
        result = scorer.score(request(), chain_index=chain_index)
        assert result.evidence_cutoff == result.entry_date

    def test_an_earlier_decision_tightens_the_cutoff(self, scorer, chain_index):
        early = scorer.calendar.shift(EVENT, -5)
        result = scorer.score(request(as_of=early), chain_index=chain_index)
        assert result.evidence_cutoff == early

    def test_an_early_entry_wins_over_a_later_decision(self, scorer, chain_index):
        """STR-RUNUP opens 14 days before the close a caller may name as `as_of`."""
        result = ScoreResult(
            ticker=TICKER, strategy="STR-RUNUP", as_of=EVENT,
            entry_date=scorer.calendar.shift(EVENT, -14),
        )
        cutoff = min(d for d in (result.as_of, result.entry_date) if d is not None)
        assert cutoff == result.entry_date
        assert cutoff < EVENT


class TestDeterminism:
    def test_the_same_request_gives_a_byte_identical_result(self, scorer, chain_index):
        """Guide acceptance test 2."""
        a = scorer.score(request(), chain_index=chain_index)
        b = scorer.score(request(), chain_index=chain_index)
        assert a.digest() == b.digest()

    def test_a_different_alpha_gives_a_different_result(self, scorer, chain_index):
        a = scorer.score(request(fill=MID), chain_index=chain_index)
        b = scorer.score(request(fill=WORST), chain_index=chain_index)
        assert a.digest() != b.digest()

    def test_the_snapshot_travels_with_the_result(self, scorer, chain_index):
        assert scorer.score(request(), chain_index=chain_index).snapshot_hash == "snap-test"


class TestScoreCalendarStrikes:
    """The alternative-strike ladder `score_calendar` can emit."""

    def test_offsets_are_symmetric_around_atm(self):
        # The construction `score_calendar` uses, checked in isolation so the
        # ladder's shape does not depend on a full board run.
        alt = 2
        offsets = [None] + [
            step * sign
            for step in (0.025 * k for k in range(1, alt + 1))
            for sign in (-1.0, 1.0)
        ]
        assert offsets[0] is None
        assert offsets[1:] == [-0.025, 0.025, -0.05, 0.05]

    def test_default_is_atm_only(self):
        alt = 0
        offsets = [None] + [
            step * sign
            for step in (0.025 * k for k in range(1, alt + 1))
            for sign in (-1.0, 1.0)
        ]
        assert offsets == [None]


class TestScoreResult:
    def test_serializes_dates_as_strings(self, scorer, chain_index):
        doc = scorer.score(request(), chain_index=chain_index).as_dict()
        assert doc["event_date"] == "2024-05-02"
        assert doc["fill"] == 0.5

    def test_flag_is_idempotent(self):
        result = ScoreResult(ticker="X", strategy="STR-THRU", as_of=EVENT)
        result.flag("NO_CHAIN")
        result.flag("NO_CHAIN")
        assert result.flags == ["NO_CHAIN"]

    def test_request_key_distinguishes_requests(self):
        assert request().key() != request(fill=WORST).key()
        assert request().key() == request().key()

    def test_every_flag_used_is_declared(self, scorer, chain_index):
        result = scorer.score(request(), chain_index=chain_index)
        for flag in result.flags:
            assert flag in score_mod.FLAGS


class TestStructureParams:
    """A structure whose SHAPE varies per event, priced through the live scorer.

    Until this existed the scorer could only price a strategy's default
    parameterisation, so every variant — a different back DTE, a shifted
    anchor, a per-event tent width — had to be a separate replay and could
    never appear on the board.
    """

    def test_params_reach_the_legs(self, scorer, chain_index):
        from engine.structures import twin_peak

        narrow = scorer._structure(request(strategy="CAL-P"))
        assert narrow.params.get("back_dte") == 20          # the default
        wide = scorer._structure(
            request(strategy="CAL-P", structure_params={"back_dte": 45}))
        assert wide.params.get("back_dte") == 45

    def test_an_unknown_parameter_raises_rather_than_being_ignored(self, scorer):
        """A structure quietly priced at its default while the row claims
        otherwise is the failure this must not have."""
        with pytest.raises(ValueError, match="not accepted by its factory"):
            scorer._structure(request(strategy="CAL-P",
                                      structure_params={"nonsense": 1}))

    def test_a_bad_value_still_raises_from_the_factory(self, scorer):
        with pytest.raises(Exception):
            scorer._structure(request(strategy="TWIN-P",
                                      structure_params={"width_moneyness": -1.0}))

    def test_different_params_are_different_trades(self):
        """They must not share an identity — or a deterministic bootstrap seed."""
        a = ScoreRequest(ticker="X", strategy="TWIN-P",
                         structure_params={"width_moneyness": 0.03})
        b = ScoreRequest(ticker="X", strategy="TWIN-P",
                         structure_params={"width_moneyness": 0.05})
        none = ScoreRequest(ticker="X", strategy="TWIN-P")
        assert len({a.key(), b.key(), none.key()}) == 3

    def test_the_key_does_not_depend_on_dict_ordering(self):
        a = ScoreRequest(ticker="X", strategy="TWIN-P",
                         structure_params={"steps": 1, "anchor_offset": 2})
        b = ScoreRequest(ticker="X", strategy="TWIN-P",
                         structure_params={"anchor_offset": 2, "steps": 1})
        assert a.key() == b.key()

    def test_no_params_is_unchanged(self, scorer):
        """The nightly path passes nothing, and must behave exactly as before."""
        assert (scorer._structure(request(strategy="STR-THRU")).to_dict()
                == scorer._structure(
                    request(strategy="STR-THRU", structure_params=None)).to_dict())


# --------------------------------------------------------------------------
# forecast-sized structures and arithmetic entry rules (Tier 4, steps 4-6)
# --------------------------------------------------------------------------

#: A one-point ladder, so TWIN-P's seven mirrored strikes are all listed.
LADDER = tuple(float(k) for k in range(60, 141))


@pytest.fixture
def dense_chain():
    """A dense ladder with CONVEX put prices.

    The convexity is load-bearing. TWIN-P's net contracts sum to zero and its
    strikes are symmetric about the anchor, so under a price that is LINEAR in
    strike every term cancels and the debit is exactly zero — which is the
    structure's own tail-cancelling property, and which would make `cost < w`
    vacuously true. A quadratic price is the smallest thing that gives the
    trade a debit at all.
    """
    rows = []
    for strike in LADDER:
        mid = 100.0 * (strike / 100.0) ** 2
        for right in ("C", "P"):
            rows.append(
                {
                    "ticker": TICKER, "obs_date": ENTRY,
                    "expiry": pd.Timestamp("2024-05-03"), "dte": 1,
                    "strike": strike, "right": right,
                    "bid": mid - 0.05, "ask": mid + 0.05, "spot": 100.0,
                    "quote_repaired": False,
                }
            )
    entry = pd.DataFrame(rows)
    exit_rows = entry.copy()
    exit_rows["obs_date"] = EXIT
    return ChainIndex({(TICKER, ENTRY): entry, (TICKER, EXIT): exit_rows})


@pytest.fixture
def forecast(monkeypatch):
    """Inject a fold model instead of fitting one — the VALUE is the subject.

    Returns a mutable dict so a test can change the forecast and re-score,
    which is the only way to assert that the shape follows the forecast rather
    than being fixed by the strategy.
    """
    from engine.data.features import tier4 as tier4_mod

    state = {"value": 6.0}

    class Fake:
        model_id = "size_fake"
        fold_start = pd.Timestamp("2024-05-01")
        tier3_snapshot = "snap-test"
        features = ("mean_prior_abs_move",)

        def predict(self, frame):
            return np.full(len(frame), state["value"])

        def interval(self, predictions):
            values = np.asarray(predictions, dtype=float)
            if state.get("band") is False:  # the fold has no residual pool yet
                nan = np.full(values.shape, np.nan)
                return nan, nan, nan, nan
            return (
                np.maximum(values - 2.0, 0.0),
                values + 3.0,
                np.full(values.shape, 1.5),
                np.full(values.shape, 900.0),
            )

    monkeypatch.setattr(tier4_mod, "serving_model", lambda *a, **k: Fake())
    return state


def twin(**kwargs) -> ScoreRequest:
    return request(strategy="TWIN-P", **kwargs)


class TestForecastSizedStructures:
    """The forecast sets the shape, BEFORE anything is priced.

    The ordering is the whole point of Tier 4. `_features` reads `entry_cost`,
    `spot` and `dte_entry`; pricing needs the strikes; the strikes need `w`;
    `w` needs the forecast. That is a cycle unless the forecast comes from
    features that touch no chain, which is exactly what the size model's
    fourteen inputs are. These tests fix that ordering in place.
    """

    @pytest.fixture(autouse=True)
    def _twin_p_is_champion(self, monkeypatch):
        """These tests are about MECHANICS, not about which shape is live.

        TWIN-P was superseded by TWIN-P5 on 2026-09-04, so the scorer now
        returns SUPERSEDED for it before pricing anything. Pinning the family
        champion back keeps these covering the seven-strike geometry they were
        written against; the supersede path has its own class below.
        """
        monkeypatch.setattr("engine.score.superseded_by", lambda strategy: None)

    def test_the_forecast_sets_the_width_before_pricing(
        self, scorer, dense_chain, forecast
    ):
        # 6.0% forecast / 1.5 (the plateau centre) = 4.0% of a 100 spot.
        result = scorer.score(twin(), chain_index=dense_chain)
        assert result.forecast_abs_move == pytest.approx(6.0)
        assert result.forecast_model == "size_fake"
        assert result.structure_params["width_moneyness"] == pytest.approx(0.04)
        assert result.structure_width == pytest.approx(4.0)
        assert result.entry_cost is not None and result.entry_cost > 0

    def test_a_different_forecast_produces_a_different_structure(
        self, scorer, dense_chain, forecast
    ):
        narrow = scorer.score(twin(), chain_index=dense_chain)
        forecast["value"] = 12.0
        wide = scorer.score(twin(), chain_index=dense_chain)
        assert narrow.structure_width == pytest.approx(4.0)
        assert wide.structure_width == pytest.approx(8.0)
        # Two events priced at different shapes are different TRADES, so they
        # must not share an identity.
        assert wide.digest() != narrow.digest()

    def test_the_width_is_read_off_the_listed_strikes_not_the_request(
        self, scorer, dense_chain, forecast
    ):
        # 6.3% / 1.5 asks for a $4.20 tent; the ladder lists whole points, so
        # the trade that EXISTS is $4 wide. A rule testing the requested width
        # would be testing a trade nobody can place — and on a coarse real
        # ladder the gap is far larger than this.
        forecast["value"] = 6.3
        result = scorer.score(twin(), chain_index=dense_chain)
        assert result.structure_params["width_moneyness"] == pytest.approx(0.042)
        assert result.structure_width == pytest.approx(4.0)

    def test_no_forecast_means_no_structure_and_no_price(
        self, scorer, dense_chain, forecast
    ):
        # Declining is correct, not degraded. Pricing the factory's DEFAULT
        # width instead would put a real number on the board for a trade nobody
        # chose, on a row claiming to be forecast-sized.
        forecast["value"] = float("nan")
        result = scorer.score(twin(), chain_index=dense_chain)
        assert "NO_FORECAST" in result.flags
        assert result.entry_cost is None
        assert result.structure_width is None
        assert result.gate_pass is None

    def test_an_absurd_forecast_is_refused_rather_than_clipped(
        self, scorer, dense_chain, forecast
    ):
        forecast["value"] = 400.0
        result = scorer.score(twin(), chain_index=dense_chain)
        assert "NO_FORECAST" in result.flags
        assert result.entry_cost is None

    def test_a_caller_supplied_shape_wins_over_the_forecast(
        self, scorer, dense_chain, forecast
    ):
        # An explicit `structure_params` is a deliberate override — an
        # experiment replaying one width across many events — and must not be
        # silently replaced by whatever the model happens to say today.
        result = scorer.score(
            twin(structure_params={"width_moneyness": 0.08}), chain_index=dense_chain
        )
        assert result.forecast_abs_move is None
        assert result.structure_width == pytest.approx(8.0)


class TestArithmeticEntryRule:
    """A strategy with no registered gate is decided by its rule, or not at all."""

    @pytest.fixture(autouse=True)
    def _twin_p_is_champion(self, monkeypatch):
        """These tests are about MECHANICS, not about which shape is live.

        TWIN-P was superseded by TWIN-P5 on 2026-09-04, so the scorer now
        returns SUPERSEDED for it before pricing anything. Pinning the family
        champion back keeps these covering the seven-strike geometry they were
        written against; the supersede path has its own class below.
        """
        monkeypatch.setattr("engine.score.superseded_by", lambda strategy: None)

    def test_the_rule_decides_and_labels_itself(self, scorer, dense_chain, forecast):
        result = scorer.score(twin(), chain_index=dense_chain)
        assert result.model_versions.get("gate") == "entry-rule:TWIN-P"
        # A rule has a verdict, not a score. Reporting a number here would
        # invite a threshold comparison that means nothing.
        assert result.gate_score is None
        assert result.gate_threshold is None
        assert "TWIN-P entry rule" in (result.detail or "")

    def test_a_small_name_fails_the_rule_and_says_which_term(
        self, scorer, dense_chain, forecast
    ):
        # The fixture panel carries mcap 5e9, under the registered $10B floor.
        result = scorer.score(twin(), chain_index=dense_chain)
        assert result.gate_pass is False
        assert "market cap" in result.detail

    def test_a_strategy_with_neither_gate_nor_rule_gets_no_verdict(self, scorer):
        # The rule is a floor, never an override: `_apply_entry_rule` is only
        # reached when the registry has nothing, and it returns silently when
        # the strategy has no rule either.
        from engine.entry_rules import rule_for

        assert rule_for("STR-THRU") is None
        result = scorer.score(request(strategy="STR-THRU"))
        assert "entry-rule" not in str(result.model_versions.get("gate", ""))

    def test_an_unpriced_row_is_undetermined_rather_than_declined(
        self, scorer, forecast
    ):
        # No chain, so no cost and no width. "We could not tell" must not
        # render as a decision against the trade.
        result = scorer.score(twin())
        assert result.gate_pass is None
        assert "undetermined" in (result.detail or "")

    def test_rel_spread_is_the_mean_over_every_entry_leg(
        self, scorer, dense_chain, forecast
    ):
        result = scorer.score(twin(), chain_index=dense_chain)
        strikes = (100.0, 104.0, 108.0, 116.0, 96.0, 92.0, 84.0)
        expected = float(np.mean([0.10 / (100.0 * (k / 100.0) ** 2) for k in strikes]))
        assert result.rel_spread == pytest.approx(expected)


class TestTheForecastBandOnTheBoard:
    """A forecast without a width is half an answer; a fabricated one is worse."""

    @pytest.fixture(autouse=True)
    def _twin_p_is_champion(self, monkeypatch):
        """These tests are about MECHANICS, not about which shape is live.

        TWIN-P was superseded by TWIN-P5 on 2026-09-04, so the scorer now
        returns SUPERSEDED for it before pricing anything. Pinning the family
        champion back keeps these covering the seven-strike geometry they were
        written against; the supersede path has its own class below.
        """
        monkeypatch.setattr("engine.score.superseded_by", lambda strategy: None)

    def test_the_band_travels_with_the_forecast(self, scorer, dense_chain, forecast):
        result = scorer.score(twin(), chain_index=dense_chain)
        assert result.forecast_abs_move == pytest.approx(6.0)
        assert result.forecast_p10 == pytest.approx(4.0)
        assert result.forecast_p90 == pytest.approx(9.0)
        assert result.forecast_sd == pytest.approx(1.5)

    def test_the_band_is_ordered_around_the_point(self, scorer, dense_chain, forecast):
        result = scorer.score(twin(), chain_index=dense_chain)
        assert result.forecast_p10 <= result.forecast_abs_move <= result.forecast_p90

    def test_a_fold_with_no_pool_yields_no_band_but_still_a_forecast(
        self, scorer, dense_chain, forecast
    ):
        # Tier 4 unbuilt, or the earliest folds. The structure must still be
        # sized — the width comes from the point forecast, not from the band.
        forecast["band"] = False
        result = scorer.score(twin(), chain_index=dense_chain)
        assert result.forecast_abs_move == pytest.approx(6.0)
        assert result.structure_width == pytest.approx(4.0)
        assert result.forecast_p10 is None
        assert result.forecast_sd is None

    def test_the_band_reaches_the_rendered_board(self):
        from engine.dashboard.render import _BOARD_FIELDS

        for field in ("forecast_abs_move", "forecast_p10", "forecast_p90", "forecast_sd"):
            assert field in _BOARD_FIELDS


class TestStructureChampionOnTheBoard:
    """A superseded shape says so, and says what beat it.

    The distinction this class defends is the one that made the first attempt
    wrong: putting a beaten structure into `DISABLED_STRATEGIES` takes it off
    the board but stamps `UNVALIDATED_STRUCTURE`, which for TWIN-P — three
    completed experiments — reports the opposite of the truth. Superseded is a
    deployment decision about which of two validated shapes is live; disabled
    is a statement that nobody has shown a shape works at all.
    """

    def test_a_superseded_structure_is_not_scored_and_names_its_successor(
        self, scorer, dense_chain, forecast
    ):
        from engine.structure_registry import superseded_by

        assert superseded_by("TWIN-P").strategy == "TWIN-P5"
        result = scorer.score(twin(), chain_index=dense_chain)
        assert result.flags == ["SUPERSEDED"]
        assert not result.scored
        assert "TWIN-P5" in result.detail

    def test_superseded_is_not_the_unvalidated_flag(self, scorer, dense_chain, forecast):
        """The whole reason the flag exists. TWIN-P is the most-measured
        structure in the program; reporting it as unvalidated would misstate
        the evidence in order to state the decision."""
        result = scorer.score(twin(), chain_index=dense_chain)
        assert "UNVALIDATED_STRUCTURE" not in result.flags
        from engine.score import DISABLED_STRATEGIES

        assert "TWIN-P" not in DISABLED_STRATEGIES

    def test_the_champion_itself_is_scored_normally(self):
        from engine.structure_registry import superseded_by

        assert superseded_by("TWIN-P5") is None

    def test_a_structure_in_no_family_is_untouched(self):
        from engine.structure_registry import family_of, superseded_by

        assert family_of("STR-THRU") is None
        assert superseded_by("STR-THRU") is None

    def test_the_default_board_universe_drops_superseded_members(self):
        """`score_calendar` defaults to every registered structure. Without
        this filter the board would price both shapes of one idea on the same
        events — twice the position for one thesis."""
        from engine.structure_registry import live_strategies
        from engine.structures import STRUCTURES

        live = live_strategies(sorted(STRUCTURES))
        assert "TWIN-P5" in live
        assert "TWIN-P" not in live

    def test_the_beaten_shape_stays_in_the_registry(self):
        """Superseded, not deleted: EXP-123/124/125 replay against TWIN-P, and
        a promotion that cannot be rolled back is not a decision, it is a
        one-way door."""
        from engine.structures import STRUCTURES

        assert "TWIN-P" in STRUCTURES
