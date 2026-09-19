"""P5-4: the frozen board analog matcher (``engine.v2.models.analog_artifact``).

Claims, all on a synthetic full universe built through the REAL legacy code
(a legacy ``Scorer`` over a synthetic panel/daily/trades world):

1. the dataset reader (``tools.phase5_datasets.board_analog_trades``) gives
   exactly the analog population legacy ``Scorer.trades`` holds;
2. native-from-artifact equals legacy ``AnalogMatcher.match`` bit for bit --
   every summary field, the bootstrap interval included -- across strategies,
   alphas, cutoffs (none, empty slice, thin slice, full) and queries with
   missing dimensions, directly and through ``application.score_one``;
3. a context-scoped legacy Scorer (a narrower panel, as the nightly builds)
   gives a DIFFERENT matcher state and different answers, and the release
   pin refuses it: the context-width defect cannot happen on the frozen path;
4. a missing or mismatched artifact is MODEL_NOT_READY, never a rebuild;
5. the verified loader, release member, lineage, no-fit guard, the training
   job (``--state board_analog_matcher``) and the P5-6 consumer probe.

Everything lives under ``tmp_path``; nothing reads or writes ``data/``.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.calendar import TradingCalendar
from engine.features import FeatureContext
from engine.models.registry import Registry
from engine.v2.contracts import ScoreRequest
from engine.v2.models.analog_artifact import (
    AnalogArtifactError,
    BoardAnalogPoolArtifact,
    analog_artifact_from_document,
)
from engine.v2.models.frozen_release import member_kind, release_member
from engine.v2.models.frozen_state import (
    FrozenStateError,
    FrozenStateLoader,
    FrozenStateRef,
    serialize_frozen_state,
)
from engine.v2.models.lineage import DataDependency, Lineage, propagate_corrections, state_node
from engine.v2.models.no_fit import RuntimeFitForbidden, no_fit_guard
from engine.v2.models.training.analogs import (
    build_board_analog_pool_artifact,
    iter_board_analog_pool_artifacts,
    population_implied_edges,
)
from engine.v2.scoring import application
from engine.v2.scoring.native_analog import evaluate_frozen_analogs
from engine.v2.scoring.source_inputs import SourceBundle, build_native_score_inputs
from tools import phase5_datasets as data
from tools import phase5_training_job as job

TICKERS = [f"T{i}" for i in range(16)]
EVENTS = pd.date_range("2019-01-15", periods=24, freq="47D")
STRATEGIES = ("STR-THRU", "BFLY-P")
ALPHAS = (0.5, 0.25)
SNAPSHOT = "snap-analog"
#: none, before any exit (empty slice), early (thin: < 30 finite ratios),
#: mid, and after every exit.
CUTOFFS = (None, "2018-06-01", "2019-03-20", "2020-09-01", "2023-12-31")
ANALOG_COLUMNS = list(data.BOARD_ANALOG_COLUMNS)


# --------------------------------------------------------------------------
# the synthetic world
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(31)
    caps = rng.choice([4e8, 5e9, 6e10], size=len(TICKERS))
    rows = []
    for t, ticker in enumerate(TICKERS):
        for k, date in enumerate(EVENTS):
            rows.append({"ticker": ticker, "date": date, "year": date.year, "n_prior": k + 4,
                         "abs_move": float(rng.uniform(0.5, 12.0)),
                         "or_implied": float(rng.uniform(2.0, 10.0)),
                         "mean_prior_or_implied": float(rng.uniform(3.0, 8.0)),
                         "mcap_usd": float(caps[t])})
    frame = pd.DataFrame(rows)
    frame.loc[5, "mcap_usd"] = np.nan              # an unmatchable market cap
    frame.loc[9, "mean_prior_or_implied"] = 0.0    # ratio -> NaN (prior 0)
    return frame


@pytest.fixture(scope="module")
def daily():
    dates = pd.bdate_range("2018-12-01", "2022-06-30")
    rng = np.random.default_rng(32)
    frames = []
    for ticker in TICKERS[:-2]:                   # two tickers: no entry-date reading
        n = len(dates)
        frames.append(pd.DataFrame({
            "ticker": ticker, "date": dates, "src_iv": "orats",
            "implied_move": rng.uniform(2.0, 10.0, n),
            "iv10": rng.uniform(20, 60, n), "iv30": rng.uniform(20, 60, n),
            "exern_iv10": 30.0, "exern_iv30": 28.0, "iee": 1.2, "skew": 0.3,
            "contango": 0.1, "fwd90_30": 32.0, "fexern90_30": 30.0, "rvol30": 38.0,
            "spot": 100.0, "mcap_log": np.log(5e9)}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def trades(panel):
    """Tier-2-shaped trades with legs: two strategies x two alphas, varied
    DTE/moneyness, a few non-finite returns and non-replay rows."""
    rng = np.random.default_rng(33)
    rows = []
    for i, event in enumerate(panel.itertuples(index=False)):
        for strategy in STRATEGIES:
            for alpha in ALPHAS:
                spot = float(rng.uniform(20, 200))
                dte = int(rng.choice([2, 2, 3, 7, 15]))
                legs = {"spot_entry": spot, "dte_entry": dte,
                        "spot_exit": spot * float(rng.uniform(0.8, 1.2)), "entry": [], "exit": []}
                exit_days = int(rng.integers(1, 20))
                rows.append({
                    "trade_id": f"{event.ticker}:{event.date.date()}:{strategy}:{alpha}",
                    "kind": "sim", "strategy": strategy, "variant": "v",
                    "ticker": event.ticker, "event_id": f"{event.ticker}_{event.date.date()}",
                    "event_date": event.date, "year": event.date.year,
                    "legs": json.dumps(legs), "entry_date": event.date - pd.Timedelta(days=1),
                    "exit_date": event.date + pd.Timedelta(days=exit_days),
                    "strike": spot * float(rng.choice([1.0, 1.0, 1.03, 1.08])),
                    "expiry": event.date + pd.Timedelta(days=exit_days),
                    "fill_alpha": alpha, "entry_cost": 3.0, "exit_value": 1.0,
                    "ret": float(rng.normal(0.02, 0.4)),
                    "provenance": "engine.replay" if i % 23 else "paper.log",
                })
    frame = pd.DataFrame(rows)
    frame.loc[[4, 40], "ret"] = np.nan
    frame.loc[77, "ret"] = np.inf
    return frame


@pytest.fixture(scope="module")
def calendar():
    return TradingCalendar(pd.bdate_range("2018-01-01", "2024-12-31"))


def _legacy_scorer(trades, panel, daily, calendar):
    from engine.score import Scorer

    context = FeatureContext(panel=panel, daily=daily, calendar=calendar)
    return Scorer(registry=Registry(entries=[]), trades=trades, context=context,
                  snapshot=SNAPSHOT, analog_daily=daily)


@pytest.fixture(scope="module")
def legacy(trades, panel, daily, calendar):
    """The full-universe legacy Scorer (its matcher is the parity oracle)."""
    return _legacy_scorer(trades, panel, daily, calendar)


@pytest.fixture(scope="module")
def scoped(trades, panel, daily, calendar):
    """A bounded legacy Scorer: its context loaded four tickers' panel rows."""
    return _legacy_scorer(trades, panel[panel["ticker"].isin(TICKERS[:4])], daily, calendar)


@pytest.fixture(scope="module")
def population(trades, panel, daily):
    """The analog population the training job reads."""
    return data.board_analog_trades(trades=trades, panel=panel, analog_daily=daily)


def _artifact(population, strategy="STR-THRU", alpha=0.5, cutoff="2020-09-01"):
    return build_board_analog_pool_artifact(population, strategy=strategy, alpha=alpha,
                                            cutoff=cutoff)


# --------------------------------------------------------------------------
# 1. the dataset is legacy's population
# --------------------------------------------------------------------------


def test_dataset_is_the_legacy_scorer_population(legacy, population):
    expected = legacy.trades[ANALOG_COLUMNS].reset_index(drop=True)
    pd.testing.assert_frame_equal(population.reset_index(drop=True), expected)
    assert population.attrs["implied_edges"] == legacy.matcher.implied_edges
    assert population_implied_edges(population) == legacy.matcher.implied_edges
    assert expected["mcap_bucket"].notna().sum() > 0.9 * len(expected)


# --------------------------------------------------------------------------
# 2. parity: native-from-artifact == legacy AnalogMatcher.match
# --------------------------------------------------------------------------

_FIELDS = (("n", "n_analogs"), ("mean", "exp_pnl_analog"), ("median", "median"),
           ("win_rate", "win_analog"), ("p10", "p10"), ("p90", "p90"),
           ("ci_low", "ci_low"), ("ci_high", "ci_high"), ("widened", "widened"),
           ("dropped", "dropped"), ("unavailable", "unavailable"), ("thin", "thin"))


def _queries(seed: int, n: int):
    rng = np.random.default_rng(seed)
    for index in range(n):
        yield {
            "mcap_usd": rng.choice([None, 4e8, 5e9, 6e10]),
            "dte": rng.choice([None, 2, 3, 7, 15, 60]),
            "moneyness_pct": rng.choice([None, 0.5, 3.0, 8.0]),
            "implied_ratio": rng.choice([None, 0.45, 0.9, 1.0, 1.3, 2.4]),
            "request_key": f"req-{seed}-{index}",
        }


def _legacy_match(matcher, strategy, alpha, cutoff, query, *, bootstrap, min_analogs):
    buckets = matcher.buckets_for(mcap_usd=query["mcap_usd"], dte=query["dte"],
                                  moneyness_pct=query["moneyness_pct"],
                                  implied_ratio=query["implied_ratio"])
    result = matcher.match(strategy, buckets, alpha=alpha, as_of=cutoff, bootstrap=bootstrap,
                           min_analogs=min_analogs, request_key=query["request_key"])
    return buckets, result


def _native_query(buckets, query) -> dict:
    return {"mcap_bucket": buckets["mcap_bucket"], "dte_band": buckets["dte_band"],
            "moneyness_band": buckets["moneyness_band"],
            "implied_ratio": query["implied_ratio"]}


def _recipe(cutoff, query=None, *, bootstrap=200, min_analogs=30, **extra) -> dict:
    recipe = {"cutoff": cutoff, "min_analogs": min_analogs, "bootstrap_draws": bootstrap,
              "ci_quantiles": (0.05, 0.95), "seed_snapshot": SNAPSHOT,
              "request_key": (query or {}).get("request_key", "req")}
    recipe.update(extra)
    return recipe


def _assert_legacy_unreachable(exc, buckets, native) -> None:
    """A recorded legacy defect, not a parity loosening.

    ``AnalogMatcher.match`` loops ``len(WIDENING_ORDER) + 1 - len(unavailable)``
    times, but ``unavailable`` may hold ``mcap_bucket``, which is not in the
    widening order: with no market cap AND a slice too thin to reach
    ``min_analogs`` after dropping every dimension, legacy runs out of
    iterations and raises ``AssertionError("unreachable")``. The native paths
    (the declared-rows recipe and this artifact) finish the ladder and
    answer with every dimension dropped. Only that exact case is allowed.
    """
    assert "unreachable" in str(exc)
    assert buckets["mcap_bucket"] is None
    assert set(native.dropped) == {"mcap_bucket", "moneyness_band", "dte_band",
                                   "implied_tercile"}


def _assert_same(legacy_set, native) -> None:
    for legacy_name, native_name in _FIELDS:
        mine, theirs = getattr(native, native_name), getattr(legacy_set, legacy_name)
        assert mine == theirs, (legacy_name, mine, theirs)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_native_from_artifact_equals_legacy_match_bit_for_bit(legacy, population, strategy,
                                                              alpha, cutoff):
    artifact = _artifact(population, strategy, alpha, cutoff)
    matched = 0
    for min_analogs in (30, 8):
        seed = sum(map(ord, f"{strategy}{alpha}{cutoff}{min_analogs}"))
        for query in _queries(seed, 40):
            buckets = legacy.matcher.buckets_for(
                mcap_usd=query["mcap_usd"], dte=query["dte"],
                moneyness_pct=query["moneyness_pct"], implied_ratio=query["implied_ratio"])
            native, flag = evaluate_frozen_analogs(
                artifact=artifact,
                recipe=_recipe(cutoff, query, min_analogs=min_analogs),
                query_features=_native_query(buckets, query), strategy=strategy, alpha=alpha)
            assert flag is None
            try:
                _, expected = _legacy_match(legacy.matcher, strategy, alpha, cutoff, query,
                                            bootstrap=200, min_analogs=min_analogs)
            except AssertionError as exc:
                _assert_legacy_unreachable(exc, buckets, native)
                continue
            _assert_same(expected, native)
            matched += expected.n > 0
    if cutoff != "2018-06-01":
        assert matched > 0
    else:
        assert not artifact.rows and artifact.causal_edges is None


def test_legacy_unreachable_case_is_the_only_recorded_difference(legacy, population):
    """No market cap, and a ladder that cannot reach ``min_analogs``: legacy
    raises, native answers on the fully-widened slice (every row)."""
    query = {"mcap_usd": None, "dte": 60, "moneyness_pct": 8.0, "implied_ratio": 2.4,
             "request_key": "unreachable"}
    buckets = legacy.matcher.buckets_for(mcap_usd=None, dte=60, moneyness_pct=8.0,
                                         implied_ratio=2.4)
    with pytest.raises(AssertionError, match="unreachable"):
        _legacy_match(legacy.matcher, "STR-THRU", 0.5, "2020-09-01", query,
                      bootstrap=0, min_analogs=100_000)
    artifact = _artifact(population)
    native, flag = evaluate_frozen_analogs(
        artifact=artifact, recipe=_recipe("2020-09-01", query, bootstrap=0,
                                          min_analogs=100_000),
        query_features=_native_query(buckets, query), strategy="STR-THRU", alpha=0.5)
    assert flag is None and native.thin
    assert native.n_analogs == sum(row[7] is not None for row in artifact.rows)
    _assert_legacy_unreachable(AssertionError("unreachable"), buckets, native)


def test_planted_defect_in_the_frozen_rows_breaks_parity(legacy, population):
    """The comparator is live: one changed return in the artifact moves the
    native answer away from legacy's."""
    from engine.v2.models.analog_artifact import make_board_analog_pool_artifact

    cutoff = "2020-09-01"
    artifact = _artifact(population, cutoff=cutoff)
    rows = [list(row) for row in artifact.rows]
    first = next(i for i, row in enumerate(rows) if row[7] is not None and row[3] is not None)
    rows[first][7] += 1.0
    planted = make_board_analog_pool_artifact(
        strategy="STR-THRU", alpha=0.5, cutoff=cutoff,
        population_edges=artifact.population_edges, causal_edges=artifact.causal_edges,
        rows=rows, lineage=artifact.lineage)
    # Market cap is never dropped; an unreachable min_analogs widens every
    # other dimension away, so the planted row is in the selected set.
    cap = {"<1B": 4e8, "1-10B": 5e9, ">=10B": 6e10}[rows[first][3]]
    query = {"mcap_usd": cap, "dte": 2, "moneyness_pct": 0.5, "implied_ratio": 1.0,
             "request_key": "planted"}
    buckets, expected = _legacy_match(legacy.matcher, "STR-THRU", 0.5, cutoff, query,
                                      bootstrap=0, min_analogs=100_000)
    for item, should_match in ((artifact, True), (planted, False)):
        native, _ = evaluate_frozen_analogs(
            artifact=item, recipe=_recipe(cutoff, query, bootstrap=0, min_analogs=100_000),
            query_features=_native_query(buckets, query), strategy="STR-THRU", alpha=0.5)
        if should_match:
            _assert_same(expected, native)
        else:
            with pytest.raises(AssertionError):
                _assert_same(expected, native)


def test_the_fixture_exercises_every_legacy_branch(population):
    thin = _artifact(population, cutoff="2019-03-20")
    assert 0 < len(thin.rows) and thin.causal_edges == (0.9, 1.1)   # < 30 finite ratios
    mid = _artifact(population, cutoff="2020-09-01")
    assert mid.causal_edges != (0.9, 1.1) and mid.causal_edges != mid.population_edges
    full = _artifact(population, cutoff="2023-12-31")
    assert any(row[7] is None for row in _artifact(population, cutoff=None).rows)
    assert any(row[3] is None for row in full.rows)                 # no market cap
    assert any(row[6] is None for row in full.rows)                 # no implied ratio


# --------------------------------------------------------------------------
# through the scoring stage
# --------------------------------------------------------------------------

_QUOTES = {("C", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 3.0},
           ("P", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 3.0}}


def _request(alpha=0.5, strategy="STR-THRU") -> ScoreRequest:
    return ScoreRequest(
        event_id="evt-analog", calendar_revision="cal-1", strategy_version=strategy,
        deployment_id="dep-1", decision_clock_id="entry-close",
        requested_decision_at="2026-09-16", snapshot_id="snap-1", mode="replay",
        fill_model={"alpha": alpha},
    )


def _bundle(**overrides) -> SourceBundle:
    base: dict = dict(
        source_ref="board-analog-bundle",
        context={"ticker": "AAA", "event_date": "2026-09-16", "entry_date": "2026-09-16",
                 "exit_date": "2026-09-17", "expiry": "2026-09-18", "spot": 100.0},
        raw_quotes=_QUOTES, feature_vector={}, feature_missing_mask={},
        model_identity={"driver": {"model_id": "m1"}},
        forecast_recipes={"driver_prediction": {"intercept": 1.0, "coefficients": {}}},
        model_artifact_refs={"driver_prediction": "sha256:m1"},
        residual_recipe={}, analog_recipe={},
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}}, "threshold": 0.0},
    )
    base.update(overrides)
    return SourceBundle(**base)


_OUTPUTS = ("exp_pnl_analog", "win_analog", "ci_low", "ci_high", "n_analogs")


def _outputs(record) -> tuple:
    return tuple(record.resolved_request.get(name) for name in _OUTPUTS)


def _frozen_bundle(artifact, query, cutoff, **recipe_extra):
    return _bundle(analog_query=query, analog_artifact=artifact,
                   analog_artifact_recipe=_recipe(cutoff, bootstrap=2000, **recipe_extra))


def test_score_one_from_the_frozen_artifact_equals_legacy(legacy, population):
    cutoff = "2020-09-01"
    artifact = _artifact(population, cutoff=cutoff)
    query = {"mcap_usd": 5e9, "dte": 2, "moneyness_pct": 0.5, "implied_ratio": 1.0,
             "request_key": "req"}
    buckets, expected = _legacy_match(legacy.matcher, "STR-THRU", 0.5, cutoff, query,
                                      bootstrap=2000, min_analogs=30)
    record = application.score_one(_request(), build_native_score_inputs(
        _frozen_bundle(artifact, _native_query(buckets, query), cutoff,
                       content_hash=artifact.content_hash)))
    assert "MODEL_NOT_READY" not in record.reason_codes
    assert expected.n >= 30 and expected.ci_low is not None
    assert _outputs(record) == (expected.mean, expected.win_rate, expected.ci_low,
                                expected.ci_high, expected.n)


def test_analog_receipt_binds_the_artifact_identity(population):
    universe = _artifact(population)
    other = _artifact(population.iloc[1:])
    receipts = []
    for artifact in (universe, other, universe):
        record = application.score_one(_request(), build_native_score_inputs(_bundle(
            analog_query=_query(), analog_artifact=artifact,
            analog_artifact_recipe=_recipe("2020-09-01", bootstrap=0))))
        receipts.append({item["stage"]: item["input_hash"]
                         for item in record.resolved_request["native_stage_receipts"]})
    assert receipts[0]["analogs"] == receipts[2]["analogs"]
    assert receipts[0]["analogs"] != receipts[1]["analogs"]


def _query():
    return {"mcap_bucket": "1-10B", "dte_band": "1-3", "moneyness_band": "ATM",
            "implied_ratio": 1.0}


@pytest.mark.parametrize("change", [
    {"cutoff": "2020-09-02"},                      # a later (leaked) cutoff
    {"cutoff": "2020-08-31"},
    {"content_hash": "sha256:" + "0" * 64},        # a different pinned release state
])
def test_any_key_or_pin_disagreement_is_model_not_ready(population, change):
    artifact = _artifact(population)
    recipe = {**_recipe("2020-09-01", bootstrap=0), **change}
    record = application.score_one(_request(), build_native_score_inputs(_bundle(
        analog_query=_query(), analog_artifact=artifact, analog_artifact_recipe=recipe)))
    assert "MODEL_NOT_READY" in record.reason_codes
    assert record.resolved_request.get("n_analogs") is None


def test_wrong_alpha_strategy_or_missing_artifact_is_model_not_ready(population):
    artifact = _artifact(population)
    recipe = _recipe("2020-09-01", bootstrap=0)
    cases = [
        (_request(alpha=0.25), artifact),                       # alpha
        (_request(), _artifact(population, strategy="BFLY-P")),  # strategy
        (_request(), None),                                      # declared, unresolved
    ]
    for request, item in cases:
        record = application.score_one(request, build_native_score_inputs(_bundle(
            analog_query=_query(), analog_artifact=item, analog_artifact_recipe=recipe)))
        assert "MODEL_NOT_READY" in record.reason_codes
        assert record.resolved_request.get("n_analogs") is None
    missing_cutoff = {k: v for k, v in recipe.items() if k != "cutoff"}
    result, flag = evaluate_frozen_analogs(artifact=artifact, recipe=missing_cutoff,
                                           query_features=_query(), strategy="STR-THRU",
                                           alpha=0.5)
    assert (result, flag) == (None, "MODEL_NOT_READY")


def test_frozen_and_declared_rows_paths_cannot_combine(population):
    artifact = _artifact(population)
    with pytest.raises(ValueError, match="cannot combine"):
        build_native_score_inputs(_bundle(
            analog_query=_query(), analog_artifact=artifact,
            analog_artifact_recipe=_recipe("2020-09-01"),
            analog_source_rows=({"row_id": "r", "mcap_bucket": "1-10B"},)))
    with pytest.raises(ValueError, match="BoardAnalogPoolArtifact"):
        build_native_score_inputs(_bundle(analog_artifact=object(),
                                          analog_artifact_recipe=_recipe("2020-09-01")))
    with pytest.raises(ValueError, match="unsupported recipe fields"):
        build_native_score_inputs(_bundle(analog_artifact=artifact,
                                          analog_artifact_recipe={"population_hash": "x"}))


def test_undeclared_bundle_analog_block_is_unchanged():
    inputs = build_native_score_inputs(_bundle())
    assert inputs.analogs == {"recipe": None}


# --------------------------------------------------------------------------
# 3. the context-width defect is impossible on the frozen path
# --------------------------------------------------------------------------


def test_context_scoped_legacy_state_differs_and_the_release_pin_refuses_it(
        legacy, scoped, population):
    cutoff = "2020-09-01"
    universe = _artifact(population, cutoff=cutoff)
    scoped_population = scoped.trades[ANALOG_COLUMNS]
    narrow = build_board_analog_pool_artifact(scoped_population, strategy="STR-THRU",
                                              alpha=0.5, cutoff=cutoff)
    # Same causal key, different content: the scoped Scorer lost the market
    # cap and implied ratio of every ticker outside its four.
    assert narrow.key == universe.key
    assert narrow.content_hash != universe.content_hash
    assert (sum(row[3] is None for row in narrow.rows)
            > sum(row[3] is None for row in universe.rows))

    # ... and legacy's answers really move with the context (the defect).
    moved = 0
    for query in _queries(7, 60):
        if query["mcap_usd"] is None:
            continue  # the recorded legacy "unreachable" case; see _assert_legacy_unreachable
        _, full = _legacy_match(legacy.matcher, "STR-THRU", 0.5, cutoff, query,
                                bootstrap=0, min_analogs=30)
        _, bounded = _legacy_match(scoped.matcher, "STR-THRU", 0.5, cutoff, query,
                                   bootstrap=0, min_analogs=30)
        moved += (full.n, full.mean) != (bounded.n, bounded.mean)
    assert moved > 0

    pinned = _recipe(cutoff, bootstrap=0, content_hash=universe.content_hash)
    ok = application.score_one(_request(), build_native_score_inputs(_bundle(
        analog_query=_query(), analog_artifact=universe, analog_artifact_recipe=pinned)))
    refused = application.score_one(_request(), build_native_score_inputs(_bundle(
        analog_query=_query(), analog_artifact=narrow, analog_artifact_recipe=pinned)))
    assert "MODEL_NOT_READY" not in ok.reason_codes and ok.resolved_request["n_analogs"] > 0
    assert "MODEL_NOT_READY" in refused.reason_codes
    assert refused.resolved_request.get("n_analogs") is None


def test_request_context_cannot_change_the_frozen_state(population):
    """The builder sees only the rows it is given; row order and any request
    never enter the state, and scoring reads the artifact as-is."""
    artifact = _artifact(population)
    shuffled = population.sample(frac=1.0, random_state=5)
    shuffled.attrs = dict(population.attrs)
    assert _artifact(shuffled).content_hash == artifact.content_hash
    first = application.score_one(_request(), build_native_score_inputs(_bundle(
        analog_query=_query(), analog_artifact=artifact,
        analog_artifact_recipe=_recipe("2020-09-01", bootstrap=0))))
    second = application.score_one(_request(), build_native_score_inputs(_bundle(
        context={"ticker": "ZZZ", "event_date": "2026-09-16", "entry_date": "2026-09-16",
                 "exit_date": "2026-09-17", "expiry": "2026-09-18", "spot": 100.0},
        analog_query=_query(), analog_artifact=artifact,
        analog_artifact_recipe=_recipe("2020-09-01", bootstrap=0))))
    assert _outputs(first) == _outputs(second) and first.resolved_request["n_analogs"] > 0


# --------------------------------------------------------------------------
# 5. loader, release member, lineage, no-fit
# --------------------------------------------------------------------------


def test_verified_loader_round_trip_and_corruption(tmp_path, population):
    artifact = _artifact(population)
    payload = serialize_frozen_state(artifact)
    (tmp_path / "a.json").write_bytes(payload)
    ref = FrozenStateRef(path="a.json", content_hash=artifact.content_hash)
    loaded = FrozenStateLoader(tmp_path).load(ref)
    assert isinstance(loaded, BoardAnalogPoolArtifact) and loaded == artifact
    assert member_kind(loaded) == "residual"
    assert release_member(loaded, "board_analog_matcher").content_hash == artifact.content_hash

    document = json.loads(payload)
    document["rows"][0][7] = 99.0
    (tmp_path / "b.json").write_text(json.dumps(document))
    with pytest.raises(FrozenStateError, match="hash mismatch"):
        FrozenStateLoader(tmp_path).load(FrozenStateRef(path="b.json",
                                                        content_hash=artifact.content_hash))
    later = dict(json.loads(payload), cutoff="2020-01-01")   # rows closing after it
    with pytest.raises(AnalogArtifactError, match="on/after its own cutoff"):
        analog_artifact_from_document(later)


def test_lineage_is_declared_hashed_and_propagates(population):
    artifact = _artifact(population)
    assert artifact.lineage.declared
    other = build_board_analog_pool_artifact(
        population, strategy="STR-THRU", alpha=0.5, cutoff="2020-09-01",
        lineage=Lineage(data=(DataDependency(table="tier2.trades"),)))
    assert other.content_hash != artifact.content_hash
    report = propagate_corrections([state_node("board_analog_matcher/x", artifact)], ())
    assert report.valid == ("board_analog_matcher/x",)
    with pytest.raises(ValueError, match="lineage"):
        build_board_analog_pool_artifact(population, strategy="STR-THRU", alpha=0.5,
                                         cutoff=None, lineage=Lineage())


def test_builder_is_rigged_and_frozen_scoring_is_not(population):
    artifact = _artifact(population)
    with no_fit_guard():
        with pytest.raises(RuntimeFitForbidden):
            _artifact(population)
        with pytest.raises(RuntimeFitForbidden):
            next(iter_board_analog_pool_artifacts(population, [("STR-THRU", 0.5, None)]))
        record = application.score_one(_request(), build_native_score_inputs(_bundle(
            analog_query=_query(), analog_artifact=artifact,
            analog_artifact_recipe=_recipe("2020-09-01", bootstrap=0))))
    assert "MODEL_NOT_READY" not in record.reason_codes
    assert record.resolved_request["n_analogs"] > 0


# --------------------------------------------------------------------------
# the training job and the P5-6 consumer probe
# --------------------------------------------------------------------------


def test_training_job_plan_only_then_writes_the_direct_builder_bytes(tmp_path, population):
    out = tmp_path / "analog"
    plan = job.run_board_analog_job(out, alpha=0.5, cutoffs=["2020-09-01", "2021-03-01"],
                                    plan_only=True, trades=population)
    assert plan["status"] == "planned" and len(plan["keys"]) == 4
    assert plan["rss"]["estimated_peak_gb"] > 0 and not list(out.glob("board_*__*.json"))

    summary = job.run_board_analog_job(out, alpha=0.5, cutoffs=["2020-09-01", "2021-03-01"],
                                       strategies=["STR-THRU"], trades=population)
    assert summary["status"] == "written" and len(summary["files"]) == 2
    for item in summary["files"]:
        strategy, alpha, cutoff = item["key"]
        direct = build_board_analog_pool_artifact(population, strategy=strategy, alpha=alpha,
                                                  cutoff=cutoff)
        assert (out / item["file"]).read_bytes() == serialize_frozen_state(direct)
    again = job.run_board_analog_job(out, alpha=0.5, cutoffs=["2020-09-01", "2021-03-01"],
                                     strategies=["STR-THRU"], trades=population)
    assert again["status"] == "resumed"

    from tools import phase5_prepare_release as prep

    found = prep.frozen_state_payloads([out])
    assert sorted(found["board_analog_matcher"]) == [
        "STR-THRU|0.5000|2020-09-01", "STR-THRU|0.5000|2021-03-01"]


def test_training_job_cli_requires_alpha_and_cutoff(tmp_path):
    with pytest.raises(SystemExit):
        job.main(["--state", "board_analog_matcher", "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit):
        job.main(["--state", "paired_residual_pool", "--strategy", "STR-THRU",
                  "--out", str(tmp_path / "y")])


def test_consumer_probe_resolves_and_refuses(population):
    from checks.phase5_consumers import CONSUMERS, ReleaseContext

    artifacts = [_artifact(population), _artifact(population, strategy="BFLY-P")]
    ctx = ReleaseContext(release_root=Path("."), model_release=None,
                         states={"board_analog_matcher": {"artifacts": artifacts}})
    rows = CONSUMERS["analogs.board_analog_matcher"](ctx)
    assert len(rows) == 2
    assert all(row["resolved"] and row["refused_when_missing"] for row in rows), rows
    blocked = CONSUMERS["analogs.board_analog_matcher"](dataclasses.replace(ctx, states={}))
    assert blocked == [{"consumer": "analogs.board_analog_matcher",
                        "member_id": "board_analog_matcher", "blocked": True}]
