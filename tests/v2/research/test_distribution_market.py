"""Synthetic native pricing and real CAS import contracts; no provider payloads."""
from copy import deepcopy

import pandas as pd
import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import (
    ArtifactError,
    ArtifactStore,
    canonical_json,
    from_document,
    to_document,
)
from engine.v2.models.payoff_artifact import make_payoff_line_artifact, serialize_payoff_artifact
from engine.v2.ops.errors import OpsError
from engine.v2.research._chains import ChainIndex
from engine.v2.research._pricing import ChainSnapshot, FillModel, TradingCalendar, price_structure
from experiments import distribution_market as market
from experiments.magnitude_search_shards import write_section


def economics(**changes):
    return dict(currency="USD", multiplier=100, commission=.65, exchange_fee=.05,
        minimum_side_fee=5, slippage_per_share_per_leg=.01, max_relative_spread=2,
        capacity_policy="not_established_no_depth", adjusted_contracts="unsupported_native_standard") | changes


def chain(day, spot, call, put):
    rows = []
    for strike in (100, 120):
        for right, spread in (("C", call), ("P", put)):
            bid, ask = spread if strike == 100 else (1, 2)
            rows.append(dict(expiry=pd.Timestamp("2026-01-09"), dte=(pd.Timestamp("2026-01-09") - pd.Timestamp(day)).days,
                strike=strike, right=right, bid=bid, ask=ask, spot=spot))
    return pd.DataFrame(rows)


@pytest.fixture
def quotes(monkeypatch):
    frames = {("SYN", pd.Timestamp(day)): chain(day, spot, call, put) for day, spot, call, put in [
        ("2026-01-05", 100, (4, 6), (3, 5)), ("2026-01-06", 120, (6, 8), (2, 4)),
        ("2026-01-07", 120, (7, 9), (1, 3))]}
    calls = []
    def load(repository, snapshot, keys):
        keys = tuple(keys)
        calls.append(keys)
        return ChainIndex({key: frames[key] for key in keys if key in frames})
    calendar = TradingCalendar(pd.bdate_range("2026-01-02", "2026-01-09"))
    monkeypatch.setattr(market, "trading_calendar_from_snapshot", lambda *args, **kwargs: calendar)
    monkeypatch.setattr(market, "load_chain_index", load)
    # Native SIZE target-event shape: session lives on each forecast, not event.
    event = dict(event_id="e", ticker="SYN", event_date="2026-01-06", forecasts=[
        dict(horizon_sessions=1, session="AMC", source_session_ceiling="2026-01-05")])
    return event, frames, calls


def test_native_context_session_and_decision_only_chain_ceiling(quotes):
    event, _, calls = quotes
    before = deepcopy(event)
    decisions = market.decision_quotes(None, None, [event], [.25], economics())[.25]
    row, decision = decisions["e"]
    assert "session" not in event and event == before
    assert row["session"] == "AMC"
    assert row["decision_date"] == pd.Timestamp("2026-01-05")
    assert calls == [(("SYN", pd.Timestamp("2026-01-05")),)]
    assert [leg.strike for leg in decision.legs] == [100, 100]
    assert decision.cost == 10


def test_actual_native_prices_keep_d_minus_one_pins_despite_changed_d0_atm(quotes):
    event, frames, _ = quotes
    decisions = market.decision_quotes(None, None, [event], [.25], economics())[.25]
    row, decision = decisions["e"]
    unpinned = price_structure(market.STRUCTURE, ChainSnapshot(ticker="SYN", obs_date=row["entry_date"],
        event_date=row["event_date"], session="AMC", rows=frames["SYN", row["entry_date"]]), FillModel(.25))
    assert [leg.strike for leg in unpinned.legs] == [120, 120]
    result = market.outcome_quotes(None, None, decisions, .25, economics(), 2)["e"]
    assert result == dict(entry_cash=2209, exit_cash=1791, exit_at="2026-01-07", contracts=2, multiplier=100)
    assert [leg.strike for leg in decision.legs] == [100, 100]


def test_missing_future_exit_preserves_decision_and_funded_entry_quote(quotes):
    event, frames, calls = quotes
    first = market.decision_quotes(None, None, [event], [.25], economics())[.25]
    del frames["SYN", pd.Timestamp("2026-01-07")]
    second = market.decision_quotes(None, None, [event], [.25], economics())[.25]
    assert first == second
    assert calls[0] == calls[1] == (("SYN", pd.Timestamp("2026-01-05")),)
    result = market.outcome_quotes(None, None, second, .25, economics(), 2)["e"]
    assert result["entry_cash"] == 2209 and result["exit_cash"] is None


def test_missing_entry_and_missing_decision_are_distinct(quotes):
    event, frames, _ = quotes
    decisions = market.decision_quotes(None, None, [event], [.25], economics())[.25]
    del frames["SYN", pd.Timestamp("2026-01-06")]
    assert market.outcome_quotes(None, None, decisions, .25, economics(), 2)["e"]["entry_cash"] is None
    del frames["SYN", pd.Timestamp("2026-01-05")]
    decisions = market.decision_quotes(None, None, [event], [.25], economics())[.25]
    assert decisions["e"][1] is None
    assert market.outcome_quotes(None, None, decisions, .25, economics(), 2) == {}


@pytest.mark.parametrize("day,field", [("2026-01-06", "entry_cash"), ("2026-01-07", "exit_cash")])
def test_missing_pinned_contract_never_reselects_a_surviving_new_atm(quotes, day, field):
    event, frames, _ = quotes
    decisions = market.decision_quotes(None, None, [event], [.25], economics())[.25]
    key = "SYN", pd.Timestamp(day)
    frames[key] = frames[key][frames[key]["strike"] == 120]
    result = market.outcome_quotes(None, None, decisions, .25, economics(), 2)
    assert result["e"][field] is None


@pytest.mark.parametrize("alpha,entry,exit", [(0, 12, 8), (.25, 11, 9), (.5, 10, 10), (1, 8, 12)])
def test_fill_alpha_prices_spread_once_with_two_leg_fees_and_slippage(quotes, alpha, entry, exit):
    event, _, _ = quotes
    decisions = market.decision_quotes(None, None, [event], [alpha], economics())[alpha]
    result = market.outcome_quotes(None, None, decisions, alpha, economics(), 2)["e"]
    side = max(5, 2 * 2 * (.65 + .05)) + 2 * 2 * 100 * .01
    assert side == market.side_cost(economics(), 2) == 9
    assert result["entry_cash"] == entry * 200 + side
    assert result["exit_cash"] == exit * 200 - side


def test_commission_branch_above_minimum_and_integer_quantity_scaling():
    assert market.side_cost(economics(minimum_side_fee=0), 3) == pytest.approx(2 * 3 * .7 + 2 * 3 * 100 * .01)


def test_mismatched_source_ceiling_refuses_instead_of_using_later_quote(quotes):
    event, _, _ = quotes
    event["forecasts"][0]["source_session_ceiling"] = "2026-01-06"
    with pytest.raises(OpsError):
        market.decision_quotes(None, None, [event], [.25], economics())


def test_decision_wide_quote_abstains_without_dropping_event(quotes):
    event, _, _ = quotes
    decisions = market.decision_quotes(None, None, [event], [.25], economics(max_relative_spread=.1))[.25]
    assert set(decisions) == {"e"} and decisions["e"][1] is None


def bridge_entry(store, **changes):
    kwargs = dict(strategy="STR-THRU", driver="abs_move", alpha=.25, cutoff="2025-12-31",
        window=("2024-01-01", "2025-12-30")) | changes
    artifact = make_payoff_line_artifact(dict(n=3, resid_sd=.1, residuals=[-.1, 0, .1],
        intercept=.1, slope=.01), **kwargs)
    ref = store.publish_bytes(serialize_payoff_artifact(artifact), schema_ref=artifact.schema_version)
    return dict(outer_year="2026", alpha=.25, ref=to_document(ref))


def test_real_frozen_bridge_loader_matches_alpha_and_strict_training_window(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    artifact = market.read_bridge(store, bridge_entry(store), dict(year="2026", training_labels_before="2026-01-01"))
    assert artifact.intercept == .1 and artifact.residuals == (-.1, 0, .1)
    assert artifact.window_end < artifact.cutoff < "2026-01-01"


@pytest.mark.parametrize("changes", [dict(alpha=.5), dict(strategy="STR-RUNUP"), dict(driver="signed_move"),
    dict(cutoff=None), dict(cutoff="2026-01-02"), dict(window=None),
    dict(window=("2025-12-31", "2025-12-31")), dict(window=("2025-12-30", "2024-01-01"))])
def test_bridge_incompatible_alpha_cutoff_driver_or_window_refuses(tmp_path, changes):
    store = ArtifactStore(tmp_path / "store")
    with pytest.raises(OpsError):
        market.read_bridge(store, bridge_entry(store, **changes), dict(year="2026", training_labels_before="2026-01-01"))


def benchmark_entry(store, rows, **changes):
    section = write_section(store, "distribution_benchmark", rows, lambda row: (row["event_id"],))
    document = dict(schema_version="distribution_benchmark.v1", outer_year="2026",
        training_labels_before="2026-01-01", forecast_horizon=1,
        decision_policy="native_h1_source_session_ceiling", rows=section) | changes
    ref = store.publish_bytes(canonical_json(document).encode(), schema_ref="distribution_benchmark.v1")
    return dict(outer_year="2026", ref=to_document(ref))


def baseline(event="e", draws=None, gate=True):
    return dict(event_id=event, incumbent_magnitudes=[0, 2, 2, 20] if draws is None else draws, gate_selected=gate)


def test_real_benchmark_cas_shards_exact_cohort_and_detached_replay(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    rows = [baseline(), baseline("z")]
    entry = benchmark_entry(store, rows)
    fold = dict(year="2026", training_labels_before="2026-01-01", evaluation_event_ids=["z", "e"])
    result = market.read_benchmark(store, entry, fold)
    assert result == {row["event_id"]: row for row in rows}
    result["e"]["incumbent_magnitudes"].append(100)
    assert market.read_benchmark(store, entry, fold)["e"] == rows[0]
    assert market.read_benchmark(store, None, fold) == {}


@pytest.mark.parametrize("draws", [[-1], [float("inf")], [float("nan")], [True], [], "1"])
def test_benchmark_nonnegative_finite_nonempty_draw_contract(tmp_path, draws):
    store = ArtifactStore(tmp_path / "store")
    entry = benchmark_entry(store, [baseline(draws=draws)])
    with pytest.raises(OpsError):
        market.read_benchmark(store, entry, dict(year="2026", training_labels_before="2026-01-01", evaluation_event_ids=["e"]))


@pytest.mark.parametrize("changes", [dict(outer_year="2027"), dict(training_labels_before="2026-01-02"),
    dict(training_labels_before="20260101"), dict(schema_version="other"), dict(forecast_horizon=3),
    dict(decision_policy="entry_close"), dict(forecast_horizon=True)])
def test_benchmark_fold_or_date_mismatch_refuses(tmp_path, changes):
    store = ArtifactStore(tmp_path / "store")
    entry = benchmark_entry(store, [baseline()], **changes)
    with pytest.raises(OpsError):
        market.read_benchmark(store, entry, dict(year="2026", training_labels_before="2026-01-01", evaluation_event_ids=["e"]))


@pytest.mark.parametrize("events", [[], ["other"], ["e", "missing"]])
def test_benchmark_must_cover_exact_frozen_cohort(tmp_path, events):
    store = ArtifactStore(tmp_path / "store")
    entry = benchmark_entry(store, [baseline()])
    with pytest.raises(OpsError):
        market.read_benchmark(store, entry, dict(year="2026", training_labels_before="2026-01-01", evaluation_event_ids=events))


def test_all_alphas_share_one_native_plan_calendar_and_decision_chain_load(quotes, monkeypatch):
    event, _, calls = quotes
    plan_calls, original = [], market.plan_events
    def planned(*args):
        plan_calls.append(1)
        return original(*args)
    monkeypatch.setattr(market, "plan_events", planned)
    calendar = TradingCalendar(pd.bdate_range("2026-01-02", "2026-01-09"))
    def forbidden(*args, **kwargs):
        pytest.fail("supplied run-scoped calendar must not be rebuilt")
    monkeypatch.setattr(market, "trading_calendar_from_snapshot", forbidden)
    result = market.decision_quotes(None, None, [event], [0, .25, .5], economics(), calendar=calendar)
    assert set(result) == {0, .25, .5} and len(plan_calls) == len(calls) == 1
    assert [result[alpha]["e"][1].cost for alpha in (0, .25, .5)] == [11, 10, 9]


def test_native_no_session_or_calendar_plan_is_not_invented(quotes):
    event, _, calls = quotes
    event["forecasts"][0]["session"] = None
    assert market.decision_quotes(None, None, [event], [.25], economics()) == {.25: {}}
    assert calls == [()]


def test_bridge_cutoff_equal_to_fold_cutoff_uses_strict_prior_window(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    artifact = market.read_bridge(store, bridge_entry(store, cutoff="2026-01-01"),
        dict(year="2026", training_labels_before="2026-01-01"))
    assert artifact.window_end < artifact.cutoff == "2026-01-01"


def test_benchmark_explicit_unavailable_entries_are_not_fabricated(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    row = dict(event_id="e", incumbent_magnitudes=None, gate_selected=None)
    entry = benchmark_entry(store, [row])
    assert market.read_benchmark(store, entry, dict(year="2026", training_labels_before="2026-01-01",
        evaluation_event_ids=["e"])) == {"e": row}


@pytest.mark.parametrize("gate", [1, "true", [], {}])
def test_benchmark_gate_selection_requires_plain_bool_or_unavailable(tmp_path, gate):
    store = ArtifactStore(tmp_path / "store")
    entry = benchmark_entry(store, [baseline(gate=gate)])
    with pytest.raises(OpsError):
        market.read_benchmark(store, entry, dict(year="2026", training_labels_before="2026-01-01", evaluation_event_ids=["e"]))


def test_benchmark_shard_reference_is_reverified_on_each_import(tmp_path, monkeypatch):
    import json

    store = ArtifactStore(tmp_path / "store")
    entry = benchmark_entry(store, [baseline()])
    root = from_document(ArtifactRef, entry["ref"])
    document = json.loads(store.read_verified(root))
    shard = from_document(ArtifactRef, document["rows"]["shards"][0]["ref"])
    calls, original = [], store.read_verified
    def counted(ref):
        calls.append(ref.content_hash)
        return original(ref)
    monkeypatch.setattr(store, "read_verified", counted)
    fold = dict(year="2026", training_labels_before="2026-01-01", evaluation_event_ids=["e"])
    market.read_benchmark(store, entry, fold)
    market.read_benchmark(store, entry, fold)
    assert calls == [root.content_hash, shard.content_hash] * 2
    # A fresh manifest cannot silently attach a false size to an existing CAS object.
    document["rows"]["shards"][0]["ref"]["byte_size"] += 1
    entry["ref"] = to_document(store.publish_bytes(canonical_json(document).encode(), schema_ref="distribution_benchmark.v1"))
    with pytest.raises((ArtifactError, OpsError)):
        market.read_benchmark(store, entry, fold)


def test_benchmark_duplicate_shard_keys_are_refused_not_deduplicated(tmp_path):
    import json

    store = ArtifactStore(tmp_path / "store")
    entry = benchmark_entry(store, [baseline()])
    document = json.loads(store.read_verified(from_document(ArtifactRef, entry["ref"])))
    document["rows"]["row_count"] = 2
    document["rows"]["shards"] *= 2
    entry["ref"] = to_document(store.publish_bytes(canonical_json(document).encode(), schema_ref="distribution_benchmark.v1"))
    with pytest.raises(OpsError):
        market.read_benchmark(store, entry, dict(year="2026", training_labels_before="2026-01-01", evaluation_event_ids=["e", "z"]))


@pytest.mark.parametrize("kind", ["bridge", "benchmark"])
@pytest.mark.parametrize("size,accepted", [(0, False), (16 * 1024 * 1024 + 1, False),
                                           (1, True), (16 * 1024 * 1024, True)])
def test_public_import_byte_bounds_precede_store_read(tmp_path, monkeypatch, kind, size, accepted):
    store = ArtifactStore(tmp_path / "store")
    entry = bridge_entry(store) if kind == "bridge" else benchmark_entry(store, [baseline()])
    fold = dict(year="2026", training_labels_before="2026-01-01", evaluation_event_ids=["e"])
    original = store.read_verified
    root = from_document(ArtifactRef, entry["ref"])
    payload = original(root)
    entry["ref"]["byte_size"] = size
    calls = []

    def read(ref):
        calls.append(ref)
        if not accepted:
            pytest.fail("invalid root byte size reached storage")
        # Tiny fake root payload isolates boundary admission; the real store's
        # independent size/hash verification is not replaced in production.
        return payload if ref.content_hash == root.content_hash else original(ref)

    monkeypatch.setattr(store, "read_verified", read)
    reader = market.read_bridge if kind == "bridge" else market.read_benchmark
    if accepted:
        assert reader(store, entry, fold)
        assert calls[0].byte_size == size
    else:
        with pytest.raises(OpsError) as caught:
            reader(store, entry, fold)
        assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and calls == []


@pytest.mark.parametrize("contracts", [0, -1, True, 2.5, "2", None])
@pytest.mark.parametrize("reader", ["side", "outcome"])
def test_invalid_quantity_refuses_before_quote_reads(monkeypatch, contracts, reader):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid quantity reached quote storage")
    monkeypatch.setattr(market, "load_chain_index", forbidden)
    with pytest.raises(OpsError) as caught:
        if reader == "side":
            market.side_cost(economics(), contracts)
        else:
            market.outcome_quotes(None, None, {}, .25, economics(), contracts)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("changes", [dict(currency="EUR"), dict(multiplier=50),
    dict(multiplier=100.0), dict(multiplier=True), dict(commission=1e308, exchange_fee=1e308)] +
    [{key: value} for key in ("commission", "exchange_fee", "minimum_side_fee", "slippage_per_share_per_leg")
     for value in (-1, float("inf"), float("nan"), True)])
@pytest.mark.parametrize("reader", ["side", "outcome"])
def test_consumed_cost_units_and_values_refuse_before_quote_reads(monkeypatch, changes, reader):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid economics reached quote storage")
    monkeypatch.setattr(market, "load_chain_index", forbidden)
    with pytest.raises(OpsError) as caught:
        if reader == "side":
            market.side_cost(economics(**changes), 2)
        else:
            market.outcome_quotes(None, None, {}, .25, economics(**changes), 2)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("spread", [-1, float("inf"), float("nan"), True])
@pytest.mark.parametrize("reader", ["decision", "outcome"])
def test_invalid_spread_refuses_before_calendar_or_chain_reads(monkeypatch, spread, reader):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid spread reached native inputs")
    monkeypatch.setattr(market, "trading_calendar_from_snapshot", forbidden)
    monkeypatch.setattr(market, "load_chain_index", forbidden)
    with pytest.raises(OpsError) as caught:
        if reader == "decision":
            market.decision_quotes(None, None, [], [.25], economics(max_relative_spread=spread))
        else:
            market.outcome_quotes(None, None, {}, .25, economics(max_relative_spread=spread), 2)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_bridge_outer_year_mismatch_refuses_before_cas(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "store")
    entry = bridge_entry(store)
    def forbidden(*args, **kwargs):
        pytest.fail("mismatched outer year reached CAS")
    monkeypatch.setattr(store, "read_verified", forbidden)
    with pytest.raises(OpsError) as caught:
        market.read_bridge(store, entry, dict(year="2027", training_labels_before="2027-01-01"))
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("outer_year", ["2025", "2027"])
def test_benchmark_wrapper_year_refuses_before_cas_even_with_matching_document(tmp_path, monkeypatch, outer_year):
    store = ArtifactStore(tmp_path / "store")
    row = baseline()
    entry = benchmark_entry(store, [row])
    fold = dict(year="2026", training_labels_before="2026-01-01", evaluation_event_ids=["e"])
    assert market.read_benchmark(store, entry, fold) == {"e": row}
    entry["outer_year"] = outer_year
    def forbidden(*args, **kwargs):
        pytest.fail("mismatched benchmark wrapper reached CAS")
    monkeypatch.setattr(store, "read_verified", forbidden)
    with pytest.raises(OpsError) as caught:
        market.read_benchmark(store, entry, fold)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
