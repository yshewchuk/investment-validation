"""The Polygon real-trade pull and its normalizer.

The live path spends ~100 minutes of rate budget on ~9.3k calls, so the
planning arithmetic (universe derivation, date windows, cache-skipping,
ordering) and the bar parser are tested here against fake adapters and
synthetic trades — the live run is where the network seam itself gets checked.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from engine.data import store
from engine.data.fetch import Fetcher
from engine.data.normalize import n_option_daily
from engine.data.pulls import polygon_fills
from engine.data.pulls.polygon_fills import (
    ENTRY_BUFFER_DAYS,
    POLYGON_OPTIONS_START,
    ContractJob,
    build_plan,
    collect_contracts,
    execute,
    render_dry_run,
)
from engine.data.sources.base import Response
from engine.data.sources.polygon import option_ticker
from engine.data.throttle import SourceConfig, Throttle


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _leg(right="C", strike=100.0, expiry="2024-09-06", **extra):
    base = {"name": "leg", "right": right, "side": "buy", "qty": 1.0,
            "expiry": expiry, "strike": strike, "dte": 10}
    base.update(extra)
    return base


def make_trades(rows):
    """A trades frame shaped like the Tier-2 table's pricing columns."""
    out = []
    for i, row in enumerate(rows):
        out.append(
            {
                "trade_id": f"t{i}",
                "ticker": row["ticker"],
                "entry_date": row.get("entry_date"),
                "exit_date": row.get("exit_date"),
                "legs": json.dumps({"entry": row["entry"], "exit": row["exit"]}),
            }
        )
    return pd.DataFrame(out)


class FakeFetcher:
    """Reports a configurable set of contracts as already cached."""

    def __init__(self, cached_contracts=()):
        self.cached = set(cached_contracts)

    def has(self, source, endpoint, params=None, *, live=False):
        contract = n_option_daily.contract_from_endpoint(endpoint)
        return contract in self.cached


class FakePolygonAdapter:
    """Replays one scripted body per endpoint."""

    name = "polygon"

    def __init__(self, bodies):
        self.bodies = dict(bodies)
        self.calls = []

    def request(self, endpoint, params, timeout):
        self.calls.append((endpoint, dict(params)))
        body = self.bodies.get(endpoint, b'{"results": [], "resultsCount": 0}')
        return Response(200, body, {}, f"https://api.polygon.io/{endpoint}")

    def quota_from(self, response):
        return None

    def is_auth_failure(self, response):
        return False


def make_fetcher(tmp_root, bodies):
    adapter = FakePolygonAdapter(bodies)
    throttle = Throttle(
        {"polygon": SourceConfig("polygon", 0.0, 0.0, 2, 5.0, single_process=False)},
        sleep_fn=lambda s: None,
    )
    return Fetcher(throttle=throttle, adapters={"polygon": adapter}), adapter


def aggs_body(bars):
    return json.dumps({"results": bars, "resultsCount": len(bars)}).encode()


def bar(day, close, *, o=None, h=None, l=None, v=10, vw=None, n=3):
    ts = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
    return {"o": o or close, "h": h or close, "l": l or close, "c": close,
            "v": v, "vw": vw or close, "t": ts, "n": n}


# --------------------------------------------------------------------------
# contract parsing
# --------------------------------------------------------------------------


class TestContractParsing:
    def test_round_trips_with_the_occ_builder(self):
        contract = option_ticker("TSLA", "2024-09-06", "C", 210.0)
        assert contract == "O:TSLA240906C00210000"
        parsed = n_option_daily.parse_contract(contract)
        assert parsed == {
            "ticker": "TSLA",
            "expiry": pd.Timestamp("2024-09-06"),
            "right": "C",
            "strike": 210.0,
        }

    def test_puts_and_fractional_strikes_round_trip(self):
        contract = option_ticker("MRVL", "2025-01-17", "P", 72.5)
        parsed = n_option_daily.parse_contract(contract)
        assert parsed["right"] == "P"
        assert parsed["strike"] == 72.5

    def test_a_malformed_id_parses_to_none(self):
        assert n_option_daily.parse_contract("TSLA240906C00210000") is None
        assert n_option_daily.parse_contract("O:TSLA240906X00210000") is None
        assert n_option_daily.parse_contract("") is None

    def test_endpoint_recovery(self):
        ep = "v2/aggs/ticker/O:TSLA240906C00210000/range/1/day"
        assert n_option_daily.contract_from_endpoint(ep) == "O:TSLA240906C00210000"
        assert n_option_daily.contract_from_endpoint("v3/reference/options/contracts") is None
        assert n_option_daily.contract_from_endpoint("v2/aggs/ticker/TSLA/range/1/day") == "TSLA"


# --------------------------------------------------------------------------
# bar parsing
# --------------------------------------------------------------------------


class TestBarsToFrame:
    CONTRACT = "O:TSLA240906C00210000"

    def test_bars_become_tier2_rows(self):
        bars = [bar("2024-08-27", 6.4, v=120, vw=6.35, n=11),
                bar("2024-08-28", 6.9, v=80, vw=6.8, n=7)]
        frame, report = n_option_daily.bars_to_frame(self.CONTRACT, bars, source_id="src")
        assert report["rows_out"] == 2
        row = frame.iloc[0]
        assert row["close"] == 6.4
        assert row["vwap"] == 6.35
        assert row["volume"] == 120
        assert row["n_trades"] == 11
        assert row["obs_date"] == pd.Timestamp("2024-08-27")
        assert row["ticker"] == "TSLA"
        assert row["strike"] == 210.0
        assert row["right"] == "C"
        assert row["src"] == "polygon.v2.aggs"

    def test_impossible_bars_are_excluded_and_counted(self):
        bars = [
            bar("2024-08-27", 6.4),
            bar("2024-08-28", 6.9, h=1.0, l=2.0),  # high below low
            {"o": 1, "h": 1, "l": 1, "c": -0.05, "v": 1, "vw": 1,
             "t": int(pd.Timestamp("2024-08-29", tz="UTC").timestamp() * 1000)},
        ]
        frame, report = n_option_daily.bars_to_frame(self.CONTRACT, bars, source_id="src")
        assert report["rows_out"] == 1
        assert report["excluded"] == 2

    def test_empty_results_are_a_legitimate_answer(self):
        frame, report = n_option_daily.bars_to_frame(self.CONTRACT, [], source_id="src")
        assert frame.empty
        assert report["reason"] == "empty results"

    def test_an_unparseable_contract_is_reported_not_guessed(self):
        frame, report = n_option_daily.bars_to_frame("garbage", [bar("2024-08-27", 1.0)],
                                                     source_id="src")
        assert frame.empty
        assert "unparseable" in report["reason"]


# --------------------------------------------------------------------------
# universe derivation
# --------------------------------------------------------------------------


class TestCollectContracts:
    def test_contracts_come_from_both_phases(self):
        trades = make_trades([
            {"ticker": "TSLA",
             "entry_date": pd.Timestamp("2024-08-27"),
             "exit_date": pd.Timestamp("2024-09-06"),
             "entry": [_leg("C", 210.0, "2024-09-06"), _leg("P", 210.0, "2024-09-06")],
             "exit": [_leg("C", 210.0, "2024-09-06"), _leg("P", 210.0, "2024-09-06")]},
        ])
        info = collect_contracts(trades)
        assert set(info) == {
            option_ticker("TSLA", "2024-09-06", "C", 210.0),
            option_ticker("TSLA", "2024-09-06", "P", 210.0),
        }
        call = info[option_ticker("TSLA", "2024-09-06", "C", 210.0)]
        assert len(call["dates"]) == 2  # entry and exit days both observed

    def test_pre_window_trades_are_not_pulled(self):
        trades = make_trades([
            {"ticker": "TSLA",
             "entry_date": pd.Timestamp("2024-08-01"),
             "exit_date": pd.Timestamp("2024-08-16"),
             "entry": [_leg()], "exit": [_leg()]},
        ])
        assert collect_contracts(trades) == {}

    def test_repeat_trades_collapse_to_one_contract(self):
        rows = [
            {"ticker": "TSLA",
             "entry_date": pd.Timestamp(d),
             "exit_date": pd.Timestamp(d) + pd.Timedelta(days=1),
             "entry": [_leg()], "exit": [_leg()]}
            for d in ("2024-09-03", "2024-09-10")
        ]
        info = collect_contracts(make_trades(rows))
        assert len(info) == 1
        rec = next(iter(info.values()))
        assert len(rec["dates"]) == 4

    def test_a_leg_shape_we_do_not_understand_adds_no_job(self):
        trades = make_trades([
            {"ticker": "TSLA",
             "entry_date": pd.Timestamp("2024-08-27"),
             "exit_date": pd.Timestamp("2024-09-06"),
             "entry": [{"name": "mystery"}], "exit": []},
        ])
        assert collect_contracts(trades) == {}


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


class TestBuildPlan:
    def trades(self):
        return make_trades([
            {"ticker": "TSLA",
             "entry_date": pd.Timestamp("2024-08-27"),
             "exit_date": pd.Timestamp("2024-09-06"),
             "entry": [_leg("C", 210.0, "2024-09-06")],
             "exit": [_leg("C", 210.0, "2024-09-06")]},
        ])

    def test_one_job_per_contract_with_buffer_and_expiry_range(self):
        plan = build_plan(self.trades(), fetcher=FakeFetcher())
        assert plan.n_calls == 1
        job = plan.jobs[0]
        assert job.start == (pd.Timestamp("2024-08-27") - pd.Timedelta(days=ENTRY_BUFFER_DAYS)).strftime("%Y-%m-%d")
        assert job.end == "2024-09-06"

    def test_cached_contracts_cost_nothing(self):
        contract = option_ticker("TSLA", "2024-09-06", "C", 210.0)
        plan = build_plan(self.trades(), fetcher=FakeFetcher(cached_contracts={contract}))
        assert plan.n_calls == 0
        assert plan.skipped_cached == 1

    def test_max_calls_truncates_and_reports(self):
        rows = [
            {"ticker": "TSLA",
             "entry_date": pd.Timestamp(f"2024-09-{d:02d}"),
             "exit_date": pd.Timestamp(f"2024-09-{d:02d}") + pd.Timedelta(days=1),
             "entry": [_leg("C", 200.0 + d, "2024-09-20")],
             "exit": [_leg("C", 200.0 + d, "2024-09-20")]}
            for d in (3, 4, 5)
        ]
        plan = build_plan(make_trades(rows), fetcher=FakeFetcher(), max_calls=2)
        assert plan.n_calls == 2
        assert plan.truncated_at_limit

    def test_most_observed_contracts_are_fetched_first(self):
        rows = [
            {"ticker": "AAA",
             "entry_date": pd.Timestamp("2024-09-03"),
             "exit_date": pd.Timestamp("2024-09-04"),
             "entry": [_leg("C", 100.0, "2024-09-13")],
             "exit": [_leg("C", 100.0, "2024-09-13")]},
        ]
        rows += [
            {"ticker": "BBB",
             "entry_date": pd.Timestamp(d),
             "exit_date": pd.Timestamp(d) + pd.Timedelta(days=1),
             "entry": [_leg("C", 50.0, "2024-09-20")],
             "exit": [_leg("C", 50.0, "2024-09-20")]}
            for d in ("2024-09-03", "2024-09-10")
        ]
        # Both contracts sit inside the recent window of the reference date,
        # so the within-tier rule applies: BBB has four obs dates vs AAA's two.
        plan = build_plan(make_trades(rows), fetcher=FakeFetcher())
        assert plan.jobs[0].ticker == "BBB"
        assert plan.jobs[0].n_obs_dates == 4

    def test_recent_contracts_lead_even_with_fewer_observations(self):
        # The newest evidence is what live decisions read, so a contract
        # observed once last month outranks one observed four times in autumn.
        rows = [
            {"ticker": "OLD",
             "entry_date": pd.Timestamp(d),
             "exit_date": pd.Timestamp(d) + pd.Timedelta(days=1),
             "entry": [_leg("C", 100.0, "2024-09-20")],
             "exit": [_leg("C", 100.0, "2024-09-20")]}
            for d in ("2024-09-03", "2024-09-10")
        ]
        rows += [
            {"ticker": "NEW",
             "entry_date": pd.Timestamp("2024-12-02"),
             "exit_date": pd.Timestamp("2024-12-03"),
             "entry": [_leg("C", 110.0, "2024-12-20")],
             "exit": [_leg("C", 110.0, "2024-12-20")]},
        ]
        plan = build_plan(make_trades(rows), fetcher=FakeFetcher())
        assert [j.ticker for j in plan.jobs] == ["NEW", "OLD"]
        assert plan.recent_jobs == 1

    def test_recent_prioritization_can_be_disabled(self):
        rows = [
            {"ticker": "OLD",
             "entry_date": pd.Timestamp(d),
             "exit_date": pd.Timestamp(d) + pd.Timedelta(days=1),
             "entry": [_leg("C", 100.0, "2024-09-20")],
             "exit": [_leg("C", 100.0, "2024-09-20")]}
            for d in ("2024-09-03", "2024-09-10")
        ]
        rows += [
            {"ticker": "NEW",
             "entry_date": pd.Timestamp("2024-12-02"),
             "exit_date": pd.Timestamp("2024-12-03"),
             "entry": [_leg("C", 110.0, "2024-12-20")],
             "exit": [_leg("C", 110.0, "2024-12-20")]},
        ]
        plan = build_plan(make_trades(rows), fetcher=FakeFetcher(), recent_days=0)
        assert [j.ticker for j in plan.jobs] == ["OLD", "NEW"]  # obs-count order
        assert plan.recent_jobs == 0

    def test_dry_run_names_the_entitlement_limit(self):
        plan = build_plan(self.trades(), fetcher=FakeFetcher())
        text = render_dry_run(plan)
        assert "DRY RUN" in text
        assert "NOT entitled" in text
        assert POLYGON_OPTIONS_START in text

    def test_job_params_are_deterministic(self):
        plan = build_plan(self.trades(), fetcher=FakeFetcher())
        job = plan.jobs[0]
        # The date range travels as path segments (query params 404 live), so
        # the params carry only the fixed options and the endpoint carries the
        # range.
        assert job.params == {"adjusted": "false", "limit": polygon_fills.AGGS_LIMIT}
        assert job.endpoint == (
            f"v2/aggs/ticker/{job.contract_ticker}/range/1/day/{job.start}/{job.end}"
        )


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


class RecordingFetcher:
    def __init__(self, fail_contracts=()):
        self.fail = set(fail_contracts)
        self.fetched = []

    def fetch(self, source, endpoint, params, **kwargs):
        contract = n_option_daily.contract_from_endpoint(endpoint)
        if contract in self.fail:
            raise RuntimeError("boom")
        self.fetched.append(contract)

        class _Record:
            from_cache = False

        return _Record()


class TestExecute:
    def test_execute_fetches_every_planned_job(self):
        plan = build_plan(
            make_trades([
                {"ticker": "TSLA",
                 "entry_date": pd.Timestamp("2024-08-27"),
                 "exit_date": pd.Timestamp("2024-09-06"),
                 "entry": [_leg()], "exit": [_leg()]}
            ]),
            fetcher=FakeFetcher(),
        )
        fetcher = RecordingFetcher()
        report = execute(plan, fetcher=fetcher)
        assert report["fetched"] == 1
        assert report["failed"] == 0
        assert fetcher.fetched == [plan.jobs[0].contract_ticker]

    def test_a_failing_contract_does_not_end_the_run(self):
        plan = build_plan(
            make_trades([
                {"ticker": "TSLA",
                 "entry_date": pd.Timestamp("2024-09-03"),
                 "exit_date": pd.Timestamp("2024-09-04"),
                 "entry": [_leg("C", 100.0 + i, "2024-09-13")],
                 "exit": [_leg("C", 100.0 + i, "2024-09-13")]}
                for i in range(3)
            ]),
            fetcher=FakeFetcher(),
        )
        fail = {plan.jobs[0].contract_ticker}
        report = execute(plan, fetcher=RecordingFetcher(fail_contracts=fail))
        assert report["failed"] == 1
        assert report["fetched"] == 2
        assert "boom" in report["errors"][0]


# --------------------------------------------------------------------------
# end to end: fetch store → normalizer → Tier 2
# --------------------------------------------------------------------------


class TestFetchStoreRoundTrip:
    def test_payloads_land_in_tier2_through_the_wrapper(self, tmp_root):
        contract = option_ticker("TSLA", "2024-09-06", "C", 210.0)
        job = ContractJob(contract, "TSLA", "2024-08-06", "2024-09-06", 2)
        body = aggs_body([bar("2024-08-27", 6.4, v=120, vw=6.35, n=11)])
        fetcher, adapter = make_fetcher(tmp_root, {job.endpoint: body})

        fetcher.fetch("polygon", job.endpoint, job.params, note="test")
        # A non-aggs payload in the same store must be skipped, not misread.
        fetcher.fetch("polygon", "v3/reference/options/contracts",
                      {"underlying_ticker": "TSLA"}, note="test")

        entries = list(n_option_daily.iter_aggs_sources())
        assert len(entries) == 1

        frame, report = n_option_daily.normalize_entry(entries[0])
        assert report["rows_out"] == 1
        store.write_table(frame, "option_daily")
        out = store.read_table("option_daily")
        assert len(out) == 1
        row = out.iloc[0]
        assert row["contract_ticker"] == contract
        assert row["close"] == 6.4
        assert row["src_file"].startswith("fetch:polygon/")

    def test_a_second_fetch_is_a_cache_hit(self, tmp_root):
        contract = option_ticker("TSLA", "2024-09-06", "C", 210.0)
        job = ContractJob(contract, "TSLA", "2024-08-06", "2024-09-06", 1)
        fetcher, adapter = make_fetcher(
            tmp_root, {job.endpoint: aggs_body([bar("2024-08-27", 6.4)])}
        )
        fetcher.fetch("polygon", job.endpoint, job.params)
        again = fetcher.fetch("polygon", job.endpoint, job.params)
        assert again.from_cache
        assert len(adapter.calls) == 1  # the resume promise, made literal

    def test_a_zero_trade_contract_is_empty_not_unrecognized(self, tmp_root):
        # Polygon omits the results key entirely when a contract never traded;
        # that is a legitimate zero-row answer, not a malformed payload.
        contract = option_ticker("TSLA", "2024-09-06", "P", 210.0)
        job = ContractJob(contract, "TSLA", "2024-08-06", "2024-09-06", 1)
        body = json.dumps(
            {"ticker": contract, "queryCount": 0, "resultsCount": 0,
             "adjusted": False, "status": "OK"}
        ).encode()
        fetcher, _ = make_fetcher(tmp_root, {job.endpoint: body})
        fetcher.fetch("polygon", job.endpoint, job.params)
        entries = list(n_option_daily.iter_aggs_sources())
        frame, report = n_option_daily.normalize_entry(entries[0])
        assert frame.empty
        assert report["reason"] == "empty results"
