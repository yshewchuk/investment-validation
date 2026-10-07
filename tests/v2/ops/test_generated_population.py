"""``ops plan nightly`` generates its expected population from the pinned snapshot.

Synthetic ``earnings_events`` only, built the way ``test_v2_ops_nightly_raw_rows`` builds
them (a real catalog + ArtifactStore snapshot). The pin step is stubbed where a test only
needs the plan wiring: its own reference-input catalog is covered elsewhere.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from engine.v2.foundation import ArtifactStore
from engine.v2.ops import cli, nightly
from engine.v2.ops import snapshot_planning as planning
from engine.v2.ops.errors import OpsError
from tests.data_scan_support import (
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)
from tests.ops_support import catalog

_EVENTS = contract_for("earnings_events")
_EVENTS_REF = contract_ref_for(_EVENTS)
_DAILY = contract_for("daily_market")
_AS_OF = "2026-12-20"  # window 2026-12-20 .. 2027-01-24 crosses a partition year
_STRATEGIES = ("BFLY-P", "BFLY-P5", "CND-PS", "CTR5", "DYN-SV", "RAMP7", "STR-RUNUP",
               "STR-THRU", "TWIN-P", "TWIN-P5")  # native-covered + the DYN-SV meta-row


def _event_row(ticker, event_date, session="BMO", suffix=""):
    d = pd.Timestamp(event_date).to_pydatetime()
    return dict(
        event_id=f"{ticker}_{d.date()}{suffix}", ticker=ticker, event_date=d, year=d.year,
        session=session, session_src="orats", annc_tod=None, src_orats=False,
        src_oquants=False, src_nasdaq=True, src_yfinance=False, date_agree=True,
        date_conflict=False, updated_at=None, event_cluster_id=None, claim_count=None,
        reconciliation=None)


def _build(conn, clock, store, rows, *, with_events=True):
    by_year: dict[str, list[dict]] = {}
    for row in rows:
        by_year.setdefault(str(row["year"]), []).append(row)
    tables, contracts = {"daily_market": []}, {"daily_market": _DAILY}
    if with_events:
        tables["earnings_events"] = [publish_and_inspect(store, _EVENTS, _EVENTS_REF,
                                                       sorted(part, key=lambda r: r["event_id"]), year)
                                     for year, part in sorted(by_year.items())]
        contracts["earnings_events"] = _EVENTS
    commit_tables(conn, clock, tables, contracts, store=store)


def _keys(ticker, day):
    return [f"{ticker}|{strategy}|{day}" for strategy in _STRATEGIES]


_ROWS = [
    _event_row("BBB", "2027-01-24"),            # last day of the window: included
    _event_row("AAA", "2026-12-30"),
    _event_row("AAA", "2026-12-30", "AMC", "-dup"),  # same key twice: de-duplicated
    _event_row("AAA", "2026-12-20"),            # as_of day: included
    _event_row("CCC", "2027-01-25"),            # one day past the horizon: excluded
    _event_row("CCC", "2026-12-19"),            # before as_of: excluded
    _event_row("DDD", "2026-12-31", None),      # no session: excluded
]
_EXPECTED = sorted(_keys("AAA", "2026-12-20") + _keys("AAA", "2026-12-30")
                   + _keys("BBB", "2027-01-24"))


@pytest.fixture()
def env(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _build(conn, clock, store, _ROWS)
    return conn, clock, store, tmp_path


def _generate(env, tickers=("AAA", "BBB", "CCC", "DDD"), as_of=_AS_OF, **kwargs):
    conn, clock, store, _ = env
    return planning.generated_population(conn, store, "shadow", as_of=as_of, tickers=tickers,
                                         clock=clock, **kwargs)


def test_generated_population_is_the_sorted_deduplicated_window_keys(env):
    population, snapshot_id = _generate(env)

    assert list(population) == _EXPECTED  # (a) edges in, outside out, sorted, no duplicates
    assert snapshot_id == env[0].execute(
        "SELECT snapshot_id FROM data_snapshot_heads WHERE scope = 'shadow'").fetchone()[0]


def test_generation_is_restricted_to_the_planned_tickers(env):
    population, _ = _generate(env, tickers=("BBB",))

    assert list(population) == sorted(_keys("BBB", "2027-01-24"))


def test_an_empty_window_is_a_typed_refusal_not_an_empty_plan(env):
    with pytest.raises(OpsError) as raised:
        _generate(env, as_of="2026-06-01")  # the window holds no event

    assert raised.value.code == "INVALID_REQUEST"
    assert "no earnings events" in raised.value.problem.message


def test_no_planned_tickers_is_refused(env):
    with pytest.raises(OpsError) as raised:
        _generate(env, tickers=())

    assert raised.value.code == "INVALID_REQUEST"


def test_a_snapshot_without_an_events_table_refuses_with_the_contract_error(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _build(conn, clock, store, [], with_events=False)

    with pytest.raises(OpsError) as raised:
        _generate((conn, clock, store, tmp_path))

    assert raised.value.code == "INPUT_CHANGED"
    assert raised.value.problem.details["data_code"] == "CONTRACT_MISMATCH"


def test_a_moved_head_is_refused_against_the_expected_snapshot_id(env):
    with pytest.raises(OpsError) as raised:
        _generate(env, expected_snapshot_id="not-the-head")

    assert raised.value.code == "INPUT_CHANGED"


class _Pin:
    """Stands in for ``pin_snapshot_inputs`` and records what the plan handed it."""

    def __init__(self):
        self.calls = []

    def __call__(self, conn, store, scope, **kwargs):
        self.calls.append(kwargs)
        return {"scope": scope, "snapshot_ref_artifact_id": "ref-1", "snapshot_id": "snap-1",
                "materialization_request_ref": "request-1"}


def _plan(env, monkeypatch, *extra, mode="snapshot"):
    conn, clock, _, root = env
    pin = _Pin()
    monkeypatch.setattr(planning, "pin_snapshot_inputs", pin)
    argv = ["plan", "nightly", "--as-of", _AS_OF, "--tickers", "AAA,BBB", "--input-mode", mode]
    if mode == "snapshot":
        argv += ["--snapshot-scope", "shadow"]
    plan = cli._plan_command(cli.parser().parse_args([*argv, *extra]), root, conn, clock)["plan"]
    return plan, pin


def test_a_plan_without_a_file_records_the_generated_population(env, monkeypatch):
    plan, pin = _plan(env, monkeypatch)

    expected = sorted(_keys("AAA", "2026-12-20") + _keys("AAA", "2026-12-30")
                      + _keys("BBB", "2027-01-24"))
    assert plan["expected_population"] == expected
    assert pin.calls[0]["expected_population"] == tuple(expected)
    assert pin.calls[0]["expected_snapshot_id"]  # pinned to the snapshot that was scanned


def test_a_supplied_file_overrides_generation(env, monkeypatch):
    supplied = env[3] / "population.json"
    supplied.write_text(json.dumps(["ZZZ|STR-THRU|2026-12-22"]))

    plan, pin = _plan(env, monkeypatch, "--expected-population", str(supplied))

    assert plan["expected_population"] == ["ZZZ|STR-THRU|2026-12-22"]
    assert pin.calls[0]["expected_snapshot_id"] is None  # nothing was generated or pre-resolved


def test_a_symlinked_or_malformed_override_keeps_its_refusals(env, monkeypatch):
    real = env[3] / "real.json"
    real.write_text(json.dumps(["AAA|STR-THRU|2026-12-20"]))
    link = env[3] / "link.json"
    link.symlink_to(real)
    bad = env[3] / "bad.json"
    bad.write_text(json.dumps({"not": "a list"}))

    with pytest.raises(OpsError) as symlinked:
        _plan(env, monkeypatch, "--expected-population", str(link))
    with pytest.raises(OpsError) as malformed:
        _plan(env, monkeypatch, "--expected-population", str(bad))

    assert symlinked.value.code == "INPUT_CHANGED"
    assert malformed.value.code == "INVALID_REQUEST"


def test_legacy_input_mode_generates_nothing(env, monkeypatch):
    plan, _ = _plan(env, monkeypatch, mode="legacy")

    assert plan["expected_population"] == []
    assert "planned_population" in plan["blocked_prerequisites"]


def test_the_same_snapshot_and_as_of_give_the_same_population_and_scope_hash(env, monkeypatch):
    first, pin_one = _plan(env, monkeypatch)
    second, pin_two = _plan(env, monkeypatch)
    snapshot = pin_one(None, None, "shadow")

    def scope(plan):
        return nightly._scope_hash(plan["tickers"], plan["year_start"], plan["year_end"],
                                   plan["expected_population"], snapshot)

    assert first["expected_population"] == second["expected_population"]
    assert first["plan_hash"] == second["plan_hash"]
    assert scope(first) == scope(second)
    assert pin_one.calls[0]["expected_snapshot_id"] == pin_two.calls[0]["expected_snapshot_id"]
