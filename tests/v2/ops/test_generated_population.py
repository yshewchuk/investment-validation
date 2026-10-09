"""``ops plan nightly`` generates its expected population from the pinned snapshot.

Synthetic ``earnings_events`` plus the real synthetic ``daily_market``/``option_chains``
rows the carried-set resolution reads, built the way ``test_v2_ops_nightly_raw_rows``
builds them (a real catalog + ArtifactStore snapshot). The pin step is stubbed where a
test only needs the plan wiring: its own reference-input catalog is covered elsewhere.
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from engine.v2.data import catalog as data_catalog
from engine.v2.data import manifests
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore
from engine.v2.ops import cli, nightly
from engine.v2.ops import snapshot_planning as planning
from engine.v2.ops.errors import OpsError
from engine.v2.ops.nightly_raw_rows import scan_forward_board_requests
from tests.data_scan_support import (
    RECEIPT,
    commit_tables,
    contract_for,
    contract_ref_for,
    fake_hash,
    publish_and_inspect,
)
from tests.ops_support import catalog

_EVENTS = contract_for("earnings_events")
_EVENTS_REF = contract_ref_for(_EVENTS)
_DAILY = contract_for("daily_market")
_CHAINS = contract_for("option_chains")
_AS_OF = "2026-12-20"  # window 2026-12-20 .. 2027-01-24 crosses a partition year
_CARRY_DAY = dt.date(2026, 12, 15)  # inside 2025-01-01 .. 2026-12-20, the carried window
_STRATEGIES = ("BFLY-P", "BFLY-P5", "CAL-P", "CND-P", "CND-PS", "CTR5", "RAMP7", "STR-RUNUP",
               "STR-THRU", "TWIN-P", "TWIN-P5")  # what the legacy score stage emits per event


def _event_row(ticker, event_date, session="BMO", suffix=""):
    d = pd.Timestamp(event_date).to_pydatetime()
    return dict(
        event_id=f"{ticker}_{d.date()}{suffix}", ticker=ticker, event_date=d, year=d.year,
        session=session, session_src="orats", annc_tod=None, src_orats=False,
        src_oquants=False, src_nasdaq=True, src_yfinance=False, date_agree=True,
        date_conflict=False, updated_at=None, event_cluster_id=None, claim_count=None,
        reconciliation=None)


def _daily(ticker, day):
    """A schema-complete ``daily_market`` row (adapted from ``test_carried_set``)."""
    if isinstance(day, dt.date) and not isinstance(day, dt.datetime):
        day = dt.datetime(day.year, day.month, day.day)
    return dict(
        ticker=ticker, date=day, year=day.year, spot=100.0, iv10=30.0, iv30=32.0,
        exern_iv10=29.0, exern_iv30=31.0, implied_move=5.0, implied_reconstructed=False,
        rvol30=28.0, skew=1.1, contango=0.5, fwd90_30=33.0, fexern90_30=34.0, iee=0.2,
        mcap_usd=1e9, mcap_log=20.7, mcap_asof=dt.datetime(day.year, 1, 2), mcap_age_days=0.0,
        src_spot="orats", src_iv="orats", src_mcap="orats")


def _chain(ticker, obs_date):
    """A schema-complete ``option_chains`` row (adapted from ``test_carried_set``)."""
    if isinstance(obs_date, dt.date) and not isinstance(obs_date, dt.datetime):
        obs_date = dt.datetime(obs_date.year, obs_date.month, obs_date.day)
    return dict(
        ticker=ticker, obs_date=obs_date, year=obs_date.year,
        expiry=obs_date + dt.timedelta(days=30), dte=30, strike=100.0, right="C", bid=1.0,
        ask=1.2, mid=1.1, iv=30.0, delta=0.5, spot=100.0, src="orats", src_file="f.parquet",
        chain_kind="entry", volume=None, open_interest=None, bid_size=None, ask_size=None,
        quote_repaired=False)


def _fragments(store, contract, rows):
    """One published, inspected fragment per year, in ascending partition-key order."""
    ref = contract_ref_for(contract)
    by_year: dict[str, list[dict]] = {}
    for row in rows:
        by_year.setdefault(str(row["year"]), []).append(row)
    return [publish_and_inspect(
                store, contract, ref,
                sorted(part, key=lambda row: tuple(row[c] for c in contract.primary_key)), year)
            for year, part in sorted(by_year.items())]


def _build(conn, clock, store, rows, *, with_events=True, with_chains=True):
    tables = {"daily_market": _fragments(
        store, _DAILY, [_daily("AAA", _CARRY_DAY), _daily("BBB", _CARRY_DAY)])}
    contracts = {"daily_market": _DAILY}
    if with_chains:
        tables["option_chains"] = _fragments(
            store, _CHAINS, [_chain("AAA", _CARRY_DAY), _chain("BBB", _CARRY_DAY)])
        contracts["option_chains"] = _CHAINS
    if with_events:
        tables["earnings_events"] = _fragments(store, _EVENTS, list(rows))
        contracts["earnings_events"] = _EVENTS
    commit_tables(conn, clock, tables, contracts, store=store)


def _head_id(conn):
    return conn.execute(
        "SELECT snapshot_id FROM data_snapshot_heads WHERE scope = 'shadow'").fetchone()[0]


def _advance_head(conn, clock, store):
    """Commit a child snapshot (one more events partition) and move the ``shadow`` head to it."""
    head = conn.execute("SELECT snapshot_id, generation FROM data_snapshot_heads "
                        "WHERE scope = 'shadow'").fetchone()
    parent = Repository(conn, store).resolve(head["snapshot_id"])
    records = {name: list(Repository(conn, store).fragment_records(parent, name))
               for name in ("daily_market", "option_chains", "earnings_events")}
    records["earnings_events"].append(publish_and_inspect(
        store, _EVENTS, _EVENTS_REF, [_event_row("EEE", "2028-01-05")], "2028"))
    contracts = {"daily_market": _DAILY, "option_chains": _CHAINS, "earnings_events": _EVENTS}
    table_manifests = {name: manifests.dataset_manifest(
        contract_ref_for(contracts[name]), recs, knowledge_mode="reconstructed",
        coverage_receipt_refs=(RECEIPT,), availability_evidence_refs=())
        for name, recs in records.items()}
    child = manifests.snapshot_ref(
        table_manifests, calendar_version="cal.v1", source_priority_version="prio.v1",
        finality_receipt_refs=(RECEIPT,), parent_snapshot_id=parent.snapshot_id)
    data_catalog.commit_snapshot(
        conn, scope="shadow", request_hash=fake_hash("advance"), contracts=list(contracts.values()),
        objects=[r.object_ref for recs in records.values() for r in recs],
        records=[r for recs in records.values() for r in recs], manifests=list(table_manifests.values()),
        snapshot=child, expected_head_snapshot_id=head["snapshot_id"],
        expected_head_generation=head["generation"], receipt_id="r-advance", attempt_id="att-advance",
        fence=1, fence_check=lambda _c: None, clock=clock, store=store)


def _keys(ticker, day):
    return [f"{ticker}|{strategy}|{day}" for strategy in _STRATEGIES]


_ROWS = [
    _event_row("BBB", "2027-01-24"),            # last day of the window: included
    _event_row("EEE", "2027-01-03"),            # in-window event, absent from carried tables
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


def _generate(env, tickers=("AAA", "BBB", "CCC", "DDD", "EEE"), as_of=_AS_OF, **kwargs):
    conn, clock, store, _ = env
    return planning.generated_population(conn, store, "shadow", as_of=as_of, tickers=tickers,
                                         clock=clock, **kwargs)


def test_generated_population_is_the_sorted_deduplicated_window_keys(env):
    population, snapshot_id, _ = _generate(env)

    assert list(population) == _EXPECTED  # (a) edges in, outside out, sorted, no duplicates
    assert snapshot_id == _head_id(env[0])


def test_generation_is_restricted_to_the_planned_tickers(env):
    population, _, _ = _generate(env, tickers=("BBB",))

    assert list(population) == sorted(_keys("BBB", "2027-01-24"))


def test_all_uncarried_candidates_are_in_refusal_evidence(env):
    with pytest.raises(OpsError) as raised:
        _generate(env, tickers=("EEE",))

    assert raised.value.code == "INVALID_REQUEST"
    assert raised.value.problem.details["candidate_exclusions"] == [{
        "ticker": "EEE", "reason_code": "UNCARRIED_TICKER",
        "missing_tables": ["daily_market", "option_chains"],
    }]


def test_an_empty_window_is_a_typed_refusal_not_an_empty_plan(env):
    with pytest.raises(OpsError) as raised:
        _generate(env, as_of="2026-06-01")  # the window holds no event

    assert raised.value.code == "INVALID_REQUEST"
    assert "no eligible earnings events" in raised.value.problem.message


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


def test_missing_eligibility_table_uses_existing_snapshot_refusal(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _build(conn, clock, store, [_event_row("AAA", "2026-12-22")], with_chains=False)

    with pytest.raises(OpsError) as raised:
        _generate((conn, clock, store, tmp_path), tickers=("AAA",))

    assert raised.value.code == "INPUT_CHANGED"
    assert raised.value.problem.details["data_code"] == "CONTRACT_MISMATCH"


def test_a_scope_without_a_head_refuses_with_the_not_ready_error(tmp_path):
    conn, clock, _ = catalog(tmp_path)  # nothing committed: the scope has no head

    with pytest.raises(OpsError) as raised:
        _generate((conn, clock, ArtifactStore(tmp_path), tmp_path))

    assert raised.value.code == "INPUT_CHANGED"
    assert raised.value.problem.details["data_code"] == "SNAPSHOT_NOT_READY"


def test_a_head_that_moves_after_the_scan_is_refused_by_the_pin(env):
    conn, clock, store, _ = env
    population, scanned, _ = _generate(env)
    _advance_head(conn, clock, store)
    assert _head_id(conn) != scanned

    with pytest.raises(OpsError) as raised:  # the real pin re-resolves and compares
        planning.pin_snapshot_inputs(
            conn, store, "shadow", tickers=("AAA", "BBB"), year_start=2026, year_end=2027,
            expected_population=population, clock=clock, session=_AS_OF,
            expected_snapshot_id=scanned)

    assert raised.value.code == "INPUT_CHANGED"
    assert "moved" in raised.value.problem.message


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


def _plan(env, monkeypatch, *extra, mode="snapshot", tickers="AAA,BBB,EEE"):
    conn, clock, _, root = env
    pin = _Pin()
    monkeypatch.setattr(planning, "pin_snapshot_inputs", pin)
    argv = ["plan", "nightly", "--as-of", _AS_OF, "--input-mode", mode]
    if tickers:
        argv += ["--tickers", tickers]
    if mode == "snapshot":
        argv += ["--snapshot-scope", "shadow"]
    plan = cli._plan_command(cli.parser().parse_args([*argv, *extra]), root, conn, clock)["plan"]
    return plan, pin


def test_a_plan_without_a_file_records_the_generated_population(env, monkeypatch):
    plan, pin = _plan(env, monkeypatch)

    expected = sorted(_keys("AAA", "2026-12-20") + _keys("AAA", "2026-12-30")
                      + _keys("BBB", "2027-01-24"))
    assert plan["expected_population"] == expected
    assert plan["candidate_exclusions"] == [{
        "ticker": "EEE", "reason_code": "UNCARRIED_TICKER",
        "missing_tables": ["daily_market", "option_chains"],
    }]
    assert pin.calls[0]["expected_population"] == tuple(expected)
    assert pin.calls[0]["expected_snapshot_id"] == _head_id(env[0])  # the snapshot that was scanned


def test_generation_follows_the_watchlist_not_the_wider_context(env, monkeypatch):
    plan, _ = _plan(env, monkeypatch, "--context-tickers", "AAA,BBB,CCC,DDD", tickers="AAA")

    assert plan["expected_population"] == sorted(_keys("AAA", "2026-12-20") + _keys("AAA", "2026-12-30"))


def test_plan_filters_legacy_and_native_enumerators_to_the_same_carried_events(env, monkeypatch):
    plan, _ = _plan(env, monkeypatch)
    score_parameters = nightly._legacy_params(
        "legacy_score", plan, plan["tickers"], plan["year_start"], plan["year_end"],
        {"finality": "job-finality", "features": "job-features"},
        context_tickers=plan["context_tickers"])
    eligible_tickers = tuple(sorted({key.split("|")[0] for key in plan["expected_population"]}))

    assert score_parameters["tickers"] == eligible_tickers == ("AAA", "BBB")
    assert score_parameters["context_tickers"] == tuple(plan["context_tickers"])
    conn, clock, store, _ = env
    snapshot = Repository(conn, store).resolve(_head_id(conn))
    native_requests = scan_forward_board_requests(
        Repository(conn, store), snapshot, as_of=_AS_OF,
        horizon_days=planning.GENERATED_HORIZON_DAYS, tickers=score_parameters["tickers"])
    native_events = {(request.ticker, request.event_date.date().isoformat())
                     for request in native_requests}
    planned_events = {(key.split("|")[0], key.split("|")[2])
                      for key in plan["expected_population"]}

    assert native_events == planned_events
    assert all(ticker != "EEE" for ticker, _ in native_events)


def test_a_plan_without_a_watchlist_is_refused_not_widened_to_the_context(env, monkeypatch):
    with pytest.raises(OpsError) as raised:  # the score stage scores --tickers only
        _plan(env, monkeypatch, "--context-tickers", "AAA,BBB", tickers="")

    assert raised.value.code == "INVALID_REQUEST"
    assert "--tickers" in raised.value.problem.message


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


def test_a_supplied_empty_list_is_refused_by_the_pin_and_blocks_a_legacy_plan(env, monkeypatch):
    conn, clock, store, root = env
    empty = root / "empty.json"
    empty.write_text("[]")

    with pytest.raises(OpsError) as raised:  # the REAL pin: nothing generated to fall back on
        planning.pin_snapshot_inputs(
            conn, store, "shadow", tickers=("AAA",), year_start=2026, year_end=2027,
            expected_population=(), clock=clock, session=_AS_OF)
    plan, _ = _plan(env, monkeypatch, "--expected-population", str(empty), mode="legacy")

    assert raised.value.code == "INVALID_REQUEST"
    assert plan["expected_population"] == []
    assert "planned_population" in plan["blocked_prerequisites"]


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
    assert first["candidate_exclusions"] == second["candidate_exclusions"]
    assert first["plan_hash"] == second["plan_hash"]
    assert scope(first) == scope(second)
    assert pin_one.calls[0]["expected_snapshot_id"] == pin_two.calls[0]["expected_snapshot_id"]


# --- the legacy score stage's own population check ---------------------------------------------


def _score_events():
    """``earnings_events`` as the legacy ``score_calendar`` reads it, for the same synthetic rows."""
    return pd.DataFrame([{"event_id": r["event_id"], "ticker": r["ticker"],
                          "event_date": pd.Timestamp(r["event_date"]), "session": r["session"]}
                         for r in _ROWS])


class _FakeCalendar:
    def resolve_offsets(self, *args, **kwargs):
        raise KeyError("synthetic: no calendar range")  # plan_events skips: no chain index needed


class _FakeScorer:
    analog_entry_coverage = 1.0

    def __init__(self, *args, **kwargs):
        self.calendar = _FakeCalendar()
        self._live_features_cache = {}

    def score(self, request, chain_index=None):
        from engine.score import ScoreResult
        return ScoreResult(ticker=request.ticker, strategy=request.strategy,
                           as_of=pd.Timestamp(_AS_OF), event_date=request.event_date,
                           session=request.session, spot=100.0, exp_pnl_sim=0.01)


def _score_stage(monkeypatch, tmp_path, population, *, events=None):
    """Run the real ``_action_score`` (so the real ``score_calendar`` and population check) on
    synthetic events; only the scorer, the feature load and the strike ladder are stubbed."""
    import engine.dashboard.nightly as nightly_module
    import engine.features as features_module
    import engine.score as score_module
    from engine.v2.ops import legacy_adapter

    frame = _score_events() if events is None else events
    monkeypatch.setattr(score_module, "Scorer", _FakeScorer)
    monkeypatch.setattr(score_module.store, "read_table", lambda *a, **k: frame.copy())
    monkeypatch.setattr(features_module.FeatureContext, "load", staticmethod(lambda t, years: object()))
    monkeypatch.setattr(nightly_module, "strike_ladder", lambda *a, **k: [])
    monkeypatch.setattr(
        legacy_adapter, "_check_features_current", lambda root, parameters=None: None
    )
    (tmp_path / "finality.json").write_text(json.dumps({
        "date": _AS_OF, "is_final": True, "market_wide": True, "daily_share": 1.0,
        "chain_share": 1.0, "covered": 1, "detail": "final"}))
    legacy_adapter._action_score({
        "tickers": ["AAA", "BBB", "CCC", "DDD"], "year_start": 2026, "year_end": 2027,
        "session": _AS_OF, "horizon_days": 35, "alt_strikes": 1,
        "expected_population": tuple(population)}, tmp_path)
    return json.loads((tmp_path / "score.json").read_text())


def test_the_generated_population_is_what_the_score_stage_emits(env, monkeypatch, tmp_path):
    population, _, _ = _generate(env)

    document = _score_stage(monkeypatch, tmp_path, population)  # raises if the check fails

    observed = set(document["observed_population"])
    chooser = {key for key in observed if key.split("|")[1] == "DYN-SV"}
    assert chooser  # the real score_calendar did append chooser rows, which no plan can list
    assert observed - chooser == set(population)


def test_a_native_board_population_would_have_failed_the_score_stage(env, monkeypatch, tmp_path):
    from engine.v2.ops.errors import OpsError
    from engine.v2.ops.native_board_universe import _COVERED_STRATEGIES

    population, _, _ = _generate(env)
    native = [key for key in population if key.split("|")[1] in _COVERED_STRATEGIES]
    assert len(native) < len(population)  # the legacy set has CAL-P and CND-P on top

    with pytest.raises(OpsError, match="differs from planned"):
        _score_stage(monkeypatch, tmp_path, native)


def test_the_score_stage_still_refuses_chooser_rows_it_cannot_tie_to_a_planned_event(
        env, monkeypatch, tmp_path):
    from engine.v2.ops.errors import OpsError

    population, _, _ = _generate(env)
    without_one_event = [key for key in population if not key.startswith("BBB|")]

    with pytest.raises(OpsError, match="differs from planned") as raised:
        _score_stage(monkeypatch, tmp_path, without_one_event)

    assert "BBB|DYN-SV|2027-01-24" in raised.value.problem.details["unplanned"]


def test_a_planned_chooser_key_must_still_be_observed(env, monkeypatch, tmp_path):
    from engine.v2.ops.errors import OpsError

    population, _, _ = _generate(env)

    with pytest.raises(OpsError, match="differs from planned") as raised:
        _score_stage(monkeypatch, tmp_path, [*population, "ZZZ|DYN-SV|2026-12-22"])

    assert raised.value.problem.details["missing"] == ["ZZZ|DYN-SV|2026-12-22"]
