"""Spec s4b Change 5: the native forward-calendar store's pure planning.

A fake ``Fetcher``-shaped stub and hand-built ``RefreshUnit`` fixtures -- no
network, no catalog. The session tie-break is compared against the untouched
legacy ``engine.calendar.SESSION_PRIORITY`` module as an oracle.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from engine.calendar import SESSION_PRIORITY
from engine.v2.contracts import KeyPredicate, SnapshotRef, TableContract, TableContractRef
from engine.v2.data import generic_incremental
from engine.v2.data.catalog import commit_snapshot as data_commit_snapshot
from engine.v2.data.legacy_mapping import build_legacy_mapping
from engine.v2.data.manifests import dataset_manifest, snapshot_ref
from engine.v2.data.objects import inspect_fragment
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, content_hash, from_document
from engine.v2.ops import forward_calendar_store
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError
from engine.v2.ops.forward_calendar_store import (
    SESSION_BY_TIME,
    _fence_check_for,
    date_units,
    horizon_dates,
    plan_forward_calendar,
    resolve_session_claims,
    ticker_units,
)
from engine.v2.ops.incremental_data import classify_response
from engine.v2.ops.lifecycle import request_cancel
from tests.ops_support import catalog, enqueue_claim

AS_OF = "2026-09-18"


def _snapshot(snapshot_id="snap-parent"):
    return SnapshotRef(
        snapshot_id=snapshot_id, manifest_hash="sha256:" + "a" * 64,
        parent_snapshot_id=None, table_versions={}, calendar_version="cal-v1",
        source_priority_version="priority-v1", finality_receipt_refs=(),
        knowledge_mode_by_table={})


def _cached(unit):
    return classify_response(200, unit.expected_keys, returned_keys=unit.expected_keys,
                             request_id=unit.request_id, receipt_ref="cache:" + unit.request_id,
                             cache_hit=True)


def _seeded_parent(conn, store, clock, *, scope: str, daily_market_rows=None,
                   extra_event_rows=()):
    """A minimal, synthetic ``earnings_events`` base snapshot (no real
    ``data/`` dependency -- built the same way
    ``tests/test_v2_data_generic_incremental.py``'s
    ``test_generic_refresh_worker_commits_json_timestamp_rows`` does),
    committed at generation 0 -> 1, for exercising ``_commit_claims``
    directly against a real resolvable parent snapshot. ``daily_market_rows``,
    when provided (an empty sequence seeds a zero-row table), additionally
    seeds a real ``daily_market`` table pinned in the same snapshot.
    ``extra_event_rows``: additional ``earnings_events`` rows (year 2025) in
    the same fragment."""
    receipt_ref = content_hash({"fixture": "forward_calendar_fence_" + scope})

    def _seeded_table(contract, rows, partition):
        contract_ref = TableContractRef(contract_id=contract.contract_id,
                                        definition_hash=contract.definition_hash)
        base_bytes = generic_incremental._parquet_bytes(contract, rows)
        published = store.publish_bytes(base_bytes, schema_ref="parquet_fragment.v1.0")
        obj = generic_incremental.ObjectRef(kind="parquet_fragment",
                                            object_id=published.artifact_id,
                                            content_hash=published.content_hash,
                                            byte_size=published.byte_size)
        inspection = inspect_fragment(store, obj, contract, contract_ref, partition)
        # An empty fragment carries no key bounds, and a fragment record
        # requires a non-empty partition, so it commits with no record.
        records = ()
        if inspection.primary_key_min is not None:
            records = (generic_incremental.manifests.fragment_record(
                inspection, contract_ref, input_receipt_refs=(receipt_ref,),
                import_request_hash=receipt_ref),)
        manifest = dataset_manifest(
            contract_ref, records, knowledge_mode="reconstructed",
            coverage_receipt_refs=(receipt_ref,), availability_evidence_refs=())
        return contract, obj, records, manifest

    contract = from_document(TableContract, build_legacy_mapping()["tables"]["earnings_events"])
    base_row = {
        "event_id": "event-1", "ticker": "AAA", "event_date": datetime(2025, 1, 15),
        "year": 2025, "session": "AMC", "session_src": None, "annc_tod": None,
        "src_orats": True, "src_oquants": False, "src_nasdaq": False, "src_yfinance": False,
        "date_agree": True, "date_conflict": False, "updated_at": None,
        "event_cluster_id": None, "claim_count": None, "reconciliation": None,
    }
    tables = [_seeded_table(contract, (base_row, *extra_event_rows), "2025")]
    if daily_market_rows is not None:
        daily_contract = from_document(TableContract,
                                       build_legacy_mapping()["tables"]["daily_market"])
        daily = tuple(daily_market_rows)
        partition = ("/".join(str(daily[0][name]) for name in daily_contract.partition_columns)
                     if daily else "0")
        tables.append(_seeded_table(daily_contract, daily, partition))
    contracts = tuple(item[0] for item in tables)
    table_objects = tuple(item[1] for item in tables)
    records = sum((item[2] for item in tables), ())
    table_manifests = tuple(item[3] for item in tables)
    parent_ref = snapshot_ref(
        {item[0].table_name: item[3] for item in tables}, calendar_version="cal.v1",
        source_priority_version="fixture", finality_receipt_refs=(receipt_ref,))
    data_commit_snapshot(
        conn, scope=scope, request_hash=content_hash({"base": scope}),
        contracts=contracts, objects=table_objects, records=records, manifests=table_manifests,
        snapshot=parent_ref, expected_head_snapshot_id=None, expected_head_generation=0,
        receipt_id="base-receipt-" + scope, attempt_id="base-attempt-" + scope, fence=1,
        fence_check=lambda _conn: None, clock=clock, store=store)
    return Repository(conn, store).resolve_full(parent_ref.snapshot_id)


def test_commit_claims_refuses_a_cancelled_attempt_and_does_not_move_the_head(tmp_path):
    """Runner-level regression (CodeRabbit, PR #55 round 1): ``_commit_claims``
    -- the function ``run_forward_calendar_refresh`` actually calls to commit
    -- must itself refuse a cancelled attempt's fence, and the snapshot head
    row must not change. Complements, without duplicating, the direct
    ``_fence_check_for`` unit tests below: this proves the runner's OWN
    commit path actually wires its ``fence_check`` into
    ``generic_incremental.commit_generic_table_candidate``, not merely that
    the helper works in isolation."""
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    scope = "fwd-cal-fence-test"
    parent = _seeded_parent(conn, store, clock, scope=scope)
    claim = enqueue_claim(conn, clock, supervisor)
    request_cancel(conn, claim.job_id, claim.attempt_id, clock=clock)
    before = conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = ?",
        (scope,)).fetchone()

    claims = {("BBB", "2026-10-01"): {"nasdaq": "BMO"}}
    with pytest.raises(OpsError) as err:
        forward_calendar_store._commit_claims(
            conn, store, parent, claims, {}, scope=scope, clock=clock,
            expected_head_generation=1, expected_head_snapshot_id=parent.snapshot.snapshot_id,
            attempt_id=claim.attempt_id, fence=claim.fence)
    assert err.value.code == "CANCELLED"

    after = conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = ?",
        (scope,)).fetchone()
    assert tuple(after) == tuple(before)


def test_fence_check_for_matches_the_real_verify_fence_and_keeps_the_lease_check(tmp_path):
    """``_fence_check_for`` must build a callable ``verify_fence`` accepts
    with its REAL signature (``conn, attempt_id, fence, now`` -- no
    ``check_lease_time`` keyword). It must also still enforce the production
    wall-clock lease-expiry check -- never a skipped check, no matter who
    calls this store (issue #52, mirroring
    ``computed_moves_store``'s own identical test)."""
    conn, clock, supervisor = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, supervisor)
    check = _fence_check_for(claim.attempt_id, claim.fence, clock)

    with transaction(conn):
        job, attempt = check(conn)
        assert job["fence"] == claim.fence
        assert attempt["fence"] == claim.fence

    clock.advance(10 ** 6)  # long past any lease_expires_at
    with transaction(conn):
        with pytest.raises(OpsError) as err:
            check(conn)
        assert err.value.code == "LEASE_LOST"


def test_fence_check_for_refuses_a_cancelled_attempt(tmp_path):
    """A job whose cancellation invalidates the fence (``request_cancel``
    sets the job to ``cancelling``) must refuse the commit's fence check
    with ``CANCELLED``, even though the snapshot head has not moved (issue
    #52's exact scenario: the head is unchanged, but the issuing attempt's
    lease is no longer live)."""
    conn, clock, supervisor = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, supervisor)
    request_cancel(conn, claim.job_id, claim.attempt_id, clock=clock)
    check = _fence_check_for(claim.attempt_id, claim.fence, clock)

    with transaction(conn):
        with pytest.raises(OpsError) as err:
            check(conn)
        assert err.value.code == "CANCELLED"


def test_fence_check_for_with_no_staged_attempt_is_a_noop():
    """No staged attempt (e.g. a manual/ad-hoc invocation with nothing to
    fence against): the returned callable does nothing and returns
    ``None``, matching ``computed_moves_store``'s own identical contract."""
    check = _fence_check_for(None, None, clock=None)
    assert check(object()) is None


def test_claims_carry_session_priority_same_as_legacy():
    """Two conflicting forward claims: Nasdaq says BMO, yfinance says AMC.
    The winner is what the untouched legacy SESSION_PRIORITY oracle picks."""
    claims = {"nasdaq": "BMO", "yfinance": "AMC"}
    expected_src = next(name for name in SESSION_PRIORITY if claims.get(name))
    assert resolve_session_claims(claims) == ("AMC", expected_src)

    # ORATS, once the event is past, still beats both forward sources.
    assert resolve_session_claims({"orats": "BMO", "yfinance": "AMC",
                                   "nasdaq": "AMC"}) == ("BMO", "orats")
    # Unknown-or-missing sessions fall through in the same order.
    assert resolve_session_claims({"nasdaq": "BMO"}) == ("BMO", "nasdaq")
    assert resolve_session_claims({}) == (None, None)
    assert set(SESSION_BY_TIME.values()) == {"BMO", "AMC"}


def test_merged_row_marks_a_sessionless_nasdaq_claim_without_attributing_it():
    """Nasdaq listed the event but SESSION_BY_TIME resolved no session: the row
    still records src_nasdaq, and nasdaq never becomes the session source."""
    row = forward_calendar_store._merged_row(
        None, "AAPL", "2026-09-21", {"nasdaq": None},
        updated_at="2026-09-18T00:00:00+00:00")

    assert row["src_nasdaq"] is True
    assert row["session"] is None
    assert row["session_src"] is None


def test_horizon_dates_falls_back_to_weekdays_without_a_calendar():
    days = horizon_dates(pd.Timestamp("2026-09-18"), 7)
    assert [str(day.date()) for day in days] == [
        "2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24",
        "2026-09-25"]
    fake = type("Calendar", (), {"days": [pd.Timestamp("2026-09-21")]})()
    assert [str(day.date()) for day in horizon_dates("2026-09-18", 7, calendar=fake)] == [
        "2026-09-21"]


def test_provider_calls_go_through_the_shared_budget():
    dates = [pd.Timestamp("2026-09-21"), pd.Timestamp("2026-09-22"),
             pd.Timestamp("2026-09-23")]
    units = date_units(dates, as_of=AS_OF)
    nasdaq_cached = {units[0].request_id: _cached(units[0])}

    nasdaq, yfinance = plan_forward_calendar(
        _snapshot(), dates, ("AAPL",), as_of=AS_OF, cached_nasdaq=nasdaq_cached,
        cached_yfinance={})

    assert nasdaq.provider_calls == len(nasdaq.fetch_units) * nasdaq.max_attempts == 2 * 3
    assert nasdaq.provider_account == "nasdaq"
    assert yfinance.provider_calls == len(yfinance.fetch_units) * yfinance.max_attempts == 1 * 3
    assert yfinance.provider_account == "yfinance"


def test_unit_ids_carry_the_as_of_date():
    """Spec R4: a new session's units are new ids, so its receipts are refetched."""
    [day_unit] = date_units([pd.Timestamp("2026-09-21")], as_of=AS_OF)
    assert day_unit.request_id == "nasdaq:calendar/earnings:2026-09-21:" + AS_OF
    [ticker_unit] = ticker_units(["AAPL"], as_of=AS_OF)
    assert ticker_unit.request_id == "yfinance:earnings:AAPL:" + AS_OF
    [next_day] = date_units([pd.Timestamp("2026-09-21")], as_of="2026-09-19")
    assert next_day.request_id != day_unit.request_id


def test_native_calendar_fallback_is_recorded_as_a_warning(monkeypatch):
    """Spec s4c: the weekday fallback is job-result evidence, not only a log line."""
    monkeypatch.setattr(forward_calendar_store, "daily_sessions",
                        lambda repository, snapshot: {})
    parent = type("Parent", (), {"snapshot": object()})()
    calendar, warnings = forward_calendar_store._native_calendar(
        object(), parent, as_of=AS_OF, horizon_days=21)
    assert calendar is None
    assert len(warnings) == 1
    assert warnings[0].startswith("weekday calendar fallback:")
    assert "daily_market session" in warnings[0]


def test_native_calendar_extends_through_the_requested_horizon(monkeypatch):
    """A stale panel (an old last observed session) must not truncate a
    normal horizon: the projection extends through as_of + horizon_days even
    when that crosses last + 400 days."""
    old_last = pd.Timestamp("2020-01-02")
    frame = pd.DataFrame({"date": [old_last]})
    monkeypatch.setattr(forward_calendar_store, "daily_sessions",
                        lambda repository, snapshot: {"AAPL": frame})
    parent = type("Parent", (), {"snapshot": object()})()
    calendar, warnings = forward_calendar_store._native_calendar(
        object(), parent, as_of=AS_OF, horizon_days=21)
    assert warnings == ()
    horizon_end = pd.Timestamp(AS_OF).normalize() + pd.Timedelta(days=21)
    assert max(calendar.days) >= horizon_end


def test_no_provider_call_when_all_cached():
    """Every unit has a satisfying cached outcome: zero reserved calls and
    the fake fetcher is never touched (cache-first, not fetch-then-discard)."""
    dates = [pd.Timestamp("2026-09-21"), pd.Timestamp("2026-09-22"),
             pd.Timestamp("2026-09-23")]
    units = date_units(dates, as_of=AS_OF)
    cached = {unit.request_id: _cached(unit) for unit in units}

    nasdaq, _yfinance = plan_forward_calendar(
        _snapshot(), dates, (), as_of=AS_OF, cached_nasdaq=cached, cached_yfinance={})
    assert nasdaq.provider_calls == 0
    assert nasdaq.fetch_units == ()


def _poison_fetcher(*_args, **_kwargs):
    raise AssertionError("a provider fetcher must never be called before validation passes")


def _valid_kwargs(tmp_path):
    """A structurally valid call: ``catalog_path``/``objects_root`` are a
    REAL (but empty/unusable) file and directory, so those two checks pass
    and whichever single field a test overrides is the one that fails.
    Every other field is well-typed and well-shaped, and the fetchers
    explode if called. If validation runs before any I/O (as required), an
    invalid override raises OpsError(INVALID_REQUEST) without ever opening
    the catalog connection or calling a fetcher -- either of which would
    raise a DIFFERENT exception (sqlite3.OperationalError / AssertionError)
    instead, which pytest.raises(OpsError) below would not swallow.
    """
    catalog_path = tmp_path / "forward_calendar_catalog.db"
    catalog_path.write_bytes(b"")
    objects_root = tmp_path / "forward_calendar_objects"
    objects_root.mkdir()
    return dict(
        catalog_path=str(catalog_path),
        objects_root=str(objects_root),
        parent_snapshot_id="snap-parent",
        refresh_plan_hash="sha256:" + "a" * 64,
        as_of=AS_OF,
        tickers=("AAPL",),
        horizon_days=21,
        scope="shadow",
        expected_head_generation=0,
        nasdaq_fetcher=_poison_fetcher,
        earnings_fetcher=_poison_fetcher,
    )


def _refused(tmp_path, **overrides):
    """Run with one field overridden; assert INVALID_REQUEST, no I/O reached."""
    kwargs = dict(_valid_kwargs(tmp_path), **overrides)
    with pytest.raises(OpsError) as exc_info:
        forward_calendar_store.run_forward_calendar_refresh(**kwargs)
    assert exc_info.value.code == "INVALID_REQUEST"


def test_as_of_none_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, as_of=None)


def test_as_of_unparseable_string_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, as_of="not-a-date")


def test_as_of_bare_number_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, as_of=20260918)


def test_as_of_bool_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, as_of=True)


def test_as_of_nat_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, as_of=pd.NaT)


def test_as_of_timezone_aware_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, as_of=pd.Timestamp(AS_OF, tz="UTC"))


def test_tickers_bare_str_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, tickers="AAPL")


def test_tickers_non_iterable_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, tickers=123)


def test_tickers_empty_string_element_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, tickers=("AAPL", ""))


def test_horizon_days_non_int_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, horizon_days="21")


def test_horizon_days_bool_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, horizon_days=True)


def test_horizon_days_below_range_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, horizon_days=0)


def test_horizon_days_above_range_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, horizon_days=forward_calendar_store.MAX_HORIZON_DAYS + 1)


def test_scope_missing_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, scope=None)


def test_scope_outside_the_standard_namespaces_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, scope="production")


def test_expected_head_generation_missing_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, expected_head_generation=None)


def test_expected_head_generation_negative_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, expected_head_generation=-1)


def test_catalog_path_that_does_not_exist_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, catalog_path=str(tmp_path / "does_not_exist.db"))


def test_objects_root_that_does_not_exist_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, objects_root=str(tmp_path / "does_not_exist_dir"))


def test_parent_snapshot_id_empty_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, parent_snapshot_id="")


def test_refresh_plan_hash_malformed_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, refresh_plan_hash="not-a-sha256-hash")


def test_expected_head_snapshot_id_empty_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, expected_head_snapshot_id="")


def test_expected_head_snapshot_id_too_long_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, expected_head_snapshot_id="s" * 129)


def test_attempt_id_empty_string_is_refused_before_any_io(tmp_path):
    """An empty ``attempt_id`` is refused, not silently treated as ``None``."""
    _refused(tmp_path, attempt_id="")


def test_attempt_id_non_str_is_refused_before_any_io(tmp_path):
    """A non-``str`` ``attempt_id`` is refused before any I/O."""
    _refused(tmp_path, attempt_id=123)


@pytest.mark.parametrize("bad_fence", [True, 0, -1, 1.5, "1"])
def test_fence_is_refused_before_any_io(tmp_path, bad_fence):
    """``fence`` must be a real ``int >= 1``: a ``bool`` (an ``int`` subclass
    in Python), zero, negative, a float, and a numeric string are each
    refused before any I/O."""
    _refused(tmp_path, fence=bad_fence)


def test_fence_set_without_attempt_id_is_refused_before_any_io(tmp_path):
    """Opus gate finding on PR #55: a bare ``fence`` with no ``attempt_id``
    would make ``_fence_check_for`` a no-op (fail-open -- the commit goes
    through unfenced). Refused up front instead of silently no-op'ing."""
    _refused(tmp_path, fence=3)


def test_attempt_id_set_without_fence_is_refused_before_any_io(tmp_path):
    """Opus gate finding on PR #55: a bare ``attempt_id`` with no ``fence``
    would previously only fail later, inside ``verify_fence``, after the
    network fetch. Refused up front instead."""
    _refused(tmp_path, attempt_id="attempt-1")


def test_attempt_fence_pair_both_none_is_valid_the_legacy_default():
    """Both ``None`` is the default every legacy/no-live-job caller uses --
    it must not be refused (Opus gate finding on PR #55)."""
    forward_calendar_store._validated_attempt_fence_pair(None, None)


def test_attempt_fence_pair_both_set_is_valid():
    """Both set (a live job attempt fencing the commit) is the other valid
    shape."""
    forward_calendar_store._validated_attempt_fence_pair("attempt-1", 3)


def _capture_scan_queries(monkeypatch, repository):
    captured = []
    original_scan = repository.scan

    def _scan(query, **kwargs):
        captured.append(query)
        return original_scan(query, **kwargs)

    monkeypatch.setattr(repository, "scan", _scan)
    return captured


def _capture_population_scan(monkeypatch, repository, population_bound):
    monkeypatch.setattr(
        repository, "scan_population_bound", lambda *_a, **_k: population_bound)
    return _capture_scan_queries(monkeypatch, repository)


def _scan_fixture(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    parent = _seeded_parent(conn, store, clock, scope="fwd-cal-scan-bound")
    repository = Repository(conn, store)
    return repository, parent.snapshot


def test__scan_rows_keeps_limit_when_population_bound_is_at_or_above_current(tmp_path,
                                                                             monkeypatch):
    """A population bound at or above the current scan limit keeps that limit.
    With ``TableContract.maximum_result_rows`` gone, the request limit is the
    module's own ``MAX_SCAN_ROWS``, so it is pinned at the shared fixture's
    real recorded row count: a bound above the repository's own pinned
    population is not a bound and is refused ``QUERY_NOT_BOUNDED``."""
    repository, snapshot = _scan_fixture(tmp_path)
    table_name = "earnings_events"
    columns = ("ticker", "event_date", "year", "src_orats")
    years = tuple(sorted({int(record.partition_key)
                          for record in repository.fragment_records(snapshot, table_name)}))
    pinned = repository.scan_population_bound(
        snapshot.snapshot_id, table_name=table_name,
        table_contract_ref=snapshot.table_versions[table_name].table_contract_ref,
        key_filter=(KeyPredicate(column="year", operator="in", values=years),),
        time_interval=None)
    monkeypatch.setattr(forward_calendar_store, "MAX_SCAN_ROWS", pinned)
    baseline = []
    for lease in forward_calendar_store._scan_rows(
            repository, snapshot, table_name, columns):
        with lease as batch:
            baseline.extend(batch)
    contract = repository.table_contract(snapshot, table_name)
    expected = forward_calendar_store.MAX_SCAN_ROWS
    captured = _capture_population_scan(monkeypatch, repository, expected)

    result = []
    for lease in forward_calendar_store._scan_rows(
            repository, snapshot, table_name, columns):
        with lease as batch:
            result.extend(batch)

    assert result == baseline
    assert result == [{"ticker": "AAA", "event_date": datetime(2025, 1, 15),
                       "year": 2025, "src_orats": True}]
    assert captured[0].max_result_rows == expected
    assert captured[0].max_batch_rows == min(contract.maximum_batch_rows, 50_000, expected)


def test__scan_rows_uses_smaller_selected_population_bound(tmp_path, monkeypatch):
    """PR 360 positive case: the REAL selected-fragment bound (no mock) --
    the sum of the recorded row counts of the fragments the scan's year
    predicate selects -- is below the existing limit, and the prepared query
    carries exactly ``min(existing_limit, bound)`` with its batch limit kept
    at or below the result limit."""
    repository, snapshot = _scan_fixture(tmp_path)
    table_name = "earnings_events"
    columns = ("ticker", "event_date", "year", "src_orats")
    contract = repository.table_contract(snapshot, table_name)
    existing = forward_calendar_store.MAX_SCAN_ROWS
    years = tuple(sorted({int(record.partition_key)
                          for record in repository.fragment_records(snapshot, table_name)}))
    bound = repository.scan_population_bound(
        snapshot.snapshot_id, table_name=table_name,
        table_contract_ref=snapshot.table_versions[table_name].table_contract_ref,
        key_filter=(KeyPredicate(column="year", operator="in", values=years),),
        time_interval=None)
    assert 0 < bound < existing
    baseline = []
    for lease in forward_calendar_store._scan_rows(
            repository, snapshot, table_name, columns):
        with lease as batch:
            baseline.extend(batch)
    captured = _capture_scan_queries(monkeypatch, repository)

    result = []
    for lease in forward_calendar_store._scan_rows(
            repository, snapshot, table_name, columns):
        with lease as batch:
            result.extend(batch)

    assert result == baseline
    assert result == [{"ticker": "AAA", "event_date": datetime(2025, 1, 15),
                       "year": 2025, "src_orats": True}]
    assert captured[0].max_result_rows == min(existing, bound)
    assert captured[0].max_batch_rows <= captured[0].max_result_rows
    assert captured[0].max_batch_rows == min(contract.maximum_batch_rows, 50_000,
                                             min(existing, bound))


def test_daily_by_ticker_matches_old_scan_and_bounds_live_rows(tmp_path, monkeypatch):
    """Retention parity over a real pinned ``daily_market``: the reader-backed
    ``_scan_rows`` returns exactly the rows the PRE-PR materializing scan
    returned -- the pinned snapshot's own contract ref, every represented year
    partition, the selected population bound as the result limit and the
    primary-key order, nulls and the populated corrected ``implied_move``
    included -- while ``daily_by_ticker`` converts each real lease only while
    it is charged: no batch is retained past ``MAX_SCAN_ROWS``, each run's
    account peaks at exactly the cap and returns to zero, and a second run
    serializes byte-identical per-ticker frames (R6)."""
    from engine.v2.contracts import DataQuery
    from engine.v2.ops import pinned_partition_reader

    daily_contract = from_document(TableContract,
                                   build_legacy_mapping()["tables"]["daily_market"])
    defaults = {column.name: None for column in daily_contract.columns if column.nullable}

    def _daily_row(ticker, day, implied_move):
        row = dict(defaults)
        row.update(ticker=ticker, date=day, year=2026, implied_move=implied_move)
        return row

    rows = (
        _daily_row("AAA", datetime(2026, 1, 2), None),
        _daily_row("AAA", datetime(2026, 3, 16), 4.5),
        _daily_row("AAA", datetime(2026, 6, 15), 3.0),
        _daily_row("AAA", datetime(2026, 12, 31), None),
        _daily_row("BBB", datetime(2026, 1, 2), 7.0),
        _daily_row("BBB", datetime(2026, 6, 15), None),
        _daily_row("BBB", datetime(2026, 12, 31), 1.25),
    )
    conn, clock, _supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    parent = _seeded_parent(conn, store, clock, scope="fwd-cal-retention-parity",
                            daily_market_rows=rows)
    repository = Repository(conn, store)
    snapshot = parent.snapshot

    # The PRE-PR ``_scan_rows`` query, rebuilt straight from the repository:
    # the pinned contract ref, every represented sorted year partition, the
    # selected population bound (never this run's lowered retention guard) and
    # the table's primary-key order, flattened from real ``repository.scan``
    # batches -- never the changed ``_scan_rows`` and never a mocked scan.
    table_name = "daily_market"
    contract = repository.table_contract(snapshot, table_name)
    contract_ref = snapshot.table_versions[table_name].table_contract_ref
    key_filter = (KeyPredicate(column="year", operator="in", values=tuple(sorted(
        {int(record.partition_key)
         for record in repository.fragment_records(snapshot, table_name)}))),)
    bound = repository.scan_population_bound(
        snapshot.snapshot_id, table_name=table_name, table_contract_ref=contract_ref,
        key_filter=key_filter, time_interval=None)
    old_max_scan_rows = forward_calendar_store.MAX_SCAN_ROWS

    def old_scan(columns):
        max_result_rows = min(old_max_scan_rows, bound)
        max_batch_rows = min(contract.maximum_batch_rows, 50_000)
        if max_result_rows > 0:
            max_batch_rows = min(max_batch_rows, max_result_rows)
        query = DataQuery(
            snapshot_id=snapshot.snapshot_id, table_contract_ref=contract_ref,
            columns=tuple(columns), key_filter=key_filter,
            order_by=tuple(contract.primary_key),
            max_batch_rows=max_batch_rows,
            max_result_rows=max_result_rows)
        flat = []
        for batch in repository.scan(query, table_name=table_name):
            flat.extend(batch.to_pylist())
        return flat

    # With the retained cap at 2 the streamed read still yields all seven rows:
    # exact list equality with the old scan proves the rewrite changed nothing
    # about what a caller sees, and 7 > cap proves the cap is no longer a
    # result limit.
    monkeypatch.setattr(forward_calendar_store, "MAX_SCAN_ROWS", 2)
    collected = []
    for lease in forward_calendar_store._scan_rows(
            repository, snapshot, table_name, ("ticker", "date", "implied_move")):
        with lease as batch:
            assert lease.released is False
            collected.extend(batch)
        assert lease.released is True
    assert collected == old_scan(("ticker", "date", "implied_move"))
    assert len(collected) == 7 > forward_calendar_store.MAX_SCAN_ROWS

    # ``daily_by_ticker`` over the same pin: a fresh per-run account minted by
    # the real shared-reader ``RetainedRowCount`` (captured as it is created),
    # and a pandas proxy that records every conversion and asserts it happened
    # while exactly that live batch was charged, at or under the retained cap.
    real_counter = pinned_partition_reader.RetainedRowCount
    accounts: list = []
    charged: dict = {}

    def _fresh_counter():
        account = real_counter()
        accounts.append(account)
        charged["current"] = account
        return account

    conversions: list = []

    class _ProxyPandas:
        def __getattr__(self, name):
            return getattr(pd, name)

        def DataFrame(self, data=None, *args, **kwargs):
            account = charged["current"]
            assert isinstance(data, list), type(data)
            assert account.live_rows == len(data) <= forward_calendar_store.MAX_SCAN_ROWS
            conversions.append(len(data))
            return pd.DataFrame(data, *args, **kwargs)

    monkeypatch.setattr(pinned_partition_reader, "RetainedRowCount", _fresh_counter)
    monkeypatch.setattr(forward_calendar_store, "pd", _ProxyPandas())

    expected_frame = pd.DataFrame(old_scan(("ticker", "date")))
    expected_frame["date"] = pd.to_datetime(expected_frame["date"])
    expected = {str(ticker): group
                for ticker, group in expected_frame.groupby("ticker")}

    before = len(accounts)
    first = forward_calendar_store.daily_by_ticker(repository, snapshot)
    [first_account] = accounts[before:]
    assert set(first) == set(expected) == {"AAA", "BBB"}
    for ticker, group in expected.items():
        pd.testing.assert_frame_equal(first[ticker], group)
    first_bytes = {ticker: frame.to_json(orient="split", date_format="iso").encode()
                   for ticker, frame in first.items()}
    assert sum(conversions) == 7
    assert all(length <= 2 for length in conversions)
    assert first_account.peak_rows == 2
    assert first_account.live_rows == 0
    conversions.clear()

    before = len(accounts)
    repeat = forward_calendar_store.daily_by_ticker(repository, snapshot)
    [repeat_account] = accounts[before:]
    repeat_bytes = {ticker: frame.to_json(orient="split", date_format="iso").encode()
                    for ticker, frame in repeat.items()}
    for ticker, frame in first.items():
        pd.testing.assert_frame_equal(repeat[ticker], frame)
    assert repeat_bytes == first_bytes
    assert sum(conversions) == 7
    assert all(length <= 2 for length in conversions)
    assert repeat_account.peak_rows == 2
    assert repeat_account.live_rows == 0


def test_daily_by_ticker_preserves_typed_refusals_and_discards_partial_batches(tmp_path,
                                                                               monkeypatch):
    """Typed refusals and partial-read discipline over two real pinned
    snapshots (R1-R5). An events-only pin carries no ``daily_market``:
    ``daily_by_ticker`` refuses with the repository's own ``CONTRACT_MISMATCH``
    missing-table code, never an empty grouped frame or a newer source. Over a
    ``daily_market`` pin whose scan dies on a pre-created ``MANIFEST_CORRUPT``
    after its first real one-row batch, the identical error object propagates
    terminal (exactly one scan, no retry), the one query is the pinned
    snapshot's own id, its contract ref and its year selection, the first
    lease is charged (``peak_rows == 1``) and every provisional lease row is
    discharged on unwind (``live_rows == 0``) with nothing returned."""
    from engine.v2.data import errors
    from engine.v2.ops import pinned_partition_reader

    missing_root = tmp_path / "missing-pin"
    missing_root.mkdir()
    missing_repository, missing_snapshot = _scan_fixture(missing_root)
    with pytest.raises(errors.DataError) as missing:
        forward_calendar_store.daily_by_ticker(missing_repository, missing_snapshot)
    assert missing.value.code == "CONTRACT_MISMATCH"
    assert missing.value.problem.details == {"table_name": "daily_market"}

    daily_contract = from_document(TableContract,
                                   build_legacy_mapping()["tables"]["daily_market"])
    defaults: dict = {}
    for column in daily_contract.columns:
        if column.name in ("ticker", "date", "year"):
            continue
        if column.nullable:
            defaults[column.name] = None
        elif column.physical_type == "string":
            defaults[column.name] = "synthetic"
        elif column.physical_type == "float64":
            defaults[column.name] = 1.0
        elif column.physical_type == "int64":
            defaults[column.name] = 1
        elif column.physical_type == "bool":
            defaults[column.name] = False
        elif column.physical_type.startswith("timestamp["):
            defaults[column.name] = datetime(2026, 1, 2)
        else:
            raise AssertionError(column.physical_type)

    def _daily_row(day):
        row = dict(defaults)
        row.update(ticker="AAA", date=day, year=2026)
        return row

    rows = (_daily_row(datetime(2026, 1, 2)), _daily_row(datetime(2026, 3, 16)),
            _daily_row(datetime(2026, 6, 15)))
    mid_root = tmp_path / "mid-scan-pin"
    mid_root.mkdir()
    conn, clock, _supervisor = catalog(mid_root)
    store = ArtifactStore(mid_root / "objects")
    parent = _seeded_parent(conn, store, clock, scope="fwd-cal-typed-refusals",
                            daily_market_rows=rows)
    mid_repository = Repository(conn, store)
    mid_snapshot = parent.snapshot

    incompatible = errors.fail("CONTRACT_MISMATCH", "synthetic incompatible pinned contract")
    real_table_contract = mid_repository.table_contract

    def _incompatible_pin(snapshot, table_name):
        if table_name == "daily_market":
            raise incompatible
        return real_table_contract(snapshot, table_name)

    monkeypatch.setattr(mid_repository, "table_contract", _incompatible_pin)
    incompatible_result = None
    with pytest.raises(errors.DataError) as incompatible_refusal:
        incompatible_result = forward_calendar_store.daily_by_ticker(
            mid_repository, mid_snapshot)
    assert incompatible_refusal.value is incompatible
    assert incompatible_result is None
    monkeypatch.setattr(mid_repository, "table_contract", real_table_contract)

    scan_error = errors.fail("MANIFEST_CORRUPT", "synthetic mid-scan integrity refusal")
    queries: list = []
    real_scan = mid_repository.scan

    def _scan(query, **kwargs):
        queries.append(query)
        for batch in real_scan(query, **kwargs):
            yield batch
            raise scan_error

    monkeypatch.setattr(mid_repository, "scan", _scan)

    real_counter = pinned_partition_reader.RetainedRowCount
    accounts: list = []

    def _fresh_counter():
        account = real_counter()
        accounts.append(account)
        return account

    monkeypatch.setattr(pinned_partition_reader, "RetainedRowCount", _fresh_counter)
    monkeypatch.setattr(forward_calendar_store, "MAX_SCAN_ROWS", 1)

    before = len(accounts)
    result = None
    with pytest.raises(errors.DataError) as refused:
        result = forward_calendar_store.daily_by_ticker(mid_repository, mid_snapshot)
    [account] = accounts[before:]
    [query] = queries
    assert refused.value is scan_error
    assert result is None
    assert len(queries) == 1
    assert query.snapshot_id == mid_snapshot.snapshot_id
    assert (query.table_contract_ref
            == mid_snapshot.table_versions["daily_market"].table_contract_ref)
    assert query.key_filter == (KeyPredicate(column="year", operator="in", values=(2026,)),)
    assert account.peak_rows == 1
    assert account.live_rows == 0


def test_daily_by_ticker_skips_an_empty_lease(monkeypatch):
    """A genuine reader lease can carry zero rows: ``daily_by_ticker`` must
    skip it instead of appending a columnless ``pd.DataFrame`` chunk -- a
    ``concat`` of which has no ``date`` column to convert, dying with
    ``KeyError("date")``. The lease itself is real (the shared reader mints
    it against a fresh account); only ``_scan_rows`` is patched, with opaque
    sentinels standing in for the repository and snapshot it never touches.
    """
    from engine.v2.ops.pinned_partition_reader import RetainedBatch, RetainedRowCount

    account = RetainedRowCount()
    lease = RetainedBatch([], account)

    def _scan_rows(_repository, _snapshot, _table_name, _columns):
        yield lease

    monkeypatch.setattr(forward_calendar_store, "_scan_rows", _scan_rows)

    grouped = forward_calendar_store.daily_by_ticker(object(), object())

    assert grouped == {}
    assert lease.released is True
    assert account.live_rows == 0


def test_daily_by_ticker_returns_empty_for_empty_pinned_history(tmp_path, monkeypatch):
    """A valid pinned ``daily_market`` with zero represented year partitions
    keeps its empty result empty: the pin is real (the snapshot carries its
    own ``daily_market`` version and contract ref), but with no fragment
    record at all the reader's year selection is empty and yields no lease,
    so the real, unpatched ``daily_by_ticker`` returns ``{}`` while the
    fresh shared-reader account it minted is never charged -- ``peak_rows``
    and ``live_rows`` both stay zero. Empty input, not a refusal: a missing
    table still raises ``CONTRACT_MISMATCH`` (the test above)."""
    from engine.v2.ops import pinned_partition_reader

    conn, clock, _supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    parent = _seeded_parent(conn, store, clock, scope="fwd-cal-empty-history",
                            daily_market_rows=())
    repository = Repository(conn, store)
    snapshot = parent.snapshot
    assert "daily_market" in snapshot.table_versions
    assert repository.fragment_records(snapshot, "daily_market") == ()

    real_counter = pinned_partition_reader.RetainedRowCount
    accounts: list = []

    def _fresh_counter():
        account = real_counter()
        accounts.append(account)
        return account

    monkeypatch.setattr(pinned_partition_reader, "RetainedRowCount", _fresh_counter)

    before = len(accounts)
    result = forward_calendar_store.daily_by_ticker(repository, snapshot)
    assert result == {}
    [account] = accounts[before:]
    assert account.peak_rows == 0
    assert account.live_rows == 0
