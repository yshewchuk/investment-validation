"""Spec s4b Change 5: the native forward-calendar store's pure planning.

A fake ``Fetcher``-shaped stub and hand-built ``RefreshUnit`` fixtures -- no
network, no catalog. The session tie-break is compared against the untouched
legacy ``engine.calendar.SESSION_PRIORITY`` module as an oracle.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.calendar import SESSION_PRIORITY
from engine.v2.contracts import SnapshotRef
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
    monkeypatch.setattr(forward_calendar_store, "daily_by_ticker",
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
    monkeypatch.setattr(forward_calendar_store, "daily_by_ticker",
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
    _refused(tmp_path, attempt_id="")


def test_attempt_id_non_str_is_refused_before_any_io(tmp_path):
    _refused(tmp_path, attempt_id=123)


@pytest.mark.parametrize("bad_fence", [True, 0, -1, 1.5, "1"])
def test_fence_is_refused_before_any_io(tmp_path, bad_fence):
    _refused(tmp_path, fence=bad_fence)
