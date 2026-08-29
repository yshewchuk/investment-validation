"""Throttle, backoff, quota guard, and the single-process Polygon lock."""
from __future__ import annotations

import os

import pytest

from engine.data.throttle import (
    ORATS_RESERVE_FLOOR,
    SOURCES,
    PolygonBusy,
    QuotaExhausted,
    SourceConfig,
    Throttle,
    polygon_lock,
)


class FakeClock:
    """Deterministic time, so pacing is asserted rather than waited out."""

    def __init__(self):
        self.now = 1000.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def throttle(clock):
    return Throttle(sleep_fn=clock.sleep, time_fn=clock.time)


class TestConfiguration:
    def test_polygon_carries_the_six_and_a_half_second_gate(self):
        # ~10 req/min on this plan; the gate is what kept multi-hour pulls alive.
        assert SOURCES["polygon"].min_interval == pytest.approx(6.5)
        assert SOURCES["polygon"].backoff_base == pytest.approx(65.0)
        assert SOURCES["polygon"].max_retries == 6

    def test_only_polygon_is_single_process(self):
        single = {n for n, c in SOURCES.items() if c.single_process}
        assert single == {"polygon"}

    def test_orats_reserves_calls_for_live_operations(self):
        assert SOURCES["orats"].monthly_quota == 20_000
        assert SOURCES["orats"].reserve_floor == ORATS_RESERVE_FLOOR == 3_000

    def test_orats_timeout_covers_batch_pulls(self):
        # 10 big tickers is ~43k rows / ~2.8 MB / ~10s; the guide requires >=120s.
        assert SOURCES["orats"].timeout >= 120

    def test_unknown_source_is_rejected(self, throttle):
        with pytest.raises(KeyError, match="unknown source"):
            throttle.config("bloomberg")


class TestPacing:
    def test_the_first_call_does_not_wait(self, throttle):
        assert throttle.acquire("polygon") == 0.0

    def test_consecutive_calls_are_spaced_by_the_gate(self, throttle, clock):
        throttle.acquire("polygon")
        waited = throttle.acquire("polygon")
        assert waited == pytest.approx(6.5)
        assert clock.slept == [pytest.approx(6.5)]

    def test_time_already_elapsed_counts_against_the_gate(self, throttle, clock):
        throttle.acquire("polygon")
        clock.now += 4.0  # work happened between calls
        assert throttle.acquire("polygon") == pytest.approx(2.5)

    def test_no_wait_once_the_interval_has_passed(self, throttle, clock):
        throttle.acquire("polygon")
        clock.now += 10.0
        assert throttle.acquire("polygon") == 0.0

    def test_sources_are_paced_independently(self, throttle):
        throttle.acquire("polygon")
        assert throttle.acquire("orats") == 0.0

    def test_orats_is_paced_lightly_because_quota_binds_not_rate(self, throttle):
        throttle.acquire("orats")
        assert throttle.acquire("orats") == pytest.approx(0.3)


class TestBackoff:
    def test_backoff_grows_with_the_attempt(self, throttle, clock):
        first = throttle.backoff("polygon", 0)
        second = throttle.backoff("polygon", 1)
        assert first == pytest.approx(65.0)
        assert second == pytest.approx(130.0)
        assert second > first

    def test_backoff_counts_as_a_call_for_pacing(self, throttle):
        throttle.backoff("polygon", 0)
        assert throttle.acquire("polygon") == pytest.approx(6.5)


class TestQuotaGuard:
    def test_ample_quota_passes(self, throttle):
        throttle.check_quota("orats", 12_000)

    def test_spending_into_the_reserve_is_refused(self, throttle):
        with pytest.raises(QuotaExhausted, match="reserve"):
            throttle.check_quota("orats", 500)

    def test_the_boundary_is_the_floor_itself(self, throttle):
        throttle.check_quota("orats", ORATS_RESERVE_FLOOR)  # exactly at floor: allowed
        with pytest.raises(QuotaExhausted):
            throttle.check_quota("orats", ORATS_RESERVE_FLOOR - 1)

    def test_the_reserve_can_be_spent_deliberately(self, throttle, monkeypatch):
        monkeypatch.setenv("ORATS_ALLOW_RESERVE", "1")
        throttle.check_quota("orats", 10)

    def test_an_unknown_quota_does_not_block(self, throttle):
        # ORATS omits quota headers on CDN-cached responses; the guard cannot
        # treat "unknown" as "exhausted" without stalling every such run.
        throttle.check_quota("orats", None)

    def test_unmetered_sources_are_never_blocked(self, throttle):
        throttle.check_quota("polygon", 0)
        throttle.check_quota("yfinance", 0)


class TestPolygonLock:
    def test_the_lock_is_held_and_released(self, tmp_path):
        path = tmp_path / "poly.lock"
        with polygon_lock(path):
            assert path.exists()
        assert not path.exists()

    def test_a_live_holder_blocks_a_second_process(self, tmp_path):
        # Two Polygon processes split one 10/min budget and both stall.
        path = tmp_path / "poly.lock"
        path.write_text(str(os.getpid() + 0))  # our own pid reads as "not another"
        path.write_text("1")  # pid 1 always exists
        with pytest.raises(PolygonBusy, match="one at a time"):
            with polygon_lock(path):
                pass

    def test_a_stale_lock_is_taken_over(self, tmp_path):
        # A WSL2 host that sleeps mid-job leaves locks behind; a lock nobody can
        # break is worse than no lock at all.
        path = tmp_path / "poly.lock"
        path.write_text("999999")  # a pid that does not exist
        with polygon_lock(path):
            assert path.read_text() == str(os.getpid())
        assert not path.exists()

    def test_an_unparseable_lock_is_taken_over(self, tmp_path):
        path = tmp_path / "poly.lock"
        path.write_text("not-a-pid")
        with polygon_lock(path):
            assert path.exists()


class TestCustomConfigs:
    def test_a_caller_supplied_config_is_honoured(self, clock):
        throttle = Throttle(
            {"test": SourceConfig("test", min_interval=2.0, backoff_base=5.0,
                                  max_retries=2, timeout=10.0)},
            sleep_fn=clock.sleep,
            time_fn=clock.time,
        )
        throttle.acquire("test")
        assert throttle.acquire("test") == pytest.approx(2.0)
