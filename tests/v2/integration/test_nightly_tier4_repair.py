"""Automatic full Tier-4 repair when the incremental build leaves a gap (#494).

Contract: engine/dashboard/ARCHITECTURE.md § "Tier-4 gap repair". An
out-of-window ``tier4_gap_partial`` from the incremental Tier-4 build is no
longer just flagged — the tiers step immediately runs a full Tier-4 rebuild
(no ``--since``) and re-checks the gap, so a repaired night reports no flag
and ``steps.tiers.tier4_repair`` records the repair. The repair raising,
exceeding its wall-clock budget, or leaving the gap stops the run with a
typed code; a night with no gap skips the repair entirely.

Every rebuild here is a synthetic fake — no real Tier-3/Tier-4 rebuild runs.
Driving ``run_nightly`` end-to-end needs a full store, ledger and scorer, so
these unit-test ``_repair_tier4_gap`` (and the call-site guard) directly.
"""
# packages: engine.v2.dashboard

from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.dashboard.nightly import (
    NightlyReport,
    NightlyStop,
    _repair_tier4_gap,
    _repair_tier4_gap_if_needed,
)


def _report() -> NightlyReport:
    report = NightlyReport(as_of="2026-10-13")
    report.steps["tiers"] = {"rebuilt": ["panel", "tier4"],
                             "tier4_since": "2026-10-10"}
    return report


def _result(gap_fill: dict) -> SimpleNamespace:
    return SimpleNamespace(reports={"tier4": {"gap_fill": gap_fill}},
                           elapsed_s=12.0)


def _gapped() -> dict:
    return {"pred_abs_move": {"out_of_window_gap": [
        {"ticker": "AAPL", "event_date": "2026-10-01"}]}}


def test_planted_gap_runs_a_full_rebuild_and_records_the_repair():
    report = _report()
    calls = []

    def fake_rebuild_tables(tables, **kwargs):
        calls.append((tables, kwargs))
        return _result({})  # the repair closed the gap

    readings = iter([0, 7.5])
    _repair_tier4_gap(report, fake_rebuild_tables, clock=lambda: next(readings))

    assert calls == [(("tier4",), {"tier4_since": None})], \
        "panel was just rebuilt; the repair is Tier-4 only, with no --since"
    assert report.flags == [], "a repaired night carries no tier4_gap_partial"
    repair = report.steps["tiers"]["tier4_repair"]
    assert repair["ran"] is True
    assert repair["elapsed_s"] == 7.5
    assert "degraded" not in report.steps["tiers"]


def test_repair_raising_stops_the_run_with_a_typed_code():
    def boom(tables, **kwargs):
        raise RuntimeError("disk on fire")

    with pytest.raises(NightlyStop) as caught:
        _repair_tier4_gap(_report(), boom)

    assert caught.value.step == "tier4_repair"
    assert caught.value.detail.startswith(
        "TIER4_REPAIR_FAILED: RuntimeError: disk on fire")


def test_repair_that_still_leaves_the_gap_stops_the_run():
    def fake_rebuild_tables(tables, **kwargs):
        return _result(_gapped())

    with pytest.raises(NightlyStop) as caught:
        _repair_tier4_gap(_report(), fake_rebuild_tables)

    assert caught.value.step == "tier4_repair"
    assert caught.value.detail.startswith("TIER4_GAP_REMAINS: ")
    assert "pred_abs_move" in caught.value.detail


@pytest.mark.parametrize("elapsed", [99.0, 100.0, 101.0])
def test_repair_budget_is_measured_on_the_injected_clock(monkeypatch, elapsed):
    from engine.dashboard import nightly

    monkeypatch.setattr(nightly, "TIER4_REPAIR_BUDGET_S", 100)

    def fake_rebuild_tables(tables, **kwargs):
        return _result({})

    readings = iter([0, elapsed])
    report = _report()

    if elapsed > 100:
        with pytest.raises(NightlyStop) as caught:
            _repair_tier4_gap(report, fake_rebuild_tables, clock=lambda: next(readings))
        assert caught.value.step == "tier4_repair"
        assert caught.value.detail.startswith("TIER4_REPAIR_BUDGET: ")
    else:
        _repair_tier4_gap(report, fake_rebuild_tables, clock=lambda: next(readings))
        assert report.steps["tiers"]["tier4_repair"]["elapsed_s"] == elapsed


def test_a_night_with_no_gap_never_calls_rebuild_a_second_time():
    report = _report()
    calls = []

    def fake_rebuild_tables(tables, **kwargs):
        calls.append((tables, kwargs))
        return _result({})

    # The call-site guard from run_nightly's tiers block: the repair runs
    # only when the incremental build actually left an out-of-window gap.
    tier_result = fake_rebuild_tables(("panel", "tier4"), tier4_since="2026-10-10")
    _repair_tier4_gap_if_needed(report, tier_result, fake_rebuild_tables)

    assert len(calls) == 1, "no gap means no second (full) rebuild"
    assert "tier4_repair" not in report.steps["tiers"]


def test_an_incremental_result_with_a_gap_runs_the_repair():
    report = _report()
    calls = []

    def fake_rebuild_tables(tables, **kwargs):
        calls.append((tables, kwargs))
        return _result({})  # the repair closed the gap

    _repair_tier4_gap_if_needed(report, _result(_gapped()), fake_rebuild_tables)

    assert calls == [(("tier4",), {"tier4_since": None})], \
        "the repair is Tier-4 only, with no --since"
    assert report.steps["tiers"]["tier4_repair"]["ran"] is True
    assert report.flags == []


def test_rerunning_the_guard_after_a_repair_does_nothing():
    report = _report()
    calls = []

    def fake_rebuild_tables(tables, **kwargs):
        calls.append((tables, kwargs))
        return _result({})

    _repair_tier4_gap_if_needed(report, _result(_gapped()), fake_rebuild_tables)
    _repair_tier4_gap_if_needed(report, _result({}), fake_rebuild_tables)

    assert len(calls) == 1, "the rerun's gap-free result must not repair again"
    assert report.steps["tiers"]["tier4_repair"]["ran"] is True


def test_repair_call_site_sits_after_the_rebuild_and_outside_its_degraded_handler():
    from pathlib import Path

    from engine.dashboard import nightly

    source = Path(nightly.__file__).read_text()
    block = source.split("# -- 2b. Tier 3 and Tier 4", 1)[1]
    block = block.split("# -- earnings-date changes", 1)[0]

    call = "_repair_tier4_gap_if_needed(report, tier_result, rebuild_tables)"
    assert call in block
    call_at = block.index(call)
    assert call_at > block.index('rebuild_tables(("panel", "tier4")')
    assert call_at < block.index("_panel_staleness_flags(as_of)")
    assert call_at > block.index('"kind": "tiers_degraded"')
