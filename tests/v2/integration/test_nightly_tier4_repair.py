"""Automatic full Tier-4 repair when the incremental build leaves a gap (#494).

Contract: engine/dashboard/ARCHITECTURE.md § "Tier-4 gap repair". An
out-of-window ``tier4_gap_partial`` from the incremental Tier-4 build is no
longer just flagged — the tiers step immediately runs a full Tier-4 rebuild
(no ``--since``) and re-checks the gap, so a repaired night reports no flag
and ``steps.tiers.tier4_repair`` records the repair. The repair raising,
exceeding its wall-clock budget, or leaving the gap stops the run with a
typed code; a night with no gap skips the repair entirely.

The snapshot is deferred: the incremental rebuild runs with
``finalize=False`` and the snapshot is written only once the gap check passes
(or the full repair finalizes it). A night whose repair is refused therefore
leaves no snapshot for the native import to read.

Every rebuild here is a synthetic fake — no real Tier-3/Tier-4 rebuild runs.
The orchestration cases drive ``run_nightly`` with every heavy step patched.
"""
# packages: engine.v2.ops, engine.v2.data

from __future__ import annotations

import contextlib
import stat
from types import SimpleNamespace

import pytest

from engine.dashboard.nightly import (
    NightlyReport,
    NightlyStop,
    repair_tier4_gap,
    repair_tier4_gap_if_needed,
)


def _report() -> NightlyReport:
    report = NightlyReport(as_of="2026-10-13")
    report.steps["tiers"] = {"rebuilt": ["panel", "tier4"],
                             "tier4_since": "2026-10-10"}
    return report


def _result(gap_fill: dict, elapsed_s: float = 12.0) -> SimpleNamespace:
    return SimpleNamespace(reports={"tier4": {"gap_fill": gap_fill}},
                           elapsed_s=elapsed_s)


def _gapped() -> dict:
    return {"pred_abs_move": {"out_of_window_gap": [
        {"ticker": "AAPL", "event_date": "2026-10-01"}]}}


# --------------------------------------------------------------------------
# the repair helper, directly
# --------------------------------------------------------------------------


def test_planted_gap_runs_a_full_rebuild_and_records_the_repair():
    report = _report()
    calls = []
    finalized = []

    repaired = _result({})  # the repair closed the gap

    def fake_rebuild_tables(tables, **kwargs):
        calls.append((tables, kwargs))
        return repaired

    readings = iter([0, 7.5])
    repair_tier4_gap(report, fake_rebuild_tables, clock=lambda: next(readings),
                      finalize=finalized.append)

    assert calls == [(("tier4",), {"tier4_since": None, "finalize": False})], \
        "panel was just rebuilt; the repair is Tier-4 only, with no --since"
    assert report.flags == [], "a repaired night carries no tier4_gap_partial"
    repair = report.steps["tiers"]["tier4_repair"]
    assert repair["ran"] is True
    assert repair["elapsed_s"] == 7.5
    assert finalized == [repaired], "the repair result is finalized exactly once"
    assert "degraded" not in report.steps["tiers"]


def test_repair_raising_stops_the_run_with_a_typed_code():
    finalized = []

    def boom(tables, **kwargs):
        raise RuntimeError("disk on fire")

    with pytest.raises(NightlyStop) as caught:
        repair_tier4_gap(_report(), boom, finalize=finalized.append)

    assert caught.value.step == "tier4_repair"
    assert caught.value.detail.startswith(
        "TIER4_REPAIR_FAILED: RuntimeError: disk on fire")
    assert finalized == [], "a refused repair must not finalize anything"


def test_repair_that_still_leaves_the_gap_stops_the_run():
    finalized = []

    def fake_rebuild_tables(tables, **kwargs):
        return _result(_gapped())

    with pytest.raises(NightlyStop) as caught:
        repair_tier4_gap(_report(), fake_rebuild_tables, finalize=finalized.append)

    assert caught.value.step == "tier4_repair"
    assert caught.value.detail.startswith("TIER4_GAP_REMAINS: ")
    assert "pred_abs_move" in caught.value.detail
    assert finalized == [], "a repair that still leaves the gap is not finalized"


def test_repair_budget_reports_the_complete_detail_string(monkeypatch):
    from engine.dashboard import nightly

    monkeypatch.setattr(nightly, "TIER4_REPAIR_BUDGET_S", 100)

    def fake_rebuild_tables(tables, **kwargs):
        return _result({})

    finalized = []
    readings = iter([0, 101.0])
    report = _report()

    with pytest.raises(NightlyStop) as caught:
        repair_tier4_gap(report, fake_rebuild_tables, clock=lambda: next(readings),
                          finalize=finalized.append)

    assert caught.value.step == "tier4_repair"
    assert caught.value.detail == (
        "TIER4_REPAIR_BUDGET: full Tier-4 rebuild took 101.0s, over the 100s budget"
    )
    assert finalized == [], "an over-budget repair is refused before finalizing"


def test_repair_at_exactly_the_budget_succeeds(monkeypatch):
    """The budget check is ``>``, so exactly TIER4_REPAIR_BUDGET_S is accepted."""
    from engine.dashboard import nightly

    monkeypatch.setattr(nightly, "TIER4_REPAIR_BUDGET_S", 100)

    def fake_rebuild_tables(tables, **kwargs):
        return _result({})

    finalized = []
    readings = iter([0, 100.0])
    report = _report()

    repair_tier4_gap(report, fake_rebuild_tables, clock=lambda: next(readings),
                      finalize=finalized.append)

    repair = report.steps["tiers"]["tier4_repair"]
    assert repair["ran"] is True, "exactly-on-budget repair is not refused"
    assert repair["elapsed_s"] == 100.0
    assert len(finalized) == 1, "the on-budget repair is finalized once"


def test_repair_just_over_the_budget_raises(monkeypatch):
    """One tick past the budget crosses the ``>`` check and stops the run."""
    from engine.dashboard import nightly

    monkeypatch.setattr(nightly, "TIER4_REPAIR_BUDGET_S", 100)

    def fake_rebuild_tables(tables, **kwargs):
        return _result({})

    finalized = []
    readings = iter([0, 100.001])
    report = _report()

    with pytest.raises(NightlyStop) as caught:
        repair_tier4_gap(report, fake_rebuild_tables, clock=lambda: next(readings),
                          finalize=finalized.append)

    assert caught.value.step == "tier4_repair"
    assert caught.value.detail.startswith("TIER4_REPAIR_BUDGET: ")
    assert finalized == [], "a just-over-budget repair is refused before finalizing"
    assert "tier4_repair" not in report.steps["tiers"]


def test_the_helper_finalizes_a_gap_free_incremental_result_once():
    report = _report()
    calls = []
    finalized = []

    def fake_rebuild_tables(tables, **kwargs):
        calls.append((tables, kwargs))
        return _result({})

    # This exercises the HELPER only (not run_nightly's call site): with no
    # out-of-window gap the helper finalizes the incremental result and does
    # not run the repair.
    tier_result = fake_rebuild_tables(("panel", "tier4"), tier4_since="2026-10-10",
                                      finalize=False)
    repair_tier4_gap_if_needed(report, tier_result, fake_rebuild_tables,
                                finalize=finalized.append)

    assert len(calls) == 1, "no gap means no second (full) rebuild"
    assert finalized == [tier_result], "the incremental result is finalized exactly once"
    assert "tier4_repair" not in report.steps["tiers"]


def test_an_incremental_result_with_a_gap_runs_the_repair_and_finalizes_it():
    report = _report()
    calls = []
    finalized = []

    repaired = _result({})  # the repair closed the gap

    def fake_rebuild_tables(tables, **kwargs):
        calls.append((tables, kwargs))
        return repaired

    repair_tier4_gap_if_needed(report, _result(_gapped()), fake_rebuild_tables,
                                finalize=finalized.append)

    assert calls == [(("tier4",), {"tier4_since": None, "finalize": False})], \
        "the repair is Tier-4 only, with no --since"
    assert finalized == [repaired], \
        "only the repair result is finalized; the gapped incremental one is not"
    assert report.steps["tiers"]["tier4_repair"]["ran"] is True
    assert report.flags == []


def test_a_refused_repair_never_finalizes_the_incremental_result():
    report = _report()
    finalized = []

    def boom(tables, **kwargs):
        raise RuntimeError("disk on fire")

    with pytest.raises(NightlyStop) as caught:
        repair_tier4_gap_if_needed(report, _result(_gapped()), boom,
                                    finalize=finalized.append)

    assert caught.value.detail.startswith("TIER4_REPAIR_FAILED: ")
    assert finalized == [], "a refused repair must leave no snapshot to finalize"


def test_rerunning_the_guard_after_a_repair_does_nothing():
    report = _report()
    calls = []
    finalized = []

    def fake_rebuild_tables(tables, **kwargs):
        calls.append((tables, kwargs))
        return _result({})

    repair_tier4_gap_if_needed(report, _result(_gapped()), fake_rebuild_tables,
                                finalize=finalized.append)
    gap_free = _result({})
    repair_tier4_gap_if_needed(report, gap_free, fake_rebuild_tables,
                                finalize=finalized.append)

    assert len(calls) == 1, "the rerun's gap-free result must not repair again"
    assert len(finalized) == 2, \
        "the repair result and the gap-free rerun are each finalized once"
    assert finalized[-1] is gap_free, "the gap-free rerun is finalized"
    assert report.steps["tiers"]["tier4_repair"]["ran"] is True


def test_repair_finalize_failure_stops_the_run_with_a_typed_code():
    def boom(result):
        raise RuntimeError("disk full")

    def fake_rebuild_tables(tables, **kwargs):
        return _result({})

    report = _report()

    with pytest.raises(NightlyStop) as caught:
        repair_tier4_gap(report, fake_rebuild_tables, finalize=boom)

    assert caught.value.step == "tier4_repair"
    assert caught.value.detail.startswith(
        "TIER4_REPAIR_FAILED: RuntimeError: disk full")
    assert "tier4_repair" not in report.steps["tiers"], \
        "a refused finalize records no tier4_repair step"


def test_gap_free_finalize_failure_degrades_the_tiers_step(monkeypatch):
    # The degrade path writes the deferred snapshot via finalize_current_store;
    # stub it so this unit test never writes the real SNAPSHOT/MANIFEST.
    monkeypatch.setattr("engine.data.rebuild.finalize_current_store",
                        lambda: SimpleNamespace(snapshot="snap-degraded"))
    report = _report()

    def boom(result):
        raise RuntimeError("disk full")

    repair_tier4_gap_if_needed(report, _result({}), lambda *a, **k: _result({}),
                                finalize=boom)

    assert report.steps["tiers"]["degraded"] is True
    assert "RuntimeError: disk full" in report.steps["tiers"]["error"]
    degraded = [f for f in report.flags if f.get("kind") == "tiers_degraded"]
    assert len(degraded) == 1, "the gap-free finalize failure raises one flag"


def test_tiers_degraded_scrubs_local_paths_from_both_fields(monkeypatch):
    """A degraded tiers step publishes no absolute local path (#490).

    The flag detail and the step error reach the published bundle, so an
    exception carrying a host-local path must have it redacted while the
    ``type(exc).__name__`` prefix stays.
    """
    from engine.dashboard.nightly import mark_tiers_degraded

    # The degrade path also writes the deferred snapshot; stub it out so this
    # unit test never touches the real store.
    monkeypatch.setattr("engine.data.rebuild.finalize_current_store", lambda: None)

    report = _report()

    mark_tiers_degraded(report, RuntimeError("failed to read /root/secret/x"))

    assert report.steps["tiers"]["degraded"] is True
    error = report.steps["tiers"]["error"]
    assert "/root/secret/x" not in error, "the step error must not leak the path"
    assert error.startswith("RuntimeError: "), "the exception type stays visible"
    degraded = [f for f in report.flags if f.get("kind") == "tiers_degraded"]
    assert len(degraded) == 1
    detail = degraded[0]["detail"]
    assert "/root/secret/x" not in detail, "the flag detail must not leak the path"
    assert "RuntimeError: " in detail, "the exception type stays visible"


# --------------------------------------------------------------------------
# manifest/snapshot atomicity: a failed write leaves the old file untouched
# --------------------------------------------------------------------------


def _raise_disk_full(*_args, **_kwargs):
    raise OSError("disk full")


def test_write_snapshot_is_atomic_when_the_replace_fails(tmp_path, monkeypatch):
    """A failed replace leaves the pre-existing SNAPSHOT and no temp file."""
    from engine.data import manifest as manifest_mod

    target = tmp_path / "SNAPSHOT"
    target.write_text("PRE-EXISTING SNAPSHOT")
    monkeypatch.setattr(manifest_mod, "_panel_digest", lambda: None)
    monkeypatch.setattr(manifest_mod, "_tier4_digest", lambda: None)
    monkeypatch.setattr(manifest_mod.os, "replace", _raise_disk_full)

    with pytest.raises(OSError, match="disk full"):
        manifest_mod.write_snapshot({"panel": {"content_hash": "abc"}}, path=target)

    assert target.read_text() == "PRE-EXISTING SNAPSHOT", \
        "a failed write must leave the previous SNAPSHOT unchanged"
    assert sorted(p.name for p in tmp_path.glob("*.tmp*")) == [], \
        "a failed write must leave no temp file behind"


def test_write_manifest_is_atomic_when_the_replace_fails(tmp_path, monkeypatch):
    """A failed replace leaves the pre-existing MANIFEST and no temp file."""
    from engine.data import manifest as manifest_mod

    target = tmp_path / "MANIFEST.md"
    target.write_text("PRE-EXISTING MANIFEST")
    stats = {"panel": {"rows": 1, "years": "2020", "partitions": 1, "files": 1,
                       "bytes": 1, "content_hash": "abc"}}
    monkeypatch.setattr(manifest_mod, "snapshot_hash", lambda _stats: "deadbeef")
    monkeypatch.setattr(manifest_mod, "_panel_digest", lambda: None)
    monkeypatch.setattr(manifest_mod.os, "replace", _raise_disk_full)

    with pytest.raises(OSError, match="disk full"):
        manifest_mod.write_manifest(stats, path=target)

    assert target.read_text() == "PRE-EXISTING MANIFEST", \
        "a failed write must leave the previous MANIFEST unchanged"
    assert sorted(p.name for p in tmp_path.glob("*.tmp*")) == [], \
        "a failed write must leave no temp file behind"


def test_write_snapshot_lands_mode_0644(tmp_path, monkeypatch):
    """The atomic SNAPSHOT write keeps the 0644 that ``write_text`` gave.

    ``tempfile.mkstemp`` creates its temp file 0600, so without an explicit
    chmod the swap would publish a SNAPSHOT more restrictive than before.
    """
    from engine.data import manifest as manifest_mod

    target = tmp_path / "SNAPSHOT"
    monkeypatch.setattr(manifest_mod, "_panel_digest", lambda: None)
    monkeypatch.setattr(manifest_mod, "_tier4_digest", lambda: None)

    manifest_mod.write_snapshot({"panel": {"content_hash": "abc"}}, path=target)

    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_write_manifest_lands_mode_0644(tmp_path, monkeypatch):
    """The atomic MANIFEST write keeps the 0644 that ``write_text`` gave."""
    from engine.data import manifest as manifest_mod

    target = tmp_path / "MANIFEST.md"
    stats = {"panel": {"rows": 1, "years": "2020", "partitions": 1, "files": 1,
                       "bytes": 1, "content_hash": "abc"}}
    monkeypatch.setattr(manifest_mod, "snapshot_hash", lambda _stats: "deadbeef")
    monkeypatch.setattr(manifest_mod, "_panel_digest", lambda: None)

    manifest_mod.write_manifest(stats, path=target)

    assert stat.S_IMODE(target.stat().st_mode) == 0o644


# --------------------------------------------------------------------------
# rebuild()/finalize_rebuild(): the snapshot is deferred, not written early
# --------------------------------------------------------------------------


def test_rebuild_defers_the_snapshot_until_finalize(monkeypatch):
    from engine.data import rebuild as rebuild_mod

    writes = []
    monkeypatch.setattr(rebuild_mod.paths, "ensure_dirs", lambda: None)
    monkeypatch.setattr(rebuild_mod, "build_panel_table", lambda: {"rows": 3})
    monkeypatch.setattr(rebuild_mod, "build_tier4_table",
                        lambda since=None: {"rows": 3, "producers": {}})
    monkeypatch.setattr(rebuild_mod.manifest, "collect_stats",
                        lambda: {"panel": "hash"})
    monkeypatch.setattr(rebuild_mod.manifest, "write_snapshot",
                        lambda stats: writes.append(("snapshot", stats)) or "snap-1")
    monkeypatch.setattr(rebuild_mod.manifest, "write_manifest",
                        lambda stats: writes.append(("manifest", stats)) or "manifest-1")

    result = rebuild_mod.rebuild(("panel", "tier4"), finalize=False)

    assert writes == [], "finalize=False must not write the snapshot or manifest"
    assert result.snapshot is None

    finalize_rebuild = rebuild_mod.finalize_rebuild
    rebuilt = finalize_rebuild(result)

    assert rebuilt is result
    assert result.snapshot == "snap-1"
    assert writes == [("manifest", {"panel": "hash"}), ("snapshot", {"panel": "hash"})]

    writes.clear()
    default = rebuild_mod.rebuild(("panel", "tier4"))

    assert default.snapshot == "snap-1"
    assert writes == [("manifest", {"panel": "hash"}), ("snapshot", {"panel": "hash"})], \
        "the default rebuild path still finalizes exactly once"


def test_rebuild_stamps_elapsed_when_it_defers_finalize(monkeypatch):
    """finalize=False still stamps elapsed_s from the build's start clock."""
    from engine.data import rebuild as rebuild_mod

    monkeypatch.setattr(rebuild_mod.paths, "ensure_dirs", lambda: None)
    monkeypatch.setattr(rebuild_mod, "build_panel_table", lambda: {"rows": 3})
    monkeypatch.setattr(rebuild_mod, "build_tier4_table",
                        lambda since=None: {"rows": 3, "producers": {}})

    readings = iter([100.0, 107.5])
    monkeypatch.setattr(rebuild_mod, "time",
                        SimpleNamespace(time=lambda: next(readings)))

    result = rebuild_mod.rebuild(("panel", "tier4"), finalize=False)

    assert result.snapshot is None, "finalize=False writes no snapshot"
    assert result.elapsed_s > 0, "elapsed_s is stamped before the deferred finalize"
    assert result.elapsed_s == pytest.approx(7.5)


def test_finalize_writes_manifest_before_snapshot(monkeypatch):
    """A manifest failure must leave no snapshot: manifest is written first."""
    from engine.data import rebuild as rebuild_mod

    writes = []

    def fake_write_manifest(stats):
        writes.append(("manifest", stats))
        raise OSError("disk full")

    monkeypatch.setattr(rebuild_mod.paths, "ensure_dirs", lambda: None)
    monkeypatch.setattr(rebuild_mod, "build_panel_table", lambda: {"rows": 3})
    monkeypatch.setattr(rebuild_mod, "build_tier4_table",
                        lambda since=None: {"rows": 3, "producers": {}})
    monkeypatch.setattr(rebuild_mod.manifest, "collect_stats",
                        lambda: {"panel": "hash"})
    monkeypatch.setattr(rebuild_mod.manifest, "write_snapshot",
                        lambda stats: writes.append(("snapshot", stats)) or "snap-1")
    monkeypatch.setattr(rebuild_mod.manifest, "write_manifest", fake_write_manifest)

    result = rebuild_mod.rebuild(("panel", "tier4"), finalize=False)

    with pytest.raises(OSError, match="disk full"):
        rebuild_mod.finalize_rebuild(result)

    assert writes == [("manifest", {"panel": "hash"})], \
        "a manifest failure must not have written the snapshot"

    # Success path: the manifest lands first and the snapshot last.
    monkeypatch.setattr(rebuild_mod.manifest, "write_manifest",
                        lambda stats: writes.append(("manifest", stats)) or "manifest-1")
    writes.clear()
    rebuilt = rebuild_mod.finalize_rebuild(result)

    assert rebuilt.snapshot == "snap-1"
    assert [kind for kind, _ in writes] == ["manifest", "snapshot"]


# --------------------------------------------------------------------------
# orchestration through run_nightly
# --------------------------------------------------------------------------


class _Scored(Exception):
    """A private sentinel: raised the moment the scorer is actually consulted."""


class _ScorerSentinel:
    """A scorer stand-in whose every attribute use raises :class:`_Scored`.

    It records that it was touched, so a stopped run can prove the scorer was
    never reached.
    """

    def __init__(self):
        object.__setattr__(self, "touched", False)

    def __getattr__(self, name):
        object.__setattr__(self, "touched", True)
        raise _Scored(name)


def _marker(tmp_path):
    """The file standing in for SNAPSHOT in the orchestration fakes."""
    return tmp_path / "snapshot.marker"


def _events_stub(tickers, *, as_of="2026-10-13", ahead_days=2):
    """One upcoming, session-stamped event per ticker for the store stub.

    A non-empty calendar is what makes ``run_nightly`` run its refresh step
    (``if refresh and calendar_tickers``); the empty stub the older cases use
    skips refresh and finality entirely. The date lands inside the horizon and
    carries a ``session``, so the event counts as upcoming and scoreable.
    """
    import pandas as pd

    stamp = pd.Timestamp(as_of) + pd.Timedelta(days=ahead_days)
    return pd.DataFrame({
        "event_id": [f"evt-{t}" for t in tickers],
        "ticker": list(tickers),
        "event_date": [stamp] * len(tickers),
        "session": ["bmo"] * len(tickers),
        "date_conflict": [False] * len(tickers),
    })


def _refresh_recorder(record):
    """A stand-in ``refresh_calendar_data`` that records its ``finalize`` kwarg.

    Returns the deferred-snapshot shape the refresh rebuild now reports when it
    is told not to finalize (``rebuild.snapshot`` is ``None``).
    """

    def recorder(tickers, as_of, **kwargs):
        record.append({"tickers": list(tickers), "finalize": kwargs.get("finalize")})
        return {"calls": 0, "rebuild": {"snapshot": None, "elapsed_s": 0}}

    return recorder


def _patch_run_nightly(monkeypatch, *, rebuild_results, finalize_calls, marker_path,
                       events=None):
    """Patch everything run_nightly touches up to the scorer.

    Returns ``(nightly, calls, captured, marker_writes)`` where ``calls``
    records every fake rebuild invocation and ``captured["report"]`` is the
    ``NightlyReport`` the run built. ``marker_writes`` counts snapshot writes:
    the fake ``rebuild`` writes ``marker_path`` when it finalizes (the
    default), the fake ``finalize_rebuild`` writes the same file, and the fake
    ``finalize_current_store`` — the degrade path's snapshot write — does too.

    ``events`` is the earnings-events table the store stub returns. It defaults
    to empty (no calendar, so refresh and finality are skipped); pass
    :func:`_events_stub` to drive a non-empty calendar through refresh, finality
    and the tiers step. ``resolve_final_session`` and ``trading_calendar`` are
    stubbed too, so a non-empty calendar proceeds without touching the network
    or the calendar cache.
    """
    import pandas as pd

    from engine.dashboard import nightly
    from engine.data import store
    from engine.data.pulls import computed_moves

    if events is None:
        events = pd.DataFrame({
            "event_id": pd.Series(dtype="object"),
            "ticker": pd.Series(dtype="object"),
            "event_date": pd.Series(dtype="datetime64[ns]"),
            "session": pd.Series(dtype="object"),
            "date_conflict": pd.Series(dtype="bool"),
        })
    monkeypatch.setattr(store, "read_table", lambda *a, **k: events.copy())
    monkeypatch.setattr(
        "engine.data.finality.resolve_final_session",
        lambda requested, tickers, *, calendar: SimpleNamespace(
            date=str(pd.Timestamp(requested).normalize().date()),
            as_dict=lambda: {"is_final": True, "detail": "stub"},
        ),
    )
    monkeypatch.setattr(
        "engine.calendar.trading_calendar",
        lambda: SimpleNamespace(is_trading_day=lambda stamp: True,
                                shift=lambda stamp, n: stamp),
    )
    monkeypatch.setattr(nightly, "validate_refresh",
                        lambda *a, **k: [{"name": "no_calendar", "passed": True,
                                          "detail": "no tickers to validate"}])
    monkeypatch.setattr(nightly, "_panel_staleness_flags", lambda as_of: [])
    monkeypatch.setattr(computed_moves, "read_state", lambda: {})
    monkeypatch.setattr(computed_moves, "target_tickers", lambda **kwargs: ([], {}))

    calls = []
    marker_writes = []

    def fake_rebuild(tables, **kwargs):
        calls.append((tables, kwargs))
        outcome = rebuild_results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if kwargs.get("finalize", True):
            marker_writes.append(tables)
            marker_path.write_text("snapshot")
        return outcome

    def fake_finalize_rebuild(result):
        finalize_calls.append(result)
        marker_writes.append("finalize")
        marker_path.write_text("snapshot")

    def fake_finalize_current_store():
        marker_writes.append("degrade")
        marker_path.write_text("snapshot")

    monkeypatch.setattr("engine.data.rebuild.rebuild", fake_rebuild)
    monkeypatch.setattr("engine.data.rebuild.finalize_rebuild", fake_finalize_rebuild)
    monkeypatch.setattr("engine.data.rebuild.finalize_current_store",
                        fake_finalize_current_store)

    captured = {}

    class _RecordingReport(nightly.NightlyReport):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["report"] = self

    monkeypatch.setattr(nightly, "NightlyReport", _RecordingReport)

    return nightly, calls, captured, marker_writes


def _run(monkeypatch, tmp_path, *, rebuild_results, finalize_calls):
    nightly, calls, captured, marker_writes = _patch_run_nightly(
        monkeypatch, rebuild_results=rebuild_results, finalize_calls=finalize_calls,
        marker_path=_marker(tmp_path))
    sentinel = _ScorerSentinel()
    with pytest.raises(_Scored):
        nightly.run_nightly(
            as_of="2026-10-13", tickers=[], refresh=True, tiers=True,
            publish=False, scorer=sentinel,
            bundle_dir=tmp_path / "bundle", target=tmp_path / "target",
        )
    return calls, sentinel, captured, marker_writes


def test_run_nightly_repairs_a_gap_then_scores(monkeypatch, tmp_path):
    finalize_calls = []
    repaired = _result({})
    calls, sentinel, captured, marker_writes = _run(
        monkeypatch, tmp_path,
        rebuild_results=[_result(_gapped()), repaired],
        finalize_calls=finalize_calls,
    )

    assert calls == [
        (("panel", "tier4"), {"tier4_since": None, "finalize": False}),
        (("tier4",), {"tier4_since": None, "finalize": False}),
    ], "the incremental build is not finalized; the repair defers its snapshot too"
    assert finalize_calls == [repaired], "only the repair result is finalized"
    assert _marker(tmp_path).exists(), "the clean repair writes the snapshot marker"
    assert len(marker_writes) == 1, "the snapshot marker is written exactly once"
    report = captured.get("report")
    assert report is not None, "run_nightly never constructed a report"
    assert report.steps["tiers"]["tier4_repair"]["ran"] is True
    assert not any(f.get("kind") == "tier4_gap_partial" for f in report.flags)
    assert sentinel.touched, "the run proceeded to the scorer"


def test_run_nightly_stops_when_the_repair_raises(monkeypatch, tmp_path):
    finalize_calls = []
    nightly, _calls, _captured, marker_writes = _patch_run_nightly(
        monkeypatch,
        rebuild_results=[_result(_gapped()), RuntimeError("disk on fire")],
        finalize_calls=finalize_calls,
        marker_path=_marker(tmp_path),
    )
    sentinel = _ScorerSentinel()

    with pytest.raises(NightlyStop) as caught:
        nightly.run_nightly(
            as_of="2026-10-13", tickers=[], refresh=True, tiers=True,
            publish=False, scorer=sentinel,
            bundle_dir=tmp_path / "bundle", target=tmp_path / "target",
        )

    assert caught.value.step == "tier4_repair"
    assert caught.value.detail.startswith("TIER4_REPAIR_FAILED: ")
    assert not sentinel.touched, "a refused repair must never reach the scorer"
    assert finalize_calls == [], "a refused repair must leave no snapshot"
    assert marker_writes == []
    assert not _marker(tmp_path).exists(), "a refused repair writes no snapshot marker"


def test_run_nightly_stops_when_the_repair_leaves_the_gap(monkeypatch, tmp_path):
    finalize_calls = []
    nightly, _calls, _captured, marker_writes = _patch_run_nightly(
        monkeypatch,
        rebuild_results=[_result(_gapped()), _result(_gapped())],
        finalize_calls=finalize_calls,
        marker_path=_marker(tmp_path),
    )
    sentinel = _ScorerSentinel()

    with pytest.raises(NightlyStop) as caught:
        nightly.run_nightly(
            as_of="2026-10-13", tickers=[], refresh=True, tiers=True,
            publish=False, scorer=sentinel,
            bundle_dir=tmp_path / "bundle", target=tmp_path / "target",
        )

    assert caught.value.detail.startswith("TIER4_GAP_REMAINS: ")
    assert not sentinel.touched, "a remaining gap must never reach the scorer"
    assert finalize_calls == []
    assert not _marker(tmp_path).exists(), "a remaining gap leaves no snapshot marker"


def test_run_nightly_leaves_no_snapshot_when_the_repair_exceeds_the_budget(
        monkeypatch, tmp_path):
    from engine.dashboard import nightly as nightly_mod

    monkeypatch.setattr(nightly_mod, "TIER4_REPAIR_BUDGET_S", -1)
    finalize_calls = []
    nightly, _calls, _captured, marker_writes = _patch_run_nightly(
        monkeypatch,
        rebuild_results=[_result(_gapped()), _result({})],
        finalize_calls=finalize_calls,
        marker_path=_marker(tmp_path),
    )
    sentinel = _ScorerSentinel()

    with pytest.raises(NightlyStop) as caught:
        nightly.run_nightly(
            as_of="2026-10-13", tickers=[], refresh=True, tiers=True,
            publish=False, scorer=sentinel,
            bundle_dir=tmp_path / "bundle", target=tmp_path / "target",
        )

    assert caught.value.step == "tier4_repair"
    assert caught.value.detail.startswith("TIER4_REPAIR_BUDGET: ")
    assert not sentinel.touched, "an over-budget repair must never reach the scorer"
    assert finalize_calls == []
    assert marker_writes == []
    assert not _marker(tmp_path).exists(), "an over-budget repair writes no snapshot"


def test_run_nightly_finalizes_a_gap_free_incremental_result(monkeypatch, tmp_path):
    finalize_calls = []
    incremental = _result({}, elapsed_s=12.5)
    calls, sentinel, captured, marker_writes = _run(
        monkeypatch, tmp_path,
        rebuild_results=[incremental],
        finalize_calls=finalize_calls,
    )

    assert calls == [(("panel", "tier4"), {"tier4_since": None, "finalize": False})], \
        "a gap-free night rebuilds once"
    assert finalize_calls == [incremental], "the incremental result is finalized once"
    assert _marker(tmp_path).exists(), "the no-gap path writes the snapshot marker"
    assert len(marker_writes) == 1, "the snapshot marker is written exactly once"
    assert captured["report"].steps["tiers"]["elapsed_s"] == 12.5, \
        "the tiers step records the incremental build's stamped elapsed_s"
    assert sentinel.touched, "the run proceeded to the scorer"


def _snap(value):
    """A stand-in finalize result carrying the SNAPSHOT hash it produced."""
    return SimpleNamespace(snapshot=value)


def _run_once(nightly, tmp_path, sentinel):
    with pytest.raises(_Scored):
        nightly.run_nightly(
            as_of="2026-10-13", tickers=[], refresh=True, tiers=True,
            publish=False, scorer=sentinel,
            bundle_dir=tmp_path / "bundle", target=tmp_path / "target",
        )


def test_run_nightly_records_the_gap_free_finalize_snapshot(monkeypatch, tmp_path):
    """A gap-free night records the SNAPSHOT its finalize returned."""
    finalize_calls = []
    nightly, _calls, captured, _marker_writes = _patch_run_nightly(
        monkeypatch,
        rebuild_results=[_result({})],
        finalize_calls=finalize_calls,
        marker_path=_marker(tmp_path),
    )
    snap = _snap("snap-gap-free")

    def fake_finalize(result):
        finalize_calls.append(result)
        _marker(tmp_path).write_text("snapshot")
        return snap

    monkeypatch.setattr("engine.data.rebuild.finalize_rebuild", fake_finalize)
    _run_once(nightly, tmp_path, _ScorerSentinel())

    report = captured["report"]
    assert report.steps["tiers"]["snapshot"] == "snap-gap-free", \
        "the gap-free path records the snapshot its finalize returned"


def test_run_nightly_records_the_repair_finalize_snapshot(monkeypatch, tmp_path):
    """A gapped night records the SNAPSHOT the repair's finalize returned."""
    finalize_calls = []
    nightly, _calls, captured, _marker_writes = _patch_run_nightly(
        monkeypatch,
        rebuild_results=[_result(_gapped()), _result({})],
        finalize_calls=finalize_calls,
        marker_path=_marker(tmp_path),
    )
    snap = _snap("snap-repaired")

    def fake_finalize(result):
        finalize_calls.append(result)
        _marker(tmp_path).write_text("snapshot")
        return snap

    monkeypatch.setattr("engine.data.rebuild.finalize_rebuild", fake_finalize)
    _run_once(nightly, tmp_path, _ScorerSentinel())

    report = captured["report"]
    assert report.steps["tiers"]["tier4_repair"]["ran"] is True
    assert report.steps["tiers"]["snapshot"] == "snap-repaired", \
        "the repair path records the snapshot its finalize returned"


def test_run_nightly_records_the_degraded_finalize_snapshot(monkeypatch, tmp_path):
    """A degraded night records the SNAPSHOT its degrade-path finalize returned."""
    finalize_calls = []
    nightly, _calls, captured, _marker_writes = _patch_run_nightly(
        monkeypatch,
        rebuild_results=[RuntimeError("boom")],
        finalize_calls=finalize_calls,
        marker_path=_marker(tmp_path),
    )
    snap = _snap("snap-degraded")

    def fake_finalize_current_store():
        _marker(tmp_path).write_text("snapshot")
        return snap

    monkeypatch.setattr("engine.data.rebuild.finalize_current_store",
                        fake_finalize_current_store)
    _run_once(nightly, tmp_path, _ScorerSentinel())

    report = captured["report"]
    assert report.steps["tiers"]["degraded"] is True
    assert report.steps["tiers"]["snapshot"] == "snap-degraded", \
        "the degrade path records the snapshot its finalize returned"


def test_run_nightly_is_idempotent_across_two_runs(monkeypatch, tmp_path):
    """A rerun over a closed gap repairs nothing and finalizes its own result.

    Run 1: the incremental build leaves the planted gap, so the repair runs and
    finalizes. Run 2: the gap is closed, so the incremental build finalizes with
    no repair. Three rebuild calls and two finalize calls in total, and neither
    report carries a ``tier4_gap_partial`` flag.
    """
    finalize_calls = []
    nightly, calls, captured, _marker_writes = _patch_run_nightly(
        monkeypatch,
        rebuild_results=[_result(_gapped()), _result({}), _result({})],
        finalize_calls=finalize_calls,
        marker_path=_marker(tmp_path),
    )
    sentinel = _ScorerSentinel()

    reports = []
    for _ in range(2):
        _run_once(nightly, tmp_path, sentinel)
        reports.append(captured["report"])

    assert len(calls) == 3, \
        "run 1 rebuilds incrementally then repairs; run 2 only rebuilds incrementally"
    assert len(finalize_calls) == 2, "each run finalizes its own result exactly once"
    for report in reports:
        assert not any(f.get("kind") == "tier4_gap_partial" for f in report.flags), \
            "a repaired or clean night carries no tier4_gap_partial"


def test_run_nightly_degrades_when_a_gap_free_finalize_raises(monkeypatch, tmp_path):
    finalize_calls = []
    nightly, _calls, captured, marker_writes = _patch_run_nightly(
        monkeypatch,
        rebuild_results=[_result({})],
        finalize_calls=finalize_calls,
        marker_path=_marker(tmp_path),
    )

    def boom(result):
        finalize_calls.append(result)
        raise RuntimeError("disk full")

    monkeypatch.setattr("engine.data.rebuild.finalize_rebuild", boom)
    sentinel = _ScorerSentinel()

    with pytest.raises(_Scored):
        nightly.run_nightly(
            as_of="2026-10-13", tickers=[], refresh=True, tiers=True,
            publish=False, scorer=sentinel,
            bundle_dir=tmp_path / "bundle", target=tmp_path / "target",
        )

    report = captured.get("report")
    assert report is not None, "run_nightly never constructed a report"
    assert report.steps["tiers"]["degraded"] is True
    assert "RuntimeError: disk full" in report.steps["tiers"]["error"]
    assert "snapshot_error" not in report.steps["tiers"]
    assert any(f.get("kind") == "tiers_degraded" for f in report.flags)
    assert sentinel.touched, "the run proceeded to the scorer"
    assert marker_writes == ["degrade"], \
        "the degrade path writes exactly one snapshot"
    assert _marker(tmp_path).exists(), \
        "the degrade path still writes the deferred refresh snapshot"


def test_run_nightly_degrades_when_the_incremental_rebuild_raises(monkeypatch, tmp_path):
    finalize_calls = []
    calls, sentinel, captured, marker_writes = _run(
        monkeypatch, tmp_path,
        rebuild_results=[RuntimeError("boom")],
        finalize_calls=finalize_calls,
    )

    assert calls == [(("panel", "tier4"), {"tier4_since": None, "finalize": False})], \
        "a failed incremental build is never repaired"
    assert finalize_calls == [], "the repair rebuild and finalize are never called"
    report = captured.get("report")
    assert report is not None, "run_nightly never constructed a report"
    assert report.steps["tiers"]["degraded"] is True
    assert "RuntimeError: boom" in report.steps["tiers"]["error"]
    assert "snapshot_error" not in report.steps["tiers"]
    assert any(f.get("kind") == "tiers_degraded" for f in report.flags)
    assert sentinel.touched, "the run proceeded to the scorer"
    assert marker_writes == ["degrade"], \
        "the degrade path writes exactly one snapshot"
    assert _marker(tmp_path).exists(), \
        "the degrade path still writes the deferred refresh snapshot"


def test_run_nightly_survives_a_failing_finalize_current_store(monkeypatch, tmp_path):
    """A failed degrade-path snapshot write is recorded, not raised.

    The refresh rebuild deferred its snapshot to the tiers step; when the
    tiers step degrades, ``mark_tiers_degraded`` writes that snapshot. If the
    write itself fails the run still scores: the failure is recorded as
    ``steps.tiers.snapshot_error`` (type only) and swallowed.
    """
    finalize_calls = []
    nightly, _calls, captured, marker_writes = _patch_run_nightly(
        monkeypatch,
        rebuild_results=[RuntimeError("boom")],
        finalize_calls=finalize_calls,
        marker_path=_marker(tmp_path),
    )

    def boom():
        raise RuntimeError("disk full")

    monkeypatch.setattr("engine.data.rebuild.finalize_current_store", boom)
    sentinel = _ScorerSentinel()

    with pytest.raises(_Scored):
        nightly.run_nightly(
            as_of="2026-10-13", tickers=[], refresh=True, tiers=True,
            publish=False, scorer=sentinel,
            bundle_dir=tmp_path / "bundle", target=tmp_path / "target",
        )

    report = captured.get("report")
    assert report is not None, "run_nightly never constructed a report"
    assert report.steps["tiers"]["degraded"] is True
    assert report.steps["tiers"]["snapshot_error"] == "RuntimeError", \
        "only the exception type is recorded, no text"
    assert any(f.get("kind") == "tiers_degraded" for f in report.flags)
    assert sentinel.touched, "the run still scored despite the failed snapshot write"
    assert marker_writes == []
    assert not _marker(tmp_path).exists(), "a failed snapshot write leaves no marker"


# --------------------------------------------------------------------------
# the refresh rebuild's snapshot is deferred exactly when tiers will finalize
# --------------------------------------------------------------------------


def test_run_nightly_defers_the_refresh_snapshot_when_tiers_will_finalize(
        monkeypatch, tmp_path):
    """With a real calendar and tiers on, refresh must NOT finalize.

    The refresh rebuild is told ``finalize=False``; the tiers step's own
    ``finalize_rebuild`` writes the one snapshot after the Tier-4 gap check.
    """
    refresh_calls = []
    nightly, calls, _captured, _marker_writes = _patch_run_nightly(
        monkeypatch,
        rebuild_results=[_result({})],
        finalize_calls=[],
        marker_path=_marker(tmp_path),
        events=_events_stub(["AAA"]),
    )
    monkeypatch.setattr(nightly, "refresh_calendar_data", _refresh_recorder(refresh_calls))
    sentinel = _ScorerSentinel()

    with pytest.raises(_Scored):
        nightly.run_nightly(
            as_of="2026-10-13", tickers=["AAA"], refresh=True, tiers=True,
            publish=False, scorer=sentinel,
            bundle_dir=tmp_path / "bundle", target=tmp_path / "target",
        )

    assert refresh_calls == [{"tickers": ["AAA"], "finalize": False}], \
        "with tiers on, the refresh rebuild defers; the tiers step finalizes"
    assert calls == [(("panel", "tier4"), {"tier4_since": None, "finalize": False})], \
        "the refresh deferral leaves the one finalize to the tiers step"
    assert sentinel.touched, "the run proceeded to the scorer"


def test_run_nightly_finalizes_the_refresh_snapshot_when_tiers_is_off(
        monkeypatch, tmp_path):
    """With tiers off there is no later finalize, so refresh finalizes as before."""
    refresh_calls = []
    nightly, calls, _captured, _marker_writes = _patch_run_nightly(
        monkeypatch,
        rebuild_results=[],
        finalize_calls=[],
        marker_path=_marker(tmp_path),
        events=_events_stub(["AAA"]),
    )
    monkeypatch.setattr(nightly, "refresh_calendar_data", _refresh_recorder(refresh_calls))
    sentinel = _ScorerSentinel()

    with pytest.raises(_Scored):
        nightly.run_nightly(
            as_of="2026-10-13", tickers=["AAA"], refresh=True, tiers=False,
            publish=False, scorer=sentinel,
            bundle_dir=tmp_path / "bundle", target=tmp_path / "target",
        )

    assert refresh_calls == [{"tickers": ["AAA"], "finalize": True}], \
        "with tiers off, the refresh rebuild finalizes as before"
    assert calls == [], "no tiers rebuild runs when tiers is off"
    assert sentinel.touched, "the run proceeded to the scorer"


def test_run_nightly_refused_repair_writes_no_snapshot_with_calendar_names(
        monkeypatch, tmp_path):
    """A refused repair leaves no SNAPSHOT even when refresh ran first.

    The refresh rebuild deferred its snapshot (``finalize=False``); when the
    repair is refused neither the refresh nor the incremental result is
    finalized, so nothing writes the marker.
    """
    finalize_calls = []
    nightly, _calls, _captured, marker_writes = _patch_run_nightly(
        monkeypatch,
        rebuild_results=[_result(_gapped()), RuntimeError("disk on fire")],
        finalize_calls=finalize_calls,
        marker_path=_marker(tmp_path),
        events=_events_stub(["AAA"]),
    )
    monkeypatch.setattr(nightly, "refresh_calendar_data", _refresh_recorder([]))
    sentinel = _ScorerSentinel()

    with pytest.raises(NightlyStop) as caught:
        nightly.run_nightly(
            as_of="2026-10-13", tickers=["AAA"], refresh=True, tiers=True,
            publish=False, scorer=sentinel,
            bundle_dir=tmp_path / "bundle", target=tmp_path / "target",
        )

    assert caught.value.step == "tier4_repair"
    assert caught.value.detail.startswith("TIER4_REPAIR_FAILED: ")
    assert not sentinel.touched, "a refused repair must never reach the scorer"
    assert finalize_calls == [], "neither the refresh nor the tiers result finalizes"
    assert marker_writes == []
    assert not _marker(tmp_path).exists(), "a refused repair leaves no SNAPSHOT"


# --------------------------------------------------------------------------
# the REAL refresh_calendar_data's finalize plumbing
# --------------------------------------------------------------------------


class _StubFetchRecord:
    """A minimal ``Fetcher`` record: not from cache, one synthetic row."""

    from_cache = False

    def json(self):
        return {"data": [{"ticker": "AAA"}]}


class _StubFetcher:
    """A ``Fetcher`` stand-in for refresh_calendar_data's own ORATS calls."""

    def __init__(self):
        self.calls = []

    def fetch(self, source, endpoint, params, **kwargs):
        self.calls.append((source, endpoint, params))
        return _StubFetchRecord()


def test_real_refresh_calendar_data_forwards_finalize_to_the_rebuild(monkeypatch):
    """The REAL refresh_calendar_data delegates its ``finalize`` flag to rebuild().

    Every network/IO dependency below the real function's own signature is
    stubbed — the forward-calendar pull, the history backfill, the recently-
    printed lookup, the chain pull and the ``Fetcher`` — so the only behaviour
    under test is the ``finalize`` plumbing into ``engine.data.rebuild.rebuild``.
    """
    from engine.dashboard import nightly
    from engine.data.pulls import forward_calendar

    monkeypatch.setattr(
        forward_calendar, "refresh_forward_calendar",
        lambda *a, **k: SimpleNamespace(as_dict=lambda: {"stub": True}))
    monkeypatch.setattr(nightly, "backfill_ticker_history",
                        lambda *a, **k: {"calls": 0})
    monkeypatch.setattr(nightly, "_recently_printed", lambda as_of, tickers: [])
    monkeypatch.setattr(nightly, "refresh_forward_chains",
                        lambda *a, **k: {"calls": 0})

    recorded = []

    class _RebuildResult:
        snapshot = "snap-1"
        elapsed_s = 1.5

    def fake_rebuild(tables, **kwargs):
        recorded.append(kwargs.get("finalize"))
        return _RebuildResult()

    monkeypatch.setattr("engine.data.rebuild.rebuild", fake_rebuild)

    fetcher = _StubFetcher()
    out = nightly.refresh_calendar_data(
        ["AAA"], "2026-10-13", fetcher=fetcher, finalize=False)

    assert recorded == [False], "finalize=False must reach the rebuild"
    assert out["rebuild"] == {"snapshot": "snap-1", "elapsed_s": 1.5}

    nightly.refresh_calendar_data(["AAA"], "2026-10-13", fetcher=fetcher)

    assert recorded == [False, True], "the default finalize=True reaches the rebuild"


# --------------------------------------------------------------------------
# the nightly CLI surfaces a typed stop as exit code 1
# --------------------------------------------------------------------------


def test_cli_returns_1_and_prints_the_code_when_run_nightly_stops(
        monkeypatch, capsys):
    from engine.dashboard import nightly

    def boom(*args, **kwargs):
        raise NightlyStop("tier4_repair", "TIER4_REPAIR_FAILED: x")

    monkeypatch.setattr(nightly, "run_nightly", boom)
    monkeypatch.setattr(nightly, "single_run_lock",
                        lambda: contextlib.nullcontext())

    result = nightly.main([
        "--as-of", "2026-10-13",
        "--no-refresh", "--no-tiers", "--no-publish", "--no-backfill",
    ])

    assert result == 1, "a typed NightlyStop is an exit-1 run"
    assert "TIER4_REPAIR_FAILED: x" in capsys.readouterr().err, \
        "the typed code must reach stderr"
