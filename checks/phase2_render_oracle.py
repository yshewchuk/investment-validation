#!/usr/bin/env python3
"""D19 oracle: the legacy-way render bundle, built directly from v1 helpers.

Moved out of ``tests/test_v2_ops_render_parity.py`` (P2-C19 review note: "one
oracle, callable from the test and the real comparator"). This module is a
pure builder over *explicit, already-loaded* inputs -- it never resolves a
job, opens a catalog, or touches ``INVESTING_PLAN_ROOT`` itself. Two callers
share it:

* ``tests/test_v2_ops_render_parity.py`` (D19, synthetic, tier 0) calls it
  in-process against a monkeypatched panel/trades/registry.
* ``checks/rearchitecture_phase2_render_parity.py``'s bounded subprocess
  worker calls it against real inputs loaded from a render job's recorded
  bindings, with ``INVESTING_PLAN_ROOT`` already pointed at the private
  legacy root by its caller.

``engine/dashboard/nightly.py:1580-1622`` is the v1 call site this mirrors;
line numbers are quoted in the docstrings below so a future drift in v1's own
argument order is easy to notice.

``checks/`` may import both ``engine.v2.ops`` and legacy ``engine.dashboard``/
``engine.data`` modules (phase-2 guide layering: checks is verification
tooling, not a production layer) -- this module does both, which is exactly
why it may not live under ``engine/v2/ops`` (that package may not import
``engine.v2.diagnosis``, and more to the point, ``render_inputs.py``'s own
docstring reserves legacy calls to ``legacy_adapter`` alone, to keep exactly
one adapter module per package).
"""
from __future__ import annotations

import pandas as pd

__all__ = ["legacy_way_bundle", "v1_flags_for_scenario"]


def v1_flags_for_scenario(*, requested_as_of, resolved_as_of, finality, tickers,
                          horizon_days, evidence) -> list[dict]:
    """The full v2 render flag list, reconstructed independently of
    ``legacy_adapter``/``render_inputs.render_flags`` -- v1's own helpers
    (``_panel_staleness_flags``, ``_date_conflict_flag``) are called directly
    on the SAME scenario, so a real v2 omission shows up as a diff instead of
    being echoed by a synthetic expected side that trusts the same adapter it
    is meant to check (P2-C08 decision 4).
    """
    from engine.dashboard.nightly import _date_conflict_flag, _panel_staleness_flags
    from engine.data import store
    from engine.v2.ops.render_inputs import (
        absent_stage_flags,
        model_evidence_stale_flag,
        unknown_operational_flags,
    )
    from engine.v2.ops.session_resolution import walk_back_flag

    resolved_ts = pd.Timestamp(resolved_as_of)
    flags = list(absent_stage_flags())
    back = walk_back_flag(requested_as_of, resolved_as_of, finality)
    if back is not None:
        flags.append(back)
    flags.extend(_panel_staleness_flags(resolved_ts))
    events = store.read_table(
        "earnings_events",
        columns=["event_id", "ticker", "event_date", "session", "date_conflict"])
    events["event_date"] = pd.to_datetime(events["event_date"])
    horizon = resolved_ts + pd.Timedelta(days=int(horizon_days))
    window = events[(events["event_date"] >= resolved_ts) & (events["event_date"] <= horizon)
                    & events["session"].notna()]
    if tickers:
        window = window[window["ticker"].isin(set(tickers))]
    conflict = _date_conflict_flag(window)
    if conflict is not None:
        flags.append(conflict)
    stale = model_evidence_stale_flag(evidence)
    if stale is not None:
        flags.append(stale)
    flags.extend(unknown_operational_flags())
    return flags


def legacy_way_bundle(out_dir, *, scores, panel, trades, registry, finality,
                      requested_as_of, resolved_as_of, tickers, horizon_days,
                      evidence=None, selfcheck_report=None):
    """``engine/dashboard/nightly.py:1580-1622``, called directly on explicit,
    already-loaded inputs -- the same scores/panel/trades/registry a v2
    render job used, so the two paths read the same ledger generation when
    ``INVESTING_PLAN_ROOT`` is pointed at the same staged/materialized root.
    Flags and health are rebuilt independently too
    (:func:`v1_flags_for_scenario`), never by trusting v2's own adapter
    output.
    """
    from engine.dashboard.render import (
        build_health,
        build_meta,
        freshness_summary,
        quota_state,
        render_bundle,
        size_model_mae_from_ledger,
    )
    from engine.v2.ops.render_inputs import unknown_selfcheck_report

    resolved_ts = pd.Timestamp(resolved_as_of)
    meta = build_meta(scores, as_of=resolved_ts, horizon_days=horizon_days, fill_alpha=0.5,
                      alt_strikes=1, freshness=freshness_summary(resolved_ts),
                      quota=quota_state(), registry=registry)
    meta["execution_clock"] = {"requested_as_of": str(pd.Timestamp(requested_as_of).date()),
                               "resolved_as_of": str(resolved_ts.date()), "finality": finality}
    flags = v1_flags_for_scenario(
        requested_as_of=requested_as_of, resolved_as_of=resolved_as_of, finality=finality,
        tickers=tickers, horizon_days=horizon_days, evidence=evidence or {})
    health = build_health(as_of=resolved_ts, size_mae=size_model_mae_from_ledger(panel=panel),
                          selfcheck_report=selfcheck_report or unknown_selfcheck_report())
    return render_bundle(scores, out_dir, as_of=resolved_ts, horizon_days=horizon_days,
                         fill_alpha=0.5, alt_strikes=1, panel=panel, trades=trades,
                         meta=meta, health=health, flags=flags, registry=registry)
