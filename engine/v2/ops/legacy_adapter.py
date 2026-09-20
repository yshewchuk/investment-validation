"""Audited bridge to the frozen legacy tree."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from engine.data.finality import _coverage_frame as _legacy_coverage_frame
from engine.data.finality import _market_wide_complete as _legacy_market_wide_complete
from engine.data.finality import covered_tickers as _legacy_covered_tickers
from engine.data.finality import resolve_final_session as _legacy_resolve_final_session
from engine.data.finality import session_finality as _legacy_session_finality
from engine.v2.foundation import safe_relative_path
from engine.v2.ops import worker_progress
from engine.v2.ops.decision_replay import score_row_id as _score_row_id
from engine.v2.ops.errors import fail
from engine.v2.ops.finality import covered_tickers, resolve_final_session, session_finality

_LEGACY_FINALITY_ORIGINALS = {
    "session_finality": _legacy_session_finality,
    "resolve_final_session": _legacy_resolve_final_session,
    "covered_tickers": _legacy_covered_tickers,
    # R3B-7 fix: the two private helpers ``session_finality``/``covered_tickers``
    # actually read data through (below) need the SAME monkeypatch seam as the
    # public names above. Before this, a test that patched only
    # ``engine.data.finality._market_wide_complete``/``_coverage_frame`` (the
    # realistic shape -- those are what a fixture patches to fake data without
    # faking every public entry point) had no effect on the native v2 path:
    # ``covered_tickers`` alone was untouched, so ``finality_compatibility``
    # left it on the NATIVE implementation, which reads real ORATS cache /
    # ``engine.data.store`` through ``finality_market_wide_complete``/
    # ``finality_coverage_frame`` below -- both fail closed (empty/None) with
    # no real data present, so every ticker came back uncovered.
    "_market_wide_complete": _legacy_market_wide_complete,
    "_coverage_frame": _legacy_coverage_frame,
}

__all__ = ["copy_read_set", "invoke_evaluate", "invoke_nightly_helper",
           "invoke_price_refresh", "invoke_score_calendar", "iter_raw_fetch_cache",
           "manifest_files", "overlay_read_set", "projected_trading_sessions",
           "run_engineering_gate", "run_legacy_rebuild", "run_legacy_script",
           "run_security_scan", "verify_export_generation"]


def projected_trading_sessions(start, end) -> tuple[str, ...]:
    """``engine.calendar.projected_trading_days`` -- a PURE weekday/US-market-
    holiday rule, no ``engine.paths`` dependency (module docstring's
    distinction: this is safe to call without ``_rooted_import``, unlike
    ``trading_calendar()`` itself, which resolves ``engine.paths.GSPC_DAILY``
    off the process-global root).

    2026-09-14 (capture-inputs): the real observed calendar CSV can lag
    "today" by weeks (whatever the last successful calendar pull wrote), so
    the finality lookback window must extend PAST the last observed date the
    same way ``trading_calendar()`` itself does (``calendar.py``'s own
    ``extend_days`` construction) -- an observed-only walk silently resolves
    a stale multi-week-old window instead of the real recent sessions.
    """
    from engine.calendar import projected_trading_days
    return tuple(str(d.date()) for d in projected_trading_days(start, end))


def iter_raw_fetch_cache(root: Path | str, source: str):
    """``engine.data.fetch.iter_cached``, rooted at an explicit directory.

    2026-09-14 (capture-inputs, task brief): never ``engine.paths.RAW_FETCH``'s
    process-global default -- ``engine.v2.ops.capture_inputs`` runs against an
    arbitrary ``--source-root`` in a process that may already have imported
    ``engine.paths`` (bound to this checkout's own root at first import,
    module-level, never re-read). ``iter_cached``'s own ``root=`` parameter
    sidesteps that entirely; this wrapper only fixes the directory shape it
    expects (``<root>/source``, matching ``engine.paths.RAW_FETCH / source``).
    """
    from engine.data.fetch import iter_cached
    return iter_cached(source, root=Path(root) / "data" / "raw" / "fetch")


def finality_compatibility(name, native):
    """Expose the legacy finality seam to the native v2 implementation."""
    legacy = sys.modules["engine.data.finality"]
    original = _LEGACY_FINALITY_ORIGINALS[name]
    current = getattr(legacy, name)
    return current if original is not None and current is not original else native


def finality_market_wide_complete(stamp) -> bool:
    """Read the legacy ORATS cache used by the finality barrier."""
    from engine.data.fetch import iter_cached

    target = str(stamp.date())
    found = set()
    for entry in iter_cached("orats"):
        if entry.endpoint in {"hist/summaries", "hist/cores"} \
                and str(entry.params.get("tradeDate")) == target \
                and int(entry.meta.get("status", 0)) == 200:
            found.add(entry.endpoint)
    return found == {"hist/summaries", "hist/cores"}


def finality_coverage_frame(table: str, column: str, stamp):
    """Read bounded legacy coverage columns for native finality."""
    from engine.data.store import read_table

    try:
        return read_table(table, years=sorted({stamp.year - 1, stamp.year}),
                          columns=["ticker", column])
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return None


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def manifest_files(root: Path | str, paths: tuple[str, ...] | list[str]) -> dict:
    """Return a complete, immutable read-set manifest and reject indirection."""
    base = Path(root).resolve()
    result = {}
    for relative in paths:
        safe_relative_path(relative)
        source = base / relative
        if source.is_symlink() or not source.is_file():
            raise fail("INPUT_CHANGED", "legacy read-set member is missing or indirect",
                       details={"path": relative})
        result[relative] = {"content_hash": _digest(source),
                            "byte_size": source.stat().st_size}
    return result


def copy_read_set(source_root: Path | str, private_root: Path | str,
                  paths: tuple[str, ...] | list[str]) -> dict:
    """Copy declared inputs privately, preserving bytes without links."""
    source_raw = Path(source_root)
    target_raw = Path(private_root)
    if source_raw.is_symlink() or target_raw.is_symlink():
        raise fail("INTEGRITY_FAILED", "private root may not be a symlink")
    source = source_raw.resolve()
    target = target_raw.resolve()
    manifest = manifest_files(source, paths)
    target.mkdir(parents=True, exist_ok=True)
    for relative, expected in manifest.items():
        current = source
        for component in Path(relative).parts[:-1]:
            current = current / component
            if current.is_symlink():
                raise fail("INPUT_CHANGED", "legacy read-set ancestor is indirect")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        probe = target
        for component in Path(relative).parts[:-1]:
            probe = probe / component
            if probe.is_symlink():
                raise fail("INTEGRITY_FAILED", "private destination ancestor is a symlink")
        if destination.exists() and destination.is_symlink():
            raise fail("INTEGRITY_FAILED", "private destination is a symlink")
        shutil.copyfile(source / relative, destination)
        if destination.is_symlink() or _digest(destination) != expected["content_hash"]:
            raise fail("INTEGRITY_FAILED", "private legacy copy failed verification",
                       details={"path": relative})
        destination.chmod(0o444)
    return manifest


def overlay_read_set(materialization_root: Path | str, private_root: Path | str) -> None:
    """Mirror a verified, read-only materialization root into a fresh
    writable private directory with one symlink per file (real shadow
    nightly attempt 19 fix).

    ``legacy_render`` needs a legacy tree it can write into (the bound
    model-evidence artifact and ledger generation land at their legacy paths,
    ``render_inputs.stage_model_evidence``/``stage_ledger_generation``), but
    ``materialization_root`` is shared across every attempt bound to the same
    request hash and its own integrity check (``snapshot_roots._walk``)
    requires every file to keep ``st_nlink == 1`` -- so it must never be
    written into, and a file inside it must never gain a second (hard) link.
    A symlink is not a link in that sense (it does not touch the target's
    link count) and costs no bytes, so this makes the SAME pinned bytes score
    read reachable at a writable path for free, without copying ~600+ curated
    files. Never touches ``materialization_root``; raises if ``private_root``
    already exists, since it must be a fresh per-attempt directory.
    """
    source = Path(materialization_root)
    if source.is_symlink():
        raise fail("INTEGRITY_FAILED", "materialization root may not be a symlink")
    source = source.resolve()
    target = Path(private_root)
    if target.exists() or target.is_symlink():
        raise fail("INTEGRITY_FAILED", "overlay destination must not already exist")
    target.mkdir(parents=True)
    for entry in sorted(source.rglob("*")):
        relative = entry.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if entry.is_dir():
            destination.mkdir(exist_ok=True)
        else:
            destination.symlink_to(entry)


def _rooted_import(root: Path | str):
    path = Path(root).resolve()
    os.environ["INVESTING_PLAN_ROOT"] = str(path)


def invoke_score_calendar(root, as_of, *, scorer, tickers=None, horizon_days=35):
    """Call the existing scorer with its public legacy arguments."""
    _rooted_import(root)
    from engine.score import score_calendar
    return score_calendar(as_of, horizon_days=horizon_days, alt_strikes=0,
                          scorer=scorer, tickers=tickers, progress_every=10)


def invoke_price_refresh(session, *, dry_run: bool = False):
    """Plan (and, unless ``dry_run``, run) one ``ops price-refresh`` pass.

    ``engine.v2.ops.cli`` may not import ``engine.data.*`` directly -- one
    adapter module per package (``checks/import_layers.py`` §4.2) -- so this
    is the whole crossing: ``engine.data.pulls.price_refresh`` (the actual
    planning/fetch logic, this task's own new module) plus
    ``engine.data.fetch.Fetcher`` for a real run. Never rooted via
    ``_rooted_import``: unlike the snapshot-materialization callers above,
    price-refresh has no ``--store-root``/``--source-root`` of its own yet
    (out of this task's scope) and always reads/writes the checkout it
    actually runs in.
    """
    from engine.data.pulls.price_refresh import (
        load_events,
        load_fetch_history,
        load_price_universe,
        plan_refresh,
        run_refresh,
    )

    plan = plan_refresh(session, events=load_events(), price_universe=load_price_universe(),
                        fetch_history=load_fetch_history())
    if dry_run:
        return {"plan": plan, "report": None}
    from engine.data.fetch import Fetcher
    return {"plan": plan, "report": run_refresh(plan, Fetcher())}


def _write_action(root, name, value):
    import json

    from engine.v2.foundation import content_hash, tag_nonfinite

    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    # Real shadow nightly attempt 14: engine.dashboard.model_evidence can
    # legitimately produce a NaN (a Spearman correlation on a constant
    # input, engine/dashboard/model_evidence.py:104), and legacy writes it
    # as a raw JSON `NaN` literal (model_evidence.py:412's bare
    # ``json.dumps``, no ``allow_nan=False``) -- Python's own json.load reads
    # that literal back as float('nan') without complaint. Dumping the SAME
    # raw ``value`` here with ``allow_nan=False`` doesn't reject the literal
    # the way legacy's own writer would -- ``json.dumps`` has no way to
    # express a Python NaN as anything else, so it raises ValueError instead
    # (worker.py then had no typed mapping for that and it surfaced as a
    # retried WORKER_FAILED). ``tag_nonfinite`` swaps every NaN/Infinity for
    # the ``{"__nonfinite__": repr(value)}`` marker canonical_json/
    # content_hash already use for the SAME value (contracts §2.1) before
    # this ever reaches ``json.dumps``, so the artifact stays valid strict
    # JSON and ``allow_nan=False`` has nothing left to refuse.  A reader
    # that needs the real float back (v2's own, or a legacy consumer that
    # must see exactly what legacy itself would have produced) applies
    # ``untag_nonfinite`` at its own read boundary -- see
    # ``_action_render``'s ``evidence`` load and
    # ``render_inputs.stage_model_evidence``.
    path.write_text(json.dumps(tag_nonfinite(value), indent=2, sort_keys=True,
                               default=str, allow_nan=False))
    return {"path": str(path.relative_to(root)), "hash": content_hash(value)}


def legacy_action(action, parameters, staging, legacy_root=None, cross_check=None):
    """Execute one registered legacy stage inside a fresh worker process.

    ``legacy_root``: a snapshot-backed stage's verified materialization root
    (P2-6 §9.3); ``None`` keeps the barrier path's ``staging/legacy``.
    ``cross_check`` (last read-set gap fix, 2026-09-15): only ever non-empty
    for ``legacy_finality``, when its launch resolved a ``finality_check``
    materialization (``snapshot_stages.prepare_launch``) -- see
    ``_action_finality``."""
    root = Path(staging).resolve()
    _rooted_import(Path(legacy_root) if legacy_root else root / "legacy")
    actions = {
        "legacy_finality": _action_finality,
        "legacy_features": _action_features,
        "legacy_score": _action_score,
        "legacy_decisions": _action_decisions,
        "legacy_settlement": _action_settlement,
        "legacy_model_evidence": _action_model_evidence,
        "legacy_render": _action_render,
        "legacy_selfcheck": _action_selfcheck,
        "legacy_score_requests": _action_score_requests,
        "legacy_decision_replay": _action_decision_replay,
    }
    if action not in actions:
        raise fail("INVALID_REQUEST", "legacy action is not allowlisted")
    # A step spanning the whole action, so every action has at least one
    # step event even where nothing inside it is instrumented more finely
    # below (Do §1: "instrument every legacy_adapter action").
    with worker_progress.step(action):
        if action == "legacy_finality":
            return _action_finality(parameters, root, cross_check=cross_check)
        return actions[action](parameters, root)


def _materialized_curated_frame(materialization_root, table, column, stamp):
    """``{ticker, column}`` rows a verified materialization root carries for
    ``table``, read directly off its curated year partitions -- never through
    ``engine.data.store``/``engine.paths``: this worker process is already
    rooted at the barrier legacy tree (``_rooted_import`` binds
    ``engine.paths`` once, at first import, per this module's own
    ``run_legacy_rebuild`` docstring), so a SECOND root cannot be read
    through that same high-level API in the same process.

    Mirrors ``engine.data.finality._coverage_frame``'s own
    ``{ticker, column}``/``{stamp.year - 1, stamp.year}`` shape exactly, so
    the two are directly comparable row for row (last read-set gap fix,
    2026-09-15). Returns an EMPTY frame, never ``None``, when a year
    partition is missing: ``_exact_share``'s own ``frame is None`` branch
    falls back to reading the BARRIER's live tree, which would silently
    defeat this cross-check by comparing the barrier against itself.
    """
    import pandas as pd

    root = Path(materialization_root)
    frames = []
    for year in sorted({stamp.year - 1, stamp.year}):
        year_dir = root / "data" / "curated" / table / f"year={year}"
        if not year_dir.is_dir():
            continue
        for part in sorted(year_dir.glob("part-*.parquet")):
            frames.append(pd.read_parquet(part, columns=["ticker", column]))
    if not frames:
        return pd.DataFrame(columns=["ticker", column])
    return pd.concat(frames, ignore_index=True)


def _cross_check_finality_against_materialization(parameters, result, materialization_root):
    """Read-set gap fix, decision (b) for ``legacy_finality`` (task facts:
    a byte comparison is the wrong cross-check here -- the 3 differing
    ``option_chains`` parquet files real attempt 19/20 measured are a
    projection artifact, not drift; compare CONTENT within each consumer's
    own declared projection instead).

    ``result`` was resolved against the barrier's live-tree read
    (``engine.data.finality.resolve_final_session``, ``_action_finality``
    below). This recomputes ``engine.data.finality.session_finality`` for the
    SAME resolved date and tickers, sourced from the run's own committed
    materialization instead of the live tree, reusing the exact coverage math
    (``_exact_share``, via ``session_finality``'s own ``frames=`` parameter)
    finality already applies to its own declared projection
    (``ticker``+``date``/``obs_date`` only) -- so a change confined to a
    column finality never reads (a projection-only difference) can never
    trip this, and a changed IN-SCOPE row (a ticker covered at the resolved
    date in one source and not the other) always does.

    Refuses only when the materialization does NOT independently confirm the
    same session final for the same tickers -- the real risk the task names:
    the live tree grows daily, so the barrier can resolve a session the
    pinned materialization has no rows for at all, while
    ``legacy_score``/``legacy_render``/``legacy_selfcheck`` read ONLY the
    materialization.
    """
    import pandas as pd

    stamp = pd.Timestamp(result.date)
    frames = {
        "daily_market": _materialized_curated_frame(materialization_root, "daily_market",
                                                     "date", stamp),
        "option_chains": _materialized_curated_frame(materialization_root, "option_chains",
                                                      "obs_date", stamp),
    }
    materialized = session_finality(result.date, parameters["tickers"], frames=frames)
    if not materialized.is_final:
        raise fail(
            "SOURCE_NOT_FINAL",
            "the resolved session is final on the live legacy tree but this run's own "
            "materialization does not independently confirm it",
            details={"date": result.date, "reason": "finality_snapshot_drift",
                     "materialized_detail": materialized.detail,
                     "materialized_daily_share": materialized.daily_share,
                     "materialized_chain_share": materialized.chain_share,
                     "materialized_covered": materialized.covered})


def _action_finality(parameters, root, cross_check=None):
    from engine.calendar import trading_calendar

    # P2-C03: v1 never falls back to the requested date when no session
    # qualifies (engine/dashboard/nightly.py:1223-1232 stops the run); the
    # supervised equivalent is an explicit, registered refusal so nothing
    # downstream can mistake a missing walk-back for a scored session.
    try:
        result = resolve_final_session(parameters["session"], parameters["tickers"],
                                       calendar=trading_calendar())
    except RuntimeError as exc:
        raise fail("SOURCE_NOT_FINAL", str(exc)) from exc
    # Last read-set gap fix (2026-09-15): only set when this attempt's launch
    # resolved a verified materialization for this exact run
    # (``snapshot_stages``'s ``finality_check`` mode) -- absent for a plain
    # legacy-mode nightly, unchanged from before this fix.
    materialization_root = (cross_check or {}).get("materialization_root")
    if materialization_root:
        _cross_check_finality_against_materialization(parameters, result, materialization_root)
    # finality.json's dict is embedded verbatim into ledger rows (v1 parity);
    # per-ticker coverage is a SEPARATE output, never a key added here.
    primary = _write_action(root, "finality.json", result.as_dict())
    coverage = _write_action(root, "finality_coverage.json", {
        "schema_version": "finality_coverage.v1.0", "date": result.date,
        "covered_tickers": covered_tickers(result.date, parameters["tickers"])})
    primary["extra"] = [{"name": "legacy_finality_coverage", "path": coverage["path"],
                         "schema": "finality_coverage.v1.0"}]
    return primary


def _load_features(root):
    path = root / "features.json"
    if not path.is_file():
        raise fail("FEATURES_MISSING",
                   "no features receipt for this session — the features stage must "
                   "run (and record the panel/tier4 hashes it built) before score")
    return json.loads(path.read_text())


def _current_features_hashes():
    """The Tier-3 panel's and Tier-4 forecast table's CURRENT content
    hashes, read off whatever root this worker process is presently rooted
    at (``_rooted_import`` -- the barrier staging copy, or a verified
    materialization). ``None`` for a file that does not exist, mirroring
    ``engine.data.manifest``'s own optional digests."""
    from engine import paths
    from engine.data import store

    return {
        "panel_sha256": store.file_sha256(paths.PANEL) if paths.PANEL.exists() else None,
        "tier4_sha256": store.file_sha256(paths.TIER4) if paths.TIER4.exists() else None,
    }


def _check_features_current(root):
    """P6-2: refuse a score launch whose features receipt does not match
    what the panel and Tier-4 forecast table actually are right now.

    Two distinct refusals, on purpose (identity must not be able to no-op,
    the way ``_phase4_tier4_digest`` and the pre-registration guard's
    ``spec_hash_checked`` did -- a hash written and never read):

    * ``FEATURES_MISSING`` -- no ``features.json`` in this session's root at
      all (``_load_features``). This is the common, dangerous case (a
      resumed run, a fresh catalog, a skipped stage) and must never
      silently pass.
    * ``FEATURES_STALE`` -- a receipt exists but the panel and/or tier4 file
      it recorded no longer matches what is on disk (the features stage
      ran, but something rebuilt or replaced Tier 3/4 afterward).

    Called unconditionally at the top of :func:`_action_score` -- never
    gated behind ``input_mode``, unlike
    ``snapshot_stages._check_tier4_coverage`` -- because the real, default
    nightly plan is ``input_mode="legacy"`` (``plans.nightly_plan``'s own
    default), which never reaches ``snapshot_stages.prepare_launch`` at all
    (``prepare_launch`` returns ``None`` for ``launch_mode(spec) ==
    "legacy"``). A check that lived only in ``snapshot_stages.py`` would be
    exactly the kind of no-op this task exists to close. See
    ``tools/phase6_capabilities.toml`` row ``nightly-features``.
    """
    receipt = _load_features(root)
    current = _current_features_hashes()
    mismatches = {key: {"receipt": receipt.get(key), "current": current[key]}
                 for key in ("panel_sha256", "tier4_sha256") if receipt.get(key) != current[key]}
    if mismatches:
        raise fail("FEATURES_STALE",
                   "the panel and/or tier4 forecast table on disk no longer matches "
                   "this session's features receipt",
                   details={"mismatches": mismatches})


def _action_features(parameters, root):
    """The Tier-3 panel and Tier-4 forecast rebuild (P6-2), wrapped as a
    supervised legacy action: ``engine.data.rebuild``'s own panel/tier4
    builders, run inside this worker's rooted staging tree
    (``legacy_action``'s ``_rooted_import``), then a ``features.json``
    receipt binding the exact files a downstream ``legacy_score`` must see
    unchanged (``_check_features_current``) -- the identity
    ``nightly-features`` never had before this task.
    """
    from engine.data import rebuild

    worker_progress.step_start("panel_rebuild")
    panel_report = rebuild.build_panel_table()
    worker_progress.step_end("panel_rebuild")
    worker_progress.step_start("tier4_rebuild")
    tier4_report = rebuild.build_tier4_table(parameters.get("tier4_since"))
    worker_progress.step_end("tier4_rebuild")
    hashes = _current_features_hashes()
    return _write_action(root, "features.json", {
        "schema_version": "features.v1.0",
        "panel_sha256": hashes["panel_sha256"],
        "tier4_sha256": hashes["tier4_sha256"],
        "panel_rows": panel_report.get("rows"),
        "tier4_rows": tier4_report.get("rows"),
    })


def _scoring_context(parameters, *, action):
    """P2-C04's single implementation of the scorer's evidence universe
    (``context_tickers``) plus its ``years`` window, used by every action
    that builds ``Scorer(context=FeatureContext.load(...))`` -- score,
    decision_replay, render and selfcheck alike -- so they cannot drift.

    External review finding (2026-09-14 fix): render and self-check built
    their ``FeatureContext`` off the bare watchlist (``parameters["tickers"]``)
    instead of ``context_tickers``, unlike score/decision_replay. A subset
    run (e.g. watchlist AAA, context AAA+BBB) could then score correctly yet
    fail render/self-check's own rescoring, blocking publication.

    ``legacy_score``/``legacy_decision_replay`` predate ``context_tickers``
    and default a caller that omits it to its own ``tickers`` -- unchanged,
    still exactly what ``test_action_score_context_tickers_defaults_to_the_watchlist``
    pins. ``legacy_render``/``legacy_selfcheck`` never had that grandfather
    clause: ``build_legacy_job_requests`` (``engine/v2/ops/nightly.py``
    ``_legacy_params``) has ALWAYS threaded ``context_tickers`` onto every
    stage's parameters, so a render/selfcheck job actually missing it can
    only mean a caller bypassed the plan builder or a payload regression
    reintroduced this exact defect -- refused typed here, never silently
    narrowed back to the watchlist (that silent narrowing is the bug).
    """
    context_tickers = parameters.get("context_tickers")
    if not context_tickers:
        if action in ("legacy_render", "legacy_selfcheck"):
            raise fail("VALIDATION_FAILED",
                      "legacy plan payload is missing context_tickers for this stage",
                      details={"action": action})
        context_tickers = parameters["tickers"]
    years = range(int(parameters["year_start"]), int(parameters["year_end"]) + 1)
    return sorted(set(context_tickers)), years


def _action_score(parameters, root):
    import pandas as pd

    from engine.dashboard.nightly import strike_ladder
    from engine.features import FeatureContext
    from engine.jsonio import json_safe
    from engine.score import Scorer, score_calendar
    from engine.v2.ops.session_resolution import resolve_effective_session

    _check_features_current(root)
    worker_progress.step_start("inputs_load")
    session = resolve_effective_session(_load_finality(root), parameters["session"])
    tickers = sorted(set(parameters["tickers"]))
    # P2-C04: the historical EVIDENCE universe (analog pools, registered
    # champions) is ``context_tickers``, never the direct watchlist — a
    # narrow watchlist must not shrink the context it is scored against.
    # Defaults to ``tickers`` for a caller that predates this parameter.
    context_tickers, years = _scoring_context(parameters, action="legacy_score")
    worker_progress.step_end("inputs_load")
    worker_progress.step_start("scorer_build")
    scorer = Scorer(context=FeatureContext.load(context_tickers, years=years))
    worker_progress.step_end("scorer_build")
    worker_progress.step_start("score_calendar")
    frame = score_calendar(pd.Timestamp(session),
                          horizon_days=int(parameters.get("horizon_days", 35)),
                          alt_strikes=0, scorer=scorer, tickers=tickers,
                          progress_every=10)
    worker_progress.step_end("score_calendar", units=len(tickers))
    rows = json_safe(frame.to_dict(orient="records"), round_to=None)
    for row in rows:
        row["row_id"] = _score_row_id(row)
    expected = tuple(parameters.get("expected_population", ()))
    if not expected:
        raise fail("VALIDATION_FAILED", "score population must be planned before execution")
    if len(set(expected)) != len(expected):
        raise fail("VALIDATION_FAILED", "planned population has duplicate keys")
    observed_keys = {_population_key(row) for row in rows}
    missing = sorted(set(expected) - observed_keys)
    unplanned = sorted(observed_keys - set(expected))
    if missing or unplanned:
        raise fail("VALIDATION_FAILED", "score population differs from planned inputs",
                   details={"missing": missing, "unplanned": unplanned})
    ladder = strike_ladder(frame, scorer=scorer,
                           alt_strikes=int(parameters.get("alt_strikes", 1)),
                           as_of=pd.Timestamp(session))
    worker_progress.step_start("write_outputs")
    output = _write_action(root, "score.json", {
        "rows": rows, "expected_population": list(expected),
        "observed_population": sorted(observed_keys), "ladder": json_safe(ladder, round_to=None),
        "tickers": tickers, "context_tickers": context_tickers,
        "analog_entry_coverage": scorer.analog_entry_coverage,
        "session": session, "requested_session": parameters["session"]})
    worker_progress.step_end("write_outputs")
    return output


def _action_score_requests(parameters, root):
    import dataclasses
    import json

    import pandas as pd

    from engine.features import FeatureContext
    from engine.jsonio import json_safe
    from engine.score import UNSCORABLE, FillModel, Scorer, ScoreRequest, unscorable_result

    entries = json.loads((root / parameters["requests_path"]).read_text())
    requests = [entry["request"] for entry in entries]
    tickers = sorted({str(row["ticker"]) for row in requests})
    years = range(int(parameters["year_start"]), int(parameters["year_end"]) + 1)
    worker_progress.step_start("scorer_build")
    scorer = Scorer(context=FeatureContext.load(tickers, years=years))
    worker_progress.step_end("scorer_build")
    rows = []
    fields = {field.name for field in dataclasses.fields(ScoreRequest)}
    dates = {"as_of", "event_date", "expiry", "chain_as_of"}
    worker_progress.step_start("scoring_loop")
    for entry, request in zip(entries, requests):
        values = {key: value for key, value in request.items() if key in fields}
        for key in dates:
            if isinstance(values.get(key), str):
                values[key] = pd.Timestamp(values[key])
        if isinstance(values.get("fill"), dict):
            values["fill"] = FillModel(alpha=float(values["fill"]["alpha"]))
        score_request = ScoreRequest(**values)
        try:
            result = scorer.score(score_request)
        except UNSCORABLE as exc:
            result = unscorable_result(score_request, as_of=score_request.as_of,
                                       snapshot=scorer.snapshot, exc=exc)
        record = json_safe(result.as_dict(), round_to=None)
        rows.append({"request_id": entry["canary_id"],
                     "record": record})
    worker_progress.step_end("scoring_loop", units=len(requests))
    worker_progress.step_start("write_outputs")
    output = _write_action(root, "score_requests.json", {"rows": rows,
                                                          "expected_population": len(requests)})
    worker_progress.step_end("write_outputs")
    return output


def _population_key(row):
    """A2: the planned population is keyed before strike/expiry are known —
    they only exist after scoring, so the plan cannot name them in advance."""
    return "|".join(str(row.get(key, "")) for key in ("ticker", "strategy", "event_date"))


def _load_action_frame(root):
    import json

    import pandas as pd

    path = root / "score.json"
    if not path.is_file():
        raise fail("INPUT_CHANGED", "score artifact is missing")
    return pd.DataFrame(json.loads(path.read_text())["rows"])


def _load_finality(root):
    import json

    path = root / "finality.json"
    if not path.is_file():
        raise fail("INPUT_CHANGED", "finality artifact is missing")
    return json.loads(path.read_text())


def _load_score_document(root):
    path = root / "score.json"
    if not path.is_file():
        raise fail("INPUT_CHANGED", "score artifact is missing")
    return json.loads(path.read_text())


def _empty_replay(session, requested_session):
    from engine.v2.foundation import content_hash
    return {"schema_version": "decision_replay.v1.0", "session": session,
            "requested_session": requested_session,
            "population": [], "source_rows": [], "replayed_rows": [],
            "source_rows_hash": content_hash([]), "replayed_rows_hash": content_hash([]),
            "findings": []}


def _action_decision_replay(parameters, root):
    """B1b: re-score the decision-eligible board rows through the SAME public
    entrypoint the score stage used, in a fresh process.

    Never rebuilds a ``ScoreRequest`` by hand: a DYN-SV chooser row is not a
    request the engine can score on its own (``DYN-SV`` names no structure,
    only ``score_calendar``'s own menu step can produce one), so the only
    faithful replay is re-running ``score_calendar`` itself and letting the
    chooser rank the menu again.

    ``FeatureContext`` loads ``context_tickers`` (P2-C04) — the historical
    EVIDENCE universe, defaulting to ``parameters["tickers"]`` for a caller
    that predates this parameter — analog pools and the registered
    gate/chooser champions are read off that context, not off
    ``score_calendar``'s own ``tickers`` argument, so a narrower context
    would be a different computation. ``score_calendar``'s own ``tickers``
    argument, by contrast, only restricts which events are enumerated and
    which chains are pre-loaded (read ``engine.score.score_calendar``: the
    events table is filtered by ticker before anything cross-request is
    built, and the DYN-SV chooser groups by event, never across tickers) — so
    it is safe, and far cheaper, to restrict it to just the eligible rows'
    own tickers rather than rescoring the whole board.
    """
    from engine.jsonio import json_safe
    from engine.v2.foundation import content_hash
    from engine.v2.ops.decision_replay import compare_rows, decision_population, population_key
    from engine.v2.ops.session_resolution import resolve_effective_session

    session = resolve_effective_session(_load_finality(root), parameters["session"])
    population = decision_population(_load_score_document(root), session)
    if not population:
        return _write_action(root, "replay.json", _empty_replay(session, parameters["session"]))

    import pandas as pd

    from engine.features import FeatureContext
    from engine.score import Scorer, score_calendar

    worker_progress.step_start("scorer_build")
    context_tickers, years = _scoring_context(parameters, action="legacy_decision_replay")
    scorer = Scorer(context=FeatureContext.load(context_tickers, years=years))
    worker_progress.step_end("scorer_build")
    eligible_tickers = sorted({str(row["ticker"]) for row in population})
    worker_progress.step_start("replay")
    frame = score_calendar(pd.Timestamp(session),
                          horizon_days=int(parameters.get("horizon_days", 35)),
                          alt_strikes=0, scorer=scorer, tickers=eligible_tickers)
    rows = json_safe(frame.to_dict(orient="records"), round_to=None)
    for row in rows:
        row["row_id"] = _score_row_id(row)
    eligible_keys = {population_key(row) for row in population}
    replayed = [row for row in rows if population_key(row) in eligible_keys]
    worker_progress.step_end("replay", units=len(eligible_tickers))
    worker_progress.step_start("write_outputs")
    output = _write_action(root, "replay.json", {
        "schema_version": "decision_replay.v1.0", "session": session,
        "requested_session": parameters["session"],
        "population": [population_key(row) for row in population],
        "source_rows": population, "replayed_rows": replayed,
        "source_rows_hash": content_hash(population),
        "replayed_rows_hash": content_hash(replayed),
        "findings": compare_rows(population, replayed)})
    worker_progress.step_end("write_outputs")
    return output


def _action_decisions(parameters, root):
    from engine.ledger import build_prediction_rows

    frame = _load_action_frame(root)
    finality = _load_finality(root)
    plan_path = root / "decision_plan.json"
    if not plan_path.is_file():
        raise fail("VALIDATION_FAILED", "decision plan artifact is missing")
    plan = json.loads(plan_path.read_text())
    # The plan's own ``session`` is decision_evidence's finality-resolved
    # date (P2-C03) — never re-derived here, so a candidate row's ``as_of``
    # always agrees with what the commit-time validator checks it against.
    rows = build_prediction_rows(frame, as_of=plan.get("session"),
                                 decision_ts=plan.get("decision_clock"),
                                 finality=finality, entry_dated_only=True)
    for row in rows:
        row["written_at"] = plan.get("decision_clock")
        row["decision_ts"] = plan.get("decision_clock")
        if not row.get("event_id"):
            row["event_id"] = (row.get("score") or {}).get("event_id")
    return _write_action(root, "decisions.json", {"rows": rows, "expected_rows": len(rows)})


def _action_settlement(parameters, root):
    import base64

    from engine.ledger import score_outcomes
    from engine.v2.ops.session_resolution import resolve_effective_session

    session = resolve_effective_session(_load_finality(root), parameters["session"])
    directory = root / "legacy" / "ledger" / "outcomes"
    before = {path: path.stat().st_size for path in directory.glob("*.jsonl")}
    result = score_outcomes(through=session)
    captured = []
    for path in sorted(directory.glob("*.jsonl")):
        data = path.read_bytes()[before.get(path, 0):]
        for raw in data.splitlines(keepends=True):
            captured.append({"original_b64": base64.b64encode(raw).decode("ascii"),
                             "row": json.loads(raw)})
    return _write_action(root, "settlement.json", {
        "result": result, "rows": captured,
        "session": session, "requested_session": parameters["session"]})


def legacy_ledger_schema_version() -> int:
    """The legacy ledger's own schema version constant
    (``engine.ledger.SCHEMA_VERSION``), bridged through the one declared
    adapter module for ``engine.v2.ops`` (§4.2 rule 2 -- every v2 -> legacy
    dependency is confined to this module, never imported directly from
    another ``engine.v2.ops`` module).

    Used to decide whether a recorded prediction is "current-schema" or
    "grandfathered" for settlement finality-proof purposes -- see
    ``engine.v2.ops.decision_commit._validate_settlement_state``.
    """
    from engine.ledger import SCHEMA_VERSION

    return SCHEMA_VERSION


def _action_model_evidence(parameters, root):
    """P2-C08: preserve v1's degraded/stale state as data, not an exception.

    v1 (``engine/dashboard/nightly.py:1552-1575``) rebuilds the evidence
    inline and, on a raised exception, degrades to whatever cached table is
    already on disk and raises ``model_evidence_stale``. v2 runs this rebuild
    and the render job in separate processes, so a raised exception here must
    become a *field* on the artifact the render job reads back, or the
    degraded/stale information never reaches it (render_inputs.model_evidence_stale_flag).

    Phase-level step events (last read-set gap fix, part 2, 2026-09-15: real
    incident job_caaed30eb5d1745ce87ed22a55dbc2e3, RESOURCE_LIMIT_EXCEEDED at
    4.38 GiB after 6.1s with the outer ``legacy_model_evidence`` step as the
    ONLY event -- no phase-level signal existed to show whether the peak came
    from the cheap fingerprint-cache read or the expensive full rebuild).
    ``model_evidence_cache_probe`` and ``model_evidence_build`` bracket
    exactly the same ``build_model_evidence()`` call ``build_model_evidence``
    itself makes internally (its own ``load_model_evidence()``/fingerprint
    check first, full rebuild only on a miss) -- no new legacy symbol crossing
    (both functions were already declared adapter symbols before this fix),
    so a killed attempt's ``diagnostics/steps.ndjson`` now shows start/end RSS
    for the read-only probe separately from the (possibly never-finishing)
    rebuild, and ``model_evidence_write`` isolates the JSON write.
    """
    from engine.dashboard.model_evidence import build_model_evidence, load_model_evidence

    with worker_progress.step("model_evidence_cache_probe"):
        # Read-only, same file build_model_evidence() itself reads first;
        # never mutates anything, purely so a step event marks how much RSS
        # was already resident before the (possibly expensive) build call.
        load_model_evidence()
    try:
        with worker_progress.step("model_evidence_build"):
            result = dict(build_model_evidence(force=bool(parameters.get("force", False))))
        result.setdefault("degraded", False)
    except Exception as exc:  # noqa: BLE001 -- v1 parity: stale evidence beats a dark board
        cached = load_model_evidence() or {}
        result = dict(cached)
        result["degraded"] = True
        result["degraded_reason"] = f"{type(exc).__name__}: {exc}"[:300]
    with worker_progress.step("model_evidence_write"):
        return _write_action(root, "model_evidence.json", result)


def _panel_lag_flags(as_of):
    """Class (a): v1's Tier-3 panel staleness check
    (``engine/dashboard/nightly.py::_panel_staleness_flags``), called
    directly against the resolved session -- no logic re-derived here."""
    from engine.dashboard.nightly import _panel_staleness_flags
    return _panel_staleness_flags(as_of)


def _calendar_conflict_flags(tickers, as_of, horizon_days):
    """Class (a): v1's per-event calendar-conflict flag
    (``engine/dashboard/nightly.py::_date_conflict_flag``), called on the
    same "upcoming" window v1 builds (``as_of`` .. ``as_of+horizon_days``,
    this render's own tickers), read fresh from the earnings_events store.
    """
    import pandas as pd

    from engine.dashboard.nightly import _date_conflict_flag
    from engine.data.store import read_table

    events = read_table(
        "earnings_events",
        columns=["event_id", "ticker", "event_date", "session", "date_conflict"])
    events["event_date"] = pd.to_datetime(events["event_date"])
    as_of_ts = pd.Timestamp(as_of)
    horizon = as_of_ts + pd.Timedelta(days=int(horizon_days))
    window = events[(events["event_date"] >= as_of_ts) & (events["event_date"] <= horizon)
                    & events["session"].notna()]
    if tickers:
        window = window[window["ticker"].isin(set(tickers))]
    flag = _date_conflict_flag(window)
    return [flag] if flag else []


def _render_meta_and_health(root, *, scores, scorer, as_of, requested_as_of,
                            resolved_as_of, finality, horizon_days, alt_strikes,
                            fill_alpha, tickers, evidence):
    """Assemble ``build_meta``/``build_health`` plus the full P2-C08 flag set."""
    from engine.dashboard.render import (
        build_health,
        build_meta,
        freshness_summary,
        quota_state,
        size_model_mae_from_ledger,
    )
    from engine.v2.ops.render_inputs import render_flags, resolve_prior_selfcheck

    meta = build_meta(scores, as_of=as_of, horizon_days=horizon_days,
                      fill_alpha=fill_alpha, alt_strikes=alt_strikes,
                      freshness=freshness_summary(as_of), quota=quota_state(),
                      registry=scorer.registry)
    meta["execution_clock"], flags = render_flags(
        requested_as_of=requested_as_of, resolved_as_of=resolved_as_of, finality=finality,
        panel_lag=_panel_lag_flags(as_of),
        calendar_conflict=_calendar_conflict_flags(tickers, as_of, horizon_days),
        model_evidence=evidence)
    health = build_health(as_of=as_of, selfcheck_report=resolve_prior_selfcheck(root),
                          size_mae=size_model_mae_from_ledger(panel=scorer.context.panel))
    return meta, health, flags


def _action_render(parameters, root):
    """P2-5/D19: render at parity with the legacy nightly's own render call.

    Carries the full legacy argument set (guide §9.4 item 2): the score
    artifact's ``rows`` + ``ladder`` concatenated exactly as the legacy
    nightly does (:func:`render_inputs.assemble_scores`), the model-evidence
    artifact placed at its legacy path, the bound ledger generation staged as
    ``legacy/ledger`` (never the mutable staged copy), and ``meta``/``health``
    built from those same legacy helpers — mirroring
    ``engine/dashboard/nightly.py:1580-1622``.
    """
    import tarfile

    import pandas as pd

    from engine.dashboard.render import render_bundle
    from engine.features import FeatureContext
    from engine.score import Scorer
    from engine.v2.foundation import untag_nonfinite
    from engine.v2.ops.render_inputs import (
        ABSENT_STAGES,
        assemble_scores,
        bundle_content_hash,
        stage_ledger_generation,
        stage_model_evidence,
    )
    from engine.v2.ops.session_resolution import resolve_effective_session

    score_document = _load_score_document(root)
    scores = assemble_scores(score_document)
    finality = _load_finality(root)
    requested_as_of = parameters["session"]
    resolved_as_of = resolve_effective_session(finality, requested_as_of)

    evidence_path = root / "model_evidence.json"
    if not evidence_path.is_file():
        raise fail("VALIDATION_FAILED", "model evidence artifact is missing")
    # v2's own read boundary: undo _write_action's {"__nonfinite__": ...} tag
    # so a NaN reads as a real float here, matching legacy's own reader.
    evidence = untag_nonfinite(json.loads(evidence_path.read_text()))
    stage_model_evidence(evidence_path, root / "legacy")

    ledger_tar = root / "ledger_generation.tar"
    if not ledger_tar.is_file():
        raise fail("VALIDATION_FAILED", "ledger generation not bound")
    stage_ledger_generation(ledger_tar, root / "legacy")
    # ``tickers`` (the direct watchlist) stays separate from the scorer's
    # evidence context below -- it only bounds _calendar_conflict_flags'
    # "this render's own tickers" window, never the historical evidence a
    # subset run's Scorer/FeatureContext must load (P2-C04 fix).
    tickers = sorted(set(parameters["tickers"]))
    context_tickers, years = _scoring_context(parameters, action="legacy_render")
    scorer = Scorer(context=FeatureContext.load(context_tickers, years=years))

    as_of = resolved_as_of
    horizon_days = int(parameters.get("horizon_days", 35))
    alt_strikes = int(parameters.get("alt_strikes", 1))
    board = pd.DataFrame(score_document.get("rows") or [])
    fill_alpha = float(board["fill"].iloc[0]) if len(board) and "fill" in board else 0.5

    meta, health, flags = _render_meta_and_health(
        root, scores=scores, scorer=scorer, as_of=as_of, requested_as_of=requested_as_of,
        resolved_as_of=resolved_as_of, finality=finality, horizon_days=horizon_days,
        alt_strikes=alt_strikes, fill_alpha=fill_alpha, tickers=tickers, evidence=evidence)
    output = root / "bundle"
    result = render_bundle(scores, output, as_of=as_of, horizon_days=horizon_days,
                           fill_alpha=fill_alpha, alt_strikes=alt_strikes,
                           panel=scorer.context.panel, trades=scorer.trades,
                           meta=meta, health=health, flags=flags,
                           registry=scorer.registry)
    _write_action(root, "meta.json", meta)
    _write_action(root, "health.json", health)
    with tarfile.open(root / "bundle.tar", "w") as archive:
        archive.add(output, arcname="bundle")
    return _write_action(root, "render.json", {
        "bundle_archive": "bundle.tar", "bundle_content_hash": bundle_content_hash(output),
        "absent_stages": list(ABSENT_STAGES), "result": result}) | {"path": "bundle.tar"}


def _action_selfcheck(parameters, root):
    import tarfile

    from engine.dashboard.selfcheck import DEFAULT_N, scrub_mismatches, selfcheck
    from engine.features import FeatureContext
    from engine.score import Scorer

    context_tickers, years = _scoring_context(parameters, action="legacy_selfcheck")
    scorer = Scorer(context=FeatureContext.load(context_tickers, years=years))
    archive = root / "bundle.tar"
    if archive.is_file() and not (root / "bundle").is_dir():
        with tarfile.open(archive) as stream:
            stream.extractall(root)
    # v1 parity: no plan builder threads a "sample" parameter (grep confirms
    # engine/v2/ops/nightly.py never sets it), so every real selfcheck job
    # fell through to a hard-coded 10 here -- half of legacy nightly.py's own
    # `selfcheck(bundle_dir, scorer=engine)` call, which defaults to the
    # guide's DEFAULT_N=20 (engine/dashboard/selfcheck.py). Match it, so the
    # supervised job checks the same number of rows the in-process nightly
    # would.
    result = selfcheck(root / "bundle", n=int(parameters.get("sample", DEFAULT_N)),
                       scorer=scorer)
    value = result.as_dict() if hasattr(result, "as_dict") else vars(result)
    if not value.get("ok"):
        # The result is discarded below (never written as an artifact), so
        # without this the ONLY record of a failed selfcheck was a bare
        # traceback in private worker.stderr -- no row, no field, no reason.
        # Scrub to row key/field path/reason before it leaves this process:
        # `details` lands in the attempt's `diagnostics/failure_details.json`
        # (worker.py `_write_failure_details`), which other agents and the
        # supervisor read, so it must never carry board/engine values.
        raise fail("VALIDATION_FAILED", "serialized legacy bundle selfcheck failed",
                  details={"n_checked": value.get("n_checked"),
                           "n_board_rows": value.get("n_board_rows"),
                           "snapshot_ok": value.get("snapshot_ok"),
                           "mismatches": scrub_mismatches(value.get("mismatches", []))})
    return _write_action(root, "selfcheck.json", value)


def invoke_nightly_helper(root, helper, *args, **kwargs):
    """Call one explicitly named read/compute helper, never ``run_nightly``."""
    _rooted_import(root)
    from engine.dashboard.nightly import refresh_calendar_data, strike_ladder, validate_refresh
    allowed = {"validate_refresh": validate_refresh,
               "strike_ladder": strike_ladder,
               "refresh_calendar_data": refresh_calendar_data}
    if helper not in allowed:
        raise fail("INVALID_REQUEST", "legacy nightly helper is not audited",
                   details={"helper": helper})
    return allowed[helper](*args, **kwargs)


def invoke_evaluate(root, spec, trades, *, run_dir, **kwargs):
    """Use the repository evaluator for a genuine experiment report."""
    _rooted_import(root)
    from engine.evaluate import evaluate
    return evaluate(spec, trades, run_dir=run_dir, **kwargs)


REGISTERED_RUNNERS = frozenset({
    "experiments/EXP-182_d_1_gated_execution_parity_registered/run.py",
})


def run_legacy_script(root, script, args=()):
    """Run a registered legacy runner in a private root with smoke protection."""
    _rooted_import(root)
    base = Path(root).resolve()
    relative = str(Path(script))
    script_path = (base / relative).resolve()
    if relative not in REGISTERED_RUNNERS or not script_path.is_relative_to(base):
        raise fail("INVALID_REQUEST", "legacy experiment runner is unaudited")
    if tuple(args):
        raise fail("INVALID_REQUEST", "legacy runner may not enable ledger writes")
    import subprocess
    command = [sys.executable, "-u", str(script_path), "--no-ledger"]
    return subprocess.run(command, cwd=base, check=False,
                          capture_output=True, text=True, timeout=3600)


# --------------------------------------------------------------------------
# P2-5/Task5: the effects-graph coordinator's own audited subprocess edges.
#
# ``checks/import_layers.py``'s ``check_runtime_edges`` allows exactly two
# ``engine/v2/ops`` modules to call ``subprocess`` at all: this one and
# ``executor.py``. ``engine.v2.ops.effects_graph`` needs three isolated
# subprocess calls of its own — none of them the frozen legacy tree, but all
# of them process boundaries a coordinator must not cross in its own
# long-lived process (a fresh ``INVESTING_PLAN_ROOT``-scoped compatibility
# read, and two crossings into ``checks/*``, which production code may never
# import directly). They live here, audited, rather than adding a third
# process owner.
# --------------------------------------------------------------------------

_VERIFY_GENERATION_SCRIPT = (
    "import json\n"
    "from engine import ledger\n"
    "print(json.dumps({'predictions': len(ledger.read_predictions(resolve_supersedes=False)),\n"
    "                   'outcomes': len(ledger.read_outcomes())}))\n"
)


def verify_export_generation(generation_dir, repo_root, *, timeout=120):
    """Read one export generation back through the real compatibility reader.

    ``engine.paths.ROOT`` is fixed at first import from ``INVESTING_PLAN_ROOT``,
    so this always runs in a fresh subprocess, never the caller's own
    long-lived process. ``generation_dir`` must directly contain
    ``predictions/`` and/or ``outcomes/`` (an export generation's own shape);
    a private symlinked root makes that true without copying any byte.
    """
    import os
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="ledger-export-verify-") as scratch:
        link_root = Path(scratch) / "root"
        link_root.mkdir()
        os.symlink(Path(generation_dir).resolve(), link_root / "ledger")
        env = dict(os.environ, INVESTING_PLAN_ROOT=str(link_root))
        result = subprocess.run([sys.executable, "-c", _VERIFY_GENERATION_SCRIPT],
                                cwd=str(repo_root), env=env, capture_output=True, text=True,
                                timeout=timeout)
    return _json_stdout(result, "export generation failed the compatibility read-back")


def run_legacy_rebuild(candidate_root, repo_root, *, tables=None, sample=None, timeout=3600):
    """Run the real legacy rebuild, rooted at a private candidate directory
    (phase-2 guide §10): a fresh subprocess with ``INVESTING_PLAN_ROOT``
    pointed at ``candidate_root``, so every write lands there alone.
    """
    import subprocess

    command = [sys.executable, "-m", "engine.data.rebuild"]
    for table in tables or ():
        command += ["--table", table]
    if sample is not None:
        command += ["--sample", str(sample)]
    env = dict(os.environ, INVESTING_PLAN_ROOT=str(candidate_root))
    result = subprocess.run(command, cwd=str(repo_root), env=env, capture_output=True,
                            text=True, timeout=timeout)
    if result.returncode != 0:
        raise fail("VALIDATION_FAILED", "legacy rebuild candidate subprocess did not complete",
                   details={"stderr": result.stderr[-2000:]})
    return {"schema_version": "legacy_rebuild_report.v1.0", "returncode": result.returncode}


def run_engineering_gate(repo_root, *, timeout=600):
    """Run the Phase 1 structural/engineering gate over ``repo_root``.

    ``checks/*`` is verification tooling, never importable from
    ``engine/v2/**`` — this is a subprocess boundary, not a Python import.
    """
    import subprocess

    script = Path(repo_root) / "checks" / "rearchitecture_phase1_gate.py"
    result = subprocess.run([sys.executable, str(script)], cwd=str(repo_root),
                            capture_output=True, text=True, timeout=timeout)
    return _json_stdout(result, "engineering gate produced no JSON")


_SECURITY_SCAN_SCRIPT = (
    "import json, sys, tarfile\n"
    "from pathlib import Path\n"
    "from checks.repo_hygiene import check_bundle, load_secrets\n"
    "from engine.dashboard.render import RENDERED_DATA_STEMS\n"
    "bundle, env_root = Path(sys.argv[1]), Path(sys.argv[2])\n"
    "files = {}\n"
    "with tarfile.open(bundle) as archive:\n"
    "    for member in archive.getmembers():\n"
    "        if member.isfile():\n"
    "            files[member.name] = archive.extractfile(member).read()\n"
    "declared = frozenset(\n"
    "    f'bundle/data/{stem}.{ext}' for stem in RENDERED_DATA_STEMS for ext in ('json', 'js')\n"
    ")\n"
    "needles = load_secrets(env_root / '.env')\n"
    "report = check_bundle(files, needles, declared=declared)\n"
    "print(json.dumps({'ok': report.ok, 'checked': report.checked,\n"
    "                   'secrets_loaded': report.secrets_loaded,\n"
    "                   'violations': [[v.path, v.rule, v.detail] for v in report.violations]}))\n"
)


def run_security_scan(bundle_path, repo_root, env_root=None, *, timeout=120):
    """Secret-scan a release bundle tar with ``checks.repo_hygiene``, isolated.

    ``repo_root`` is the CODE checkout the subprocess runs from (it must have
    ``checks/`` and ``engine/`` importable from its cwd) -- it is never where
    secrets are read from: a snapshot-backed/frozen worktree run from here
    carries no ``.env`` at all. ``env_root`` is the store/source checkout
    that holds the real ``.env`` (``Service.store_root`` / the ops CLI's
    ``--store-root``, "the legacy checkout"), passed explicitly. It defaults
    to ``repo_root`` only for a caller that genuinely has one combined
    checkout (tests, a single-directory dev setup); production always passes
    the two apart. ``check_bundle`` itself refuses (0 needles is a
    violation) rather than silently scanning with an empty needle set, so an
    ``env_root`` that turns out to have no ``.env`` fails the gate instead of
    passing it open.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", _SECURITY_SCAN_SCRIPT, str(bundle_path),
         str(env_root if env_root is not None else repo_root)],
        cwd=str(repo_root), capture_output=True, text=True, timeout=timeout)
    return _json_stdout(result, "security scan produced no JSON")


def _json_stdout(result, message):
    try:
        return json.loads(result.stdout)
    except ValueError:
        raise fail("VALIDATION_FAILED", message,
                   details={"stderr": result.stderr[-2000:]}) from None
