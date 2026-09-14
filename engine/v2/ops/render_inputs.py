"""Pure logic for the supervised ``legacy_render`` action (P2-5, D19).

Nothing here imports a legacy (``engine.*``) symbol — that would open a
second adapter module inside the ``engine.v2.ops`` package, which
``checks/import_layers.py`` §4.2 forbids (one adapter module per package).
Legacy calls (``render_bundle``, ``build_meta``, ``build_health``, ...) stay
in ``engine.v2.ops.legacy_adapter``; this module only assembles and compares
their inputs and outputs.
"""
from __future__ import annotations

import json
import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath

import pandas as pd

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail

__all__ = [
    "ABSENT_STAGES",
    "EXECUTION_METADATA_FIELDS",
    "RENDER_FLAG_SOURCES_V1",
    "absent_stage_flags",
    "assemble_scores",
    "bundle_content_hash",
    "diff_bundles",
    "execution_clock_and_flags",
    "model_evidence_stale_flag",
    "normalized_bundle_entries",
    "render_flags",
    "resolve_prior_selfcheck",
    "stage_ledger_generation",
    "stage_model_evidence",
    "unknown_operational_flags",
    "unknown_selfcheck_report",
]

#: guides/rearchitecture_phase2_data_access.md §9.4: "Stages not carried:
#: refresh, the validate-refresh battery, the Tier 3/4 rebuild, missed-night
#: backfill, and calibration flags stay outside the shadow graph. Every
#: shadow receipt lists them as absent so a shadow board is never mistaken
#: for a production one."
ABSENT_STAGES = ("refresh", "validate_refresh", "tier34_rebuild", "backfill",
                  "calibration_flags")

#: Top-level keys of ``meta.json`` / ``health.json`` that carry wall-clock or
#: host state rather than board content: ``generated_at`` stamps the instant
#: ``build_meta``/``build_health`` ran; ``freshness`` embeds per-source ages
#: and the last network-call timestamp off disk; ``quota`` embeds the quota
#: ledger's own ``ts``; ``cron`` names the host's schedule entry (legacy
#: nightly interpolates ``paths.ROOT.name`` into it). None of the four is
#: reproducible from (scores, ladder, model evidence, ledger, finality)
#: alone, so D19 parity excludes them rather than freezing a clock.
EXECUTION_METADATA_FIELDS = {
    "meta": ("generated_at", "freshness", "quota", "cron"),
    "health": ("generated_at",),
}


def absent_stage_flags() -> list[dict]:
    """One board flag per :data:`ABSENT_STAGES` entry — the shadow disclosure."""
    return [{"kind": "shadow_stage_absent", "stage": stage,
             "detail": f"{stage} does not run in the supervised shadow render"}
            for stage in ABSENT_STAGES]


def execution_clock_and_flags(requested_as_of, resolved_as_of, finality: dict) -> tuple[dict, list]:
    """P2-C03: v1's ``execution_clock`` meta plus the render flag list.

    ``requested_as_of``/``resolved_as_of`` mirror
    ``engine/dashboard/nightly.py:1592-1594``'s ``meta["execution_clock"]``
    exactly; the walk-back flag (same ``as_of_resolved`` kind/detail as
    ``engine/dashboard/nightly.py:1244``) is appended to
    :func:`absent_stage_flags` only when the two dates differ.
    """
    from engine.v2.ops.session_resolution import walk_back_flag

    clock = {"requested_as_of": str(pd.Timestamp(requested_as_of).date()),
            "resolved_as_of": str(pd.Timestamp(resolved_as_of).date()), "finality": finality}
    flags = absent_stage_flags()
    flag = walk_back_flag(requested_as_of, resolved_as_of, finality)
    if flag is not None:
        flags.append(flag)
    return clock, flags


#: P2-C08 (guide §9.4 item 2, review "the review's requirement"): every v1
#: ``report.flags`` kind found in ``engine/dashboard/nightly.py``, classified:
#:
#: * ``"a"`` -- derivable from artifacts this render job binds or can read
#:   (score, finality, model evidence, ledger generation, the pinned
#:   earnings/panel store) -- carried at v1's own kind/detail via a direct
#:   call to the legacy helper that built it in v1, never a re-derivation.
#: * ``"b"`` -- only ever produced by a stage v2 deliberately does not run in
#:   the shadow render (refresh/validate_refresh/tier34_rebuild/backfill/
#:   calibration_flags, already declared by :data:`ABSENT_STAGES`), or by a
#:   stage that runs as a hard pass/fail v2 job rather than a soft degrade
#:   (settlement, selfcheck, publication, backup) -- never a render input in
#:   either version.
#: * ``"c"`` -- sourced from mutable host state or a previous run's persisted
#:   board that this render job does not own -- rendered as an explicit
#:   unknown (:func:`unknown_operational_flags`), never silently omitted and
#:   never implied healthy.
RENDER_FLAG_SOURCES_V1 = (
    {"kind": "as_of_resolved", "v1_ref": "engine/dashboard/nightly.py:1239-1244",
     "class": "a", "v2_source": "session_resolution.walk_back_flag, called from "
     "execution_clock_and_flags on the bound finality artifact"},
    {"kind": "panel_stale", "v1_ref": "engine/dashboard/nightly.py:756-767",
     "class": "a", "v2_source": "legacy_adapter._panel_lag_flags calls "
     "engine.dashboard.nightly._panel_staleness_flags directly on the "
     "resolved session; no logic re-derived"},
    {"kind": "panel_missing_prints", "v1_ref": "engine/dashboard/nightly.py:769-780",
     "class": "a", "v2_source": "same call as panel_stale -- one legacy "
     "helper returns both kinds"},
    {"kind": "panel_coverage_unknown", "v1_ref": "engine/dashboard/nightly.py:743-745",
     "class": "a", "v2_source": "same call as panel_stale"},
    {"kind": "calendar_date_conflict", "v1_ref": "engine/dashboard/nightly.py:1027-1046,1532-1534",
     "class": "a", "v2_source": "legacy_adapter._calendar_conflict_flags reads "
     "earnings_events for this render's own tickers/window and calls "
     "engine.dashboard.nightly._date_conflict_flag directly"},
    {"kind": "model_evidence_stale", "v1_ref": "engine/dashboard/nightly.py:1552-1575",
     "class": "a", "v2_source": "model_evidence_stale_flag reads the "
     "'degraded'/'degraded_reason' fields legacy_adapter._action_model_evidence "
     "now preserves on a failed rebuild, in place of catching a live "
     "exception in this process"},
    {"kind": "no_upcoming_events", "v1_ref": "engine/dashboard/nightly.py:1178-1180",
     "class": "b", "v2_source": "absent_stage_flags['refresh']"},
    {"kind": "refresh_degraded", "v1_ref": "engine/dashboard/nightly.py:1210-1212",
     "class": "b", "v2_source": "absent_stage_flags['refresh']"},
    {"kind": "session_not_final", "v1_ref": "engine/dashboard/nightly.py:1223-1232",
     "class": "b", "v2_source": "superseded, not carried: legacy_finality "
     "refuses with SOURCE_NOT_FINAL instead (P2-C03); render never runs on a "
     "session that failed finality"},
    {"kind": "validation_red", "v1_ref": "engine/dashboard/nightly.py:1258-1264",
     "class": "b", "v2_source": "absent_stage_flags['validate_refresh']"},
    {"kind": "moves_degraded", "v1_ref": "engine/dashboard/nightly.py:1324-1330",
     "class": "b", "v2_source": "absent_stage_flags['tier34_rebuild']"},
    {"kind": "tiers_degraded", "v1_ref": "engine/dashboard/nightly.py:1340-1349",
     "class": "b", "v2_source": "absent_stage_flags['tier34_rebuild']"},
    {"kind": "backfill_skipped_closed", "v1_ref": "engine/dashboard/nightly.py:1502-1507",
     "class": "b", "v2_source": "absent_stage_flags['backfill']"},
    {"kind": "backfill_gap", "v1_ref": "engine/dashboard/nightly.py:1508-1512",
     "class": "b", "v2_source": "absent_stage_flags['backfill']"},
    {"kind": "late_backfill", "v1_ref": "engine/dashboard/nightly.py:1513-1518",
     "class": "b", "v2_source": "absent_stage_flags['backfill']"},
    {"kind": "calibration_drift", "v1_ref": "engine/dashboard/nightly.py:979-998,1541",
     "class": "b", "v2_source": "absent_stage_flags['calibration_flags']"},
    {"kind": "settle_failed", "v1_ref": "engine/dashboard/nightly.py:1443-1452",
     "class": "b", "v2_source": "settlement runs as its own v2 job "
     "(legacy_settlement); a failure fails that job rather than degrading "
     "into a soft render flag, so render carries no equivalent"},
    {"kind": "selfcheck_red", "v1_ref": "engine/dashboard/nightly.py:1631-1632",
     "class": "b", "v2_source": "selfcheck runs as its own v2 job "
     "(legacy_selfcheck), strictly after render, and refuses "
     "(VALIDATION_FAILED) on a mismatch rather than degrading; not a render "
     "input in either version"},
    {"kind": "publish_failed", "v1_ref": "engine/dashboard/nightly.py:1656",
     "class": "b", "v2_source": "publication runs as its own stage after "
     "render (P2-5/Task5); not a render input"},
    {"kind": "backup_failed", "v1_ref": "engine/dashboard/nightly.py:1698-1716",
     "class": "b", "v2_source": "backup runs as its own optional stage after "
     "render; not a render input"},
    {"kind": "earnings_date_changed", "v1_ref": "engine/dashboard/nightly.py:1004-1023,1365-1368",
     "class": "c", "v2_source": "needs a previous nightly's persisted "
     "'calendar' state, which this render job does not bind; folded into "
     "the 'prior_run_state_unknown' explicit-unknown flag"},
    {"kind": "new_gate_triggers", "v1_ref": "engine/dashboard/nightly.py:1521-1530",
     "class": "c", "v2_source": "needs a previous nightly's persisted "
     "'gate_triggers' state; folded into 'prior_run_state_unknown'"},
    {"kind": "quota_below_reserve", "v1_ref": "engine/dashboard/nightly.py:963-975,1536-1538",
     "class": "c", "v2_source": "reads the live ORATS quota ledger, mutable "
     "host state the refresh stage owns; rendered as the explicit "
     "'quota_unknown' flag, never omitted and never implied healthy"},
)


def model_evidence_stale_flag(evidence: dict) -> dict | None:
    """Class (a): v1's ``model_evidence_stale`` kind/detail
    (``engine/dashboard/nightly.py:1568-1574``), derived from the bound
    model-evidence artifact's own ``degraded``/``degraded_reason`` fields
    (``legacy_adapter._action_model_evidence`` sets them on a failed
    rebuild) rather than a live exception in this process -- the shadow
    render's model-evidence rebuild and its render run in separate jobs.
    """
    if not isinstance(evidence, dict) or not evidence.get("degraded"):
        return None
    reason = evidence.get("degraded_reason", "")
    detail = ("evidence rebuild failed; the bundle carries the cached table, "
             f"which may predate the current champions — {reason}")[:300]
    return {"kind": "model_evidence_stale", "detail": detail}


def unknown_operational_flags() -> list[dict]:
    """Class (c): v1 flags sourced from mutable host state or a previous
    run's persisted board that this render job does not own. Never silently
    dropped -- each renders as an explicit unknown, distinct from both "flag
    absent" (nothing wrong) and a declared absent stage.
    """
    return [
        {"kind": "quota_unknown", "detail": "ORATS quota is tracked by the "
         "refresh stage (an absent stage here); headroom cannot be "
         "evaluated in this render"},
        {"kind": "freshness_unknown", "detail": "source freshness is "
         "measured against the refresh clock (an absent stage here)"},
        {"kind": "prior_run_state_unknown", "detail": "no prior nightly "
         "state is bound to this render; earnings-date-change and "
         "new-gate-trigger detection need last night's persisted board"},
    ]


def unknown_selfcheck_report() -> dict:
    """Class (c): no prior selfcheck bound to this render. ``build_health``
    would otherwise serialize a bare ``None`` into ``last_selfcheck`` --
    indistinguishable from "never checked, assume fine" -- so this is passed
    instead: an explicit unknown state ``build_health`` cannot mistake for
    healthy.
    """
    return {"ok": None, "known": False,
            "detail": "no prior selfcheck bound to this render"}


def resolve_prior_selfcheck(root: Path) -> dict:
    """The bound ``prior_selfcheck.json`` (a previous run's committed
    selfcheck artifact) when the plan supplied one, else the explicit
    unknown state -- never a bare ``None``.
    """
    path = Path(root) / "prior_selfcheck.json"
    if not path.is_file():
        return unknown_selfcheck_report()
    return json.loads(path.read_text())


def render_flags(*, requested_as_of, resolved_as_of, finality, panel_lag,
                 calendar_conflict, model_evidence) -> tuple[dict, list]:
    """The full v2 render flag list (:data:`RENDER_FLAG_SOURCES_V1`): the
    execution clock, the absent-stage disclosures and walk-back flag, every
    class (a) flag derived from a bound artifact, and the class (c) explicit
    unknowns. ``panel_lag`` and ``calendar_conflict`` are precomputed by
    ``legacy_adapter`` (both call a legacy helper this module may not
    import, per this module's own docstring); ``model_evidence`` is the
    already-loaded model-evidence document.
    """
    clock, flags = execution_clock_and_flags(requested_as_of, resolved_as_of, finality)
    flags.extend(panel_lag)
    flags.extend(calendar_conflict)
    stale = model_evidence_stale_flag(model_evidence)
    if stale is not None:
        flags.append(stale)
    flags.extend(unknown_operational_flags())
    return clock, flags


def assemble_scores(score_document: dict) -> pd.DataFrame:
    """Reproduce the legacy nightly's board+ladder frame from ``score.json``.

    ``engine/dashboard/nightly.py`` (quoted, its own line numbers)::

        1412  scores = score_calendar(
        1413      as_of, horizon_days=horizon_days, alt_strikes=0,
        1414      scorer=engine, tickers=tickers,
        1415      progress_every=10,
        1416  )
        1422  board_scores = scores
        ...
        1456  ladder = strike_ladder(
        1457      board_scores, scorer=engine, alt_strikes=alt_strikes, as_of=as_of
        1458  )
        1459  if ladder:
        1460      scores = pd.concat([scores, pd.DataFrame(ladder)], ignore_index=True)

    ``_action_score`` already ran ``score_calendar`` and ``strike_ladder`` and
    wrote both frames into the score artifact (``rows`` is ``board_scores``,
    ``ladder`` is the strike-ladder rows) — so this performs the *same*
    ``pd.concat([scores, pd.DataFrame(ladder)], ignore_index=True)`` over
    those two frames rather than re-deriving either one.
    """
    board = pd.DataFrame(score_document.get("rows") or [])
    ladder_rows = score_document.get("ladder") or []
    if not ladder_rows:
        return board
    return pd.concat([board, pd.DataFrame(ladder_rows)], ignore_index=True)


def stage_model_evidence(evidence_path: Path, legacy_root: Path) -> Path:
    """Place the bound model-evidence artifact at its legacy on-disk path.

    ``render_bundle`` reads it back through
    ``engine.dashboard.model_evidence.load_model_evidence()``, which is
    ``paths.FEATURES / "model_evidence.json"`` — i.e.
    ``<legacy_root>/data/features/model_evidence.json`` once
    ``INVESTING_PLAN_ROOT`` is ``legacy_root``.
    """
    destination = Path(legacy_root) / "data" / "features" / "model_evidence.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    shutil.copyfile(Path(evidence_path), destination)
    return destination


def stage_ledger_generation(tar_path: Path, legacy_root: Path) -> Path:
    """Replace ``<legacy_root>/ledger`` with exactly one export generation.

    Any staged copy of the *mutable* ledger (e.g. a stray general read-set
    copy) is removed first — the book view and the live size-model MAE must
    read the bound, immutable generation, never a mutable ledger file that
    happened to be sitting in staging.
    """
    ledger_dir = Path(legacy_root) / "ledger"
    if ledger_dir.exists() or ledger_dir.is_symlink():
        shutil.rmtree(ledger_dir)
    ledger_dir.mkdir(parents=True)
    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise fail("INTEGRITY_FAILED",
                           "ledger generation archive member is unsafe",
                           details={"member": member.name})
            if not member.isfile() and not member.isdir():
                raise fail("INTEGRITY_FAILED",
                           "ledger generation archive member is not a plain file",
                           details={"member": member.name})
            if name.parts and name.parts[0] not in ("predictions", "outcomes"):
                raise fail("INTEGRITY_FAILED",
                           "ledger generation archive has an unexpected member",
                           details={"member": member.name})
        archive.extractall(ledger_dir)
    return ledger_dir


_JS_ASSIGNMENT = re.compile(r"^window\.[A-Za-z0-9_]+ = ")


def _strip_metadata(stem: str, payload):
    fields = EXECUTION_METADATA_FIELDS.get(stem)
    if not fields or not isinstance(payload, dict):
        return payload
    return {key: value for key, value in payload.items() if key not in fields}


def _normalized_js(stem: str, text: str):
    match = _JS_ASSIGNMENT.match(text)
    if not match:
        return text
    body = text[match.end():]
    if body.endswith(";\n"):
        body = body[:-2]
    try:
        payload = json.loads(body)
    except ValueError:
        return text
    stripped = _strip_metadata(stem, payload)
    return match.group(0) + json.dumps(stripped, sort_keys=True, default=str) + ";\n"


def normalized_bundle_entries(bundle_dir: Path) -> dict:
    """Every file in a rendered bundle, keyed by relative path.

    A declared metadata file (``meta.json``/``meta.js``, ``health.json``/
    ``health.js``) is parsed and stripped of :data:`EXECUTION_METADATA_FIELDS`
    before it enters the map; every other file is a hash of its raw bytes.
    Two bundles compare equal under this map exactly when D19 requires:
    identical everywhere except declared execution-metadata fields.
    """
    bundle_dir = Path(bundle_dir)
    entries: dict[str, object] = {}
    for path in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(bundle_dir).as_posix()
        stem = path.stem
        if path.suffix == ".json" and stem in EXECUTION_METADATA_FIELDS:
            entries[relative] = _strip_metadata(stem, json.loads(path.read_text()))
        elif path.suffix == ".js" and stem in EXECUTION_METADATA_FIELDS:
            entries[relative] = _normalized_js(stem, path.read_text())
        else:
            import hashlib

            entries[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return entries


def bundle_content_hash(bundle_dir: Path) -> str:
    """The D19 identity hash: every bundle file, metadata fields removed."""
    return content_hash(normalized_bundle_entries(bundle_dir))


def diff_bundles(dir_a: Path, dir_b: Path) -> list[str]:
    """Name every file (and, for meta/health, every field) that differs."""
    left = normalized_bundle_entries(dir_a)
    right = normalized_bundle_entries(dir_b)
    diffs = []
    for name in sorted(set(left) | set(right)):
        a, b = left.get(name), right.get(name)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            changed = sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))
            diffs.append(f"{name}: {', '.join(changed) or 'missing'}")
        else:
            diffs.append(name)
    return diffs
