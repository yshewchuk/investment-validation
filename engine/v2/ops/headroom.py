"""Nightly limit headroom check that reads no data rows and writes no data
or artifacts, but -- like every ops command -- opens the ops catalog, which
may run the outcome-session backfill: usage against fixed repo limits.

One ``headroom_report`` pass answers "how close is tonight's plan to a hard
limit" from catalog metadata only -- no object payloads beyond the plan and
its bound legacy manifest, no data-row reads, so it fits the nightly window.
Statuses: ``OK`` at or below 80%, ``WARN`` above 80%, ``FAIL`` above 95%; a
limit the check could not measure is ``UNMEASURED`` rather than ``OK`` and
drives exit 1 so a skipped check is never mistaken for a clean one. The
document's overall status is the worst limit's and the CLI exits 0/1/2 to
match (``UNMEASURED`` ranks with ``WARN``). Anything else the check does not
measure appears in ``not_covered``.

The scratch check reuses the same helpers as ``nightly.plan_scratch_problems``
-- the ``store_barrier.domains_of`` read-domain test, the ``_legacy_resource``
profile routing and the manifest ``file_refs`` byte total -- but reports the
measured usage per profile instead of only refusing an oversize plan, so it is
not a mirror of the refusal logic. No covered limit has growth history yet, so
every entry keeps a ``projected_days_to_limit`` of ``None`` to hold the
``headroom.v1`` shape stable.
"""
from __future__ import annotations

import json
import math
import sys
import time

from engine.v2.foundation import ArtifactError, ArtifactStore
from engine.v2.ops import nightly
from engine.v2.ops.checkpoints import registered_artifact
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.stages import registry
from engine.v2.ops.store_barrier import domains_of

#: The three-tier cut points; 80.0 is OK, 95.0 is WARN (strictly above fails).
_WARN_PCT = 80.0
_FAIL_PCT = 95.0
_EXIT = {"OK": 0, "UNMEASURED": 1, "WARN": 1, "FAIL": 2}
#: The worst-status order for the document; UNMEASURED counts as exit 1
#: (like WARN) but ranks below a measured WARN so a real signal shows.
_RANK = {"OK": 0, "UNMEASURED": 1, "WARN": 2, "FAIL": 3}

#: How many of the most recent finished attempts the memory limits scan.
_RECENT_ATTEMPTS = 500
_GIB = 1024 ** 3
#: Bounded-read cap for JSON artifact loads (repo convention, cf.
#: d14_inputs.MAX_CALENDAR_BYTES): refuse before verified read of an oversize blob.
_MAX_JSON_BYTES = 16 * 1024 * 1024
#: Bounded cap for the stored ArtifactRef metadata ``registered_artifact``
#: decodes (cf. native_sweep_collection's MAX_NATIVE_EVIDENCE_BYTES pass):
#: an oversize or malformed ref row is an unreadable artifact, refused
#: INVALID_REQUEST, not a payload-size signal.
_MAX_METADATA_BYTES = 1024 * 1024

_NOT_COVERED = (
    "max-registration-bytes (run-script flag, not in repo)",
    "chains._RESULT_CAP (2_000_000)",
    "legacy_bundle.MAX_ARTIFACT_BYTES",
    "release_layout.MAX_CATALOG_BYTES",
    "deployment.MAX_MANIFEST_BYTES",
    "serving_fold MAX_FOLD_BYTES",
    "d14_inputs.MAX_CALENDAR_BYTES",
    "serving.projections._LOOKUP_RESULT_CAP",
    "models.inventory MAX_ARTIFACT_BYTES/MAX_REGISTRY_BYTES",
    "iv_crush_inputs.MAX_TARGET_OBJECT_BYTES",
    "event_history.MAX_OUTCOME_OBJECT_BYTES",
)


def status_for(pct: float) -> str:
    if pct > _FAIL_PCT:
        return "FAIL"
    if pct > _WARN_PCT:
        return "WARN"
    return "OK"


def _entry(name, usage, limit, unit, detail):
    # status_for runs on the unrounded ratio: 80.04% displayed as 80.0 is
    # still WARN, and a rounding-to-boundary must never soften a tier.
    ratio = 100.0 * usage / limit if limit > 0 else 0.0
    return {"name": name, "usage": int(usage), "limit": int(limit), "unit": unit,
            "pct": round(ratio, 1), "status": status_for(ratio),
            "projected_days_to_limit": None, "detail": detail}


def _unmeasured(name, limit, unit, detail):
    """A limit the pass could not measure: no usage, no pct, never OK.

    It ranks with ``WARN`` (exit 1) so a skipped check is not read as clean;
    the ``detail`` names why nothing was measured.
    """
    return {"name": name, "usage": 0, "limit": int(limit), "unit": unit,
            "pct": None, "status": "UNMEASURED",
            "projected_days_to_limit": None, "detail": detail}


def _scratch_entry(profile, needed, detail):
    return _entry(f"scratch_bytes:{profile.name}", needed, profile.scratch_bytes,
                  "bytes", detail)


def _json_object(conn, store, artifact_id, label):
    """Load a registered artifact as a JSON object, or refuse.

    A bad ``--plan`` id (unregistered) or a payload that is not a JSON object
    is an operator input error on this write-free command, so it becomes an
    ``INVALID_REQUEST`` refusal rather than a traceback. But a registered
    artifact whose bytes fail content-address verification
    (``ArtifactError`` from ``read_verified``) is corruption, not input error,
    so it becomes ``INTEGRITY_FAILED``. The ref row is read with bounded
    metadata (``max_metadata_bytes=_MAX_METADATA_BYTES``, cf.
    native_sweep_collection): oversize or malformed reference metadata is
    refused as unreadable, like an unknown id; only the payload cap keeps
    ``RESOURCE_LIMIT_EXCEEDED``, and both size checks run on catalog metadata
    before any bytes are hashed.
    """
    try:
        ref = registered_artifact(conn, artifact_id,
                                  max_metadata_bytes=_MAX_METADATA_BYTES)
    except OpsError:
        raise fail("INVALID_REQUEST", f"the {label} artifact is not readable") from None
    if ref.byte_size > _MAX_JSON_BYTES:
        raise fail("RESOURCE_LIMIT_EXCEEDED",
                   f"the {label} artifact exceeds {_MAX_JSON_BYTES} bytes")
    try:
        raw = store.read_verified(ref)
    except ArtifactError:
        raise fail("INTEGRITY_FAILED",
                   f"the {label} artifact bytes fail verification") from None
    try:
        document = json.loads(raw)
    except ValueError:
        raise fail("INVALID_REQUEST", f"the {label} artifact is not valid JSON") from None
    if not isinstance(document, dict):
        raise fail("INVALID_REQUEST", f"the {label} must be a JSON object")
    return document


def _manifest_total(conn, store, manifest_id, totals):
    if manifest_id not in totals:
        manifest = _json_object(conn, store, manifest_id, "manifest")
        file_refs = manifest.get("file_refs")
        if not isinstance(file_refs, list):
            raise fail("INVALID_REQUEST", "the manifest artifact has no file_refs list")
        sizes = []
        for item in file_refs:
            size = item.get("byte_size") if isinstance(item, dict) else None
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise fail("INVALID_REQUEST",
                           "the manifest file_refs need non-negative integer byte_size")
            sizes.append(size)
        totals[manifest_id] = sum(sizes)
    return totals[manifest_id]


def _snapshot_estimate(snapshot):
    """The validated pinned ``scratch_estimate_bytes``, or an INVALID_REQUEST refusal."""
    estimate = (snapshot or {}).get("scratch_estimate_bytes")
    if isinstance(estimate, bool) or not isinstance(estimate, int) or estimate < 0:
        raise fail("INVALID_REQUEST",
                   "snapshot scratch_estimate_bytes must be a non-negative integer")
    return estimate


def _scratch_entries(conn, store, plan):
    """One ``scratch_bytes:<profile>`` entry per profile the plan stages against.

    Mirrors ``nightly.plan_scratch_problems``: ``legacy_materialize`` is
    checked against its pinned ``scratch_estimate_bytes``; every kind with a
    declared legacy-store read domain is checked against its bound
    ``legacy_manifest.json`` (the plan's ``input_manifest_ref``) byte total;
    snapshot-mode stages that read the materialized root declare no read
    domain and are skipped, exactly as ``domains_of`` decides at claim time.
    """
    if plan.get("kind") != "nightly":
        return [_unmeasured("scratch_bytes", 0, "bytes",
                            "not a nightly plan; no scratch budget to measure")]
    snapshot = (plan.get("snapshot_inputs") or {}
                if plan.get("input_mode") == "snapshot" else None)
    refresh_mode = plan.get("refresh_mode", "legacy")
    reg = registry()
    manifest_id = plan.get("input_manifest_ref")
    totals: dict[str, int] = {}
    seen: set[str] = set()
    entries = []
    for stage in nightly._stage_sequence(plan, False, snapshot, refresh_mode):
        kind = nightly._action_for(stage, refresh_mode=refresh_mode)
        if kind == "legacy_materialize":
            needed = _snapshot_estimate(snapshot)
            detail = "pinned snapshot scratch_estimate_bytes (legacy_materialize)"
        elif snapshot is not None and stage in nightly.SNAPSHOT_STAGES:
            continue
        elif manifest_id and any(mode == "read" for _, mode in domains_of(reg, kind)):
            needed = _manifest_total(conn, store, manifest_id, totals)
            detail = f"legacy read-set manifest total (stage {stage})"
        else:
            continue
        profile = profile_named(DEFAULT_POLICY, nightly._legacy_resource(kind))
        if profile.name in seen:
            continue
        seen.add(profile.name)
        entries.append(_scratch_entry(profile, needed, detail))
    if not entries:
        entries = [_unmeasured("scratch_bytes", 0, "bytes",
                               "plan carries neither a legacy input manifest nor a "
                               "snapshot scratch estimate; nothing to measure")]
    return entries


def _memory_entries(conn, max_rss_gb):
    """``memory_peak:<profile>`` entries from the most recent finished attempts.

    The enforced memory limit and the profile name live on the attempt's
    ``resource_reservations`` row (``scheduler._create_attempt`` writes both
    beside ``resources_json``); rows with a NULL peak are skipped, as are
    attempts outside the recent window. With no recorded peak at all the
    limits are ``UNMEASURED``, never an OK zero: a catalog that has never
    run an attempt is not a catalog with idle memory.
    """
    entries = []
    rows = conn.execute(
        "SELECT rr.profile AS profile, rr.memory_bytes AS memory_bytes, "
        "MAX(a.memory_peak_bytes) AS peak FROM attempts a "
        "JOIN resource_reservations rr ON rr.attempt_id = a.attempt_id "
        "WHERE a.attempt_id IN (SELECT attempt_id FROM attempts "
        "WHERE ended_at IS NOT NULL AND memory_peak_bytes IS NOT NULL "
        "ORDER BY COALESCE(ended_at, created_at) DESC LIMIT ?) "
        "GROUP BY rr.profile, rr.memory_bytes ORDER BY peak DESC",
        (_RECENT_ATTEMPTS,)).fetchall()
    if not rows:
        entries.append(_unmeasured("memory_peak", 0, "bytes",
                                   "no finished attempt has a recorded memory peak"))
    for row in rows:
        entries.append(_entry(f"memory_peak:{row['profile']}", row["peak"],
                              row["memory_bytes"], "bytes",
                              "max memory_peak_bytes of recent finished attempts "
                              "in this reservation profile"))
    if max_rss_gb is not None:
        # "most recent session" is the latest supervisor epoch: attempts carry
        # their submitting supervisor's epoch_id, so this bounds the scan to
        # one daemon lifetime.
        peak = conn.execute(
            "SELECT MAX(memory_peak_bytes) AS peak FROM attempts "
            "WHERE memory_peak_bytes IS NOT NULL AND supervisor_epoch = "
            "(SELECT supervisor_epoch FROM attempts "
            "WHERE ended_at IS NOT NULL AND memory_peak_bytes IS NOT NULL "
            "ORDER BY COALESCE(ended_at, created_at) DESC LIMIT 1)").fetchone()["peak"]
        if peak is None:
            entries.append(_unmeasured("rss_cap_gb", int(max_rss_gb * _GIB), "bytes",
                                       "no finished attempt has a recorded memory peak"))
        else:
            entries.append(_entry("rss_cap_gb", int(peak), int(max_rss_gb * _GIB), "bytes",
                                  "largest attempt memory peak of the most recent "
                                  "supervisor epoch vs --max-rss-gb"))
    return entries


def headroom_report(conn, root, plan, *, max_rss_gb=None):
    # Validate the cap here, not only through cli.py's argparse type: library
    # callers bypass the parser, and bool/NaN/inf/0/negatives would otherwise
    # reach the rss arithmetic (True silently meaning "1 GiB").
    if max_rss_gb is not None and (
            not isinstance(max_rss_gb, float)
            or not math.isfinite(max_rss_gb) or max_rss_gb <= 0.0):
        raise fail("INVALID_REQUEST",
                   "max_rss_gb must be a finite number greater than 0")
    if not isinstance(plan, dict):
        raise fail("INVALID_REQUEST", "plan must be a JSON object")
    started = time.monotonic()
    print("[headroom] start", file=sys.stderr)
    store = ArtifactStore(root)
    limits = [*_scratch_entries(conn, store, plan),
              *_memory_entries(conn, max_rss_gb)]
    status = max((item["status"] for item in limits), key=_RANK.__getitem__, default="OK")
    document = {"schema": "headroom.v1", "status": status, "exit_code": _EXIT[status],
                "limits": limits, "not_covered": list(_NOT_COVERED)}
    for item in limits:
        pct = "unmeasured" if item["pct"] is None else f"{item['pct']}%"
        print(f"[headroom] {item['status']} {item['name']}: {item['usage']}/"
              f"{item['limit']} {item['unit']} ({pct})", file=sys.stderr)
    print(f"[headroom] done in {time.monotonic() - started:.1f}s: {status}", file=sys.stderr)
    return document


def headroom_command(args, root, conn, clock):
    """``ops headroom --plan <plan_ref>``: load the plan artifact, report."""
    store = ArtifactStore(root)
    plan = _json_object(conn, store, args.plan, "plan")
    return headroom_report(conn, root, plan, max_rss_gb=args.max_rss_gb)
