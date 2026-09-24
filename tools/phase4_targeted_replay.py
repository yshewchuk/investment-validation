#!/usr/bin/env python3
"""Targeted strict Phase 4 replay diagnostic — read-only, per-row evidence.

A small, read-only diagnostic for iterating on specific Phase 4 sign-off fixes
without rebuilding the frozen Tier-0 capture or paying the full
``checks/phase4_real.py`` gate on every edit. It replays the exact fixtures you
name, through the EXACT repository code the full gate uses
(``checks.phase4_real._verified_trace_bundle`` / ``_replayed_member`` /
``_replayed_chooser`` / ``_record_checks``), and emits value-free per-row
evidence as JSON: the dimension checks, the numeric finding FIELD NAMES, the
key / flag / null-mask differences, and typed incomparability reasons. When a
compared row's ``contracts`` check fails it also reports a compact,
deterministic ``contract_differences`` naming which projected attributes differ
(leg name/right/side/quantity/strike/expiry/price/cash_flow, or an
entry/exit/execution timeline date) with both sides' values — the exact
``_contract_projection`` the checker compares, so the diagnostic can never
drift from the verdict it explains. Those projected leg/timeline terms are the
structural subject of the contracts check, not the numeric parity values the
rest of the evidence keeps value-free; a multi-member DYN-SV row indexes each
difference by member.

**This never claims sign-off.** It runs no acceptance battery, computes no
overall Phase 4 status, writes no final evidence artifact, and modifies nothing
under the corpus. The full ``checks/phase4_real.py`` gate remains the only
thing that may sign Phase 4 off; this tool exists to gather targeted evidence
that a specific fix moved a specific row.

Why the loader is mandatory. A raw pair's ``input_trace`` stores shared frozen
subtrees by reference (``{"$shared": "sha256:..."}``) and translation rows by
reference (``{"$rows": ...}``), so ``content_hash`` over the raw JSON does NOT
reproduce ``trace_hash`` — a hand-decoded pair fails
``input_trace.trace_hash: content hash mismatch`` before it ever scores. This
CLI therefore hydrates through ``checks.tier0_corpus.load`` (the same
hydration/integrity path the gate uses), which resolves every reference back to
the expanded logical value, and verifies each selected row's recomputed payload
digest AND its manifest-declared ``payload_hash``/``request_hash``/
``record_kind``/``covers`` (``case_manifest``'s requirements, per row, through
``compare_records``) and its ``trace_hash`` through that repository code — never
a weaker hand-decoded hash. The repository loader resolves references across the
whole corpus in one pass, so selected-row verification requires a full-corpus
``load()``; the memory cost of that is reported honestly in the output rather
than dodged with a partial parse. ``--output`` is rejected (before any load) if
it resolves inside the corpus — through a symlinked parent or otherwise — so
this read-only tool can never overwrite its own oracle.

Usage::

    python3 tools/phase4_targeted_replay.py \
        --corpus fixtures/tier0 \
        --fixture-id 012_RAMP7-AAPL-2023-11-02_d610d6a5 \
        [--fixture-id OTHER_ID ...] [--output /tmp/targeted.json]

Clean JSON goes to stdout (or ``--output``); the progress/ETA line stream goes
to stderr so stdout stays machine-parseable. The heartbeat labels the whole-
corpus hydration as ``phase=loading`` (``load ETA unknown``, since a memory
watchdog killing a multi-gigabyte load is not a parity finding) separately from
``phase=replay``, whose provisional ETA covers only the replay loop and
accounts for the row currently in flight.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import phase4_real  # noqa: E402
from checks import tier0_corpus as t0  # noqa: E402
from engine.v2.foundation import content_hash  # noqa: E402

__all__ = ["run_targeted", "main", "BASELINE_SECONDS_PER_ROW", "SCHEMA_VERSION"]

SCHEMA_VERSION = "phase4_targeted_replay.v1.0"

#: The prior native-parity run took ~32 minutes for 20 rows. Until this run has
#: observed completed rows of its own, the provisional ETA is extrapolated from
#: this rate; afterwards it is recomputed from observed per-row times.
BASELINE_SECONDS_PER_ROW = (32 * 60.0) / 20.0

#: Dispositions a row can take. Deliberately a subset of the full gate's: this
#: diagnostic compares one row at a time and never folds them into a verdict.
COMPARED = "compared"
EXCLUDED = "excluded"
INCOMPARABLE = "incomparable"

#: The five numeric dimensions ``_record_checks`` compares by field name.
_NUMERIC_DIMENSIONS = (
    "forecasts", "simulation", "financial_diagnostics", "verdicts", "analogs",
)

#: The per-leg attributes ``phase4_real._contract_projection`` normalizes onto
#: every leg, in the order a contract difference is reported (fixed, so the
#: JSON is deterministic regardless of any source dict's key ordering).
_LEG_ATTRIBUTES = (
    "name", "right", "side", "quantity", "strike", "expiry", "price", "cash_flow",
)

#: The whole-structure trade-timeline dates the projection carries alongside the
#: legs, in report order.
_TIMELINE_ATTRIBUTES = ("entry_date", "exit_date", "execution_date")


class _SelectionError(ValueError):
    """The requested selection or corpus is not something this tool can act on
    (unknown fixture id, no corpus, an unsafe output path) — a usage error, not
    a per-row finding."""


def _reject_output_inside_corpus(corpus_root: Path, resolved: Path, output) -> None:
    """Refuse to write ``--output`` anywhere inside the corpus this tool is
    meant to read without touching. The atomic write would ``replace()`` a
    corpus file (e.g. its own ``INDEX.json``), silently mutating the supposedly
    read-only oracle. Two physical locations must be checked against BOTH the
    given root and the ``resolve_corpus`` target:

    * the fully-resolved output path — catches a deep file inside the corpus
      and an output whose PARENT directory is a symlink into the corpus; and
    * the directory the atomic write actually operates in joined with the
      output's own name — because ``Path.replace()`` swaps the entry inside
      ``output.parent``, so an ``output`` that is itself a symlink INTO the
      corpus (however far outside its target resolves) would still clobber the
      corpus entry and leave a temp file beside it. ``resolve()`` on the whole
      path follows that final symlink and would otherwise hide it.
    """
    if output is None:
        return
    out = Path(output)
    roots = {Path(corpus_root).resolve(), Path(resolved).resolve()}
    write_location = out.parent.resolve() / out.name
    for candidate in {out.resolve(), write_location}:
        for root in roots:
            if candidate == root or root in candidate.parents:
                raise _SelectionError(
                    f"--output must not write inside the read-only corpus "
                    f"({candidate} -> {root}); choose a path outside it "
                    "(e.g. under /tmp)."
                )


def _declared_map(corpus) -> Mapping[str, Any] | None:
    declared = corpus.index.get("pairs")
    return declared if isinstance(declared, Mapping) else None


def _is_excluded(corpus, declared, fixture_id: str) -> str | None:
    """The exclusion reason for a selected row, mirroring
    ``phase4_real._release_population``: only a named kind that BOTH the
    loaded payload and (on a manifest-bound corpus) the manifest row agree on
    is dropped. Everything else stays expected and compares (or fails)."""
    pair = corpus.pairs.get(fixture_id)
    kind = ((pair or {}).get("payload") or {}).get("record_kind")
    if kind not in phase4_real.PHASE4_EXCLUDED_RECORD_KINDS:
        return None
    row = declared.get(fixture_id) if declared is not None else {"record_kind": kind}
    if isinstance(row, Mapping) and row.get("record_kind") == kind:
        return phase4_real.PHASE4_EXCLUDED_RECORD_KINDS[kind]
    return None


def _manifest_projection(declared, fixture_id: str, pair) -> tuple[dict, dict]:
    """The (manifest-side, pair-file-side) four-field view of one fixture,
    shaped exactly as ``checks.tier0_corpus.case_manifest`` builds it, so a
    selected row is bound to the same ``payload_hash``/``request_hash``/
    ``record_kind``/``covers`` contract the full battery enforces corpus-wide."""
    row = declared.get(fixture_id)
    payload = pair.get("payload") or {}
    if not isinstance(row, Mapping):
        return None, None
    manifest_side = {
        "payload_hash": row.get("payload_hash"),
        "request_hash": row.get("request_hash"),
        "record_kind": row.get("record_kind"),
        "covers": row.get("covers"),
    }
    file_side = {
        "payload_hash": pair.get("payload_hash"),
        "request_hash": pair.get("request_hash"),
        "record_kind": payload.get("record_kind"),
        "covers": pair.get("covers"),
    }
    return manifest_side, file_side


def _verify_manifest_fields(declared, fixture_id: str, pair) -> list[str]:
    """Typed reasons for a declared row that is malformed or disagrees with its
    pair file on any of the four manifest fields. Mirrors ``case_manifest``'s
    per-pair requirement through the same ``compare_records`` helper; a missing
    ``payload_hash`` (or any field) is a refusal, never a silent pass."""
    manifest_side, file_side = _manifest_projection(declared, fixture_id, pair)
    if manifest_side is None:
        return ["release manifest: declared entry is not an object"]
    reasons = []
    if manifest_side["payload_hash"] is None:
        reasons.append("release manifest: declared row is missing payload_hash")
    comparison = t0.compare_records(
        manifest_side, file_side, comparison_kind="tier0_manifest_row",
        left_ref="INDEX.json", right_ref="pair-file",
    )
    if comparison.verdict != t0.AGREE:
        fields = sorted({finding.field_path for finding in comparison.findings})
        reasons.append(
            "release manifest: metadata disagrees with pair file (" + ", ".join(fields) + ")"
        )
    return reasons


def _verify_payload(corpus, declared, fixture_id: str) -> dict[str, Any]:
    """Verify a loaded pair's integrity through the repository's own code: the
    recomputed payload digest (``content_hash``, per-row analog of
    ``checks.tier0_corpus.case_digest``) AND — on a manifest-bound corpus —
    agreement with the declared row's four metadata fields (per-row analog of
    ``case_manifest``). ``fragments=corpus.fragments`` is required so a hydrated
    shared subtree hashes back to the stored digest exactly (byte-identical —
    the fragments protocol's own contract)."""
    pair = corpus.pairs[fixture_id]
    stored = pair.get("payload_hash")
    recomputed = content_hash(pair["payload"], fragments=corpus.fragments)
    reasons = []
    if recomputed != stored:
        reasons.append("payload_hash: digest mismatch (stored vs recomputed)")
    if declared is not None:
        reasons.extend(_verify_manifest_fields(declared, fixture_id, pair))
    return {
        "payload_verified": not reasons,
        "payload_hash_stored": stored,
        "payload_hash_recomputed": recomputed,
        "reasons": reasons,
    }


def _row_diffs(members) -> dict[str, Any]:
    """Fold per-member ``_record_checks`` differences into value-free row
    evidence. Regular rows have one member; a chooser row has one entry per
    ranked member, so non-empty per-member dicts are collected into a list."""
    key_diffs, flag_diffs, mask_diffs, never_ran = [], [], [], []
    for diff in (m["differences"] for m in members):
        if diff.get("key_differences"):
            key_diffs.append(diff["key_differences"])
        if diff.get("flag_differences"):
            flag_diffs.append(diff["flag_differences"])
        if diff.get("null_mask_differences"):
            mask_diffs.append(diff["null_mask_differences"])
        if diff.get("never_ran_dimensions"):
            never_ran.append(diff["never_ran_dimensions"])
    single = len(members) == 1

    def _pick(items):
        if not items:
            return None
        return items[0] if single else items

    return {
        "key_differences": _pick(key_diffs),
        "flag_differences": _pick(flag_diffs),
        "null_mask_differences": _pick(mask_diffs),
        "never_ran_dimensions": _pick(never_ran),
    }


def _contract_side_projections(native, record) -> tuple[dict, dict]:
    """Both sides of one member's ``contracts`` projection, built through the
    EXACT helper the checker compares (``phase4_real._contract_projection``) and
    with the identical caller-side extraction ``_record_checks`` uses (native's
    ``legs``/``entry_exit_plan``/``quote_provenance``; legacy record's
    ``legs``/``entry_date``/``exit_date``/``quote_date``). Reusing the projection
    is the whole point: this can never disagree with the checker's own
    ``contracts`` verdict, so a reported difference always matches a real one."""
    native_projection = phase4_real._contract_projection(
        native.legs,
        entry_date=native.entry_exit_plan.get("entry_date"),
        exit_date=native.entry_exit_plan.get("exit_date"),
        execution_date=native.quote_provenance.get("quote_date"),
    )
    legacy_projection = phase4_real._contract_projection(
        record.get("legs") or (),
        entry_date=record.get("entry_date"),
        exit_date=record.get("exit_date"),
        execution_date=record.get("quote_date"),
    )
    return native_projection, legacy_projection


def _projected_contract_diff(native_projection, legacy_projection) -> dict | None:
    """Attribute-level differences between two contract projections, or ``None``
    when they are equal. Walks the aligned legs attribute-by-attribute in the
    fixed report order, flags a leg-count mismatch as a scalar (never dumping an
    entire leg), and names any timeline date that differs — always with BOTH
    sides' values, which is what lets a reader see the actual contract drift.
    Only structural leg/timeline terms are reported, so this stays bounded."""
    native_legs, legacy_legs = native_projection["legs"], legacy_projection["legs"]
    count_mismatch = len(native_legs) != len(legacy_legs)
    leg_diffs = []
    for index in range(min(len(native_legs), len(legacy_legs))):
        native_leg, legacy_leg = native_legs[index], legacy_legs[index]
        for attribute in _LEG_ATTRIBUTES:
            native_value = native_leg.get(attribute)
            legacy_value = legacy_leg.get(attribute)
            if native_value != legacy_value:
                leg_diffs.append({
                    "index": index, "attribute": attribute,
                    "native": native_value, "legacy": legacy_value,
                })
    timeline_diffs = []
    for attribute in _TIMELINE_ATTRIBUTES:
        native_value = native_projection.get(attribute)
        legacy_value = legacy_projection.get(attribute)
        if native_value != legacy_value:
            timeline_diffs.append({
                "attribute": attribute,
                "native": native_value, "legacy": legacy_value,
            })
    if not (count_mismatch or leg_diffs or timeline_diffs):
        return None
    diff: dict[str, Any] = {}
    if count_mismatch:
        diff["leg_count"] = {"native": len(native_legs), "legacy": len(legacy_legs)}
    if leg_diffs:
        diff["legs"] = leg_diffs
    if timeline_diffs:
        diff["timeline"] = timeline_diffs
    return diff


def _member_contract_diff(member) -> dict | None:
    """One member's contract difference (``None`` when its contracts agree)."""
    native_projection, legacy_projection = _contract_side_projections(
        member["native"], member["record"])
    return _projected_contract_diff(native_projection, legacy_projection)


def _row_contract_diffs(members):
    """Row-level ``contract_differences``: a single member's dict when there is
    one, else a member-INDEXED list (``{"member": i, ...}``, ascending, only the
    members that differ) so a multi-member DYN-SV row shows which ranked member's
    contract drifted rather than a positionally-ambiguous list."""
    per_member = [_member_contract_diff(m) for m in members]
    if len(members) == 1:
        return per_member[0]
    entries = [{"member": index, **diff}
               for index, diff in enumerate(per_member) if diff is not None]
    return entries or None


def _replay_regular_member(record, verified):
    """Native-replay one verified (non-chooser) trace and run the full gate's
    per-record comparison. Raises whatever the repository replay/verify code
    raises (e.g. a runtime-receipt mismatch); the caller records it honestly.
    ``trace_verified`` reflects that the caller already obtained a bundle whose
    ``trace_hash`` matched — that check happened in ``_verified_trace_bundle``."""
    native, receipts, _identities = phase4_real._replayed_member(verified)
    checks, numeric, differences = phase4_real._record_checks(record, native)
    return {
        "record": record,
        "native": native,
        "checks": checks,
        "numeric": numeric,
        "differences": differences,
        "trace_verified": True,
        "trace_hash": verified["trace_hash"],
        "runtime_stages": len(receipts),
    }


def _replay_chooser_members(pair, corpus_root: Path):
    """Verify + natively replay every ranked member of a ``dyn_sv_choice`` pair
    and compute the chooser-selection check, returning ``(rows, selection)``.

    Each member's own ``trace_hash`` is verified inside ``_replayed_chooser``.
    ``rows`` carry ONLY the ordinary per-member checks — the ``chooser``
    dimension is a single selection verdict over the combined native, added by
    the caller AFTER aggregating every member, exactly as ``_native_parity``
    does. Stamping it onto one member's checks would make a >1-member row raise
    ``KeyError('chooser')`` when the aggregation walks member 0's keys.
    """
    members, choice = phase4_real._replayed_chooser(pair, corpus_root)
    summary_record = pair["payload"]["record"]
    rows = []
    for member_record, verified, native, receipts, _identities in members:
        checks, numeric, differences = phase4_real._record_checks(member_record, native)
        rows.append({
            "record": member_record,
            "native": native,
            "checks": checks,
            "numeric": numeric,
            "differences": differences,
            "trace_verified": True,
            "trace_hash": verified["trace_hash"],
            "runtime_stages": len(receipts),
        })
    selection = phase4_real._chooser_selection_checks(summary_record, choice)
    return rows, selection


def _incomparable(row, *, reason, trace_verified):
    return {**row, "disposition": INCOMPARABLE, "trace_verified": trace_verified,
            "reason": reason}


def _build_row(corpus, declared, fixture_id: str) -> dict[str, Any]:
    """Compute one selected row's value-free evidence: manifest-bound payload
    integrity, then the strict trace verification + native replay + per-record
    comparison, each reported separately. Never raises: any failure becomes a
    typed incomparability reason, with ``trace_verified`` distinguishing "the
    trace hash did not match" from "the trace verified but its replay
    refused"."""
    loaded = fixture_id in corpus.pairs
    row: dict[str, Any] = {
        "fixture_id": fixture_id,
        "declared": declared is not None and fixture_id in declared,
        "loaded": loaded,
    }
    if declared is not None and not row["declared"]:
        return _incomparable(row, reason="release manifest: undeclared pair file",
                             trace_verified=None)
    if not loaded:
        return _incomparable(row, reason="release manifest: declared pair file missing",
                             trace_verified=None)

    payload = _verify_payload(corpus, declared, fixture_id)
    row.update(payload)
    if not payload["payload_verified"]:
        return _incomparable(row, reason="; ".join(payload["reasons"]),
                             trace_verified=None)

    excluded_reason = _is_excluded(corpus, declared, fixture_id)
    if excluded_reason is not None:
        return {**row, "disposition": EXCLUDED, "trace_verified": None,
                "reason": excluded_reason}

    pair = corpus.pairs[fixture_id]
    if payload_trace_gap(pair):
        return _incomparable(row, reason=str(pair["payload"]["strict_trace_gap"]),
                             trace_verified=False)
    chooser = pair["payload"].get("record_kind") == "dyn_sv_choice"
    selection = None

    if chooser:
        try:
            members, selection = _replay_chooser_members(pair, corpus.root)
        except Exception as exc:  # noqa: BLE001 -- honest incomparability
            # ``_replayed_chooser`` verifies each member's trace AND replays it,
            # so a raise cannot be attributed to the trace hash alone: report
            # verification status as UNKNOWN (None), never a false False.
            return _incomparable(row, reason=f"{type(exc).__name__}: {exc}",
                                 trace_verified=None)
        trace_verified = all(m["trace_verified"] for m in members)
        trace_hash = [m["trace_hash"] for m in members]
        runtime = [m["runtime_stages"] for m in members]
    else:
        try:
            verified = phase4_real._verified_trace_bundle(pair, corpus.root)
        except Exception as exc:  # noqa: BLE001 -- the trace hash itself failed
            return _incomparable(row, reason=f"{type(exc).__name__}: {exc}",
                                 trace_verified=False)
        try:
            members = [_replay_regular_member(pair["payload"]["record"], verified)]
        except Exception as exc:  # noqa: BLE001 -- trace verified; replay refused
            return _incomparable({**row, "trace_hash": verified["trace_hash"]},
                                 reason=f"{type(exc).__name__}: {exc}",
                                 trace_verified=True)
        trace_verified = True
        trace_hash = verified["trace_hash"]
        runtime = members[0]["runtime_stages"]

    # Aggregate the ORDINARY per-member checks first, then — for a chooser row
    # only — add the single ``chooser`` selection verdict computed over the
    # combined native, exactly as ``_native_parity`` does.
    checks = {name: all(m["checks"][name] for m in members) for name in members[0]["checks"]}
    if chooser:
        checks["chooser"] = all(selection.values())
    numeric_findings = {
        dimension: [field for m in members for field in m["numeric"][dimension]["finding_fields"]]
        for dimension in _NUMERIC_DIMENSIONS
    }
    diffs = _row_diffs(members)
    advisory = {
        "legacy": sorted(set(pair["payload"]["record"].get("flags") or ())
                         & phase4_real._ADVISORY_FLAGS),
        "native": sorted(set(members[0]["native"].reason_codes)
                         & phase4_real._ADVISORY_FLAGS),
    }
    result = {
        **row,
        "disposition": COMPARED,
        "trace_verified": trace_verified,
        "trace_hash": trace_hash,
        "runtime_receipt_stages": runtime,
        "checks": checks,
        "checks_failed": sorted(name for name, ok in checks.items() if not ok),
        "numeric_findings": numeric_findings,
        "advisory_flags": advisory,
        "members": len(members),
    }
    for key in ("key_differences", "flag_differences", "null_mask_differences",
                "never_ran_dimensions"):
        if diffs[key] is not None:
            result[key] = diffs[key]
    # A contracts mismatch already failed the row's check; explain WHICH
    # projected attribute drifted (reusing the checker's own projection, so the
    # reported difference can never disagree with the ``contracts`` verdict).
    if not checks.get("contracts", True):
        contract_diffs = _row_contract_diffs(members)
        if contract_diffs is not None:
            result["contract_differences"] = contract_diffs
    if chooser:
        result["chooser_findings"] = sorted(n for n, ok in selection.items() if not ok)
    return result


def payload_trace_gap(pair) -> bool:
    """True when a captured pair records that strict tracing produced no
    honest trace (``strict_trace_gap``) rather than a verifiable trace."""
    payload = pair.get("payload") or {}
    return payload.get("strict_trace_gap") is not None and not payload.get("input_trace")


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts by disposition and the lists of rows that did not fully agree.
    Deliberately NOT a verdict: there is no overall pass/fail here — only the
    full gate may fold rows into one."""
    return {
        "requested": len(rows),
        "counts": {
            COMPARED: sum(r["disposition"] == COMPARED for r in rows),
            EXCLUDED: sum(r["disposition"] == EXCLUDED for r in rows),
            INCOMPARABLE: sum(r["disposition"] == INCOMPARABLE for r in rows),
        },
        "discrepancies": sorted(
            r["fixture_id"] for r in rows
            if r["disposition"] == COMPARED and r.get("checks_failed")
        ),
        "incomparable": sorted(
            r["fixture_id"] for r in rows if r["disposition"] == INCOMPARABLE
        ),
    }


def run_targeted(corpus_root: Path, fixture_ids, *, output=None,
                 progress_stream=None, progress_interval: float = 45.0,
                 baseline_seconds_per_row: float = BASELINE_SECONDS_PER_ROW,
                 ) -> dict[str, Any]:
    """Load the corpus once through the repository hydration path, verify each
    selected fixture is declared/loaded, replay it through the full gate's
    comparison code, and return the value-free per-row report.

    Never runs the battery, never writes a file, never claims sign-off.
    ``output`` is only inspected (never written) to reject a path inside the
    read-only corpus before any work happens; the actual write stays in
    ``main``. ``progress_interval``/``baseline_seconds_per_row`` only tune the
    stderr heartbeat (and let a test observe a heartbeat while a single row
    blocks); they change nothing about what is verified."""
    progress_stream = progress_stream or sys.stderr
    requested = [str(fid) for fid in fixture_ids]
    if not requested:
        raise _SelectionError("no --fixture-id given")

    resolved = t0.resolve_corpus(Path(corpus_root))
    if not (resolved / "INDEX.json").is_file():
        raise _SelectionError(f"no tier-0 corpus at {resolved}")
    _reject_output_inside_corpus(Path(corpus_root), resolved, output)

    progress = _ProgressReporter(len(requested), progress_stream,
                                 interval=progress_interval,
                                 baseline=baseline_seconds_per_row)
    with progress:
        progress.begin_load()
        corpus = t0.load(resolved)
        progress.end_load()
        declared = _declared_map(corpus)
        unknown = [fid for fid in requested
                   if fid not in corpus.pairs
                   and not (declared is not None and fid in declared)]
        if unknown:
            raise _SelectionError(
                "selected fixture ids are neither declared nor loaded: "
                + ", ".join(unknown))
        rows = []
        for fixture_id in requested:
            progress.begin_row(fixture_id)
            started = time.perf_counter()
            rows.append(_build_row(corpus, declared, fixture_id))
            progress.end_row(fixture_id, time.perf_counter() - started)

    manifest_bound = declared is not None
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic": "phase4_targeted_replay",
        # Hard "do not mistake this for sign-off" markers first.
        "claims_sign_off": False,
        "sign_off": False,
        "overall_phase4_status": "not_evaluated",
        "full_phase4_gate_required": True,
        "scope_note": (
            "Targeted per-row diagnostic only. It replays the named fixtures "
            "through checks.phase4_real's comparison code but runs no "
            "acceptance battery and computes no overall Phase 4 status; the "
            "full checks/phase4_real.py gate remains required for sign-off."
        ),
        "memory_note": (
            "Loaded the full corpus once via checks.tier0_corpus.load(); "
            "reference hydration is corpus-wide, so selected-row verification "
            "costs a whole-corpus load. peak_rss_mb below is this process's "
            "high-water mark; no weaker hand-decoded hash path was used."
        ),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "corpus_root": str(resolved),
        "corpus_hash": corpus.index.get("corpus_hash"),
        "manifest_bound": manifest_bound,
        "requested": requested,
        "rows": rows,
        "summary": _summary(rows),
    }


class _ProgressReporter:
    """A watchdog progress stream for a long targeted replay.

    Emits a flushed heartbeat to the progress stream at least once every
    ``interval`` seconds. The whole operation has TWO phases, reported
    distinctly because the real corpus has been measured taking minutes and
    gigabytes to *hydrate* (``checks.tier0_corpus.load``) before a single row
    is even scored, and a whole-corpus load that a memory watchdog kills is not
    a parity finding:

    * ``phase=loading`` — until the load returns. No row-level ETA is implied:
      it prints ``load ETA unknown`` (we have no measured load prior) plus the
      row count still queued, so a slow/huge load never masquerades as replay
      progress.
    * ``phase=replay`` — after the load. A ``provisional ETA (replay, after
      load)`` covering ONLY the replay loop. It folds the currently-active
      row's elapsed time into the remaining estimate (so it ticks down and is
      never a static figure) and flags ``overdue`` once that row exceeds its
      own prior. Before any row completes the per-row rate is the documented
      ~32m/20-rows baseline; afterwards it is the observed mean.

    ``interval`` and ``clock`` are injectable so the phase/ETA logic is
    testable against a fake clock without real sleeping."""

    def __init__(self, total: int, stream, *, interval: float = 45.0,
                 baseline: float = BASELINE_SECONDS_PER_ROW,
                 clock=time.perf_counter) -> None:
        self._total = max(int(total), 0)
        self._stream = stream
        self._interval = interval
        self._baseline = baseline
        self._clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._completed = 0
        self._observed = 0.0
        self._current: str | None = None
        self._started = 0.0
        self._phase = "replay"
        self._load_start = 0.0
        self._load_seconds: float | None = None
        self._row_started: float | None = None

    def __enter__(self) -> "_ProgressReporter":
        self._started = self._clock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5.0)
        self._emit(self._clock())
        return False

    def begin_load(self) -> None:
        """Enter the loading phase (before ``t0.load``)."""
        with self._lock:
            self._phase = "loading"
            self._load_start = self._clock()

    def end_load(self) -> None:
        """Leave the loading phase (after ``t0.load`` returns)."""
        with self._lock:
            self._load_seconds = self._clock() - self._load_start
            self._phase = "replay"

    def begin_row(self, fixture_id: str) -> None:
        with self._lock:
            self._current = fixture_id
            self._row_started = self._clock()

    def end_row(self, fixture_id: str, seconds: float) -> None:
        with self._lock:
            self._completed += 1
            self._observed += max(seconds, 0.0)
            self._current = None
            self._row_started = None

    def _format(self, now: float) -> str:
        with self._lock:
            phase = self._phase
            completed, total = self._completed, self._total
            current = self._current
            row_started = self._row_started
            rate = (self._observed / completed) if completed else self._baseline
            source = "observed" if completed else "baseline ~32m/20 rows"
            load_seconds = self._load_seconds
            baseline = self._baseline
            started = self._started
        elapsed = now - started
        stamp = time.strftime("%H:%M:%S")
        if phase == "loading":
            return (f"[targeted-replay {stamp}] phase=loading; {total} row(s) "
                    f"queued; elapsed {elapsed:.0f}s; load ETA unknown")
        active = current is not None
        active_elapsed = (now - row_started) if (active and row_started is not None) else 0.0
        not_started = max(total - completed - (1 if active else 0), 0)
        leftover_active = max(rate - active_elapsed, 0.0) if active else 0.0
        eta_seconds = leftover_active + not_started * rate
        overdue = active and active_elapsed > rate
        where = f", current: {current}" if active else ""
        load = f"; load {load_seconds:.0f}s" if load_seconds is not None else ""
        tail = " (replay, after load)" if load_seconds is not None else ""
        flag = ", overdue" if overdue else ""
        return (f"[targeted-replay {stamp}] phase=replay {completed}/{total} "
                f"rows{where}; elapsed {elapsed:.0f}s{load}; provisional ETA{tail} "
                f"{eta_seconds / 60:.1f}m (rate {rate:.0f}s/row, {source}{flag})")

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._emit(self._clock())

    def _emit(self, now: float) -> None:
        print(self._format(now), file=self._stream, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only targeted strict Phase 4 replay diagnostic "
                    "(never claims sign-off).")
    parser.add_argument("--corpus", default="fixtures/tier0",
                        help="corpus root (a directory with INDEX.json, or a "
                             "CURRENT-pointer root); read-only.")
    parser.add_argument("--fixture-id", dest="fixture_ids", action="append",
                        default=[], help="an exact fixture id to replay; repeat "
                                         "for multiple.")
    parser.add_argument("--output", default=None,
                        help="write the JSON report here atomically instead of "
                             "stdout; a partial artifact is never left behind.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output) if args.output else None
    try:
        # run_targeted validates the output path (rejecting one inside the
        # read-only corpus) before it loads or replays anything.
        report = run_targeted(Path(args.corpus), args.fixture_ids, output=output)
    except _SelectionError as exc:
        print(f"targeted-replay: {exc}", file=sys.stderr)
        return 2
    except t0.CorpusFormatError as exc:
        # A hard loader refusal (malformed shared reference, digest mismatch):
        # reported honestly, never downgraded to a partial or weaker result.
        print(f"targeted-replay: corpus integrity failure: {exc}", file=sys.stderr)
        return 3
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        _atomic_write(output, text)
        print(f"targeted-replay: wrote report to {output} "
              f"(summary: {report['summary']['counts']})", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
