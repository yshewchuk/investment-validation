"""Read-only summary of the native_parity stage's own report artifact (P?-?).

``native_parity_summary`` never re-compares a record: every number it
returns is an aggregate over fields ``engine.v2.ops.native_parity_report``
already computed (``mismatches``/``only_legacy``/``only_native``/
``native_refused*``). ``engine/v2/serving`` (layer 7.0) and
``engine/v2/ops`` (layer 7.0) are equal-layer peers -- neither may import
the other -- so this module reads the report file directly as JSON rather
than importing anything from ``engine.v2.ops`` or ``engine.v2.parity``.

See root ``ARCHITECTURE.md`` section 4, "Native parity summary projection",
for the condition-to-outcome failure-semantics table this implements.
"""
from __future__ import annotations

import errno
import json
import os
import stat
from collections import Counter
from http import HTTPStatus
from pathlib import Path
from typing import Any

__all__ = [
    "NATIVE_PARITY_SUMMARY_V1",
    "NATIVE_PARITY_REPORT_MALFORMED",
    "CAPTURED_COMPARISON_V1",
    "native_parity_summary",
]

NATIVE_PARITY_SUMMARY_V1 = "native_parity_summary.v1.0"
NATIVE_PARITY_REPORT_MALFORMED = "NATIVE_PARITY_REPORT_MALFORMED"

_REQUIRED_LIST_FIELDS = ("compared", "only_legacy", "only_native", "mismatches")
_OPTIONAL_LIST_FIELDS = ("native_refused", "native_refused_unmatched")

CAPTURED_COMPARISON_V1 = "captured_native_comparison.v1.0"
_CAPTURED_SCOPE = "selected_saved_replay"
_CAPTURED_STRATEGY = "STR-THRU"
_CAPTURED_FLAGS = ("full_population_verified", "cutover_qualified", "current_board")
_CAPTURED_IDENTITY_FIELDS = ("ticker", "strategy", "event_date", "session", "as_of",
                             "entry_date", "exit_date")
_CAPTURED_CLOCK_FIELDS = ("corpus_as_of", "requested_decision_at", "decision_as_of",
                          "quote_as_of", "event_date", "session")
_CAPTURED_PROVENANCE_FIELDS = ("corpus_hash", "fixture_id", "payload_hash",
                               "legacy_request_hash", "native_request_hash", "trace_hash",
                               "same_input_receipt", "frozen_release_id",
                               "native_snapshot_ref")
_CAPTURED_GROUPS = ("forecasts", "simulation", "financial_diagnostics", "verdicts", "analogs")
_STRING_LIMIT = 512
_NAME_LIMIT = 128
_ROW_LIMIT = 100


def _field_mismatch_counts(mismatches: list[dict[str, Any]]) -> dict[str, int]:
    """How many mismatch entries name each field, across every dimension."""
    counts: Counter[str] = Counter()
    for entry in mismatches:
        counts.update(entry["finding_fields"])
    return dict(sorted(counts.items()))


def _dimension_mismatch_counts(mismatches: list[dict[str, Any]]) -> dict[str, int]:
    """How many mismatch entries (one per row+dimension) name each dimension."""
    counts = Counter(entry["dimension"] for entry in mismatches)
    return dict(sorted(counts.items()))


def _worst_rows(mismatches: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """The rows with the most mismatched fields, most first.

    One row can carry several dimension mismatches; a row's total is the
    sum of ``len(finding_fields)`` across all of them. Ties break on
    ``row_key`` so the result is deterministic.
    """
    totals: dict[str, dict[str, Any]] = {}
    for entry in mismatches:
        key = entry["row_key"]
        row = totals.setdefault(
            key, {"row_key": key, "mismatched_field_count": 0, "dimensions": []})
        row["mismatched_field_count"] += len(entry["finding_fields"])
        row["dimensions"].append(entry["dimension"])
    ordered = sorted(
        totals.values(),
        key=lambda row: (-row["mismatched_field_count"], row["row_key"]))
    return ordered[:limit]


def _reason_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(entry["refusal_code"] for entry in entries)
    return dict(sorted(counts.items()))


def _open_regular_no_follow(path: Path):
    """Open ``path`` for reading without ever following a symlink.

    Raises ``FileNotFoundError`` for a missing path, a symlink, or anything
    that is not a plain regular file -- the caller's ``"no_report"``
    outcome. Any other ``OSError`` propagates as the caller's
    ``"unavailable"`` outcome. The symlink/regular-file check and the read
    happen on the SAME open file descriptor, so a path swapped for a
    symlink between the check and the read can never be followed
    (TOCTOU-safe).
    """
    flags = os.O_RDONLY | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ELOOP, errno.ENOTDIR):
            raise FileNotFoundError(str(path)) from exc
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise FileNotFoundError(str(path))
    except BaseException:
        os.close(fd)
        raise
    return os.fdopen(fd, "r")


def _validate_mismatches(mismatches: list[Any]) -> None:
    for entry in mismatches:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("row_key"), str)
            or not isinstance(entry.get("dimension"), str)
            or not isinstance(entry.get("finding_fields"), list)
            or not all(isinstance(field, str) for field in entry["finding_fields"])
        ):
            raise ValueError("native parity mismatch entry is malformed")


def _validate_optional_lists(report: dict[str, Any]) -> None:
    for field in _OPTIONAL_LIST_FIELDS:
        if field in report:
            value = report[field]
            if not isinstance(value, list):
                raise ValueError(f"native parity report field {field!r} is present but not a list")
            for entry in value:
                if not isinstance(entry, dict) or not isinstance(entry.get("refusal_code"), str):
                    raise ValueError(f"native parity report field {field!r} entry is malformed")


def _validate_compared(compared: list[Any], mismatches: list[dict[str, Any]]) -> None:
    """``compared`` must be unique strings, and every mismatch's ``row_key``
    must be one of them.

    ``matched_row_count`` is derived as
    ``len(compared) - len(distinct mismatch row keys)``; without this check a
    malformed report (duplicate ``compared`` entries, or a mismatch row_key
    that is not in ``compared``) can make that arithmetic wrong -- including
    negative -- while still passing as ``200 available``.
    """
    if not all(isinstance(key, str) for key in compared):
        raise ValueError("native parity report field 'compared' has a non-string entry")
    if len(compared) != len(set(compared)):
        raise ValueError("native parity report field 'compared' has duplicate keys")
    compared_set = set(compared)
    for entry in mismatches:
        if entry["row_key"] not in compared_set:
            raise ValueError("native parity mismatch row_key is not in 'compared'")


def _load_report(handle) -> dict[str, Any]:
    """Parse and shape-check the report; raises on anything malformed."""
    report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError("native parity report is not a JSON object")
    if not isinstance(report.get("schema_version"), str):
        raise ValueError("native parity report has no string schema_version")
    for field in _REQUIRED_LIST_FIELDS:
        if not isinstance(report.get(field), list):
            raise ValueError(f"native parity report field {field!r} is missing or not a list")
    _validate_mismatches(report["mismatches"])
    _validate_compared(report["compared"], report["mismatches"])
    _validate_optional_lists(report)
    return report


def _captured_string(value: Any, limit: int, label: str) -> str:
    if not isinstance(value, str) or not (1 <= len(value) <= limit):
        raise ValueError(f"captured comparison {label} is not a bounded non-empty string")
    return value


def _captured_number(value: Any) -> Any:
    """Accept null and finite JSON numbers; reject bools, strings and non-finite floats.

    Finiteness is a strict open interval against the float infinities, so
    NaN -- which fails every comparison -- is rejected too. The check runs
    only on floats: arbitrary-precision JSON integers are finite and must
    be preserved exactly.
    """
    if value is None:
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("captured comparison value is neither null nor a finite number")
    if isinstance(value, float) and not (float("-inf") < value < float("inf")):
        raise ValueError("captured comparison value is neither null nor a finite number")
    return value


def _captured_aligned_group(group: str, left: Any, right: Any) -> None:
    """One legacy/native group: a non-empty aligned object of bounded names."""
    if (not isinstance(left, dict) or not isinstance(right, dict)
            or not left or set(left) != set(right)):
        raise ValueError(f"captured comparison group {group!r} is not an aligned object")
    if any(not isinstance(field, str) or not (1 <= len(field) <= _NAME_LIMIT)
           for field in left):
        raise ValueError(f"captured comparison group {group!r} has a bad field name")


def _captured_metadata_section(block: dict[str, Any], key: str,
                               fields: tuple[str, ...]) -> dict[str, Any]:
    """An exact-key metadata object of bounded non-empty strings; return it."""
    section = block.get(key)
    if not isinstance(section, dict) or set(section) != set(fields):
        raise ValueError(f"captured comparison {key} key set is not exact")
    for name in fields:
        _captured_string(section[name], _STRING_LIMIT, f"{key} {name!r}")
    return section


def _captured_rows(legacy: Any, native: Any) -> list[dict[str, Any]]:
    if (not isinstance(legacy, dict) or not isinstance(native, dict)
            or set(legacy) != set(_CAPTURED_GROUPS)
            or set(native) != set(_CAPTURED_GROUPS)):
        raise ValueError("captured comparison groups are not the exact five")
    for group in _CAPTURED_GROUPS:
        _captured_aligned_group(group, legacy[group], native[group])
    if sum(len(legacy[group]) for group in _CAPTURED_GROUPS) > _ROW_LIMIT:
        raise ValueError("captured comparison row count exceeds the bound")
    return [{"group": group, "field": field,
             "legacy": _captured_number(legacy[group][field]),
             "native": _captured_number(native[group][field])}
            for group in _CAPTURED_GROUPS for field in sorted(legacy[group])]


def _project_captured_comparison(block: Any) -> dict[str, Any]:
    """Validate the exporter's optional captured block and project paired rows.

    Raises ``ValueError`` (the existing malformed-report refusal) on any
    deviation from the v1 schema. Never re-verifies provenance, never
    recomputes a hash or compares the two request hashes, and never rounds,
    deltas or substitutes a captured numeric value. The raw
    ``checks``/``numeric_comparisons``/``runtime_stage_count`` diagnostics
    are not exposed as comparison logic.
    """
    if not isinstance(block, dict):
        raise ValueError("captured comparison is not a JSON object")
    if block.get("schema_version") != CAPTURED_COMPARISON_V1:
        raise ValueError("captured comparison schema_version is not the v1 literal")
    if block.get("scope") != _CAPTURED_SCOPE:
        raise ValueError("captured comparison scope is not the selected-replay literal")
    for flag in _CAPTURED_FLAGS:
        if block.get(flag) is not False:
            raise ValueError(f"captured comparison flag {flag!r} is not exactly false")
    identity = _captured_metadata_section(block, "identity", _CAPTURED_IDENTITY_FIELDS)
    if identity["strategy"] != _CAPTURED_STRATEGY:
        raise ValueError("captured comparison identity strategy is not STR-THRU")
    clocks = _captured_metadata_section(block, "clocks", _CAPTURED_CLOCK_FIELDS)
    for clock, field in (("decision_as_of", "as_of"), ("event_date", "event_date"),
                         ("session", "session")):
        if clocks[clock] != identity[field]:
            raise ValueError(f"captured comparison clock {clock!r} contradicts identity")
    provenance = _captured_metadata_section(block, "provenance", _CAPTURED_PROVENANCE_FIELDS)
    rows = _captured_rows(block.get("legacy"), block.get("native"))
    return {
        "schema_version": CAPTURED_COMPARISON_V1,
        "scope": _CAPTURED_SCOPE,
        "full_population_verified": False,
        "cutover_qualified": False,
        "current_board": False,
        "identity": {name: identity[name] for name in _CAPTURED_IDENTITY_FIELDS},
        "clocks": {name: clocks[name] for name in _CAPTURED_CLOCK_FIELDS},
        "provenance": {name: provenance[name] for name in _CAPTURED_PROVENANCE_FIELDS},
        "rows": rows,
    }


def native_parity_summary(report_path: Path | str, *, worst_limit: int = 10) -> tuple[HTTPStatus, dict]:
    """Load ``report_path`` and return a compact, dashboard-ready summary.

    See the module docstring and root ``ARCHITECTURE.md`` section 4 for the
    failure semantics: missing file, a symlink, or a non-regular path ->
    ``"no_report"`` (200, the everyday state today, no production job
    writes this artifact yet); malformed content, including a present but
    non-list/wrongly-typed required or optional field -> ``"unavailable"``
    (503); a valid report simply missing the optional
    ``native_refused``/``native_refused_unmatched`` KEYS (pre-refusal
    schema) -> ``"available"`` with ``partial: true``. A key present with a
    null or otherwise non-list value is malformed, never treated as absent.
    """
    path = Path(report_path)
    try:
        handle = _open_regular_no_follow(path)
    except FileNotFoundError:
        return HTTPStatus.OK, {
            "schema_version": NATIVE_PARITY_SUMMARY_V1,
            "status": "no_report",
        }
    except OSError:
        return HTTPStatus.SERVICE_UNAVAILABLE, {
            "schema_version": NATIVE_PARITY_SUMMARY_V1,
            "status": "unavailable",
            "reason_code": NATIVE_PARITY_REPORT_MALFORMED,
        }
    try:
        with handle:
            report = _load_report(handle)
        mismatches = report["mismatches"]
        mismatched_row_count = len({entry["row_key"] for entry in mismatches})
        matched_row_count = len(report["compared"]) - mismatched_row_count
        field_mismatch_counts = _field_mismatch_counts(mismatches)
        dimension_mismatch_counts = _dimension_mismatch_counts(mismatches)
        worst_rows = _worst_rows(mismatches, worst_limit)
        has_refused = "native_refused" in report
        has_refused_unmatched = "native_refused_unmatched" in report
        native_refused = report.get("native_refused", [])
        native_refused_unmatched = report.get("native_refused_unmatched", [])
        partial = not (has_refused and has_refused_unmatched)
        reason_counts = _reason_counts(native_refused + native_refused_unmatched)
        captured = (
            _project_captured_comparison(report["captured_comparison"])
            if "captured_comparison" in report else None)
    except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError):
        return HTTPStatus.SERVICE_UNAVAILABLE, {
            "schema_version": NATIVE_PARITY_SUMMARY_V1,
            "status": "unavailable",
            "reason_code": NATIVE_PARITY_REPORT_MALFORMED,
        }
    summary = {
        "schema_version": NATIVE_PARITY_SUMMARY_V1,
        "status": "available",
        "partial": partial,
        "source_schema_version": report["schema_version"],
        "compared_count": len(report["compared"]),
        "only_legacy_count": len(report["only_legacy"]),
        "only_native_count": len(report["only_native"]),
        "matched_row_count": matched_row_count,
        "mismatched_row_count": mismatched_row_count,
        "field_mismatch_counts": field_mismatch_counts,
        "dimension_mismatch_counts": dimension_mismatch_counts,
        "worst_rows": worst_rows,
        "native_refused_count": len(native_refused),
        "native_refused_unmatched_count": len(native_refused_unmatched),
        "native_refused_reasons": reason_counts,
    }
    if captured is not None:
        summary["captured_comparison"] = captured
    return HTTPStatus.OK, summary
