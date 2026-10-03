"""Read-only summary of the native_parity stage's own report artifact (P?-?).

``native_parity_summary`` never re-compares a record: every number it
returns is an aggregate over fields ``engine.v2.ops.native_parity_report``
already computed (``mismatches``/``only_legacy``/``only_native``/
``native_refused*``). ``engine/v2/serving`` (layer 7.0) and
``engine/v2/ops`` (layer 7.0) are equal-layer peers -- neither may import
the other -- so this module reads the report file directly as JSON rather
than importing anything from ``engine.v2.ops``; it takes only the checker's
public field groups from ``engine.v2.parity.dimensions`` for the per-mismatch
detail screen.

See root ``ARCHITECTURE.md`` section 4, "Native parity summary projection",
for the condition-to-outcome failure-semantics table this implements.
"""
from __future__ import annotations

import errno
import json
import os
import sqlite3
import stat
from http import HTTPStatus

import engine.v2.parity.dimensions as parity_dimensions
from engine.v2.foundation import parse_timestamp

__all__ = [
    "NATIVE_PARITY_SUMMARY_V1",
    "NATIVE_PARITY_REPORT_MALFORMED",
    "CAPTURED_COMPARISON_V1",
    "native_parity_snapshot",
    "native_parity_summary",
    "native_parity_items",
    "native_parity_freshness",
]

NATIVE_PARITY_SUMMARY_V1 = "native_parity_summary.v1.0"
NATIVE_PARITY_REPORT_MALFORMED = "NATIVE_PARITY_REPORT_MALFORMED"

_V12_SCHEMA = "native_parity_report.v1.2"

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
_ITEM_GROUP_FIELDS: dict[str, tuple[str, ...]] = dict(zip(
    _CAPTURED_GROUPS,
    (parity_dimensions.FORECAST_FIELDS, parity_dimensions.SIMULATION_FIELDS,
     parity_dimensions.FINANCIAL_FIELDS, parity_dimensions.GATE_FIELDS,
     parity_dimensions.ANALOG_FIELDS),
))
_CAPTURED_BOOL_FIELDS = frozenset({("verdicts", "gate_pass")})
_STRING_LIMIT = 512
_NAME_LIMIT = 128
_ROW_LIMIT = 100


def _counted(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _field_mismatch_counts(mismatches: list[dict[str, object]]) -> dict[str, int]:
    """How many mismatch entries name each field, across every dimension."""
    return _counted(field for entry in mismatches for field in entry["finding_fields"])


def _dimension_mismatch_counts(mismatches: list[dict[str, object]]) -> dict[str, int]:
    """How many mismatch entries (one per row+dimension) name each dimension."""
    return _counted(entry["dimension"] for entry in mismatches)


def _worst_rows(mismatches: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    """The rows with the most mismatched fields, most first.

    One row can carry several dimension mismatches; a row's total is the
    sum of ``len(finding_fields)`` across all of them. Ties break on
    ``row_key`` so the result is deterministic.
    """
    totals: dict[str, dict[str, object]] = {}
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


def _reason_counts(entries: list[dict[str, object]]) -> dict[str, int]:
    return _counted(entry["refusal_code"] for entry in entries)


def _open_regular_no_follow(path: str | os.PathLike):
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


def _validate_mismatches(mismatches: list[object]) -> None:
    for entry in mismatches:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("row_key"), str)
            or not isinstance(entry.get("dimension"), str)
            or not isinstance(entry.get("finding_fields"), list)
            or not all(isinstance(field, str) for field in entry["finding_fields"])
        ):
            raise ValueError("native parity mismatch entry is malformed")


def _validate_optional_lists(report: dict[str, object]) -> None:
    for field in _OPTIONAL_LIST_FIELDS:
        if field in report:
            value = report[field]
            if not isinstance(value, list):
                raise ValueError(f"native parity report field {field!r} is present but not a list")
            for entry in value:
                if not isinstance(entry, dict) or not isinstance(entry.get("refusal_code"), str):
                    raise ValueError(f"native parity report field {field!r} entry is malformed")


def _validate_compared(compared: list[object], mismatches: list[dict[str, object]]) -> None:
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


def _validate_report_values(mismatches: list[dict[str, object]]) -> None:
    """Every finding names a field of its dimension's group and carries the supplied values pair."""
    for entry in mismatches:
        group = _ITEM_GROUP_FIELDS.get(entry["dimension"])
        if group is None or not set(entry["finding_fields"]).issubset(group):
            raise ValueError("native parity mismatch dimension or finding field is unknown")
        values = entry.get("values")
        if not isinstance(values, dict):
            raise ValueError("native parity mismatch values is not an object")
        for name in entry["finding_fields"]:
            saved = values.get(name)
            if (not isinstance(saved, dict)
                    or "legacy" not in saved or "native" not in saved):
                raise ValueError("native parity mismatch value pair is incomplete")
            json.dumps(saved, allow_nan=False)


def _validate_run_identity(report: dict[str, object]) -> None:
    """Validate modern run identity: both keys, parsable timestamps, policy, values."""
    has_as_of = "as_of" in report
    has_generated_at = "generated_at" in report
    if not (report["schema_version"] == _V12_SCHEMA or has_as_of or has_generated_at):
        return
    if has_as_of != has_generated_at:
        raise ValueError("native parity report must carry both as_of and generated_at")
    if has_as_of:
        parse_timestamp(report["generated_at"])
        as_of = report["as_of"]
        if as_of is not None:
            if not isinstance(as_of, str) or len(as_of) != 10:
                raise ValueError("native parity as_of is not an ISO date")
            parse_timestamp(as_of + "T00:00:00.000000Z")
    policy = report.get("tolerance_policy_id")
    if not isinstance(policy, str) or not policy:
        raise ValueError("native parity tolerance_policy_id is missing")
    _validate_report_values(report["mismatches"])


def _load_report(handle) -> dict[str, object]:
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
    for field in ("only_legacy", "only_native"):
        if not all(isinstance(key, str) for key in report[field]):
            raise ValueError(f"native parity report field {field!r} has a non-string entry")
    _validate_optional_lists(report)
    _validate_run_identity(report)
    return report


def _captured_string(value: object, limit: int, label: str) -> str:
    if not isinstance(value, str) or not (1 <= len(value) <= limit):
        raise ValueError(f"captured comparison {label} is not a bounded non-empty string")
    return value


def _captured_number(value: object) -> object:
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


_MAX_SAFE_INTEGER = 2 ** 53 - 1


def _captured_display(value: object) -> object:
    """JSON-safe rendering form of an already-validated captured number.

    The browser's ``Response.json`` converts the payload to IEEE-754, so a
    JSON integer beyond ``2**53 - 1`` would silently lose exactness (and a
    400-digit one become ``Infinity``). Null, zero, finite floats and safe
    ints keep the original value; only an ``int`` whose absolute value
    exceeds the safe range becomes its exact base-10 ``str``. The exporter's
    canonical boolean verdict (``verdicts.gate_pass``) passes through here
    unchanged -- bools are never numeric inputs -- and the original
    ``legacy``/``native`` fields always keep the source value.
    """
    if isinstance(value, int) and abs(value) > _MAX_SAFE_INTEGER:
        return str(value)
    return value


def _captured_value(group: str, field: str, value: object) -> object:
    """One captured row value under the schema's exact per-field typing.

    Every group/field is numeric except the single canonical boolean gate
    verdict: at ``verdicts.gate_pass`` only ``True``/``False``/``None`` are
    accepted -- numeric ``0``/``1``, floats and strings are refused so a
    bool can never masquerade as a number and vice versa. ``False`` in any
    other field/group is refused by the strict ``_captured_number``.
    """
    if (group, field) in _CAPTURED_BOOL_FIELDS:
        if value is None or isinstance(value, bool):
            return value
        raise ValueError("captured comparison boolean verdict is not true/false/null")
    return _captured_number(value)


def _captured_row(group: str, field: str, legacy_value: object,
                  native_value: object) -> dict[str, object]:
    """Validate each side once; emit the original and additive display fields."""
    legacy = _captured_value(group, field, legacy_value)
    native = _captured_value(group, field, native_value)
    return {"group": group, "field": field,
            "legacy": legacy, "native": native,
            "legacy_display": _captured_display(legacy),
            "native_display": _captured_display(native)}


def _captured_aligned_group(group: str, left: object, right: object) -> None:
    """One legacy/native group: a non-empty aligned object of bounded names."""
    if (not isinstance(left, dict) or not isinstance(right, dict)
            or not left or set(left) != set(right)):
        raise ValueError(f"captured comparison group {group!r} is not an aligned object")
    if any(not isinstance(field, str) or not (1 <= len(field) <= _NAME_LIMIT)
           for field in left):
        raise ValueError(f"captured comparison group {group!r} has a bad field name")


def _captured_metadata_section(block: dict[str, object], key: str,
                               fields: tuple[str, ...]) -> dict[str, object]:
    """An exact-key metadata object of bounded non-empty strings; return it."""
    section = block.get(key)
    if not isinstance(section, dict) or set(section) != set(fields):
        raise ValueError(f"captured comparison {key} key set is not exact")
    for name in fields:
        _captured_string(section[name], _STRING_LIMIT, f"{key} {name!r}")
    return section


def _captured_rows(legacy: object, native: object) -> list[dict[str, object]]:
    if (not isinstance(legacy, dict) or not isinstance(native, dict)
            or set(legacy) != set(_CAPTURED_GROUPS)
            or set(native) != set(_CAPTURED_GROUPS)):
        raise ValueError("captured comparison groups are not the exact five")
    for group in _CAPTURED_GROUPS:
        _captured_aligned_group(group, legacy[group], native[group])
    if sum(len(legacy[group]) for group in _CAPTURED_GROUPS) > _ROW_LIMIT:
        raise ValueError("captured comparison row count exceeds the bound")
    return [_captured_row(group, field, legacy[group][field], native[group][field])
            for group in _CAPTURED_GROUPS for field in sorted(legacy[group])]


def _project_captured_comparison(block: object) -> dict[str, object]:
    """Validate the exporter's optional captured block and project paired rows.

    Raises ``ValueError`` (the existing malformed-report refusal) on any
    deviation from the v1 schema. Never re-verifies provenance, never
    recomputes a hash or compares the two request hashes, and never rounds,
    deltas or substitutes a captured numeric value. The additive
    ``legacy_display``/``native_display`` row fields carry the original
    value, or the exact base-10 string of an int beyond the browser's
    safe-integer range, so unsafe integer display values remain decimal
    strings through browser parsing. The raw
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


def _mismatch_item(entry: dict[str, object]) -> dict[str, object]:
    fields = {name: {"status": "agree"}
              for name in _ITEM_GROUP_FIELDS.get(entry["dimension"], ())}
    values = entry.get("values")
    for name in entry["finding_fields"]:
        fields[name] = {"status": "differ"}
        saved = values.get(name) if isinstance(values, dict) else None
        if isinstance(saved, dict) and "legacy" in saved and "native" in saved:
            fields[name].update(legacy=saved["legacy"], native=saved["native"])
    return {"row_key": entry["row_key"], "dimension": entry["dimension"],
            "fields": fields}


def native_parity_items(report: dict, section: str, *,
                        side: str | None = None,
                        row_key: str | None = None) -> list:
    """One detail collection; a row_key outside the report population raises LookupError."""
    population = set(report["compared"]) | set(report["only_legacy"]) | set(report["only_native"])
    if row_key is not None and row_key not in population:
        raise LookupError(row_key)
    if section == "unpaired":
        keys = report["only_" + side]
        return sorted(key for key in keys if row_key is None or key == row_key)
    if section != "mismatches":
        raise ValueError(f"unknown native parity detail section {section!r}")
    selected = [entry for entry in report["mismatches"]
                if row_key is None or entry["row_key"] == row_key]
    selected.sort(key=lambda entry: (entry["row_key"], entry["dimension"]))
    return [_mismatch_item(entry) for entry in selected]


def native_parity_freshness(serving_db, resolve_current, as_of: str | None, open_index,
                            read_release, resolver_errors: tuple[type[BaseException], ...] = ()) -> str:
    """Only a genuinely earlier run date is stale; unknown current stays available."""
    if as_of is None:
        return "available"
    try:
        release_id = resolve_current()
    except (sqlite3.OperationalError,) + tuple(resolver_errors):
        return "available"
    if release_id is None:
        return "available"
    try:
        conn = open_index(serving_db)
    except sqlite3.OperationalError:
        return "available"
    try:
        release = read_release(conn, release_id)
    except sqlite3.OperationalError:
        release = None
    finally:
        conn.close()
    return "stale" if release is not None and as_of < release.resolved_as_of else "available"


def _empty_snapshot(unavailable: bool = False) -> tuple[HTTPStatus, dict, None]:
    summary = {"schema_version": NATIVE_PARITY_SUMMARY_V1,
               "status": "unavailable" if unavailable else "no_report"}
    if unavailable:
        summary["reason_code"] = NATIVE_PARITY_REPORT_MALFORMED
    return (HTTPStatus.SERVICE_UNAVAILABLE if unavailable else HTTPStatus.OK), summary, None


def native_parity_snapshot(report_path: str | os.PathLike | None, *,
                           worst_limit: int = 10) -> tuple[HTTPStatus, dict, dict | None]:
    """``(status, summary, report_or_None)``; report is present only when available."""
    if report_path is None:
        return _empty_snapshot()
    path = os.fspath(report_path)
    try:
        handle = _open_regular_no_follow(path)
    except FileNotFoundError:
        return _empty_snapshot()
    except OSError:
        return _empty_snapshot(unavailable=True)
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
        return _empty_snapshot(unavailable=True)
    summary = {
        "schema_version": NATIVE_PARITY_SUMMARY_V1,
        "status": "available",
        "partial": partial,
        "source_schema_version": report["schema_version"],
        "as_of": report.get("as_of"),
        "generated_at": report.get("generated_at"),
        "tolerance_policy_id": report.get("tolerance_policy_id"),
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
    return HTTPStatus.OK, summary, report


def native_parity_summary(report_path: str | os.PathLike | None, *,
                          worst_limit: int = 10) -> tuple[HTTPStatus, dict]:
    """Summary-only wrapper over :func:`native_parity_snapshot`'s first two elements."""
    return native_parity_snapshot(report_path, worst_limit=worst_limit)[:2]
