"""Per-night native-vs-legacy parity report (spec_ns_c, G2/G4).

This module is the reporting half of the native shadow-serving seam
(``engine.v2.ops.native_shadow_render`` / ``engine.v2.serving.native_shadow_render``):
it compares the legacy rows the projection already carries with the native rows
``build_native_bundle_rows`` built, and writes the differences down.  It is
deliberately NOT a second comparator: every per-dimension verdict comes from
``engine.v2.parity.dimensions.compare_dimension`` -- the same tolerance policy,
the same ``engine.v2.diagnosis`` machinery and the same field groups the Phase
4 corpus comparison uses -- so the nightly report can never drift away from the
checker it reports on.  That module is where the checker's comparator moved
(``spec_ns_c`` part c: production never imports ``checks/``, and "never
duplicate the comparison logic here"); no comparison rule is re-implemented in
this file.

G2 (report, never reconcile): ``compare_native_vs_legacy`` classifies and
returns.  It has no code path that copies a native value into a legacy row (or
the reverse), and a mismatch is a return-value finding, never an exception.

G4 (no hashed payload changes): this module only reads rows and writes a NEW,
separate report artifact at the caller's explicit path.  It never touches
``score.json``'s ``rows``/``ladder``, any ``ScoreRecord`` field, or the corpus
``checks/phase4_real.py`` hashes.

The ``native_parity`` stage is OPTIONAL in ``engine.v2.ops.nightly.GRAPH``: a
parity-report failure degrades the shadow board's receipt, it never blocks it
(G2: a difference is reported, not turned into a hard gate).  In ``"legacy"``
serving mode there are no native rows to compare, so
:func:`native_parity_handler` returns ``{"status": "not_applicable"}`` instead
of the stage being omitted from the graph (a conditionally absent stage is
exactly the DAG-shape drift ``nightly.GRAPH`` exists to avoid).
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from engine.v2.foundation import Clock, SystemClock, format_timestamp
from engine.v2.ops.decision_validation import population_key
from engine.v2.ops.errors import fail
from engine.v2.ops.native_shadow_render import native_shadow_serving_mode
from engine.v2.parity.dimensions import (
    ANALOG_FIELDS,
    FINANCIAL_FIELDS,
    FORECAST_FIELDS,
    GATE_FIELDS,
    NEVER_RAN_DIMENSIONS,
    SIMULATION_FIELDS,
    compare_dimension,
)
from engine.v2.parity.tolerance import SCORE_RECORD_V1, TolerancePolicy

__all__ = [
    "PARITY_DIMENSIONS",
    "SCHEMA_VERSION",
    "apply_native_refusals",
    "compare_native_vs_legacy",
    "native_parity_handler",
    "run_native_parity_worker",
    "write_parity_report",
]

#: v1.1 -> v1.2: adds the top-level run identity ``as_of``/``generated_at``
#: and, inside each ``mismatches`` entry, ``values`` -- that entry's own
#: mismatched fields only, keyed ``{legacy, native}``.
SCHEMA_VERSION = "native_parity_report.v1.2"

#: The exact ``schema_version`` tags
#: ``native_score_batch._native_score_batch_documents`` writes for its v2.0
#: ``records.json``/``refusals.json`` pair. A worker fed anything else refuses
#: up front rather than silently misreading a pre-v2.0 document as if it were
#: keyed.
_RECORDS_SCHEMA_VERSION = "native_score_batch_records.v2.0"
_REFUSALS_SCHEMA_VERSION = "native_score_batch_refusals.v2.0"

#: The checker's own numeric field groups, reused by name -- see
#: ``checks/phase4_real._compare_numeric_outputs``, whose dimension names and
#: field tuples these are.  A dimension outside this map is refused, never
#: compared against an empty view.
_DIMENSION_FIELDS: dict[str, tuple[str, ...]] = {
    "forecasts": FORECAST_FIELDS,
    "simulation": SIMULATION_FIELDS,
    "financial_diagnostics": FINANCIAL_FIELDS,
    "verdicts": GATE_FIELDS,
    "analogs": ANALOG_FIELDS,
}

#: Every one of the checker's five numeric field groups.  ``forecasts`` and
#: ``financial_diagnostics`` carry the served numbers, so they are compared
#: too; the ``NEVER_RAN_DIMENSIONS`` groups (``analogs``, ``simulation``,
#: ``verdicts``) stay in, including their typed placeholder defaults, rather
#: than being silently excluded from the report.
PARITY_DIMENSIONS: tuple[str, ...] = tuple(sorted(_DIMENSION_FIELDS))
if not set(NEVER_RAN_DIMENSIONS) <= set(PARITY_DIMENSIONS):
    raise RuntimeError("PARITY_DIMENSIONS must cover every never-ran dimension")


def _dimension_fields(dimension: str) -> tuple[str, ...]:
    """The checker's field group for ``dimension``, or a typed refusal."""
    fields = _DIMENSION_FIELDS.get(dimension)
    if fields is None:
        raise fail("INVALID_REQUEST", "unknown native parity dimension",
                   details={"dimension": dimension,
                            "known": sorted(_DIMENSION_FIELDS)})
    return fields


def _dimension_view(row: Mapping[str, Any], dimension: str) -> dict[str, Any]:
    """One row's values for one dimension, read by the checker's field names."""
    return {name: row.get(name) for name in _dimension_fields(dimension)}


def _row_mismatches(key: str, legacy: Mapping[str, Any], native: Mapping[str, Any],
                    dimensions: tuple[str, ...], tolerance_policy: TolerancePolicy,
                    ) -> list[dict[str, Any]]:
    """Every dimension of one shared key that does not agree, with its receipt."""
    mismatches = []
    for dimension in dimensions:
        result = compare_dimension(
            _dimension_view(legacy, dimension), _dimension_view(native, dimension), dimension,
            tolerance_policy=tolerance_policy)
        if not result["agree"]:
            mismatches.append({
                "row_key": key,
                "dimension": dimension,
                "finding_fields": list(result["finding_fields"]),
                "receipt": result["receipt"],
                "values": result["values"],
            })
    return mismatches


def _refuse_empty_inputs(legacy_rows: Mapping[str, Any], native_rows: Mapping[str, Any],
                         dimensions: tuple[str, ...]) -> None:
    """Fail closed: an empty comparison must never report ``compared``."""
    for dimension in dimensions:
        _dimension_fields(dimension)
    if not dimensions:
        raise fail("VALIDATION_FAILED", "native parity report has no dimensions")
    if not legacy_rows:
        raise fail("VALIDATION_FAILED", "native parity report has no legacy rows")
    if not native_rows:
        raise fail("VALIDATION_FAILED", "native parity report has no native rows")


def _population_key_from_board_request_key(key: str) -> str:
    """Project one ``records.json``/``refusals.json`` canonical key (the
    4-field ``f"{ticker}|{strategy}|{event_date_iso}|{session}"`` string
    ``native_score_batch._board_request_key`` builds) down to the 3-field
    ``population_key`` format :func:`engine.v2.ops.decision_validation.
    population_key` already uses for legacy rows.

    Splits on ``"|"`` into exactly 4 parts and reuses ``population_key``'s
    own join format for the first three (never string-concatenating a
    fourth time), so the two sides can never silently drift onto two
    different separators or field orders. ``session`` (the 4th part) is
    validated as present but never folded into the key -- legacy rows carry
    no ``session`` field to join against.

    Raises ``engine.v2.ops.errors.fail("VALIDATION_FAILED", ...)`` (an
    ``OpsError``) if ``key`` does not split into exactly 4 parts.
    """
    parts = key.split("|")
    if len(parts) != 4:
        raise fail("VALIDATION_FAILED",
                   "native canonical key does not have exactly 4 parts",
                   details={"key": key, "parts": len(parts)})
    ticker, strategy, event_date, _session = parts
    return population_key({"ticker": ticker, "strategy": strategy, "event_date": event_date})


def _native_comparison_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one ``records.json`` entry into the checker's per-dimension
    field shape.

    ``record`` is ``native_score_batch``'s ``to_document(ScoreRecord)``
    value, which PRESERVES :class:`~engine.v2.contracts.ScoreRecord`'s own
    nested per-dimension dicts (``forecasts``, ``uncertainty``,
    ``resolved_request``, ``financial_diagnostics``, ``gate_terms``), while
    :func:`_dimension_view` looks every field up at the row's TOP level.
    This projection reads each dimension's fields from the same nested
    sources ``checks/phase4_real._numeric_views`` reads them from on the
    native side -- ``forecasts`` first, then ``uncertainty``/
    ``resolved_request``, in its exact fallback order -- so the nightly
    report's view of a record can never drift from the checker's. It is a
    DATA-SHAPE projection only: every comparison decision still comes from
    ``compare_dimension``.

    The never-ran groups follow
    ``engine.v2.serving.native_render.native_display_row``'s convention: a
    field the record does not carry reads as ``None`` here, and the
    absent-vs-zero distinction for ``n_analogs`` is preserved (a record that
    carries no ``n_analogs`` key never invents ``0``; memory
    ``n-analogs-int-default-mismatch``).  ``engine.v2.serving`` is a layer
    ABOVE ``engine.v2.ops`` and can never be imported here, so this narrow
    projection is local to this module -- exactly as
    :func:`_population_key_from_board_request_key` re-derives its own
    serving-layer concept.
    """
    forecasts = record.get("forecasts") or {}
    uncertainty = record.get("uncertainty") or {}
    resolved = record.get("resolved_request") or {}
    financial = record.get("financial_diagnostics") or {}
    gate = record.get("gate_terms") or {}
    row: dict[str, Any] = {}
    for name in FORECAST_FIELDS:
        row[name] = forecasts.get(name, uncertainty.get(name, resolved.get(name)))
    for name in SIMULATION_FIELDS:
        row[name] = forecasts.get(name, resolved.get(name))
    for name in FINANCIAL_FIELDS:
        row[name] = financial.get(name)
    for name in GATE_FIELDS:
        row[name] = gate.get(name)
    for name in ANALOG_FIELDS:
        row[name] = resolved.get(name)
    return row


def _native_rows_and_refusals(
    records_document: Mapping[str, Any],
    refusals_document: Mapping[str, Any],
) -> tuple[dict[str, dict], dict[str, str], tuple[Mapping[str, Any], ...]]:
    """Project ``native_score_batch``'s v2.0 ``records.json``/
    ``refusals.json`` canonical keys down to ``population_key``, returning
    ``(native_rows, native_refusals, unkeyable_refusals)``.

    ``records_document["records"]`` and ``refusals_document["refusals"]``
    are both ``{canonical_key: value}`` objects (the v2.0 shape). Every key
    from BOTH, together, is projected through
    :func:`_population_key_from_board_request_key`; the projection is LOSSY
    (it drops ``session``), so two DISTINCT canonical keys -- the same
    ``(ticker, strategy, event_date)`` under two different ``session``
    values -- can collide onto the SAME ``population_key``. Any
    ``population_key`` produced by more than one distinct canonical key
    raises ``fail("VALIDATION_FAILED", ...)`` for the WHOLE call, before
    ``native_rows``/``native_refusals`` are built -- never a silent
    last-write-wins overwrite.

    ``refusals_document["unkeyable_refusals"]`` is accessed with `[...]`,
    never ``.get(..., ())``: this function's only caller is only ever
    reached for a document already confirmed ``v2.0``-shaped (see
    ARCHITECTURE.md), and the v2.0 writer always emits this key, even as
    ``[]`` for a batch with none -- its absence means the file is
    malformed, and the resulting ``KeyError`` is meant to propagate and be
    caught by the caller alongside its other decode failures, never
    silently treated as "no unkeyable refusals." Its entries are returned
    unchanged: an ``INVALID_KEY_FIELD`` row never had a ``population_key``
    to compute, so there is nothing here to project or collision-check for
    it.

    ``native_rows`` maps ``population_key -> the record's own document``
    (the ``records_document["records"]`` value, untouched). ``native_refusals``
    maps ``population_key -> refusal code string`` (``refusals_document
    ["refusals"][canonical_key]["code"]``).
    """
    unkeyable_refusals = refusals_document["unkeyable_refusals"]
    projected: dict[str, str] = {}

    def _project(source_key: str) -> str:
        population = _population_key_from_board_request_key(source_key)
        prior = projected.get(population)
        if prior is not None and prior != source_key:
            raise fail(
                "VALIDATION_FAILED",
                "native population key collision after projection",
                details={"population_key": population,
                         "keys": sorted([source_key, prior])})
        projected[population] = source_key
        return population

    native_rows: dict[str, dict] = {}
    for source_key, record in records_document["records"].items():
        native_rows[_project(source_key)] = record

    native_refusals: dict[str, str] = {}
    for source_key, refusal in refusals_document["refusals"].items():
        native_refusals[_project(source_key)] = refusal["code"]

    return native_rows, native_refusals, tuple(unkeyable_refusals)


def compare_native_vs_legacy(
    legacy_rows: dict[str, dict],
    native_rows: dict[str, dict],
    dimensions: tuple[str, ...],
    *,
    tolerance_policy: TolerancePolicy = SCORE_RECORD_V1,
) -> dict:
    """Classify every row key against the checker's per-dimension comparator.

    Every key present in EITHER side is visited, never only the intersection:
    a key present on only one side is itself a finding (``only_legacy``/
    ``only_native``), never silently skipped.  A shared key is ``compared``,
    and each ``dimensions`` entry that does not agree becomes one entry in
    ``mismatches`` carrying the checker's own finding fields and receipt.
    ``dimensions`` entries outside the checker's numeric field groups are
    refused up front with ``INVALID_REQUEST``.  Empty ``dimensions``, an empty side, or no shared key at all is refused with ``VALIDATION_FAILED``: a report that compared nothing never claims ``"compared"``.

    ``tolerance_policy`` is the ONE config policy every numeric dimension's
    comparison reads (never a per-dimension or per-field override threaded
    in some other way): it defaults to ``engine.v2.parity.tolerance.SCORE_RECORD_V1``,
    which declares zero rules, so the default is exact for every field and
    this parameter's addition changes no existing caller's behavior. A
    caller that has a user-approved, per-field ``TolerancePolicy`` for the
    native-vs-legacy comparison specifically (never for the Phase 4
    checker's own tier-0-style exact check, which does not pass this
    argument) supplies it here; this module invents no field's tolerance
    value itself.

    G2: this function classifies and returns; it has no path that writes a
    native value into a legacy row or the reverse, and a mismatch never raises.
    """
    _refuse_empty_inputs(legacy_rows, native_rows, dimensions)
    compared: list[str] = []
    only_legacy: list[str] = []
    only_native: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for key in sorted(set(legacy_rows) | set(native_rows)):
        legacy = legacy_rows.get(key)
        native = native_rows.get(key)
        if legacy is None:
            only_native.append(key)
        elif native is None:
            only_legacy.append(key)
        else:
            compared.append(key)
            mismatches.extend(_row_mismatches(key, legacy, native, dimensions, tolerance_policy))
    if not compared:
        raise fail("VALIDATION_FAILED", "native parity report shares no row key",
                   details={"only_legacy": len(only_legacy), "only_native": len(only_native)})
    return {
        "schema_version": SCHEMA_VERSION,
        "tolerance_policy_id": tolerance_policy.policy_id,
        "compared": compared,
        "only_legacy": only_legacy,
        "only_native": only_native,
        "mismatches": mismatches,
    }


def _empty_native_report(
    legacy_rows: Mapping[str, dict],
    native_rows: Mapping[str, dict],
    dimensions: tuple[str, ...],
    tolerance_policy: TolerancePolicy,
) -> dict:
    """The report ``compare_native_vs_legacy`` would return for a
    would-be comparison that shares no key, built WITHOUT calling
    ``compare_native_vs_legacy`` (there is no numeric comparison to make:
    nothing shared was scored against anything). Every key of
    ``legacy_rows`` is ``only_legacy`` and every key of ``native_rows`` is
    ``only_native`` -- by construction, on the path this function is used
    for, the two sets share no key.

    ``dimensions`` is accepted for signature symmetry with
    ``compare_native_vs_legacy`` (a caller can pass the same arguments to
    either) and is not otherwise used: no dimension comparison happens
    when nothing is shared.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "tolerance_policy_id": tolerance_policy.policy_id,
        "compared": [],
        "only_legacy": sorted(legacy_rows),
        "only_native": sorted(native_rows),
        "mismatches": [],
    }


def apply_native_refusals(
    report: dict,
    native_refusals: Mapping[str, str],
    unkeyable_refusals: tuple[Mapping[str, Any], ...] = (),
) -> dict:
    """Move refused native rows out of ``report["only_legacy"]`` into two
    new, additive report fields -- never mutates ``report`` in place;
    returns a new dict.

    ``native_refusals`` maps a ``population_key`` string to a refusal code
    string. Any key that is also present in ``report["only_legacy"]``
    moves there (removed from ``only_legacy``, appended to a new
    ``"native_refused"`` list as ``{"row_key": key, "refusal_code": code}``).
    A ``native_refusals`` key that is NOT in ``only_legacy`` is appended
    instead to a new ``"native_refused_unmatched"`` list, same
    ``{"row_key": key, "refusal_code": code}`` shape -- a native refusal
    with no legacy counterpart to explain (a row native's own board
    universe found and refused that legacy's ``score.json`` never carried).
    Iterates ``native_refusals`` in ``sorted()`` key order, so the output
    order is deterministic regardless of the input mapping's own order.

    Every entry of ``unkeyable_refusals`` (each already shaped
    ``{"key": {...raw ticker/strategy/event_date/session fields...},
    "code": ..., "detail": ...}``) is appended unconditionally to the SAME
    ``"native_refused_unmatched"`` list, in the given order, as
    ``{"row_key": entry["key"], "refusal_code": entry["code"]}`` -- using
    its raw structured key instead of a population-key string, since an
    unkeyable refusal has no ``population_key`` to project.

    Every native refusal -- keyed or not, legacy-matched or not -- lands
    in exactly one of ``native_refused``/``native_refused_unmatched``;
    none is ever silently dropped.
    """
    only_legacy = list(report["only_legacy"])
    only_legacy_set = set(only_legacy)
    native_refused: list[dict[str, Any]] = []
    native_refused_unmatched: list[dict[str, Any]] = []
    for key in sorted(native_refusals):
        code = native_refusals[key]
        if key in only_legacy_set:
            only_legacy.remove(key)
            native_refused.append({"row_key": key, "refusal_code": code})
        else:
            native_refused_unmatched.append({"row_key": key, "refusal_code": code})
    for entry in unkeyable_refusals:
        native_refused_unmatched.append(
            {"row_key": entry["key"], "refusal_code": entry["code"]})
    updated = dict(report)
    updated["only_legacy"] = only_legacy
    updated["native_refused"] = native_refused
    updated["native_refused_unmatched"] = native_refused_unmatched
    return updated


def _stamp_report_identity(report: dict, *, as_of: str | None, clock: Clock) -> dict:
    """Add this report's run identity -- never set inside
    compare_native_vs_legacy/_empty_native_report, which stay pure and
    wall-clock-free (R6, engine/v2/parity/ARCHITECTURE.md). ``as_of`` is the
    nightly session/as-of date this comparison was produced for (``None``
    when the caller has none to give -- never invented). ``generated_at`` is
    always this call's own wall-clock write time. Returns a new dict; never
    mutates ``report`` in place, matching every other function in this
    module.
    """
    updated = dict(report)
    updated["as_of"] = as_of
    updated["generated_at"] = format_timestamp(clock.now())
    return updated


def write_parity_report(report: dict, path: Path | str) -> Path:
    """Write ``report`` as deterministic JSON; any filesystem error propagates.

    The caller names the path explicitly (the nightly plan's private root,
    alongside ``run_shadow_nightly``'s own ``receipt_path`` convention) --
    never ``ledger/`` and never the legacy board's output directory.
    """
    path = Path(path)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return path


def _as_of_from_expected_ids(expected_ids) -> str | None:
    """The run/session as-of date one opaque expected-id carries, or ``None``.

    Only an id that actually contains the ``|`` separator yields a real date;
    an opaque id with no separator stays ``None`` rather than masquerading as
    one. Never raises.
    """
    if not expected_ids:
        return None
    candidate, separator, _scope_hash = expected_ids[0].partition("|")
    return candidate if separator else None


def run_native_parity_worker(parameters: Mapping[str, Any], root: Path, *,
                             clock: Clock = SystemClock()) -> dict[str, Any]:
    """The ``native_parity`` job kind's worker entrypoint.

    Reads three job-bound inputs already staged into ``root`` by the
    generic input-binding mechanism: ``score.json`` (the paired legacy
    "score" job's output), ``records.json``/``refusals.json`` (the paired
    ``native_score_batch`` job's v2.0 outputs). Both native documents must
    declare exactly the ``schema_version`` tags
    ``native_score_batch._native_score_batch_documents`` writes -- anything
    else is refused with ``VALIDATION_FAILED`` before a single row is read,
    so the generic job-submission API can never route a pre-v2.0 pair past
    this worker. Builds ``legacy_rows`` via
    :func:`engine.v2.ops.nightly.legacy_parity_rows` and
    ``native_rows``/``native_refusals``/``unkeyable_refusals`` via
    :func:`_native_rows_and_refusals`, projects every native record through
    :func:`_native_comparison_row` (its nested per-dimension dicts flattened
    to the shape ``_dimension_view`` reads), then classifies every row via
    :func:`compare_native_vs_legacy` -- or, when nothing shared but a
    refusal explains why, :func:`_empty_native_report` -- and layers
    :func:`apply_native_refusals` on top before writing
    ``native_parity_report.json``, additively stamped with this run's
    ``as_of``/``generated_at`` by :func:`_stamp_report_identity`. See
    ARCHITECTURE.md's "Cutover PR-4 (redo)" section for the rationale.
    """
    from engine.v2.ops.nightly import legacy_parity_rows

    expected_ids = parameters["expected_ids"]
    as_of = _as_of_from_expected_ids(expected_ids)
    score_document = json.loads((root / "score.json").read_text())
    records_document = json.loads((root / "records.json").read_text())
    refusals_document = json.loads((root / "refusals.json").read_text())
    if not isinstance(records_document, dict):
        raise fail("VALIDATION_FAILED",
                   "native_score_batch records.json is not a JSON mapping",
                   details={"schema_version": None})
    if records_document.get("schema_version") != _RECORDS_SCHEMA_VERSION:
        raise fail("VALIDATION_FAILED",
                   "native_score_batch records.json has an unsupported schema_version",
                   details={"schema_version": records_document.get("schema_version")})
    if not isinstance(refusals_document, dict):
        raise fail("VALIDATION_FAILED",
                   "native_score_batch refusals.json is not a JSON mapping",
                   details={"schema_version": None})
    if refusals_document.get("schema_version") != _REFUSALS_SCHEMA_VERSION:
        raise fail("VALIDATION_FAILED",
                   "native_score_batch refusals.json has an unsupported schema_version",
                   details={"schema_version": refusals_document.get("schema_version")})
    legacy_rows = legacy_parity_rows(score_document)
    native_rows, native_refusals, unkeyable_refusals = _native_rows_and_refusals(
        records_document, refusals_document)
    native_rows = {key: _native_comparison_row(record)
                   for key, record in native_rows.items()}
    shared = set(legacy_rows) & set(native_rows)
    if legacy_rows and not shared:
        fully_refused = set(legacy_rows) <= set(native_refusals)
        nothing_keyable_at_all = (
            not native_rows and not native_refusals and bool(unkeyable_refusals))
        refusal_explains_absence = fully_refused or nothing_keyable_at_all
    else:
        refusal_explains_absence = False
    if refusal_explains_absence:
        report = _empty_native_report(
            legacy_rows, native_rows, PARITY_DIMENSIONS, SCORE_RECORD_V1)
    else:
        report = compare_native_vs_legacy(
            legacy_rows, native_rows, PARITY_DIMENSIONS,
            tolerance_policy=SCORE_RECORD_V1)
    report = apply_native_refusals(report, native_refusals, unkeyable_refusals)
    report = _stamp_report_identity(report, as_of=as_of, clock=clock)
    (root / "native_parity_report.json").write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")))
    return {
        "outputs": [
            {"name": "report", "path": "native_parity_report.json",
             "schema": SCHEMA_VERSION},
        ],
        "completed_ids": list(parameters["expected_ids"]),
        "no_work": not parameters["expected_ids"],
    }


def native_parity_handler(
    plan: Mapping[str, Any],
    *,
    legacy_rows: dict[str, dict],
    native_rows: dict[str, dict],
    report_path: Path | str,
    dimensions: tuple[str, ...] = PARITY_DIMENSIONS,
    tolerance_policy: TolerancePolicy = SCORE_RECORD_V1,
    clock: Clock = SystemClock(),
) -> Callable[[dict], dict]:
    """Build the ``nightly.GRAPH`` handler for the ``native_parity`` stage.

    The plan's explicit ``shadow_serving_scorer`` (G5) decides whether there
    is anything to compare: ``"native"`` compares, writes the report and
    returns ``{"status": "compared"}``; ``"legacy"`` has no native rows and
    returns ``{"status": "not_applicable"}`` without writing.  ``compare``/
    ``write`` failures propagate: ``nightly._run_stage`` already degrades an
    OPTIONAL stage rather than blocking the board.

    ``tolerance_policy`` defaults to ``SCORE_RECORD_V1`` (exact for every
    field, unchanged from before this parameter existed) and is passed
    straight through to :func:`compare_native_vs_legacy` -- see that
    function's docstring for what a caller may plug in here.

    The written report is stamped additively with this run's identity --
    ``as_of`` from the running stage accumulator's ``"session"`` (``None``
    when absent) and ``clock``'s write time as ``generated_at`` -- by
    :func:`_stamp_report_identity`.
    """
    path = Path(report_path)

    def handler(value: dict) -> dict:
        if native_shadow_serving_mode(plan) != "native":
            return {**value, "native_parity": {"status": "not_applicable"}}
        report = compare_native_vs_legacy(
            legacy_rows, native_rows, dimensions, tolerance_policy=tolerance_policy)
        report = apply_native_refusals(report, {}, ())
        report = _stamp_report_identity(report, as_of=value.get("session"), clock=clock)
        write_parity_report(report, path)
        return {**value, "native_parity": {
            "status": "compared",
            "report_path": str(path),
            "compared": len(report["compared"]),
            "only_legacy": len(report["only_legacy"]),
            "only_native": len(report["only_native"]),
            "mismatches": len(report["mismatches"]),
        }}

    return handler
