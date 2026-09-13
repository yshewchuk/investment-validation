"""Snapshot-to-legacy materialization planning — phase-2 guide §9.1, §9.2, D13/D14.

Pure logic only: this module never opens a file, a database connection or a
legacy ``engine.*`` symbol (Layer 1, ``system_rearchitecture.md`` §4.1). It
imports only ``engine.v2.contracts``, ``engine.v2.foundation``, and this
package's own ``repository`` (for its ``table_contract``/``explain_dependencies``
public reads). ``engine/v2/data/legacy_adapter.py`` — the package's one
legacy-importing module — imports *from* this module, never the other way,
so there is no import cycle between the two.

:data:`LEGACY_SCORE_READ_PLAN_V1` is this task's central judgement call (task
brief decision 1): which curated tables, in what scope, plus which model/
calendar files, a legacy score batch (``engine.features.FeatureContext.load``
+ ``engine.score.Scorer.__init__``/``.score``/``score_calendar``) actually
reads. Derived by grepping every ``iter_table``/``read_table``/``_read_part``/
``paths.<CONST>`` call site reachable from those four entry points (review
round 2 redid this grep from scratch after round 1 missed a call site — see
below), not guessed:

**Tables** (call sites, in the order this module found them):

* ``earnings_events`` — WHOLE, unconditionally. ``engine.features.
  _session_index``, ``._next_event``, ``.Scorer._crush_frame``'s fallback, and
  ``engine.score.score_calendar`` itself all call
  ``store.read_table("earnings_events", ...)`` with no ``years``/``tickers``
  bound.
* ``trades`` — WHOLE, unconditionally. ``engine.score._load_trades_without_legs``
  streams ``store.iter_table("trades", ...)`` with no bound; this is exactly
  "the broader historical analog population required by current behavior"
  the guide's §9.2 names, read at ``Scorer.__init__`` before any board scores.
* ``daily_market`` — evidence ticker/year scoped, matching
  ``FeatureContext.load(tickers, years=years)``'s own bound (``engine/
  features.py`` ``FeatureContext.load``, ``daily_state_frame``). A SECOND site,
  ``Scorer._entry_implied_move`` (chunked by ``trades``'s own ticker/year
  span, unconditionally exercised at construction), needs a still-broader
  span this module cannot derive without reading ``trades`` first — the
  caller's ``evidence_scope`` is trusted to already cover it (memory note:
  "the evidence scope must be the score job's FULL ticker set and year
  range"); ``build_materialization_request``/``materialize_tree`` now PROVE
  this rather than trust it (decision 4 below), refusing when it does not
  hold.
* ``option_chains`` — evidence ticker/year scoped, matching
  ``load_chain_index``'s own bound. **Missed in review round 1.**
  ``engine.replay.load_chain_index(keys, years=None, ...)`` reads
  ``store.iter_table("option_chains", years=years, columns=list(_CHAIN_COLUMNS))``
  (``years`` derived from ``keys`` when not given; ticker filtering happens
  in pandas AFTER the read, so the store-level read is years-only — same
  shape as ``daily_market``'s own real behavior). Two call sites reach it:
  ``engine.score.score_calendar`` pre-loads one index for the whole board
  (``keys`` = every strategy's ``plan_events(...).chain_keys``, i.e. the
  board's own tickers) and passes it into every ``Scorer.score(request,
  chain_index=index)`` call; ``Scorer._price_entry`` (inside ``.score``) loads
  its OWN index on demand whenever a caller passes ``chain_index=None``
  instead — the coordinator's "score_calendar(alt_strikes=0) path" is this
  pre-load. Columns: ``engine.replay._CHAIN_COLUMNS`` (11 of 21 contract
  columns: ticker, obs_date, expiry, dte, strike, right, bid, ask, delta,
  spot, quote_repaired) — a provable subset, plus ``"year"`` (not read by the
  loader, but required for this module's own year-partitioned write and for
  legacy ``coerce()`` to accept the file back, since ``year`` is a
  non-nullable column).
* ``feature_panel``/``tier4_forecasts`` — WHOLE FILE, unconditionally
  (``engine.features.load_panel()``, ``engine.data.features.tier4.
  load_forecasts()`` both take no scope argument; the latter is read from
  ``tier4.serving_model``/``._pool_before``, unconditionally reachable from
  ``Scorer.score()``).
* ``securities``/``option_daily`` — RE-VERIFIED EXCLUDED (review round 2): an
  explicit grep for the literal strings ``"securities"``/``"option_daily"``
  across every ``.py`` file under ``engine/`` (excluding ``engine/v2``) finds
  zero occurrences as a table-name argument. Neither is read by
  ``FeatureContext.load``, ``Scorer.__init__``, ``Scorer.score``,
  ``score_calendar``, or anything they call.

**Conditional/optional reads found, NOT added to the plan (documented, not
silently dropped):**

* ``engine.data.features.tier4``'s ``im_t1_feature_model``/
  ``runup_move_feature_model``/``iv_crush_feature_model`` each carry a
  ``prepare(panel)`` closure that reads ``earnings_events``/``daily_market``
  over a FIXED historical window (``IM_T1_YEARS = range(2017, 2027)``,
  independent of ``evidence_scope``). This only executes when
  ``tier4.serving_model``'s joblib cache misses — a real possibility in a
  fresh private materialization, since this task does not pin any
  pre-existing ``data/models/tier4/*.joblib`` cache file. ``earnings_events``
  is already whole-table (covers any window); ``daily_market`` is not — a
  caller whose ``evidence_scope`` years exclude 2017–2026 under-covers this
  path for ``pred_im_t1_d14``/``pred_runup_abs_move_d14``/
  ``pred_iv_crush_30``. ``pred_abs_move`` (``size_feature_model``, the most
  common producer) is exempt: its ``prepare`` only touches the
  already-loaded ``panel`` DataFrame.
* ``engine.score.Scorer._chooser_analog_pool`` reads
  ``paths.FEATURES / "chooser_analog_pool.parquet"`` for ``DYNAMIC_MENU``
  strategies, wrapped in a bare ``try/except Exception: return None`` — a
  missing file silently degrades the chooser features to NaN rather than
  raising. Pinned anyway, alongside ``structures.json``, for the same
  "exact parity over silent degradation" reasoning (see the pinned-ref list
  below) rather than left optional.

**Registry/model/calendar inputs** are plain pinned files, never
Repository-scanned tables — not expressible as a bounded ``DataQuery``, so
they travel as ``"path::content_hash"`` pinned refs instead (judgement call
2 below), each now verified to actually resolve in the store before the
request is built (decision 5, review round 2):
``engine/models/registry.json`` (``engine.models.registry.REGISTRY_PATH``,
read unconditionally by ``Scorer.__init__``'s ``load_registry()``),
``engine/models/structures.json`` (``engine.structure_registry.
CHAMPIONS_PATH`` — optional in legacy code, pinned here for exact
structure-champion parity), ``data/features/chooser_analog_pool.parquet``
(optional in legacy code, pinned for the same reason), each champion's
joblib artifact (``RegistryEntry.artifact``/``.artifact_sha256``, under
``data/models/...``), and ``data/raw/polygon/gspc_daily.csv``
(``engine.paths.GSPC_DAILY``, ``engine.calendar.trading_calendar``'s sole
input).

No STOP finding. Every read this audit found is either a boundable
``DataQuery``, a nameable pinned file, or a documented conditional/optional
caveat above. The ``trades``-span-vs-``evidence_scope`` caveat is now a
PROVEN refusal (decision 4), not a trusted assumption.
"""
from __future__ import annotations

import dataclasses
import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from engine.v2.contracts.data import (
    DataQuery,
    KeyPredicate,
    LegacyMaterializationRequest,
    SnapshotRef,
    TableContract,
    TimeInterval,
)
from engine.v2.foundation import CONTENT_HASH_PREFIX, content_hash, to_document

from . import errors, time_formats
from . import query as query_mod

__all__ = [
    "LEGACY_LAYOUT_VERSION",
    "LEGACY_SCORE_READ_PLAN_V1",
    "SCORE_READ_PLAN_TABLES",
    "TABLE_OUTPUT_KIND",
    "MaterializedTree",
    "assert_rows_match",
    "build_materialization_request",
    "evidence_scope_covers_trades",
    "format_pinned_ref",
    "lock_down",
    "materialize_tree",
    "narrow_query_to_year",
    "parse_pinned_ref",
    "read_plan_complete",
    "scanned_rows",
    "trades_span",
]

LEGACY_LAYOUT_VERSION = "legacy_curated_layout.v1"

#: Tables a legacy score batch actually reads, in the order they are written.
#: ``"whole_table"`` means the loader reads the table unconditionally, with no
#: ticker/year bound; ``"evidence_scoped"`` means it is bounded by the
#: request's ``evidence_scope`` tickers/years.
LEGACY_SCORE_READ_PLAN_V1: dict[str, object] = {
    "schema_version": "legacy_score_read_plan.v1",
    "tables": {
        "earnings_events": {
            "scope": "whole_table", "columns": "full", "output": "curated",
            "reason": "engine.features._session_index/_next_event, engine.score.score_calendar "
                      "all call store.read_table('earnings_events', ...) with no years/tickers bound",
        },
        "daily_market": {
            "scope": "evidence_scoped", "columns": "full", "output": "curated",
            "reason": "engine.features.FeatureContext.load(tickers, years=years) bounds its own read "
                      "to the caller's tickers/years; engine.score.Scorer._entry_implied_move's own "
                      "second read site needs the broader trades-ticker span — PROVEN inside "
                      "evidence_scope by evidence_scope_covers_trades(), not trusted",
        },
        "trades": {
            "scope": "whole_table", "columns": "full", "output": "curated",
            "reason": "engine.score._load_trades_without_legs streams store.iter_table('trades', ...) "
                      "with no years/tickers bound; this IS the broader historical analog population "
                      "the guide's §9.2 names, read unconditionally at Scorer.__init__",
        },
        "option_chains": {
            "scope": "evidence_scoped",
            "columns": ("ticker", "obs_date", "year", "expiry", "dte", "strike", "right", "bid", "ask",
                       "delta", "spot", "quote_repaired"),
            "output": "curated",
            "reason": "engine.replay.load_chain_index(keys, years=None, ...) reads "
                      "store.iter_table('option_chains', years=years, columns=_CHAIN_COLUMNS); "
                      "engine.score.score_calendar pre-loads one index for the whole board and passes "
                      "it to every Scorer.score(chain_index=...) call (missed in review round 1). "
                      "columns = _CHAIN_COLUMNS plus 'year' (not read by the loader, but required for "
                      "this module's own year partitioning and for legacy coerce() to accept the file)",
        },
        "feature_panel": {
            "scope": "whole_table", "columns": "full", "output": "single_file",
            "reason": "engine.features.load_panel() takes no scope argument",
        },
        "tier4_forecasts": {
            "scope": "whole_table", "columns": "full", "output": "single_file",
            "reason": "engine.data.features.tier4.load_forecasts() takes no scope argument; read by "
                      "serving_model/_pool_before, unconditionally reachable from score()",
        },
    },
    "excluded_tables": ("securities", "option_daily"),
    "excluded_reason": "re-verified review round 2: zero occurrences of either literal table name as a "
                        "store.read_table/iter_table argument anywhere under engine/ (excluding v2)",
    "registry_and_model_refs": {
        "engine/models/registry.json": "engine.models.registry.REGISTRY_PATH, load_registry() default",
        "engine/models/structures.json": "engine.structure_registry.CHAMPIONS_PATH (optional in legacy "
                                          "code; pinned here for exact structure-champion parity)",
        "data/features/chooser_analog_pool.parquet": "engine.score.Scorer._chooser_analog_pool "
                                          "(optional in legacy code, silently degrades chooser features "
                                          "if absent; pinned here for the same parity reasoning)",
        "data/models/...": "each champion's joblib artifact named by RegistryEntry.artifact/"
                            ".artifact_sha256 inside registry.json",
    },
    "calendar_refs": {
        "data/raw/polygon/gspc_daily.csv": "engine.paths.GSPC_DAILY, engine.calendar.trading_calendar's "
                                            "sole input",
    },
}

SCORE_READ_PLAN_TABLES: tuple[str, ...] = tuple(LEGACY_SCORE_READ_PLAN_V1["tables"])
_EVIDENCE_SCOPED_TABLES = frozenset(
    name for name, spec in LEGACY_SCORE_READ_PLAN_V1["tables"].items() if spec["scope"] == "evidence_scoped")
#: table_name -> legacy output shape: "curated" (year-partitioned) or
#: "single_file" (one file, no year partitioning) — ``legacy_adapter.materialize``'s
#: own routing, kept here so it is derived from the plan rather than restated.
TABLE_OUTPUT_KIND: dict[str, str] = {
    name: spec["output"] for name, spec in LEGACY_SCORE_READ_PLAN_V1["tables"].items()}

_PLACEHOLDER_HASH = CONTENT_HASH_PREFIX + "0" * 64
#: Judgement call 2 (task brief): a "pinned ref" for a plain file that is not
#: a Repository-tracked dataset (a registry/model/calendar file) is a single
#: string combining its legacy-relative destination path and its content
#: hash — ``LegacyMaterializationRequest.registry_and_model_refs``/
#: ``.calendar_refs`` are ``tuple[str, ...]`` in the contract, so this is the
#: one self-contained encoding that lets ``materialize`` resolve bytes from
#: the object store using only ``(repository, store, request, dest_root)``.
_REF_SEP = "::"


def format_pinned_ref(relative_path: str, ref_content_hash: str) -> str:
    """Build one ``registry_and_model_refs``/``calendar_refs`` entry."""
    if _REF_SEP in relative_path:
        raise ValueError(f"a legacy relative path may not contain {_REF_SEP!r}: {relative_path!r}")
    return f"{relative_path}{_REF_SEP}{ref_content_hash}"


def parse_pinned_ref(ref: str) -> tuple[str, str]:
    """``(relative_path, content_hash)`` from one pinned-ref string."""
    if _REF_SEP not in ref:
        raise ValueError(f"not a pinned ref (missing {_REF_SEP!r}): {ref!r}")
    path, _, ref_content_hash = ref.partition(_REF_SEP)
    if not path or not ref_content_hash:
        raise ValueError(f"pinned ref has an empty path or hash: {ref!r}")
    return path, ref_content_hash


def _next_representable(value: str) -> str:
    """The smallest value strictly after ``value``, in ``value``'s own
    encoding (review round 2, decision 3) — a naive timestamp advances by one
    microsecond (the format's own resolution, ``time_formats.
    NAIVE_TIMESTAMP_FORMAT``'s six-digit fraction); a bare date advances by
    one day."""
    if time_formats.is_naive_timestamp(value):
        parsed = datetime.strptime(value, time_formats.NAIVE_TIMESTAMP_FORMAT)
        return time_formats.format_naive_timestamp(parsed + timedelta(microseconds=1))
    return (date.fromisoformat(value) + timedelta(days=1)).isoformat()


def _whole_table_bounds(repository, snapshot_ref: SnapshotRef, table_name: str,
                        contract: TableContract) -> tuple[tuple[KeyPredicate, ...], TimeInterval | None]:
    """Decision 3 (review round 2): a whole-table read names an explicit
    interval derived from the PINNED MANIFEST's own fragment records — the
    minimum ``time_min`` and the maximum ``time_max`` across every fragment,
    end made exclusive via :func:`_next_representable` — never a sentinel
    made up out of thin air. A table with no ``observation_time_column`` uses
    a key predicate over the partition years the manifest actually carries
    instead (none of this plan's tables hit that branch today, but the rule
    is general)."""
    records = repository.fragment_records(snapshot_ref, table_name)
    if not records:
        raise errors.fail("CONTRACT_MISMATCH",
                  "a whole-table read needs at least one fragment to derive its bound",
                  details={"table_name": table_name})
    if not contract.observation_time_column:
        years = tuple(sorted({int(r.partition_key) for r in records}))
        return (KeyPredicate(column="year", operator="in", values=years),), None
    time_mins = [r.time_min for r in records if r.time_min is not None]
    time_maxs = [r.time_max for r in records if r.time_max is not None]
    if not time_mins or not time_maxs:
        raise errors.fail("CONTRACT_MISMATCH",
                  "a whole-table read's observation_time_column has no recorded fragment time bounds",
                  details={"table_name": table_name})
    interval = TimeInterval(column=contract.observation_time_column,
                            start_inclusive=min(time_mins), end_exclusive=_next_representable(max(time_maxs)))
    return (), interval


def _scope_bounds(repository, snapshot_ref: SnapshotRef, table_name: str,
                  contract: TableContract,
                  evidence_scope: dict) -> tuple[tuple[KeyPredicate, ...], TimeInterval | None]:
    if table_name not in _EVIDENCE_SCOPED_TABLES:
        return _whole_table_bounds(repository, snapshot_ref, table_name, contract)
    column = contract.observation_time_column
    tickers = sorted(set(evidence_scope["tickers"]))
    years = sorted(int(y) for y in evidence_scope["years"])
    key_filter = (KeyPredicate(column="ticker", operator="in", values=tuple(tickers)),)
    interval = TimeInterval(column=column, start_inclusive=f"{years[0]}-01-01",
                            end_exclusive=f"{years[-1] + 1}-01-01")
    return key_filter, interval


def _plan_columns(table_name: str, contract: TableContract) -> tuple[str, ...]:
    """The plan's declared column set for ``table_name`` — the full contract
    column list unless :data:`LEGACY_SCORE_READ_PLAN_V1` names a provable
    subset (``option_chains``: ``engine.replay._CHAIN_COLUMNS`` plus the
    ``year`` partition column this module's own write needs)."""
    spec_columns = LEGACY_SCORE_READ_PLAN_V1["tables"][table_name]["columns"]
    if spec_columns == "full":
        return tuple(c.name for c in contract.columns)
    declared = {c.name for c in contract.columns}
    unknown = sorted(set(spec_columns) - declared)
    if unknown:
        raise errors.fail("CONTRACT_MISMATCH", f"plan names undeclared column(s) {unknown}",
                  details={"table_name": table_name})
    return tuple(spec_columns)


def _build_table_query(repository, snapshot_ref: SnapshotRef, table_name: str,
                       evidence_scope: dict) -> DataQuery:
    contract = repository.table_contract(snapshot_ref, table_name)
    contract_ref = snapshot_ref.table_versions[table_name].table_contract_ref
    key_filter, time_interval = _scope_bounds(repository, snapshot_ref, table_name, contract,
                                              evidence_scope)
    columns = _plan_columns(table_name, contract)
    probe = DataQuery(
        snapshot_id=snapshot_ref.snapshot_id, table_contract_ref=contract_ref, columns=columns,
        key_filter=key_filter, time_interval=time_interval, order_by=contract.primary_key,
        max_batch_rows=contract.maximum_batch_rows, max_result_rows=contract.maximum_result_rows)
    plan = repository.explain_dependencies(probe, table_name=table_name)
    # Decision 2: max_result_rows is the pinned row count this exact scope
    # touches, derived from the resolved snapshot's own fragment records
    # (DependencyEntry.estimated_rows == FragmentRecord.row_count) — finite
    # and manifest-derived, never the table's generic cap. max_batch_rows is
    # capped to the same bound (never > max_result_rows, per §5.3's own rule).
    row_bound = max(1, sum(entry.estimated_rows for entry in plan.dependencies))
    max_result_rows = min(row_bound, contract.maximum_result_rows)
    max_batch_rows = min(contract.maximum_batch_rows, max_result_rows)
    return dataclasses.replace(probe, max_batch_rows=max_batch_rows, max_result_rows=max_result_rows)


def _verify_pinned_ref_exists(store, ref: str) -> None:
    """Decision 5 (review round 2): a pinned ref must actually resolve in
    ``store`` — with the exact recorded hash — before it goes into a request,
    never trusted on the strength of its own string alone."""
    _, ref_hash = parse_pinned_ref(ref)
    _resolve_pinned_bytes(store, ref_hash)


def build_materialization_request(
    repository, store, snapshot_ref: SnapshotRef, legacy_snapshot_object_ref,
    *, direct_scope: dict, evidence_scope: dict, registry_and_model_refs, calendar_refs,
    expected_population: dict,
) -> LegacyMaterializationRequest:
    """One ``DataQuery`` per :data:`SCORE_READ_PLAN_TABLES` entry, hashed as a
    Command (decision 2): ``request_hash`` covers everything except itself.

    ``repository``/``store`` are not in the brief's own abbreviated signature
    but are required here (recorded as a deviation in this task's report):
    computing a manifest-derived ``max_result_rows``/whole-table interval per
    table needs ``Repository.explain_dependencies``/``.fragment_records``,
    and verifying every pinned ref (decision 5) needs the object store — both
    only a live instance can answer. Publishing the registry/model/calendar
    bytes those refs name is NOT this function's job (see this task's
    report for who is expected to: a snapshot/reference-data import job).
    """
    for ref in (*registry_and_model_refs, *calendar_refs):
        _verify_pinned_ref_exists(store, ref)
    table_queries = {name: _build_table_query(repository, snapshot_ref, name, evidence_scope)
                     for name in SCORE_READ_PLAN_TABLES}
    placeholder = LegacyMaterializationRequest(
        request_hash=_PLACEHOLDER_HASH, snapshot_ref=snapshot_ref,
        legacy_snapshot_object_ref=legacy_snapshot_object_ref, direct_scope=dict(direct_scope),
        evidence_scope=dict(evidence_scope), table_queries=table_queries,
        registry_and_model_refs=tuple(registry_and_model_refs), calendar_refs=tuple(calendar_refs),
        legacy_layout_version=LEGACY_LAYOUT_VERSION, expected_population=dict(expected_population))
    doc = to_document(placeholder)
    del doc["request_hash"]
    return dataclasses.replace(placeholder, request_hash=content_hash(doc))


def _scope_superset(evidence_scope: dict, direct_scope: dict) -> bool:
    for key in ("tickers", "years"):
        if key not in direct_scope:
            continue
        if key not in evidence_scope:
            return False
        if not set(direct_scope[key]) <= set(evidence_scope[key]):
            return False
    return True


def trades_span(repository, request: LegacyMaterializationRequest) -> dict:
    """The ACTUAL ``(ticker, year)`` span ``trades`` carries under this
    request's own query (decision 4, review round 2) — scanned, never
    assumed. Projects ``ticker``/``year`` plus whatever the query's own
    ``time_interval``/``key_filter`` columns are (``repository.scan``'s row
    matching reads a predicate's column straight from the projected row, so
    dropping it out of ``columns`` would silently fail every row rather than
    matching it) — still a lean pass over the same rows ``materialize_tree``
    writes anyway, never the ``legs`` blob."""
    query = request.table_queries["trades"]
    predicate_columns = {p.column for p in query.key_filter}
    if query.time_interval is not None:
        predicate_columns.add(query.time_interval.column)
    columns = tuple(dict.fromkeys(("ticker", "year", *sorted(predicate_columns))))
    narrow = dataclasses.replace(query, columns=columns)
    tickers: set[str] = set()
    years: set[int] = set()
    for batch in repository.scan(narrow, table_name="trades"):
        for row in batch.to_pylist():
            tickers.add(row["ticker"])
            years.add(row["year"])
    return {"tickers": tickers, "years": years}


def evidence_scope_covers_trades(repository, request: LegacyMaterializationRequest) -> bool:
    """True iff ``trades``'s real span sits inside ``evidence_scope`` — the
    proof ``Scorer._entry_implied_move``'s own ``daily_market`` read (chunked
    by exactly this span) will not silently under-read."""
    span = trades_span(repository, request)
    evidence_tickers = set(request.evidence_scope.get("tickers", ()))
    evidence_years = {int(y) for y in request.evidence_scope.get("years", ())}
    return span["tickers"] <= evidence_tickers and span["years"] <= evidence_years


def read_plan_complete(request: LegacyMaterializationRequest, repository) -> bool:
    """Decision 4: True only if every plan entry has a query or ref, the
    evidence scope is a superset of the direct scope, and (review round 2)
    ``trades``'s real span sits inside the evidence scope too (guide §9.2's
    refusal: "if the adapter cannot prove its read plan complete, it refuses
    checkpoint reuse"). ``repository`` is required, unlike round 1's version,
    because that last proof needs an actual scan.
    """
    if set(request.table_queries) != set(SCORE_READ_PLAN_TABLES):
        return False
    if not request.registry_and_model_refs or not request.calendar_refs:
        return False
    if not _scope_superset(request.evidence_scope, request.direct_scope):
        return False
    return evidence_scope_covers_trades(repository, request)


# ==========================================================================
# materialize_tree — everything that touches no legacy symbol (§9.1)
# ==========================================================================
#
# ``legacy_adapter.materialize()`` is the only place a legacy reader is
# actually invoked (fan-out budget note in that module's docstring); every
# other piece of the write — dest_root safety, Parquet writing, content
# hashing, row comparison, lock-down — lives here instead, since none of it
# touches a legacy ``engine.*`` symbol.


@dataclasses.dataclass(frozen=True)
class MaterializedTree:
    """What :func:`materialize_tree` wrote, before validation or lock-down.

    ``curated_files``/``single_files`` name exactly the paths
    ``legacy_adapter.materialize`` must re-read with the legacy loaders;
    ``manifest`` is the full ``{relative_path: content_hash}`` this task's
    contract returns.
    """

    manifest: dict[str, str]
    curated_files: dict[str, dict[int, Path]]
    single_files: dict[str, Path]


def materialize_tree(repository, store, request: LegacyMaterializationRequest,
                     dest_root) -> MaterializedTree:
    """Write every declared path under ``dest_root``. Never chmods (decision
    3: lock-down is a separate, later step) and never opens a legacy
    ``engine.*`` symbol."""
    dest_root = Path(dest_root)
    _check_dest_root(store, dest_root)
    if not evidence_scope_covers_trades(repository, request):
        raise errors.fail("EVIDENCE_SCOPE_INCOMPLETE",
                  "trades's real (ticker, year) span is not inside evidence_scope; "
                  "Scorer._entry_implied_move's own daily_market read would under-cover it")
    manifest: dict[str, str] = {}
    curated_files: dict[str, dict[int, Path]] = {}
    single_files: dict[str, Path] = {}
    for table_name, query in request.table_queries.items():
        contract = repository.table_contract(request.snapshot_ref, table_name)
        if TABLE_OUTPUT_KIND[table_name] == "curated":
            year_paths = _write_curated_table(repository, query, table_name, dest_root)
            curated_files[table_name] = year_paths
            for path in year_paths.values():
                manifest[str(path.relative_to(dest_root))] = _file_content_hash(path)
        else:
            rel = _single_file_relative_path(table_name)
            path = dest_root / rel
            _write_single_file(repository, query, table_name, contract, path)
            single_files[table_name] = path
            manifest[rel] = _file_content_hash(path)
    snapshot_rel = "data/features/SNAPSHOT"
    snapshot_path = dest_root / snapshot_rel
    _write_pinned_bytes(store, request.legacy_snapshot_object_ref.content_hash, snapshot_path)
    manifest[snapshot_rel] = _file_content_hash(snapshot_path)
    for ref in (*request.registry_and_model_refs, *request.calendar_refs):
        rel, ref_hash = parse_pinned_ref(ref)
        path = dest_root / rel
        _write_pinned_bytes(store, ref_hash, path)
        manifest[rel] = _file_content_hash(path)
    return MaterializedTree(manifest=manifest, curated_files=curated_files, single_files=single_files)


def _single_file_relative_path(table_name: str) -> str:
    name = "panel.parquet" if table_name == "feature_panel" else "tier4_forecasts.parquet"
    return f"data/features/{name}"


# -- dest_root safety --------------------------------------------------------


def _check_dest_root(store, dest_root: Path) -> None:
    if dest_root.is_symlink():
        raise errors.fail("DEST_ROOT_UNSAFE", "materialization dest_root must not be a symlink")
    store_root = Path(store.root).resolve()
    resolved = dest_root.resolve() if dest_root.exists() else dest_root.absolute()
    if resolved == store_root or store_root in resolved.parents:
        raise errors.fail("DEST_ROOT_UNSAFE",
                  "materialization dest_root must not be inside the object store")
    if not dest_root.exists():
        dest_root.mkdir(parents=True)
        return
    if not dest_root.is_dir():
        raise errors.fail("DEST_ROOT_UNSAFE", "materialization dest_root must be a directory")
    if any(dest_root.iterdir()):
        raise errors.fail("DEST_ROOT_NOT_EMPTY", "materialization dest_root must be empty")


# -- curated (year-partitioned) tables ---------------------------------------


def _write_curated_table(repository, query: DataQuery, table_name: str,
                         dest_root: Path) -> dict[int, Path]:
    curated_root = dest_root / "data" / "curated" / table_name
    writers: dict[int, pq.ParquetWriter] = {}
    paths: dict[int, Path] = {}
    for batch in repository.scan(query, table_name=table_name):
        _split_batch_by_year(batch, curated_root, writers, paths)
    for writer in writers.values():
        writer.close()
    return paths


def _split_batch_by_year(batch: pa.RecordBatch, curated_root: Path,
                         writers: dict[int, pq.ParquetWriter], paths: dict[int, Path]) -> None:
    years = batch.column(batch.schema.get_field_index("year")).to_pylist()
    for year in sorted(set(years)):
        sub = batch.filter(pa.array([y == year for y in years]))
        writer = writers.get(year)
        if writer is None:
            part_dir = curated_root / f"year={year}"
            part_dir.mkdir(parents=True, exist_ok=True)
            path = part_dir / "part-0000.parquet"
            writer = pq.ParquetWriter(path, batch.schema)
            writers[year] = writer
            paths[year] = path
        writer.write_batch(sub)


# -- single-file tables (feature_panel / tier4_forecasts) --------------------


def _write_single_file(repository, query: DataQuery, table_name: str,
                       contract: TableContract, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for batch in repository.scan(query, table_name=table_name):
            if writer is None:
                writer = pq.ParquetWriter(dest_path, batch.schema)
            writer.write_batch(batch)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        _write_empty_table(query, contract, dest_path)


def _write_empty_table(query: DataQuery, contract: TableContract, dest_path: Path) -> None:
    physical = {c.name: c.physical_type for c in contract.columns}
    schema = pa.schema([(name, query_mod.arrow_type_for(physical[name])) for name in query.columns])
    empty = pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)
    pq.write_table(empty, dest_path)


# -- SNAPSHOT bytes and pinned registry/model/calendar refs ------------------


def _resolve_pinned_bytes(store, ref_hash: str) -> bytes:
    """Resolve ``ref_hash`` against ``store``'s own content-addressed object
    pool (the same ``objects/<hash[:2]>/<hash>`` layout
    ``engine.v2.foundation.ArtifactStore`` commits to), verifying it exists
    with exactly that hash. Shared by the build-time existence check
    (decision 5) and the actual write, so the two can never disagree on what
    "resolves" means."""
    digest = ref_hash.removeprefix(CONTENT_HASH_PREFIX)
    object_path = Path(store.root) / "objects" / digest[:2] / digest
    try:
        data = object_path.read_bytes()
    except OSError as exc:
        raise errors.fail("OBJECT_CORRUPT", "a pinned ref names no object in the store") from exc
    if CONTENT_HASH_PREFIX + hashlib.sha256(data).hexdigest() != ref_hash:
        raise errors.fail("OBJECT_CORRUPT", "a pinned ref's bytes do not match its recorded hash")
    return data


def _write_pinned_bytes(store, ref_hash: str, dest_path: Path) -> None:
    """Resolve ``ref_hash`` and copy it to ``dest_path``. Never a symlink/hard
    link — always a fresh byte copy."""
    data = _resolve_pinned_bytes(store, ref_hash)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(data)


# -- row comparison, hashing, lock-down --------------------------------------


def narrow_query_to_year(query: DataQuery, contract: TableContract, year: int) -> DataQuery:
    """``query`` restricted to one partition year — how validation re-scans
    exactly one already-written curated file without holding the whole table."""
    return dataclasses.replace(query, time_interval=TimeInterval(
        column=contract.observation_time_column, start_inclusive=f"{year}-01-01",
        end_exclusive=f"{year + 1}-01-01"))


def scanned_rows(repository, query: DataQuery, table_name: str) -> list[dict]:
    return [row for batch in repository.scan(query, table_name=table_name) for row in batch.to_pylist()]


def _normalize_value(value):
    if value is None:
        return None
    try:
        if value != value:  # NaN / NaT, the only values unequal to themselves
            return None
    except (TypeError, ValueError):
        pass
    to_pydatetime = getattr(value, "to_pydatetime", None)
    return to_pydatetime() if to_pydatetime is not None else value


def assert_rows_match(scanned: list[dict], legacy_frame, columns: tuple[str, ...]) -> None:
    """Row-for-row equality between a fresh Repository scan and a legacy
    ``_read_part`` frame of the same materialized file (D14: full-precision
    values, not an approximation)."""
    legacy_rows = legacy_frame.to_dict("records") if len(legacy_frame) else []
    if len(scanned) != len(legacy_rows):
        raise errors.fail("CONTRACT_MISMATCH",
                  f"materialized row count {len(legacy_rows)} != scanned row count {len(scanned)}")
    for scanned_row, legacy_row in zip(scanned, legacy_rows):
        for name in columns:
            if _normalize_value(scanned_row.get(name)) != _normalize_value(legacy_row.get(name)):
                raise errors.fail("CONTRACT_MISMATCH",
                          f"materialized value for column {name!r} disagrees with the scanned row")


def _file_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return CONTENT_HASH_PREFIX + digest.hexdigest()


def lock_down(dest_root) -> None:
    """chmod every file 0444 and every directory 0555 (decision 3), including
    ``dest_root`` itself."""
    dest_root = Path(dest_root)
    for path in dest_root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((p for p in dest_root.rglob("*") if p.is_dir()), reverse=True):
        path.chmod(0o555)
    dest_root.chmod(0o555)
