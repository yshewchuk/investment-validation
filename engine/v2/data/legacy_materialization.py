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
``paths.<CONST>`` call site reachable from those four entry points, redone
from scratch across three review rounds as each missed something the last
one didn't, not guessed:

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
* ``daily_market`` — WHOLE, unconditionally (review round 3, item 2 — was
  "evidence ticker/year scoped" through round 2; see this table's own
  ``"reason"`` entry in the dict below for the full derivation, including
  why option (a), pinning the serving-model artifacts to prevent a cache
  miss, is provably NOT viable here).
* ``option_chains`` — evidence ticker/year scoped, matching
  ``load_chain_index``'s own bound. **Missed in review round 1.**
  ``engine.replay.load_chain_index(keys, years=None, ...)`` reads
  ``store.iter_table("option_chains", years=years, columns=list(_CHAIN_COLUMNS))``
  (``years`` derived from ``keys`` when not given; ticker filtering happens
  in pandas AFTER the read, so the store-level read is years-only). Two call
  sites reach it: ``engine.score.score_calendar`` pre-loads one index for the
  whole board (``keys`` = every strategy's ``plan_events(...).chain_keys``,
  i.e. the board's own tickers) and passes it into every ``Scorer.score(
  request, chain_index=index)`` call; ``Scorer._price_entry`` (inside
  ``.score``) loads its OWN index on demand whenever a caller passes
  ``chain_index=None`` instead — the coordinator's "score_calendar(
  alt_strikes=0) path" is this pre-load. Columns:
  ``engine.replay._CHAIN_COLUMNS`` (11 of 21 contract columns: ticker,
  obs_date, expiry, dte, strike, right, bid, ask, delta, spot,
  quote_repaired) — a provable subset, plus ``"year"`` (not read by the
  loader, but required for this module's own year-partitioned write and for
  legacy ``coerce()`` to accept the file back, since ``year`` is a
  non-nullable column). No known cache-miss-style unconditional-widening
  finding applies to this table the way it does to ``daily_market``: nothing
  else reachable from the four entry points reads ``option_chains``.
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

**Review round 3, item 2 — why ``daily_market`` moved to whole_table.**
``engine.data.features.tier4.serving_model(fold_start, model=...)`` is called,
unconditionally reachable from ``Scorer.score()``, for whichever Tier-4
producer a given row needs (``engine.score.py`` picks ``produces`` — one of
``pred_abs_move``/``pred_im_t1_d14``/``pred_runup_abs_move_d14``/
``pred_iv_crush_30`` — per column, at runtime, so materialize() cannot know
in advance which will be exercised). On a joblib cache MISS it calls
``fit_fold`` -> ``training_frames(panel, model)`` -> ``model.prepare(panel)``:

* ``im_t1_feature_model``/``runup_move_feature_model``'s own ``prepare``
  closures read ``store.read_table("daily_market", years=range(min(
  IM_T1_YEARS) - 1, max(IM_T1_YEARS) + 1), columns=[...])`` where
  ``IM_T1_YEARS = range(2017, 2027)`` — a FIXED window, no ticker filter,
  independent of ``evidence_scope``.
* ``iv_crush_feature_model``'s ``prepare = engine.models.training.
  iv_crush.prepare`` calls ``crush_frame()`` with NO arguments, which reads
  ``store.read_table("daily_market", columns=[...])`` with NO years bound
  and NO ticker filter at all — worse than a fixed window, truly unbounded.

**Review round 4 revisits this.** Round 3 concluded the cache miss was
provably always reachable because ``materialize_tree`` could not predict
``store.file_sha256(paths.PANEL)`` before writing it (a pyarrow rewrite gives
the file new bytes, hence a new hash, every time). Decision 1 below removes
that premise for ``feature_panel`` specifically: it is now copied
byte-for-byte from its own immutable object, so its hash IS the source
object's ``content_hash`` — predictable before any byte is written, and
pinnable in a ``registry_and_model_refs`` ref by an exact filename (decision
2). That reopens the question decision 2 asks: does a cache HIT (now
reachable, since a pinned joblib's embedded hash can finally match) return
before ``prepare()``/``crush_frame()`` runs? **No — proven the other way,
one level deeper than round 3 looked.** Quoting ``engine/data/features/
tier4.py`` verbatim (line numbers as of this commit; confirmed unchanged
from the parent branch by a plain diff of that file)::

    # serving_model, the cache-HIT branch (tier4.py:1280-1292):
        if cache and path.exists():
            import joblib

            stored = joblib.load(path)
            if (
                stored.get("model_id") == model.model_id
                and pd.Timestamp(stored.get("fold_start")) == fold
                and stored.get("tier3_snapshot") == snapshot
                and tuple(stored.get("features", ())) == tuple(model.features)
            ):
                pool_pred, pool_res = _pool_before(
                    fold, model, load_panel() if panel is None else panel
                )

    # _pool_before, called on THAT SAME cache-hit branch above (tier4.py:1243-1246):
        earlier = stored[stored[point].notna() & (stored[fold_col] < pd.Timestamp(fold))]
        if earlier.empty:
            return empty
        _, trainable = training_frames(panel, model)   # <- calls model.prepare(panel)

``_pool_before`` — called UNCONDITIONALLY on the cache-hit path, before
``serving_model`` ever returns — reads back ``tier4_forecasts`` (``stored =
load_forecasts()``) and, whenever that table already has an earlier row for
this producer's column before the current fold (``earlier`` non-empty), it
calls ``training_frames(panel, model)`` itself — which calls
``model.prepare(panel)`` regardless of whether the SERVING MODEL's own
``estimator`` was a cache hit or miss. Since ``tier4_forecasts`` is
materialized as the real production forecasts table (whole file), and a
column that has ever been forecast historically will have SOME row before
almost any realistic fold, ``earlier`` is non-empty for realistic scoring in
practice — so ``iv_crush_feature_model``'s ``prepare`` (``crush_frame()``,
unbounded) and ``im_t1_feature_model``/``runup_move_feature_model``'s
``prepare`` (the fixed ``IM_T1_YEARS`` window) both run independent of
whether their OWN joblib cache hits. No pinned artifact can prevent this:
``training_frames``'s own memoization (``_PREPARED: dict[str, tuple[...]]``,
keyed by ``model.model_id`` alone, with the panel object itself stored
alongside so a hit also requires ``hit[0] is panel`` — tier4.py:541,553-554)
is an IN-PROCESS dict, not a file, so it cannot be pre-seeded by a pinned ref
either. Option (a) therefore still fails — not for
round 3's reason (an unpredictable hash), but because ``_pool_before``'s own
unconditional call reaches ``prepare()`` regardless of the serving-model
cache's hit/miss status. ``daily_market`` stays ``whole_table``: it is the
only read provably complete for ``iv_crush_feature_model`` (and
``im_t1``/``runup_move``) via this path. Decision 2's pinned cache refs are
still implemented and still valuable — they avoid ``fit_fold`` re-fitting a
DIFFERENT estimator than production's committed one (a real parity risk on
their own), just not the ``daily_market`` scope question.

**Registry/model/calendar inputs** are plain pinned files, never
Repository-scanned tables. They are not expressible as a bounded ``DataQuery``,
so they travel as ``"path::content_hash"`` pinned refs instead (judgement call
2 below). Each ref is verified to resolve in the store before the request is
built (decision 5, review round 2). Which files they are, and their legacy
paths, is ``engine.v2.data.reference_inputs.LEGACY_REFERENCE_INPUTS_V1``:
every path there comes from the legacy constant that defines it. The snapshot
import pins and publishes them per import receipt
(``engine.v2.data.reference_catalog``), and snapshot planning reads them back
from there.

No STOP finding. Every read this audit found is either a boundable
``DataQuery`` or a nameable pinned file. ``daily_market`` moving to
whole_table also makes ``Scorer._entry_implied_move``'s trades-ticker-span
need (round 2) and the tier4 cache-miss windows above (round 3) all
trivially satisfied by construction; ``evidence_scope_covers_trades()``
still runs and still proves it, now vacuously, rather than being removed —
a regression guard costs nothing to keep.
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
    "tier4_cache_refs_match_panel",
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
            "scope": "whole_table", "columns": "full", "output": "curated",
            "reason": "review round 3, item 2: engine.features.FeatureContext.load(tickers, "
                      "years=years) bounds ITS OWN read narrowly, but two other unconditionally "
                      "reachable read sites need far more than evidence_scope can promise, so this "
                      "table is whole_table rather than evidence_scoped. (1) "
                      "engine.data.features.tier4's im_t1_feature_model/runup_move_feature_model "
                      "prepare() closures read years=range(2016, 2027) (IM_T1_YEARS padded one year) "
                      "with NO ticker filter, on a tier4.serving_model() cache miss — proven always "
                      "reachable below (option (b) was chosen: a pinned joblib cannot prevent the "
                      "miss, since its cache key embeds store.file_sha256(paths.PANEL), which is the "
                      "PRIVATE MATERIALIZED panel's own hash and essentially never matches any "
                      "pre-existing cache file's embedded hash). (2) WORSE: iv_crush_feature_model's "
                      "prepare = engine.models.training.iv_crush.prepare calls crush_frame() with NO "
                      "arguments, which reads store.read_table('daily_market', columns=[...]) with "
                      "NO years bound and NO ticker filter AT ALL — truly unbounded, not just a fixed "
                      "window. Since materialize() cannot know in advance which of "
                      "pred_abs_move/pred_im_t1_d14/pred_runup_abs_move_d14/pred_iv_crush_30 a given "
                      "scoring run will need (engine.score.py picks the producer per Tier-4 column at "
                      "runtime), whole_table is the only read that is provably never incomplete. "
                      "engine.score.Scorer._entry_implied_move's own trades-ticker-span need (round 2) "
                      "is trivially subsumed by this — evidence_scope_covers_trades() still runs and "
                      "still passes, now vacuously.",
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
    "reference_inputs": "engine.v2.data.reference_inputs.LEGACY_REFERENCE_INPUTS_V1: the model "
                        "registry, structure champions, champion artifacts, Tier-4 serving caches, "
                        "chooser analog pool, calendar CSV and legacy SNAPSHOT, pinned per import "
                        "receipt and carried here as registry_and_model_refs/calendar_refs",
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


def _manifest_row_count(repository, snapshot_ref: SnapshotRef, table_name: str) -> int:
    """The pinned manifest's own total row count for ``table_name`` — the
    sum of every fragment's ``row_count``, never a scan."""
    return sum(r.row_count for r in repository.fragment_records(snapshot_ref, table_name))


def _interval_covers(query_interval: TimeInterval | None,
                     manifest_interval: TimeInterval | None) -> bool:
    """True iff ``query_interval`` is not narrower than ``manifest_interval``
    (review P2-C05, decision 2's "no time bound narrower than the manifest
    bounds"): both absent, or the same column with a start at or before the
    manifest's own start and an end at or after the manifest's own end.
    Naive-timestamp/date strings compare correctly lexicographically."""
    if manifest_interval is None:
        return query_interval is None
    if query_interval is None or query_interval.column != manifest_interval.column:
        return False
    starts_ok = (query_interval.start_inclusive is None
                or (manifest_interval.start_inclusive is not None
                    and query_interval.start_inclusive <= manifest_interval.start_inclusive))
    ends_ok = (query_interval.end_exclusive is None
              or (manifest_interval.end_exclusive is not None
                  and query_interval.end_exclusive >= manifest_interval.end_exclusive))
    return starts_ok and ends_ok


def _whole_table_copy_eligible(repository, snapshot_ref: SnapshotRef, table_name: str,
                               contract: TableContract, query: DataQuery) -> bool:
    """Review P2-C05, decision 2: True only if ``query`` is provably a full,
    unfiltered read of ``table_name``'s pinned manifest — the one case a
    verified byte-for-byte object copy may satisfy instead of a scan. A
    request whose declared projection, predicates or ceiling cover fewer
    rows than the whole table must take the scan-and-rewrite path (or
    refuse), never the copy path, no matter how the static read plan
    classifies the table."""
    if set(query.columns) != {c.name for c in contract.columns}:
        return False
    manifest_filter, manifest_interval = _whole_table_bounds(repository, snapshot_ref, table_name, contract)
    if set(query.key_filter) != set(manifest_filter):
        return False
    if not _interval_covers(query.time_interval, manifest_interval):
        return False
    return query.max_result_rows >= _manifest_row_count(repository, snapshot_ref, table_name)


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
    # and manifest-derived, never the table's generic cap.
    row_bound = max(1, sum(entry.estimated_rows for entry in plan.dependencies))
    # Review fix P2-C05, decision 3: a whole_table output is never fed to
    # Repository.scan() (materialize_tree byte-copies it — see
    # _whole_table_copy_eligible), so the contract's scan-result cap does
    # not bound it. Clamping it here anyway was the "clamp a population to
    # a smaller limit and then copy all rows" bug the review named: the
    # request claimed <= contract.maximum_result_rows while the byte-copy
    # path silently wrote every manifest row regardless. The honest ceiling
    # for a whole_table output is the exact pinned manifest row count.
    # evidence_scoped outputs are always scanned, so they keep the cap.
    if table_name in _EVIDENCE_SCOPED_TABLES:
        max_result_rows = min(row_bound, contract.maximum_result_rows)
    else:
        max_result_rows = row_bound
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
    evidence scope is a superset of the direct scope, (review round 2)
    ``trades``'s real span sits inside the evidence scope too (guide §9.2's
    refusal: "if the adapter cannot prove its read plan complete, it refuses
    checkpoint reuse"), and (review round 4, decision 2) every pinned Tier-4
    serving-model cache ref carries this request's own panel-hash prefix.
    ``repository`` is required, unlike round 1's version, because those last
    two proofs need an actual scan/fragment lookup.
    """
    if set(request.table_queries) != set(SCORE_READ_PLAN_TABLES):
        return False
    if not request.registry_and_model_refs or not request.calendar_refs:
        return False
    if not _scope_superset(request.evidence_scope, request.direct_scope):
        return False
    if not evidence_scope_covers_trades(repository, request):
        return False
    return tier4_cache_refs_match_panel(repository, request)


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
    contract returns. ``copied_tables`` (review round 4, decision 1) names
    every table written by a verified byte-for-byte object copy rather than
    a Repository-scan rewrite — ``legacy_adapter.materialize`` uses it to
    skip the expensive re-scan-and-compare validation for those tables
    (their bytes are already proven correct by the copy's own hash check)
    and only re-read them once, lightly, to prove the unchanged legacy
    reader still opens them.
    """

    manifest: dict[str, str]
    curated_files: dict[str, dict[int, list[Path]]]
    single_files: dict[str, Path]
    copied_tables: frozenset[str] = frozenset()
    #: Review P2-C05, decision 4 (accounting): the actual materialized row
    #: count this call verified for every table it wrote, keyed by
    #: ``table_name`` — recorded on the result, not just checked in passing.
    row_counts: dict[str, int] = dataclasses.field(default_factory=dict)


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
    _check_tier4_cache_refs(repository, request.snapshot_ref, request.registry_and_model_refs)
    manifest: dict[str, str] = {}
    curated_files: dict[str, dict[int, list[Path]]] = {}
    single_files: dict[str, Path] = {}
    copied_tables: set[str] = set()
    row_counts: dict[str, int] = {}
    for table_name, query in request.table_queries.items():
        contract = repository.table_contract(request.snapshot_ref, table_name)
        eligible = (table_name not in _EVIDENCE_SCOPED_TABLES
                   and _whole_table_copy_eligible(repository, request.snapshot_ref, table_name,
                                                  contract, query))
        if TABLE_OUTPUT_KIND[table_name] == "curated":
            year_paths, copied, rows = _materialize_curated(repository, store, request, table_name,
                                                             query, eligible, dest_root)
            curated_files[table_name] = year_paths
            for paths_for_year in year_paths.values():
                for path in paths_for_year:
                    manifest[str(path.relative_to(dest_root))] = _file_content_hash(path)
        else:
            path = dest_root / _single_file_relative_path(table_name)
            copied, rows = _materialize_single(repository, store, request, table_name, query,
                                               contract, eligible, path)
            single_files[table_name] = path
            manifest[str(path.relative_to(dest_root))] = _file_content_hash(path)
        row_counts[table_name] = rows
        if copied:
            copied_tables.add(table_name)
    from . import reference_inputs  # call-time import: see _tier4_cache_dir

    snapshot_rel = reference_inputs.LEGACY_SNAPSHOT_PATH
    snapshot_path = dest_root / snapshot_rel
    _write_pinned_bytes(store, request.legacy_snapshot_object_ref.content_hash, snapshot_path)
    manifest[snapshot_rel] = _file_content_hash(snapshot_path)
    for ref in (*request.registry_and_model_refs, *request.calendar_refs):
        rel, ref_hash = parse_pinned_ref(ref)
        path = dest_root / rel
        _write_pinned_bytes(store, ref_hash, path)
        manifest[rel] = _file_content_hash(path)
    return MaterializedTree(manifest=manifest, curated_files=curated_files, single_files=single_files,
                            copied_tables=frozenset(copied_tables), row_counts=row_counts)


def _parquet_row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def _verify_materialized_rows(actual: int, expected: int, ceiling: int, table_name: str) -> None:
    """Review P2-C05, decision 2/4: the row count actually on disk must
    equal what this output was expected to hold — the pinned manifest count
    for a byte copy, the post-filter scan count for a rewrite — and never
    exceed the request's own ceiling for this table. Catches both a copy
    that silently wrote more/fewer rows than its manifest declares and a
    clamped ceiling a copy ignored; never a clamp-and-copy-anyway."""
    if actual != expected or actual > ceiling:
        raise errors.fail("RESULT_LIMIT_EXCEEDED",
                  "materialized row count does not match this table's expected population",
                  details={"table_name": table_name, "materialized_rows": actual,
                           "expected_rows": expected, "ceiling": ceiling})


def _materialize_curated(repository, store, request: LegacyMaterializationRequest, table_name: str,
                         query: DataQuery, eligible: bool, dest_root: Path):
    if eligible:
        year_paths = _copy_whole_table_curated(repository, request.snapshot_ref, table_name, store,
                                               dest_root)
        expected = _manifest_row_count(repository, request.snapshot_ref, table_name)
        copied = True
    else:
        year_paths, expected = _write_curated_table(repository, query, table_name, dest_root)
        copied = False
    actual = sum(_parquet_row_count(p) for ps in year_paths.values() for p in ps)
    _verify_materialized_rows(actual, expected, query.max_result_rows, table_name)
    return year_paths, copied, actual


def _materialize_single(repository, store, request: LegacyMaterializationRequest, table_name: str,
                        query: DataQuery, contract: TableContract, eligible: bool, path: Path):
    copied = eligible and _copy_whole_single_file(repository, request.snapshot_ref, table_name, store,
                                                  path)
    if copied:
        expected = _manifest_row_count(repository, request.snapshot_ref, table_name)
    else:
        expected = _write_single_file(repository, query, table_name, contract, path)
    actual = _parquet_row_count(path)
    _verify_materialized_rows(actual, expected, query.max_result_rows, table_name)
    return copied, actual


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
                         dest_root: Path) -> tuple[dict[int, list[Path]], int]:
    """Scan-and-rewrite one curated table. Returns the written paths plus the
    post-filter row count the scan itself produced (decision 2's "expected"
    count for a rewrite) — a plain running total, not a re-read."""
    curated_root = dest_root / "data" / "curated" / table_name
    writers: dict[int, pq.ParquetWriter] = {}
    paths: dict[int, Path] = {}
    scanned = 0
    for batch in repository.scan(query, table_name=table_name):
        scanned += batch.num_rows
        _split_batch_by_year(batch, curated_root, writers, paths)
    for writer in writers.values():
        writer.close()
    return {year: [path] for year, path in paths.items()}, scanned


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


# -- review round 4, decision 1: byte-identical copy for whole outputs ------
#
# A whole_table read never filters a row out of any fragment it touches (the
# manifest-derived interval is built FROM those fragments' own time_min/
# time_max, so every row in a surviving fragment is in range by
# construction — see _whole_table_bounds). That makes the object bytes
# themselves the legacy output: no re-encoding, no re-scan, and — critically
# for feature_panel — a byte-identical panel.parquet keeps
# store.file_sha256(paths.PANEL) equal to the SOURCE panel object's own
# hash, which is what lets a Tier-4 serving-model cache reference be pinned
# by an exact, predictable filename (decision 2 below) instead of guaranteed
# to miss.


def _copy_verified_object(store, object_ref, dest_path: Path) -> None:
    """Stream-copy one immutable object's bytes to ``dest_path``, hashing as
    it goes and refusing a mismatch against ``object_ref``'s own recorded
    hash/size — never re-encoded, never loaded whole into memory."""
    digest = object_ref.content_hash.removeprefix(CONTENT_HASH_PREFIX)
    source_path = Path(store.root) / "objects" / digest[:2] / digest
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    size = 0
    try:
        with open(source_path, "rb") as src, open(dest_path, "wb") as dst:
            while chunk := src.read(1 << 20):
                hasher.update(chunk)
                size += len(chunk)
                dst.write(chunk)
    except OSError as exc:
        dest_path.unlink(missing_ok=True)
        raise errors.fail("OBJECT_CORRUPT", "a copied object could not be read from the store") from exc
    if size != object_ref.byte_size or CONTENT_HASH_PREFIX + hasher.hexdigest() != object_ref.content_hash:
        dest_path.unlink(missing_ok=True)
        raise errors.fail("OBJECT_CORRUPT", "a copied object's bytes do not match its recorded hash")


def _copy_whole_table_curated(repository, snapshot_ref: SnapshotRef, table_name: str, store,
                              dest_root: Path) -> dict[int, list[Path]]:
    """Every fragment of a whole_table curated table, copied to its own
    ``part-NNNN.parquet`` in manifest order, per partition year — the exact
    ``part-*.parquet`` glob legacy ``iter_table``/``_partition_files`` already
    concatenates, so a multi-fragment year needs no new legacy-side support."""
    curated_root = dest_root / "data" / "curated" / table_name
    year_paths: dict[int, list[Path]] = {}
    for record in repository.fragment_records(snapshot_ref, table_name):
        year = int(record.partition_key)
        ordinal = len(year_paths.get(year, ()))
        path = curated_root / f"year={year}" / f"part-{ordinal:04d}.parquet"
        _copy_verified_object(store, record.object_ref, path)
        year_paths.setdefault(year, []).append(path)
    return year_paths


def _copy_whole_single_file(repository, snapshot_ref: SnapshotRef, table_name: str, store,
                            dest_path: Path) -> bool:
    """Copy ``table_name``'s single fragment byte-for-byte to ``dest_path``.

    Returns False (nothing written) when the logical partition has more than
    one fragment: legacy ``load_panel``/``load_forecasts`` open ONE fixed
    file path, never a ``part-*`` directory, so a genuinely multi-fragment
    ``feature_panel``/``tier4_forecasts`` cannot be represented as a single
    verified copy — the caller falls back to the existing rewrite path,
    which stays correct (not byte-identical) for that rare case.
    """
    records = repository.fragment_records(snapshot_ref, table_name)
    if len(records) != 1:
        return False
    _copy_verified_object(store, records[0].object_ref, dest_path)
    return True


# --------------------------------------------------------------------------
# review round 4, decision 2: pinned Tier-4 serving-model cache refs
# --------------------------------------------------------------------------
#
# engine.data.features.tier4._serving_path(model_id, fold, snapshot) names
# SERVING_DIR / f"{model_id}_{fold:%Y%m}_{snapshot[:12]}.joblib", where
# snapshot = store.file_sha256(paths.PANEL). The directory's legacy-relative
# path is reference_inputs.TIER4_SERVING_DIR, so a pinned cache ref reads
#
#     <TIER4_SERVING_DIR>/<model_id>_<fold:%Y%m>_<panel_sha256[:12]>.joblib
#
# The snapshot import pins every cache file whose name carries the panel hash
# it imported (reference_inputs.resolve_reference_files). What this module
# checks, structurally: every pinned ref that looks like one of these cache
# files carries the CORRECT panel-hash prefix for THIS request's own panel
# object. A stale ref copied from a different snapshot is refused before it is
# ever trusted, exactly like a missing one already is via
# _verify_pinned_ref_exists.


def _tier4_cache_dir() -> str:
    """``reference_inputs.TIER4_SERVING_DIR``, imported at call time.

    ``reference_inputs`` imports ``legacy_adapter``, which imports this
    module, so a module-level import here would be an import cycle.
    """
    from . import reference_inputs

    return reference_inputs.TIER4_SERVING_DIR


def _panel_object_ref(repository, snapshot_ref: SnapshotRef):
    """The single fragment's ``ObjectRef`` behind ``feature_panel``, or
    ``None`` when it is genuinely multi-fragment (no one predictable hash to
    check a cache ref's filename against)."""
    records = repository.fragment_records(snapshot_ref, "feature_panel")
    return records[0].object_ref if len(records) == 1 else None


def _tier4_cache_hash_prefix(relative_path: str) -> str | None:
    """The ``<panel_sha256[:12]>`` segment of a Tier-4 serving-cache ref's
    filename, or ``None`` if ``relative_path`` is not shaped like one."""
    prefix = f"{_tier4_cache_dir()}/"
    if not relative_path.startswith(prefix) or not relative_path.endswith(".joblib"):
        return None
    stem = relative_path[len(prefix):-len(".joblib")]
    if "_" not in stem:
        return None
    return stem.rsplit("_", 1)[-1]


def _check_tier4_cache_refs(repository, snapshot_ref: SnapshotRef, registry_and_model_refs) -> None:
    panel_object_ref = _panel_object_ref(repository, snapshot_ref)
    if panel_object_ref is None:
        return
    expected = panel_object_ref.content_hash.removeprefix(CONTENT_HASH_PREFIX)[:12]
    for ref in registry_and_model_refs:
        relative_path, _ = parse_pinned_ref(ref)
        prefix = _tier4_cache_hash_prefix(relative_path)
        if prefix is not None and prefix != expected:
            raise errors.fail("TIER4_CACHE_STALE",
                      "a pinned Tier-4 serving-model cache ref's panel-hash prefix does not match "
                      "this materialization's own panel object",
                      details={"path": relative_path})


def tier4_cache_refs_match_panel(repository, request: LegacyMaterializationRequest) -> bool:
    """True iff every pinned Tier-4 cache ref (if any) carries this request's
    own panel-hash prefix — see :func:`_tier4_cache_dir` above."""
    try:
        _check_tier4_cache_refs(repository, request.snapshot_ref, request.registry_and_model_refs)
    except errors.DataError:
        return False
    return True


# -- single-file tables (feature_panel / tier4_forecasts) --------------------


def _write_single_file(repository, query: DataQuery, table_name: str,
                       contract: TableContract, dest_path: Path) -> int:
    """Scan-and-rewrite one single-file table. Returns the post-filter row
    count the scan produced (decision 2's "expected" count for a rewrite)."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    scanned = 0
    try:
        for batch in repository.scan(query, table_name=table_name):
            if writer is None:
                writer = pq.ParquetWriter(dest_path, batch.schema)
            writer.write_batch(batch)
            scanned += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        _write_empty_table(query, contract, dest_path)
    return scanned


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
