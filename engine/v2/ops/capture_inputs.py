"""The supported capture for the shadow nightly's ``--input-manifest``.

``python3 -m engine.v2.ops capture-inputs`` (wired in ``cli.py``). Builds one
:class:`~engine.v2.contracts.LegacyInputManifest` covering every family
:data:`engine.v2.data.legacy_nightly_read_plan.LEGACY_NIGHTLY_READ_PLAN_V1`
declares for the six barrier-only nightly kinds (``legacy_finality``,
``legacy_decisions``, ``legacy_settlement``, ``legacy_model_evidence``,
``legacy_render``, ``legacy_selfcheck``) -- the union, since one manifest is
what a real ``ops plan nightly --input-manifest`` submits for the WHOLE
barrier-mode nightly, not one kind at a time. Because ``score_context``
(reused from ``LEGACY_SCORE_READ_PLAN_V1``) is also exactly what
``legacy_score``/``legacy_score_requests``/``legacy_decision_replay`` need
when THEY run against the barrier (the default ``input_mode="legacy"``),
this same manifest is sufficient for those three too -- a useful side effect
of taking the union, not a claim this module verifies independently.

Why a plain command rather than a ``plan nightly --capture-from`` flag: the
capture has its own refusal surface (a missing required family, a symlinked
legacy file) that is orthogonal to planning a job graph, and the task brief's
example CLI already spells it as a stand-alone verb. A flag on ``plan
nightly`` would have to thread ``--source-root`` through an unrelated
command and blur "planning failed" with "capture failed" into one refusal
path.

**Never relies on ``engine.paths.ROOT``.** That module binds ``ROOT`` from
``INVESTING_PLAN_ROOT`` once, at first import, and every legacy-touching v2
module (``engine.v2.data.legacy_adapter``, and transitively
``reference_inputs``/``legacy_mapping``) already imports it at module load
time -- long before this module's ``capture()`` runs, and in the SAME
process as everything else ``engine.v2.ops`` does. Every filesystem read here
therefore takes an EXPLICIT ``root`` and never calls a legacy helper that
resolves a path off the process-global ``engine.paths.ROOT`` without a
``root=`` override (``engine.calendar.trading_calendar()`` is exactly that
kind of helper, which is why the trading-session walk below re-reads the
calendar CSV directly rather than calling it). ``engine.data.fetch.iter_cached``
is the one legacy call this module needs, and it DOES accept a ``root=``
override; it is reached through ``engine.v2.ops.legacy_adapter.iter_raw_fetch_cache``
(the package's one declared adapter, ``checks/legacy_adapters.json``), never
imported directly here.

**Scope note on grandfathered quota logs.** ``engine.data.throttle.latest_quota``
best-effort-reads ``engine.paths.RAW_ORATS_QUOTA_LOG``
(``earnings_predictions/data/raw/orats/quota_log.csv``) in addition to the
Tier-1 ``quota_log.csv`` this module captures. That path sits inside the
GRANDFATHERED ``earnings_predictions`` research tree (``engine/paths.py:42-68``,
``GRANDFATHERED = (EP, BT, RAW_POLYGON_LEGACY)``) -- a much larger, differently
governed tree this task's read-only approval does not cover capturing
wholesale. ``latest_quota`` degrades to ``remaining: None`` when a log is
absent, so omitting it degrades ``legacy_render``'s quota display, never
fails the stage; documented here rather than silently dropped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from engine.v2.contracts import LegacyFileRef, LegacyInputManifest
from engine.v2.data import reference_inputs
from engine.v2.data.legacy_mapping import PANEL_RELATIVE_PATH, TIER4_RELATIVE_PATH
from engine.v2.data.legacy_materialization import LEGACY_SCORE_READ_PLAN_V1
from engine.v2.data.legacy_nightly_read_plan import (
    BARRIER_KINDS,
    FAMILIES,
    NIGHTLY_CAPTURE_IMPLEMENTATION_REF,
    required_families,
)
from engine.v2.foundation import CONTENT_HASH_PREFIX, content_hash, to_document
from engine.v2.ops.errors import fail
from engine.v2.ops.legacy_adapter import iter_raw_fetch_cache, manifest_files

__all__ = ["capture", "write_manifest"]

_DATA_DIR = reference_inputs.DATA_DIR  # "data" -- engine.paths.DATA, relative to ROOT
_CALENDAR_RELATIVE_PATH = reference_inputs.LEGACY_REFERENCE_INPUTS_V1["inputs"]["calendar"]["path"]


@dataclass(frozen=True)
class _Scope:
    as_of: pd.Timestamp
    tickers: tuple[str, ...]
    context_tickers: tuple[str, ...]
    year_start: int
    year_end: int


def _curated_dir(table: str) -> str:
    return f"{_DATA_DIR}/curated/{table}"


def _enumerate_table_years(root: Path, table: str, years: set[int] | None) -> list[str]:
    """Relative ``part-*`` paths for ``table``, restricted to ``years`` when given.

    Mirrors ``engine.data.store.table_years``/``read_table``'s own scope
    exactly (``years=None`` means every ``year=YYYY`` directory present, the
    same "whole table" a legacy read site with no ``years=`` bound gets) --
    never a raw recursive glob past the table's own directory.
    """
    table_dir = root / _curated_dir(table)
    if not table_dir.is_dir() or table_dir.is_symlink():
        return []
    out: list[str] = []
    for year_dir in sorted(table_dir.iterdir(), key=lambda p: p.name):
        if not year_dir.is_dir() or year_dir.is_symlink() or not year_dir.name.startswith("year="):
            continue
        try:
            year = int(year_dir.name.split("=", 1)[1])
        except ValueError:
            continue
        if years is not None and year not in years:
            continue
        for part in sorted(year_dir.iterdir(), key=lambda p: p.name):
            if part.is_file() and not part.is_symlink() and part.name.startswith("part-"):
                out.append(f"{_curated_dir(table)}/{year_dir.name}/{part.name}")
    return out


def _enumerate_ledger_glob(root: Path, directory: str, as_of: pd.Timestamp) -> list[str]:
    """``.jsonl`` files under ``directory``, bounded to strictly BEFORE
    session ``as_of``.

    2026-09-15 fix (real 2026-09-10 attempt 11 failure): an unbounded glob
    picked up ``ledger/predictions/2026-09-10.jsonl`` -- session S's own
    predictions, which v2 must produce and commit itself, never read
    pre-computed from the staged legacy ledger -- and
    ``ledger/outcomes/2026-09-11.jsonl``/``2026-09-12.jsonl``, dated after
    the session (outcomes files are named by settle date). Both leaked into
    ``legacy_settlement``, which then tried to commit outcome lines naming
    predictions never committed in the catalog and refused
    ``VALIDATION_FAILED``.

    2026-09-15 correction (send-back on 7d235f8): ``ledger/outcomes/S.jsonl``
    must be excluded too, not only files dated AFTER S. It is S's own
    nightly settlement output (legacy's ``score_outcomes(through=S)`` writes
    it same-day, keyed by ``resolved_at``) -- staging it makes legacy's own
    ``_unresolved`` treat those rows as already settled and skip them, so v2
    never produces or commits them itself. That is exactly the predictions
    case: S's own output belongs to v2, not to a staged copy of legacy's.
    Predictions and outcomes therefore share ONE bound -- excluded strictly
    AT and after the session, matching ``ops ledger import-history
    --through``'s own exclusive bound. A file whose name is not a bare
    ``YYYY-MM-DD.jsonl`` date is left out rather than guessed at -- an
    unrecognized name is not proof it is in-bounds.
    """
    ledger_dir = root / directory
    if not ledger_dir.is_dir() or ledger_dir.is_symlink():
        return []
    cutoff = pd.Timestamp(as_of).normalize()
    out = []
    for p in ledger_dir.iterdir():
        if not (p.is_file() and not p.is_symlink() and p.name.endswith(".jsonl")):
            continue
        try:
            file_date = pd.Timestamp(p.name[:-len(".jsonl")])
        except ValueError:
            continue
        if file_date >= cutoff:
            continue
        out.append(f"{directory}/{p.name}")
    return sorted(out)


def _lookback_sessions(root: Path, as_of: pd.Timestamp, *, max_sessions: int = 15) -> tuple[str, ...]:
    """Up to ``max_sessions`` trading-session dates at/before ``as_of``.

    Replicates ``engine.calendar.trading_calendar()``'s own source
    (``engine.paths.GSPC_DAILY``, ``calendar.py:358-372``: ``pd.read_csv(path,
    skiprows=3, header=None, usecols=[0], names=["date"])``) directly against
    an explicit ``root`` rather than calling ``trading_calendar()`` itself --
    that helper resolves ``engine.paths.GSPC_DAILY`` off the process-global
    ``engine.paths.ROOT`` with no override, which this module must never rely
    on (module docstring).

    2026-09-14 correction: the observed series alone is NOT enough. Measured
    against the real repo, the observed CSV lagged "today" by roughly two
    weeks (whatever the last successful calendar pull wrote), so an
    observed-only walk silently resolved a stale window -- 15 sessions
    ending 2026-08-27 for an ``as_of`` of 2026-09-10, missing the ORATS cache
    entries that actually exist for the real recent dates. ``trading_calendar()``
    itself never has this gap because it always extends past the last
    observed date with ``projected_trading_days`` (rule-based weekdays minus
    US market holidays, ``calendar.py``'s own ``extend_days`` construction) --
    a PURE function with no ``engine.paths`` dependency, reached here through
    ``engine.v2.ops.legacy_adapter.projected_trading_sessions`` so this walk
    matches the real calendar for any ``as_of`` at or beyond the last
    observed date, not only within observed history.
    """
    from engine.v2.ops.legacy_adapter import projected_trading_sessions

    path = root / _CALENDAR_RELATIVE_PATH
    if not path.is_file() or path.is_symlink():
        raise fail("INPUT_CHANGED", "trading calendar source is missing",
                  details={"path": _CALENDAR_RELATIVE_PATH})
    frame = pd.read_csv(path, skiprows=3, header=None, usecols=[0], names=["date"])
    observed = pd.to_datetime(frame["date"], errors="coerce").dropna().dt.normalize()
    stamp = pd.Timestamp(as_of).normalize()
    last_observed = max(observed) if len(observed) else stamp
    projected = (pd.to_datetime(list(projected_trading_sessions(last_observed, stamp)))
                if stamp > last_observed else pd.DatetimeIndex([]))
    days = sorted(set(observed) | set(projected), reverse=True)
    at_or_before = [d for d in days if d <= stamp]
    return tuple(str(d.date()) for d in at_or_before[:max_sessions])


def _enumerate_raw_fetch_orats(root: Path, sessions: tuple[str, ...],
                               endpoints: tuple[str, ...]) -> list[str]:
    """Every cached ``(meta, body)`` pair for ``orats``/``endpoints`` whose
    ``tradeDate`` param falls in ``sessions`` -- the exact filter
    ``engine.data.finality._market_wide_complete`` applies (finality.py:57-67),
    scoped to the finality lookback window rather than the whole cache.
    """
    wanted_endpoints = set(endpoints)
    wanted_sessions = set(sessions)
    out: list[str] = []
    for entry in iter_raw_fetch_cache(root, "orats"):
        if entry.endpoint not in wanted_endpoints:
            continue
        if str(entry.params.get("tradeDate")) not in wanted_sessions:
            continue
        body = entry.path
        meta = body.with_name(body.name.replace(".body.gz", ".meta.json"))
        for candidate in (meta, body):
            if candidate.is_file() and not candidate.is_symlink():
                out.append(candidate.resolve().relative_to(root.resolve()).as_posix())
    return sorted(set(out))


def _single_file(root: Path, relative: str) -> str | None:
    path = root / relative
    if path.is_file() and not path.is_symlink():
        return relative
    return None


def _score_context_paths(root: Path, scope: _Scope) -> tuple[list[str], str]:
    """The ``score_context`` bundle's curated-table paths, plus
    ``feature_panel``'s own content hash (``reference_inputs.resolve_reference_files``
    needs it to select the matching Tier-4 serving caches)."""
    tables = LEGACY_SCORE_READ_PLAN_V1["tables"]
    paths: list[str] = []
    for name, spec in tables.items():
        if spec["output"] == "single_file":
            continue
        years = None if spec["scope"] == "whole_table" else set(
            range(scope.year_start, scope.year_end + 1))
        paths.extend(_enumerate_table_years(root, name, years))
    panel_relative = f"{_DATA_DIR}/{PANEL_RELATIVE_PATH}"
    tier4_relative = f"{_DATA_DIR}/{TIER4_RELATIVE_PATH}"
    for relative in (panel_relative, tier4_relative):
        if _single_file(root, relative) is None:
            raise fail("INPUT_CHANGED", "score_context single-file table is missing",
                      details={"path": relative})
        paths.append(relative)
    panel_hash = manifest_files(root, [panel_relative])[panel_relative]["content_hash"]
    return paths, panel_hash


def _file_ref_for(root: Path):
    def _make(_root: Path, relative: str) -> LegacyFileRef:
        info = manifest_files(root, [relative])[relative]
        return LegacyFileRef(path=relative, content_hash=info["content_hash"],
                             byte_size=info["byte_size"])
    return _make


def _to_refs(hashed: dict) -> tuple[LegacyFileRef, ...]:
    return tuple(LegacyFileRef(path=path, content_hash=info["content_hash"],
                               byte_size=info["byte_size"])
                for path, info in sorted(hashed.items()))


def _capture_raw_fetch_window(root: Path, family: str, spec: dict, scope: _Scope) -> list[str]:
    sessions = _lookback_sessions(root, scope.as_of, max_sessions=spec["lookback_sessions"])
    found = _enumerate_raw_fetch_orats(root, sessions, spec["endpoints"])
    if not found:
        raise fail("SOURCE_EMPTY",
                  "no cached ORATS market-wide files for any finality lookback session",
                  details={"family": family, "sessions": list(sessions),
                           "endpoints": list(spec["endpoints"])})
    return found


def _capture_whole_curated_table(family: str, spec: dict, root: Path) -> list[str]:
    found = _enumerate_table_years(root, spec["table"], None)
    if not found:
        raise fail("INPUT_CHANGED", "required curated table has no data on disk",
                  details={"family": family, "table": spec["table"]})
    return found


def _capture_scoped_curated_table(family: str, spec: dict, root: Path, scope: _Scope,
                                  finality_years: set[int]) -> list[str]:
    years = finality_years if family.startswith("finality_") else set(
        range(scope.year_start, scope.year_end + 1))
    # A genuinely empty window (no partitions in range) is a real
    # finality/settlement verdict (SOURCE_NOT_FINAL / no pending rows), not
    # a capture defect -- not refused here.
    return _enumerate_table_years(root, spec["table"], years)


def _capture_single_file(family: str, spec: dict, root: Path) -> list[str]:
    found = _single_file(root, spec["path"])
    if found is None and spec.get("required", True):
        raise fail("INPUT_CHANGED", "required file is missing",
                  details={"family": family, "path": spec["path"]})
    return [found] if found is not None else []


def _capture_family(root: Path, family: str, scope: _Scope, finality_years: set[int],
                    panel_hash: str | None) -> tuple[list[str], str | None]:
    """One family's candidate paths, plus an updated ``panel_hash`` (only the
    ``score_context_bundle`` family sets it; every other kind passes it
    through unchanged).
    """
    spec = FAMILIES[family]
    kind = spec["kind"]
    if kind == "score_context_bundle":
        paths, panel_hash = _score_context_paths(root, scope)
        return paths, panel_hash
    if kind == "reference_calendar":
        return [], panel_hash  # folded into the reference-input bundle, below
    if kind == "raw_fetch_window":
        return _capture_raw_fetch_window(root, family, spec, scope), panel_hash
    if kind == "whole_curated_table":
        return _capture_whole_curated_table(family, spec, root), panel_hash
    if kind == "scoped_curated_table":
        return _capture_scoped_curated_table(family, spec, root, scope, finality_years), panel_hash
    if kind == "ledger_glob":
        return _enumerate_ledger_glob(root, spec["directory"], scope.as_of), panel_hash
    if kind == "single_file":
        return _capture_single_file(family, spec, root), panel_hash
    return [], panel_hash


def _resolve_reference_bundle(root: Path, families: list[str], panel_hash: str | None):
    """The reference-input bundle (calendar/registry/structures/chooser
    pool/SNAPSHOT/champion artifacts/tier4 serving caches), resolved once,
    keyed to the score_context panel hash when that bundle is in scope, or a
    freshly-computed one otherwise (legacy_finality alone still needs the
    calendar file; ``resolve_reference_files``'s "exact" entries do not
    depend on the panel hash at all). ``None`` (not needed) when no family
    in scope uses the bundle.
    """
    needs_reference = any(FAMILIES[name]["kind"] in ("score_context_bundle", "reference_calendar")
                          for name in families)
    if not needs_reference:
        return None
    if panel_hash is None:
        panel_relative = f"{_DATA_DIR}/{PANEL_RELATIVE_PATH}"
        panel_hash = (manifest_files(root, [panel_relative])[panel_relative]["content_hash"]
                     if _single_file(root, panel_relative) is not None
                     else CONTENT_HASH_PREFIX + "0" * 64)
    return reference_inputs.resolve_reference_files(
        root, panel_content_hash=panel_hash, file_ref=_file_ref_for(root))


def _build_manifest(file_refs, registry_and_model_refs, calendar_ref,
                    scope: _Scope) -> LegacyInputManifest:
    fields = dict(
        file_refs=file_refs, table_contract_refs=(),
        registry_and_model_refs=registry_and_model_refs, calendar_ref=calendar_ref,
        selected_session=str(scope.as_of.date()), finality_receipt_refs=(),
        knowledge_mode_by_table={}, availability_evidence_refs=(), read_set_complete=True,
        capture_implementation_ref=NIGHTLY_CAPTURE_IMPLEMENTATION_REF)
    placeholder = LegacyInputManifest(manifest_id="pending", **fields)
    digest = content_hash(to_document(placeholder)).removeprefix(CONTENT_HASH_PREFIX)[:32]
    return LegacyInputManifest(manifest_id=f"nightly_capture_{digest}", **fields)


def capture(source_root, *, as_of, tickers=(), context_tickers=(), year_start: int,
           year_end: int, kinds: tuple[str, ...] = BARRIER_KINDS) -> LegacyInputManifest:
    """Enumerate, hash and pin every declared family for ``kinds`` (default:
    every barrier kind). Read-only: every path under ``source_root`` is
    opened for stat/hash only, never written.
    """
    root = Path(source_root).resolve()
    stamp = pd.Timestamp(as_of).normalize()
    scope = _Scope(as_of=stamp, tickers=tuple(sorted(tickers)),
                   context_tickers=tuple(sorted(context_tickers or tickers)),
                   year_start=int(year_start), year_end=int(year_end))
    families = sorted({name for kind in kinds for name in required_families(kind)})
    finality_years = {scope.as_of.year - 1, scope.as_of.year}

    candidate_paths: list[str] = []
    panel_hash: str | None = None
    for family in families:
        paths, panel_hash = _capture_family(root, family, scope, finality_years, panel_hash)
        candidate_paths.extend(paths)

    reference_refs = _resolve_reference_bundle(root, families, panel_hash) or ()
    candidate_paths.extend(ref.path for ref in reference_refs)

    unique_paths = sorted(set(candidate_paths))
    file_refs = _to_refs(manifest_files(root, unique_paths))
    registry_and_model_refs, calendar_ref = reference_inputs.manifest_pins(reference_refs)
    return _build_manifest(file_refs, registry_and_model_refs, calendar_ref, scope)


def write_manifest(manifest: LegacyInputManifest, output) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_document(manifest), indent=2, sort_keys=True))
    return path
