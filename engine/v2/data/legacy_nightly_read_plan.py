"""Declared read plans for the shadow nightly's barrier-only legacy stages.

2026-09-14 debug investigation (task brief): a real 4-ticker nightly failed
``legacy_finality`` with ``SOURCE_NOT_FINAL`` because the operator passed
``engine.v2.data.import_snapshot``'s *snapshot-import* manifest
(``capture_implementation_ref="snapshot_import_plan.v1"``) as ``--input-manifest``.
That manifest declares exactly ``LEGACY_SCORE_READ_PLAN_V1``'s tables plus
Tier-3/reference inputs (``legacy_materialization.py``'s own docstring) — it
never includes Tier-1 ``data/raw/fetch/**``, which ``engine.data.finality``
needs and no tool had ever declared. This module is the missing declaration:
what every *barrier-only* ``legacy_*`` nightly kind reads from the real
legacy tree, derived by reading the code each kind actually runs, not from
memory.

**Barrier kinds, enumerated from the registry, not typed by hand.**
``engine/v2/ops/stages.py::registry()`` gives every ``legacy_*`` job kind a
``store_domains=(("legacy_store", "read"),)`` lease (the loop building
``kinds`` for ``ACTION_NAMES`` minus the four ``ledger_export``-style pure
stages, stages.py ~217-224). ``SNAPSHOT_BACKED_KINDS = {"legacy_score",
"legacy_score_requests", "legacy_decision_replay"}`` (stages.py ~112-113) are
the only three kinds ``input_mode_problems`` (stages.py ~127-138) ever lets
run against a verified snapshot materialization instead of the barrier —
every other ``legacy_*`` kind stays on the barrier in EVERY nightly plan,
snapshot-mode or not (``engine/v2/ops/nightly.py`` ~282-284:
``SNAPSHOT_STAGES = frozenset({"score", "decision_replay"})``, "Every other
stage keeps the Phase 1 barrier"). ``stages.py``'s own
``BARRIER_ONLY_REASONS`` dict names three of these six
(``legacy_finality``/``legacy_model_evidence``/``legacy_selfcheck``) with a
one-line reason each; ``legacy_decisions``/``legacy_settlement``/
``legacy_render`` are barrier-only for the identical structural reason —
absent from ``SNAPSHOT_BACKED_KINDS`` — the dict simply never grew a comment
for them. :data:`BARRIER_KINDS` below is the full, structurally-derived set
of six.

**Families.** Rather than list raw path globs per kind, each kind declares
which named :data:`FAMILIES` it reads; a family is one bounded, reviewable
unit (a curated table scope, a reference-input bundle, a raw-fetch lookback
window, a ledger glob, or a single file). ``engine.v2.ops.capture_inputs``
enumerates each family concretely (it is the one module allowed to touch the
real filesystem); this module stays pure — no legacy import, no I/O — so it
can be unit-tested and read as the single source of truth for "what does
kind X read."

**score_context.** ``legacy_model_evidence`` (the ``gate_forecast_analog``
rebuild path, ``engine/dashboard/model_evidence.py:240-250`` builds a
``Scorer(context=FeatureContext.load(...))`` for its analog join),
``legacy_render`` and ``legacy_selfcheck`` (both call
``Scorer(context=FeatureContext.load(tickers, years=years))`` directly —
``engine/v2/ops/legacy_adapter.py``'s ``_action_render``/``_action_selfcheck``)
all share the identical ``engine.features.FeatureContext.load`` +
``engine.score.Scorer`` dependency that ``engine/v2/data/legacy_materialization.py``
already derived, reviewed across four rounds, as ``LEGACY_SCORE_READ_PLAN_V1``.
:data:`FAMILIES["score_context"]` REUSES that plan by reference (never
re-derives it) so the two cannot silently drift apart.
"""
from __future__ import annotations

from .legacy_materialization import LEGACY_SCORE_READ_PLAN_V1

__all__ = [
    "BARRIER_KINDS", "FAMILIES", "LEGACY_NIGHTLY_READ_PLAN_V1",
    "NIGHTLY_CAPTURE_IMPLEMENTATION_REF", "required_families", "manifest_problems",
]

#: ``capture_implementation_ref`` written by ``engine.v2.ops.capture_inputs``
#: (deliverable 2) — the plan-time guard (deliverable 3) refuses any other
#: value, which is exactly how the real 2026-09-14 failure would have been
#: caught before submission: ``snapshot_import_plan.v1`` is not this.
NIGHTLY_CAPTURE_IMPLEMENTATION_REF = "legacy_nightly_capture.v1"

#: The six ``legacy_*`` nightly kinds that never run in snapshot input mode —
#: see the module docstring for the structural derivation from ``stages.py``.
BARRIER_KINDS: tuple[str, ...] = (
    "legacy_finality", "legacy_decisions", "legacy_settlement",
    "legacy_model_evidence", "legacy_render", "legacy_selfcheck",
)

#: Named, bounded read families. ``kind`` selects how ``capture_inputs``
#: enumerates it; every family cites the exact legacy call site it exists for.
FAMILIES: dict[str, dict] = {
    "score_context": {
        "kind": "score_context_bundle",
        "reason": "engine.features.FeatureContext.load + engine.score.Scorer, already fully "
                  "derived as LEGACY_SCORE_READ_PLAN_V1 (legacy_materialization.py) across four "
                  "review rounds; reused here by reference for legacy_model_evidence's "
                  "gate_forecast_analog Scorer rebuild (model_evidence.py:247-250), "
                  "legacy_render's and legacy_selfcheck's own Scorer(FeatureContext.load(...)) "
                  "(engine/v2/ops/legacy_adapter.py:_action_render, :_action_selfcheck)",
    },
    "finality_calendar": {
        "kind": "reference_calendar",
        "reason": "engine.calendar.trading_calendar() reads engine.paths.GSPC_DAILY "
                  "(calendar.py:358-372); engine/v2/ops/legacy_adapter.py:_action_finality calls "
                  "trading_calendar() to pass as resolve_final_session's calendar=",
    },
    "finality_raw_fetch_orats": {
        "kind": "raw_fetch_window",
        "source": "orats",
        "endpoints": ("hist/summaries", "hist/cores"),
        "lookback_sessions": 15,
        "path_prefix": "data/raw/fetch/orats/",
        "reason": "engine/data/finality.py:57-67 _market_wide_complete scans "
                  "fetch.iter_cached('orats') for hist/summaries and hist/cores at each candidate "
                  "date; engine/data/finality.py:203-229 resolve_final_session walks at most 15 "
                  "trading sessions at/before the requested as-of, via engine.calendar.trading_calendar",
    },
    "finality_daily_market_window": {
        "kind": "scoped_curated_table",
        "table": "daily_market",
        "year_window": "as_of.year-1..as_of.year",
        "reason": "engine/data/finality.py:70-84 _coverage_frame('daily_market', 'date', stamp) "
                  "reads store.read_table(years=sorted({stamp.year-1, stamp.year})); "
                  "resolve_final_session (finality.py:216-219) reads this ONCE, before the walk, "
                  "keyed to the ORIGINAL requested as-of, not the walked-back candidate",
    },
    "finality_option_chains_window": {
        "kind": "scoped_curated_table",
        "table": "option_chains",
        "year_window": "as_of.year-1..as_of.year",
        "reason": "engine/data/finality.py:70-84 _coverage_frame('option_chains', 'obs_date', "
                  "stamp), same call site and window as daily_market above; also read by "
                  "covered_tickers (finality.py:182-200) for the resolved date's year",
    },
    "earnings_events_whole": {
        "kind": "whole_curated_table",
        "table": "earnings_events",
        "reason": "engine/ledger.py:302-310 _event_ids (build_prediction_rows' event-id join, "
                  "legacy_decisions) and engine/ledger.py:516-519 _settlement_calendar "
                  "(score_outcomes, legacy_settlement) both call "
                  "store.read_table('earnings_events', ...) with no years/tickers bound",
    },
    "ledger_predictions": {
        "kind": "ledger_glob",
        "directory": "ledger/predictions",
        "required": False,
        "reason": "engine/ledger.py:466-... _unresolved(through) scans predictions_dir().glob("
                  "'*.jsonl') for rows still owed an outcome (score_outcomes, legacy_settlement) "
                  "-- an empty ledger (no predictions ever recorded) is a legitimate first-run "
                  "state, not a capture defect, so this family is not required",
    },
    "ledger_outcomes": {
        "kind": "ledger_glob",
        "directory": "ledger/outcomes",
        "required": False,
        "reason": "engine/v2/ops/legacy_adapter.py:_action_settlement reads "
                  "root/legacy/ledger/outcomes/*.jsonl directly (captured outcome rows); "
                  "engine/dashboard/render.py:size_model_mae_from_ledger calls "
                  "engine.ledger.read_outcomes() (legacy_render's build_health) -- an empty "
                  "ledger is a legitimate first-run state, not a capture defect",
    },
    "settlement_option_chains_window": {
        "kind": "scoped_curated_table",
        "table": "option_chains",
        "year_window": "year_start..year_end",
        "reason": "engine/ledger.py:score_outcomes replays each pending row's recorded structure "
                  "through engine.replay (chain lookups) to simulate the settlement fill; bounded "
                  "by the operator's own --year-start/--year-end, matching every other "
                  "operator-scoped family in this plan (judgement call: the exact exit-date span "
                  "of pending ledger rows cannot be known without reading the ledger first)",
    },
    "model_evidence_cache": {
        "kind": "single_file",
        "path": "data/features/model_evidence.json",
        "required": False,
        "reason": "engine/dashboard/model_evidence.py:evidence_path/load_model_evidence: the "
                  "degraded-fallback cache _action_model_evidence reads when build_model_evidence() "
                  "raises (P2-C08 parity, engine/v2/ops/legacy_adapter.py:_action_model_evidence) "
                  "-- absent on a first-ever run, so not required",
    },
    "fetch_log": {
        "kind": "single_file",
        "path": "data/raw/fetch/fetch_log.csv",
        "required": False,
        "reason": "engine/dashboard/render.py:freshness_summary reads engine.paths.FETCH_LOG for "
                  "per-source last-network-call timestamps (legacy_render's build_meta)",
    },
    "quota_log": {
        "kind": "single_file",
        "path": "data/raw/fetch/quota_log.csv",
        "required": False,
        "reason": "engine/data/throttle.py:latest_quota reads paths.QUOTA_LOG (and, best-effort, "
                  "the grandfathered earnings_predictions quota ledger -- out of scope, see "
                  "capture_inputs module docstring); engine/dashboard/render.py:quota_state calls "
                  "it for legacy_render's build_meta",
    },
}

#: Per-kind family membership plus a short pointer to the adapter action that
#: reads it. Cite ``engine/v2/ops/legacy_adapter.py``'s ``_action_*`` for the
#: single entry point; each family's own ``reason`` above cites the deeper call.
LEGACY_NIGHTLY_READ_PLAN_V1: dict[str, object] = {
    "schema_version": "legacy_nightly_read_plan.v1",
    "kinds": {
        "legacy_finality": {
            "families": ("finality_calendar", "finality_raw_fetch_orats",
                        "finality_daily_market_window", "finality_option_chains_window"),
            "reason": "engine/v2/ops/legacy_adapter.py:_action_finality -> "
                      "engine.data.finality.resolve_final_session/covered_tickers",
        },
        "legacy_decisions": {
            "families": ("earnings_events_whole",),
            "reason": "engine/v2/ops/legacy_adapter.py:_action_decisions -> "
                      "engine.ledger.build_prediction_rows -> _event_ids",
        },
        "legacy_settlement": {
            "families": ("earnings_events_whole", "ledger_predictions", "ledger_outcomes",
                        "settlement_option_chains_window"),
            "reason": "engine/v2/ops/legacy_adapter.py:_action_settlement -> "
                      "engine.ledger.score_outcomes -> _unresolved/_settlement_calendar/replay",
        },
        "legacy_model_evidence": {
            "families": ("score_context", "model_evidence_cache"),
            "reason": "engine/v2/ops/legacy_adapter.py:_action_model_evidence -> "
                      "engine.dashboard.model_evidence.build_model_evidence/load_model_evidence",
        },
        "legacy_render": {
            "families": ("score_context", "earnings_events_whole", "ledger_outcomes",
                        "finality_calendar", "fetch_log", "quota_log"),
            "reason": "engine/v2/ops/legacy_adapter.py:_action_render -> "
                      "render_bundle/build_meta/build_health/_panel_lag_flags/"
                      "_calendar_conflict_flags",
        },
        "legacy_selfcheck": {
            "families": ("score_context",),
            "reason": "engine/v2/ops/legacy_adapter.py:_action_selfcheck -> "
                      "Scorer(context=FeatureContext.load(tickers, years=years))",
        },
    },
    #: Tables/reference inputs the score_context bundle carries, restated
    #: here (rather than re-imported at call sites) purely for readability of
    #: this plan document; the authoritative source stays
    #: ``LEGACY_SCORE_READ_PLAN_V1``.
    "score_context_tables": tuple(LEGACY_SCORE_READ_PLAN_V1["tables"]),
}


def required_families(kind: str, *, only_required: bool = False) -> tuple[str, ...]:
    """Families :data:`LEGACY_NIGHTLY_READ_PLAN_V1` declares for ``kind``.

    ``only_required=True`` drops families explicitly marked
    ``"required": False`` (an absent-on-first-run cache or best-effort log).
    """
    entry = LEGACY_NIGHTLY_READ_PLAN_V1["kinds"].get(kind)
    if entry is None:
        return ()
    families = entry["families"]
    if not only_required:
        return tuple(families)
    return tuple(name for name in families if FAMILIES[name].get("required", True))


def _present_score_context_bundle(spec, paths, manifest) -> bool:
    tables = set(LEGACY_NIGHTLY_READ_PLAN_V1["score_context_tables"])
    have_table = any(f"data/curated/{table}/" in path or path.endswith(f"/{table}.parquet")
                     for path in paths for table in tables)
    have_reference = bool(manifest.get("calendar_ref")) and bool(manifest.get("registry_and_model_refs"))
    return have_table and have_reference


def _present_reference_calendar(spec, paths, manifest) -> bool:
    return bool(manifest.get("calendar_ref"))


def _present_prefixed(spec, paths, manifest) -> bool:
    prefix = spec.get("path_prefix") or f"data/curated/{spec.get('table', '')}/"
    return any(path.startswith(prefix) for path in paths)


def _present_ledger_glob(spec, paths, manifest) -> bool:
    prefix = spec["directory"] + "/"
    return any(path.startswith(prefix) for path in paths)


def _present_single_file(spec, paths, manifest) -> bool:
    if not spec.get("required", True):
        return True
    return spec["path"] in paths


#: One presence checker per family ``kind`` (task brief §1's declared shapes).
#: A dispatch table rather than an if/elif chain: it keeps this module's own
#: complexity budget (checks/code_budgets.py) low as families are added.
_PRESENCE_CHECKS = {
    "score_context_bundle": _present_score_context_bundle,
    "reference_calendar": _present_reference_calendar,
    "raw_fetch_window": _present_prefixed,
    "scoped_curated_table": _present_prefixed,
    "whole_curated_table": _present_prefixed,
    "ledger_glob": _present_ledger_glob,
    "single_file": _present_single_file,
}


def _family_present(family_name: str, manifest: dict) -> bool:
    """Pure presence check: does ``manifest`` (a decoded ``LegacyInputManifest``
    document, or an equivalent plain dict) contain at least one declared file
    for this family? A presence check, not a full completeness re-derivation
    (the capture tool and the completeness test own that); this is the cheap
    guard against the real defect class -- a family missing ENTIRELY.
    """
    spec = FAMILIES[family_name]
    paths = [ref["path"] for ref in manifest.get("file_refs", ())]
    checker = _PRESENCE_CHECKS.get(spec["kind"])
    return checker(spec, paths, manifest) if checker else False


def manifest_problems(manifest: dict, *, kinds: tuple[str, ...] = BARRIER_KINDS) -> list[dict]:
    """Pure plan-time check (deliverable 3): problems with ``manifest`` for
    running ``kinds`` against the barrier. Empty means the manifest is fine.

    Every problem is ``{"kind": ..., "family": ..., "reason": ...}`` (family
    is ``None`` for the provenance check) -- a typed shape a caller can log
    or fold into a ``Problem.details`` payload without inventing structure.
    """
    problems: list[dict] = []
    ref = manifest.get("capture_implementation_ref")
    if ref != NIGHTLY_CAPTURE_IMPLEMENTATION_REF:
        problems.append({
            "kind": None, "family": None,
            "reason": f"capture_implementation_ref {ref!r} is not a nightly capture "
                     f"(expected {NIGHTLY_CAPTURE_IMPLEMENTATION_REF!r}) -- the real "
                     "2026-09-14 failure was exactly this: a snapshot_import_plan.v1 manifest "
                     "passed as --input-manifest to a barrier-mode nightly",
        })
        return problems  # the family checks below are meaningless against the wrong manifest
    for kind in kinds:
        for family in required_families(kind, only_required=True):
            if not _family_present(family, manifest):
                problems.append({
                    "kind": kind, "family": family,
                    "reason": FAMILIES[family]["reason"],
                })
    return problems
