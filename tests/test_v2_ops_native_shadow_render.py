"""spec_ns_b: native scoring as the v2 shadow-serving row source (G1/G5).

Every case drives the real production path: real ``checks/phase4_real.py``
source bundles through ``engine.v2.scoring.application.score_one``, the real
``engine.v2.serving.projections.build_candidate`` against a real temp
``serving.sqlite`` and a real Phase 2 catalog, and a real list-appending
observer for the S9H analog display channel.  No guard is monkeypatched and no
fixture id is hand-injected into a row; the only fixed data is the declared
source material the scorer computes from.

``build_native_bundle_rows`` returns the spec's ``{native_row_key: row}``
mapping.  The real ``bridge.build_bridges`` consumes the
``{ticker: [row, ...]}`` shape ``legacy_bundle`` already provides, so the seam
function under test (``shadow_serving_row_source``) is what regroups it for
``build_candidate`` -- the only shape adaptation between the two documented
contracts.
"""
from __future__ import annotations

import ast
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from checks import phase4_real, tier0_corpus
from checks.phase4_real import _numerical_independence_source, _request
from engine.v2.contracts import PreviewRelease
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, content_hash
from engine.v2.ops import native_shadow_render as ops_shadow
from engine.v2.ops.errors import OpsError
from engine.v2.ops.native_parity_report import (
    PARITY_DIMENSIONS,
    compare_native_vs_legacy,
    native_parity_handler,
    write_parity_report,
)
from engine.v2.ops.native_shadow_render import native_shadow_serving_mode
from engine.v2.ops.nightly import GRAPH, OPTIONAL, build_nightly_plan, run_shadow_nightly
from engine.v2.scoring import application
from engine.v2.scoring.source_inputs import build_native_score_inputs
from engine.v2.scoring.stages import analog_display_fields
from engine.v2.serving import native_shadow_render as serving_shadow
from engine.v2.serving import projections
from engine.v2.serving.native_render import NATIVE_NEVER_COMPUTED
from engine.v2.serving.native_shadow_render import (
    NativeShadowConfigError,
    build_native_bundle_rows,
    shadow_serving_row_source,
)
from tests.test_phase4_capture_strict import _artifact, _full_strict_candidate
from tests.test_v2_serving_projections import (
    _event_row,
    _events_snapshot,
    _preview_input,
)
from tools.capture_tier0_corpus import canonical_v2_request, write

_TICKERS = ("PHASE4", "PHASE5")
_EVENT_DATE = "2026-09-16"
_NATIVE_PLAN = {"shadow_serving_scorer": "native"}
_LEGACY_PLAN = {"shadow_serving_scorer": "legacy"}
_EMPTY_SCORE_DOC: dict = {"rows": [], "ladder": [], "expected_population": []}

#: G1: the write paths a shadow row source must never name, from either half
#: of the layer-7-peer seam (``engine.dashboard`` wholesale covers the legacy
#: board's own ``nightly`` entrypoint).
_FORBIDDEN_IMPORT_PREFIXES = (
    "engine.v2.ops.decision_commit",
    "engine.v2.ops.ledger_history_import",
    "engine.v2.ops.legacy_actions",
    "engine.dashboard",
)


def _pairs() -> dict[str, tuple]:
    """Two real ``(ScoreRequest, NativeScoreInputs)`` pairs, distinct tickers."""
    base = _numerical_independence_source()
    pairs: dict[str, tuple] = {}
    for index, ticker in enumerate(_TICKERS):
        request = _request(event_id=f"native-shadow-{index}")
        inputs = build_native_score_inputs(
            replace(base, context={**base.context, "ticker": ticker}))
        pairs[f"{ticker}|STR-THRU|{_EVENT_DATE}"] = (request, inputs)
    return pairs


def _shadow_rows(tmp_path):
    """Build rows through the real seam and project them, exactly like tools."""
    (tmp_path / "phase2").mkdir()
    conn, store, snap = _events_snapshot(
        tmp_path / "phase2",
        [_event_row(f"e{index}", ticker, datetime(2026, 9, 16))
         for index, ticker in enumerate(_TICKERS)],
        year="2026")
    repository = Repository(conn, store)
    serving_root = tmp_path / "serving"
    serving_root.mkdir()
    serving_store = ArtifactStore(serving_root / "objects")
    serving_conn = projections.connect(str(serving_root / "serving.sqlite"))
    pairs = _pairs()
    bundle = shadow_serving_row_source(_NATIVE_PLAN, _EMPTY_SCORE_DOC, {}, pairs)
    score_doc = {
        "rows": [row for rows in bundle.values() for row in rows],
        "ladder": [],
        "expected_population": sorted(pairs),
    }
    release = projections.build_candidate(
        _preview_input(), score_doc, bundle,
        repository=repository, snapshot_ref=snap, store=serving_store,
        conn=serving_conn, requested_as_of=_EVENT_DATE, resolved_as_of=_EVENT_DATE)
    return release, serving_conn, serving_store, pairs


def test_real_native_rows_reach_the_serving_index(tmp_path):
    """Test 1: score_one -> seam -> build_candidate -> real SELECT."""
    release, serving_conn, _, pairs = _shadow_rows(tmp_path)
    assert isinstance(release, PreviewRelease)
    summaries = serving_conn.execute(
        "SELECT e.ticker, s.n_analogs, s.selected_row_ids "
        "FROM serving_score_summary s "
        "JOIN serving_event_summary e ON e.release_id = s.release_id "
        "AND e.event_id = s.event_id WHERE s.release_id = ? ORDER BY e.ticker",
        (release.release_id,)).fetchall()
    assert {row["ticker"] for row in summaries} == set(_TICKERS)

    expected: dict[str, dict] = {}
    for key, (request, inputs) in pairs.items():
        observations: list = []
        record = application.score_one(request, inputs, observer=observations.append)
        fields = analog_display_fields(observations)
        expected[key.split("|", 1)[0]] = {
            "n_analogs": record.resolved_request.get("n_analogs"),
            "selected_row_ids": [str(value) for value in fields["selected_row_ids"]],
        }
    for row in summaries:
        real = expected[row["ticker"]]
        assert real["n_analogs"] is not None  # the real analog stage ran
        assert real["selected_row_ids"]  # and matched a real population
        assert row["n_analogs"] == real["n_analogs"]  # served verbatim, not a fixture
        assert json.loads(row["selected_row_ids"]) == real["selected_row_ids"]
    serving_conn.close()


def test_build_native_bundle_rows_keys_each_row_by_native_row_key():
    """The spec's ``{native_row_key: display_row}`` contract, on real records."""
    pairs = _pairs()
    rows = build_native_bundle_rows(_EMPTY_SCORE_DOC, pairs)
    assert set(rows) == set(pairs)
    for key, row in rows.items():
        assert key == "|".join(
            (str(row["ticker"]), str(row["strategy"]), str(row["event_date"])))
        assert row["gate_pass"] is True
        assert row["exp_pnl_model"] is not None


def test_g1_no_write_path_import_is_statically_named():
    """Test 2: static AST scan of both halves; a runtime mock cannot pass it."""
    for module in (ops_shadow, serving_shadow):
        imported = _imported_dotted_names(module)
        for prefix in _FORBIDDEN_IMPORT_PREFIXES:
            offenders = sorted(
                name for name in imported
                if name == prefix or name.startswith(prefix + "."))
            assert offenders == [], (module.__name__, prefix, offenders)


def _imported_dotted_names(module) -> set[str]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_g5_switch_is_explicit_and_typed():
    """Test 3: both halves validate the same two strings, never a bare error."""
    assert native_shadow_serving_mode({"shadow_serving_scorer": "legacy"}) == "legacy"
    assert native_shadow_serving_mode({"shadow_serving_scorer": "native"}) == "native"
    assert native_shadow_serving_mode({}) == "native"
    with pytest.raises(OpsError) as error:
        native_shadow_serving_mode({"shadow_serving_scorer": "auto"})
    assert error.value.code == "INVALID_REQUEST"
    assert error.value.problem.category == "validation"

    with pytest.raises(NativeShadowConfigError) as row_error:
        shadow_serving_row_source({"shadow_serving_scorer": "auto"}, {}, {}, {})
    assert row_error.value.code == "INVALID_REQUEST"


def test_legacy_switch_returns_the_bundle_rows_unchanged():
    legacy = {"AAA": [{"ticker": "AAA", "strategy": "STR-THRU"}]}
    assert shadow_serving_row_source(_LEGACY_PLAN, {}, legacy, {}) is legacy


def test_g3_never_computed_band_stays_absent_through_the_full_path(tmp_path):
    """Test 4: no column or display/engine row invents a legacy band value."""
    release, serving_conn, serving_store, _ = _shadow_rows(tmp_path)
    score_ids = [row["score_id"] for row in serving_conn.execute(
        "SELECT score_id FROM serving_score_summary WHERE release_id = ?",
        (release.release_id,)).fetchall()]
    assert len(score_ids) == len(_TICKERS)
    for score_id in score_ids:
        bridge = projections.get_score_detail(
            serving_conn, serving_store, release.release_id, score_id)
        assert set(NATIVE_NEVER_COMPUTED).isdisjoint(bridge.display_record)
        assert set(NATIVE_NEVER_COMPUTED).isdisjoint(bridge.engine_record)
    columns = {row[1] for row in serving_conn.execute(
        "PRAGMA table_info(serving_score_summary)")}
    assert set(NATIVE_NEVER_COMPUTED).isdisjoint(columns)
    serving_conn.close()


def test_the_nightly_plan_records_the_shadow_serving_scorer():
    """G5: the switch is a recorded plan field, not an env var or CLI flag."""
    repo = str(Path(__file__).resolve().parents[1])
    plan = build_nightly_plan(repo, _EVENT_DATE)
    assert plan["shadow_serving_scorer"] == "native"
    assert native_shadow_serving_mode(plan) == "native"
    legacy = build_nightly_plan(repo, _EVENT_DATE, shadow_serving_scorer="legacy")
    assert native_shadow_serving_mode(legacy) == "legacy"


# --------------------------------------------------------------------------
# spec_ns_c: the per-night parity report (G2) and its G4 corpus control.
# --------------------------------------------------------------------------


def _p4_corpus_comparison_hash(corpus_root: Path) -> str:
    """The real P4 corpus comparison's own content hash, from real machinery.

    ``checks/phase4_real._native_parity`` is the corpus comparison
    ``checks/phase4_real.build_evidence`` runs (its ``native_parity`` stage);
    ``comparison_receipt`` is ``content_hash(rows)`` over every row it
    compared, so it changes if any hashed corpus payload does.
    """
    release, parity = phase4_real._native_parity(tier0_corpus.load(corpus_root))
    assert parity["population"]["compared"] == 1, release["dispositions"]
    return release["comparison_receipt"]


def _g4_strict_corpus(tmp_path: Path) -> Path:
    """A real one-pair tier-0 corpus with a strict Phase 4 trace."""
    path, digest = _artifact(tmp_path)
    candidate = _full_strict_candidate(
        fixture_id="case-g4", ticker="AAA", driver_vector={"x": 2.0},
        gate_vector={"x": 9.0, "n_prior": 5.0}, path=path, digest=digest)
    checkpoint = candidate["legacy_trace"]["checkpoints"]["source_inputs"]
    # The frozen bridge verifies each binding's own decision clock against the
    # request the strict probe derives from the legacy request; align the
    # fixture bindings with the derived clock rather than the placeholder
    # default ``_model_binding`` stamps.
    clock = canonical_v2_request(candidate, "snapshot-1").decision_clock_id
    for binding in checkpoint["value"]["model_bindings"]:
        binding["decision_clock"] = clock
    checkpoint["value"]["native_recipes"]["simulation"] = {
        "terminal_spots": [95.0, 105.0], "capital_at_risk": 3.0}
    checkpoint["content_hash"] = content_hash(checkpoint["value"])
    write(tmp_path / "tier0", [candidate], {}, pd.Timestamp("2026-09-16"),
          "snapshot-1", strict_trace=True)
    return tmp_path / "tier0"


def _exercise_new_modules(tmp_path: Path) -> None:
    """Call the new modules the way the nightly seam does -- G4's 'after'."""
    assert ops_shadow.native_shadow_serving_mode(_NATIVE_PLAN) == "native"
    rows = build_native_bundle_rows(_EMPTY_SCORE_DOC, _pairs())
    report = compare_native_vs_legacy(rows, {key: dict(row) for key, row in rows.items()},
                                      PARITY_DIMENSIONS)
    assert report["mismatches"] == []
    write_parity_report(report, tmp_path / "parity_report.json")


def test_g4_p4_corpus_comparison_hash_is_byte_identical_across_the_new_modules(
        tmp_path, monkeypatch):
    """G4: exercising the new modules never perturbs what phase4_real hashes.

    The REAL ``checks/phase4_real.py`` corpus comparison runs before and after
    importing/exercising ``engine.v2.ops.native_shadow_render`` and
    ``engine.v2.ops.native_parity_report`` in this one test process; its own
    content hash must be byte-identical across both runs.  Neither module
    touches ``score.json``'s ``rows``/``ladder`` or any ``ScoreRecord`` field,
    so a difference here would be a real G4 regression, not a flake.
    """
    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
    corpus_root = _g4_strict_corpus(tmp_path)

    before = _p4_corpus_comparison_hash(corpus_root)
    _exercise_new_modules(tmp_path)
    after = _p4_corpus_comparison_hash(corpus_root)
    assert after == before


def test_native_parity_handler_status_is_explicit_and_the_report_is_separate(tmp_path):
    """The handler returns ``compared``/``not_applicable`` -- never a bool."""
    rows = build_native_bundle_rows(_EMPTY_SCORE_DOC, _pairs())
    report_path = tmp_path / "parity_report.json"

    legacy = native_parity_handler(
        _LEGACY_PLAN, legacy_rows={}, native_rows=rows,
        report_path=report_path)({"session": _EVENT_DATE})
    assert legacy["native_parity"]["status"] == "not_applicable"
    assert not report_path.exists()  # no native rows -> nothing written

    native = native_parity_handler(
        _NATIVE_PLAN, legacy_rows={key: dict(row) for key, row in rows.items()},
        native_rows=rows, report_path=report_path)({"session": _EVENT_DATE})
    assert native["native_parity"]["status"] == "compared"
    report = json.loads(report_path.read_text())
    assert report["schema_version"] == "native_parity_report.v1.0"
    assert sorted(report["compared"]) == sorted(rows)
    assert report["only_legacy"] == [] and report["only_native"] == []
    assert report["mismatches"] == []


def test_compare_reports_a_mismatch_without_raising_or_reconciling():
    """G2: a difference is a return-value finding, never an exception."""
    rows = build_native_bundle_rows(_EMPTY_SCORE_DOC, _pairs())
    key = sorted(rows)[0]
    legacy = {key: {**rows[key], "exp_pnl_model": rows[key]["exp_pnl_model"] + 1.0}}

    report = compare_native_vs_legacy(legacy, rows, PARITY_DIMENSIONS)

    assert report["compared"] == [key]
    assert report["mismatches"] and report["mismatches"][0]["row_key"] == key
    assert report["mismatches"][0]["dimension"] == "simulation"
    assert "exp_pnl_model" in report["mismatches"][0]["finding_fields"]
    # No code path copies either side's value into the other row.
    assert rows[key]["exp_pnl_model"] != legacy[key]["exp_pnl_model"]


def test_compare_refuses_an_unknown_dimension_and_write_propagates_oserror(tmp_path):
    with pytest.raises(OpsError) as error:
        compare_native_vs_legacy({}, {}, ("not_a_dimension",))
    assert error.value.code == "INVALID_REQUEST"
    with pytest.raises(OSError):
        write_parity_report({"schema_version": "native_parity_report.v1.0"},
                            tmp_path / "missing" / "parity_report.json")


def test_native_parity_stage_runs_optional_in_the_real_shadow_graph(tmp_path):
    """The stage is in the fixed DAG, optional, and reports through a real run."""
    assert GRAPH["native_parity"] == ("score",)
    assert "native_parity" in OPTIONAL
    rows = build_native_bundle_rows(_EMPTY_SCORE_DOC, _pairs())
    source = tmp_path / "source"
    source.mkdir()
    private = tmp_path / "private"
    report_path = private / "parity_report.json"
    handlers = {stage: (lambda value, stage=stage: {**value, stage: "ok"})
                for stage in GRAPH}
    handlers["native_parity"] = native_parity_handler(
        _NATIVE_PLAN, legacy_rows={key: dict(row) for key, row in rows.items()},
        native_rows=rows, report_path=report_path)

    receipt = run_shadow_nightly(source, private, _EVENT_DATE, handlers=handlers)

    assert receipt["status"] == "succeeded"
    stage_receipt = next(row for row in receipt["stages"]
                         if row["stage_id"] == "native_parity")
    assert stage_receipt["status"] == "succeeded"
    assert json.loads(report_path.read_text())["schema_version"] == "native_parity_report.v1.0"


# --------------------------------------------------------------------------
# Review fixes: fail-closed parity, registered stage, no-job stage, one switch.
# --------------------------------------------------------------------------

def _native_rows():
    return build_native_bundle_rows(_EMPTY_SCORE_DOC, _pairs())


def _differing_legacy(rows):
    return {key: {**row, "exp_pnl_model": (row.get("exp_pnl_model") or 0.0) + 1.0}
            for key, row in rows.items()}


@pytest.mark.parametrize("case", ["no_rows", "no_legacy", "no_native", "no_dimensions",
                                  "no_shared_key"])
def test_parity_report_refuses_an_empty_comparison(case):
    rows = _native_rows()
    legacy, native, dims = dict(rows), dict(rows), PARITY_DIMENSIONS
    if case == "no_rows":
        legacy, native = {}, {}
    elif case == "no_legacy":
        legacy = {}
    elif case == "no_native":
        native = {}
    elif case == "no_dimensions":
        dims = ()
    else:
        legacy = {"other-key": next(iter(rows.values()))}
    with pytest.raises(OpsError) as error:
        compare_native_vs_legacy(legacy, native, dims)
    assert error.value.code == "VALIDATION_FAILED"


def test_parity_dimensions_cover_all_five_field_groups():
    assert PARITY_DIMENSIONS == ("analogs", "financial_diagnostics", "forecasts",
                                 "simulation", "verdicts")


def test_parity_compares_forecasts_and_financial_diagnostics():
    rows = _native_rows()
    key = sorted(rows)[0]
    for field, dimension in (("driver_prediction", "forecasts"),
                             ("entry_cost_pct", "financial_diagnostics")):
        legacy = {key: {**rows[key], field: (rows[key].get(field) or 0.0) + 1.0}}
        report = compare_native_vs_legacy(legacy, rows, PARITY_DIMENSIONS)
        assert [item["dimension"] for item in report["mismatches"]] == [dimension], field
        assert field in report["mismatches"][0]["finding_fields"]


def _all_but_parity():
    return {stage: (lambda value, stage=stage: {**value, stage: "ok"})
            for stage in GRAPH if stage != "native_parity"}


def test_legacy_mode_nightly_still_succeeds_with_the_registered_parity_stage(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    plan = build_nightly_plan(Path.cwd(), _EVENT_DATE, shadow_serving_scorer="legacy")
    receipt = run_shadow_nightly(source, tmp_path / "private", _EVENT_DATE,
                                 handlers=_all_but_parity(), plan=plan)
    assert receipt["status"] == "succeeded"
    stage = next(row for row in receipt["stages"] if row["stage_id"] == "native_parity")
    assert stage["status"] == "succeeded" and stage["error_code"] is None
    assert not (tmp_path / "private" / "native_parity_report.json").exists()


def test_native_mode_parity_difference_is_reported_not_a_status_change(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = _native_rows()
    receipt = run_shadow_nightly(source, tmp_path / "private", _EVENT_DATE,
                                 handlers=_all_but_parity(), plan=dict(_NATIVE_PLAN),
                                 parity_rows=(_differing_legacy(rows), rows))
    assert receipt["status"] == "succeeded"
    report = json.loads((tmp_path / "private" / "native_parity_report.json").read_text())
    assert report["compared"] and report["mismatches"]


def test_native_mode_without_rows_degrades_only_the_optional_parity_stage(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    receipt = run_shadow_nightly(source, tmp_path / "private", _EVENT_DATE,
                                 handlers=_all_but_parity(), plan=dict(_NATIVE_PLAN))
    degraded = [row for row in receipt["stages"] if row["status"] == "degraded"]
    assert [row["stage_id"] for row in degraded] == ["native_parity"]
    assert degraded[0]["error_code"] == "OpsError"


def test_native_parity_is_never_a_job_request():
    from engine.v2.ops.nightly import build_legacy_job_requests
    plan = build_nightly_plan(Path.cwd(), _EVENT_DATE)
    assert "native_parity" in plan["order"]
    requests = build_legacy_job_requests(plan, tickers=("AAA",), year_start=2024,
                                         year_end=2024, include_prerequisites=True)
    kinds = {request.job.kind for request in requests}
    assert "legacy_native_parity" not in kinds and "native_parity" not in kinds
    assert not any(request.idempotency_key.endswith(":native_parity") for request in requests)


def test_one_switch_source_of_truth_for_both_halves():
    from engine.v2.contracts import serving as contracts_serving
    assert ops_shadow.SHADOW_SERVING_SCORERS is contracts_serving.SHADOW_SERVING_SCORERS
    assert serving_shadow.SHADOW_SERVING_SCORERS is contracts_serving.SHADOW_SERVING_SCORERS
    assert contracts_serving.DEFAULT_SHADOW_SERVING_SCORER == "native"
    assert native_shadow_serving_mode({}) == "native"
    with pytest.raises(OpsError) as ops_error:
        native_shadow_serving_mode({"shadow_serving_scorer": "bogus"})
    assert ops_error.value.code == "INVALID_REQUEST"
    with pytest.raises(NativeShadowConfigError) as serving_error:
        serving_shadow.shadow_serving_row_source({"shadow_serving_scorer": "bogus"},
                                                 _EMPTY_SCORE_DOC, {}, {})
    assert serving_error.value.code == "INVALID_REQUEST"


def test_phase4_rebinding_of_compare_records_reaches_compare_dimension(monkeypatch):
    class Rebound(Exception):
        pass

    def rebound(*args, **kwargs):
        raise Rebound

    monkeypatch.setattr(phase4_real, "compare_records", rebound)
    with pytest.raises(Rebound):
        phase4_real._compare_dimension({}, {}, "forecasts")


def _tool_args(tmp_path, plan=None, native_inputs=None):
    import argparse
    plan_path = None
    if plan is not None:
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan))
    return argparse.Namespace(nightly_plan=plan_path, native_inputs=native_inputs)


def test_dashboard_project_serves_native_rows_through_the_seam(tmp_path):
    from engine.v2.foundation import to_document
    from tools import v2_dashboard_project as tool
    legacy_rows = {"AAA": [{"ticker": "AAA"}]}
    assert tool._shadow_rows(_tool_args(tmp_path), _EMPTY_SCORE_DOC, legacy_rows) is legacy_rows
    assert tool._shadow_rows(_tool_args(tmp_path, {"shadow_serving_scorer": "legacy"}),
                             _EMPTY_SCORE_DOC, legacy_rows) is legacy_rows
    with pytest.raises(OpsError) as error:
        tool._shadow_rows(_tool_args(tmp_path, dict(_NATIVE_PLAN)), _EMPTY_SCORE_DOC, legacy_rows)
    assert error.value.code == "INVALID_REQUEST"
    pairs = _pairs()
    inputs_path = tmp_path / "native_inputs.json"
    inputs_path.write_text(json.dumps({key: {"request": to_document(request),
                                             "inputs": to_document(inputs)}
                                       for key, (request, inputs) in pairs.items()},
                                      default=str))
    served = tool._shadow_rows(_tool_args(tmp_path, dict(_NATIVE_PLAN), inputs_path),
                               _EMPTY_SCORE_DOC, legacy_rows)
    expected = serving_shadow.shadow_serving_row_source(_NATIVE_PLAN, _EMPTY_SCORE_DOC,
                                                        legacy_rows, pairs)
    assert served == expected and served != legacy_rows
