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

import pytest

from checks.phase4_real import _numerical_independence_source, _request
from engine.v2.contracts import PreviewRelease
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore
from engine.v2.ops import native_shadow_render as ops_shadow
from engine.v2.ops.errors import OpsError
from engine.v2.ops.native_shadow_render import native_shadow_serving_mode
from engine.v2.ops.nightly import build_nightly_plan
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
from tests.test_v2_serving_projections import (
    _event_row,
    _events_snapshot,
    _preview_input,
)

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
