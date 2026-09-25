"""S9H end to end: the analog stage's selected row ids reach the serving index.

The analog stage emits ``selected_row_ids``/``contributing_row_ids`` only on
``StageObservation.display_document`` -- the display channel ``_emit_stage``
never hashes. This test scores the real ``_bundle`` fixture, takes the ids off
that channel through ``analog_display_fields``, merges them onto the row the
real ``render_bundle`` renders, reads the bundle back with
``load_legacy_bundle`` and builds one serving candidate through the real bridge
and ``projections`` index. The stored summary row must carry the ids the
analog stage actually selected, in order, while the score/engine row does not
and every stage receipt stays byte-identical to a run whose display channel
was stripped.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime

import pandas as pd

from engine.dashboard.render import render_bundle
from engine.v2.contracts import PreviewRelease
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, to_document
from engine.v2.scoring import application
from engine.v2.scoring.source_inputs import build_native_score_inputs
from engine.v2.scoring.stages import analog_display_fields
from engine.v2.serving import projections
from engine.v2.serving.legacy_bundle import load_legacy_bundle
from tests.test_v2_scoring_source_inputs import _bundle as _source_bundle, _request
from tests.test_v2_serving_projections import (
    _event_row,
    _events_snapshot,
    _preview_input,
    _score_doc,
)

EVENT_DATE = "2026-09-16"


def _score(monkeypatch, *, strip_display: bool):
    """Score the fixed source bundle, optionally with the matcher's ids stripped."""
    if strip_display:
        import engine.v2.scoring.native_analog as native_analog

        real = native_analog.evaluate_analogs

        def without_ids(**kwargs):
            return replace(real(**kwargs), selected_row_ids=(),
                           contributing_row_ids=())

        monkeypatch.setattr(native_analog, "evaluate_analogs", without_ids)
    observations = []
    record = application.score_one(
        _request(0.5), build_native_score_inputs(_source_bundle()),
        observer=observations.append)
    return record, observations


def _stage_hashes(record) -> tuple:
    return tuple(
        (item["stage"], item["input_hash"], item["output_hash"])
        for item in record.resolved_request["native_stage_receipts"])


def _served_release(tmp_path, score_doc, bundle_rows_by_ticker):
    (tmp_path / "phase2").mkdir()
    conn, store, snap = _events_snapshot(
        tmp_path / "phase2", [_event_row("e1", "TEST", datetime(2026, 9, 16))],
        year="2026")
    serving_root = tmp_path / "serving"
    serving_root.mkdir()
    serving_store = ArtifactStore(serving_root / "objects")
    serving_conn = projections.connect(str(serving_root / "serving.sqlite"))
    release = projections.build_candidate(
        _preview_input(), score_doc, bundle_rows_by_ticker,
        repository=Repository(conn, store), snapshot_ref=snap,
        store=serving_store, conn=serving_conn,
        requested_as_of=EVENT_DATE, resolved_as_of=EVENT_DATE)
    return release, serving_conn


def test_analog_ids_reach_the_serving_index_without_moving_stage_hashes(tmp_path, monkeypatch):
    # The ids-present run first, then the stripped control: the golden pattern
    # is that the display channel cannot move any hashed stage payload.
    record, observations = _score(monkeypatch, strip_display=False)
    stripped, stripped_observations = _score(monkeypatch, strip_display=True)

    ids = analog_display_fields(observations)
    assert ids == {"selected_row_ids": ("a1", "a2"),
                   "contributing_row_ids": ("a1", "a2")}
    assert analog_display_fields(stripped_observations) == {
        "selected_row_ids": (), "contributing_row_ids": ()}
    assert _stage_hashes(record) == _stage_hashes(stripped)
    assert record.score_id == stripped.score_id

    # The score artifact row (hashed into the bridge's score_id) carries no ids.
    engine_row = to_document(dict(record.resolved_request)) | {"strike_offset": None}
    assert not set(ids) & set(engine_row)
    display_row = {**engine_row,
                   **{name: list(value) for name, value in ids.items()}}

    # The real renderer writes the display row into the bundle; the real
    # bundle reader is what the bridge joins against.
    bundle_root = tmp_path / "bundle"
    render_bundle(pd.DataFrame([display_row]), bundle_root,
                  as_of=pd.Timestamp(EVENT_DATE))
    bundle_rows_by_ticker, _manifest = load_legacy_bundle(bundle_root)
    rendered = bundle_rows_by_ticker["TEST"][0]
    assert rendered["selected_row_ids"] == ["a1", "a2"]
    assert rendered["contributing_row_ids"] == ["a1", "a2"]

    release, serving_conn = _served_release(
        tmp_path, _score_doc(rows=[engine_row]), bundle_rows_by_ticker)
    assert isinstance(release, PreviewRelease)
    try:
        stored = serving_conn.execute(
            "SELECT selected_row_ids, contributing_row_ids FROM serving_score_summary "
            "WHERE release_id = ?", (release.release_id,)).fetchone()
    finally:
        serving_conn.close()

    assert json.loads(stored["selected_row_ids"]) == ["a1", "a2"]
    assert json.loads(stored["contributing_row_ids"]) == ["a1", "a2"]
