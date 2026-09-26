"""S9H: native history-analogs view over the serving summary rows.

``analog_document`` is the whole body ``GET /analogs.json`` serves, so these
are the pure-document controls: the available branch reads every score/
strategy row's persisted ids and count straight off ``serving_score_summary``
(never recomputing a count from the ids), ordered deterministically by
strategy then ``score_id`` and optionally filtered by strategy; an unknown
pair is refused with a named reason code rather than an empty success. HTTP
reachability through the preview launcher is covered by
``tests/test_v2_dashboard_preview_analogs.py``.
"""
from __future__ import annotations

import json
from http import HTTPStatus

from engine.v2.serving import projections
from engine.v2.serving.analog_projection import (
    ANALOG_VIEW_SCHEMA_V1,
    EVENT_NOT_FOUND,
    analog_document,
)


def _summary(conn, *, release_id="r1", score_id="s1", event_id="e1",
             strategy="STR-THRU", selected=("row-a", "row-b"),
             contributing=("row-a",), n_analogs=2):
    conn.execute(
        "INSERT OR IGNORE INTO serving_release "
        "(release_id, document_json, findings_json, status, written_at) "
        "VALUES (?, '{}', '{}', 'candidate', '2024-01-01T00:00:00Z')", (release_id,))
    conn.execute(
        "INSERT OR IGNORE INTO serving_object (artifact_id, ref_json) VALUES ('obj-1', '{}')")
    conn.execute(
        "INSERT INTO serving_score_summary "
        "(release_id, score_id, event_id, strategy, selected_row_ids, "
        "contributing_row_ids, n_analogs, detail_artifact_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'obj-1')",
        (release_id, score_id, event_id, strategy, json.dumps(list(selected)),
         json.dumps(list(contributing)), n_analogs))


def _conn(tmp_path):
    return projections.connect(str(tmp_path / "serving.sqlite"))


def test_analog_document_available_reads_the_persisted_row(tmp_path):
    conn = _conn(tmp_path)
    try:
        _summary(conn, selected=("row-a", "row-b"), contributing=("row-a",), n_analogs=7)
        status, body = analog_document(conn, None, "r1", "e1")
    finally:
        conn.close()

    assert status == HTTPStatus.OK
    assert body == {
        "schema_version": ANALOG_VIEW_SCHEMA_V1,
        "status": "available",
        "release_id": "r1",
        "event_id": "e1",
        "analogs": [{
            "score_id": "s1",
            "strategy": "STR-THRU",
            "n_analogs": 7,  # the row's own count, never len(selected_row_ids)
            "selected_row_ids": ["row-a", "row-b"],
            "contributing_row_ids": ["row-a"],
        }],
    }
    assert not ({"exp_pnl_analog", "win_analog", "ci_low", "ci_high"} & set(body))
    assert not ({"exp_pnl_analog", "win_analog", "ci_low", "ci_high"}
                & set(body["analogs"][0]))


def test_every_score_and_strategy_of_the_event_is_returned_ordered(tmp_path):
    """No ``LIMIT 1``: two BWB rows and one STR-THRU row all come back, in
    strategy-then-score-id order regardless of insert order."""
    conn = _conn(tmp_path)
    try:
        _summary(conn, score_id="s9", strategy="STR-THRU", n_analogs=1)
        _summary(conn, score_id="s2", strategy="BWB", n_analogs=2)
        _summary(conn, score_id="s1", strategy="BWB", n_analogs=3)
        status, body = analog_document(conn, None, "r1", "e1")
    finally:
        conn.close()

    assert status == HTTPStatus.OK
    assert [(row["strategy"], row["score_id"], row["n_analogs"])
            for row in body["analogs"]] == [
        ("BWB", "s1", 3), ("BWB", "s2", 2), ("STR-THRU", "s9", 1)]


def test_strategy_filter_selects_only_that_strategys_rows(tmp_path):
    conn = _conn(tmp_path)
    try:
        _summary(conn, score_id="s9", strategy="STR-THRU", n_analogs=1)
        _summary(conn, score_id="s2", strategy="BWB", n_analogs=2)
        _summary(conn, score_id="s1", strategy="BWB", n_analogs=3)
        status, body = analog_document(conn, None, "r1", "e1", "BWB")
        missing_status, missing_body = analog_document(conn, None, "r1", "e1", "NOPE")
    finally:
        conn.close()

    assert status == HTTPStatus.OK
    assert [row["score_id"] for row in body["analogs"]] == ["s1", "s2"]
    assert missing_status == HTTPStatus.NOT_FOUND
    assert missing_body["reason_code"] == EVENT_NOT_FOUND


def test_null_n_analogs_stays_null_and_a_real_zero_stays_zero(tmp_path):
    conn = _conn(tmp_path)
    try:
        _summary(conn, score_id="s1", n_analogs=None)
        _summary(conn, score_id="s2", strategy="BWB", n_analogs=0)
        status, body = analog_document(conn, None, "r1", "e1")
    finally:
        conn.close()

    assert status == HTTPStatus.OK
    assert [(row["score_id"], row["n_analogs"]) for row in body["analogs"]] == [
        ("s2", 0), ("s1", None)]


def test_analog_document_unknown_event_refuses(tmp_path):
    conn = _conn(tmp_path)
    try:
        _summary(conn)
        status, body = analog_document(conn, None, "r1", "missing")
    finally:
        conn.close()

    assert status == HTTPStatus.NOT_FOUND
    assert body == {
        "schema_version": ANALOG_VIEW_SCHEMA_V1,
        "status": "refused",
        "reason_code": EVENT_NOT_FOUND,
    }


def test_analog_document_event_under_another_release_refuses(tmp_path):
    conn = _conn(tmp_path)
    try:
        _summary(conn, release_id="r1")
        status, body = analog_document(conn, None, "r2", "e1")
    finally:
        conn.close()

    assert status == HTTPStatus.NOT_FOUND
    assert body["reason_code"] == EVENT_NOT_FOUND
