"""The real score stage and serving bridge share one population verdict."""
from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.contracts import EventRef
from engine.v2.foundation.score_population import population_difference, population_key
from engine.v2.ops import legacy_adapter
from engine.v2.ops.errors import OpsError
from engine.v2.serving.bridge import build_bridges
from engine.v2.serving.legacy_bundle import load_score_document

BASE = "AAA|STR-THRU|2026-01-15"
CHOOSER = "AAA|DYN-SV|2026-01-15"


def _row(key):
    ticker, strategy, event_date = key.split("|")
    return dict(ticker=ticker, strategy=strategy, event_date=event_date,
                as_of="2026-01-14", strike=None, expiry=None,
                strike_offset=None, chosen_strategy="STR-THRU")


def _bundle(rows):
    bundle = {}
    for row in rows:
        display = {**row, "row_id": population_key(row), "digest": "synthetic",
                   "scored": False, "rank": None}
        bundle.setdefault(row["ticker"], []).append(display)
    return bundle


def _bridge(document, bundle=None):
    refs = {(row["ticker"], row["event_date"]): EventRef(
        event_id=f"event:{row['ticker']}:{row['event_date']}", calendar_revision="calendar")
        for row in document["rows"]}
    return build_bridges(
        document, _bundle(document["rows"]) if bundle is None else bundle, refs,
        score_batch_ref="batch", snapshot_ref="snapshot",
        model_registry_artifact_refs=(), request_provenance_refs=())


def _score_stage(monkeypatch, root, planned, rows):
    """Stub source/scorer I/O, never either population check or score writer."""
    import engine.dashboard.nightly as nightly
    import engine.features as features
    import engine.score as score

    monkeypatch.setattr(features.FeatureContext, "load", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(score, "Scorer", lambda **k: SimpleNamespace(analog_entry_coverage=1.0))
    monkeypatch.setattr(score, "score_calendar", lambda *a, **k: pd.DataFrame(rows))
    monkeypatch.setattr(nightly, "strike_ladder", lambda *a, **k: [])
    monkeypatch.setattr(legacy_adapter, "_check_features_current", lambda *a, **k: None)
    (root / "finality.json").write_text(json.dumps({
        "date": "2026-01-14", "is_final": True, "market_wide": True,
        "daily_share": 1.0, "chain_share": 1.0, "covered": 1, "detail": "synthetic"}))
    return legacy_adapter._action_score({
        "tickers": ["AAA"], "year_start": 2026, "year_end": 2026,
        "session": "2026-01-14", "expected_population": planned}, root)


@pytest.mark.parametrize("planned, observed, missing, unplanned", [
    ([BASE], [BASE], [], []),
    ([BASE], [BASE, CHOOSER], [], []),
    ([BASE], [BASE, "BBB|DYN-SV|2026-01-15"], [], ["BBB|DYN-SV|2026-01-15"]),
    ([BASE], [BASE, "AAA|DYN-SV|2026-01-16"], [], ["AAA|DYN-SV|2026-01-16"]),
    ([BASE], [BASE, "AAA|STR-RUNUP|2026-01-15"], [], ["AAA|STR-RUNUP|2026-01-15"]),
    ([BASE, CHOOSER], [BASE], [CHOOSER], []),
    ([BASE], [CHOOSER], [BASE], []),
    ([CHOOSER], [CHOOSER, BASE], [], [BASE]),
], ids=["exact", "derived-chooser", "unplanned-ticker", "unplanned-date",
        "unplanned-strategy", "missing-explicit-chooser", "missing-base", "chooser-only-plan"])
def test_score_stage_and_real_bridge_agree(monkeypatch, tmp_path, planned, observed,
                                           missing, unplanned):
    rows = [_row(key) for key in observed]
    if missing or unplanned:
        with pytest.raises(OpsError) as raised:
            _score_stage(monkeypatch, tmp_path, planned, rows)
        assert raised.value.code == "VALIDATION_FAILED"
        assert raised.value.problem.details == {"missing": missing, "unplanned": unplanned}
        assert not (tmp_path / "score.json").exists()
        # The rejected producer rows are the bridge's identical synthetic input.
        (tmp_path / "score.json").write_text(json.dumps({
            "rows": rows, "ladder": [], "expected_population": planned}))
    else:
        _score_stage(monkeypatch, tmp_path, planned, rows)
    document = load_score_document(tmp_path / "score.json")
    before = deepcopy(document)
    bridges, findings = _bridge(document)
    assert document == before
    assert len(bridges) == len(observed)
    assert findings.ok == (not missing and not unplanned)
    assert [(finding.code, finding.details["population_key"]) for finding in findings.findings] == [
        *[("PLANNED_ROW_MISSING", key) for key in missing],
        *[("SCORED_ROW_UNPLANNED", key) for key in unplanned],
    ]
    assert _bridge(document) == (bridges, findings)


def test_chooser_exception_keeps_other_real_bridge_checks(tmp_path):
    rows = [_row(BASE), _row(CHOOSER)]
    document = {"rows": rows, "ladder": [], "expected_population": [BASE]}
    path = tmp_path / "score.json"
    path.write_text(json.dumps(document))
    bundle = _bundle(rows)
    bundle["AAA"][1]["chosen_strategy"] = "STR-RUNUP"
    _, findings = _bridge(load_score_document(path), bundle)
    assert not findings.ok
    assert [(finding.code, finding.field_name) for finding in findings.findings] == [
        ("VALUE_MISMATCH", "chosen_strategy")]


def test_shared_rule_is_deterministic_read_only_and_conservative():
    planned = [BASE, "ZZZ|STR-THRU|2026-01-15", CHOOSER]
    observed = [BASE, "ZZZ|DYN-SV|2026-01-15|extra", "BBB|DYN-SV|2026-01-15"]
    before = deepcopy((planned, observed))
    expected = ([CHOOSER, "ZZZ|STR-THRU|2026-01-15"],
                ["BBB|DYN-SV|2026-01-15", "ZZZ|DYN-SV|2026-01-15|extra"])
    assert population_difference(planned, observed) == expected
    assert population_difference(reversed(planned), reversed(observed)) == expected
    assert (planned, observed) == before
    assert population_key({"ticker": "AAA", "strategy": "STR-THRU",
                           "event_date": "2026-01-15", "strike": 99, "expiry": "later"}) == BASE
    assert population_key({"ticker": None}) == "None||"  # Preserve existing encoding.


@pytest.mark.parametrize("extra, accepted", [(CHOOSER, True), ("BBB|DYN-SV|2026-01-15", False)])
def test_real_candidate_keeps_unplanned_refusal(tmp_path, extra, accepted):
    from datetime import datetime

    from engine.v2.contracts import ObjectRef, PreviewInput, PreviewRelease, Problem
    from engine.v2.data.repository import Repository
    from engine.v2.foundation import ArtifactStore
    from engine.v2.serving import projections
    from tests.data_scan_support import (
        catalog_and_store,
        commit_tables,
        contract_for,
        contract_ref_for,
        publish_and_inspect,
    )

    catalog, clock, source_store = catalog_and_store(tmp_path)
    contract = contract_for("earnings_events")
    event_date = datetime(2026, 1, 15)
    event_rows = [dict(event_id=f"evt-{ticker}", ticker=ticker, event_date=event_date,
                       year=2026, session="BMO", session_src="synthetic", src_orats=True,
                       src_oquants=False, src_nasdaq=False, src_yfinance=False,
                       date_agree=True, date_conflict=False) for ticker in ("AAA", "BBB")]
    fragment = publish_and_inspect(source_store, contract, contract_ref_for(contract), event_rows, "2026")
    snapshot = commit_tables(catalog, clock, {"earnings_events": [fragment]}, {"earnings_events": contract})
    rows = [_row(BASE), _row(extra)]
    path = tmp_path / "score.json"
    path.write_text(json.dumps({"rows": rows, "ladder": [], "expected_population": [BASE]}))
    preview = PreviewInput(
        source_release_id="source", source_release_manifest_ref="manifest",
        snapshot_ref=snapshot.snapshot_id,
        legacy_snapshot_object_ref=ObjectRef(kind="legacy_snapshot", object_id="source",
                                           content_hash="sha256:" + "1" * 64, byte_size=1),
        score_batch_ref="batch", score_job_input_refs=(), bundle_manifest_ref="bundle",
        model_registry_artifact_refs=(), finality_ref="finality", expected_population_ref="planned",
        score_comparison_receipt_ref="score-comparison", render_comparison_receipt_ref="render-comparison",
        source_code_hash="sha256:" + "2" * 64, source_environment_hash="sha256:" + "3" * 64)
    conn = projections.connect(str(tmp_path / "serving.sqlite"))
    try:
        result = projections.build_candidate(
            preview, load_score_document(path), _bundle(rows), repository=Repository(catalog, source_store),
            snapshot_ref=snapshot, store=ArtifactStore(tmp_path / "serving-objects"), conn=conn,
            requested_as_of="2026-01-14", resolved_as_of="2026-01-14", clock=clock)
        if accepted:
            assert isinstance(result, PreviewRelease)
            assert projections.get_release(conn, result.release_id) == result
        else:
            assert isinstance(result, Problem)
            assert result.code == "PROJECTION_REFUSED"
            assert [finding["code"] for finding in result.details["findings"]["findings"]] == [
                "SCORED_ROW_UNPLANNED"]
            assert projections.get_release(conn, result.details["release_id"]) is None
    finally:
        conn.close()
        catalog.close()
