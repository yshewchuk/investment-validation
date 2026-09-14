"""P3-1a: serving contracts and the offline legacy score bridge.

Synthetic ``score.json`` rows and a synthetic rendered bundle with the real
key shapes (`engine/dashboard/render.py` ``_BOARD_FIELDS``/``compact_row``,
`engine/v2/ops/legacy_adapter.py` ``_action_score``/``_population_key``) --
never the real renderer or a real panel. ``engine/v2/serving/bridge.py`` may
not import legacy `engine.*` or `engine.v2.ops`, so this file builds its own
tiny stand-in for ``compact_row`` rather than calling the real one.

* **L03** -- strict contract round trips; malformed refs/unsupported schema
  versions fail; the bridge's ``score_id`` is deterministic and excludes
  nothing operational while geometry/model/snapshot changes do change it.
* **L04** -- mapping covers the planned population with separate ladder
  counts; an unmatched event, a duplicate key and an omitted refusal each
  produce independent findings in one receipt; an empty compared population
  cannot be ok.
* **L05** -- ``engine_record`` keeps full precision, ``display_record`` keeps
  the rendered values; rounded-geometry-as-join-key, null-to-zero, wrong
  units and an altered DYN-SV choice each fail locally with their own finding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts import (  # noqa: E402
    EventPage,
    EventPageItem,
    EventRef,
    EventScoreSummary,
    Finding,
    LegacyScoreBridge,
    ObjectRef,
    PreviewCapabilities,
    PreviewInput,
    PreviewRelease,
    ProjectionFindings,
    RowIdentity,
)
from engine.v2.foundation import DocumentError, from_document, to_document  # noqa: E402
from engine.v2.serving.bridge import LEGACY_DISPLAY_MAPPING_V1, build_bridges  # noqa: E402

EVENT_REF = EventRef(event_id="evt_1", calendar_revision="cal_1")
EVENT_REFS = {("AAPL", "2026-01-15"): EVENT_REF}


# --------------------------------------------------------------------------
# synthetic fixtures -- real key shapes, no legacy import
# --------------------------------------------------------------------------


def _row(*, ticker="AAPL", strategy="STR-THRU", event_date="2026-01-15",
          strike=100.0, expiry="2026-01-16", strike_offset=None, **overrides):
    """One ``ScoreResult.as_dict()``-shaped row, as ``score.json`` writes it."""
    base = dict(
        ticker=ticker, strategy=strategy, event_date=event_date, as_of="2026-01-14",
        strike=strike, expiry=expiry, strike_offset=strike_offset,
        requested_strike=strike, spot=99.5, entry_cost=1.234567891234,
        entry_cost_pct=1.24, model_fair_pct=1.1, premium_vs_fair=1.13,
        exp_pnl_model=0.05, win_model=0.55, model_p10=-0.1, model_p90=0.2,
        exp_pnl_analog=None, win_analog=None, ci_low=None, ci_high=None,
        n_analogs=0, analog_widened=0, gate_score=0.9, gate_threshold=0.5,
        gate_pass=True, legs=[{"right": "P", "strike": strike, "side": "buy", "qty": 1}],
        structure_params={"width_legs": 5.0}, structure_width=5.0,
        cost_over_width=0.25, rel_spread=0.05, model_versions={"m": "v1"},
        flags=[], extrapolated=False, driver_name="abs_move",
        driver_prediction=6.0, implied_move=9.0, model_vs_market=0.43,
        fill=0.5, detail="ok", quote_date="2026-01-13", quote_age_sessions=1,
        quote_max_age_sessions=3, dte_entry=2, entry_date="2026-01-14",
        exit_date="2026-01-16",
    )
    base.update(overrides)
    return base


def _compact(row, **overrides):
    """A ``compact_row``-shaped display row: the same field names, the
    renderer's own rounding (6dp, `REPLAY_INPUT_FIELDS` exempt), plus the
    renderer-only computed keys (``row_id``, ``digest``, ``scored``, ``rank``).
    """
    exact = {"structure_params", "requested_strike", "strike", "strike_offset",
             "fill", "quote_max_age_sessions"}
    display = {k: (v if k in exact or not isinstance(v, float) else round(v, 6))
               for k, v in row.items()}
    display["row_id"] = "|".join(str(row.get(k)) for k in
                                  ("ticker", "strategy", "event_date", "strike"))
    display["digest"] = "sha256:" + "0" * 64
    display["scored"] = row.get("exp_pnl_model") is not None or row.get("exp_pnl_analog") is not None
    display["rank"] = 1
    display["payoff_curve"] = {"x": [1.0], "y": [2.0], "max": 2.0, "min": 0.0,
                                "strikes": [], "shape": "twin"}
    display.update(overrides)
    return display


def _score_doc(rows=(), ladder=(), expected=None):
    rows = list(rows)
    expected = expected if expected is not None else sorted({
        f"{r['ticker']}|{r['strategy']}|{r['event_date']}" for r in rows})
    return {"expected_population": expected, "rows": rows, "ladder": list(ladder)}


def _bundle(*rows):
    by_ticker: dict[str, list[dict]] = {}
    for row in rows:
        by_ticker.setdefault(str(row["ticker"]), []).append(row)
    return by_ticker


def _kwargs(**overrides):
    kwargs = dict(score_batch_ref="batch_1", snapshot_ref="snap_1",
                  model_registry_artifact_refs=("model_1",),
                  request_provenance_refs=("req_1",))
    kwargs.update(overrides)
    return kwargs


# --------------------------------------------------------------------------
# L03: contract round trips, malformed input, schema versions, score_id
# --------------------------------------------------------------------------


def _preview_input():
    obj = ObjectRef(kind="legacy_snapshot", object_id="o1",
                     content_hash="sha256:" + "1" * 64, byte_size=10)
    return PreviewInput(
        source_release_id="rel_1", source_release_manifest_ref="m1",
        snapshot_ref="snap_1", legacy_snapshot_object_ref=obj,
        score_batch_ref="batch_1", score_job_input_refs=("in_1",),
        bundle_manifest_ref="bm_1", model_registry_artifact_refs=("model_1",),
        finality_ref="fin_1", expected_population_ref="pop_1",
        score_comparison_receipt_ref="sc_1", render_comparison_receipt_ref="rc_1",
        source_code_hash="sha256:" + "2" * 64, source_environment_hash="sha256:" + "3" * 64,
    )


def _preview_release():
    return PreviewRelease(
        release_id="rel_2", source_release_id="rel_1", projection_manifest_ref="pm_1",
        snapshot_ref="snap_1", score_batch_ref="batch_1", bundle_manifest_ref="bm_1",
        model_registry_artifact_refs=("model_1",), comparison_receipt_refs=("cr_1",),
        source_code_hash="h1", projection_code_hash="h2",
        requested_as_of="2026-01-14", resolved_as_of="2026-01-14", clock_ids=("c1",),
        coverage_summary={"planned": 1.0}, stale_or_degraded_reasons=(),
    )


def _bridge_sample():
    return LegacyScoreBridge(
        score_id="sha256:" + "4" * 64, event_ref=EVENT_REF, clock_id="2026-01-14",
        legacy_row_id="AAPL|STR-THRU|2026-01-15|100.0|2026-01-16",
        score_batch_ref="batch_1", source_row_key="AAPL|STR-THRU|2026-01-15",
        source_record_hash="sha256:" + "5" * 64, request_provenance_refs=("req_1",),
        snapshot_ref="snap_1", model_registry_artifact_refs=("model_1",),
        engine_record={"strike": 100.0}, display_record={"strike": 100.0},
    )


def _event_page_sample():
    summary = EventScoreSummary(score_id="sc_1", strategy="STR-THRU", verdict="TRADE")
    item = EventPageItem(event_ref=EVENT_REF, ticker="AAPL", event_date="2026-01-15",
                          clock_id="2026-01-14", readiness="ready", scores=(summary,))
    return EventPage(release_id="rel_2", query_hash="qh_1", items=(item,), total_matching=1)


@pytest.mark.parametrize("sample", [
    _preview_input(), _preview_release(), _bridge_sample(), _event_page_sample(),
    RowIdentity(ticker="AAPL", event_date="2026-01-15", strategy="STR-THRU"),
    Finding(code="X", category="missing", message="m"),
    ProjectionFindings(planned_population=1, rendered_main_population=1,
                       rendered_ladder_population=0, matched_population=1,
                       compared_population=1, findings=(), ok=True),
    PreviewCapabilities(read=True, submit_jobs=False, collect_live=False),
], ids=lambda s: type(s).__name__)
def test_l03_every_new_contract_round_trips_exactly(sample):
    assert from_document(type(sample), to_document(sample)) == sample


def test_l03_unknown_field_is_refused():
    doc = to_document(_bridge_sample())
    doc["not_a_real_field"] = 1
    with pytest.raises(DocumentError) as err:
        from_document(LegacyScoreBridge, doc)
    assert err.value.code == "UNKNOWN_FIELD"


def test_l03_required_ref_cannot_be_null():
    doc = to_document(_preview_input())
    doc["legacy_snapshot_object_ref"] = None
    with pytest.raises(DocumentError):
        from_document(PreviewInput, doc)


def test_l03_wrong_major_schema_version_is_refused():
    doc = to_document(_bridge_sample())
    doc["schema_version"] = "legacy_score_bridge.v2.0"
    with pytest.raises(DocumentError) as err:
        from_document(LegacyScoreBridge, doc)
    assert err.value.code == "UNSUPPORTED_VERSION"


def test_l03_newer_minor_schema_version_is_refused():
    doc = to_document(_bridge_sample())
    doc["schema_version"] = "legacy_score_bridge.v1.9"
    with pytest.raises(DocumentError) as err:
        from_document(LegacyScoreBridge, doc)
    assert err.value.code == "UNSUPPORTED_VERSION"


def test_l03_score_id_is_deterministic():
    row = _row()
    score_doc = _score_doc(rows=[row])
    bundle = _bundle(_compact(row))
    a, _ = build_bridges(score_doc, bundle, EVENT_REFS, **_kwargs())
    b, _ = build_bridges(score_doc, bundle, EVENT_REFS, **_kwargs())
    assert a[0].score_id == b[0].score_id


def test_l03_score_id_changes_with_geometry():
    row = _row()
    other = _row(strike=105.0)
    bridges_a, _ = build_bridges(_score_doc(rows=[row]), _bundle(_compact(row)),
                                  EVENT_REFS, **_kwargs())
    bridges_b, _ = build_bridges(_score_doc(rows=[other]), _bundle(_compact(other)),
                                  EVENT_REFS, **_kwargs())
    assert bridges_a[0].score_id != bridges_b[0].score_id


def test_l03_score_id_changes_with_model_ref():
    row = _row()
    score_doc, bundle = _score_doc(rows=[row]), _bundle(_compact(row))
    a, _ = build_bridges(score_doc, bundle, EVENT_REFS, **_kwargs(model_registry_artifact_refs=("model_1",)))
    b, _ = build_bridges(score_doc, bundle, EVENT_REFS, **_kwargs(model_registry_artifact_refs=("model_2",)))
    assert a[0].score_id != b[0].score_id


def test_l03_score_id_changes_with_snapshot_ref():
    row = _row()
    score_doc, bundle = _score_doc(rows=[row]), _bundle(_compact(row))
    a, _ = build_bridges(score_doc, bundle, EVENT_REFS, **_kwargs(snapshot_ref="snap_1"))
    b, _ = build_bridges(score_doc, bundle, EVENT_REFS, **_kwargs(snapshot_ref="snap_2"))
    assert a[0].score_id != b[0].score_id


# --------------------------------------------------------------------------
# L04: population mapping, ladder counts, independent findings
# --------------------------------------------------------------------------


def test_l04_full_population_maps_with_separate_ladder_counts():
    main_row = _row()
    ladder_row = _row(strike=102.5, strike_offset=2.5)
    score_doc = _score_doc(rows=[main_row], ladder=[ladder_row])
    bundle = _bundle(_compact(main_row), _compact(ladder_row))
    bridges, findings = build_bridges(score_doc, bundle, EVENT_REFS, **_kwargs())
    assert len(bridges) == 2
    assert findings.ok is True
    assert findings.rendered_main_population == 1
    assert findings.rendered_ladder_population == 1
    assert findings.matched_population == 2
    assert findings.compared_population == 2
    assert findings.planned_population == 1


def test_l04_unmatched_event_is_an_independent_finding():
    row = _row()
    bridges, findings = build_bridges(
        _score_doc(rows=[row]), _bundle(_compact(row)), {}, **_kwargs())
    assert findings.ok is False
    assert any(f.category == "unresolved_event" and f.code == "EVENT_UNMAPPED"
               for f in findings.findings)


def test_l04_ambiguous_event_is_an_independent_finding():
    row = _row()
    ambiguous_refs = {("AAPL", "2026-01-15"): None}
    _, findings = build_bridges(
        _score_doc(rows=[row]), _bundle(_compact(row)), ambiguous_refs, **_kwargs())
    assert findings.ok is False
    assert any(f.category == "unresolved_event" and f.code == "EVENT_AMBIGUOUS"
               for f in findings.findings)


def test_l04_duplicate_rendered_key_is_an_independent_finding():
    row = _row()
    bundle = _bundle(_compact(row), _compact(row))
    _, findings = build_bridges(_score_doc(rows=[row]), bundle, EVENT_REFS, **_kwargs())
    assert findings.ok is False
    assert any(f.category == "duplicate" for f in findings.findings)


def test_l04_duplicate_source_key_is_an_independent_finding():
    row = _row()
    score_doc = _score_doc(rows=[row, dict(row)])
    _, findings = build_bridges(score_doc, _bundle(_compact(row)), EVENT_REFS, **_kwargs())
    assert findings.ok is False
    assert any(f.category == "duplicate" and f.code == "SOURCE_ROW_DUPLICATE_KEY"
               for f in findings.findings)


def test_l04_omitted_refusal_and_unmatched_event_are_independent_in_one_receipt():
    """Two unrelated defects in one call: a declined row the render dropped,
    and a second event the caller could not map -- both must appear, neither
    masking the other (guide §5.3 point 7: "never stopping at the first")."""
    kept = _row(ticker="AAPL", strategy="STR-THRU")
    declined = _row(ticker="AAPL", strategy="STR-RUNUP", gate_pass=False,
                    exp_pnl_model=None, win_model=None)
    unmapped = _row(ticker="MSFT", strategy="STR-THRU", event_date="2026-02-01")
    score_doc = _score_doc(rows=[kept, declined, unmapped])
    bundle = _bundle(_compact(kept))  # declined + unmapped never rendered
    _, findings = build_bridges(score_doc, bundle, EVENT_REFS, **_kwargs())
    assert findings.ok is False
    categories = {f.category for f in findings.findings}
    assert "missing" in categories  # the declined row, dropped from the render
    assert "unresolved_event" in categories  # MSFT, never mapped


def test_l04_empty_compared_population_cannot_be_ok():
    _, findings = build_bridges(_score_doc(rows=[]), {}, {}, **_kwargs())
    assert findings.compared_population == 0
    assert findings.ok is False


def test_l04_dyn_sv_kept_as_its_own_result_with_its_choice():
    dyn_row = _row(strategy="DYN-SV", chosen_strategy="STR-THRU",
                    chosen_margin=0.31, menu_size=3)
    score_doc = _score_doc(rows=[dyn_row])
    bridges, findings = build_bridges(score_doc, _bundle(_compact(dyn_row)),
                                       EVENT_REFS, **_kwargs())
    assert findings.ok is True
    assert bridges[0].engine_record["chosen_strategy"] == "STR-THRU"
    assert bridges[0].engine_record["menu_size"] == 3


def test_l04_row_without_ticker_or_event_date_stops_instead_of_inventing_an_id():
    malformed = {"strategy": "STR-THRU"}
    with pytest.raises(ValueError, match="event_id"):
        build_bridges(_score_doc(rows=[malformed], expected=[]), {}, {}, **_kwargs())


# --------------------------------------------------------------------------
# L05: precision, display fidelity, and local per-field failures
# --------------------------------------------------------------------------


def test_l05_engine_record_keeps_full_precision():
    row = _row(entry_cost=1.234567891234)
    bridges, findings = build_bridges(
        _score_doc(rows=[row]), _bundle(_compact(row)), EVENT_REFS, **_kwargs())
    assert findings.ok is True
    assert bridges[0].engine_record["entry_cost"] == 1.234567891234


def test_l05_display_record_keeps_the_rendered_values():
    row = _row(entry_cost=1.234567891234)
    display = _compact(row)
    bridges, _ = build_bridges(
        _score_doc(rows=[row]), _bundle(display), EVENT_REFS, **_kwargs())
    assert bridges[0].display_record == display
    assert bridges[0].display_record["entry_cost"] == round(1.234567891234, 6)


def test_l05_rounded_geometry_used_as_a_join_key_fails_locally():
    row = _row(strike=100.123456789)
    # A render that rounded the join-key strike instead of keeping it exact.
    bad_display = _compact(row, strike=round(100.123456789, 6))
    score_doc = _score_doc(rows=[row])
    bridges, findings = build_bridges(score_doc, _bundle(bad_display), EVENT_REFS, **_kwargs())
    assert findings.ok is False
    assert not bridges
    assert any(f.category == "missing" for f in findings.findings)


def test_l05_null_turned_to_zero_fails_locally():
    row = _row(ci_low=None)
    bad_display = _compact(row, ci_low=0.0)
    _, findings = build_bridges(_score_doc(rows=[row]), _bundle(bad_display),
                                 EVENT_REFS, **_kwargs())
    assert findings.ok is False
    assert any(f.category == "null_mask" and f.field_name == "ci_low"
               for f in findings.findings)


def test_l05_wrong_units_fails_locally():
    row = _row(win_model=0.55)
    bad_display = _compact(row, win_model=55.0)  # fraction rendered as percent
    _, findings = build_bridges(_score_doc(rows=[row]), _bundle(bad_display),
                                 EVENT_REFS, **_kwargs())
    assert findings.ok is False
    assert any(f.category == "value" and f.field_name == "win_model"
               for f in findings.findings)


def test_l05_altered_dyn_sv_choice_fails_locally():
    row = _row(strategy="DYN-SV", chosen_strategy="STR-THRU", chosen_margin=0.2, menu_size=3)
    bad_display = _compact(row, chosen_strategy="STR-RUNUP")
    _, findings = build_bridges(_score_doc(rows=[row]), _bundle(bad_display),
                                 EVENT_REFS, **_kwargs())
    assert findings.ok is False
    assert any(f.category == "value" and f.field_name == "chosen_strategy"
               for f in findings.findings)


def test_l05_mapping_spec_covers_every_display_key_in_a_synthetic_compact_row():
    """Independent transcription of ``dashboard/render.py`` ``_BOARD_FIELDS``,
    so a typo in one copy (this test's or ``bridge.py``'s own) is caught
    rather than the spec trivially covering itself."""
    synthetic_compact_row_keys = {
        "row_id", "ticker", "strategy", "as_of", "event_date", "session",
        "entry_date", "exit_date", "strike", "strike_offset", "expiry",
        "quote_date", "quote_age_sessions", "quote_max_age_sessions", "requested_strike",
        "dte_entry", "spot", "entry_cost", "entry_cost_pct",
        "model_fair_pct", "premium_vs_fair",
        "exp_pnl_model", "win_model", "model_p10", "model_p90",
        "exp_pnl_analog", "win_analog", "ci_low", "ci_high",
        "n_analogs", "analog_widened",
        "gate_score", "gate_threshold", "gate_pass",
        "forecast_abs_move", "forecast_p10", "forecast_p90", "forecast_sd",
        "forecast_model", "forecast_fold", "structure_params",
        "structure_width", "cost_over_width", "rel_spread",
        "legs", "payoff_curve",
        "exp_pnl_sim", "win_sim",
        "chosen_strategy", "chosen_margin", "menu_size",
        "extrapolated", "flags", "model_versions",
        "driver_name", "driver_prediction", "driver_p10", "driver_p90",
        "runup_move_prediction", "runup_move_p10", "runup_move_p90",
        "runup_move_days", "runup_move_scale",
        "implied_move", "implied_move_at_entry", "model_vs_market", "model_input_as_of",
        "chain_last_obs", "chain_age_days",
        "scored", "rank", "fill", "detail", "digest",
    }
    spec_fields = {spec.field for spec in LEGACY_DISPLAY_MAPPING_V1}
    assert synthetic_compact_row_keys <= spec_fields
