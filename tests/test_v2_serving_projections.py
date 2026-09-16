"""P3-1b: the minimal serving index, immutable detail objects, and the
offline projection coordinator — rearchitecture phase-3 guide §5.4.

A synthetic Phase 2 catalog (real ``earnings_events`` fragments, the same
technique ``tests/test_v2_data_events_chains.py`` uses) plus synthetic
``score.json``/render-bundle pairs shaped like ``tests/test_v2_serving_
bridge.py`` — never a real panel, a real renderer, or ``engine.v2.ops``.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts import (  # noqa: E402
    ObjectRef,
    PreviewInput,
    PreviewRelease,
    Problem,
)
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import ArtifactStore  # noqa: E402
from engine.v2.serving import projections  # noqa: E402
from tests.data_scan_support import (  # noqa: E402
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)

_EVENTS = contract_for("earnings_events")
_EVENTS_REF = contract_ref_for(_EVENTS)


# --------------------------------------------------------------------------
# synthetic Phase 2 catalog: real earnings_events fragments
# --------------------------------------------------------------------------


def _event_row(event_id: str, ticker: str, event_date: datetime) -> dict:
    return dict(event_id=event_id, ticker=ticker, event_date=event_date, year=event_date.year,
               session="BMO", session_src="orats", annc_tod=None, src_orats=True,
               src_oquants=True, src_nasdaq=False, src_yfinance=False, date_agree=True,
               date_conflict=False, updated_at=None, event_cluster_id=None,
               claim_count=None, reconciliation=None)


def _events_snapshot(tmp_path, rows, *, year="2024"):
    conn, clock, store = catalog_and_store(tmp_path)
    record = publish_and_inspect(store, _EVENTS, _EVENTS_REF, rows, year)
    snap = commit_tables(conn, clock, {"earnings_events": [record]}, {"earnings_events": _EVENTS})
    return conn, store, snap


# --------------------------------------------------------------------------
# synthetic score.json / render-bundle rows — the same shapes
# tests/test_v2_serving_bridge.py uses
# --------------------------------------------------------------------------


def _row(*, ticker="AAA", strategy="STR-THRU", event_date="2024-01-05",
         strike=100.0, expiry="2024-01-06", strike_offset=None, **overrides):
    base = dict(
        ticker=ticker, strategy=strategy, event_date=event_date, as_of="2024-01-04",
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
        fill=0.5, detail="ok", quote_date="2024-01-03", quote_age_sessions=1,
        quote_max_age_sessions=3, dte_entry=2, entry_date="2024-01-04",
        exit_date="2024-01-06",
    )
    base.update(overrides)
    return base


def _compact(row: dict, **overrides) -> dict:
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


def _score_doc(rows=(), ladder=(), expected=None) -> dict:
    rows = list(rows)
    expected = expected if expected is not None else sorted({
        f"{r['ticker']}|{r['strategy']}|{r['event_date']}" for r in rows})
    return {"expected_population": expected, "rows": rows, "ladder": list(ladder)}


def _bundle(*rows) -> dict:
    by_ticker: dict[str, list[dict]] = {}
    for row in rows:
        by_ticker.setdefault(str(row["ticker"]), []).append(row)
    return by_ticker


def _preview_input(**overrides) -> PreviewInput:
    obj = ObjectRef(kind="legacy_snapshot", object_id="o1",
                    content_hash="sha256:" + "1" * 64, byte_size=10)
    kwargs = dict(
        source_release_id="rel_1", source_release_manifest_ref="m1",
        snapshot_ref="snap_1", legacy_snapshot_object_ref=obj,
        score_batch_ref="batch_1", score_job_input_refs=("in_1",),
        bundle_manifest_ref="bm_1", model_registry_artifact_refs=("model_1",),
        finality_ref="fin_1", expected_population_ref="pop_1",
        score_comparison_receipt_ref="sc_1", render_comparison_receipt_ref="rc_1",
        source_code_hash="sha256:" + "2" * 64, source_environment_hash="sha256:" + "3" * 64)
    kwargs.update(overrides)
    return PreviewInput(**kwargs)


def _serving(tmp_path, name="serving"):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(root / "objects")
    conn = projections.connect(str(root / "serving.sqlite"))
    return conn, store


def _build(preview_input, score_doc, bundle, *, repository, snap, serving_conn, serving_store,
          requested_as_of="2024-01-04", resolved_as_of="2024-01-04", fault=None):
    return projections.build_candidate(
        preview_input, score_doc, bundle, repository=repository, snapshot_ref=snap,
        store=serving_store, conn=serving_conn, requested_as_of=requested_as_of,
        resolved_as_of=resolved_as_of, fault=fault)


# --------------------------------------------------------------------------
# resolve_event_refs — direct coverage of the resolver this task adds
# --------------------------------------------------------------------------


def test_resolve_event_refs_maps_a_known_pair(tmp_path):
    conn, store, snap = _events_snapshot(tmp_path, [_event_row("e1", "AAA", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    result = projections.resolve_event_refs(repo, snap, {("AAA", "2024-01-05")})
    ref = result[("AAA", "2024-01-05")]
    assert ref.event_id == "e1"
    assert ref.calendar_revision == snap.table_versions["earnings_events"].dataset_version_id


def test_resolve_event_refs_zero_matches_is_absent_not_none(tmp_path):
    conn, store, snap = _events_snapshot(tmp_path, [_event_row("e1", "AAA", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    result = projections.resolve_event_refs(repo, snap, {("ZZZ", "2024-01-05")})
    assert ("ZZZ", "2024-01-05") not in result


def test_resolve_event_refs_two_ids_for_one_pair_is_ambiguous(tmp_path):
    conn, store, snap = _events_snapshot(tmp_path, [
        _event_row("e1", "AAA", datetime(2024, 1, 5)),
        _event_row("e2", "AAA", datetime(2024, 1, 5)),
    ])
    repo = Repository(conn, store)
    result = projections.resolve_event_refs(repo, snap, {("AAA", "2024-01-05")})
    assert result[("AAA", "2024-01-05")] is None


# --------------------------------------------------------------------------
# build_candidate: idempotency, change detection, refusal, crash recovery
# --------------------------------------------------------------------------


def test_same_inputs_twice_give_identical_release_id_and_row_counts(tmp_path):
    conn, store, snap = _events_snapshot(tmp_path, [_event_row("e1", "AAA", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    serving_conn, serving_store = _serving(tmp_path)
    row = _row()
    score_doc, bundle = _score_doc(rows=[row]), _bundle(_compact(row))
    preview_input = _preview_input()

    first = _build(preview_input, score_doc, bundle, repository=repo, snap=snap,
                   serving_conn=serving_conn, serving_store=serving_store)
    second = _build(preview_input, score_doc, bundle, repository=repo, snap=snap,
                    serving_conn=serving_conn, serving_store=serving_store)

    assert isinstance(first, PreviewRelease)
    assert first.release_id == second.release_id
    n_release = serving_conn.execute("SELECT COUNT(*) FROM serving_release").fetchone()[0]
    n_events = serving_conn.execute("SELECT COUNT(*) FROM serving_event_summary").fetchone()[0]
    n_scores = serving_conn.execute("SELECT COUNT(*) FROM serving_score_summary").fetchone()[0]
    assert (n_release, n_events, n_scores) == (1, 1, 1)


def test_changed_score_gives_new_release_and_old_release_stays_readable(tmp_path):
    conn, store, snap = _events_snapshot(tmp_path, [_event_row("e1", "AAA", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    serving_conn, serving_store = _serving(tmp_path)
    preview_input = _preview_input()
    row = _row()
    old_score, old_bundle = _score_doc(rows=[row]), _bundle(_compact(row))
    old_release = _build(preview_input, old_score, old_bundle, repository=repo, snap=snap,
                         serving_conn=serving_conn, serving_store=serving_store)

    other = _row(strike=105.0)
    new_score, new_bundle = _score_doc(rows=[other]), _bundle(_compact(other))
    new_release = _build(preview_input, new_score, new_bundle, repository=repo, snap=snap,
                         serving_conn=serving_conn, serving_store=serving_store)

    assert new_release.release_id != old_release.release_id
    reread_old = projections.get_release(serving_conn, old_release.release_id)
    assert reread_old == old_release
    reread_new = projections.get_release(serving_conn, new_release.release_id)
    assert reread_new == new_release


def test_changed_bundle_gives_a_new_release_id(tmp_path):
    conn, store, snap = _events_snapshot(tmp_path, [_event_row("e1", "AAA", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    serving_conn, serving_store = _serving(tmp_path)
    preview_input = _preview_input()
    row = _row()
    score_doc = _score_doc(rows=[row])

    a = _build(preview_input, score_doc, _bundle(_compact(row)), repository=repo, snap=snap,
              serving_conn=serving_conn, serving_store=serving_store)
    # payoff_curve is a `derived` field (LEGACY_DISPLAY_MAPPING_V1) the bridge
    # never checks against a source, so changing only it keeps findings.ok
    # True -- isolating "the bundle's own bytes are part of release identity"
    # from "a mismatched bundle refuses the candidate" (separately covered above).
    changed = _compact(row, payoff_curve={"x": [1.0], "y": [3.0], "max": 3.0, "min": 0.0,
                                          "strikes": [], "shape": "twin"})
    b = _build(preview_input, score_doc, _bundle(changed), repository=repo, snap=snap,
              serving_conn=serving_conn, serving_store=serving_store)
    assert isinstance(a, PreviewRelease)
    assert isinstance(b, PreviewRelease)
    assert a.release_id != b.release_id


def test_changed_verified_input_provenance_gives_a_new_release_id(tmp_path):
    conn, store, snap = _events_snapshot(tmp_path, [_event_row("e1", "AAA", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    serving_conn, serving_store = _serving(tmp_path)
    row = _row()
    score_doc, bundle = _score_doc(rows=[row]), _bundle(_compact(row))

    first = _build(_preview_input(finality_ref="fin_1"), score_doc, bundle, repository=repo, snap=snap,
                   serving_conn=serving_conn, serving_store=serving_store)
    second = _build(_preview_input(finality_ref="fin_2"), score_doc, bundle, repository=repo, snap=snap,
                    serving_conn=serving_conn, serving_store=serving_store)

    assert isinstance(first, PreviewRelease)
    assert isinstance(second, PreviewRelease)
    assert first.release_id != second.release_id


def test_findings_failure_leaves_no_candidate_rows_and_writes_the_findings_receipt(tmp_path):
    conn, store, snap = _events_snapshot(tmp_path, [_event_row("e1", "AAA", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    serving_conn, serving_store = _serving(tmp_path)
    preview_input = _preview_input()
    row = _row()
    # A row scored but never rendered -- ProjectionFindings.ok is False.
    score_doc = _score_doc(rows=[row])
    result = _build(preview_input, score_doc, {}, repository=repo, snap=snap,
                    serving_conn=serving_conn, serving_store=serving_store)

    assert isinstance(result, Problem)
    assert result.code == "PROJECTION_REFUSED"
    assert result.details["findings"]["ok"] is False
    n_release = serving_conn.execute("SELECT COUNT(*) FROM serving_release").fetchone()[0]
    assert n_release == 0
    # The findings receipt was still published: a content-addressed object
    # under the store's own hash path, safe to leave unreferenced by any index row.
    digest = result.diagnostic_ref.removeprefix("sha256:")
    object_path = serving_store.root / "objects" / digest[:2] / digest
    assert object_path.is_file()


def test_ambiguous_event_produces_a_finding_and_refuses(tmp_path):
    conn, store, snap = _events_snapshot(tmp_path, [
        _event_row("e1", "AAA", datetime(2024, 1, 5)),
        _event_row("e2", "AAA", datetime(2024, 1, 5)),
    ])
    repo = Repository(conn, store)
    serving_conn, serving_store = _serving(tmp_path)
    row = _row()
    result = _build(_preview_input(), _score_doc(rows=[row]), _bundle(_compact(row)),
                    repository=repo, snap=snap, serving_conn=serving_conn, serving_store=serving_store)
    assert isinstance(result, Problem)
    codes = {f["code"] for f in result.details["findings"]["findings"]}
    assert "EVENT_AMBIGUOUS" in codes


def test_unresolved_event_produces_a_finding_and_refuses(tmp_path):
    conn, store, snap = _events_snapshot(tmp_path, [_event_row("e1", "ZZZ", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    serving_conn, serving_store = _serving(tmp_path)
    row = _row(ticker="AAA")
    result = _build(_preview_input(), _score_doc(rows=[row]), _bundle(_compact(row)),
                    repository=repo, snap=snap, serving_conn=serving_conn, serving_store=serving_store)
    assert isinstance(result, Problem)
    codes = {f["code"] for f in result.details["findings"]["findings"]}
    assert "EVENT_UNMAPPED" in codes


def test_crash_after_objects_before_index_commit_leaves_no_partial_release(tmp_path):
    conn, store, snap = _events_snapshot(tmp_path, [_event_row("e1", "AAA", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    serving_conn, serving_store = _serving(tmp_path)
    preview_input = _preview_input()
    row = _row()
    score_doc, bundle = _score_doc(rows=[row]), _bundle(_compact(row))

    def boom(point: str) -> None:
        if point == "index_rows_written":
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        _build(preview_input, score_doc, bundle, repository=repo, snap=snap,
              serving_conn=serving_conn, serving_store=serving_store, fault=boom)

    assert serving_conn.execute("SELECT COUNT(*) FROM serving_release").fetchone()[0] == 0
    assert serving_conn.execute("SELECT COUNT(*) FROM serving_event_summary").fetchone()[0] == 0
    assert serving_conn.execute("SELECT COUNT(*) FROM serving_score_summary").fetchone()[0] == 0

    rerun = _build(preview_input, score_doc, bundle, repository=repo, snap=snap,
                   serving_conn=serving_conn, serving_store=serving_store)
    assert isinstance(rerun, PreviewRelease)
    assert serving_conn.execute("SELECT COUNT(*) FROM serving_release").fetchone()[0] == 1


# --------------------------------------------------------------------------
# read helpers: pagination, event->scores, score detail
# --------------------------------------------------------------------------


def _five_event_setup(tmp_path):
    event_rows = [_event_row(f"e{i}", f"T{i}", datetime(2024, 1, i + 1)) for i in range(5)]
    conn, store, snap = _events_snapshot(tmp_path, event_rows)
    repo = Repository(conn, store)
    serving_conn, serving_store = _serving(tmp_path)
    score_rows = [_row(ticker=f"T{i}", event_date=f"2024-01-0{i + 1}", strike=100.0 + i)
                 for i in range(5)]
    score_doc = _score_doc(rows=score_rows)
    bundle = _bundle(*[_compact(row) for row in score_rows])
    release = _build(_preview_input(), score_doc, bundle, repository=repo, snap=snap,
                     serving_conn=serving_conn, serving_store=serving_store)
    assert isinstance(release, PreviewRelease)
    return serving_conn, serving_store, release


def test_pagination_covers_a_complete_ordered_non_duplicated_set(tmp_path):
    serving_conn, _, release = _five_event_setup(tmp_path)
    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        page = projections.list_events(serving_conn, release.release_id, limit=2, cursor=cursor)
        assert page.total_matching == 5
        seen.extend(item.event_ref.event_id for item in page.items)
        pages += 1
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
        assert pages <= 10  # a runaway cursor would hang the test, not just fail it
    assert len(seen) == 5
    assert len(set(seen)) == 5
    assert seen == sorted(seen, key=lambda eid: int(eid[1:]))


def test_event_scores_and_detail_ref_resolve(tmp_path):
    serving_conn, serving_store, release = _five_event_setup(tmp_path)
    page = projections.list_events(serving_conn, release.release_id, limit=10)
    item = page.items[0]
    scores = projections.event_scores(serving_conn, release.release_id, item.event_ref.event_id)
    assert len(scores) == 1
    assert scores == item.scores
    detail = projections.get_score_detail(serving_conn, serving_store, release.release_id,
                                          scores[0].score_id)
    assert detail is not None
    assert detail.event_ref == item.event_ref
    assert detail.score_id == scores[0].score_id


def test_score_summary_fields_equal_the_mapped_display_values_for_the_exact_row(tmp_path):
    """No summary field may diverge from what the mapping spec already
    checked into ``display_record`` -- this compares against that exact
    rendered row, not a hand-picked expectation, and pins the review fix:
    ``verdict`` is the row's own raw ``gate_pass``, never an invented word,
    and ``expected_return`` stays null (no single headline field upstream)."""
    serving_conn, _, release = _five_event_setup(tmp_path)
    page = projections.list_events(serving_conn, release.release_id, limit=10)
    summary = page.items[0].scores[0]
    row = _row(ticker="T0", event_date="2024-01-01", strike=100.0)
    display = _compact(row)

    assert summary.strategy == display["strategy"]
    assert summary.verdict == ("true" if display["gate_pass"] else "false")
    assert summary.refusal_reason is None  # gate_pass is True on this row
    assert summary.driver_forecast == display["driver_prediction"]
    assert summary.market_implied_move == display["implied_move"]
    assert summary.entry_premium == display["entry_cost"]
    assert summary.expected_return is None
    assert summary.expected_return_model == display["exp_pnl_model"]
    assert summary.expected_return_analog == display["exp_pnl_analog"]
    assert summary.expected_return_sim == display.get("exp_pnl_sim")


def test_refusal_reason_carries_the_rows_own_detail_when_gate_declines(tmp_path):
    conn, store, snap = _events_snapshot(tmp_path, [_event_row("e1", "AAA", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    serving_conn, serving_store = _serving(tmp_path)
    row = _row(gate_pass=False, exp_pnl_model=None, win_model=None, detail="gate score below threshold")
    release = _build(_preview_input(), _score_doc(rows=[row]), _bundle(_compact(row)),
                     repository=repo, snap=snap, serving_conn=serving_conn, serving_store=serving_store)
    assert isinstance(release, PreviewRelease)
    page = projections.list_events(serving_conn, release.release_id, limit=10)
    summary = page.items[0].scores[0]
    assert summary.verdict == "false"
    assert summary.refusal_reason == "gate score below threshold"


def test_expected_return_stays_null_when_only_analog_is_present(tmp_path):
    """§5.2/§5.3: the serving index never chooses between rendered values.
    A row scored only by the analog layer (model null) must not have its
    analog figure promoted into `expected_return` -- that would be exactly
    the invented precedence the review flagged."""
    conn, store, snap = _events_snapshot(tmp_path, [_event_row("e1", "AAA", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    serving_conn, serving_store = _serving(tmp_path)
    row = _row(exp_pnl_model=None, win_model=None, exp_pnl_analog=0.08, win_analog=0.6)
    release = _build(_preview_input(), _score_doc(rows=[row]), _bundle(_compact(row)),
                     repository=repo, snap=snap, serving_conn=serving_conn, serving_store=serving_store)
    assert isinstance(release, PreviewRelease)
    page = projections.list_events(serving_conn, release.release_id, limit=10)
    summary = page.items[0].scores[0]
    assert summary.expected_return is None
    assert summary.expected_return_model is None
    assert summary.expected_return_analog == 0.08
    assert summary.expected_return_sim is None


# --------------------------------------------------------------------------
# CLI: end to end on tmp dirs
# --------------------------------------------------------------------------


def test_cli_runs_end_to_end_on_tmp_dirs(tmp_path, capsys):
    import json

    from engine.v2.foundation import to_document
    from tools.v2_dashboard_project import main

    (tmp_path / "phase2").mkdir()
    conn, store, snap = _events_snapshot(tmp_path / "phase2", [_event_row("e1", "AAA", datetime(2024, 1, 5))])
    conn.close()

    row = _row()
    score_path = tmp_path / "score.json"
    score_path.write_text(json.dumps(_score_doc(rows=[row])))
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "AAA.json").write_text(json.dumps([_compact(row)]))
    preview_input_path = tmp_path / "preview_input.json"
    preview_input_path.write_text(json.dumps(to_document(_preview_input())))

    exit_code = main([
        "--preview-input", str(preview_input_path),
        "--score-json", str(score_path),
        "--bundle-dir", str(bundle_dir),
        "--bundle-format", "flat",
        "--snapshot-id", snap.snapshot_id,
        "--catalog", str(tmp_path / "phase2" / "catalog.sqlite"),
        "--store-root", str(tmp_path / "phase2" / "store"),
        "--serving-root", str(tmp_path / "serving_out"),
        "--requested-as-of", "2024-01-04",
        "--resolved-as-of", "2024-01-04",
    ])
    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out["ok"] is True
    assert "release_id" in out
    assert out["findings"]["ok"] is True
