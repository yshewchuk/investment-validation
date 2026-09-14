"""P3-4: the real legacy render-bundle adapter.

Builds one small, real-shaped bundle by calling the actual
``engine.dashboard.render.render_bundle`` on synthetic ``ScoreResult`` rows
(the same technique ``tests/test_dashboard.py`` already uses) — never a real
feature panel, never real market data. The baked bundle is rendered ONCE
(module-scoped) and copied per test, so refusal tests mutate their own copy
without paying the render cost again.

``engine.v2.serving.legacy_bundle`` may not import legacy ``engine.*`` or
``engine.v2.ops`` — only this test file does, to produce a bundle with the
real on-disk shape.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.dashboard.render import render_bundle  # noqa: E402
from engine.jsonio import json_safe  # noqa: E402
from engine.score import ScoreResult  # noqa: E402
from engine.v2.contracts import EventRef, ObjectRef, PreviewInput  # noqa: E402
from engine.v2.foundation import content_hash, to_document  # noqa: E402
from engine.v2.serving.bridge import build_bridges  # noqa: E402
from engine.v2.serving.legacy_bundle import (  # noqa: E402
    LegacyBundleError,
    load_legacy_bundle,
    load_score_document,
)
from tests.data_scan_support import catalog_and_store  # noqa: E402

AS_OF = pd.Timestamp("2026-08-10")
EVENT = pd.Timestamp("2026-08-12")


# --------------------------------------------------------------------------
# a small, real-shaped bundle — rendered once, module-scoped
# --------------------------------------------------------------------------


def _score_result(ticker: str, strategy: str, strike: float, **kwargs) -> ScoreResult:
    base = dict(
        as_of=AS_OF, event_date=EVENT, session="AMC",
        exp_pnl_model=0.031, win_model=0.56, model_p10=-0.22, model_p90=0.41,
        exp_pnl_analog=0.028, win_analog=0.54, ci_low=-0.01, ci_high=0.07,
        n_analogs=120, gate_score=0.71, gate_threshold=0.6, gate_pass=True,
        entry_date=EVENT, exit_date=EVENT + pd.Timedelta(days=1),
        strike=strike, expiry=EVENT + pd.Timedelta(days=2),
        entry_cost=6.0, spot=100.0, dte_entry=2,
        payoff={"intercept": 0.01, "slope": 0.006, "n": 400},
        driver_name="abs_move", driver_prediction=7.5,
        model_versions={"size": "size_v13"}, snapshot_hash="snap-test",
        legs=[{"right": "P", "strike": strike, "side": "buy", "qty": 1}],
    )
    base.update(kwargs)
    return ScoreResult(ticker=ticker, strategy=strategy, **base)


def _source_row(result: ScoreResult, *, strike_offset: float | None) -> dict:
    """What ``score.json`` actually stores: ``json_safe(..., round_to=None)``
    (``engine.v2.ops.legacy_adapter._action_score``), plus ``strike_offset``
    added exactly as ``strike_ladder``/``assemble_scores`` add it."""
    return json_safe(result.as_dict(), round_to=None) | {"strike_offset": strike_offset}


@pytest.fixture(scope="module")
def baked_bundle(tmp_path_factory) -> dict:
    """Render one real bundle: two tickers, a main row each, one ladder row.

    Module-scoped so ``render_bundle`` (real registry/ledger/portfolio reads,
    all gracefully empty here — no panel, no trades passed) runs once; every
    test that needs to mutate bytes copies this tree into its own ``tmp_path``.
    """
    aaa_main = _score_result("AAA", "STR-THRU", 100.0)
    bbb_main = _score_result("BBB", "STR-THRU", 50.0)
    aaa_ladder = _score_result("AAA", "STR-THRU", 102.5)

    main_rows = [_source_row(aaa_main, strike_offset=None),
                 _source_row(bbb_main, strike_offset=None)]
    ladder_rows = [_source_row(aaa_ladder, strike_offset=2.5)]

    root = tmp_path_factory.mktemp("baked") / "bundle"
    frame = pd.DataFrame(main_rows + ladder_rows)
    summary = render_bundle(frame, root, as_of=AS_OF)

    event_date = main_rows[0]["event_date"]
    assert event_date == str(EVENT.date())  # sanity: json_safe's own contract
    return {
        "root": root, "main_rows": main_rows, "ladder_rows": ladder_rows,
        "event_date": event_date, "summary": summary,
    }


def _copy_bundle(baked: dict, tmp_path: Path) -> Path:
    dest = tmp_path / "bundle"
    shutil.copytree(baked["root"], dest)
    return dest


def _event_refs(baked: dict) -> dict:
    event_date = baked["event_date"]
    return {
        ("AAA", event_date): EventRef(event_id="evt_AAA", calendar_revision="cal_1"),
        ("BBB", event_date): EventRef(event_id="evt_BBB", calendar_revision="cal_1"),
    }


def _score_doc(baked: dict) -> dict:
    main_rows, ladder_rows = baked["main_rows"], baked["ladder_rows"]
    expected = sorted({f"{r['ticker']}|{r['strategy']}|{r['event_date']}" for r in main_rows})
    return {"rows": main_rows, "ladder": ladder_rows, "expected_population": expected}


# --------------------------------------------------------------------------
# bundle forms actually found: JSON and its JS wrapper
# --------------------------------------------------------------------------


def test_render_bundle_emits_both_json_and_js_forms(baked_bundle):
    root = baked_bundle["root"]
    for stem in ("board", "meta", "health", "flags", "strategies", "book", "models"):
        assert (root / "data" / f"{stem}.json").is_file()
        assert (root / "data" / f"{stem}.js").is_file()
    for ticker in ("AAA", "BBB"):
        assert (root / "data" / "tickers" / f"{ticker}.json").is_file()
        assert (root / "data" / "tickers" / f"{ticker}.js").is_file()
    board_js = (root / "data" / "board.js").read_text()
    assert board_js.startswith("window.BOARD = ")
    ticker_js = (root / "data" / "tickers" / "AAA.js").read_text()
    assert 'window.TICKER_DATA["AAA"] = ' in ticker_js


# --------------------------------------------------------------------------
# round trip: real bundle -> load_legacy_bundle -> build_bridges
# --------------------------------------------------------------------------


def test_real_bundle_round_trips_through_load_legacy_bundle_and_build_bridges(baked_bundle):
    bundle_rows_by_ticker, manifest = load_legacy_bundle(baked_bundle["root"])

    assert set(bundle_rows_by_ticker) == {"AAA", "BBB"}
    # Ladder rows sit inside their parent event's `rows`, identified only by
    # a non-null strike_offset -- never a separate top-level list.
    aaa_rows = bundle_rows_by_ticker["AAA"]
    assert sum(1 for r in aaa_rows if r.get("strike_offset") is None) == 1
    assert sum(1 for r in aaa_rows if r.get("strike_offset") is not None) == 1
    bbb_rows = bundle_rows_by_ticker["BBB"]
    assert sum(1 for r in bbb_rows if r.get("strike_offset") is None) == 1

    assert manifest["data/board.json"].startswith("sha256:")
    assert "data/tickers/AAA.json" in manifest
    assert "data/tickers/BBB.json" in manifest

    bridges, findings = build_bridges(
        _score_doc(baked_bundle), bundle_rows_by_ticker, _event_refs(baked_bundle),
        score_batch_ref="batch_1", snapshot_ref="snap_1",
        model_registry_artifact_refs=("model_1",), request_provenance_refs=("req_1",))

    assert findings.ok is True, findings.findings
    assert len(bridges) == 3
    assert findings.rendered_main_population == len(baked_bundle["main_rows"])
    assert findings.rendered_ladder_population == len(baked_bundle["ladder_rows"])
    # Ties the bridge's own funnel back to what render_bundle itself reported.
    assert (findings.rendered_main_population + findings.rendered_ladder_population
            == baked_bundle["summary"]["n_rows"])


def test_js_only_ticker_file_loads_the_same_rows(baked_bundle, tmp_path):
    """"accepting the JSON form and, if emitted, the JS-wrapper form"."""
    root = _copy_bundle(baked_bundle, tmp_path)
    (root / "data" / "tickers" / "BBB.json").unlink()

    bundle_rows_by_ticker, _ = load_legacy_bundle(root)
    baseline, _ = load_legacy_bundle(baked_bundle["root"])
    assert bundle_rows_by_ticker["BBB"] == baseline["BBB"]


# --------------------------------------------------------------------------
# manifest design: byte-bound, path-keyed content hashes
# --------------------------------------------------------------------------


def test_a_changed_byte_in_one_ticker_file_changes_the_manifest_hash(baked_bundle, tmp_path):
    root = _copy_bundle(baked_bundle, tmp_path)
    _, manifest_before = load_legacy_bundle(root)

    ticker_path = root / "data" / "tickers" / "AAA.json"
    payload = json.loads(ticker_path.read_text())
    # A ticker-payload row is the raw engine record, not compact_row's
    # output -- `snapshot_hash` is a real ScoreResult field outside
    # `_BOARD_FIELD_NAMES`, so bridge.py's display comparison never looks at
    # it, isolating "the manifest is byte-bound" from "a mismatch refuses".
    payload["events"][0]["rows"][0]["snapshot_hash"] = "snap-test-mutated"
    ticker_path.write_text(json.dumps(payload))
    # Keep the .js sibling in agreement -- this test isolates the manifest's
    # byte-sensitivity from BUNDLE_FORM_MISMATCH, covered separately below.
    (root / "data" / "tickers" / "AAA.js").write_text(
        "window.TICKER_DATA = window.TICKER_DATA || {};\n"
        f'window.TICKER_DATA["AAA"] = {json.dumps(payload)};\n')

    _, manifest_after = load_legacy_bundle(root)

    assert manifest_before["data/tickers/AAA.json"] != manifest_after["data/tickers/AAA.json"]
    assert manifest_before["data/tickers/AAA.js"] != manifest_after["data/tickers/AAA.js"]
    assert manifest_before["data/board.json"] == manifest_after["data/board.json"]
    assert content_hash(manifest_before) != content_hash(manifest_after)


# --------------------------------------------------------------------------
# both forms present: read both, hash both, and require them to agree --
# the browser (legacy dashboard, compatibility preview) loads the .js
# wrapper, so a projection built only from .json could silently diverge
# from what a user actually sees.
# --------------------------------------------------------------------------


def test_matching_json_and_js_pair_loads_with_both_hashes_in_the_manifest(baked_bundle):
    """The baked bundle's real render_bundle output: every board/ticker file
    has an untouched, agreeing .js sibling -- both must be read and hashed."""
    _, manifest = load_legacy_bundle(baked_bundle["root"])

    assert "data/board.json" in manifest and "data/board.js" in manifest
    assert manifest["data/board.json"] != manifest["data/board.js"]  # different bytes, same content
    for ticker in ("AAA", "BBB"):
        assert f"data/tickers/{ticker}.json" in manifest
        assert f"data/tickers/{ticker}.js" in manifest


def test_mismatched_board_js_is_refused(baked_bundle, tmp_path):
    root = _copy_bundle(baked_bundle, tmp_path)
    payload = json.loads((root / "data" / "board.json").read_text())
    payload["n_rows"] = payload["n_rows"] + 1000  # disagrees with board.json
    (root / "data" / "board.js").write_text(f"window.BOARD = {json.dumps(payload)};\n")

    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(root)
    assert err.value.code == "BUNDLE_FORM_MISMATCH"
    assert err.value.details["json_path"] == "data/board.json"
    assert err.value.details["js_path"] == "data/board.js"


def test_mismatched_ticker_js_is_refused(baked_bundle, tmp_path):
    root = _copy_bundle(baked_bundle, tmp_path)
    payload = json.loads((root / "data" / "tickers" / "AAA.json").read_text())
    payload["history"] = [{"planted": "disagreement"}]  # disagrees with AAA.json
    (root / "data" / "tickers" / "AAA.js").write_text(
        "window.TICKER_DATA = window.TICKER_DATA || {};\n"
        f'window.TICKER_DATA["AAA"] = {json.dumps(payload)};\n')

    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(root)
    assert err.value.code == "BUNDLE_FORM_MISMATCH"
    assert err.value.details["json_path"] == "data/tickers/AAA.json"
    assert err.value.details["js_path"] == "data/tickers/AAA.js"


# --------------------------------------------------------------------------
# refusals: typed errors, never a silent skip
# --------------------------------------------------------------------------


def test_symlinked_ticker_file_is_refused(baked_bundle, tmp_path):
    root = _copy_bundle(baked_bundle, tmp_path)
    victim = root / "data" / "tickers" / "AAA.json"
    target = root / "data" / "tickers" / "BBB.json"
    victim.unlink()
    os.symlink(target, victim)

    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(root)
    assert err.value.code == "SYMLINK_REFUSED"


def test_path_traversal_ticker_is_refused(tmp_path):
    root = tmp_path / "evil_bundle"
    (root / "data").mkdir(parents=True)
    (root / "data" / "board.json").write_text(json.dumps(
        {"as_of": "2026-01-01", "n_rows": 1, "rows": [{"ticker": "../evil"}]}))

    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(root)
    assert err.value.code == "PATH_TRAVERSAL"


def test_ticker_listed_on_board_with_no_rendered_file_is_refused(baked_bundle, tmp_path):
    root = _copy_bundle(baked_bundle, tmp_path)
    (root / "data" / "tickers" / "BBB.json").unlink()
    (root / "data" / "tickers" / "BBB.js").unlink()

    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(root)
    assert err.value.code == "MISSING_TICKER_FILE"
    assert "BBB" in err.value.details["missing_tickers"]


def test_duplicate_ticker_is_refused(baked_bundle, tmp_path):
    root = _copy_bundle(baked_bundle, tmp_path)
    dup = root / "data" / "tickers" / "AAA_dup.json"
    dup.write_text(json.dumps(
        {"ticker": "AAA", "as_of": "2026-08-10", "events": [], "history": [], "analogs": []}))

    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(root)
    assert err.value.code == "DUPLICATE_TICKER"


def test_malformed_js_wrapper_is_refused(baked_bundle, tmp_path):
    root = _copy_bundle(baked_bundle, tmp_path)
    (root / "data" / "tickers" / "AAA.json").unlink()
    (root / "data" / "tickers" / "AAA.js").write_text("not a window.* assignment at all\n")

    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(root)
    assert err.value.code == "MALFORMED_WRAPPER"


def test_unknown_top_level_key_is_refused(baked_bundle, tmp_path):
    root = _copy_bundle(baked_bundle, tmp_path)
    board_path = root / "data" / "board.json"
    # Isolate this test's own claim from BUNDLE_FORM_MISMATCH: drop the .js
    # sibling so only the mutated .json form is read.
    (root / "data" / "board.js").unlink()
    payload = json.loads(board_path.read_text())
    payload["surprise_field"] = 1
    board_path.write_text(json.dumps(payload))

    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(root)
    assert err.value.code == "UNKNOWN_STRUCTURE"


def test_symlinked_bundle_root_is_refused(baked_bundle, tmp_path):
    link = tmp_path / "bundle_link"
    os.symlink(baked_bundle["root"], link)
    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(link)
    assert err.value.code == "SYMLINK_REFUSED"


# --------------------------------------------------------------------------
# load_score_document
# --------------------------------------------------------------------------


def test_load_score_document_accepts_the_real_shape(baked_bundle, tmp_path):
    path = tmp_path / "score.json"
    path.write_text(json.dumps(_score_doc(baked_bundle)))
    doc = load_score_document(path)
    assert set(doc) == {"rows", "ladder", "expected_population"}
    assert len(doc["rows"]) == 2
    assert len(doc["ladder"]) == 1


def test_load_score_document_refuses_unknown_key(tmp_path):
    path = tmp_path / "score.json"
    path.write_text(json.dumps({"rows": [], "ladder": [], "expected_population": [], "extra": 1}))
    with pytest.raises(LegacyBundleError) as err:
        load_score_document(path)
    assert err.value.code == "UNKNOWN_STRUCTURE"


def test_load_score_document_refuses_missing_required_key(tmp_path):
    path = tmp_path / "score.json"
    path.write_text(json.dumps({"rows": [], "ladder": []}))
    with pytest.raises(LegacyBundleError) as err:
        load_score_document(path)
    assert err.value.code == "UNKNOWN_STRUCTURE"


def test_load_score_document_refuses_symlink(tmp_path):
    real = tmp_path / "real_score.json"
    real.write_text(json.dumps({"rows": [], "ladder": [], "expected_population": []}))
    link = tmp_path / "score.json"
    os.symlink(real, link)
    with pytest.raises(LegacyBundleError) as err:
        load_score_document(link)
    assert err.value.code == "SYMLINK_REFUSED"


def test_load_score_document_refuses_missing_file(tmp_path):
    with pytest.raises(LegacyBundleError) as err:
        load_score_document(tmp_path / "does_not_exist.json")
    assert err.value.code == "MISSING_FILE"


def test_load_score_document_refuses_a_json_array(tmp_path):
    path = tmp_path / "score.json"
    path.write_text(json.dumps([]))
    with pytest.raises(LegacyBundleError) as err:
        load_score_document(path)
    assert err.value.code == "UNKNOWN_STRUCTURE"


def test_load_score_document_refuses_a_non_list_required_key(tmp_path):
    path = tmp_path / "score.json"
    path.write_text(json.dumps({"rows": {}, "ladder": [], "expected_population": []}))
    with pytest.raises(LegacyBundleError) as err:
        load_score_document(path)
    assert err.value.code == "UNKNOWN_STRUCTURE"


# --------------------------------------------------------------------------
# load_legacy_bundle: every remaining branch (malformed JSON, a JS wrapper
# missing its terminator, a non-object payload, an absent board/ticker file,
# a symlinked tickers/ directory, a board with no rows and no tickers/ dir,
# and a malformed events/rows shape) -- not just the headline refusal list.
# --------------------------------------------------------------------------


def _write_board(root: Path, rows: list) -> None:
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data" / "board.json").write_text(
        json.dumps({"as_of": "2026-01-01", "n_rows": len(rows), "rows": rows}))


def test_direct_read_and_hash_missing_file_is_refused(tmp_path):
    """The defensive branch in ``_read_and_hash`` (unreachable through the
    public API, whose only caller already confirmed existence first)."""
    from engine.v2.serving import legacy_bundle as lb

    with pytest.raises(LegacyBundleError) as err:
        lb._read_and_hash(tmp_path, tmp_path / "nope.json", {})
    assert err.value.code == "MISSING_FILE"


def test_board_with_invalid_json_syntax_is_refused(tmp_path):
    root = tmp_path / "bundle"
    (root / "data").mkdir(parents=True)
    (root / "data" / "board.json").write_text("{not valid json")

    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(root)
    assert err.value.code == "UNKNOWN_STRUCTURE"


def test_board_that_is_a_json_array_is_refused(tmp_path):
    root = tmp_path / "bundle"
    (root / "data").mkdir(parents=True)
    (root / "data" / "board.json").write_text(json.dumps([]))

    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(root)
    assert err.value.code == "UNKNOWN_STRUCTURE"


def test_board_rows_not_a_list_is_refused(tmp_path):
    root = tmp_path / "bundle"
    (root / "data").mkdir(parents=True)
    (root / "data" / "board.json").write_text(
        json.dumps({"as_of": "2026-01-01", "n_rows": 0, "rows": {}}))

    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(root)
    assert err.value.code == "UNKNOWN_STRUCTURE"


def test_board_row_without_ticker_key_is_refused(tmp_path):
    root = tmp_path / "bundle"
    _write_board(root, [{"strategy": "STR-THRU"}])

    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(root)
    assert err.value.code == "UNKNOWN_STRUCTURE"


def test_neither_board_json_nor_board_js_exists_is_refused(tmp_path):
    root = tmp_path / "bundle"
    (root / "data").mkdir(parents=True)

    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(root)
    assert err.value.code == "MISSING_FILE"


def test_empty_board_with_no_tickers_directory_loads_cleanly(tmp_path):
    """No tickers referenced, no ``data/tickers/`` at all -- a legitimate
    empty bundle, not a refusal."""
    root = tmp_path / "bundle"
    _write_board(root, [])

    bundle_rows_by_ticker, manifest = load_legacy_bundle(root)
    assert bundle_rows_by_ticker == {}
    assert "data/board.json" in manifest


def test_symlinked_tickers_directory_is_refused(baked_bundle, tmp_path):
    root = _copy_bundle(baked_bundle, tmp_path)
    tickers_dir = root / "data" / "tickers"
    real_dir = tmp_path / "elsewhere_tickers"
    shutil.move(str(tickers_dir), str(real_dir))
    os.symlink(real_dir, tickers_dir)

    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(root)
    assert err.value.code == "SYMLINK_REFUSED"


def test_ticker_js_wrapper_missing_terminator_is_refused(baked_bundle, tmp_path):
    root = _copy_bundle(baked_bundle, tmp_path)
    (root / "data" / "tickers" / "AAA.json").unlink()
    payload = json.loads((baked_bundle["root"] / "data" / "tickers" / "AAA.json").read_text())
    (root / "data" / "tickers" / "AAA.js").write_text(
        f'window.TICKER_DATA["AAA"] = {json.dumps(payload)}')  # no trailing ';'

    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(root)
    assert err.value.code == "MALFORMED_WRAPPER"


def test_ticker_payload_that_is_a_json_array_is_refused(baked_bundle, tmp_path):
    root = _copy_bundle(baked_bundle, tmp_path)
    # Isolate this test's own claim from BUNDLE_FORM_MISMATCH: drop the .js
    # sibling so only the mutated .json form is read.
    (root / "data" / "tickers" / "AAA.js").unlink()
    (root / "data" / "tickers" / "AAA.json").write_text(json.dumps([]))

    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(root)
    assert err.value.code == "UNKNOWN_STRUCTURE"


def test_ticker_events_not_a_list_is_refused(baked_bundle, tmp_path):
    root = _copy_bundle(baked_bundle, tmp_path)
    (root / "data" / "tickers" / "AAA.js").unlink()
    ticker_path = root / "data" / "tickers" / "AAA.json"
    payload = json.loads(ticker_path.read_text())
    payload["events"] = {}
    ticker_path.write_text(json.dumps(payload))

    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(root)
    assert err.value.code == "UNKNOWN_STRUCTURE"


def test_ticker_event_missing_rows_is_refused(baked_bundle, tmp_path):
    root = _copy_bundle(baked_bundle, tmp_path)
    (root / "data" / "tickers" / "AAA.js").unlink()
    ticker_path = root / "data" / "tickers" / "AAA.json"
    payload = json.loads(ticker_path.read_text())
    payload["events"] = [{"event_date": "2026-08-12", "session": "AMC"}]  # no 'rows'
    ticker_path.write_text(json.dumps(payload))

    with pytest.raises(LegacyBundleError) as err:
        load_legacy_bundle(root)
    assert err.value.code == "UNKNOWN_STRUCTURE"


# --------------------------------------------------------------------------
# CLI: --bundle-format legacy end to end, and release-id change on a real
# bundle-byte change
# --------------------------------------------------------------------------


def _event_row(event_id: str, ticker: str, event_date: datetime) -> dict:
    return dict(event_id=event_id, ticker=ticker, event_date=event_date, year=event_date.year,
               session="BMO", session_src="orats", annc_tod=None, src_orats=True,
               src_oquants=True, src_nasdaq=False, src_yfinance=False, date_agree=True,
               date_conflict=False, updated_at=None, event_cluster_id=None,
               claim_count=None, reconciliation=None)


def _preview_input() -> PreviewInput:
    obj = ObjectRef(kind="legacy_snapshot", object_id="o1",
                    content_hash="sha256:" + "1" * 64, byte_size=10)
    return PreviewInput(
        source_release_id="rel_1", source_release_manifest_ref="m1",
        snapshot_ref="snap_1", legacy_snapshot_object_ref=obj,
        score_batch_ref="batch_1", score_job_input_refs=("in_1",),
        bundle_manifest_ref="unused-overridden-by-cli", model_registry_artifact_refs=("model_1",),
        finality_ref="fin_1", expected_population_ref="pop_1",
        score_comparison_receipt_ref="sc_1", render_comparison_receipt_ref="rc_1",
        source_code_hash="sha256:" + "2" * 64, source_environment_hash="sha256:" + "3" * 64)


def test_cli_end_to_end_legacy_format_and_byte_change_moves_release_id(tmp_path, capsys):
    from tests.data_scan_support import commit_tables, contract_for, contract_ref_for, publish_and_inspect
    from tools.v2_dashboard_project import main as cli_main

    events = contract_for("earnings_events")
    events_ref = contract_ref_for(events)
    (tmp_path / "phase2").mkdir()
    conn, clock, store = catalog_and_store(tmp_path / "phase2")
    record = publish_and_inspect(store, events, events_ref,
                                 [_event_row("e1", "AAA", EVENT.to_pydatetime())], "2026")
    snap = commit_tables(conn, clock, {"earnings_events": [record]}, {"earnings_events": events})
    conn.close()

    single = _score_result("AAA", "STR-THRU", 100.0)
    row = _source_row(single, strike_offset=None)
    bundle_root = tmp_path / "bundle"
    render_bundle(pd.DataFrame([row]), bundle_root, as_of=AS_OF)

    score_path = tmp_path / "score.json"
    score_path.write_text(json.dumps(
        {"rows": [row], "ladder": [],
         "expected_population": [f"{row['ticker']}|{row['strategy']}|{row['event_date']}"]}))
    preview_input_path = tmp_path / "preview_input.json"
    preview_input_path.write_text(json.dumps(to_document(_preview_input())))

    def run(serving_name: str) -> dict:
        exit_code = cli_main([
            "--preview-input", str(preview_input_path),
            "--score-json", str(score_path),
            "--bundle-dir", str(bundle_root),
            "--bundle-format", "legacy",
            "--snapshot-id", snap.snapshot_id,
            "--catalog", str(tmp_path / "phase2" / "catalog.sqlite"),
            "--store-root", str(tmp_path / "phase2" / "store"),
            "--serving-root", str(tmp_path / serving_name),
            "--requested-as-of", str(row["as_of"]),
            "--resolved-as-of", str(row["as_of"]),
        ])
        out = json.loads(capsys.readouterr().out)
        assert exit_code == 0, out
        return out

    first = run("serving_out_1")
    assert first["ok"] is True
    assert first["findings"]["ok"] is True

    ticker_path = bundle_root / "data" / "tickers" / "AAA.json"
    payload = json.loads(ticker_path.read_text())
    # Same reasoning as the manifest-hash test: `snapshot_hash` is outside
    # `_BOARD_FIELD_NAMES`, so this bundle-byte change never becomes a
    # VALUE_MISMATCH -- findings stay ok, isolating "the release id binds to
    # the bundle's actual bytes" from "a mismatch refuses the candidate".
    payload["events"][0]["rows"][0]["snapshot_hash"] = "snap-test-mutated"
    ticker_path.write_text(json.dumps(payload))
    (bundle_root / "data" / "tickers" / "AAA.js").write_text(
        "window.TICKER_DATA = window.TICKER_DATA || {};\n"
        f'window.TICKER_DATA["AAA"] = {json.dumps(payload)};\n')

    second = run("serving_out_2")
    assert second["ok"] is True
    assert second["findings"]["ok"] is True
    assert second["release_id"] != first["release_id"]
