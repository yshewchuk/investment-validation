"""CAPTURE_DUMP_SELECTED: an opt-in env var that pickles exactly the
post-selection inputs attach_strict_probe/write need, so
tools/capture_attach_probe.py can replay the attach step offline without
rebuilding the Scorer or re-running candidate selection (see
tools/capture_attach_probe.py's module docstring). Fully synthetic: no real
Scorer, no panel, no chain table, and the dump never carries real data in
these tests."""
from __future__ import annotations

import pickle

import pandas as pd

import tools.capture_tier0_corpus as capture

from tests.test_capture_tier0_release_scorer import _args, _FakeScorer, _install_empty_pipeline


def test_dump_selected_round_trips_chosen_index_and_snapshot(tmp_path):
    chosen = [{"fixture_id": "case-0", "kind": "score_result",
              "request": {"a": 1}, "record": {"b": 2}, "duration": 0.1,
              "legacy_trace": {"x": 1}}]
    index = {"strategy:STR-THRU": ["case-0"]}
    as_of = pd.Timestamp("2026-01-01")
    snapshot = "snapshot-1"
    path = tmp_path / "dump.pkl"

    capture._dump_selected(path, chosen, index, as_of, snapshot)

    with path.open("rb") as fh:
        payload = pickle.load(fh)
    assert payload == {"chosen": chosen, "index": index, "as_of": as_of,
                       "snapshot": snapshot}


def test_main_writes_the_dump_when_the_env_var_is_set(monkeypatch, tmp_path):
    scorer = _FakeScorer()
    chosen_in = [{"fixture_id": "case-0", "kind": "score_result",
                 "request": {}, "record": {}, "duration": 0.1}]
    _install_empty_pipeline(monkeypatch, chosen_in)
    monkeypatch.setattr(capture.score_mod, "Scorer", lambda: scorer)
    monkeypatch.setattr(capture, "write", lambda *a, **k: {
        "corpus_hash": "sha256:" + "0" * 64, "required_axes": [], "uncovered_axes": [],
    })
    dump_path = tmp_path / "dump" / "selected.pkl"
    monkeypatch.setenv("CAPTURE_DUMP_SELECTED", str(dump_path))

    rc = capture.main(["--out", str(tmp_path / "out")])

    assert rc == 0
    assert dump_path.is_file()
    with dump_path.open("rb") as fh:
        payload = pickle.load(fh)
    assert payload["chosen"] == chosen_in
    assert payload["snapshot"] == "snapshot-1"


def test_main_writes_no_dump_when_the_env_var_is_unset(monkeypatch, tmp_path):
    scorer = _FakeScorer()
    chosen_in = [{"fixture_id": "case-0", "kind": "score_result",
                 "request": {}, "record": {}, "duration": 0.1}]
    _install_empty_pipeline(monkeypatch, chosen_in)
    monkeypatch.setattr(capture.score_mod, "Scorer", lambda: scorer)
    monkeypatch.setattr(capture, "write", lambda *a, **k: {
        "corpus_hash": "sha256:" + "0" * 64, "required_axes": [], "uncovered_axes": [],
    })
    monkeypatch.delenv("CAPTURE_DUMP_SELECTED", raising=False)

    rc = capture.main(["--out", str(tmp_path / "out")])

    assert rc == 0
    assert not (tmp_path / "dump").exists()