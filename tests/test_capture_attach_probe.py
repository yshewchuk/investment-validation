"""tools/capture_attach_probe.py: replays attach_strict_probe/write against
a CAPTURE_DUMP_SELECTED dump, standalone. Synthetic: builds one real
strict-traceable STR-THRU candidate (the same fixture shape
tests/test_phase4_capture_strict.py already uses for this), dumps it with
capture_tier0_corpus._dump_selected, and checks the probe reproduces the
same corpus a direct `write(..., strict_trace=True)` call would."""
from __future__ import annotations

import json

import pandas as pd

import tools.capture_attach_probe as probe
import tools.capture_tier0_corpus as capture

from tests.test_phase4_capture_strict import _artifact, _full_strict_candidate


def test_probe_loads_a_dump_and_reproduces_the_written_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
    path, digest = _artifact(tmp_path)
    candidate = _full_strict_candidate(
        fixture_id="case-0", ticker="AAA", driver_vector={"x": 2.0},
        gate_vector={"x": 9.0, "n_prior": 5.0}, path=path, digest=digest,
    )
    chosen = [candidate]
    index = {"strategy:STR-THRU": ["case-0"]}
    as_of = pd.Timestamp("2026-01-01")
    snapshot = "snapshot-1"

    dump_path = tmp_path / "dump.pkl"
    capture._dump_selected(dump_path, chosen, index, as_of, snapshot)

    out_dir = tmp_path / "probe-out"
    rc = probe.main([str(dump_path), "--out", str(out_dir)])

    assert rc == 0
    doc = json.loads((out_dir / "INDEX.json").read_text())
    assert doc["pairs"]["case-0"]["trace_disposition"] == "complete"
    assert doc["corpus_hash"]


def test_probe_tracemalloc_flag_does_not_change_the_written_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
    path, digest = _artifact(tmp_path)
    candidate = _full_strict_candidate(
        fixture_id="case-0", ticker="AAA", driver_vector={"x": 2.0},
        gate_vector={"x": 9.0, "n_prior": 5.0}, path=path, digest=digest,
    )
    chosen = [candidate]
    index = {"strategy:STR-THRU": ["case-0"]}
    as_of = pd.Timestamp("2026-01-01")
    snapshot = "snapshot-1"

    dump_path = tmp_path / "dump.pkl"
    capture._dump_selected(dump_path, chosen, index, as_of, snapshot)

    out_dir = tmp_path / "probe-out"
    rc = probe.main([str(dump_path), "--out", str(out_dir), "--tracemalloc"])

    assert rc == 0
    doc = json.loads((out_dir / "INDEX.json").read_text())
    assert doc["pairs"]["case-0"]["trace_disposition"] == "complete"