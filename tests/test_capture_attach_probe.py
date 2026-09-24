"""tools/capture_attach_probe.py: replays attach_strict_probe/write against
a CAPTURE_DUMP_SELECTED dump, standalone. Synthetic: builds one real
strict-traceable STR-THRU candidate (the same fixture shape
tests/test_phase4_capture_strict.py already uses for this), dumps it with
capture_tier0_corpus._dump_selected, and checks the probe reproduces the
same corpus a direct `write(..., strict_trace=True)` call would."""
from __future__ import annotations

import json
import pickle

import pandas as pd
import pytest

import tools.capture_attach_probe as probe
import tools.capture_tier0_corpus as capture

from tests.test_capture_heartbeat import FakeClock
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


# ---------------------------------------------------------------------------
# blocker 2 (spec_2): the offline recapture path carries the SAME lifecycle,
# with a provisional profile of ONLY attach + write (~62 min) -- it never
# gathers, so the 81-min full-capture prior would over-state the wait.
# ---------------------------------------------------------------------------


def test_probe_heartbeat_profile_is_attach_write_only_and_load_is_outstanding():
    """Driving the probe's factory directly: while the (potentially slow)
    dump load blocks, the whole-operation ETA is the prior-run attach+write
    (~62m) -- NOT the 81m full capture, and load itself is shown as
    completed/total so a stuck load is visible, not silent."""
    clock = FakeClock()
    lines: list[str] = []
    hb = probe._new_heartbeat(clock=clock, emit=lines.append)
    with hb:
        hb.begin_stage("load", units_total=1)
        hb.phase("load-dump", total=1)  # exactly what probe.main() does
        load_line = hb.beat()
    assert "phase=load-dump" in load_line
    assert "done=0/1" in load_line
    assert "~62m" in load_line           # attach 41 + write 21, gather excluded
    assert "~81m" not in load_line and "~82m" not in load_line
    assert "prior-run estimate" in load_line  # nothing observed yet
    assert "gather" not in probe.PROBE_STAGE_ORDER


def test_probe_main_instruments_the_replay_and_preserves_the_corpus(tmp_path, monkeypatch):
    """The full offline path: main() emits heartbeat lines through every
    stage (load, attach, write-pairs, finalize/publish) yet the written
    corpus is byte-clean -- no heartbeat text leaks into any fixture file
    (stdout/logging is the ONLY thing the heartbeat may add) -- and the
    capture-module singleton is restored afterwards."""
    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
    path, digest = _artifact(tmp_path)
    candidate = _full_strict_candidate(
        fixture_id="case-0", ticker="AAA", driver_vector={"x": 2.0},
        gate_vector={"x": 9.0, "n_prior": 5.0}, path=path, digest=digest,
    )
    chosen = [candidate]
    dump_path = tmp_path / "dump.pkl"
    capture._dump_selected(dump_path, chosen, {"strategy:STR-THRU": ["case-0"]},
                           pd.Timestamp("2026-01-01"), "snapshot-1")

    lines: list[str] = []
    real = probe._new_heartbeat
    created: dict = {}

    def factory(**kw):
        kw.setdefault("clock", FakeClock())
        kw.setdefault("emit", lines.append)
        created["hb"] = real(**kw)
        return created["hb"]

    monkeypatch.setattr(probe, "_new_heartbeat", factory)
    # Force a beat DURING the (potentially slow) dump load so the load phase
    # is exercised deterministically, exactly as the per-minute thread would
    # once it fires while a real multi-GB pickle is being read.
    real_load = pickle.load

    def load_and_beat(fh):
        payload = real_load(fh)
        created["hb"].beat()
        return payload

    monkeypatch.setattr(pickle, "load", load_and_beat)
    out_dir = tmp_path / "probe-out"
    rc = probe.main([str(dump_path), "--out", str(out_dir)])

    assert rc == 0
    assert lines, "probe emitted no heartbeat line"
    # The t=0 beat (before begin_stage) already carries the 62-min attach/
    # write profile, never gather; and the load blocks are logged too.
    assert any("[probe] heartbeat" in line and "gather" not in line for line in lines)
    assert any("phase=load-dump" in line and "eta=~62m" in line for line in lines)
    for json_file in out_dir.rglob("*.json"):
        assert "heartbeat" not in json_file.read_text()
    doc = json.loads((out_dir / "INDEX.json").read_text())
    assert doc["pairs"]["case-0"]["trace_disposition"] == "complete"
    assert capture.heartbeat() is not created["hb"]


def test_probe_stops_the_heartbeat_and_restores_state_on_failure(tmp_path, monkeypatch):
    """A replay that raises still tears the thread down and hands the
    capture-module singleton back -- no orphan heartbeat, no leaked state."""
    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
    path, digest = _artifact(tmp_path)
    candidate = _full_strict_candidate(
        fixture_id="case-0", ticker="AAA", driver_vector={"x": 2.0},
        gate_vector={"x": 9.0, "n_prior": 5.0}, path=path, digest=digest,
    )
    dump_path = tmp_path / "dump.pkl"
    capture._dump_selected(dump_path, [candidate], {"strategy:STR-THRU": ["case-0"]},
                           pd.Timestamp("2026-01-01"), "snapshot-1")

    created: dict = {}
    real = probe._new_heartbeat
    monkeypatch.setattr(probe, "_new_heartbeat",
                        lambda **kw: created.setdefault("hb", real(**kw)))

    def _boom(*args, **kwargs):
        raise RuntimeError("attach died mid-replay")

    monkeypatch.setattr(capture, "write", _boom)
    with pytest.raises(RuntimeError):
        probe.main([str(dump_path), "--out", str(tmp_path / "out")])

    assert created["hb"].running is False
    assert not created["hb"].is_alive
    assert capture.heartbeat() is not created["hb"]