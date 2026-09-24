"""The tier-0 corpus check, and its negative controls.

`component_contracts.md` §9.5 and §15.3 define what the check must prove; each
case in :mod:`checks.tier0_corpus` is proved here by corrupting a synthetic
corpus and asserting the case goes red, and each coverage axis has a crafted
positive and negative record.

A synthetic corpus rather than the real one, deliberately: the real corpus
carries licensed quotes and is not in this repository, so a test that needed
it would not run on a clean checkout. The real corpus is exercised by
``checks/rearchitecture_phase0_gate.py``, which reports honestly when it is
absent.
"""
from __future__ import annotations

import copy
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import tier0_corpus as t0  # noqa: E402
from engine.v2.diagnosis import AGREE, DIFFER, INCOMPARABLE, content_hash  # noqa: E402

MENU = ["TWIN-P5", "BFLY-P", "CND-PS"]
AXIS_INPUTS = {
    "structures": ["BFLY-P", "CAL-P", "CND-PS", "STR-RUNUP", "STR-THRU", "TWIN-P5"],
    "dynamic_strategy": "DYN-SV",
    "menu": MENU,
    "disabled": ["CAL-P"],
    "model_roles": ["size", "implied_t1", "runup_move", "iv_crush", "gate", "chooser"],
    "refusal_code_mapping": {code: code for code in (
        "UNVALIDATED_STRUCTURE", "NO_CHAIN", "BAD_QUOTE", "COARSE_LADDER", "NO_FORECAST")},
}
LEGS = [{"name": "dn1", "strike": 240.0, "qty": 1},
        {"name": "atm", "strike": 250.0, "qty": -2},
        {"name": "up1", "strike": 260.0, "qty": 1}]
WIDTH = 0.05123456789012


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def request(strategy: str, **fields) -> dict:
    base = {"ticker": "MTN", "strategy": strategy, "event_date": "2026-09-28",
            "session": "AMC", "structure_params": None, "strike": None,
            "fill": {"policy_id": "legacy.fill_alpha.v1", "alpha": 0.5}}
    base.update(fields)
    base["identity_key"] = f"MTN|{strategy}|{json.dumps(fields, sort_keys=True)}"
    return base


def priced(strategy: str, width: float = WIDTH, **fields) -> dict:
    record = {"ticker": "MTN", "strategy": strategy, "event_date": "2026-09-28",
              "session": "AMC", "entry_date": "2026-09-28", "exit_date": "2026-09-29",
              "legs": copy.deepcopy(LEGS), "entry_cost": 3.45, "spot": 249.88,
              "forecast_abs_move": 5.218743916, "forecast_p10": 2.91,
              "forecast_p90": 8.0, "forecast_sd": 1.99, "forecast_model": "size_v1_4",
              "forecast_fold": "2025-01-01", "ci_low": -0.01193117,
              "ci_high": 0.04217742, "structure_params": {"width_moneyness": width},
              "flags": []}
    record.update(fields)
    return record


def refusal(strategy: str, flag: str, **fields) -> dict:
    record = {"ticker": "MTN", "strategy": strategy, "event_date": "2026-09-28",
              "session": "BMO", "structure_params": {}, "legs": [], "flags": [flag]}
    record.update(fields)
    return record


def pair(fid: str, req: dict, rec: dict, kind: str = "score_result",
         relations: dict | None = None) -> dict:
    payload = {"request": req, "record": rec, "record_kind": kind}
    if relations:
        payload["relations"] = relations
    return {
        "schema_version": "tier0_pair.v1.1",
        "fixture_id": fid,
        "covers": t0.derive_covers(rec, req, kind, AXIS_INPUTS, relations),
        "notes": "",
        "payload": payload,
        "payload_hash": content_hash(payload),
        "request_hash": content_hash(req),
        "envelope": {"captured_at": "2026-09-12T00:00:00.000000+00:00",
                     "worker_ref": "test:1", "duration_seconds": 0.01},
    }


def standard_pairs() -> list[dict]:
    """Six pairs that together carry every seeded control and a pinned relation."""
    selector = request("TWIN-P5")
    return [
        pair("000_TWIN-P5", selector, priced("TWIN-P5")),
        pair("001_TWIN-P5-pinned", request("TWIN-P5", structure_params={"width_moneyness": WIDTH}),
             priced("TWIN-P5"), relations={"pinned_from": content_hash(selector)}),
        pair("002_STR-THRU", request("STR-THRU", strike=14.7615), refusal("STR-THRU", "NO_CHAIN")),
        pair("003_CAL-P", request("CAL-P"), refusal("CAL-P", "UNVALIDATED_STRUCTURE")),
        pair("004_BFLY-P", request("BFLY-P", strike=250.0), priced("BFLY-P", 0.061246010132397256)),
        pair("005_CND-PS", request("CND-PS"), priced("CND-PS", 0.0234567891234)),
    ]


def build(root: Path, pairs: list[dict]) -> Path:
    (root / "pairs").mkdir(parents=True, exist_ok=True)
    for old in (root / "pairs").glob("*.json"):
        old.unlink()
    coverage: dict[str, list[str]] = {}
    for p in pairs:
        (root / "pairs" / f"{p['fixture_id']}.json").write_text(
            json.dumps(p, indent=2, sort_keys=True) + "\n")
        for axis in p["covers"]:
            coverage.setdefault(axis, []).append(p["fixture_id"])
    (root / "INDEX.json").write_text(json.dumps({
        "schema_version": "tier0_corpus.v1.1",
        "pairs": {p["fixture_id"]: {"payload_hash": p["payload_hash"],
                                    "request_hash": p["request_hash"],
                                    "record_kind": p["payload"]["record_kind"],
                                    "covers": p["covers"]} for p in pairs},
        "axis_inputs": AXIS_INPUTS,
        "refusal_code_mapping": AXIS_INPUTS["refusal_code_mapping"],
        "coverage": coverage,
        "required_axes": sorted(coverage),
        "uncovered_axes": [],
        "corpus_hash": content_hash({p["fixture_id"]: p["payload_hash"] for p in pairs}),
    }, indent=2, sort_keys=True) + "\n")
    return root


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return build(tmp_path / "tier0", standard_pairs())


def _pair_path(root: Path, fixture_id: str) -> Path:
    return root / "pairs" / f"{fixture_id}.json"


def _rewrite(root: Path, fixture_id: str, mutate) -> None:
    """Edit a pair file WITHOUT re-hashing it — a tampered or corrupted file."""
    path = _pair_path(root, fixture_id)
    doc = json.loads(path.read_text())
    mutate(doc)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def covers(record: dict, req: dict | None = None, kind: str = "score_result",
           relations: dict | None = None) -> set[str]:
    return set(t0.derive_covers(record, req or request(record["strategy"]), kind,
                                AXIS_INPUTS, relations))


# --------------------------------------------------------------------------
# a healthy corpus
# --------------------------------------------------------------------------


def test_a_healthy_corpus_agrees(corpus):
    merged, cases = t0.run(corpus)
    assert merged.verdict == AGREE, merged.summary()
    assert set(cases) == {"manifest", "corpus_replay", "coverage", "pinned_counterparts",
                          "seeded_controls", "batch_vs_single", "fresh_process"}
    assert all(r.verdict == AGREE for r in cases.values())


def test_it_runs_well_inside_the_ten_second_budget(corpus):
    started = time.monotonic()
    t0.run(corpus)
    assert time.monotonic() - started < t0.TIME_BUDGET_SECONDS


def test_it_loads_no_panel_and_opens_no_socket(corpus):
    """§7.2: no network, no panel load, no fitting on replay."""
    script = (
        "import sys, json; sys.path.insert(0, %r)\n"
        "from pathlib import Path\n"
        "from checks import tier0_corpus as t0\n"
        "merged, _ = t0.run(Path(%r))\n"
        "print(json.dumps({'verdict': merged.verdict,\n"
        "  'pandas': 'pandas' in sys.modules,\n"
        "  'score': 'engine.score' in sys.modules}))\n"
    ) % (str(ROOT), str(corpus))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, cwd=ROOT, check=True)
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == {"verdict": AGREE, "pandas": False, "score": False}


def test_the_network_guard_actually_refuses():
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from checks import tier0_corpus as t0\n"
        "t0._forbid_network()\n"
        "import socket\n"
        "try:\n"
        "    socket.socket()\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__)\n"
    ) % str(ROOT)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, cwd=ROOT, check=True)
    assert "_NetworkUsed" in proc.stdout


def test_the_corpus_is_released_before_the_fresh_process_forks(tmp_path, monkeypatch):
    """Regression for the v20/v21 OOM kills: on the real corpus, the parent
    held its fully-loaded ``Corpus`` (~5.87 GB) while ``case_fresh_process``
    forked a subprocess that loaded a full second copy -- ~11.7 GB peak on a
    10.7 GB box. ``_run_loaded`` must drop its own last reference to
    ``corpus`` before that subprocess call, so the object is actually
    collectable (not merely ``del``-ed while something else still holds it).

    ``t0.subprocess.run`` is stubbed so this never spawns a real process, and
    the synthetic ``corpus`` fixture (a handful of tiny fixtures) stands in
    for the real 3.2 GB corpus -- this test proves the LIFETIME property, not
    a real memory measurement (the caller runs that against the real corpus
    separately).

    Deliberately does NOT bind the loaded ``Corpus`` to a name in this test's
    own frame: a named local here would keep its own reference alive across
    the ``_run_loaded`` call regardless of anything ``_run_loaded`` does
    internally (verified empirically: an inline call argument has exactly
    one owner -- the callee's parameter -- while a caller-side named local
    is a second, independent owner that survives the callee's ``del``). This
    mirrors exactly how ``_run`` and ``main()`` call ``_run_loaded`` in
    ``checks/tier0_corpus.py``. A test that merely asserted ``del`` was
    called, without this, would pass even if the fix did nothing.
    """
    import gc
    import tempfile
    import weakref

    root = build(tmp_path / "tier0", standard_pairs())
    observed: dict[str, bool] = {}

    def fake_subprocess_run(*_args, **_kwargs):
        gc.collect()
        observed["corpus_alive_at_fork"] = ref_box[0]() is not None
        return subprocess.CompletedProcess(_args, 0, stdout=f"{AGREE}\n", stderr="")

    monkeypatch.setattr(t0.subprocess, "run", fake_subprocess_run)

    ref_box: list = []

    def _load_and_track():
        # Returned straight into `_run_loaded(...)` below with no
        # intervening `corpus = ...` in THIS frame.
        loaded = t0.load(root)
        ref_box.append(weakref.ref(loaded))
        return loaded

    with tempfile.TemporaryDirectory(prefix="tier0-corpus-lifetime-") as tmp:
        merged, cases, report_extra = t0._run_loaded(
            _load_and_track(), root, Path(tmp))

    assert "corpus_alive_at_fork" in observed, "the stubbed subprocess call never fired"
    assert observed["corpus_alive_at_fork"] is False, (
        "the corpus was still resident when case_fresh_process forked")
    gc.collect()
    assert ref_box[0]() is None, "the corpus is still referenced somewhere after the run"
    assert cases["fresh_process"].verdict == AGREE
    assert report_extra is not None and report_extra["pairs"] == len(standard_pairs())
    assert merged.verdict == AGREE, merged.summary()


# --------------------------------------------------------------------------
# progress: a battery that takes tens of minutes on the real corpus must
# never go a minute without a line (see checks.tier0_corpus.Progress)
# --------------------------------------------------------------------------


class _FakeClock:
    def __init__(self, start=100.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _Recorder:
    """Duck-typed ``Progress`` consumer: records begins, counts ticks and
    beats, prints nothing."""

    def __init__(self):
        self.begins = []
        self.ticks = 0
        self.beats = 0

    def begin(self, phase, total, unit="cases"):
        self.begins.append((phase, total))

    def tick(self, n=1):
        self.ticks += n

    def beat(self):
        self.beats += 1


def test_progress_line_reports_completed_total_elapsed_and_flushes():
    stream = io.StringIO()
    ticker = t0.Progress(name="corpus_load", prefix="[phase4_real]",
                         stream=stream, clock=_FakeClock())
    ticker.begin("load", 20, unit="pairs")
    ticker.tick()
    assert stream.getvalue().splitlines() == [
        "[phase4_real] PROGRESS corpus_load load 1/20 pairs elapsed=0.0s"]


def test_progress_lines_are_gated_to_at_most_one_per_interval():
    clock = _FakeClock()
    stream = io.StringIO()
    ticker = t0.Progress(stream=stream, clock=clock, interval=60.0)
    ticker.begin("digest", 4, unit="pairs")
    seen_lines = []
    for _ in range(4):
        ticker.tick()
        seen_lines.append(len(stream.getvalue().splitlines()))
        clock.advance(40.0)
    assert seen_lines == [1, 1, 2, 2]
    lines = stream.getvalue().splitlines()
    assert lines[0].startswith("[tier0] PROGRESS digest 1/4 pairs")
    assert lines[1].startswith("[tier0] PROGRESS digest 3/4 pairs")


def test_eta_appears_only_with_a_known_total_and_enough_completed_cases():
    stream = io.StringIO()
    clock = _FakeClock()
    ticker = t0.Progress(stream=stream, clock=clock, interval=0.0)
    ticker.begin("fixtures", 6)
    for _ in range(6):
        ticker.tick()
        clock.advance(10.0)
    # ticks at t=0..50: the done=1/2 lines have too few samples for an
    # average, the done=3 line extrapolates (20/3)*3=20.0s, and the final
    # 6/6 line has nothing left to estimate. No stage profile -> no
    # provisional eta either.
    lines = stream.getvalue().splitlines()
    assert "eta=" not in lines[0] and "eta=" not in lines[1]
    assert "3/6 cases elapsed=20.0s phase_eta=20.0s" in lines[2]
    assert "6/6 cases" in lines[5] and "eta=" not in lines[5]
    unknown = io.StringIO()
    ticker = t0.Progress(stream=unknown, clock=_FakeClock(), interval=0.0)
    ticker.begin("mystery", 0)
    for _ in range(4):
        ticker.tick()
    lines = unknown.getvalue().splitlines()
    assert "1/? cases" in lines[0] and "4/? cases" in lines[-1]
    assert all("eta=" not in line for line in lines)


def test_provisional_eta_shows_from_stage_start_then_gives_way_to_observed():
    clock = _FakeClock()
    stream = io.StringIO()
    ticker = t0.Progress(stream=stream, clock=clock, interval=0.0,
                         stage_seconds=100.0)
    ticker.begin("fixtures", 6)
    ticker.tick()
    assert "eta~=100.0s" in stream.getvalue().splitlines()[-1]
    clock.advance(10.0)
    ticker.tick()
    assert "eta~=90.0s" in stream.getvalue().splitlines()[-1]
    clock.advance(10.0)
    ticker.tick()  # done=3: the observed phase estimate takes over
    last = stream.getvalue().splitlines()[-1]
    assert "eta~=" not in last and "elapsed=20.0s phase_eta=20.0s" in last
    clock.advance(1000.0)
    ticker.begin("phase_two", 3)
    ticker.tick()
    # profile overrun with too few samples for a phase estimate: the
    # provisional now COUNTS UP past the spent budget (best effort, still
    # labeled) instead of dropping the ETA.
    last = stream.getvalue().splitlines()[-1]
    assert "eta~=920.0s" in last and "phase_eta=" not in last


def test_gate_eta_rides_every_line_and_beat_beside_the_phase_estimate():
    clock = _FakeClock()
    stream = io.StringIO()

    class _StubGate:
        def __init__(self):
            self.consulted = []

        def remaining(self, ticker, now):
            self.consulted.append(ticker.done)
            return 9999.0

    gate = _StubGate()
    ticker = t0.Progress(name="native_parity", stream=stream, clock=clock,
                         interval=0.0, gate=gate, stage_seconds=1234.0)
    ticker.begin("fixtures", 6)
    ticker.tick()
    ticker.tick()
    ticker.tick()
    ticker.beat()  # a blocked unit still shows the whole-gate estimate
    lines = stream.getvalue().splitlines()
    assert all("gate_eta=9999.0s" in line for line in lines)
    assert "phase_eta=" not in lines[0] and "phase_eta=" in lines[2]
    assert gate.consulted == [1, 2, 3, 3]  # every line consults the gate
    assert not any("eta~=" in line for line in lines)  # the gate wins


def test_heartbeat_prints_interim_lines_while_a_single_unit_blocks():
    """The P1 property: ticks alone cannot bound the silent gap -- one
    100 MB pair or one heavy native replay can span many minutes. The
    timer beats regardless, and counts stay honest: every interim line
    shows the blocked unit NOT yet completed."""
    stream = io.StringIO()
    ticker = t0.Progress(stream=stream, interval=0.08)
    ticker.begin("round_trip", 3, unit="pairs")
    ticker.start_heartbeat()
    assert ticker.heartbeat_active()
    try:
        time.sleep(0.3)  # one unit spanning many heartbeat checks
    finally:
        ticker.stop_heartbeat()
    lines = stream.getvalue().splitlines()
    assert len(lines) >= 2, lines
    assert all("round_trip 0/3 pairs" in line for line in lines)
    assert not ticker.heartbeat_active()
    ticker.start_heartbeat()
    ticker.start_heartbeat()  # double start keeps exactly one timer
    ticker.stop_heartbeat()
    ticker.stop_heartbeat()   # and stop is idempotent
    assert not ticker.heartbeat_active()
    ticker.tick()
    assert ticker.done == 1  # counts accurate before/after either timer


def test_heartbeat_leaks_no_thread_when_the_owned_block_raises():
    ticker = t0.Progress(stream=io.StringIO(), interval=0.08)
    ticker.begin("fixtures", 3)
    ticker.start_heartbeat()
    with pytest.raises(RuntimeError, match="body died"):
        try:
            raise RuntimeError("body died")
        finally:
            ticker.stop_heartbeat()
    assert not ticker.heartbeat_active()


def test_standalone_progress_flag_keeps_stdout_verdict_only(corpus):
    """`--progress` may run this CLI in the fresh-process CHILD role,
    where stdout is the parsed verdict: it must stay one clean line even
    while the heartbeat and the load ticks are active."""
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from checks import tier0_corpus as t0\n"
        "raise SystemExit(t0.main(['--corpus', %r, '--emit-verdict', '--progress']))\n"
    ) % (str(ROOT), str(corpus))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, cwd=ROOT, check=True, timeout=60)
    assert proc.stdout.strip().splitlines()[-1] == AGREE


def test_fresh_process_reaps_the_child_when_the_parent_wait_raises(monkeypatch,
                                                                   tmp_path):
    """P2 regression: if communicate() or a beat raises, the child must be
    terminated and reaped before the ORIGINAL exception propagates -- an
    instrumented run must not orphan a corpus-loading process."""
    events = {}

    class _Child:
        stdout = io.StringIO()

        def __init__(self, command, **kwargs):
            pass

        def communicate(self, timeout=None):
            raise RuntimeError("parent-side wait blew up")

        def terminate(self):
            events["terminated"] = True

        def kill(self):
            events["killed"] = True

        def wait(self, timeout=None):
            events["reaped"] = True
            return 0

    monkeypatch.setattr(t0.subprocess, "Popen", _Child)
    with pytest.raises(RuntimeError, match="blew up"):
        t0.case_fresh_process(tmp_path, AGREE, _Recorder())
    assert events == {"terminated": True, "reaped": True}


def test_fresh_process_escalates_to_kill_when_the_child_ignores_term(monkeypatch,
                                                                     tmp_path):
    events = {"waits": 0}

    class _StubbornChild:
        stdout = io.StringIO()

        def __init__(self, command, **kwargs):
            pass

        def communicate(self, timeout=None):
            raise RuntimeError("a beat raised instead")

        def terminate(self):
            events["terminated"] = True

        def kill(self):
            events["killed"] = True

        def wait(self, timeout=None):
            events["waits"] += 1
            if events["waits"] == 1:
                raise subprocess.TimeoutExpired("child", timeout)
            events["reaped"] = True
            return -9

    monkeypatch.setattr(t0.subprocess, "Popen", _StubbornChild)
    with pytest.raises(RuntimeError, match="a beat raised instead"):
        t0.case_fresh_process(tmp_path, AGREE, _Recorder())
    assert events["terminated"] and events["killed"] and events.get("reaped")


def test_beat_keeps_the_line_moving_without_counting_a_case():
    clock = _FakeClock()
    stream = io.StringIO()
    ticker = t0.Progress(stream=stream, clock=clock, interval=60.0)
    ticker.begin("fresh_process", 1, unit="subprocess")
    ticker.beat()
    clock.advance(30.0)
    ticker.beat()
    clock.advance(30.0)
    ticker.beat()
    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    assert all("fresh_process 0/1 subprocess" in line for line in lines)


def test_count_suffix_and_the_no_progress_default():
    ticker = t0.Progress(stream=io.StringIO(), clock=_FakeClock())
    assert ticker.count_suffix() == ", 0/? cases"
    ticker.begin("fixtures", 4)
    ticker.tick(2)
    assert ticker.count_suffix() == ", 2/4 cases"
    t0.NO_PROGRESS.begin("x", 1)
    t0.NO_PROGRESS.tick()
    t0.NO_PROGRESS.beat()
    assert t0.NO_PROGRESS.count_suffix() == ""


def test_load_ticks_one_case_per_pair_file(corpus):
    rec = _Recorder()
    loaded = t0.load(corpus, progress=rec)
    assert rec.begins == [("load", len(standard_pairs()))]
    assert rec.ticks == len(loaded.pairs)


def test_per_pair_battery_cases_tick_once_per_pair(corpus):
    loaded = t0.load(corpus)
    rec = _Recorder()
    t0.seeded_controls(loaded, rec)
    assert rec.begins == [("seeded_controls", len(loaded.pairs))]
    assert rec.ticks == len(loaded.pairs)
    rec = _Recorder()
    t0.derived_uncovered(loaded, rec)
    assert rec.begins == [("coverage", len(loaded.pairs))]
    assert rec.ticks == len(loaded.pairs)
    rec = _Recorder()
    t0.case_digest(loaded, rec)
    assert rec.begins == [("digest", len(loaded.pairs))]
    assert rec.ticks == len(loaded.pairs)


def test_fresh_process_without_a_ticker_stays_a_plain_captured_run(monkeypatch,
                                                                   tmp_path):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout=f"{AGREE}\n",
                                           stderr="")

    monkeypatch.setattr(t0.subprocess, "run", fake_run)
    receipt = t0.case_fresh_process(tmp_path, AGREE)
    assert receipt.verdict == AGREE
    assert "--progress" not in seen["command"]
    assert seen["kwargs"]["capture_output"] is True


def test_fresh_process_with_a_ticker_beats_while_the_child_streams(monkeypatch,
                                                                   tmp_path):
    """With a ticker the child is launched with ``--progress`` and an
    INHERITED stderr (it emits its own case lines straight to the
    operator), while its stdout stays pipe-captured because that last
    stdout line IS the compared verdict. Each timeout of the parent's
    poll loop beats the ticker, so the wait is never a silent minute."""
    calls = {}

    class _Child:
        stdout = io.StringIO()

        def __init__(self, command, **kwargs):
            calls["command"] = command
            calls.update(kwargs)
            self._timeouts_left = 1

        def communicate(self, timeout=None):
            if self._timeouts_left:
                self._timeouts_left -= 1
                raise subprocess.TimeoutExpired(calls["command"], timeout)
            return f"{AGREE}\n", None

    monkeypatch.setattr(t0.subprocess, "Popen", _Child)
    rec = _Recorder()
    receipt = t0.case_fresh_process(tmp_path, AGREE, rec)
    assert receipt.verdict == AGREE
    assert calls["command"][-1] == "--progress"
    assert calls["stderr"] is None
    assert calls["stdout"] is subprocess.PIPE
    assert rec.begins == [("fresh_process", 1)]
    assert rec.beats == 1 and rec.ticks == 0


def test_run_threads_one_ticker_through_every_per_pair_pass(corpus, monkeypatch):
    def fake_fresh(root, this_verdict, progress=t0.NO_PROGRESS):
        progress.begin("fresh_process", 1, unit="subprocess")
        progress.beat()
        return t0.compare_records(
            {"verdict": this_verdict}, {"verdict": this_verdict},
            comparison_kind="tier0_fresh_process",
            left_ref="this-process", right_ref="subprocess")

    monkeypatch.setattr(t0, "case_fresh_process", fake_fresh)
    rec = _Recorder()
    merged, cases = t0.run(corpus, rec)
    assert merged.verdict == AGREE, merged.summary()
    assert cases["fresh_process"].verdict == AGREE
    assert rec.begins[0] == ("load", len(standard_pairs()))
    phases = [phase for phase, _ in rec.begins]
    for expected in ("addressing", "digest", "round_trip", "batch_singles",
                     "coverage", "seeded_controls", "fresh_process"):
        assert expected in phases, phases
    assert rec.ticks >= 8 * len(standard_pairs())
    assert rec.beats == 1


# --------------------------------------------------------------------------
# integrity: addressing, digest, manifest
# --------------------------------------------------------------------------


def test_a_rounded_request_stops_addressing_its_record(corpus):
    """`b33036c`: the lookup key is the hash of the FULL-PRECISION request."""
    _rewrite(corpus, "000_TWIN-P5", lambda d: d["payload"]["request"].update(
        {"identity_key": d["payload"]["request"]["identity_key"] + "-rounded"}))
    merged, cases = t0.run(corpus)
    assert merged.verdict == DIFFER
    assert cases["corpus_replay"].verdict == DIFFER
    assert {"request_hash", "resolves_to[0]"} & {f.field_path for f in merged.findings}


def test_a_file_that_disagrees_with_its_digest_fails(corpus):
    """`6b9d5cf`: rounding reapplied after the exemption."""
    _rewrite(corpus, "004_BFLY-P", lambda d: d["payload"]["record"]["structure_params"].update(
        {"width_moneyness": 0.061246}))
    merged, _ = t0.run(corpus)
    assert merged.verdict == DIFFER
    assert any(f.field_path == "payload_hash" for f in merged.findings)


def test_an_empty_corpus_is_incomparable_not_agreement(tmp_path):
    root = tmp_path / "tier0"
    (root / "pairs").mkdir(parents=True)
    (root / "INDEX.json").write_text(json.dumps(
        {"pairs": {}, "coverage": {}, "required_axes": [], "uncovered_axes": []}))
    merged, _ = t0.run(root)
    assert merged.verdict == INCOMPARABLE


def test_a_missing_corpus_exits_nonzero_rather_than_passing(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "checks" / "tier0_corpus.py"),
         "--corpus", str(tmp_path / "nothing")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 1
    assert "CORPUS_MISSING" in proc.stderr or "INCOMPARABLE" in proc.stderr


def test_batch_and_single_agree_on_a_corrupted_corpus_too(corpus):
    """The batch must not hide a finding the singles would have reported."""
    _rewrite(corpus, "003_CAL-P", lambda d: d["payload"]["record"].update({"ci_low": -9.9}))
    merged, cases = t0.run(corpus)
    assert cases["batch_vs_single"].verdict == AGREE
    assert merged.verdict == DIFFER


def test_deleting_fixtures_and_keeping_the_index_is_not_a_pass(corpus):
    """The 2026-09-12 review probe: keep one file, leave the index alone."""
    keep = "000_TWIN-P5"
    deleted = []
    for path in (corpus / "pairs").glob("*.json"):
        if path.stem != keep:
            deleted.append(path.stem)
            path.unlink()
    merged, cases = t0.run(corpus)
    assert cases["manifest"].verdict == DIFFER
    paths = {f.field_path for f in cases["manifest"].findings}
    assert any(p.startswith("pair_ids") for p in paths)
    for fid in deleted:
        assert f"pairs.{fid}.payload_hash" in paths
    assert merged.verdict != AGREE
    assert cases["coverage"].verdict == DIFFER


def test_an_extra_undeclared_file_is_a_finding(corpus):
    extra = json.loads(_pair_path(corpus, "000_TWIN-P5").read_text())
    extra["fixture_id"] = "999_UNDECLARED"
    (corpus / "pairs" / "999_UNDECLARED.json").write_text(json.dumps(extra))
    _, cases = t0.run(corpus)
    assert cases["manifest"].verdict == DIFFER


def test_a_tampered_manifest_hash_is_a_finding(corpus):
    index = json.loads((corpus / "INDEX.json").read_text())
    index["pairs"]["003_CAL-P"]["payload_hash"] = "sha256:" + "0" * 64
    (corpus / "INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    _, cases = t0.run(corpus)
    assert cases["manifest"].verdict == DIFFER


def test_a_tampered_corpus_hash_is_a_finding(corpus):
    index = json.loads((corpus / "INDEX.json").read_text())
    index["corpus_hash"] = "sha256:" + "0" * 64
    (corpus / "INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    _, cases = t0.run(corpus)
    assert cases["manifest"].verdict == DIFFER


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------


def test_a_missing_strategy_axis_claim_fails_coverage(corpus):
    index = json.loads((corpus / "INDEX.json").read_text())
    index["coverage"].pop("strategy:CAL-P")
    (corpus / "INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    _, cases = t0.run(corpus)
    assert cases["coverage"].verdict == DIFFER
    assert any(f.field_path.startswith("axes.strategy:CAL-P")
               for f in cases["coverage"].findings)


def test_a_missing_refusal_code_claim_fails_coverage(corpus):
    index = json.loads((corpus / "INDEX.json").read_text())
    index["coverage"].pop("refusal:NO_CHAIN")
    (corpus / "INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    _, cases = t0.run(corpus)
    assert cases["coverage"].verdict == DIFFER


def test_deleting_the_only_priced_fixture_loses_its_axis(corpus):
    _pair_path(corpus, "004_BFLY-P").unlink()
    _, cases = t0.run(corpus)
    paths = {f.field_path for f in cases["coverage"].findings}
    assert any(p.startswith("axes.priced:BFLY-P") for p in paths)


def test_a_bad_quote_priced_row_covers_priced_but_not_priced_clean():
    """Legacy prices the entry (legs + cost resolve) before checking
    BAD_QUOTE, so a BAD_QUOTE row is `priced` without ever reaching
    simulation/gate -- exactly the STR-THRU/ISPR corpus case this axis
    exists to make `select()` stop treating as sufficient."""
    got = covers(priced("STR-THRU", flags=["BAD_QUOTE"]))
    assert "priced:STR-THRU" in got
    assert "priced_clean:STR-THRU" not in got


def test_a_priced_row_with_no_early_exit_flag_covers_priced_clean():
    got = covers(priced("STR-THRU"))
    assert {"priced:STR-THRU", "priced_clean:STR-THRU"} <= got


def test_an_unpriced_refusal_covers_neither_priced_axis():
    got = covers(refusal("STR-THRU", "BAD_QUOTE"))
    assert "priced:STR-THRU" not in got
    assert "priced_clean:STR-THRU" not in got


def test_deleting_the_only_priced_clean_fixture_loses_its_axis(corpus):
    _pair_path(corpus, "004_BFLY-P").unlink()
    _, cases = t0.run(corpus)
    paths = {f.field_path for f in cases["coverage"].findings}
    assert any(p.startswith("axes.priced_clean:BFLY-P") for p in paths)


def test_an_inflated_covers_list_is_a_finding(corpus):
    _rewrite(corpus, "002_STR-THRU", lambda d: d["covers"].append("geometry:round_listed_strike"))
    _, cases = t0.run(corpus)
    assert cases["manifest"].verdict == DIFFER
    assert cases["coverage"].verdict == DIFFER


def test_uncovered_axes_are_derived_from_the_records_not_read_from_the_index(corpus):
    """An index claiming no gaps cannot hide one: the gate reads the derivation."""
    index = json.loads((corpus / "INDEX.json").read_text())
    index["required_axes"].append("dyn_sv:tie")
    (corpus / "INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    assert t0.derived_uncovered(t0.load(corpus)) == ["dyn_sv:tie"]


def test_a_refusal_with_a_computed_strike_does_not_cover_round_listed_strike():
    """The first corpus's round-listed-strike fixture was exactly this row."""
    got = covers(refusal("TWIN-P5", "NO_CHAIN"), request("TWIN-P5", strike=14.7615))
    assert "geometry:round_listed_strike" not in got


def test_a_priced_row_at_one_of_its_listed_strikes_covers_round_listed_strike():
    assert "geometry:round_listed_strike" in covers(
        priced("TWIN-P5"), request("TWIN-P5", strike=250.0))


def test_a_priced_row_whose_legs_snapped_away_from_the_request_does_not():
    assert "geometry:round_listed_strike" not in covers(
        priced("TWIN-P5"), request("TWIN-P5", strike=251.3))


def test_a_pinned_request_covers_pinned_only_beside_its_named_source():
    rec, req = priced("TWIN-P5"), request("TWIN-P5", structure_params={"width_moneyness": WIDTH})
    assert not {"geometry:pinned", "geometry:selector"} & covers(rec, req)
    assert "geometry:pinned" in covers(rec, req, relations={"pinned_from": "sha256:x"})


def test_a_selector_resolved_priced_row_covers_selector_and_computed_width():
    assert {"geometry:selector", "geometry:computed_width"} <= covers(priced("TWIN-P5"))


def test_geometry_axes_need_a_priced_row():
    rec = refusal("TWIN-P5", "NO_FORECAST", structure_params={"width_moneyness": WIDTH})
    assert not {axis for axis in covers(rec) if axis.startswith("geometry:")}


def test_the_coarse_ladder_refusal_covers_its_axis():
    assert "geometry:coarse_ladder" in covers(refusal("TWIN-P5", "COARSE_LADDER"))


def test_an_even_ladder_is_an_exact_mirror_and_an_uneven_one_is_not():
    assert "geometry:exact_mirror" in covers(priced("BFLY-P"))
    uneven = priced("BFLY-P", legs=[{"strike": 240.0}, {"strike": 250.0}, {"strike": 265.0}])
    assert "geometry:exact_mirror" not in covers(uneven)


def test_bad_quote_is_one_refusal_code_not_two():
    got = covers(refusal("STR-THRU", "BAD_QUOTE"))
    assert "refusal:BAD_QUOTE" in got
    assert not any("COST_PCT" in axis for axis in got)


def test_a_disabled_refusal_and_a_research_replay_cover_their_axes():
    assert "disabled:CAL-P:refused" in covers(refusal("CAL-P", "UNVALIDATED_STRUCTURE"))
    assert "disabled:CAL-P:research_replay" in covers(
        {"strategy": "CAL-P", "rows": []}, {"kind": "research_replay"}, kind="research_replay")


def test_boundaries_come_from_the_trade_window():
    across = priced("STR-RUNUP", entry_date="2025-12-22", exit_date="2026-01-06")
    assert {"boundary:year", "boundary:month"} <= covers(across)
    assert not {"boundary:year", "boundary:month"} & covers(priced("TWIN-P5"))


def dyn_record(**fields) -> dict:
    record = {"strategy": "DYN-SV", "chosen_strategy": "BFLY-P", "ticker": "MTN",
              "event_date": "2026-09-28", "session": "AMC", "menu_size": 2,
              "chosen_margin": 0.01, "chooser_score": 0.3,
              "legs": copy.deepcopy(LEGS), "entry_cost": 1.0, "flags": []}
    record.update(fields)
    return record


def dyn_request(strategies: list[str]) -> dict:
    return {"kind": "dyn_sv_resolution", "menu": MENU,
            "frame_rows": [{"request": request(s), "record": {"strategy": s}}
                           for s in strategies]}


def test_a_tie_between_two_structures_covers_tie():
    got = covers(dyn_record(chosen_margin=0.0), dyn_request(["BFLY-P", "TWIN-P5"]),
                 kind="dyn_sv_choice")
    assert {"dyn_sv:tie", "dyn_sv:partial_menu", "priced:DYN-SV"} <= got


def test_a_frame_carrying_one_structure_twice_covers_no_dyn_sv_axis():
    """The first corpus's only tie: BFLY-P against its own pinned re-score."""
    got = covers(dyn_record(chosen_margin=0.0), dyn_request(["BFLY-P", "BFLY-P"]),
                 kind="dyn_sv_choice")
    assert not {axis for axis in got if axis.startswith("dyn_sv:")}


def test_a_nonfinite_chooser_score_on_a_full_menu_is_the_fallback():
    got = covers(dyn_record(chooser_score={"__nonfinite__": "nan"}, menu_size=3),
                 dyn_request(MENU), kind="dyn_sv_choice")
    assert {"dyn_sv:fallback", "dyn_sv:full_menu"} <= got
    assert "dyn_sv:tie" not in got


# --------------------------------------------------------------------------
# pinned counterparts — e845f3e on real-shaped data
# --------------------------------------------------------------------------


def test_a_pinned_fixture_without_its_source_fails(tmp_path):
    pairs = [p for p in standard_pairs() if p["fixture_id"] != "000_TWIN-P5"]
    _, cases = t0.run(build(tmp_path / "tier0", pairs))
    assert cases["pinned_counterparts"].verdict == DIFFER


def test_a_pinned_fixture_that_lost_its_forecast_fails(tmp_path):
    pairs = standard_pairs()
    pinned = pairs[1]
    record = copy.deepcopy(pinned["payload"]["record"])
    record["forecast_abs_move"] = None
    pairs[1] = pair(pinned["fixture_id"], pinned["payload"]["request"], record,
                    relations=pinned["payload"]["relations"])
    _, cases = t0.run(build(tmp_path / "tier0", pairs))
    assert cases["pinned_counterparts"].verdict == DIFFER
    assert any(f.field_path.endswith("forecast_abs_move")
               for f in cases["pinned_counterparts"].findings)


# --------------------------------------------------------------------------
# seeded controls over the corpus
# --------------------------------------------------------------------------


def test_the_seeded_controls_detect_every_cause_on_distinct_pairs(corpus):
    summary = t0.seeded_controls(t0.load(corpus))
    targets = [control["target"] for control in summary["controls"].values()]
    assert None not in targets and len(set(targets)) == len(targets)
    assert all(control["problems"] == [] for control in summary["controls"].values()), summary
    assert summary["untargeted_findings"] == []
    assert summary["field_set_mismatches"] == []
    assert summary["one_pass_verdict"] == DIFFER


def test_the_seeded_controls_fail_when_no_pair_can_carry_a_cause(tmp_path):
    pairs = [p for p in standard_pairs() if p["fixture_id"] != "005_CND-PS"]
    _, cases = t0.run(build(tmp_path / "tier0", pairs))
    assert cases["seeded_controls"].verdict == DIFFER


# --------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------


def test_the_cli_reports_json(corpus):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "checks" / "tier0_corpus.py"),
         "--corpus", str(corpus), "--json"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["verdict"] == AGREE
    assert out["uncovered_axes"] == []
    assert out["pairs"] == out["declared_pairs"] == 6
    assert all(c["problems"] == [] for c in out["seeded_controls"]["controls"].values())


# --------------------------------------------------------------------------
# shared frozen documents (a pair's payload may embed one by reference)
# --------------------------------------------------------------------------


def write_shared(root: Path, value, *, expanded=None) -> str:
    """Write ``value`` (the on-disk storage form, which may itself carry
    nested ``$shared`` references) under ``root/shared/<hex>.json``. The
    digest is the content hash of ``expanded`` (the fully expanded logical
    document, defaulting to ``value`` itself when it carries no nested
    reference) -- the same value a real writer computes, never the hash of
    the reference-shaped storage form.
    """
    digest = content_hash(expanded if expanded is not None else value)
    shared_dir = root / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": t0.SHARED_DOCUMENT_SCHEMA_VERSION,
            "digest": digest, "value": value}
    (shared_dir / f"{digest.split(':', 1)[-1]}.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n")
    return digest


def test_a_shared_reference_resolves_to_one_object_shared_by_every_occurrence(tmp_path):
    """Two DIFFERENT pairs, and two occurrences within one of them, all
    referencing the same fold pool: `load` must read `shared/<hex>.json`
    exactly once and hand every occurrence the SAME Python object back."""
    pool = {"predictions": [0.1, 0.2, 0.3], "tag": "pool"}
    digest = write_shared(tmp_path / "tier0", pool)
    ref = {t0.SHARED_REF_KEY: digest}
    p1 = pair("100_shared_a", request("STR-THRU"),
              priced("STR-THRU", extra_field={"a": ref, "b": ref}))
    p2 = pair("101_shared_b", request("BFLY-P"), priced("BFLY-P", extra_field=ref))
    root = build(tmp_path / "tier0", [p1, p2])
    corpus = t0.load(root)
    a = corpus.pairs["100_shared_a"]["payload"]["record"]["extra_field"]
    b = corpus.pairs["101_shared_b"]["payload"]["record"]["extra_field"]
    assert a["a"] is a["b"] is b
    assert a["a"] == pool


def test_an_unknown_reference_shape_refuses_loudly(tmp_path):
    pool = {"tag": "pool"}
    digest = write_shared(tmp_path / "tier0", pool)
    bad_ref = {t0.SHARED_REF_KEY: digest, "extra": 1}
    p = pair("102_bad_shape", request("STR-THRU"), priced("STR-THRU", extra_field=bad_ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_missing_shared_file_refuses_loudly(tmp_path):
    ref = {t0.SHARED_REF_KEY: "sha256:" + "a" * 64}
    p = pair("103_missing", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_digest_mismatched_shared_file_refuses_loudly(tmp_path):
    pool = {"tag": "pool"}
    digest = write_shared(tmp_path / "tier0", pool)
    # Tamper with the shared file's content after its digest was computed.
    shared_path = tmp_path / "tier0" / "shared" / f"{digest.split(':', 1)[-1]}.json"
    doc = json.loads(shared_path.read_text())
    doc["value"]["tag"] = "tampered"
    shared_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    ref = {t0.SHARED_REF_KEY: digest}
    p = pair("104_tampered", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_shared_document_may_reference_another(tmp_path):
    """Nested sharing: one shared document embeds a reference to another."""
    inner = {"tag": "inner-pool"}
    inner_digest = write_shared(tmp_path / "tier0", inner)
    outer_stored = {"nested": {t0.SHARED_REF_KEY: inner_digest}, "tag": "outer"}
    outer_expanded = {"nested": inner, "tag": "outer"}
    outer_digest = write_shared(tmp_path / "tier0", outer_stored, expanded=outer_expanded)
    ref = {t0.SHARED_REF_KEY: outer_digest}
    p = pair("105_nested", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    corpus = t0.load(root)
    resolved = corpus.pairs["105_nested"]["payload"]["record"]["extra_field"]
    assert resolved["nested"] == inner
    assert resolved["tag"] == "outer"


def test_an_old_format_corpus_has_no_shared_directory_and_loads_unchanged(corpus):
    assert not (corpus / "shared").exists()
    loaded = t0.load(corpus)
    assert loaded.pairs["000_TWIN-P5"]["payload"]["record"]["strategy"] == "TWIN-P5"


# --------------------------------------------------------------------------
# Change B — fragments into case_digest / case_addressing
# --------------------------------------------------------------------------


def _pair_with_shared_ref(fid: str, req: dict, record_with_ref: dict,
                          record_expanded: dict) -> dict:
    """A pair whose STORED payload carries ``{$shared: ...}`` markers
    (``record_with_ref``) but whose ``payload_hash`` is taken over the fully
    EXPANDED value (``record_expanded``) -- the real writer's contract
    (``tools/capture_tier0_corpus.py``'s ``make_pair``/``_SharedDocumentWriter``:
    the digest is computed before shared subtrees are substituted for disk
    references), and what ``load()`` must reproduce after resolving those
    references back. The plain ``pair()`` helper above hashes whatever it is
    given as-is, which is correct for every OTHER test here (none of them
    embed a ``$shared`` reference) but would be the wrong value for one that
    does -- this helper exists so the fragments tests below assert a
    genuinely correct ``payload_hash``, not an artifact of the shortcut.
    """
    kind = "score_result"
    return {
        "schema_version": "tier0_pair.v1.1",
        "fixture_id": fid,
        "covers": t0.derive_covers(record_expanded, req, kind, AXIS_INPUTS, None),
        "notes": "",
        "payload": {"request": req, "record": record_with_ref, "record_kind": kind},
        "payload_hash": content_hash({"request": req, "record": record_expanded,
                                      "record_kind": kind}),
        "request_hash": content_hash(req),
        "envelope": {"captured_at": "2026-09-12T00:00:00.000000+00:00",
                     "worker_ref": "test:1", "duration_seconds": 0.01},
    }


def _shared_fragment_corpus(tmp_path: Path):
    """Two pairs, three occurrences of one shared pool between them (two in
    one pair's own record, one in the other's), with correct payload_hashes
    (see :func:`_pair_with_shared_ref`) so the full case battery can
    legitimately AGREE on them."""
    pool = {"predictions": [0.11, 0.22, 0.33], "tag": "shared-pool"}
    digest = write_shared(tmp_path / "tier0", pool)
    ref = {t0.SHARED_REF_KEY: digest}
    req_a = request("STR-THRU")
    req_b = request("BFLY-P")
    rec_a_ref = priced("STR-THRU", extra_field={"a": ref, "b": ref})
    rec_a_expanded = priced("STR-THRU", extra_field={"a": pool, "b": pool})
    rec_b_ref = priced("BFLY-P", extra_field=ref)
    rec_b_expanded = priced("BFLY-P", extra_field=pool)
    p1 = _pair_with_shared_ref("200_shared_a", req_a, rec_a_ref, rec_a_expanded)
    p2 = _pair_with_shared_ref("201_shared_b", req_b, rec_b_ref, rec_b_expanded)
    root = build(tmp_path / "tier0", [p1, p2])
    return t0.load(root), pool


def test_fragments_do_not_change_the_hash(tmp_path):
    """Required proof for Change B: `content_hash(value, fragments=...)` is
    byte-identical to `content_hash(value)` over the same (real, shared)
    subtree -- `engine/v2/foundation/canonical.py`'s own contract, pinned
    here rather than just trusted."""
    corpus, pool = _shared_fragment_corpus(tmp_path)
    assert corpus.fragments is not None

    record_a = corpus.pairs["200_shared_a"]["payload"]["record"]["extra_field"]
    record_b = corpus.pairs["201_shared_b"]["payload"]["record"]["extra_field"]
    assert record_a["a"] is record_a["b"] is record_b  # one shared object, by identity
    assert record_a["a"] == pool

    for fixture_id in corpus.ordered_ids:
        payload = corpus.pairs[fixture_id]["payload"]
        assert (content_hash(payload, fragments=corpus.fragments)
                == content_hash(payload))
        req = corpus.request_of(fixture_id)
        assert (content_hash(req, fragments=corpus.fragments) == content_hash(req))


def test_case_digest_and_addressing_pass_the_corpus_fragments(tmp_path, monkeypatch):
    """`case_digest`/`case_addressing` must actually pass `fragments=
    corpus.fragments` through to `content_hash` -- not just be capable of
    it if called that way. Spies on the module's `content_hash` name (both
    functions call it as a bare name, so patching the module attribute
    intercepts every call) and asserts the corpus's own fragments object
    was used, and that both cases still AGREE."""
    corpus, _pool = _shared_fragment_corpus(tmp_path)
    assert corpus.fragments is not None

    seen_fragments = []
    real_content_hash = t0.content_hash

    def spy(value, *, fragments=None):
        seen_fragments.append(fragments)
        return real_content_hash(value, fragments=fragments)

    monkeypatch.setattr(t0, "content_hash", spy)
    digest_receipts = t0.case_digest(corpus)
    addressing_receipts = t0.case_addressing(corpus)

    assert corpus.fragments in seen_fragments
    assert all(r.verdict == AGREE for r in digest_receipts)
    assert all(r.verdict == AGREE for r in addressing_receipts)


# --------------------------------------------------------------------------
# shared translation row tables (input_translation.mappings by reference)
# --------------------------------------------------------------------------


def translation_row(shared_path: list, native_path: list, value) -> dict:
    return {"shared_path": shared_path, "native_path": native_path,
            "value_hash": content_hash(value)}


def write_translation_table(root: Path, rows: list) -> str:
    """Write ``rows`` under ``root/shared/translations/<hex>.json``, the
    format ``tools/capture_tier0_corpus.py``'s ``_TranslationTableWriter``
    writes. Returns the digest (``content_hash(rows)`` over the exact array
    stored, per ``SHARED_TRANSLATION_TABLE_SCHEMA_VERSION``).
    """
    digest = content_hash(rows)
    tables_dir = root / "shared" / "translations"
    tables_dir.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": t0.SHARED_TRANSLATION_TABLE_SCHEMA_VERSION,
            "digest": digest, "rows": rows}
    (tables_dir / f"{digest.split(':', 1)[-1]}.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n")
    return digest


def test_an_identity_reference_returns_the_table_rows_exactly(tmp_path):
    """The table itself may hold rows in ANY order (here deliberately
    REVERSED path order): an "identity" reference must return exactly that,
    unchanged -- never re-sorted."""
    table_rows = [translation_row(["p", i], ["p", i], i) for i in range(50, 0, -1)]
    digest = write_translation_table(tmp_path / "tier0", table_rows)
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_IDENTITY}
    p = pair("199_identity", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    corpus = t0.load(root)
    resolved = corpus.pairs["199_identity"]["payload"]["record"]["extra_field"]
    assert resolved == table_rows
    assert content_hash(resolved) == content_hash(table_rows)


def test_a_positions_reference_reconstructs_realistically_unsorted_order(tmp_path):
    """2026-09-20: the real capture's mappings order turned out to be
    NEITHER sorted-by-path NOR consistent across members -- the probe on
    real data refused an earlier, sort-based design. This fixture is
    deliberately adversarial to that assumption: the table holds rows in
    REVERSED path order, and the member's OWN order interleaves an ascending
    even-path pass, one of its own rows not in the table at all, a
    descending odd-path pass, and a second own-only row -- proving
    reconstruction depends on nothing but the explicit ``sequence``, never
    on re-deriving order from content.
    """
    table_rows = [translation_row(["p", i], ["p", i], i) for i in range(199, -1, -1)]
    digest = write_translation_table(tmp_path / "tier0", table_rows)
    table_index = {content_hash(row): pos for pos, row in enumerate(table_rows)}

    own_row_a = translation_row(["p", 500], ["p", 500], "own-a")
    own_row_b = translation_row(["p", 501], ["p", 501], "own-b")
    member_rows = (
        [translation_row(["p", i], ["p", i], i) for i in range(0, 200, 2)]
        + [own_row_a]
        + [translation_row(["p", i], ["p", i], i) for i in range(199, 0, -2)]
        + [own_row_b]
    )
    tokens = []
    literals = []
    for row in member_rows:
        key = content_hash(row)
        position = table_index.get(key)
        if position is not None:
            tokens.append(str(position))
        else:
            tokens.append(f"L{len(literals)}")
            literals.append(row)
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_POSITIONS,
           "sequence": ",".join(tokens), "literals": literals}
    p = pair("200_rows", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    corpus = t0.load(root)
    resolved = corpus.pairs["200_rows"]["payload"]["record"]["extra_field"]
    assert resolved == member_rows  # exact order, including the two literals
    assert content_hash(resolved) == content_hash(member_rows)  # hash-oracle equivalence


def test_an_unknown_rows_reference_shape_refuses_loudly(tmp_path):
    digest = write_translation_table(
        tmp_path / "tier0", [translation_row(["p", 0], ["p", 0], 0)])
    bad_ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_IDENTITY, "surprise": 1}
    p = pair("201_bad_shape", request("STR-THRU"), priced("STR-THRU", extra_field=bad_ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_missing_translation_table_refuses_loudly(tmp_path):
    ref = {t0.ROWS_REF_KEY: "sha256:" + "b" * 64, "order": t0.ROWS_ORDER_IDENTITY}
    p = pair("202_missing_table", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_tampered_translation_table_refuses_loudly(tmp_path):
    table_rows = [translation_row(["p", i], ["p", i], i) for i in range(5)]
    digest = write_translation_table(tmp_path / "tier0", table_rows)
    table_path = (tmp_path / "tier0" / "shared" / "translations"
                  / f"{digest.split(':', 1)[-1]}.json")
    doc = json.loads(table_path.read_text())
    doc["rows"][0]["value_hash"] = "sha256:" + "c" * 64
    table_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_IDENTITY}
    p = pair("203_tampered_table", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_an_unsupported_row_order_refuses_loudly(tmp_path):
    digest = write_translation_table(
        tmp_path / "tier0", [translation_row(["p", 0], ["p", 0], 0)])
    ref = {t0.ROWS_REF_KEY: digest, "order": "shared_path"}  # the retired v1 scheme
    p = pair("204_bad_order", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_malformed_sequence_token_refuses_loudly(tmp_path):
    digest = write_translation_table(
        tmp_path / "tier0", [translation_row(["p", 0], ["p", 0], 0)])
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_POSITIONS,
           "sequence": "0,not-a-token", "literals": []}
    p = pair("205_bad_token", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_sequence_table_index_out_of_range_refuses_loudly(tmp_path):
    digest = write_translation_table(
        tmp_path / "tier0", [translation_row(["p", 0], ["p", 0], 0)])
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_POSITIONS,
           "sequence": "1", "literals": []}
    p = pair("206_index_oob", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_sequence_literal_index_out_of_range_refuses_loudly(tmp_path):
    digest = write_translation_table(
        tmp_path / "tier0", [translation_row(["p", 0], ["p", 0], 0)])
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_POSITIONS,
           "sequence": "L0", "literals": []}
    p = pair("207_literal_oob", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_an_unused_literal_refuses_loudly(tmp_path):
    digest = write_translation_table(
        tmp_path / "tier0", [translation_row(["p", 0], ["p", 0], 0)])
    unused = translation_row(["p", 1], ["p", 1], 1)
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_POSITIONS,
           "sequence": "0", "literals": [unused]}
    p = pair("208_unused_literal", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_duplicate_table_position_in_a_sequence_refuses_loudly(tmp_path):
    digest = write_translation_table(
        tmp_path / "tier0", [translation_row(["p", 0], ["p", 0], 0)])
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_POSITIONS,
           "sequence": "0,0", "literals": []}
    p = pair("209_dup_position", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_duplicate_literal_use_in_a_sequence_refuses_loudly(tmp_path):
    """Distinct from a duplicate TABLE position: here the same literal index
    is referenced twice by the sequence, which is just as ambiguous as
    replaying one table row twice.
    """
    digest = write_translation_table(
        tmp_path / "tier0", [translation_row(["p", 0], ["p", 0], 0)])
    literal = translation_row(["p", 1], ["p", 1], 1)
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_POSITIONS,
           "sequence": "L0,L0", "literals": [literal]}
    p = pair("210_dup_literal", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_duplicate_row_within_a_shared_table_refuses_loudly(tmp_path):
    """A stored table with a repeated row is ambiguous for both overlap
    matching and position reconstruction -- the writer never produces this,
    but a hand-edited or corrupted table must still be refused, not silently
    accepted.
    """
    row = translation_row(["p", 0], ["p", 0], 0)
    table_rows = [row, dict(row)]
    tables_dir = tmp_path / "tier0" / "shared" / "translations"
    tables_dir.mkdir(parents=True, exist_ok=True)
    # Write directly (bypassing `write_translation_table`'s digest-over-rows
    # convention isn't needed here) so the digest matches this exact,
    # duplicate-carrying row list.
    digest = content_hash(table_rows)
    body = {"schema_version": t0.SHARED_TRANSLATION_TABLE_SCHEMA_VERSION,
            "digest": digest, "rows": table_rows}
    (tables_dir / f"{digest.split(':', 1)[-1]}.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n")
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_IDENTITY}
    p = pair("211_dup_table_row", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_an_unparseable_translation_table_file_refuses_loudly(tmp_path):
    digest = "sha256:" + "d" * 64
    tables_dir = tmp_path / "tier0" / "shared" / "translations"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / f"{digest.split(':', 1)[-1]}.json").write_text("{not json")
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_IDENTITY}
    p = pair("212_bad_json", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_translation_table_with_the_wrong_schema_version_refuses_loudly(tmp_path):
    table_rows = [translation_row(["p", 0], ["p", 0], 0)]
    digest = content_hash(table_rows)
    tables_dir = tmp_path / "tier0" / "shared" / "translations"
    tables_dir.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": "tier0_shared_translation_table.v9.9",
            "digest": digest, "rows": table_rows}
    (tables_dir / f"{digest.split(':', 1)[-1]}.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n")
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_IDENTITY}
    p = pair("213_bad_schema", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_a_translation_table_whose_rows_is_not_a_list_refuses_loudly(tmp_path):
    digest = content_hash({"not": "a list"})
    tables_dir = tmp_path / "tier0" / "shared" / "translations"
    tables_dir.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": t0.SHARED_TRANSLATION_TABLE_SCHEMA_VERSION,
            "digest": digest, "rows": {"not": "a list"}}
    (tables_dir / f"{digest.split(':', 1)[-1]}.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n")
    ref = {t0.ROWS_REF_KEY: digest, "order": t0.ROWS_ORDER_IDENTITY}
    p = pair("214_rows_not_list", request("STR-THRU"), priced("STR-THRU", extra_field=ref))
    root = build(tmp_path / "tier0", [p])
    with pytest.raises(t0.CorpusFormatError):
        t0.load(root)


def test_an_old_format_plain_mappings_list_loads_unchanged(tmp_path):
    """A pair captured before this fix carries `mappings` as a plain list,
    with no `shared/translations/` directory anywhere -- CURRENT is still an
    old-format corpus. It must load byte-for-byte unchanged, order included
    (deliberately reversed here, not ascending-by-path).
    """
    mappings = [translation_row(["p", i], ["p", i], i) for i in range(4, -1, -1)]
    p = pair("206_plain_mappings", request("STR-THRU"),
             priced("STR-THRU", extra_field={"mappings": mappings}))
    root = build(tmp_path / "tier0", [p])
    assert not (root / "shared").exists()
    corpus = t0.load(root)
    loaded = corpus.pairs["206_plain_mappings"]["payload"]["record"]["extra_field"]["mappings"]
    assert loaded == mappings
