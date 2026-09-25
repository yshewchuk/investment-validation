"""tools/bounded_run.py coordination: heavy reservations, slots, admission.

Each test drives the real runner in a subprocess with the memory readers
injected (``_available_mb`` / ``_vmrss_mb`` patched in the wrapper) and poll
intervals shortened 100x, using tiny ``time.sleep`` children, so none of it
touches real memory pressure.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(state: Path, extra: dict[str, str] | None) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("BOUNDED_RUN_")}
    env["BOUNDED_RUN_STATE_DIR"] = str(state)
    env.update(extra or {})
    return env


def _wrapper(argv: list[str], *, available_mb: float, rss_code: str | None,
             pre: list[str] | None) -> str:
    lines = [
        "import sys",
        "import time",
        f"sys.path.insert(0, {str(ROOT)!r})",
        "import tools.bounded_run as br",
        "br.SLOT_POLL_S = 0.05",
        "br.ADMISSION_POLL_S = 0.05",
        "br.HEAVY_POLL_S = 0.05",
        f"br._available_mb = lambda: {available_mb!r}",
        rss_code or "br._vmrss_mb = lambda pid: 0.0",
    ]
    lines.extend(pre or [])
    lines.append(f"sys.argv = {argv!r}")
    lines.append("sys.exit(br.main())")
    return "\n".join(lines)


def _popen(state: Path, command: list[str], *, available_mb: float = 10240.0,
           rss_code: str | None = None, max_wait: str = "30",
           max_rss_gb: str = "0.5", heavy: bool = False,
           pre: list[str] | None = None,
           extra_env: dict[str, str] | None = None) -> subprocess.Popen:
    argv = ["bounded_run.py", "--cores", "1",
            "--max-rss-gb", max_rss_gb, "--min-free-gb", "0.05",
            "--poll-s", "0.05", "--max-wait-s", max_wait]
    if heavy:
        argv.append("--heavy")
    argv.extend(["--", *command])
    return subprocess.Popen(
        [sys.executable, "-c", _wrapper(argv, available_mb=available_mb,
                                        rss_code=rss_code, pre=pre)],
        env=_env(state, extra_env),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def _run(state: Path, command: list[str], *, timeout: float = 30,
         **kwargs) -> tuple[int, str]:
    proc = _popen(state, command, **kwargs)
    out, _ = proc.communicate(timeout=timeout)
    return proc.returncode, out


def _wait_marker(fname: str, label: str, marker: Path) -> list[str]:
    """Wrapper code that tags every ``_resource_wait`` with a marker line."""
    path = str(marker)
    return [
        f"def _mark_{fname}():\n"
        f"    open({path!r}, 'a').write({label!r} + ' ' "
        "+ repr(time.time()) + '\\n')\n"
        "_original_wait = br._resource_wait\n"
        f"def _wait_{fname}(what, poll_s):\n"
        f"    _mark_{fname}()\n"
        "    _original_wait(what, poll_s)\n"
        f"br._resource_wait = _wait_{fname}",
    ]


def _child(label: str, sleep_s: float, marker: Path) -> list[str]:
    path = str(marker)
    code = (
        "import time\n"
        f"open({path!r}, 'a').write({label!r} + '-start ' "
        "+ repr(time.time()) + '\\n')\n"
        f"time.sleep({sleep_s!r})\n"
        f"open({path!r}, 'a').write({label!r} + '-end ' "
        "+ repr(time.time()) + '\\n')\n"
        f"print('CHILD-{label}')\n"
    )
    return [sys.executable, "-c", code]


def _hold_child(label: str, marker: Path, release: Path) -> list[str]:
    """A child that stays alive until the test creates ``release`` (20 s cap)."""
    path, rel = str(marker), str(release)
    code = (
        "import os, time\n"
        f"open({path!r}, 'a').write({label!r} + '-start ' "
        "+ repr(time.time()) + '\\n')\n"
        "deadline = time.time() + 20\n"
        f"while not os.path.exists({rel!r}) and time.time() < deadline:\n"
        "    time.sleep(0.02)\n"
        f"open({path!r}, 'a').write({label!r} + '-end ' "
        "+ repr(time.time()) + '\\n')\n"
        f"print('CHILD-{label}')\n"
    )
    return [sys.executable, "-c", code]


def _markers(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            name, _, raw = line.rpartition(" ")
            if raw:
                out[name] = float(raw)
    return out


def _wait_for(predicate, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition never became true")


def _hold_slot(state: Path, index: int = 0) -> int:
    state.mkdir(parents=True, exist_ok=True)
    fd = os.open(state / f"slot-{index}.lock", os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def _hold_heavy(state: Path, reserve_gb: float = 0.1):
    state.mkdir(parents=True, exist_ok=True)
    path = state / f"heavy-{os.getpid()}.json"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path.write_text(json.dumps({
        "pid": os.getpid(), "reserve_gb": reserve_gb,
        "started_at": time.time(), "argv0": "test",
    }))
    return fd, path


def test_slot_limit_one_serializes_a_second_job(tmp_path):
    state = tmp_path / "state"
    marker = tmp_path / "markers"
    release = tmp_path / "release"
    slots = {"BOUNDED_RUN_SLOTS": "1"}
    first = _popen(state, _hold_child("first", marker, release),
                   extra_env=slots)
    _wait_for(lambda: "first-start" in _markers(marker))
    second = _popen(state, _child("second", 0.0, marker), extra_env=slots,
                    pre=_wait_marker("second", "second-wait", marker))
    _wait_for(lambda: "second-wait" in _markers(marker))
    assert "second-start" not in _markers(marker)
    assert first.poll() is None
    release.write_text("go")
    out_first, _ = first.communicate(timeout=30)
    out_second, _ = second.communicate(timeout=30)
    assert first.returncode == 0 and second.returncode == 0
    assert "RESOURCE WAIT" in out_second and "slot" in out_second
    assert "RESOURCE WAIT" not in out_first
    times = _markers(marker)
    assert times["second-start"] >= times["first-end"]


def test_under_live_heavy_the_slot_count_drops(tmp_path):
    state = tmp_path / "state"
    marker = tmp_path / "markers"
    release = tmp_path / "release"
    env = {"BOUNDED_RUN_SLOTS": "4", "BOUNDED_RUN_SLOTS_UNDER_HEAVY": "1"}
    fd, _ = _hold_heavy(state)
    try:
        first = _popen(state, _hold_child("a", marker, release), extra_env=env)
        _wait_for(lambda: "a-start" in _markers(marker))
        second = _popen(state, _child("b", 0.0, marker), extra_env=env,
                        pre=_wait_marker("b", "b-wait", marker))
        _wait_for(lambda: "b-wait" in _markers(marker))
        assert "b-start" not in _markers(marker)
        release.write_text("go")
        out_a, _ = first.communicate(timeout=30)
        out_b, _ = second.communicate(timeout=30)
    finally:
        os.close(fd)
        release.write_text("go")
    assert first.returncode == 0 and second.returncode == 0
    assert "RESOURCE WAIT" in out_b and "slot" in out_b
    assert "RESOURCE WAIT" not in out_a
    times = _markers(marker)
    assert times["b-start"] >= times["a-end"]


def test_stale_heavy_file_is_ignored_and_removed(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    stale = state / "heavy-999999.json"
    stale.write_text(json.dumps({
        "pid": 999999, "reserve_gb": 8.0,
        "started_at": time.time(), "argv0": "crashed",
    }))
    code, out = _run(state, [sys.executable, "-c", "print('CHILD-RAN')"])
    assert code == 0, out
    assert "CHILD-RAN" in out
    assert "RESOURCE WAIT" not in out
    assert not stale.exists()


def test_second_heavy_waits_for_the_first(tmp_path):
    state = tmp_path / "state"
    marker = tmp_path / "markers"
    release = tmp_path / "release"
    first = _popen(state, _hold_child("h1", marker, release), heavy=True)
    _wait_for(lambda: bool(list(state.glob("heavy-*.json"))))
    second = _popen(state, _child("h2", 0.0, marker), heavy=True,
                    pre=_wait_marker("h2", "h2-wait", marker))
    _wait_for(lambda: "h2-wait" in _markers(marker))
    assert "h2-start" not in _markers(marker)
    assert first.poll() is None
    release.write_text("go")
    out_first, _ = first.communicate(timeout=30)
    out_second, _ = second.communicate(timeout=30)
    assert first.returncode == 0 and second.returncode == 0
    assert "RESOURCE WAIT" in out_second and "heavy" in out_second
    times = _markers(marker)
    assert times["h2-start"] >= times["h1-end"]
    assert not list(state.glob("heavy-*.json"))


def test_heavy_reservation_file_reports_its_contract(tmp_path):
    state = tmp_path / "state"
    marker = tmp_path / "markers"
    heavy = _popen(state, _child("heavy", 0.5, marker), heavy=True,
                   max_rss_gb="3")
    _wait_for(lambda: bool(list(state.glob("heavy-*.json"))))
    path = next(state.glob("heavy-*.json"))
    data = json.loads(path.read_text())
    assert data["pid"] == heavy.pid
    assert data["reserve_gb"] == 3
    assert data["argv0"] == sys.executable
    assert isinstance(data["started_at"], float)
    out, _ = heavy.communicate(timeout=30)
    assert heavy.returncode == 0, out
    assert not path.exists()


def test_admission_waits_until_heavy_rss_grows(tmp_path):
    state = tmp_path / "state"
    marker = tmp_path / "markers"
    release = tmp_path / "release"
    rss_file = tmp_path / "heavy_rss_mb"
    rss_file.write_text("1024")
    heavy = _popen(state, _hold_child("heavy", marker, release), heavy=True,
                   available_mb=8192.0, max_rss_gb="4")
    _wait_for(lambda: bool(list(state.glob("heavy-*.json"))))
    heavy_pid = json.loads(next(state.glob("heavy-*.json")).read_text())["pid"]
    rss_code = (
        "import pathlib\n"
        f"br._vmrss_mb = lambda pid: "
        f"float(pathlib.Path({str(rss_file)!r}).read_text()) "
        f"if pid == {heavy_pid} else 0.0"
    )
    waiter = _popen(state, _child("waiter", 0.0, marker),
                    available_mb=1024.0, rss_code=rss_code,
                    pre=_wait_marker("waiter", "waiter-wait", marker))
    _wait_for(lambda: "waiter-wait" in _markers(marker))
    assert "waiter-start" not in _markers(marker)
    with open(marker, "a") as handle:
        handle.write(f"update {time.time()!r}\n")
    rss_file.write_text("4096")
    _wait_for(lambda: "waiter-start" in _markers(marker))
    release.write_text("go")
    out_heavy, _ = heavy.communicate(timeout=30)
    out_waiter, _ = waiter.communicate(timeout=30)
    assert heavy.returncode == 0 and waiter.returncode == 0
    assert "RESOURCE WAIT" in out_waiter and "heavy" in out_waiter
    times = _markers(marker)
    assert times["waiter-start"] >= times["update"]


def test_max_wait_expiry_exits_75(tmp_path):
    # The child source text must not literally contain "CHILD-RAN": bounded_run
    # echoes the full command line ("[bounded] command: ...") before waiting
    # for admission, so a literal marker would show up even when the child
    # never actually runs. Split the literal so the echoed argv can't match it.
    state = tmp_path / "state"
    fd = _hold_slot(state)
    try:
        code, out = _run(state,
                         [sys.executable, "-c", "print('CHILD' + '-RAN')"],
                         max_wait="0.5",
                         extra_env={"BOUNDED_RUN_SLOTS": "1"})
    finally:
        os.close(fd)
    assert code == 75, out
    assert "RESOURCE WAIT timed out" in out
    assert "CHILD-RAN" not in out


def test_nested_bypasses_slots(tmp_path):
    state = tmp_path / "state"
    fd = _hold_slot(state)
    try:
        code, out = _run(state, [sys.executable, "-c", "print('CHILD-RAN')"],
                         extra_env={"BOUNDED_RUN_SLOTS": "1",
                                    "BOUNDED_RUN_NESTED": "1"})
    finally:
        os.close(fd)
    assert code == 0, out
    assert "CHILD-RAN" in out
    assert "RESOURCE WAIT" not in out


def test_child_receives_nested_env(tmp_path):
    child = ("import os; "
             "print('NESTED=' + str(os.environ.get('BOUNDED_RUN_NESTED')))")
    code, out = _run(tmp_path / "state",
                     [sys.executable, "-c", child])
    assert code == 0, out
    assert "NESTED=1" in out


def test_slot_count_env_defaults(tmp_path, monkeypatch):
    import tools.bounded_run as br
    monkeypatch.setenv("BOUNDED_RUN_SLOTS", "4")
    monkeypatch.setenv("BOUNDED_RUN_SLOTS_UNDER_HEAVY", "1")
    assert br._slot_count([]) == 4
    assert br._slot_count([{"pid": 1}]) == 1