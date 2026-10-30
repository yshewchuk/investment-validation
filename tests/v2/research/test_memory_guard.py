"""RSS watchdog enforces resident memory, never address space."""
import mmap
import resource
import signal
import time

import pytest

from engine.v2.ops.errors import OpsError
from experiments import size_search
from experiments.memory_guard import rss_bytes, rss_watchdog, start_rss_watchdog


def test_rss_bytes_matches_getrusage_order():
    current = rss_bytes()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    assert 0.2 * peak <= current <= 5.0 * peak


def test_address_space_above_budget_is_not_refused():
    with rss_watchdog(rss_bytes() + 400 * 2 ** 20):
        mapping = mmap.mmap(-1, 1 << 30)
        time.sleep(0.3)
        mapping.close()


def test_rss_over_budget_is_refused():
    with pytest.raises(MemoryError):
        with rss_watchdog(rss_bytes() + 64 * 2 ** 20):
            block = bytearray(256 * 2 ** 20)
            block[::4096] = b"\1" * len(range(0, len(block), 4096))
            for _ in range(300):
                time.sleep(0.01)


def test_swallowed_memory_error_is_raised_again():
    try:
        count, hold = 0, None
        with rss_watchdog(rss_bytes() + 32 * 2 ** 20, interval=0.01):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    if hold is None:
                        hold = bytearray(128 * 2 ** 20)
                        hold[::4096] = b"\1" * len(range(0, len(hold), 4096))
                    time.sleep(0.01)
                except MemoryError:
                    count += 1
                    if count >= 2:
                        hold = None
                        break
        assert count >= 2
    except MemoryError:
        pass


def test_nonpositive_interval_is_refused():
    with pytest.raises(OpsError):
        with rss_watchdog(2 ** 40, interval=0):
            pass


def test_stopped_watchdog_ignores_late_signal():
    stop = start_rss_watchdog(rss_bytes() + 4 * 2 ** 30)
    stop()
    signal.raise_signal(signal.SIGUSR1)


def parse_status():
    fields = {}
    with open("/proc/self/status", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("VmSize:"):
                fields["VmSize"] = int(line.split()[1]) * 1024
            if line.startswith("VmRSS:"):
                fields["VmRSS"] = int(line.split()[1]) * 1024
    return fields


def test_measure_vmsize_vs_rss_on_arrow_workload(tmp_path):
    numpy = pytest.importorskip("numpy")
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")

    path = tmp_path / "random.parquet"
    values = numpy.random.default_rng(0).standard_normal(50 * 1024 * 1024 // 8)
    parquet.write_table(pyarrow.table({"value": values}), path)
    table = parquet.read_table(path, memory_map=True)

    _ = table.num_rows
    fields = parse_status()
    assert fields["VmSize"] - fields["VmRSS"] >= 100 * 2 ** 20


def limits(parent, child):
    return {"request": {"limits": {"parent": {"memory_bytes": parent, "wall_seconds": 60},
        "child": {"memory_bytes": child}, "job": {"wall_seconds": 60}}}}


def test_size_search_contain_sets_no_address_space_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(size_search.resource, "setrlimit", lambda *args: calls.append(args))
    monkeypatch.setattr("experiments.magnitude_search_request_versions.compile_request",
        lambda request: limits(2 ** 33, 2 ** 32))
    try:
        assert size_search._contain({}) == 2 ** 33
        assert calls == []
    finally:
        signal.alarm(0)
    monkeypatch.setattr("experiments.magnitude_search_request_versions.compile_request",
        lambda request: limits(2 ** 33, 2 ** 34))
    with pytest.raises(OpsError) as caught:
        size_search._contain({})
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
