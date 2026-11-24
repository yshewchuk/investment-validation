"""Smoke test for tools/bench_chain_index.py: the ``--tiny`` run is fast and passes.

Synthetic only: the benchmark generates its own chains in memory and never
reads data/ or any store.
"""
from __future__ import annotations

# packages: engine.v2.ops

import time

import tools.bench_chain_index as bench


def test_bench_tiny_runs_fast_and_identical(capsys):
    start = time.perf_counter()
    rc = bench.main(["--tiny"])
    elapsed = time.perf_counter() - start
    captured = capsys.readouterr()
    assert rc == 0
    assert elapsed < 2.0, elapsed
    assert "speedup" in captured.out
