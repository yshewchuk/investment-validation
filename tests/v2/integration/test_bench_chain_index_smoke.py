"""Smoke test for tools/bench_chain_index.py: the ``--tiny`` run is fast and passes.

Synthetic only: the benchmark generates its own chains in memory and never
reads data/ or any store.
"""
from __future__ import annotations

# packages: engine.v2.ops

import dataclasses
import re
import time
from functools import partial

import pytest

import tools.bench_chain_index as bench
from engine.fills import FillModel
from engine.structures import STRUCTURES, price_structure


def test_bench_tiny_runs_fast_and_identical(capsys):
    start = time.perf_counter()
    rc = bench.main(["--tiny"])
    elapsed = time.perf_counter() - start
    captured = capsys.readouterr()
    assert rc == 0
    assert elapsed < 2.0, elapsed
    assert "speedup" in captured.out
    match = re.search(r"^summary: ok_rows=(\d+)", captured.out, re.MULTILINE)
    assert match is not None, captured.out
    assert int(match.group(1)) >= 1, captured.out


def test_bench_case_row_and_identity_control():
    # (i) one real case runs, is identical to legacy, and reports a speedup.
    rows = bench.make_chain(40, 1, "CP", bench.case_seed(40, 1, "CP"))
    structure = STRUCTURES["STR-THRU"]()
    fill = FillModel(0.0)
    reference = bench.outcome(partial(price_structure, structure, bench.make_snapshot(rows), fill))
    assert reference[0] == "ok"
    out = bench.run_case(40, 1, "CP", ["STR-THRU"], (0.0,), 1)
    assert len(out) == 1
    assert out[0]["ok"] is True
    assert out[0]["speedup"] == pytest.approx(out[0]["legacy_ms"] / out[0]["warm_ms"])

    # (ii) control: a numerically altered result must be caught by assert_identical.
    a = reference[1]
    b = dataclasses.replace(a, spot=a.spot + 1.0)
    with pytest.raises(AssertionError):
        bench.assert_identical("x", ("ok", a), ("ok", b))


def test_run_case_refuses_zero_reps():
    with pytest.raises(ValueError):
        bench.run_case(40, 1, "CP", ["STR-THRU"], (0.0,), 0)


def test_assert_identical_error_branches():
    # (a) ok vs err status mismatch
    with pytest.raises(AssertionError):
        bench.assert_identical("x", ("ok", 1), ("err", ValueError, "m"))
    # (b) same type, different message
    with pytest.raises(AssertionError):
        bench.assert_identical("x", ("err", ValueError, "m"), ("err", ValueError, "other"))
    # (c) same message, different type
    with pytest.raises(AssertionError):
        bench.assert_identical("x", ("err", ValueError, "m"), ("err", KeyError, "m"))
    # (d) identical errors do not raise
    bench.assert_identical("x", ("err", ValueError, "m"), ("err", ValueError, "m"))
