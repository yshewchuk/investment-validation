"""P6-6 resource-measurement recorder: peak parsing, kill classification,
flag pass-through and evidence writing.

Everything runs against ``tests/fixtures/v2_resource_measurement_bounded_run.py``,
a scripted stand-in for ``tools/bounded_run.py`` -- no real heavy job is ever
launched, and the recorder's argv (not just its record) is asserted from the
argv file the stub writes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import v2_resource_measurement as rm

STUB = Path(__file__).resolve().parent / "fixtures" / "v2_resource_measurement_bounded_run.py"
LABEL = "test-workload"
WORKLOAD = ["python3", "-c", "print('workload')"]


def _watchdog(rss_gb: float, elapsed_min: float = 1.0) -> str:
    return (f"[watchdog]   {elapsed_min:.1f}m rss {rss_gb:5.2f}G pss ({rss_gb:.2f}G vm, 2 procs) "
            f"= 10% of cap; swap  0.00G; box free 5.00G")


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """Monkeypatch the bounded-run path and return the script setter."""
    argv_path = tmp_path / "child-argv.json"
    monkeypatch.setenv("V2_RESOURCE_STUB_ARGV_PATH", str(argv_path))
    monkeypatch.setattr(rm, "BOUNDED_RUN", STUB)

    def script(lines, exit_code):
        monkeypatch.setenv("V2_RESOURCE_STUB_LINES", json.dumps(lines))
        monkeypatch.setenv("V2_RESOURCE_STUB_EXIT_CODE", str(exit_code))
        return argv_path

    return script


def _measure(tmp_path, **kwargs):
    defaults = dict(workload_label=LABEL, max_rss_gb=2.0, cache_state="warm",
                    command=WORKLOAD, evidence_dir=tmp_path / "evidence")
    defaults.update(kwargs)
    return rm.measure(**defaults)


def _child_argv(argv_path: Path) -> list[str]:
    return json.loads(argv_path.read_text())[1:]  # drop the stub's own argv[0]


# --------------------------------------------------------------------- peaks


def test_peak_is_the_maximum_watchdog_reading(tmp_path, stub):
    stub([_watchdog(0.5), _watchdog(1.75), _watchdog(0.9)], 0)

    record = _measure(tmp_path)

    assert record["peak_rss_gb"] == 1.75
    assert record["exit_code"] == 0
    assert record["killed"] is False
    assert record["kill_reason"] is None


def test_no_watchdog_line_records_null_not_zero(tmp_path, stub):
    stub(["[bounded] command: workload", "plain output, no heartbeat"], 0)

    assert _measure(tmp_path)["peak_rss_gb"] is None


def test_watchdog_rss_values_ignores_the_startup_reading():
    assert rm.watchdog_rss_values(_watchdog(0.01, elapsed_min=0.0)) == []


def test_watchdog_rss_values_keeps_only_full_interval_readings():
    output = "\n".join([_watchdog(0.01, elapsed_min=0.0),
                        _watchdog(1.23, elapsed_min=1.0)])

    assert rm.watchdog_rss_values(output) == [1.23]


def test_measure_fast_clean_exit_has_no_false_peak(tmp_path, monkeypatch):
    monkeypatch.setattr(rm, "_stream",
                        lambda command: ([_watchdog(0.01, elapsed_min=0.0) + "\n"], 0))

    record = _measure(tmp_path)

    assert record["peak_rss_gb"] is None
    assert record["exit_code"] == 0
    assert record["killed"] is False


def test_measure_uses_a_real_reading_not_the_startup_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(rm, "_stream", lambda command: (
        [_watchdog(0.01, elapsed_min=0.0) + "\n", _watchdog(2.5, elapsed_min=2.0) + "\n"], 0))

    assert _measure(tmp_path)["peak_rss_gb"] == 2.5


# ---------------------------------------------------------- kill classification


@pytest.mark.parametrize("breach", rm.KILL_REASONS)
def test_each_breach_string_classifies_the_kill(tmp_path, stub, breach):
    stub([_watchdog(7.9), f"[watchdog] ... - {breach}, SIGKILL"], 137)

    record = _measure(tmp_path)

    assert record["killed"] is True
    assert record["kill_reason"] == breach


def test_exit_137_without_a_breach_string_has_no_reason(tmp_path, stub):
    stub([_watchdog(0.5)], 137)

    record = _measure(tmp_path)

    assert record["exit_code"] == 137
    assert record["killed"] is True
    assert record["kill_reason"] is None


def test_breach_text_without_exit_137_is_never_a_kill(tmp_path, stub):
    stub([_watchdog(0.5), "SWAP BREACH printed but the child exited 1"], 1)

    record = _measure(tmp_path)

    assert record["killed"] is False
    assert record["kill_reason"] is None


# ----------------------------------------------------------------- pass-through


def test_optional_flags_reach_the_child_only_when_given(tmp_path, stub):
    argv_path = stub([_watchdog(0.5)], 0)
    _measure(tmp_path, max_swap_gb=1.5, cores=4, cpu_set="2-3")
    assert _child_argv(argv_path) == [
        "--max-rss-gb", "2.0", "--max-swap-gb", "1.5", "--cores", "4",
        "--cpu-set", "2-3", "--", *WORKLOAD,
    ]

    argv_path = stub([_watchdog(0.5)], 0)
    _measure(tmp_path)
    assert _child_argv(argv_path) == ["--max-rss-gb", "2.0", "--", *WORKLOAD]


# -------------------------------------------------------------------- record


def test_writes_one_timestamped_record_equal_to_the_returned_one(tmp_path, stub):
    stub([_watchdog(0.5)], 0)

    record = _measure(tmp_path, workload_label="nightly-shadow")

    files = sorted((tmp_path / "evidence").glob("nightly-shadow-*.json"))
    assert len(files) == 1
    assert files[0].name.endswith("Z.json")
    assert json.loads(files[0].read_text()) == record


def test_record_has_the_declared_schema_and_fields(tmp_path, stub):
    stub([_watchdog(0.5)], 0)

    record = _measure(tmp_path, max_swap_gb=6.0, cores=8, cpu_set="0-7",
                      capabilities_covered=["nightly-score", "nightly-publish"])

    assert record["schema_version"] == "v2_resource_measurement_record.v1.0"
    assert record["command"] == WORKLOAD
    assert record["cache_state"] == "warm"
    assert record["max_rss_gb_cap"] == 2.0
    assert record["max_swap_gb_cap"] == 6.0
    assert record["cores"] == 8
    assert record["cpu_set"] == "0-7"
    assert record["capabilities_covered"] == ["nightly-score", "nightly-publish"]
    assert record["wall_seconds"] >= 0
    assert record["started_at"] <= record["ended_at"]
    assert set(record["contention"]) == {"other_heavy_jobs"}


def test_collect_contention_records_agents_md_bracket_patterns(monkeypatch):
    seen = []

    def fake_pgrep(pattern):
        seen.append(pattern)
        return [11]

    contention = rm.collect_contention(fake_pgrep)

    assert seen == [pattern for _, pattern in rm.HEAVY_JOB_PATTERNS]
    assert seen == ["[b]ounded_run.py", "[s]erve_monitor", "[c]orpus_parity.py run"]
    assert contention == {"other_heavy_jobs": {
        "bounded_run.py": [11], "serve_monitor": [11], "corpus_parity.py run": [11]}}


def test_free_parser_reads_the_available_column():
    text = ("              total        used        free      shared  buff/cache   available\n"
            "Mem:          10240        5120        1024         200        4096        6144\n"
            "Swap:         20480           0       20480\n")
    assert rm.parse_free_available_gb(text) == 6.0
    assert rm.parse_free_available_gb("") is None


# ----------------------------------------------------------------------- CLI


def test_cli_exits_with_the_childs_own_code_and_prints_the_record(tmp_path, stub, capsys):
    stub([_watchdog(0.5)], 137)

    code = rm.main(["--workload-label", LABEL, "--max-rss-gb", "2",
                    "--cache-state", "cold", "--evidence-dir", str(tmp_path / "evidence"),
                    "--", *WORKLOAD])

    assert code == 137
    out = capsys.readouterr().out
    record = json.loads(out[out.index("{"):])
    assert record["exit_code"] == 137
    assert record["killed"] is True


def test_cli_parses_comma_separated_capabilities(tmp_path, stub, capsys):
    stub([], 0)

    code = rm.main(["--workload-label", LABEL, "--max-rss-gb", "2",
                    "--cache-state", "unknown",
                    "--capabilities-covered", "nightly-score, nightly-publish",
                    "--evidence-dir", str(tmp_path / "evidence"), "--", *WORKLOAD])

    assert code == 0
    out = capsys.readouterr().out
    assert json.loads(out[out.index("{"):])["capabilities_covered"] == [
        "nightly-score", "nightly-publish"]


def test_cli_refuses_an_empty_command(tmp_path, stub, capsys):
    assert rm.main(["--workload-label", LABEL, "--max-rss-gb", "2",
                    "--cache-state", "cold"]) == 2
    assert "no workload command" in capsys.readouterr().err


def test_cli_refuses_a_label_that_cannot_name_a_file(tmp_path, stub):
    assert rm.main(["--workload-label", "../escape", "--max-rss-gb", "2",
                    "--cache-state", "cold", "--", *WORKLOAD]) == 2
