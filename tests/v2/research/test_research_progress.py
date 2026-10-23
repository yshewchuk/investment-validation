"""Progress logging (#953): bounded cadence, stderr only; pinned build is output-identical and cheaper."""
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest

from engine.v2.data.errors import DataError, fail
from engine.v2.data.objects import verify_object_path
from engine.v2.data.repository import Repository
from engine.v2.foundation import to_document
from engine.v2.research import progress
from engine.v2.research._chains import load_chain_index, read_chains_for_years
from engine.v2.research.experiment_population import load_population
from experiments import magnitude_history_store as entry
from tests.data_scan_support import catalog_and_store, commit_tables, contract_for, contract_ref_for, publish_and_inspect
from tests.v2.research.magnitude_trial_support import (
    HISTORY_REQUEST,
    calendar_sessions,
    commit_history_inputs,
    history_event,
    outcome_move,
)
from tests.v2.research.scan_progress_support import assert_scan_boundaries, observe_scan_outcomes


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_progress_cadence_is_bounded_by_interval_not_items():
    clock, lines = Clock(), []
    reporter = progress.Progress("x", 1000, min_interval=30, clock=clock, emit=lines.append)
    for _ in range(1000):
        clock.now += 1.0
        reporter.tick()
    assert len(lines) == 33  # one per 30 s of the 1000 s pass, never one per item
    for number, line in enumerate(lines, start=1):
        done = number * 30
        assert line.startswith(f"  [x] {done:,}/1,000, {done}s elapsed, 1.00/s, ETA {1000 - done}s, peak RSS ")


def test_progress_is_silent_for_short_passes_and_honours_min_items():
    clock, lines = Clock(), []
    short = progress.Progress("x", 5, min_interval=30, clock=clock, emit=lines.append)
    for _ in range(5):
        clock.now += 1.0
        short.tick()
    assert lines == []
    gated = progress.Progress("y", 100, min_interval=1, min_items=10, clock=clock, emit=lines.append)
    for _ in range(25):
        clock.now += 5.0
        gated.tick()
    assert len(lines) == 2 and lines[0].startswith("  [y] 10/100")


def test_stage_logs_start_done_and_failure_and_reraises():
    clock, lines = Clock(), []
    with progress.stage("a", clock=clock, emit=lines.append):
        clock.now += 3
    assert len(lines) == 2 and lines[0].startswith("  [a] start, peak RSS ")
    assert lines[1].startswith("  [a] done in 3s, peak RSS ")
    lines.clear()
    with pytest.raises(RuntimeError, match="^boom$"):
        with progress.stage("b", clock=clock, emit=lines.append):
            raise RuntimeError("boom")
    assert lines[0].startswith("  [b] start") and lines[1].startswith("  [b] failed after 0s")


def test_quiet_stage_prints_only_when_slow():
    clock, lines = Clock(), []
    with progress.stage("q", quiet_under=5.0, clock=clock, emit=lines.append):
        clock.now += 4
    assert lines == []
    with progress.stage("q", quiet_under=5.0, clock=clock, emit=lines.append):
        clock.now += 6
    assert len(lines) == 1 and lines[0].startswith("  [q] done in 6s")


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("code", ["HOLDOUT_ACCESS_DENIED", "CONTRACT_MISMATCH"])
def test_native_refusal_logging_is_opt_in_and_never_includes_details(monkeypatch, enabled, code):
    lines, clock = [], Clock()
    refusal = fail(code, "requested events are excluded", details={"event_id": "private-event"})
    monkeypatch.setattr(progress, "_rss", lambda: ", peak RSS 12 MB")
    with pytest.raises(DataError) as caught:
        with progress.stage("admission", clock=clock, emit=lines.append, report_refusals=enabled):
            clock.now += 3
            raise refusal
    assert caught.value is refusal
    assert lines == ["  [admission] start, peak RSS 12 MB", "  [admission] failed after 3s, peak RSS 12 MB"
                     + (f"; refused {code}" if enabled else "")]
    assert "requested events are excluded" not in "".join(lines)


@pytest.mark.parametrize("failure", [RuntimeError("private detail"), KeyboardInterrupt("private detail"), SystemExit(7)])
def test_enabled_refusal_reporting_preserves_non_data_errors_and_interruptions(monkeypatch, failure):
    lines, clock = [], Clock()
    monkeypatch.setattr(progress, "_rss", lambda: ", peak RSS 12 MB")
    with pytest.raises(type(failure)) as caught:
        with progress.stage("admission", clock=clock, emit=lines.append, report_refusals=True):
            clock.now += 3
            raise failure
    assert caught.value is failure
    assert lines == ["  [admission] start, peak RSS 12 MB", "  [admission] failed after 3s, peak RSS 12 MB"]


def test_emit_failure_never_breaks_a_pass(monkeypatch):
    attempts = []

    def broken(*args, **kwargs):
        attempts.append(args)
        raise OSError("closed")
    monkeypatch.setattr("builtins.print", broken)
    with progress.stage("c"):
        result = 7
    assert result == 7 and len(attempts) == 2  # start and done were both attempted


@pytest.mark.parametrize("mode", ["filtered", "cumulative", "none_filtered", "refusal", "early_refusal", "broken_stderr"])
def test_shared_scan_counts_decoded_rows_against_bound_and_preserves_results(monkeypatch, capsys, mode):
    """Exercise both supported chain readers without private fault injection."""
    clock, calls = Clock(), []
    original_progress, original_stage = progress.Progress, progress.stage
    monkeypatch.setattr(progress, "Progress", lambda name, total: original_progress(name, total, clock=clock))
    monkeypatch.setattr(progress, "stage", lambda name, **kwargs: original_stage(name, clock=clock, **kwargs))
    monkeypatch.setattr(progress, "_rss", lambda: ", peak RSS 12 MB")
    refusal = fail("OBJECT_CORRUPT", "synthetic later scan refusal")
    attempts = []
    discarded = []
    if mode == "broken_stderr":
        original_print = print
        def broken(*args, **kwargs):
            if kwargs.get("file") is not None:
                attempts.append(args)
                raise OSError("closed stderr")
            return original_print(*args, **kwargs)
        monkeypatch.setattr("builtins.print", broken)
    class Repository:
        def table_contract(self, *args):
            return replace(contract_for("option_chains"), maximum_batch_rows=2)
        def fragment_records(self, *args):
            return [SimpleNamespace(partition_key="2024")]
        def scan_population_bound(self, *args, **kwargs):
            calls.append("bound")
            return 6  # Candidate fragments can contain rows the scan predicate excludes.
        def scan(self, query, *, table_name):
            calls.append("scan")
            assert query.max_result_rows == 6 and query.max_batch_rows == 2
            assert query.columns == ("ticker", "obs_date", "expiry", "dte", "strike", "right",
                                     "bid", "ask", "delta", "spot", "quote_repaired")
            assert [(p.column, p.operator, p.values) for p in query.key_filter if p.column == "year"] == [
                ("year", "eq", (2024,))]
            if mode == "early_refusal":
                clock.now += 7
                raise refusal
            edge_day = "2024-01-01" if mode == "cumulative" else "2024-01-02"
            rows = pa.RecordBatch.from_pydict({"ticker": ["A", "B", "A", "A"],
                "obs_date": pd.to_datetime([edge_day, "2024-01-01", "2024-01-01", edge_day]),
                "bid": [1., 2., 3., 4.]})
            for start, count, elapsed in ((0, 1, 10), (1, 2, 20), (3, 1, 30)):
                clock.now += elapsed
                yield rows.slice(start, count)
            if mode == "refusal":
                raise refusal
    snapshot = SimpleNamespace(snapshot_id="synthetic", table_versions={
        "option_chains": SimpleNamespace(table_contract_ref="synthetic-contract")})
    def read():
        if mode == "none_filtered":
            def filter_or_none(frame):
                filtered = frame[frame["bid"] == 3.]
                if filtered.empty:
                    discarded.append(len(frame))
                    return None
                return filtered
            return read_chains_for_years(Repository(), snapshot, iter(("2024",)), batch_filter=filter_or_none)
        return load_chain_index(Repository(), snapshot, [("A", "2024-01-01"), ("B", "2024-01-02")])
    if mode in ("refusal", "early_refusal"):
        with pytest.raises(DataError) as caught:
            read()
        assert caught.value is refusal
    else:
        result = read()
        if mode != "none_filtered":
            assert result.keys == {("A", pd.Timestamp("2024-01-01"))}
            result = result.get("A", "2024-01-01")
        assert result.to_dict("records") == [
            {"ticker": "A", "obs_date": pd.Timestamp("2024-01-01"), "bid": bid}
            for bid in ([1., 3., 4.] if mode == "cumulative" else [3.])]
    assert calls == ["bound", "scan"]
    assert discarded == ([1, 1] if mode == "none_filtered" else [])
    captured = capsys.readouterr()
    assert captured.out == ("" if mode in ("none_filtered", "refusal", "early_refusal") else
                            "  [replay] chain index: year 2024: " + ("3" if mode == "cumulative" else "1")
                            + " rows\n  [replay] chain index: 1 of 2 requested keys present\n")
    if mode == "broken_stderr":
        assert captured.err == "" and len(attempts) == 7  # Boundaries, two ticks/decisions and one partition.
    else:
        assert captured.err.splitlines() == [
            "  [read option_chains rows] start, peak RSS 12 MB",
        ] + ([] if mode == "early_refusal" else [
            "  [read option_chains rows/candidate bound; batch limit 2] 3/6, 30s elapsed, 0.10/s, ETA 30s, peak RSS 12 MB",
            "  [read option_chains filter] " + ("retained=2, skipped=1" if mode == "cumulative" else "retained=1, skipped=2"),
            "  [read option_chains rows/candidate bound; batch limit 2] 4/6, 60s elapsed, 0.07/s, ETA 30s, peak RSS 12 MB",
            "  [read option_chains filter] " + ("retained=3, skipped=1" if mode == "cumulative" else "retained=1, skipped=3"),
        ]) + ([
            "  [read option_chains rows] failed after " + ("7" if mode == "early_refusal" else "60") + "s, peak RSS 12 MB",
        ] if mode in ("refusal", "early_refusal") else [
            "  [read option_chains rows] done in 60s, peak RSS 12 MB",
            "  [read option_chains partitions] 1/1, 60s elapsed, 0.02/s, ETA 0s, peak RSS 12 MB"])


@pytest.mark.parametrize("years", [[], (), iter(())], ids=["list", "tuple", "iterator"])
def test_public_chain_reader_rejects_empty_years_before_access(years, capsys):
    class UntouchedRepository:
        def __getattr__(self, name):
            raise AssertionError(f"repository accessed before year admission: {name}")
    with pytest.raises(ValueError, match="^option_chains: years must not be empty$"):
        read_chains_for_years(UntouchedRepository(), None, years)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("years", ["2024", "", b"2024", b""])
def test_public_chain_reader_rejects_scalar_year_strings_before_access(years, capsys):
    class UntouchedRepository:
        def __getattr__(self, name):
            raise AssertionError(f"repository accessed before year admission: {name}")
    with pytest.raises(ValueError, match="^option_chains: years must be a non-string iterable$"):
        read_chains_for_years(UntouchedRepository(), None, years)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("state", ["start", "done in 0s", "failed after 0s"])
def test_cli_boundary_oracle_independently_requires_rss(state):
    finish = state if state != "start" else "done in 0s"
    lines = [f"  [read earnings_events rows] {s}" + ("" if s == state else ", peak RSS 12 MB")
             for s in ("start", finish)]
    expected = [("earnings_events", "failed" if finish.startswith("failed") else "done")]
    with pytest.raises(AssertionError):
        assert_scan_boundaries("\n".join(lines) + "\n", tables={"earnings_events"},
                               expected_scans=expected, allow_failures=True)


@pytest.mark.parametrize("state", ["start", "done in 0s", "failed after 0s"])
def test_cli_boundary_oracle_independently_rejects_trailing_noise(state):
    finish = state if state != "start" else "done in 0s"
    lines = [f"  [read earnings_events rows] {s}, peak RSS 12 MB" + ("; secret" if s == state else "")
             for s in ("start", finish)]
    expected = [("earnings_events", "failed" if finish.startswith("failed") else "done")]
    with pytest.raises(AssertionError):
        assert_scan_boundaries("\n".join(lines) + "\n", tables={"earnings_events"},
                               expected_scans=expected, allow_failures=True)


@pytest.mark.parametrize("stderr", [
    "Traceback\n",
    "  [read earnings_events rows] start, peak RSS 12 MB; secret\n",
    "  [read earnings_events rows] done in 0s, peak RSS 12 MB\n",
    "  [read earnings_events rows] start, peak RSS 12 MB\n",
    "  [read earnings_events rows] start, peak RSS 12 MB\n  [read daily_market rows] done in 0s, peak RSS 12 MB\n",
    "  [read earnings_events rows] start, peak RSS 12 MB\n  [read earnings_events rows] failed after 0s, peak RSS 12 MB\n",
    "  [read earnings_events rows] start, peak RSS 12 MB\n  [read earnings_events rows] done in secret\n",
])
def test_short_cli_stderr_oracle_rejects_noise_unpaired_boundaries_and_failures(stderr):
    with pytest.raises(AssertionError):
        assert_scan_boundaries(stderr, tables={"earnings_events", "daily_market"},
                               expected_scans=[("earnings_events", "done")], required_tables={"earnings_events"})


@pytest.mark.parametrize("table,stderr", [
    ("private_table", "  [read private_table rows] start, peak RSS 12 MB\n"
     "  [read private_table rows] done in 0s, peak RSS 12 MB\n"),
    ("earnings_events", "  [read earnings_events rows] start, peak RSS 12 MB\n  [read earnings_events rows] start, peak RSS 12 MB\n"
     "  [read earnings_events rows] done in 0s, peak RSS 12 MB\n"),
])
def test_short_cli_stderr_oracle_independently_rejects_unknown_tables_and_nested_starts(table, stderr):
    # With either guard deleted its transcript otherwise completes successfully.
    with pytest.raises(AssertionError):
        assert_scan_boundaries(stderr, tables={"earnings_events"},
                               expected_scans=[(table, "done")], required_tables={table})


@pytest.mark.parametrize("stderr,expected", [
    ("  [read earnings_events rows] start, peak RSS 12 MB\n  [read earnings_events rows] done in 0s, peak RSS 12 MB\n"
     "  [read earnings_events rows] start, peak RSS 12 MB\n", ("earnings_events", "done")),
    ("  [read earnings_events rows] start, peak RSS 12 MB\n"
     "  [read daily_market rows] done in 0s, peak RSS 12 MB\n", ("daily_market", "done")),
    ("  [read earnings_events rows] start, peak RSS 12 MB\n"
     "  [read earnings_events rows] failed after 0s, peak RSS 12 MB\n", ("earnings_events", "failed")),
], ids=["unclosed-after-completion", "cross-table-completion", "disallowed-failure"])
def test_short_cli_stderr_oracle_isolates_closing_and_failure_guards(stderr, expected):
    # Each transcript satisfies the required-table and exact-outcome checks.
    # Only its unclosed, wrong-table, or disallowed-failure guard rejects it.
    with pytest.raises(AssertionError):
        assert_scan_boundaries(stderr, tables={"earnings_events", "daily_market"},
                               expected_scans=[expected], required_tables={expected[0]})


def test_scan_observer_records_real_integrity_failure(tmp_path, monkeypatch, capsys):
    conn, clock, store = catalog_and_store(tmp_path)
    path = original = None
    try:
        contract = contract_for("earnings_events")
        event = history_event("2024-06-17")
        record = publish_and_inspect(store, contract, contract_ref_for(contract), [event], "2024")
        snapshot = commit_tables(conn, clock, {"earnings_events": [record]}, {"earnings_events": contract}, store=store)
        repository = Repository(conn, store)
        path = verify_object_path(store, record.object_ref)
        original = path.read_bytes()
        path.chmod(0o644)
        path.write_bytes(b"!" * len(original))
        original_stage = progress.stage
        monkeypatch.setattr(progress, "stage", lambda name, **kw: original_stage(name, clock=Clock(), **kw))
        monkeypatch.setattr(progress, "_rss", lambda: ", peak RSS 12 MB")
        capsys.readouterr()
        with observe_scan_outcomes({"earnings_events"}) as outcomes:
            with pytest.raises(DataError) as caught:
                load_population(repository, snapshot, as_of_month="2025-01", event_ids=[event["event_id"]])
        assert to_document(caught.value.problem) == {
            "code": "OBJECT_CORRUPT", "category": "integrity", "retryable": False,
            "message": "published object bytes do not match its recorded hash",
            "stage": None, "trace_id": None, "dependency_refs": [], "retry_after_seconds": None,
            "diagnostic_ref": None, "details": {}, "schema_version": "problem.v1.0",
        }
        assert outcomes == [("earnings_events", "failed")]
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ("  [read earnings_events rows] start, peak RSS 12 MB\n"
                                "  [read earnings_events rows] failed after 0s, peak RSS 12 MB\n")
        assert_scan_boundaries(captured.err, tables={"earnings_events"}, expected_scans=outcomes, allow_failures=True)
    finally:
        if original is not None:
            path.write_bytes(original)
            path.chmod(0o444)
        conn.close()


@pytest.mark.parametrize("stderr", [
    "",
    "  [read earnings_events rows] start, peak RSS 12 MB\n  [read earnings_events rows] done in 0s, peak RSS 12 MB\n",
])
def test_short_cli_stderr_oracle_requires_every_expected_table(stderr):
    with pytest.raises(AssertionError, match="missing scan boundaries"):
        assert_scan_boundaries(stderr, tables={"earnings_events", "daily_market"},
                               expected_scans=[("earnings_events", "done"), ("daily_market", "done")])


@pytest.mark.parametrize("state", ["done in 0s", "failed after 0s"])
def test_pre_scan_refusal_oracle_rejects_any_scan_boundary(state):
    with pytest.raises(AssertionError, match="pre-scan refusal emitted"):
        assert_scan_boundaries("  [read earnings_events rows] start, peak RSS 12 MB\n"
                               f"  [read earnings_events rows] {state}, peak RSS 12 MB\n",
                               tables={"earnings_events"}, expected_scans=[], required_tables=set(), allow_failures=True)


@pytest.mark.parametrize("outcome", ["done", "failed"])
def test_pre_scan_refusal_oracle_rejects_unlogged_repository_access(outcome):
    with pytest.raises(AssertionError, match="pre-scan refusal accessed"):
        assert_scan_boundaries("", tables={"earnings_events"},
                               expected_scans=[("earnings_events", outcome)], required_tables=set(), allow_failures=True)


@pytest.mark.parametrize("observed", [[], [("earnings_events", "done")] * 2, [("earnings_events", "failed")]])
def test_short_cli_stderr_oracle_rejects_missing_extra_or_wrong_scan_outcomes(observed):
    with pytest.raises(AssertionError, match="do not match repository calls"):
        assert_scan_boundaries("  [read earnings_events rows] start, peak RSS 12 MB\n"
                               "  [read earnings_events rows] done in 0s, peak RSS 12 MB\n",
                               tables={"earnings_events"}, expected_scans=observed, allow_failures=True)


def test_short_cli_stderr_oracle_accepts_only_complete_allowed_boundaries():
    assert_scan_boundaries("", tables={"earnings_events"}, expected_scans=[], required_tables=set())  # Pre-scan refusal.
    assert_scan_boundaries("  [read earnings_events rows] start, peak RSS 12 MB\n"
                           "  [read earnings_events rows] done in 2s, peak RSS 13 MB\n",
                           tables={"earnings_events"}, expected_scans=[("earnings_events", "done")])
    assert_scan_boundaries("  [read earnings_events rows] start, peak RSS 12 MB\n"
                           "  [read earnings_events rows] failed after 0s, peak RSS 12 MB\n",
                           tables={"earnings_events"}, expected_scans=[("earnings_events", "failed")], allow_failures=True)
    assert_scan_boundaries("  [read earnings_events rows] start, peak RSS 12 MB\n"
                           "  [read earnings_events rows] done in 0s, peak RSS 12 MB\n"
                           "  [read daily_market rows] start, peak RSS 12 MB\n  [read daily_market rows] done in 1s, peak RSS 12 MB\n",
                           tables={"earnings_events", "daily_market"},
                           expected_scans=[("earnings_events", "done"), ("daily_market", "done")])
    assert_scan_boundaries(("  [read earnings_events rows] start, peak RSS 12 MB\n"
                            "  [read earnings_events rows] done in 0s, peak RSS 12 MB\n") * 2,
                           tables={"earnings_events"}, expected_scans=[("earnings_events", "done")] * 2)


def _source(tmp_path):
    days = [d.date().isoformat() for d in pd.date_range("2022-06-02", "2023-05-30", freq="BMS")][:8]
    events = [history_event(day) for day in (*days, "2024-06-17")]
    moves = [outcome_move(e, float(n + 1)) for n, e in enumerate(events[:-1])] + [outcome_move(events[-1], None)]
    conn, repository, snapshot = commit_history_inputs(
        tmp_path, events, moves, days=calendar_sessions("2022-05-25", "2024-06-18").days)
    return conn, repository, snapshot


def test_pinned_build_is_output_identical_cheaper_and_logs_only_to_stderr(tmp_path, monkeypatch, capsys):
    decodes = []
    original = Repository._record

    def counted(self, *args, **kwargs):
        decodes.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Repository, "_record", counted)
    results = {}
    for name, pinned in (("plain", False), ("pinned", True)):
        conn, repository, snapshot = _source(tmp_path / name)
        try:
            repository = Repository(conn, repository._store, pin_snapshots=pinned)
            decodes.clear()
            capsys.readouterr()
            build = entry.build(repository, snapshot, **HISTORY_REQUEST)
            captured = capsys.readouterr()
            stored = entry.read(repository, snapshot,
                                repository.resolve(build["feature_snapshot"]["snapshot_id"]), **HISTORY_REQUEST)
            results[name] = (len(decodes), build["context"], stored["rows"], captured)
        finally:
            conn.close()
    assert results["pinned"][1] == results["plain"][1]
    assert results["pinned"][2] == results["plain"][2]
    assert results["plain"][2] and any(row["n_prior"] > 0 for row in results["plain"][2])
    assert results["pinned"][0] < results["plain"][0] / 2
    for _, _, _, captured in results.values():
        assert captured.out == ""
        for line in ("[history preflight] start", "[history preflight] done", "[history prepare] done",
                     "[history materialize] start", "[history materialize] done"):
            assert line in captured.err
