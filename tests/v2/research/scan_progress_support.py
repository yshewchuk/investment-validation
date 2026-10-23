"""Strict stderr boundary oracle for short synthetic CLI scans."""
import re
from contextlib import contextmanager
from unittest.mock import patch

from engine.v2.data.repository import Repository


@contextmanager
def observe_scan_outcomes(tables):
    """Observe public scan calls independently of the telemetry implementation."""
    original, outcomes = Repository.scan, []
    def observed(repository, query, *, table_name, **kwargs):
        if table_name not in tables:
            yield from original(repository, query, table_name=table_name, **kwargs)
            return
        index = len(outcomes)
        outcomes.append((table_name, "pending"))
        try:
            yield from original(repository, query, table_name=table_name, **kwargs)
        except BaseException:
            outcomes[index] = (table_name, "failed")
            raise
        else:
            outcomes[index] = (table_name, "done")
    with patch.object(Repository, "scan", observed):
        yield outcomes


def assert_scan_boundaries(stderr, *, tables, expected_scans, required_tables=None, allow_failures=False):
    """Match every observed CLI scan with RSS; a pre-scan refusal permits no output."""
    required = tables if required_tables is None else required_tables
    if not required:
        assert stderr == "", "pre-scan refusal emitted scan boundaries"
        assert expected_scans == [], "pre-scan refusal accessed the repository"
        return
    completed = []
    active = None
    for line in stderr.splitlines():
        match = re.fullmatch(
            r"  \[read ([a-z_]+) rows\] (start|done in [0-9]+s|failed after [0-9]+s)"
            r", peak RSS [0-9]+ MB", line)
        assert match is not None, line
        table, state = match.groups()
        assert table in tables, table
        if state == "start":
            assert active is None, line
            active = table
        else:
            assert active == table, line
            assert allow_failures or state.startswith("done in "), line
            completed.append((table, "done" if state.startswith("done in ") else "failed"))
            active = None
    assert active is None, "unclosed scan boundary"
    seen = {table for table, _ in completed}
    assert required <= seen, f"missing scan boundaries: {required - seen}"
    assert completed == expected_scans, "scan boundaries do not match repository calls"
