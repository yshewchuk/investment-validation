"""P6-6 session-evidence completeness: the per-row ``evidence`` field resolved
against a fake ops catalog, a route-probe receipt and resource-measurement
records (no real evidence tree or heavy run is required).

Each test isolates the row shape it is about, so a failure names exactly the
mechanism that broke: a ``job:`` row needs a succeeded attempt inside the
window (a FAIL receipt is not evidence -- Opus review #1), a ``route:`` row
needs a 2xx from THIS session's probe, a ``cli:`` row needs an exit-0,
not-killed measurement inside the window, ``exempt``/``open`` rows never need
evidence, and a row with no ``evidence`` key fails closed.
"""
from __future__ import annotations

import json
import sqlite3
import textwrap
from pathlib import Path

from tools import v2_session_evidence_check as sec
from tools.phase6_inventory import DECLARATIONS, load_declarations

SESSION = "s1"
START = "2026-09-25T00:00:00Z"
END = "2026-09-25T23:59:59Z"
IN_WINDOW = "2026-09-25T12:00:00.000000Z"
OUT_OF_WINDOW = "2026-09-24T12:00:00.000000Z"


# ------------------------------------------------------------------- fixtures


def _row(row_id, evidence, *, disposition="native"):
    return textwrap.dedent(f'''
        [[row]]
        id = "{row_id}"
        area = "board"
        capability = "c"
        new = []
        producer = "p"
        identity = "i"
        tests = []
        disposition = "{disposition}"
        owner = "P6-4"
        evidence = "{evidence}"
    ''')


def _row_without_evidence(row_id):
    return textwrap.dedent(f'''
        [[row]]
        id = "{row_id}"
        area = "board"
        capability = "c"
        new = []
        producer = "p"
        identity = "i"
        tests = []
        disposition = "native"
        owner = "P6-4"
    ''')


def _row_list(row_id, evidences):
    rendered = ", ".join(f'"{entry}"' for entry in evidences)
    return textwrap.dedent(f'''
        [[row]]
        id = "{row_id}"
        area = "board"
        capability = "c"
        new = []
        producer = "p"
        identity = "i"
        tests = []
        disposition = "native"
        owner = "P6-4"
        evidence = [{rendered}]
    ''')


def _declarations(tmp_path, *blocks):
    path = tmp_path / "capabilities.toml"
    path.write_text("".join(blocks))
    return path


def _catalog(path, *, jobs=(), attempts=(), outbox=()):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE jobs (job_id TEXT PRIMARY KEY, kind TEXT NOT NULL)")
    conn.execute("CREATE TABLE attempts (attempt_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, "
                 "state TEXT NOT NULL, started_at TEXT)")
    conn.execute("CREATE TABLE outbox (effect_id TEXT PRIMARY KEY, kind TEXT NOT NULL, "
                 "state TEXT NOT NULL)")
    conn.executemany("INSERT INTO jobs VALUES (?, ?)", jobs)
    conn.executemany("INSERT INTO attempts VALUES (?, ?, ?, ?)", attempts)
    conn.executemany("INSERT INTO outbox VALUES (?, ?, ?)", outbox)
    conn.commit()
    conn.close()
    return path


def _check(tmp_path, declarations, *, evidence=None, catalog=None, session=SESSION):
    return sec.check(
        session=session, window_start=START, window_end=END,
        catalog=catalog if catalog is not None else _catalog(tmp_path / "catalog.sqlite"),
        declarations=declarations,
        evidence_dir=evidence if evidence is not None else tmp_path / "evidence")


def _route_receipt(evidence, *, session=SESSION, routes=(("GET", "/y", 200),),
                   generated_at=IN_WINDOW):
    path = evidence / "route_probe" / f"{session}-route_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": "route_probe_receipt.v1.0",
        "session": session,
        "generated_at": generated_at,
        "routes": [{"method": method, "path": concrete, "status": status}
                   for method, concrete, status in routes],
        "all_2xx": all(200 <= status < 300 for _, _, status in routes),
    }))
    return path


def _resource_record(evidence, *, command, exit_code=0, killed=False,
                     started_at=IN_WINDOW, name="run.json"):
    directory = evidence / "resource_measurement"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps({
        "schema_version": "v2_resource_measurement_record.v1.0",
        "command": list(command), "exit_code": exit_code, "killed": killed,
        "started_at": started_at, "ended_at": started_at,
    }))


# ----------------------------------------------------------------- job/effect


def test_succeeded_attempt_inside_the_window_covers_the_row(tmp_path):
    declarations = _declarations(tmp_path, _row("job-row", "job:legacy_score"))
    catalog = _catalog(tmp_path / "catalog.sqlite", jobs=[("j1", "legacy_score")],
                       attempts=[("a1", "j1", "succeeded", IN_WINDOW)])

    result = _check(tmp_path, declarations, catalog=catalog)

    assert result["rows_uncovered"] == []
    assert result["rows_covered"] == 1
    assert result["rows"][0]["source"] == "job"
    assert result["rows"][0]["window_checked"] is True


def test_window_end_with_no_fraction_covers_the_whole_last_second(tmp_path):
    declarations = _declarations(tmp_path, _row("job-row", "job:legacy_score"))
    catalog = _catalog(tmp_path / "catalog.sqlite", jobs=[("j1", "legacy_score")],
                       attempts=[("a1", "j1", "succeeded", "2026-09-25T23:59:59.412000Z")])

    result = sec.check(
        session=SESSION, window_start=START, window_end="2026-09-25T23:59:59Z",
        catalog=catalog, declarations=declarations, evidence_dir=tmp_path / "evidence")

    assert result["rows_uncovered"] == []
    assert result["rows_covered"] == 1
    assert result["rows"][0]["status"] == "covered"
    assert result["rows"][0]["source"] == "job"


def test_window_end_with_explicit_zero_fraction_is_not_widened(tmp_path):
    declarations = _declarations(tmp_path, _row("job-row", "job:legacy_score"))
    catalog = _catalog(tmp_path / "catalog.sqlite", jobs=[("j1", "legacy_score")],
                       attempts=[("a1", "j1", "succeeded", "2026-09-25T23:59:59.412000Z")])

    result = sec.check(
        session=SESSION, window_start=START, window_end="2026-09-25T23:59:59.000000Z",
        catalog=catalog, declarations=declarations, evidence_dir=tmp_path / "evidence")

    assert result["rows_uncovered"] == ["job-row"]
    assert result["rows_covered"] == 0
    assert result["rows"][0]["status"] == "uncovered"
    assert result["rows"][0]["source"] == "job"


def test_succeeded_attempt_outside_the_window_is_not_evidence(tmp_path):
    declarations = _declarations(tmp_path, _row("job-row", "job:legacy_score"))
    catalog = _catalog(tmp_path / "catalog.sqlite", jobs=[("j1", "legacy_score")],
                       attempts=[("a1", "j1", "succeeded", OUT_OF_WINDOW)])

    result = _check(tmp_path, declarations, catalog=catalog)

    assert result["rows_uncovered"] == ["job-row"]


def test_failed_attempt_inside_the_window_is_not_evidence(tmp_path):
    declarations = _declarations(tmp_path, _row("job-row", "job:legacy_score"))
    catalog = _catalog(tmp_path / "catalog.sqlite", jobs=[("j1", "legacy_score")],
                       attempts=[("a1", "j1", "failed", IN_WINDOW)])

    result = _check(tmp_path, declarations, catalog=catalog)

    assert result["rows_uncovered"] == ["job-row"]


def test_delivered_outbox_row_outside_the_window_is_not_evidence(tmp_path):
    declarations = _declarations(tmp_path, _row("effect-row", "job:publication"))
    catalog = _catalog(tmp_path / "catalog.sqlite",
                       outbox=[("e1", "publication", "delivered")])

    result = _check(tmp_path, declarations, catalog=catalog)

    assert result["rows_uncovered"] == ["effect-row"]
    assert result["rows"][0]["source"] == "job"
    assert "no succeeded 'publication' attempt inside the window" in result["rows"][0]["detail"]


def test_absent_catalog_fails_closed_for_job_rows(tmp_path):
    declarations = _declarations(tmp_path, _row("job-row", "job:legacy_score"))

    result = _check(tmp_path, declarations, catalog=tmp_path / "absent.sqlite")

    assert result["rows_uncovered"] == ["job-row"]
    assert "catalog unavailable" in result["rows"][0]["detail"]


# ---------------------------------------------------------------------- route


def test_route_receipt_2xx_covers_the_row(tmp_path):
    declarations = _declarations(tmp_path, _row("route-row", "route:GET /y"))
    evidence = tmp_path / "evidence"
    _route_receipt(evidence)

    result = _check(tmp_path, declarations, evidence=evidence)

    assert result["rows_uncovered"] == []
    assert result["rows"][0]["source"] == "route"


def test_route_receipt_non_2xx_is_not_evidence(tmp_path):
    declarations = _declarations(tmp_path, _row("route-row", "route:GET /y"))
    evidence = tmp_path / "evidence"
    _route_receipt(evidence, routes=(("GET", "/y", 500),))

    result = _check(tmp_path, declarations, evidence=evidence)

    assert result["rows_uncovered"] == ["route-row"]


def test_route_receipt_for_another_session_is_not_cross_counted(tmp_path):
    declarations = _declarations(tmp_path, _row("route-row", "route:GET /y"))
    evidence = tmp_path / "evidence"
    _route_receipt(evidence, session="other")

    result = _check(tmp_path, declarations, evidence=evidence)

    assert result["rows_uncovered"] == ["route-row"]


def test_route_receipt_generated_outside_the_window_is_not_evidence(tmp_path):
    declarations = _declarations(tmp_path, _row("route-row", "route:GET /y"))
    evidence = tmp_path / "evidence"
    _route_receipt(evidence, generated_at=OUT_OF_WINDOW)

    result = _check(tmp_path, declarations, evidence=evidence)

    assert result["rows_uncovered"] == ["route-row"]
    assert "outside the session window" in result["rows"][0]["detail"]


def test_route_receipt_naming_another_session_inside_is_refused(tmp_path):
    declarations = _declarations(tmp_path, _row("route-row", "route:GET /y"))
    evidence = tmp_path / "evidence"
    path = evidence / "route_probe" / f"{SESSION}-route_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"session": "other", "generated_at": IN_WINDOW,
                                "routes": [
        {"method": "GET", "path": "/y", "status": 200}]}))

    result = _check(tmp_path, declarations, evidence=evidence)

    assert result["rows_uncovered"] == ["route-row"]
    assert "names session 'other'" in result["rows"][0]["detail"]


# ------------------------------------------------------------------------ cli


def test_clean_cli_measurement_covers_the_row(tmp_path):
    declarations = _declarations(tmp_path, _row("cli-row", "cli:tools/z.py"))
    evidence = tmp_path / "evidence"
    _resource_record(evidence, command=["python3", "tools/z.py", "--run"])

    result = _check(tmp_path, declarations, evidence=evidence)

    assert result["rows_uncovered"] == []
    assert result["rows"][0]["window_checked"] is True


def test_cli_measurement_with_nonzero_exit_is_not_evidence(tmp_path):
    declarations = _declarations(tmp_path, _row("cli-row", "cli:tools/z.py"))
    evidence = tmp_path / "evidence"
    _resource_record(evidence, command=["python3", "tools/z.py"], exit_code=1)

    result = _check(tmp_path, declarations, evidence=evidence)

    assert result["rows_uncovered"] == ["cli-row"]


def test_cli_measurement_killed_despite_exit_zero_is_not_evidence(tmp_path):
    declarations = _declarations(tmp_path, _row("cli-row", "cli:tools/z.py"))
    evidence = tmp_path / "evidence"
    _resource_record(evidence, command=["python3", "tools/z.py"],
                     exit_code=0, killed=True)

    result = _check(tmp_path, declarations, evidence=evidence)

    assert result["rows_uncovered"] == ["cli-row"]


def test_cli_measurement_outside_the_window_is_not_evidence(tmp_path):
    declarations = _declarations(tmp_path, _row("cli-row", "cli:tools/z.py"))
    evidence = tmp_path / "evidence"
    _resource_record(evidence, command=["python3", "tools/z.py"],
                     started_at=OUT_OF_WINDOW)

    result = _check(tmp_path, declarations, evidence=evidence)

    assert result["rows_uncovered"] == ["cli-row"]


def test_ops_subcommand_evidence_matches_the_module_argv(tmp_path):
    declarations = _declarations(tmp_path, _row("cli-row", "cli:ops ledger calibrate"))
    evidence = tmp_path / "evidence"
    _resource_record(evidence, command=["python3", "-m", "engine.v2.ops",
                                        "ledger", "calibrate"])

    result = _check(tmp_path, declarations, evidence=evidence)

    assert result["rows_uncovered"] == []


def test_unlisted_dotted_module_is_not_reduced_to_its_last_name(tmp_path):
    declarations = _declarations(tmp_path, _row("cli-row", "cli:ops ledger status"))
    evidence = tmp_path / "evidence"
    _resource_record(evidence, command=["python3", "-m", "foo.ops",
                                        "ledger", "status"])

    result = _check(tmp_path, declarations, evidence=evidence)

    assert result["rows_uncovered"] == ["cli-row"]


def test_allowlisted_engine_ops_module_matches_ops_evidence(tmp_path):
    declarations = _declarations(tmp_path, _row("cli-row", "cli:ops ledger status"))
    evidence = tmp_path / "evidence"
    _resource_record(evidence, command=["python3", "-m", "engine.v2.ops",
                                        "ledger", "status"])

    result = _check(tmp_path, declarations, evidence=evidence)

    assert result["rows_uncovered"] == []


def test_list_evidence_needs_every_entry_and_names_the_missing_one(tmp_path):
    declarations = _declarations(
        tmp_path, _row_list("tools-row", ["cli:tools/x.py", "cli:tools/y.py"]))
    evidence = tmp_path / "evidence"
    _resource_record(evidence, command=["python3", "tools/x.py"], name="x.json")

    result = _check(tmp_path, declarations, evidence=evidence)

    assert result["rows_uncovered"] == ["tools-row"]
    detail = result["rows"][0]["detail"]
    assert "tools/y.py" in detail
    assert "tools/x.py" not in detail


def test_list_evidence_is_covered_when_every_entry_ran(tmp_path):
    declarations = _declarations(
        tmp_path, _row_list("tools-row", ["cli:tools/x.py", "cli:tools/y.py"]))
    evidence = tmp_path / "evidence"
    _resource_record(evidence, command=["python3", "tools/x.py"], name="x.json")
    _resource_record(evidence, command=["python3", "tools/y.py"], name="y.json")

    result = _check(tmp_path, declarations, evidence=evidence)

    assert result["rows_uncovered"] == []
    assert result["rows_covered"] == 1


def test_empty_evidence_list_fails_closed(tmp_path):
    declarations = _declarations(tmp_path, _row_list("empty-row", []))

    result = _check(tmp_path, declarations)

    assert result["rows_uncovered"] == ["empty-row"]
    assert "no usable evidence field" in result["rows"][0]["detail"]


def test_non_string_evidence_element_fails_closed(tmp_path):
    block = textwrap.dedent('''
        [[row]]
        id = "mixed-row"
        area = "board"
        capability = "c"
        new = []
        producer = "p"
        identity = "i"
        tests = []
        disposition = "native"
        owner = "P6-4"
        evidence = ["cli:x", 3]
    ''')
    declarations = _declarations(tmp_path, block)

    result = _check(tmp_path, declarations)

    assert result["rows_uncovered"] == ["mixed-row"]
    assert "no usable evidence field" in result["rows"][0]["detail"]


def test_unreadable_resource_record_is_reported_not_fatal(tmp_path):
    declarations = _declarations(tmp_path, _row("cli-row", "cli:tools/z.py"))
    evidence = tmp_path / "evidence"
    directory = evidence / "resource_measurement"
    directory.mkdir(parents=True)
    (directory / "broken.json").write_text("{ not json")
    _resource_record(evidence, command=["python3", "tools/z.py"])

    result = _check(tmp_path, declarations, evidence=evidence)

    assert result["rows_uncovered"] == []
    assert result["unreadable_evidence_files"] == [str(directory / "broken.json")]


# -------------------------------------------------------- exempt, open, absent


def test_exempt_rows_never_need_evidence(tmp_path):
    declarations = _declarations(
        tmp_path,
        _row("explicit-exempt", "exempt"),
        _row("dormant-row", "exempt", disposition="dormant-historical"),
        _row("missing-row", "exempt", disposition="missing"))

    result = _check(tmp_path, declarations)

    assert result["rows_uncovered"] == []
    assert result["rows_exempt"] == 3
    assert result["rows_covered"] == 0


def test_open_rows_are_reported_open_never_uncovered_and_fail_the_verdict(tmp_path):
    declarations = _declarations(tmp_path, _row("open-row", "open"))

    result = _check(tmp_path, declarations)

    assert result["rows_uncovered"] == []
    assert result["rows_open"] == ["open-row"]
    assert result["verdict"] == "FAIL"


def test_row_missing_the_evidence_key_fails_closed(tmp_path):
    declarations = _declarations(tmp_path, _row_without_evidence("undeclared"))

    result = _check(tmp_path, declarations)

    assert result["rows_uncovered"] == ["undeclared"]
    assert "no usable evidence field" in result["rows"][0]["detail"]


def test_user_decision_entries_are_not_rows(tmp_path):
    user_decision = textwrap.dedent('''
        [[user_decision]]
        id = "UD-9"
        rows = ["exempt-row"]
        question = "accept?"
    ''')
    declarations = _declarations(tmp_path, _row("exempt-row", "exempt"), user_decision)

    result = _check(tmp_path, declarations)

    assert result["rows_total"] == 1
    assert result["rows_exempt"] == 1


# ------------------------------------------------- real declarations guard


def test_real_declarations_never_require_a_post_route():
    """The route probe requests GET routes only, so a ``route:POST ...`` row
    could never be covered; the shipped declarations must not contain one."""
    declared = load_declarations(sec.ROOT / DECLARATIONS)

    for row in declared.get("row", []):
        evidence = row.get("evidence")
        entries = evidence if isinstance(evidence, list) else [evidence]
        assert not any(isinstance(entry, str) and entry.startswith("route:POST")
                       for entry in entries), row.get("id")


# --------------------------------------------------------------- CLI surface


def test_all_covered_session_passes_and_exits_zero(tmp_path, capsys):
    declarations = _declarations(
        tmp_path,
        _row("job-row", "job:legacy_score"),
        _row("route-row", "route:GET /y"),
        _row("cli-row", "cli:tools/z.py"),
        _row("exempt-row", "exempt"))
    catalog = _catalog(tmp_path / "catalog.sqlite", jobs=[("j1", "legacy_score")],
                       attempts=[("a1", "j1", "succeeded", IN_WINDOW)])
    evidence = tmp_path / "evidence"
    _route_receipt(evidence)
    _resource_record(evidence, command=["python3", "tools/z.py"])

    code = sec.main(["--session", SESSION, "--window-start", START, "--window-end", END,
                     "--catalog", str(catalog), "--declarations", str(declarations),
                     "--evidence-dir", str(evidence), "--json"])

    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["verdict"] == "PASS"
    assert result["rows_uncovered"] == [] and result["rows_open"] == []
    assert result["rows_covered"] == 3 and result["rows_exempt"] == 1


def test_human_output_names_uncovered_and_open_rows(tmp_path, capsys):
    declarations = _declarations(
        tmp_path,
        _row("open-row", "open"),
        _row("job-row", "job:legacy_score"))

    code = sec.main(["--session", SESSION, "--window-start", START, "--window-end", END,
                     "--catalog", str(_catalog(tmp_path / "catalog.sqlite")),
                     "--declarations", str(declarations),
                     "--evidence-dir", str(tmp_path / "evidence")])

    assert code == 1
    out = capsys.readouterr().out
    assert "1 open, 0 covered, 1 uncovered -> FAIL" in out
    assert "UNCOVERED job-row" in out
    assert "OPEN open-row" in out


def test_naive_window_is_refused(tmp_path, capsys):
    declarations = _declarations(tmp_path, _row("exempt-row", "exempt"))

    code = sec.main(["--session", SESSION, "--window-start", "2026-09-25T00:00:00",
                     "--window-end", END,
                     "--catalog", str(_catalog(tmp_path / "catalog.sqlite")),
                     "--declarations", str(declarations)])

    assert code == 2
    assert "refused" in capsys.readouterr().err
