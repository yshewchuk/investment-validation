"""Fresh-process planning with permitted, synthetic-only ambient dotenv loading."""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from tests.v2.ops.test_massive_plan import cache as cache
from tests.v2.ops.test_massive_plan import cli_args, inventory, mutate_receipt, seed

ROOT = Path(__file__).resolve().parents[3]
KEYS = ("MASSIVE_API_KEY", "MASSIVE_S3_FILE_KEY_ID", "MASSIVE_S3_FILE_KEY_SECRET")
GUARDS = r'''
import contextlib, io, json, os, pathlib, runpy, sys
sys.path.insert(0, sys.argv[1])
dotenv_reads = []

def forbidden(message):
    pathlib.Path("guard-violation").write_text(message)
    raise AssertionError(message)

def audit(event, args):
    if event in ("socket.connect", "socket.sendto", "socket.getaddrinfo", "socket.gethostbyname"):
        forbidden("planner attempted network access")
    if event == "open" and isinstance(args[0], str) and pathlib.Path(args[0]).name == ".env":
        if pathlib.Path(args[0]).resolve() != pathlib.Path.cwd() / ".env" or dotenv_reads:
            forbidden("unexpected dotenv read")
        dotenv_reads.append(True)

sys.addaudithook(audit)
assert "engine" not in sys.modules
import engine
assert dotenv_reads == [True]
assert os.environ["PLANNER_STARTUP_SENTINEL"] == "synthetic-startup"
assert "engine.v2.ops.massive_plan" not in sys.modules
assert "tools.v2_massive_plan" not in sys.modules
original_getitem = type(os.environ).__getitem__

def guarded_getitem(self, key):
    if key.startswith(("MASSIVE_", "ORATS_", "POLYGON_")):
        forbidden("provider lookup")
    return original_getitem(self, key)

type(os.environ).__getitem__ = guarded_getitem
'''
CHILD = GUARDS + r'''
import subprocess
assert "MASSIVE_API_KEY" in engine.env.loaded_keys
arguments = json.loads(sys.argv[2])
guards, replacement, guard_worker = sys.argv[3:6]
source_root = sys.argv[1]
original_run, workers = subprocess.run, []
original_popen, processes = subprocess.Popen, []

def track_popen(*args, **kwargs):
    process = original_popen(*args, **kwargs)
    processes.append(process)
    return process

def guarded_worker(command, **kwargs):
    command = list(command)
    assert command[1:4] == ["-I", "-B", "-c"]
    if guard_worker == "yes":
        command[4] = guards + replacement + command[4]
        command.append(source_root)
    workers.append(True)
    result = original_run(command, **kwargs)
    assert "AssertionError" not in result.stderr, result.stderr
    return result

subprocess.run = guarded_worker
subprocess.Popen = track_popen
exec(replacement)
out, err = io.StringIO(), io.StringIO()
sys.argv = ["tools.v2_massive_plan", *arguments]
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
    try:
        if replacement:
            from tools import v2_massive_plan as cli
            cli.PLAN_TIMEOUT_SECONDS = 5
            raise SystemExit(cli.main(arguments))
        else:
            runpy.run_module("tools.v2_massive_plan", run_name="__main__")
    except SystemExit as exc:
        code = exc.code
assert dotenv_reads == [True]
assert workers == [True]
assert len(processes) == 1 and processes[0].poll() is not None
assert not pathlib.Path("guard-violation").exists()
print(json.dumps({"code": code, "stdout": out.getvalue(), "stderr": err.getvalue()}))
'''


def _run(root, arguments, revision, replacement="", *, guard_worker=True):
    markers = [f"synthetic-planner-{revision}-{index}" for index in range(len(KEYS))]
    (root / ".env").write_text("PLANNER_STARTUP_SENTINEL=synthetic-startup\n" + "".join(
        f"{key}={value}\n" for key, value in zip(KEYS, markers, strict=True)))
    with subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", CHILD, str(ROOT), json.dumps(arguments),
         GUARDS, replacement, "yes" if guard_worker else "no"],
        cwd=root, env={"HOME": str(root), "INVESTING_PLAN_ROOT": str(root)},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=15 if replacement else 40)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
    assert process.returncode == 0, stderr
    assert all(marker not in stdout + stderr for marker in markers)
    return json.loads(stdout), markers


@pytest.mark.parametrize("mode", [
    "help", "invalid", "s3", "missing", "empty", "cached", "fifo", "nul_catalog", "nul_store",
    "receipt_corrupt", "bytes_corrupt", "object_missing",
    "cwd_shadow",
    "missing_flags", "missing_value", "bad_adjustment", "bad_cap", "unsupported_transport", "reversed_dates",
])
def test_ambient_startup_never_becomes_provider_io_or_credential_output(cache, mode):
    conn, _, root = cache
    if mode == "cwd_shadow":
        (root / "json.py").write_text('raise RuntimeError("unrelated cwd module")\n')
    if mode == "cached":
        for unit in inventory(cache)["units"]:
            seed(cache, unit["request"])
    if mode in ("receipt_corrupt", "bytes_corrupt", "object_missing"):
        record = seed(cache, inventory(datasets=["daily"])["units"][0]["request"])
        digest = record.raw_hash.removeprefix("sha256:")
        path = root / "store" / "objects" / digest[:2] / digest
        if mode == "receipt_corrupt":
            mutate_receipt(cache, record, "raw_receipt_id", "raw_" + "0" * 32)
        elif mode == "bytes_corrupt":
            path.chmod(0o600)
            path.write_bytes(b"tampered")
        else:
            path.unlink()
    arguments = cli_args(root)
    if mode == "help":
        arguments = ["--help"]
    elif mode == "invalid":
        arguments += ["--unknown-option", "synthetic-private-unknown"]
    elif mode == "missing_flags":
        arguments = []
    elif mode == "missing_value":
        arguments += ["--ticker"]
    elif mode == "bad_adjustment":
        arguments = cli_args(root, adjustment="synthetic-private-choice")
    elif mode == "bad_cap":
        arguments = cli_args(root, **{"max-calls-per-request": "synthetic-private-cap"})
    elif mode == "reversed_dates":
        arguments = cli_args(root, start="2026-02-01", end="2026-01-31")
    elif mode == "unsupported_transport":
        arguments = cli_args(root, transport="synthetic-private-transport")
    elif mode == "s3":
        arguments = cli_args(root, transport="s3")
    elif mode == "missing":
        arguments = cli_args(root, catalog=root / "missing.sqlite")
    elif mode == "fifo":
        fifo = root / "catalog.fifo"
        os.mkfifo(fifo)
        arguments = cli_args(root, catalog=fifo)
    elif mode == "nul_catalog":
        arguments = cli_args(root, catalog=str(root / "bad") + "\0")
    elif mode == "nul_store":
        arguments = cli_args(root, **{"store-root": str(root / "bad") + "\0"})
    before = tuple(conn.iterdump())
    objects = {str(path.relative_to(root)): path.read_bytes()
               for path in (root / "store").rglob("*") if path.is_file()}
    first, markers = _run(root, arguments, "first")
    second, more_markers = _run(root, arguments, "second")
    assert first == second
    assert first["code"] == (0 if mode in ("help", "empty", "cached", "cwd_shadow") else 2)
    if mode in ("empty", "cached", "cwd_shadow"):
        plan = json.loads(first["stdout"])
        assert plan["plan_only"] is True
        assert plan["minimum_initial_calls"] == (0 if mode == "cached" else 3)
        if mode == "cached":
            assert all(unit["raw_receipt_id"] for unit in plan["units"])
            assert plan["maximum_http_attempts"] == 0
    if mode in ("missing", "fifo", "nul_catalog", "nul_store"):
        assert json.loads(first["stdout"]) == {"refused": "RESOURCE_UNAVAILABLE"}
        assert first["stderr"] == ""
        assert not (root / "missing.sqlite").exists()
    if mode in ("s3", "unsupported_transport"):
        assert json.loads(first["stdout"]) == {
            "refused": "INVALID_REQUEST", "reason": "CAPABILITY_UNAVAILABLE"}
        assert first["stderr"] == ""
    if mode in ("invalid", "missing_flags", "missing_value", "bad_adjustment", "bad_cap", "reversed_dates"):
        assert json.loads(first["stdout"]) == {"refused": "INVALID_REQUEST", "reason": None}
        assert first["stderr"] == ""
    if mode == "help":
        assert first["stdout"].startswith("usage: tools.v2_massive_plan ")
        assert "--max-calls-per-request" in first["stdout"]
        assert first["stderr"] == ""
    if mode in ("receipt_corrupt", "bytes_corrupt", "object_missing"):
        assert json.loads(first["stdout"]) == {
            "refused": "MISSING" if mode == "object_missing" else "INTEGRITY_FAILED",
            "reason": None}
        assert first["stderr"] == ""
    assert tuple(conn.iterdump()) == before
    assert objects == {str(path.relative_to(root)): path.read_bytes()
                       for path in (root / "store").rglob("*") if path.is_file()}
    assert all(marker not in str(before) + str(objects) for marker in markers + more_markers)


@pytest.mark.parametrize("cached", [False, True])
def test_unmodified_worker_bootstrap_succeeds_outside_source_tree(cache, cached):
    if cached:
        for unit in inventory(cache)["units"]:
            seed(cache, unit["request"])
    output, _ = _run(cache[2], cli_args(cache[2]), "unmodified", guard_worker=False)
    assert output["code"] == 0
    assert output["stderr"] == ""
    assert json.loads(output["stdout"]) == inventory(cache)


@pytest.mark.parametrize("failure", ["replacement", "partial_timeout", "crash"])
def test_catalog_replacement_with_fifo_has_a_bounded_refusal(cache, failure):
    conn, _, root = cache
    catalog, saved = root / "ops.sqlite", root / "saved.sqlite"
    before = tuple(conn.iterdump())
    replacement = f'''
def replace_catalog(event, args):
    if event == "sqlite3.connect" and str(args[0]).endswith("?mode=ro"):
        if {failure!r} != "replacement":
            print("discard partial stdout", flush=True)
            print("discard private diagnostic", file=sys.stderr, flush=True)
        if {failure!r} == "crash":
            raise RuntimeError("synthetic worker crash")
        os.replace({str(catalog)!r}, {str(saved)!r})
        os.mkfifo({str(catalog)!r})
sys.addaudithook(replace_catalog)
'''
    try:
        output, _ = _run(root, cli_args(root), "replacement", replacement)
        if failure != "crash":
            assert saved.is_file(), "replacement must occur after preflight and before SQLite open"
        assert output == {"code": 2, "stdout": '{"refused": "RESOURCE_UNAVAILABLE"}\n',
                          "stderr": ""}
        assert tuple(conn.iterdump()) == before
    finally:
        if saved.exists():
            catalog.unlink()
            saved.rename(catalog)
