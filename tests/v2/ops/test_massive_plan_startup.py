"""Fresh-process planning with permitted, synthetic-only ambient dotenv loading."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.v2.ops.test_massive_plan import cache as cache
from tests.v2.ops.test_massive_plan import cli_args, inventory, seed

ROOT = Path(__file__).resolve().parents[3]
KEYS = ("MASSIVE_API_KEY", "MASSIVE_S3_FILE_KEY_ID", "MASSIVE_S3_FILE_KEY_SECRET")
CHILD = r'''
import contextlib, io, json, os, pathlib, runpy, sys
sys.path.insert(0, sys.argv[1])
arguments = json.loads(sys.argv[2])
dotenv_reads = []

def audit(event, args):
    if event in ("socket.connect", "socket.sendto", "socket.getaddrinfo", "socket.gethostbyname"):
        raise AssertionError("planner attempted network access")
    if event == "open" and isinstance(args[0], str) and pathlib.Path(args[0]).name == ".env":
        assert pathlib.Path(args[0]).resolve() == pathlib.Path.cwd() / ".env"
        dotenv_reads.append(True)

sys.addaudithook(audit)
assert "engine" not in sys.modules
import engine
assert dotenv_reads == [True]
assert os.environ["PLANNER_STARTUP_SENTINEL"] == "synthetic-startup"
assert "MASSIVE_API_KEY" in engine.env.loaded_keys
assert "engine.v2.ops.massive_plan" not in sys.modules
assert "tools.v2_massive_plan" not in sys.modules
original_getitem = type(os.environ).__getitem__

def guarded_getitem(self, key):
    assert not key.startswith(("MASSIVE_", "ORATS_", "POLYGON_")), "provider lookup"
    return original_getitem(self, key)

type(os.environ).__getitem__ = guarded_getitem
out, err = io.StringIO(), io.StringIO()
sys.argv = ["tools.v2_massive_plan", *arguments]
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
    try:
        runpy.run_module("tools.v2_massive_plan", run_name="__main__")
    except SystemExit as exc:
        code = exc.code
assert dotenv_reads == [True]
print(json.dumps({"code": code, "stdout": out.getvalue(), "stderr": err.getvalue()}))
'''


def _run(root, arguments, revision):
    markers = [f"synthetic-planner-{revision}-{index}" for index in range(len(KEYS))]
    (root / ".env").write_text("PLANNER_STARTUP_SENTINEL=synthetic-startup\n" + "".join(
        f"{key}={value}\n" for key, value in zip(KEYS, markers, strict=True)))
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", CHILD, str(ROOT), json.dumps(arguments)],
        cwd=root, env={"HOME": str(root), "INVESTING_PLAN_ROOT": str(root)},
        capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert all(marker not in result.stdout + result.stderr for marker in markers)
    return json.loads(result.stdout), markers


@pytest.mark.parametrize("mode", ["help", "invalid", "s3", "missing", "empty", "cached"])
def test_ambient_startup_never_becomes_provider_io_or_credential_output(cache, mode):
    conn, _, root = cache
    if mode == "cached":
        for unit in inventory(cache)["units"]:
            seed(cache, unit["request"])
    arguments = cli_args(root)
    if mode == "help":
        arguments = ["--help"]
    elif mode == "invalid":
        arguments = ["--unknown-option"]
    elif mode == "s3":
        arguments = cli_args(root, transport="s3")
    elif mode == "missing":
        arguments = cli_args(root, catalog=root / "missing.sqlite")
    before = tuple(conn.iterdump())
    objects = {str(path.relative_to(root)): path.read_bytes()
               for path in (root / "store").rglob("*") if path.is_file()}
    first, markers = _run(root, arguments, "first")
    second, more_markers = _run(root, arguments, "second")
    assert first == second
    assert first["code"] == (0 if mode in ("help", "empty", "cached") else 2)
    if mode in ("empty", "cached"):
        plan = json.loads(first["stdout"])
        assert plan["plan_only"] is True
        assert plan["minimum_initial_calls"] == (0 if mode == "cached" else 3)
        if mode == "cached":
            assert all(unit["raw_receipt_id"] for unit in plan["units"])
            assert plan["maximum_http_attempts"] == 0
    if mode == "missing":
        assert json.loads(first["stdout"]) == {"refused": "RESOURCE_UNAVAILABLE"}
        assert not (root / "missing.sqlite").exists()
    if mode == "s3":
        assert json.loads(first["stdout"]) == {
            "refused": "INVALID_REQUEST", "reason": "CAPABILITY_UNAVAILABLE"}
    assert tuple(conn.iterdump()) == before
    assert objects == {str(path.relative_to(root)): path.read_bytes()
                       for path in (root / "store").rglob("*") if path.is_file()}
    assert all(marker not in str(before) + str(objects) for marker in markers + more_markers)
