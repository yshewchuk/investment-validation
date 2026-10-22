"""tools/bounded_run.py disk-used ceiling (#1119): synthetic statvfs only.

``_statvfs`` is monkeypatched everywhere a filesystem would be read, so no
test touches a real disk; children are tiny (``sleep``/``true``) and
coordination state goes to ``tmp_path``, never the shared /tmp state dir.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

_spec = importlib.util.spec_from_file_location(
    "bounded_run", ROOT / "tools" / "bounded_run.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

_rm_spec = importlib.util.spec_from_file_location(
    "v2_resource_measurement", ROOT / "tools" / "v2_resource_measurement.py")
rm = importlib.util.module_from_spec(_rm_spec)
_rm_spec.loader.exec_module(rm)

GIB = 1024**3


@pytest.fixture(autouse=True)
def _isolated_state_dir(tmp_path, monkeypatch):
    """Never touch the shared /tmp state dir or an ambient ceiling env var."""
    monkeypatch.setenv("BOUNDED_RUN_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("BOUNDED_RUN_NESTED", raising=False)
    monkeypatch.delenv(mod.DISK_ENV, raising=False)


def _st(f_blocks: int, f_bfree: int, f_frsize: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(f_blocks=f_blocks, f_bfree=f_bfree,
                                 f_frsize=f_frsize)


def _parse(argv: list[str]):
    parser = mod._build_parser()
    args = parser.parse_args(argv)
    mod._validate(args, parser)
    return args


def _run_argv(monkeypatch, *flags: str, command: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["bounded_run.py", "--cores", "1",
                                      "--min-free-gb", "0",
                                      "--max-swap-gb", "-1",
                                      "--poll-s", "0.05", *flags,
                                      "--", *command])


# ---------------------------------------------------------------- validation

def test_ceiling_off_by_default():
    args = _parse([])
    assert args.max_disk_used_gb is None
    assert args.disk_path == []


def test_env_var_used_when_flag_absent(monkeypatch):
    monkeypatch.setenv(mod.DISK_ENV, "512.5")
    assert _parse([]).max_disk_used_gb == 512.5


def test_flag_overrides_env(monkeypatch):
    monkeypatch.setenv(mod.DISK_ENV, "512.5")
    assert _parse(["--max-disk-used-gb", "7"]).max_disk_used_gb == 7.0


@pytest.mark.parametrize("raw", ["abc", "0", "-2", "nan", "inf", "", "  "])
def test_bad_env_value_is_usage_error(monkeypatch, raw, capsys):
    monkeypatch.setenv(mod.DISK_ENV, raw)
    parser = mod._build_parser()
    args = parser.parse_args([])
    with pytest.raises(SystemExit) as excinfo:
        mod._validate(args, parser)
    assert excinfo.value.code == 2
    assert ("BOUNDED_RUN_MAX_DISK_USED_GB must be a positive finite number"
            in capsys.readouterr().err)


@pytest.mark.parametrize("raw", ["0", "-2", "nan", "inf"])
def test_bad_flag_value_is_usage_error(raw, capsys):
    parser = mod._build_parser()
    args = parser.parse_args(["--max-disk-used-gb", raw])
    with pytest.raises(SystemExit) as excinfo:
        mod._validate(args, parser)
    assert excinfo.value.code == 2
    assert ("--max-disk-used-gb must be a positive finite number"
            in capsys.readouterr().err)


def test_more_than_the_path_cap_is_usage_error(capsys):
    paths = [x for i in range(mod.DISK_PATHS_MAX + 1)
             for x in ("--disk-path", f"/p{i}")]
    parser = mod._build_parser()
    args = parser.parse_args(paths)
    with pytest.raises(SystemExit) as excinfo:
        mod._validate(args, parser)
    assert excinfo.value.code == 2
    assert "at most 8" in capsys.readouterr().err


def test_path_cap_itself_is_accepted():
    paths = [x for i in range(mod.DISK_PATHS_MAX)
             for x in ("--disk-path", f"/p{i}")]
    args = _parse(paths)
    assert len(args.disk_path) == mod.DISK_PATHS_MAX


# ------------------------------------------------------------------- helpers

def test_used_gb_math(monkeypatch):
    monkeypatch.setattr(mod, "_statvfs",
                        lambda path: _st(1000, 250, 2048))
    assert mod._disk_used_gb("/x") == pytest.approx((1000 - 250) * 2048 / GIB)


def test_disk_sample_returns_the_worst_target(monkeypatch):
    values = {"/small": _st(100, 90, GIB), "/big": _st(100, 10, GIB)}
    monkeypatch.setattr(mod, "_statvfs", lambda path: values[path])
    used, path = mod._disk_sample([("/small", 1), ("/big", 2)])
    assert used == pytest.approx(90.0)
    assert path == "/big"


def test_disk_targets_dedupe_by_device(monkeypatch):
    devs = {"/cwd": 4, "/a": 4, "/b": 7, "/c": 4}
    monkeypatch.setattr(mod.os, "getcwd", lambda: "/cwd")
    monkeypatch.setattr(mod.os, "stat",
                        lambda p: types.SimpleNamespace(st_dev=devs[p]))
    assert mod._disk_targets(["/a", "/b", "/c"]) == [("/cwd", 4), ("/b", 7)]


def test_disk_targets_dedupe_repeats_of_the_same_path():
    cwd = os.getcwd()
    assert [p for p, _dev in mod._disk_targets([cwd, cwd])] == [cwd]


# --------------------------------------------------------------- _check_disk

def test_check_disk_over_ceiling_kills(monkeypatch):
    kills = []
    monkeypatch.setattr(mod, "_terminate_with_grace",
                        lambda proc, grace, poll, floor: kills.append(
                            (proc, grace, poll, floor)))
    monkeypatch.setattr(mod, "_statvfs", lambda path: _st(1000, 100, GIB))
    args = _parse(["--max-disk-used-gb", "500", "--min-free-gb", "2"])
    args.disk_guard = mod._DiskGuard(args.max_disk_used_gb, [("/data", 7)],
                                     heavy=False)
    assert mod._check_disk("PROC", args, elapsed=60.0) is True
    assert kills == [("PROC", args.kill_grace_s, args.poll_s, 2048.0)]


def test_check_disk_under_ceiling_records_and_does_not_kill(monkeypatch):
    monkeypatch.setattr(mod, "_statvfs", lambda path: _st(1000, 999, GIB))
    args = _parse(["--max-disk-used-gb", "500"])
    guard = mod._DiskGuard(args.max_disk_used_gb, [("/data", 7)], heavy=False)
    args.disk_guard = guard
    assert mod._check_disk("PROC", args, elapsed=1.0) is False
    assert guard.last_used_gb == pytest.approx(1.0)
    assert guard.fails == 0


def test_check_disk_exactly_at_ceiling_does_not_kill(monkeypatch):
    kills = []
    monkeypatch.setattr(mod, "_terminate_with_grace",
                        lambda *a: kills.append(a))
    monkeypatch.setattr(mod, "_statvfs", lambda path: _st(500, 0, GIB))
    args = _parse(["--max-disk-used-gb", "500"])
    guard = mod._DiskGuard(args.max_disk_used_gb, [("/data", 7)], heavy=False)
    args.disk_guard = guard
    assert mod._check_disk("PROC", args, elapsed=1.0) is False
    assert kills == []
    assert guard.last_used_gb == pytest.approx(500.0)


def test_check_disk_fail_counter_resets_on_good_sample(monkeypatch):
    kills = []
    monkeypatch.setattr(mod, "_terminate_with_grace",
                        lambda *a: kills.append(a))
    good = _st(1000, 999, GIB)
    seq: list = []

    def fake(path):
        action = seq.pop(0)
        if isinstance(action, OSError):
            raise action
        return action

    monkeypatch.setattr(mod, "_statvfs", fake)
    args = _parse(["--max-disk-used-gb", "500"])
    guard = mod._DiskGuard(args.max_disk_used_gb, [("/x", 1)], heavy=True)
    args.disk_guard = guard
    errs = [OSError("e") for _ in range(4)]
    seq.extend(errs)
    for _ in range(4):
        assert mod._check_disk("P", args, elapsed=1.0) is False
    assert guard.fails == 4
    seq.append(good)
    assert mod._check_disk("P", args, elapsed=1.0) is False
    assert guard.fails == 0
    seq.extend(errs)
    for _ in range(4):
        assert mod._check_disk("P", args, elapsed=1.0) is False
    assert kills == []
    seq.append(OSError("boom"))
    assert mod._check_disk("P", args, elapsed=1.0) is True
    assert len(kills) == 1


# ------------------------------------------------------------------ main()

def test_watch_level_breach_exits_76_and_kills_the_tree(monkeypatch, capsys):
    monkeypatch.setattr(mod, "_statvfs", lambda path: _st(1000, 100, GIB))
    _run_argv(monkeypatch, "--max-disk-used-gb", "500",
              command=["sleep", "30"])
    assert mod.main() == mod.DISK_LIMIT_EXIT == 76
    captured = capsys.readouterr()
    assert " disk-cap=500G" in captured.out
    assert "used 900.0G > ceiling 500G" in captured.out
    assert os.getcwd() in captured.out
    assert "killed at the disk limit" in captured.out


def test_heavy_refuses_to_start_when_statvfs_unreadable(tmp_path, monkeypatch,
                                                        capsys):
    def boom(path):
        raise OSError("filesystem gone")

    spawned = []
    monkeypatch.setattr(mod, "_statvfs", boom)
    monkeypatch.setattr(mod, "_spawn", lambda argv, env: spawned.append(argv))
    _run_argv(monkeypatch, "--max-disk-used-gb", "500", "--heavy",
              command=["sleep", "30"])
    assert mod.main() == mod.DISK_UNREADABLE_EXIT == 69
    assert spawned == []
    assert not list(tmp_path.glob("heavy-*.json"))
    captured = capsys.readouterr()
    assert "DISK LIMIT unreadable" in captured.err
    assert "refusing to start --heavy" in captured.err


def test_light_run_survives_unreadable_statvfs_with_the_check_off(
        monkeypatch, capsys):
    calls = {"n": 0}

    def boom(path):
        calls["n"] += 1
        raise OSError("filesystem gone")

    monkeypatch.setattr(mod, "_statvfs", boom)
    _run_argv(monkeypatch, "--max-disk-used-gb", "500", command=["true"])
    assert mod.main() == 0
    assert calls["n"] == 1
    captured = capsys.readouterr()
    assert "running with the disk check off" in captured.err
    assert "disk-cap" not in captured.out


def test_heavy_fails_closed_after_five_unreadable_ticks(monkeypatch, capsys):
    calls = {"n": 0}

    def fake(path):
        calls["n"] += 1
        if calls["n"] == 1:
            return _st(1000, 999, GIB)
        raise OSError("filesystem gone")

    monkeypatch.setattr(mod, "_statvfs", fake)
    _run_argv(monkeypatch, "--max-disk-used-gb", "500", "--heavy",
              command=["sleep", "30"])
    assert mod.main() == 76
    captured = capsys.readouterr()
    assert "DISK LIMIT unreadable for 5 ticks" in captured.out
    assert "killed at the disk limit" in captured.out
    assert captured.err.count("DISK LIMIT unreadable") == 1


def test_heavy_single_unreadable_tick_does_not_kill(monkeypatch, capsys):
    calls = {"n": 0}

    def fake(path):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("blip")
        return _st(1000, 999, GIB)

    monkeypatch.setattr(mod, "_statvfs", fake)
    _run_argv(monkeypatch, "--max-disk-used-gb", "500", "--heavy",
              command=["sleep", "0.3"])
    assert mod.main() == 0
    assert calls["n"] >= 3
    captured = capsys.readouterr()
    assert "killed at the disk limit" not in captured.out
    assert captured.err.count("DISK LIMIT unreadable") == 1
    assert "disk used 1.0G/500G" in captured.out


# -------------------------------------------------- kill classification, exit 76

def test_disk_limit_exit_76_classifies_and_other_exits_do_not():
    assert rm.classify_kill("[watchdog] 1.0m DISK LIMIT: /x used 600.0G"
                            " > ceiling 500G", 76) == "DISK LIMIT"
    assert rm.classify_kill("DISK LIMIT", 1) is None
    assert rm.classify_kill("CAP BREACH", 137) == "CAP BREACH"
