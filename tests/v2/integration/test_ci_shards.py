"""tests/shards.toml: every test file is in exactly one CI shard (mirrors the
mutation module-ownership test in test_mutation_ci.py), the loader rejects bad
declarations, and tools/ci_shards.py prints the matrix and per-shard files."""
from __future__ import annotations

# packages: ops

import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

import ci_shards  # noqa: E402

REAL_FILES = ci_shards.discover_test_files()
REAL_SHARDS = ci_shards.load_shards()


def _toml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "shards.toml"
    path.write_text(text)
    return path


def test_discovery_matches_the_git_tracked_test_inventory():
    # Independent oracle: git's own file list, not the discovery being checked.
    tracked = subprocess.run(
        ["git", "ls-files", "tests/**/*.py", "tests/*.py"],
        cwd=ROOT, capture_output=True, text=True, check=True, timeout=60,
    ).stdout.split()
    tracked = [p for p in tracked
               if Path(p).name.startswith("test_") or Path(p).name.endswith("_test.py")]
    if not tracked:
        pytest.skip("not a git checkout")
    assert set(REAL_FILES) >= set(tracked), sorted(set(tracked) - set(REAL_FILES))


def test_every_test_file_is_in_exactly_one_shard():
    parts = ci_shards.assign(REAL_FILES, REAL_SHARDS)
    flat = [f for files in parts.values() for f in files]
    assert sorted(flat) == REAL_FILES
    assert len(flat) == len(set(flat))
    assert REAL_FILES, "no test files discovered"


def test_every_explicit_file_in_research_a_and_research_b_exists():
    by_name = {s.name: s for s in REAL_SHARDS}
    for name in ("research-a", "research-b"):
        for pattern in by_name[name].patterns:
            assert (ROOT / pattern).is_file(), f"{name}: {pattern}"


def _first_unclaimed_pattern(shards, files):
    """The first (shard, pattern) that claims no file an earlier pattern left over."""
    claimed: set[str] = set()
    for shard in shards[:-1]:
        for pat in shard.patterns:
            fresh = {f for f in files if fnmatch.fnmatchcase(f, pat)} - claimed
            if not fresh:
                return shard.name, pat
            claimed |= fresh
    return None


def test_every_declared_pattern_claims_a_file_nothing_earlier_claimed():
    assert _first_unclaimed_pattern(REAL_SHARDS, REAL_FILES) is None


def test_a_duplicate_or_shadowed_pattern_inside_one_shard_is_found(tmp_path):
    files = ["tests/a/test_x.py", "tests/b/test_y.py"]
    dup = _toml(tmp_path, '[[shard]]\nname="a"\npatterns=["tests/a/*", "tests/a/*"]\n'
                          '[[shard]]\nname="r"\nremainder=true\n')
    assert _first_unclaimed_pattern(ci_shards.load_shards(dup), files) == ("a", "tests/a/*")
    shadow = _toml(tmp_path, '[[shard]]\nname="a"\npatterns=["tests/*", "tests/a/*"]\n'
                             '[[shard]]\nname="r"\nremainder=true\n')
    assert _first_unclaimed_pattern(ci_shards.load_shards(shadow), files) == ("a", "tests/a/*")


def test_a_pattern_duplicated_or_shadowed_across_shards_is_found(tmp_path):
    files = ["tests/a/test_x.py", "tests/x/test_y.py"]
    dup = _toml(tmp_path, '[[shard]]\nname="a"\npatterns=["tests/a/*"]\n'
                          '[[shard]]\nname="b"\npatterns=["tests/a/*", "tests/x/*"]\n'
                          '[[shard]]\nname="r"\nremainder=true\n')
    assert _first_unclaimed_pattern(ci_shards.load_shards(dup), files) == ("b", "tests/a/*")
    shadow = _toml(tmp_path, '[[shard]]\nname="a"\npatterns=["tests/*"]\n'
                             '[[shard]]\nname="b"\npatterns=["tests/x/*"]\n'
                             '[[shard]]\nname="r"\nremainder=true\n')
    assert _first_unclaimed_pattern(ci_shards.load_shards(shadow), files) == ("b", "tests/x/*")
    clean = _toml(tmp_path, '[[shard]]\nname="a"\npatterns=["tests/a/*"]\n'
                            '[[shard]]\nname="b"\npatterns=["tests/x/*"]\n'
                            '[[shard]]\nname="r"\nremainder=true\n')
    assert _first_unclaimed_pattern(ci_shards.load_shards(clean), files) is None


def test_discovery_finds_both_pytest_naming_conventions(tmp_path):
    for rel in ("tests/test_a.py", "tests/v2/x/b_test.py", "tests/v2/x/helper.py",
                "tests/__pycache__/test_c.py"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    assert ci_shards.discover_test_files(tmp_path) == ["tests/test_a.py", "tests/v2/x/b_test.py"]


def test_selection_ignores_unknown_paths_and_directories(tmp_path, capsys):
    sel = tmp_path / "sel.txt"
    sel.write_text("tests/nope/test_missing.py\ntests/v2\n")
    for shard in REAL_SHARDS:
        assert ci_shards.main(["files", shard.name, "--selection", str(sel)]) == 0
        assert capsys.readouterr().out == ""


def test_every_shard_owns_at_least_one_file():
    for name, files in ci_shards.assign(REAL_FILES, REAL_SHARDS).items():
        assert files, f"shard {name!r} is empty"


def test_unclaimed_file_goes_to_the_remainder_shard():
    assert ci_shards.owner("tests/v2/brand_new_pkg/test_x.py", REAL_SHARDS) == REAL_SHARDS[-1].name


def test_first_matching_shard_owns_a_file_two_shards_match(tmp_path):
    path = _toml(tmp_path, '[[shard]]\nname="a"\npatterns=["tests/*"]\n'
                           '[[shard]]\nname="b"\npatterns=["tests/x/*"]\n'
                           '[[shard]]\nname="r"\nremainder=true\n')
    shards = ci_shards.load_shards(path)
    assert ci_shards.owner("tests/x/test_y.py", shards) == "a"


R = '[[shard]]\nname="r"\nremainder=true\n'


@pytest.mark.parametrize(("text", "message"), [
    ('[[shard]]\nname="a"\npatterns=["x"]\n', "exactly one remainder shard is required"),
    ('[[shard]]\nname="a"\nremainder=true\n[[shard]]\nname="b"\npatterns=["x"]\n',
     "exactly one remainder shard is required"),
    ('[[shard]]\nname="a"\npatterns=["x"]\n[[shard]]\nname="a"\nremainder=true\n',
     "duplicate shard names"),
    ('[[shard]]\nname="a"\nremainder=true\n[[shard]]\nname="b"\nremainder=true\n',
     "exactly one remainder shard is required"),
    ('[[shard]]\nname="a"\n' + R, "shard 'a' has no patterns and is not the remainder"),
    ('[[shard]]\nname="a"\npatterns=["x"]\nremainder=true\n',
     "remainder shard 'a' must not list patterns"),
    ('[[shard]]\npatterns=["x"]\n' + R, "shard #0 has no name"),
    ('shard=["bad"]\n', "shard #0 must be a table"),
    ('shard="bad"\n', "`shard` must be an array"),
    ('[[shard]]\nname="a"\npatterns="x"\n' + R, "shard 'a': patterns must be a list of strings"),
    ('[[shard]]\nname="a"\npatterns=["x"]\n[[shard]]\nname="r"\nremainder="false"\n',
     "shard 'r': remainder must be a boolean"),
    ('[[shard]]\nname="a"\npatterns=["x"]\nextra=1\n' + R, "shard #0 has unknown keys: ['extra']"),
    ('[[shard]]\nname=1\npatterns=["x"]\n' + R, "shard #0 has no name"),
    ('[[shard]]\nname=""\npatterns=["x"]\n' + R, "shard #0 has no name"),
    ('[[shard]]\nname="a"\npatterns=[1]\n' + R, "shard 'a': patterns must be a list of strings"),
    ('[[shard]]\nname="a"\npatterns=["x", 2]\n' + R,
     "shard 'a': patterns must be a list of strings"),
])
def test_loader_rejects_a_bad_declaration_with_the_specific_refusal(tmp_path, text, message):
    with pytest.raises(ValueError) as excinfo:
        ci_shards.load_shards(_toml(tmp_path, text))
    assert message in str(excinfo.value)


def test_ownership_matches_an_independently_written_table(tmp_path):
    shards = ci_shards.load_shards(_toml(
        tmp_path,
        '[[shard]]\nname="a"\npatterns=["tests/a/*", "tests/shared/*"]\n'
        '[[shard]]\nname="b"\npatterns=["tests/b/*", "tests/shared/*", "tests/*_x.py"]\n' + R))
    expected = {
        "tests/a/test_1.py": "a",
        "tests/shared/test_2.py": "a",  # second pattern of a, also matched by b: a is first
        "tests/b/test_3.py": "b",
        "tests/test_q_x.py": "b",  # third pattern of b
        "tests/c/test_4.py": "r",  # unclaimed: remainder
    }
    for path, shard in expected.items():
        assert ci_shards.owner(path, shards) == shard, path
    assert ci_shards.assign(list(expected), shards) == {
        "a": ["tests/a/test_1.py", "tests/shared/test_2.py"],
        "b": ["tests/b/test_3.py", "tests/test_q_x.py"],
        "r": ["tests/c/test_4.py"],
    }


@pytest.mark.parametrize(("path", "shard"), [
    ("tests/v2/ops/test_a.py", "ops"),
    ("tests/v2/ops/deep/test_a.py", "ops"),
    ("tests/test_v2_ops_foo.py", "ops"),
    ("tests/v2/ops/test_gate_sweep.py", "ops-sweep"),
    ("tests/v2/ops/test_native_gate_worker.py", "ops-sweep"),
    ("tests/v2/ops/test_computed_moves_bounded_retention.py", "ops-sweep"),
    ("tests/v2/research/test_a.py", "research"),
    ("tests/v2/research/test_gate_report.py", "research-a"),
    ("tests/v2/research/test_magnitude_history_model.py", "research-b"),
    ("tests/v2/integration/test_a.py", "integration"),
    ("tests/test_v2_research_foo.py", "research"),
    ("tests/test_checks_phase2_gate.py", "checks"),
    ("tests/test_v2_data_foo.py", "rest"),
    ("tests/v2/serving/test_a.py", "rest"),
    ("tests/test_audit.py", "rest"),
])
def test_real_declaration_owns_representative_paths_as_documented(path, shard):
    assert ci_shards.owner(path, REAL_SHARDS) == shard


def test_matrix_lists_every_shard_name(capsys):
    assert ci_shards.main(["matrix"]) == 0
    expected = ",".join('{"shard":"%s"}' % s.name for s in REAL_SHARDS)
    assert capsys.readouterr().out == '{"include":[' + expected + ']}\n'


def test_files_prints_the_shard_files(capsys):
    name = REAL_SHARDS[0].name
    assert ci_shards.main(["files", name]) == 0
    expected = ci_shards.assign(REAL_FILES, REAL_SHARDS)[name]
    assert capsys.readouterr().out == "".join(f + "\n" for f in expected)


def test_selection_lines_are_stripped_and_blank_lines_ignored(tmp_path, capsys):
    shard = REAL_SHARDS[-1].name
    target = ci_shards.assign(REAL_FILES, REAL_SHARDS)[shard][0]
    sel = tmp_path / "sel.txt"
    sel.write_text(f"\n   \n  {target}  \n\n")
    assert ci_shards.main(["files", shard, "--selection", str(sel)]) == 0
    assert capsys.readouterr().out == target + "\n"


def test_all_marker_only_counts_as_the_first_selection_line(tmp_path, capsys):
    shard = REAL_SHARDS[-1].name
    sel = tmp_path / "sel.txt"
    sel.write_text("tests/nope/test_x.py\n__ALL__\n")
    assert ci_shards.main(["files", shard, "--selection", str(sel)]) == 0
    assert capsys.readouterr().out == ""


def _cli(args, root):
    env = dict(os.environ, INVESTING_PLAN_ROOT=str(root))
    return subprocess.run([sys.executable, str(ROOT / "tools" / "ci_shards.py"), *args],
                          env=env, capture_output=True, text=True, timeout=60, cwd=ROOT)


def test_investing_plan_root_override_is_where_the_cli_reads_shards_and_tests(tmp_path):
    (tmp_path / "tests" / "pkg").mkdir(parents=True)
    (tmp_path / "tests" / "shards.toml").write_text(
        '[[shard]]\nname="only_pkg"\npatterns=["tests/pkg/*"]\n'
        '[[shard]]\nname="everything_else"\nremainder=true\n')
    for rel in ("tests/pkg/test_b.py", "tests/test_a.py", "tests/pkg/c_test.py"):
        (tmp_path / rel).write_text("")
    matrix = _cli(["matrix"], tmp_path)
    assert matrix.returncode == 0, matrix.stderr
    assert matrix.stdout == '{"include":[{"shard":"only_pkg"},{"shard":"everything_else"}]}\n'
    pkg = _cli(["files", "only_pkg"], tmp_path)
    assert pkg.stdout == "tests/pkg/c_test.py\ntests/pkg/test_b.py\n"
    rest = _cli(["files", "everything_else"], tmp_path)
    assert rest.stdout == "tests/test_a.py\n"
    # Without the override the same CLI reads the real checkout's declaration.
    real = subprocess.run([sys.executable, str(ROOT / "tools" / "ci_shards.py"), "matrix"],
                          env={k: v for k, v in os.environ.items() if k != "INVESTING_PLAN_ROOT"},
                          capture_output=True, text=True, timeout=60, cwd=ROOT)
    names = ",".join('{"shard":"%s"}' % s.name for s in REAL_SHARDS)
    assert real.stdout == '{"include":[' + names + ']}\n'


def test_pr_selection_universe_matches_shard_discovery_on_both_naming_conventions():
    import mutation_pilot as pilot

    tracked = {"tests/a/test_x.py", "tests/a/y_test.py", "tests/a/helper.py",
               "tests/conftest.py", "tests/a/test_w.txt", "engine/test_z.py", "engine/z_test.py"}
    assert pilot.pytest_test_files(tracked) == ["tests/a/test_x.py", "tests/a/y_test.py"]
    assert set(pilot.pytest_test_files(set(REAL_FILES))) == set(REAL_FILES)


def test_unknown_shard_exits_2(capsys):
    assert ci_shards.main(["files", "nope"]) == 2
    assert capsys.readouterr().err == "unknown shard 'nope'\n"


def test_selection_all_empty_and_subset(tmp_path, capsys):
    parts = ci_shards.assign(REAL_FILES, REAL_SHARDS)
    first, last = REAL_SHARDS[0].name, REAL_SHARDS[-1].name
    sel = tmp_path / "sel.txt"
    sel.write_text("__ALL__\n")
    assert ci_shards.main(["files", first, "--selection", str(sel)]) == 0
    assert capsys.readouterr().out == "".join(f + "\n" for f in parts[first])
    sel.write_text("")
    assert ci_shards.main(["files", first, "--selection", str(sel)]) == 0
    assert capsys.readouterr().out == ""
    picked = [parts[first][0], parts[last][0]]
    sel.write_text("\n".join(picked) + "\n")
    assert ci_shards.main(["files", first, "--selection", str(sel)]) == 0
    assert capsys.readouterr().out == picked[0] + "\n"
    assert ci_shards.main(["files", last, "--selection", str(sel)]) == 0
    assert capsys.readouterr().out == picked[1] + "\n"
