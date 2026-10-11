"""tests/shards.toml: every test file is in exactly one CI shard (mirrors the
mutation module-ownership test in test_mutation_ci.py), the loader rejects bad
declarations, and tools/ci_shards.py prints the matrix and per-shard files."""
from __future__ import annotations

# packages: ops

import fnmatch
import json
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


@pytest.mark.parametrize("text", [
    '[[shard]]\nname="a"\npatterns=["x"]\n',
    '[[shard]]\nname="a"\nremainder=true\n[[shard]]\nname="b"\npatterns=["x"]\n',
    '[[shard]]\nname="a"\npatterns=["x"]\n[[shard]]\nname="a"\nremainder=true\n',
    '[[shard]]\nname="a"\nremainder=true\n[[shard]]\nname="b"\nremainder=true\n',
    '[[shard]]\nname="a"\n[[shard]]\nname="r"\nremainder=true\n',
    '[[shard]]\nname="a"\npatterns=["x"]\nremainder=true\n',
    '[[shard]]\npatterns=["x"]\n[[shard]]\nname="r"\nremainder=true\n',
    'shard=["bad"]\n',
    'shard="bad"\n',
    '[[shard]]\nname="a"\npatterns="x"\n[[shard]]\nname="r"\nremainder=true\n',
    '[[shard]]\nname="a"\npatterns=["x"]\n[[shard]]\nname="r"\nremainder="false"\n',
    '[[shard]]\nname="a"\npatterns=["x"]\nextra=1\n[[shard]]\nname="r"\nremainder=true\n',
])
def test_loader_rejects_a_bad_declaration(tmp_path, text):
    with pytest.raises(ValueError):
        ci_shards.load_shards(_toml(tmp_path, text))


def test_matrix_lists_every_shard_name(capsys):
    assert ci_shards.main(["matrix"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data == {"include": [{"shard": s.name} for s in REAL_SHARDS]}


def test_files_prints_the_shard_files(capsys):
    name = REAL_SHARDS[0].name
    assert ci_shards.main(["files", name]) == 0
    assert capsys.readouterr().out.split() == ci_shards.assign(REAL_FILES, REAL_SHARDS)[name]


def test_unknown_shard_exits_2(capsys):
    assert ci_shards.main(["files", "nope"]) == 2
    assert "unknown shard" in capsys.readouterr().err


def test_selection_all_empty_and_subset(tmp_path, capsys):
    parts = ci_shards.assign(REAL_FILES, REAL_SHARDS)
    first, last = REAL_SHARDS[0].name, REAL_SHARDS[-1].name
    sel = tmp_path / "sel.txt"
    sel.write_text("__ALL__\n")
    assert ci_shards.main(["files", first, "--selection", str(sel)]) == 0
    assert capsys.readouterr().out.split() == parts[first]
    sel.write_text("")
    assert ci_shards.main(["files", first, "--selection", str(sel)]) == 0
    assert capsys.readouterr().out == ""
    picked = [parts[first][0], parts[last][0]]
    sel.write_text("\n".join(picked) + "\n")
    assert ci_shards.main(["files", first, "--selection", str(sel)]) == 0
    assert capsys.readouterr().out.split() == [picked[0]]
    assert ci_shards.main(["files", last, "--selection", str(sel)]) == 0
    assert capsys.readouterr().out.split() == [picked[1]]