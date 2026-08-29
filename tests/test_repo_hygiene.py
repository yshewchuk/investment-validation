"""Repo hygiene — the last gate before a public push.

A secret that reaches a public remote is compromised permanently, no matter how
fast the commit is removed. These tests are the reason to trust the hook.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checks.repo_hygiene import (  # noqa: E402
    MAX_BYTES,
    Report,
    check_files,
    load_secrets,
    parse_env,
    secret_needles,
)

SECRET = "sk-live-abcdefghijklmnop1234567890"
ENV_BODY = f"""
# a comment
export ORATS_API_KEY={SECRET}
export OQUANTS_COOKIE_NAME=__Secure-better-auth.session_token
POLYGON_API_KEY="quoted-secret-value-9876543210"
"""


@pytest.fixture
def needles():
    return secret_needles(parse_env(ENV_BODY))


class TestEnvParsing:
    def test_export_and_bare_assignments_both_parse(self):
        env = parse_env(ENV_BODY)
        assert env["ORATS_API_KEY"] == SECRET
        assert env["POLYGON_API_KEY"] == "quoted-secret-value-9876543210"

    def test_quotes_are_stripped(self):
        assert parse_env('A="xyz"')["A"] == "xyz"
        assert parse_env("A='xyz'")["A"] == "xyz"

    def test_comments_and_blanks_are_skipped(self):
        assert parse_env("# nope\n\nA=1") == {"A": "1"}

    def test_an_unparseable_line_does_not_fail_the_whole_file(self):
        # A hygiene check must never fail open because of a formatting surprise.
        assert parse_env("garbage line\nA=1") == {"A": "1"}

    def test_structurally_non_secret_variables_are_not_searched_for(self, needles):
        # The cookie NAME is not a secret; searching for it would flag every
        # file that documents the auth flow, AGENTS.md included.
        assert not any("OQUANTS_COOKIE_NAME" in label for label in needles.values())

    def test_short_values_are_not_searched_for(self):
        # A two-character "secret" would match nearly every file.
        assert secret_needles({"K": "ab"}) == {}


class TestSecretDetection:
    def test_a_pasted_key_blocks_the_commit(self, needles):
        report = check_files({"notes.py": f'KEY = "{SECRET}"'.encode()}, needles)
        assert not report.ok
        assert report.violations[0].rule == "secret"

    def test_the_secret_value_is_never_echoed_in_the_failure(self, needles):
        report = check_files({"notes.py": SECRET.encode()}, needles)
        message = str(report.violations[0])
        assert SECRET not in message
        assert "ORATS_API_KEY" in message  # names the variable, not the value

    def test_a_base64_encoded_key_is_caught(self, needles):
        import base64

        blob = b"payload=" + base64.b64encode(SECRET.encode())
        report = check_files({"cfg.py": blob}, needles)
        assert not report.ok

    def test_a_url_encoded_key_is_caught(self, needles):
        import urllib.parse

        blob = f"https://api/x?token={urllib.parse.quote(SECRET, safe='')}".encode()
        assert not check_files({"u.py": blob}, needles).ok

    def test_the_key_is_caught_in_any_file_type(self, needles):
        for name in ("a.py", "b.md", "c.yaml", "d.html", "notes.txt"):
            assert not check_files({name: SECRET.encode()}, needles).ok

    def test_clean_content_passes(self, needles):
        assert check_files({"a.py": b"import os\nKEY = os.environ['ORATS_API_KEY']\n"}, needles).ok

    def test_credential_shapes_are_caught_without_any_env_file(self):
        # A token pasted from somewhere else entirely, with .env absent.
        #
        # These fixtures are assembled at runtime rather than written as
        # literals, because the checker scans every tracked file including this
        # one — and it is right to. Allowlisting test files to quiet it would
        # mean a genuine secret pasted into a test could reach the remote.
        blobs = {
            "a.py": ("token = 'github_" + "pat_11" + "A" * 22 + "'").encode(),
            "b.py": ("aws = '" + "AKIA" + "IOSFODNN7EXAMPLE" + "'").encode(),
            "c.pem": ("-" * 5 + "BEGIN RSA " + "PRIVATE KEY" + "-" * 5).encode(),
        }
        for name, blob in blobs.items():
            report = check_files({name: blob}, {})
            assert not report.ok, name

    def test_a_missing_env_file_yields_no_needles(self, tmp_path):
        assert load_secrets(tmp_path / "absent.env") == {}


class TestDataAndSizePolicy:
    @pytest.mark.parametrize(
        "name",
        ["trades.csv", "panel.parquet", "chain.json.gz", "x.jsonl", "db.sqlite",
         "model.pkl", "fig.png", "run.log"],
    )
    def test_data_extensions_are_blocked(self, name):
        report = check_files({name: b"x"}, {})
        assert not report.ok
        assert any(v.rule == "data-extension" for v in report.violations)

    def test_an_oversize_file_is_blocked(self):
        report = check_files({"big.py": b"x" * (MAX_BYTES + 1)}, {})
        assert any(v.rule == "oversize" for v in report.violations)

    def test_a_file_at_the_limit_passes(self):
        assert check_files({"ok.py": b"x" * MAX_BYTES}, {}).ok

    @pytest.mark.parametrize(
        "path",
        [
            "data/curated/x.py",
            "ledger/entries.py",
            "earnings_predictions/src/build.py",
            "bt/straddle/x.py",
            "polygon_cache/x.py",
            "config/thesis/universe.py",
            "reports/phase0.py",
        ],
    )
    def test_forbidden_trees_are_blocked_even_for_code(self, path):
        report = check_files({path: b"print(1)"}, {})
        assert not report.ok
        assert any(v.rule == "forbidden-path" for v in report.violations)

    @pytest.mark.parametrize("path", ["a/results/x.py", "b/figures/y.py", "c/cache/z.py"])
    def test_forbidden_directory_segments_are_blocked_anywhere(self, path):
        assert not check_files({path: b"print(1)"}, {}).ok

    def test_the_env_file_itself_is_blocked_by_name(self):
        report = check_files({".env": b""}, {})
        assert any(v.rule == "forbidden-name" for v in report.violations)

    def test_the_env_template_is_allowed(self):
        assert check_files({".env.example": b"export ORATS_API_KEY=\n"}, {}).ok

    def test_ordinary_code_and_docs_pass(self):
        files = {
            "engine/fills.py": b"class FillModel: pass",
            "guides/phase0.md": b"# guide",
            "README.md": b"# readme",
            "spec.yaml": b"hypothesis: x",
        }
        assert check_files(files, {}).ok


class TestReporting:
    def test_every_violation_is_reported_not_just_the_first(self, needles):
        files = {"a.csv": SECRET.encode(), "b.py": b"x" * (MAX_BYTES + 1)}
        report = check_files(files, needles)
        rules = {v.rule for v in report.violations}
        assert {"secret", "data-extension", "oversize"} <= rules

    def test_a_clean_report_is_ok(self):
        report = Report()
        assert report.ok

    def test_checked_count_tracks_the_input(self):
        report = check_files({"a.py": b"1", "b.py": b"2"}, {})
        assert report.checked == 2


class TestCli:
    """The hook has to work in a bare clone, with no engine and no deps."""

    def _run(self, root: Path, *args):
        return subprocess.run(
            [sys.executable, str(root / "checks" / "repo_hygiene.py"), *args],
            capture_output=True,
            text=True,
        )

    @pytest.fixture
    def repo(self, tmp_path):
        root = tmp_path / "repo"
        (root / "checks").mkdir(parents=True)
        source = Path(__file__).resolve().parents[1] / "checks" / "repo_hygiene.py"
        (root / "checks" / "repo_hygiene.py").write_text(source.read_text())
        (root / ".env").write_text(ENV_BODY)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
        return root

    def test_clean_staged_content_exits_zero(self, repo):
        (repo / "ok.py").write_text("print(1)\n")
        subprocess.run(["git", "add", "ok.py"], cwd=repo, check=True)
        assert self._run(repo, "--repo-root", str(repo)).returncode == 0

    def test_a_staged_secret_exits_nonzero(self, repo):
        (repo / "leak.py").write_text(f'K = "{SECRET}"\n')
        subprocess.run(["git", "add", "-f", "leak.py"], cwd=repo, check=True)
        result = self._run(repo, "--repo-root", str(repo))
        assert result.returncode == 1
        assert "ORATS_API_KEY" in result.stderr
        assert SECRET not in result.stderr

    def test_a_staged_data_file_exits_nonzero(self, repo):
        (repo / "trades.csv").write_text("a,b\n1,2\n")
        subprocess.run(["git", "add", "-f", "trades.csv"], cwd=repo, check=True)
        assert self._run(repo, "--repo-root", str(repo)).returncode == 1

    def test_a_staged_oversize_file_exits_nonzero(self, repo):
        (repo / "big.py").write_text("x" * (2 * MAX_BYTES))
        subprocess.run(["git", "add", "-f", "big.py"], cwd=repo, check=True)
        assert self._run(repo, "--repo-root", str(repo)).returncode == 1

    def test_it_warns_when_no_secrets_could_be_loaded(self, repo):
        (repo / ".env").unlink()
        result = self._run(repo, "--repo-root", str(repo))
        assert "WARNING" in result.stdout

    def test_it_checks_the_staged_blob_not_the_worktree(self, repo):
        # Stage clean content, then dirty the worktree. The commit would carry
        # the staged version, so that is what must be checked.
        (repo / "f.py").write_text("clean\n")
        subprocess.run(["git", "add", "f.py"], cwd=repo, check=True)
        (repo / "f.py").write_text(f'K = "{SECRET}"\n')
        assert self._run(repo, "--repo-root", str(repo)).returncode == 0
