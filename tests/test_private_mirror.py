"""The private-mirror allowlist.

This script decides what leaves the machine for a second remote. Private is not
the same as safe — a credential in any remote is a credential to rotate — so the
secret scan is as load-bearing here as in the public hygiene gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import paths  # noqa: E402
from tools import private_mirror  # noqa: E402
from tools.private_mirror import (  # noqa: E402
    EXCLUDE_PARTS,
    INCLUDE,
    INCLUDE_FILES,
    collect,
    scan_for_secrets,
    sync,
)


class TestAllowlistPolicy:
    def test_the_irreplaceable_trees_are_all_declared(self):
        directories = {directory for directory, _ in INCLUDE}
        for required in ("ledger", "reports", "config/thesis", "earnings_predictions"):
            assert required in directories, f"{required} is not mirrored"

    def test_the_research_findings_docs_are_declared(self):
        for required in ("AGENTS.md", "STRATEGY.md", "ADVISOR_BRIEF.md"):
            assert required in INCLUDE_FILES

    def test_market_data_directories_are_not_mirrored(self):
        # Re-pullable at quota cost; 57k files do not belong in git. The gap is
        # accepted and documented in RECOVERY.md.
        directories = {directory for directory, _ in INCLUDE}
        for excluded in ("data", "polygon_cache", "data/raw", "data/curated"):
            assert excluded not in directories

    def test_only_code_and_doc_globs_are_pulled_from_the_research_trees(self):
        # `earnings_predictions/**` would sweep in 57k data files.
        for directory, pattern in INCLUDE:
            if directory in ("earnings_predictions", "bt", "dashboard"):
                assert pattern.rsplit(".", 1)[-1] in ("py", "md", "html", "css", "js"), (
                    f"{directory}/{pattern} is not restricted to code or docs"
                )

    def test_build_artifacts_are_excluded(self):
        for part in ("__pycache__", ".git", ".venv"):
            assert part in EXCLUDE_PARTS


class TestCollection:
    def test_it_finds_real_files_and_stays_small(self):
        files, _ = collect()
        assert files, "the mirror collected nothing"
        logs = [p for p in files if p.name.startswith("transactions_")]
        rest = sum(p.stat().st_size for p in files if p not in set(logs))
        total = sum(p.stat().st_size for p in files)
        # Docs, code and result artifacts. The transaction logs are budgeted
        # separately because they are legitimately a few MB each; everything
        # else ballooning means a data glob has crept into the allowlist.
        assert rest < 10_000_000, f"non-log payload is {rest:,} bytes — check the allowlist"
        assert total < 40_000_000, f"mirror is {total:,} bytes"

    def test_no_market_data_files_are_collected(self):
        """No market data — but the irreplaceable non-code artifacts DO ship.

        The ban is on re-pullable market data (chain partitions, caches), not
        on the two record files the mirror exists for: the multiple-testing
        ledger and the prediction ledger. Those are append-only records that
        cannot be regenerated at any price, which is the whole reason there is
        a private mirror at all.
        """
        files, _ = collect()
        banned = {".csv", ".parquet", ".gz", ".pkl", ".jsonl", ".sqlite", ".npy"}
        allowed = {
            paths.ROOT / "experiments" / "LEDGER.csv",   # multiple-testing record
        }
        offenders = [
            str(p) for p in files
            if p.suffix.lower() in banned
            and Path(p) not in allowed
            # transactions_*.csv — the per-trade audit trail behind a report's
            # equity curve. Licensed quotes, so public is out; without it a
            # published chart cannot be checked against anything.
            and not p.name.startswith("transactions_")
            # ledger/predictions/*.jsonl + ledger/outcomes/*.jsonl — the
            # out-of-time validator; nothing can regenerate a frozen prediction.
            and paths.LEDGER not in Path(p).parents
        ]
        assert not offenders, f"market-data files in the mirror: {offenders[:5]}"

    def test_no_pycache_is_collected(self):
        files, _ = collect()
        assert not [p for p in files if "__pycache__" in p.parts]

    def test_the_env_file_is_never_collected(self):
        files, _ = collect()
        assert not [p for p in files if p.name == ".env"]


class TestSecretScan:
    def test_the_current_env_tree_scans_clean(self):
        files, _ = collect()
        assert scan_for_secrets(files).ok

    def test_a_planted_secret_is_caught(self, tmp_path):
        from checks.repo_hygiene import check_blob, Report

        report = Report()
        check_blob("x.py", b"KEY = 'super-secret-value-123456'", {b"super-secret-value-123456": "TEST_KEY"}, report)
        assert not report.ok
        assert report.violations[0].rule == "secret"

    def test_data_extensions_do_not_block_the_mirror(self):
        # Reports and ledgers are exactly what this mirror is for, so the
        # public repo's data-extension rule must NOT apply here.
        files, _ = collect()
        report = scan_for_secrets(files)
        assert all(v.rule in ("secret", "credential-shape", "oversize") for v in report.violations)


class TestSync:
    def test_it_copies_files_preserving_layout(self, tmp_path):
        root = private_mirror.ROOT
        files = [root / "README.md"]
        target = tmp_path / "mirror"
        copied = sync(target, files)
        assert copied == 1
        assert (target / "README.md").read_bytes() == (root / "README.md").read_bytes()

    def test_unchanged_files_are_not_recopied(self, tmp_path):
        root = private_mirror.ROOT
        files = [root / "README.md"]
        target = tmp_path / "mirror"
        sync(target, files)
        assert sync(target, files) == 0

    def test_changed_files_are_recopied(self, tmp_path):
        root = private_mirror.ROOT
        files = [root / "README.md"]
        target = tmp_path / "mirror"
        sync(target, files)
        (target / "README.md").write_text("stale")
        assert sync(target, files) == 1
