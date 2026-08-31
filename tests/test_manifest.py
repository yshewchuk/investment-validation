"""The manifest and the snapshot hash.

The snapshot hash is the identifier every report's provenance block pins. It has
one job: answer "was this produced from the same data?" — which means it must
change when the data changes and not otherwise.
"""
from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest


@pytest.fixture
def modules(tmp_root):
    from engine.data import manifest as manifest_mod
    from engine.data import store as store_mod

    importlib.reload(store_mod)
    importlib.reload(manifest_mod)
    return manifest_mod, store_mod


def make_chains(bid=1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "obs_date": pd.to_datetime(["2024-05-01", "2024-05-01"]),
            "year": [2024, 2024],
            "expiry": pd.to_datetime(["2024-05-17", "2024-05-17"]),
            "dte": [16, 16],
            "strike": [100.0, 105.0],
            "right": ["C", "P"],
            "bid": [bid, 1.0],
            "ask": [1.4, 1.4],
        }
    )


class TestSnapshotHash:
    def test_it_is_stable_across_recomputation(self, modules):
        manifest, store = modules
        store.write_table(make_chains(), "option_chains")
        assert manifest.snapshot_hash() == manifest.snapshot_hash()

    def test_it_changes_when_the_data_changes(self, modules):
        manifest, store = modules
        store.write_table(make_chains(bid=1.0), "option_chains")
        before = manifest.snapshot_hash()
        store.write_table(make_chains(bid=1.2), "option_chains")
        assert manifest.snapshot_hash() != before

    def test_it_does_not_change_on_a_rewrite_of_identical_data(self, modules):
        manifest, store = modules
        frame = make_chains()
        store.write_table(frame, "option_chains")
        before = manifest.snapshot_hash()
        store.write_table(frame, "option_chains")
        assert manifest.snapshot_hash() == before

    def test_it_covers_every_table_not_just_one(self, modules):
        manifest, store = modules
        store.write_table(make_chains(), "option_chains")
        before = manifest.snapshot_hash()
        store.write_table(
            pd.DataFrame(
                {
                    "event_id": ["AAA_2024-05-08"],
                    "ticker": ["AAA"],
                    "event_date": pd.to_datetime(["2024-05-08"]),
                    "year": [2024],
                    "src_orats": [True],
                    "src_oquants": [True],
                    "src_nasdaq": [False],
                    "src_yfinance": [False],
                    "date_agree": [True],
                    "date_conflict": [False],
                }
            ),
            "earnings_events",
        )
        assert manifest.snapshot_hash() != before

    def test_an_empty_store_still_hashes(self, modules):
        manifest, _ = modules
        assert len(manifest.snapshot_hash()) == 64


class TestSnapshotFile:
    def test_it_records_the_hash_and_the_tables(self, modules):
        manifest, store = modules
        store.write_table(make_chains(), "option_chains")
        digest = manifest.write_snapshot()
        payload = manifest.read_snapshot()
        assert payload["snapshot"] == digest
        assert "option_chains" in payload["tables"]
        assert payload["format"] in ("parquet", "csv.gz")

    def test_a_missing_snapshot_reads_as_none(self, modules):
        manifest, _ = modules
        assert manifest.read_snapshot() is None

    def test_a_corrupt_snapshot_reads_as_none_rather_than_raising(self, modules):
        manifest, _ = modules
        from engine import paths

        paths.SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        paths.SNAPSHOT_FILE.write_text("{not json")
        assert manifest.read_snapshot() is None


class TestManifestRendering:
    def test_it_declares_itself_generated(self, modules):
        manifest, store = modules
        store.write_table(make_chains(), "option_chains")
        body = manifest.write_manifest().read_text()
        assert "do not edit by hand" in body.lower()

    def test_it_reports_the_row_counts_it_measured(self, modules):
        manifest, store = modules
        store.write_table(make_chains(), "option_chains")
        body = manifest.write_manifest().read_text()
        assert "option_chains" in body
        assert manifest.snapshot_hash() in body

    def test_it_carries_the_conventions_and_source_priority(self, modules):
        # These are decisions, not facts. The manifest is where a future reader
        # finds out that ORATS mktCap has three unit eras.
        manifest, store = modules
        store.write_table(make_chains(), "option_chains")
        body = manifest.write_manifest().read_text()
        assert "orats_mktcap_units" in body
        assert "Source priority" in body

    def test_extra_sections_are_appended(self, modules):
        manifest, store = modules
        store.write_table(make_chains(), "option_chains")
        body = manifest.write_manifest(
            extra_sections={"Custom Section": "some content"}
        ).read_text()
        assert "## Custom Section" in body
        assert "some content" in body

    def test_a_clean_store_says_so(self, modules):
        manifest, store = modules
        store.write_table(make_chains(), "option_chains")
        assert "No quarantine flags" in manifest.write_manifest().read_text()

    def test_quarantine_flags_are_surfaced(self, modules):
        manifest, store = modules
        from engine.data.validate import quarantine

        store.write_table(make_chains(), "option_chains")
        quarantine("2024-05-01_b0.json.gz", "crossed everything")
        body = manifest.write_manifest().read_text()
        assert "2024-05-01_b0.json.gz" in body
        assert "crossed everything" in body
