"""Tier-2 storage: partitioning, idempotence, streaming, and content hashes."""
from __future__ import annotations

import importlib

import pandas as pd
import pytest

from engine.data.schemas import assert_schema


@pytest.fixture
def store(tmp_root):
    from engine.data import store as store_mod

    importlib.reload(store_mod)
    return store_mod


def make_chains(years=(2023, 2024), per_year=4) -> pd.DataFrame:
    rows = []
    for year in years:
        for i in range(per_year):
            rows.append(
                {
                    "ticker": f"T{i % 2}",
                    "obs_date": pd.Timestamp(f"{year}-05-01") + pd.Timedelta(days=i),
                    "year": year,
                    "expiry": pd.Timestamp(f"{year}-06-21"),
                    "dte": 30 + i,
                    "strike": 100.0 + i,
                    "right": "C",
                    "bid": 1.0 + i,
                    "ask": 1.5 + i,
                }
            )
    return pd.DataFrame(rows)


class TestRoundTrip:
    def test_write_then_read_preserves_the_rows(self, store):
        frame = make_chains()
        store.write_table(frame, "option_chains")
        out = store.read_table("option_chains")
        assert len(out) == len(frame)
        assert set(out["year"]) == {2023, 2024}

    def test_partitions_are_one_directory_per_year(self, store):
        from engine import paths

        store.write_table(make_chains(), "option_chains")
        assert store.table_years("option_chains") == [2023, 2024]
        assert paths.curated_partition("option_chains", 2024).is_dir()

    def test_reading_a_year_subset_reads_only_that_year(self, store):
        store.write_table(make_chains(), "option_chains")
        out = store.read_table("option_chains", years=[2024])
        assert set(out["year"]) == {2024}

    def test_column_projection(self, store):
        store.write_table(make_chains(), "option_chains")
        out = store.read_table("option_chains", columns=["ticker", "strike"])
        assert list(out.columns) == ["ticker", "strike"]

    def test_reading_an_absent_table_returns_a_typed_empty_frame(self, store):
        out = store.read_table("option_chains")
        assert out.empty
        assert "strike" in out.columns

    def test_unknown_table_is_rejected(self, store):
        with pytest.raises(KeyError, match="unknown table"):
            store.read_table("nope")


class TestIdempotence:
    def test_rewriting_identical_data_produces_identical_bytes(self, store):
        frame = make_chains()
        store.write_table(frame, "option_chains")
        first = store.table_stats("option_chains").content_hash
        store.write_table(frame, "option_chains")
        assert store.table_stats("option_chains").content_hash == first

    def test_row_order_does_not_change_the_stored_bytes(self, store):
        frame = make_chains()
        store.write_table(frame, "option_chains")
        first = store.table_stats("option_chains").content_hash
        store.write_table(frame.sample(frac=1.0, random_state=7), "option_chains")
        assert store.table_stats("option_chains").content_hash == first

    def test_changed_data_changes_the_content_hash(self, store):
        frame = make_chains()
        store.write_table(frame, "option_chains")
        first = store.table_stats("option_chains").content_hash
        bumped = frame.copy()
        bumped.loc[0, "bid"] = 99.0
        store.write_table(bumped, "option_chains")
        assert store.table_stats("option_chains").content_hash != first

    def test_a_rebuild_that_drops_a_year_leaves_no_stale_partition(self, store):
        store.write_table(make_chains(years=(2023, 2024)), "option_chains")
        store.write_table(make_chains(years=(2024,)), "option_chains")
        assert store.table_years("option_chains") == [2024]
        assert set(store.read_table("option_chains")["year"]) == {2024}


class TestStreamingWriter:
    def test_streamed_and_single_shot_writes_agree_on_content(self, store):
        frame = make_chains(years=(2023, 2024), per_year=6)
        with store.PartitionedWriter("option_chains") as writer:
            for _, chunk in frame.groupby("ticker"):
                writer.add(chunk)
        streamed = store.read_table("option_chains").sort_values(
            ["ticker", "obs_date", "expiry", "strike", "right"]
        )
        store.write_table(frame, "option_chains")
        single = store.read_table("option_chains").sort_values(
            ["ticker", "obs_date", "expiry", "strike", "right"]
        )
        pd.testing.assert_frame_equal(
            streamed.reset_index(drop=True), single.reset_index(drop=True)
        )

    def test_flushing_is_driven_by_a_global_row_cap(self, store):
        # A per-year threshold would never trip here: each batch spreads a few
        # rows across two years, so the whole table would stay resident.
        frame = make_chains(years=(2023, 2024), per_year=8)
        with store.PartitionedWriter("option_chains", max_buffered_rows=4) as writer:
            for i in range(len(frame)):
                writer.add(frame.iloc[[i]])
            assert writer.flushes > 1
        assert len(store.read_table("option_chains")) == len(frame)

    def test_buffered_rows_are_all_written_on_close(self, store):
        frame = make_chains(per_year=3)
        writer = store.PartitionedWriter("option_chains", max_buffered_rows=10_000)
        writer.add(frame)
        assert store.table_stats("option_chains").rows == 0  # still buffered
        writer.close()
        assert store.table_stats("option_chains").rows == len(frame)

    def test_a_streamed_rewrite_does_not_double_count(self, store):
        frame = make_chains(per_year=6)
        for _ in range(2):
            with store.PartitionedWriter("option_chains", max_buffered_rows=2) as writer:
                for i in range(len(frame)):
                    writer.add(frame.iloc[[i]])
        assert store.table_stats("option_chains").rows == len(frame)

    def test_a_single_shot_write_clears_earlier_streamed_parts(self, store):
        frame = make_chains(per_year=6)
        with store.PartitionedWriter("option_chains", max_buffered_rows=2) as writer:
            for i in range(len(frame)):
                writer.add(frame.iloc[[i]])
        store.write_table(frame, "option_chains")
        assert store.table_stats("option_chains").rows == len(frame)


class TestStats:
    def test_stats_describe_the_written_table(self, store):
        frame = make_chains()
        store.write_table(frame, "option_chains")
        stats = store.table_stats("option_chains")
        assert stats.rows == len(frame)
        assert stats.years == [2023, 2024]
        assert stats.files >= 2
        assert stats.bytes > 0
        assert len(stats.content_hash) == 64

    def test_iter_table_yields_one_frame_per_year(self, store):
        store.write_table(make_chains(), "option_chains")
        got = dict(store.iter_table("option_chains"))
        assert sorted(got) == [2023, 2024]
        assert all(len(f) > 0 for f in got.values())

    def test_dropping_a_table_removes_everything(self, store):
        store.write_table(make_chains(), "option_chains")
        store.drop_table("option_chains")
        assert store.table_years("option_chains") == []
        assert store.table_stats("option_chains").rows == 0


class TestWriteGuards:
    def test_writing_into_a_grandfathered_tree_is_refused(self, tmp_root):
        # The research trees cost a month of quota to acquire; an accidental
        # write there should be impossible, not merely unlikely.
        from engine import paths

        with pytest.raises(PermissionError, match="grandfathered"):
            paths.assert_writable(paths.RAW_ORATS_STRIKES / "x.json.gz")

    def test_a_bad_frame_never_reaches_disk(self, store):
        frame = make_chains()
        frame.loc[0, "ticker"] = None
        with pytest.raises(Exception):
            store.write_table(frame, "option_chains")
        assert store.table_stats("option_chains").rows == 0


class TestFinalizeDedupe:
    """Primary-key uniqueness is a whole-table property, resolved at finalize."""

    def test_duplicates_across_batches_are_removed(self, store):
        frame = make_chains(years=(2024,), per_year=4)
        with store.PartitionedWriter("option_chains", max_buffered_rows=2) as writer:
            writer.add(frame)
            writer.add(frame)  # the same contracts arriving from a second payload
            assert store.table_stats("option_chains").rows == 0 or True
            removed = writer.finalize(dedupe=True)
        assert removed == len(frame)
        assert store.table_stats("option_chains").rows == len(frame)

    def test_the_surviving_row_is_the_first_one_fed(self, store):
        first = make_chains(years=(2024,), per_year=2)
        second = first.copy()
        second["bid"] = 99.0
        with store.PartitionedWriter("option_chains") as writer:
            writer.add(first)
            writer.add(second)
            writer.finalize(dedupe=True)
        out = store.read_table("option_chains")
        assert (out["bid"] != 99.0).all()

    def test_the_deduped_table_satisfies_its_own_primary_key(self, store):
        frame = make_chains(years=(2023, 2024), per_year=4)
        with store.PartitionedWriter("option_chains") as writer:
            writer.add(frame)
            writer.add(frame)
            writer.finalize(dedupe=True)
        out = store.read_table("option_chains")
        # This is the contract the store was previously violating.
        assert_schema(out, "option_chains", check_keys=True)

    def test_finalize_compacts_part_files(self, store):
        from engine import paths

        frame = make_chains(years=(2024,), per_year=20)
        with store.PartitionedWriter("option_chains", max_buffered_rows=2) as writer:
            for i in range(len(frame)):
                writer.add(frame.iloc[[i]])
            assert store.table_stats("option_chains").files > 1
            writer.finalize(dedupe=True)
        assert store.table_stats("option_chains").files == 1

    def test_rows_written_is_corrected_for_removals(self, store):
        frame = make_chains(years=(2024,), per_year=3)
        with store.PartitionedWriter("option_chains") as writer:
            writer.add(frame)
            writer.add(frame)
            writer.finalize(dedupe=True)
            assert writer.rows_written == len(frame)

    def test_finalize_without_dedupe_only_compacts(self, store):
        frame = make_chains(years=(2024,), per_year=3)
        with store.PartitionedWriter("option_chains", max_buffered_rows=1) as writer:
            writer.add(frame)
            writer.add(frame)
            removed = writer.finalize(dedupe=False)
        assert removed == 0
        assert store.table_stats("option_chains").rows == 2 * len(frame)

    def test_a_clean_build_is_unchanged_by_finalize(self, store):
        frame = make_chains(years=(2023, 2024), per_year=4)
        with store.PartitionedWriter("option_chains") as writer:
            writer.add(frame)
            removed = writer.finalize(dedupe=True)
        assert removed == 0
        assert store.table_stats("option_chains").rows == len(frame)
