"""The Tier-1 fetch store must be readable, not just writable.

The two pull generations store the same ORATS rows in different envelopes: the
legacy chain files wrap them as ``{entry_date, tickers, rows}``, while the fetch
wrapper stores the response verbatim as ``{"data": [...]}``. If only the legacy
shape is readable, Tier 1 is write-only for everything new — the Sep-1 pull
would land 16,000 calls that no rebuild could see, and a restored machine (which
has no legacy tree at all) would rebuild empty tables.
"""
from __future__ import annotations

import gzip
import json

import pandas as pd
import pytest

from engine.data.fetch import CachedEntry, iter_cached
from engine.data.normalize import n_chains
from engine.data.normalize.fetch_store import iter_orats_rows, orats_rows


def strike_row(**over):
    row = {
        "ticker": "AAA",
        "tradeDate": "2026-09-02",
        "expirDate": "2026-09-18",
        "dte": 16,
        "strike": 100.0,
        "stockPrice": 99.5,
        "spotPrice": 99.5,
        "callBidPrice": 2.0,
        "callAskPrice": 2.4,
        "callMidIv": 0.31,
        "putBidPrice": 1.0,
        "putAskPrice": 1.4,
        "putMidIv": 0.33,
        "delta": 0.55,
    }
    row.update(over)
    return row


def write_entry(root, source, endpoint, key, payload, params=None):
    """Write a fetch-store entry the way `Fetcher._persist` does."""
    directory = root / source / key[:2]
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / f"{key}.body.gz", "wb") as fh:
        fh.write(json.dumps(payload).encode())
    (directory / f"{key}.meta.json").write_text(
        json.dumps(
            {
                "key": key,
                "source": source,
                "endpoint": endpoint,
                "params": params or {},
                "status": 200,
            }
        )
    )


class TestEnvelopeParsing:
    def test_the_orats_data_envelope_is_unwrapped(self):
        assert orats_rows({"data": [{"a": 1}, {"a": 2}]}) == [{"a": 1}, {"a": 2}]

    def test_the_legacy_rows_envelope_is_also_accepted(self):
        assert orats_rows({"entry_date": "x", "rows": [{"a": 1}]}) == [{"a": 1}]

    def test_a_bare_list_is_accepted(self):
        assert orats_rows([{"a": 1}]) == [{"a": 1}]

    @pytest.mark.parametrize("payload", [None, {}, {"data": None}, 42, "text"])
    def test_unrecognized_shapes_yield_no_rows_rather_than_raising(self, payload):
        # One odd payload must not abort a rebuild over the whole store.
        assert orats_rows(payload) == []

    def test_non_dict_members_are_skipped(self):
        assert orats_rows({"data": [{"a": 1}, "junk", None]}) == [{"a": 1}]


class TestStoreScanning:
    def test_entries_are_discovered_and_filtered_by_endpoint(self, tmp_path):
        write_entry(tmp_path, "orats", "hist/strikes", "aa" + "0" * 62, {"data": []})
        write_entry(tmp_path, "orats", "hist/earnings", "bb" + "1" * 62, {"data": []})

        assert len(iter_cached("orats", root=tmp_path)) == 2
        strikes = iter_cached("orats", "hist/strikes", root=tmp_path)
        assert len(strikes) == 1
        assert strikes[0].endpoint == "hist/strikes"

    def test_the_scan_is_deterministically_ordered(self, tmp_path):
        for key in ("cc" + "2" * 62, "aa" + "0" * 62, "bb" + "1" * 62):
            write_entry(tmp_path, "orats", "hist/strikes", key, {"data": []})
        keys = [e.key for e in iter_cached("orats", "hist/strikes", root=tmp_path)]
        assert keys == sorted(keys)

    def test_an_absent_store_yields_nothing(self, tmp_path):
        assert iter_cached("orats", root=tmp_path / "nope") == []

    def test_an_unreadable_sidecar_is_skipped_not_fatal(self, tmp_path):
        write_entry(tmp_path, "orats", "hist/strikes", "aa" + "0" * 62, {"data": []})
        bad = tmp_path / "orats" / "zz"
        bad.mkdir(parents=True)
        (bad / ("zz" + "9" * 62 + ".meta.json")).write_text("{not json")
        assert len(iter_cached("orats", root=tmp_path)) == 1

    def test_a_sidecar_without_a_body_is_skipped(self, tmp_path):
        directory = tmp_path / "orats" / "aa"
        directory.mkdir(parents=True)
        (directory / ("aa" + "0" * 62 + ".meta.json")).write_text(
            json.dumps({"key": "aa", "source": "orats", "endpoint": "hist/strikes"})
        )
        assert iter_cached("orats", root=tmp_path) == []

    def test_the_body_loads_lazily_and_parses(self, tmp_path):
        write_entry(tmp_path, "orats", "hist/strikes", "aa" + "0" * 62, {"data": [{"x": 1}]})
        entry = iter_cached("orats", root=tmp_path)[0]
        assert entry.json() == {"data": [{"x": 1}]}

    def test_source_id_identifies_the_payload(self, tmp_path):
        write_entry(tmp_path, "orats", "hist/strikes", "aa" + "0" * 62, {"data": []})
        entry = iter_cached("orats", root=tmp_path)[0]
        assert entry.source_id.startswith("fetch:orats/hist/strikes/")

    def test_params_are_recovered_from_the_sidecar(self, tmp_path):
        write_entry(
            tmp_path, "orats", "hist/strikes", "aa" + "0" * 62, {"data": []},
            params={"tradeDate": "2026-09-02"},
        )
        assert iter_cached("orats", root=tmp_path)[0].params["tradeDate"] == "2026-09-02"


class TestBothGenerationsProduceIdenticalRows:
    """The whole point: envelope differences must not reach Tier 2."""

    def test_the_same_rows_normalize_identically_either_way(self, tmp_path):
        rows = [strike_row(), strike_row(strike=105.0)]

        legacy_path = tmp_path / "2026-09-02_b0.json.gz"
        with gzip.open(legacy_path, "wt") as fh:
            json.dump({"entry_date": "2026-09-02", "tickers": ["AAA"], "rows": rows}, fh)
        legacy, _ = n_chains.normalize_file(legacy_path)

        store_root = tmp_path / "store"
        write_entry(store_root, "orats", "hist/strikes", "aa" + "0" * 62, {"data": rows})
        source = next(iter_orats_rows("hist/strikes", root=store_root))
        fetched, _ = n_chains.normalize_fetch_rows(source)

        # Provenance and pull-kind differ by design; every price and key must not.
        compare = ["ticker", "obs_date", "expiry", "dte", "strike", "right",
                   "bid", "ask", "mid", "iv", "delta", "spot"]
        pd.testing.assert_frame_equal(
            legacy[compare].reset_index(drop=True),
            fetched[compare].reset_index(drop=True),
        )

    def test_fetch_rows_are_labelled_as_a_distinct_pull_kind(self, tmp_path):
        write_entry(tmp_path, "orats", "hist/strikes", "aa" + "0" * 62, {"data": [strike_row()]})
        source = next(iter_orats_rows("hist/strikes", root=tmp_path))
        frame, report = n_chains.normalize_fetch_rows(source)
        assert (frame["chain_kind"] == n_chains.FETCH_CHAIN_KIND).all()
        assert report["chain_kind"] == n_chains.FETCH_CHAIN_KIND

    def test_provenance_points_back_at_the_payload(self, tmp_path):
        write_entry(tmp_path, "orats", "hist/strikes", "aa" + "0" * 62, {"data": [strike_row()]})
        source = next(iter_orats_rows("hist/strikes", root=tmp_path))
        frame, _ = n_chains.normalize_fetch_rows(source)
        assert frame["src_file"].iloc[0].startswith("fetch:orats/hist/strikes/")

    def test_an_empty_payload_yields_no_rows(self, tmp_path):
        write_entry(tmp_path, "orats", "hist/strikes", "aa" + "0" * 62, {"data": []})
        assert list(iter_orats_rows("hist/strikes", root=tmp_path)) == []


class TestDailyBridge:
    def test_the_fetch_index_groups_summaries_and_cores_by_ticker(self, tmp_path, monkeypatch):
        from engine import paths
        from engine.data.normalize import n_daily

        write_entry(
            tmp_path, "orats", "hist/summaries", "aa" + "0" * 62,
            {"data": [{"ticker": "AAA", "tradeDate": "2026-09-02", "stockPrice": 10.0}]},
        )
        write_entry(
            tmp_path, "orats", "hist/cores", "bb" + "1" * 62,
            {"data": [{"ticker": "AAA", "tradeDate": "2026-09-02", "mktCap": 5_000_000}]},
        )
        monkeypatch.setattr(paths, "RAW_FETCH", tmp_path)
        n_daily.fetch_daily_index.cache_clear()
        try:
            index = n_daily.fetch_daily_index()
            assert set(index) == {"AAA"}
            assert len(index["AAA"]["summaries"]) == 1
            assert len(index["AAA"]["cores"]) == 1
        finally:
            n_daily.fetch_daily_index.cache_clear()

    def test_a_ticker_present_only_in_the_fetch_store_normalizes(self, tmp_path, monkeypatch):
        from engine import paths
        from engine.data.normalize import n_daily

        write_entry(
            tmp_path, "orats", "hist/summaries", "aa" + "0" * 62,
            {"data": [
                {"ticker": "NEW", "tradeDate": "2026-09-02", "stockPrice": 50.0, "iv30d": 0.3}
            ]},
        )
        monkeypatch.setattr(paths, "RAW_FETCH", tmp_path)
        n_daily.fetch_daily_index.cache_clear()
        try:
            # This is the restored-machine case: no legacy tree at all.
            frame, _ = n_daily.normalize_ticker(
                "NEW", summaries_dir=tmp_path / "absent", cores_dir=tmp_path / "absent"
            )
            assert len(frame) == 1
            assert frame["spot"].iloc[0] == 50.0
            assert frame["iv30"].iloc[0] == pytest.approx(30.0)
            assert "NEW" in n_daily.list_tickers(root=tmp_path / "absent")
        finally:
            n_daily.fetch_daily_index.cache_clear()


class TestRebuildConsumesBothGenerations:
    """End-to-end: a fetch-store payload must actually reach Tier 2.

    The unit tests above prove the two envelopes parse identically. This proves
    the rebuild *looks* at both — the actual defect was that it only ever
    iterated the legacy tree, so the fetch store was write-only and a Sep-1 pull
    would have bought 16,000 calls of chains that no table could see.
    """

    def test_legacy_and_fetch_rows_both_land_in_tier_2(self, tmp_root, monkeypatch):
        import importlib

        from engine import paths as paths_mod

        importlib.reload(paths_mod)
        from engine.data import rebuild as rebuild_mod
        from engine.data import store as store_mod
        from engine.data.normalize import n_daily

        importlib.reload(store_mod)
        importlib.reload(rebuild_mod)
        n_daily.fetch_daily_index.cache_clear()

        # A legacy chain file …
        legacy_dir = paths_mod.RAW_ORATS_STRIKES
        legacy_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(legacy_dir / "2026-09-01_b0.json.gz", "wt") as fh:
            json.dump(
                {
                    "entry_date": "2026-09-01",
                    "tickers": ["OLD"],
                    "rows": [strike_row(ticker="OLD", tradeDate="2026-09-01")],
                },
                fh,
            )

        # … and a fetch-store payload, exactly as the Sep-1 pull would leave it.
        write_entry(
            paths_mod.RAW_FETCH,
            "orats",
            "hist/strikes",
            "ab" + "0" * 62,
            {"data": [strike_row(ticker="NEW", tradeDate="2026-09-02")]},
            params={"tradeDate": "2026-09-02", "ticker": "NEW"},
        )

        try:
            report = rebuild_mod.build_chains_table()
        finally:
            n_daily.fetch_daily_index.cache_clear()

        assert report["legacy_files"] == 1
        assert report["fetch_payloads"] == 1

        out = store_mod.read_table("option_chains")
        assert set(out["ticker"]) == {"OLD", "NEW"}, "the fetch payload never reached Tier 2"
        # Both sides of the strike, from both generations.
        assert len(out) == 4
        assert set(out.loc[out["ticker"] == "NEW", "chain_kind"]) == {"fetch"}
        assert out.loc[out["ticker"] == "NEW", "src_file"].iloc[0].startswith("fetch:")

    def test_the_rebuild_deduplicates_across_generations(self, tmp_root, monkeypatch):
        """A re-pull of a date already held legacy must not double the rows."""
        import importlib

        from engine import paths as paths_mod

        importlib.reload(paths_mod)
        from engine.data import rebuild as rebuild_mod
        from engine.data import store as store_mod
        from engine.data.normalize import n_daily

        importlib.reload(store_mod)
        importlib.reload(rebuild_mod)
        n_daily.fetch_daily_index.cache_clear()

        row = strike_row(ticker="DUP", tradeDate="2026-09-01")
        legacy_dir = paths_mod.RAW_ORATS_STRIKES
        legacy_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(legacy_dir / "2026-09-01_b0.json.gz", "wt") as fh:
            json.dump({"entry_date": "2026-09-01", "tickers": ["DUP"], "rows": [row]}, fh)
        write_entry(
            paths_mod.RAW_FETCH, "orats", "hist/strikes", "ab" + "0" * 62, {"data": [row]}
        )

        try:
            report = rebuild_mod.build_chains_table()
        finally:
            n_daily.fetch_daily_index.cache_clear()

        out = store_mod.read_table("option_chains")
        assert len(out) == 2  # one call + one put, not two of each
        assert report["duplicates_removed"] == 2
        # Legacy is fed first, so it wins the tie and provenance stays stable.
        assert not out["src_file"].iloc[0].startswith("fetch:")
