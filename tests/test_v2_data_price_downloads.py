"""Point-in-time yfinance price versions -- the pure data-layer logic, plus
the ``Repository.scan``-backed typed read.

Covers ``engine.v2.data.price_history`` (the normalized bitemporal diff/as-of
logic the design settled on -- ``engine.v2.data.price_downloads``, the
original whole-download resolution/pin/replay module, was removed per the
2026-09-14 SEND-BACK once this module's ``as_of_view`` was confirmed to cover
the same resolution rule at the row-version grain; its stability property
test is ``test_as_of_view_property_full_history_equals_own_retrieval_and_is_stable``
below, migrated from ``price_downloads``'s own
``test_resolve_at_cutoff_property_appending_later_never_changes_earlier_cutoffs``),
``engine.v2.data.price_download_sources`` (read-only legacy parsing; Tier-1
``close_raw`` is anchored against the real ``panel._yf_history_from_tier1``'s
``close_adj`` -- the same ``Close`` column, read for a different purpose --
see ``test_tier1_close_column_matches_legacy_panels_close_adj``), and
``engine.v2.data.price_history_query``/``Repository.get_price_series``/
``get_close`` (``Repository.scan(DataQuery)`` over a real, synthetic,
committed ``price_history`` snapshot -- no network, no legacy writes).
"""
from __future__ import annotations

import gzip
import json
import random

import pandas as pd
import pytest

from engine.v2.contracts import PriceQuery
from engine.v2.data import price_download_sources as sources
from engine.v2.data import price_history
from engine.v2.data.errors import DataError
from engine.v2.data.price_history_table import PRICE_HISTORY_CONTRACT, PRICE_HISTORY_TABLE_NAME
from engine.v2.data.repository import Repository
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_ref_for,
    publish_and_inspect,
)

# --------------------------------------------------------------------------
# price_history: diff_retrieval / as_of_view
# --------------------------------------------------------------------------


_STORED_COLUMNS = ["date", "close_adj", "close_raw", "high_raw", "retrieved_at", "deleted",
                  "source_kind", "source_hash", "capture_id"]


def _empty_stored():
    return pd.DataFrame(columns=_STORED_COLUMNS)


def _retrieval(rows):
    """rows: {date: (close_adj, close_raw, high_raw)}."""
    return pd.DataFrame([{"date": d, "close_adj": v[0], "close_raw": v[1], "high_raw": v[2]}
                         for d, v in rows.items()])


def _apply(stored, rows, *, retrieved_at, source_kind="legacy_px_csv", source_hash="h",
          capture_id="c"):
    new_rows = price_history.diff_retrieval(stored, _retrieval(rows), retrieved_at=retrieved_at,
                                            source_kind=source_kind, source_hash=source_hash,
                                            capture_id=capture_id)
    return pd.concat([stored, new_rows], ignore_index=True), new_rows


def test_diff_retrieval_first_capture_adds_every_row():
    stored, new_rows = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.1),
                                                "2024-01-02": (2.0, 2.0, 2.1)},
                              retrieved_at="2024-01-05T00:00:00Z")
    assert len(new_rows) == 2
    assert not new_rows["deleted"].any()


def test_diff_retrieval_unchanged_is_empty():
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.1)},
                       retrieved_at="2024-01-05T00:00:00Z")
    _, new_rows = _apply(stored, {"2024-01-01": (1.0, 1.0, 1.1)}, retrieved_at="2024-01-06T00:00:00Z")
    assert new_rows.empty


def test_diff_retrieval_changed_value_adds_one_row():
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.1)},
                       retrieved_at="2024-01-05T00:00:00Z")
    _, new_rows = _apply(stored, {"2024-01-01": (9.0, 1.0, 1.1)}, retrieved_at="2024-01-06T00:00:00Z")
    assert len(new_rows) == 1 and new_rows.iloc[0]["close_adj"] == 9.0


def test_diff_retrieval_nan_equal_to_nan_is_unchanged():
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, float("nan"), float("nan"))},
                       retrieved_at="2024-01-05T00:00:00Z")
    _, new_rows = _apply(stored, {"2024-01-01": (1.0, float("nan"), float("nan"))},
                         retrieved_at="2024-01-06T00:00:00Z")
    assert new_rows.empty


def test_diff_retrieval_missing_date_tombstones():
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.1), "2024-01-02": (2.0, 2.0, 2.1)},
                       retrieved_at="2024-01-05T00:00:00Z")
    _, new_rows = _apply(stored, {"2024-01-01": (1.0, 1.0, 1.1)}, retrieved_at="2024-01-06T00:00:00Z")
    assert len(new_rows) == 1
    assert new_rows.iloc[0]["date"] == "2024-01-02"
    assert bool(new_rows.iloc[0]["deleted"]) is True


def test_diff_retrieval_partial_window_refused():
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.1), "2024-01-02": (2.0, 2.0, 2.1)},
                       retrieved_at="2024-01-05T00:00:00Z")
    with pytest.raises(DataError) as exc:
        _apply(stored, {"2024-01-02": (2.0, 2.0, 2.1)}, retrieved_at="2024-01-06T00:00:00Z")
    assert exc.value.code == "CONTRACT_MISMATCH"


def test_check_not_backdated_refuses():
    with pytest.raises(DataError) as exc:
        price_history.check_not_backdated(["2024-01-05T00:00:00Z"], "2024-01-03T00:00:00Z")
    assert exc.value.code == "INPUT_CHANGED"


def test_check_not_backdated_allows_strictly_later():
    price_history.check_not_backdated(["2024-01-05T00:00:00Z"], "2024-01-06T00:00:00Z")


def test_as_of_view_picks_version_at_cutoff_and_drops_tombstones():
    stored = _empty_stored()
    stored, _ = _apply(stored, {"2024-01-01": (1.0, None, None), "2024-01-02": (2.0, None, None)},
                       retrieved_at="2024-01-05T00:00:00Z")
    # 2024-01-02 unchanged and still present -- stays live.
    stored, _ = _apply(stored, {"2024-01-01": (9.0, None, None), "2024-01-02": (2.0, None, None)},
                       retrieved_at="2024-01-10T00:00:00Z")
    # 2024-01-02 dropped from this retrieval -> tombstoned here.
    stored, _ = _apply(stored, {"2024-01-01": (9.0, None, None)}, retrieved_at="2024-01-20T00:00:00Z")

    before_change = price_history.as_of_view(stored, "2024-01-07T00:00:00Z")
    assert dict(zip(before_change["date"], before_change["close_adj"])) == {
        "2024-01-01": 1.0, "2024-01-02": 2.0}

    after_change_before_tombstone = price_history.as_of_view(stored, "2024-01-15T00:00:00Z")
    assert dict(zip(after_change_before_tombstone["date"], after_change_before_tombstone["close_adj"])) == {
        "2024-01-01": 9.0, "2024-01-02": 2.0}

    after_tombstone = price_history.as_of_view(stored, "2024-01-25T00:00:00Z")
    assert dict(zip(after_tombstone["date"], after_tombstone["close_adj"])) == {"2024-01-01": 9.0}


def test_as_of_view_before_any_capture_uses_earliest():
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, None, None)},
                       retrieved_at="2024-06-01T00:00:00Z")
    stored, _ = _apply(stored, {"2024-01-01": (2.0, None, None)}, retrieved_at="2024-07-01T00:00:00Z")
    view = price_history.as_of_view(stored, "2020-01-01T00:00:00Z")
    assert dict(zip(view["date"], view["close_adj"])) == {"2024-01-01": 1.0}


def test_as_of_view_never_filters_by_source_kind_across_restatement_and_tombstone():
    """User decision 2026-09-14 (replacing an earlier two-policy design):
    ONE read rule, no source_kind filter, ever -- rows are diff-only, so an
    unchanged value across sources stays stored once, under whichever
    retrieval first saw it. A px retrieval, then a Tier-1 retrieval that adds
    a new date and restates one old date, then a Tier-1 tombstone -- assert
    the exact view at cutoffs before, between and after all three. If the
    view were (wrongly) filtered down to the chosen retrieval's own
    source_kind, 2024-01-01 (a px-only row, never restated by Tier-1) would
    silently vanish the moment a Tier-1 retrieval became the chosen one.
    """
    stored = _empty_stored()
    # T1 (px): the whole initial history.
    stored, _ = _apply(stored, {"2024-01-01": (1.0, None, None), "2024-01-02": (2.0, None, None),
                                "2024-01-03": (3.0, None, None)},
                       retrieved_at="2024-01-05T00:00:00Z", source_kind="legacy_px_csv",
                       source_hash="h1", capture_id="c1")
    # T2 (tier1): adds 2024-01-04, restates 2024-01-02, leaves 2024-01-01/03
    # untouched (diff-only: no new row for either).
    stored, _ = _apply(stored, {"2024-01-01": (1.0, None, None), "2024-01-02": (20.0, None, None),
                                "2024-01-03": (3.0, None, None), "2024-01-04": (4.0, None, None)},
                       retrieved_at="2024-02-01T00:00:00Z", source_kind="tier1_fetch",
                       source_hash="h2", capture_id="c2")
    # T3 (tier1): drops 2024-01-04 -> tombstones it; the earliest date stays
    # present so this is not a partial-window refusal.
    stored, _ = _apply(stored, {"2024-01-01": (1.0, None, None), "2024-01-02": (20.0, None, None),
                                "2024-01-03": (3.0, None, None)},
                       retrieved_at="2024-03-01T00:00:00Z", source_kind="tier1_fetch",
                       source_hash="h3", capture_id="c3")

    before_t1 = price_history.as_of_view(stored, "2024-01-01T00:00:00Z")
    assert dict(zip(before_t1["date"], before_t1["close_adj"])) == {
        "2024-01-01": 1.0, "2024-01-02": 2.0, "2024-01-03": 3.0}

    between_t1_t2 = price_history.as_of_view(stored, "2024-01-10T00:00:00Z")
    assert dict(zip(between_t1_t2["date"], between_t1_t2["close_adj"])) == {
        "2024-01-01": 1.0, "2024-01-02": 2.0, "2024-01-03": 3.0}

    between_t2_t3 = price_history.as_of_view(stored, "2024-02-15T00:00:00Z")
    assert dict(zip(between_t2_t3["date"], between_t2_t3["close_adj"])) == {
        "2024-01-01": 1.0, "2024-01-02": 20.0, "2024-01-03": 3.0, "2024-01-04": 4.0}

    after_t3 = price_history.as_of_view(stored, "2024-04-01T00:00:00Z")
    assert dict(zip(after_t3["date"], after_t3["close_adj"])) == {
        "2024-01-01": 1.0, "2024-01-02": 20.0, "2024-01-03": 3.0}


def _simulate_full_history_captures(retrievals):
    stored = _empty_stored()
    for i, (retrieved_at, values) in enumerate(retrievals):
        stored, _ = _apply(stored, {d: (v, v, v) for d, v in values.items()},
                           retrieved_at=retrieved_at, source_hash=f"h{i}", capture_id=f"c{i}")
    return stored


_DATES = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]


def _random_full_history(rng):
    dates = [_DATES[0]] + [d for d in _DATES[1:] if rng.random() < 0.7]
    return {d: round(rng.uniform(1, 100), 4) for d in dates}


def test_as_of_view_property_full_history_equals_own_retrieval_and_is_stable():
    """Migrated from ``price_downloads.resolve_at_cutoff``'s own stability
    property (removed 2026-09-14 SEND-BACK): appending a later capture never
    changes what an earlier cutoff already resolved to.
    """
    rng = random.Random(1234567)
    for _trial in range(80):
        n = rng.randint(1, 4)
        stamps = sorted({rng.randint(1, 50) for _ in range(n)})
        retrieved_ats = [f"2024-06-{s:02d}T00:00:00Z" for s in stamps]
        retrievals = [(ts, _random_full_history(rng)) for ts in retrieved_ats]
        stored = _simulate_full_history_captures(retrievals)

        for ts, values in retrievals:
            view = price_history.as_of_view(stored, ts)
            got = dict(zip(view["date"], view["close_adj"]))
            assert got == pytest.approx(values)

        later_stamp = max(stamps) + rng.randint(1, 10)
        later_ts = f"2024-06-{later_stamp:02d}T00:00:00Z"
        later_values = _random_full_history(rng)
        stored_after, _ = _apply(stored, {d: (v, v, v) for d, v in later_values.items()},
                                 retrieved_at=later_ts, source_hash="later", capture_id="later")
        for ts, _values in retrievals:
            before = price_history.as_of_view(stored, ts)
            after = price_history.as_of_view(stored_after, ts)
            pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))


# --------------------------------------------------------------------------
# price_download_sources: legacy parsing, and parse equality vs panel.py
# --------------------------------------------------------------------------


def test_read_legacy_px_csv(tmp_path):
    path = tmp_path / "px_AAPL.csv"
    path.write_text("date,close_adj,close_raw,high_raw\n2024-01-02,100.0,100.0,101.0\n"
                    "2024-01-01,99.0,99.0,99.5\n")
    frame = sources.read_legacy_px_csv(path)
    assert list(frame["date"].astype(str).str.slice(0, 10)) == ["2024-01-01", "2024-01-02"]


def test_normalize_px_csv_reshapes_and_fills_missing_columns(tmp_path):
    path = tmp_path / "px_AAPL.csv"
    path.write_text("date,close_adj\n2024-01-01,1.0\n")
    frame = sources.read_legacy_px_csv(path)
    normalized = sources.normalize_px_csv("AAPL", frame, retrieved_at="2024-01-05T00:00:00Z",
                                          source_hash="h")
    assert list(normalized.columns) == list(sources.NORMALIZED_COLUMNS)
    assert normalized.iloc[0]["ticker"] == "AAPL"
    assert pd.isna(normalized.iloc[0]["close_raw"])


def _yfinance_csv_bytes(rows, *, header="Date,Open,High,Low,Close,Adj Close,Volume", adj=None):
    """``rows``: ``[(date, close), ...]`` -- used for Open/High/Low/Close (so
    ``close_raw``/``high_raw`` both come out equal to ``close``). ``adj``,
    when given, is a parallel ``{date: adj_close}`` map so ``Adj Close`` (->
    ``close_adj``) can differ from ``Close`` -- real Tier-1 bodies always
    have both (2026-09-15 fix; see ``price_download_sources``'s module
    docstring)."""
    lines = [header]
    for d, close in rows:
        adj_close = adj[d] if adj else close
        lines.append(f"{d},{close},{close},{close},{close},{adj_close},1000")
    return ("\n".join(lines) + "\n").encode()


def test_read_tier1_body_parses_all_three_columns_correctly():
    """A real 7-column Tier-1 body: close_adj = Adj Close (dividend-adjusted),
    close_raw = Close, high_raw = High -- distinct values on each, so a
    column-swap bug would fail this (2026-09-15 fix)."""
    body = _yfinance_csv_bytes([("2024-01-01", 10.0), ("2024-01-02", 11.0)],
                               adj={"2024-01-01": 9.5, "2024-01-02": 10.6})
    frame = sources.read_tier1_body(body)
    assert list(frame["close_adj"]) == [9.5, 10.6]
    assert list(frame["close_raw"]) == [10.0, 11.0]
    assert list(frame["high_raw"]) == [10.0, 11.0]


def test_read_tier1_body_refuses_when_adj_close_column_is_missing():
    body = _yfinance_csv_bytes([("2024-01-01", 10.0)],
                               header="Date,Open,High,Low,Close,Volume")
    with pytest.raises(DataError) as exc:
        sources.read_tier1_body(body)
    assert exc.value.code == "CONTRACT_MISMATCH"


def test_read_tier1_body_refuses_when_a_column_parses_entirely_nan():
    lines = ["Date,Open,High,Low,Close,Adj Close,Volume",
            "2024-01-01,10.0,10.0,10.0,10.0,not-a-number,1000",
            "2024-01-02,11.0,11.0,11.0,11.0,also-not-a-number,1000"]
    body = ("\n".join(lines) + "\n").encode()
    with pytest.raises(DataError) as exc:
        sources.read_tier1_body(body)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details["all_nan_columns"] == ["close_adj"]


def test_tier1_close_column_matches_legacy_panels_close_adj(tmp_path, monkeypatch):
    """Legacy's own ``_yf_history_from_tier1`` (panel.py:444-474) maps Tier-1
    ``Close`` to ITS ``close_adj`` -- a different semantic (split-adjusted
    only, for run-up features; that function's own docstring says dividends
    are never adjusted there). This module's ``close_raw`` reads the SAME
    ``Close`` column, so it should match legacy's ``close_adj`` output even
    though this module's OWN ``close_adj`` (from ``Adj Close``) legitimately
    differs -- a lighter anchor than the byte-for-byte parity this test used
    to assert, retired 2026-09-15 because the two functions now compute
    genuinely different columns on purpose (see the module docstring's
    "Tier-1 fallback" section: parity with legacy is not the goal here).
    """
    from engine.data.features import panel
    from engine.data.fetch import cache_key

    fetch_root = tmp_path / "fetch"
    ticker = "ZETA"
    key = cache_key("yfinance", "history", {"ticker": ticker, "period": "max"})
    sub = fetch_root / "yfinance" / key[:2]
    sub.mkdir(parents=True)
    body = _yfinance_csv_bytes([("2024-01-01", 10.0), ("2024-01-02", 11.5), ("2024-01-03", 12.25)],
                               adj={"2024-01-01": 9.9, "2024-01-02": 11.4, "2024-01-03": 12.1})
    (sub / f"{key}.body.gz").write_bytes(gzip.compress(body, mtime=0))
    (sub / f"{key}.meta.json").write_text(json.dumps({
        "source": "yfinance", "endpoint": "history", "key": key,
        "params": {"ticker": ticker, "period": "max"},
        "fetched_at": "2024-01-04T00:00:00+00:00", "status": 200}))

    import engine.paths as engine_paths
    monkeypatch.setattr(engine_paths, "RAW_FETCH", fetch_root)

    real = panel._yf_history_from_tier1(ticker)
    ours = sources.read_tier1_body(body)

    assert list(real["close_adj"]) == list(ours["close_raw"])
    assert list(ours["close_adj"]) == [9.9, 11.4, 12.1]
    assert [str(d)[:10] for d in real["date"]] == [str(d)[:10] for d in ours["date"]]


# --------------------------------------------------------------------------
# price_history_query / Repository.get_price_series / Repository.get_close:
# a real, synthetic, committed price_history snapshot (SEND-BACK 2026-09-14
# requirement 2: built on Repository.scan(DataQuery), not a bare DataFrame).
# --------------------------------------------------------------------------


def _price_query(ticker="AAPL", session_date="2024-01-10", ceiling="2024-01-10T00:00:00Z",
                 lookback=10):
    return PriceQuery(ticker=ticker, session_date=session_date, observation_ceiling=ceiling,
                      lookback_sessions=lookback)


def _commit_price_history_snapshot(conn, clock, store, ticker_rows: dict[str, pd.DataFrame]):
    """One fresh snapshot, scope ``"test"``, whose only table is
    ``price_history`` -- one whole-ticker fragment per ``ticker_rows`` entry
    (already-diffed ``stored`` frames, e.g. built with ``_apply`` above)."""
    contract_ref = contract_ref_for(PRICE_HISTORY_CONTRACT)
    records = []
    for ticker in sorted(ticker_rows):
        ordered = ticker_rows[ticker].sort_values(["date", "retrieved_at"]).reset_index(drop=True).copy()
        ordered.insert(0, "ticker", ticker)
        rows = ordered.to_dict("records")
        records.append(publish_and_inspect(store, PRICE_HISTORY_CONTRACT, contract_ref, rows, ticker))
    return commit_tables(conn, clock, {PRICE_HISTORY_TABLE_NAME: records},
                         {PRICE_HISTORY_TABLE_NAME: PRICE_HISTORY_CONTRACT}, scope="test")


def test_get_price_series_value_before_and_after_a_changed_retrieval(tmp_path):
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.0)},
                       retrieved_at="2024-01-02T00:00:00Z", source_hash="h0")
    stored, _ = _apply(stored, {"2024-01-01": (5.0, 5.0, 5.0)}, retrieved_at="2024-01-08T00:00:00Z",
                       source_hash="h1")
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit_price_history_snapshot(conn, clock, store, {"AAPL": stored})
    repository = Repository(conn, store)

    before = repository.get_price_series(
        _price_query(session_date="2024-01-01", ceiling="2024-01-05T00:00:00Z"), snapshot)
    assert before[-1].close_adj == 1.0
    assert before[-1].source_hash == "h0"

    after = repository.get_price_series(_price_query(ceiling="2024-01-10T00:00:00Z"), snapshot)
    assert after[-1].close_adj == 5.0
    assert after[-1].source_hash == "h1"


def test_get_price_series_ceiling_before_any_retrieval_uses_earliest(tmp_path):
    stored, _ = _apply(_empty_stored(), {"2020-01-01": (1.0, 1.0, 1.0), "2020-01-02": (2.0, 2.0, 2.0),
                                        "2020-01-03": (3.0, 3.0, 3.0)},
                       retrieved_at="2024-06-05T00:00:00Z", source_hash="h0")
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit_price_history_snapshot(conn, clock, store, {"AAPL": stored})
    repository = Repository(conn, store)
    result = repository.get_price_series(
        _price_query(session_date="2020-01-02", ceiling="2020-01-02T00:00:00Z", lookback=10), snapshot)
    assert [r.close_adj for r in result] == [1.0, 2.0]
    assert result[-1].source_hash == "h0"


def test_get_price_series_drops_a_tombstoned_date(tmp_path):
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.0), "2024-01-02": (2.0, 2.0, 2.0)},
                       retrieved_at="2024-01-03T00:00:00Z")
    stored, _ = _apply(stored, {"2024-01-01": (1.0, 1.0, 1.0)}, retrieved_at="2024-01-10T00:00:00Z")
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit_price_history_snapshot(conn, clock, store, {"AAPL": stored})
    repository = Repository(conn, store)
    result = repository.get_price_series(
        _price_query(session_date="2024-01-02", ceiling="2024-01-15T00:00:00Z"), snapshot)
    assert [r.date for r in result] == ["2024-01-01"]


def test_get_price_series_refuses_query_not_bounded(tmp_path):
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.0)},
                       retrieved_at="2024-01-02T00:00:00Z")
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit_price_history_snapshot(conn, clock, store, {"AAPL": stored})
    repository = Repository(conn, store)
    with pytest.raises(DataError) as exc:
        repository.get_price_series(
            _price_query(session_date="2024-06-01", ceiling="2024-01-01T00:00:00Z"), snapshot)
    assert exc.value.code == "QUERY_NOT_BOUNDED"


def test_get_price_series_refuses_unknown_ticker(tmp_path):
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.0)},
                       retrieved_at="2024-01-02T00:00:00Z")
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit_price_history_snapshot(conn, clock, store, {"AAPL": stored})
    repository = Repository(conn, store)
    with pytest.raises(DataError) as exc:
        repository.get_price_series(_price_query(ticker="ZZZZ"), snapshot)
    assert exc.value.code == "CONTRACT_MISMATCH"


def test_get_close_returns_provenance(tmp_path):
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.0)},
                       retrieved_at="2024-01-02T00:00:00Z", source_hash="hx")
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit_price_history_snapshot(conn, clock, store, {"AAPL": stored})
    repository = Repository(conn, store)
    row = repository.get_close("AAPL", "2024-01-01", "2024-06-01T00:00:00Z", snapshot)
    assert row.close_adj == 1.0
    assert row.source_hash == "hx"
