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
import struct
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.contracts import PriceQuery
from engine.v2.data import price_download_sources as sources
from engine.v2.data import price_history, price_history_query
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


def _tombstone(stored, date, *, retrieved_at, source_kind="legacy_px_csv", source_hash="h",
               capture_id="c"):
    """Append a hand-made ``deleted=True`` version row.

    ``diff_retrieval`` refuses an omitted live date instead of tombstoning it
    (issue #719), so a test that needs a tombstone in the stored frame -- the
    shape older captures actually wrote, which the read side still honours --
    builds one directly."""
    row = {"date": date, "close_adj": float("nan"), "close_raw": float("nan"),
           "high_raw": float("nan"), "retrieved_at": retrieved_at, "deleted": True,
           "source_kind": source_kind, "source_hash": source_hash, "capture_id": capture_id}
    return pd.concat([stored, pd.DataFrame([row], columns=_STORED_COLUMNS)],
                     ignore_index=True)


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


def test_diff_retrieval_omitted_date_refuses_the_whole_retrieval():
    """Issue #719: a retrieval that drops a live stored date used to
    tombstone it; now the WHOLE response is refused ``CONTRACT_MISMATCH``,
    with the offending date named in ``details``."""
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.1), "2024-01-02": (2.0, 2.0, 2.1)},
                       retrieved_at="2024-01-05T00:00:00Z")
    with pytest.raises(DataError) as exc:
        _apply(stored, {"2024-01-01": (1.0, 1.0, 1.1)}, retrieved_at="2024-01-06T00:00:00Z")
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"missing_count": 1, "missing_dates": ["2024-01-02"]}


def test_diff_retrieval_mid_history_gap_refuses():
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.1), "2024-01-02": (2.0, 2.0, 2.1),
                                         "2024-01-03": (3.0, 3.0, 3.1)},
                       retrieved_at="2024-01-05T00:00:00Z")
    with pytest.raises(DataError) as exc:
        _apply(stored, {"2024-01-01": (1.0, 1.0, 1.1), "2024-01-03": (3.0, 3.0, 3.1)},
               retrieved_at="2024-01-06T00:00:00Z")
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"missing_count": 1, "missing_dates": ["2024-01-02"]}


def test_diff_retrieval_empty_retrieval_against_live_history_refuses():
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.1), "2024-01-02": (2.0, 2.0, 2.1)},
                       retrieved_at="2024-01-05T00:00:00Z")
    with pytest.raises(DataError) as exc:
        _apply(stored, {}, retrieved_at="2024-01-06T00:00:00Z")
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"missing_count": 2,
                                         "missing_dates": ["2024-01-01", "2024-01-02"]}


def test_diff_retrieval_omission_is_refused_even_when_the_response_is_not_shorter():
    """Omission is judged by DATE, never by row count: a retrieval that drops
    live stored dates but adds enough new later bars to be at least as long
    as the stored history is still refused whole, and nothing about the input
    frame changes (the raise happens before any write)."""
    live_dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    stored, _ = _apply(_empty_stored(), {d: (float(i), float(i), float(i))
                                         for i, d in enumerate(live_dates, start=1)},
                       retrieved_at="2024-01-05T00:00:00Z")
    before = stored.copy()
    retrieval = {"2024-01-01": (1.0, 1.0, 1.0), "2024-01-04": (4.0, 4.0, 4.0),
                 "2024-01-05": (5.0, 5.0, 5.0)}  # 3 rows >= 3 live, omits 01-02/01-03
    with pytest.raises(DataError) as exc:
        _apply(stored, retrieval, retrieved_at="2024-01-06T00:00:00Z")
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"missing_count": 2,
                                         "missing_dates": ["2024-01-02", "2024-01-03"]}
    assert stored.equals(before)


def test_diff_retrieval_starting_later_than_stored_reports_both_min_dates():
    """The other whole-response refusal (the "full history only" start rule):
    a retrieval whose earliest date is strictly later than the live stored
    history's earliest date names both minima in ``details``."""
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.1),
                                         "2024-01-02": (2.0, 2.0, 2.1)},
                       retrieved_at="2024-01-05T00:00:00Z")
    with pytest.raises(DataError) as exc:
        _apply(stored, {"2024-01-02": (2.0, 2.0, 2.1), "2024-01-03": (3.0, 3.0, 3.1)},
               retrieved_at="2024-01-06T00:00:00Z")
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"stored_min_date": "2024-01-01",
                                         "retrieval_min_date": "2024-01-02"}


def test_diff_retrieval_omitting_more_than_20_dates_reports_count_first_and_last():
    dates = [f"2024-01-{day:02d}" for day in range(1, 26)]  # 25 live dates
    stored, _ = _apply(_empty_stored(), {d: (float(i), float(i), float(i))
                                         for i, d in enumerate(dates, start=1)},
                       retrieved_at="2024-02-01T00:00:00Z")
    with pytest.raises(DataError) as exc:
        _apply(stored, {dates[0]: (1.0, 1.0, 1.0)}, retrieved_at="2024-02-02T00:00:00Z")
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"missing_count": 24,
                                         "first_missing_date": "2024-01-02",
                                         "last_missing_date": "2024-01-25"}


def test_diff_retrieval_exactly_20_missing_dates_still_reports_the_full_list():
    """Boundary of the compact-details rule: 20 missing dates is still the
    ``missing_dates`` list, not the first/last pair."""
    dates = [f"2024-01-{day:02d}" for day in range(1, 22)]  # 21 live dates
    stored, _ = _apply(_empty_stored(), {d: (float(i), float(i), float(i))
                                         for i, d in enumerate(dates, start=1)},
                       retrieved_at="2024-02-01T00:00:00Z")
    with pytest.raises(DataError) as exc:
        _apply(stored, {dates[0]: (1.0, 1.0, 1.0)}, retrieved_at="2024-02-02T00:00:00Z")
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"missing_count": 20, "missing_dates": dates[1:]}


def test_diff_retrieval_21_missing_dates_reports_first_and_last_only():
    """One past the boundary: 21 missing dates compact to
    ``first_missing_date``/``last_missing_date``."""
    dates = [f"2024-01-{day:02d}" for day in range(1, 23)]  # 22 live dates
    stored, _ = _apply(_empty_stored(), {d: (float(i), float(i), float(i))
                                         for i, d in enumerate(dates, start=1)},
                       retrieved_at="2024-02-01T00:00:00Z")
    with pytest.raises(DataError) as exc:
        _apply(stored, {dates[0]: (1.0, 1.0, 1.0)}, retrieved_at="2024-02-02T00:00:00Z")
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"missing_count": 21,
                                         "first_missing_date": "2024-01-02",
                                         "last_missing_date": "2024-01-22"}


def test_diff_retrieval_full_history_with_a_new_later_date_appends_the_new_bar():
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.1), "2024-01-02": (2.0, 2.0, 2.1)},
                       retrieved_at="2024-01-05T00:00:00Z")
    _, new_rows = _apply(stored, {"2024-01-01": (1.0, 1.0, 1.1), "2024-01-02": (2.0, 2.0, 2.1),
                                  "2024-01-03": (3.0, 3.0, 3.1)},
                         retrieved_at="2024-01-06T00:00:00Z")
    assert list(new_rows["date"]) == ["2024-01-03"]
    assert not new_rows["deleted"].any()


def test_diff_retrieval_revives_a_previously_deleted_row():
    """Tombstones written before issue #719 still exist in stored fragments;
    a later retrieval carrying that date again must revive it (never re-refuse
    on a date whose latest version is ``deleted=True``)."""
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.1), "2024-01-02": (2.0, 2.0, 2.1)},
                       retrieved_at="2024-01-05T00:00:00Z")
    stored = _tombstone(stored, "2024-01-02", retrieved_at="2024-01-06T00:00:00Z")
    _, new_rows = _apply(stored, {"2024-01-01": (1.0, 1.0, 1.1), "2024-01-02": (2.5, 2.4, 2.6)},
                         retrieved_at="2024-01-07T00:00:00Z")
    assert list(new_rows["date"]) == ["2024-01-02"]
    assert not new_rows["deleted"].any()
    revived = new_rows.iloc[0]
    assert (revived["close_adj"], revived["close_raw"], revived["high_raw"]) == (2.5, 2.4, 2.6)


def test_diff_retrieval_omitting_a_tombstoned_date_is_not_an_omission_refusal():
    """A tombstoned date is not live, so a retrieval that simply stops before
    it diffs cleanly: unchanged live rows give an empty delta, and a changed
    live row still gives exactly the one new row -- no CONTRACT_MISMATCH."""
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.1),
                                         "2024-01-02": (2.0, 2.0, 2.1),
                                         "2024-01-03": (3.0, 3.0, 3.1)},
                       retrieved_at="2024-01-05T00:00:00Z")
    stored = _tombstone(stored, "2024-01-03", retrieved_at="2024-01-06T00:00:00Z")

    unchanged, new_rows = _apply(stored, {"2024-01-01": (1.0, 1.0, 1.1),
                                          "2024-01-02": (2.0, 2.0, 2.1)},
                                 retrieved_at="2024-01-07T00:00:00Z")
    assert unchanged.equals(stored)
    assert new_rows.empty

    _, new_rows = _apply(stored, {"2024-01-01": (1.0, 1.0, 1.1),
                                  "2024-01-02": (9.0, 2.0, 2.1)},
                         retrieved_at="2024-01-08T00:00:00Z")
    assert list(new_rows["date"]) == ["2024-01-02"]
    assert len(new_rows) == 1
    assert new_rows.iloc[0]["close_adj"] == 9.0


def test_diff_retrieval_wholly_deleted_history_takes_a_fresh_retrieval_without_refusing():
    """Every stored date is tombstoned, so there is no live history left to
    omit: a fresh retrieval of never-seen dates is added exactly, with no
    refusal and no tombstone rows."""
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.1),
                                         "2024-01-02": (2.0, 2.0, 2.1)},
                       retrieved_at="2024-01-05T00:00:00Z")
    stored = _tombstone(stored, "2024-01-01", retrieved_at="2024-01-06T00:00:00Z")
    stored = _tombstone(stored, "2024-01-02", retrieved_at="2024-01-06T00:00:01Z")

    before = len(stored)
    after, new_rows = _apply(stored, {"2024-01-03": (7.0, 7.0, 7.1), "2024-01-04": (8.0, 8.0, 8.1)},
                             retrieved_at="2024-01-07T00:00:00Z")
    assert list(new_rows["date"]) == ["2024-01-03", "2024-01-04"]
    assert len(new_rows) == 2
    assert list(new_rows["close_adj"]) == [7.0, 8.0]
    assert not new_rows["deleted"].any()
    assert len(after) == before + 2


def test_diff_retrieval_reversion_to_earlier_value_diffs_against_latest_state():
    """External review reproduction (2026-09-15): 09-10 price 100, 09-11
    correction to 110, 09-12 reverts to 100 (byte-identical retrieval to
    09-10's). ``diff_retrieval`` always compares against ``latest_state`` --
    never against any single earlier retrieval's own bytes -- so the 09-12
    retrieval must diff against 110 (the CURRENT latest), not vanish as if it
    matched 09-10. This is the pure logic; the duplicate short-circuit that
    used to swallow this case lived one layer up, in
    ``engine.v2.ops.price_history_store._capture_ticker``.
    """
    stored, _ = _apply(_empty_stored(), {"2024-09-10": (100.0, 100.0, 100.0)},
                       retrieved_at="2024-09-10T00:00:00Z", source_hash="h_a")
    stored, _ = _apply(stored, {"2024-09-10": (110.0, 110.0, 110.0)},
                       retrieved_at="2024-09-11T00:00:00Z", source_hash="h_b")
    stored, new_rows = _apply(stored, {"2024-09-10": (100.0, 100.0, 100.0)},
                              retrieved_at="2024-09-12T00:00:00Z", source_hash="h_a")
    assert len(new_rows) == 1
    assert new_rows.iloc[0]["close_adj"] == 100.0
    assert new_rows.iloc[0]["retrieved_at"] == "2024-09-12T00:00:00Z"


def test_diff_retrieval_no_change_against_latest_is_empty_regardless_of_source_hash():
    """A retrieval whose bytes/hash differ from every prior retrieval, but
    whose VALUE matches the current latest state, is still ``no_change`` --
    the duplicate/no-op distinction is about the observed value, not the hash
    identity of the bytes that produced it.
    """
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.1)},
                       retrieved_at="2024-01-05T00:00:00Z", source_hash="h1")
    _, new_rows = _apply(stored, {"2024-01-01": (1.0, 1.0, 1.1)},
                         retrieved_at="2024-01-06T00:00:00Z", source_hash="h2_different_bytes")
    assert new_rows.empty


def test_as_of_view_reversion_sequence_reads_return_a_then_b_then_a():
    """A->B->A: reads at each retrieval's own cutoff return that retrieval's
    value -- the view rule (:func:`resolve_pool`/:func:`as_of_view`) already
    picks the latest retrieval at-or-before cutoff and reconstructs from it,
    so this passes today; it is asserted here as the read-side half of the
    external review's reversion scenario.
    """
    stored, _ = _apply(_empty_stored(), {"2024-09-10": (100.0, None, None)},
                       retrieved_at="2024-09-10T00:00:00Z", source_hash="h_a")
    stored, _ = _apply(stored, {"2024-09-10": (110.0, None, None)},
                       retrieved_at="2024-09-11T00:00:00Z", source_hash="h_b")
    stored, _ = _apply(stored, {"2024-09-10": (100.0, None, None)},
                       retrieved_at="2024-09-12T00:00:00Z", source_hash="h_a")

    at_a = price_history.as_of_view(stored, "2024-09-10T00:00:00Z")
    assert dict(zip(at_a["date"], at_a["close_adj"])) == {"2024-09-10": 100.0}

    at_b = price_history.as_of_view(stored, "2024-09-11T00:00:00Z")
    assert dict(zip(at_b["date"], at_b["close_adj"])) == {"2024-09-10": 110.0}

    at_a_again = price_history.as_of_view(stored, "2024-09-12T00:00:00Z")
    assert dict(zip(at_a_again["date"], at_a_again["close_adj"])) == {"2024-09-10": 100.0}

    # A cutoff after the last retrieval also still resolves to the reverted value.
    at_a_later = price_history.as_of_view(stored, "2024-12-01T00:00:00Z")
    assert dict(zip(at_a_later["date"], at_a_later["close_adj"])) == {"2024-09-10": 100.0}


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
    # 2024-01-02 dropped -- a hand-made tombstone, since ``diff_retrieval``
    # now refuses an omission instead of writing one (issue #719).
    stored = _tombstone(stored, "2024-01-02", retrieved_at="2024-01-20T00:00:00Z")

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
    # T3 (tier1): 2024-01-04 dropped -- a hand-made tombstone (a live-date
    # omission is refused by ``diff_retrieval`` since issue #719); the
    # earliest date stays present so the view is unchanged for it.
    stored = _tombstone(stored, "2024-01-04", retrieved_at="2024-03-01T00:00:00Z",
                        source_kind="tier1_fetch", source_hash="h3", capture_id="c3")

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


def _random_full_history(rng, seen=()):
    """A random subset of ``_DATES`` that always starts at ``_DATES[0]`` and
    never drops a date already seen by an earlier capture -- ``diff_retrieval``
    refuses a live-date omission (issue #719), so each later retrieval is a
    superset of the previous ones."""
    dates = [_DATES[0]] + [d for d in _DATES[1:] if d in set(seen) or rng.random() < 0.7]
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
        seen: set = set()
        retrievals = []
        for ts in retrieved_ats:
            values = _random_full_history(rng, seen)
            retrievals.append((ts, values))
            seen |= set(values)
        stored = _simulate_full_history_captures(retrievals)

        for ts, values in retrievals:
            view = price_history.as_of_view(stored, ts)
            got = dict(zip(view["date"], view["close_adj"]))
            assert got == pytest.approx(values)

        later_stamp = max(stamps) + rng.randint(1, 10)
        later_ts = f"2024-06-{later_stamp:02d}T00:00:00Z"
        later_values = _random_full_history(rng, seen)
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


def _awkward_float64s(n=500, seed=20260915):
    """Deterministic doubles built from random 64-bit patterns, restricted
    to a plausible price magnitude -- close to what real close_adj values
    (raw price times a cumulative float adjustment ratio) actually look
    like. Measured against the real shadow snapshot (snap_6ae7348...):
    ~40-47% of values built this way need the full 17 significant digits
    for an exact round trip, and 143 of 144 fetched real tickers hit at
    least one such row somewhere in their history."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        bits = rng.getrandbits(64)
        x = struct.unpack("<d", struct.pack("<Q", bits))[0]
        if x == x and 0.0001 < abs(x) < 100000:  # finite, plausible price range
            out.append(x)
    return out


def test_write_then_read_legacy_px_csv_round_trips_full_float64_precision(tmp_path):
    """The writer/reader pair ``materialize_price_series`` actually uses
    must recover the exact float64 ``close_adj`` bit pattern it wrote, for
    awkward doubles that need every significant digit -- not merely
    ``==`` up to some approximation. Regression for the 2026-09-15
    VALIDATION_FAILED defect (see
    test_default_pandas_read_csv_loses_precision_on_some_float64_values
    below for the mechanism this pins)."""
    values = _awkward_float64s()
    dates = pd.date_range("2020-01-01", periods=len(values), freq="D").strftime("%Y-%m-%d")
    rows = [SimpleNamespace(date=d, close_adj=v, close_raw=v, high_raw=v)
           for d, v in zip(dates, values)]
    path = tmp_path / "px_AAPL.csv"
    written = sources.write_legacy_px_csv(path, rows)
    readback = sources.read_legacy_px_csv(path)
    assert len(readback) == len(written)
    assert list(readback["close_adj"]) == list(written["close_adj"])
    assert list(readback["close_raw"]) == list(written["close_raw"])
    assert list(readback["high_raw"]) == list(written["high_raw"])


def test_default_pandas_read_csv_loses_precision_on_some_float64_values(tmp_path):
    """Documents the actual mechanism behind the 2026-09-15 VALIDATION_FAILED
    defect (worker.stderr on attempts att_7eaea005... / att_6059778f...):
    pandas' default C float parser (no ``float_precision`` kwarg) is not a
    true round trip for an arbitrary float64 -- it can read back a value 1
    ULP off from what ``to_csv`` wrote, for a real (not vanishingly rare)
    fraction of values. This is exactly why both
    ``price_download_sources.read_legacy_px_csv`` and
    ``engine/data/features/panel.py:533`` now pass
    ``float_precision="round_trip"``."""
    values = _awkward_float64s()
    path = tmp_path / "close_adj_only.csv"
    pd.DataFrame({"close_adj": values}).to_csv(path, index=False)
    default_read = pd.read_csv(path)["close_adj"].tolist()
    mismatches = sum(1 for w, r in zip(values, default_read) if w != r)
    assert mismatches > 0, ("expected the default pandas float parser to lose precision on at "
                            "least one of these awkward doubles -- if this stops reproducing, "
                            "the round_trip fix may no longer be necessary, but check the real "
                            "mechanism before removing it")
    round_trip_read = pd.read_csv(path, float_precision="round_trip")["close_adj"].tolist()
    assert round_trip_read == values


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


def test_get_price_series_uses_latest_pinned_retrieval_regardless_of_ceiling(tmp_path):
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.0)},
                       retrieved_at="2024-01-02T00:00:00Z", source_hash="h0")
    stored, _ = _apply(stored, {"2023-12-31": (4.0, 4.0, 4.0),
                                "2024-01-01": (5.0, 5.0, 5.0),
                                "2024-01-02": (6.0, 6.0, 6.0)},
                       retrieved_at="2024-01-08T00:00:00Z", source_hash="h1")
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit_price_history_snapshot(conn, clock, store, {"AAPL": stored})
    repository = Repository(conn, store)

    before = repository.get_price_series(
        _price_query(session_date="2024-01-01", ceiling="2024-01-01T23:59:59Z"), snapshot)
    assert [row.date for row in before] == ["2023-12-31", "2024-01-01"]
    assert before[0].close_adj == 4.0
    assert before[0].retrieved_at == "2024-01-08T00:00:00Z"
    assert before[0].source_hash == "h1"
    assert before[-1].close_adj == 5.0
    assert before[-1].source_hash == "h1"
    guard_rows = pd.DataFrame({"retrieved_at": ["2024-01-08T00:00:00Z"]})
    with pytest.raises(ValueError, match="cutoff is required"):
        price_history._chosen_retrieved_at(guard_rows, None)
    with pytest.raises(ValueError, match="cutoff must be omitted"):
        price_history._chosen_retrieved_at(
            guard_rows, "2024-01-01T23:59:59Z", latest_snapshot=True)
    close = repository.get_close("AAPL", "2024-01-01", "2024-01-01T23:59:59Z", snapshot)
    assert close.close_adj == 5.0
    assert close.retrieved_at == "2024-01-08T00:00:00Z"
    assert close.source_hash == "h1"


def test_get_price_series_uses_latest_retrieval_when_ceiling_precedes_all(tmp_path):
    stored, _ = _apply(_empty_stored(), {"2020-01-01": (1.0, 1.0, 1.0), "2020-01-02": (2.0, 2.0, 2.0),
                                        "2020-01-03": (3.0, 3.0, 3.0)},
                       retrieved_at="2024-06-05T00:00:00Z", source_hash="h0")
    stored, _ = _apply(stored, {"2020-01-01": (4.0, 4.0, 4.0), "2020-01-02": (5.0, 5.0, 5.0),
                                "2020-01-03": (6.0, 6.0, 6.0)},
                       retrieved_at="2024-06-06T00:00:00Z", source_hash="h1")
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit_price_history_snapshot(conn, clock, store, {"AAPL": stored})
    repository = Repository(conn, store)
    result = repository.get_price_series(
        _price_query(session_date="2020-01-02", ceiling="2020-01-02T00:00:00Z", lookback=10), snapshot)
    assert [r.close_adj for r in result] == [4.0, 5.0]
    assert result[-1].source_hash == "h1"


def test_get_price_series_drops_a_tombstoned_date(tmp_path):
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.0), "2024-01-02": (2.0, 2.0, 2.0)},
                       retrieved_at="2024-01-03T00:00:00Z")
    # 2024-01-02 dropped -- hand-made tombstone (``diff_retrieval`` refuses an
    # omission instead of writing one since issue #719).
    stored = _tombstone(stored, "2024-01-02", retrieved_at="2024-01-10T00:00:00Z")
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit_price_history_snapshot(conn, clock, store, {"AAPL": stored})
    repository = Repository(conn, store)
    result = repository.get_price_series(
        _price_query(session_date="2024-01-02", ceiling="2024-01-15T00:00:00Z"), snapshot)
    assert [r.date for r in result] == ["2024-01-01"]


def test_get_price_series_refuses_session_after_observation_ceiling(tmp_path):
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


def test_max_object_bytes_bounds_both_typed_reads_before_any_decode(tmp_path, monkeypatch):
    """The optional ``max_object_bytes`` reaches ``Repository.scan`` from
    ``_fetch_rows`` for BOTH typed reads: a cap under the pinned fragment's own
    ``byte_size`` refuses the repository's own bound error
    (``RESULT_LIMIT_EXCEEDED`` / ``object exceeds max_object_bytes``) before a
    fragment is opened, while the default ``None`` keeps today's
    byte-unbounded read intact."""
    stored, _ = _apply(_empty_stored(), {"2024-01-01": (1.0, 1.0, 1.0),
                                         "2024-01-02": (2.0, 2.0, 2.0)},
                       retrieved_at="2024-01-03T00:00:00Z")
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit_price_history_snapshot(conn, clock, store, {"AAPL": stored})
    repository = Repository(conn, store)
    query = _price_query(session_date="2024-01-02", ceiling="2024-01-02T00:00:00Z")
    assert [row.date for row in price_history_query.get_price_series(
        repository, query, snapshot)] == ["2024-01-01", "2024-01-02"]
    monkeypatch.setattr(repository, "_open_parquet",
                        lambda *args, **kwargs: pytest.fail("bounded read decoded a fragment"))
    with pytest.raises(DataError, match="object exceeds max_object_bytes") as caught:
        price_history_query.get_price_series(repository, query, snapshot, max_object_bytes=1)
    assert caught.value.code == "RESULT_LIMIT_EXCEEDED"
    with pytest.raises(DataError, match="object exceeds max_object_bytes") as caught:
        price_history_query.get_close(repository, "AAPL", "2024-01-02",
                                      "2024-01-02T00:00:00Z", snapshot, max_object_bytes=1)
    assert caught.value.code == "RESULT_LIMIT_EXCEEDED"
