from __future__ import annotations

from datetime import datetime

import pytest

from engine.v2.contracts import (
    CoverageKey,
    CoverageOutcome,
    RevisionCandidate,
    TableContractRef,
    TimeInterval,
)
from engine.v2.data.errors import DataError
from engine.v2.data.incremental import (
    DailyMarketRevision,
    build_completed_coverage,
    daily_market_logical_key,
    merge_daily_market,
    revision_content_hash,
)
from tests.test_v2_data_objects import _contract, _daily_market_rows


def _revision(row, *, revision_id, ordinal=1, finality="final", priority=0,
              deleted=False, source="orats"):
    ticker = row["ticker"]
    session = row["date"].date().isoformat()
    payload_row = None if deleted else row
    candidate = RevisionCandidate(
        revision_id=revision_id,
        logical_key=daily_market_logical_key(ticker, session),
        source=source,
        source_priority=priority,
        finality=finality,
        revision_ordinal=ordinal,
        received_at="2026-09-16T12:00:00.000000Z",
        content_hash=revision_content_hash(
            ticker=ticker, session_date=session, row=payload_row, deleted=deleted),
    )
    return DailyMarketRevision(
        candidate=candidate,
        ticker=ticker,
        session_date=session,
        row=payload_row,
        deleted=deleted,
        raw_receipt_id="raw-1",
        normalization_id="norm-1",
    )


def test_append_correction_and_tombstone_rewrite_only_changed_partition():
    contract = _contract("daily_market")
    prior = _daily_market_rows()
    corrected = dict(prior[0], spot=111.0)
    appended = dict(prior[0], ticker="ZZZ", date=datetime(2025, 1, 2), year=2025)
    revisions = (
        _revision(corrected, revision_id="r-correct", ordinal=2),
        _revision(appended, revision_id="r-append"),
        _revision(prior[1], revision_id="r-delete", ordinal=2, deleted=True),
    )

    merged = merge_daily_market(contract, prior, (), revisions)

    assert [change.revision_kind for change in merged.changes] == [
        "correction", "tombstone", "append"]
    assert merged.changed_partitions == ("2024", "2025")
    assert set(merged.partition_hashes) == {"2024", "2025"}
    assert len(merged.rows) == len(prior)


def test_winner_policy_is_deterministic_and_clean_rebuild_equivalent():
    contract = _contract("daily_market")
    base = _daily_market_rows()[0]
    low = _revision(dict(base, spot=101.0), revision_id="low", priority=1, ordinal=9)
    provisional = _revision(
        dict(base, spot=102.0), revision_id="provisional", priority=0,
        finality="provisional", ordinal=9)
    winner = _revision(
        dict(base, spot=103.0), revision_id="winner", priority=0,
        finality="final", ordinal=1)

    incremental = merge_daily_market(contract, [base], (), (low, provisional, winner))
    reversed_order = merge_daily_market(
        contract, [base], (), (winner, provisional, low))
    clean = merge_daily_market(contract, (), (), (low, provisional, winner))

    assert incremental.rows[0]["spot"] == 103.0
    assert incremental.rows == reversed_order.rows == clean.rows
    assert incremental.partition_hashes == reversed_order.partition_hashes


def test_equal_rank_different_content_refuses():
    contract = _contract("daily_market")
    base = _daily_market_rows()[0]
    left = _revision(dict(base, spot=101.0), revision_id="left")
    right = _revision(dict(base, spot=102.0), revision_id="right")

    with pytest.raises(DataError) as exc:
        merge_daily_market(contract, [base], (), (left, right))
    assert exc.value.code == "IDENTITY_CONFLICT"


def test_noop_has_zero_changes_partitions_and_hashes():
    contract = _contract("daily_market")
    base = _daily_market_rows()[0]
    same = _revision(base, revision_id="same")

    merged = merge_daily_market(contract, [base], (), (same,))

    assert merged.rows == (base,)
    assert merged.changes == ()
    assert merged.changed_partitions == ()
    assert merged.partition_hashes == {}


def test_coverage_requires_every_explicit_denominator_member():
    contract = _contract("daily_market")
    ref = TableContractRef(
        contract_id=contract.contract_id, definition_hash=contract.definition_hash)
    keys = (
        CoverageKey(item_key="AAA", session_date="2026-09-15", ticker="AAA"),
        CoverageKey(item_key="BBB", session_date="2026-09-15", ticker="BBB"),
    )
    one = CoverageOutcome(
        key=keys[0], status="present", receipt_id="raw-a", revision_id="rev-a",
        finality="final")
    partial = build_completed_coverage(
        ref, source="orats", endpoint="summaries",
        interval=TimeInterval(
            column="date", start_inclusive="2026-09-15", end_exclusive="2026-09-16"),
        expected=keys, outcomes=(one,), acquisition_receipt_refs=("raw-a",),
        completed_at="2026-09-16T00:00:00.000000Z")
    complete = build_completed_coverage(
        ref, source="orats", endpoint="summaries",
        interval=partial.interval, expected=keys,
        outcomes=(one, CoverageOutcome(
            key=keys[1], status="legitimate_empty", receipt_id="raw-b",
            revision_id=None, finality="final")),
        acquisition_receipt_refs=("raw-a", "raw-b"),
        completed_at="2026-09-16T00:00:00.000000Z")

    assert partial.state == "incomplete"
    assert partial.completed_at is None
    assert complete.state == "complete"
    assert complete.covered_tickers == ("AAA", "BBB")


def test_tombstone_survives_lower_priority_later_revision():
    contract = _contract("daily_market")
    base = _daily_market_rows()[0]
    tombstone = _revision(base, revision_id="delete", priority=0, deleted=True)
    stale = _revision(
        dict(base, spot=999.0), revision_id="stale", priority=1, ordinal=99)

    first = merge_daily_market(contract, [base], (), (tombstone,))
    retried = merge_daily_market(contract, first.rows, (tombstone,), (stale,))

    assert first.rows == retried.rows == ()
    assert retried.changes == ()


def test_mcap_carries_forward_from_prior_when_session_has_none():
    contract = _contract("daily_market")
    base = _daily_market_rows()[0]  # ticker AAA, date 2024-01-02, mcap_usd=1e9
    prior = [base]
    incoming_row = dict(
        base, date=datetime(2024, 1, 5), mcap_usd=None, mcap_log=None,
        mcap_asof=None, mcap_age_days=None, src_mcap=None)
    incoming = _revision(incoming_row, revision_id="fetched")

    merged = merge_daily_market(contract, prior, (), (incoming,))

    row = next(r for r in merged.rows if r["ticker"] == "AAA"
               and r["date"] == datetime(2024, 1, 5))
    assert row["mcap_usd"] == base["mcap_usd"]
    assert row["mcap_log"] == base["mcap_log"]
    assert row["mcap_asof"] == datetime(2024, 1, 2)
    assert row["mcap_age_days"] == 3.0
    assert row["src_mcap"] == "orats.cores"


def test_mcap_stays_none_when_no_prior_value_is_loaded():
    contract = _contract("daily_market")
    base = _daily_market_rows()[0]
    incoming_row = dict(
        base, ticker="ZZZ", date=datetime(2024, 1, 5), mcap_usd=None, mcap_log=None,
        mcap_asof=None, mcap_age_days=None, src_mcap=None)
    incoming = _revision(incoming_row, revision_id="fetched-zzz")

    merged = merge_daily_market(contract, [base], (), (incoming,))

    row = next(r for r in merged.rows if r["ticker"] == "ZZZ")
    assert row["mcap_usd"] is None
    assert row["mcap_asof"] is None


def test_mcap_carry_preserves_original_observation_date_across_two_hops():
    contract = _contract("daily_market")
    base = _daily_market_rows()[0]  # ticker AAA, date 2024-01-02, mcap_usd=1e9

    # First hop: 2024-01-05 has no cores data; carries from 2024-01-02.
    first_incoming_row = dict(
        base, date=datetime(2024, 1, 5), mcap_usd=None, mcap_log=None,
        mcap_asof=None, mcap_age_days=None, src_mcap=None)
    first_incoming = _revision(first_incoming_row, revision_id="hop-1")
    first_merge = merge_daily_market(contract, [base], (), (first_incoming,))
    hop1_row = next(r for r in first_merge.rows if r["date"] == datetime(2024, 1, 5))
    assert hop1_row["mcap_asof"] == datetime(2024, 1, 2)
    assert hop1_row["src_mcap"] == "orats.cores"

    # Second hop: 2024-01-09 also has no cores data. `prior` for this refresh is
    # `first_merge.rows` -- it only contains the 2024-01-02 original and the
    # 2024-01-05 carried row (no fresh 2024-01-02 row separately). The carried
    # asof must still be 2024-01-02, and the age must be measured from there
    # (7 days), not from 2024-01-05 (which would wrongly give 4 days).
    second_incoming_row = dict(
        base, date=datetime(2024, 1, 9), mcap_usd=None, mcap_log=None,
        mcap_asof=None, mcap_age_days=None, src_mcap=None)
    second_incoming = _revision(second_incoming_row, revision_id="hop-2")
    second_merge = merge_daily_market(
        contract, list(first_merge.rows), (), (second_incoming,))
    hop2_row = next(r for r in second_merge.rows if r["date"] == datetime(2024, 1, 9))

    assert hop2_row["mcap_asof"] == datetime(2024, 1, 2)
    assert hop2_row["mcap_age_days"] == 7.0
    assert hop2_row["src_mcap"] == "orats.cores"


def test_mcap_carries_within_the_same_incoming_batch():
    # Day 1 and day 2 both arrive as NEW revisions in ONE merge call, with nothing
    # previously committed (prior=()). Day 2 has no cores data; it must still carry
    # day 1's fresh value from within this same batch, not only from `prior`.
    contract = _contract("daily_market")
    base = _daily_market_rows()[0]  # ticker AAA, date 2024-01-02, mcap_usd=1e9
    day1 = _revision(base, revision_id="day1")
    day2_row = dict(
        base, date=datetime(2024, 1, 5), mcap_usd=None, mcap_log=None,
        mcap_asof=None, mcap_age_days=None, src_mcap=None)
    day2 = _revision(day2_row, revision_id="day2")

    merged = merge_daily_market(contract, (), (), (day1, day2))

    row = next(r for r in merged.rows if r["date"] == datetime(2024, 1, 5))
    assert row["mcap_usd"] == base["mcap_usd"]
    assert row["mcap_asof"] == datetime(2024, 1, 2)
    assert row["mcap_age_days"] == 3.0
    assert row["src_mcap"] == "orats.cores"


def test_mcap_carries_into_a_retained_row_with_no_cap_of_its_own():
    # A retained revision (an already-committed row being replayed) that itself has
    # mcap_usd=None must still be backfilled from `prior`, the same as a freshly
    # fetched row would be -- an incremental replay must match a clean rebuild.
    contract = _contract("daily_market")
    base = _daily_market_rows()[0]  # ticker BBB-equivalent shape; override ticker below
    earlier = dict(base, ticker="CCC", date=datetime(2024, 1, 2))
    retained_row = dict(
        base, ticker="CCC", date=datetime(2024, 1, 5), mcap_usd=None, mcap_log=None,
        mcap_asof=None, mcap_age_days=None, src_mcap=None)
    retained = _revision(retained_row, revision_id="retained-ccc")

    merged = merge_daily_market(contract, [earlier], (retained,), ())

    row = next(r for r in merged.rows if r["ticker"] == "CCC"
               and r["date"] == datetime(2024, 1, 5))
    assert row["mcap_usd"] == earlier["mcap_usd"]
    assert row["mcap_asof"] == datetime(2024, 1, 2)
    assert row["src_mcap"] == "orats.cores"


def test_mcap_carry_never_rewrites_a_row_with_no_winner_this_build():
    # An already-committed row with no cap of its own (DDD, 2024-01-05) sits in `prior`
    # alongside an eligible earlier observation (DDD, 2024-01-02) with a real cap. Nothing
    # in this build's incoming/retained revisions touches DDD at all -- only an unrelated
    # ticker (EEE) is being appended. DDD 2024-01-05 must come out of the merge byte-for-byte
    # unchanged: no carried cap, and it must not appear in `merged.changes`.
    contract = _contract("daily_market")
    base = _daily_market_rows()[0]
    earlier = dict(base, ticker="DDD", date=datetime(2024, 1, 2))
    untouched = dict(
        base, ticker="DDD", date=datetime(2024, 1, 5), mcap_usd=None, mcap_log=None,
        mcap_asof=None, mcap_age_days=None, src_mcap=None)
    unrelated_append = dict(base, ticker="EEE", date=datetime(2024, 1, 2))
    unrelated = _revision(unrelated_append, revision_id="unrelated-eee")

    merged = merge_daily_market(contract, [earlier, untouched], (), (unrelated,))

    row = next(r for r in merged.rows if r["ticker"] == "DDD"
               and r["date"] == datetime(2024, 1, 5))
    assert row == untouched
    assert all(change.revision_id != "unrelated-eee" or "DDD" not in str(change)
               for change in merged.changes)
    assert {change.revision_id for change in merged.changes} == {"unrelated-eee"}
