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
