import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.event_revisions import (
    EventRevision,
    apply_event_revision,
    event_revision_candidate,
    resolve_event_identity,
)


def _prior(event_id="AAPL_2026-01-01", ticker="AAPL", cluster="AAPL-q1"):
    return {
        "event_id": event_id, "ticker": ticker, "event_date": "2026-01-01",
        "session": "AMC", "event_cluster_id": cluster,
    }


def test_bmo_amc_and_date_correction_preserve_event_identity():
    old = _prior()
    corrected = {
        "ticker": "AAPL", "event_date": "2026-01-02", "session": "BMO",
        "event_cluster_id": "AAPL-q1",
    }
    event_id = resolve_event_identity(corrected, (old,))
    candidate = event_revision_candidate(
        event_id=event_id, row=corrected, source="nasdaq", source_priority=1,
        finality="final", revision_ordinal=2, received_at="2026-01-03T00:00:00Z")
    rows = apply_event_revision(
        EventRevision(candidate=candidate, event_id=event_id, row=corrected), (old,))
    assert rows == ({**corrected, "event_id": old["event_id"]},)


def test_date_only_mapping_and_ambiguous_cluster_are_refused():
    with pytest.raises(DataError) as no_identity:
        resolve_event_identity({"ticker": "AAPL", "event_date": "2026-01-02"}, (_prior(),))
    assert no_identity.value.code == "IDENTITY_CONFLICT"
    with pytest.raises(DataError) as ambiguous:
        resolve_event_identity(
            {"ticker": "AAPL", "event_cluster_id": "AAPL-q1"},
            (_prior("AAPL_1"), _prior("AAPL_2")))
    assert ambiguous.value.code == "IDENTITY_CONFLICT"


def test_explicit_cross_security_identity_is_refused():
    with pytest.raises(DataError) as exc:
        resolve_event_identity(
            {"event_id": "AAPL_2026-01-01", "ticker": "MSFT"}, (_prior(),))
    assert exc.value.code == "IDENTITY_CONFLICT"
