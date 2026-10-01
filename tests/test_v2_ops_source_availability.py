"""Closed quote EOD availability preflight.

Every call to ``verify_eod_availability`` must refuse: the registered source
completion/finality verifier does not exist yet, and no readable evidence
artifact, matching clock, empty domain or checkpoint-shaped document may be
mistaken for proof. These tests drive the real data catalog, the real
``Repository`` and the real ``ArtifactStore`` with public synthetic
option_chains fixtures; the only substitutions anywhere are a narrow
``read_verified`` call counter used to pin the oversized-candidate allocation
boundary, and a narrow ``os.read`` spy that wraps the real reader to record
requested lengths and actual bytes returned against the registered bound —
never proof logic.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.data import catalog, manifests  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import ArtifactError  # noqa: E402
from engine.v2.ops.checkpoints import artifact, register_artifact, registered_artifact  # noqa: E402
from engine.v2.ops.errors import OpsError  # noqa: E402
from engine.v2.ops.source_availability import verify_eod_availability  # noqa: E402
from tests.data_scan_support import (  # noqa: E402
    catalog_and_store,
    contract_for,
    contract_ref_for,
    fake_hash,
    publish_and_inspect,
)

_TABLE = "option_chains"
_SESSION = "2024-01-05"
_PRIOR_SESSION = "2024-01-04"
_FUTURE_SESSION = "2024-01-06"
_DECISION = "2024-01-05T21:00:00.000000Z"
_UNAVAILABLE = "registered source completion/finality verifier is unavailable"
_TAMPERED = "availability evidence failed full-byte verification"
_EVIDENCE_SCHEMA = "quote_availability_evidence.v1"

_CHAINS = contract_for("option_chains")
_CHAINS_REF = contract_ref_for(_CHAINS)

_SOURCE_CLAIM = json.dumps({
    "producer": "orats",
    "available_at": "2024-01-05T20:00:00.000000Z",
    "finality": "final",
    "session": _SESSION,
}).encode()
_CHECKPOINT_CLAIM = json.dumps({
    "schema_version": "checkpoint_receipt.v1",
    "stage_id": "quote_eod",
    "validation_refs": [],
    "producer_attempt_id": "att-fake",
    "producer_fence": 1,
    "committed_at": _DECISION,
}).encode()


def _chain_rows() -> list[dict]:
    obs_date = datetime(2024, 1, 5)
    expiry = datetime(2024, 2, 16)
    return [dict(ticker="AAA", obs_date=obs_date, year=obs_date.year, expiry=expiry,
                 dte=(expiry - obs_date).days, strike=100.0, right="C", bid=1.0, ask=1.2,
                 mid=1.1, iv=30.0, delta=0.5, spot=100.0, src="orats", src_file="f.parquet",
                 chain_kind="entry", volume=None, open_interest=None, bid_size=None,
                 ask_size=None, quote_repaired=False)]


def _fragment(store):
    return publish_and_inspect(store, _CHAINS, _CHAINS_REF, _chain_rows(), "2024")


def _register(conn, clock, store, payload: bytes = b"{}"):
    ref = store.publish_bytes(payload, schema_ref=_EVIDENCE_SCHEMA)
    register_artifact(conn, ref, None, clock)
    return ref


def _commit(conn, clock, store, *, availability, finality, knowledge_mode="observed",
            empty=False, scope="shadow", receipt_id="r1", expected=(None, 0)):
    records = [] if empty else [_fragment(store)]
    manifest = manifests.dataset_manifest(
        _CHAINS_REF, records, knowledge_mode=knowledge_mode,
        coverage_receipt_refs=(fake_hash("coverage"),),
        availability_evidence_refs=availability)
    snapshot = manifests.snapshot_ref(
        {_TABLE: manifest}, calendar_version="cal.v1", source_priority_version="prio.v1",
        finality_receipt_refs=finality)
    receipt = catalog.commit_snapshot(
        conn, scope=scope, request_hash=fake_hash(f"{receipt_id}-request"),
        contracts=[_CHAINS], objects=[r.object_ref for r in records], records=records,
        manifests=[manifest], snapshot=snapshot, expected_head_snapshot_id=expected[0],
        expected_head_generation=expected[1], receipt_id=receipt_id,
        attempt_id=f"att-{receipt_id}", fence=1, fence_check=lambda _conn: None,
        clock=clock, store=store)
    return receipt.snapshot_ref


class _RepositorySpy:
    """A repository that fails loudly if any read happens; validates ordering."""

    def __init__(self):
        self.calls = 0

    def resolve_full(self, snapshot_id):
        self.calls += 1
        raise AssertionError("repository must not be read")


def _refuses(conn, store, snapshot, code, *, session=_SESSION, decision=_DECISION, table=_TABLE,
             repository=None):
    with pytest.raises(OpsError) as err:
        verify_eod_availability(conn, store, repository or Repository(conn), snapshot,
                                table_name=table, session_date=session, decision_at=decision)
    assert err.value.code == code, str(err.value)
    return err.value.problem.message


def _watch_reads(store, monkeypatch):
    calls = []
    real = store.read_verified

    def spy(ref):
        calls.append(ref.artifact_id)
        return real(ref)

    monkeypatch.setattr(store, "read_verified", spy)
    return calls


# --------------------------------------------------------------------------
# no affirmative return, ever
# --------------------------------------------------------------------------


def test_closed_preflight_still_refuses_and_reads_every_candidate(tmp_path, monkeypatch):
    conn, clock, store = catalog_and_store(tmp_path)
    availability = _register(conn, clock, store, b'{"evidence": "availability"}')
    finality = _register(conn, clock, store, b'{"evidence": "finality"}')
    snapshot = _commit(conn, clock, store, availability=(availability.artifact_id,),
                       finality=(finality.artifact_id,))
    calls = _watch_reads(store, monkeypatch)
    assert _refuses(conn, store, snapshot, "VALIDATION_FAILED") == _UNAVAILABLE
    assert sorted(calls) == sorted([availability.artifact_id, finality.artifact_id])


@pytest.mark.parametrize("session", [_SESSION, _PRIOR_SESSION])
def test_session_alone_never_establishes_availability(tmp_path, session):
    conn, clock, store = catalog_and_store(tmp_path)
    evidence = _register(conn, clock, store)
    snapshot = _commit(conn, clock, store, availability=(evidence.artifact_id,),
                       finality=(evidence.artifact_id,))
    assert _refuses(conn, store, snapshot, "VALIDATION_FAILED", session=session) == _UNAVAILABLE


def test_readable_source_shaped_json_still_refuses(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    claim = _register(conn, clock, store, _SOURCE_CLAIM)
    finality = _register(conn, clock, store, b'{"finality": true}')
    snapshot = _commit(conn, clock, store, availability=(claim.artifact_id,),
                       finality=(finality.artifact_id,))
    assert _refuses(conn, store, snapshot, "VALIDATION_FAILED") == _UNAVAILABLE


def test_checkpoint_shaped_json_with_empty_validation_refs_still_refuses(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    checkpoint = _register(conn, clock, store, _CHECKPOINT_CLAIM)
    snapshot = _commit(conn, clock, store, availability=(checkpoint.artifact_id,),
                       finality=(checkpoint.artifact_id,))
    assert _refuses(conn, store, snapshot, "VALIDATION_FAILED") == _UNAVAILABLE


# --------------------------------------------------------------------------
# missing proof and empty domains never manufacture a positive
# --------------------------------------------------------------------------


def test_empty_availability_evidence_refuses(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    finality = _register(conn, clock, store)
    snapshot = _commit(conn, clock, store, availability=(), finality=(finality.artifact_id,),
                       knowledge_mode="reconstructed")
    assert _refuses(conn, store, snapshot, "VALIDATION_FAILED") \
        == "pinned availability evidence is empty"


def test_empty_finality_evidence_refuses_before_reads(tmp_path, monkeypatch):
    conn, clock, store = catalog_and_store(tmp_path)
    availability = _register(conn, clock, store)
    snapshot = _commit(conn, clock, store, availability=(availability.artifact_id,),
                       finality=())
    calls = _watch_reads(store, monkeypatch)
    assert _refuses(conn, store, snapshot, "VALIDATION_FAILED") \
        == "pinned finality evidence is empty"
    assert calls == []


def test_zero_row_dataset_with_missing_proof_refuses(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    finality = _register(conn, clock, store)
    snapshot = _commit(conn, clock, store, availability=(), finality=(finality.artifact_id,),
                       knowledge_mode="reconstructed", empty=True)
    assert _refuses(conn, store, snapshot, "VALIDATION_FAILED") \
        == "pinned availability evidence is empty"


def test_zero_row_dataset_with_unregistered_proof_refuses(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit(conn, clock, store, availability=("art_" + "0" * 32,),
                       finality=("art_" + "1" * 32,), empty=True)
    assert _refuses(conn, store, snapshot, "INTEGRITY_FAILED")


# --------------------------------------------------------------------------
# pinned snapshot identity: exact equality, never the mutable head
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mutate", [
    lambda snap: dataclasses.replace(snap, calendar_version="cal.v2"),
    lambda snap: dataclasses.replace(snap, source_priority_version="prio.v2"),
    lambda snap: dataclasses.replace(snap, finality_receipt_refs=(fake_hash("other"),)),
])
def test_same_id_different_content_refuses(tmp_path, mutate):
    conn, clock, store = catalog_and_store(tmp_path)
    evidence = _register(conn, clock, store)
    snapshot = _commit(conn, clock, store, availability=(evidence.artifact_id,),
                       finality=(evidence.artifact_id,))
    supplied = mutate(snapshot)
    assert supplied.snapshot_id == snapshot.snapshot_id
    assert _refuses(conn, store, supplied, "VALIDATION_FAILED") \
        == "the pinned snapshot does not resolve to itself"


def test_unknown_snapshot_id_refuses_as_validation_failure(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    evidence = _register(conn, clock, store)
    snapshot = _commit(conn, clock, store, availability=(evidence.artifact_id,),
                       finality=(evidence.artifact_id,))
    unknown = dataclasses.replace(snapshot, snapshot_id="snap_" + "0" * 32)
    with pytest.raises(OpsError) as err:
        verify_eod_availability(conn, store, Repository(conn), unknown, table_name=_TABLE,
                                session_date=_SESSION, decision_at=_DECISION)
    assert err.value.code == "VALIDATION_FAILED"
    assert err.value.problem.message == "the pinned snapshot is not registered"
    assert isinstance(err.value.__cause__, DataError)
    assert err.value.__cause__.code == "SNAPSHOT_NOT_FOUND"


def test_moved_head_leaves_pinned_snapshot_authoritative(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    pinned_evidence = _register(conn, clock, store, b'{"evidence": "pinned"}')
    first = _commit(conn, clock, store, availability=(pinned_evidence.artifact_id,),
                    finality=(pinned_evidence.artifact_id,), receipt_id="r1")
    moved = _commit(conn, clock, store, availability=("art_" + "f" * 32,),
                    finality=("art_" + "e" * 32,), receipt_id="r2",
                    expected=(first.snapshot_id, 1))
    assert moved.snapshot_id != first.snapshot_id
    assert _refuses(conn, store, first, "VALIDATION_FAILED") == _UNAVAILABLE
    assert _refuses(conn, store, moved, "INTEGRITY_FAILED")


# --------------------------------------------------------------------------
# request form refuses before any repository read
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [
    123, True, date(2024, 1, 5), datetime(2024, 1, 5, 21, tzinfo=timezone.utc),
    "2024-1-5", "not-a-date", "20240105",
])
def test_malformed_session_refuses_before_repository_reads(bad):
    spy = _RepositorySpy()
    _refuses(None, None, None, "INVALID_REQUEST", session=bad, repository=spy)
    assert spy.calls == 0


@pytest.mark.parametrize("bad", [
    1.0, False, datetime(2024, 1, 5, 21, tzinfo=timezone.utc), "2024-01-05T21:00:00Z",
    "2024-01-05T21:00:00.000000+00:00", "2024-01-05T21:00:00.000000",
    "2024-01-05 21:00:00.000000Z",
])
def test_noncanonical_decision_refuses_before_repository_reads(bad):
    spy = _RepositorySpy()
    _refuses(None, None, None, "INVALID_REQUEST", decision=bad, repository=spy)
    assert spy.calls == 0


def test_future_session_refuses_before_repository_reads():
    spy = _RepositorySpy()
    _refuses(None, None, None, "INVALID_REQUEST", session=_FUTURE_SESSION, repository=spy)
    assert spy.calls == 0


@pytest.mark.parametrize("table", ["securities", "option_chain", 7, None])
def test_unsupported_table_refuses(table):
    spy = _RepositorySpy()
    _refuses(None, None, None, "INVALID_REQUEST", table=table, repository=spy)
    assert spy.calls == 0


def test_non_snapshot_handle_refuses_before_repository_reads():
    spy = _RepositorySpy()
    _refuses(None, None, object(), "INVALID_REQUEST", repository=spy)
    assert spy.calls == 0


# --------------------------------------------------------------------------
# candidate artifacts: registration, bytes, duplication, allocation bound
# --------------------------------------------------------------------------


def test_unregistered_candidate_refuses(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit(conn, clock, store, availability=("art_" + "0" * 32,),
                       finality=("art_" + "1" * 32,))
    assert _refuses(conn, store, snapshot, "INTEGRITY_FAILED")


def test_duplicate_availability_ref_refuses_before_reads(tmp_path, monkeypatch):
    conn, clock, store = catalog_and_store(tmp_path)
    evidence = _register(conn, clock, store)
    snapshot = _commit(conn, clock, store,
                       availability=(evidence.artifact_id, evidence.artifact_id),
                       finality=(evidence.artifact_id,))
    calls = _watch_reads(store, monkeypatch)
    assert _refuses(conn, store, snapshot, "VALIDATION_FAILED") \
        == "pinned availability evidence repeats an artifact"
    assert calls == []


def test_duplicate_finality_ref_refuses_before_reads(tmp_path, monkeypatch):
    conn, clock, store = catalog_and_store(tmp_path)
    availability = _register(conn, clock, store, b'{"evidence": "a"}')
    finality = _register(conn, clock, store, b'{"evidence": "f"}')
    snapshot = _commit(conn, clock, store, availability=(availability.artifact_id,),
                       finality=(finality.artifact_id, finality.artifact_id))
    calls = _watch_reads(store, monkeypatch)
    assert _refuses(conn, store, snapshot, "VALIDATION_FAILED") \
        == "pinned finality evidence repeats an artifact"
    assert calls == []


def test_cross_carrier_shared_ref_is_read_once(tmp_path, monkeypatch):
    conn, clock, store = catalog_and_store(tmp_path)
    evidence = _register(conn, clock, store)
    snapshot = _commit(conn, clock, store, availability=(evidence.artifact_id,),
                       finality=(evidence.artifact_id,))
    calls = _watch_reads(store, monkeypatch)
    assert _refuses(conn, store, snapshot, "VALIDATION_FAILED") == _UNAVAILABLE
    assert calls == [evidence.artifact_id]


def test_modified_artifact_bytes_fail_full_byte_verification(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    evidence = _register(conn, clock, store, _SOURCE_CLAIM)
    snapshot = _commit(conn, clock, store, availability=(evidence.artifact_id,),
                       finality=(evidence.artifact_id,))
    path = store.verify(evidence)
    os.chmod(path, 0o644)
    original = path.read_bytes()
    path.write_bytes(original + b"!")
    with pytest.raises(OpsError) as err:
        verify_eod_availability(conn, store, Repository(conn), snapshot, table_name=_TABLE,
                                session_date=_SESSION, decision_at=_DECISION)
    assert err.value.code == "INTEGRITY_FAILED"
    assert err.value.problem.message == _TAMPERED
    assert isinstance(err.value.__cause__, ArtifactError)
    path.write_bytes(original)
    assert _refuses(conn, store, snapshot, "VALIDATION_FAILED") == _UNAVAILABLE


def test_registered_accessor_returns_ref_without_byte_integrity(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    evidence = _register(conn, clock, store, _SOURCE_CLAIM)
    snapshot = _commit(conn, clock, store, availability=(evidence.artifact_id,),
                       finality=(evidence.artifact_id,))
    path = store.verify(evidence)
    os.chmod(path, 0o644)
    original = path.read_bytes()
    path.write_bytes(original + b"!")
    try:
        assert registered_artifact(conn, evidence.artifact_id) == evidence
        with pytest.raises(ArtifactError) as err:
            artifact(conn, store, evidence.artifact_id)
        assert err.value.code == "INTEGRITY_FAILED"
        assert _refuses(conn, store, snapshot, "INTEGRITY_FAILED") == _TAMPERED
    finally:
        path.write_bytes(original)
    assert _refuses(conn, store, snapshot, "VALIDATION_FAILED") == _UNAVAILABLE


def test_oversized_candidate_refused_before_any_byte_access(tmp_path, monkeypatch):
    conn, clock, store = catalog_and_store(tmp_path)
    oversized = _register(conn, clock, store, b"x" * ((1 << 20) + 1))
    snapshot = _commit(conn, clock, store, availability=(oversized.artifact_id,),
                       finality=(oversized.artifact_id,))

    def boom(*_args, **_kwargs):
        raise AssertionError("oversized evidence must not touch store bytes")

    monkeypatch.setattr(store, "verify", boom)
    monkeypatch.setattr(store, "read_verified", boom)
    assert _refuses(conn, store, snapshot, "VALIDATION_FAILED") \
        == "availability evidence exceeds the 1 MiB limit"


def _enlarge(path, original, extra_bytes):
    """Grow the real backing file past the registered bound; the ref is untouched."""
    os.chmod(path, 0o644)
    path.write_bytes(original + b"!" * extra_bytes)


def test_backing_file_grown_past_registered_ref_refuses_before_full_accumulation(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    evidence = _register(conn, clock, store, _SOURCE_CLAIM)
    snapshot = _commit(conn, clock, store, availability=(evidence.artifact_id,),
                       finality=(evidence.artifact_id,))
    path = store.verify(evidence)
    original = path.read_bytes()
    _enlarge(path, original, (1 << 20) + 1)
    assert path.stat().st_size > (1 << 20)
    try:
        with pytest.raises(OpsError) as err:
            verify_eod_availability(conn, store, Repository(conn), snapshot, table_name=_TABLE,
                                    session_date=_SESSION, decision_at=_DECISION)
        assert err.value.code == "INTEGRITY_FAILED"
        assert err.value.problem.message == _TAMPERED
        assert isinstance(err.value.__cause__, ArtifactError)
        assert err.value.__cause__.code == "INTEGRITY_FAILED"
    finally:
        os.chmod(path, 0o644)
        path.write_bytes(original)
    assert _refuses(conn, store, snapshot, "VALIDATION_FAILED") == _UNAVAILABLE


def test_registered_bound_caps_every_read_request_and_total_accumulation(tmp_path, monkeypatch):
    conn, clock, store = catalog_and_store(tmp_path)
    evidence = _register(conn, clock, store, _SOURCE_CLAIM)
    path = store.verify(evidence)
    original = path.read_bytes()
    _enlarge(path, original, (1 << 20) + 1)

    requests: list[int] = []
    returned: list[int] = []
    real_read = os.read

    def spy(fd, size):
        requests.append(size)
        chunk = real_read(fd, size)
        returned.append(len(chunk))
        return chunk

    monkeypatch.setattr(os, "read", spy)
    try:
        with pytest.raises(ArtifactError) as err:
            store.read_verified(evidence)
    finally:
        monkeypatch.undo()
        os.chmod(path, 0o644)
        path.write_bytes(original)
    assert err.value.code == "INTEGRITY_FAILED"
    bound = evidence.byte_size
    seen = 0
    for requested, got in zip(requests, returned):
        assert requested <= bound - seen + 1
        seen += got
    assert seen <= bound + 1
    assert sum(returned) < (1 << 20)
    assert store.read_verified(evidence) == original


def test_growth_after_open_is_caught_by_the_one_byte_probe(tmp_path, monkeypatch):
    conn, clock, store = catalog_and_store(tmp_path)
    evidence = _register(conn, clock, store, _SOURCE_CLAIM)
    path = store.verify(evidence)
    original = path.read_bytes()
    os.chmod(path, 0o644)

    requests: list[int] = []
    returned: list[int] = []
    real_read = os.read

    def spy(fd, size):
        if len(requests) == 1:
            # the fd is already open at the registered size; bytes appear now
            with open(path, "ab") as sink:
                sink.write(b"!" * ((1 << 20) + 1))
        requests.append(size)
        chunk = real_read(fd, size)
        returned.append(len(chunk))
        return chunk

    monkeypatch.setattr(os, "read", spy)
    try:
        with pytest.raises(ArtifactError) as err:
            store.read_verified(evidence)
    finally:
        monkeypatch.undo()
        path.write_bytes(original)
    assert err.value.code == "INTEGRITY_FAILED"
    bound = evidence.byte_size
    assert requests == [bound + 1, 1]
    assert returned == [bound, 1]
    assert sum(returned) == bound + 1
    assert store.read_verified(evidence) == original
