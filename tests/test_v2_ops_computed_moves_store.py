"""Direct tests for ``computed_moves_store.py``'s helpers, and an end-to-end
test of ``run_computed_moves_refresh`` itself: a complete unit, a
same-session cached rerun that re-fetches nothing, and a provider failure
mapped to its typed code. The store is still unwired to the nightly job graph
(see ``engine/v2/ops/ARCHITECTURE.md``'s "not yet wired" note) -- these tests
call it directly, the same way its eventual worker will."""
from __future__ import annotations

import pandas as pd
import pytest

from engine.v2.foundation import ArtifactStore, canonical_json
from engine.v2.ops import computed_moves_store
from engine.v2.ops.catalog import transaction
from engine.v2.ops.computed_moves_store import _capture_id_for, _fence_check_for
from engine.v2.ops.errors import OpsError
from engine.v2.ops.incremental_data import RefreshParameters, RefreshUnit
from tests.data_scan_support import commit_tables, contract_for, contract_ref_for, publish_and_inspect
from tests.ops_support import catalog, enqueue_claim


def _unit(ticker: str, day: str) -> RefreshUnit:
    return RefreshUnit(request_id=f"computed_moves:{ticker}:{day}",
                       table_name="computed_moves", partition_key=ticker,
                       expected_keys=(ticker,))


def test_capture_id_is_stable_for_the_same_unit_and_differs_for_a_different_one():
    """Same (ticker, as_of) -> same capture_id, called twice -- independent of
    wall-clock time. This is the bug the ``request_id``-derived id fixes: the
    old code folded in the run's own ``created_at``, so a same-day retry never
    matched the already-logged capture and ``_insert_captures`` would have
    double-logged it."""
    unit = _unit("ABCD", "2026-09-24")
    first = _capture_id_for(unit)
    second = _capture_id_for(unit)
    assert first == second
    assert first.startswith("capture_")
    other_ticker = _capture_id_for(_unit("WXYZ", "2026-09-24"))
    other_day = _capture_id_for(_unit("ABCD", "2026-09-25"))
    assert other_ticker != first
    assert other_day != first


def test_fence_check_for_matches_the_real_verify_fence_and_keeps_the_lease_check(tmp_path):
    """``_fence_check_for`` must build a callable ``verify_fence`` accepts with
    its REAL signature (``conn, attempt_id, fence, now`` -- no
    ``check_lease_time`` keyword; ``engine.v2.ops.lifecycle.verify_fence`` has
    never had one). It must also still enforce the production wall-clock
    lease-expiry check -- never a skipped check, no matter who calls this
    store."""
    conn, clock, supervisor = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, supervisor)
    check = _fence_check_for(claim.attempt_id, claim.fence, clock)

    with transaction(conn):
        job, attempt = check(conn)
        assert job["fence"] == claim.fence
        assert attempt["fence"] == claim.fence

    clock.advance(10 ** 6)  # long past any lease_expires_at
    with transaction(conn):
        with pytest.raises(OpsError) as err:
            check(conn)
        assert err.value.code == "LEASE_LOST"


def test_fence_check_for_with_no_staged_attempt_is_a_noop():
    """No staged attempt (e.g. a manual/ad-hoc invocation with nothing to
    fence against): the returned callable does nothing and returns ``None``,
    matching ``_commit_generation``'s previous inline ternary."""
    check = _fence_check_for(None, None, clock=None)
    assert check(object()) is None


# --------------------------------------------------------------------------
# capture_id must not collide two attempts with genuinely different content
# (CodeRabbit round 2: request_id alone is not a sufficient identity)
# --------------------------------------------------------------------------


def test_capture_id_folds_in_source_hash_so_different_bytes_do_not_collide():
    """Same (ticker, as_of) request_id, but the upstream data genuinely
    differs between two attempts (a same-day yfinance correction/backfill) --
    the identity must differ too, or ``_insert_captures``'s dedup would
    silently keep only the FIRST attempt's outcome even though the second
    attempt's actual captured content was different."""
    unit = _unit("ABCD", "2026-09-24")
    same_a = _capture_id_for(unit, source_hash="hash-one")
    same_b = _capture_id_for(unit, source_hash="hash-one")
    different = _capture_id_for(unit, source_hash="hash-two")
    no_source = _capture_id_for(unit)
    assert same_a == same_b
    assert different != same_a
    assert no_source != same_a
    assert no_source != different


def test_capture_id_folds_in_the_contract_definition_hash():
    """A contract/schema change between two captures of the same unit must
    not silently reuse the old capture identity."""
    unit = _unit("ABCD", "2026-09-24")
    original = _capture_id_for(unit, source_hash="hash-one")
    other_schema = _capture_id_for(unit, source_hash="hash-one",
                                   definition_hash="sha256:" + "0" * 64)
    assert other_schema != original


# --------------------------------------------------------------------------
# run_computed_moves_refresh end to end (CodeRabbit round 2: no direct test
# of the callback existed): a real snapshot/catalog/store fixture, no network.
# --------------------------------------------------------------------------

_EVENTS = contract_for("earnings_events")
_EVENTS_REF = contract_ref_for(_EVENTS)
_DAILY = contract_for("daily_market")
_DAILY_REF = contract_ref_for(_DAILY)

_BDAYS = pd.bdate_range("2024-01-02", "2024-01-31")  # 22 business days
_EVENT_DAYS = [_BDAYS[2], _BDAYS[6], _BDAYS[10], _BDAYS[14], _BDAYS[18]]  # 5 spread-out prints
_AS_OF = "2024-02-05"


def _event_row(ticker: str, event_date) -> dict:
    d = event_date.to_pydatetime()
    return dict(
        event_id=f"{ticker}_{d.date()}", ticker=ticker, event_date=d, year=d.year,
        session="AMC", session_src="orats", annc_tod=None, src_orats=True,
        src_oquants=False, src_nasdaq=False, src_yfinance=False, date_agree=True,
        date_conflict=False, updated_at=None, event_cluster_id=None, claim_count=None,
        reconciliation=None)


def _closes_csv() -> bytes:
    lines = ["Date,Close"] + [f"{d.date()},{100.0 + i}" for i, d in enumerate(_BDAYS)]
    return "\n".join(lines).encode()


def _head_row(conn, scope="shadow"):
    return conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = ?",
        (scope,)).fetchone()


def _build_parent(conn, clock, store, *, events_rows=()):
    tables = {"daily_market": []}
    if events_rows:
        record = publish_and_inspect(store, _EVENTS, _EVENTS_REF, list(events_rows), "2024")
        tables["earnings_events"] = [record]
    else:
        tables["earnings_events"] = []
    contracts = {"earnings_events": _EVENTS, "daily_market": _DAILY}
    commit_tables(conn, clock, tables, contracts, store=store)
    return _head_row(conn)


class _CountingFetcher:
    """Records every ticker it is called for; returns the same complete CSV."""

    def __init__(self, csv_bytes: bytes):
        self._csv = csv_bytes
        self.calls: list[str] = []

    def __call__(self, ticker: str):
        self.calls.append(ticker)
        return self._csv, "complete", {}, None


def _refused_fetcher(_ticker):
    return b"not-a-real-response", "refused", {}, None


def _parameters(head, *, expected_ids, catalog_path, objects_root) -> RefreshParameters:
    return RefreshParameters(
        expected_ids=expected_ids, parent_snapshot_id=head["snapshot_id"],
        refresh_plan_hash="sha256:" + "d" * 64, provider_calls=1,
        catalog_path=str(catalog_path), objects_root=str(objects_root), scope="shadow",
        expected_head_generation=head["generation"], expected_head_snapshot_id=head["snapshot_id"])


def _write_input(root, *, catalog_path, objects_root, head, as_of=_AS_OF):
    root.mkdir(exist_ok=True)
    document = {
        "catalog_path": str(catalog_path), "objects_root": str(objects_root),
        "scope": "shadow", "expected_head_generation": head["generation"],
        "expected_head_snapshot_id": head["snapshot_id"], "as_of": as_of,
    }
    (root / computed_moves_store.INPUT_PATH).write_text(canonical_json(document))


def test_run_computed_moves_refresh_captures_one_complete_unit(tmp_path, monkeypatch):
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAAA"], {}))
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    events_rows = [_event_row("AAAA", d) for d in _EVENT_DAYS]
    head = _build_parent(conn, clock, store, events_rows=events_rows)

    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head)
    fetcher = _CountingFetcher(_closes_csv())
    parameters = _parameters(head, expected_ids=("AAAA",),
                             catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)

    result = computed_moves_store.run_computed_moves_refresh(
        parameters, root, as_of=_AS_OF, fetcher=fetcher)

    assert result.status == "complete"
    assert result.coverage_advanced is True
    assert result.completed_ids == ("AAAA",)
    assert fetcher.calls == ["AAAA"]
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 1
    row = conn.execute(
        "SELECT outcome FROM data_computed_moves_captures WHERE ticker = ?",
        ("AAAA",)).fetchone()
    assert row["outcome"] == "added"


def test_run_computed_moves_refresh_cached_rerun_refetches_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAAA"], {}))
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    events_rows = [_event_row("AAAA", d) for d in _EVENT_DAYS]
    head = _build_parent(conn, clock, store, events_rows=events_rows)

    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head)
    fetcher = _CountingFetcher(_closes_csv())
    parameters = _parameters(head, expected_ids=("AAAA",),
                             catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)
    first = computed_moves_store.run_computed_moves_refresh(
        parameters, root, as_of=_AS_OF, fetcher=fetcher)
    assert first.status == "complete"
    assert fetcher.calls == ["AAAA"]

    new_head = _head_row(conn)
    root2 = tmp_path / "attempt2"
    _write_input(root2, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=new_head)
    parameters2 = _parameters(new_head, expected_ids=("AAAA",),
                              catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)

    second = computed_moves_store.run_computed_moves_refresh(
        parameters2, root2, as_of=_AS_OF, fetcher=fetcher)

    # The rerun re-fetches NOTHING (the point of this test, and the actually
    # verified behavior): the fetcher is called exactly once, for the first
    # run, never again. The result is still "complete", not "noop" -- a
    # separate, tracked, pre-existing gap (see run_computed_moves_refresh's
    # own docstring): computed_at embeds wall-clock time, so a rebuilt row is
    # never byte-identical to the one it replaces even from cached bytes.
    assert second.status == "complete"
    assert second.coverage_advanced is True
    assert fetcher.calls == ["AAAA"]  # unchanged: the second run never re-fetched
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM data_computed_moves_captures WHERE ticker = ?",
        ("AAAA",)).fetchone()[0] == 1  # stable capture_id (round 2 fix) dedups the rerun


def test_run_computed_moves_refresh_provider_failure_maps_to_its_code(tmp_path, monkeypatch):
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["CCCC"], {}))
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _build_parent(conn, clock, store, events_rows=())

    root = tmp_path / "attempt"
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head)
    parameters = _parameters(head, expected_ids=("CCCC",),
                             catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)

    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            parameters, root, as_of=_AS_OF, fetcher=_refused_fetcher)

    assert err.value.code == "SOURCE_INVALID"


# --------------------------------------------------------------------------
# as_of must be validated before any I/O (Opus review finding on a sibling
# PR, #40, applies here too: main's RefreshParameters has no as_of field, so
# this function takes it as an explicit keyword and validates it itself)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_as_of", [
    123,
    123.5,
    True,
    pd.NaT,
    pd.Timestamp("2024-01-01", tz="UTC"),
    "not-a-date",
])
def test_run_computed_moves_refresh_refuses_a_bad_as_of_before_any_io(bad_as_of):
    """``parameters=object()`` and ``root=None`` both raise on the first real
    touch (attribute access / path use) -- so an ``OpsError`` instead of an
    ``AttributeError``/``TypeError`` proves the as_of check runs BEFORE
    ``parameters``, ``root``, or the filesystem are touched at all, exactly
    as ``run_computed_moves_refresh``'s own docstring now claims."""
    with pytest.raises(OpsError) as err:
        computed_moves_store.run_computed_moves_refresh(
            object(), None, as_of=bad_as_of, fetcher=None)
    assert err.value.code == "INVALID_REQUEST"
