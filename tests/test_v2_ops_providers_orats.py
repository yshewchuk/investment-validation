"""S4B: the native ORATS ``daily_market`` fetcher and its refresh wiring.

The fetcher lives in the ops layer (``engine.v2.ops.providers``) because it
raises ops-level acquisition codes; the ops layer injects it into the
data-layer refresh wrapper. No network is touched: every test injects
``http_get``. The end-to-end test runs ``run_daily_market_refresh`` against a
temp catalog/store with the same fixture helpers the existing refresh tests
use.
"""
from __future__ import annotations

import json

import pytest

from engine.v2.data import incremental as data_incremental
from engine.v2.data.catalog import commit_snapshot
from engine.v2.data.errors import DataError
from engine.v2.data.manifests import dataset_manifest, snapshot_ref
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, canonical_json, content_hash
from engine.v2.ops import incremental_data as ops_incremental
from engine.v2.ops.errors import OpsError
from engine.v2.ops.incremental_data import RefreshParameters, _failure_for_refresh_status
from engine.v2.ops.providers.orats_daily_market import orats_daily_market_fetcher
from tests.ops_support import catalog
from tests.test_v2_data_manifests import _DAILY_MARKET_CONTRACT, _DAILY_MARKET_REF

SESSION_DATE = "2026-05-01"
REQUEST_ID = "orats-2026-05-01"
UNIT = {"request_id": REQUEST_ID, "table_name": "daily_market",
        "partition_key": SESSION_DATE, "expected_keys": ["AAA"]}

SUMMARIES_ROW = {
    "ticker": "AAA", "tradeDate": SESSION_DATE,
    "stockPrice": 100.0, "iv10d": 0.30, "iv30d": 0.32,
    "exErnIv10d": 0.29, "exErnIv30d": 0.31, "impliedMove": 0.05,
    "rVol30": 0.28, "skewing": 1.1, "contango": 0.5,
    "fwd90_30": 0.33, "fexErn90_30": 0.34, "ieeEarnEffect": 0.2,
}
CORES_ROW = {"ticker": "AAA", "tradeDate": SESSION_DATE, "mktCap": 1_000_000.0}


def _body(rows):
    return json.dumps({"data": rows, "message": "ok"}).encode()


class _FakeHttp:
    """Endpoint-keyed canned responses; records every ``(url, timeout)`` call."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, url, *, timeout):
        self.calls.append((url, timeout))
        for endpoint, response in self.responses.items():
            if f"/{endpoint}?" in url:
                return response
        raise AssertionError(f"unexpected url {url!r}")


def _ok_fake():
    return _FakeHttp({
        "hist/summaries": (200, {}, _body([SUMMARIES_ROW])),
        "hist/cores": (200, {}, _body([CORES_ROW])),
    })


def test_credential_missing_is_credential_invalid(monkeypatch):
    monkeypatch.delenv("ORATS_API_KEY", raising=False)
    fetcher = orats_daily_market_fetcher(http_get=_ok_fake(), api_key=None)
    with pytest.raises(OpsError) as exc:
        fetcher(dict(UNIT))
    assert exc.value.code == "CREDENTIAL_INVALID"


def test_complete_response_builds_ported_ticker_rows():
    fake = _ok_fake()
    fetcher = orats_daily_market_fetcher(http_get=fake, api_key="test-key")
    assert fetcher.lookback_days == 6

    raw_bytes, kind, meta, rows = fetcher(dict(UNIT))

    assert kind == "complete"
    assert meta == {"summaries_status": 200, "cores_status": 200,
                    "trade_date": SESSION_DATE}
    assert json.loads(raw_bytes) == {
        "summaries": {"data": [SUMMARIES_ROW], "message": "ok"},
        "cores": {"data": [CORES_ROW], "message": "ok"}}
    assert fake.calls == [
        (f"https://api.orats.io/datav2/hist/summaries"
         f"?tradeDate={SESSION_DATE}&token=test-key", 30.0),
        (f"https://api.orats.io/datav2/hist/cores"
         f"?tradeDate={SESSION_DATE}&token=test-key", 30.0)]
    assert len(rows) == 1
    row = rows[0]
    assert row["ticker"] == "AAA"
    assert row["date"] == SESSION_DATE
    assert row["year"] == 2026
    assert row["spot"] == 100.0
    assert row["iv10"] == pytest.approx(30.0)
    assert row["implied_move"] == pytest.approx(5.0)
    assert row["mcap_usd"] == pytest.approx(1e9)
    assert row["mcap_log"] == pytest.approx(20.72326583694641)
    assert row["mcap_asof"] == SESSION_DATE
    assert row["mcap_age_days"] == 0.0
    assert row["implied_reconstructed"] is False
    assert row["src_iv"] == "orats.summaries"
    assert row["src_spot"] == "orats.summaries"
    assert row["src_mcap"] == "orats.cores"


def test_both_endpoints_404_is_transient_source():
    fake = _FakeHttp({"hist/summaries": (404, {}, b"not found"),
                      "hist/cores": (404, {}, b"not found")})
    fetcher = orats_daily_market_fetcher(http_get=fake, api_key="test-key")
    with pytest.raises(OpsError) as exc:
        fetcher(dict(UNIT))
    assert exc.value.code == "TRANSIENT_SOURCE"


def test_unauthorized_is_credential_invalid_and_never_echoes_the_key():
    secret = "orats-super-secret-token"
    fake = _FakeHttp({"hist/summaries": (401, {}, b'{"message": "bad token"}'),
                      "hist/cores": (200, {}, _body([CORES_ROW]))})
    fetcher = orats_daily_market_fetcher(http_get=fake, api_key=secret)
    with pytest.raises(OpsError) as exc:
        fetcher(dict(UNIT))
    assert exc.value.code == "CREDENTIAL_INVALID"
    assert secret not in str(exc.value)


def test_missing_expected_ticker_surfaces_classify_response_partial():
    unit = dict(UNIT, expected_keys=["AAA", "BBB"])
    fetcher = orats_daily_market_fetcher(http_get=_ok_fake(), api_key="test-key")
    with pytest.raises(OpsError) as exc:
        fetcher(unit)
    assert exc.value.code == _failure_for_refresh_status("partial")
    assert "partial" in str(exc.value)


def test_load_data_refresh_callback_injects_the_fetcher_without_network(monkeypatch):
    from engine.v2.ops.providers import orats_daily_market

    def explode(*args, **kwargs):
        raise AssertionError("provider HTTP ran while constructing the callback")

    monkeypatch.delenv("ORATS_API_KEY", raising=False)
    monkeypatch.setattr(orats_daily_market, "_requests_get", explode)
    callback = ops_incremental._load_data_refresh_callback()
    fetcher = callback.keywords["fetcher"]
    assert callable(fetcher)


def test_load_daily_market_fetcher_fails_closed_with_resource_unavailable():
    with pytest.raises(DataError) as exc:
        data_incremental._load_daily_market_fetcher()
    assert exc.value.code == "RESOURCE_UNAVAILABLE"


def _commit_parent(conn, store, clock):
    receipt_ref = content_hash({"s4b": "parent"})
    manifest = dataset_manifest(
        _DAILY_MARKET_REF, (), knowledge_mode="reconstructed",
        coverage_receipt_refs=(receipt_ref,), availability_evidence_refs=())
    snapshot = snapshot_ref(
        {"daily_market": manifest}, calendar_version="cal.v1",
        source_priority_version="prio.v1", finality_receipt_refs=(receipt_ref,))
    commit_snapshot(
        conn, scope="shadow", request_hash=content_hash({"s4b": "base"}),
        contracts=(_DAILY_MARKET_CONTRACT,), objects=(), records=(), manifests=(manifest,),
        snapshot=snapshot, expected_head_snapshot_id=None, expected_head_generation=0,
        receipt_id="s4b-base", attempt_id="s4b-base", fence=1,
        fence_check=lambda _conn: None, clock=clock, store=store)
    return snapshot


def _head(conn):
    return conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = ?",
        ("shadow",)).fetchone()


def test_run_daily_market_refresh_commits_the_orats_fetch_end_to_end(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    parent = _commit_parent(conn, store, clock)
    head = _head(conn)

    root = tmp_path / "attempt"
    root.mkdir()
    document = {
        "catalog_path": str(tmp_path / "ops.sqlite"),
        "objects_root": str(tmp_path),
        "scope": "shadow",
        "expected_head_generation": head["generation"],
        "expected_head_snapshot_id": head["snapshot_id"],
        "table_name": "daily_market",
    }
    (root / "incremental_refresh_input.json").write_text(canonical_json(document))
    (root / "refresh_plan.json").write_text(canonical_json({"fetch_units": [dict(UNIT)]}))
    fetcher = orats_daily_market_fetcher(http_get=_ok_fake(), api_key="test-key")
    parameters = RefreshParameters(
        expected_ids=(REQUEST_ID,), parent_snapshot_id=parent.snapshot_id,
        refresh_plan_hash="sha256:" + "b" * 64, provider_calls=1,
        catalog_path=str(tmp_path / "ops.sqlite"), objects_root=str(tmp_path),
        scope="shadow", expected_head_generation=head["generation"],
        expected_head_snapshot_id=head["snapshot_id"])

    result = data_incremental.run_daily_market_refresh(parameters, root, fetcher=fetcher)

    assert result["status"] == "complete"
    assert result["coverage_advanced"] is True
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 1
    committed = Repository(conn).resolve_full(result["candidate_snapshot_id"])
    contract = next(item for item in committed.contracts if item.table_name == "daily_market")
    fragment_ids = {ref.fragment_id
                    for ref in committed.table_manifests["daily_market"].fragment_refs}
    records = [record for record in committed.records if record.fragment_id in fragment_ids]
    rows = data_incremental.load_daily_market_rows(store, records, contract)
    assert [row["ticker"] for row in rows] == ["AAA"]
    assert rows[0]["spot"] == 100.0
    assert rows[0]["mcap_usd"] == pytest.approx(1e9)


def test_refresh_plan_table_name_mismatch_is_contract_mismatch(tmp_path):
    document = {"catalog_path": str(tmp_path / "ops.sqlite"),
                "objects_root": str(tmp_path), "table_name": "daily_market"}
    (tmp_path / "refresh_plan.json").write_text(canonical_json({
        "fetch_units": [{"request_id": "u1", "table_name": "option_chains",
                         "partition_key": SESSION_DATE, "expected_keys": ["AAA"]}]}))
    with pytest.raises(DataError) as exc:
        data_incremental._acquire_refresh_units(None, tmp_path, document, None)
    assert exc.value.code == "CONTRACT_MISMATCH"