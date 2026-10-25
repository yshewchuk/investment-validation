"""Consumer admission and predecode byte bounds over native synthetic daily objects."""
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from engine.v2.data import objects
from engine.v2.data.errors import DataError
from engine.v2.data.price_history_table import PRICE_HISTORY_TABLE_NAME
from engine.v2.research import magnitude_panel_daily as daily
from tests.v2.research.test_magnitude_panel_daily import DAILY, START, build, native  # noqa: F401


@pytest.mark.parametrize("cap", [None, 0, -1, True, 1.5, "1024"])
def test_invalid_byte_caps_refuse_before_repository(native, monkeypatch, cap):
    monkeypatch.setattr(native.repository, "resolve", lambda *a: pytest.fail("unbounded admission accessed repository"))
    with pytest.raises(DataError, match="max_object_bytes must be a positive integer") as caught:
        build(native, max_object_bytes=cap)
    assert caught.value.code == "QUERY_NOT_BOUNDED"


def test_every_ticker_year_read_gets_context_bound_cap(native, monkeypatch):
    calls, original = [], native.repository.scan
    def scan(query, *, table_name, **kwargs):
        if "spot" in query.columns:
            calls.append(kwargs["max_object_bytes"])
        yield from original(query, table_name=table_name, **kwargs)
    monkeypatch.setattr(native.repository, "scan", scan)
    first = build(native, max_object_bytes=2**20)
    assert calls == [2**20] * 3  # SYNTH, three years, shared by every horizon (SPY daily spot is no longer read).
    calls.clear()
    second = build(native, max_object_bytes=2**21)
    assert calls == [2**21] * 3
    assert first["context"]["max_object_bytes"] == 2**20
    assert second["context"]["max_object_bytes"] == 2**21
    assert first["context"]["context_id"] != second["context"]["context_id"]
    assert [(r["daily_market"], r["spy"]) for r in first["rows"]] == [
        (r["daily_market"], r["spy"]) for r in second["rows"]]


def test_regime_read_forwards_the_context_byte_cap(native, monkeypatch):
    """``_read_regime_series`` bounds its ^GSPC scan with the materialization's
    own ``max_object_bytes``, exactly like the daily-market source reads."""
    calls, original = [], native.repository.scan
    def scan(query, *, table_name, **kwargs):
        if table_name == PRICE_HISTORY_TABLE_NAME:
            calls.append(kwargs["max_object_bytes"])
        yield from original(query, table_name=table_name, **kwargs)
    monkeypatch.setattr(native.repository, "scan", scan)
    build(native, max_object_bytes=2**20)
    build(native, max_object_bytes=2**21)
    assert calls == [2**20, 2**21]


def test_regime_object_cap_refuses_before_value_decode(native, monkeypatch):
    """A tiny cap refuses the regime read with the repository's own bound error
    before any fragment object is read or decoded."""
    monkeypatch.setattr(objects, "read_verified_bytes",
        lambda *a, **k: pytest.fail("oversized regime object read"))
    monkeypatch.setattr(native.repository, "_open_parquet",
        lambda *a, **k: pytest.fail("oversized regime object opened"))
    with pytest.raises(DataError, match="object exceeds max_object_bytes") as caught:
        daily._read_regime_series(native.repository, native.regime, START, "2024-08-19", 1000, 1)
    assert caught.value.code == "RESULT_LIMIT_EXCEEDED"


@pytest.mark.parametrize("ticker", ["SPY", "SYNTH"])
def test_physical_object_cap_refuses_before_read(native, monkeypatch, ticker):
    monkeypatch.setattr(objects, "read_verified_bytes", lambda *a, **k: pytest.fail("oversized object read"))
    monkeypatch.setattr(native.repository, "_open_parquet", lambda *a, **k: pytest.fail("oversized object opened"))
    with pytest.raises(DataError, match="object exceeds max_object_bytes") as caught:
        daily._source_rows(native.repository, native.snapshot, ticker, START, "2024-08-15", 1000, 1)
    assert caught.value.code == "RESULT_LIMIT_EXCEEDED"


@pytest.mark.parametrize("ticker", ["SPY", "SYNTH"])
def test_expanded_object_cap_refuses_before_value_decode(native, monkeypatch, ticker):
    record = next(r for r in native.repository.fragment_records(native.snapshot, "daily_market")
                  if r.partition_key == "2022")  # Shared-year artifact includes both tickers.
    payload = objects.read_verified_bytes(native.repository._store, record.object_ref, max_bytes=2**20)
    metadata = pq.read_metadata(pa.BufferReader(payload))
    cap = sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups)) - 1
    assert record.object_ref.byte_size < cap
    monkeypatch.setattr(native.repository, "_iter_batches", lambda *a: pytest.fail("oversized values decoded"))
    with pytest.raises(DataError, match="parquet exceeds max_object_bytes when uncompressed") as caught:
        daily._source_rows(native.repository, native.snapshot, ticker, START, "2024-08-15", 1000, cap)
    assert caught.value.code == "RESULT_LIMIT_EXCEEDED"


@pytest.mark.parametrize("quote,status,missing", [(3.0, "present", {}), (None, "absent", {"implied_move": "no_quote"})])
def test_literal_quote_presence_and_absence(quote, status, missing):
    row = dict.fromkeys(DAILY, 1.0)
    row.update(ticker="SYNTH", date="2024-01-02", mcap_asof="2024-01-02", implied_move=quote)
    block = daily._daily([row], [row["date"]], row["date"])
    assert block["quote_status"] == status
    assert block["values"]["implied_move"] == quote
    assert block["missing"] == missing


def test_missing_iv_surface_is_filtered_before_daily_anchor(native, monkeypatch):
    original = daily._source_rows
    def no_surface(*args, **kwargs):
        return [{**row, "src_iv": None} for row in original(*args, **kwargs)]
    monkeypatch.setattr(daily, "_source_rows", no_surface)
    result = build(native)
    for row in result["rows"]:
        assert row["daily_market"] == {"values": dict.fromkeys(DAILY), "anchor": None,
            "quote_status": "unavailable", "missing": dict.fromkeys(DAILY, "unavailable")}
        assert row["spy"]["values"]["spy_ret21"] is not None
