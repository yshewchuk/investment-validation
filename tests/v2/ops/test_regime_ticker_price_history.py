"""The ``REGIME_TICKER`` (^GSPC), the single source for the nine ``spy_*``
panel regime features, is a declared ``price_history`` dependency: capture
ingests the legacy ``gspc_daily.csv`` (``source_kind="legacy_gspc_csv"``) as
one more ticker with the same required-ticker / tombstone / no-drop-history
rules, and ``SOURCE_NOT_FOUND`` refuses before any write when a required
^GSPC is absent from both the sources and the prior dataset version. The
fixture csv's Adj Close/Close/High/Low/Open columns are deliberately all
distinct (a capture reading any column but Close fails), and both stored
closes must equal Close exactly (^GSPC is never dividend-adjusted). Synthetic
px trees and a synthetic three-header gspc csv only -- no network, no real
market data. The shared fixture builders (``_base_snapshot``,
``_scan_price_history``, px writes) are local copies here rather than
cross-module imports of another test file's private helpers.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.data.pulls.gspc_refresh import HEADER_LINES
from engine.v2.contracts import DataQuery, KeyPredicate
from engine.v2.data import reference_catalog
from engine.v2.data.price_history_table import (
    CALENDAR_TICKER,
    PRICE_HISTORY_TABLE_NAME,
    REGIME_TICKER,
    REQUIRED_CAPTURE_TICKERS,
)
from engine.v2.data.reference_catalog import ReferenceInput
from engine.v2.data.repository import Repository
from engine.v2.ops import cli, nightly_trigger, price_history_store
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError
from engine.v2.ops.price_history_store import capture
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)
from tests.v2.ops.test_planned_price_refresh import (
    SESSION,
    install_fetcher,
    isolated_legacy_sources,
    shadow_root,
)

_GSPC_DATES = ("2024-01-02", "2024-01-03", "2024-01-04")
_GSPC_CLOSES = (4700.913, 4742.83, 4756.01)  # round_trip-sensitive on purpose
_SPY_DATES = ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
              "2024-01-08", "2024-01-09")

#: Local copy of ``price_history_store``'s private read ceiling (32 MiB) --
#: a test must not import another module's private names; keep in sync with
#: that module's single ``gspc_daily.csv`` cap.
_GSPC_MAX_BYTES = 32 * 1024 * 1024

_SEC = contract_for("securities")
_SEC_REF = contract_ref_for(_SEC)


def _securities_row(ticker: str, year: int) -> dict:
    return dict(ticker=ticker, year=year, first_date=None, last_date=None, mcap_usd=1.5e9,
                mcap_log=21.1, mcap_raw=1.5, mcap_unit_era="billions", mcap_quantized=False,
                n_obs=250, src="orats")


def _base_snapshot(tmp_path):
    """A committed snapshot whose only table is ``securities`` -- the base a
    price_history capture extends (local copy of the shared fixture)."""
    conn, clock, store = catalog_and_store(tmp_path)
    record = publish_and_inspect(store, _SEC, _SEC_REF, [_securities_row("AAPL", 2024)], "2024")
    snap = commit_tables(conn, clock, {"securities": [record]}, {"securities": _SEC},
                         scope="shadow", receipt_id="base-r1")
    ref = ReferenceInput(kind="calendar", legacy_path="calendar.csv", object_id="art_cal",
                         content_hash="sha256:" + "cd" * 32, byte_size=5)
    with transaction(conn):
        reference_catalog.insert_reference_inputs(conn, "base-r1", [ref])
    return conn, clock, store, snap


def _write_px(source_root, ticker: str, rows: dict) -> Path:
    """One synthetic legacy px csv (rows: ``{date: close}``); local copy of
    the shared fixture's px writer."""
    directory = Path(source_root) / "earnings_predictions" / "data" / "raw" / "yfinance"
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{"date": d, "close_adj": v, "close_raw": v, "high_raw": v}
                          for d, v in sorted(rows.items())])
    path = directory / f"px_{ticker}.csv"
    frame.to_csv(path, index=False)
    return path


def _write_sources(source_root, *, with_spy: bool) -> None:
    """An AAPL-only px tree, optionally with SPY rows on consecutive weekday
    dates (local copy of the calendar-ticker fixture's source writer)."""
    _write_px(source_root, "AAPL", {"2024-01-02": 185.0, "2024-01-03": 186.0})
    if with_spy:
        _write_px(source_root, "SPY", {d: 470.0 + i for i, d in enumerate(_SPY_DATES)})


def _scan_price_history(conn, store, snapshot_id, *, ticker: str) -> pd.DataFrame:
    """Every stored ``price_history`` row for one ticker of one committed
    snapshot (local copy of the shared scan helper)."""
    repository = Repository(conn, store)
    snapshot = repository.resolve(snapshot_id)
    dvr = snapshot.table_versions[PRICE_HISTORY_TABLE_NAME]
    key_filter = (KeyPredicate(column="ticker", operator="eq", values=(ticker,)),)
    population_bound = repository.scan_population_bound(
        snapshot.snapshot_id, table_name=PRICE_HISTORY_TABLE_NAME,
        table_contract_ref=dvr.table_contract_ref, key_filter=key_filter)
    max_result_rows = min(100, population_bound)
    max_batch_rows = min(100, max_result_rows) if max_result_rows > 0 else 100
    query = DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=dvr.table_contract_ref,
                      columns=("date", "close_adj", "close_raw", "high_raw", "retrieved_at",
                               "deleted", "source_kind", "source_hash", "capture_id"),
                      key_filter=key_filter,
                      order_by=("ticker", "date", "retrieved_at"), max_batch_rows=max_batch_rows,
                      max_result_rows=max_result_rows)
    rows = [r for batch in repository.scan(query, table_name=PRICE_HISTORY_TABLE_NAME)
            for r in batch.to_pylist()]
    return pd.DataFrame(rows)


def _gspc_path(source_root):
    return source_root / "earnings_predictions" / "data" / "raw" / "polygon" / "gspc_daily.csv"


def _write_gspc_rows(source_root, rows):
    directory = _gspc_path(source_root).parent
    directory.mkdir(parents=True, exist_ok=True)
    lines = list(HEADER_LINES) + [row + "\n" for row in rows]
    _gspc_path(source_root).write_text("".join(lines))


def _write_gspc(source_root, *, closes=_GSPC_CLOSES, dates=_GSPC_DATES) -> None:
    """The legacy layout: three header lines, then positional
    ``date,adj,close,high,low,open,volume`` rows. Every price column is
    distinct (adj = close + 100, high = close + 5, low = close - 3,
    open = close - 1), so a capture reading any column other than Close for
    the stored closes fails loudly."""
    _write_gspc_rows(source_root, [
        f"{day},{close + 100.0},{close},{close + 5.0},{close - 3.0},"
        f"{close - 1.0},{close * 1e9:.0f}"
        for day, close in zip(dates, closes, strict=True)])


def _shadow_head(conn):
    return conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope='shadow'"
                        ).fetchone()


def test_capture_ingests_gspc_daily_csv_as_the_regime_ticker(tmp_path):
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=False)
    _write_gspc(source_root)
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert report["failed_tickers"] == 0
    rows = _scan_price_history(conn, store, report["result_snapshot_id"], ticker=REGIME_TICKER)
    assert not rows.empty
    assert sorted(rows["date"]) == list(_GSPC_DATES)
    assert set(rows["source_kind"]) == {"legacy_gspc_csv"}
    assert not rows["deleted"].any()
    by_date = rows.set_index("date")
    for i, day in enumerate(_GSPC_DATES):
        # ^GSPC has no dividend adjustment: close_adj == close_raw == Close
        # EXACTLY (the distinct Adj Close column must never reach the table),
        # high_raw == High exactly.
        assert by_date.loc[day, "close_raw"] == _GSPC_CLOSES[i]
        assert by_date.loc[day, "close_adj"] == _GSPC_CLOSES[i]
        assert by_date.loc[day, "high_raw"] == _GSPC_CLOSES[i] + 5.0


def test_gspc_hash_and_retrieved_at_describe_the_bytes_read(tmp_path):
    """Hash, ``retrieved_at`` and rows all describe ONE read: the stored
    ``source_hash`` is exactly the sha256 of the file's bytes and
    ``retrieved_at`` is that read's own ``fstat`` mtime in UTC ISO -- the
    module's own convention, recomputed here from the file, never an internal
    helper call."""
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=False)
    _write_gspc(source_root)
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    rows = _scan_price_history(conn, store, report["result_snapshot_id"], ticker=REGIME_TICKER)
    path = _gspc_path(source_root)
    assert set(rows["source_hash"]) == {hashlib.sha256(path.read_bytes()).hexdigest()}
    mtime = path.stat().st_mtime
    expected = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    assert set(rows["retrieved_at"]) == {expected}


def test_gspc_null_close_or_high_row_is_dropped(tmp_path):
    """A non-numeric ``Close`` OR ``High`` coerces to NaN and its row is
    dropped: price_history.v3 nulls ``close_raw``/``high_raw`` only on a
    tombstone, so neither may be stored null. The 2024-01-02 row carries no
    High and the 2024-01-03 row no Close; only 2024-01-04 is kept."""
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=False)
    _write_gspc_rows(source_root, [
        "2024-01-02,4800.913,4700.913,,4697.913,4699.913,4700913000000",
        "2024-01-03,4842.83,,4747.83,4739.83,4741.83,4742830000000",
        "2024-01-04,4856.01,4756.01,4761.01,4753.01,4755.01,4756010000000",
    ])
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert report["failed_tickers"] == 0
    rows = _scan_price_history(conn, store, report["result_snapshot_id"], ticker=REGIME_TICKER)
    assert sorted(rows["date"]) == ["2024-01-04"]
    kept = rows.iloc[0]
    assert kept["close_raw"] == 4756.01 and kept["close_adj"] == 4756.01
    assert kept["high_raw"] == 4761.01


def test_oversize_gspc_file_refuses_before_any_byte_is_read(tmp_path, monkeypatch):
    """A file over the 32 MiB capture cap refuses on ``fstat`` BEFORE any
    read: ``os.fdopen`` is wrapped so a read of that fd is recorded -- the
    sparse-truncated file is 1 byte over the cap, and the recorded reads stay
    empty. Required, it is not a usable ticker (``SOURCE_NOT_FOUND``, nothing
    committed); unrequired, it fails only its own retrieval, reported as a
    parse error exactly like a malformed px csv."""
    conn, clock, store, base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=True)
    path = _gspc_path(source_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.truncate(_GSPC_MAX_BYTES + 1)
    gspc_stat = path.stat()
    reads: list[int] = []
    real_fdopen = os.fdopen

    def _recording_fdopen(fd, *args, **kwargs):
        handle = real_fdopen(fd, *args, **kwargs)
        info = os.fstat(handle.fileno())
        if (info.st_ino, info.st_dev) != (gspc_stat.st_ino, gspc_stat.st_dev):
            return handle
        original_read = handle.read

        def _read(*a, **k):
            reads.append(a[0] if a else -1)
            return original_read(*a, **k)

        class _ReadRecording:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                handle.close()
                return False

            def read(self, *a, **k):
                return _read(*a, **k)

        return _ReadRecording()

    monkeypatch.setattr(price_history_store.os, "fdopen", _recording_fdopen)
    with pytest.raises(OpsError) as exc:
        capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                required_tickers=(REGIME_TICKER,))
    assert exc.value.problem.code == "SOURCE_NOT_FOUND"
    assert _shadow_head(conn)["snapshot_id"] == base.snapshot_id

    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert [(f["ticker"], f["error"]) for f in report["failures"]] == \
        [(REGIME_TICKER, "ValueError")]
    assert reads == []  # the cap refusal happened on fstat: zero bytes read


def test_symlinked_gspc_file_is_skipped(tmp_path):
    """A symlinked ``gspc_daily.csv`` is never opened (discovery refuses it,
    and the read itself would ``O_NOFOLLOW`` refuse a swapped-in link):
    required it is the ``SOURCE_NOT_FOUND`` case, unrequired the ticker is
    simply not in this run's sources."""
    conn, clock, store, base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=True)
    outside = tmp_path / "real_gspc.csv"
    lines = list(HEADER_LINES) + [
        "2024-01-02,4800.913,4700.913,4705.913,4697.913,4699.913,4700913000000\n"]
    outside.write_text("".join(lines))
    path = _gspc_path(source_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(outside)
    with pytest.raises(OpsError) as exc:
        capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                required_tickers=(REGIME_TICKER,))
    assert exc.value.problem.code == "SOURCE_NOT_FOUND"
    assert exc.value.problem.details == {"tickers": [REGIME_TICKER]}
    assert _shadow_head(conn)["snapshot_id"] == base.snapshot_id

    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert report["failures"] == []


@pytest.mark.parametrize("shape", ["directory", "fifo"])
def test_nonregular_gspc_path_is_skipped(tmp_path, shape):
    """A non-regular ``gspc_daily.csv`` (a directory or a FIFO at the path)
    is skipped at discovery, never read: required, ``SOURCE_NOT_FOUND``;
    unrequired, the ticker is simply absent from the run."""
    conn, clock, store, base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=True)
    path = _gspc_path(source_root)
    if shape == "directory":
        path.mkdir(parents=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.mkfifo(path)
    with pytest.raises(OpsError) as exc:
        capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                required_tickers=(REGIME_TICKER,))
    assert exc.value.problem.code == "SOURCE_NOT_FOUND"
    assert _shadow_head(conn)["snapshot_id"] == base.snapshot_id

    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert report["failures"] == []
    assert report["tickers_seen"] == 2  # AAPL and SPY only -- ^GSPC was skipped


def test_malformed_required_regime_ticker_file_fails_like_a_malformed_px_csv(tmp_path):
    conn, clock, store, base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=True)
    _gspc_path(source_root).parent.mkdir(parents=True, exist_ok=True)
    _gspc_path(source_root).write_bytes(b"not,a,price\nfile\n")
    with pytest.raises(OpsError) as exc:
        capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                required_tickers=(REGIME_TICKER,))
    assert exc.value.problem.code == "SOURCE_NOT_FOUND"
    assert exc.value.problem.details == {"tickers": [REGIME_TICKER]}
    assert _shadow_head(conn)["snapshot_id"] == base.snapshot_id  # nothing was committed

    # Not required: the malformed file fails only its own ticker's capture,
    # exactly as a malformed px csv does, and the run still commits the rest.
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert [(f["ticker"], f["error"]) for f in report["failures"]] == \
        [(REGIME_TICKER, "EmptyDataError")]
    assert report["failed_tickers"] == 1
    assert "result_snapshot_id" in report


def test_missing_required_regime_ticker_refuses_source_not_found(tmp_path):
    conn, clock, store, base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=True)  # SPY present, ^GSPC absent
    with pytest.raises(OpsError) as exc:
        capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                 required_tickers=REQUIRED_CAPTURE_TICKERS)
    assert exc.value.problem.code == "SOURCE_NOT_FOUND"
    assert exc.value.problem.details == {"tickers": [REGIME_TICKER]}
    assert _shadow_head(conn)["snapshot_id"] == base.snapshot_id  # nothing was committed


def test_malformed_gspc_falls_back_to_the_prior_stored_history(tmp_path):
    """A malformed ``gspc_daily.csv`` never destroys an already-captured
    ^GSPC history: the failed retrieval changes nothing, the ticker stays
    usable through the prior dataset version (the required check reads its
    live rows), and the run commits the prior rows forward."""
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=True)
    _write_gspc(source_root)
    first = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                    required_tickers=REQUIRED_CAPTURE_TICKERS)
    before = _scan_price_history(conn, store, first["result_snapshot_id"],
                                 ticker=REGIME_TICKER)

    _gspc_path(source_root).write_bytes(b"not,a,price\nfile\n")
    second = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                     required_tickers=REQUIRED_CAPTURE_TICKERS)
    assert [(f["ticker"], f["error"]) for f in second["failures"]] == \
        [(REGIME_TICKER, "EmptyDataError")]  # the parse still fails and is reported
    assert "result_snapshot_id" in second  # but the prior history keeps the run committable
    after = _scan_price_history(conn, store, second["result_snapshot_id"],
                                ticker=REGIME_TICKER)
    cols = ["date", "close_adj", "close_raw", "high_raw", "retrieved_at", "deleted",
            "source_hash"]
    assert after[cols].sort_values("date").reset_index(drop=True).equals(
        before[cols].sort_values("date").reset_index(drop=True))
    assert not after["deleted"].any()


def test_cli_price_history_capture_passes_the_shared_required_tickers(tmp_path, monkeypatch):
    """The real ``ops price-history capture`` command path, with ``capture``
    stubbed at its module seam, receives exactly the shared
    ``REQUIRED_CAPTURE_TICKERS`` (imported, never restated at the call site)."""
    recorded: dict = {}

    def _stub_capture(conn, store, source_root, **kwargs):
        recorded["args"] = (conn, store, source_root)
        recorded.update(kwargs)
        return {"schema_version": "price_history_capture_report.v1.0",
                "dry_run": bool(kwargs.get("dry_run"))}

    monkeypatch.setattr(price_history_store, "capture", _stub_capture)
    source_root = tmp_path / "legacy"
    source_root.mkdir()
    args = SimpleNamespace(source_root=source_root, scope="shadow", dry_run=True)
    report = cli.price_history_command(args, tmp_path / "ops", None, None)
    assert report["dry_run"] is True
    assert REQUIRED_CAPTURE_TICKERS == (CALENDAR_TICKER, REGIME_TICKER)
    assert recorded["required_tickers"] == REQUIRED_CAPTURE_TICKERS


def test_nightly_trigger_capture_passes_the_shared_required_tickers(
        isolated_legacy_sources, shadow_root, install_fetcher, monkeypatch):
    """The real nightly price-preparation path
    (``nightly_trigger._refresh_plan_prices``, fixtures per
    ``test_planned_price_refresh``) calls capture with exactly the shared
    ``REQUIRED_CAPTURE_TICKERS``: a recording wrapper observes the real
    call-site kwargs and delegates to the untouched capture, which commits
    for real (the fixture seeds the required SPY and ^GSPC sources)."""
    state = shadow_root
    install_fetcher(state.root)
    recorded: dict = {}
    real_capture = price_history_store.capture

    def _recording_capture(*args, **kwargs):
        recorded.update(kwargs)
        return real_capture(*args, **kwargs)

    monkeypatch.setattr(price_history_store, "capture", _recording_capture)
    snapshot_id = nightly_trigger._refresh_plan_prices(
        state.root, SESSION, ("ZZNEW",), state.clock, state.base.snapshot_id)
    assert recorded["required_tickers"] == REQUIRED_CAPTURE_TICKERS
    assert snapshot_id != state.base.snapshot_id  # the real delegated capture committed
