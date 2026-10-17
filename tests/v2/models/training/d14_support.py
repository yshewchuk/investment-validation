"""Public synthetic fixtures and untouched legacy oracles shared by D14 tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.calendar import TradingCalendar, projected_trading_days
from engine.data.features import tier4
from engine.data.schemas import coerce
from engine.models.training import implied_t1
from engine.v2.data.reference_catalog import ReferenceInput, insert_reference_inputs
from engine.v2.data.repository import Repository
from engine.v2.models.training.d14_dataset import HISTORY_COLUMNS
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
    publish_bytes,
)


def input_frames():
    days = projected_trading_days("2023-10-01", "2025-03-31")
    dates = ("2024-01-02", "2024-02-01", "2024-07-05", "2024-12-02", "2025-01-02")
    events, panel, daily = [], [], []
    tickers = ("AMC", "BMO", "GAP", "NULL", "ZERO", "NEG", "NOPANEL", "NODAILY", "NOSESSION")
    for ti, ticker in enumerate(tickers):
        for ei, text in enumerate(dates):
            day = pd.Timestamp(text)
            events.append(dict(event_id=f"{ti:02}-{ei}", ticker=ticker, event_date=day,
                               year=day.year, session=None if ticker == "NOSESSION" else
                               "BMO" if ticker == "BMO" else "AMC", src_orats=True,
                               src_oquants=False, src_nasdaq=False, src_yfinance=False,
                               date_agree=True, date_conflict=False))
            if ticker != "NOPANEL":
                panel.append(dict(ticker=ticker, date=day, k=12, year=day.year,
                                  **{c: 1.5 + ti / 10 + ei / 50 for c in HISTORY_COLUMNS
                                     if c not in ("n_prior", "signed_streak")},
                                  n_prior=12, signed_streak=-2, or_iv30=9999.0))
        if ticker == "NODAILY":
            continue
        for di, day in enumerate(days):
            if ticker == "GAP" and di % 4 == 0:
                continue
            row = dict(ticker=ticker, date=day, year=day.year,
                       src_iv=None if di % 11 == 0 else "orats")
            row.update({field: 3 + di / 100 + ti / 50 for field in implied_t1.DAILY_STATE_COLUMNS
                        if "_d" not in field})
            row["implied_move"] = row.pop("im")
            row["spot"] = (0.0 if ticker == "ZERO" else -1.0 if ticker == "NEG" else 100 + di / 4)
            if ticker == "NULL":
                row["implied_move"] = None
            daily.append(row)
    return pd.DataFrame(events).sort_values(["year", "event_id"]).reset_index(drop=True), \
        pd.DataFrame(panel), pd.DataFrame(daily).sort_values(["ticker", "date"]), days


@pytest.fixture
def pinned(tmp_path, request):
    events, panel, daily, days = input_frames()
    case = getattr(request, "param", "normal")
    calendar_bytes = case.get("calendar_bytes") if isinstance(case, dict) else None
    case = case.get("case", "normal") if isinstance(case, dict) else case
    if case == "empty":
        panel = panel.iloc[:0]
    elif case == "all_null":
        panel["ema12_prior_move"] = np.nan
    elif case == "boundary":
        days = projected_trading_days("2015-11-01", days[-1])
        for year in (2016, 2017, 2026, 2027):
            event = events.iloc[0].to_dict()
            event.update(event_id=f"boundary-{year}", event_date=pd.Timestamp(f"{year}-01-02"), year=year)
            events = pd.concat([events, pd.DataFrame([event])], ignore_index=True)
            if year == 2026:
                history = panel.iloc[0].to_dict()
                history.update(date=event["event_date"], year=year, n_prior=26, mean_prior_move=26.5)
                panel = pd.concat([panel, pd.DataFrame([history])], ignore_index=True)
    elif case == "blank_session":
        events.loc[0, "session"] = ""
    elif case == "warmup":
        for index in (13, 14):
            event = events.iloc[0].to_dict()
            event.update(event_id=f"warmup-{index}", event_date=days[index], year=days[index].year)
            events = pd.concat([events, pd.DataFrame([event])], ignore_index=True)
    elif case == "early_daily":
        events = events.iloc[[0]].copy()
        events["event_date"], events["year"] = pd.Timestamp("2017-01-20"), 2017
        panel = panel.iloc[[0]].copy()
        panel["date"], panel["year"] = pd.Timestamp("2017-01-20"), 2017
        days = projected_trading_days("2016-11-01", "2017-02-01")
        template = daily.iloc[1].to_dict()
        daily = pd.DataFrame([{**template, "date": day, "year": day.year, "spot": 100.0 + i}
                              for i, day in enumerate(days)])
    conn, clock, store = catalog_and_store(tmp_path)
    frames = {"earnings_events": events, "feature_panel": panel, "daily_market": daily}
    contracts = {table: contract_for(table) for table in frames}
    records = {}
    for table, frame in frames.items():
        contract = contracts[table]
        records[table] = []
        parts = ([] if frame.empty else [(None, frame)]) if table == "feature_panel" \
            else frame.groupby("year", sort=True)
        for year, part in parts:
            part = part.sort_values(list(contract.primary_key))
            records[table].append(publish_and_inspect(
                store, contract, contract_ref_for(contract), part.astype(object).where(pd.notna(part), None).to_dict("records"),
                "all" if year is None else str(year)))
    snapshot = commit_tables(conn, clock, records, contracts, store=store)
    # Original yfinance CSV's three header rows, then dates. Exact immutable object.
    data = ("Price,Close\nTicker,^GSPC\nDate,\n" +
            "\n".join(f"{d.date()},100" for d in days) + "\n").encode()
    if case in ("calendar_at_limit", "calendar_past_limit"):
        size = 16 * 1024 * 1024 + (case == "calendar_past_limit")
        data += b"\n" * (size - len(data))
    obj = publish_bytes(store, data if calendar_bytes is None else calendar_bytes)
    ref = ReferenceInput(kind="model_registry" if case == "no_calendar" else "calendar", legacy_path="data/raw/yf/index.csv",
                         object_id=obj.object_id, content_hash=obj.content_hash, byte_size=obj.byte_size)
    if case != "no_refs":
        insert_reference_inputs(conn, "r1", (ref,))
    conn.commit()
    yield dict(repository=Repository(conn, store), snapshot=snapshot, events=events,
               panel=panel, daily=daily, days=days, conn=conn, calendar_ref=ref,
               records=records, contracts=contracts, clock=clock, store=store,
               inputs=dict(catalog=tmp_path / "catalog.sqlite", artifact_root=store.root,
                           snapshot_id=snapshot.snapshot_id, receipt_id="r1"))
    conn.close()


def legacy_frame(pinned, role, monkeypatch):
    calendar = TradingCalendar(list(pinned["days"]) + list(projected_trading_days(
        pinned["days"][-1], pinned["days"][-1] + pd.Timedelta(days=400))))
    monkeypatch.setattr(implied_t1, "trading_calendar", lambda: calendar)
    def read(name, **kwargs):
        frame = pinned["events"] if name == "earnings_events" else pinned["daily"]
        frame = frame[frame.year.isin(kwargs["years"])]
        if name == "earnings_events":
            frame = frame.sort_values(["year", "event_id"], kind="stable")
        return coerce(frame, name, only=kwargs["columns"])

    monkeypatch.setattr(tier4.store, "read_table", read)
    factory = tier4.im_t1_feature_model if role == "implied_t1" else tier4.runup_move_feature_model
    return factory().prepare(pinned["panel"]).rename(columns={"date": "event_date"})


def forbidden(*args, **kwargs):
    raise AssertionError("legacy reader reached")
