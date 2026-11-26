"""EXP-189 A1 W7c3 part b: the report-input store on a stub quotes object (synthetic values, tmp_path only; no data/ read)."""
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments import size_pnl_receipts as receipts
from experiments import size_width_f_store as st
from experiments import size_width_pricing as seam

ENTRY, EXIT, EXP = pd.Timestamp("2024-02-05"), pd.Timestamp("2024-02-08"), pd.Timestamp("2024-03-15")
ROW = {"bid": 1.0, "ask": 1.2, "eff_bid": 1.0, "eff_ask": 1.2, "V": 1.1, "status": "ok", "source": "smile", "vol": 0.2, "smvVol": 0.25,
       "rate": 0.05, "c_buy": 0.5, "c_sell": 0.5, "fill_buy": 1.1, "fill_sell": 1.05, "repaired": False, "open_interest": 10.0,
       "bid_size": 5.0, "ask_size": 6.0, "chain_kind": "chain", "assetType": "code", "spot": 100.0}


def _table(keys):
    """A quote table indexed (ticker, obs_date, expiry, right, strike) with one ROW per key."""
    index = pd.MultiIndex.from_arrays([[k[j] for k in keys] for j in range(5)], names=["ticker", "obs_date", "expiry", "right", "strike"])
    return pd.DataFrame([ROW] * len(keys), index=index)


class Stub:
    """Stands in for FQuotes: fixed tables per side, a fixed leg key, and a record of the sides asked for."""
    def __init__(self, tables):
        self.tables, self.calls, self.coverage = tables, [], []

    def table(self, chunk, *, exit=False):
        self.calls.append(exit)
        return self.tables["exit" if exit else "entry"]

    def leg_key(self, chunk, i, idx, *, exit=False):
        return ("AAA", EXIT if exit else ENTRY, EXP, "P", 100.0)


def _keys(rows):
    return pd.DataFrame(rows, columns=st._KEYS)


def _entry_table():
    return _table([("AAA", ENTRY, EXP, "P", 100.0)])


def test_a_listed_leg_row_carries_its_table_quote_and_fields(tmp_path):
    quotes = Stub({"entry": _entry_table(), "exit": _table([("AAA", EXIT, EXP, "P", 100.0)])})
    legs = st.Store(tmp_path, "inner", quotes)._legs(None, _keys([("exit", "e1", "A", 1.0, "buy", 2.0, "AAA", EXIT, EXP, "P", 100.0)]))
    row = legs.iloc[0]
    assert (row["point"], row["side"], row["qty"], row["strike"], row["right"]) == ("exit", "buy", 2.0, 100.0, "P")
    assert row["mid"] == pytest.approx(1.1) and row["year"] == 2024 and row["status"] == "ok"
    assert row["smvVol"] == 0.25 and row["source"] == "smile" and row["V"] == pytest.approx(1.1)


def test_an_unlisted_exit_leg_is_unpriceable_with_null_values(tmp_path):
    quotes = Stub({"entry": _entry_table(), "exit": _table([("AAA", EXIT, EXP, "P", 100.0)])})
    legs = st.Store(tmp_path, "inner", quotes)._legs(None, _keys([("exit", "e1", "A", 1.0, "buy", 2.0, "AAA", EXIT, EXP, "P", 105.0)]))
    row = legs.iloc[0]
    assert row["status"] == "unpriceable" and pd.isna(row["bid"]) and pd.isna(row["V"]) and pd.isna(row["source"])
    assert pd.isna(row["repaired"]) and row["strike"] == 105.0


def test_duplicate_buffered_keys_yield_one_leg_row_per_part(tmp_path):
    key = ("entry", "e1", "A", 1.0, "buy", 2.0, "AAA", ENTRY, EXP, "P", 100.0)
    store = st.Store(tmp_path, "inner", Stub({"entry": _entry_table(), "exit": _table([])}))
    store.keys = [key, key]
    store.flush(None)
    assert len(pd.read_parquet(tmp_path / "report_inputs" / "inner" / "legs_0000.parquet")) == 1
    assert store.parts["legs_0000"]["rows"] == 1 and store.n == 1 and not store.keys


def test_a_fallback_taken_row_is_labelled_with_its_phi_and_sorted_factors(tmp_path):
    store = st.Store(tmp_path, "outer", Stub({"entry": _entry_table(), "exit": _table([])}))
    pricing = SimpleNamespace(resolve=lambda chunk, i, name, pred: ([{"idx": 0, "side": "buy", "qty": 1.0}], None))
    sweep = SimpleNamespace(pricing=pricing, phi=0.5, factors={"B": 2.0, "A": 1.0})
    chunk = {"event_id": ["e1"], "right": [0], "strike": [100.0], "expiry": [EXP]}
    store.taken(sweep, chunk, pd.DataFrame({"event_id": ["e1"], "pred_move": [1.0]}),
                pd.DataFrame({"event_id": ["e1"], "structure": ["A"], "k": [1.0]}), "E")
    assert store.taken_buf[0]["arm"].iloc[0] == "fallback phi=0.5 factors=A=1.0,B=2.0"
    assert len(store.keys) == 2  # the entry and the exit leg of the one taken row


def test_close_with_nothing_recorded_writes_empty_tables_and_no_legs_part(tmp_path):
    st.Store(tmp_path, "inner", Stub({})).close("H", "W")
    receipt = receipts.load_receipt(tmp_path / "receipts" / "report_inputs_inner.json")
    assert receipt["tables"]["coverage"]["rows"] == 0 and receipt["tables"]["exit_flags"]["rows"] == 0
    assert not any(name.startswith("legs_") for name in receipt["tables"])


def test_open_store_is_none_for_current_pricing(tmp_path):
    assert st.open_store(tmp_path, "inner", seam.load()) is None


def test_entry_coverage_prints_only_entry_tables_and_the_entry_rows(tmp_path, capsys):
    quotes = Stub({"entry": _table([]), "exit": _table([])})
    quotes.coverage = [pd.DataFrame({"point": ["entry", "exit"], "year": [2024, 2024], "chain_kind": ["c", "c"], "metric": ["rows", "rows"],
                                     "value": ["", ""], "count": [3, 4]})]
    st.Store(tmp_path, "inner", quotes).print_entry_coverage([object(), object()])
    assert quotes.calls == [False, False]
    err = capsys.readouterr().err
    assert "[f_store] entry coverage" in err and "entry,2024,c,rows,,3" in err and "exit" not in err.split("entry coverage")[1]


def test_candidates_buffer_entry_legs_only_where_entry_f_priced(tmp_path):
    """e1: entry F priced (a skip counts as priced), so its entry legs are kept; e2: not priced; e3: resolve gives no legs."""
    legs = [{"idx": 0, "side": "sell", "qty": 1.0}, {"idx": 1, "side": "buy", "qty": 0.0}]
    resolved, priced = {0: legs, 1: legs, 2: None}, {0: True, 1: False}
    fills = SimpleNamespace(was_priced=lambda trade: priced.get(trade.i, False))
    sweep = SimpleNamespace(pricing=SimpleNamespace(resolve=lambda chunk, i, name, pred: (resolved[i], None), fills=fills))
    store = st.Store(tmp_path, "inner", Stub({}))
    events = pd.DataFrame({"event_id": ["e1", "e2", "e3"], "pred_move": [6.0, 6.0, 6.0]})
    store.candidates(sweep, {"event_id": ["e1", "e2", "e3"]}, events, "A", 1.0)
    assert [key[1] for key in store.keys] == ["e1"]  # the one positive-quantity entry leg of e1
    assert store.keys[0][0] == "entry"
