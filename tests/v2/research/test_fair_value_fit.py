"""Tests for fill model F slice F6a: validation, pricing, window, exclusions, privacy and the receipt (design section 9).
Synthetic fills only. Oracles are hand-written from the spec's formulas, never from fair_value_fit helpers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

import pandas as pd
import pytest

from experiments import fair_value_fit as ff
from experiments import fair_value_inputs as fvi
from experiments import fair_value_run as run

ORDER = "2030-03-04"
FILL = "2030-03-04T15:30:00-05:00"
SNAP = "2030-03-04T15:46:00-05:00"
EXP_30D = "2030-04-03"


def make_row(h, c, side, *, hit=50.0, oi=1000.0, **over):
    """Quote (1-h, 1+h) on a call with strike 100 and spot 101: V = 1, denom = h. Price = V + c*h (buy) or V - c*h (sell)."""
    price = 1.0 + c * h if side == "buy" else 1.0 - c * h
    row = {"fill_time": FILL, "snapshot_time": SNAP, "ticker": "XYZ", "expiry": ORDER, "strike": 100.0, "right": "C",
           "side": side, "qty": 1.0, "price": price, "bid": 1.0 - h, "ask": 1.0 + h, "bid_size": 50.0, "ask_size": 50.0,
           "open_interest": oi, "quote_repaired": "false", "spot": 101.0, "vol_date": None, "smvVol": None,
           "rates_date": None, "iRate5wk": None, "iRateLt": None, "borrow30": None, "borrow2yr": None, "divAmt": None,
           "divDate": None}
    row["ask_size" if side == "buy" else "bid_size"] = hit
    row.update(over)
    return row


def base_rows(n=120):
    """n unflagged fills with observed c = 0.30 (the price is built from c and the spread, so c is exact)."""
    return [make_row(0.02 + 0.28 * i / max(n - 1, 1), 0.30, "buy" if i % 2 == 0 else "sell") for i in range(n)]


def blob_of(rows) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False, na_rep="").encode()


def bs_call(s, k, t, r, sig):
    d1 = (math.log(s / k) + (r + 0.5 * sig * sig) * t) / (sig * math.sqrt(t))
    d2 = d1 - sig * math.sqrt(t)

    def n(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    return s * n(d1) - k * math.exp(-r * t) * n(d2)


def long_row(strike=100.0, **over):
    """A 30-day call from the order date; vol and rates from the previous day; smvVol 0.25; iRate 0.04; no borrow, no dividend."""
    row = make_row(0.05, 0.3, "buy", strike=strike, expiry=EXP_30D, vol_date="2030-03-03", smvVol=0.25,
                   rates_date="2030-03-03", iRate5wk=0.04, iRateLt=0.04)
    row.update(over)
    return row


def _refusal(rows, tmp_path) -> dict:
    p = tmp_path / "fills.csv"
    p.write_bytes(blob_of(rows))
    with pytest.raises(fvi.FairValueRefusal) as err:
        ff.price_fills(p.read_bytes())
    return err.value.details


# --- 1. observed c: hand-written formula, buy and sell, negative c, repaired quote, time value -------------------------

def test_observed_c_matches_hand_formula_for_buy_sell_and_negative_c():
    rows = [make_row(0.10, 0.30, "buy"), make_row(0.10, 0.30, "sell"), make_row(0.20, -0.10, "buy"), make_row(0.20, -0.10, "sell")]
    d, _ = ff.price_fills(blob_of(rows))
    assert list(d.columns) == ["c", "x", "thin_size", "thin_oi", "group", "order"]
    assert d["c"].to_numpy() == pytest.approx([0.30, 0.30, -0.10, -0.10], abs=1e-9)


def test_repaired_quote_uses_075_and_125_of_the_stored_price():
    rows = [make_row(0.05, 0.0, "buy", bid=1.0, ask=1.0, quote_repaired="true", price=1.1),
            make_row(0.05, 0.0, "sell", bid=1.0, ask=1.0, quote_repaired="true", price=0.9)]
    d, _ = ff.price_fills(blob_of(rows))
    # effective quote (0.75, 1.25) and V = 1: buy c = (1.1 - 1) / 0.25, sell c = (1 - 0.9) / 0.25
    assert d["c"].to_numpy() == pytest.approx([0.4, 0.4], abs=1e-9)


def test_time_value_call_matches_black_scholes_from_order_date():
    rows = []
    for k, side in ((99.0, "buy"), (100.0, "sell"), (101.0, "buy"), (102.0, "sell")):
        v = bs_call(101.0, k, 30 / 365, 0.04, 0.25)  # snapshot spot 101; 30 days from the order date; b = r; no dividend
        h = 0.05
        price = v + 0.3 * h if side == "buy" else v - 0.3 * h
        rows.append(long_row(strike=k, bid=v - h, ask=v + h, price=price, side=side))
    d, _ = ff.price_fills(blob_of(rows))
    assert d["c"].to_numpy() == pytest.approx([0.3] * 4, abs=1e-9)


# --- 2. group assignment from the side-hit size; unknown for empty size or open interest -------------------------------

def test_group_uses_side_hit_size_and_empty_size_or_oi_is_unknown():
    rows = [make_row(0.10, 0.3, "buy", hit=5.0),  # ask_size thin on a buy: size_only
            make_row(0.10, 0.3, "buy", bid_size=5.0),  # bid_size thin on a buy is not the side hit: unflagged
            make_row(0.10, 0.3, "sell", hit=5.0),  # bid_size thin on a sell: size_only
            make_row(0.10, 0.3, "sell", ask_size=5.0),  # ask_size thin on a sell is not the side hit: unflagged
            make_row(0.10, 0.3, "buy", hit=None),  # empty ask_size: unknown
            make_row(0.10, 0.3, "sell", oi=None)]  # empty open_interest: unknown
    d, _ = ff.price_fills(blob_of(rows))
    assert d["group"].tolist() == ["size_only", "unflagged", "size_only", "unflagged", "unknown", "unknown"]


# --- 3. window edges and offsets --------------------------------------------------------------------------------------

def test_window_edges_kept_and_one_second_over_excluded():
    rows = base_rows()
    rows.append(make_row(0.05, 0.3, "buy", snapshot_time="2030-03-04T14:30:00-05:00"))  # exactly 60 min before: kept
    rows.append(make_row(0.05, 0.3, "buy", snapshot_time="2030-03-04T16:30:00-05:00"))  # exactly 60 min after: kept
    rows.append(make_row(0.05, 0.3, "buy", snapshot_time="2030-03-04T16:30:01-05:00"))  # 60 min 1 s: excluded
    rows.append(make_row(0.05, 0.3, "sell", fill_time="2030-03-04T20:30:00+00:00", snapshot_time=SNAP))  # UTC offset
    _, receipt = ff.price_fills(blob_of(rows))
    assert receipt["counts"]["outside_window"] == 1
    assert receipt["counts"]["used"] == 123


def test_different_offsets_compare_by_instant():
    rows = base_rows()
    rows.append(make_row(0.05, 0.3, "buy", fill_time="2030-03-04T20:30:00+00:00", snapshot_time="2030-03-04T15:30:00-05:00"))
    _, receipt = ff.price_fills(blob_of(rows))
    assert receipt["counts"]["outside_window"] == 0


# --- 4. each exclusion counted once -----------------------------------------------------------------------------------

def test_each_exclusion_counted_once():
    rows = base_rows()
    rows.append(make_row(0.05, 0.3, "buy", bid=1.05, ask=0.95))  # ask < bid: unpriceable
    rows.append(long_row(rates_date="2030-02-24", smvVol=0.25))  # rates 8 days old: rate_unknown
    v = bs_call(101.0, 100.0, 30 / 365, 0.04, 0.25)  # quote around V so the row is priced, not clipped to the ask
    rows.append(long_row(rates_date="2030-02-25", bid=v - 0.05, ask=v + 0.05, price=v + 0.01))  # 7 days old: accepted
    rows.append(long_row(smvVol=None))  # no smile fallback: no_fair_value
    rows.append(make_row(0.05, 0.3, "buy", spot=105.0))  # intrinsic 5 above ask: V == ask, so small_denominator
    _, receipt = ff.price_fills(blob_of(rows))
    counts = receipt["counts"]
    assert counts["unpriceable"] == 1
    assert counts["rate_unknown"] == 1
    assert counts["no_fair_value"] == 1
    assert counts["small_denominator"] == 1
    assert counts["outside_window"] == 0


def test_out_of_range_rate_and_borrow_are_counted_and_treated_as_missing():
    rows = [long_row(iRate5wk=0.5, iRateLt=0.5),  # both rates above 0.20: both cells counted, both treated as missing
            long_row(borrow30=2.0),  # borrow above 0.2
            long_row(borrow2yr=-1.5)]  # borrow below -1.0
    _, receipt = ff.price_fills(blob_of(rows))
    assert receipt["counts"]["rate_cells_out_of_range"] == 2
    assert receipt["counts"]["borrow_cells_out_of_range"] == 2
    assert receipt["counts"]["rate_unknown"] == 1


# --- 5. causality, value and structure refusals -----------------------------------------------------------------------

def test_same_day_vol_and_rates_dates_are_not_causal(tmp_path):
    rows = base_rows()
    rows.append(long_row(vol_date=ORDER))
    assert _refusal(rows, tmp_path) == {"reason": "not_causal", "column": "vol_date", "rows": 1}
    rows = base_rows()
    rows.append(long_row(rates_date=ORDER))
    assert _refusal(rows, tmp_path) == {"reason": "not_causal", "column": "rates_date", "rows": 1}


def test_naive_and_date_only_times_are_value_invalid(tmp_path):
    rows = base_rows()
    rows.append(make_row(0.05, 0.3, "buy", fill_time="2030-03-04T15:30:00"))  # naive
    assert _refusal(rows, tmp_path) == {"reason": "value_invalid", "column": "fill_time", "rows": 1}
    rows = base_rows()
    rows.append(make_row(0.05, 0.3, "buy", snapshot_time="2030-03-04"))  # date only
    assert _refusal(rows, tmp_path) == {"reason": "value_invalid", "column": "snapshot_time", "rows": 1}


def test_non_numeric_bid_size_is_value_invalid(tmp_path):
    rows = base_rows()
    rows.append(make_row(0.05, 0.3, "buy", bid_size="abc"))
    assert _refusal(rows, tmp_path) == {"reason": "value_invalid", "column": "bid_size", "rows": 1}


def test_empty_ticker_is_value_invalid(tmp_path):
    rows = base_rows()
    rows.append(make_row(0.05, 0.3, "buy", ticker=None))  # empty cell
    assert _refusal(rows, tmp_path) == {"reason": "value_invalid", "column": "ticker", "rows": 1}


def test_bad_side_is_value_invalid(tmp_path):
    rows = base_rows()
    rows.append(make_row(0.05, 0.3, "hold"))
    assert _refusal(rows, tmp_path) == {"reason": "value_invalid", "column": "side", "rows": 1}


def test_missing_column_and_empty_file(tmp_path):
    rows = [{k: v for k, v in r.items() if k != "spot"} for r in base_rows()]
    assert _refusal(rows, tmp_path) == {"reason": "column_missing", "columns": ["spot"]}
    p = tmp_path / "empty.csv"
    p.write_bytes(pd.DataFrame(columns=list(make_row(0.05, 0.3, "buy").keys())).to_csv(index=False).encode())
    with pytest.raises(fvi.FairValueRefusal) as err:
        ff.price_fills(p.read_bytes())
    assert err.value.details["reason"] == "fills_empty"


def test_out_dir_exists_is_checked_before_reading(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(fvi.FairValueRefusal) as err:
        ff.run_fit(tmp_path / "missing_fills.csv", out)  # the fills file is absent, so a read would fail first
    assert err.value.details["reason"] == "out_dir_exists"


# --- 6. privacy: no row-level values in stdout, stderr, logs or files -------------------------------------------------

def test_privacy_success_and_refusal_leak_no_row_values(tmp_path, capsys, caplog):
    caplog.set_level(logging.INFO)
    rows = base_rows()
    for r in rows:
        r["ticker"] = "ZZPRIVATE"
    p = tmp_path / "secret_fills_name.csv"
    p.write_bytes(blob_of(rows))
    out = tmp_path / "out"
    assert ff.main(["--fills", str(p), "--out-dir", str(out)]) == 0
    captured = capsys.readouterr()
    written = "".join(f.read_text() for f in out.iterdir())
    text = captured.out + captured.err + caplog.text + written
    assert "ZZPRIVATE" not in text and str(p) not in text and "secret_fills_name" not in text

    secret = make_row(0.05, 0.3, "buy")
    secret["side"] = "SECRET123"  # set after building the row, so the dict key is not passed twice
    p2 = tmp_path / "bad.csv"
    p2.write_bytes(blob_of(base_rows() + [secret]))
    assert ff.main(["--fills", str(p2), "--out-dir", str(tmp_path / "out2")]) == 2
    refusal_out = capsys.readouterr()
    assert "SECRET123" not in refusal_out.out + refusal_out.err and str(p2) not in refusal_out.out + refusal_out.err
    assert not (tmp_path / "out2").exists()


# --- 7. receipt, zero used fills and the written file -----------------------------------------------------------------

def test_receipt_keys_and_hashes_are_exact():
    blob = blob_of(base_rows())
    _, receipt = ff.price_fills(blob)
    assert set(receipt) == {"input_sha256", "base_fvf_id", "fit_source_sha256", "window_minutes", "counts",
                            "first_order_date", "last_order_date"}
    assert receipt["input_sha256"] == hashlib.sha256(blob).hexdigest()
    assert receipt["base_fvf_id"] == run.fvf_id()
    assert receipt["fit_source_sha256"] == hashlib.sha256(Path(ff.__file__).read_bytes()).hexdigest()
    assert receipt["window_minutes"] == 60
    assert receipt["counts"]["used"] == 120
    assert receipt["first_order_date"] == receipt["last_order_date"] == ORDER


def test_zero_used_fills_is_a_receipt_not_a_refusal(tmp_path):
    p = tmp_path / "fills.csv"
    p.write_bytes(blob_of([make_row(0.05, 0.3, "buy", snapshot_time="2030-03-04T20:30:00-05:00")]))  # 5 h away: outside
    out = tmp_path / "out"
    receipt = ff.run_fit(p, out)
    assert receipt["counts"]["used"] == 0
    assert receipt["counts"]["outside_window"] == 1
    assert receipt["first_order_date"] is None and receipt["last_order_date"] is None
    assert json.loads((out / "fit_receipt.json").read_text())["counts"]["used"] == 0


def test_run_fit_writes_only_the_receipt(tmp_path):
    p = tmp_path / "fills.csv"
    p.write_bytes(blob_of(base_rows()))
    out = tmp_path / "out"
    receipt = ff.run_fit(p, out)
    assert json.loads((out / "fit_receipt.json").read_text()) == receipt
    assert sorted(x.name for x in out.iterdir()) == ["fit_receipt.json"]  # no fit_receipt.json.tmp left behind
    assert sorted(x.name for x in tmp_path.iterdir()) == ["fills.csv", "out"]


def test_refusal_leaves_no_out_dir(tmp_path):
    p = tmp_path / "fills.csv"
    p.write_bytes(blob_of(base_rows() + [make_row(0.05, 0.3, "hold")]))  # bad side: pricing refuses
    out = tmp_path / "out"
    with pytest.raises(fvi.FairValueRefusal):
        ff.run_fit(p, out)
    assert not out.exists()
    assert sorted(x.name for x in tmp_path.iterdir()) == ["fills.csv"]


def test_failed_rename_leaves_no_out_dir_and_no_staging(tmp_path, monkeypatch):
    p = tmp_path / "fills.csv"
    p.write_bytes(blob_of(base_rows()))
    out = tmp_path / "out"

    def boom(self, target):
        raise OSError("boom")

    with monkeypatch.context() as m:
        m.setattr(Path, "rename", boom)
        with pytest.raises(OSError, match="boom"):
            ff.run_fit(p, out)
    assert not out.exists()
    assert sorted(x.name for x in tmp_path.iterdir()) == ["fills.csv"]  # no .fit-* staging directory left behind
