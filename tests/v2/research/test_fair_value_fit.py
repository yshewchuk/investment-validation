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
    """Quote (1-h, 1+h) on a call with strike 100 and spot 101: V = 1, mid = 1, half = h. Price = V + c*h (buy) or V - c*h (sell)."""
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
    """n unflagged fills with price-space x = 0.30 (the price is built from x and the spread, so x is exact)."""
    return [make_row(0.02 + 0.28 * i / max(n - 1, 1), 0.30, "buy" if i % 2 == 0 else "sell") for i in range(n)]


def blob_of(rows) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False, na_rep="").encode()


def fills_file(tmp_path, blob, name="fills.csv") -> Path:
    p = tmp_path / name
    p.write_bytes(blob)
    return p


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
    p = fills_file(tmp_path, blob_of(rows))
    with pytest.raises(fvi.FairValueRefusal) as err:
        ff.price_fills(p)
    return err.value.details


# --- 1. price space: hand-written x, v, mid, half, bid, ask, spread_term for buy and sell, negative x, repaired quote, T>0 ---

def test_price_space_columns_match_hand_values_for_buy_sell_and_negative_x(tmp_path):
    rows = [make_row(0.10, 0.30, "buy"), make_row(0.10, 0.30, "sell"), make_row(0.20, -0.10, "buy"), make_row(0.20, -0.10, "sell")]
    d, _ = ff.price_fills(fills_file(tmp_path, blob_of(rows)))
    assert list(d.columns) == ["x", "v", "spread_term", "thin_size", "thin_oi", "group", "mid", "half_spread", "bid", "ask", "order"]
    # quote (0.9, 1.1), V = 1: buy x = (1.03 - 1) / 0.1; sell x = (1 - 0.97) / 0.1; v = 0 because V is the mid
    # quote (0.8, 1.2), V = 1: buy price 0.98 gives x = (0.98 - 1) / 0.2 = -0.1 (a buy below mid); sell price 1.02 gives x = -0.1
    assert d["x"].to_numpy() == pytest.approx([0.3, 0.3, -0.1, -0.1], abs=1e-9)
    assert d["v"].to_numpy() == pytest.approx([0.0] * 4, abs=1e-9)
    assert d["mid"].to_numpy() == pytest.approx([1.0] * 4, abs=1e-9)
    assert d["half_spread"].to_numpy() == pytest.approx([0.1, 0.1, 0.2, 0.2], abs=1e-9)
    assert d["bid"].to_numpy() == pytest.approx([0.9, 0.9, 0.8, 0.8], abs=1e-9)
    assert d["ask"].to_numpy() == pytest.approx([1.1, 1.1, 1.2, 1.2], abs=1e-9)
    # spread_term = min((hi - lo) / mid / 0.5, 1): 0.2 / 1 / 0.5 = 0.4 and 0.4 / 1 / 0.5 = 0.8
    assert d["spread_term"].to_numpy() == pytest.approx([0.4, 0.4, 0.8, 0.8], abs=1e-9)


def test_repaired_quote_uses_075_and_125_of_the_stored_price_and_stores_the_stored_quote(tmp_path):
    rows = [make_row(0.05, 0.0, "buy", bid=1.0, ask=1.0, quote_repaired="true", price=1.1),
            make_row(0.05, 0.0, "sell", bid=1.0, ask=1.0, quote_repaired="true", price=0.9)]
    d, _ = ff.price_fills(fills_file(tmp_path, blob_of(rows)))
    # effective quote (0.75, 1.25), so mid = 1 and half = 0.25; V = 1 is inside.
    # buy x = (1.1 - 1) / 0.25; sell x = (1 - 0.9) / 0.25; v = 0
    assert d["x"].to_numpy() == pytest.approx([0.4, 0.4], abs=1e-9)
    assert d["v"].to_numpy() == pytest.approx([0.0, 0.0], abs=1e-9)
    assert d["mid"].to_numpy() == pytest.approx([1.0, 1.0], abs=1e-9)
    assert d["half_spread"].to_numpy() == pytest.approx([0.25, 0.25], abs=1e-9)
    # the bid and ask columns hold the stored quote (P = 1.0 on both sides), not the effective one
    assert d["bid"].to_numpy() == pytest.approx([1.0, 1.0], abs=1e-9)
    assert d["ask"].to_numpy() == pytest.approx([1.0, 1.0], abs=1e-9)
    # spread_term = min((1.25 - 0.75) / 1 / 0.5, 1) = 1
    assert d["spread_term"].to_numpy() == pytest.approx([1.0, 1.0], abs=1e-9)


def test_time_value_call_matches_black_scholes_from_order_date(tmp_path):
    rows, cases = [], []  # cases: (lo, hi, V, price, buy) with V the test's own Black-Scholes value
    for k, side in ((99.0, "buy"), (100.0, "sell"), (101.0, "buy"), (102.0, "sell")):
        v = bs_call(101.0, k, 30 / 365, 0.04, 0.25)  # snapshot spot 101; 30 days from the order date; b = r; no dividend
        h = 0.05
        price = v + 0.3 * h if side == "buy" else v - 0.3 * h
        rows.append(long_row(strike=k, bid=v - h, ask=v + h, price=price, side=side))
        cases.append((v - h, v + h, v, price, side == "buy"))
    d, _ = ff.price_fills(fills_file(tmp_path, blob_of(rows)))
    # hand formulas: x = (price - mid) / half for a buy, (mid - price) / half for a sell; v likewise from V
    exp_x = [(p - (lo + hi) / 2.0) / ((hi - lo) / 2.0) if buy else ((lo + hi) / 2.0 - p) / ((hi - lo) / 2.0) for lo, hi, _, p, buy in cases]
    exp_v = [(V - (lo + hi) / 2.0) / ((hi - lo) / 2.0) if buy else ((lo + hi) / 2.0 - V) / ((hi - lo) / 2.0) for lo, hi, V, _, buy in cases]
    assert d["x"].to_numpy() == pytest.approx(exp_x, abs=1e-6)
    assert d["x"].to_numpy() == pytest.approx([0.3] * 4, abs=1e-6)
    assert d["v"].to_numpy() == pytest.approx(exp_v, abs=1e-6)
    assert d["mid"].to_numpy() == pytest.approx([(lo + hi) / 2.0 for lo, hi, _, _, _ in cases], abs=1e-9)
    assert d["half_spread"].to_numpy() == pytest.approx([(hi - lo) / 2.0 for lo, hi, _, _, _ in cases], abs=1e-9)
    assert d["bid"].to_numpy() == pytest.approx([lo for lo, _, _, _, _ in cases], abs=1e-9)
    assert d["ask"].to_numpy() == pytest.approx([hi for _, hi, _, _, _ in cases], abs=1e-9)
    assert d["spread_term"].to_numpy() == pytest.approx([min((hi - lo) / ((lo + hi) / 2.0) / 0.5, 1.0) for lo, hi, _, _, _ in cases], abs=1e-9)


# --- 2. group assignment from the side-hit size; unknown for empty size or open interest -------------------------------

def test_group_uses_side_hit_size_and_empty_size_or_oi_is_unknown(tmp_path):
    rows = [make_row(0.10, 0.3, "buy", hit=5.0),  # ask_size thin on a buy: size_only
            make_row(0.10, 0.3, "buy", bid_size=5.0),  # bid_size thin on a buy is not the side hit: unflagged
            make_row(0.10, 0.3, "sell", hit=5.0),  # bid_size thin on a sell: size_only
            make_row(0.10, 0.3, "sell", ask_size=5.0),  # ask_size thin on a sell is not the side hit: unflagged
            make_row(0.10, 0.3, "buy", hit=None),  # empty ask_size: unknown
            make_row(0.10, 0.3, "sell", oi=None)]  # empty open_interest: unknown
    d, _ = ff.price_fills(fills_file(tmp_path, blob_of(rows)))
    assert d["group"].tolist() == ["size_only", "unflagged", "size_only", "unflagged", "unknown", "unknown"]


# --- 3. window edges and offsets --------------------------------------------------------------------------------------

def test_window_edges_kept_and_one_second_over_excluded(tmp_path):
    rows = base_rows()
    rows.append(make_row(0.05, 0.3, "buy", snapshot_time="2030-03-04T14:30:00-05:00"))  # exactly 60 min before: kept
    rows.append(make_row(0.05, 0.3, "buy", snapshot_time="2030-03-04T16:30:00-05:00"))  # exactly 60 min after: kept
    rows.append(make_row(0.05, 0.3, "buy", snapshot_time="2030-03-04T16:30:01-05:00"))  # 60 min 1 s: excluded
    rows.append(make_row(0.05, 0.3, "sell", fill_time="2030-03-04T20:30:00+00:00", snapshot_time=SNAP))  # UTC offset
    _, receipt = ff.price_fills(fills_file(tmp_path, blob_of(rows)))
    assert receipt["counts"]["outside_window"] == 1
    assert receipt["counts"]["used"] == 123


def test_different_offsets_compare_by_instant(tmp_path):
    rows = base_rows()
    rows.append(make_row(0.05, 0.3, "buy", fill_time="2030-03-04T20:30:00+00:00", snapshot_time="2030-03-04T15:30:00-05:00"))
    _, receipt = ff.price_fills(fills_file(tmp_path, blob_of(rows)))
    assert receipt["counts"]["outside_window"] == 0


# --- 4. each exclusion counted once -----------------------------------------------------------------------------------

def test_each_exclusion_counted_once(tmp_path):
    rows = base_rows()
    rows.append(make_row(0.05, 0.3, "buy", bid=1.05, ask=0.95))  # ask < bid: unpriceable
    rows.append(long_row(rates_date="2030-02-24", smvVol=0.25))  # rates 8 days old: rate_unknown
    v = bs_call(101.0, 100.0, 30 / 365, 0.04, 0.25)  # quote around V so the row is priced, not clipped to the ask
    rows.append(long_row(rates_date="2030-02-25", bid=v - 0.05, ask=v + 0.05, price=v + 0.01))  # 7 days old: accepted
    rows.append(long_row(smvVol=None))  # no smile fallback: no_fair_value
    rows.append(make_row(0.0, 0.3, "buy"))  # bid == ask == 1.0, not repaired: zero_half_spread
    _, receipt = ff.price_fills(fills_file(tmp_path, blob_of(rows)))
    counts = receipt["counts"]
    assert counts["unpriceable"] == 1
    assert counts["rate_unknown"] == 1
    assert counts["no_fair_value"] == 1
    assert counts["zero_half_spread"] == 1
    assert counts["outside_window"] == 0


def test_v_clamped_to_the_quote_edge_is_used_with_v_plus_one(tmp_path):
    rows = [make_row(0.05, 0.0, "buy", spot=105.0, price=1.02),  # intrinsic 5 above the ask 1.05: V clamped to the ask
            make_row(0.05, 0.0, "sell", spot=95.0, price=0.98)]  # intrinsic 0 below the bid 0.95: V clamped to the bid
    d, receipt = ff.price_fills(fills_file(tmp_path, blob_of(rows)))
    assert receipt["counts"]["used"] == 2
    assert d["v"].to_numpy() == pytest.approx([1.0, 1.0], abs=1e-9)
    # buy x = (1.02 - 1) / 0.05; sell x = (1 - 0.98) / 0.05
    assert d["x"].to_numpy() == pytest.approx([0.4, 0.4], abs=1e-9)


def test_non_repaired_zero_half_spread_is_counted_not_used(tmp_path):
    rows = base_rows() + [make_row(0.0, 0.3, "buy"), make_row(0.0, 0.3, "sell")]  # bid == ask == 1.0, not repaired
    _, receipt = ff.price_fills(fills_file(tmp_path, blob_of(rows)))
    assert receipt["counts"]["zero_half_spread"] == 2
    assert receipt["counts"]["unpriceable"] == 0
    assert receipt["counts"]["used"] == 120


def test_one_cent_spread_is_used(tmp_path):
    rows = [make_row(0.05, 0.3, "buy", bid=1.00, ask=1.01, price=1.0075)]  # mid 1.005, half 0.005; V = 1 is inside
    d, receipt = ff.price_fills(fills_file(tmp_path, blob_of(rows)))
    assert receipt["counts"]["used"] == 1
    assert receipt["counts"]["zero_half_spread"] == 0
    # x = (1.0075 - 1.005) / 0.005 = 0.5; v = (1 - 1.005) / 0.005 = -1
    assert d["x"].to_numpy() == pytest.approx([0.5], abs=1e-9)
    assert d["v"].to_numpy() == pytest.approx([-1.0], abs=1e-9)


def test_out_of_range_rate_and_borrow_are_counted_and_treated_as_missing(tmp_path):
    rows = [long_row(iRate5wk=0.5, iRateLt=0.5),  # both rates above 0.20: both cells counted, both treated as missing
            long_row(borrow30=2.0),  # borrow above 0.2
            long_row(borrow2yr=-1.5)]  # borrow below -1.0
    _, receipt = ff.price_fills(fills_file(tmp_path, blob_of(rows)))
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
        ff.price_fills(p)
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

def test_receipt_keys_and_hashes_are_exact(tmp_path):
    blob = blob_of(base_rows())
    _, receipt = ff.price_fills(fills_file(tmp_path, blob))
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


# --- 8. read cap: enforced inside price_fills, before any parse ------------------------------------------------------

def test_fills_one_byte_over_16_mib_refused_before_parsing(tmp_path, monkeypatch):
    rows = base_rows()
    for r in rows:
        r["pad"] = ""  # an extra column that pricing ignores
    base_len = len(blob_of(rows))
    rows[0]["pad"] = "x" * (16 * 1024 * 1024 + 1 - base_len)  # the only padding, so the file is exactly 16 MiB + 1 byte
    blob = blob_of(rows)
    assert len(blob) == 16 * 1024 * 1024 + 1

    def no_parse(*args, **kwargs):
        raise AssertionError("pandas.read_csv must not be called")

    p = tmp_path / "fills.csv"
    p.write_bytes(blob)
    out = tmp_path / "out"
    with monkeypatch.context() as m:
        m.setattr(pd, "read_csv", no_parse)
        with pytest.raises(fvi.FairValueRefusal) as err:
            ff.price_fills(p)
        assert err.value.code == "RESOURCE_LIMIT_EXCEEDED"
        assert err.value.details["reason"] == "input_too_large"
        with pytest.raises(fvi.FairValueRefusal) as err:
            ff.run_fit(p, out)
        assert err.value.code == "RESOURCE_LIMIT_EXCEEDED"
        assert err.value.details["reason"] == "input_too_large"
    assert not out.exists()
