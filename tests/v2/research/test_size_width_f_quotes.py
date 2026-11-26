"""Tests for the F quote adapter (experiments/size_width_f_quotes.py): ladder rows, leg tables, admission and refusals.

Synthetic chains feed both the ladder chunk (through a fake index) and F's snapshot; tables are pinned parquet.
"""
import hashlib
import json
import logging
import math

import numpy as np
import pandas as pd
import pytest

from engine.fills import FillModel
from experiments import fair_value_fills as fvf
from experiments import fair_value_inputs as fvi
from experiments import fair_value_run as fvr
from experiments import size_pnl_ladder as lad
from experiments import size_pnl_shortvol as sv
from experiments import size_width_f_quotes as wf
from experiments.fair_value_inputs import FairValueRefusal

NAN = float("nan")
PRED = 6.0
SPOT = 100.0
ENTRY, EXIT, EXPIRY = "2024-02-05", "2024-02-08", "2024-03-15"
STRIKES = [float(k) for k in range(80, 121)]
OK = {"files": 2, "rows": 0, "unreadable": 0, "unrecognized": 0}
UNITS = {"borrow30": "percent", "borrow2yr": "percent", "divAmt": "dollars_per_share"}
LADDER_CASES = [
    (True, [("z", "P", 100.0, 0.0, 0.5), ("t", "P", 100.0, 1.0, 1.2), ("u", "C", 105.0, 1.0, 1.2)], ["t", "u"]),
    (True, [("t", "P", 100.0, 1.0, 1.2), ("z", "P", 100.0, 0.0, 0.5), ("u", "C", 105.0, 1.0, 1.2)], ["t", "u"]),
    (True, [("i", "P", 100.0, NAN, 1.0), ("z", "P", 100.0, 0.0, 0.5)], ["z"]),          # zero bid beats invalid
    (True, [("c", "P", 100.0, 2.0, 1.0), ("n", "P", 100.0, -0.1, 0.5)], ["c"]),        # both invalid: scan order
    (True, [("t1", "P", 100.0, 1.0, 1.2), ("t2", "P", 100.0, 1.0, 1.2)], ["t1"]),      # tie: scan order
    (True, [("z0", "P", 100.0, 0.0, 0.0), ("t", "P", 100.0, 0.9, 1.1)], ["t"]),        # ask 0 is invalid
    (False, [("i", "P", 100.0, NAN, 1.0), ("u", "P", 100.0, 1.0, 1.2)], ["u"]),
    (False, [("c", "P", 100.0, 2.0, 1.0), ("z", "P", 100.0, 0.0, 0.5)], ["z"]),       # exit: usable beats crossed
    (False, [("z0", "P", 100.0, 0.0, 0.0), ("t", "P", 100.0, 1.0, 1.2)], ["z0"]),     # exit tie: scan order
    (False, [("n", "P", 100.0, -0.1, 0.5), ("t", "P", 100.0, 1.0, 1.2)], ["t"]),
]


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _ts(value):
    return pd.Timestamp(value)


def _mid(k):
    return 1.0 + 0.01 * (k - 100.0) ** 2


def _bs_put(k, t, *, s=SPOT, v=0.25):
    """Closed-form Black-Scholes put at zero rate and no dividend (equals the American put here)."""
    def N(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    d1 = (math.log(s / k) + 0.5 * v * v * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    return k * N(-d2) - s * N(-d1)


def _row(ticker, obs, right, strike, bid, ask, *, expiry=EXPIRY, dte=30):
    return {"ticker": ticker, "obs_date": _ts(obs), "expiry": _ts(expiry), "dte": dte, "right": right,
            "strike": strike, "bid": bid, "ask": ask, "iv": 0.3, "spot": SPOT, "quote_repaired": False,
            "open_interest": 500.0, "bid_size": 50.0, "ask_size": 50.0, "chain_kind": "fetch"}


def _one(*args, **kwargs):
    return pd.DataFrame([_row(*args, **kwargs)])


def _listing(ticker, obs, *, expiry=EXPIRY, strikes=STRIKES, quotes=None, skip=()):
    """Listed put and call rows (bid/ask = mid -/+ 0.1); ``quotes`` overrides (right, strike) -> (bid, ask)."""
    quotes = quotes or {}
    rows = []
    for k in strikes:
        for right in ("P", "C"):
            if (right, k) in skip:
                continue
            bid, ask = quotes.get((right, k), (_mid(k) - 0.1, _mid(k) + 0.1))
            rows.append(_row(ticker, obs, right, k, bid, ask, expiry=expiry))
    return pd.DataFrame(rows)


def _frames(entry=ENTRY, exit_=EXIT):
    """Listed chains for AAA and BBB on the entry and exit dates, keyed as the ladder index is."""
    return {(t, _ts(d)): _listing(t, d) for t in ("AAA", "BBB") for d in (entry, exit_)}


def _base(frames):
    return pd.concat(frames.values(), ignore_index=True)


def _plan(*events):
    """Plan rows (event_id, ticker, entry, exit); the event falls the day after entry, AMC."""
    return pd.DataFrame([{"event_id": e, "ticker": t, "event_date": _ts(en) + pd.Timedelta(days=1), "session": "AMC",
                          "entry_date": _ts(en), "exit_date": _ts(ex)} for e, t, en, ex in events])


DEFAULT_PLAN = _plan(("E0", "AAA", ENTRY, EXIT), ("E1", "BBB", ENTRY, EXIT))


class _Index:
    def __init__(self, frames):
        self._frames = frames

    def get(self, ticker, date):
        return self._frames.get((ticker, _ts(date).normalize()))


def _chunk(frames, plan):
    index, recs = _Index(frames), []
    for row in plan.to_dict("records"):
        skip, _, rec = lad._event(row, index, lad.SELECTOR)
        assert skip is None, skip
        recs.append(rec)
    return lad._pack(recs, {})


def _idx(chunk, i, right, strike):
    code = 0 if right == "P" else 1
    return next(j for j in range(chunk["offsets"][i], chunk["offsets"][i + 1])
                if chunk["right"][j] == code and chunk["strike"][j] == strike)


def _write(tmp_path, base, rates=(5.0, 4.0)):
    """Pinned tables root: quotes and rates split by month of obs_date (Jan and Feb 2024), plus a manifest."""
    root = tmp_path / "root"
    vols = base[fvi.VOL_KEYS].drop_duplicates().assign(smvVol=25.0)
    # Zero borrow (percent) and zero dividend: carry b = r when the rate is zero, and no dividend is applied.
    carry = {"borrow30": 0.0, "borrow2yr": 0.0, "divAmt": 0.0, "divDate": pd.NaT, "annIdiv": 0.0, "annActDiv": 0.0,
             "assetType": 1.0}
    rate_frame = base[["ticker", "obs_date"]].drop_duplicates().assign(iRate5wk=rates[0], iRateLt=rates[1], **carry)
    for kind, frame in (("quotes", vols), ("rates", rate_frame)):
        for month in (1, 2):
            path = root / kind / "year=2024" / f"month={month:02d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame[frame["obs_date"].dt.month == month].to_parquet(path, index=False)
    manifest = {"version": "3", "tables": {"quotes": OK, "rates": OK}, "units": UNITS}
    (root / "manifest.json").write_bytes(json.dumps(manifest).encode())
    return root


def _pins(root):
    files = {p.relative_to(root).as_posix(): _sha(p.read_bytes()) for p in root.rglob("*.parquet")}
    return {"manifest_sha256": _sha((root / "manifest.json").read_bytes()), "files": files}


def _reader(snap, calls=None):
    """Fake read_quotes: the snapshot rows of the requested (ticker, obs_date, expiry) triples."""
    def read(expiries):
        if calls is not None:
            calls.append(expiries.copy())
        want = pd.MultiIndex.from_arrays([expiries["ticker"], expiries["obs_date"], expiries["expiry"]])
        have = pd.MultiIndex.from_arrays([snap["ticker"], pd.to_datetime(snap["obs_date"]),
                                          pd.to_datetime(snap["expiry"])])
        out = snap[have.isin(want)][fvi.QUOTE_COLUMNS]
        return out.reset_index(drop=True) if len(out) else pd.DataFrame(columns=fvi.QUOTE_COLUMNS)
    return read


def _setup(tmp_path, frames, plan, *, snap=None, read=None, calls=None, pins_edit=None, expected="snap-1",
           rates=(5.0, 4.0)):
    base = _base(frames)
    root = _write(tmp_path, base, rates)
    pins = _pins(root)
    if pins_edit is not None:
        pins_edit(pins)
    fq = wf.FQuotes(plan, read or _reader(base if snap is None else snap, calls), tables_root=root, pins=pins,
                    snapshot_id="snap-1", expected_snapshot_id=expected, vol_scale=0.01, rate_scale=0.01)
    return fq, _chunk(frames, plan)


def _match(frame, ticker, obs, right, strike):
    return ((frame["ticker"] == ticker) & (frame["obs_date"] == _ts(obs)) & (frame["right"] == right)
            & (frame["strike"] == strike))


def _set(frame, ticker, obs, right, strike, **values):
    out = frame.copy()
    out.loc[_match(out, ticker, obs, right, strike), list(values)] = list(values.values())
    return out


def _drop(frame, ticker, obs, right, strike):
    return frame[~_match(frame, ticker, obs, right, strike)]


def _resolved(chunk, i=0):
    priced, legs, reason = sv.resolve_priced(chunk, i, "CND-PS", PRED, FillModel(0.5))
    assert reason is None
    return legs


def _leg(chunk, i, right, strike, side="sell"):
    return {"idx": _idx(chunk, i, right, strike), "side": side, "qty": 1.0}


def _legs_frame(fq, chunk, legs, *, i=0, at_exit=False):
    out = fq.legs(chunk, i, legs, exit=at_exit)
    return out.assign(side=[leg["side"] for leg in legs], qty=[leg["qty"] for leg in legs])


def _refused(exc, reason):
    assert exc.value.details["reason"] == reason
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def _rows(spec):
    return pd.DataFrame([{"tag": t, "ticker": "AAA", "obs_date": _ts(ENTRY), "expiry": _ts(EXPIRY), "right": r,
                          "strike": k, "bid": b, "ask": a} for t, r, k, b, a in spec])


@pytest.mark.parametrize("entry,spec,expect", LADDER_CASES)
def test_ladder_rows_keeps_the_row_the_ladder_keeps(entry, spec, expect):
    rows = _rows(spec)
    kept, dropped = wf.ladder_rows(rows, entry=entry)
    assert kept["tag"].tolist() == expect
    assert dropped == len(spec) - len({(r, k) for _, r, k, _, _ in spec})
    raw = {t: (b, a) for t, _, _, b, a in spec}
    win = kept[(kept["right"] == "P") & (kept["strike"] == 100.0)].iloc[0]
    np.testing.assert_array_equal([win["bid"], win["ask"]], raw[win["tag"]])  # stored values, never blanked
    b, a = raw[win["tag"]]
    ok = (not entry) or (np.isfinite(b) and np.isfinite(a) and b <= a and a > 0 and b >= 0)
    got = lad._contracts(rows, entry=entry)
    got = got[(got["right"] == 0) & (got["strike"] == 100.0)].iloc[0]
    np.testing.assert_array_equal([got["bid"], got["ask"]], [b, a] if ok else [NAN, NAN])


def test_ladder_rows_counts_duplicates_and_keeps_unconvertible_keys():
    spec = [("a", "P", 100.0, 1.0, 1.2), ("a2", "P", 100.0, 1.0, 1.2), ("n1", "P", None, 1.0, 1.2),
            ("n2", "P", None, 1.0, 1.2), ("x", "P", "abc", 1.0, 1.2), ("x2", "P", "abc", 1.0, 1.2)]
    kept, dropped = wf.ladder_rows(_rows(spec), entry=True)
    assert kept["tag"].tolist() == ["a", "n1", "n2", "x", "x2"] and dropped == 1


def test_table_prices_every_listed_row_of_the_expiry_and_excludes_other_expiries(tmp_path):
    frames = _frames()
    frames[("AAA", _ts(ENTRY))] = pd.concat([
        _listing("AAA", ENTRY, quotes={("P", 90.0): (2.0, 1.0)}),
        _listing("AAA", ENTRY, expiry="2024-04-19", strikes=[100.0])], ignore_index=True)
    base = _base(frames)
    fq, chunk = _setup(tmp_path, frames, DEFAULT_PLAN, read=lambda expiries: base[fvi.QUOTE_COLUMNS])  # unfiltered
    table = fq.table(chunk)
    assert set(table["expiry"]) == {_ts(EXPIRY)}  # the 2024-04-19 row is merged away
    assert len(table[table["ticker"] == "AAA"]) == 82  # every listed row, including the crossed 90 put
    key = ("AAA", _ts(ENTRY), _ts(EXPIRY), "P", 90.0)
    assert key in table.index and table.loc[key, "bid"] == pytest.approx(2.0)


def test_table_is_built_once_per_side_and_rebuilt_for_a_new_chunk(tmp_path):
    calls, frames = [], _frames()
    fq, chunk = _setup(tmp_path, frames, DEFAULT_PLAN, calls=calls)
    first = fq.table(chunk)
    assert fq.table(chunk) is first
    assert len(calls) == 1 and set(calls[0]["obs_date"]) == {_ts(ENTRY)}  # entry never asks for exit dates
    fq.table(chunk, exit=True)
    fq.table(chunk, exit=True)
    assert len(calls) == 2 and set(calls[1]["obs_date"]) == {_ts(EXIT)}
    other = _chunk(frames, DEFAULT_PLAN)
    assert fq.table(other) is not first
    assert len(calls) == 3


def test_a_date_that_is_one_events_entry_and_another_events_exit_ranks_per_side(tmp_path):
    frames = {("AAA", _ts(ENTRY)): _listing("AAA", ENTRY),
              ("AAA", _ts(EXIT)): pd.concat([_one("AAA", EXIT, "P", 100.0, 0.0, 0.0), _listing("AAA", EXIT)],
                                            ignore_index=True),
              ("AAA", _ts("2024-02-12")): _listing("AAA", "2024-02-12")}
    plan = _plan(("E0", "AAA", ENTRY, EXIT), ("E1", "AAA", EXIT, "2024-02-12"))
    fq, chunk = _setup(tmp_path, frames, plan)
    key = ("AAA", _ts(EXIT), _ts(EXPIRY), "P", 100.0)
    assert fq.table(chunk).loc[key, "bid"] == pytest.approx(0.9)  # E1 enters here: the tradable row wins
    assert fq.table(chunk, exit=True).loc[key, "bid"] == pytest.approx(0.0)  # E0 exits here: first usable row
    fq.legs(chunk, 0, [_leg(chunk, 0, "P", 100.0)], exit=True)
    fq.legs(chunk, 1, [_leg(chunk, 1, "P", 100.0)])


def test_legs_keep_leg_order_and_feed_structure_entry_and_exit(tmp_path):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    legs = list(reversed(_resolved(chunk, 0)))
    assert any(leg["qty"] == 0 for leg in legs)
    entry = _legs_frame(fq, chunk, legs)
    assert entry["strike"].tolist() == [float(chunk["strike"][leg["idx"]]) for leg in legs]
    res = fvf.structure_entry(entry[entry["qty"] > 0])
    assert res.outcome == "skip"
    assert res.skip_reason == "fill_over_fair"
    assert res.debit_mid == pytest.approx(0.54)
    assert res.spread_sum == pytest.approx(0.80)
    assert res.debit_F > res.debit_fair
    exit_ = _legs_frame(fq, chunk, legs, at_exit=True)
    assert fvf.structure_exit(exit_[exit_["qty"] > 0]).outcome == "ok"
    for col in ("repaired", "unknown_bid_size", "unknown_ask_size", "unknown_oi", "carry_fallback", "dividend_fallback",
                "dividend_applied"):
        assert entry[col].dtype == bool and exit_[col].dtype == bool
    for i, leg in enumerate(legs):
        idx = leg["idx"]
        assert entry.loc[i, "right"] == ("P" if chunk["right"][idx] == 0 else "C")
        assert entry.loc[i, "bid"] == pytest.approx(chunk["bid_e"][idx])
        assert entry.loc[i, "ask"] == pytest.approx(chunk["ask_e"][idx])
        assert exit_.loc[i, "right"] == ("P" if chunk["right"][idx] == 0 else "C")
        assert exit_.loc[i, "bid"] == pytest.approx(chunk["bid_x"][idx])
        assert exit_.loc[i, "ask"] == pytest.approx(chunk["ask_x"][idx])


def test_blanked_entry_quote_with_a_stored_crossed_row_is_not_a_mismatch(tmp_path):
    frames = _frames()
    frames[("AAA", _ts(ENTRY))] = _listing("AAA", ENTRY, quotes={("P", 100.0): (1.2, 1.0)})
    fq, chunk = _setup(tmp_path, frames, DEFAULT_PLAN)
    idx = _idx(chunk, 0, "P", 100.0)
    assert np.isnan(chunk["bid_e"][idx]) and np.isnan(chunk["ask_e"][idx])  # blanked by the ladder
    out = fq.legs(chunk, 0, [{"idx": idx, "side": "buy", "qty": 1.0}])
    assert out.loc[0, "bid"] == pytest.approx(1.2) and out.loc[0, "ask"] == pytest.approx(1.0)  # stored, unblanked


@pytest.mark.parametrize("date,at_exit", [(ENTRY, False), (EXIT, True)], ids=["entry", "exit"])
def test_changed_stored_bid_is_f_quote_mismatch(tmp_path, date, at_exit):
    base = _base(_frames())
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN, snap=_set(base, "AAA", date, "P", 100.0, bid=0.95))
    with pytest.raises(FairValueRefusal) as exc:
        fq.legs(chunk, 0, [_leg(chunk, 0, "P", 100.0)], exit=at_exit)
    _refused(exc, "f_quote_mismatch")


@pytest.mark.parametrize("date,at_exit", [(ENTRY, False), (EXIT, True)], ids=["entry", "exit"])
def test_changed_stored_ask_is_f_quote_mismatch(tmp_path, date, at_exit):
    base = _base(_frames())
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN, snap=_set(base, "AAA", date, "P", 100.0, ask=1.05))
    with pytest.raises(FairValueRefusal) as exc:
        fq.legs(chunk, 0, [_leg(chunk, 0, "P", 100.0)], exit=at_exit)
    _refused(exc, "f_quote_mismatch")


@pytest.mark.parametrize("date,at_exit", [(ENTRY, False), (EXIT, True)], ids=["entry", "exit"])
def test_listed_chunk_quote_without_a_row_is_f_quote_mismatch(tmp_path, date, at_exit):
    base = _base(_frames())
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN, snap=_drop(base, "AAA", date, "P", 100.0))
    with pytest.raises(FairValueRefusal) as exc:
        fq.legs(chunk, 0, [_leg(chunk, 0, "P", 100.0)], exit=at_exit)
    _refused(exc, "f_quote_mismatch")


def test_exit_contract_not_listed_on_exit_date_is_unpriceable(tmp_path):
    frames = _frames()
    frames[("AAA", _ts(EXIT))] = _listing("AAA", EXIT, skip=[("P", 100.0)])
    fq, chunk = _setup(tmp_path, frames, DEFAULT_PLAN)
    legs = [_leg(chunk, 0, "P", 100.0), _leg(chunk, 0, "P", 99.0, side="buy")]
    out = _legs_frame(fq, chunk, legs, at_exit=True)
    assert out.loc[0, "status"] == "unpriceable" and out.loc[0, "thin_oi"] == fvf.UNKNOWN_THIN
    res = fvf.structure_exit(out)
    assert res.outcome == "unpriceable" and res.leg_flags[0]["repaired"] is False


def test_all_exit_quotes_missing_gives_unpriceable_exit_legs(tmp_path):
    frames = {k: v for k, v in _frames().items() if k[1] == _ts(ENTRY)}
    fq, chunk = _setup(tmp_path, frames, DEFAULT_PLAN)
    assert np.isnan(chunk["bid_x"]).all()
    assert len(fq.table(chunk, exit=True)) == 0
    out = _legs_frame(fq, chunk, _resolved(chunk, 0), at_exit=True)
    assert (out["status"] == "unpriceable").all()
    traded = out[out["qty"] > 0]
    assert fvf.structure_exit(traded).outcome == "unpriceable"
    assert fvf.structure_exit(traded).leg_flags[0]["repaired"] is False


def test_snapshot_mismatch_is_refused_before_any_read(tmp_path):
    calls = []
    with pytest.raises(FairValueRefusal) as exc:
        wf.FQuotes(DEFAULT_PLAN, _reader(_base(_frames()), calls), tables_root=tmp_path,
                   pins={"manifest_sha256": "0" * 64, "files": {}}, snapshot_id="snap-1",
                   expected_snapshot_id="snap-2", vol_scale=0.01, rate_scale=0.01)
    _refused(exc, "snapshot_binding")
    assert calls == []


@pytest.mark.parametrize("pins", [{}, None, {"manifest_sha256": "x", "files": None}], ids=["empty", "none", "files-null"])
def test_malformed_pins_are_refused_before_any_read(tmp_path, pins):
    calls = []
    with pytest.raises(FairValueRefusal) as exc:
        wf.FQuotes(DEFAULT_PLAN, _reader(_base(_frames()), calls), tables_root=tmp_path, pins=pins,
                   snapshot_id="snap-1", expected_snapshot_id="snap-1", vol_scale=0.01, rate_scale=0.01)
    _refused(exc, "pins_invalid")
    assert calls == []


BAD_PLANS = [
    pytest.param(DEFAULT_PLAN.drop(columns="event_id"), id="missing-event_id"),
    pytest.param(_plan(("E0", "AAA", ENTRY, EXIT), ("E0", "BBB", ENTRY, EXIT)), id="duplicate-event_id"),
    pytest.param(DEFAULT_PLAN.assign(entry_date=pd.NaT), id="null-entry_date"),
    pytest.param(DEFAULT_PLAN.assign(exit_date="not a date"), id="unparseable-exit_date"),
    pytest.param([1, 2], id="not-a-dataframe"),
]


@pytest.mark.parametrize("plan", BAD_PLANS)
def test_malformed_plan_is_refused_before_any_read(tmp_path, plan):
    calls = []
    with pytest.raises(FairValueRefusal) as exc:
        wf.FQuotes(plan, _reader(_base(_frames()), calls), tables_root=tmp_path,
                   pins={"manifest_sha256": "0" * 64, "files": {}}, snapshot_id="snap-1",
                   expected_snapshot_id="snap-1", vol_scale=0.01, rate_scale=0.01)
    _refused(exc, "plan_invalid")
    assert calls == []


def test_chunk_event_absent_from_the_plan_is_refused_before_any_read(tmp_path):
    frames, calls = _frames(), []
    chunk = _chunk(frames, DEFAULT_PLAN)  # two events
    fq, _ = _setup(tmp_path, frames, _plan(("E0", "AAA", ENTRY, EXIT)), calls=calls)  # plan holds only E0
    with pytest.raises(FairValueRefusal) as exc:
        fq.table(chunk)
    _refused(exc, "plan_invalid")
    assert calls == []


def test_batches_never_split_an_expiry_and_match_the_unbatched_table(tmp_path, monkeypatch, caplog):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    unbatched = fq.table(chunk)  # default BATCH_ROWS: one batch
    spied, real = [], fvf.leg_fills

    def spy(frame, vols):
        spied.append(frame.copy())
        return real(frame, vols)

    monkeypatch.setattr(fvf, "leg_fills", spy)
    monkeypatch.setattr(fvr, "BATCH_ROWS", 1)  # every expiry (82 rows) exceeds the bound: one batch each
    fq2, chunk2 = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    caplog.set_level(logging.INFO, logger="size_width_f_quotes")
    batched = fq2.table(chunk2)
    assert len(spied) == 2
    for frame in spied:
        assert len(frame) == 82 and frame["expiry"].nunique() == 1 and frame["ticker"].nunique() == 1
    pd.testing.assert_frame_equal(batched, unbatched)
    assert len([r for r in caplog.records if " batch " in r.getMessage()]) == 2


def test_duplicated_listed_rows_are_counted_per_side(tmp_path):
    base = _base(_frames())
    entry_row = base[_match(base, "AAA", ENTRY, "P", 100.0)]
    exit_row = base[_match(base, "AAA", EXIT, "P", 100.0)]
    snap = pd.concat([base, entry_row, exit_row, exit_row], ignore_index=True)
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN, snap=snap)
    fq.table(chunk)
    fq.table(chunk, exit=True)
    assert fq.counts == {"entry_duplicates": 1, "exit_duplicates": 2}


def test_table_hashes_match_the_manifest_and_the_loaded_table_files(tmp_path):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    fq.table(chunk)
    fq.table(chunk, exit=True)
    pins = _pins(tmp_path / "root")
    # Entry (Feb 5, 7-day lag reaches Jan 29): quotes and rates for Feb, rates for Jan. Exit (Feb 8, lag Feb 1): Feb only.
    loaded = ["quotes/year=2024/month=02.parquet", "rates/year=2024/month=01.parquet", "rates/year=2024/month=02.parquet"]
    assert fq.table_hashes == {"manifest.json": pins["manifest_sha256"], **{r: pins["files"][r] for r in loaded}}


def test_an_unpinned_month_is_table_absent(tmp_path):
    frames = _frames(entry="2024-01-29", exit_="2024-02-01")
    plan = _plan(("E0", "AAA", "2024-01-29", "2024-02-01"))
    fq, chunk = _setup(tmp_path, frames, plan,
                       pins_edit=lambda pins: pins["files"].pop("quotes/year=2024/month=01.parquet"))
    with pytest.raises(FairValueRefusal) as exc:
        fq.table(chunk)
    _refused(exc, "table_absent")


def test_a_wrong_table_hash_is_refused(tmp_path):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN,
                       pins_edit=lambda pins: pins["files"].update({"quotes/year=2024/month=02.parquet": "0" * 64}))
    with pytest.raises(FairValueRefusal) as exc:
        fq.table(chunk)
    _refused(exc, "table_hash")


def test_a_null_strike_in_a_requested_expiry_is_key_value(tmp_path):
    frames = _frames()
    snap = pd.concat([_base(frames), _one("AAA", ENTRY, "C", None, 1.0, 1.2)], ignore_index=True)
    fq, chunk = _setup(tmp_path, frames, DEFAULT_PLAN, snap=snap)
    with pytest.raises(FairValueRefusal) as exc:
        fq.table(chunk)
    _refused(exc, "key_value")


def test_table_prices_with_unit_scaled_vol_and_rate(tmp_path):
    frames = _frames()
    fq, chunk = _setup(tmp_path, frames, DEFAULT_PLAN)  # pinned vols 25.0 and rates 5.0 / 4.0 percent, scaled by 0.01
    got = fq.table(chunk)
    got = got[got["ticker"] == "AAA"]
    rows = frames[("AAA", _ts(ENTRY))][fvi.QUOTE_COLUMNS].reset_index(drop=True)
    days = 39  # (EXPIRY - ENTRY).days, by hand
    assert (_ts(EXPIRY) - _ts(ENTRY)).days == days
    rate = 0.05 + (days - 35) / (365 - 35) * (0.04 - 0.05)  # by hand: 5wk and long rates blended on days
    rows = rows.assign(rate=rate, borrow30=0.0, borrow2yr=0.0, divAmt=0.0, divDate=pd.NaT)  # as _write pins them
    vols = rows[fvi.VOL_KEYS].drop_duplicates().assign(smvVol=0.25)
    want = fvf.leg_fills(rows, vols)
    got = got.set_index(["right", "strike"])
    want = want.set_index(["right", "strike"])
    assert len(got) == len(want) == 82
    got = got.loc[want.index]
    for col in ("V", "vol", "rate", "fill_buy", "fill_sell"):
        np.testing.assert_allclose(got[col].to_numpy(dtype=float), want[col].to_numpy(dtype=float), err_msg=col)
    assert (got["status"] == "ok").all()
    assert (got["source"] == "smvvol").all()
    np.testing.assert_allclose(got["vol"].to_numpy(dtype=float), 0.25)
    put = got.loc[("P", 100.0)]
    assert put["bid"] <= put["V"] <= put["ask"]
    assert put["rate"] == pytest.approx(rate)


@pytest.mark.parametrize("date,at_exit", [(ENTRY, False), (EXIT, True)], ids=["entry", "exit"])
def test_leg_prices_and_fills_match_a_closed_form_oracle(tmp_path, date, at_exit):
    strikes = (95.0, 100.0, 105.0)
    frames, oracle = _frames(), {}
    for day in (ENTRY, EXIT):
        t = (_ts(EXPIRY) - _ts(day)).days / 365
        quotes = {}
        for k in strikes:
            p = _bs_put(k, t)
            bid, ask = round(p - 0.40, 2), round(p + 0.60, 2)
            assert bid > 0
            quotes[("P", k)] = (bid, ask)
            oracle[(day, k)] = p
        frames[("AAA", _ts(day))] = _listing("AAA", day, quotes=quotes)
    fq, chunk = _setup(tmp_path, frames, DEFAULT_PLAN, rates=(0.0, 0.0))
    out = fq.legs(chunk, 0, [_leg(chunk, 0, "P", k) for k in strikes], exit=at_exit)
    day = EXIT if at_exit else ENTRY
    if not at_exit:
        assert oracle[(ENTRY, 100.0)] == pytest.approx(3.26, abs=0.01)  # hard literal: ATM 39-day put at 25% vol
    assert out["status"].tolist() == ["ok"] * len(strikes)
    for i, k in enumerate(strikes):
        p = oracle[(day, k)]
        bid, ask = round(p - 0.40, 2), round(p + 0.60, 2)
        mid = (bid + ask) / 2
        s = (ask - bid) / mid
        c = min(1.0, 0.25 + 0.50 * min(s / 0.50, 1.0))
        row = out.iloc[i]
        assert row["V"] == pytest.approx(p, abs=0.01)
        assert row["fill_buy"] == pytest.approx(p + c * (ask - p), abs=0.01)
        assert row["fill_sell"] == pytest.approx(p - c * (p - bid), abs=0.01)
        assert abs(row["V"] - mid) > 0.05


@pytest.mark.parametrize("exit_", [False, True], ids=["entry", "exit"])
def test_quote_rows_at_the_cap_build_and_one_more_is_refused_before_any_pricing(tmp_path, monkeypatch, exit_):
    frames, seen = _frames(), []
    read = _reader(_base(frames))

    def counting(expiries):
        seen.append(read(expiries))
        return seen[-1]

    fq, chunk = _setup(tmp_path, frames, DEFAULT_PLAN, read=counting)
    fq.table(chunk, exit=exit_)
    count = len(seen[-1])
    monkeypatch.setattr(wf, "MAX_QUOTE_ROWS", count)
    fq, chunk = _setup(tmp_path, frames, DEFAULT_PLAN, read=counting)
    assert len(fq.table(chunk, exit=exit_)) == count  # exactly at the cap is allowed
    monkeypatch.setattr(wf, "MAX_QUOTE_ROWS", count - 1)
    calls, real = [], fvf.leg_fills

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(fvf, "leg_fills", spy)
    fq, chunk = _setup(tmp_path, frames, DEFAULT_PLAN, read=counting)
    with pytest.raises(FairValueRefusal) as exc:
        fq.table(chunk, exit=exit_)
    assert exc.value.details["reason"] == "input_too_large"
    assert exc.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert not fq.counts
    assert calls == []
