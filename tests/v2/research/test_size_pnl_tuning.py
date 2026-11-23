"""EXP-188 S6: divisor re-tune and tuning receipts. Synthetic ladders and a stub pool only; no repository."""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.portfolio import CONTRACT_MULTIPLIER
from engine.structures import BUY
from experiments import size_pnl_shortvol as sv
from experiments import size_pnl_tuning as tuning
from experiments.size_pnl_pool import build_pool
from experiments.size_pnl_straddle import lots

DRAW = 200
PRED = 6.0
BODY = {"stage": "tuning", "model_id": "M", "c": 1.1}
PNL = {0.5: 0.1, 0.6: 0.2, 0.67: 0.3, 0.75: 9.0, 0.9: 2.0, 1.0: 3.0}
CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
HOLDOUT = "HOLDOUT_ACCESS_DENIED"
UNREADABLE = "tuning_receipt_unreadable"

# The CSR chunk layout of experiments/size_pnl_ladder.py, rebuilt here so the test uses no private helper.
_SCALARS = {"event_id": object, "expiry": "datetime64[ns]", "spot_entry": np.float64,
            "spot_exit": np.float64, "dte_entry": np.int32, "dte_exit": np.int32}
_FLATS = {"right": np.int8, "strike": np.float64, "bid_e": np.float64, "ask_e": np.float64,
          "bid_x": np.float64, "ask_x": np.float64, "exit_rung": np.int8}


def pack(recs):
    """A chunk dict with per-event scalars, flat per-contract arrays, and CSR offsets."""
    offsets = np.zeros(len(recs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(r["right"]) for r in recs], dtype=np.int64)
    chunk = {"offsets": offsets, "skipped": {}, "exit_missing": {"no_exit_chain": 0, "expiry_gone_at_exit": 0}}
    chunk.update({k: np.array([r[k] for r in recs], dtype=dt) for k, dt in _SCALARS.items()})
    chunk.update({k: np.concatenate([np.zeros(0, dt)] + [np.asarray(r[k], dt) for r in recs])
                  for k, dt in _FLATS.items()})
    return chunk


def quote(k):
    """Entry (bid, ask) at strike ``k``: a convex mid of 1 + 0.01*(k-100)^2 with a 0.2 spread."""
    mid = 1.0 + 0.01 * (k - 100) ** 2
    return mid - 0.1, mid + 0.1


def rec(eid, spot, legs):
    cols = list(zip(*legs))
    return {"event_id": eid, "expiry": np.datetime64("2024-03-15"), "spot_entry": spot, "spot_exit": spot,
            "dte_entry": 30, "dte_exit": 29, "right": cols[0], "strike": cols[1], "bid_e": cols[2],
            "ask_e": cols[3], "bid_x": cols[4], "ask_x": cols[5], "exit_rung": cols[6]}


def put_legs(*, strikes=range(80, 121), entry=quote):
    """Put rows across ``strikes``, locked and rung 1 at exit."""
    return [(0, float(k), *entry(k), 1.0, 1.0, 1) for k in strikes]


def chunk_of(*eids, strikes=range(80, 121)):
    """Chunk with one put ladder per event id, spot 100."""
    return pack([rec(eid, 100.0, put_legs(strikes=strikes)) for eid in eids])


def events_for(*eids, pred=6.0, mcap=20e9, date="2024-06-10"):
    return pd.DataFrame([{"event_id": e, "event_date": pd.Timestamp(date), "pred_move": pred,
                          "pred_crush": -15.0, "pre_iv30": 40.0, "mcap_usd": mcap, "dte_exit": 9}
                         for e in eids])


def _tables(seed=7, n=600):
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 1096, n), unit="D")
    pred_move = rng.uniform(2, 8, n)
    pred_crush = rng.normal(-15, 3, n)
    index = pd.Index([f"E{i}" for i in range(n)], name="event_id")
    table = pd.DataFrame({"event_date": dates, "pred_move": pred_move, "pred_crush": pred_crush}, index=index)
    labels = pd.DataFrame({"abs_move": pred_move + rng.normal(0, 1.5, n),
                           "crush_pct_iv30": pred_crush + rng.normal(0, 5, n)}, index=index)
    return table, labels


@pytest.fixture
def pool():
    return build_pool(*_tables(), floor=None)


# ---- choose_c ---------------------------------------------------------------

def test_choose_c_takes_the_argmax():
    assert tuning.choose_c({0.75: 1.0, 0.9: 2.0, 1.0: 3.0, 1.1: 9.0}, 500) == (1.1, None)


def test_choose_c_tie_goes_to_the_c_nearest_one():
    assert tuning.choose_c({1.25: 5.0, 1.0: 5.0, 0.75: 1.0}, 300) == (1.0, None)
    assert tuning.choose_c({1.5: 5.0, 1.1: 5.0}, 300) == (1.1, None)


def test_choose_c_tie_between_0_75_and_1_1_goes_to_1_1():
    # 1.1 is nearer to 1 than 0.75 is, so the nearest rule beats the smaller-c rule.
    assert tuning.choose_c({0.75: 5.0, 1.1: 5.0}, 300) == (1.1, None)


def test_choose_c_equidistant_tie_goes_to_the_smaller_c():
    assert tuning.choose_c({1.1: 5.0, 0.9: 5.0}, 300) == (0.9, None)
    # 0.75 and 1.25 are both 0.25 from 1, so the smaller c wins.
    assert tuning.choose_c({0.75: 5.0, 1.25: 5.0}, 300) == (0.75, None)


def test_choose_c_below_min_trades_is_unity_with_reason():
    assert tuning.choose_c({0.75: 9.0, 1.0: 1.0}, 299) == (1.0, "fewer_than_300_inner_trades")
    assert tuning.choose_c({0.75: 9.0, 1.0: 1.0}, 300) == (0.75, None)


@pytest.mark.parametrize("totals", [{}, {1.0: math.nan}, {1.0: math.inf}])
def test_choose_c_refuses_empty_or_non_finite(totals):
    with pytest.raises(ValueError):
        tuning.choose_c(totals, 500)


# ---- tune: refusals and schema ----------------------------------------------

def test_tune_refuses_an_outer_year_before_scoring_anything(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("scoring must not run")
    monkeypatch.setattr(tuning, "score_events", boom)
    monkeypatch.setattr(tuning, "decide", boom)
    monkeypatch.setattr(tuning, "score", boom)
    chunk = chunk_of("A", "B")
    items = [(chunk, events_for("A"), {}), (chunk, events_for("B", date="2025-03-10"), {})]
    with pytest.raises(sv.Refusal) as caught:
        tuning.tune(items, "M")
    assert caught.value.code == HOLDOUT
    assert caught.value.details == {"reason": "outer_year_in_tuning", "year": 2025}


@pytest.mark.parametrize("date", ["2022-12-31", "2025-01-01"])
def test_outer_year_boundary_refuses(monkeypatch, date):
    def boom(*args, **kwargs):
        raise AssertionError("scoring must not run")
    monkeypatch.setattr(tuning, "score_events", boom)
    with pytest.raises(sv.Refusal) as caught:
        tuning.tune([(chunk_of("A"), events_for("A", date=date), {})], "M")
    assert caught.value.code == HOLDOUT


@pytest.mark.parametrize("date", ["2025-01-01", "2025-12-31"])
def test_outer_year_guard_refuses_string_year_2025_events(monkeypatch, date):
    # The guard reads each event's calendar year (an int), so a plan's string year "2025" cannot slip past it.
    def boom(*args, **kwargs):
        raise AssertionError("scoring must not run")
    monkeypatch.setattr(tuning, "score_events", boom)
    with pytest.raises(sv.Refusal) as caught:
        tuning.tune([(chunk_of("A"), events_for("A", date=date), {})], "M")
    assert caught.value.code == HOLDOUT
    assert caught.value.details == {"reason": "outer_year_in_tuning", "year": 2025}


class _Reached(Exception):
    """Raised by a stub scorer to show that the year guard let the items through."""


@pytest.mark.parametrize("date", ["2023-01-01", "2024-12-31"])
def test_inner_year_boundary_passes_the_guard(monkeypatch, date):
    def reached(*args, **kwargs):
        raise _Reached
    monkeypatch.setattr(tuning, "score_events", reached)
    with pytest.raises(_Reached):
        tuning.tune([(chunk_of("A"), events_for("A", date=date), {})], "M")


@pytest.mark.parametrize("kwargs, items_empty", [({"grid": (0.9, 1.1)}, False), ({}, True)])
def test_tune_refuses_a_grid_without_unity_or_no_items(kwargs, items_empty):
    items = [] if items_empty else [(chunk_of("A"), events_for("A"), {})]
    with pytest.raises(ValueError):
        tuning.tune(items, "M", **kwargs)


def test_tune_on_a_tiny_fixture_is_skipped_at_unity(pool):
    chunk, events = chunk_of("A"), events_for("A")
    sims = sv.simulate(chunk, events, pool, draws=DRAW)
    out = tuning.tune([(chunk, events, sims)], "M1")
    assert set(out) == {"stage", "model_id", "alpha", "grid", "c", "skipped", "skip_reason", "trades", "totals",
                        "declined"}
    assert out["stage"] == "tuning" and out["model_id"] == "M1" and out["alpha"] == 0.5
    assert out["grid"] == list(tuning.C_GRID)
    assert set(out["totals"]) == {str(c) for c in tuning.C_GRID}
    assert out["skipped"] is True and out["skip_reason"] == "fewer_than_300_inner_trades"
    assert out["c"] == 1.0 and out["trades"] == 0  # one event has no prior history, so nothing is taken


# ---- tune: argmax and per-c decisions with stubs ----------------------------

def test_tune_picks_the_argmax_and_recomputes_decisions_per_c(monkeypatch):
    monkeypatch.setattr(tuning, "MIN_TRADES", 1)
    calls = []  # (structure, c) of every decide call

    def fake_score_events(chunk, events, sims, name, *, alpha=0.5, c=1.0):
        return pd.DataFrame({"event_id": events["event_id"].astype(str), "structure": name, "c": c, "reason": None})

    def fake_decide(scored):
        calls.append((scored["structure"].iloc[0], scored["c"].iloc[0]))
        return scored.assign(status="taken", entry_cost=1.0, alpha=0.5)

    def fake_score(chunk, events, decisions, name, alpha, *, c=1.0):
        return pd.DataFrame({"event_id": decisions["event_id"], "structure": name,
                             "pnl_dollars": [PNL[c]] * len(decisions)})

    monkeypatch.setattr(tuning, "score_events", fake_score_events)
    monkeypatch.setattr(tuning, "decide", fake_decide)
    monkeypatch.setattr(tuning, "score", fake_score)
    out = tuning.tune([(chunk_of("A", "B"), events_for("A", "B"), {})], "M")
    assert sorted(calls) == sorted((n, c) for n in tuning.NAMES for c in tuning.C_GRID)
    assert out["c"] == 0.75 and out["skipped"] is False and out["skip_reason"] is None
    assert out["trades"] == 2 * len(tuning.NAMES)
    assert out["totals"]["0.75"] == pytest.approx(9.0 * 2 * len(tuning.NAMES))
    assert out["declined"] == {str(c): 0.0 for c in tuning.C_GRID}


# ---- grid, tie rule and width declines --------------------------------------

def test_grid_includes_the_wider_points():
    assert tuning.C_GRID == (0.5, 0.6, 0.67, 0.75, 0.9, 1.0)


def test_choose_c_tie_rule_with_new_points():
    assert tuning.choose_c({0.5: 5.0, 1.5: 5.0}, 300) == (0.5, None)  # both 0.5 from 1: smaller c wins
    assert tuning.choose_c({0.67: 5.0, 0.9: 5.0}, 300) == (0.9, None)  # 0.1 from 1 beats 0.33
    assert tuning.choose_c({0.5: 9.0, 1.0: 1.0}, 300) == (0.5, None)


def test_c_half_declines_widths_beyond_width_max_and_counts_them(pool):
    chunk, events = chunk_of("A"), events_for("A", pred=12.0)
    assert sv.resolve(chunk, 0, "TWIN-P", 12.0, c=0.5) == (None, "width_declined")  # width 0.16 > WIDTH_MAX 0.15
    sims = sv.simulate(chunk, events, pool, draws=DRAW)
    scored = sv.score_events(chunk, events, sims, "TWIN-P", c=0.5)
    assert "width_declined" in set(scored["reason"])
    out = tuning.tune([(chunk, events, sims)], "M")
    for c in (0.5, 1.0):
        rows = [sv.score_events(chunk, events, sims, name, c=c) for name in tuning.NAMES]
        expected = sum((r["reason"] == "width_declined").sum() for r in rows) / sum(len(r) for r in rows)
        assert out["declined"][str(c)] == pytest.approx(expected)
    assert 0 < out["declined"]["0.5"] <= 1
    assert out["declined"]["0.5"] > out["declined"]["1.0"]


# ---- tune: real scorers against a hand-computed oracle ----------------------

N_REAL = 300  # ~25 events a month across 2023, so every 6-month trailing window clears MIN_WINDOW = 100


def real_chunk():
    """N_REAL events, each a put ladder whose entry and exit quotes are locked (bid == ask) and scale per event."""
    recs = []
    for i in range(N_REAL):
        ent = 0.8 + 0.03 * ((i * 13) % 17)  # entry mid scale: non-monotone, cheap-entry events recur
        ext = 0.6 + 0.011 * ((i * 7) % 50)  # exit mid scale
        legs = []
        for k in range(80, 121):
            mid_e = (1.0 + 0.01 * (k - 100) ** 2) * ent
            mid_x = (1.0 + 0.01 * (k - 100) ** 2) * ext
            legs.append((0, float(k), mid_e, mid_e, mid_x, mid_x, 1))
        recs.append(rec(f"R{i:03d}", 100.0, legs))
    return pack(recs)


def real_events():
    """N_REAL events spread evenly across calendar 2023, so all sit in the inner years and span every month."""
    rows = []
    for i in range(N_REAL):
        day = int(i * 365 / N_REAL)  # strictly increasing since N_REAL < 365; max day 364 stays in 2023
        rows.append({"event_id": f"R{i:03d}", "event_date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=day),
                     "pred_move": PRED, "pred_crush": -15.0, "pre_iv30": 40.0, "mcap_usd": 20e9, "dte_exit": 9})
    return pd.DataFrame(rows)


def test_tune_matches_a_hand_computed_oracle_at_unity(pool, monkeypatch):
    monkeypatch.setattr(tuning, "MIN_TRADES", 1)
    grid = (0.9, 1.0, 1.1)
    chunk, events = real_chunk(), real_events()
    sims = sv.simulate(chunk, events, pool, draws=DRAW)
    out = tuning.tune([(chunk, events, sims)], "M", grid=grid)

    # Enumerate the taken rows at c = 1.0 for every structure, straight from the real score_events and decide.
    pos = {str(e): i for i, e in enumerate(chunk["event_id"])}
    pred = dict(zip(events["event_id"].astype(str), events["pred_move"]))
    expected_trades, expected_total = 0, 0.0
    for name in sv.NAMES:
        decisions = sv.decide(sv.score_events(chunk, events, sims, name, alpha=0.5, c=1.0))
        for row in decisions[decisions["status"] == "taken"].to_dict("records"):
            eid = str(row["event_id"])
            legs, reason = sv.resolve(chunk, pos[eid], name, pred[eid], 1.0)
            assert legs is not None, reason
            # Hand P&L: quotes read straight from the chunk. Locked quotes make the fill alpha irrelevant.
            # A BUY leg closes by selling at the exit bid, a SELL leg by buying at the exit ask.
            cost, value = 0.0, 0.0
            for leg in legs:
                if leg["qty"] <= 0:
                    continue
                k = leg["idx"]
                sign = 1.0 if leg["side"] == BUY else -1.0
                cost += sign * leg["qty"] * float(chunk["bid_e"][k])
                close = chunk["bid_x"][k] if sign > 0 else chunk["ask_x"][k]
                value += sign * leg["qty"] * float(close)
            assert cost == pytest.approx(row["entry_cost"], abs=1e-9)
            expected_total += (value - cost) * CONTRACT_MULTIPLIER * lots(cost)
            expected_trades += 1

    assert expected_trades > 0  # a vacuous fixture would prove nothing
    assert out["trades"] == expected_trades
    assert out["totals"]["1.0"] == pytest.approx(expected_total, rel=1e-9, abs=1e-9)
    assert out["skipped"] is False and out["skip_reason"] is None
    assert out["c"] == max(grid, key=lambda c: out["totals"][str(c)])


# ---- receipts ----------------------------------------------------------------

def test_receipt_hash_refuses_a_body_carrying_content_hash():
    with pytest.raises(ValueError):
        tuning.receipt_hash({"content_hash": "x"})


def test_receipt_hash_is_sha256_of_the_canonical_json():
    # The canonical text of {"a": 1} is the bytes '{"a":1}'; the digest is taken over those bytes directly.
    assert tuning.receipt_hash({"a": 1}) == hashlib.sha256(b'{"a":1}').hexdigest()


def test_write_refuses_a_non_dict_body_and_creates_no_file(tmp_path):
    path = tmp_path / "r.json"
    with pytest.raises(ValueError):
        tuning.write_receipt(path, [])
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_nested_int_keys_are_admitted_and_round_trip(tmp_path):
    # JSON stringifies the keys, so 2 and 10 sort as "10" < "2" unless the body is admitted first.
    path = tmp_path / "r.json"
    body = {"t": {2: 1, 10: 2}}
    written = tuning.write_receipt(path, body)
    assert written["t"] == {"2": 1, "10": 2}
    assert {k: v for k, v in written.items() if k != "content_hash"} == {"t": {"2": 1, "10": 2}}
    assert tuning.load_receipt(path) == written


def test_nan_body_is_refused_as_value_error(tmp_path):
    path = tmp_path / "r.json"
    with pytest.raises(ValueError):
        tuning.write_receipt(path, {"c": math.nan})
    assert not path.exists()


def test_write_then_load_round_trip(tmp_path):
    path = tmp_path / "sub" / "receipt.json"
    written = tuning.write_receipt(path, dict(BODY))
    assert written["content_hash"] == tuning.receipt_hash(BODY)
    assert {k: v for k, v in written.items() if k != "content_hash"} == BODY
    assert tuning.load_receipt(path) == written


def test_replay_with_the_same_body_leaves_the_file_untouched(tmp_path):
    path = tmp_path / "r.json"
    first = tuning.write_receipt(path, dict(BODY))
    before = path.stat()
    again = tuning.write_receipt(path, dict(BODY))
    after = path.stat()
    assert again == first
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_changed_body_refuses_and_leaves_the_file_alone(tmp_path):
    path = tmp_path / "r.json"
    tuning.write_receipt(path, dict(BODY))
    original = path.read_bytes()
    with pytest.raises(sv.Refusal) as caught:
        tuning.write_receipt(path, {**BODY, "c": 0.9})
    assert caught.value.code == CONFLICT
    assert caught.value.details == {"reason": "tuning_receipt_changed", "path": str(path)}
    assert path.read_bytes() == original


def test_corrupt_file_refuses_and_is_never_repaired(tmp_path):
    path = tmp_path / "r.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(sv.Refusal) as caught:
        tuning.write_receipt(path, dict(BODY))
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == UNREADABLE
    assert path.read_text(encoding="utf-8") == "{not json"


def test_no_temp_files_are_left_behind(tmp_path):
    path = tmp_path / "r.json"
    tuning.write_receipt(path, dict(BODY))
    tuning.write_receipt(path, dict(BODY))
    with pytest.raises(sv.Refusal):
        tuning.write_receipt(path, {**BODY, "c": 0.9})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r.json"]


def test_lost_link_race_rereads_the_winner(tmp_path, monkeypatch):
    seed = tmp_path / "seed.json"
    winner = tuning.write_receipt(seed, dict(BODY))
    path = tmp_path / "r.json"

    def race(src, dst):  # a concurrent writer lands the same receipt just before our link
        Path(dst).write_text(seed.read_text(encoding="utf-8"), encoding="utf-8")
        raise FileExistsError(dst)

    monkeypatch.setattr(tuning.os, "link", race)
    assert tuning.write_receipt(path, dict(BODY)) == winner


def test_load_missing_receipt_refuses(tmp_path):
    with pytest.raises(sv.Refusal) as caught:
        tuning.load_receipt(tmp_path / "nope.json")
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "tuning_receipt_missing"


def test_load_tampered_receipt_refuses(tmp_path):
    path = tmp_path / "r.json"
    tuning.write_receipt(path, dict(BODY))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["c"] = 0.9  # body changed, content_hash left as written
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(sv.Refusal) as caught:
        tuning.load_receipt(path)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == UNREADABLE


def test_oversize_receipt_is_refused_at_write_and_unreadable_at_load(tmp_path):
    big = {**BODY, "pad": "x" * tuning.MAX_RECEIPT_BYTES}
    path = tmp_path / "big.json"
    # The writer refuses an oversize body before touching the filesystem: no target, no temp file.
    with pytest.raises(sv.Refusal) as caught:
        tuning.write_receipt(path, big)
    assert caught.value.code == CONFLICT
    assert caught.value.details == {"reason": "tuning_receipt_too_large", "path": str(path)}
    assert list(tmp_path.iterdir()) == []
    # A well-formed, correctly hashed oversize receipt written by hand: the size cap alone must refuse it on load.
    full = dict(big, content_hash=tuning.receipt_hash(big))
    path.write_bytes((json.dumps(full, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    assert path.stat().st_size > tuning.MAX_RECEIPT_BYTES
    with pytest.raises(sv.Refusal) as caught:
        tuning.load_receipt(path)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == UNREADABLE


def test_symlinked_receipt_refuses_as_unreadable(tmp_path):
    seed = tmp_path / "seed.json"
    tuning.write_receipt(seed, dict(BODY))
    link = tmp_path / "link.json"
    os.symlink(seed, link)
    with pytest.raises(sv.Refusal) as caught:
        tuning.load_receipt(link)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == UNREADABLE


def test_missing_hash_refuses_as_unreadable(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(dict(BODY)), encoding="utf-8")
    with pytest.raises(sv.Refusal) as caught:
        tuning.load_receipt(path)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == UNREADABLE


def test_nan_receipt_text_refuses_as_unreadable(tmp_path):
    path = tmp_path / "r.json"
    path.write_text('{"a": NaN, "content_hash": "x"}', encoding="utf-8")
    with pytest.raises(sv.Refusal) as caught:
        tuning.load_receipt(path)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == UNREADABLE


@pytest.mark.parametrize("text", ["[1, 2]", '"text"', "7"])
def test_non_dict_json_refuses_as_unreadable(tmp_path, text):
    path = tmp_path / "r.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(sv.Refusal) as caught:
        tuning.load_receipt(path)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == UNREADABLE
