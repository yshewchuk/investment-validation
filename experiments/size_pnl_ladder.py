"""EXP-188 S3: ladder-quote table for admitted events, built one quarter at a time.

Per event: put and call quotes at the first post-event expiry on the entry date, and the same
contracts' quotes on the exit date (matched by right and strike). Entry quotes that fail validity (NaN,
crossed, negative bid, ask<=0) are set to NaN, never dropped, so consumers see the true strike ladder.
Each exit quote gets a mark rung
(1 usable, 2 zero-bid or wide, 3 unusable) via ``FillModel.is_wide``; no marks are computed here
(S5). Chains arrive only through the injected ``load_index``, once per quarter.

Skips depend only on entry-time facts (no entry chain, no session, selector failure, no put or no call
row in the entry chain, NaN entry dte). Future exit data never drops an event: a missing exit chain, no
rows at the expiry on the exit date, or a NaN exit dte keeps the event with every exit quote NaN, rung 3,
``dte_exit = -1`` (sentinel for "unknown") and NaN ``spot_exit`` where no spot is available. These
events are counted in the chunk's ``exit_missing`` dict so a straddle decision never depends on exit
coverage (the design's worst-case rung 3). Chunk fields: ``offsets``, ``skipped``, ``exit_missing``,
per-event scalars, and per-contract arrays.
"""
import sys
import time

import numpy as np
import pandas as pd

from engine.fills import FillModel
from engine.v2.ops.errors import fail
from engine.v2.research._pricing import ExpirySelector, StructureError
from experiments.size_pnl_readers import Admitted

_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
_DENIED = "HOLDOUT_ACCESS_DENIED"
_REASONS = ("no_entry_chain", "no_session", "structure_unresolved")
_EXIT_MISSING = ("no_exit_chain", "expiry_gone_at_exit")
_PLAN_COLS = ("event_id", "ticker", "event_date", "session", "entry_date", "exit_date")
_DATE_COLS = ("event_date", "entry_date", "exit_date")
_SESSIONS = ("AMC", "BMO")
_CODE = {"P": 0, "C": 1}
SELECTOR = ExpirySelector(kind="first_post_event")
_SCALARS = {"event_id": object, "expiry": "datetime64[ns]", "spot_entry": np.float64,
            "spot_exit": np.float64, "dte_entry": np.int32, "dte_exit": np.int32}
_FLATS = {"right": np.int8, "strike": np.float64, "bid_e": np.float64, "ask_e": np.float64,
          "bid_x": np.float64, "ask_x": np.float64, "exit_rung": np.int8}


def _check_plan(admitted, plan):
    """Refuse an unissued admission, malformed plan fields, plan ids outside it, repeated ids, and
    mismatched or inconsistent dates."""
    if not isinstance(admitted, Admitted):
        raise fail(_DENIED, "the ladder accepts only an admission issued by admit_events",
                   details={"reason": "admission_not_issued"})
    bad = not isinstance(plan, pd.DataFrame) or not set(_PLAN_COLS) <= set(plan.columns)
    if not bad:
        try:
            bad = any(pd.to_datetime(plan[c]).isna().any() for c in _DATE_COLS)
        except (ValueError, TypeError):
            bad = True
    if bad:
        raise fail(_CONFLICT, "the plan is malformed", details={"reason": "plan_malformed"})
    if any(not isinstance(t, str) or not t.strip() for t in plan["ticker"]):
        raise fail(_CONFLICT, "the plan has an empty or null ticker", details={"reason": "plan_malformed"})
    if any(not pd.isna(s) and (not isinstance(s, str) or s.upper() not in _SESSIONS)
           for s in plan["session"]):
        raise fail(_CONFLICT, "the plan has a session outside AMC/BMO", details={"reason": "plan_session_invalid"})
    entry, event, exit_ = (pd.to_datetime(plan[c]).dt.normalize() for c in ("entry_date", "event_date", "exit_date"))
    if not ((entry <= event) & (event <= exit_) & (entry < exit_)).all():
        raise fail(_CONFLICT, "the plan dates are not entry <= event <= exit with entry < exit",
                   details={"reason": "plan_dates_inconsistent"})
    dates = admitted["events"].set_index("event_id")["event_date"]
    ids = plan["event_id"].astype(str)
    if not set(ids) <= set(dates.index):
        raise fail(_DENIED, "plan event ids are not admitted", details={"reason": "key_not_admitted"})
    if ids.duplicated().any():
        raise fail(_CONFLICT, "the plan repeats an event id", details={"reason": "duplicate_event_id"})
    if (pd.to_datetime(plan["event_date"]).to_numpy() != dates.reindex(ids).to_numpy()).any():
        raise fail(_CONFLICT, "a plan event_date differs from the admitted date",
                   details={"reason": "event_date_mismatch"})


check_plan = _check_plan


def _first(values):
    """First non-null numeric value, or NaN."""
    got = pd.to_numeric(values, errors="coerce").dropna()
    return float(got.iloc[0]) if len(got) else float("nan")


def _contracts(rows, *, entry):
    """Puts/calls with a usable strike, sorted. Among rows with the same (right, strike), a valid/usable
    quote wins over an invalid one; otherwise the first row. Entry rows keep every such strike but set
    bid/ask to NaN where the quote is NaN, crossed, negative-bid or ask<=0, so the true strike ladder
    survives; exit rows keep every quote so rung 3 can record them."""
    out = pd.DataFrame({
        "right": rows["right"].map(_CODE).astype(float).to_numpy(),
        "strike": pd.to_numeric(rows["strike"], errors="coerce").astype(float).to_numpy(),
        "bid": pd.to_numeric(rows["bid"], errors="coerce").astype(float).to_numpy(),
        "ask": pd.to_numeric(rows["ask"], errors="coerce").astype(float).to_numpy(),
    })
    out = out[out["right"].notna() & out["strike"].notna()]
    if entry:
        valid = out["bid"].notna() & out["ask"].notna() & (out["bid"] <= out["ask"]) & (out["ask"] > 0) & (out["bid"] >= 0)
        tradable = valid & (out["bid"] > 0) & (out["bid"] < out["ask"])
        rank = np.where(tradable, 0, np.where(valid, 1, 2))  # 0 tradable, 1 zero-bid/locked, 2 invalid
        out = out.assign(bid=out["bid"].where(valid), ask=out["ask"].where(valid), _rank=rank)
        out = out.sort_values(["right", "strike", "_rank"], kind="mergesort").drop(columns="_rank")
    else:
        usable = np.isfinite(out["bid"]) & np.isfinite(out["ask"]) & (out["bid"] >= 0) & (out["ask"] >= out["bid"])
        out = out.assign(_rank=(~usable).astype(int))
        out = out.sort_values(["right", "strike", "_rank"], kind="mergesort").drop(columns="_rank")
    return out.drop_duplicates(["right", "strike"], keep="first").sort_values(["right", "strike"])


def _exit_rungs(bid, ask):
    """1 usable; 2 zero bid or wide; 3 NaN, crossed or negative."""
    rung = np.full(bid.shape, 3, dtype=np.int8)
    usable = np.isfinite(bid) & np.isfinite(ask) & (bid >= 0) & (ask >= bid)
    if usable.any():
        b, a = bid[usable], ask[usable]
        rung[usable] = np.where((b == 0) | FillModel(0.5).is_wide(b, a), 2, 1)
    return rung


def _event(row, index, selector):
    """(skip, gap, record). ``skip`` is a reason for a dropped event; otherwise ``gap`` is None or the
    ``_EXIT_MISSING`` key for an event kept with no exit quotes, and ``record`` holds the contract quotes
    and scalars. Skips use entry-time facts only: no entry chain, no session, selector failure, an entry
    chain with no put row or no call row (any quote validity), or NaN entry dte. Whether a short-vol
    structure has enough valid puts is not checked here; S5 applies that depth rule."""
    entry = index.get(row["ticker"], row["entry_date"])
    if entry is None or entry.empty:
        return "no_entry_chain", None, None
    if pd.isna(row["session"]):
        return "no_session", None, None
    try:
        expiry = pd.Timestamp(selector.select(entry, row["event_date"], row["session"]))
    except StructureError:
        return "structure_unresolved", None, None
    ent_rows = entry[entry["expiry"] == expiry]
    ent = _contracts(ent_rows, entry=True).rename(columns={"bid": "bid_e", "ask": "ask_e"})
    if not ((ent["right"] == 0).any() and (ent["right"] == 1).any()):
        return "structure_unresolved", None, None
    dte_e = _first(ent_rows["dte"])
    if np.isnan(dte_e):
        return "structure_unresolved", None, None
    # Exit data never skips: a missing chain, expiry or dte is recorded as a gap and the event is kept.
    exit_ = index.get(row["ticker"], row["exit_date"])
    if exit_ is None or exit_.empty:
        gap, ex_rows, spot_x, dte_x = "no_exit_chain", None, float("nan"), float("nan")
    else:
        ex_rows = exit_[exit_["expiry"] == expiry]
        spot_x, dte_x = _first(exit_["spot"]), _first(ex_rows["dte"])  # NaN dte if no rows at expiry
        gap = "expiry_gone_at_exit" if np.isnan(dte_x) else None
    if gap is None:
        ex = _contracts(ex_rows, entry=False).rename(columns={"bid": "bid_x", "ask": "ask_x"})
        ent = ent.merge(ex[["right", "strike", "bid_x", "ask_x"]], on=["right", "strike"], how="left")
        bid_x = np.array(ent["bid_x"], dtype=float)
        ask_x = np.array(ent["ask_x"], dtype=float)
    else:
        bid_x = np.full(len(ent), np.nan)
        ask_x = np.full(len(ent), np.nan)
    rung = _exit_rungs(bid_x, ask_x)
    bid_x[rung == 3] = np.nan
    ask_x[rung == 3] = np.nan
    record = {"event_id": str(row["event_id"]), "expiry": expiry.to_datetime64(),
              "spot_entry": _first(entry["spot"]), "spot_exit": spot_x,
              "dte_entry": int(dte_e), "dte_exit": -1 if gap else int(dte_x),
              "right": ent["right"].to_numpy(float), "strike": ent["strike"].to_numpy(float),
              "bid_e": ent["bid_e"].to_numpy(float), "ask_e": ent["ask_e"].to_numpy(float),
              "bid_x": bid_x, "ask_x": ask_x, "exit_rung": rung}
    return None, gap, record


def _pack(recs, skipped, exit_missing=None):
    """CSR chunk: per-event scalars, flat per-contract arrays, offsets into the flat arrays."""
    offsets = np.zeros(len(recs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(r["right"]) for r in recs], dtype=np.int64)
    gaps = dict.fromkeys(_EXIT_MISSING, 0) if exit_missing is None else dict(exit_missing)
    chunk = {"offsets": offsets, "skipped": dict(skipped), "exit_missing": gaps}
    chunk.update({k: np.array([r[k] for r in recs], dtype=dt) for k, dt in _SCALARS.items()})
    chunk.update({k: np.concatenate([np.zeros(0, dt)] + [np.asarray(r[k], dt) for r in recs])
                  for k, dt in _FLATS.items()})
    return chunk


def _rss_mib():
    """Peak RSS in MiB (Linux ru_maxrss is KiB); 0 where ``resource`` is unavailable."""
    try:
        import resource
    except ImportError:
        return 0.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _progress(label, number, quarters, events, chunk, done, total, started):
    rung = chunk["exit_rung"]
    shares = " ".join(f"rung{r}={100.0 * np.count_nonzero(rung == r) / max(rung.size, 1):.1f}%"
                      for r in (1, 2, 3))
    rate = done / max(time.monotonic() - started, 1e-9)
    eta = (total - done) / rate if rate > 0 else 0.0
    print(f"[pnl_ladder] {label} {number}/{quarters} events={events} kept={len(chunk['event_id'])} "
          f"skipped={chunk['skipped']} exit_missing={chunk['exit_missing']} "
          f"rate={rate:.1f}/s eta={eta:.0f}s rss={_rss_mib():.0f}MiB {shares}",
          file=sys.stderr)


def _quarters(plan, load_index, selector):
    labels = sorted(plan["quarter"].unique())
    total, done, started = len(plan), 0, time.monotonic()
    print(f"[pnl_ladder] start events={total} quarters={len(labels)}", file=sys.stderr)
    index = None
    for number, label in enumerate(labels, start=1):
        index = None  # release the previous quarter's index before loading the next
        part = plan[plan["quarter"] == label].sort_values("event_id", kind="mergesort")
        keys = {(str(t), pd.Timestamp(d).normalize()) for t, d in zip(part["ticker"], part["entry_date"])}
        keys |= {(str(t), pd.Timestamp(d).normalize()) for t, d in zip(part["ticker"], part["exit_date"])}
        index = load_index(keys)
        reasons = dict.fromkeys(_REASONS, 0)
        gaps = dict.fromkeys(_EXIT_MISSING, 0)
        recs = []
        for row in part.to_dict("records"):
            reason, gap, rec = _event(row, index, selector)
            if reason is not None:
                reasons[reason] += 1
                continue
            recs.append(rec)
            if gap is not None:
                gaps[gap] += 1
        chunk = _pack(recs, reasons, gaps)
        index = None  # chunk holds only numpy copies; drop the index before suspending at yield
        done += len(part)
        _progress(label, number, len(labels), len(part), chunk, done, total, started)
        yield label, chunk
    print(f"[pnl_ladder] end events={total} elapsed={time.monotonic() - started:.1f}s", file=sys.stderr)


def ladder_chunks(admitted, plan, load_index, *, expiry_selector=None):
    """Generator: yields (quarter_label, chunk) one quarter at a time, ascending."""
    _check_plan(admitted, plan)
    frame = plan.assign(event_id=plan["event_id"].astype(str),
                        event_date=pd.to_datetime(plan["event_date"]),
                        entry_date=pd.to_datetime(plan["entry_date"]),
                        exit_date=pd.to_datetime(plan["exit_date"]))
    frame = frame.assign(quarter=frame["event_date"].dt.to_period("Q").astype(str))
    yield from _quarters(frame, load_index, expiry_selector or SELECTOR)


def default_load_index(repository, snapshot):
    """Real loader: ``keys`` is a set of (ticker, Timestamp); reads only those chains."""
    from engine.v2.research._chains import load_chain_index
    return lambda keys: load_chain_index(repository, snapshot, keys)
