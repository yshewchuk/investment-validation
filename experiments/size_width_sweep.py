"""EXP-189 W4: the shared sweep over the k grid for the seven short-vol structures (design sections 3, 5).

k scales the forecast the sizing rule reads: forecast_params at k*f equals size_width_sizing.k_params at k
(bounds apply after k), so EXP-188's size_pnl_shortvol scores, decides (per-structure trailing cutoff) and marks
unchanged. The simulation always uses M0's own forecast, once per event, shared by every structure and k.
Arm E picks, per event and structure, the grid k with the highest exp_pnl_sim among feasible widths
(ties: nearest 1, then smaller); no realised outcome is read. Progress: one stderr line per month.
"""
import resource
import sys
import time

import numpy as np
import pandas as pd

from engine.pnl_sim import DRAWS
from experiments import size_pnl_shortvol as sv
from experiments.size_width_sizing import GRID

ALPHA = 0.5
E_ARM = "E"
WINDOWS = {"2023H2": (pd.Timestamp("2023-07-01"), pd.Timestamp("2024-01-01")),
           "2024": (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
           "2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"))}
INNER = (pd.Timestamp("2023-07-01"), pd.Timestamp("2025-01-01"))
_TAKEN = ["event_id", "event_date", "structure", "k", "entry_date", "exit_date", "entry_cost", "exit_value",
          "exit_value_mid", "pnl_dollars", "pnl_dollars_mid", "secured_per_contract"]


def scaled(events, k):
    """``events`` with pred_move x k (k a number or a per-row array); NaN stays NaN."""
    return events.assign(pred_move=pd.to_numeric(events["pred_move"], errors="coerce") * k)


def in_window(frame, window):
    d = pd.to_datetime(frame["event_date"])
    return frame[(d >= window[0]) & (d < window[1])]


def pick_e(by_k, ks):
    """One row per event from ``by_k`` ({k: score_events frame}): the scored row with the highest exp_pnl_sim, ties nearest 1 then smaller k,
    plus a ``k`` column. An event with no feasible k keeps its row at the k nearest 1, so its reason is production's."""
    base = min(ks, key=lambda k: (abs(k - 1.0), k))
    stack = pd.concat([f.assign(k=k) for k, f in by_k.items()], ignore_index=True)
    value = pd.to_numeric(stack["exp_pnl_sim"], errors="coerce")
    ok = stack[(stack["status"] == "scored") & np.isfinite(value)].assign(_d=lambda f: (f["k"] - 1.0).abs())
    best = (ok.sort_values(["event_id", "exp_pnl_sim", "_d", "k"], ascending=[True, False, True, True], kind="mergesort")
            .drop_duplicates("event_id").drop(columns="_d"))
    rest = by_k[base][~by_k[base]["event_id"].isin(best["event_id"])].assign(k=base)
    parts = [p for p in (best, rest) if not p.empty]
    if not parts:
        return by_k[base].assign(k=base)
    return pd.concat(parts, ignore_index=True).sort_values("event_id", kind="mergesort").reset_index(drop=True)


class Sweep:
    """Scored history (it feeds each trailing cutoff), taken rows and decline reasons per (structure, arm); arm is a grid k or E_ARM."""
    def __init__(self, ks=GRID, names=sv.NAMES):
        self.ks, self.names = tuple(ks), tuple(names)
        self.scored, self.taken, self.declined = {}, {}, {}

    def add_month(self, chunk, events, pool, *, output, sessions, draws=DRAWS):
        """Simulate once, score every (structure, k) and E, decide on the accumulated history. ``output`` False (burn-in) only adds history.
        ``sessions`` maps event_id -> (entry_date, exit_date); a missing event leaves NaT dates and the account refuses it."""
        started = time.time()
        sims = sv.simulate(chunk, events, pool, draws=draws)
        keep = np.array([sims.get(str(e)) is not None for e in events["event_id"]], dtype=bool)  # unsimulable events are dropped, as EXP-188 does
        events = events[keep]
        ids = set(events["event_id"].astype(str))
        for name in self.names:
            by_k = {k: sv.score_events(chunk, scaled(events, k), sims, name, alpha=ALPHA) for k in self.ks}
            for arm, frame in [*by_k.items(), (E_ARM, pick_e(by_k, self.ks))]:
                self._decide(chunk, events, frame, name, arm, ids, output, sessions)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"[width_sweep] month events={len(events)} output={output} elapsed={time.time() - started:.1f}s rss={rss:.0f}MiB",
              file=sys.stderr)

    def _decide(self, chunk, events, frame, name, arm, ids, output, sessions):
        """Adds ``frame`` to the history of (name, arm) and, when ``output``, decides, prices and records the taken rows."""
        key = (name, arm)
        history = self.scored.setdefault(key, [])
        history.append(frame)
        if not output:
            return
        reasons = self.declined.setdefault(key, {})
        for reason, count in frame.loc[frame["status"] != "scored", "reason"].fillna("none").value_counts().items():
            reasons[str(reason)] = reasons.get(str(reason), 0) + int(count)
        decided = sv.decide(pd.concat(history, ignore_index=True))
        mine = decided[decided["event_id"].isin(ids) & (decided["structure"] == name) & (decided["status"] == "taken")]
        if arm == E_ARM:
            kmap = dict(zip(frame["event_id"], frame["k"]))
            ks = events["event_id"].astype(str).map(kmap).fillna(1.0).to_numpy(dtype=float)
        else:
            ks = float(arm)
        ev = scaled(events, ks)
        primary = sv.score(chunk, ev, mine, name, ALPHA)
        mid = sv.score(chunk, ev, mine, name, ALPHA, mid_marks=True)
        sec = sv.secured(chunk, ev, mine, name)
        rows = (primary[["event_id", "structure", "entry_cost", "exit_value", "pnl_dollars"]]
                .merge(mid[["event_id", "exit_value", "pnl_dollars"]].rename(columns={"exit_value": "exit_value_mid",
                                                                                     "pnl_dollars": "pnl_dollars_mid"}), on="event_id")
                .merge(sec, on="event_id")
                .merge(mine[["event_id", "event_date"]], on="event_id"))
        rows["k"] = rows["event_id"].map(kmap) if arm == E_ARM else float(arm)
        rows["entry_date"] = rows["event_id"].map(lambda e: sessions.get(e, (np.nan, np.nan))[0])
        rows["exit_date"] = rows["event_id"].map(lambda e: sessions.get(e, (np.nan, np.nan))[1])
        self.taken.setdefault(key, []).append(rows[_TAKEN])

    def rows(self, pairs, window):
        """Taken rows of every (structure, arm) in ``pairs`` inside ``window`` (by event date), concatenated; empty frame with the columns when none."""
        frames = [in_window(f, window) for pair in pairs for f in self.taken.get(tuple(pair), [])]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_TAKEN)

    def scored_rows(self, name, arm, window):
        """Scored (status "scored") rows of (name, arm) inside ``window``, the candidates E or a fixed k put to the gate."""
        frames = [in_window(f[f["status"] == "scored"], window) for f in self.scored.get((name, arm), [])]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=sv.SCORE_COLS + (["k"] if arm == E_ARM else []))

    def curves(self, name, window):
        """{str(arm): mean exp_pnl_sim of the scored rows in ``window``, or None} for every k and E: the T3 expected-P&L curve."""
        out = {}
        for arm in [*self.ks, E_ARM]:
            rows = self.scored_rows(name, arm, window)
            out[str(arm)] = float(pd.to_numeric(rows["exp_pnl_sim"]).mean()) if not rows.empty else None
        return out


def feed(sweep, months, *, output_from, until=None):
    """Run ``months`` (dicts with period, chunk, events, pool, sessions; period a Timestamp, month start) through ``sweep`` in order, stopping before ``until``."""
    for m in months:
        if until is not None and m["period"] >= until:
            break
        sweep.add_month(m["chunk"], m["events"], m["pool"], output=m["period"] >= output_from, sessions=m["sessions"],
                        **({"draws": m["draws"]} if "draws" in m else {}))
