"""EXP-189 W4: shared k-grid sweep for the seven short-vol structures (design sections 3, 5); arm E = best exp_pnl_sim per event."""
import resource
import sys
import time

import numpy as np
import pandas as pd

from engine.pnl_sim import DRAWS
from experiments import size_pnl_shortvol as sv
from experiments.size_width_sizing import GRID

ALPHA, E_ARM = 0.5, "E"
WINDOWS = {"2023H2": (pd.Timestamp("2023-07-01"), pd.Timestamp("2024-01-01")),
           "2024": (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
           "2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"))}
INNER = (pd.Timestamp("2023-07-01"), pd.Timestamp("2025-01-01"))
_TAKEN = ["event_id", "event_date", "structure", "k", "entry_date", "exit_date", "entry_cost", "exit_value",
          "exit_value_mid", "pnl_dollars", "pnl_dollars_mid", "secured_per_contract"]


def admit_k(k):
    """A grid k: a real number (not a bool or a string) on GRID; anything else is a ValueError."""
    if isinstance(k, bool) or not isinstance(k, (int, float)) or k not in GRID:
        raise ValueError(f"k must be one of the grid values, got {k!r}")
    return float(k)


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
        self.ks, self.names = tuple(admit_k(k) for k in ks), tuple(names)
        if not self.ks or len(set(self.ks)) != len(self.ks): raise ValueError(f"ks must be non-empty and unique: {self.ks}")
        if not self.names or len(set(self.names)) != len(self.names) or any(n not in sv.NAMES for n in self.names): raise ValueError(f"names must be non-empty, unique and in sv.NAMES: {self.names}")
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
        decided = sv.decide(pd.concat(history, ignore_index=True))
        declined = decided[decided["event_id"].isin(ids) & (decided["structure"] == name) & (decided["status"] != "taken")]
        reasons = self.declined.setdefault(key, {})
        for reason, count in declined["reason"].fillna("none").value_counts().items():
            reasons[str(reason)] = reasons.get(str(reason), 0) + int(count)
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
        """Taken rows of every (structure, arm) in ``pairs`` inside ``window``: trades are selected by event date and must also EXIT inside the window; a trade exiting later belongs to no account of this window, so no later P&L enters a window's objective or evaluation. Empty frame with the columns when none."""
        taken = [in_window(g, window) for pair in pairs for g in self.taken.get(tuple(pair), [])]
        for f in taken:
            if f["exit_date"].isna().any(): raise ValueError("taken rows must carry exit dates")
        frames = [f[pd.to_datetime(f["exit_date"]) < window[1]] for f in taken]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_TAKEN)

    def scored_rows(self, name, arm, window, scored_only=True):
        """Scored rows of (name, arm) inside ``window`` (every status when ``scored_only`` is False): the candidates E or a fixed k put to the gate."""
        frames = [f for f in (in_window(g[g["status"] == "scored"] if scored_only else g, window) for g in self.scored.get((name, arm), [])) if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=sv.SCORE_COLS + (["k"] if arm == E_ARM else []))

    def curves(self, name, window):
        """{str(arm): mean exp_pnl_sim of the scored rows in ``window``, or None} for every k and E: the T3 expected-P&L curve."""
        out = {}
        for arm in [*self.ks, E_ARM]:
            rows = self.scored_rows(name, arm, window)
            out[str(arm)] = float(pd.to_numeric(rows["exp_pnl_sim"]).mean()) if not rows.empty else None
        return out
