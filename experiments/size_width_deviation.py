"""EXP-189 W5: realised-width deviation of today's trades and the decline table (T8, design 12.2); descriptive only."""
import numpy as np
import pandas as pd

from engine.forecast_sizing import forecast_params
from engine.structures import STRUCTURES
from experiments import size_pnl_shortvol as sv
from experiments.size_width_sweep import Sweep, in_window

TIERS = ("ge_10B", "1B_10B", "lt_1B", "unknown")
QUANTILES = (10, 25, 50, 75, 90)
BUCKETS = ("lt_0.9", "0.9_1.1", "gt_1.1")


def tier(mcap):
    """Market-cap tier in USD, lower bounds inclusive; NaN or None is "unknown"."""
    if mcap is None or pd.isna(mcap):
        return "unknown"
    return "ge_10B" if mcap >= 10e9 else "1B_10B" if mcap >= 1e9 else "lt_1B"


def bucket(ratio):
    """Bucket of a finite ratio; 0.9 and 1.1 both fall in the middle one."""
    return "lt_0.9" if ratio < 0.9 else "0.9_1.1" if ratio <= 1.1 else "gt_1.1"


def realised_ratio(chunk, i, name, pred):
    """|strike(b) - strike(a)| / (spot * width) over the width legs (a, b) of event ``i``; None when it does not resolve."""
    legs, _ = sv.resolve(chunk, i, name, pred)
    params = None if legs is None else forecast_params(name, float(pred))
    a, b = STRUCTURES[name]().params["width_legs"]
    strike = {leg["name"]: leg["strike"] for leg in legs or []}
    if params is None or a not in strike or b not in strike:
        return None
    return abs(strike[b] - strike[a]) / (float(chunk["spot_entry"][i]) * params["width_moneyness"])


class DeviationSweep(Sweep):
    """A drop-in Sweep that also records T8's raw rows: one per production (arm 1.0) taken trade."""
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.deviation, self.unavailable = [], 0

    def _decide(self, chunk, events, frame, name, arm, ids, output, sessions):
        super()._decide(chunk, events, frame, name, arm, ids, output, sessions)
        if not output or arm != 1.0:
            return
        pos = {str(e): i for i, e in enumerate(chunk["event_id"])}
        info = {str(r["event_id"]): r for r in events.to_dict("records")}
        for row in self.taken[(name, arm)][-1].to_dict("records"):
            eid, ev = str(row["event_id"]), info[str(row["event_id"])]
            ratio = realised_ratio(chunk, pos[eid], name, ev["pred_move"])
            self.unavailable += ratio is None
            self.deviation.append({"structure": name, "event_id": eid, "event_date": row["event_date"],
                                   "mcap_usd": sv._num(ev["mcap_usd"]), "ratio": float("nan") if ratio is None else ratio,
                                   "ret": row["exit_value_mid"] / row["entry_cost"] - 1.0})


def _cells(sweep, windows, column, values):
    """(window, structure, value, cell) for every non-empty cell of the finite-ratio rows; windows are half-open on event date."""
    frame = pd.DataFrame(sweep.deviation, columns=["structure", "event_id", "event_date", "mcap_usd", "ratio", "ret"])
    frame["ratio"] = pd.to_numeric(frame["ratio"], errors="coerce").astype(float)
    frame = frame[np.isfinite(frame["ratio"])].assign(tier=lambda f: f["mcap_usd"].map(tier),
                                                      bucket=lambda f: f["ratio"].map(bucket))
    for label, window in windows.items():
        inside = in_window(frame, window)
        for name in sv.NAMES:
            for v in values:
                cell = inside[(inside["structure"] == name) & (inside[column] == v)]
                if not cell.empty:
                    yield label, name, v, cell


def quantile_table(sweep, windows):
    """Numpy percentiles (linear interpolation) of the realised/target ratio per (window, structure, tier)."""
    return [{"window": w, "structure": s, "tier": t, "n": len(c),
             **{f"q{p}": float(v) for p, v in zip(QUANTILES, np.percentile(c["ratio"], QUANTILES))}}
            for w, s, t, c in _cells(sweep, windows, "tier", TIERS)]


def bucket_table(sweep, windows):
    """Mean realised return per (window, structure, bucket). Descriptive only: snap deviation correlates with price level
    and ladder spacing, so these cells test nothing on their own."""
    return [{"window": w, "structure": s, "bucket": b, "n": len(c), "mean_ret": float(c["ret"].astype(float).mean())}
            for w, s, b, c in _cells(sweep, windows, "bucket", BUCKETS)]


def decline_table(sweep, windows):
    """Share of each outcome ("taken" for a taken row) per (window, structure, tier) among the cell's candidates: every scored
    production row, decided on its own structure's history; "all" pools the seven."""
    decided = {n: sv.decide(pd.concat(f, ignore_index=True)) for n in sv.NAMES if (f := sweep.scored.get((n, 1.0)))}
    decided = {**decided, "all": pd.concat(list(decided.values()), ignore_index=True)} if decided else decided
    rows = []
    for label, window in windows.items():
        for name, frame in decided.items():
            cells = in_window(frame, window).assign(tier=lambda f: f["mcap_usd"].map(tier),
                                                    reason=lambda f: f["reason"].fillna("taken"))
            totals = cells.groupby("tier").size()
            for (t, reason), n in cells.groupby(["tier", "reason"]).size().items():
                rows.append({"window": label, "structure": name, "tier": t, "reason": reason, "n": int(n),
                             "share": float(n / totals[t])})
    return rows
