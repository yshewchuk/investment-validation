"""EXP-189 report diagnostics D1 and D2 (reported, never deciding): win rates and Spearman/quintile checks of predicted vs realised return, from the stored taken rows of an F or model A run. Refusals come before anything is written (README)."""
import hashlib
import io
import os
import stat
import sys
import time

import numpy as np
import pandas as pd

from experiments import size_pnl_decisions as decisions
from experiments import size_pnl_receipts as receipts
from experiments import size_width_sweep as sweep
from experiments.size_pnl_ladder import _rss_mib
from experiments.size_pnl_report import md_table
from experiments.size_pnl_shortvol import NAMES, Refusal

CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
PERIODS, LABELS, FIXED = ("2025", "2023H2", "2024"), ("production", "A", "B", "D", "E"), ("production", "A", "B", "D")
REQUIRED = ["event_id", "event_date", "exit_date", "structure", "arm", "exp_pnl_sim", "entry_cost", "entry_cost_mid", "exit_value",
            "exit_value_mid", "pnl_dollars", "pnl_dollars_mid"]
PAIRS = (("mid vs mid", "pred_mid", "real_mid"), ("mid vs F", "pred_mid", "real_fill"), ("F vs F", "pred_fill", "real_fill"))
D1_HEAD = "### D1 beside T5: win rate by arm, period and structure (pnl > 0; reported, never deciding; 2025 NOT blind for widths)"
D2_HEAD = "### D2 beside T5: prediction rank check at fixed width, Spearman of predicted vs realised return (reported, never deciding; 2025 NOT blind for widths)"
Q_HEAD = "### D2 beside T5: predicted quintiles, mean predicted and mean realised return % (reported, never deciding; 2025 NOT blind for widths)"


def _num(series):
    return pd.to_numeric(series, errors="coerce").astype(float)


def _stage(out, stage, inner):
    """The taken part frames of one stage: each named by its receipt, its bytes checked by SHA-256 before they are parsed."""
    receipt = receipts.load_receipt(out / "receipts" / f"report_inputs_{stage}.json")
    if (receipt.get("stage"), receipt.get("inputs_hash"), receipt.get("scope")) != (stage, inner["inputs_hash"], inner["scope"]):
        raise Refusal(CONFLICT, f"report inputs receipt {stage} is bound to another stage, inputs or scope", {"reason": "tuning_receipt_changed"})
    cap, frames = decisions.MAX_DECISIONS_BYTES, []
    for name in sorted(n for n in receipt["tables"] if n.startswith("taken_")):
        try:  # one no-follow, non-blocking read of a regular file of at most cap bytes, at most cap + 1 read
            with os.fdopen(os.open(out / "report_inputs" / stage / f"{name}.parquet", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as h:
                st = os.fstat(h.fileno())
                data = h.read(cap + 1) if stat.S_ISREG(st.st_mode) and st.st_size <= cap else None
        except OSError:
            data = None
        if data is None or len(data) > cap or hashlib.sha256(data).hexdigest() != receipt["tables"][name]["sha256"]:
            raise Refusal(CONFLICT, f"report input {stage}/{name} is missing, not a regular file, oversized or changed", {"reason": "report_input_changed"})
        try:
            frame = pd.read_parquet(io.BytesIO(data))
        except (OSError, ValueError):
            raise Refusal(CONFLICT, f"report input {stage}/{name} is not a readable table", {"reason": "report_input_changed"}) from None
        missing = [c for c in REQUIRED if c not in frame.columns]
        if missing:
            raise Refusal(CONFLICT, f"report input {stage}/{name} lacks required columns {missing}", {"reason": "diag_columns_missing", "columns": missing})
        frames.append(frame)
    return frames


def _taken(out, inner):
    """(taken rows of both stages, part count); refuses a non-finite exp_pnl_sim or a repeated (arm, event_id, structure)."""
    frames = [f for stage in ("inner", "outer") for f in _stage(out, stage, inner)]
    taken = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=REQUIRED)
    if not np.isfinite(_num(taken["exp_pnl_sim"])).all() or taken.duplicated(["arm", "event_id", "structure"]).any():
        raise Refusal(CONFLICT, "taken rows have a non-finite exp_pnl_sim or a repeated (arm, event_id, structure)", {"reason": "diag_rows_invalid"})
    return taken, len(frames)


def _window(taken, factors, period):
    """The label's rows for ``period``, T5's way: arm is the factor for the row's structure (fallback rows never match), event date
    inside the window and exit before its end."""
    rows = taken[taken["arm"] == taken["structure"].map({n: str(factors[n]) for n in NAMES})]
    window = sweep.WINDOWS[period]
    got = sweep.in_window(rows, window)
    return got[pd.to_datetime(got["exit_date"]) < window[1]]


def _rho(x, y):
    """Spearman (pearson of the average ranks); None with fewer than 2 rows or a constant side."""
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2: return None
    return float(x.rank().corr(y.rank()))


def _within(p, x, y):
    """Spearman in each (structure, year) cell, averaged with weight the cell's row count over the cells where it is defined."""
    got = [(r, len(c)) for _, c in p.groupby(["structure", "year"]) for r in [_rho(c[x], c[y])] if r is not None]
    return sum(r * n for r, n in got) / sum(n for _, n in got) if got else None


def _pairs(rows):
    """Per row: structure, year, and predicted and realised returns at fill (F) and at mid; NaN where undefined."""
    cost, mid = _num(rows["entry_cost"]).to_numpy(), _num(rows["entry_cost_mid"]).to_numpy()
    exp, ex, exm = _num(rows["exp_pnl_sim"]).to_numpy(), _num(rows["exit_value"]).to_numpy(), _num(rows["exit_value_mid"]).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.DataFrame({"structure": rows["structure"].to_numpy(), "year": pd.to_datetime(rows["event_date"]).dt.year.to_numpy(),
                             "pred_fill": exp, "pred_mid": np.where(mid > 0, (exp + 1) * cost / mid - 1, np.nan),
                             "real_fill": np.where(cost > 0, ex / cost - 1, np.nan), "real_mid": np.where(mid > 0, exm / mid - 1, np.nan)})


def section(out, inner, labels_factors):
    """Markdown for D1 (win rates), D2 (Spearman by pair) and the D2 quintile table, from the stored taken rows of ``out``."""
    started = time.monotonic()
    taken, parts = _taken(out, inner)
    d1, d2, quint = [], [], []
    for label in [x for x in LABELS if x in labels_factors]:
        for p in PERIODS:
            got = _window(taken, labels_factors[label], p)
            for name, g in [("all", got), *((n, got[got["structure"] == n]) for n in NAMES)]:
                pct = [("n/a" if len(g) == 0 else f"{100 * float((_num(g[c]) > 0).mean()):.2f}") for c in ("pnl_dollars", "pnl_dollars_mid")]
                d1.append([label, p, name, len(g), *pct])
    for label in [x for x in FIXED if x in labels_factors]:
        pairs = _pairs(pd.concat([_window(taken, labels_factors[label], p) for p in PERIODS], ignore_index=True))
        for name, x, y in PAIRS:
            p = pairs[np.isfinite(pairs[x]) & np.isfinite(pairs[y])]
            yearly = [("n/a" if (r := _rho(p.loc[p["year"] == yr, x], p.loc[p["year"] == yr, y])) is None else f"{r:.3f}") for yr in (2023, 2024, 2025)]
            whole, within = _rho(p[x], p[y]), _within(p, x, y)
            d2.append([label, name, len(p), *[("n/a" if v is None else f"{v:.3f}") for v in (whole, within)], *yearly])
            if len(p) >= 5:
                code = pd.qcut(p[x], 5, labels=False, duplicates="drop").to_numpy()
                for c in np.unique(code):
                    cell = p[code == c]
                    quint.append([label, name, int(c) + 1, len(cell), f"{100 * cell[x].mean():.2f}", f"{100 * cell[y].mean():.2f}"])
    print(f"[width_diag] parts={parts} rows={len(taken)} elapsed={time.monotonic() - started:.1f}s rss={_rss_mib():.0f}MiB", file=sys.stderr)
    return "\n\n".join([
        D1_HEAD + "\n\n" + md_table(["arm", "period", "structure", "trades", "win % at F fills", "win % at mid"], d1),
        D2_HEAD + "\n\n" + md_table(["arm", "predicted vs realised", "trades", "pooled", "within structure x year (event-weighted)",
                                     "2023", "2024", "2025"], d2),
        Q_HEAD + "\n\n" + md_table(["arm", "predicted vs realised", "quintile", "trades", "mean predicted %", "mean realised %"], quint)])
