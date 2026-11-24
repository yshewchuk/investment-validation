"""EXP-189 W5: tolerance bands (design 12.1), the tuning_band receipt, the rho ladder and the pre-stated adoption rule (12.4)."""
import math
from pathlib import Path

import experiments.size_pnl_receipts as receipts
from experiments.size_pnl_shortvol import NAMES, Refusal

PHIS = (0.80, 0.90, 0.95)
RHO_STEP = 0.05
MIN_RECOVERED = 100
ALPHA_MARGIN = 0.05
WINDOW = "2023H2-2024"
_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
WINDOW_KEYS = ("2025", "2023H2", "2024")


def _walk(grid, i, step, thr):
    """(edge k, open) from index ``i`` stepping by ``step`` over passing points; a not_scored neighbour ends it, a failing one is interpolated."""
    while 0 <= i + step < len(grid) and not grid[i + step][1]["not_scored"] and grid[i + step][1]["ret"] >= thr:
        i += step
    j = i + step
    if not 0 <= j < len(grid):
        return grid[i][0], True
    if grid[j][1]["not_scored"]:
        return grid[i][0], False
    (k0, r0), (k1, r1) = (grid[i][0], grid[i][1]["ret"]), (grid[j][0], grid[j][1]["ret"])
    return k0 + (k1 - k0) * (r0 - thr) / (r0 - r1), False


def band(scores, phi):
    """The tolerance band of one tuning curve at ``phi``: the contiguous run around the peak with ret >= phi * peak ret; open ends are flagged."""
    grid = sorted(((float(k), s) for k, s in scores.items()), key=lambda p: p[0])
    scored = [i for i, (_k, s) in enumerate(grid) if not s["not_scored"]]
    peak = max(scored, key=lambda i: (grid[i][1]["ret"], -abs(grid[i][0] - 1.0), -grid[i][0]), default=None)
    if peak is None or grid[peak][1]["ret"] <= 0:
        return {"lo": None, "hi": None, "peak_k": None, "flags": ["no_positive_peak"]}
    thr = phi * grid[peak][1]["ret"]
    lo, lo_open = _walk(grid, peak, -1, thr)
    hi, hi_open = _walk(grid, peak, 1, thr)
    return {"lo": lo, "hi": hi, "peak_k": grid[peak][0], "flags": ["band_open"] if lo_open or hi_open else []}


def build_bands(curves):
    """Bands per structure. B and C multiply production's width (k is a ratio to production); the anchor's k multiplies the ACTUAL move, so its band is in actual-move ratios."""
    missing = [n for n in NAMES if n not in curves]
    if missing:
        raise ValueError(f"no tuning curves for {missing}")
    out = {}
    for name in NAMES:
        found = {label: curves[name].get(label) for label in ("B", "C", "anchor")}
        found = {label: s for label, s in found.items() if s is not None}
        if "B" not in found:
            raise ValueError(f"{name} has no B curve")
        out[name] = {"primary": "C" if "C" in found else "B",
                     "curves": {label: {str(p): band(s, p) for p in PHIS} for label, s in found.items()}}
    return out


def write_bands(out, bands, inputs_hash):
    """Publishes ``bands`` as the write-once tuning_band receipt under ``out``; returns the receipt."""
    if not _well_formed(bands):
        raise ValueError("tuning_band body is malformed")
    return receipts.write_receipt(Path(out) / "receipts" / "tuning_band.json",
                                  {"experiment": "EXP-189", "arm": "band", "inputs_hash": inputs_hash, "window": WINDOW, "body": bands})


def _well_formed(body):
    """True when ``body`` holds every structure with a B curve (plus optional C and anchor), a primary of C when present else B, and a valid cell per label and phi: no_positive_peak alone with None fields, else 0 < lo <= peak_k <= hi with flags [] or ["band_open"]."""
    def real(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    try:
        for n in NAMES:
            entry = body[n]
            curves = entry["curves"]
            if "B" not in curves or not set(curves) <= {"B", "C", "anchor"} or entry["primary"] != ("C" if "C" in curves else "B"):
                return False
            for curve in curves.values():
                for p in PHIS:
                    cell = curve[str(p)]
                    if not isinstance(cell, dict) or not isinstance(cell["flags"], list):
                        return False
                    if "no_positive_peak" in cell["flags"]:
                        if cell["flags"] != ["no_positive_peak"] or (cell["lo"], cell["hi"], cell["peak_k"]) != (None, None, None):
                            return False
                    elif not (real(cell["lo"]) and real(cell["hi"]) and real(cell["peak_k"])) \
                            or not 0 < cell["lo"] <= cell["peak_k"] <= cell["hi"] or cell["flags"] not in ([], ["band_open"]):
                        return False
    except (KeyError, TypeError, AttributeError, ValueError):
        return False
    return True


def frozen_bands(out, inputs_hash):
    """The frozen tuning_band body under ``out``; refuses when the receipt is missing, malformed, for other inputs, or changed."""
    data = receipts.load_receipt(Path(out) / "receipts" / "tuning_band.json")
    if ((data.get("experiment"), data.get("arm"), data.get("window")) != ("EXP-189", "band", WINDOW)
            or data.get("inputs_hash") != inputs_hash or not _well_formed(data.get("body"))):
        raise Refusal(_CONFLICT, "tuning band receipt does not match this run's inputs", {"reason": "tuning_receipt_changed"})
    return data["body"]


def band_range(body, structure, phi):
    """(lo, hi) of the primary curve of ``structure`` at ``phi``, or None when that curve has no positive peak."""
    entry = body[structure]
    cell = entry["curves"][entry["primary"]][str(phi)]
    return None if "no_positive_peak" in cell["flags"] else (cell["lo"], cell["hi"])


def rho_ladder(lo, hi, k):
    """The arm's rho candidates: multiples of RHO_STEP in [lo/k, hi/k] except 1.0, nearest 1 first (smaller rho on ties)."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k!r}")
    first, last = max(1, math.ceil(lo / k / RHO_STEP - 1e-9)), math.floor(hi / k / RHO_STEP + 1e-9)
    rhos = {round(j * RHO_STEP, 10) for j in range(first, last + 1)} - {1.0}
    return sorted(rhos, key=lambda r: (abs(math.log(r)), r))


def choose_phi(recovery):
    """The widest band (smallest phi first) whose pooled 2023-24 recovery holds MIN_RECOVERED and does not lose to taken."""
    for phi in PHIS:
        r = recovery.get(phi)
        if r is None or r["recovered"] < MIN_RECOVERED or r["recovered_ret"] is None or r["taken_ret"] is None:
            continue
        if r["recovered_ret"] >= r["taken_ret"]:
            return {"phi": phi, "flags": []}
    return {"phi": None, "flags": ["insufficient_recovery"]}


def adopt(phi, accounts, alpha):
    """The pre-stated rule: the fallback beats base in every window, and its 2025 breakeven alpha is within ALPHA_MARGIN of base's."""
    if phi is None:
        return {"adopt": False, "phi": None, "beats": False, "alpha_ok": False, "reason": "insufficient_recovery"}
    beats = all([accounts[w]["fallback"] > accounts[w]["base"] for w in WINDOW_KEYS])
    fb, base = alpha["fallback"], alpha["base"]
    alpha_ok = fb is not None and base is not None and fb <= base + ALPHA_MARGIN + 1e-12
    return {"adopt": beats and alpha_ok, "phi": phi, "beats": beats, "alpha_ok": alpha_ok, "reason": None}


def tier(mcap):
    """Market-cap tier in USD: ge_10B, 1B_10B or lt_1B; unknown for None or NaN."""
    if mcap is None or math.isnan(mcap):
        return "unknown"
    if mcap >= 10e9:
        return "ge_10B"
    return "1B_10B" if mcap >= 1e9 else "lt_1B"
