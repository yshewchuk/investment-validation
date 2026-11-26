"""EXP-189 Amendment 1, slice W7c1: the report side of F. Reads the beside-F views W7b2 stores in each stage summary's results
(account: all six views; structures: F, mid, harsh) and renders the beside-F tables and the POST-HOC label (design A1.1, A1.2, A1.10)."""
import math

from experiments.fair_value_fills import VARIANTS
from experiments.size_pnl_report import md_table
from experiments.size_pnl_shortvol import NAMES

LABEL = "POST-HOC (Amendment 1, F fills)"
ACCOUNT_VIEWS = ("F", "mid", "harsh")
VIEWS = (*ACCOUNT_VIEWS, *VARIANTS)


def _real(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _entry_ok(entry, names):
    """One account or structure entry: exactly ``names`` as views; a non-bool int trade count >= 0; a finite total_return;
    every view walks the taken set (wanted == trades) with a finite total_return; mean_ret is None exactly when there are
    no trades and else a finite real; and the F view is the deciding total_return."""
    if set(entry["views"]) != set(names): return False
    trades = entry["trades"]
    if isinstance(trades, bool) or not isinstance(trades, int) or trades < 0: return False
    if not _real(entry["total_return"]): return False
    for name in names:
        cell = entry["views"][name]
        if cell["wanted"] != trades or not _real(cell["total_return"]): return False
        mean = cell["mean_ret"]
        if (mean is None) != (trades == 0): return False
        if mean is not None and not _real(mean): return False
    return math.isclose(entry["views"]["F"]["total_return"], entry["total_return"], rel_tol=1e-12, abs_tol=1e-12)


def admit(results, labels, periods):
    """True only when, for every label and period, the account passes _entry_ok over VIEWS and every structure of NAMES
    passes it over ACCOUNT_VIEWS. Missing or malformed input is False, never an exception."""
    try:
        for label in labels:
            for period in periods:
                res = results[label][period]
                if not _entry_ok(res["account"], VIEWS): return False
                if not all(_entry_ok(res["structures"][n], ACCOUNT_VIEWS) for n in NAMES): return False
        return True
    except (KeyError, TypeError, AttributeError, OverflowError):
        return False


def has_views(results):
    """True when any results[label][period] has a "views" key in its account or in any structure entry (False on the plain non-F shape)."""
    return any("views" in res["account"] or any("views" in entry for entry in res["structures"].values())
               for by in results.values() for res in by.values())


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.2f}"


def t5_markdown(labels, periods, results, stats):
    """T5 at F fills (the deciding cell), then the beside-F account and per-trade tables by view."""
    keys = [(label, p) for label in labels for p in periods]
    def view(label, p, v): return results[label][p]["account"]["views"][v]
    deciding = [[label, p, results[label][p]["account"]["trades"], _pct(view(label, p, "F")["total_return"]),
                 _pct(view(label, p, "F")["mean_ret"]), "n/a" if stats[label][p]["x"] is None else f"{stats[label][p]['x']:.3f}"]
                for label, p in keys]
    def section(title, key): return f"### {title}\n\n" + md_table(["arm", "period", *VIEWS], [[label, p, *(_pct(view(label, p, v)[key]) for v in VIEWS)] for label, p in keys])
    return "\n\n".join([
        "### T5 production versus arms ($200,000 account at F fills; 2025 NOT blind for widths)\n\n"
        + md_table(["arm", "period", "trades", "account return %", "return per trade %", "breakeven alpha"], deciding),
        section("T5 beside F: seven-structure $200,000 account return % by view (reported, never deciding; 2025 NOT blind for widths)", "total_return"),
        section("T5 beside F: return per trade % by view (reported, never deciding; 2025 NOT blind for widths)", "mean_ret")])


def t6_markdown(labels, results):
    """T6 per-structure stand-alone accounts in 2025 at mid and at harsh (reported, never deciding)."""
    return "\n\n".join(
        f"### T6 beside F: per-structure stand-alone accounts at {view} (2025, return %; reported, never deciding; 2025 NOT blind for widths)\n\n"
        + md_table(["structure", *labels], [[n, *(_pct(results[label]["2025"]["structures"][n]["views"][view]["total_return"]) for label in labels)] for n in NAMES])
        for view in ("mid", "harsh"))


def label_headings(text):
    """``text`` with the label appended to every heading line (a line starting with "#")."""
    return "\n".join(f"{line} [{LABEL}]" if line.startswith("#") else line for line in text.split("\n"))
