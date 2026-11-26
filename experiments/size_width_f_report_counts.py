"""EXP-189 Amendment 1, slice W7c2 part 1: report side of the A1.4 non-trade counts and the A1.3 non-positive-debit counts
(design A1.3, A1.4). Reads only stored results; size_width_f_counts writes them and only size_width_report consumes this module."""
import math

from experiments.size_pnl_report import md_table
from experiments.size_pnl_shortvol import NAMES
from experiments.size_width_f_fills import ENTRY_REASONS
from experiments.size_width_success import LADDER

KEYS = ("candidates", "f_skipped", "reasons", "only_arm_f_skipped", "only_production_f_skipped")
COUNTS = ("candidates", "f_skipped", "only_arm_f_skipped", "only_production_f_skipped")
ALPHA_KEYS = tuple(f"{a:.2f}" for a in LADDER)


def _count(x):
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _share(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1


def _nontrades_ok(nt, label):
    """One structure's non-trade counts of ``label``: exactly the stored keys (any_k_f_skipped_share only for E); each count a
    non-bool int >= 0; the F reasons sum to f_skipped; f_skipped <= candidates, only_arm <= f_skipped and only_production <=
    candidates - f_skipped; E's share is None exactly when there are no candidates, else a finite number in [0, 1]."""
    keys = set(KEYS) | ({"any_k_f_skipped_share"} if label == "E" else set())
    if not isinstance(nt, dict) or set(nt) != keys: return False
    c, f, arm, prod, reasons = nt["candidates"], nt["f_skipped"], nt["only_arm_f_skipped"], nt["only_production_f_skipped"], nt["reasons"]
    if not all(_count(nt[k]) for k in COUNTS): return False
    if not isinstance(reasons, dict) or not all(_count(v) for v in reasons.values()): return False
    if sum(reasons.get(r, 0) for r in ENTRY_REASONS) != f or f > c or arm > f or prod > c - f: return False
    if label != "E": return True
    share = nt["any_k_f_skipped_share"]
    return share is None if c == 0 else _share(share)


def _debit_ok(account):
    """The account's non-positive debit counts: exactly the ladder's alpha keys, each a count no larger than the account's trades."""
    debit = account["nonpositive_debit"]
    return isinstance(debit, dict) and set(debit) == set(ALPHA_KEYS) and all(_count(debit[a]) and debit[a] <= account["trades"] for a in ALPHA_KEYS)


def admit(results, labels, periods):
    """True only when, for every label and period, every structure of NAMES passes _nontrades_ok and the account passes _debit_ok.
    Labels and periods are admitted by size_width_f_views.admit first. Missing, extra or malformed input is False, never an exception."""
    try:
        for label in labels:
            for p in periods:
                res = results[label][p]
                if not _debit_ok(res["account"]): return False
                if not all(_nontrades_ok(res["structures"][n]["nontrades"], label) for n in NAMES): return False
        return True
    except (KeyError, TypeError, AttributeError, OverflowError):
        return False


def _cell(n, candidates):
    """A count as "N (P)", P its share of the candidates to 2 decimals; "0 (n/a)" when there are no candidates."""
    return "0 (n/a)" if candidates == 0 else f"{n} ({100 * n / candidates:.2f})"


def counts_markdown(title, labels, periods, results):
    """T2c and T5c: per label, period and structure, the candidates, each F entry reason and the F skips as "N (P)", the two
    comparisons with production as counts, and E's any-k share as a percent. Non-F reasons are not shown."""
    headers = ["arm", "year", "structure", "candidates", *ENTRY_REASONS, "F-skipped", "only this arm F-skipped",
               "only production F-skipped", "E: events with a k F-skipped %"]
    rows = []
    for label in labels:
        for p in periods:
            for n in NAMES:
                nt = results[label][p]["structures"][n]["nontrades"]
                c = nt["candidates"]
                share = nt.get("any_k_f_skipped_share") if label == "E" else None
                rows.append([label, p, n, c, *(_cell(nt["reasons"].get(r, 0), c) for r in ENTRY_REASONS), _cell(nt["f_skipped"], c),
                             nt["only_arm_f_skipped"], nt["only_production_f_skipped"], "n/a" if share is None else f"{100 * share:.2f}"])
    return f"### {title}\n\n" + md_table(headers, rows)


def nonpositive_markdown(labels, periods, results):
    """T5d: per label and period, the taken F trades and, per alpha, the taken trades with no ladder return (repriced debit not positive)."""
    rows = [[label, p, results[label][p]["account"]["trades"], *(results[label][p]["account"]["nonpositive_debit"][a] for a in ALPHA_KEYS)]
            for label in labels for p in periods]
    return ("### T5d F-selected taken trades with no ladder return (repriced debit not positive), seven structures, by arm and alpha "
            "(2025 NOT blind for widths)\n\n" + md_table(["arm", "year", "trades", *(f"alpha {a}" for a in ALPHA_KEYS)], rows))
