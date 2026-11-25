"""EXP-188 Amendment 6 (slice A6-3a): M0's G_PROD gate agreement with production's stored 2025 gate decisions, the mandatory M0 gate section. Research-only: reached from size_pnl_gprod_report. Pure functions over published decision rows; the one read is the bounded stored-gate CSV."""
import io
import math
import os
import stat

import numpy as np
import pandas as pd

from engine.v2.ops.errors import fail
from experiments import size_pnl_report as report
from experiments.size_pnl_gate_fold import REGISTERED_THRESHOLD

# The agreement bar is the owner's choice (design section 18 names none). Documented, never tuned.
MIN_JACCARD = 0.5
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
MAX_STORED_BYTES = 64 << 20
HEADING = "### G_PROD M0 agreement with production's stored 2025 gate decisions"
AS_RUN = "decisions (M0 gate_score as run)"
_INVALID = "INVALID_EXPERIMENT_SPEC"
NOTE = ("covered = M0's determined 2025 events in E (the transactions file holds selected rows only, so a covered event absent from it is "
        "read as not selected; stored events outside the covered set are counted, not used). Diagnostics are never rules and no threshold "
        "is tuned. Known training-universe difference: EXP-187 refits on v2-snapshot trades, the fold CLI on legacy replay trades.")
NO_LADDER = ("Decomposition steps (a) band from production's full-history pool and (b) unrestricted analog pool were not supplied via "
             "--gate-ladder.")
COUNT_HEADERS = ["covered", "gprod_selected", "production_selected", "both", "only_production", "only_gprod", "stored_outside_covered",
                 "rate_gprod", "rate_production", "jaccard", "agreement_rate", "bar"]
LIVE_HEADERS = ["events_excluded", "production_removed", "live_jaccard", "bar"]
STRADDLE_PREFIX = "Straddle verdicts (H2, and H3 through its straddle leg): "
NOT_COMPUTED = STRADDLE_PREFIX + "production agreement not computed in this run."


def _refuse(reason, message):
    return fail(_INVALID, message, details={"reason": reason})


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_stored(path):
    """{"selected": {event_id: ret_mid}} from EXP-187's transactions CSV (selected rows only). The file is opened without following links."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except (OSError, ValueError):  # a NUL byte in the path raises ValueError
        raise _refuse("stored_gate_unreadable", "the stored gate file is missing or not a plain file") from None
    try:
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STORED_BYTES:
                raise _refuse("stored_gate_unreadable", "the stored gate file is not a regular file within its size cap")
            data = handle.read(MAX_STORED_BYTES + 1)
    except OSError:
        raise _refuse("stored_gate_unreadable", "the stored gate file could not be read") from None
    if len(data) > MAX_STORED_BYTES:
        raise _refuse("stored_gate_unreadable", "the stored gate file is larger than its size cap")
    try:
        table = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    except Exception:  # empty, malformed or not CSV
        raise _refuse("stored_gate_unreadable", "the stored gate file cannot be parsed as CSV") from None
    if not {"event_id", "ret"} <= set(table.columns):
        raise _refuse("stored_gate_columns", "the stored gate file lacks event_id or ret")
    ids = list(table["event_id"])
    rets = []
    for text in table["ret"]:
        try:
            rets.append(float(text))
        except ValueError:
            raise _refuse("stored_gate_invalid", "the stored gate file has a non-numeric or non-finite ret") from None
    if len(set(ids)) != len(ids):
        raise _refuse("stored_gate_invalid", "the stored gate file repeats an event id")
    stored = {"selected": dict(zip(ids, rets))}
    if not valid(stored):
        raise _refuse("stored_gate_invalid", "the stored gate file has an empty or whitespace-only event id or a non-finite ret")
    return stored


def valid(stored) -> bool:
    """True when ``stored`` is {"selected": {event_id: finite ret}} with non-blank string ids (bool values refused)."""
    if not (isinstance(stored, dict) and set(stored) == {"selected"} and isinstance(stored["selected"], dict)):
        return False
    return all(isinstance(e, str) and e.strip() and _finite(v) for e, v in stored["selected"].items())


def ladder_valid(ladder) -> bool:
    """True when ``ladder`` is {label: {event_id: finite score}} with str labels (not AS_RUN) and non-empty str event ids."""
    return isinstance(ladder, dict) and AS_RUN not in ladder and all(
        isinstance(label, str) and isinstance(steps, dict) and all(isinstance(e, str) and e and _finite(s) for e, s in steps.items())
        for label, steps in ladder.items())


def _sets(m0, selected):
    covered = set(m0["event_id"])
    gprod = set(m0.loc[m0["status"] == "taken", "event_id"])
    return covered, gprod, set(selected) & covered


def _mean(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def stats(m0, selected):
    """Agreement counts, rates, Jaccard and mean mid returns of M0's determined events against the stored selected set."""
    covered, gprod, prod = _sets(m0, selected)
    both, only_prod, only_gprod = gprod & prod, prod - gprod, gprod - prod
    n, union = len(covered), len(gprod | prod)
    taken = m0[m0["status"] == "taken"]
    ret_g = dict(zip(taken["event_id"], taken["exit_value_mid"] / taken["entry_cost"] - 1))
    return {"covered": n, "n_gprod": len(gprod), "n_prod": len(prod), "both": len(both), "only_prod": len(only_prod),
            "only_gprod": len(only_gprod), "outside": len(set(selected) - covered),
            "rate_gprod": len(gprod) / n if n else None, "rate_prod": len(prod) / n if n else None,
            "jaccard": len(both) / union if union else None,
            "agreement_rate": (n - len(only_prod) - len(only_gprod)) / n if n else None,
            "ret_both_prod": _mean(selected[e] for e in both), "ret_only_prod": _mean(selected[e] for e in only_prod),
            "ret_both_gprod": _mean(ret_g[e] for e in both), "ret_only_gprod": _mean(ret_g[e] for e in only_gprod)}


def quantile_table(m0, threshold):
    """Quantiles, min, max, the registered threshold and the share of finite gate scores at or above it."""
    s = m0["gate_score"].to_numpy(dtype=float)
    s = s[np.isfinite(s)]
    rows = [["finite_scores", int(s.size)]] + [[f"q{p:g}", float(np.quantile(s, p)) if s.size else None] for p in QUANTILES]
    rows += [["min", float(s.min()) if s.size else None], ["max", float(s.max()) if s.size else None],
             ["registered threshold", threshold], ["share >= threshold", float((s >= threshold).mean()) if s.size else None]]
    return report.md_table(["statistic", "value"], rows)


def ladder_table(steps, covered, prod, threshold):
    """One row per step: its finite scores on covered events, those at or above the threshold, and the overlap with production."""
    rows = []
    for label, scores in steps.items():
        scored = {e for e, v in scores.items() if e in covered and math.isfinite(v)}
        selected = {e for e in scored if scores[e] >= threshold}
        overlap, union = len(selected & prod), len(selected | prod)
        rows.append([label, len(scored), len(selected), len(selected) / len(scored) if scored else None, overlap,
                     overlap / union if union else None])
    return report.md_table(["step", "scored", "selected", "rate", "overlap_with_production", "jaccard"], rows)


def admit_gap(accepted_gap):
    """Admits ``accepted_gap``: None or a non-blank str. Raises accepted_gap_invalid otherwise."""
    if accepted_gap is not None and not (isinstance(accepted_gap, str) and accepted_gap.strip()):
        raise _refuse("accepted_gap_invalid", "accepted_gap must be a non-empty string")


def admit_selected(selected):
    """Admits ``selected``: {event_id: finite ret} with non-blank ids. Raises stored_gate_invalid otherwise."""
    if not valid({"selected": selected}):
        raise _refuse("stored_gate_invalid", "selected must map non-empty event ids to finite returns")


def check(st, accepted_gap):
    """Raises when agreement is below the bar and no gap is accepted; returns the label ("" when at or above the bar)."""
    admit_gap(accepted_gap)
    jac = st["jaccard"]
    if not (st["covered"] and jac is not None and jac >= MIN_JACCARD):
        if accepted_gap is None:
            raise _refuse("gate_agreement_below_bar", f"G_PROD agreement with production is below the bar: covered {st['covered']}, "
                          f"gprod selected {st['n_gprod']}, production selected {st['n_prod']}, overlap {st['both']}, jaccard {jac} "
                          f"(bar {MIN_JACCARD}); pass --accept-gate-gap to report the gap")
        return f"not production-faithful, agreement jaccard {jac} (bar {MIN_JACCARD}); accepted gap: {accepted_gap}"
    return ""


def live_domain(m0, selected):
    """The live domain: M0's covered events less those with reason universe_filter, removed from both sides (the m0 rows and
    production's selected ids). ``jaccard`` is None when the live union is empty."""
    excluded = set(m0.loc[m0["reason"] == "universe_filter", "event_id"])
    kept = m0[~m0["event_id"].isin(excluded)]
    live_selected = {e: v for e, v in selected.items() if e not in excluded}
    return {"excluded": len(excluded), "prod_removed": len(set(selected) & excluded), "jaccard": stats(kept, live_selected)["jaccard"]}


def agreement_parts(m0, selected, accepted_gap):
    """The one computation behind the M0 section and the straddle label: the as-run stats, the live domain, and check's label
    (raises as check does). Admits both inputs before anything computes."""
    admit_gap(accepted_gap)
    admit_selected(selected)
    st = stats(m0, selected)
    return st, live_domain(m0, selected), check(st, accepted_gap)


def _live_text(live):
    return "not computable inside the live domain" if live is None else f"{live:.2f} inside the live domain"


def straddle_label(m0, selected, accepted_gap):
    """The straddle verdicts' label, from agreement_parts. Below the bar it names the live jaccard and the accepted gap (stripped);
    at or above the bar it carries no acceptance wording."""
    st, live, gap = agreement_parts(m0, selected, accepted_gap)
    live_jaccard = live["jaccard"]  # the live domain is a dict; its jaccard is None when the live union is empty
    if gap:
        return f"{STRADDLE_PREFIX}agreement {_live_text(live_jaccard)}, {accepted_gap.strip()}; not fully production-faithful"
    return f"{STRADDLE_PREFIX}as-run agreement {st['jaccard']:.2f} meets the {MIN_JACCARD} bar; {_live_text(live_jaccard)}."


def section(m0, selected, *, accepted_gap=None, ladder=None):
    """The M0 agreement section (markdown). ``m0``: M0's determined 2025 G_PROD rows in E, one per event; ``selected``: {id: ret_mid}.
    check runs first, so a failing bar raises before anything is rendered."""
    admit_selected(selected)  # before the ladder check, so selected is refused first
    if ladder is not None and not ladder_valid(ladder):
        raise _refuse("ladder_invalid", "ladder must map labels (not the as-run label) to {event_id: finite score}")
    st, live, label = agreement_parts(m0, selected, accepted_gap)
    covered, gprod, prod = _sets(m0, selected)
    counts = [[st["covered"], st["n_gprod"], st["n_prod"], st["both"], st["only_prod"], st["only_gprod"], st["outside"],
               st["rate_gprod"], st["rate_prod"], st["jaccard"], st["agreement_rate"], MIN_JACCARD]]
    rets = [["both, production stored ret", st["both"], st["ret_both_prod"]],
            ["only production, stored ret", st["only_prod"], st["ret_only_prod"]],
            ["both, G_PROD ret", st["both"], st["ret_both_gprod"]],
            ["only G_PROD, G_PROD ret", st["only_gprod"], st["ret_only_gprod"]]]
    steps = {AS_RUN: dict(zip(m0["event_id"], m0["gate_score"]))}
    steps.update(ladder or {})
    ladder_text = ladder_table(steps, covered, prod, REGISTERED_THRESHOLD) + ("" if ladder else "\n\n" + NO_LADDER)
    parts = [HEADING] + ([label] if label else []) + [NOTE,
             "#### Counts\n\n" + report.md_table(COUNT_HEADERS, counts),
             "#### Live domain (universe_filter removed from both sides)\n\n"
             + report.md_table(LIVE_HEADERS, [[live["excluded"], live["prod_removed"], live["jaccard"], MIN_JACCARD]]),
             "#### Mean return at mid\n\n" + report.md_table(["group", "events", "mean_ret_mid"], rets),
             "#### Score distribution\n\n" + quantile_table(m0, REGISTERED_THRESHOLD),
             "#### Decomposition ladder (diagnostic, never a rule)\n\n" + ladder_text]
    return "\n\n".join(parts)
