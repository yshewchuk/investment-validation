"""The report generator: the only way a result becomes a record.

Phases 1-3 and 5 all emit through this module, so "the report" means one
format with one fixed section order and one provenance contract — consumers
can rely on where a number lives, and a report that cannot be regenerated
from its provenance block is a bug (the acceptance suite proves it).

Markdown is assembled from string templates (no jinja2 dependency); figures
are matplotlib with the Agg backend, one function per standard figure so the
style stays uniform, every caption stating what would falsify the result it
shows. Determinism: fixed seeds, fixed axis policies, no timestamps inside
PNGs — timestamps live in the markdown.

Fixed section order (consumers rely on it): Verdict → Headline table → Sample
funnel → Equity/DD → By-year → Monte Carlo → Stress grid → Calibration →
Accuracy checklist → Provenance → Additional analyses → Appendix → Glossary.

**One formatting contract.** ``METRIC_SPEC`` maps every metric key to its
formatter, its label and its definition, so a return renders as ``+4.04%``
everywhere and cannot render as ``0.0404`` in one table and ``+4.04%`` in the
next. Missing values render as ``n/a``, never as a dash that reads like a
number. The glossary is generated from the same mapping, so a metric cannot
reach a table without a definition existing.

**The verdict block** (§0) is derived, never composed: every line maps to a
field of the results dict, and its warnings are computed here rather than
passed in, so a caller cannot quiet one by omitting it.

**Accuracy checklist — auto-evaluated.** Each item renders PASS / FAIL / N/A
*with an evidence pointer*, and the generator computes them rather than taking
the caller's word: real prices only, leak audit ran, headline = walk-forward
OOS only, fill sensitivity present, multiple-testing ledger cited,
survivorship caveat included, pre-registration valid. Any FAIL renders a red
banner at the top — the report can exist as a diagnostic, but promotion and
publish paths treat FAIL as blocking.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from engine import paths

__all__ = ["GENERATOR_VERSION", "ChecklistItem", "Advisory", "accuracy_checklist",
           "advisories", "build_provenance", "Report", "Verdict", "verdict",
           "compute_warnings", "sample_funnel", "stress_state", "METRIC_SPEC",
           "GLOSSARY", "SECTIONS_BY_KIND", "cell", "COMPOSITE_METRIC_KEYS", "fmt_metric", "metric_label", "MISSING",
           "MIN_STRESS_COVERAGE", "MIN_BRIER_SKILL",
           "pct", "prob", "ratio", "num", "count", "money_x", "pp"]

GENERATOR_VERSION = "1.0.0"

#: The engine modules a report's provenance pins. These are the files whose
#: bytes determine every number in the report; hashing them is what makes
#: "regenerated from the provenance block" checkable.
_CODE_MODULES = (
    "engine/__init__.py",
    "engine/paths.py",
    "engine/fills.py",
    "engine/structures.py",
    "engine/replay.py",
    "engine/audit.py",
    "engine/evaluate.py",
    "engine/report.py",
)

SURVIVORSHIP_NOTE = (
    "Survivorship caveat: the trade universe is built from the CURRENTLY "
    "listed names with data in the store. Delisted names (acquisitions, "
    "bankruptcies) are under-represented, which biases a long-vol program "
    "mildly in its favor — the worst prints of a name that no longer exists "
    "are the ones missing from the sample."
)

#: Sources whose quotes are licensed for P&L. Anything else — most importantly
#: oquants model-fitted marks — is banned (standing rule, VERDICT_2026-08-27).
REAL_PRICE_SOURCES = ("orats", "polygon", "engine.replay")


def _sha256(path: Path) -> str:
    from engine.data.store import file_sha256

    return file_sha256(Path(path))


def _file_fingerprint(path: Path) -> dict[str, Any]:
    """Provenance entry for one input file.

    Files over 100 MB hash only their first megabyte plus size and mtime —
    hashing six-million-row chain partitions in full on every report would
    cost more than the evaluation itself, and a silent same-size edit of a
    partition changes the first MB in practice (row-group headers move).
    """
    p = Path(path)
    info: dict[str, Any] = {"path": str(p)}
    if not p.exists():
        info["missing"] = True
        return info
    size = p.stat().st_size
    info["bytes"] = size
    if size > 100 * (1 << 20):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            h.update(fh.read(1 << 20))
        info["first_mb_sha256"] = h.hexdigest()
        info["mtime"] = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
        info["note"] = ">100MB: size+mtime+first-MB hash"
    else:
        info["sha256"] = _sha256(p)
    return info


def build_provenance(spec_hash: str | None = None,
                     seeds: Mapping[str, Any] | None = None,
                     input_files: Sequence[Path | str] = ()) -> dict[str, Any]:
    """The regeneration contract, shared by evaluations and promotions.

    Input files + hashes, the data snapshot, the seeds, the code state (sha256
    of every engine module a report depends on), the quota state, and the
    generator version. A report that cannot be regenerated from this block is
    a bug — the acceptance suite proves it.
    """
    snapshot_hash = None
    if paths.SNAPSHOT_FILE.exists():
        try:
            snapshot_hash = json.loads(paths.SNAPSHOT_FILE.read_text()).get("snapshot")
        except (ValueError, OSError):
            snapshot_hash = None

    from engine.data.throttle import latest_quota

    q = latest_quota()
    quota_state = (
        f"{q['remaining']} remaining at {q['ts']} ({q['source']})"
        if q["remaining"] is not None else None
    )

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION,
        "spec_hash": spec_hash,
        "data_snapshot": snapshot_hash,
        "seeds": dict(seeds or {}),
        "inputs": [_file_fingerprint(Path(p)) for p in input_files],
        "code": {m: _sha256(paths.ROOT / m) for m in _CODE_MODULES if (paths.ROOT / m).exists()},
        "quota_state": quota_state,
    }


# --------------------------------------------------------------------------
# accuracy checklist
# --------------------------------------------------------------------------


class ChecklistItem:
    def __init__(self, name: str, status: str, evidence: str):
        assert status in ("PASS", "FAIL", "N/A")
        self.name = name
        self.status = status
        self.evidence = evidence

    def row(self) -> str:
        return f"| {self.name} | **{self.status}** | {self.evidence} |"


def accuracy_checklist(results: Mapping[str, Any], spec: Mapping[str, Any],
                       ledger_path: Path | None = None) -> list[ChecklistItem]:
    """The seven-item evidence standard, computed from the results themselves."""
    items: list[ChecklistItem] = []

    # 1. Real prices only.
    source = str(spec.get("price_source", "")).lower()
    if any(s in source for s in REAL_PRICE_SOURCES):
        items.append(ChecklistItem("Real prices only", "PASS", f"price_source={spec.get('price_source')!r}"))
    elif source:
        items.append(ChecklistItem("Real prices only", "FAIL",
                                   f"price_source={spec.get('price_source')!r} is not a sanctioned source"))
    else:
        items.append(ChecklistItem("Real prices only", "N/A", "spec declares no price_source"))

    # 2. Leak audit ran on this evaluation.
    audit = (results.get("walk_forward") or {}).get("audit")
    receipt = (audit or {}).get("receipt")
    if receipt:
        items.append(ChecklistItem(
            "Leak audit ran", "PASS",
            f"receipt: {receipt['paths'][0]}, {receipt['n_folds_checked']} fold(s), "
            f"{receipt['n_rows_checked']:,} row(s), latest fit year "
            f"{receipt['max_fit_year']}, min margin {receipt['min_margin_years']} year(s) "
            f"before the traded year"))
    elif audit and audit.get("fit_years_seen"):
        items.append(ChecklistItem("Leak audit ran", "PASS",
                                   f"fits saw max year per fold: {audit['fit_years_seen']}"))
    elif audit:
        # A receipt with no fits means no gate ran — there was nothing to
        # audit. That is honestly N/A, not a PASS.
        items.append(ChecklistItem("Leak audit ran", "N/A",
                                   "no gate fitted in this evaluation"))
    else:
        items.append(ChecklistItem("Leak audit ran", "N/A", "no walk-forward audit receipt"))

    # 3. Headline numbers are walk-forward OOS only.
    stage = results.get("headline_stage")
    items.append(ChecklistItem(
        "Headline = walk-forward OOS", "PASS" if stage == "wf_oos" else "FAIL",
        f"headline_stage={stage!r}"))

    # 4. Fill sensitivity shown.
    sweep = (results.get("headline") or {}).get("alpha_sweep") or (results.get("backtest") or {}).get("alpha_sweep") or {}
    be = (results.get("headline") or {}).get("breakeven_alpha", "missing")
    if len(sweep) >= 3 and "breakeven_alpha" in (results.get("headline") or {}):
        items.append(ChecklistItem("Fill sensitivity", "PASS",
                                   f"{len(sweep)} alphas swept; breakeven alpha={num(be)}"))
    else:
        items.append(ChecklistItem("Fill sensitivity", "FAIL",
                                   f"only {len(sweep)} alpha(s) swept; breakeven={be}"))

    # 5. Multiple-testing ledger cited — the spec itself must appear in it,
    # not merely the file exist.
    sha = results.get("spec_hash", "")
    if ledger_path is not None and Path(ledger_path).exists():
        lines = Path(ledger_path).read_text().splitlines()
        total = max(len(lines) - 1, 0)
        if total == 0:
            items.append(ChecklistItem(
                "Multiple-testing ledger", "N/A", "no experiments tried yet (ledger empty)"))
        else:
            rows = [ln for ln in lines[1:] if sha[:16] in ln]
            if rows:
                items.append(ChecklistItem(
                    "Multiple-testing ledger", "PASS",
                    f"spec {sha[:12]}… appears in {len(rows)} ledger row(s); "
                    f"{total} spec(s) tried overall"))
            else:
                items.append(ChecklistItem(
                    "Multiple-testing ledger", "FAIL",
                    f"spec {sha[:12]}… never registered in the ledger ({total} row(s) exist)"))
    else:
        items.append(ChecklistItem("Multiple-testing ledger", "FAIL", "no LEDGER.csv attached"))

    # 6. Survivorship caveat included.
    items.append(ChecklistItem("Survivorship caveat", "PASS", "auto-included; current-listed universe"))

    # 7. Preregistration valid.
    prereg = results.get("preregistration") or {}
    if prereg.get("valid"):
        detail = f"preregistered_at={prereg.get('preregistered_at', 'stamp present')}"
        items.append(ChecklistItem("Preregistration", "PASS", detail))
    elif prereg.get("enforced") is False:
        items.append(ChecklistItem("Preregistration", "N/A", "run not attached to an experiment dir"))
    else:
        items.append(ChecklistItem("Preregistration", "FAIL", "no valid preregistered_at"))

    return items


# --------------------------------------------------------------------------
# formatting contract
# --------------------------------------------------------------------------
#
# One mapping from a metric key to its unit, its label and its definition.
# Every table cell, every glossary row and every verdict sentence renders
# through this, because two places computing "percent" differently is exactly
# the bug this contract exists to prevent: the generator used to print
# ``mean/trade 0.0404`` in the headline while a hand-appended appendix printed
# the same number as ``+4.04%``.
#
# ``MISSING`` is the only token a formatter emits for an absent value. It is
# never a bare em dash: a dash reads as a rendered number and hides the
# difference between "measured zero" and "never measured" (the stale-date
# stress shipped a NaN as ``—`` for exactly that reason).

MISSING = "n/a"


def _f(x: Any) -> float | None:
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if np.isfinite(xf) else None


def pct(x: Any, nd: int = 2, signed: bool = True) -> str:
    """A return: signed percent. ``0.0404`` -> ``+4.04%``."""
    xf = _f(x)
    if xf is None:
        return MISSING
    return f"{xf * 100:+.{nd}f}%" if signed else f"{xf * 100:.{nd}f}%"


def prob(x: Any, nd: int = 1) -> str:
    """A rate or probability: unsigned percent. ``0.3978`` -> ``39.8%``."""
    xf = _f(x)
    return MISSING if xf is None else f"{xf * 100:.{nd}f}%"


def ratio(x: Any, nd: int = 2) -> str:
    """A multiple. ``1.2438`` -> ``1.24×``."""
    xf = _f(x)
    if xf is None:
        return MISSING
    xf = xf + 0.0 if xf else 0.0          # -0.0 must not render as "-0.00×"
    return f"{xf:.{nd}f}×"


def num(x: Any, nd: int = 2) -> str:
    """A unitless statistic (Sharpe, Sortino, monotonicity)."""
    xf = _f(x)
    return MISSING if xf is None else f"{xf:.{nd}f}"


def count(x: Any) -> str:
    """A count, always with thousands separators."""
    xf = _f(x)
    return MISSING if xf is None else f"{int(round(xf)):,}"


def money_x(x: Any) -> str:
    """Terminal equity as a multiple of the start, clamped.

    A report that prints ``45,528,139,812,307,560×`` teaches its reader to
    distrust the document. The exact value stays in ``results/metrics_*.json``.
    """
    xf = _f(x)
    if xf is None:
        return MISSING
    if abs(xf) >= 1e6:
        return ">1e6×"
    return f"{xf:,.0f}×" if abs(xf) >= 100 else f"{xf:,.2f}×"


def pp(x: Any, nd: int = 1) -> str:
    """A difference between two rates, in percentage points."""
    xf = _f(x)
    return MISSING if xf is None else f"{xf * 100:+.{nd}f} pp"


#: key -> (formatter, label, one-line definition). The single source for the
#: canonical metrics table, the glossary and the verdict block, so a metric
#: cannot appear in a table without a definition existing (checked by
#: ``checks/phase4_checks.py::glossary_complete``).
METRIC_SPEC: dict[str, tuple[Any, str, str]] = {
    "n": (count, "n", "number of trades in the sample"),
    "mean": (pct, "mean/trade", "average return per trade, on the trade's own premium"),
    "median": (pct, "median/trade", "middle return per trade — below the mean when a few large wins carry the sample"),
    "std": (lambda x, nd=2: pct(x, nd=nd, signed=False), "std",
            "standard deviation of per-trade returns"),
    "win_rate": (prob, "win rate", "share of trades with return > 0"),
    "profit_factor": (ratio, "profit factor", "gross wins / gross losses; 1.00× is break-even"),
    "sharpe_trade": (num, "Sharpe (trade)", "mean/std of per-trade returns × √(trades per year)"),
    "sharpe_equity": (num, "Sharpe (equity)", "daily equity-curve Sharpe × √252, on the 5%-sized walk-forward curve"),
    "sortino": (num, "Sortino", "mean / downside deviation vs 0 — Sharpe counting only losses as risk"),
    "max_dd": (prob, "max drawdown", "worst peak-to-trough fall on the 5%-sized equity curve"),
    "tail_ratio": (ratio, "tail ratio", "|p95 win| / |p95 loss|; below 1.00× the losing tail is the bigger one"),
    "dollar_weighted": (pct, "return on capital",
                        "total P&L / total premium paid — what the average DOLLAR "
                        "returned, as opposed to the average trade"),
    "breakeven_alpha": (num, "breakeven alpha", "the fill quality at which the strategy makes exactly zero"),
}

#: Canonical headline keys that are nested blocks rather than scalar cells:
#: they carry their own sections and tables, so they need no cell formatter.
COMPOSITE_METRIC_KEYS = ("by_year", "capacity", "deployment", "mc")

#: Program vocabulary that is not a metrics-table key. Term -> (definition, why
#: it is in the report at all).
GLOSSARY: dict[str, tuple[str, str]] = {
    "fill alpha (α)": (
        "Where inside the bid-ask spread a trade is assumed to fill: 0 = worst "
        "(buy the ask, sell the bid), 0.5 = mid, 1 = best.",
        "Worst-case fills kill every strategy in this program and mid fills flip "
        "all three positive, so no P&L number means anything without its alpha."),
    "breakeven alpha": (
        "The alpha at which mean return per trade crosses zero, interpolated "
        "across the swept alpha grid.",
        "It is the margin of safety on the mid-fill assumption — the single "
        "biggest risk in the program, and what Phase 5 measures for real."),
    "walk-forward OOS": (
        "Expanding-window evaluation: fit on years < Y, trade year Y, then "
        "concatenate the traded years. Nothing sees the year it trades.",
        "Headline numbers come only from this stage; in-sample numbers are "
        "diagnostics and are labelled as such."),
    "anti-selection guard": (
        "Any statistic computed on a model-selected subset is reported beside "
        "the same statistic on the unselected universe.",
        "A gate that merely re-labels an already-profitable universe looks "
        "identical to a gate with real skill until you print both."),
    "block bootstrap": (
        "Monte Carlo that resamples contiguous blocks of 20 trades rather than "
        "single trades.",
        "Earnings cluster in weeks; independent resampling would destroy that "
        "clustering and understate drawdowns."),
    "deployment / marked equity": (
        "Deployment is total premium at risk divided by equity; marked equity is "
        "cash plus open positions carried at cost.",
        "Per-trade sizing times concurrency is implicit leverage — a 5% trade "
        "with 133 positions open is not a 5% book."),
    "Brier skill": (
        "1 − Brier / Brier(base rate). Positive means the predicted "
        "probabilities beat always predicting the sample's base rate; negative "
        "means they are worse than that constant.",
        "It is the test of whether a win rate is a probability or only a "
        "ranking. The program's floor is −0.05."),
    "spec hash": (
        "SHA-256 of the pre-registered spec — the identity a result is filed "
        "under in the multiple-testing ledger.",
        "It is what makes 'this spec was registered before the OOS run' "
        "checkable rather than asserted."),
    "data snapshot hash": (
        "Hash of the Tier-3 feature snapshot the evaluation read.",
        "Two reports with the same snapshot hash saw the same data; a report "
        "that cannot name its snapshot cannot be regenerated."),
    "model layer / analog layer": (
        "The model layer pushes a champion model's prediction through the "
        "structure's payoff; the analog layer is the empirical distribution of "
        "matched historical trades.",
        "They are always shown side by side and never averaged — disagreement "
        "between them is information, not noise."),
    "EXTRAPOLATED": (
        "A label on any score away from the at-the-money strikes the evidence "
        "actually covers.",
        "Edge decay across moneyness is unmeasured until the Phase 2 backlog "
        "item lands; unlabelled extrapolation would be a silent accuracy claim."),
    "INCONCLUSIVE": (
        "A stress stage that ran but had too little chain coverage (<5% of "
        "trades) or produced a non-finite statistic.",
        "It is reported as its own state so a stress that did nothing cannot "
        "read as a stress that passed."),
}

#: Which sections each report kind renders. Evaluations render everything;
#: other kinds skip the sections their results have no data for rather than
#: printing tables of "n/a".
SECTIONS_BY_KIND: dict[str, tuple[str, ...]] = {
    "evaluation": ("verdict", "headline", "funnel", "equity", "by_year", "mc",
                   "stress", "calibration", "checklist", "provenance", "extras",
                   "appendix", "glossary"),
    "calibration": ("verdict", "funnel", "calibration", "checklist", "provenance",
                    "extras", "appendix", "glossary"),
    "audit": ("verdict", "funnel", "checklist", "provenance", "extras", "appendix",
              "glossary"),
    "forward_test": ("verdict", "headline", "funnel", "equity", "by_year",
                     "calibration", "checklist", "provenance", "extras", "appendix",
                     "glossary"),
}

#: A stress stage below this chain coverage has not, in any useful sense, been
#: performed. Judgement call, deliberately visible as one.
MIN_STRESS_COVERAGE = 0.05

#: The Brier-skill floor from the Phase 1 decision record
#: (reports/phase1_decision_calibration_reclassification.md).
MIN_BRIER_SKILL = -0.05


def cell(value: Any) -> str:
    """One markdown table cell, safe to paste free text into.

    A pipe inside a cell silently splits the row into extra columns, so a
    caller writing a perfectly reasonable ``max |Δ| 3.6e-15`` would corrupt the
    table it was documenting. Escaped here, once, rather than remembered at
    every call site.
    """
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def fmt_metric(key: str, value: Any) -> str:
    """Render a metric through its declared formatter."""
    formatter = METRIC_SPEC.get(key, (num, key, ""))[0]
    return formatter(value)


def metric_label(key: str) -> str:
    return METRIC_SPEC.get(key, (num, key, ""))[1]


# --------------------------------------------------------------------------
# stress states
# --------------------------------------------------------------------------


def stress_state(stage: Mapping[str, Any] | None,
                 *, coverage: float | None = None,
                 statistic: Any = None) -> tuple[str, str]:
    """MEASURED / INCONCLUSIVE / N/A for one stress stage, with its reason.

    ``available: True`` carrying a NaN is the worst of both worlds: it renders
    as a result and contains none. A stage that ran on 0.1% of trades gets its
    own state so the report cannot imply the stress passed.
    """
    if not stage:
        return "N/A", "stage not run"
    if stage.get("available") is False:
        return "N/A", str(stage.get("note") or "not applicable")
    cov = _f(coverage if coverage is not None else stage.get("coverage"))
    if cov is not None and cov < MIN_STRESS_COVERAGE:
        return ("INCONCLUSIVE",
                f"only {prob(cov, nd=2)} of trades had an adjacent cached chain; "
                "this stress has not been performed")
    if statistic is not None and _f(statistic) is None:
        return "INCONCLUSIVE", "the stage ran but produced no finite statistic"
    return "MEASURED", ""


# --------------------------------------------------------------------------
# sample funnel
# --------------------------------------------------------------------------


def sample_funnel(results: Mapping[str, Any], spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Which universe every n in the report belongs to.

    Upstream stages (calendar events, chain availability, pricing drops) are
    only knowable to the caller that built the trades, so they arrive through
    ``spec['universe_funnel']``; the evaluation's own stages are derived here.
    The row the headline is computed on is flagged, in every report.
    """
    rows: list[dict[str, Any]] = []
    for row in (spec.get("universe_funnel") or []):
        rows.append({"stage": row.get("stage", "?"), "events": row.get("events"),
                     "note": row.get("note", ""), "headline": False})

    backtest = results.get("backtest") or {}
    if backtest.get("n_events") is not None:
        rows.append({"stage": "priced into trades", "events": backtest["n_events"],
                     "note": "unique events with both legs priced at every alpha",
                     "headline": False})

    wf = (results.get("walk_forward") or {}).get("diagnostics") or []
    if wf:
        n_test = sum(int(d.get("n_test", 0)) for d in wf)
        n_selected = sum(int(d.get("n_selected", 0)) for d in wf)
        gated = [d for d in wf if not d.get("ungated")]
        rows.append({"stage": "walk-forward out-of-sample", "events": n_test,
                     "note": f"{len(wf)} folds; {len(wf) - len(gated)} ungated "
                             f"(no fitted gate, all rows kept)",
                     "headline": n_selected == n_test})
        if n_selected != n_test:
            rows.append({"stage": "selected by the gate", "events": n_selected,
                         "note": "rows without complete gate features are unscoreable "
                                 "and are not selected",
                         "headline": True})
    return rows


# --------------------------------------------------------------------------
# the verdict block
# --------------------------------------------------------------------------


class Advisory:
    """A non-blocking readout: how far the evidence stretches.

    Distinct from :class:`ChecklistItem`, which is about whether the evidence
    is admissible at all. Advisories never block promotion; FAILs still do.
    """

    def __init__(self, name: str, status: str, evidence: str):
        assert status in ("OK", "WARN", "N/A")
        self.name = name
        self.status = status
        self.evidence = evidence

    def row(self) -> str:
        return f"| {self.name} | **{self.status}** | {self.evidence} |"


class Verdict:
    def __init__(self, call: str, sentence: str, rows: list[tuple[str, str, str]],
                 warnings: list[str]):
        self.call = call
        self.sentence = sentence
        self.rows = rows
        self.warnings = warnings


def _falsification_clause(hypothesis: str) -> str | None:
    """The spec's own falsifier, verbatim — never paraphrased."""
    text = " ".join(str(hypothesis or "").split())
    for marker in ("Falsified if", "falsified if"):
        idx = text.find(marker)
        if idx >= 0:
            return text[idx:].strip()
    return None


def compute_warnings(results: Mapping[str, Any], calibration: Mapping[str, Any] | None,
                     checklist: Sequence[ChecklistItem] = ()) -> list[str]:
    """Every warning the verdict block carries, computed from the results.

    Warnings are derived, never passed in — a caller cannot quiet one by
    omitting it.
    """
    warns: list[str] = []
    headline = results.get("headline") or {}

    skill = _f((calibration or {}).get("brier_skill"))
    if skill is not None and skill < MIN_BRIER_SKILL:
        warns.append("win-rate calibration below the base rate "
                     f"(Brier skill {num(skill, nd=3)})")

    predicted = _f((calibration or {}).get("predicted_mean_pnl"))
    realized = _f((calibration or {}).get("realized_mean_pnl"))
    if predicted is not None and realized is not None:
        gap = predicted - realized
        if abs(gap) > 0.02 or (predicted > 0 and realized <= 0):
            warns.append(
                f"expected P&L misses realized by {pp(gap)} "
                f"(predicted {pct(predicted)}, realized {pct(realized)})")

    stress = results.get("stress") or {}
    inconclusive = []
    slippage = stress.get("slippage") or {}
    for shift, s in (slippage.get("shifts") or {}).items():
        state, _ = stress_state(slippage, coverage=s.get("coverage"),
                                statistic=s.get("delta_mean"))
        if state == "INCONCLUSIVE":
            inconclusive.append(f"slippage {shift}")
    stale = stress.get("stale_dates") or {}
    if stale:
        state, _ = stress_state(stale, coverage=stale.get("coverage"),
                                statistic=stale.get("delta_mean"))
        if state == "INCONCLUSIVE":
            inconclusive.append("stale dates")
    if inconclusive:
        warns.append(f"{len(inconclusive)} stress stage(s) INCONCLUSIVE "
                     f"({', '.join(inconclusive)})")

    mean_v, dollar_v = _f(headline.get("mean")), _f(headline.get("dollar_weighted"))
    if mean_v is not None and dollar_v is not None and mean_v - dollar_v > 0.01:
        warns.append(f"equal-weighted {pct(mean_v)} but capital-weighted {pct(dollar_v)} "
                     "— the edge sits in the cheapest contracts")

    ungated = _f(headline.get("ungated_share"))
    if ungated is not None and ungated > 0.25:
        warns.append(f"{prob(ungated)} of headline trades are from ungated years "
                     "(base exposure, not the gate)")

    spread_pnl = ((headline.get("capacity") or {}).get("pnl_by_spread") or {})
    widest = _f(spread_pnl.get("widest_quintile_share"))
    if widest is not None and widest > 0.5:
        warns.append(f"{prob(widest)} of net P&L comes from the widest-quoted fifth of "
                     "markets, where mid is least achievable")

    conc = ((results.get("transaction_log") or {}).get("concentration") or {})
    share = _f(conc.get("top10_net_share"))
    if share is not None and share > 0.5:
        warns.append(f"10 trades carry {prob(share)} of the net result "
                     f"({count(conc.get('trades_for_half_the_gains'))} winners make "
                     "half the gains)")

    concurrency = _f(headline.get("max_concurrency"))
    if concurrency is not None and concurrency > 1 and (results.get("mc") or {}).get("by_fraction"):
        warns.append(f"MC sizing ignores the {int(concurrency)}-position overlap "
                     "(its terminal column is an upper bound)")

    by_year = headline.get("by_year") or {}
    thin = [y for y, s in by_year.items() if _f(s.get("n")) is not None and s["n"] < 30]
    if thin:
        warns.append(f"{len(thin)} headline year(s) with n < 30 ({', '.join(sorted(thin))})")

    be = _f(headline.get("breakeven_alpha"))
    if be is not None and be > 0.45:
        warns.append(f"breakeven alpha {num(be)} leaves under 5 points of margin below mid")

    fails = [i.name for i in checklist if getattr(i, "status", "") == "FAIL"]
    if fails:
        warns.append(f"accuracy checklist FAIL: {', '.join(fails)}")
    return warns


def verdict(results: Mapping[str, Any], spec: Mapping[str, Any],
            calibration: Mapping[str, Any] | None = None,
            checklist: Sequence[ChecklistItem] = ()) -> Verdict:
    """The report's first section: what this evaluation says, in one screen.

    Every line is template-derived from the results dict. The function may
    only say things the results can support — the moment it starts composing
    sentences of its own the report stops being reproducible.

    ``spec['type']`` ("confirmatory" | "descriptive") decides whether the call
    is a pass/fail on a hypothesis or a statement of what was measured; a spec
    that omits it is descriptive when it has no promotion target and its
    hypothesis opens with the word "descriptive".
    """
    headline = results.get("headline") or {}
    warns = compute_warnings(results, calibration, checklist)
    rows: list[tuple[str, str, str]] = []

    mean = _f(headline.get("mean"))
    n = _f(headline.get("n"))
    by_year = headline.get("by_year") or {}
    years = sorted(by_year)
    span = f"{years[0]}–{years[-1]}" if years else "the sample"
    if mean is not None:
        rows.append((
            "Does it make money at mid fills?",
            f"**{'Yes' if mean > 0 else 'No'}** — {pct(mean)}/trade over "
            f"{count(n)} OOS trades ({span})", "§1"))

    be = _f(headline.get("breakeven_alpha"))
    if be is not None:
        margin = (0.5 - be) * 100
        rows.append((
            "How much fill quality does it need?",
            f"Breakeven at **α = {num(be)}** — needs {prob(be)} of the spread; "
            f"mid is 50%. Margin: {margin:+.1f} points", "§1"))

    if by_year:
        positive = [y for y in years if (_f(by_year[y].get("mean")) or 0) > 0]
        weakest = min(years, key=lambda y: _f(by_year[y].get("mean")) or 0.0)
        rows.append((
            "Is it positive every year?",
            f"**{len(positive)} of {len(years)}** OOS years positive, weakest "
            f"{weakest} ({pct(by_year[weakest].get('mean'))})", "§3"))

    stress = results.get("stress") or {}
    regimes = stress.get("regimes") or {}
    if regimes:
        pos = [k for k, s in regimes.items() if (_f(s.get("mean")) or 0) > 0]
        detail = f"{len(pos)}/{len(regimes)} crisis regimes positive"
        incon = [w for w in warns if "INCONCLUSIVE" in w]
        rows.append((
            "Does it survive the stress battery?",
            (f"**Partly** — {detail}; " + incon[0] if incon
             else f"**{'Yes' if len(pos) == len(regimes) else 'Partly'}** — {detail}"), "§5"))

    skill = _f((calibration or {}).get("brier_skill"))
    if skill is not None:
        ok = skill >= MIN_BRIER_SKILL
        rows.append((
            "Are the win rates trustworthy?",
            f"**{'Yes' if ok else 'No'}** — Brier skill {num(skill, nd=3)}"
            + ("" if ok else ", worse than the base rate. Treat win_rate as a "
                            "ranking, not a probability"), "§6"))

    predicted = _f((calibration or {}).get("predicted_mean_pnl"))
    realized = _f((calibration or {}).get("realized_mean_pnl"))
    if predicted is not None and realized is not None:
        ok = abs(predicted - realized) <= 0.02 and not (predicted > 0 >= realized)
        rows.append((
            "Does expected P&L match what happened?",
            f"**{'Yes' if ok else 'No'}** — predicted {pct(predicted)}/trade, realized "
            f"{pct(realized)}/trade ({pp(predicted - realized)})", "§6"))

    concurrency = _f(headline.get("max_concurrency"))
    if (results.get("mc") or {}).get("by_fraction"):
        if concurrency is not None and concurrency > 1:
            rows.append((
                "What sizing does the evidence support?",
                f"Undetermined here — MC ignores the {int(concurrency)}-position "
                "overlap; see the deployment block", "§4"))
        else:
            mc5 = (results.get("mc") or {}).get("by_fraction", {}).get("0.05") or {}
            rows.append((
                "What sizing does the evidence support?",
                f"At 5%: P(final loss) {prob(mc5.get('p_loss'))}, drawdown p95 "
                f"{prob(mc5.get('dd_p95'))}", "§4"))

    falsifier = _falsification_clause(spec.get("hypothesis"))
    if falsifier:
        rows.append(("What would falsify it?", falsifier, "spec.yaml"))

    kind = str(spec.get("type") or "").lower()
    if not kind:
        hyp = " ".join(str(spec.get("hypothesis") or "").split()).lower()
        kind = "descriptive" if (not spec.get("promotion_target")
                                 and hyp.startswith("descriptive")) else "confirmatory"

    if kind == "descriptive":
        call = "DESCRIPTIVE"
        sentence = ("This experiment measures rather than proposes: it carries no "
                    "promotion target, and the rows below are what it measured, "
                    "not a pass/fail.")
    else:
        positive_mean = mean is not None and mean > 0
        margin_ok = be is not None and be < 0.5
        if positive_mean and margin_ok:
            call = "SUPPORTED"
        elif positive_mean or margin_ok:
            call = "MIXED"
        else:
            call = "NOT SUPPORTED"
        sentence = (
            f"Mean {pct(mean)}/trade at mid over {count(n)} walk-forward OOS trades, "
            f"breakeven α = {num(be)} against a mid assumption of 0.50."
            if mean is not None else "No headline trades were produced.")
    if warns and call in ("SUPPORTED", "DESCRIPTIVE"):
        call = f"{call} (with {len(warns)} warning{'s' if len(warns) != 1 else ''})"
    return Verdict(call, sentence, rows, warns)


def advisories(results: Mapping[str, Any], spec: Mapping[str, Any],
               calibration: Mapping[str, Any] | None,
               funnel: Sequence[Mapping[str, Any]] = ()) -> list[Advisory]:
    """The second, non-blocking table: how far this evidence stretches."""
    out: list[Advisory] = []
    headline = results.get("headline") or {}

    skill = _f((calibration or {}).get("brier_skill"))
    if skill is None:
        out.append(Advisory("Calibration state", "N/A",
                            str((calibration or {}).get("reason")
                                or "no OOS probabilities to calibrate")))
    elif skill < MIN_BRIER_SKILL:
        out.append(Advisory("Calibration state", "WARN",
                            f"Brier skill {num(skill, nd=3)} (below the {num(MIN_BRIER_SKILL)} floor) "
                            "— win rates rank, they do not measure"))
    else:
        out.append(Advisory("Calibration state", "OK", f"Brier skill {num(skill, nd=3)}"))

    predicted = _f((calibration or {}).get("predicted_mean_pnl"))
    realized = _f((calibration or {}).get("realized_mean_pnl"))
    if predicted is not None and realized is not None:
        gap = predicted - realized
        bad = abs(gap) > 0.02 or (predicted > 0 >= realized)
        out.append(Advisory("Expected vs realized P&L", "WARN" if bad else "OK",
                            f"predicted {pct(predicted)} vs realized {pct(realized)} "
                            f"({pp(gap)})"))

    stress = results.get("stress") or {}
    stages, dead = 0, 0
    slippage = stress.get("slippage") or {}
    for _shift, s in (slippage.get("shifts") or {}).items():
        stages += 1
        if stress_state(slippage, coverage=s.get("coverage"),
                        statistic=s.get("delta_mean"))[0] == "INCONCLUSIVE":
            dead += 1
    for key in ("stale_dates", "tail_injection"):
        stage = stress.get(key)
        if stage:
            stages += 1
            if stress_state(stage, coverage=stage.get("coverage"))[0] == "INCONCLUSIVE":
                dead += 1
    if stress.get("regimes"):
        stages += 1
    if stages:
        out.append(Advisory("Stress coverage", "WARN" if dead else "OK",
                            f"{dead} of {stages} stage(s) INCONCLUSIVE" if dead
                            else f"{stages} stage(s), all measured or N/A"))

    if funnel:
        unexplained = [r for r in funnel if not r.get("note")]
        out.append(Advisory(
            "Sample funnel disclosed", "WARN" if unexplained else "OK",
            f"{len(funnel)} stage(s)"
            + (f"; {len(unexplained)} without a reason" if unexplained
               else "; every drop carries a reason")))
    else:
        out.append(Advisory("Sample funnel disclosed", "WARN",
                            "no funnel supplied — the reader cannot tell which "
                            "universe the headline n belongs to"))

    concurrency = _f(headline.get("max_concurrency"))
    if concurrency is not None and concurrency > 1:
        out.append(Advisory("Concurrency vs sizing", "WARN",
                            f"max {int(concurrency)} concurrent positions; MC sizing "
                            "is an upper bound and its P(loss) a lower bound"))
    elif concurrency is not None:
        out.append(Advisory("Concurrency vs sizing", "OK", "no overlapping positions"))

    by_year = headline.get("by_year") or {}
    if by_year:
        ns = [int(s.get("n", 0)) for s in by_year.values()]
        thin = [y for y, s in by_year.items() if int(s.get("n", 0)) < 30]
        out.append(Advisory("Sample size per headline year", "WARN" if thin else "OK",
                            f"min year n = {count(min(ns))}"
                            + (f"; thin: {', '.join(sorted(thin))}" if thin else "")))
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _finish(fig, path: Path, caption: str, *, bottom: float = 0.22) -> Path:
    """Reserve space for the caption, then save.

    Captions used to be dropped at ``y=0.005`` on top of whatever the axes had
    already drawn there, so every figure in every report overprinted its own
    x-axis label. The margin is reserved first; the caption goes into it.
    """
    fig.subplots_adjust(bottom=bottom)
    fig.text(0.01, 0.02, caption, fontsize=7, color="gray", wrap=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _figure_data(path: Path, data: Mapping[str, Any]) -> None:
    """Persist the arrays behind a figure next to the PNG.

    Golden-file tests compare these, not pixels: font metrics drift between
    matplotlib versions, the numbers do not.
    """
    payload = {k: (list(v) if isinstance(v, (list, tuple, np.ndarray, pd.Series)) else v)
               for k, v in data.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(payload, indent=1, default=str))


def _auto_yscale(ax, values: Sequence[float], label: str) -> str:
    """Log scale once a series spans more than two orders of magnitude.

    A 25,000× equity curve on a linear axis is a flat line for the first seven
    of its nine years — the reader sees nothing until the last months.
    """
    finite = [v for v in (_f(v) for v in values) if v is not None and v > 0]
    if finite and max(finite) / min(finite) > 100:
        ax.set_yscale("log")
        ax.set_ylabel(f"{label} (log scale)")
        return "log"
    ax.set_ylabel(label)
    return "linear"


def fig_equity(equity: pd.Series, path: Path, title: str,
               *, ungated_years: Sequence[int] = ()) -> Path:
    plt = _matplotlib()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(equity.index, equity.values, lw=1.2, color="tab:blue")
    scale = _auto_yscale(ax1, equity.values, "equity (× start)")
    ax1.set_title(title)

    peak = equity.cummax()
    dd = (equity / peak - 1.0)
    ax2.fill_between(equity.index, dd.values, 0.0, color="tab:red", alpha=0.6)
    ax2.set_ylabel("drawdown")
    ax2.yaxis.set_major_formatter(lambda v, _pos: f"{v:.0%}")

    # Mark the worst drawdown, and the years that ran ungated — a reader
    # otherwise cannot see which part of the curve the gate was not selecting.
    if len(dd):
        trough = dd.idxmin()
        ax2.axvline(trough, color="k", lw=0.8, ls=":")
        ax2.annotate(f"max DD {dd.min():.0%}", xy=(trough, dd.min()),
                     xytext=(4, 6), textcoords="offset points", fontsize=7)
    for year in ungated_years:
        ax1.axvspan(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"),
                    color="gray", alpha=0.10)
    if len(ungated_years):
        ax1.text(0.01, 0.95, "shaded = ungated years (all rows kept)",
                 transform=ax1.transAxes, fontsize=7, color="gray", va="top")
    for boundary in pd.date_range(equity.index.min(), equity.index.max(), freq="YS"):
        ax1.axvline(boundary, color="gray", lw=0.4, alpha=0.5)
    _figure_data(path, {"date": [str(d.date()) for d in pd.DatetimeIndex(equity.index)],
                        "equity": list(map(float, equity.values)),
                        "drawdown": list(map(float, dd.values)), "yscale": scale})
    return _finish(fig, path, "Falsified if: a re-run with the same seed and trades "
                              "does not reproduce this curve.", bottom=0.16)


def fig_by_year(by_year: Mapping[str, Mapping[str, float]], path: Path, title: str) -> Path:
    plt = _matplotlib()
    years = sorted(by_year)
    means = [by_year[y]["mean"] for y in years]
    counts = [by_year[y].get("n") for y in years]
    colors = ["tab:green" if (m is not None and m >= 0) else "tab:red" for m in means]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(years, means, color=colors)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("mean return / trade")
    ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:.0%}")
    span = max((abs(m) for m in means if m is not None), default=0.0)
    for x, (m, c) in enumerate(zip(means, counts)):
        if m is None:
            continue
        ax.text(x, m + (0.03 * span if m >= 0 else -0.06 * span), f"n={c:,}",
                ha="center", fontsize=7, color="gray")
    ax.set_title(title)
    _figure_data(path, {"year": years, "mean": means, "n": counts})
    return _finish(fig, path, "Falsified if: the edge concentrates in so few years "
                              "that removing one flips the sign.")


def fig_mc_fan(mc: Mapping[str, Any], path: Path, title: str) -> Path:
    plt = _matplotlib()
    frac_stats = mc.get("by_fraction", {})
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fractions, p05, p50, p95, ploss = [], [], [], [], []
    for key in sorted(frac_stats):
        s = frac_stats[key]
        fractions.append(float(key))
        p05.append(s["terminal_p05"])
        p50.append(s["terminal_p50"])
        p95.append(s["terminal_p95"])
        ploss.append(s["p_loss"])
    ax.plot(fractions, p05, "o-", label="terminal p05", color="tab:red")
    ax.plot(fractions, p50, "o-", label="terminal p50", color="tab:blue")
    ax.plot(fractions, p95, "o-", label="terminal p95", color="tab:green")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    # p05 and p50 are invisible beside a p95 four orders of magnitude above
    # them on a linear axis — which is what the first version of this chart
    # shipped.
    scale = _auto_yscale(ax, p05 + p50 + p95, "terminal equity (× start)")
    ax.set_xlabel("sizing fraction per trade")
    # Ticks on the sizings actually simulated — interpolated ticks (8%, 12%)
    # invite reading a value off a curve that was never evaluated there.
    ax.set_xticks(fractions)
    ax.xaxis.set_major_formatter(lambda v, _pos: f"{v:.0%}")
    if 0.05 in fractions:
        ax.axvline(0.05, color="tab:orange", lw=1.0, ls=":",
                   label="5% sizing (the plan's base)")
    ax.set_title(title)
    ax2 = ax.twinx()
    ax2.plot(fractions, ploss, "s--", color="gray", label="P(loss)")
    ax2.set_ylabel("P(final loss)")
    ax2.set_ylim(0, 1)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    _figure_data(path, {"fraction": fractions, "p05": p05, "p50": p50, "p95": p95,
                        "p_loss": ploss, "yscale": scale})
    return _finish(fig, path, "Falsified if: live season equity leaves the p05-p95 band "
                              "(the MC is then mis-specified). Overlap is ignored: read "
                              "with the deployment block.")


def fig_alpha_curve(sweep: Mapping[str, Mapping[str, float]], breakeven: float | None,
                    path: Path, title: str) -> Path:
    plt = _matplotlib()
    points = sorted((float(a), s["mean"]) for a, s in sweep.items()
                    if np.isfinite(s.get("mean", np.nan)))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    xs: tuple = ()
    if points:
        xs, ys = zip(*points)
        ax.plot(xs, ys, "o-", color="tab:blue")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.axvline(0.5, color="tab:blue", lw=1.0, ls=":", label="mid (α=0.50)")
        if breakeven is not None:
            ax.axvline(breakeven, color="tab:red", lw=1.0, ls="--",
                       label=f"breakeven α={breakeven:.2f}")
            # The gap between breakeven and mid IS the margin of safety.
            ax.axvspan(min(breakeven, 0.5), max(breakeven, 0.5), color="tab:green",
                       alpha=0.12)
            ax.text((breakeven + 0.5) / 2, ax.get_ylim()[1] * 0.9,
                    f"margin {abs(0.5 - breakeven) * 100:.0f} pts", fontsize=7,
                    ha="center", color="tab:green")
        ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("fill alpha (0 = worst, 1 = best)")
    ax.set_ylabel("mean return / trade")
    ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:.0%}")
    ax.set_title(title)
    _figure_data(path, {"alpha": list(xs), "mean": [p[1] for p in points],
                        "breakeven": breakeven})
    return _finish(fig, path, "Falsified if: measured live fill quality (Phase 5 "
                              "alpha-hat) lands left of breakeven.")


def fig_stress_grid(regimes: Mapping[str, Mapping[str, float]], path: Path, title: str) -> Path:
    plt = _matplotlib()
    names = list(regimes)
    means = [regimes[n].get("mean", np.nan) for n in names]
    counts = [regimes[n].get("n", 0) for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    data = np.array([[m if np.isfinite(m) else 0.0 for m in means]])
    # Centred on zero: a diverging map whose midpoint drifts makes a positive
    # cell read as a bad one.
    bound = max(abs(np.nanmin(means)), abs(np.nanmax(means)), 1e-9)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-bound, vmax=bound)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_yticks([])
    for i, (m, c) in enumerate(zip(means, counts)):
        ax.text(i, -0.18, f"{m:+.1%}" if np.isfinite(m) else "n/a", ha="center",
                va="center", fontsize=9, weight="bold")
        ax.text(i, 0.18, f"n={c:,}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, label="mean return")
    cbar.ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:+.0%}")
    _figure_data(path, {"regime": names, "mean": list(map(float, means)), "n": counts})
    return _finish(fig, path, "Falsified if: a crisis regime shows losses the sizing "
                              "rules do not survive.", bottom=0.26)


def fig_reliability(calibration: Mapping[str, Any], path: Path, title: str) -> Path:
    """Reliability curve: predicted win rate vs realized, per decile.

    ``calibration`` needs ``deciles``: a list of rows with ``predicted`` and
    ``realized`` (and optionally ``n``). The diagonal is perfect calibration.
    """
    plt = _matplotlib()
    deciles = calibration.get("deciles") or []
    base = _f(calibration.get("base_rate"))
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    pred: list[float] = []
    real: list[float] = []
    counts: list[float] = []
    if deciles:
        pred = [float(d.get("predicted", np.nan)) for d in deciles]
        real = [float(d.get("realized", np.nan)) for d in deciles]
        counts = [float(d.get("n", 1)) for d in deciles]
        ax.scatter(pred, real, s=[max(12, 4 * np.sqrt(c)) for c in counts],
                   color="tab:blue", label="model (point size ∝ n)")
        # Shade where the model promises more than it delivers: "all points
        # below the diagonal" should be a statement the figure makes, not an
        # inference the reader has to draw.
        ax.fill_between([0, 1], [0, 1], [0, 0], color="tab:red", alpha=0.06)
        ax.text(0.72, 0.12, "over-promising", fontsize=7, color="tab:red")
    if base is not None:
        ax.axhline(base, color="tab:orange", lw=1.0, ls=":",
                   label=f"base rate {base:.1%}")
    ax.set_xlabel("predicted win rate")
    ax.set_ylabel("realized win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(title)
    _figure_data(path, {"predicted": pred, "realized": real, "n": counts,
                        "base_rate": base})
    return _finish(fig, path, "Falsified if: points stay off the diagonal as ledger "
                              "events accrue (then the shipped win rate is not the "
                              "win rate).", bottom=0.15)


def fig_mc_fan_paths(bands: Mapping[str, Any], path: Path, title: str) -> Path:
    """MC fan chart: percentile bands of the equity paths over the trade index.

    ``bands`` needs ``p05``, ``p50``, ``p95`` arrays aligned on the trade index
    — the true fan over the equity path, as distinct from the sizing curve
    (terminal percentiles vs fraction).
    """
    plt = _matplotlib()
    idx = np.arange(len(bands.get("p50", [])))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if len(idx):
        ax.fill_between(idx, bands["p05"], bands["p95"], color="tab:blue", alpha=0.2,
                        label="p05-p95")
        ax.plot(idx, bands["p50"], color="tab:blue", lw=1.2, label="p50")
        ax.plot(idx, bands["p05"], color="tab:red", lw=0.8, ls=":", label="p05")
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        _auto_yscale(ax, list(bands["p05"]) + list(bands["p95"]), "equity (× start)")
    ax.set_xlabel("trade sequence index (chronological)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    _figure_data(path, {"p05": list(bands.get("p05", [])), "p50": list(bands.get("p50", [])),
                        "p95": list(bands.get("p95", []))})
    return _finish(fig, path, "Falsified if: the realized forward-test equity path "
                              "leaves the p05-p95 band.")


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------


def _fmt(x: Any, pct: bool = False, nd: int = 4) -> str:
    """Legacy raw formatter.

    Retained only where a number genuinely has no unit and no METRIC_SPEC
    entry. New code renders through :func:`fmt_metric` or one of the named
    formatters — a bare four-decimal float in a report body is a defect the
    acceptance suite fails on.
    """
    if x is None:
        return MISSING
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return MISSING
    return f"{xf:+.{nd}f}{'%' if pct else ''}" if pct else f"{xf:.{nd}f}"


class Report:
    """A standard report: REPORT.md + figures/, rendered from a context dict.

    Use :meth:`from_eval` for evaluation results; the plain constructor takes
    an already-assembled context for other phases (calibration, forward-test
    reviews, data audits) that emit through the same format.

    Fixed section order — consumers rely on it:
    0 Verdict → 1 Headline → 1.5 Sample funnel → 2 Equity → 3 By-year →
    4 Monte Carlo → 5 Stress → 6 Calibration → 7 Checklist → 8 Provenance →
    8.5 Additional analyses → 9 Appendix → 10 Glossary.
    """

    def __init__(self, context: Mapping[str, Any]):
        self.context = dict(context)

    # -- assembly -----------------------------------------------------------

    @classmethod
    def from_eval(cls, result, input_files: Sequence[Path | str] = (),
                  extra_sections: Sequence[Mapping[str, Any]] = ()) -> "Report":
        results = result.results
        spec = result.spec
        headline = results.get("headline", {})
        backtest = results.get("backtest", {})

        provenance = build_provenance(
            spec_hash=results.get("spec_hash"),
            seeds={"monte_carlo": (results.get("mc") or {}).get("seed", 0),
                   "equity_mode": results.get("equity_mode")},
            input_files=input_files,
        )

        ledger_path = paths.ROOT / "experiments" / "LEDGER.csv"
        checklist = accuracy_checklist(results, spec, ledger_path=ledger_path)

        calibration = results.get("calibration")
        return cls({
            "kind": "evaluation",
            "spec": spec,
            "results": results,
            "headline": headline,
            "backtest": backtest,
            "checklist": checklist,
            "provenance": provenance,
            "survivorship_note": SURVIVORSHIP_NOTE,
            "funnel": sample_funnel(results, spec),
            "extra_sections": list(extra_sections),
            # Only a calibration that was actually measured reaches the report
            # (and thus the reliability figure); an unavailable one renders as
            # the section's plain-text note instead of a phantom plot.
            "calibration": calibration if calibration and calibration.get("available") else None,
            "calibration_raw": calibration,
        })

    # -- rendering -----------------------------------------------------------

    @property
    def any_fail(self) -> bool:
        return any(item.status == "FAIL" for item in self.context.get("checklist", []))

    @property
    def verdict(self) -> Verdict:
        return verdict(self.context.get("results", {}), self.context.get("spec", {}),
                       self.context.get("calibration"), self.context.get("checklist", []))

    def write(self, out_dir: Path | str, filename: str = "REPORT.md") -> Path:
        """Render to ``out_dir/filename`` with figures beside it.

        ``filename`` exists for the report paths the plan pins by name
        (``reports/phase0_data_audit.md``); experiment folders take the
        default.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # The figures directory is created by the savers, so a report with no
        # figures (a data audit) does not leave an empty one behind.
        figures = self._render_figures(out_dir / "figures")
        if self.context.get("kind") == "promotion":
            md = self._render_promotion_markdown(figures)
        else:
            md = self._render_markdown(figures)
        path = out_dir / filename
        path.write_text(md)
        return path

    def _render_figures(self, fig_dir: Path) -> dict[str, Path]:
        results = self.context.get("results", {})
        headline = self.context.get("headline", {})
        figures: dict[str, Path] = {}

        raw = results.get("equity_curve_series")
        eq = None
        if isinstance(raw, pd.Series):
            eq = raw
        elif isinstance(raw, Mapping) and raw.get("date"):
            eq = pd.Series(raw["equity"], index=pd.to_datetime(raw["date"]))
        if eq is not None and len(eq) > 1:
            ungated = [int(d["year"]) for d in (results.get("walk_forward") or {}).get("diagnostics", [])
                       if d.get("ungated")]
            figures["equity"] = fig_equity(eq, fig_dir / "equity_drawdown.png",
                                           "Equity curve (5% sizing) and drawdown",
                                           ungated_years=ungated)

        by_year = headline.get("by_year") or {}
        if by_year:
            figures["by_year"] = fig_by_year(by_year, fig_dir / "by_year.png",
                                             "Mean return per trade, by year (OOS)")

        if results.get("mc", {}).get("by_fraction"):
            figures["mc"] = fig_mc_fan(results["mc"], fig_dir / "mc_fan.png",
                                       "Monte Carlo sizing curve (block bootstrap)")

        sweep = headline.get("alpha_sweep") or {}
        if sweep:
            figures["alpha"] = fig_alpha_curve(sweep, headline.get("breakeven_alpha"),
                                               fig_dir / "alpha_breakeven.png",
                                               "Fill-quality degradation curve")

        regimes = (results.get("stress") or {}).get("regimes") or {}
        if regimes:
            figures["stress"] = fig_stress_grid(regimes, fig_dir / "stress_grid.png",
                                                "Regime replays: mean return per trade")

        bands = ((results.get("mc") or {}).get("path_bands") or {}).get("0.05")
        if bands and bands.get("p50"):
            figures["mc_fan"] = fig_mc_fan_paths(bands, fig_dir / "mc_fan_paths.png",
                                                 "MC equity fan (5% sizing, p05/p50/p95)")

        cal = self.context.get("calibration") or {}
        if cal.get("deciles"):
            figures["reliability"] = fig_reliability(cal, fig_dir / "reliability.png",
                                                     "Reliability: predicted vs realized win rate")
        return figures

    def _render_promotion_markdown(self, figures: Mapping[str, Path]) -> str:
        spec = self.context.get("spec", {})
        results = self.context.get("results", {})
        headline = self.context.get("headline", {})
        champion = results.get("champion", {})
        prov = self.context.get("provenance", {})
        checklist: list[ChecklistItem] = self.context.get("checklist", [])
        ctx = results.get("ledger_context", {})

        def view(doc: Mapping[str, Any]) -> dict[str, Any]:
            full = "headline" in doc
            h = (doc.get("headline") if full else doc) or {}
            mc = doc.get("mc") or {}
            mc5 = (mc.get("by_fraction") or {}).get("0.05") or ({} if full else mc)
            return {"n": h.get("n"), "mean": h.get("mean"), "win_rate": h.get("win_rate"),
                    "sharpe_trade": h.get("sharpe_trade"), "sharpe_equity": h.get("sharpe_equity"),
                    "max_dd": h.get("max_dd"), "p_loss_5": mc5.get("p_loss")}

        c, h = view({"headline": headline}), view(champion)

        lines: list[str] = []
        add = lines.append
        if self.any_fail:
            add("> **⚠ ACCURACY CHECKLIST HAS FAILING ITEMS — diagnostic only.**")
            add("")
        add(f"# Promotion report — {spec.get('id', 'EXP-?')}")
        add("")
        add(f"*{results.get('decided_at', '')} by engine.report v{prov.get('generator_version')}.*")
        add("")
        tried = ctx.get("specs_tried", 0)
        rows_here = ctx.get("this_spec_rows", 0)
        add(f"**Decision: {results.get('decision', 'PROMOTED')}.** {tried} spec(s) were tried "
            f"against this snapshot before this one (this spec appears in {rows_here} ledger "
            "row(s)) — that count is the multiple-testing context this promotion was earned under.")
        if spec.get("hypothesis"):
            add("")
            add(f"**Hypothesis:** {str(spec['hypothesis']).strip()}")
        add("")
        add("## Rules")
        add("")
        for r in results.get("reasons", []):
            add(f"- {r}")
        add("")
        add("## Challenger vs champion (walk-forward OOS, mid fills)")
        add("")
        add("| metric | challenger | champion |")
        add("|---|---|---|")
        for key in ("n", "mean", "win_rate", "sharpe_trade", "sharpe_equity", "max_dd", "p_loss_5"):
            label = "MC P(loss)@5%" if key == "p_loss_5" else metric_label(key)
            fmt = prob if key == "p_loss_5" else (lambda v, _k=key: fmt_metric(_k, v))
            add(f"| {label} | {fmt(c.get(key))} | {fmt(h.get(key))} |")
        add("")
        if "by_year" in figures:
            add(f"![by year](figures/{figures['by_year'].name})")
            add("")
        if checklist:
            add("## Accuracy-evidence checklist (challenger's evaluation)")
            add("")
            add("| check | status | evidence |")
            add("|---|---|---|")
            for item in checklist:
                add(item.row())
            add("")
        add("## Provenance")
        add("")
        add(f"- spec hash: `{prov.get('spec_hash')}`")
        add(f"- data snapshot: `{prov.get('data_snapshot')}`")
        for f in prov.get("inputs", []):
            detail = f.get("sha256") or f.get("first_mb_sha256") or "MISSING"
            add(f"- input: `{f['path']}` — {detail}")
        add("- code state (sha256):")
        for module, digest in prov.get("code", {}).items():
            add(f"  - `{module}` — {digest[:16]}…")
        add("")
        return "\n".join(lines) + "\n"

    # -- section renderers ---------------------------------------------------

    def _promoted_rows(self) -> list[tuple[str, str, str]]:
        """Verdict rows contributed by extra sections.

        EXP-102's defined-risk falsification is the experiment's headline
        finding; leaving it on page four because it was computed by run.py
        rather than by evaluate() is a formatting accident, not a judgement.
        """
        rows = []
        for i, section in enumerate(self.context.get("extra_sections") or [], start=1):
            if section.get("promote_to_verdict") and section.get("verdict_row"):
                question, answer, *_where = section["verdict_row"]
                # The pointer is computed, never quoted: a caller that hard-codes
                # "§8.5.3" is wrong the moment a section is inserted above it.
                rows.append((question, answer, f"§8.5.{i}"))
        return rows

    def _render_verdict(self, lines: list[str]) -> None:
        v = self.verdict
        add = lines.append
        add("## 0. Verdict — read this first")
        add("")
        add(f"**{v.call}.** {v.sentence}")
        add("")
        rows = list(v.rows)
        promoted = self._promoted_rows()
        if promoted:
            # Promoted findings sit above the falsifier row, which always ends
            # the table.
            insert_at = len(rows) - 1 if rows and rows[-1][2] == "spec.yaml" else len(rows)
            rows[insert_at:insert_at] = promoted
        if rows:
            add("| Question | Answer | Where |")
            add("|---|---|---|")
            for question, answer, where in rows:
                add(f"| {question} | {answer} | {where} |")
            add("")
        if v.warnings:
            add("**Warnings:** " + " · ".join(v.warnings) + ".")
        else:
            add("**Warnings:** none — every advisory in §7 reads OK.")
        add("")

    def _render_headline(self, lines: list[str], figures: Mapping[str, Path]) -> None:
        headline = self.context.get("headline", {})
        backtest = self.context.get("backtest", {})
        results = self.context.get("results", {})
        add = lines.append

        add("## 1. Headline (walk-forward OOS, worst/mid/best fills) "
            "([definitions](#10-glossary))")
        add("")
        sweep = headline.get("alpha_sweep") or backtest.get("alpha_sweep") or {}
        add("| fill | alpha | n | mean/trade | win rate |")
        add("|---|---|---|---|---|")
        for label, a in (("worst", "0.00"), ("mid", "0.50"), ("best", "1.00")):
            s = sweep.get(a)
            if s:
                add(f"| {label} | {a} | {count(s['n'])} | {pct(s['mean'])} | "
                    f"{prob(s['win_rate'])} |")
        be = headline.get("breakeven_alpha")
        add("")
        if be is not None:
            margin = (0.5 - float(be)) * 100
            add(f"**Breakeven alpha: {num(be)}** — the strategy needs {prob(be)} of the "
                f"spread to break even; the mid-fill assumption is 50%, so the margin of "
                f"safety is {margin:+.1f} points.")
        else:
            add("**Breakeven alpha:** none in [0, 1] — the alpha sweep never crosses zero.")
        base = headline.get("base_unselected") or {}
        if base:
            add("")
            add(f"Anti-selection guard — the same statistics on the UNSELECTED universe "
                f"(every replayed event, gate ignored): n={count(base.get('n'))}, "
                f"mean {pct(base.get('mean'))}, win {prob(base.get('win_rate'))}.")
        keys = ("mean", "median", "dollar_weighted", "std", "win_rate", "profit_factor",
                "sharpe_trade", "sharpe_equity", "sortino", "max_dd", "tail_ratio")
        add("")
        add("Canonical metrics (mid fills, walk-forward OOS, on the selected set):")
        add("")
        add("| " + " | ".join(metric_label(k) for k in keys) + " |")
        add("|" + "---|" * len(keys))
        add("| " + " | ".join(fmt_metric(k, headline.get(k)) for k in keys) + " |")
        add("")
        mean_v, dollar_v = _f(headline.get("mean")), _f(headline.get("dollar_weighted"))
        if mean_v is not None and dollar_v is not None:
            gap = mean_v - dollar_v
            if abs(gap) > 0.01:
                add(f"**Equal-weighted {pct(mean_v)} vs capital-weighted "
                    f"{pct(dollar_v)} ({pp(gap)}).** The mean counts a cheap contract "
                    "and an expensive one alike; the capital-weighted number counts "
                    "dollars. When the mean is the larger of the two, the edge sits in "
                    "the cheapest contracts — and fixed-fraction sizing buys the most "
                    "of exactly those, which is a capacity claim rather than a return.")
                add("")

        ungated = _f(headline.get("ungated_share"))
        if ungated is not None and ungated > 0.01:
            add(f"**{prob(ungated)} of these trades come from ungated years** "
                "(folds without enough training history keep every row). To that "
                "extent this headline describes the base exposure, not the gate — "
                "§3 splits it by year and §9 says which folds were ungated.")
            add("")

        capacity = headline.get("capacity") or {}
        if capacity.get("available"):
            wide = capacity.get("wide_market_frac")
            add(f"**Capacity:** mean relative spread at the traded strikes "
                f"{prob(capacity.get('mean_rel_spread'), nd=2)}, p95 "
                f"{prob(capacity.get('p95_rel_spread'), nd=2)}"
                + (f", wide-market fraction {prob(wide)}" if wide is not None else "")
                + f" — {capacity.get('note', '')}")
            spread_pnl = capacity.get("pnl_by_spread") or {}
            if spread_pnl:
                add("")
                add(f"**Where the P&L sits, by quoted width:** the widest fifth of "
                    f"markets (median relative spread "
                    f"{prob(spread_pnl.get('median_rel_spread_widest'))}) supplies "
                    f"{prob(spread_pnl.get('widest_quintile_share'))} of net P&L; the "
                    f"two tightest fifths (median "
                    f"{prob(spread_pnl.get('median_rel_spread_tightest'))}) supply "
                    f"{prob(spread_pnl.get('tightest_two_quintiles_share'))}. Mid is a "
                    "real price only where the market is tight enough for mid to mean "
                    "something, so an edge concentrated in the widest names is an edge "
                    "in the fill assumption.")
        else:
            add("**Capacity:** not measurable on this trade set"
                + (f" — {capacity.get('note')}" if capacity.get("note") else "") + ".")
        add("")
        if results.get("equity_mode") == "sequential":
            add("*Equity mode `sequential` (EXP-050 reference construction): overlap is ignored;*")
            add("*the `cashflow` construction is the default for new experiments.*")
            add("")

    def _render_funnel(self, lines: list[str]) -> None:
        funnel = self.context.get("funnel") or []
        add = lines.append
        add("## 1.5 Sample funnel — which universe each n belongs to")
        add("")
        if not funnel:
            add("*No funnel supplied. Every n in this report is the evaluation's own "
                "trade set; the upstream calendar-to-priced-trades drops are not "
                "disclosed here.*")
            add("")
            return
        add("| stage | events | note |")
        add("|---|---:|---|")
        for row in funnel:
            marker = " ← **headline**" if row.get("headline") else ""
            add(f"| {cell(row['stage'])}{marker} | {count(row.get('events'))} | "
                f"{cell(row.get('note', ''))} |")
        add("")
        add("Every §1–§5 number is computed on the row marked **headline**; a drop "
            "between stages that carries no reason is a defect, not a rounding.")
        add("")

    def _render_equity(self, lines: list[str], figures: Mapping[str, Path]) -> None:
        headline = self.context.get("headline", {})
        add = lines.append
        add("## 2. Equity curve & drawdown (5% sizing)")
        add("")
        if "equity" in figures:
            add(f"![equity](figures/{figures['equity'].name})")
        else:
            add("*No equity series recorded.*")
        add("")
        add(f"Max drawdown: {prob(headline.get('max_dd'))}. Max concurrent positions: "
            f"{count(headline.get('max_concurrency'))}.")
        dep = headline.get("deployment") or {}
        if dep:
            cap = dep.get("cap")
            add("")
            add(f"Deployment at 5% sizing: peak {ratio(dep.get('peak'))} equity, worst cash "
                f"{ratio(dep.get('worst_cash'))} equity"
                + (f", capped at {ratio(cap)} "
                   f"({count(dep.get('constrained_entries', 0))} entries constrained)"
                   if cap else ", UNCAPPED — per-trade sizing times concurrency is "
                               "implicit leverage")
                + ".")
        log = (self.context.get("results", {}) or {}).get("transaction_log") or {}
        if log.get("rows"):
            add("")
            add(f"**Transaction log — every trade behind this curve:** "
                f"`{log.get('path', 'results/transactions_*.csv')}` "
                f"({count(log.get('rows'))} rows"
                + (f", sha256 `{str(log['sha256'])[:16]}…`" if log.get("sha256") else "")
                + "). One row per trade in the order the equity engine processed it, "
                "carrying the quotes it was priced from (per-leg bid/ask, strike, "
                "expiry, DTE at both ends), the contracts bought, the equity it was "
                "sized off, and what it contributed to the final number. A chart "
                "nobody can audit row by row is an assertion, not evidence.")
            add("")
            if log.get("reconciles"):
                add(f"*Reconciled: the {count(log.get('rows'))} contributions sum to "
                    f"{ratio(log.get('implied_final'))} against the curve's "
                    f"{ratio(log.get('final_equity'))} (error "
                    f"{log.get('abs_error', 0):.2e}).*")
                add("")
                add("**To re-derive this curve from the log** (and get the same "
                    "drawdown): process events chronologically, **exits before "
                    "entries on the same date**; an exit credits "
                    "`contracts × exit_value`, an entry debits "
                    "`contracts × entry_cost`; mark equity as "
                    "`cash + Σ contracts × entry_cost` over the still-open positions; "
                    "and keep **one mark per date — the last one**. That last step is "
                    "load-bearing: marking after every event instead of every date "
                    "reads intra-day orderings as troughs and reports a deeper "
                    "drawdown than the daily series the chart plots. The plotted "
                    "series itself is in `figures/equity_drawdown.json`, so the chart "
                    "can also be checked directly, without replaying anything.")
            else:
                add(f"> **⚠ The log does NOT reconcile with the curve:** contributions "
                    f"imply {ratio(log.get('implied_final'))} against "
                    f"{ratio(log.get('final_equity'))}. Treat both as suspect until "
                    "this is explained — a log that does not add up is worse than no "
                    "log, because it looks like evidence.")
        conc = (log.get("concentration") or {}) if log else {}
        if conc.get("trades_for_half_the_gains"):
            add("")
            share = conc.get("top10_net_share")
            add(f"**Concentration:** the 10 largest contributions account for "
                f"{prob(share) if share is not None else 'n/a'} of the net result, and "
                f"{count(conc['trades_for_half_the_gains'])} of "
                f"{count(conc.get('n_winners'))} winning trades make half the gains. "
                "A mean return cannot say this; the transaction log can, and a curve "
                "carried by a handful of trades is a different asset from one that is "
                "not — the same edge, spread thinner, survives a bad week that this "
                "one may not.")
        if "alpha" in figures:
            add("")
            add(f"Fill-quality degradation: ![alpha](figures/{figures['alpha'].name})")
        add("")

    def _render_by_year(self, lines: list[str], figures: Mapping[str, Path]) -> None:
        headline = self.context.get("headline", {})
        add = lines.append
        add("## 3. By year (walk-forward OOS, mid fills)")
        add("")
        by_year = headline.get("by_year") or {}
        add("| year | n | mean/trade | win rate |")
        add("|---|---:|---:|---:|")
        for y in sorted(by_year):
            s = by_year[y]
            add(f"| {y} | {count(s['n'])} | {pct(s['mean'])} | {prob(s['win_rate'])} |")
        add("")
        if "by_year" in figures:
            add(f"![by year](figures/{figures['by_year'].name})")
            add("")

    def _render_mc(self, lines: list[str], figures: Mapping[str, Path]) -> None:
        results = self.context.get("results", {})
        headline = self.context.get("headline", {})
        add = lines.append
        mc = results.get("mc", {})
        add("## 4. Monte Carlo & sizing (block bootstrap on the OOS sequence)")
        add("")
        add("**These are properties of the trade sequence, not forecasts.** The MC "
            "column compounds trades one after another; the deterministic column is the "
            "actual walk-forward book at that sizing, with the deployment cap applied. "
            "Where they diverge, the divergence is overlap: this set ran up to "
            f"{count(headline.get('max_concurrency'))} positions at once, so the MC "
            "terminal column is an upper bound and its P(loss) a lower bound. The sizing "
            "decision belongs to the Phase 5 go/no-go memo, not to this table.")
        add("")
        add(f"block={mc.get('block')}, paths={mc.get('paths')}, seed={mc.get('seed')}, "
            f"n_trades={count(mc.get('n_trades'))}.")
        add("")
        deterministic = results.get("equity_curves") or {}
        add("| sizing | MC terminal p50 (overlap ignored) | deterministic, capped | "
            "MC P(loss) | terminal p05 | p95 | DD p50 | DD p95 |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|")
        for f in sorted(mc.get("by_fraction", {})):
            s = mc["by_fraction"][f]
            det = (deterministic.get(f"{float(f):.2f}") or {}).get("final")
            add(f"| {float(f):.0%} | {money_x(s['terminal_p50'])} | {money_x(det)} | "
                f"{prob(s['p_loss'], nd=2)} | {money_x(s['terminal_p05'])} | "
                f"{money_x(s['terminal_p95'])} | {prob(s['dd_p50'])} | "
                f"{prob(s['dd_p95'])} |")
        add("")
        add("*Terminal equity above 1e6× renders as `>1e6×`; the exact values are in "
            "`results/metrics_*.json`.*")
        add("")
        for key, label in (("mc", "MC sizing curve"), ("mc_fan", "MC equity fan")):
            if key in figures:
                add(f"![{label}](figures/{figures[key].name})")
                add("")
        recon = (self.context.get("spec") or {}).get("reconciliation")
        if recon:
            add(f"> **Reconciliation with the plan's quoted figure:** {recon}")
            add("")

    def _render_stress(self, lines: list[str], figures: Mapping[str, Path]) -> None:
        results = self.context.get("results", {})
        add = lines.append
        add("## 5. Stress battery")
        add("")
        add("Each stage reports one of three states: **MEASURED** (a result), "
            "**INCONCLUSIVE** (it ran but on too little data to mean anything — under "
            f"{prob(MIN_STRESS_COVERAGE, nd=0)} chain coverage), or **N/A** (structurally "
            "inapplicable). An INCONCLUSIVE stage is an honest gap, not a pass.")
        add("")
        stress = results.get("stress", {})
        regimes = stress.get("regimes") or {}
        if regimes:
            add("| regime | n | mean/trade | win rate |")
            add("|---|---:|---:|---:|")
            for name, s in regimes.items():
                add(f"| {name} | {count(s['n'])} | {pct(s.get('mean'))} | "
                    f"{prob(s.get('win_rate'))} |")
            if "stress" in figures:
                add("")
                add(f"![stress grid](figures/{figures['stress'].name})")
            add("")
        iv = stress.get("iv_regime") or {}
        if iv.get("split_by"):
            hi, lo = iv.get("high", {}), iv.get("low", {})
            add(f"**IV-regime split** ({iv['split_by']}): high-vol n={count(hi.get('n'))} "
                f"mean {pct(hi.get('mean'))} vs low-vol n={count(lo.get('n'))} mean "
                f"{pct(lo.get('mean'))}.")
            add("")

        tail = stress.get("tail_injection") or {}
        if tail.get("available") is False:
            flag = "**REQUIRED and missing**" if tail.get("required") else "N/A"
            add(f"**Tail injection:** {flag} — {tail.get('note', 'not applicable')}.")
        elif tail:
            mc_t = tail.get("mc", {}).get("0.05", {})
            add(f"**Tail injection** (worst 1% moves doubled): MEASURED — shocked worst "
                f"trade {pct(tail.get('shocked_worst_trade'))} (base "
                f"{pct(tail.get('base_worst_trade'))}), MC P(loss)@5% "
                f"{prob(mc_t.get('p_loss'), nd=2)}.")
        add("")

        slip = stress.get("slippage") or {}
        if slip.get("available"):
            for shift, s in slip.get("shifts", {}).items():
                state, why = stress_state(slip, coverage=s.get("coverage"),
                                          statistic=s.get("delta_mean"))
                if state == "MEASURED":
                    add(f"**Slippage {shift}:** MEASURED — mean {pct(s.get('mean'))} "
                        f"(Δ {pp(s.get('delta_mean'))}), coverage "
                        f"{prob(s.get('coverage'))}.")
                else:
                    add(f"**Slippage {shift}:** {state} — {why}.")
        else:
            add(f"**Slippage days:** N/A — {slip.get('note', 'no repricer')}.")
        stale = stress.get("stale_dates") or {}
        if stale.get("available"):
            state, why = stress_state(stale, coverage=stale.get("coverage"),
                                      statistic=stale.get("delta_mean"))
            if state == "MEASURED":
                add(f"**Stale dates** (1% mis-dated): MEASURED — Δmean "
                    f"{pp(stale.get('delta_mean'))} on {count(stale.get('n_misdated'))} "
                    f"events.")
            else:
                add(f"**Stale dates** (1% mis-dated): {state} — {why} "
                    f"({count(stale.get('n_misdated'))} events attempted).")
        else:
            add(f"**Stale dates:** N/A — {stale.get('note', 'no repricer')}.")
        add("")

    def _render_calibration(self, lines: list[str], figures: Mapping[str, Path]) -> None:
        cal = self.context.get("calibration")
        raw = self.context.get("calibration_raw") or {}
        add = lines.append
        add("## 6. Calibration — are the predicted win rates real probabilities?")
        add("")
        if not cal:
            reason = raw.get("reason") or ("this evaluation fitted no gate that emits "
                                           "probabilities")
            add(f"**Not measured here** — {reason}.")
            add("")
            add("Predicted-vs-realized win rate for the shipped scorer is measured by the "
                "ledger calibration report (`ledger/calibration/REPORT.md`), which "
                "regenerates every 50 newly scored predictions.")
            add("")
            return

        skill = _f(cal.get("brier_skill"))
        base = _f(cal.get("base_rate"))
        if skill is not None and skill < MIN_BRIER_SKILL:
            add(f"**No.** Brier skill **{num(skill, nd=3)}** (Brier {num(cal.get('brier'), nd=4)} "
                f"vs {num(cal.get('brier_base_rate'), nd=4)} for always predicting the base "
                f"rate of {prob(base)}, n={count(cal.get('n'))}). The predicted win "
                "probabilities are worse than that constant.")
            add("")
            add("**What this does and does not invalidate:** the P&L numbers in §1–§5 are "
                "realized returns and are unaffected. What is affected is any use of "
                "`win_rate` as a probability — sizing off it, or a dashboard reading it as "
                "\"a 58% chance of a win\". Rank order is usable; the level is not. This is "
                "a known, accepted and tracked state, not a fresh surprise: see "
                "`reports/phase1_decision_calibration_reclassification.md`, which "
                "reclassified calibration as an optimization target rather than a gate.")
        else:
            add(f"**Yes, within the program's floor.** Brier skill **{num(skill, nd=3)}** "
                f"(floor {num(MIN_BRIER_SKILL)}), Brier {num(cal.get('brier'), nd=4)} vs "
                f"{num(cal.get('brier_base_rate'), nd=4)} for the base-rate forecaster "
                f"({prob(base)}), n={count(cal.get('n'))}.")
        add("")
        add(f"Reliability monotonicity {num(cal.get('reliability_monotonicity'))} — how "
            "consistently a higher predicted decile delivers a higher realized win rate "
            "(1.00 is perfectly ordered).")
        add("")
        predicted = _f(cal.get("predicted_mean_pnl"))
        realized = _f(cal.get("realized_mean_pnl"))
        if predicted is not None and realized is not None:
            gap = predicted - realized
            verdict_word = ("matches" if abs(gap) <= 0.02 and not (predicted > 0 >= realized)
                            else "does NOT match")
            add(f"**Expected P&L {verdict_word} realized P&L:** predicted "
                f"{pct(predicted)}/trade against realized {pct(realized)}/trade, a gap of "
                f"{pp(gap)}. Win-rate calibration and P&L calibration are different "
                "claims: a model can rank trades correctly and still misstate what they "
                "pay.")
            add("")
        deciles = cal.get("deciles") or []
        if deciles:
            add("| decile | predicted | realized | n | error |")
            add("|---:|---:|---:|---:|---:|")
            for i, d in enumerate(deciles, start=1):
                err = _f(d.get("realized"))
                pred_v = _f(d.get("predicted"))
                gap = None if (err is None or pred_v is None) else err - pred_v
                add(f"| {i} | {prob(d.get('predicted'))} | {prob(d.get('realized'))} | "
                    f"{count(d.get('n'))} | {pp(gap)} |")
            add("")
        if "reliability" in figures:
            add(f"![reliability](figures/{figures['reliability'].name})")
            add("")

    def _render_checklist(self, lines: list[str]) -> None:
        add = lines.append
        checklist: list[ChecklistItem] = self.context.get("checklist", [])
        add("## 7. Accuracy-evidence checklist")
        add("")
        if checklist:
            add("**Mandatory** — whether this evidence is admissible at all. Any FAIL "
                "blocks promotion and publication.")
            add("")
            add("| check | status | evidence |")
            add("|---|---|---|")
            for item in checklist:
                add(item.row())
        else:
            add("**Mandatory** — the seven-item evidence standard applies to evaluations "
                "of a trading result; this report carries none, so there is nothing for "
                "it to admit or refuse.")
        add("")
        adv = advisories(self.context.get("results", {}), self.context.get("spec", {}),
                         self.context.get("calibration"), self.context.get("funnel") or [])
        if adv:
            add("**Advisory** — how far this evidence stretches. Never blocking; read "
                "with §0.")
            add("")
            add("| advisory | status | detail |")
            add("|---|---|---|")
            for item in adv:
                add(item.row())
            add("")

    def _render_provenance(self, lines: list[str]) -> None:
        prov = self.context.get("provenance", {})
        add = lines.append
        add("## 8. Provenance")
        add("")
        add("Everything needed to regenerate this report. A report that cannot be "
            "regenerated from this block is a bug.")
        add("")
        add(f"- spec hash: `{prov.get('spec_hash')}`")
        add(f"- data snapshot: `{prov.get('data_snapshot')}`")
        add(f"- seeds: {json.dumps(prov.get('seeds', {}), default=str)}")
        if prov.get("quota_state"):
            add(f"- quota state (last log row): {prov['quota_state']}")
        add("- input files:")
        for f in prov.get("inputs", []):
            detail = f.get("sha256") or f.get("first_mb_sha256") or "MISSING"
            add(f"  - `{f['path']}` — {detail}" + (f" ({f['note']})" if f.get("note") else ""))
        add("- code state (sha256):")
        for module, digest in prov.get("code", {}).items():
            add(f"  - `{module}` — {digest[:16]}…")
        add("")

    def _render_extra_sections(self, lines: list[str]) -> None:
        extras = self.context.get("extra_sections") or []
        if not extras:
            return
        add = lines.append
        add("## 8.5 Additional analyses")
        add("")
        add("*Spec-specific analyses, rendered through the generator so they carry the "
            "same units, ordering and provenance as everything above.*")
        add("")
        for i, section in enumerate(extras, start=1):
            add(f"### 8.5.{i} {section.get('title', 'analysis')}")
            add("")
            if section.get("note"):
                add(str(section["note"]))
                add("")
            columns = section.get("columns")
            rows = section.get("rows")
            if columns and rows is not None:
                aligns = section.get("align") or ["---"] * len(columns)
                add("| " + " | ".join(cell(c) for c in columns) + " |")
                add("|" + "|".join(aligns) + "|")
                for row in rows:
                    add("| " + " | ".join(cell(c) for c in row) + " |")
                add("")
            for line in section.get("body", []) or []:
                add(str(line))
            if section.get("body"):
                add("")
            if section.get("falsifies"):
                add(f"*Falsified if: {section['falsifies']}*")
                add("")

    def _render_appendix(self, lines: list[str]) -> None:
        results = self.context.get("results", {})
        backtest = self.context.get("backtest", {})
        add = lines.append
        note = self.context.get("survivorship_note", SURVIVORSHIP_NOTE)
        diagnostics = (results.get("walk_forward") or {}).get("diagnostics", [])
        if not (note or diagnostics or backtest.get("by_year") or
                self.context.get("grid_results")):
            return          # an empty appendix is a heading, not a section
        add("## 9. Appendix")
        add("")
        if note:
            add(note)
            add("")
        if diagnostics:
            add("Walk-forward diagnostics — how many rows each fold trained on, tested "
                "and selected:")
            add("")
            add("| year | n train | n test | n selected | ungated |")
            add("|---|---:|---:|---:|---|")
            for d in diagnostics:
                add(f"| {d['year']} | {count(d['n_train'])} | {count(d['n_test'])} | "
                    f"{count(d['n_selected'])} | {'yes' if d.get('ungated') else 'no'} |")
            add("")
        by_year = backtest.get("by_year") or {}
        if by_year:
            add("Unselected universe by year (mid fills) — the anti-selection baseline "
                "the §3 table is measured against:")
            add("")
            add("| year | n | mean/trade | win rate |")
            add("|---|---:|---:|---:|")
            for y in sorted(by_year):
                s = by_year[y]
                add(f"| {y} | {count(s.get('n'))} | {pct(s.get('mean'))} | "
                    f"{prob(s.get('win_rate'))} |")
            add("")
        grid = self.context.get("grid_results")
        if grid:
            add("Grid / secondary results (NOT the headline — the primary spec is):")
            add("")
            add("```json")
            add(json.dumps(grid, indent=1, default=str))
            add("```")
            add("")

    def _render_glossary(self, lines: list[str]) -> None:
        add = lines.append
        add("## 10. Glossary")
        add("")
        add("| term | what it is | why it is in this report |")
        add("|---|---|---|")
        for key, (_formatter, label, definition) in METRIC_SPEC.items():
            if definition:
                add(f"| {label} | {definition} | canonical metric — comparable across "
                    f"every strategy |")
        for term, (definition, why) in GLOSSARY.items():
            add(f"| {term} | {definition} | {why} |")
        add("")

    def _render_markdown(self, figures: Mapping[str, Path]) -> str:
        spec = self.context.get("spec", {})
        prov = self.context.get("provenance", {})

        lines: list[str] = []
        add = lines.append

        if self.any_fail:
            add("> **⚠ ACCURACY CHECKLIST HAS FAILING ITEMS — this report is a diagnostic,")
            add("> not evidence. Promotion and publish paths refuse it until every item passes.**")
            add("")

        add(f"# {spec.get('id', 'EVALUATION')} — {spec.get('title', 'evaluation report')}")
        add("")
        add(f"*Generated {prov.get('generated_at')} by engine.report v{prov.get('generator_version')}.*")
        add("")
        if spec.get("hypothesis"):
            add(f"**Hypothesis:** {str(spec['hypothesis']).strip()}")
            add("")

        wanted = SECTIONS_BY_KIND.get(str(self.context.get("kind", "evaluation")),
                                      SECTIONS_BY_KIND["evaluation"])
        renderers = (
            ("verdict", lambda: self._render_verdict(lines)),
            ("headline", lambda: self._render_headline(lines, figures)),
            ("funnel", lambda: self._render_funnel(lines)),
            ("equity", lambda: self._render_equity(lines, figures)),
            ("by_year", lambda: self._render_by_year(lines, figures)),
            ("mc", lambda: self._render_mc(lines, figures)),
            ("stress", lambda: self._render_stress(lines, figures)),
            ("calibration", lambda: self._render_calibration(lines, figures)),
            ("checklist", lambda: self._render_checklist(lines)),
            ("provenance", lambda: self._render_provenance(lines)),
            ("extras", lambda: self._render_extra_sections(lines)),
            ("appendix", lambda: self._render_appendix(lines)),
            ("glossary", lambda: self._render_glossary(lines)),
        )
        # A section that does not apply is SKIPPED, not rendered full of "n/a":
        # a calibration report has no Monte Carlo and a data audit has no
        # equity curve. Numbering stays fixed either way, so §6 is the
        # calibration section in every report that has one.
        for name, render in renderers:
            if name in wanted:
                render()
        return "\n".join(lines) + "\n"
