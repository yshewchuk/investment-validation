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

Fixed section order (consumers rely on it): Headline table → Equity/DD →
By-year → Monte Carlo → Stress grid → Calibration → Accuracy checklist →
Provenance → Appendix.

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

__all__ = ["GENERATOR_VERSION", "ChecklistItem", "accuracy_checklist",
           "build_provenance", "Report"]

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

    quota_state = None
    if paths.QUOTA_LOG.exists():
        lines = paths.QUOTA_LOG.read_text().splitlines()
        quota_state = lines[-1] if len(lines) > 1 else None

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
    if audit and audit.get("fit_years_seen"):
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
                                   f"{len(sweep)} alphas swept; breakeven alpha={be}"))
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
# figures
# --------------------------------------------------------------------------


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def fig_equity(equity: pd.Series, path: Path, title: str) -> Path:
    plt = _matplotlib()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(equity.index, equity.values, lw=1.2, color="tab:blue")
    ax1.set_ylabel("equity (× start)")
    ax1.set_title(title)
    peak = equity.cummax()
    ax2.fill_between(equity.index, (equity / peak - 1.0).values, 0.0, color="tab:red", alpha=0.6)
    ax2.set_ylabel("drawdown")
    fig.text(0.01, 0.005,
             "Falsified if: a re-run with the same seed and trades does not reproduce this curve.",
             fontsize=7, color="gray")
    return _save(fig, path)


def fig_by_year(by_year: Mapping[str, Mapping[str, float]], path: Path, title: str) -> Path:
    plt = _matplotlib()
    years = sorted(by_year)
    means = [by_year[y]["mean"] for y in years]
    colors = ["tab:green" if (m is not None and m >= 0) else "tab:red" for m in means]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(years, means, color=colors)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("mean return / trade")
    ax.set_title(title)
    fig.text(0.01, 0.005,
             "Falsified if: the edge concentrates in so few years that removing one flips the sign.",
             fontsize=7, color="gray")
    return _save(fig, path)


def fig_mc_fan(mc: Mapping[str, Any], path: Path, title: str) -> Path:
    plt = _matplotlib()
    frac_stats = mc.get("by_fraction", {})
    fig, ax = plt.subplots(figsize=(7, 4))
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
    ax.set_xlabel("sizing fraction per trade")
    ax.set_ylabel("terminal equity (× start)")
    ax.set_title(title)
    ax2 = ax.twinx()
    ax2.plot(fractions, ploss, "s--", color="gray", label="P(loss)")
    ax2.set_ylabel("P(final loss)")
    ax2.set_ylim(0, 1)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    fig.text(0.01, 0.005,
             "Falsified if: live season equity leaves the p05-p95 band (the MC is then mis-specified).",
             fontsize=7, color="gray")
    return _save(fig, path)


def fig_alpha_curve(sweep: Mapping[str, Mapping[str, float]], breakeven: float | None,
                    path: Path, title: str) -> Path:
    plt = _matplotlib()
    points = sorted((float(a), s["mean"]) for a, s in sweep.items() if np.isfinite(s.get("mean", np.nan)))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    if points:
        xs, ys = zip(*points)
        ax.plot(xs, ys, "o-", color="tab:blue")
        ax.axhline(0.0, color="k", lw=0.8)
        if breakeven is not None:
            ax.axvline(breakeven, color="tab:red", lw=1.0, ls="--",
                       label=f"breakeven α={breakeven:.2f}")
            ax.legend(fontsize=8)
    ax.set_xlabel("fill alpha (0 = worst, 1 = best)")
    ax.set_ylabel("mean return / trade")
    ax.set_title(title)
    fig.text(0.01, 0.005,
             "Falsified if: measured live fill quality (Phase 5 alpha-hat) lands left of breakeven.",
             fontsize=7, color="gray")
    return _save(fig, path)


def fig_stress_grid(regimes: Mapping[str, Mapping[str, float]], path: Path, title: str) -> Path:
    plt = _matplotlib()
    names = list(regimes)
    means = [regimes[n].get("mean", np.nan) for n in names]
    counts = [regimes[n].get("n", 0) for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.0))
    data = np.array([[m if np.isfinite(m) else 0.0 for m in means]])
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn",
                   vmin=-max(abs(np.nanmin(means)), abs(np.nanmax(means)), 1e-9),
                   vmax=max(abs(np.nanmin(means)), abs(np.nanmax(means)), 1e-9))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\nn={c}" for n, c in zip(names, counts)], fontsize=8)
    ax.set_yticks([])
    for i, m in enumerate(means):
        ax.text(i, 0, f"{m:+.1%}" if np.isfinite(m) else "—", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="mean return")
    fig.text(0.01, 0.005,
             "Falsified if: a crisis regime shows losses the sizing rules do not survive.",
             fontsize=7, color="gray")
    return _save(fig, path)


def fig_reliability(calibration: Mapping[str, Any], path: Path, title: str) -> Path:
    """Reliability curve: predicted win rate vs realized, per decile.

    ``calibration`` needs ``deciles``: a list of rows with ``predicted`` and
    ``realized`` (and optionally ``n``). The diagonal is perfect calibration.
    """
    plt = _matplotlib()
    deciles = calibration.get("deciles") or []
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    if deciles:
        pred = [float(d.get("predicted", np.nan)) for d in deciles]
        real = [float(d.get("realized", np.nan)) for d in deciles]
        counts = [float(d.get("n", 1)) for d in deciles]
        ax.scatter(pred, real, s=[max(12, 4 * np.sqrt(c)) for c in counts],
                   color="tab:blue", label="model")
    ax.set_xlabel("predicted win rate")
    ax.set_ylabel("realized win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.text(0.01, 0.005,
             "Falsified if: points stay off the diagonal as ledger events accrue "
             "(then the shipped win rate is not the win rate).",
             fontsize=7, color="gray")
    return _save(fig, path)


def fig_mc_fan_paths(bands: Mapping[str, Any], path: Path, title: str) -> Path:
    """MC fan chart: percentile bands of the equity paths over the trade index.

    ``bands`` needs ``p05``, ``p50``, ``p95`` arrays aligned on the trade index
    — the true fan over the equity path, as distinct from the sizing curve
    (terminal percentiles vs fraction).
    """
    plt = _matplotlib()
    idx = np.arange(len(bands.get("p50", [])))
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(idx):
        ax.fill_between(idx, bands["p05"], bands["p95"], color="tab:blue", alpha=0.2,
                        label="p05-p95")
        ax.plot(idx, bands["p50"], color="tab:blue", lw=1.2, label="p50")
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("trade sequence index (chronological)")
    ax.set_ylabel("equity (× start)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.text(0.01, 0.005,
             "Falsified if: the realized forward-test equity path leaves the p05-p95 band.",
             fontsize=7, color="gray")
    return _save(fig, path)


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------


def _fmt(x: Any, pct: bool = False, nd: int = 4) -> str:
    if x is None:
        return "—"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return "—"
    return f"{xf:+.{nd}f}{'%' if pct else ''}" if pct else f"{xf:.{nd}f}"


class Report:
    """A standard report: REPORT.md + figures/, rendered from a context dict.

    Use :meth:`from_eval` for evaluation results; the plain constructor takes
    an already-assembled context for other phases (calibration, forward-test
    reviews) that emit through the same format.
    """

    def __init__(self, context: Mapping[str, Any]):
        self.context = dict(context)

    # -- assembly -----------------------------------------------------------

    @classmethod
    def from_eval(cls, result, input_files: Sequence[Path | str] = ()) -> "Report":
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

        return cls({
            "kind": "evaluation",
            "spec": spec,
            "results": results,
            "headline": headline,
            "backtest": backtest,
            "checklist": checklist,
            "provenance": provenance,
            "survivorship_note": SURVIVORSHIP_NOTE,
        })

    # -- rendering -----------------------------------------------------------

    @property
    def any_fail(self) -> bool:
        return any(item.status == "FAIL" for item in self.context.get("checklist", []))

    def write(self, out_dir: Path | str) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "figures").mkdir(exist_ok=True)

        figures = self._render_figures(out_dir / "figures")
        if self.context.get("kind") == "promotion":
            md = self._render_promotion_markdown(figures)
        else:
            md = self._render_markdown(figures)
        path = out_dir / "REPORT.md"
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
            figures["equity"] = fig_equity(eq, fig_dir / "equity_drawdown.png",
                                           "Equity curve (5% sizing) and drawdown")

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
            label = "MC P(loss)@5%" if key == "p_loss_5" else key
            add(f"| {label} | {_fmt(c.get(key))} | {_fmt(h.get(key))} |")
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

    def _render_markdown(self, figures: Mapping[str, Path]) -> str:
        spec = self.context.get("spec", {})
        results = self.context.get("results", {})
        headline = self.context.get("headline", {})
        backtest = self.context.get("backtest", {})
        prov = self.context.get("provenance", {})
        checklist: list[ChecklistItem] = self.context.get("checklist", [])

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

        # 1. Headline table -------------------------------------------------
        add("## 1. Headline (walk-forward OOS, worst/mid/best fills)")
        add("")
        sweep = headline.get("alpha_sweep") or backtest.get("alpha_sweep") or {}
        add("| fill | alpha | n | mean/trade | win rate |")
        add("|---|---|---|---|---|")
        for label, a in (("worst", "0.00"), ("mid", "0.50"), ("best", "1.00")):
            s = sweep.get(a)
            if s:
                add(f"| {label} | {a} | {s['n']} | {_fmt(s['mean'])} | {_fmt(s['win_rate'])} |")
        be = headline.get("breakeven_alpha")
        add("")
        add(f"**Breakeven alpha:** {_fmt(be)} — the margin of safety on the mid-fill assumption."
            if be is not None else
            "**Breakeven alpha:** none in [0, 1] — the alpha sweep never crosses zero.")
        base = headline.get("base_unselected") or {}
        if base:
            add("")
            add(f"Anti-selection guard — unselected universe: n={base.get('n')}, "
                f"mean {_fmt(base.get('mean'))}, win {_fmt(base.get('win_rate'))}.")
        keys = ("mean", "median", "std", "win_rate", "profit_factor", "sharpe_trade",
                "sharpe_equity", "sortino", "max_dd", "tail_ratio")
        add("")
        add("Canonical metrics (mid fills, OOS):")
        add("")
        add("| " + " | ".join(keys) + " |")
        add("|" + "---|" * len(keys))
        add("| " + " | ".join(_fmt(headline.get(k)) for k in keys) + " |")
        add("")
        if results.get("equity_mode") == "sequential":
            add("*Equity mode `sequential` (EXP-050 reference construction): overlap is ignored;*")
            add("*the `cashflow` construction is the default for new experiments.*")
            add("")

        # 2. Equity / drawdown ---------------------------------------------
        add("## 2. Equity curve & drawdown (5% sizing)")
        add("")
        if "equity" in figures:
            add(f"![equity](figures/{figures['equity'].name})")
        else:
            add("*No equity series recorded.*")
        eq5 = (headline.get("mc") or {}).get("0.05") or {}
        add("")
        add(f"Max drawdown: {_fmt(headline.get('max_dd'))}. Max concurrent positions: "
            f"{headline.get('max_concurrency', '—')}.")
        if "alpha" in figures:
            add("")
            add(f"Fill-quality degradation: ![alpha](figures/{figures['alpha'].name})")
        add("")

        # 3. By year ---------------------------------------------------------
        add("## 3. By year (OOS)")
        add("")
        by_year = headline.get("by_year") or {}
        add("| year | n | mean | win rate |")
        add("|---|---|---|---|")
        for y in sorted(by_year):
            s = by_year[y]
            add(f"| {y} | {s['n']} | {_fmt(s['mean'])} | {_fmt(s['win_rate'])} |")
        add("")
        if "by_year" in figures:
            add(f"![by year](figures/{figures['by_year'].name})")
            add("")

        # 4. Monte Carlo ------------------------------------------------------
        add("## 4. Monte Carlo (block bootstrap on the OOS sequence)")
        add("")
        mc = results.get("mc", {})
        add(f"block={mc.get('block')}, paths={mc.get('paths')}, seed={mc.get('seed')}, "
            f"n_trades={mc.get('n_trades')}.")
        add("")
        add("| sizing | P(loss) | terminal p05 | p50 | p95 | DD p50 | DD p95 |")
        add("|---|---|---|---|---|---|---|")
        for f in sorted(mc.get("by_fraction", {})):
            s = mc["by_fraction"][f]
            add(f"| {float(f):.0%} | {_fmt(s['p_loss'])} | {_fmt(s['terminal_p05'], nd=2)} | "
                f"{_fmt(s['terminal_p50'], nd=2)} | {_fmt(s['terminal_p95'], nd=2)} | "
                f"{_fmt(s['dd_p50'])} | {_fmt(s['dd_p95'])} |")
        add("")
        if "mc" in figures:
            add(f"![MC fan](figures/{figures['mc'].name})")
            add("")
        if "mc_fan" in figures:
            add(f"![MC equity fan](figures/{figures['mc_fan'].name})")
            add("")

        # 5. Stress grid ------------------------------------------------------
        add("## 5. Stress battery")
        add("")
        stress = results.get("stress", {})
        regimes = stress.get("regimes") or {}
        if regimes:
            add("| regime | n | mean | win rate |")
            add("|---|---|---|---|")
            for name, s in regimes.items():
                add(f"| {name} | {s['n']} | {_fmt(s.get('mean'))} | {_fmt(s.get('win_rate'))} |")
            if "stress" in figures:
                add("")
                add(f"![stress grid](figures/{figures['stress'].name})")
            add("")
        iv = stress.get("iv_regime") or {}
        if iv.get("split_by"):
            hi, lo = iv.get("high", {}), iv.get("low", {})
            add(f"IV-regime split ({iv['split_by']}): high-vol n={hi.get('n')} mean {_fmt(hi.get('mean'))} "
                f"vs low-vol n={lo.get('n')} mean {_fmt(lo.get('mean'))}.")
            add("")
        tail = stress.get("tail_injection") or {}
        if tail.get("available") is False:
            flag = "**REQUIRED and missing**" if tail.get("required") else "N/A"
            add(f"Tail injection: {flag} — {tail.get('note', '')}")
        else:
            mc_t = tail.get("mc", {}).get("0.05", {})
            add(f"Tail injection (worst 1% moves doubled): shocked worst trade "
                f"{_fmt(tail.get('shocked_worst_trade'))} (base {_fmt(tail.get('base_worst_trade'))}), "
                f"MC P(loss)@5% {_fmt(mc_t.get('p_loss'))}.")
        add("")
        slip = stress.get("slippage") or {}
        if slip.get("available"):
            for shift, s in slip.get("shifts", {}).items():
                add(f"Slippage {shift}: coverage {_fmt(s.get('coverage'))}, mean {_fmt(s.get('mean'))} "
                    f"(Δ {_fmt(s.get('delta_mean'))}).")
        else:
            add(f"Slippage days: N/A — {slip.get('note', 'no repricer')}.")
        stale = stress.get("stale_dates") or {}
        if stale.get("available"):
            add(f"Stale dates (1% mis-dated): Δmean {_fmt(stale.get('delta_mean'))} on "
                f"{stale.get('n_misdated')} events.")
        else:
            add(f"Stale dates: N/A — {stale.get('note', 'no repricer')}.")
        add("")

        # 6. Calibration -------------------------------------------------------
        add("## 6. Calibration")
        add("")
        cal = self.context.get("calibration")
        if cal:
            add(json.dumps(cal, indent=1, default=str))
            if "reliability" in figures:
                add("")
                add(f"![reliability](figures/{figures['reliability'].name})")
        else:
            add("No calibration block for this evaluation (predicted-vs-realized win rate is")
            add("reported by the Phase 1 calibration reports once ledger events accrue).")
        add("")

        # 7. Accuracy checklist -------------------------------------------------
        add("## 7. Accuracy-evidence checklist")
        add("")
        add("| check | status | evidence |")
        add("|---|---|---|")
        for item in checklist:
            add(item.row())
        add("")

        # 8. Provenance ----------------------------------------------------------
        add("## 8. Provenance")
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

        # 9. Appendix --------------------------------------------------------------
        add("## 9. Appendix")
        add("")
        add(self.context.get("survivorship_note", SURVIVORSHIP_NOTE))
        add("")
        add("Walk-forward diagnostics:")
        add("")
        add("| year | n_train | n_test | n_selected | ungated |")
        add("|---|---|---|---|---|")
        for d in (results.get("walk_forward") or {}).get("diagnostics", []):
            add(f"| {d['year']} | {d['n_train']} | {d['n_test']} | {d['n_selected']} | "
                f"{bool(d.get('ungated'))} |")
        add("")
        add(f"Backtest (unselected) alpha sweep, by year at mid: "
            f"{json.dumps(backtest.get('by_year', {}), default=str)}")
        add("")
        grid = self.context.get("grid_results")
        if grid:
            add("Grid / secondary results (NOT the headline — the primary spec is):")
            add("")
            add("```json")
            add(json.dumps(grid, indent=1, default=str))
            add("```")
            add("")
        return "\n".join(lines) + "\n"
