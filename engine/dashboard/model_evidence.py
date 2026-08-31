"""What each model input actually does to the output.

    python3 -m engine.dashboard.model_evidence

A feature list with one-line notes says what an input *is*. It does not say
whether the model had any reason to look at it. This module rebuilds each
champion's own training set and measures, per input:

* **Correlation with the target**, Pearson and Spearman side by side. The pair
  matters: a monotone but curved relationship shows up in Spearman and hides in
  Pearson, and reading only the linear one would call a real signal noise.
* **A decile table** — the feature cut into ten buckets, with the mean target in
  each. This is the shape, and it is the part a correlation cannot show: whether
  the relationship is monotone, flat in the middle, or driven entirely by one
  tail.
* **Coverage** — how much of the training set even had the value.

Read it as description, not attribution. These are marginal relationships in
the training data: a feature can correlate strongly and contribute nothing once
the others are in (collinearity), or correlate near zero and matter through an
interaction a tree model found. It answers "what does this input look like
against the outcome", which is the question a reader actually has, and it is
honest about not being a causal or even a model-attribution claim.

Cached under Tier 3 keyed by the artifact hash, because it changes only when a
champion changes — not nightly.
"""
from __future__ import annotations

import gc
import json
import time
from typing import Any

import numpy as np
import pandas as pd

from engine import paths

__all__ = ["build_model_evidence", "evidence_path", "load_model_evidence", "DECILES"]

#: Buckets per feature. Ten is enough to show a shape and few enough to read.
DECILES = 10

#: Below this many usable rows a feature's statistics are reported as unusable
#: rather than as a number nobody should act on.
MIN_ROWS = 200

#: Rows sampled per model before the statistics are computed. A correlation and
#: a decile shape are settled long before half a million rows — the implied_t1
#: set is 577k — and holding every training set at once is what got this
#: OOM-killed. Sampling is recorded in the output, never silent.
MAX_ROWS = 150_000

#: Events sampled before the implied_t1 dataset is BUILT (nine rows each).
MAX_EVENTS = 20_000

#: Deterministic sample, so two runs on the same store agree.
SAMPLE_SEED = 7


def evidence_path() -> "paths.Path":
    return paths.FEATURES / "model_evidence.json"


def load_model_evidence() -> dict | None:
    path = evidence_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _feature_stats(x: pd.Series, y: pd.Series) -> dict:
    """Correlations, coverage and the decile shape for one input."""
    x = pd.to_numeric(x, errors="coerce")
    ok = x.notna() & y.notna()
    n = int(ok.sum())
    out: dict[str, Any] = {"n": n, "coverage": round(float(ok.mean()), 4)}
    if n < MIN_ROWS:
        out["usable"] = False
        out["reason"] = f"only {n} rows with both the input and the outcome"
        return out

    xs, ys = x[ok], y[ok]
    out["usable"] = True
    out["pearson"] = round(float(xs.corr(ys)), 4)
    out["spearman"] = round(float(xs.corr(ys, method="spearman")), 4)
    out["mean"] = round(float(xs.mean()), 4)
    out["std"] = round(float(xs.std()), 4)
    out["p10"] = round(float(xs.quantile(0.10)), 4)
    out["p50"] = round(float(xs.quantile(0.50)), 4)
    out["p90"] = round(float(xs.quantile(0.90)), 4)

    # The shape. `duplicates="drop"` because a feature like n_prior or
    # signed_streak is lumpy and cannot always be cut into ten distinct bins —
    # fewer honest buckets beat ten fabricated ones.
    try:
        bins = pd.qcut(xs, DECILES, labels=False, duplicates="drop")
    except (ValueError, IndexError):
        return out
    grouped = pd.DataFrame({"bin": bins, "x": xs, "y": ys}).groupby("bin", observed=True)
    out["deciles"] = [
        {
            "bin": int(b) + 1,
            "n": int(len(g)),
            "x_lo": round(float(g["x"].min()), 4),
            "x_hi": round(float(g["x"].max()), 4),
            "y_mean": round(float(g["y"].mean()), 4),
        }
        for b, g in grouped
    ]
    if out["deciles"]:
        first, last = out["deciles"][0]["y_mean"], out["deciles"][-1]["y_mean"]
        out["decile_spread"] = round(float(last - first), 4)
    return out


def _daily_subset(tickers, years=None) -> pd.DataFrame:
    """``daily_market`` for a bounded set of tickers, one partition at a time.

    ``store.read_table("daily_market")`` is 8.9M rows and peaked at 6.9 GB on a
    7 GB box — it OOM-killed this build twice before the first statistic was
    computed. Streaming per year and keeping only the tickers a model actually
    needs is the difference between "does not run" and a few hundred MB.
    """
    from engine.data import store

    wanted = set(tickers)
    kept = []
    for _, frame in store.iter_table("daily_market", years=years):
        chunk = frame[frame["ticker"].isin(wanted)]
        if len(chunk):
            kept.append(chunk)
    if not kept:
        return pd.DataFrame()
    return pd.concat(kept, ignore_index=True)


def _dataset_for(role: str, strategy: str, *, panel, daily, trades):
    """Rebuild the rows a champion was trained on, and name its target.

    Each model learns from a different table, at a different scale, which is
    itself part of the answer to "why not one model": the size model sees every
    earnings event in the panel, while a gate only sees events whose chains
    exist to price a trade from — an order of magnitude fewer rows.
    """
    if role == "size":
        from engine.models.training import size_model

        return size_model.prepare(panel), size_model.TARGET, list(size_model.FEATURES)
    if role == "gate":
        from engine.models.training import gate

        rows = trades[trades["strategy"] == strategy]
        if rows.empty:
            return None, None, []
        years = sorted(pd.to_datetime(rows["entry_date"]).dt.year.unique().tolist())
        daily = _daily_subset(rows["ticker"].unique(), years=years)
        data = gate.build_dataset(rows, panel=panel, daily=daily)
        return data, gate.TARGET, list(gate.FEATURES)
    if role == "implied_t1":
        from engine.models.training import implied_t1
        from engine.models.training.train_all import _events_with_session

        # One row per (event, decision day) across nine decision days: 577k rows
        # over the full calendar, built in a Python loop. A correlation and a
        # decile shape are settled long before that, so the EVENTS are sampled
        # first — sampling the output would still pay for the whole build. The
        # sample is recorded in the output, never silent.
        events = _events_with_session()
        sampled = None
        if len(events) > MAX_EVENTS:
            sampled = {"events": MAX_EVENTS, "of": int(len(events)), "seed": SAMPLE_SEED}
            events = events.sample(MAX_EVENTS, random_state=SAMPLE_SEED)
        years = sorted(pd.to_datetime(events["event_date"]).dt.year.unique().tolist())
        daily = _daily_subset(events["ticker"].unique(), years=years)
        data = implied_t1.build_dataset(events, panel=panel, daily=daily)
        if sampled is not None:
            data.attrs["sampled_events"] = sampled
        return data, implied_t1.TARGET, list(implied_t1.FEATURES)
    return None, None, []


def build_model_evidence(*, registry=None, force: bool = False) -> dict:
    """Per-champion input evidence, cached by artifact hash.

    Rebuilt only when a champion changes: the numbers describe a model's
    training set, which does not move on a nightly cadence, and the implied_t1
    dataset alone is over half a million rows.
    """
    from engine.data import store
    from engine.features import feature_note, load_panel
    from engine.models.registry import load_registry

    registry = registry or load_registry()
    cached = load_model_evidence() or {}
    started = time.time()

    champions = []
    for entry in registry.entries:
        if entry.champion:
            champions.append(entry)

    fingerprint = {e.id: (e.artifact_sha256 or "") for e in champions}
    if not force and cached.get("fingerprint") == fingerprint:
        return cached

    panel = load_panel()
    daily = None  # loaded per model, filtered to the tickers it needs
    trades = store.read_table("trades")
    trades = trades[trades["provenance"].astype(str) == "engine.replay"]

    models: dict[str, Any] = {}
    for entry in champions:
        try:
            data, target, features = _dataset_for(
                entry.role, entry.strategy, panel=panel, daily=daily, trades=trades
            )
        except Exception as exc:  # one model's dataset must not lose the others
            models[entry.id] = {
                "id": entry.id, "role": entry.role, "strategy": entry.strategy,
                "target": entry.target, "available": False,
                "reason": f"rebuilding the training set raised {type(exc).__name__}: {exc}"[:300],
            }
            continue
        block: dict[str, Any] = {
            "id": entry.id,
            "role": entry.role,
            "strategy": entry.strategy,
            "target": entry.target,
            "kind": _model_kind(entry, registry),
        }
        if data is None or not len(data) or target not in data.columns:
            block["available"] = False
            block["reason"] = "the training set could not be rebuilt from the store"
            models[entry.id] = block
            continue

        block["n_rows"] = int(len(data))
        if data.attrs.get("sampled_events"):
            block["sampled"] = data.attrs["sampled_events"]
        if len(data) > MAX_ROWS:
            data = data.sample(MAX_ROWS, random_state=SAMPLE_SEED)
            block["sampled"] = {"rows": MAX_ROWS, "of": block["n_rows"], "seed": SAMPLE_SEED}

        y = pd.to_numeric(data[target], errors="coerce")
        block["available"] = True
        block["target_mean"] = round(float(y.mean()), 4)
        block["target_std"] = round(float(y.std()), 4)
        block["inputs"] = []
        for name in entry.features:
            if name not in data.columns:
                block["inputs"].append(
                    {"name": name, "note": feature_note(name), "usable": False,
                     "reason": "not present in the rebuilt training set"}
                )
                continue
            stats = _feature_stats(data[name], y)
            stats["name"] = name
            stats["note"] = feature_note(name)
            block["inputs"].append(stats)

        # Strongest marginal relationships first — the reader's way in. The
        # model's own feature order is preserved in `strategies.json`.
        block["inputs"].sort(
            key=lambda s: abs(s.get("spearman") or 0.0), reverse=True
        )
        models[entry.id] = block
        # The rebuilt sets are large and are not needed once measured.
        del data, y
        gc.collect()

    out = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "fingerprint": fingerprint,
        "deciles": DECILES,
        "elapsed_s": round(time.time() - started, 1),
        "models": models,
        "caveat": (
            "Marginal relationships in each model's own training set, not "
            "attributions. A feature can correlate strongly and add nothing once "
            "the others are present, or correlate near zero and matter through an "
            "interaction. Pearson is the linear reading, Spearman the monotone "
            "one; the decile table is the shape neither number can show."
        ),
    }
    path = paths.assert_writable(evidence_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=str))
    return out


def _model_kind(entry, registry) -> dict:
    """What the thing actually is — a blend, a tree ensemble, a linear fit."""
    try:
        _, artifact = registry.load_champion(entry.role, entry.strategy, verify=False)
    except Exception:
        return {"type": "unknown"}
    model = artifact.model
    name = type(model).__name__
    kind = {"type": name, "params": {}}
    inner = model
    if hasattr(model, "steps"):
        kind["pipeline"] = [type(step).__name__ for _, step in model.steps]
        inner = model.steps[-1][1]
    for attr in ("n_estimators", "max_depth", "learning_rate", "hidden_layer_sizes"):
        if hasattr(inner, attr):
            kind["params"][attr] = str(getattr(inner, attr))
    if artifact.params:
        kind["params"].update({k: str(v) for k, v in artifact.params.items()})
    kind["residuals_n"] = int(len(artifact.residuals))
    return kind


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = parser.parse_args(argv)

    out = build_model_evidence(force=args.force)
    for model_id, block in sorted(out["models"].items()):
        if not block.get("available"):
            print(f"{model_id:26s} unavailable — {block.get('reason')}")
            continue
        top = block["inputs"][0]
        print(
            f"{model_id:26s} {block['n_rows']:>8,} rows  target={block['target']:9s} "
            f"strongest: {top['name']} (spearman {top.get('spearman')})"
        )
    print(f"\nwrote {evidence_path()} in {out['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
