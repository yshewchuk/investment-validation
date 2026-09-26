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
import multiprocessing
import re
import time
from typing import Any

import numpy as np
import pandas as pd

from engine import paths

#: Mirrors ``engine.dashboard.publish.SECRET_PATTERNS``' literal ``/root/``
#: check: any exception text embedded in a reason string must never carry an
#: absolute local filesystem path, so a champion evidence block that fails
#: to rebuild (e.g. a FileNotFoundError inside a worker's private code
#: snapshot under /root/phase2-shadow-ops/code/<hash>/...) cannot leak one
#: into the published dashboard bundle. Defense in depth: the real fix is
#: giving the worker snapshot the files it needs (see
#: engine/v2/ops/fingerprints.py CODE_ASSET_FILES), but a reason string is
#: built from arbitrary exception text and must be scrubbed regardless of
#: why the rebuild failed.
_ABS_PATH_RE = re.compile(r"/root/\S*", re.IGNORECASE)


def _sanitize_reason(text: str) -> str:
    """Strip any absolute local filesystem path from ``text`` before it can
    reach a cached or rendered evidence reason string. A path under
    ``paths.ROOT`` is relativized (still informative); anything else under
    ``/root/`` (e.g. a worker's private code-snapshot root) is redacted
    generically, since it is host-local and never meaningful to a reader of
    the published dashboard."""
    root = str(paths.ROOT)
    text = text.replace(root, "<repo>")
    return _ABS_PATH_RE.sub("<local path>", text)


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

#: Raw points kept per input for the scatter. Enough to read the spread and the
#: shape, small enough that fifty inputs across four models stay a file a phone
#: will load.
SCATTER_POINTS = 300

#: Bumped whenever a fix changes what build_model_evidence produces or how a
#: cache hit is judged, in a way that must force every previously cached
#: entry to rebuild even though the champion artifacts it was keyed on have
#: not changed. Bumped 2026-09-26: a champion's cached "reason" string could
#: carry an absolute local filesystem path (see _sanitize_reason above) from
#: before that fix existed -- the fingerprint alone (artifact_sha256 per
#: champion) cannot see that difference, since the underlying artifact never
#: changed, only what this module does with a rebuild failure.
EVIDENCE_SCHEMA_VERSION = 2


def _release_free_pages() -> None:
    """Hand memory Python has finished with back to the OS.

    Same technique and rationale as ``engine.score._release_free_pages``
    (not imported from there to avoid pulling that module's own weight into
    this one): glibc keeps freed heap in its own arenas rather than
    returning it, so a step that allocates a gigabyte and drops it stays a
    gigabyte of RSS. This rebuild's champions run sequentially in one
    process — chooser, two gate variants, iv_crush, implied_t1, runup_move,
    size — each dropping a multi-hundred-MB-to-multi-GB training frame, so
    the fragmentation compounds across the loop rather than resetting per
    model. Best-effort; a platform without ``malloc_trim`` is slightly
    fatter, not broken.
    """
    import ctypes

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


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

    # The V-shape measure, and it is not optional. A signed input against a
    # magnitude outcome — mean_prior_move against |move| — runs high at both
    # ends and low in the middle, which is a real and strong relationship that
    # BOTH correlations above score at approximately zero. Ranking inputs by
    # correlation alone therefore buries exactly the ones that matter most.
    # Distance from the centre is the reading that sees it.
    centre = float(xs.median())
    out["magnitude_spearman"] = round(
        float((xs - centre).abs().corr(ys, method="spearman")), 4
    )
    out["centre"] = round(centre, 4)

    # The straight line a linear model would fit, for the scatter overlay. It
    # is drawn precisely so a reader can SEE when the line explains nothing
    # that the decile means clearly do.
    if float(xs.std()) > 0:
        slope = float(xs.cov(ys) / xs.var())
        out["ols"] = {
            "slope": round(slope, 6),
            "intercept": round(float(ys.mean() - slope * float(xs.mean())), 6),
        }
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
        means = [d["y_mean"] for d in out["deciles"]]
        first, last = means[0], means[-1]
        # End-to-end: what a monotone reading sees.
        out["decile_spread"] = round(float(last - first), 4)
        # Best-to-worst across ALL deciles: what a V or an inverted U actually
        # spans. When this dwarfs the end-to-end number the relationship is
        # real and non-monotone, and the UI badges it.
        out["decile_range"] = round(float(max(means) - min(means)), 4)
        out["monotone"] = bool(
            means == sorted(means) or means == sorted(means, reverse=True)
        )
        out["extreme_bin"] = int(
            out["deciles"][means.index(max(means))]["bin"]
        )

    out["scatter"] = _scatter_sample(xs, ys)
    return out


def _scatter_sample(xs: pd.Series, ys: pd.Series) -> list[list[float]]:
    """A bounded sample of the raw points, for the scatter.

    Deliberately a sample and not the whole set: plotting 115,000 points is
    overplotting that hides density rather than showing it, and the bundle has
    a size budget. The decile means carry the shape; these points carry the
    spread around it, which is the part a summary always flatters.
    """
    n = min(SCATTER_POINTS, len(xs))
    if n <= 0:
        return []
    idx = np.random.default_rng(SAMPLE_SEED).choice(len(xs), size=n, replace=False)
    sx, sy = xs.to_numpy()[idx], ys.to_numpy()[idx]
    return [[round(float(a), 4), round(float(b), 4)] for a, b in zip(sx, sy)]


def _replay_trades() -> pd.DataFrame:
    """``trades`` filtered to ``engine.replay`` provenance, one year at a time.

    ``store.read_table("trades")`` reads and concatenates every partition
    (~526k rows / 18 columns, ~1.9 GB) BEFORE the provenance filter ever
    drops a row, then the boolean-mask filter builds a second, near-full-size
    frame (~506k rows survive — provenance excludes only ~4%) while the first
    is still reachable through the coercion pandas does internally. Measured
    2026-09-15 on three real forced ``--force`` rebuilds, 1-1.5s PSS sampling:
    this was the single largest step in the whole rebuild, peaking the tree at
    4.1-4.99 GiB PSS in the FIRST 20-45 seconds — before any model's dataset
    build even starts — because the reassignment frees the old frame at the
    Python level but glibc does not hand freed arenas back to the OS before
    the next allocation, so PSS does not drop with it (confirmed with a
    standalone profile: the same two-frame read+filter step, isolated, showed
    the identical jump from ~1.9 GiB to ~4.1 GiB PSS). Streaming per year and
    filtering each partition before concatenating — the same shape
    :func:`_daily_subset` already uses for ``daily_market`` below — keeps only
    one year's raw partition and the growing filtered result live at once, so
    the unfiltered and filtered copies of the whole table never coexist.
    """
    from engine.data import store
    from engine.data.schemas import coerce, empty_frame

    kept = []
    for _, frame in store.iter_table("trades"):
        chunk = frame[frame["provenance"].astype(str) == "engine.replay"]
        if len(chunk):
            kept.append(chunk)
    if not kept:
        return empty_frame("trades")
    out = coerce(pd.concat(kept, ignore_index=True), "trades")
    _release_free_pages()
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
    out = pd.concat(kept, ignore_index=True)
    _release_free_pages()
    return out


def _dataset_for(role: str, strategy: str, *, panel, daily, trades, features=()):
    """Rebuild the rows a champion was trained on, and name its target.

    Each model learns from a different table, at a different scale, which is
    itself part of the answer to "why not one model": the size model sees every
    earnings event in the panel, while a gate only sees events whose chains
    exist to price a trade from — an order of magnitude fewer rows.

    ``features`` is the champion's REGISTERED feature list, and a gate needs
    it to pick its training module: the STR-THRU champion may be the
    forecast_analog variant, whose set extends the incumbent's with the
    Tier-4 forecast and analog columns. Rebuilding that champion through the
    incumbent's ``gate.build_dataset`` produces a frame missing every extended
    input, and the evidence table then reports each of them as "not present in
    the rebuilt training set" — the dashboard silently loses the new
    champion's parameters. Dispatch on the registered set, the same way
    ``engine.score`` serves whatever ``artifact.features`` names.
    """
    if role == "size":
        from engine.models.training import size_model

        return size_model.prepare(panel), size_model.TARGET, list(size_model.FEATURES)
    if role == "chooser":
        from engine.models.training import chooser

        data, target, feats = chooser.build_dataset()
        return data, target, feats
    if role == "gate":
        from engine.models.training import gate

        module = gate
        if features and set(features) != set(gate.FEATURES):
            from engine.models.training import gate_forecast_analog as gate_fa

            if set(features) == set(gate_fa.FEATURES):
                module = gate_fa
        rows = trades[trades["strategy"] == strategy]
        if rows.empty:
            return None, None, []
        years = sorted(pd.to_datetime(rows["entry_date"]).dt.year.unique().tolist())
        daily = _daily_subset(rows["ticker"].unique(), years=years)
        if module is not gate:
            # The forecast_analog rebuild constructs a Scorer-like context for
            # its analog join, and a Scorer's default context reads the ENTIRE
            # daily_market (~8.9M rows, ~6.9 GB peak) on top of everything
            # this builder already holds — two rebuilds were OOM-killed at
            # exactly that point. Bound the context to what the trades can
            # actually look up: their own tickers, their own years.
            #
            # This used to call FeatureContext.load(rows["ticker"].unique(),
            # years=years), which re-reads daily_market for the SAME
            # tickers/years `daily` above was just built from, filtered a
            # second time. Measured 2026-09-15 (1s PSS sampling, real forced
            # rebuild): that second read was the single largest transient
            # peak in this champion's whole evidence build, 5.0-5.4 GiB,
            # bigger than any other individual step including trades loading
            # and _daily_subset combined. `daily` already carries every
            # column FeatureContext.load's own restricted read would have
            # (it is read with no column filter, so it is a strict superset
            # of ORATS_FEATURES/DAILY_STATE_FIELDS) and the identical
            # ticker/year filter, so building the context directly from it
            # is the same data, not an approximation -- just without paying
            # for daily_market twice.
            from engine.calendar import trading_calendar
            from engine.features import FeatureContext

            context = FeatureContext(panel=panel, daily=daily, calendar=trading_calendar())
            data = module.build_dataset(rows, panel=panel, daily=daily, context=context)
        else:
            data = module.build_dataset(rows, panel=panel, daily=daily)
        return data, module.TARGET, list(module.FEATURES)
    if role == "iv_crush":
        from engine.models.training import iv_crush

        # Panel row joined to the realized crush, which is read from Tier 2
        # rather than the panel: the target lives on BOTH sides of the print and
        # Tier 3 by construction holds only the pre-print side. That is the leak
        # rule working, not a gap.
        return iv_crush.prepare(panel), iv_crush.TARGET, list(iv_crush.FEATURES)
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
    if role == "runup_move":
        from engine.models.training import runup_move
        from engine.models.training.train_all import _events_with_session

        events = _events_with_session()
        sampled = None
        if len(events) > MAX_EVENTS:
            sampled = {"events": MAX_EVENTS, "of": int(len(events)), "seed": SAMPLE_SEED}
            events = events.sample(MAX_EVENTS, random_state=SAMPLE_SEED)
        years = sorted(pd.to_datetime(events["event_date"]).dt.year.unique().tolist())
        daily = _daily_subset(events["ticker"].unique(), years=years)
        data = runup_move.build_dataset(events, panel=panel, daily=daily)
        if sampled is not None:
            data.attrs["sampled_events"] = sampled
        return data, runup_move.TARGET, list(runup_move.FEATURES)
    return None, None, []


def _isolated_entrypoint(func, args, kwargs, conn) -> None:
    """Child side of :func:`_run_isolated`: call ``func`` and send back
    ``(True, value)`` or ``(False, description)``, never letting an
    exception escape unreported."""
    try:
        value = func(*args, **kwargs)
        conn.send((True, value))
    except Exception as exc:  # the parent decides what an isolated failure means
        conn.send((False, f"{type(exc).__name__}: {exc}"[:300]))
    finally:
        conn.close()


def _run_isolated(func, *args, **kwargs):
    """Run ``func(*args, **kwargs)`` in a freshly spawned subprocess and
    return ``(ok, value_or_description)``.

    A long-lived process that builds and drops several multi-GiB frames in a
    row does not return that memory to the OS the way a process EXIT does --
    ``gc.collect()`` only frees Python's own references, and even
    ``_release_free_pages()``'s ``malloc_trim`` is best-effort against glibc
    arena growth that compounds across repeated large alloc/free cycles
    (measured 2026-09-15: five real forced rebuilds in one process peaked
    4.85-5.55 GiB PSS even after that fix, while each champion's build
    measured only ~3.5-3.6 GiB in isolation -- see
    ``engine/v2/ops/profiles.py``'s module docstring, v8 entry, for the
    full basis). Running each champion in its own subprocess makes "in
    isolation" the actual shape of the real rebuild: nothing from one
    champion's build can still be resident, freed-but-fragmented or
    otherwise, when the next one starts.

    ``spawn`` (not the platform default ``fork`` on Linux) so the child
    starts with an empty heap rather than a copy-on-write snapshot of
    whatever the parent already holds.

    ``func`` and every value in ``args``/``kwargs`` must be picklable
    (importable by reference, for ``func`` -- a module-level function, not a
    closure) and the return value must be small and picklable too: it
    crosses back over a pipe, not shared memory.
    """
    ctx = multiprocessing.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_isolated_entrypoint, args=(func, args, kwargs, send_conn))
    proc.start()
    send_conn.close()  # the parent only reads; close its copy of the write end
    try:
        outcome = recv_conn.recv()
    except EOFError:
        outcome = (False, f"subprocess exited {proc.exitcode} without a result")
    proc.join()
    return outcome


def _champion_block_impl(entry, registry) -> dict[str, Any]:
    """One champion's evidence block. Runs inside :func:`_run_isolated`'s
    subprocess -- loads its own panel and (only for a gate role, the only
    role that reads it) its own trades, fresh, rather than sharing whatever
    the caller already has loaded."""
    from engine.features import feature_note, load_panel

    panel = load_panel()
    trades = None
    if entry.role == "gate":
        # _replay_trades() returns every strategy's replayed trades across
        # every year -- 982.9 MiB measured 2026-09-15 -- but a single gate
        # only ever trains on its own strategy's rows (STR-THRU: 99.95 MiB
        # of that, a tenth). `_dataset_for`'s gate branch re-filters by
        # strategy too, so filtering here and releasing the rest immediately
        # (rather than holding the whole table resident for this champion's
        # entire build, which runs for minutes once analog matching starts)
        # does not change what gets trained on, only how much of the other
        # seven strategies' trades stays resident while it happens.
        trades = _replay_trades()
        trades = trades[trades["strategy"] == entry.strategy].reset_index(drop=True)
        _release_free_pages()

    try:
        data, target, _features = _dataset_for(
            entry.role, entry.strategy, panel=panel, daily=None, trades=trades,
            features=entry.features,
        )
    except Exception as exc:  # one model's dataset must not lose the others
        reason = _sanitize_reason(
            f"rebuilding the training set raised {type(exc).__name__}: {exc}"
        )[:300]
        return {
            "id": entry.id, "role": entry.role, "strategy": entry.strategy,
            "target": entry.target, "available": False,
            "reason": reason,
        }

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
        return block

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

    # Strongest marginal relationship first, where "strongest" takes the
    # LARGER of the monotone and the magnitude readings. Sorting on
    # correlation alone put mean_prior_move — an 8.35 → 4.60 → 7.82 V
    # against |move| — near the bottom of the size model's table on a
    # Spearman of +0.013.
    block["inputs"].sort(
        key=lambda s: max(
            abs(s.get("spearman") or 0.0), abs(s.get("magnitude_spearman") or 0.0)
        ),
        reverse=True,
    )
    return block


def _champion_block(entry, registry) -> dict[str, Any]:
    """``_champion_block_impl`` run in its own subprocess (see
    :func:`_run_isolated`) so this champion's peak memory cannot compound
    with any other's."""
    ok, result = _run_isolated(_champion_block_impl, entry, registry)
    if ok:
        return result
    return {
        "id": entry.id, "role": entry.role, "strategy": entry.strategy,
        "target": entry.target, "available": False,
        "reason": _sanitize_reason(f"rebuilding the training set raised {result}"),
    }


def build_model_evidence(*, registry=None, force: bool = False) -> dict:
    """Per-champion input evidence, cached by artifact hash.

    Rebuilt only when a champion changes: the numbers describe a model's
    training set, which does not move on a nightly cadence, and the implied_t1
    dataset alone is over half a million rows.

    Each champion's dataset is built in its own subprocess (see
    :func:`_champion_block`) -- this function itself never loads a panel,
    trades table or daily_market slice, so its own memory footprint is just
    the registry plus whatever the champions' returned blocks weigh (small:
    scalars, decile tables, a bounded scatter sample -- never a DataFrame).
    """
    from engine.models.registry import load_registry

    registry = registry or load_registry()
    cached = load_model_evidence() or {}
    started = time.time()

    champions = []
    for entry in registry.entries:
        if entry.champion:
            champions.append(entry)

    fingerprint = {e.id: (e.artifact_sha256 or "") for e in champions}
    if (
        not force
        and cached.get("fingerprint") == fingerprint
        and cached.get("schema_version") == EVIDENCE_SCHEMA_VERSION
    ):
        return cached

    models: dict[str, Any] = {}
    for entry in champions:
        models[entry.id] = _champion_block(entry, registry)

    out = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "fingerprint": fingerprint,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
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
        _, artifact = registry.load_champion(
            entry.role,
            entry.strategy,
            decision_offset=entry.decision_offset,
            verify=False,
        )
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
