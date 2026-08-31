"""``score_calendar()`` output → the self-contained dashboard bundle.

The renderer is the ONE place a ScoreResult becomes display data. Three rules
from the guide are enforced here rather than documented:

* **The UI never computes.** Every number shown — including derived display
  values like the per-strategy rank, the entry premium as a fraction of spot,
  and the model-fair premium comparison — is computed here, so the nightly
  self-check covers it. The client formats; it does not derive.
* **One rendering path.** The local server, the phone snapshot and the engine
  all read what this module wrote. ``.json`` files carry the contract;
  ``.js`` wrappers are generated from the same payload so the bundle also
  opens from ``file://`` (where ``fetch()`` is blocked) without a second code
  path that could drift.
* **No secrets, no internal paths.** :func:`engine.dashboard.publish.secret_scan`
  re-reads the bundle before any publish; the renderer itself never embeds a
  filesystem path or credential, only data the store already publishes.

Bundle layout::

    {out}/index.html  {out}/assets/app.js  {out}/assets/app.css
    {out}/data/board.json    {out}/data/board.js
    {out}/data/meta.json     {out}/data/meta.js
    {out}/data/health.json   {out}/data/health.js
    {out}/data/flags.json    {out}/data/flags.js
    {out}/data/strategies.json  {out}/data/strategies.js
    {out}/data/tickers/{T}.json   {out}/data/tickers/{T}.js
"""
from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from engine import paths
from engine.jsonio import json_safe

__all__ = [
    "BOARD_MAX_BYTES",
    "RENDER_VERSION",
    "compact_row",
    "row_digest",
    "render_bundle",
    "write_single_file",
    "build_meta",
    "build_strategies",
    "build_health",
    "freshness_summary",
    "quota_state",
    "size_model_mae_from_ledger",
]

#: Bumped when the bundle's on-disk shape changes, so a stale published bundle
#: and a new client (or the reverse) are detectable from meta.json alone.
RENDER_VERSION = 1

#: The guide's mobile budget: board.json should stay under ~1 MB. Above this
#: the renderer flags the bundle rather than failing — the board still works,
#: it just loads slowly on a phone.
BOARD_MAX_BYTES = 1_200_000

#: Fields a board row carries. Everything else stays in the per-ticker file,
#: which loads lazily — the board is the only file every visit pays for.
_BOARD_FIELDS = (
    "row_id", "ticker", "strategy", "event_date", "session",
    "entry_date", "exit_date", "strike", "strike_offset", "expiry",
    "dte_entry", "spot", "entry_cost", "entry_cost_pct",
    "model_fair_pct", "premium_vs_fair",
    "exp_pnl_model", "win_model", "model_p10", "model_p90",
    "exp_pnl_analog", "win_analog", "ci_low", "ci_high",
    "n_analogs", "analog_widened",
    "gate_score", "gate_threshold", "gate_pass",
    "extrapolated", "flags", "model_versions",
    "driver_name", "driver_prediction",
    "chain_last_obs", "chain_age_days",
    "scored", "rank", "fill", "detail", "digest",
)

#: Per-ticker evidence: at most this many historical prints and analog trades,
#: newest first. The explorer is a view, not a data export.
MAX_HISTORY_EVENTS = 40
MAX_ANALOG_TRADES = 50


# --------------------------------------------------------------------------
# json hygiene
# --------------------------------------------------------------------------


#: The bundle's display precision. Six places is more than any cell renders and
#: still small enough to keep board.json inside the mobile budget.
BUNDLE_PRECISION = 6


def _clean(value: Any) -> Any:
    """Make a value strict-JSON safe at the bundle's display precision.

    ``NaN`` is valid JavaScript but not valid JSON, and a strict parser on the
    other end of a published bundle is entitled to reject it — so missing
    numbers travel as ``null``.
    """
    return json_safe(value, round_to=BUNDLE_PRECISION)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _js_name(stem: str) -> str:
    return {
        "board": "BOARD", "meta": "META", "health": "HEALTH", "flags": "FLAGS",
        "strategies": "STRATEGIES", "models": "MODELS",
    }.get(stem)


def _write_pair(directory: Path, stem: str, payload: Any, *, js_expr: str | None = None) -> Path:
    """Write ``stem.json`` plus its ``stem.js`` wrapper from the SAME payload.

    The wrapper exists because ``fetch()`` is blocked on ``file://`` pages —
    the offline requirement — while the JSON file is what the self-check, the
    API and any future consumer read. Generating both from one payload is the
    only way they cannot disagree.
    """
    payload = _clean(payload)
    text = json.dumps(payload, sort_keys=True, default=str)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{stem}.json"
    json_path.write_text(text + "\n")
    expr = js_expr if js_expr is not None else f"window.{_js_name(stem) or stem.upper()} = "
    (directory / f"{stem}.js").write_text(expr + text + ";\n")
    return json_path


def _digest_safe(value: Any) -> Any:
    """Canonicalize one field for :func:`row_digest`.

    Unlike :func:`_clean` this does NOT round: the digest must stay sensitive
    to every significant figure the engine produced. It only erases the
    distinctions a DataFrame round-trip invents, because those are not
    differences in the score, and treating them as such would fail the
    self-check every night on any board where one row has a strike and another
    does not:

    * ``None`` becomes ``nan`` in a column another row gave a number to;
    * ``int`` becomes ``numpy.int64``, and then ``float`` (2 → 2.0) as soon as
      one row in the column is missing — pandas has no nullable-int default.

    Both sides of the comparison pass through here, so the normalization is
    consistent rather than lossy in one direction: what it gives up is the
    ability to distinguish ``2`` from ``2.0``, which the board cannot display
    differently anyway.
    """
    out = json_safe(value)
    return _integral_floats(out)


def _integral_floats(value: Any) -> Any:
    """``2.0`` → ``2``, recursively. The digest's own rule, and only its own."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _integral_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_integral_floats(v) for v in value]
    return value


def row_digest(record: Mapping[str, Any]) -> str:
    """The canonical digest of one scored row.

    ``score_calendar`` appends ``strike_offset`` to ``ScoreResult.as_dict()``;
    the digest is defined on the result alone, so the extra key is removed
    before hashing. Both sides of the self-check hash through THIS function —
    the stored row and the fresh :class:`~engine.score.ScoreResult` — so the
    comparison asks whether the engine still produces this score, not whether
    a value travelled through pandas on its way here.
    """
    import hashlib

    core = {
        str(k): _digest_safe(v) for k, v in record.items() if k != "strike_offset"
    }
    return hashlib.sha256(
        json.dumps(core, sort_keys=True, default=str).encode()
    ).hexdigest()


# --------------------------------------------------------------------------
# derived display values — computed HERE so the self-check covers them
# --------------------------------------------------------------------------


def _model_fair_pct(record: Mapping[str, Any]) -> float | None:
    """The payoff map's exit value at the predicted driver, as % of spot.

    ``exit_value / spot = intercept + slope * driver`` — the structure's fair
    premium per the model layer, which the board compares against the premium
    the chain actually quotes.
    """
    payoff = record.get("payoff") or {}
    driver = record.get("driver_prediction")
    intercept, slope = payoff.get("intercept"), payoff.get("slope")
    if driver is None or intercept is None or slope is None:
        return None
    fair = (float(intercept) + float(slope) * float(driver)) * 100.0
    return max(0.0, fair)


def compact_row(record: Mapping[str, Any], rank: int | None = None) -> dict:
    """One ``score_calendar`` row → the board's compact row.

    Keeps exactly :data:`_BOARD_FIELDS`; the full record lives in the
    per-ticker file. The digest is carried so the self-check can verify this
    row against a fresh engine re-score without shipping the whole record.
    """
    spot = record.get("spot")
    entry_cost = record.get("entry_cost")
    entry_cost_pct = (
        float(entry_cost) / float(spot) * 100.0
        if entry_cost is not None and spot
        else None
    )
    fair_pct = _model_fair_pct(record)
    premium_vs_fair = (
        entry_cost_pct / fair_pct
        if entry_cost_pct is not None and fair_pct not in (None, 0.0)
        else None
    )
    row = {
        "row_id": "|".join(
            [
                str(record.get("ticker")),
                str(record.get("strategy")),
                str(record.get("event_date")),
                "atm" if record.get("strike") is None else f"{float(record['strike']):.4f}",
            ]
        ),
        "entry_cost_pct": entry_cost_pct,
        "model_fair_pct": fair_pct,
        "premium_vs_fair": premium_vs_fair,
        "scored": record.get("exp_pnl_model") is not None
        or record.get("exp_pnl_analog") is not None,
        "rank": rank,
        "digest": row_digest(record),
    }
    for field in _BOARD_FIELDS:
        if field in record:
            row[field] = record[field]
    return {k: row[k] for k in _BOARD_FIELDS if k in row}


def _count_events(records: Sequence[Mapping[str, Any]]) -> int:
    """Distinct (ticker, event_date), ignoring rows that carry no date.

    A row without an event date is not a second event on that ticker — counting
    it as one double-counted every name on the board back when disabled
    strategies returned before resolving their event.
    """
    return len(
        {
            (r.get("ticker"), str(r.get("event_date")))
            for r in records
            if r.get("event_date") is not None and not pd.isna(r.get("event_date"))
        }
    )


def _rank_rows(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Per-strategy rank: gate passers first, then gate score, then model PnL.

    The rank is a display order, computed once here; the client sorts columns
    on request but never re-derives a ranking of its own.
    """
    ranks: dict[str, int] = {}
    by_strategy: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_strategy.setdefault(str(record.get("strategy")), []).append(record)

    def order(record: Mapping[str, Any]) -> tuple:
        gate_pass = 1.0 if record.get("gate_pass") else 0.0
        gate = record.get("gate_score")
        pnl = record.get("exp_pnl_model")
        return (
            gate_pass,
            float(gate) if gate is not None else float("-inf"),
            float(pnl) if pnl is not None else float("-inf"),
        )

    for strategy, rows in by_strategy.items():
        ranked = sorted(rows, key=order, reverse=True)
        for i, record in enumerate(ranked, start=1):
            ranks[str(record.get("row_id") or _row_identity(record))] = i
    return ranks


def _row_identity(record: Mapping[str, Any]) -> str:
    strike = record.get("strike")
    return "|".join(
        [
            str(record.get("ticker")),
            str(record.get("strategy")),
            str(record.get("event_date")),
            "atm" if strike is None else f"{float(strike):.4f}",
        ]
    )


# --------------------------------------------------------------------------
# meta / health inputs
# --------------------------------------------------------------------------


def freshness_summary(as_of=None) -> dict:
    """Per-source freshness for meta.json, computed from what is on disk.

    Staleness is made visible here instead of being silent: when the refresh
    step cannot spend quota, the board still renders — with the age of its
    data on show.
    """
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    out: dict[str, Any] = {"as_of": str(as_of.date())}

    from engine.data import store

    try:
        years = sorted({as_of.year - 1, as_of.year})
        daily = store.read_table("daily_market", years=years, columns=["ticker", "date"])
        if len(daily):
            last = pd.to_datetime(daily["date"]).max()
            out["daily_market_last_date"] = str(pd.Timestamp(last).date())
            out["daily_market_age_days"] = int((as_of - pd.Timestamp(last).normalize()).days)
        else:
            out["daily_market_last_date"] = None
            out["daily_market_age_days"] = None
    except Exception as exc:  # a freshness report must never take the board down
        out["daily_market_last_date"] = None
        out["daily_market_error"] = f"{type(exc).__name__}: {exc}"[:200]

    fetch_last: dict[str, str] = {}
    if paths.FETCH_LOG.exists():
        try:
            with open(paths.FETCH_LOG, newline="") as fh:
                for row in csv.DictReader(fh):
                    source = row.get("source") or "?"
                    if row.get("from_cache") in ("True", "true", "1"):
                        continue
                    fetch_last[source] = row.get("ts", "")
        except OSError:
            pass
    out["fetch_last_network_call"] = fetch_last
    return out


def quota_state() -> dict:
    """The ORATS quota picture from the quota ledger's last entry."""
    from engine.data.throttle import ORATS_RESERVE_FLOOR

    state: dict[str, Any] = {"reserve_floor": ORATS_RESERVE_FLOOR, "remaining": None, "ts": None}
    if not paths.QUOTA_LOG.exists():
        return state
    try:
        with open(paths.QUOTA_LOG, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            last = rows[-1]
            remaining = last.get("quota_remaining")
            state["remaining"] = int(remaining) if remaining not in (None, "") else None
            state["ts"] = last.get("ts")
            state["below_reserve"] = (
                state["remaining"] is not None and state["remaining"] < ORATS_RESERVE_FLOOR
            )
    except (OSError, ValueError):
        pass
    return state


def size_model_mae_from_ledger(panel: pd.DataFrame | None = None) -> dict:
    """Live size-model MAE against the implied-move baseline, from the ledger.

    For every resolved prediction whose driver is ``abs_move``: the model's
    frozen ``driver_prediction`` versus the realized |move| from the panel,
    and the baseline ``or_implied`` (the quoted implied move at the last
    pre-print close — the number the market itself printed) versus the same
    realized move. This is the daily-scored comparison the model-health view
    leads with.
    """
    from engine import ledger

    outcomes = {
        o["row_id"]: o
        for o in ledger.read_outcomes()
        if o.get("status") == "resolved" and o.get("event_date")
    }
    if not outcomes:
        return {"available": False, "reason": "no resolved ledger outcomes yet"}

    if panel is None:
        try:
            from engine.features import load_panel

            panel = load_panel()
        except FileNotFoundError:
            return {"available": False, "reason": "Tier-3 panel not built"}

    cols = ["ticker", "date", "abs_move"]
    if "or_implied" in panel.columns:
        cols.append("or_implied")
    realized: dict[tuple[str, pd.Timestamp], tuple[float | None, float | None]] = {}
    for record in panel[cols].itertuples(index=False):
        fields = record._asdict()
        if pd.isna(fields["abs_move"]):
            continue
        implied = fields.get("or_implied")
        realized[(fields["ticker"], pd.Timestamp(fields["date"]).normalize())] = (
            float(fields["abs_move"]),
            float(implied) if implied is not None and pd.notna(implied) else None,
        )

    pairs: list[dict] = []
    for row in ledger.read_predictions():
        outcome = outcomes.get(row["row_id"])
        if outcome is None:
            continue
        score = row.get("score") or {}
        if score.get("driver_name") != "abs_move":
            continue
        predicted = score.get("driver_prediction")
        if predicted is None:
            continue
        key = (row["ticker"], pd.Timestamp(row["event_date"]).normalize())
        realized_pair = realized.get(key)
        if realized_pair is None or realized_pair[0] is None:
            continue
        pairs.append(
            {
                "as_of": row["as_of"],
                "ticker": row["ticker"],
                "event_date": row["event_date"],
                "predicted": float(predicted),
                "realized": realized_pair[0],
                "implied": realized_pair[1],
            }
        )
    if not pairs:
        return {"available": False, "reason": "no resolved size-model predictions yet"}

    frame = pd.DataFrame(pairs)
    frame["model_err"] = (frame["predicted"] - frame["realized"]).abs()
    out = {
        "available": True,
        "n": int(len(frame)),
        "model_mae_pp": float(frame["model_err"].mean()),
    }
    has_baseline = frame["implied"].notna()
    if has_baseline.any():
        baseline_err = (frame.loc[has_baseline, "implied"] - frame.loc[has_baseline, "realized"]).abs()
        out["implied_baseline_mae_pp"] = float(baseline_err.mean())
        out["n_baseline"] = int(has_baseline.sum())
    series = []
    for as_of, group in frame.groupby("as_of"):
        entry = {
            "as_of": as_of,
            "n": int(len(group)),
            "model_mae_pp": float(group["model_err"].mean()),
        }
        if group["implied"].notna().any():
            entry["implied_baseline_mae_pp"] = float(
                (group["implied"] - group["realized"]).abs().mean()
            )
        series.append(entry)
    out["series"] = series
    return out


def build_strategies(registry=None) -> dict:
    """How each strategy's number is derived — the reader's way in.

    Everything here is read off the objects that actually produce the score:
    the structure definition for the legs and the entry/exit offsets, the model
    registry for the champion and its walk-forward metrics, ``PAYOFF_DRIVER``
    for what the model predicts. Nothing is transcribed by hand, so the page
    cannot describe a pipeline the engine no longer runs.

    Per-row inputs are not here — they belong to the row, and travel in the
    per-ticker files as ``model_inputs``. This is the shape; those are the
    values that went through it.
    """
    from engine.features import DRIVER_NOTES, feature_note
    from engine.payoff import PAYOFF_DRIVER
    from engine.score import DISABLED_STRATEGIES
    from engine.structures import STRUCTURES

    if registry is None:
        try:
            from engine.models.registry import load_registry

            registry = load_registry()
        except Exception:
            registry = None

    def champion(role: str, strategy: str):
        if registry is None:
            return None
        for candidate in (strategy, "*"):
            try:
                return registry.champion(role, strategy=candidate)
            except TypeError:
                try:
                    return registry.champion(role)
                except Exception:
                    return None
            except Exception:
                continue
        return None

    def describe(entry) -> dict | None:
        if entry is None:
            return None
        metrics = entry.eval if isinstance(entry.eval, dict) else {}
        return {
            "id": entry.id,
            "role": entry.role,
            "target": entry.target,
            "train_window": entry.train_window,
            "promoted": str(getattr(entry, "promoted", "") or ""),
            "evidence": str(getattr(entry, "evidence", "") or ""),
            "threshold": getattr(entry, "threshold", None),
            "artifact_sha256": (entry.artifact_sha256 or "")[:16],
            # Headline walk-forward numbers only. The full block lives in the
            # registry; a page nobody reads is not transparency.
            "oos": {
                k: metrics.get(k)
                for k in ("n", "r", "mae", "rmse", "bias", "decile_spread", "oos_years")
                if metrics.get(k) is not None
            },
            "features": [
                {"name": name, "note": feature_note(name)} for name in entry.features
            ],
        }

    out: dict[str, Any] = {}
    for name in sorted(STRUCTURES):
        structure = STRUCTURES[name]()
        driver = PAYOFF_DRIVER.get(name)
        legs = [
            {
                "right": "call" if str(leg.right).upper().startswith("C") else "put",
                "qty": float(getattr(leg, "qty", 1.0)),
                "side": "long" if float(getattr(leg, "qty", 1.0)) > 0 else "short",
            }
            for leg in getattr(structure, "legs", [])
        ]
        out[name] = {
            "name": name,
            "enabled": name not in DISABLED_STRATEGIES,
            "disabled_reason": DISABLED_STRATEGIES.get(name),
            "structure": {
                "legs": legs,
                "entry_offset": int(structure.entry_offset),
                "exit_offset": int(structure.exit_offset),
                "entry_note": _offset_note(structure.entry_offset),
                "exit_note": _offset_note(structure.exit_offset),
            },
            "driver": driver,
            "driver_note": DRIVER_NOTES.get(driver or "", ""),
            "model": describe(champion("size" if driver == "abs_move" else "implied_t1", name))
            if driver
            else None,
            "gate": describe(champion("gate", name)),
            "layers": {
                "model": (
                    "The champion predicts the driver from the features below. That "
                    "prediction is pushed through the structure's payoff map — a line "
                    "fitted on real replayed trades, exit_value/spot = intercept + "
                    "slope x driver — and simulated against the premium actually "
                    "quoted, twice-randomised: once for how wrong the driver "
                    "prediction may be, once for how much the payoff line fails to "
                    "explain. Expected PnL is the mean of those draws; the win rate "
                    "is the share above zero."
                ),
                "analog": (
                    "The empirical distribution of matched historical trades — same "
                    "strategy, matched on market-cap bucket, implied-move level, DTE "
                    "and moneyness — with a bootstrap CI. It uses no model at all, "
                    "which is why it is shown beside the model layer instead of "
                    "averaged into it: when the two disagree that is information, and "
                    "the row is flagged LAYER_DISAGREE."
                ),
            },
        }
    return out


def _offset_note(offset: int) -> str:
    """Plain English for a trading-day offset relative to the print."""
    offset = int(offset)
    if offset == 0:
        return "the last close before the announcement"
    if offset == 1:
        return "the first close after the announcement"
    if offset < 0:
        return f"{abs(offset)} trading days before the last pre-print close"
    return f"{offset} trading days after the announcement"


def build_meta(
    scores: pd.DataFrame,
    *,
    as_of,
    horizon_days: int,
    fill_alpha: float,
    alt_strikes: int,
    freshness: dict | None = None,
    quota: dict | None = None,
    late_as_ofs: Sequence[str] | None = None,
    registry=None,
) -> dict:
    """meta.json: identity, versions, and freshness of one bundle."""
    as_of = pd.Timestamp(as_of).normalize()
    records = scores.to_dict(orient="records")
    events = _count_events(records)

    model_versions: dict[str, str] = {}
    if registry is None:
        try:
            from engine.models.registry import load_registry

            registry = load_registry()
        except Exception:
            registry = None
    if registry is not None:
        for role in ("size", "implied_t1", "gate"):
            try:
                model_versions[role] = registry.champion(role).id
            except Exception:
                continue

    from engine.score import DISABLED_STRATEGIES
    from engine.structures import STRUCTURES

    strategies = {
        name: (
            {"enabled": False, "detail": DISABLED_STRATEGIES[name]}
            if name in DISABLED_STRATEGIES
            else {"enabled": True}
        )
        for name in sorted(STRUCTURES)
    }

    snapshot = next(
        (r.get("snapshot_hash") for r in records if r.get("snapshot_hash")), ""
    )
    return {
        "render_version": RENDER_VERSION,
        "as_of": str(as_of.date()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizon_days": int(horizon_days),
        "fill_alpha": float(fill_alpha),
        "alt_strikes": int(alt_strikes),
        "snapshot_hash": snapshot,
        "n_rows": len(records),
        "n_events": events,
        "n_tickers": len({r.get("ticker") for r in records}),
        "strategies": strategies,
        "model_versions": model_versions,
        "freshness": freshness if freshness is not None else {},
        "quota": quota if quota is not None else {},
        "late_as_ofs": list(late_as_ofs or []),
        "cron": {
            "entry": "daily post-close (see dashboard/README.md)",
            "idempotent": True,
        },
    }


def build_health(
    *,
    as_of=None,
    selfcheck_report: Any = None,
    size_mae: dict | None = None,
    ledger_health: dict | None = None,
    drift_brier_skill_floor: float = -0.05,
) -> dict:
    """health.json: calibration drift, live MAE, and the last self-check.

    Reads the ledger's ``health.json`` when present — the ledger owns
    calibration; the dashboard only displays it.
    """
    if ledger_health is None:
        from engine import ledger

        path = ledger.health_path()
        if path.exists():
            try:
                ledger_health = json.loads(path.read_text())
            except (ValueError, OSError):
                ledger_health = None

    drift_reasons: list[str] = []
    if ledger_health:
        for strategy, block in sorted((ledger_health.get("per_strategy") or {}).items()):
            if not isinstance(block, dict) or not block.get("available"):
                continue
            skill = block.get("brier_skill")
            if skill is not None and float(skill) < drift_brier_skill_floor:
                drift_reasons.append(
                    f"{strategy}: Brier skill {float(skill):.3f} < {drift_brier_skill_floor}"
                )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": str(pd.Timestamp(as_of).date()) if as_of is not None else None,
        "ledger": ledger_health or {"available": False},
        "size_model": size_mae or {"available": False},
        "last_selfcheck": (
            selfcheck_report.as_dict()
            if hasattr(selfcheck_report, "as_dict")
            else selfcheck_report
        ),
        "calibration_drift": {"flagged": bool(drift_reasons), "reasons": drift_reasons},
    }


# --------------------------------------------------------------------------
# the renderer
# --------------------------------------------------------------------------


def _static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


def _copy_static(out: Path, static_dir: Path | None = None) -> list[str]:
    """Copy the client (index.html + assets) into the bundle.

    The client ships with the engine, not with the data: a bundle re-rendered
    on another machine from the same commit is byte-identical in its code.
    """
    src = Path(static_dir) if static_dir is not None else _static_dir()
    if not src.exists():
        raise FileNotFoundError(f"dashboard static templates missing at {src}")
    files: list[str] = []
    (out / "assets").mkdir(parents=True, exist_ok=True)
    for name in ("index.html",):
        shutil.copyfile(src / name, out / name)
        files.append(name)
    assets = src / "assets"
    if assets.exists():
        for path in sorted(assets.iterdir()):
            if path.is_file():
                shutil.copyfile(path, out / "assets" / path.name)
                files.append(f"assets/{path.name}")
    return files


def _ticker_payload(
    ticker: str,
    records: list[dict],
    *,
    as_of,
    panel: pd.DataFrame | None,
    trades: pd.DataFrame | None,
) -> dict:
    """The strike explorer + the ticker's evidence panel."""
    events: list[dict] = []
    if records:
        frame = pd.DataFrame(records)
        grouped = frame.groupby("event_date", sort=True)
    else:
        grouped = []
    for event_date, group in grouped:
        rows = sorted(
            group.to_dict(orient="records"),
            key=lambda r: (
                r.get("strategy") or "",
                float("inf") if r.get("strike") is None else float(r["strike"]),
            ),
        )
        events.append(
            {
                "event_date": event_date,
                "session": rows[0].get("session") if rows else None,
                "rows": [_clean(r) for r in rows],
            }
        )

    history: list[dict] = []
    if panel is not None and len(panel):
        hist = panel[panel["ticker"] == ticker].copy()
        if len(hist):
            hist["date"] = pd.to_datetime(hist["date"])
            hist = hist.sort_values("date").tail(MAX_HISTORY_EVENTS)
            cols = [
                c
                for c in ("date", "k", "implied_move", "or_implied", "move", "abs_move", "mcap_usd")
                if c in hist.columns
            ]
            history = [_clean(dict(r._asdict())) for r in hist[cols].itertuples(index=False)]
            history.reverse()

    analogs: list[dict] = []
    if trades is not None and len(trades):
        own = trades[trades["ticker"] == ticker].copy()
        if len(own):
            own["event_date"] = pd.to_datetime(own["event_date"])
            own = own.sort_values("event_date").tail(MAX_ANALOG_TRADES)
            cols = [
                c
                for c in (
                    "strategy", "event_date", "entry_date", "exit_date",
                    "fill_alpha", "entry_cost", "exit_value", "ret",
                )
                if c in own.columns
            ]
            analogs = [_clean(dict(r._asdict())) for r in own[cols].itertuples(index=False)]
            analogs.reverse()

    return {
        "ticker": ticker,
        "as_of": str(pd.Timestamp(as_of).date()),
        "events": events,
        "history": history,
        "analogs": analogs,
    }


def write_single_file(bundle: Path | str, out: Path | str | None = None) -> Path:
    """Pack a rendered bundle into ONE self-contained ``.html``.

    Not a second renderer: it inlines the bytes ``render_bundle`` already
    wrote — the same ``app.js``, the same data payloads — so it cannot show a
    number the bundle does not. What it drops is laziness: every per-ticker
    file is inlined up front instead of loaded on click, which is the whole
    point (one file has nowhere to load from).

    For the case where a served bundle is awkward to reach — a container
    without a published port, a phone, an email to yourself — and for keeping a
    dated copy of what the board said on a day.
    """
    bundle = Path(bundle)
    out = Path(out) if out is not None else bundle.parent / f"earnings-board-{_bundle_as_of(bundle)}.html"
    html = (bundle / "index.html").read_text()

    css = (bundle / "assets" / "app.css").read_text()
    html = html.replace(
        '<link rel="stylesheet" href="assets/app.css">', f"<style>\n{css}\n</style>"
    )

    # Every data script the page references, inlined where it sat.
    referenced = set()
    for src in re.findall(r'<script src="(data/[^"]+\.js)"></script>', html):
        referenced.add(src)
        payload = (bundle / src).read_text()
        html = html.replace(f'<script src="{src}"></script>', f"<script>\n{payload}</script>")

    # Plus every payload the page loads ON DEMAND. models.js is ~2 MB and is
    # deliberately not on the first-paint path, so a build that inlined only
    # what index.html references would ship a one-file board whose Models area
    # is permanently empty.
    lazy = [
        path for path in sorted((bundle / "data").glob("*.js"))
        if f"data/{path.name}" not in referenced
    ]
    if lazy:
        blocks = "\n".join(f"<script>\n{path.read_text()}</script>" for path in lazy)
        html = html.replace("</body>", f"{blocks}\n</body>") if "</body>" in html else html + blocks

    # Every ticker, inlined where the client already looks first: its lazy
    # loader checks `TICKER_DATA[ticker]` before injecting a script, so a
    # pre-populated map makes the explorer work with nothing to fetch.
    tickers = sorted((bundle / "data" / "tickers").glob("*.json"))
    blocks = ["window.TICKER_DATA = window.TICKER_DATA || {};"]
    for path in tickers:
        blocks.append(f'window.TICKER_DATA[{json.dumps(path.stem)}] = {path.read_text().strip()};')
    html = html.replace(
        "window.TICKER_DATA = window.TICKER_DATA || {};", "\n".join(blocks)
    )

    app = (bundle / "assets" / "app.js").read_text()
    html = html.replace('<script src="assets/app.js"></script>', f"<script>\n{app}\n</script>")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


def _bundle_as_of(bundle: Path) -> str:
    try:
        return str(json.loads((bundle / "data" / "meta.json").read_text()).get("as_of", "snapshot"))
    except (OSError, ValueError):
        return "snapshot"


def render_bundle(
    scores: pd.DataFrame,
    out: Path | str,
    *,
    as_of=None,
    horizon_days: int = 21,
    fill_alpha: float = 0.5,
    alt_strikes: int = 0,
    panel: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    meta: dict | None = None,
    health: dict | None = None,
    flags: dict | list | None = None,
    static_dir: Path | str | None = None,
    registry=None,
) -> dict:
    """Render one bundle. Deterministic in ``(scores, meta, health)``.

    ``panel`` and ``trades`` feed the per-ticker evidence panel; when omitted
    the evidence sections render empty rather than failing — a bundle must be
    producible even on a machine mid-rebuild.
    """
    out = paths.assert_writable(Path(out))
    out.mkdir(parents=True, exist_ok=True)
    as_of = (
        pd.Timestamp(as_of).normalize()
        if as_of is not None
        else pd.Timestamp.today().normalize()
    )

    records = [_clean(dict(r)) | {"digest": row_digest(r)} for r in scores.to_dict(orient="records")]
    ranks = _rank_rows(
        [r | {"row_id": _row_identity(r)} for r in records]
    )
    board_rows = [
        compact_row(r, rank=ranks.get(_row_identity(r))) for r in records
    ]
    board_rows.sort(
        key=lambda r: (
            r.get("event_date") or "",
            r.get("ticker") or "",
            r.get("strategy") or "",
            float("inf") if r.get("strike") is None else float(r["strike"]),
        )
    )

    board = {
        "as_of": str(as_of.date()),
        "n_rows": len(board_rows),
        "rows": board_rows,
    }
    board_path = _write_pair(out / "data", "board", board)
    board_bytes = board_path.stat().st_size

    by_ticker: dict[str, list[dict]] = {}
    for record in records:
        by_ticker.setdefault(str(record.get("ticker")), []).append(record)

    tickers_dir = out / "data" / "tickers"
    if tickers_dir.exists():
        shutil.rmtree(tickers_dir)
    tickers_dir.mkdir(parents=True, exist_ok=True)
    for ticker in sorted(by_ticker):
        payload = _ticker_payload(
            ticker, by_ticker[ticker], as_of=as_of, panel=panel, trades=trades
        )
        json_text = json.dumps(_clean(payload), sort_keys=True, default=str)
        (tickers_dir / f"{ticker}.json").write_text(json_text + "\n")
        (tickers_dir / f"{ticker}.js").write_text(
            "window.TICKER_DATA = window.TICKER_DATA || {};\n"
            f'window.TICKER_DATA["{ticker}"] = {json_text};\n'
        )

    meta_payload = meta if meta is not None else build_meta(
        scores,
        as_of=as_of,
        horizon_days=horizon_days,
        fill_alpha=fill_alpha,
        alt_strikes=alt_strikes,
        registry=registry,
    )
    meta_payload = dict(meta_payload)
    meta_payload.setdefault("board_bytes", board_bytes)
    meta_payload["board_oversized"] = board_bytes > BOARD_MAX_BYTES
    _write_pair(out / "data", "meta", meta_payload)

    health_payload = health if health is not None else build_health(as_of=as_of)
    _write_pair(out / "data", "health", health_payload)

    flags_payload = {"as_of": str(as_of.date()), "flags": flags if flags is not None else []}
    _write_pair(out / "data", "flags", flags_payload)

    _write_pair(out / "data", "strategies", build_strategies(registry=registry))

    # Per-input evidence, if it has been built. Absent is a legitimate state —
    # it is rebuilt only when a champion changes, not nightly — and the view
    # says so rather than rendering an empty table.
    from engine.dashboard.model_evidence import load_model_evidence

    _write_pair(
        out / "data", "models",
        load_model_evidence() or {"models": {}, "available": False},
    )

    static_files = _copy_static(out, static_dir)

    return {
        "out": str(out),
        "as_of": str(as_of.date()),
        "n_rows": len(board_rows),
        "n_events": _count_events(records),
        "n_tickers": len(by_ticker),
        "board_bytes": board_bytes,
        "board_oversized": board_bytes > BOARD_MAX_BYTES,
        "files": static_files,
    }
