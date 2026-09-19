"""Real-data dataset builders for the Phase 5 frozen serving states.

Legacy refits these states live inside every Scorer (P5-1
``NON_MODEL_STATE_ITEMS``). This module reads the SAME rows legacy fits them
on, through the legacy loaders, read-only, so ``tools/phase5_training_job.py``
can freeze them. Legacy readers live here, under ``tools/``, because
``engine/v2`` may not import legacy numerics (``checks/import_layers.py``);
no new v2 -> legacy adapter edge is needed.

Which legacy selection each builder mirrors:

* :func:`payoff_trades` -- ``Scorer.trades`` as ``Scorer.payoff`` /
  ``Scorer.runup_payoff`` hand it to ``engine.payoff.fit_payoff`` /
  ``fit_runup_payoff``: ``_load_trades_without_legs`` (spot_entry/spot_exit
  parsed from the legs blob, a partition at a time), ``provenance ==
  "engine.replay"``, then ``Scorer._enrich``'s left merge of the full panel's
  ``abs_move``/``or_implied`` and ``im_t1 = or_implied``. Only the columns the
  payoff recipes read are kept, and only STR-THRU/STR-RUNUP rows (the recipes
  filter on strategy anyway); row order is legacy's, which matters because it
  fixes which residuals the seeded subsample keeps. The analog-only parts of
  ``_enrich`` (entry-date implied move, ``bucket_frame``) add columns and
  neither drop nor reorder rows, so they are skipped: they are the part that
  reads ``daily_market``.
* :func:`recalibration_pairs` -- ``engine.recalibrate.load_pairs()``, the
  cached pairs table ``Scorer.recalibration`` fits on. Rebuilding that table
  (``recalibrate.build_pairs``) re-scores thousands of events and writes into
  ``data/``; legacy scoring never does it, so neither does this.
* :func:`crush_table` -- ``engine.models.training.iv_crush.crush_frame()``
  over the FULL universe (legacy's ``context.daily is None`` branch of
  ``Scorer._crush_table``), computed a ticker chunk at a time. That is exact:
  ``crush_frame`` pairs each event with adjacent rows of its own ticker's
  series only.
* :func:`paired_pool_inputs` -- ``Scorer._residual_pool``'s three inputs:
  stored Tier-4 forecasts, the panel's realized ``abs_move`` and the crush
  table above. **Difference from legacy:** a bounded live Scorer (the
  nightly's) scopes the crush table to the tickers its context loaded, so its
  pool is a function of the board; this one never is.
* :func:`board_analog_trades` -- ``Scorer.trades``, the population
  ``engine.analogs.AnalogMatcher`` is built on: ``_load_trades_without_legs``,
  ``provenance == "engine.replay"``, then legacy ``Scorer._enrich`` itself
  (panel merge, entry-date implied move, ``bucket_frame``) run against the
  FULL panel. **Difference from legacy:** a bounded live Scorer (the
  nightly's) merges the panel its context loaded, so tickers outside the
  board lose their market-cap and implied-ratio buckets and the matcher's
  population moves with the board; this one never does.
* :func:`champion_driver_pool` -- the full-refit champion's embedded pool
  (``ModelArtifact.residuals``/``residual_buckets``) that
  ``ModelArtifact.residual_draws`` serves in ``Scorer._score_model`` /
  ``_score_runup_model``, loaded through ``Registry.load_champion``.
* :func:`pnl_sim_history` -- ``engine.pnl_sim.load_history()``, the stored
  ``exp_pnl_sim`` series ``Scorer._simulated_pnl`` reads for the entry-rule
  gate's trailing cutoff (``pnl_sim.trailing_cutoff``). ``None`` when the
  file was never built, which legacy serves as "no bar".

Nothing here fits anything or writes anything.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "BOARD_ANALOG_COLUMNS",
    "DRIVER_ROLES",
    "PAYOFF_STRATEGIES",
    "champion_driver_pool",
    "crush_table",
    "board_analog_trades",
    "file_digest",
    "paired_pool_inputs",
    "payoff_trades",
    "pnl_sim_history",
    "recalibration_pairs",
]

PAYOFF_STRATEGIES = ("STR-THRU", "STR-RUNUP")
#: Catalog member ``driver_residual_pool:<role>`` -> the champion strategy key.
DRIVER_ROLES = ("size", "implied_t1", "runup_move")

#: The trades columns the payoff recipes (and their receipts) read.
_TRADE_COLUMNS = ("event_id", "ticker", "strategy", "fill_alpha", "event_date",
                  "entry_date", "exit_date", "strike", "exit_value", "provenance")
_CRUSH_DAILY_COLUMNS = ["ticker", "date", "iv10", "iv30", "exern_iv10", "exern_iv30"]


def _log(message: str) -> None:
    print(f"[p5-data] {message}", flush=True)


def file_digest(path) -> str | None:
    """sha256 of a source file, streamed (``None`` when absent)."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


# --------------------------------------------------------------------------
# payoff: Scorer.trades
# --------------------------------------------------------------------------


def _trade_partitions():
    from engine.data import store

    yield from store.iter_table("trades", columns=[*_TRADE_COLUMNS, "legs"])


def payoff_trades(partitions: Iterable | None = None, panel: pd.DataFrame | None = None,
                  strategies=PAYOFF_STRATEGIES) -> pd.DataFrame:
    """The ``Scorer.trades`` rows the payoff fits read, in legacy order.

    ``partitions`` yields ``(year, frame)`` like ``store.iter_table`` (tests
    inject it); each frame carries ``legs``. Peak memory is one partition's
    legs blob plus the slim rows kept so far, never the whole blob.
    """
    from engine.data.schemas import coerce
    from engine.replay import legs_exit_spot, legs_spot_dte

    parts = _trade_partitions() if partitions is None else partitions
    blocks = []
    for year, part in parts:
        keep = ((part["provenance"].astype(str) == "engine.replay")
                & part["strategy"].astype(str).isin(set(strategies))).to_numpy()
        part = part[keep]
        spot_entry, _dte = legs_spot_dte(part)
        spot_exit = legs_exit_spot(part)
        block = part.drop(columns=["legs"])
        block = coerce(block, "trades", only=[c for c in _TRADE_COLUMNS if c in block.columns],
                       allow_extra=True)
        block["spot_entry"] = spot_entry.to_numpy()
        block["spot_exit"] = spot_exit.to_numpy()
        blocks.append(block)
        _log(f"trades {year}: kept {len(block):,} of {int(keep.size):,} rows")
        del part, spot_entry, spot_exit
    if not blocks:
        raise SystemExit("the trades table is empty; nothing to calibrate")
    out = pd.concat(blocks, ignore_index=True)
    del blocks
    out["event_date"] = pd.to_datetime(out["event_date"])
    if panel is None:
        from engine.features import load_panel

        panel = load_panel()
    columns = [c for c in ("ticker", "date", "or_implied", "abs_move") if c in panel.columns]
    out = out.merge(panel[columns].rename(columns={"date": "event_date"}),
                    on=["ticker", "event_date"], how="left")
    out["im_t1"] = out["or_implied"]
    return out


# --------------------------------------------------------------------------
# recalibration: the cached pairs table
# --------------------------------------------------------------------------


def recalibration_pairs(path=None) -> pd.DataFrame:
    """``recalibrate.load_pairs(path)``: the exact table ``Scorer.recalibration`` reads."""
    from engine import recalibrate

    pairs = recalibrate.load_pairs(path)
    if pairs.empty:
        raise SystemExit("recalibration pairs table is missing or empty "
                         f"({path or recalibrate.PAIRS_PATH}); legacy ships the raw win")
    return pairs


# --------------------------------------------------------------------------
# trailing pnl_sim cutoff: the stored history
# --------------------------------------------------------------------------


def pnl_sim_history(path=None) -> pd.DataFrame | None:
    """``pnl_sim.load_history(path)``, the two columns the cutoff reads."""
    from engine import pnl_sim

    history = pnl_sim.load_history(path)
    if history is None:
        return None
    return history[["event_date", "exp_pnl_sim"]]


# --------------------------------------------------------------------------
# paired residual pool: Scorer._residual_pool over the full universe
# --------------------------------------------------------------------------


def _daily_chunk(tickers: set[str], daily_parts) -> pd.DataFrame:
    from engine.data import store
    from engine.data.schemas import coerce

    parts = (daily_parts() if daily_parts is not None
             else store.iter_table("daily_market", columns=_CRUSH_DAILY_COLUMNS))
    kept = []
    for _, frame in parts:
        chunk = frame[frame["ticker"].astype(str).isin(tickers)]
        if len(chunk):
            kept.append(chunk[_CRUSH_DAILY_COLUMNS])
    if not kept:
        return coerce(pd.DataFrame(columns=_CRUSH_DAILY_COLUMNS), "daily_market",
                      only=_CRUSH_DAILY_COLUMNS)
    return coerce(pd.concat(kept, ignore_index=True), "daily_market", only=_CRUSH_DAILY_COLUMNS)


def crush_table(*, events: pd.DataFrame | None = None, daily_parts=None,
                ticker_chunk: int = 1000) -> pd.DataFrame:
    """``iv_crush.crush_frame()`` for every event, a ticker chunk at a time.

    ``daily_parts`` is a zero-argument callable returning a fresh
    ``(year, frame)`` iterator (default: ``store.iter_table("daily_market")``,
    re-read once per chunk). The result equals ``crush_frame()`` on the whole
    table row for row, in the same ``(ticker, event_date)`` order.
    """
    from engine.data import store
    from engine.models.training import iv_crush

    if events is None:
        events = store.read_table("earnings_events", columns=["ticker", "event_date", "session"])
    tickers = sorted(events.dropna(subset=["session"])["ticker"].astype(str).unique())
    frames = []
    for start in range(0, len(tickers), ticker_chunk):
        chunk = set(tickers[start:start + ticker_chunk])
        daily = _daily_chunk(chunk, daily_parts)
        frames.append(iv_crush.crush_frame(
            events=events[events["ticker"].astype(str).isin(chunk)], daily=daily))
        _log(f"crush tickers {start + len(chunk):,}/{len(tickers):,}: "
             f"daily rows {len(daily):,}, events paired {len(frames[-1]):,}")
        del daily
    if not frames:
        return iv_crush.crush_frame(events=events.iloc[:0],
                                    daily=_daily_chunk(set(), lambda: iter(())))
    return pd.concat(frames, ignore_index=True)


def _single_model_id(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        raise SystemExit(f"tier4 forecasts have no {column!r}; cannot key the paired pool")
    ids = sorted({str(v) for v in frame[column].dropna().unique()})
    if not ids:
        raise SystemExit(f"{column}: no producer id on any poolable forecast row")
    if len(ids) > 1:
        # Legacy pools every stored row whatever producer stamped it; the key
        # names them all rather than pretending the pool has one producer.
        _log(f"{column}: {len(ids)} producer ids in the pooled rows; keyed as their '+' join")
    return "+".join(ids)


def paired_pool_inputs(*, forecasts: pd.DataFrame | None = None,
                       panel: pd.DataFrame | None = None, crush: pd.DataFrame | None = None,
                       ticker_chunk: int = 1000) -> dict:
    """The three frames ``Scorer._residual_pool`` joins, over the full universe.

    Returns ``{"forecasts", "outcomes", "crush", "move_model_id",
    "crush_model_id"}``. The two model ids are the Tier-4 producers stamped
    on the forecast rows that can enter the pool ('+'-joined if mixed).
    """
    if forecasts is None:
        from engine.data.features import tier4

        forecasts = tier4.load_forecasts()
        if forecasts is None:
            raise SystemExit("tier4_forecasts is missing; the paired pool cannot be built")
    if panel is None:
        from engine.features import load_panel

        panel = load_panel()
    if crush is None:
        crush = crush_table(ticker_chunk=ticker_chunk)
    cols = ["ticker", "event_date", "pred_abs_move", "pred_iv_crush_30"]
    ids = [c for c in ("pred_abs_move_model_id", "pred_iv_crush_30_model_id") if c in forecasts]
    fc = forecasts[cols + ids].copy()
    fc["event_date"] = pd.to_datetime(fc["event_date"])
    usable = fc[np.isfinite(pd.to_numeric(fc["pred_abs_move"], errors="coerce"))
                & np.isfinite(pd.to_numeric(fc["pred_iv_crush_30"], errors="coerce"))]
    outcomes = panel[["ticker", "date", "abs_move"]].rename(columns={"date": "event_date"})
    return {
        "forecasts": fc[cols],
        "outcomes": outcomes,
        "crush": crush[["ticker", "event_date", "crush_pct_iv30"]],
        "move_model_id": _single_model_id(usable, "pred_abs_move_model_id"),
        "crush_model_id": _single_model_id(usable, "pred_iv_crush_30_model_id"),
    }


# --------------------------------------------------------------------------
# board analog matcher: Scorer.trades over the full panel
# --------------------------------------------------------------------------

#: What the analog builder reads (``engine.v2.models.training.analogs``).
BOARD_ANALOG_COLUMNS = ("trade_id", "strategy", "fill_alpha", "event_date", "exit_date",
                        "mcap_bucket", "dte_band", "moneyness_band", "implied_ratio", "ret")
_PANEL_ENRICH_COLUMNS = ["ticker", "date", "mcap_usd", "or_implied", "mean_prior_or_implied",
                         "abs_move", "n_prior"]


def board_analog_trades(*, trades: pd.DataFrame | None = None, panel: pd.DataFrame | None = None,
                        analog_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """``Scorer.trades`` as the analog matcher sees it, over the FULL panel.

    Runs legacy ``Scorer._enrich`` unchanged on a bare scorer whose context
    holds the whole panel (only the columns ``_enrich`` merges). With no
    ``analog_daily`` the entry-date implied move is read from
    ``daily_market`` a ticker chunk at a time (legacy's own chunked path),
    so no request context narrows it. ``trades``/``panel``/``analog_daily``
    are injectable for tests. Returns the analog columns plus
    ``implied_edges`` in ``attrs`` (legacy's population edges).
    """
    import types

    from engine import score as score_mod

    if trades is None:
        trades = score_mod._load_trades_without_legs()
    engine_rows = trades[trades["provenance"].astype(str) == "engine.replay"]
    if panel is None:
        from engine.features import load_panel

        panel = load_panel()
    scorer = score_mod.Scorer.__new__(score_mod.Scorer)
    scorer.context = types.SimpleNamespace(
        panel=panel[[c for c in _PANEL_ENRICH_COLUMNS if c in panel.columns]])
    scorer._analog_daily = analog_daily
    enriched = score_mod.Scorer._enrich(scorer, engine_rows)
    _log(f"analog trades: {len(enriched):,} engine.replay rows; entry-date implied "
         f"coverage {scorer.analog_entry_coverage}")
    out = enriched[list(BOARD_ANALOG_COLUMNS)].copy()
    out.attrs["implied_edges"] = enriched.attrs.get("implied_edges")
    return out


# --------------------------------------------------------------------------
# driver residual pools: the champions' embedded pools
# --------------------------------------------------------------------------


def champion_driver_pool(role: str, registry=None) -> dict:
    """``{"model_id", "artifact_sha256", "residuals", "buckets"}`` for a champion.

    ``Registry.load_champion(role, "*", verify=True)``: the hash-verified
    pickle ``Scorer.model(role)`` serves, read-only.
    """
    from engine.models.registry import load_registry

    if role not in DRIVER_ROLES:
        raise SystemExit(f"{role}: not a driver residual pool role ({', '.join(DRIVER_ROLES)})")
    registry = load_registry() if registry is None else registry
    entry, artifact = registry.load_champion(role, "*", verify=True)
    return {
        "model_id": entry.id,
        "artifact_sha256": entry.artifact_sha256,
        "residuals": np.asarray(artifact.residuals, dtype=float),
        "buckets": getattr(artifact, "residual_buckets", None),
    }
