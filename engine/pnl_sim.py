"""Expected P&L for a structure, by simulation — the gate that replaced arithmetic.

TWIN-P5's reward term was ``cost < peak / 2``: max profit beats max loss, decided
without reference to where the print was likely to land. EXP-129 replaced it with
the quantity that term is a proxy for. Each leg's exit value is a deterministic
function of where spot lands, what implied vol survives the print, and how much
time is left, and two of those three have out-of-sample forecasts with calibrated
error distributions in Tier 4. So the expectation is an integral over two random
variables, evaluated here by simulation.

**Why not closed form.** ``PnL(m, c)`` is a sum of Black-Scholes prices carrying
``N(d1)`` and ``N(d2)``, where ``d1`` depends on ``log(S(1+m)/K)`` and on
``sigma(1+c)sqrt(T)``. Integrating the normal CDF against a distribution over
both a shift in spot and a multiplier on vol has no elementary antiderivative. It
collapses only at ``T_exit = 0`` — and the exit is not at expiry: median 9 DTE
remain when TWIN-P5 closes, and only 15% of exits fall within a day of it.

**What EXP-129 and EXP-131 established, and what they did not.** The gate beats
``cost < peak/2`` at every matched selectivity, and on 2023-2026 held out from
its own selection it compounded 1.0 to 2.53 against the incumbent's 1.68, at
Sharpe 1.54 against 1.15. It also did NOT clear its distinguishability bar: a
block bootstrap on the held-out CAGR difference spans [-28.99, +43.71]pp. Four
years cannot separate these rules, and this module is live on a decision the
statistics could not make. See ``guides/pnl_gate_promotion.md``.

**The mechanism is simpler than the machinery suggests.** A post-hoc 2x2 found
that the MOVE's variance is the whole effect: holding it at its point forecast
costs ~9pp of CAGR, while the crush's variance moves nothing in either
direction. The crush model earns its place as a LEVEL — it sets exit vol — and
its distribution does not. The paired draw is kept because it costs nothing and
because ``independent`` measured 3.3pp of CAGR worse, but nobody should describe
this as joint integration doing the work.
"""
from __future__ import annotations

import hashlib
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    "DRAWS", "MIN_VOL", "MIN_SPOT_FRACTION", "MIN_POOL", "WINDOW_MONTHS",
    "QUANTILE", "black_scholes_put", "ResidualPool", "expected_pnl",
    "trailing_cutoff",
    "load_history",
    "HISTORY_PATH",
]

#: Registered in EXP-129's spec before it ran. `draws_500` measured the estimate
#: as stable at a fifth of this; 4,000 is kept because it costs milliseconds.
DRAWS = 4000

#: Vol floor after a crush draw, in decimals. A crush at or past -100% is inside
#: the residual pool's support and is not a market state.
MIN_VOL = 0.01

#: Floor on ``S_exit / S_entry``. A move draw past -100% is likewise inside the
#: pool's support and outside the world's. Left untruncated it produced a
#: negative spot, NaN through ``log(S/K)``, and silently poisoned the mean of
#: every affected event — found in EXP-129 before any result was reported.
MIN_SPOT_FRACTION = 1e-4

#: Below this many paired residuals an event is not simulated at all. Same
#: reasoning as Tier 4's MIN_RESIDUALS: an expectation over forty errors is not
#: an expectation, and a number shaped like one gets read as one.
MIN_POOL = 250

#: The trailing window the gate's cutoff is computed over, and the share it
#: admits. Chosen in EXP-131 on 2018-2022 alone and confirmed on 2023-2026.
#:
#: Six months rather than twelve is NOT a performance choice — no window from 6
#: to 36 months is distinguishable from any other on returns. It is a VOLUME
#: choice: the realized share of candidates admitted has a yearly SD of 0.027 at
#: six months against 0.051 at twelve and 0.074 for a calendar-year rule. The
#: requirement it serves is a predictable, bounded trade count.
WINDOW_MONTHS = 6
QUANTILE = 0.20


def black_scholes_put(spot, strike, years, vol):
    """Put value, vectorised over draws. Zero rates and dividends.

    Omitted deliberately rather than forgotten: over the ~9 DTE that remain at
    exit, ``exp(-rT)`` differs from 1 by under 0.1% at any plausible rate, an
    order of magnitude below this function's own repricing error. Measured
    against 2,893 known outcomes: median error $0.216 on a $5.70 median exit,
    r = 0.998, and a +$0.043 bias that points the wrong way for a gate at zero.
    """
    spot = np.asarray(spot, dtype=float)
    vol = np.maximum(np.asarray(vol, dtype=float), MIN_VOL)
    years = float(years)
    if years <= 0:
        return np.maximum(strike - spot, 0.0)
    sT = vol * np.sqrt(years)
    d1 = (np.log(spot / strike) + 0.5 * vol**2 * years) / sT
    return strike * norm.cdf(-(d1 - sT)) - spot * norm.cdf(-d1)


class ResidualPool:
    """Paired ``(move error, crush error)`` draws from strictly-earlier events.

    Both residuals come from the SAME historical event, so the dependence
    between them travels with the pair and nothing has to be estimated. That
    matters because the dependence is invisible to correlation: Pearson r is
    +0.028 across 115,195 events, while Spearman is -0.095, the median crush
    walks -13.5% to -18.5% across move deciles, and the conditional SD nearly
    doubles. A copula fitted on r = 0.028 would find none of it.

    Causality is the constraint: a pool for an event dated D holds only errors
    from events that had already printed by D.
    """

    def __init__(self, history: pd.DataFrame, buckets: int = 10) -> None:
        need = ["event_date", "pred_abs_move", "err_move", "err_crush"]
        missing = [c for c in need if c not in history.columns]
        if missing:
            raise ValueError(f"residual history is missing {missing}")
        h = history.dropna(subset=need).sort_values("event_date").reset_index(drop=True)
        self._dates = pd.to_datetime(h["event_date"]).to_numpy()
        self._pred = h["pred_abs_move"].to_numpy(dtype=float)
        self._move = h["err_move"].to_numpy(dtype=float)
        self._crush = h["err_crush"].to_numpy(dtype=float)
        self._buckets = int(buckets)
        #: Row dicts, built once and reused by reference — see `_all_rows`.
        self._row_cache: list[Mapping] | None = None
        #: Plain-dict copy of EVERY row, built once for the pool's life — see
        #: `_documented_rows`. `documented_population` slices this fresh on
        #: every call; it never rebuilds a per-cutoff copy and never retains
        #: one either — see that method's docstring for why.
        self._documented_all: list[dict] | None = None

    def __len__(self) -> int:
        return len(self._dates)

    def before(self, cutoff) -> int:
        return int(np.searchsorted(self._dates, np.datetime64(pd.Timestamp(cutoff)), "left"))

    def _all_rows(self) -> list[Mapping]:
        """Every row, JSON-safe, built once and cached for the pool's life.

        A strict-trace capture calls :meth:`evidence_rows` once per request
        that gates on a simulation, and ``range(cutoff_index)`` is always a
        PREFIX of this same date-sorted pool (cutoff differs per request, the
        ordering does not). Rebuilding ``cutoff_index`` fresh dicts (each with
        its own ``pd.Timestamp(...).isoformat()`` string) on every one of
        those requests — one 85k-row pool, a dozen-plus boundary/pinned/
        coarse rescores in a single capture run — was the repeated-allocation
        driver of a multi-GB transient that OOM-killed the bounded Phase 4
        strict-capture check. Building the dicts once and slicing/indexing
        the cached list on every subsequent call reuses the same objects
        instead of reallocating them; the returned VALUES are identical
        either way.
        """
        if self._row_cache is None:
            # Read-only: this list is shared by reference across every
            # caller and every future call, so a row a consumer could mutate
            # in place would corrupt the cache (and therefore every OTHER
            # request's checkpoint) rather than just its own copy. A plain
            # dict allows exactly that silently; `MappingProxyType` raises
            # instead. `_document`/`_normalize` (the only paths that touch
            # these rows before they are hashed or written) read Mappings
            # generically and rebuild plain dicts, so nothing downstream of
            # here ever needs write access.
            self._row_cache = [
                MappingProxyType({
                    "event_date": pd.Timestamp(self._dates[i]).isoformat(),
                    "pred_abs_move": float(self._pred[i]),
                    "err_move": float(self._move[i]),
                    "err_crush": float(self._crush[i]),
                })
                for i in range(len(self))
            ]
        return self._row_cache

    def _documented_rows(self) -> list[dict]:
        """Plain-dict copy of every pool row, built ONCE and shared by reference.

        One pass over the whole pool, not one pass per distinct cutoff. A
        strict-trace forward pass hits a distinct `cutoff_index` per distinct
        forward `event_date` — up to one per event, not shared like a
        boundary/pinned/coarse rescore's cutoff is — so caching per-cutoff
        COPIES here (the previous shape) meant every new forward date paid
        another full `[dict(row) for row in ...]` pass and retained it
        forever: measured on a synthetic 20k-row pool, ~91% of the retained
        growth over 40 distinct near-end cutoffs traced to that one list
        comprehension (see the marginal-cost regression test in
        `tests/test_pnl_sim_evidence.py`), unbounded because a 40-event run
        never revisits a cutoff often enough for any small eviction cap to
        bind.

        Building this list once and having `documented_population` SLICE it
        fixes that: `full[:cutoff_index]` is a new list object holding
        REFERENCES to these same dicts (list slicing does not copy elements),
        so its marginal cost is ~8 bytes/row (one pointer), not one dict
        rebuild per row. The dicts themselves exist once, period, regardless
        of how many distinct cutoffs the run visits.

        Contract unchanged from before: plain ``dict`` (not `_all_rows`'s
        read-only ``MappingProxyType`` — the checkpoint sink's JSON writer
        needs an actual ``dict``), shared by reference across every cutoff,
        every candidate, and the pool's whole life, and never mutated in
        place by any consumer. That audit already covers this list unchanged:
        `engine.score.Phase4TraceCollector._document` only reads Mappings
        (dict/list/scalar) to rebuild ITS OWN copies; `capture_simulation`
        wraps the value in `_Predocumented` specifically so `_document` does
        not re-copy it; nothing writes to a row in place.
        """
        if self._documented_all is None:
            self._documented_all = [dict(row) for row in self._all_rows()]
        return self._documented_all

    def documented_population(self, cutoff_index: int) -> list[dict]:
        """Plain-dict rows before ``cutoff_index``, sliced from one shared list.

        ``full[:cutoff_index]`` is a NEW list object, but its elements are
        REFERENCES to the same dicts `_documented_rows` built once -- list
        slicing does not copy elements -- so the marginal cost of a call at a
        never-seen cutoff is ~8 bytes/row (one pointer), not a row rebuild.

        Deliberately NOT cached per cutoff. A strict-trace forward pass hits
        close to one distinct `cutoff_index` per forward event (40 events,
        ~40 distinct cutoffs, no bounded cache -- 64 slots, say -- ever fills
        enough to evict); caching the SLICE would still retain one growing
        list per distinct cutoff for the run's whole life, just ~70x cheaper
        per entry than the old per-cutoff dict-rebuild. Returning a fresh,
        uncached slice means nothing outlives the caller that asked for it:
        steady-state memory is the ONE `_documented_rows()` list, full stop,
        regardless of how many distinct cutoffs the run visits.

        Every rescored variant of the SAME boundary event (pinned, strike,
        coarse-ladder) shares that event's date, hence the same cutoff INDEX
        -- those calls still get lists with IDENTICAL, reference-shared
        elements; they just are not the same wrapping list object. Callers
        that want ``_document`` (or anything downstream of it) to skip
        re-copying this list must wrap it (e.g. in
        ``engine.score._Predocumented``) before handing it off -- that
        contract lives with the caller, not here.
        """
        return self._documented_rows()[:cutoff_index]

    def evidence_rows(self, indices=None) -> list[Mapping]:
        """JSON-safe residual rows for an explicit evidence selection.

        This is intentionally separate from :meth:draw so ordinary simulation
        does not allocate row dictionaries. A trace caller must supply the
        exact indices emitted by draw evidence; an invalid index is a caller
        error rather than an opportunity to silently trim provenance.
        """
        if indices is None:
            indices = range(len(self._dates))
        cache = self._all_rows()
        rows = []
        for index in indices:
            if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < len(self):
                raise IndexError(f"residual evidence index out of range: {index!r}")
            rows.append(cache[int(index)])
        return rows

    def draw(self, cutoff, prediction: float, n: int, rng, *, evidence: dict | None = None):
        """Draw paired residuals, optionally recording the exact selection path.

        ``evidence`` is a caller-owned sink.  Keeping it ``None`` preserves the
        original allocation and random-number path; the lists below are built
        only for explicit evidence capture.
        """
        end = self.before(cutoff)
        if end < MIN_POOL:
            if evidence is not None:
                evidence.clear()
                evidence.update({
                    "schema_version": "residual_draw_evidence.v1",
                    "status": "refused",
                    "refusal": "THIN_RESIDUAL_POOL",
                    "cutoff": pd.Timestamp(cutoff).isoformat(),
                    "cutoff_index": end,
                    "prediction": float(prediction),
                    "bucket_count": self._buckets,
                    "bucket_index": None,
                    "bucket_edges": [],
                    "causal_indices": list(range(end)),
                    "eligible_indices": [],
                    "fallback_used": False,
                    "fallback_indices": [],
                    "selected_indices": [],
                })
            return np.empty(0), np.empty(0)
        pred = self._pred[:end]
        # Deciles from the pool that exists AT THIS CUTOFF, never the whole
        # history — that would be a leak wearing a full-sample refit's hat.
        edges = np.quantile(pred, np.linspace(0, 1, self._buckets + 1)[1:-1])
        index = int(np.searchsorted(edges, prediction, side="right"))
        rows = np.flatnonzero(np.searchsorted(edges, pred, side="right") == index)
        eligible = rows
        fallback_used = rows.size < MIN_POOL
        if fallback_used:
            rows = np.arange(end)
        chosen = rows[rng.integers(0, rows.size, size=n)]
        if evidence is not None:
            evidence.clear()
            evidence.update({
                "schema_version": "residual_draw_evidence.v1",
                "status": "selected",
                "refusal": None,
                "cutoff": pd.Timestamp(cutoff).isoformat(),
                "cutoff_index": end,
                "prediction": float(prediction),
                "bucket_count": self._buckets,
                "bucket_index": index,
                "bucket_edges": edges.tolist(),
                "causal_indices": list(range(end)),
                "eligible_indices": eligible.tolist(),
                "fallback_used": fallback_used,
                "fallback_indices": rows.tolist() if fallback_used else [],
                "selected_indices": chosen.tolist(),
            })
        return self._move[chosen], self._crush[chosen]


def expected_pnl(
    *,
    exit_legs,
    spot: float,
    entry_cost: float,
    pre_iv30: float,
    pred_abs_move: float,
    pred_iv_crush: float,
    dte_exit: float,
    event_date,
    pool: ResidualPool,
    key: str = "",
    draws: int = DRAWS,
    evidence: dict | None = None,
) -> dict | None:
    """Expected return, win probability and band, or ``None`` if unsimulable.

    ``None`` is the third outcome and it is load-bearing: an event with no
    forecast, no pre-print vol or too thin a pool is UNDETERMINED, never
    rejected. Collapsing "we could not tell" into "no" is how a data gap becomes
    a silent permanent decline that looks like a decision.
    """
    if evidence is not None:
        evidence.clear()
        evidence.update({
            "schema_version": "expected_pnl_evidence.v1",
            "status": "pending",
            "refusal": None,
            "event_date": pd.Timestamp(event_date).isoformat(),
            "draw_count": int(draws),
        })
    if not exit_legs:
        if evidence is not None:
            evidence.update({"status": "refused", "refusal": "MISSING_EXIT_LEGS"})
        return None
    values = (spot, entry_cost, pre_iv30, pred_abs_move, pred_iv_crush, dte_exit)
    if not all(v is not None and np.isfinite(v) for v in values):
        if evidence is not None:
            evidence.update({"status": "refused", "refusal": "NONFINITE_INPUT"})
        return None
    if pre_iv30 <= 0 or entry_cost <= 0 or dte_exit < 0:
        if evidence is not None:
            evidence.update({"status": "refused", "refusal": "INVALID_INPUT"})
        return None

    # SHA-256, not hash(): Python salts string hashing per process, and the
    # first implementation drew different samples on every run — 7 events and
    # 0.26pp of mean apart, which is this estimator's noise floor.
    seed_material = f"{key}|{event_date}"
    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    if evidence is None:
        err_move, err_crush = pool.draw(event_date, pred_abs_move, draws, rng)
    else:
        draw_evidence = {}
        err_move, err_crush = pool.draw(
            event_date,
            pred_abs_move,
            draws,
            rng,
            evidence=draw_evidence,
        )
        evidence.update({
            "seed": seed,
            "seed_algorithm": "sha256-first-8-bytes-big-endian",
            "seed_material": seed_material,
            "residual_draw": draw_evidence,
        })
    if err_move.size == 0:
        if evidence is not None:
            evidence.update({"status": "refused", "refusal": "THIN_RESIDUAL_POOL"})
        return None

    move = np.maximum(pred_abs_move + err_move, 0.0)
    crush = pred_iv_crush + err_crush
    # The sign is drawn rather than modelled, and that is EXACT rather than an
    # approximation: TWIN-P5 is symmetric about its anchor, so its payoff
    # depends on |move| only.
    sign = rng.choice((-1.0, 1.0), size=move.size)
    spot_exit = spot * np.maximum(1.0 + sign * move / 100.0, MIN_SPOT_FRACTION)
    vol_exit = (pre_iv30 / 100.0) * (1.0 + crush / 100.0)

    value = np.zeros(move.size)
    for leg in exit_legs:
        strike, qty = leg.get("strike"), float(leg.get("qty", 0.0))
        if strike is None or not np.isfinite(float(strike)) or qty == 0:
            continue
        # `sell` at exit means the position is LONG that leg and receives its
        # value; `buy` closes a short and pays it.
        side = 1.0 if str(leg.get("side", "")).lower() == "sell" else -1.0
        value += side * qty * black_scholes_put(spot_exit, float(strike), dte_exit / 365.0, vol_exit)

    ret = (value - entry_cost) / entry_cost
    result = {
        "exp_pnl_sim": float(np.mean(ret)),
        "win_sim": float(np.mean(ret > 0)),
        "sim_p10": float(np.quantile(ret, 0.10)),
        "sim_p90": float(np.quantile(ret, 0.90)),
        "pool_n": int(pool.before(event_date)),
    }
    if evidence is not None:
        evidence.update({"status": "completed", "refusal": None})
    return result


#: Where the gate's trailing history lives. Model OUTPUT, like Tier 4, so it
#: sits beside the panel rather than inside it — Tier 3 is a deterministic
#: function of Tier 2 and `data_snapshot` pins it, and a simulated expectation
#: in there would make a champion promotion invalidate experiments that never
#: read one.
HISTORY_PATH = "data/features/pnl_sim_history.parquet"


def load_history(path: str | None = None) -> pd.DataFrame | None:
    """The stored ``exp_pnl_sim`` series the cutoff is computed from.

    ``None`` when it has never been built, which makes every gate verdict
    UNDETERMINED rather than admitting or rejecting on a bar that does not
    exist. Seeded from EXP-129's simulated universe (2,802 events, 2018-2026)
    and extended by the replay as new events price.
    """
    from pathlib import Path
    from engine import paths

    target = Path(path) if path else paths.ROOT / HISTORY_PATH
    if not target.exists():
        return None
    frame = pd.read_parquet(target)
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    return frame


def trailing_cutoff(history: pd.DataFrame, as_of, *, window_months: int = WINDOW_MONTHS,
                    quantile: float = QUANTILE, min_window: int = 100) -> float | None:
    """The bar an event must clear: the top ``quantile`` of the trailing window.

    ``history`` needs ``event_date`` and ``exp_pnl_sim``. The window is
    ``[as_of - window_months, as_of)`` — strictly before, so an event is never
    ranked against itself or anything later. ``None`` when the window is too
    thin, which makes the event UNDETERMINED rather than admitted by default.
    """
    if history is None or history.empty:
        return None
    start = pd.Timestamp(as_of).to_period("M").to_timestamp()
    window_start = start - pd.DateOffset(months=window_months)
    dates = pd.to_datetime(history["event_date"])
    prior = history[(dates >= window_start) & (dates < start)]["exp_pnl_sim"].dropna()
    if len(prior) < min_window:
        return None
    return float(np.quantile(prior.to_numpy(dtype=float), 1.0 - quantile))
