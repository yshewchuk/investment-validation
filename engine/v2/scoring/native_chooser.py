"""The DYN-SV chooser's feature vector from declared primitive inputs.

``engine/score.py`` ``Scorer._chooser_frame`` assembles the champion's 67
inputs. :func:`derive_chooser_columns` assembles the 50 it computes, from the
native scoring pass and the chooser block's declared frozen state; the other
17 (event history, market and regime block, ``dte_entry``) are primitive
features the bundle declares in ``feature_vector``:

* the direct columns (``exp_pnl_sim``, ``exp_pnl_sim_select``, the ``is_*``
  one-hots, ``quote_repaired``, ``wide_market``) and ``entry_cost_pct``;
* the sizing forecast and its band, from ``fold_pools["pred_abs_move"]``
  (the served size fold's ``pool_pred``/``pool_res``/``interval_floor``);
* the geometry ratios, ``rel_spread`` and the payoff schematics, from the
  priced legs, spot and entry cost;
* ``n_admissible``: the chain depth of the priced quote domain, looked up in
  the frozen ``AdmissibleDepthTable`` the recipe pins;
* the k-NN ``analog_*`` block, from the frozen ``ChooserAnalogPoolArtifact``
  the recipe pins;
* the ``pred_im_t1_d14``/``pred_runup_abs_move_d14`` producers: their
  Tier-4 fold bindings (``producers``) and pools (``fold_pools``);
* ``tier4_forecast_edge``: the sizing forecast minus ``or_implied``.

A frozen state the recipe declares but that is missing or carries another
key is ``MODEL_NOT_READY``, never a silent substitute. A state the recipe
does not declare leaves its columns NaN -- legacy without the pool file or
fold -- and the chooser then declines ``CHOOSER_MISSING_FEATURES`` unless the
bundle declared those columns in ``feature_vector`` (the compatibility path).
"""
from __future__ import annotations

from typing import Any, Mapping

from engine.v2.models.admissible_table import AdmissibleDepthTable, n_admissible_for
from engine.v2.models.chooser_analog_pool import (
    ChooserAnalogPoolArtifact,
    chooser_analog_pool_key,
)
from engine.v2.scoring import native_chooser_features as columns
from engine.v2.scoring.native_gate_features import chooser_direct_columns

__all__ = [
    "ADMISSIBLE_KEY_FIELD",
    "ADMISSIBLE_TABLE_FIELD",
    "ANALOG_KEY_FIELD",
    "ANALOG_POOL_FIELD",
    "FOLD_POOLS_FIELD",
    "FOLD_POOL_HASHES_FIELD",
    "PRODUCERS_FIELD",
    "derive_chooser_columns",
    "identity_view",
]

MODEL_NOT_READY = "MODEL_NOT_READY"
#: Chooser-block fields (see ``source_inputs._chooser_block``).
PRODUCERS_FIELD = "producers"
FOLD_POOLS_FIELD = "fold_pools"
#: Content hash of each declared fold pool, computed once at build time.
FOLD_POOL_HASHES_FIELD = "fold_pool_hashes"
ANALOG_POOL_FIELD = "analog_pool"
ANALOG_KEY_FIELD = "analog_pool_key"
ADMISSIBLE_TABLE_FIELD = "admissible_table"
ADMISSIBLE_KEY_FIELD = "admissible_table_key"

_NAN = float("nan")


class _NotReady(Exception):
    """A declared frozen state cannot serve this request."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        super().__init__(",".join(reasons))
        self.reasons = reasons


def _declared_state(block: Mapping[str, Any], state_field: str, key_field: str,
                    kind: type, key_of) -> Any:
    """The declared frozen state after its causal-key check, or ``None``
    when the recipe declares none. Mirrors ``native_residuals._mismatch``:
    every key part must be declared and equal, and a pinned content hash
    must match."""
    state, expected = block.get(state_field), block.get(key_field)
    if state is None and not expected:
        return None
    if not isinstance(state, kind) or not isinstance(expected, Mapping):
        raise _NotReady((MODEL_NOT_READY,))
    try:
        wanted = key_of(expected)
    except KeyError as exc:
        raise _NotReady((MODEL_NOT_READY,)) from exc
    pinned = expected.get("content_hash")
    if state.key != wanted or (pinned is not None and pinned != state.content_hash):
        raise _NotReady((MODEL_NOT_READY,))
    return state


def _analog_key(expected: Mapping[str, Any]):
    return chooser_analog_pool_key(expected["pool_id"], expected["cutoff"])


def _table_key(expected: Mapping[str, Any]):
    return (str(expected["table_id"]), str(expected["version"]))


def _producer_prediction(executor: Any, facts: Mapping[str, Any]) -> float | None:
    """One Tier-4 fold's prediction; ``None`` where legacy serves NaN (a
    missing or non-finite feature: ``ServingModel.predict``, or the
    ``any(f not in features.columns ...)`` skip)."""
    try:
        return next(iter(executor.predict(facts).values()), None)
    except (TypeError, ValueError, KeyError) as exc:
        reasons = tuple(getattr(exc, "reason_codes", ()) or ())
        if "MISSING_FEATURES" in reasons:
            return None
        raise _NotReady(reasons or ("INVALID_CHOOSER_PRODUCER",)) from exc


def _as_float(value: Any) -> float:
    if value is None:
        return _NAN
    try:
        return float(value)
    except (TypeError, ValueError):
        return _NAN


def _pricing_columns(values: Mapping[str, Any], m: float,
                     band: Mapping[str, float]) -> dict[str, float]:
    legs, spot = values.get("legs") or (), values.get("spot")
    out = dict(columns.geometry_columns(legs, spot, m))
    out["rel_spread"] = columns.rel_spread(legs)
    out.update(columns.schematic_columns(
        legs, spot, values.get("entry_cost"), m, band["pred_abs_move_sd"]))
    return out


def _n_admissible(block, quotes, values, m: float, s: float) -> float:
    table = _declared_state(block, ADMISSIBLE_TABLE_FIELD, ADMISSIBLE_KEY_FIELD,
                            AdmissibleDepthTable, _table_key)
    if table is None:
        return _NAN
    legs = values.get("legs") or ()
    expiry = legs[0].get("expiry") if legs else None
    depth = columns.chain_depth(quotes, expiry, values.get("spot"), m, s)
    return n_admissible_for(table, depth)


def _producers(block, facts) -> dict[str, float]:
    executors = block.get(PRODUCERS_FIELD) or {}
    pools = block.get(FOLD_POOLS_FIELD) or {}
    out: dict[str, float] = {}
    for output in columns.PRODUCER_COLUMNS:
        executor = executors.get(output)
        prediction = None if executor is None else _producer_prediction(executor, facts)
        out.update(columns.producer_columns(output, prediction, pools.get(output)))
    return out


def _derive(block, facts, strategy, values, quotes, flags) -> dict[str, float]:
    out = chooser_direct_columns(strategy, values, flags)
    out["entry_cost_pct"] = columns.entry_cost_pct(values.get("entry_cost"),
                                                   values.get("spot"))
    m = _as_float(values.get("forecast_abs_move"))
    band = columns.size_band_columns(m, (block.get(FOLD_POOLS_FIELD) or {})
                                     .get("pred_abs_move"))
    out.update(band)
    out.update(_pricing_columns(values, m, band))
    out["n_admissible"] = _n_admissible(block, quotes, values, m,
                                        band["pred_abs_move_sd"])
    pool = _declared_state(block, ANALOG_POOL_FIELD, ANALOG_KEY_FIELD,
                           ChooserAnalogPoolArtifact, _analog_key)
    out.update(columns.knn_analog_columns(pool, strategy, facts.get("entry_date"), out))
    out.update(_producers(block, facts))
    out["tier4_forecast_edge"] = columns.forecast_edge(
        m, _as_float(facts.get("or_implied")))
    return out


def derive_chooser_columns(block: Mapping[str, Any], facts: Mapping[str, Any],
                           strategy: str, values: Mapping[str, Any],
                           quotes: Mapping[Any, Any] | None,
                           flags: list[str]) -> dict[str, float] | None:
    """The chooser's derived columns, or ``None`` after flagging the reason
    a declared frozen input cannot serve this request."""
    try:
        return _derive(block, facts, strategy, values, quotes, flags)
    except _NotReady as exc:
        for reason in exc.reasons:
            if reason not in flags:
                flags.append(reason)
        return None


def identity_view(block: Mapping[str, Any]) -> dict[str, Any]:
    """The chooser block's frozen inputs by identity, for stage receipts."""
    view: dict[str, Any] = {}
    for name in (PRODUCERS_FIELD,):
        view[name] = {key: str(value) for key, value in (block.get(name) or {}).items()}
    for name in (ANALOG_POOL_FIELD, ADMISSIBLE_TABLE_FIELD):
        state = block.get(name)
        view[name] = None if state is None else str(state)
    for name in (ANALOG_KEY_FIELD, ADMISSIBLE_KEY_FIELD):
        view[name] = dict(block.get(name) or {})
    view[FOLD_POOLS_FIELD] = dict(block.get(FOLD_POOL_HASHES_FIELD) or {})
    return view
