"""Native score-request identity must reproduce legacy's bootstrap seed.

Legacy (``engine/score.py``) seeds every bootstrap from
``sha256(f"{snapshot}|{request.key()}")``. The native pipeline rebuilds that
seed in ``engine.v2.scoring.stages._model_seed`` from request-level identity
captured into the score context; a rendering divergence would silently draw a
different Monte Carlo path for the same trade. These tests pin the native
functions to the legacy original across nulls, zeros, empty strings, four
decimal rounding and structure parameters -- including the one optional field.
"""
from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from engine.fills import MID, FillModel
from engine.score import ScoreRequest
from engine.v2.scoring.identity import bootstrap_seed, score_request_key


def _context_from_request(request, snapshot):
    def _date_or_none(value):
        return None if value is None else str(pd.Timestamp(value).date())

    context = {
        "ticker": request.ticker,
        "strategy": request.strategy,
        "snapshot": snapshot,
        "requested_as_of": _date_or_none(request.as_of),
        "requested_event_date": _date_or_none(request.event_date),
        "requested_strike": request.strike,
        "requested_expiry": _date_or_none(request.expiry),
        "fill_alpha": request.fill.alpha,
        "variant": request.variant,
        "decision_offset": request.decision_offset,
        "quote_max_age_sessions": request.quote_max_age_sessions,
        "chain_as_of": _date_or_none(request.chain_as_of),
    }
    if request.structure_params:
        context["requested_structure_params"] = dict(request.structure_params)
    return context


def _legacy_seed(request, snapshot):
    return int.from_bytes(
        hashlib.sha256(f"{snapshot}|{request.key()}".encode()).digest()[:8], "big"
    )


def _assert_matches_legacy(request, snapshot):
    context = _context_from_request(request, snapshot)
    key = score_request_key(context)
    assert key == request.key()
    assert bootstrap_seed(snapshot, key) == _legacy_seed(request, snapshot)


def _minimal_request(**changes):
    return ScoreRequest(ticker="AAPL", strategy="STR-THRU", **changes)


@pytest.mark.parametrize(
    ("score_request", "snapshot"),
    [
        pytest.param(_minimal_request(), "snap-abc", id="minimal"),
        pytest.param(
            ScoreRequest(
                ticker="MSFT",
                strategy="TWIN-P",
                as_of=pd.Timestamp("2026-01-15"),
                event_date=pd.Timestamp("2026-01-20"),
                strike=123.456789,
                expiry=pd.Timestamp("2026-02-20"),
                fill=FillModel(0.73),
                variant="t2",
                decision_offset=-2,
                quote_max_age_sessions=3,
                chain_as_of=pd.Timestamp("2026-01-14"),
                structure_params={"width": 0.05, "tent": 3},
            ),
            "snap-xyz",
            id="fully-populated",
        ),
        pytest.param(_minimal_request(decision_offset=0), "snap-abc",
                     id="decision-offset-zero"),
        pytest.param(_minimal_request(decision_offset=5), "snap-abc",
                     id="decision-offset-five"),
        pytest.param(_minimal_request(variant=""), "snap-abc", id="variant-empty"),
        pytest.param(_minimal_request(strike=95.5), "snap-abc", id="strike-only"),
        pytest.param(_minimal_request(expiry=pd.Timestamp("2026-03-20")), "snap-abc",
                     id="expiry-only"),
        pytest.param(_minimal_request(), "", id="empty-snapshot"),
    ],
)
def test_native_key_and_seed_match_legacy(score_request, snapshot):
    _assert_matches_legacy(score_request, snapshot)


def test_default_fill_is_mid_and_renders_four_decimals():
    request = _minimal_request()
    assert request.fill == MID
    key = score_request_key(_context_from_request(request, "snap-abc"))
    assert "|0.5000|" in key
    assert key == request.key()


@pytest.mark.parametrize("offset", [0, 5])
def test_decision_offset_renders_signed_and_never_empty(offset):
    request = _minimal_request(decision_offset=offset)
    key = score_request_key(_context_from_request(request, "snap-abc"))
    assert f"|d+{offset}|" in key
    assert key == request.key()


def test_empty_variant_and_none_variant_render_identically():
    with_empty = _minimal_request(variant="")
    with_none = _minimal_request(variant=None)
    key_empty = score_request_key(_context_from_request(with_empty, "snap-abc"))
    key_none = score_request_key(_context_from_request(with_none, "snap-abc"))
    assert key_empty == key_none
    assert key_empty == with_empty.key()


def test_strike_and_expiry_render_independently():
    strike_only = _minimal_request(strike=95.5)
    expiry_only = _minimal_request(expiry=pd.Timestamp("2026-03-20"))
    no_strike = score_request_key(_context_from_request(strike_only, "snap-abc"))
    no_expiry = score_request_key(_context_from_request(expiry_only, "snap-abc"))
    assert no_strike == strike_only.key()
    assert "95.5000||" in no_strike
    assert no_expiry == expiry_only.key()
    assert "|2026-03-20|" in no_expiry
    assert no_strike != no_expiry


def test_absent_required_field_raises_keyerror():
    context = _context_from_request(_minimal_request(), "snap-abc")
    del context["requested_as_of"]
    with pytest.raises(KeyError, match="requested_as_of"):
        score_request_key(context)


def test_absent_structure_params_matches_present_none():
    request = _minimal_request(strike=95.5)
    absent = _context_from_request(request, "snap-abc")
    assert "requested_structure_params" not in absent
    present_none = dict(absent, requested_structure_params=None)
    assert score_request_key(absent) == score_request_key(present_none)
    assert score_request_key(absent) == request.key()
    assert score_request_key(present_none).endswith("|")
