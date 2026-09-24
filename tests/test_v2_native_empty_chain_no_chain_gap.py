"""Phase 4 parity gap (Tier-0 fixtures 002 DYN-SV LUXE member 2, 019 DYN-SV
LEN member 2): an OBSERVED EMPTY quote lookup must classify as NO_CHAIN.

Root cause: ``engine/score.py::_price_entry`` flags ``NO_CHAIN`` the moment
the chain rows come back empty (engine/score.py:2347-2351), BEFORE any expiry
resolution, and the strict capture records that ran-and-found-nothing lookup
as an explicitly empty ``quotes`` mapping, accepted only because legacy said
WHY (``tools/capture_tier0_corpus.py::_quote_map``: ``quote_status="empty"``;
a never-recorded empty domain is refused at capture, and a ``"not_reached"``
row is refused upstream and never reaches the geometry gate with a spot).
Native's geometry gate (``engine/v2/scoring/stages.py::_resolve_geometry``)
only asked whether an expiry was resolvable at all, so it stamped
``MISSING_EXPIRY`` for those rows -- conflating "the lookup ran and saw
nothing" (legacy's ``NO_CHAIN``) with "quotes exist but carry no eligible
expiration".

The fix is a classification keyed off the capture evidence itself (was a
``quotes`` domain recorded, and is it empty?), never off ticker/fixture/date:
observed-empty -> ``NO_CHAIN``; non-empty with nothing selectable ->
``MISSING_EXPIRY``; missing or malformed evidence -> fail closed on
``MISSING_EXPIRY``, exactly as before (the controls below pin the behavior
the 2026-09-24 ``has_resolvable_expiry`` gate landed with).
"""
from engine.v2.scoring.stages import (
    STAGE_NAMES,
    NativeScoreInputs,
    StageReceipt,
    _observed_empty_quote_lookup,
    assemble_native_values,
)

_RECEIPTS = tuple(
    StageReceipt(stage, "declared-input", "declared-output")
    for stage in STAGE_NAMES if stage != "diagnostics"
)
_NO_QUOTES = object()


def _inputs(*, strategy="STR-THRU", quotes=_NO_QUOTES):
    """A captured-shaped row legacy never priced: spot present (a DYN-SV
    menu member's own context), no expiry, and a quote domain that is empty
    BECAUSE the lookup ran and found nothing."""
    context = {
        "ticker": "AAA", "strategy": strategy, "event_date": "2026-09-16",
        "entry_date": "2026-09-16", "exit_date": "2026-09-18",
        "session": "AMC", "spot": 100.0,
    }
    if quotes is not _NO_QUOTES:
        context["quotes"] = quotes
    return NativeScoreInputs(
        context=context,
        features={"model_inputs": {}},
        forecast={"driver_name": "abs_move",
                  "models": {"driver_prediction": {"intercept": 0.0,
                                                   "coefficients": {}}}},
        geometry=None,
        pricing=None,
        analogs={"recipe": None},
        simulation={"mode": "not_applicable"},
        gate={"mode": "not_applicable"},
        chooser={},
        diagnostics={},
        source_ref="empty-chain-no-chain-gap",
        stage_receipts=_RECEIPTS,
    )


def _flags(strategy="STR-THRU", quotes=_NO_QUOTES):
    return set(assemble_native_values(
        _inputs(strategy=strategy, quotes=quotes), strategy=strategy,
    )["flags"])


# -- the evidence reader ------------------------------------------------------


def test_evidence_reader_distinguishes_recorded_from_absent_lookup():
    assert _observed_empty_quote_lookup(_inputs(quotes={})) is True
    assert _observed_empty_quote_lookup(_inputs(
        quotes={("P", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 2.0}},
    )) is False
    # Missing or malformed capture evidence is NOT an observed empty domain.
    assert _observed_empty_quote_lookup(_inputs(quotes=None)) is False
    assert _observed_empty_quote_lookup(_inputs(quotes=[("P", 100.0)])) is False
    assert _observed_empty_quote_lookup(_inputs(quotes=_NO_QUOTES)) is False


# -- observed empty domain -> legacy-equivalent NO_CHAIN ----------------------


def test_observed_empty_domain_refuses_no_chain_not_missing_expiry():
    flags = _flags(quotes={})
    assert "NO_CHAIN" in flags
    assert "MISSING_EXPIRY" not in flags


def test_dyn_sv_member_with_observed_empty_domain_refuses_no_chain():
    """The fixture 002/019 member shape: a DYN-SV menu member whose capture
    recorded the empty lookup. The gate fires for DYN-SV like any enabled
    strategy, so the classification -- not a strategy exemption -- is what
    produces legacy's NO_CHAIN here."""
    flags = _flags(strategy="DYN-SV", quotes={})
    assert "NO_CHAIN" in flags
    assert "MISSING_EXPIRY" not in flags


def test_no_chain_row_still_reaches_every_stage_and_stays_unpriced():
    """Downstream reachability: the geometry refusal carries through pricing
    (all stage receipts still emit, execution graph whole) and the row is
    never priced."""
    values = assemble_native_values(_inputs(quotes={}), strategy="STR-THRU")
    flags = set(values["flags"])
    assert "NO_CHAIN" in flags
    assert {row["stage"] for row in values["native_stage_receipts"]} \
        == set(STAGE_NAMES)
    assert not values.get("legs")
    assert values.get("entry_cost") is None


# -- controls: MISSING_EXPIRY keeps its genuine cases --------------------------


def test_nonempty_quotes_without_eligible_expiry_keep_missing_expiry():
    """Quotes present but no common call/put expiration to trade: the
    observed chain is NOT empty, so this stays MISSING_EXPIRY, never an
    inferred NO_CHAIN."""
    flags = _flags(quotes={("C", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 2.0}})
    assert "MISSING_EXPIRY" in flags
    assert "NO_CHAIN" not in flags


def test_absent_capture_evidence_stays_fail_closed_missing_expiry():
    """The previously landed negative control (a bundle that recorded no
    quote lookup at all): unchanged MISSING_EXPIRY."""
    flags = _flags()
    assert "MISSING_EXPIRY" in flags
    assert "NO_CHAIN" not in flags


def test_malformed_capture_evidence_stays_fail_closed_missing_expiry():
    """A non-mapping under the ``quotes`` key is a capture defect, not an
    observed empty domain: it must not earn the (weaker-claiming) NO_CHAIN
    classification."""
    flags = _flags(quotes=None)
    assert "MISSING_EXPIRY" in flags
    assert "NO_CHAIN" not in flags
