"""Regression test for the missing ``implied_move`` capture: the real
tier0 corpus never carried it. ``engine/score.py``'s ``Scorer._score``
computes ``implied_move`` (the market's quote NOW, at the decision date)
and ``implied_move_at_entry`` (the quote at THIS trade's own entry date)
around lines 1960-1966, but the request-scoped
``capture_source_bundle(context={...})`` call that runs unconditionally
for every row (~line 1847) fires BEFORE that point, when neither field
exists yet -- so no captured row's ``source_inputs.context`` ever held
either one, only legacy's own ``record``.

Fixed by adding a second, narrower ``capture_source_bundle`` call right
after both are computed, folding whichever of the two is not ``None``
into the same ``context`` group (``capture_source_bundle`` merges into
the persistent ``_source_bundle["context"]`` dict across calls -- see its
docstring). Neither field is a scoring answer: absent from
``Phase4TraceCollector._source_answer_fields``,
``engine/v2/scoring/source_inputs.py``'s ``_ANSWER_FIELDS``, and
``engine/v2/scoring/frozen_inputs.py``'s ``_ANSWER_FIELDS`` (the answer-free
gate's map, moved out of ``checks/phase4_frozen_bridge.py`` for Phase 6).

``tools/capture_tier0_corpus.py``'s ``_captured_blocks`` takes
``source.get("context")`` into native's ``context`` block verbatim (no
per-field allowlist), so nothing there needs to change for the new keys
to reach a captured bundle.
"""
from __future__ import annotations

from engine.score import Phase4TraceCollector


def test_trace_collector_captures_implied_move_as_a_source_fact():
    """The exact mechanism ``Scorer._score`` now uses, right after both
    quotes are computed: a second ``capture_source_bundle(context={...})``
    call that merges into the same context group the earlier
    resolve_context call already started. Proves it is not rejected as a
    scoring answer and round-trips through the collector unchanged."""
    trace = Phase4TraceCollector(retain_full_trace=False)
    trace.capture_source_bundle(
        context={
            "ticker": "AAA",
            "strategy": "STR-THRU",
            "entry_date": "2026-09-16",
            "exit_date": "2026-09-18",
        },
        quote_status="not_reached",
    )
    # The later call: both quotes now known, only the market-quote-now
    # value present (mirrors a row where the entry-date quote was too old
    # to clear MIN_QUOTED_IMPLIED_MOVE and so was never set on ``result``).
    trace.capture_source_bundle(context={"implied_move": 6.25})

    frozen_context = trace._checkpoint_groups["source_inputs"]["context"]

    assert frozen_context["implied_move"] == 6.25
    assert frozen_context["ticker"] == "AAA"  # the earlier call's facts survive


def test_trace_collector_captures_both_implied_move_quotes():
    trace = Phase4TraceCollector(retain_full_trace=False)
    trace.capture_source_bundle(
        context={"ticker": "AAA", "strategy": "STR-THRU"},
        quote_status="not_reached",
    )
    trace.capture_source_bundle(
        context={"implied_move_at_entry": 5.5, "implied_move": 6.25},
    )

    frozen_context = trace._checkpoint_groups["source_inputs"]["context"]

    assert frozen_context["implied_move_at_entry"] == 5.5
    assert frozen_context["implied_move"] == 6.25
