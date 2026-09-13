"""Pure decision-replay helpers: eligible population and row-diff.

``guides/rearchitecture_phase1_operations.md`` §5 defines replay as invoking
computation from frozen inputs in a fresh process — never comparing stored
output to a copy of itself. This module holds only the legacy-free logic that
serves that definition: which board rows are decision-eligible, and how two
scored rows disagree. It imports nothing from ``engine.*`` (the legacy tree)
— only other v2/ops modules.

``engine.v2.ops.legacy_adapter._action_decision_replay`` is the one caller
that does the actual re-scoring — through ``engine.score.score_calendar``,
the SAME public entrypoint the score stage used, never a hand-rebuilt
``ScoreRequest``. A DYN-SV chooser row is not a request the engine can score
on its own (``DYN-SV`` names no structure); the only faithful replay of one
is re-running the whole menu through the calendar's own chooser and letting
it pick again. That step needs the legacy engine and lives in
``legacy_adapter.py``, not here.
"""
from __future__ import annotations

from engine.v2.ops.decision_validation import population_key
from engine.v2.ops.errors import fail

__all__ = ["compare_rows", "decision_population", "population_key"]

#: Fields ``engine.v2.ops.legacy_adapter._action_score`` adds to a row AFTER
#: scoring, on top of whatever ``ScoreResult.as_dict()`` produced:
#: - ``row_id``: computed from the row itself, in ``_score_row_id``.
#: - ``strike_offset``: appended by ``score_calendar`` for every row (always
#:   ``None`` here, since both the score stage and the replay call it with
#:   ``alt_strikes=0``).
#: Neither is evidence the SCORER produced — both are the calendar's own
#: bookkeeping, recomputed identically by any caller and never a genuine
#: content disagreement. Replay now runs the real ``score_calendar`` (never a
#: hand-rebuilt request), so a DYN-SV row's ``strategy``, ``detail``,
#: ``chosen_strategy``, ``chosen_margin`` and ``menu_size`` are all genuinely
#: recomputed by the same chooser the board used, and stay compared.
_ADDED_FIELDS = frozenset({"row_id", "strike_offset"})


def _date_only(value):
    """The date portion of a stored ISO date/datetime string, or ``None``."""
    if value is None:
        return None
    text = str(value)
    return text[:10] if len(text) >= 10 else text or None


def decision_population(score_doc, session):
    """The rows ``build_prediction_rows(..., entry_dated_only=True)`` would
    record from this score document, sorted by population key.

    Mirrors ``engine/ledger.py::build_prediction_rows``'s own filter exactly:
    a row is eligible only when ``as_of`` (or, failing that, ``decision_date``
    then ``entry_date`` — the same fallback chain) names the replay session.
    Board rows never carry ``decision_date``, so in practice this reads
    ``as_of``, but the fallback is kept so a synthetic frame that does is
    still read the way the ledger reads it.

    Also excludes ladder rows (``strike_offset`` not ``None``) the same way
    ``engine/v2/ops/decision_validation.py::_validate_candidate`` refuses
    them — a ladder row is an alternative strike for a structure, not a
    decision to trade, and never belongs in this population.

    Duplicate population keys among the eligible rows fail loudly: a
    replayable population is keyed uniquely by construction, and a collision
    means the score document is malformed, not that one candidate should
    silently win.
    """
    rows = score_doc.get("rows") if isinstance(score_doc, dict) else score_doc
    if not isinstance(rows, list):
        raise fail("VALIDATION_FAILED", "score document has no row list")
    session_date = _date_only(session)
    eligible = []
    for row in rows:
        if not isinstance(row, dict) or row.get("strike_offset") is not None:
            continue
        decision_date = row.get("as_of") or row.get("decision_date") or row.get("entry_date")
        if decision_date is None or _date_only(decision_date) != session_date:
            continue
        eligible.append(row)
    keys = [population_key(row) for row in eligible]
    if len(set(keys)) != len(keys):
        duplicated = sorted({key for key in keys if keys.count(key) > 1})
        raise fail("VALIDATION_FAILED", "decision population has duplicate keys",
                   details={"keys": duplicated})
    return sorted(eligible, key=population_key)


def compare_rows(source_rows, replayed_rows):
    """Findings where a source row and its replay disagree, by population key.

    Exact canonical-JSON comparison over every field except ``_ADDED_FIELDS``.
    A source key with no matching replayed row is itself a finding (the
    caller selects ``replayed_rows`` by population key before calling this,
    so a key the fresh calendar run never produced shows up here as missing,
    never as a silently-empty pass). A finding names the population key and
    the field path — never the value on either side, so a finding can be
    logged and reported without leaking a private score.
    """
    from engine.v2.foundation import canonical_json

    source = {population_key(row): row for row in source_rows}
    replayed = {population_key(row): row for row in replayed_rows}
    findings = []
    for key in sorted(set(source) | set(replayed)):
        left, right = source.get(key), replayed.get(key)
        if left is None or right is None:
            findings.append({"key": key, "field": "<row>", "reason": "row_missing"})
            continue
        for field in sorted((set(left) | set(right)) - _ADDED_FIELDS):
            if canonical_json(left.get(field)) != canonical_json(right.get(field)):
                findings.append({"key": key, "field": field, "reason": "value_mismatch"})
    return findings
