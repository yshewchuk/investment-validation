"""Native-path parity row inputs, extracted as a leaf module.

Holds :func:`legacy_parity_rows`, which keys a legacy ``score.json``
document's own ``rows`` array by
``engine.v2.foundation.score_population.population_key``.  It depends only on
``engine.v2.foundation.score_population`` and ``engine.v2.ops.errors``, never
on ``engine.v2.ops.nightly`` or ``engine.v2.ops.native_parity_report``.
"""
from __future__ import annotations

from typing import Any, Mapping

from engine.v2.foundation.score_population import population_key
from engine.v2.ops.errors import fail


def legacy_parity_rows(score_document: Mapping[str, Any]) -> dict[str, dict]:
    """Key a legacy ``score.json`` document's own ``rows`` array by
    ``engine.v2.foundation.score_population.population_key``'s
    ``"ticker|strategy|event_date"`` format.

    Pure: no filesystem, no clock. A ``score_document`` with no ``"rows"``
    key returns ``{}`` (not a refusal) -- ``native_parity_handler``'s own
    ``_refuse_empty_inputs`` already turns an empty ``legacy_rows`` into a
    ``VALIDATION_FAILED`` refusal downstream.

    Does NOT reuse ``population_key``'s own ``.get(key, "")`` empty-string
    substitution for a missing ``ticker``/``strategy``/``event_date``: every
    row's three key fields are validated present and non-empty before any
    key is computed, and any two rows sharing one ``population_key`` value
    are rejected as a collision -- never silently keyed by whichever key a
    dict comprehension iterates last. Every one of these checks is a
    batch-level refusal for the WHOLE call, raised before any dict is
    constructed -- never a per-row skip.

    Raises ``engine.v2.ops.errors.fail("VALIDATION_FAILED", ...)`` (an
    ``OpsError``) when:
    - ``score_document["rows"]`` is present but is not a list;
    - any element of that list is not a mapping;
    - any row is missing, or has a non-string or empty/falsy, ``ticker``,
      ``strategy``, or ``event_date``;
    - any row's ``ticker`` or ``strategy`` contains the ``"|"``
      ``population_key`` delimiter;
    - two rows produce the same ``population_key`` value.
    """
    if "rows" not in score_document:
        return {}
    rows = score_document["rows"]
    if not isinstance(rows, list):
        raise fail("VALIDATION_FAILED", "legacy parity rows is not a list",
                   details={"type": type(rows).__name__})
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise fail("VALIDATION_FAILED", "legacy parity row is not a mapping",
                       details={"index": index, "type": type(row).__name__})
    required = ("ticker", "strategy", "event_date")
    delimiter_checked = ("ticker", "strategy")
    for index, row in enumerate(rows):
        for field_name in required:
            value = row.get(field_name)
            if not isinstance(value, str) or not value:
                raise fail("VALIDATION_FAILED", "legacy parity row missing required field",
                           details={"index": index, "field": field_name})
            if field_name in delimiter_checked and "|" in value:
                raise fail("VALIDATION_FAILED",
                           "legacy parity row field contains the population key delimiter",
                           details={"index": index, "field": field_name})
    keys = [population_key(row) for row in rows]
    seen: dict[str, int] = {}
    for index, key in enumerate(keys):
        if key in seen:
            raise fail("VALIDATION_FAILED", "legacy parity rows have duplicate population key",
                       details={"population_key": key, "indices": [seen[key], index]})
        seen[key] = index
    return dict(zip(keys, rows))
