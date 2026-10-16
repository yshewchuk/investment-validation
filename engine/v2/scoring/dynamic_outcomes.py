"""Pure, immutable accounting for one DYN-SV event's candidate outcomes.

These are caller-supplied summaries, not proof of record provenance, source
admission or rankability. There is no record lookup, scoring or selection here.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

from engine.v2.registry import DYNAMIC_MENU

__all__ = [
    "DYNAMIC_MENU_REFUSAL_CODES", "DynamicCandidateOutcome", "DynamicMenuIdentity",
    "DynamicMenuRefusal", "validate_dynamic_menu",
]

DYNAMIC_MENU_REFUSAL_CODES = frozenset({
    "INVALID_CHOOSER_MENU", "CHOOSER_MENU_INCOMPLETE",
})
_INVALID_DETAIL = "candidate outcomes do not form one valid event menu"


class DynamicMenuRefusal(ValueError):
    """A malformed join or incomplete barrier, never a candidate exclusion."""

    def __init__(self, code: str, *, missing_members: tuple[str, ...] = ()) -> None:
        if code not in DYNAMIC_MENU_REFUSAL_CODES:
            raise ValueError("unknown dynamic menu refusal code")
        self.code = code
        self.missing_members = missing_members
        self.detail = (_INVALID_DETAIL if code == "INVALID_CHOOSER_MENU" else
                       "missing candidate outcomes: " + ", ".join(missing_members))
        super().__init__(f"{code}: {self.detail}")


def _invalid() -> None:
    raise DynamicMenuRefusal("INVALID_CHOOSER_MENU")


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True, slots=True, kw_only=True)
class DynamicMenuIdentity:
    """Shared, opaque identity supplied by the caller's pinned event context.

    Dates/instants are compared exactly, never reduced to days or reconstructed.
    Fill/ATM hashes identify the complete declared policies, using the existing
    foundation content hash at the caller. No hash is verified by this helper.
    """

    ticker: str
    event_date: str
    session: str
    event_id: str
    calendar_revision: str
    snapshot_id: str
    release_id: str
    deployment_id: str
    decision_clock_id: str
    requested_decision_at: str
    mode: str
    fill_model_hash: str
    atm_policy_hash: str
    event_revision: str | None = None

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "event_revision" and value is None:
                continue
            if not _text(value):
                _invalid()


@dataclass(frozen=True, slots=True, kw_only=True)
class DynamicCandidateOutcome:
    """One actual record summary OR one explicit producer/assembly refusal.

    Record readiness/status/reasons remain verbatim, including refused records.
    A refusal may precede request construction, so its request hash is optional.
    A record summary always names both request and score. Neither variant implies
    eligibility. Reasons and warnings are detached immutable tuples of strings.
    """

    strategy: str
    identity: DynamicMenuIdentity
    request_hash: str | None = None
    score_id: str | None = None
    validation_status: str | None = None
    readiness: str | None = None
    refusal_code: str | None = None
    reason_codes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.identity, DynamicMenuIdentity) or not _text(self.strategy):
            _invalid()
        for name in ("request_hash", "score_id", "validation_status", "readiness",
                     "refusal_code"):
            value = getattr(self, name)
            if value is not None and not _text(value):
                _invalid()
        for name in ("reason_codes", "warnings"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or not all(map(_text, values)):
                _invalid()
            object.__setattr__(self, name, tuple(values))
        self._validate_variant()

    def _validate_variant(self) -> None:
        record_fields = (self.score_id, self.validation_status, self.readiness)
        if self.refusal_code is not None:
            if any(value is not None for value in record_fields):
                _invalid()
        elif self.request_hash is None or any(value is None for value in record_fields):
            _invalid()


def validate_dynamic_menu(
    expected: DynamicMenuIdentity,
    outcomes: Sequence[DynamicCandidateOutcome],
) -> tuple[DynamicCandidateOutcome, ...]:
    """Require one outcome per registered member and return registry order.

    Invalid identities, unexpected members, duplicates (including record/refusal
    overlap) and malformed shapes refuse INVALID_CHOOSER_MENU before incomplete
    accounting is considered. Missing members refuse CHOOSER_MENU_INCOMPLETE,
    with only their fixed registry names in the detail. All seven may be refused.
    The input must be a list/tuple, never a keyed mapping that hides duplicates.
    No inputs are mutated, no ranking is performed and no partial result escapes.
    """
    if (not isinstance(expected, DynamicMenuIdentity)
            or not isinstance(outcomes, (tuple, list))):
        _invalid()
    by_strategy: dict[str, DynamicCandidateOutcome] = {}
    request_ids: set[str] = set()
    score_ids: set[str] = set()
    for outcome in outcomes:
        if not isinstance(outcome, DynamicCandidateOutcome):
            _invalid()
        if (outcome.strategy not in DYNAMIC_MENU or outcome.strategy in by_strategy
                or outcome.identity != expected):
            _invalid()
        for value, seen in ((outcome.request_hash, request_ids), (outcome.score_id, score_ids)):
            if value is not None:
                if value in seen:
                    _invalid()
                seen.add(value)
        by_strategy[outcome.strategy] = outcome
    missing = tuple(member for member in DYNAMIC_MENU if member not in by_strategy)
    if missing:
        raise DynamicMenuRefusal("CHOOSER_MENU_INCOMPLETE", missing_members=missing)
    return tuple(by_strategy[member] for member in DYNAMIC_MENU)
