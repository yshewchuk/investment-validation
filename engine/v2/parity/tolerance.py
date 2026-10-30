"""Per-field tolerance policies. There is no global tolerance, by construction.

§3.2 and §7.3: exact comparison for IDs, selected contracts, integer
quantities, gate verdicts, flags, null masks and frozen canonical records,
except where a field has an explicitly declared tolerance; declared
**per-field** tolerances for independently recomputed numeric values; never a
global tolerance, and never one widened to make a test pass.

The default here is therefore *exact*, and a policy is a mapping of explicit
field patterns to explicit tolerances. That inverts the usual arrangement on
purpose: with a global default, widening it is a one-character edit that turns
a whole class of findings off silently. With this arrangement, an undeclared
float that drifts is a finding, and making it not-a-finding costs a named line
in a named policy that shows up in a diff.

Tolerances are declared from the arithmetic that produces the number, before
the first run. `rearchitecture_phase0_baseline.md` §11: "Tolerances chosen to
make the first run pass" is a listed failure mode.
"""
from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = ["Tolerance", "TolerancePolicy", "EXACT", "SCORE_RECORD_V1"]


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _nonnegative_number(value: Any) -> bool:
    return _finite_number(value) and value >= 0


def _nonempty_str_tuple(value: Any, length: int | None = None) -> bool:
    return (
        isinstance(value, tuple)
        and (length is None or len(value) == length)
        and all(isinstance(item, str) and item for item in value)
    )


def _n_analogs_matches(left_value: Any, right_value: Any) -> bool:
    return (
        type(left_value) is int
        and type(right_value) is int
        and left_value >= 0
        and right_value >= 0
        and left_value == right_value
    )


def _context_value_matches(name: str, left_value: Any, right_value: Any) -> bool:
    if name == "n_analogs":
        return _n_analogs_matches(left_value, right_value)
    return (
        left_value is not None
        and right_value is not None
        and type(left_value) is type(right_value)
        and left_value == right_value
    )


def _resolve_scaled_absolute(
    tolerance: "Tolerance", low: Any, high: Any
) -> float | None:
    if not (_finite_number(low) and _finite_number(high)):
        return None
    try:
        width = high - low
        if not math.isfinite(width):
            return None
    except (OverflowError, TypeError):
        return None
    if high < low:
        return None
    try:
        resolved = tolerance.absolute + tolerance.relative * width
    except (OverflowError, TypeError):
        return None
    if not math.isfinite(resolved):
        return None
    return resolved


@dataclass(frozen=True)
class Tolerance:
    """How far two independently computed values of one field may differ."""

    #: Absolute tolerance, in the field's own unit.
    absolute: float = 0.0
    #: Relative tolerance, as a fraction of ``max(|left|, |right|)``.
    relative: float = 0.0
    #: Why this number and not a smaller one. Required: a tolerance without a
    #: stated derivation is a tolerance nobody can argue with later.
    reason: str = "exact"
    #: ``(low_field, high_field)`` in the LEFT/legacy record. If present,
    #: ``relative`` is the allowed fraction of ``high - low`` rather than of
    #: the maximum endpoint magnitude.
    scale_fields: tuple[str, str] | None = None
    #: Fields that must be equal across left and right for this rule to apply.
    require_equal_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("absolute", "relative"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be an int or float")
            if not _nonnegative_number(value):
                raise ValueError(f"{name} must be finite and >= 0")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a nonempty string")
        if self.scale_fields is not None and not _nonempty_str_tuple(
            self.scale_fields, 2
        ):
            raise ValueError("scale_fields must be a tuple of two nonempty strings")
        if not _nonempty_str_tuple(self.require_equal_fields):
            raise ValueError("require_equal_fields must be a tuple of nonempty strings")

    @property
    def is_exact(self) -> bool:
        return self.absolute == 0.0 and self.relative == 0.0

    def exceeded_by(self, left: float, right: float) -> float:
        """How far outside tolerance the pair is; ``0.0`` when inside."""
        delta = abs(left - right)
        allowed = self.absolute + self.relative * max(abs(left), abs(right))
        return 0.0 if delta <= allowed else delta - allowed


EXACT = Tolerance()


def _context_matches(
    tolerance: Tolerance,
    left_record: Mapping[str, Any] | None,
    right_record: Mapping[str, Any] | None,
) -> bool:
    if not tolerance.require_equal_fields:
        return True
    if not isinstance(left_record, Mapping) or not isinstance(right_record, Mapping):
        return False
    for name in tolerance.require_equal_fields:
        if name not in left_record or name not in right_record:
            return False
        if not _context_value_matches(name, left_record[name], right_record[name]):
            return False
    return True


def _scaled_tolerance(
    tolerance: Tolerance,
    left_record: Mapping[str, Any] | None,
) -> Tolerance | None:
    if tolerance.scale_fields is None:
        return None
    if not isinstance(left_record, Mapping):
        return EXACT
    low_field, high_field = tolerance.scale_fields
    if low_field not in left_record or high_field not in left_record:
        return EXACT
    resolved_absolute = _resolve_scaled_absolute(
        tolerance, left_record[low_field], left_record[high_field]
    )
    if resolved_absolute is None:
        return EXACT
    return Tolerance(
        absolute=resolved_absolute,
        relative=0.0,
        reason=tolerance.reason,
    )


@dataclass(frozen=True)
class TolerancePolicy:
    """An ordered list of ``(field glob, tolerance)``. First match wins."""

    policy_id: str
    rules: tuple[tuple[str, Tolerance], ...] = field(default_factory=tuple)

    def for_field(
        self,
        field_path: str,
        *,
        left_record: Mapping[str, Any] | None = None,
        right_record: Mapping[str, Any] | None = None,
    ) -> Tolerance:
        for pattern, tol in self.rules:
            if not fnmatch.fnmatchcase(field_path, pattern):
                continue
            if not _context_matches(tol, left_record, right_record):
                return EXACT
            scaled = _scaled_tolerance(tol, left_record)
            return scaled if scaled is not None else tol
        return EXACT

    def declared_fields(self) -> tuple[str, ...]:
        return tuple(pattern for pattern, _ in self.rules)


#: The policy the tier-0 corpus compares under.
#:
#: It is deliberately almost empty. A tier-0 replay compares a frozen record
#: against itself through a serialization round trip, and a round trip that
#: needs a tolerance has already lost information — which is the defect, not
#: something to absorb. Per-field tolerances belong to the *parity* comparisons
#: of later phases, where v2 recomputes a float by a different route; each one
#: is added there with its derivation, one line at a time.
SCORE_RECORD_V1 = TolerancePolicy(policy_id="score_record.exact.v1", rules=())
