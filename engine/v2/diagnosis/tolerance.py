"""Per-field tolerance policies. There is no global tolerance, by construction.

§3.2 and §7.3: exact comparison for IDs, selected contracts, integer
quantities, gate verdicts, flags, null masks and frozen canonical records;
declared **per-field** tolerances for independently recomputed floats; never a
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
from dataclasses import dataclass, field

__all__ = ["Tolerance", "TolerancePolicy", "EXACT", "SCORE_RECORD_V1"]


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

    @property
    def is_exact(self) -> bool:
        return self.absolute == 0.0 and self.relative == 0.0

    def exceeded_by(self, left: float, right: float) -> float:
        """How far outside tolerance the pair is; ``0.0`` when inside."""
        delta = abs(left - right)
        allowed = self.absolute + self.relative * max(abs(left), abs(right))
        return 0.0 if delta <= allowed else delta - allowed


EXACT = Tolerance()


@dataclass(frozen=True)
class TolerancePolicy:
    """An ordered list of ``(field glob, tolerance)``. First match wins."""

    policy_id: str
    rules: tuple[tuple[str, Tolerance], ...] = field(default_factory=tuple)

    def for_field(self, field_path: str) -> Tolerance:
        for pattern, tol in self.rules:
            if fnmatch.fnmatchcase(field_path, pattern):
                return tol
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
