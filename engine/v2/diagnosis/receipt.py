"""``ComparisonReceipt``, ``Finding`` and the shared failure envelope.

The phase-0 subset of [component contracts §15.2](../../../guides/component_contracts.md).
Full schema fidelity is not required here; the two behavioural properties are,
and they are what the shapes below exist to make possible:

* **Stage-localized** — every :class:`Finding` names a stage and a field path,
  and :class:`ComparisonReceipt` carries per-stage input and output hashes so
  "first differing stage" is a computed fact rather than a claim.
* **Complete, not first-wins** — ``findings`` is a set of independent findings
  from one pass. ``independent_of`` records which of them the comparator
  *proved* are not consequences of one another, so several can be fixed at
  once. Findings it could not separate are reported without that link rather
  than silently merged.

``verdict: incomparable`` is a first-class outcome. A missing artifact, an
unresolvable reference or a zero-row population is not agreement — and
:class:`Population` is the field that prevents the reciprocal failure, where a
comparison over a collapsed set reports ``agree``.

The envelope (``started_at``, ``duration``, ``worker_ref``) is excluded from
every content hash, per contracts §2.5: a replay reproduces a payload without
reproducing its elapsed time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "SCHEMA_VERSION",
    "Verdict",
    "Finding",
    "StageHashes",
    "Population",
    "Envelope",
    "ComparisonReceipt",
    "problem",
    "PROBLEM_CATEGORIES",
]

SCHEMA_VERSION = "comparison_receipt.v1.0"

#: The three outcomes. There is no fourth, and no "mostly agrees".
AGREE = "agree"
DIFFER = "differ"
INCOMPARABLE = "incomparable"
Verdict = str

#: contracts §2.4.
PROBLEM_CATEGORIES = (
    "validation", "dependency", "source", "resource", "integrity", "internal",
)


@dataclass(frozen=True)
class Finding:
    """One independent difference, localized to a stage and a field."""

    finding_id: str
    first_differing_stage: str
    field_path: str
    left_value: Any = None
    right_value: Any = None
    delta: float | None = None
    unit: str | None = None
    tolerance_applied: str = "exact"
    exceeded_by: float | None = None
    null_mask_left: bool = False
    null_mask_right: bool = False
    #: What kind of disagreement this is: ``value``, ``null_mask``,
    #: ``missing_field``, ``type`` or ``length``. Separate from the field path
    #: because "present but different" and "absent on one side" are different
    #: bugs, and five of the six defects this corpus exists to catch were
    #: invisible in the non-null values alone.
    kind: str = "value"
    source_rows_ref: str | None = None
    recipe_and_artifact_versions: dict[str, Any] = field(default_factory=dict)
    affected_count: int = 1
    #: Findings this one was PROVED not to be a consequence of. Empty means
    #: "not separated", never "definitely dependent".
    independent_of: tuple[str, ...] = ()

    def describe(self) -> str:
        return (
            f"{self.first_differing_stage}: {self.field_path} "
            f"({self.kind}) {self.left_value!r} != {self.right_value!r}"
        )


@dataclass(frozen=True)
class StageHashes:
    """One row of the receipt's ordered stage table."""

    stage_id: str
    left_input_hash: str
    left_output_hash: str
    right_input_hash: str
    right_output_hash: str
    agrees: bool

    @property
    def inputs_agree(self) -> bool:
        return self.left_input_hash == self.right_input_hash


@dataclass(frozen=True)
class Population:
    """Expected, supported, compared and skipped — recorded separately.

    A fixture whose universe collapsed compares zero rows. Reporting that as
    agreement is the vacuous pass §11 names; keeping the three counts apart is
    what makes a collapse independently visible rather than a silent denominator
    change.
    """

    expected: int = 0
    supported: int = 0
    compared: int = 0
    skipped_with_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def collapsed(self) -> bool:
        return self.compared == 0


@dataclass(frozen=True)
class Envelope:
    """Operational metadata. Excluded from every content hash (contracts §2.5)."""

    started_at: str | None = None
    duration_seconds: float | None = None
    worker_ref: str | None = None
    diagnostic_ref: str | None = None


@dataclass(frozen=True)
class ComparisonReceipt:
    """Evidence that one comparison happened, and what it found."""

    receipt_id: str
    comparison_kind: str
    tier: int
    left_ref: str
    right_ref: str
    stage_plan_ref: str
    tolerance_policy_ref: str
    verdict: Verdict
    stage_hashes: tuple[StageHashes, ...] = ()
    findings: tuple[Finding, ...] = ()
    population: Population = field(default_factory=Population)
    envelope: Envelope = field(default_factory=Envelope)
    problems: tuple[dict[str, Any], ...] = ()
    schema_version: str = SCHEMA_VERSION

    @property
    def agrees(self) -> bool:
        return self.verdict == AGREE

    @property
    def first_differing_stage(self) -> str | None:
        """The earliest stage whose inputs agreed and outputs did not."""
        for row in self.stage_hashes:
            if row.inputs_agree and not row.agrees:
                return row.stage_id
        return None

    def stages_named(self) -> tuple[str, ...]:
        """Every distinct stage the findings name, in receipt order."""
        order = [row.stage_id for row in self.stage_hashes]
        named = {f.first_differing_stage for f in self.findings}
        ranked = sorted(named, key=lambda s: order.index(s) if s in order else 1e9)
        return tuple(ranked)

    def payload(self) -> dict[str, Any]:
        """The deterministic part — everything but the envelope."""
        out = asdict(self)
        out.pop("envelope", None)
        return out

    def summary(self) -> str:
        lines = [
            f"{self.comparison_kind} [tier {self.tier}] -> {self.verdict.upper()}",
            f"  population: expected={self.population.expected} "
            f"supported={self.population.supported} "
            f"compared={self.population.compared} "
            f"skipped={self.population.skipped_with_reasons or '{}'}",
        ]
        for finding in self.findings:
            lines.append(f"  - {finding.describe()}")
        for prob in self.problems:
            lines.append(f"  ! {prob['code']}: {prob['message']}")
        return "\n".join(lines)


def problem(
    code: str,
    message: str,
    *,
    category: str = "validation",
    stage: str | None = None,
    retryable: bool = False,
    dependency_refs: tuple[str, ...] = (),
) -> dict[str, Any]:
    """The shared failure envelope of contracts §2.4.

    A refusal is a business result, not a crash: an incomparable receipt
    carries one of these and is still a receipt. Provider credentials and raw
    licensed responses never appear in one.
    """
    if category not in PROBLEM_CATEGORIES:
        raise ValueError(f"{category!r} is not one of {PROBLEM_CATEGORIES}")
    return {
        "schema_version": "problem.v1.0",
        "code": code,
        "category": category,
        "retryable": retryable,
        "message": message,
        "stage": stage,
        "dependency_refs": list(dependency_refs),
    }
