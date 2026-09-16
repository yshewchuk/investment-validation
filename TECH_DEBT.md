# Technical debt

## TD-P2-3A-001 - Phase 2 provenance blocks Phase 3A launch acceptance

**Status:** Open. Phase 3A is not acceptance-complete while this item remains
open.

The available Phase 3A candidate was built from a Phase 2 handoff whose D14
corpus receipt remains an explained strict differ and whose D15 provenance
cannot supply the verified, release-matching score and render receipt refs
required by PreviewInput. The old candidate also used a zero bundle manifest
hash and placeholder not-yet-produced receipt identifiers. These are not
valid immutable inputs for a launched projection.

**Why this is debt, not a cosmetic gate issue:** Phase 3A can only claim a
served PreviewRelease when its source release, snapshot, score batch, bundle
manifest, finality, expected-population and D14/D15 evidence refs are retained
and verifiable. A browser/API receipt cannot repair missing upstream
provenance.

**Required resolution:**

1. Rebuild the bounded Phase 2 candidate from retained cached inputs under a
   new private root; do not fetch providers or rewrite the official dataset.
2. Produce release-matching D15 score-parity and D19 render-parity receipts,
   then construct a non-placeholder PreviewInput from their actual retained
   refs and bundle manifest.
3. Resolve the full 121-row source-to-bridge precision discrepancy before
   accepting browser/API parity for that population.
4. Run two distinct fenced publication generations and a fresh rollback
   generation, retaining each bound projection and publication evidence.
5. Rebuild the Phase 3A manifest and require the strengthened gate to pass
   without placeholder, lineage, population, or rollback-binding findings.

**D14 note:** The existing Phase 2 guide records D14 as an explained strict
differ caused by a historical corpus/archive boundary and says a corpus
refresh alone will not make it agree. Do not erase or relabel that finding.
If a fully strict Phase 2 prerequisite is required for Phase 3A, ownership
must explicitly extend to resolving that corpus/archive boundary; otherwise
the Phase 3A report must continue to surface it as retained prerequisite debt,
not launch acceptance.

**Evidence:** Phase 2 D14 disposition, Phase 3A sections 5.1 and 10, and
Phase 3 gate enforcement committed in 54a7b20.
