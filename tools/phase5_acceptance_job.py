"""Run release-scope Phase 5 acceptance for one staged candidate (S2).

The worker logic of the ``phase5_accept`` ops job
(``engine/v2/ops/retrain_check.py`` is the thin job wrapper, as
``engine/v2/ops/state_stage.py`` wraps ``tools/phase5_state_stage_job.py``).
Calls the existing acceptance gate with no Phase 4 corpus/checkpoint and writes
its evidence as ``acceptance.json`` in the attempt output directory. Never marks
staging succeeded, never writes the live deployment pointer, and never replays
Phase 4; the existing acceptance rehearsal writes only on its scratch
pointer-state copy.
Contract: ``engine/v2/models/ARCHITECTURE.md`` section 2.3 (S2).
"""
from __future__ import annotations

import json
from pathlib import Path

from checks.phase5_acceptance import build_evidence

ACCEPTANCE_FILE = "acceptance.json"


def run_acceptance(release_root: Path, output_dir: Path,
                   candidate_release_id: str | None = None) -> tuple[dict, Path]:
    """Build release-scope evidence and write it as ``acceptance.json``.

    An explicit candidate id selects that release's local catalog; without one
    the gate keeps its root-selector default."""
    evidence = build_evidence(Path(release_root),
                              candidate_release_id=candidate_release_id)
    output = Path(output_dir) / ACCEPTANCE_FILE
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return evidence, output
