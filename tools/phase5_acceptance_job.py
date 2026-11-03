"""Run release-scope Phase 5 acceptance for one staged candidate (S2).

The worker logic of the ``phase5_accept`` ops job
(``engine/v2/ops/retrain_check.py`` is the thin job wrapper, as
``engine/v2/ops/state_stage.py`` wraps ``tools/phase5_state_stage_job.py``).
Calls the existing acceptance gate with no Phase 4 corpus/checkpoint and writes
its evidence as ``acceptance.json`` in the attempt output directory. Never marks
staging succeeded, writes a deployment pointer, or replays Phase 4.
Contract: ``engine/v2/models/ARCHITECTURE.md`` section 2.3 (S2).
"""
from __future__ import annotations

import json
from pathlib import Path

from checks.phase5_acceptance import build_evidence

ACCEPTANCE_FILE = "acceptance.json"


def run_acceptance(release_root: Path, output_dir: Path) -> tuple[dict, Path]:
    """Build release-scope evidence and write it as ``acceptance.json``."""
    evidence = build_evidence(Path(release_root))
    output = Path(output_dir) / ACCEPTANCE_FILE
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return evidence, output
