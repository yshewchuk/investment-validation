"""Installed-code-bound entrypoint for reusable experiment history artifacts."""
from pathlib import Path

from engine.v2.ops.fingerprints import environment_identity, source_closure
from engine.v2.research import eligible_history as history
from engine.v2.research.eligible_history import admit_source as admit_source


def _producer():
    root = Path(__file__).resolve().parents[1]
    return {"source_closure": source_closure(root, ["experiments/eligible_history.py"]),
            "environment": environment_identity(1)}


def materialize(repository, snapshot, **request):
    """Never accept caller assertions about which installed code builds/reuses rows."""
    return history.materialize(repository, snapshot, producer=_producer(), **request)


def read(repository, source, feature_snapshot, *, columns, **request):
    """Re-derive producer/source/policy identity before admitting model inputs."""
    return history.read(repository, source, feature_snapshot, columns=columns,
                        producer=_producer(), **request)
