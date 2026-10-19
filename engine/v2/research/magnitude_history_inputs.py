"""Consumer-only metadata admission for the original stored history artifact."""
from engine.v2.research.magnitude_history_store import preflight_stored


def admit_stored(repository, snapshot, context):
    """Prove receipt, contract and whole-snapshot isolation without opening payloads."""
    preflight_stored(repository, snapshot, context)
