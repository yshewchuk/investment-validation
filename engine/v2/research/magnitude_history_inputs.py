"""Consumer-only metadata admission for the original stored history artifact."""
from engine.v2.data import errors
from engine.v2.research.magnitude_history_store import _existing, _preflight_features
from engine.v2.research.magnitude_history_table import CONTRACT, TABLE


def admit_stored(repository, snapshot, context):
    """Prove receipt, contract and whole-snapshot isolation without opening payloads."""
    if _existing(repository, context) != snapshot or set(snapshot.table_versions) != {TABLE}:
        raise errors.fail("IDENTITY_CONFLICT", "feature snapshot is not the committed context output")
    if repository.table_contract(snapshot, TABLE) != CONTRACT:
        raise errors.fail("CONTRACT_MISMATCH", "stored history contract differs")
    _preflight_features(repository, snapshot, context)
