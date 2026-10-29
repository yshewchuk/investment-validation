"""Installed-code-bound entrypoint for reusable experiment history artifacts."""
from pathlib import Path

from engine.v2.data.errors import fail
from engine.v2.foundation import canonical_json
from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts, calendar_date
from engine.v2.ops.fingerprints import environment_identity, source_closure
from engine.v2.research import eligible_history as history
from engine.v2.research.eligibility_policy import eligibility_policy
from engine.v2.research.eligible_history import admit_source as admit_source
from engine.v2.research.eligible_history_table import FEATURES
from engine.v2.research.snapshot_inputs import snapshot_reference


def validate_request(request, *, allow_all_events=False):
    """Pure closed request validation; no source/producer admission capability."""
    if type(request) is dict and "producer" in request:
        raise fail("HOLDOUT_ACCESS_DENIED", "installed producer identity cannot be supplied by callers")
    if type(request) is not dict or set(request) != {
            "as_of_month", "policy", "event_ids", "horizons", "history_start", "limits"}:
        raise fail("CONTRACT_MISMATCH", "explicit closed history request required")
    policy, ids, horizons = request["policy"], request["event_ids"], request["horizons"]
    template = eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30)
    if (type(policy) is not dict or set(policy) != set(template)
            or any(type(policy[k]) is not type(v) for k, v in template.items())
            or policy != eligibility_policy(version=policy["schema_version"], embargo_days=policy["embargo_days"])):
        raise fail("CONTRACT_MISMATCH", "invalid history eligibility policy")
    history._limits(request["limits"])
    if (not (allow_all_events and ids is None) and (type(ids) is not list or not ids
            or any(type(key) is not str or not key or key != key.strip() for key in ids)
            or len(set(ids)) != len(ids))):
        raise fail("CONTRACT_MISMATCH", "explicit unique canonical targets required")
    if type(horizons) is not list or any(type(h) is not int for h in horizons) or horizons != [1, 3, 5]:
        raise fail("CONTRACT_MISMATCH", "ordered native horizons required")
    try:
        ExperimentHoldouts(request["as_of_month"])
        calendar_date(request["history_start"])
    except (ValueError, TypeError, OverflowError):
        raise fail("CONTRACT_MISMATCH", "canonical history start and holdout month required") from None


def _producer():
    root = Path(__file__).resolve().parents[1]
    return {"source_closure": source_closure(root, ["experiments/eligible_history.py"]),
            "environment": environment_identity(1)}


_IMPORTED_PRODUCER = canonical_json(_producer())


def _admitted_producer(request):
    if "producer" in request:
        raise fail("HOLDOUT_ACCESS_DENIED", "installed producer identity cannot be supplied by callers")
    producer = _producer()
    if canonical_json(producer) != _IMPORTED_PRODUCER:
        raise fail("INPUT_CHANGED", "installed producer drifted since import; restart from a fixed code tree")
    return producer


def materialize(repository, snapshot, **request):
    """Never accept caller assertions about which installed code builds/reuses rows."""
    validate_request(request)
    snapshot_reference(snapshot)
    return history._materialize(repository, snapshot, producer=_admitted_producer(request), **request)


def read(repository, source, feature_snapshot, *, columns=None, **request):
    """Re-derive producer/source/policy identity before admitting model inputs."""
    validate_request(request)
    snapshot_reference(source)
    snapshot_reference(feature_snapshot)
    if (type(columns) not in (list, tuple) or not columns
            or any(type(c) is not str or c not in FEATURES for c in columns) or len(set(columns)) != len(columns)):
        raise fail("HOLDOUT_ACCESS_DENIED", "experiment fields require exact eligible-history provenance")
    return history._read(repository, source, feature_snapshot, columns=columns,
                         producer=_admitted_producer(request), **request)


def iter_read(repository, source, feature_snapshot, *, columns=None, **request):
    """Admit history metadata once and yield committed rows lazily for bounded consumers."""
    validate_request(request)
    snapshot_reference(source)
    snapshot_reference(feature_snapshot)
    if (type(columns) not in (list, tuple) or not columns
            or any(type(c) is not str or c not in FEATURES for c in columns) or len(set(columns)) != len(columns)):
        raise fail("HOLDOUT_ACCESS_DENIED", "experiment fields require exact eligible-history provenance")
    admitted = history._iter_read(repository, source, feature_snapshot, columns=columns,
                                  producer=_admitted_producer(request), **request)
    return admitted
