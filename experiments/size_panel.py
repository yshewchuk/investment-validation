"""Installed-code-bound qualified SIZE panel entrypoint; no fitting authority."""
import sys
from pathlib import Path

from engine.v2.data.errors import fail
from engine.v2.foundation import canonical_json
from engine.v2.ops.fingerprints import environment_identity, source_closure
from engine.v2.research import magnitude_panel_qualified as panel
from engine.v2.research.eligible_history_table import FEATURES
from engine.v2.research.magnitude_panel_qualified_table import admit_columns
from engine.v2.research.magnitude_panel_table import canonical_day
from engine.v2.research.snapshot_inputs import snapshot_reference
from experiments import eligible_history


def _fingerprint():
    return {"source_closure": source_closure(Path(__file__).resolve().parents[1],
            ["experiments/size_panel.py"]), "environment": environment_identity(1)}


_IMPORTED_PRODUCER = canonical_json(_fingerprint())


def _producer(request):
    if any(name in request for name in ("producer", "history_producer", "history_admitted")):
        raise fail("HOLDOUT_ACCESS_DENIED", "installed producer identity cannot be supplied by callers")
    producer = _fingerprint()
    if canonical_json(producer) != _IMPORTED_PRODUCER:
        raise fail("INPUT_CHANGED", "panel producer changed since import; restart before admission")
    return producer


_HISTORY_KEYS = {"as_of_month", "policy", "event_ids", "horizons", "history_start", "limits"}
_PANEL_KEYS = {"history_snapshot", "price_history_snapshot", "source_start", "metadata_limits"}
_METADATA_LIMITS = {"max_catalog_tables", "max_catalog_fragments", "max_catalog_bytes",
    "max_fragments", "max_partitions", "max_tickers", "max_scan_rows", "max_object_bytes",
    "max_total_object_bytes", "max_batch_rows", "max_retained_rows",
    "max_retained_metadata_bytes", "max_proof_bytes"}


def _request(source, request):
    if any(name in request for name in ("producer", "history_producer", "history_admitted")):
        raise fail("HOLDOUT_ACCESS_DENIED", "installed producer identity cannot be supplied by callers")
    if set(request) != _HISTORY_KEYS | _PANEL_KEYS:
        raise fail("CONTRACT_MISMATCH", "qualified panel requires exact explicit request fields")
    eligible_history.validate_request({k: request[k] for k in _HISTORY_KEYS})
    panel._policy(request["policy"])
    canonical_day(request["history_start"])
    canonical_day(request["source_start"])
    limits = request["metadata_limits"]
    if (type(limits) is not dict or limits.keys() != _METADATA_LIMITS
            or any(type(v) is not int or not 0 < v < sys.maxsize for v in limits.values())):
        raise fail("CONTRACT_MISMATCH", "panel requires exact positive metadata limits")
    for snapshot in (source, request["history_snapshot"], request["price_history_snapshot"]):
        snapshot_reference(snapshot)


def _history(repository, source, request):
    panel._policy(request["policy"])
    return eligible_history.iter_read(repository, source, request["history_snapshot"], columns=list(FEATURES),
        **{k: request[k] for k in ("as_of_month", "policy", "event_ids", "horizons", "history_start", "limits")})


def materialize(repository, source, **request):
    """Publish immutable complete reconstructed rows after source qualification."""
    _request(source, request)
    return panel._materialize(repository, source, producer=_producer(request),
                              history_admitted=_history(repository, source, request), **request)


def read(repository, source, feature_snapshot, *, columns=None, **request):
    """Re-admit exact stored rows; projections cannot relax grouped eligibility."""
    admit_columns(columns)
    _request(source, request)
    snapshot_reference(feature_snapshot)
    producer = _producer(request)
    history_admitted = _history(repository, source, request)
    for _ in history_admitted["rows"]:
        pass
    return panel._read(repository, source, feature_snapshot, columns=columns,
                       producer=producer, history_admitted=history_admitted, **request)
