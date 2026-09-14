"""Tier-4 serving-cache coverage for snapshot-backed scoring — P2-C02 (Phase 2
review closeout, §12.2): "Before a snapshot-backed scoring or replay stage
launches, compute the Tier-4 ``(model_id, fold, panel sha256)`` serving
caches its planned population needs. Refuse with a stable code that lists
every missing triple when the pinned reference inputs don't cover them. Do
not fit on a miss in Phase 2."

Three pure functions, legacy-free:

* :func:`champion_producer_models` — the champions with a Tier-4 ``produces``
  from an already-parsed list of champion registry entries
  (``reference_inputs.champion_entries``); model_id/features/produces only,
  never the registry's own artifact path or hyperparameters.
* :func:`required_serving_triples` — one ``(model_id, fold_yyyymm, panel_sha)``
  per (event, Tier-4 producer) the planned population needs. The fold itself
  comes from ``legacy_adapter.legacy_serving_fold`` — the scorer's own
  ``min(event_fold, decision_fold)`` rule (``engine/data/features/tier4.py``
  ``serving_fold``) — never reimplemented here.
* :func:`missing_triples` — which required triples the pinned
  ``tier4_serving_cache`` refs do not actually cover, and why. A triple's
  filename alone (``<model_id>_<fold:%Y%m>_<panel_sha[:12]>.joblib``) decides
  ``absent`` vs. ``fold_mismatch``; a filename match still reads the file's
  own header (through ``legacy_adapter.legacy_tier4_serving_header`` — bounded
  and hash-verified before it is ever unpickled) to catch ``panel_mismatch``
  and ``features_mismatch``, since the filename's panel segment is only a
  12-character prefix and carries no features at all.
"""
from __future__ import annotations

from engine.v2.foundation import CONTENT_HASH_PREFIX

from . import legacy_adapter, reference_inputs

__all__ = [
    "MAX_HEADER_BYTES",
    "champion_producer_models",
    "missing_triples",
    "required_serving_triples",
]

#: Refuse to unpickle a pinned serving-cache file bigger than this. A real
#: cache is one small fitted estimator plus a handful of scalars/strings —
#: kilobytes to a few megabytes; a file far past that is not read to find out
#: why (task brief decision 1).
MAX_HEADER_BYTES = 64 * 1024 * 1024


def champion_producer_models(champions) -> tuple[dict, ...]:
    """``(model_id, features, produces)`` for every champion with a Tier-4
    ``produces`` — ``champions`` is ``reference_inputs.champion_entries``'s
    own return, so a caller never parses the registry a second way."""
    out = []
    for entry in champions:
        produces = entry.get("produces")
        if produces is None:
            continue
        out.append({"model_id": entry.get("id"), "features": tuple(entry.get("features") or ()),
                   "produces": produces})
    return tuple(out)


def required_serving_triples(population, registry_models, panel_sha: str) -> frozenset:
    """Every ``(model_id, fold_yyyymm, panel_sha)`` the population needs.

    ``population`` is an iterable of ``(event_date, as_of)`` pairs — the
    scorer serves every Tier-4 producer at the SAME fold for a given event
    (``Scorer._serving`` keys its cache on ``(fold, produces)``, and the fold
    itself does not depend on ``produces``), so the fold is computed once per
    distinct ``(event_date, as_of)`` pair and then crossed with every
    producer model.
    """
    folds: dict[tuple, str] = {}
    triples = set()
    for event_date, as_of in population:
        key = (event_date, as_of)
        fold = folds.get(key)
        if fold is None:
            fold = legacy_adapter.legacy_serving_fold(event_date, as_of).strftime("%Y%m")
            folds[key] = fold
        for model in registry_models:
            triples.add((model["model_id"], fold, panel_sha))
    return frozenset(triples)


def _parse_cache_path(relative_path: str) -> tuple[str, str, str] | None:
    """``(model_id, fold_yyyymm, panel_sha12)`` from a Tier-4 serving-cache
    relative path, or ``None`` if it is not shaped like one. ``model_id`` may
    itself contain underscores, so this splits from the right — the fold
    (6 digits) and panel prefix (12 hex chars) have a fixed, recognizable
    shape and the rest is the model id, whatever it contains."""
    prefix = reference_inputs.TIER4_SERVING_DIR + "/"
    if not relative_path.startswith(prefix) or not relative_path.endswith(".joblib"):
        return None
    stem = relative_path[len(prefix):-len(".joblib")]
    parts = stem.rsplit("_", 2)
    if len(parts) != 3 or not all(parts):
        return None
    model_id, fold, panel12 = parts
    if len(fold) != 6 or not fold.isdigit():
        return None
    if len(panel12) != 12 or any(c not in "0123456789abcdef" for c in panel12.lower()):
        return None
    return model_id, fold, panel12


def _index_pinned(pinned_cache_refs) -> dict[tuple[str, str], dict[str, str]]:
    """``(model_id, panel_sha12) -> {fold_yyyymm: content_hash}``, from every
    pinned ref shaped like a Tier-4 serving-cache path. A ref that is not
    shaped like one (or lives elsewhere) is silently not an index entry —
    the same "not recognized as a Tier-4 cache ref at all" rule
    ``legacy_materialization._tier4_cache_hash_prefix`` already uses."""
    index: dict[tuple[str, str], dict[str, str]] = {}
    for path, digest in pinned_cache_refs.items():
        parsed = _parse_cache_path(path)
        if parsed is None:
            continue
        model_id, fold, panel12 = parsed
        index.setdefault((model_id, panel12), {})[fold] = digest
    return index


def _cache_object_path(store, digest: str):
    hex_digest = digest.removeprefix(CONTENT_HASH_PREFIX)
    return store.root / "objects" / hex_digest[:2] / hex_digest


def _missing_entry(model_id: str, fold: str, panel12: str, reason: str) -> dict:
    return {"model_id": model_id, "fold": fold, "panel_sha12": panel12, "reason": reason}


def missing_triples(required, pinned_cache_refs, store, *, registry_models,
                    max_header_bytes: int = MAX_HEADER_BYTES) -> list[dict]:
    """Which of ``required`` (:func:`required_serving_triples`'s own output)
    the pinned refs do not actually cover, sorted for a stable report.

    ``pinned_cache_refs`` is ``{relative_path: content_hash}`` — every pinned
    ``tier4_serving_cache`` ref of the materialization request being launched.
    ``store`` resolves a content hash to bytes the same way every other
    object read in this codebase does: ``store.root/objects/<xx>/<hash>``
    (``ArtifactStore``'s own layout; see ``snapshot_planning.scratch_estimate``
    for the same pattern).
    """
    index = _index_pinned(pinned_cache_refs)
    features_by_model = {m["model_id"]: tuple(m["features"]) for m in registry_models}
    out = []
    for model_id, fold, panel_sha in sorted(required):
        panel12 = panel_sha[:12]
        candidates = index.get((model_id, panel12), {})
        digest = candidates.get(fold)
        if digest is None:
            reason = "fold_mismatch" if candidates else "absent"
            out.append(_missing_entry(model_id, fold, panel12, reason))
            continue
        header = legacy_adapter.legacy_tier4_serving_header(
            _cache_object_path(store, digest), expected_sha256_hex=digest.removeprefix(CONTENT_HASH_PREFIX),
            max_bytes=max_header_bytes)
        if header is None:
            out.append(_missing_entry(model_id, fold, panel12, "absent"))
        elif header.get("tier3_snapshot") != panel_sha:
            out.append(_missing_entry(model_id, fold, panel12, "panel_mismatch"))
        elif tuple(header.get("features") or ()) != features_by_model.get(model_id, ()):
            out.append(_missing_entry(model_id, fold, panel12, "features_mismatch"))
    return out
