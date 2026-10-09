"""Native score-input decoding, extracted from ``engine.v2.ops.cli`` as a leaf.

Holds the ops layer's canonical decoder for an already-captured
``to_document(NativeScoreInputs)`` document.  ``_load_native_score_inputs``
composes :func:`_restored_model_block`, which rehydrates the typed payoff
artifact inside a model block, because ``from_document`` cannot reconstruct
``NativeScoreInputs``' ``Mapping[...]`` blocks or its typed model layer.

Imports only ``engine.v2.foundation`` at module level; the model/domain/scoring
imports stay inside the functions (those packages are lazy-load only).  The
CLI, the worker and the dashboard projector all consume these functions, so
they live here rather than in the entry-point ``engine.v2.ops.cli``.
"""
from __future__ import annotations

from engine.v2.foundation import DocumentError, from_document


def _restored_model_block(model: dict) -> dict:
    """C3 (partial): rehydrate the typed payoff artifact inside a model block.

    ``to_document(NativeScoreInputs)`` flattens a nested
    ``PayoffLineArtifact``/``PayoffSurfaceArtifact`` into a plain JSON dict,
    but the native model stage requires the concrete type
    (``stages._artifact_key_mismatch``), so a valid serialized document would
    otherwise score MODEL_NOT_READY. The dict is restored with
    ``payoff_artifact._artifact_from_document`` (its ``schema_version`` picks
    line vs surface) and the declared ``content_hash`` must be a string equal
    to the restored artifact's recomputed hash. Refusals are ``DocumentError``
    at ``$.model.payoff_artifact`` (or a child path) and never echo input
    values. The caller's block is never mutated: a fresh dict is returned.
    """
    from engine.v2.models.payoff_artifact import (
        PayoffArtifactError,
        _artifact_from_document,
    )

    path = "$.model.payoff_artifact"
    artifact_doc = model["payoff_artifact"]
    if not isinstance(artifact_doc, dict):
        raise DocumentError("BAD_TYPE", path, "expected a payoff artifact object")
    if "content_hash" not in artifact_doc:
        raise DocumentError("MISSING_FIELD", f"{path}.content_hash", "required")
    if not isinstance(artifact_doc["content_hash"], str):
        raise DocumentError("BAD_TYPE", f"{path}.content_hash", "expected a string")
    try:
        artifact = _artifact_from_document(artifact_doc)
    except KeyError as exc:
        field = exc.args[0] if exc.args and isinstance(exc.args[0], str) else ""
        raise DocumentError("MISSING_FIELD",
                            f"{path}.{field}" if field else path, "required") from None
    except PayoffArtifactError:
        raise DocumentError("BAD_SCHEMA_VERSION", f"{path}.schema_version",
                            "not a supported payoff artifact schema_version") from None
    except (TypeError, ValueError, OverflowError):
        raise DocumentError("BAD_TYPE", path, "malformed payoff artifact document") from None
    if artifact_doc["content_hash"] != artifact.content_hash:
        raise DocumentError("CONTENT_HASH_MISMATCH", f"{path}.content_hash",
                            "declared hash does not match the artifact payload")
    block = dict(model)
    block["payoff_artifact"] = artifact
    return block


def _load_native_score_inputs(doc: dict):
    from engine.v2.domain.generation import Geometry, Pricing
    from engine.v2.scoring.stages import NativeScoreInputs, StageReceipt

    geometry = from_document(Geometry, doc["geometry"]) if doc.get("geometry") is not None else None
    pricing = from_document(Pricing, doc["pricing"]) if doc.get("pricing") is not None else None
    receipts = tuple(from_document(StageReceipt, row) for row in doc["stage_receipts"])
    kwargs = dict(
        context=doc["context"], features=doc["features"], forecast=doc["forecast"],
        geometry=geometry, pricing=pricing, analogs=doc["analogs"], simulation=doc["simulation"],
        gate=doc["gate"], chooser=doc["chooser"], diagnostics=doc.get("diagnostics", {}),
        source_ref=doc["source_ref"], stage_receipts=receipts,
    )
    if "model" in doc:
        model = doc["model"]
        if isinstance(model, dict) and model.get("payoff_artifact") is not None:
            model = _restored_model_block(model)
        kwargs["model"] = model
    return NativeScoreInputs(**kwargs)
