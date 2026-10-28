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

from math import isfinite

from engine.v2.foundation import DocumentError, from_document


def _restore_model_artifact(
    artifact_doc, *, path, parser, artifact_error, kind
):
    """Validate and restore one typed artifact document under ``path``."""
    if not isinstance(artifact_doc, dict):
        raise DocumentError("BAD_TYPE", path, f"expected a {kind} artifact object")
    if "content_hash" not in artifact_doc:
        raise DocumentError("MISSING_FIELD", f"{path}.content_hash", "required")
    if not isinstance(artifact_doc["content_hash"], str):
        raise DocumentError("BAD_TYPE", f"{path}.content_hash", "expected a string")
    try:
        artifact = parser(artifact_doc)
    except KeyError as exc:
        field = exc.args[0] if exc.args and isinstance(exc.args[0], str) else ""
        raise DocumentError(
            "MISSING_FIELD",
            f"{path}.{field}" if field else path,
            "required",
        ) from None
    except artifact_error:
        raise DocumentError(
            "BAD_SCHEMA_VERSION",
            f"{path}.schema_version",
            f"not a supported {kind} artifact schema_version",
        ) from None
    except (TypeError, ValueError, OverflowError):
        raise DocumentError(
            "BAD_TYPE", path, f"malformed {kind} artifact document"
        ) from None
    if artifact_doc["content_hash"] != artifact.content_hash:
        raise DocumentError(
            "CONTENT_HASH_MISMATCH",
            f"{path}.content_hash",
            "declared hash does not match the artifact payload",
        )
    return artifact


def _restored_model_block(model: dict) -> dict:
    """C3 (partial): rehydrate the typed artifacts inside a model block.

    ``to_document(NativeScoreInputs)`` flattens nested
    ``PayoffLineArtifact``/``PayoffSurfaceArtifact`` and
    ``RecalibrationMapArtifact`` values into plain JSON dicts, but the native
    model stage requires the concrete types, so a valid serialized document
    would otherwise score MODEL_NOT_READY. Each present artifact is restored
    with its module's ``_artifact_from_document`` (the payoff one picks line vs
    surface from ``schema_version``) and its declared ``content_hash`` must be a
    string equal to the restored artifact's recomputed hash. Refusals are
    ``DocumentError`` under ``$.model.<field>`` (or a child path) and never
    echo input values. The caller's block is never mutated: a fresh dict is
    returned, and absent or ``None`` fields stay absent or ``None``.
    """
    block = dict(model)
    if model.get("payoff_artifact") is not None:
        from engine.v2.models.payoff_artifact import (
            PayoffArtifactError,
            _artifact_from_document,
        )

        block["payoff_artifact"] = _restore_model_artifact(
            model["payoff_artifact"],
            path="$.model.payoff_artifact",
            parser=_artifact_from_document,
            artifact_error=PayoffArtifactError,
            kind="payoff",
        )
    if model.get("recalibration_artifact") is not None:
        from engine.v2.models.recalibration_artifact import (
            RecalibrationArtifactError,
        )
        from engine.v2.models.recalibration_artifact import (
            _artifact_from_document as _recalibration_from_document,
        )

        block["recalibration_artifact"] = _restore_model_artifact(
            model["recalibration_artifact"],
            path="$.model.recalibration_artifact",
            parser=_recalibration_from_document,
            artifact_error=RecalibrationArtifactError,
            kind="recalibration",
        )
    return block


def _load_native_score_inputs(doc: dict):
    from engine.v2.domain.generation import Geometry, Pricing
    from engine.v2.scoring.stages import NativeScoreInputs, StageReceipt

    geometry = from_document(Geometry, doc["geometry"]) if doc.get("geometry") is not None else None
    pricing = from_document(Pricing, doc["pricing"]) if doc.get("pricing") is not None else None
    receipts = tuple(from_document(StageReceipt, row) for row in doc["stage_receipts"])
    context = doc["context"]
    if isinstance(context, dict) and "chain_spot" in context:
        raw_spot = context["chain_spot"]
        try:
            chain_spot = None if isinstance(raw_spot, bool) else float(raw_spot)
        except (TypeError, ValueError, OverflowError):
            chain_spot = None
        if chain_spot is None or not isfinite(chain_spot) or chain_spot <= 0.0:
            raise DocumentError(
                "BAD_VALUE", "$.context.chain_spot",
                "expected a finite positive spot")
        context = dict(context)
        context["chain_spot"] = chain_spot
    kwargs = dict(
        context=context, features=doc["features"], forecast=doc["forecast"],
        geometry=geometry, pricing=pricing, analogs=doc["analogs"], simulation=doc["simulation"],
        gate=doc["gate"], chooser=doc["chooser"], diagnostics=doc.get("diagnostics", {}),
        source_ref=doc["source_ref"], stage_receipts=receipts,
    )
    if "model" in doc:
        model = doc["model"]
        if isinstance(model, dict) and (
            model.get("payoff_artifact") is not None
            or model.get("recalibration_artifact") is not None
        ):
            model = _restored_model_block(model)
        kwargs["model"] = model
    return NativeScoreInputs(**kwargs)
