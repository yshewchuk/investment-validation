import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from checks import phase4_real
from checks.tier0_corpus import Corpus
from engine.v2.contracts import SCORE_REQUEST_V1
from engine.v2.foundation import content_hash, to_document
from engine.v2.scoring.identity import request_hash
from engine.v2.scoring.identity import with_score_id
from tools.phase4_request_translation import canonical_request_from_legacy


#: The pair's saved LEGACY request (payload.request); the traced V2 request
#: below is its translation (tools.phase4_request_translation).
LEGACY_REQUEST = {
    "ticker": "ABC", "strategy": "STR-THRU", "as_of": "2026-09-16",
    "event_date": "2026-09-17", "session": "AMC",
    "fill": {"policy_id": "legacy.fill_alpha.v1", "alpha": 0.5},
}


def _request_document():
    bound = to_document(canonical_request_from_legacy(
        LEGACY_REQUEST, event_id="event-1", snapshot="snapshot-1",
    ))
    return {
        "event_id": "event-1",
        "calendar_revision": bound["calendar_revision"],
        "strategy_version": "STR-THRU",
        "deployment_id": "deployment-1",
        "decision_clock_id": "entry-close-1",
        "requested_decision_at": "2026-09-16",
        "snapshot_id": "snapshot-1",
        "mode": "replay",
        "fill_model": bound["fill_model"],
        "event_revision": bound["event_revision"],
        "contract_override": None,
        "geometry_override": None,
        "dependency_refs": ["residual:1"],
        "model_artifact_refs": [],
        "residual_state_ref": None,
        "analog_state_ref": None,
        "calibration_state_ref": None,
        "schema_version": SCORE_REQUEST_V1,
    }


def _stage_rows():
    rows = {}
    prior = "root"
    for stage in phase4_real._REQUIRED_TRACE_STAGES:
        inputs = {"prior": prior, "stage": stage}
        output = {"stage": stage, "value": 1}
        rows[stage] = {
            "input": inputs,
            "output": output,
            "input_hash": content_hash(inputs),
            "output_hash": content_hash(output),
            "owner": "engine.score",
        }
        prior = rows[stage]["output_hash"]
    return rows


def _translation(shared_inputs, native_inputs):
    native_document = {
        "request": shared_inputs["request"],
        "native_inputs": {
            key: value for key, value in native_inputs.items()
            if key != "source_ref"
        },
    }
    shared_leaves = phase4_real._leaf_values(shared_inputs)
    native_leaves = phase4_real._leaf_values(native_document)
    assert shared_leaves == native_leaves
    mappings = [{
        "shared_path": list(path),
        "native_path": list(path),
        "value_hash": content_hash(value),
    } for path, value in sorted(shared_leaves.items(), key=lambda item: repr(item[0]))]
    body = {
        "schema_version": phase4_real._TRANSLATION_SCHEMA,
        "shared_input_hash": content_hash(shared_inputs),
        "native_input_hash": content_hash(native_inputs),
        "mappings": mappings,
        "derived": [{
            "native_path": ["native_inputs", "source_ref"],
            "operation": "shared_input_hash",
            "value_hash": content_hash(content_hash(shared_inputs)),
        }],
    }
    return {**body, "translation_hash": content_hash(body)}


def _pair(tmp_path, *, extra_model_inputs=None):
    """``extra_model_inputs``: merged into ``features.model_inputs`` on both
    ``native_inputs`` and its embedded copy in ``shared_inputs``, so
    ``_translation``'s own ``phase4_real._leaf_values`` call picks it up and
    builds correct mappings for it automatically -- used to plant a
    ``{"__nonfinite__": ...}``-tagged leaf (simulating the on-disk,
    post-``tag_nonfinite`` form a real capture stores) at top level, nested,
    or inside a list, without duplicating the rest of this fixture.
    """
    request_doc = _request_document()
    residuals = [{"event_date": "2025-01-01", "err_move": 0.1}]
    raw = json.dumps(residuals, sort_keys=True).encode()
    (tmp_path / "residuals.json").write_bytes(raw)
    native_inputs = {
        "context": {"strategy": "STR-THRU"},
        "features": {"model_inputs": dict(extra_model_inputs or {})},
        "forecast": {},
        "geometry": None,
        "pricing": None,
        "analogs": {},
        "simulation": {"residuals": residuals},
        "gate": {},
        "chooser": {},
        "diagnostics": {},
        "source_ref": "pending",
    }
    shared_inputs = {
        "request": request_doc,
        "native_inputs": {
            key: copy.deepcopy(value) for key, value in native_inputs.items()
            if key != "source_ref"
        },
    }
    shared_hash = content_hash(shared_inputs)
    native_inputs["source_ref"] = shared_hash
    traced_inputs = copy.deepcopy(native_inputs)
    traced_inputs["simulation"]["residuals"] = {"$resource": "residuals"}
    trace = {
        "schema_version": phase4_real._TRACE_SCHEMA,
        "request": request_doc,
        "request_hash": request_hash(
            phase4_real.from_document(phase4_real.ScoreRequest, request_doc)
        ),
        "shared_inputs": shared_inputs,
        "shared_input_hash": shared_hash,
        "native_input_hash": content_hash(native_inputs),
        "native_inputs": traced_inputs,
        "native_inputs_hash": content_hash(native_inputs),
        "input_translation": _translation(shared_inputs, native_inputs),
        "stages": _stage_rows(),
        "resources": [{
            "resource_id": "residuals",
            "ref": "residual:1",
            "kind": "sidecar",
            "path": "residuals.json",
            "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "content_hash": content_hash(residuals),
        }],
    }
    trace["trace_hash"] = content_hash(trace)
    return {
        "payload_hash": content_hash({"record": {}}),
        "payload": {
            "request": copy.deepcopy(LEGACY_REQUEST),
            "record": {},
            "legacy_input_hash": shared_hash,
            "input_trace": trace,
            "input_trace_hash": trace["trace_hash"],
        },
    }


def _resign(pair):
    trace = pair["payload"]["input_trace"]
    trace["trace_hash"] = content_hash({
        key: value for key, value in trace.items() if key != "trace_hash"
    })
    pair["payload"]["input_trace_hash"] = trace["trace_hash"]


def _native_with_captured_receipts(pair, tmp_path):
    verified = phase4_real._verified_trace_bundle(pair, tmp_path)
    native = phase4_real.application.score_one(
        verified["request"], verified["inputs"],
    )
    receipts = tuple({
        "stage": stage,
        "input_hash": row["input_hash"],
        "output_hash": row["output_hash"],
        "owner": row["owner"],
    } for stage, row in pair["payload"]["input_trace"]["stages"].items())
    resolved = dict(native.resolved_request)
    resolved["native_stage_receipts"] = receipts
    return with_score_id(replace(native, resolved_request=resolved))


def test_complete_trace_reconstructs_exact_request_and_inputs(tmp_path):
    pair = _pair(tmp_path)

    verified = phase4_real._verified_trace_bundle(pair, tmp_path)

    assert verified["request"].event_id == "event-1"
    assert verified["input_hash"] == pair["payload"]["legacy_input_hash"]
    assert verified["inputs"].source_ref == verified["input_hash"]
    assert verified["inputs"].simulation["residuals"][0]["err_move"] == 0.1
    assert verified["translation_hash"].startswith("sha256:")
    assert tuple(receipt.stage for receipt in verified["inputs"].stage_receipts) == (
        phase4_real._REQUIRED_TRACE_STAGES
    )


def test_repeated_shared_hash_over_different_native_document_is_rejected(tmp_path):
    pair = _pair(tmp_path)
    trace = pair["payload"]["input_trace"]
    trace["shared_inputs"] = {
        "request": copy.deepcopy(trace["request"]),
        "source_rows": ["row-1"],
    }
    shared_hash = content_hash(trace["shared_inputs"])
    trace["shared_input_hash"] = shared_hash
    trace["native_input_hash"] = shared_hash
    trace["native_inputs"]["source_ref"] = shared_hash
    native_inputs = copy.deepcopy(trace["native_inputs"])
    native_inputs["simulation"]["residuals"] = [{
        "event_date": "2025-01-01", "err_move": 0.1,
    }]
    trace["native_inputs_hash"] = content_hash(native_inputs)
    pair["payload"]["legacy_input_hash"] = shared_hash
    _resign(pair)

    with pytest.raises(
        phase4_real._TraceError,
        match="input_trace.native_input_hash: mismatch",
    ):
        phase4_real._verified_trace_bundle(pair, tmp_path)


def test_structurally_different_documents_cannot_claim_complete_translation(
    tmp_path,
):
    pair = _pair(tmp_path)
    trace = pair["payload"]["input_trace"]
    trace["shared_inputs"] = {
        "request": copy.deepcopy(trace["request"]),
        "source_rows": ["row-1"],
    }
    shared_hash = content_hash(trace["shared_inputs"])
    trace["shared_input_hash"] = shared_hash
    trace["native_inputs"]["source_ref"] = shared_hash
    native_inputs = copy.deepcopy(trace["native_inputs"])
    native_inputs["simulation"]["residuals"] = [{
        "event_date": "2025-01-01", "err_move": 0.1,
    }]
    native_hash = content_hash(native_inputs)
    trace["native_input_hash"] = native_hash
    trace["native_inputs_hash"] = native_hash
    pair["payload"]["legacy_input_hash"] = shared_hash
    translation = trace["input_translation"]
    translation["shared_input_hash"] = shared_hash
    translation["native_input_hash"] = native_hash
    translation["mappings"] = [{
        "shared_path": ["request", "event_id"],
        "native_path": ["request", "event_id"],
        "value_hash": content_hash(trace["request"]["event_id"]),
    }]
    translation["derived"][0]["value_hash"] = content_hash(shared_hash)
    translation["translation_hash"] = content_hash({
        key: value for key, value in translation.items()
        if key != "translation_hash"
    })
    _resign(pair)

    with pytest.raises(phase4_real._TraceError, match="leaf coverage mismatch"):
        phase4_real._verified_trace_bundle(pair, tmp_path)


#: R176: a mapping's shared_path/native_path is recorded against the
#: pre-serialization document (tools/phase4_release_assembler.py::_leaves
#: runs before tools/capture_tier0_corpus.py's _prepare_normalized_shared
#: tags a nonfinite float for the stored JSON as
#: {"__nonfinite__": repr(value)} -- one segment deeper). checks/phase4_real
#: reads that same on-disk (already-tagged) form, so its own leaf-walk
#: (_leaf_values) must treat the exact-sentinel dict as the atomic leaf it
#: encodes, or every recorded path to a nonfinite feature is one segment
#: short of a "leaf" by the walker's own (container-recursing) definition.
@pytest.mark.parametrize("extra_model_inputs,leaf_lookup,tagged_repr", [
    pytest.param(
        {"nonfinite_feature": {phase4_real.NONFINITE_KEY: "nan"}},
        ("nonfinite_feature",), "nan", id="top_level-nan",
    ),
    pytest.param(
        {"a": {"b": {phase4_real.NONFINITE_KEY: "inf"}}},
        ("a", "b"), "inf", id="nested-inf",
    ),
    pytest.param(
        {"series": [1.0, {phase4_real.NONFINITE_KEY: "-inf"}, 3.0]},
        ("series", 1), "-inf", id="in_list-neg_inf",
    ),
])
def test_nonfinite_tagged_leaf_resolves_at_its_recorded_path(
    tmp_path, extra_model_inputs, leaf_lookup, tagged_repr,
):
    pair = _pair(tmp_path, extra_model_inputs=extra_model_inputs)

    verified = phase4_real._verified_trace_bundle(pair, tmp_path)

    # Reaching here without _TraceError already proves the recorded mapping
    # resolved at the sentinel-tagged path; also confirm the resolved native
    # document still carries the tag unchanged (this checker never decodes
    # it -- that is a separate downstream concern) at exactly the leaf
    # _leaf_values addressed.
    node = verified["inputs"].features["model_inputs"]
    for segment in leaf_lookup:
        node = node[segment]
    assert node == {phase4_real.NONFINITE_KEY: tagged_repr}


def test_lookalike_one_key_dict_is_not_swallowed_as_a_leaf(tmp_path):
    """Guard against the fix over-matching: a genuine one-key dict whose key
    is NOT the nonfinite sentinel must still fail as a real defect when a
    mapping addresses it one level short of its actual leaf -- the same
    failure shape the sentinel fix resolves, but for a case that is a real
    mismatch and must stay rejected."""
    extra = {"lookalike": {"not_the_sentinel": "oops"}}
    pair = _pair(tmp_path, extra_model_inputs=extra)
    trace = pair["payload"]["input_trace"]
    translation = trace["input_translation"]
    mappings = translation["mappings"]
    target_index = next(
        i for i, row in enumerate(mappings)
        if row["native_path"] == [
            "native_inputs", "features", "model_inputs",
            "lookalike", "not_the_sentinel",
        ]
    )
    row = mappings[target_index]
    row["shared_path"] = row["shared_path"][:-1]
    row["native_path"] = row["native_path"][:-1]
    translation["translation_hash"] = content_hash({
        key: value for key, value in translation.items()
        if key != "translation_hash"
    })
    _resign(pair)

    with pytest.raises(
        phase4_real._TraceError,
        match=r"input_translation\.mappings\[\d+\]: path is not a document leaf",
    ):
        phase4_real._verified_trace_bundle(pair, tmp_path)


def test_stage_hash_mismatch_is_incomparable(tmp_path):
    pair = _pair(tmp_path)
    pair["payload"]["input_trace"]["stages"]["forecast"]["output"]["value"] = 2
    _resign(pair)

    with pytest.raises(phase4_real._TraceError, match="content hash mismatch"):
        phase4_real._verified_trace_bundle(pair, tmp_path)


def test_runtime_receipts_and_final_identities_are_verified(tmp_path):
    pair = _pair(tmp_path)
    verified = phase4_real._verified_trace_bundle(pair, tmp_path)
    native = _native_with_captured_receipts(pair, tmp_path)

    receipts, identities = phase4_real._verify_runtime_execution(
        verified, native,
    )

    assert receipts[-1]["stage"] == "serialization"
    assert identities == {
        "serialization_hash": receipts[-1]["output_hash"],
        "payload_hash": native.payload_hash,
        "request_hash": native.request_hash,
        "score_id": native.score_id,
    }


def test_resigned_captured_stage_receipt_mismatch_is_incomparable(
    tmp_path, monkeypatch,
):
    pair = _pair(tmp_path)
    native = _native_with_captured_receipts(pair, tmp_path)
    forecast = pair["payload"]["input_trace"]["stages"]["forecast"]
    forecast["output"] = {"stage": "forecast", "value": 2}
    forecast["output_hash"] = content_hash(forecast["output"])
    _resign(pair)
    called = False

    def score_one(*args, **kwargs):
        nonlocal called
        called = True
        return native

    monkeypatch.setattr(phase4_real.application, "score_one", score_one)
    corpus = Corpus(
        root=tmp_path,
        index={"corpus_hash": content_hash({"release": 1})},
        pairs={"pair-1": pair},
    )

    release, parity = phase4_real._native_parity(corpus)

    assert called is True
    assert release["population"]["compared"] == 0
    assert release["population"]["incomparable"] == 1
    assert "execution.forecast.output_hash" in release["dispositions"][0]["reason"]
    assert parity["complete"] is False


@pytest.mark.parametrize(
    ("label", "stage", "block", "corruption"),
    (
        ("context", "resolve_context", "output",
         {"decision_cutoff": "2026-09-18T20:00:00Z"}),
        ("features", "features", "output",
         {"model_inputs": {"spot": 999.0}}),
        ("forecast role", "forecast", "input",
         {"role": "unverified-driver", "artifact_ref": "model:corrupt"}),
        ("contracts", "geometry", "input",
         {"eligible_contracts_ref": "contracts:corrupt"}),
        ("geometry", "geometry", "output",
         {"width": 25.0, "selected_contracts": ["wrong-contract"]}),
        ("quotes", "pricing", "input",
         {"quotes_ref": "quotes:corrupt", "fill_model": {"alpha": 1.0}}),
        ("pricing", "pricing", "output", {"entry_cost": 999.0}),
        ("analog population", "analogs", "input",
         {"population_hash": "sha256:" + "0" * 64}),
        ("residual population", "simulation", "input",
         {"residual_ref": "residual:corrupt"}),
        ("exit horizon", "simulation", "input",
         {"exit_date": "2026-09-18", "remaining_dte": 0}),
        ("gate", "gate", "output",
         {"gate_score": 0.0, "gate_pass": False}),
        ("chooser", "chooser", "output",
         {"selected_strategy": "CND-PS"}),
        ("serialization", "serialization", "output",
         {"payload_hash": "sha256:" + "0" * 64}),
    ),
)
def test_each_trace_stage_and_financial_input_corruption_is_rejected(
    tmp_path, label, stage, block, corruption,
):
    pair = _pair(tmp_path)
    pair["payload"]["input_trace"]["stages"][stage][block] = corruption
    _resign(pair)

    message = rf"input_trace\.stages\.{stage}\.{block}: content hash mismatch"
    with pytest.raises(phase4_real._TraceError, match=message):
        phase4_real._verified_trace_bundle(pair, tmp_path)


def _add_resource(pair, tmp_path, *, resource_id, ref, kind, raw):
    path = f"{resource_id}.bin" if kind == "artifact" else f"{resource_id}.json"
    content_digest = None
    if kind == "sidecar":
        content_digest = content_hash(json.loads(raw))
    (tmp_path / path).write_bytes(raw)
    resource = {
        "resource_id": resource_id,
        "ref": ref,
        "kind": kind,
        "path": path,
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }
    if content_digest is not None:
        resource["content_hash"] = content_digest
    pair["payload"]["input_trace"]["resources"].append(resource)
    _resign(pair)
    return path, len(pair["payload"]["input_trace"]["resources"]) - 1


@pytest.mark.parametrize(
    ("label", "resource_id", "ref", "kind", "raw"),
    (
        ("contracts", "contracts", "contracts:1", "sidecar",
         b"""[{"contract":"AAA260918C00100000","strike":100.0}]"""),
        ("quotes", "quotes", "quotes:1", "sidecar",
         b"""[{"ask":3.0,"bid":2.0,"contract":"AAA260918C00100000"}]"""),
        ("analog population", "analogs", "analog:1", "sidecar",
         b"""[{"event_id":"analog-1","return":0.12}]"""),
        ("residual population", "residual_copy", "residual:copy", "sidecar",
         b"""[{"err_crush":-0.02,"err_move":0.1,"event_date":"2025-01-01","pred_abs_move":0.07}]"""),
        ("forecast artifact", "driver_model", "model:driver:1", "artifact",
         b"frozen-driver-artifact"),
    ),
)
def test_each_resource_class_byte_corruption_is_rejected(
    tmp_path, label, resource_id, ref, kind, raw,
):
    pair = _pair(tmp_path)
    path, resource_index = _add_resource(
        pair, tmp_path, resource_id=resource_id, ref=ref, kind=kind, raw=raw,
    )
    (tmp_path / path).write_bytes(b"poisoned-resource")

    with pytest.raises(
        phase4_real._TraceError,
        match=rf"resources\[{resource_index}\]\.sha256: mismatch",
    ):
        phase4_real._verified_trace_bundle(pair, tmp_path)


def test_saved_request_corruption_is_incomparable_before_execution(
    tmp_path, monkeypatch,
):
    pair = _pair(tmp_path)
    trace = pair["payload"]["input_trace"]
    trace["request"] = copy.deepcopy(trace["request"])
    trace["request"]["event_id"] = "event-corrupt"
    _resign(pair)
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("score_one must not run")

    monkeypatch.setattr(phase4_real.application, "score_one", forbidden)
    corpus = Corpus(
        root=tmp_path,
        index={"corpus_hash": content_hash({"release": 1})},
        pairs={"pair-1": pair},
    )
    release, parity = phase4_real._native_parity(corpus)

    assert called is False
    assert release["population"]["compared"] == 0
    assert release["population"]["incomparable"] == 1
    assert "not the translation of the saved legacy request" in (
        release["dispositions"][0]["reason"])
    assert parity["complete"] is False


def test_sidecar_byte_hash_mismatch_is_incomparable(tmp_path):
    pair = _pair(tmp_path)
    (tmp_path / "residuals.json").write_text("[]")

    with pytest.raises(phase4_real._TraceError, match="sha256: mismatch"):
        phase4_real._verified_trace_bundle(pair, tmp_path)


def test_resource_path_escape_is_incomparable(tmp_path):
    pair = _pair(tmp_path)
    pair["payload"]["input_trace"]["resources"][0]["path"] = "../outside.json"
    _resign(pair)

    with pytest.raises(phase4_real._TraceError, match="escapes release root"):
        phase4_real._verified_trace_bundle(pair, tmp_path)


def test_trace_free_release_never_executes_or_completes(tmp_path, monkeypatch):
    pair = _pair(tmp_path)
    del pair["payload"]["input_trace"]
    del pair["payload"]["input_trace_hash"]
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("score_one must not run")

    monkeypatch.setattr(phase4_real.application, "score_one", forbidden)
    corpus = Corpus(
        root=tmp_path,
        index={"corpus_hash": content_hash({"release": 1})},
        pairs={"pair-1": pair},
    )

    release, parity = phase4_real._native_parity(corpus)

    assert called is False
    assert release["complete"] is False
    assert release["population"]["compared"] == 0
    assert release["population"]["incomparable"] == 1
    assert "input_trace: missing" in release["dispositions"][0]["reason"]
    assert parity["same_input_hashes"] is False
    assert parity["stages"] == ()
    assert parity["complete"] is False


def test_verified_bundle_is_passed_to_canonical_execution(tmp_path, monkeypatch):
    pair = _pair(tmp_path)
    seen = []

    def record_call(request, inputs):
        seen.append((request, inputs))
        raise RuntimeError("stop after boundary assertion")

    monkeypatch.setattr(phase4_real.application, "score_one", record_call)
    corpus = Corpus(
        root=tmp_path,
        index={"corpus_hash": content_hash({"release": 1})},
        pairs={"pair-1": pair},
    )

    release, parity = phase4_real._native_parity(corpus)

    assert len(seen) == 1
    assert seen[0][0].event_id == "event-1"
    assert seen[0][1].source_ref == pair["payload"]["legacy_input_hash"]
    assert release["population"]["incomparable"] == 1
    assert parity["complete"] is False


def test_expected_population_comes_from_release_manifest_members(tmp_path):
    corpus = Corpus(
        root=tmp_path,
        index={
            "corpus_hash": content_hash({"release": 1}),
            "pairs": {
                "pair-1": {"payload_hash": content_hash({"pair": 1})},
                "pair-2": {"payload_hash": content_hash({"pair": 2})},
            },
        },
        pairs={},
    )

    release, parity = phase4_real._native_parity(corpus)

    assert release["population"] == {
        "expected": 2,
        "supported": 0,
        "agreed": 0,
        "manifest_bound": True,
        "compared": 0,
        "refused_as_expected": 0,
        "incomparable": 2,
        "excluded": {"research_replay": 0},
    }
    assert [row["fixture_id"] for row in release["dispositions"]] == [
        "pair-1", "pair-2",
    ]
    assert all(
        "declared pair file missing" in row["reason"]
        for row in release["dispositions"]
    )
    assert parity["population"]["expected"] == 2
    assert parity["population"]["manifest_bound"] is True
    assert parity["complete"] is False


# --------------------------------------------------------------------------
# R4-3: the comparator must not silently omit gate_score/gate_threshold,
# ci_low/ci_high/n_analogs, or the contract's entry/exit/execution dates.
# Each field gets its own synthetic pair proving a change in ONLY that
# field is caught, per checks/phase4_real.py's own newly-added dimensions
# ("verdicts", "analogs") and the extended _contract_projection.
# --------------------------------------------------------------------------


def _r43_record(**overrides):
    base = {
        "gate_score": 0.7, "gate_threshold": 0.6, "gate_pass": True,
        "ci_low": -0.0123, "ci_high": 0.0456, "n_analogs": 42,
    }
    base.update(overrides)
    return base


def _r43_native(**overrides):
    base = dict(
        forecasts={}, uncertainty={},
        resolved_request={"ci_low": -0.0123, "ci_high": 0.0456, "n_analogs": 42},
        financial_diagnostics={},
        gate_terms={"gate_score": 0.7, "gate_threshold": 0.6, "gate_pass": True},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_verdicts_and_analogs_dimensions_agree_on_identical_pairs():
    comparison = phase4_real._compare_numeric_outputs(_r43_record(), _r43_native())
    assert comparison["verdicts"]["agree"] is True
    assert comparison["verdicts"]["finding_fields"] == []
    assert comparison["analogs"]["agree"] is True
    assert comparison["analogs"]["finding_fields"] == []


def test_gate_score_only_difference_is_caught():
    native = _r43_native(gate_terms={
        "gate_score": 0.71, "gate_threshold": 0.6, "gate_pass": True,
    })
    comparison = phase4_real._compare_numeric_outputs(_r43_record(), native)
    assert comparison["verdicts"]["agree"] is False
    assert comparison["verdicts"]["finding_fields"] == ["gate_score"]


def test_gate_threshold_only_difference_is_caught():
    native = _r43_native(gate_terms={
        "gate_score": 0.7, "gate_threshold": 0.61, "gate_pass": True,
    })
    comparison = phase4_real._compare_numeric_outputs(_r43_record(), native)
    assert comparison["verdicts"]["agree"] is False
    assert comparison["verdicts"]["finding_fields"] == ["gate_threshold"]


def test_gate_pass_only_difference_is_caught():
    native = _r43_native(gate_terms={
        "gate_score": 0.7, "gate_threshold": 0.6, "gate_pass": False,
    })
    comparison = phase4_real._compare_numeric_outputs(_r43_record(), native)
    assert comparison["verdicts"]["agree"] is False
    assert comparison["verdicts"]["finding_fields"] == ["gate_pass"]


def test_ci_low_only_difference_is_caught():
    native = _r43_native(resolved_request={
        "ci_low": -0.0999, "ci_high": 0.0456, "n_analogs": 42,
    })
    comparison = phase4_real._compare_numeric_outputs(_r43_record(), native)
    assert comparison["analogs"]["agree"] is False
    assert comparison["analogs"]["finding_fields"] == ["ci_low"]


def test_ci_high_only_difference_is_caught():
    native = _r43_native(resolved_request={
        "ci_low": -0.0123, "ci_high": 0.0999, "n_analogs": 42,
    })
    comparison = phase4_real._compare_numeric_outputs(_r43_record(), native)
    assert comparison["analogs"]["agree"] is False
    assert comparison["analogs"]["finding_fields"] == ["ci_high"]


def test_n_analogs_only_difference_is_caught():
    native = _r43_native(resolved_request={
        "ci_low": -0.0123, "ci_high": 0.0456, "n_analogs": 41,
    })
    comparison = phase4_real._compare_numeric_outputs(_r43_record(), native)
    assert comparison["analogs"]["agree"] is False
    assert comparison["analogs"]["finding_fields"] == ["n_analogs"]


def _r43_legs():
    return [{
        "name": "call", "right": "C", "side": "long", "quantity": 1.0,
        "strike": 100.0, "expiry": "2026-09-18", "fill": 1.5,
        "cash_flow": -150.0,
    }]


def test_contract_projection_identical_pair_agrees():
    left = phase4_real._contract_projection(
        _r43_legs(), entry_date="2026-09-16", exit_date="2026-09-17",
        execution_date="2026-09-16",
    )
    right = phase4_real._contract_projection(
        copy.deepcopy(_r43_legs()), entry_date="2026-09-16",
        exit_date="2026-09-17", execution_date="2026-09-16",
    )
    assert left == right


def test_contract_projection_catches_entry_date_only_difference():
    legs = _r43_legs()
    left = phase4_real._contract_projection(
        legs, entry_date="2026-09-16", exit_date="2026-09-17",
        execution_date="2026-09-16",
    )
    right = phase4_real._contract_projection(
        legs, entry_date="2026-09-15", exit_date="2026-09-17",
        execution_date="2026-09-16",
    )
    assert left != right
    assert left["legs"] == right["legs"]
    assert left["entry_date"] != right["entry_date"]


def test_contract_projection_catches_exit_date_only_difference():
    legs = _r43_legs()
    left = phase4_real._contract_projection(
        legs, entry_date="2026-09-16", exit_date="2026-09-17",
        execution_date="2026-09-16",
    )
    right = phase4_real._contract_projection(
        legs, entry_date="2026-09-16", exit_date="2026-09-18",
        execution_date="2026-09-16",
    )
    assert left != right
    assert left["legs"] == right["legs"]
    assert left["exit_date"] != right["exit_date"]


def test_contract_projection_catches_execution_date_only_difference():
    legs = _r43_legs()
    left = phase4_real._contract_projection(
        legs, entry_date="2026-09-16", exit_date="2026-09-17",
        execution_date="2026-09-16",
    )
    right = phase4_real._contract_projection(
        legs, entry_date="2026-09-16", exit_date="2026-09-17",
        execution_date="2026-09-17",
    )
    assert left != right
    assert left["legs"] == right["legs"]
    assert left["execution_date"] != right["execution_date"]


def test_contract_projection_dates_absent_on_both_sides_are_none_not_dropped():
    """A missing date is compared as an explicit None, not silently omitted."""
    projection = phase4_real._contract_projection(_r43_legs())
    assert projection["entry_date"] is None
    assert projection["exit_date"] is None
    assert projection["execution_date"] is None
    assert set(projection) == {"legs", "entry_date", "exit_date", "execution_date"}
