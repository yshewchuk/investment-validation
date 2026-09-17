from __future__ import annotations

import json

import pandas as pd

from tools.capture_tier0_corpus import write


def test_writer_persists_bounded_checkpoint_manifest(tmp_path) -> None:
    checkpoint = {
        "schema_version": "phase4_legacy_diagnostic_checkpoint.v1.0",
        "disposition": {"status": "completed", "flags": []},
        "checkpoints": {
            "features": {
                "value": {
                    "feature_vector": {"spot": 100.0},
                    "missing_mask": {"spot": False},
                    "model_identity": {"model_id": "m1"},
                },
                "content_hash": "sha256:" + "0" * 64,
            },
        },
    }
    candidate = {
        "fixture_id": "case-1",
        "covers": ["strategy:STR-THRU"],
        "request": {"strategy": "STR-THRU", "ticker": "ABC"},
        "record": {"strategy": "STR-THRU", "ticker": "ABC"},
        "kind": "score_result",
        "duration": 0.1,
        "legacy_trace": checkpoint,
    }

    release = tmp_path / "release"
    write(
        release,
        [candidate],
        {"strategy:STR-THRU": ["case-1"]},
        pd.Timestamp("2026-01-01"),
        "snapshot-1",
    )

    manifest = json.loads(
        (release / "checkpoints" / "manifest.json").read_text()
    )
    case = json.loads(
        (release / "checkpoints" / "cases" / "case-1.json").read_text()
    )

    assert manifest["cases"] == [{
        "case_id": "case-1",
        "sha256": manifest["cases"][0]["sha256"],
    }]
    assert case["checkpoint"] == checkpoint
    assert "record" not in case
    assert json.loads((release / "INDEX.json").read_text())[
        "diagnostic_checkpoint_manifest"
    ] == "checkpoints/manifest.json"
