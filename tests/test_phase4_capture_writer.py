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


def test_writer_keeps_infinite_residuals_in_the_canonical_form(tmp_path) -> None:
    """Legacy ResidualPool keeps a +/-inf residual (R4-20 gap 1), so the
    captured residual_population must carry it. It is written as the repo's
    canonical {"__nonfinite__": ...} tag (engine.v2.foundation.canonical),
    under the content hash already recorded, never as null or bare Infinity."""
    from engine.v2.foundation import content_hash
    from engine.v2.foundation.canonical import untag_nonfinite

    population = [
        {"event_date": "2025-01-02", "pred_abs_move": 4.0,
         "err_move": float("inf"), "err_crush": -1.0},
        {"event_date": "2025-01-03", "pred_abs_move": 5.0,
         "err_move": 0.5, "err_crush": float("-inf")},
    ]
    value = {"residual_population": population, "draw_count": 4}
    checkpoint = {
        "schema_version": "phase4_legacy_diagnostic_checkpoint.v1.0",
        "disposition": {"status": "completed", "flags": []},
        "checkpoints": {"simulation": {"value": value,
                                       "content_hash": content_hash(value)}},
    }
    candidate = {
        "fixture_id": "case-inf", "covers": ["strategy:STR-THRU"],
        "request": {"strategy": "STR-THRU", "ticker": "ABC"},
        "record": {"strategy": "STR-THRU", "ticker": "ABC"},
        "kind": "score_result", "duration": 0.1, "legacy_trace": checkpoint,
    }
    release = tmp_path / "release"
    write(release, [candidate], {"strategy:STR-THRU": ["case-inf"]},
          pd.Timestamp("2026-01-01"), "snapshot-1")

    def reject(constant):
        raise AssertionError(f"bare {constant} written")

    case = json.loads((release / "checkpoints" / "cases" / "case-inf.json").read_text(),
                      parse_constant=reject)
    pair = json.loads((release / "pairs" / "case-inf.json").read_text(),
                      parse_constant=reject)
    for trace in (case["checkpoint"], pair["payload"]["legacy_trace"]):
        row = trace["checkpoints"]["simulation"]
        written = row["value"]["residual_population"]
        assert written[0]["err_move"] == {"__nonfinite__": "inf"}
        assert written[1]["err_crush"] == {"__nonfinite__": "-inf"}
        assert row["content_hash"] == content_hash(row["value"]) == content_hash(value)
        decoded = untag_nonfinite(written)
        assert decoded[0]["err_move"] == float("inf")
        assert decoded[1]["err_crush"] == float("-inf")
        assert decoded[0]["err_crush"] == -1.0
