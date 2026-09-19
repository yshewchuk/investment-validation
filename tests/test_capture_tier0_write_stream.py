"""`write()`'s streaming pair/case writer: byte- and hash-identical to the
OLD in-memory `json.dumps`/`content_hash` path, for a synthetic multi-member
chooser whose members share one pool by identity, with NaN/Inf leaves and a
second, differently-nested reference to the same shared pool (nested
fragments). The old algorithm is reconstructed here ONLY as the oracle --
production streams via `capture._write_pair_file`/`_prepare_normalized_shared`
and `DiskCheckpointSink.write_case`'s streaming path.
"""
from __future__ import annotations

import re

import pandas as pd

import tools.capture_tier0_corpus as capture
from engine.v2.foundation.canonical import content_hash
from tools.phase4_checkpoint_sink import _json_bytes


def _shared_pool(rows: int = 40) -> dict:
    """A pool with NaN/Inf, -0.0, big/small exponents and unicode -- the
    kinds of leaves `tag_nonfinite`/`_scalar` treat specially.
    """
    return {
        "predictions": [0.001 * i for i in range(rows)],
        "residuals": [
            float("nan") if i % 11 == 0 else
            (float("inf") if i % 17 == 0 else -0.002 * i)
            for i in range(rows)
        ],
        "meta": {
            "tag": "pool",
            "note": "unicode éé emoji \U0001F600",
            "neg_zero": -0.0,
            "big": 1e21,
            "small": 1e-7,
        },
    }


def _chooser_legacy_trace(members: int = 11) -> dict:
    """One `legacy_trace` shaped like a `dyn_sv_choice` pair's: the SAME pool
    object embedded once per ranked member, PLUS a second, differently
    nested reference to it (nested fragments) -- two independent sharing
    paths to the one object.
    """
    pool = _shared_pool()
    return {
        "schema_version": "phase4_legacy_diagnostic_checkpoint.v1.0",
        "disposition": {"status": "completed", "flags": []},
        "checkpoints": {
            "chooser": {
                "value": {
                    "members": [
                        {
                            "member_index": i,
                            "pool": pool,  # identity-shared across members
                            "rank": i,
                            "score": None if i == 0 else 1.5 - 0.01 * i,
                        }
                        for i in range(members)
                    ],
                    # a second, differently nested path to the same object
                    "summary": {"by_fold": {"fold-0": {"details": [{"pool_ref": pool}]}}},
                },
                "content_hash": "sha256:" + "2" * 64,
            },
        },
    }


_CAPTURED_AT = re.compile(r'"captured_at": "[^"]*"')


def _without_wall_clock(text: str) -> str:
    """`envelope.captured_at` is deliberately wall-clock (contracts §2.2,
    excluded from `payload_hash`) -- it legitimately differs between two
    independent `make_pair` calls for the same candidate. Blank it out so
    the rest of the byte-identity comparison is exact.
    """
    return _CAPTURED_AT.sub('"captured_at": "REDACTED"', text)


def _oracle_pair_text(pair: dict) -> str:
    """The pair-file bytes `write()` produced BEFORE streaming: one
    `json.dumps(tag_nonfinite(pair), indent=2, sort_keys=True,
    allow_nan=False)` call. Kept here only as the test oracle.
    """
    import json
    return json.dumps(capture.tag_nonfinite(pair), indent=2, sort_keys=True,
                       allow_nan=False) + "\n"


def _oracle_payload_hash(payload: dict) -> str:
    """`payload_hash` BEFORE streaming: the batch `content_hash`. Kept here
    only as the test oracle -- production now calls `stream_content_hash`.
    """
    return content_hash(payload, fragments=capture._SHARED_TRACE_DOCUMENTS)


def _rebuild_pair(candidate: dict) -> dict:
    """Rebuild exactly the `pair` `write()`'s loop body constructs for one
    candidate, using a fresh cache the same way `write()` does. Deterministic:
    `_prepare_normalized_shared` is a pure function of its input, so this
    matches what `write()` built internally byte for byte, regardless of
    cache reuse.
    """
    cache: dict = {}
    checkpoint = candidate.get("legacy_trace")
    if checkpoint is not None:
        checkpoint = capture._prepare_normalized_shared(checkpoint, cache)
    return capture.make_pair(
        candidate["fixture_id"], candidate["covers"], candidate["request"],
        candidate["record"], record_kind=candidate["kind"],
        duration=candidate["duration"], legacy_trace=checkpoint,
    ), checkpoint


def test_write_streams_a_multi_member_chooser_byte_identical(tmp_path) -> None:
    trace = _chooser_legacy_trace(members=11)
    request = {"strategy": "dynamic_short_vol", "ticker": "ABC"}
    record = {"strategy": "dynamic_short_vol", "ticker": "ABC"}
    candidate = {
        "fixture_id": "dyn_sv_choice_001",
        "covers": ["strategy:dynamic_short_vol"],
        "request": request, "record": record, "kind": "dyn_sv_choice",
        "duration": 0.2,
        "legacy_trace": trace,
    }
    try:
        out_dir = tmp_path / "corpus"
        doc = capture.write(
            out_dir, [candidate],
            {"strategy:dynamic_short_vol": ["dyn_sv_choice_001"]},
            pd.Timestamp("2026-01-01"), "snap-1",
        )

        pair, _ = _rebuild_pair(candidate)
        written_text = (out_dir / "pairs" / "dyn_sv_choice_001.json").read_text()

        assert _without_wall_clock(written_text) == _without_wall_clock(_oracle_pair_text(pair))
        assert pair["payload_hash"] == _oracle_payload_hash(pair["payload"])
        assert doc["pairs"]["dyn_sv_choice_001"]["payload_hash"] == pair["payload_hash"]

        # The written file genuinely carries the pool once per member plus
        # the summary's own reference -- real duplicated output, not a
        # dropped/skipped occurrence, confirming the fix streams rather than
        # silently thinning the document.
        assert written_text.count('"tag": "pool"') == 12
    finally:
        capture._cleanup_trace_spill()


def test_write_streams_case_file_byte_identical_for_the_same_chooser(tmp_path) -> None:
    trace = _chooser_legacy_trace(members=11)
    request = {"strategy": "dynamic_short_vol", "ticker": "ABC"}
    record = {"strategy": "dynamic_short_vol", "ticker": "ABC"}
    candidate = {
        "fixture_id": "dyn_sv_choice_002",
        "covers": ["strategy:dynamic_short_vol"],
        "request": request, "record": record, "kind": "dyn_sv_choice",
        "duration": 0.2,
        "legacy_trace": trace,
    }
    try:
        out_dir = tmp_path / "corpus"
        capture.write(
            out_dir, [candidate],
            {"strategy:dynamic_short_vol": ["dyn_sv_choice_002"]},
            pd.Timestamp("2026-01-01"), "snap-1",
        )

        _, checkpoint = _rebuild_pair(candidate)
        oracle_case = {
            "case_id": "dyn_sv_choice_002",
            "request": request,
            "strategy": record.get("strategy"),
            "covers": ["strategy:dynamic_short_vol"],
            "record_kind": "dyn_sv_choice",
            "checkpoint": checkpoint,
        }
        oracle_bytes = _json_bytes(oracle_case)
        written_bytes = (
            out_dir / "checkpoints" / "cases" / "dyn_sv_choice_002.json"
        ).read_bytes()
        assert written_bytes == oracle_bytes
    finally:
        capture._cleanup_trace_spill()


def test_shared_pool_is_one_object_not_duplicated_after_preparing(tmp_path) -> None:
    """`_prepare_normalized_shared` must return the SAME prepared object for
    every occurrence of a shared container -- the whole point of the fix.
    A regression here (falling back to plain `tag_nonfinite` semantics)
    would silently reintroduce the per-member duplication this task exists
    to remove, while every byte-identity test above would still pass.
    """
    pool = _shared_pool(rows=5)
    value = {"members": [{"pool": pool} for _ in range(4)],
             "elsewhere": {"deep": {"ref": pool}}}
    cache: dict = {}
    prepared = capture._prepare_normalized_shared(value, cache)
    prepared_pools = [m["pool"] for m in prepared["members"]]
    prepared_pools.append(prepared["elsewhere"]["deep"]["ref"])
    first = prepared_pools[0]
    assert all(p is first for p in prepared_pools)
