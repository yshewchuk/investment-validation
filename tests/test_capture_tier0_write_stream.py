"""`write()`'s streaming pair/case writer: byte- and hash-identical to the
OLD in-memory `json.dumps`/`content_hash` path, for a synthetic multi-member
chooser whose members share one pool by identity, with NaN/Inf leaves and a
second, differently-nested reference to the same shared pool (nested
fragments). The old algorithm is reconstructed here ONLY as the oracle --
production streams via `capture._write_pair_file`/`_prepare_normalized_shared`
and `DiskCheckpointSink.write_case`'s streaming path.

The tests below this point cover the SEPARATE fix that stores a
`_SHARED_TRACE_DOCUMENTS`-registered document once per corpus version under
`shared/<digest>.json`, referenced (`{"$shared": digest}`) at every
occurrence instead of expanded again -- the fix for a chooser pair file that
carried a full copy of the same fold pool once per ranked member. The
fixtures above never call `register_shared`, so `_prepare_normalized_shared`
still fully expands them (deduped only by LOCAL identity within one call);
that is deliberate and unaffected by the fix below, which only activates for
a node genuinely registered as shared.
"""
from __future__ import annotations

import json
import re

import pandas as pd
import pytest

import checks.tier0_corpus as t0
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


# --------------------------------------------------------------------------
# a REGISTERED shared document is hoisted into shared/ and referenced
# --------------------------------------------------------------------------


def _registered_chooser_candidate(fixture_id: str, members: int = 11) -> tuple[dict, dict]:
    """A `_chooser_legacy_trace`, but the pool is registered with
    `_SHARED_TRACE_DOCUMENTS` the way `engine.score`'s real collector would
    (`Phase4TraceCollector._document`'s `shared()` helper) -- the identity
    check `write()`'s new hoisting logic keys off. Returns ``(candidate,
    pool)``.
    """
    trace = _chooser_legacy_trace(members=members)
    pool = trace["checkpoints"]["chooser"]["value"]["members"][0]["pool"]
    capture._SHARED_TRACE_DOCUMENTS.register_shared((pool,))
    request = {"strategy": "dynamic_short_vol", "ticker": "ABC"}
    record = {"strategy": "dynamic_short_vol", "ticker": "ABC"}
    return {
        "fixture_id": fixture_id, "covers": ["strategy:dynamic_short_vol"],
        "request": request, "record": record, "kind": "dyn_sv_choice",
        "duration": 0.2, "legacy_trace": trace,
    }, pool


def test_a_registered_shared_pool_is_hoisted_and_referenced(tmp_path) -> None:
    """The pair file written for a REGISTERED shared pool carries zero
    copies of it (only references); `shared/` carries exactly one file;
    `payload_hash` is byte-identical to the OLD, fully-expanded oracle; and
    loading the pair back resolves every occurrence to ONE shared Python
    object whose content matches the original pool -- the identity and
    hash-value proofs the fix exists to satisfy.
    """
    candidate, pool = _registered_chooser_candidate("dyn_sv_shared_001")
    try:
        out_dir = tmp_path / "corpus"
        doc = capture.write(
            out_dir, [candidate],
            {"strategy:dynamic_short_vol": ["dyn_sv_shared_001"]},
            pd.Timestamp("2026-01-01"), "snap-1",
        )

        oracle_pair, _ = _rebuild_pair(candidate)
        written_text = (out_dir / "pairs" / "dyn_sv_shared_001.json").read_text()

        # No expanded copy of the pool anywhere in the pair file: every
        # occurrence is a small reference instead.
        assert written_text.count('"tag": "pool"') == 0
        assert written_text.count(capture.SHARED_REF_KEY) == 12  # 11 members + summary
        # The old fully-expanded pair would carry 12 copies of the ~1.7KB
        # pool (>20KB); the referenced pair carries none of it.
        assert len(written_text) < 6000

        shared_files = sorted((out_dir / "shared").glob("*.json"))
        assert len(shared_files) == 1
        assert doc["shared_documents"] == [
            "sha256:" + shared_files[0].stem
        ]

        written_pair = json.loads(written_text)
        assert written_pair["payload_hash"] == oracle_pair["payload_hash"]
        assert doc["pairs"]["dyn_sv_shared_001"]["payload_hash"] == oracle_pair["payload_hash"]

        corpus = t0.load(out_dir)
        loaded_payload = corpus.pairs["dyn_sv_shared_001"]["payload"]
        # The re-hash proof `checks/tier0_corpus.py` itself relies on
        # (its "digest" check, §7.3 item 3): plain content_hash, no
        # fragments, over the RESOLVED payload reproduces payload_hash.
        assert content_hash(loaded_payload) == oracle_pair["payload_hash"]

        members = loaded_payload["legacy_trace"]["checkpoints"]["chooser"]["value"]["members"]
        pools = [m["pool"] for m in members]
        summary = loaded_payload["legacy_trace"]["checkpoints"]["chooser"]["value"][
            "summary"]["by_fold"]["fold-0"]["details"][0]["pool_ref"]
        pools.append(summary)
        first = pools[0]
        assert all(p is first for p in pools)  # one shared object, not 12 copies
        # Its expanded recompute matches the source pool -- tag_nonfinite'd,
        # since that is the JSON-safe form a real file round-trips through.
        assert first == capture.tag_nonfinite(pool)
    finally:
        capture._cleanup_trace_spill()


def test_the_case_document_also_references_the_shared_pool(tmp_path) -> None:
    candidate, pool = _registered_chooser_candidate("dyn_sv_shared_002")
    try:
        out_dir = tmp_path / "corpus"
        capture.write(
            out_dir, [candidate],
            {"strategy:dynamic_short_vol": ["dyn_sv_shared_002"]},
            pd.Timestamp("2026-01-01"), "snap-1",
        )
        case_text = (
            out_dir / "checkpoints" / "cases" / "dyn_sv_shared_002.json"
        ).read_text()
        assert case_text.count('"tag":"pool"') == 0  # compact separators, no spaces
        assert case_text.count(capture.SHARED_REF_KEY) == 12
    finally:
        capture._cleanup_trace_spill()


def test_two_pairs_served_by_the_same_fold_write_the_shared_file_once(tmp_path) -> None:
    """Two DIFFERENT DYN-SV choosers sharing the SAME served fold pool (the
    measured 09-19 scenario: both 11-member choosers) write it once, not
    once per pair."""
    trace_a = _chooser_legacy_trace(members=11)
    pool = trace_a["checkpoints"]["chooser"]["value"]["members"][0]["pool"]
    capture._SHARED_TRACE_DOCUMENTS.register_shared((pool,))
    trace_b = {
        "schema_version": "phase4_legacy_diagnostic_checkpoint.v1.0",
        "disposition": {"status": "completed", "flags": []},
        "checkpoints": {
            "chooser": {
                "value": {"members": [{"member_index": i, "pool": pool, "rank": i}
                                       for i in range(11)],
                          "summary": {}},
                "content_hash": "sha256:" + "3" * 64,
            },
        },
    }
    candidate_a = {
        "fixture_id": "dyn_sv_a", "covers": ["strategy:dynamic_short_vol"],
        "request": {"strategy": "dynamic_short_vol", "ticker": "A"},
        "record": {"strategy": "dynamic_short_vol", "ticker": "A"},
        "kind": "dyn_sv_choice", "duration": 0.2, "legacy_trace": trace_a,
    }
    candidate_b = {
        "fixture_id": "dyn_sv_b", "covers": ["strategy:dynamic_short_vol"],
        "request": {"strategy": "dynamic_short_vol", "ticker": "B"},
        "record": {"strategy": "dynamic_short_vol", "ticker": "B"},
        "kind": "dyn_sv_choice", "duration": 0.2, "legacy_trace": trace_b,
    }
    try:
        out_dir = tmp_path / "corpus"
        doc = capture.write(
            out_dir, [candidate_a, candidate_b],
            {"strategy:dynamic_short_vol": ["dyn_sv_a", "dyn_sv_b"]},
            pd.Timestamp("2026-01-01"), "snap-1",
        )
        assert len(doc["shared_documents"]) == 1
        assert len(list((out_dir / "shared").glob("*.json"))) == 1

        corpus = t0.load(out_dir)
        pool_a = corpus.pairs["dyn_sv_a"]["payload"]["legacy_trace"][
            "checkpoints"]["chooser"]["value"]["members"][0]["pool"]
        pool_b = corpus.pairs["dyn_sv_b"]["payload"]["legacy_trace"][
            "checkpoints"]["chooser"]["value"]["members"][0]["pool"]
        assert pool_a is pool_b
    finally:
        capture._cleanup_trace_spill()


# --------------------------------------------------------------------------
# input_translation.mappings row tables (separate fix: 09-19)
#
# A DYN-SV chooser's native members each carry a near-identical
# `input_translation.mappings` row list (measured: ~359,406 rows per member,
# 99.997% identical, differing by ~14 rows apiece) -- whole-node identity
# sharing above never matches on it, since `_translation` builds a fresh row
# list per member. `_TranslationTableWriter` stores the row content once and
# gives every occurrence a small `{"$rows": ...}` delta instead.
# --------------------------------------------------------------------------


def _translation_row(shared_path: list, native_path: list, value) -> dict:
    return {"shared_path": shared_path, "native_path": native_path,
            "value_hash": content_hash(value)}


def _chooser_translation_candidate(fixture_id: str, members: int = 7,
                                    rows: int = 400, differs: int = 14):
    """A synthetic `dyn_sv_choice` candidate whose `members[i].input_trace.
    input_translation.mappings` mimics a real DYN-SV chooser trace: every
    member carries a near-identical row list, `differs` rows apart from its
    siblings -- the measured shape a real frozen-chooser fold pool produces.
    Bypasses `attach_strict_probe`/a real Scorer entirely (like the
    `legacy_trace`-only candidates above): `write()` only needs
    `make_pair`'s two checks (`legacy_input_hash` == `input_trace.
    shared_input_hash`, and `input_trace.trace_hash` present), so a synthetic
    trace with placeholder hashes elsewhere is enough to exercise the
    row-table compaction it triggers. Returns ``(candidate, member_row_lists)``
    where ``member_row_lists[i]`` is member i's ORIGINAL, full, unmutated row
    list -- kept for later comparison after `write()` mutates the candidate's
    own `input_trace` in place (storage-side compaction).
    """
    # `_translation` (`tools/phase4_release_assembler.py`) always hands the
    # real writer an already-`repr(shared_path)`-sorted list (the only
    # `mappings` this capture ever builds is auto-derived, never
    # caller-ordered) -- `_TranslationTableWriter.reference` relies on and
    # enforces that invariant, so this fixture must respect it too, exactly
    # as a real capture would.
    base_rows = sorted(
        (_translation_row(["c", i], ["c", i], i) for i in range(rows)),
        key=capture._translation_row_sort_key,
    )
    member_row_lists = []
    for m in range(members):
        member_rows = list(base_rows)
        for offset in range(differs):
            idx = (m * 37 + offset * 7) % rows
            member_rows[idx] = _translation_row(
                member_rows[idx]["shared_path"], member_rows[idx]["native_path"],
                f"member-{m}-{offset}")
        member_row_lists.append(member_rows)
    member_traces = []
    for m, member_rows in enumerate(member_row_lists):
        translation = {
            "schema_version": capture.TRANSLATION_SCHEMA,
            "shared_input_hash": "sha256:" + f"{m:064d}",
            "native_input_hash": "sha256:" + f"{m:064d}",
            "mappings": member_rows,
            "derived": [],
            "translation_hash": "sha256:" + f"{m:064d}",
        }
        member_traces.append({
            "member_index": m,
            "input_trace": {
                "schema_version": "phase4_input_trace.v1.0",
                "input_translation": translation,
                "trace_hash": "sha256:" + f"{m:064d}",
            },
        })
    input_trace = {
        "schema_version": "phase4_chooser_trace.v1.0",
        "members": member_traces,
        "shared_input_hash": "sha256:" + "0" * 64,
        "trace_hash": "sha256:" + "0" * 64,
    }
    candidate = {
        "fixture_id": fixture_id, "covers": ["strategy:dynamic_short_vol"],
        "request": {"strategy": "dynamic_short_vol", "ticker": "ABC"},
        "record": {"strategy": "dynamic_short_vol", "ticker": "ABC"},
        "kind": "dyn_sv_choice", "duration": 0.2,
        "input_trace": input_trace, "legacy_input_hash": "sha256:" + "0" * 64,
    }
    return candidate, member_row_lists


def _oracle_pair_for_input_trace(candidate: dict) -> dict:
    """`make_pair`'s result over `candidate`'s CURRENT `input_trace`, with no
    storage-side compaction applied -- the same call `write()`'s loop makes
    internally before `_compact_translation_mappings` ever runs. Callers
    must capture the returned `payload_hash` (a string) BEFORE calling
    `capture.write(...)` on the same candidate: `write()` mutates
    `candidate["input_trace"]` in place (storage-side reference swap).
    """
    return capture.make_pair(
        candidate["fixture_id"], candidate["covers"], candidate["request"],
        candidate["record"], record_kind=candidate["kind"],
        duration=candidate["duration"], input_trace=candidate["input_trace"],
        legacy_input_hash=candidate["legacy_input_hash"],
    )


def test_translation_mappings_are_compacted_and_reconstruct_exactly(tmp_path) -> None:
    """A synthetic 7-member DYN-SV chooser candidate: `write()` stores the
    row content ONCE under `shared/translations/`, the written pair file
    carries no full copy per member, `t0.load()` reconstructs each member's
    EXACT original row list (order included), and `payload_hash` is
    unchanged from the pre-compaction oracle -- hashes are taken over the
    LOGICAL fully expanded document, storage-side compaction never touches
    them.
    """
    candidate, member_row_lists = _chooser_translation_candidate("dyn_sv_rows_001")
    oracle_pair = _oracle_pair_for_input_trace(candidate)
    oracle_payload_hash = oracle_pair["payload_hash"]
    try:
        out_dir = tmp_path / "corpus"
        doc = capture.write(
            out_dir, [candidate],
            {"strategy:dynamic_short_vol": ["dyn_sv_rows_001"]},
            pd.Timestamp("2026-01-01"), "snap-1",
        )
        written_text = (out_dir / "pairs" / "dyn_sv_rows_001.json").read_text()
        assert written_text.count(capture.ROWS_REF_KEY) == len(member_row_lists)
        # One member's worth of rows on disk (the first table written), not
        # `members` copies of it: the old fully-expanded pair would be
        # several times larger than one member's own row list alone.
        assert len(written_text) < 300_000

        tables = sorted((out_dir / "shared" / "translations").glob("*.json"))
        assert len(tables) == 1
        assert doc["shared_translation_tables"] == ["sha256:" + tables[0].stem]
        assert doc["pairs"]["dyn_sv_rows_001"]["payload_hash"] == oracle_payload_hash

        corpus = t0.load(out_dir)
        loaded_payload = corpus.pairs["dyn_sv_rows_001"]["payload"]
        loaded_members = loaded_payload["input_trace"]["members"]
        assert len(loaded_members) == len(member_row_lists)
        for member, expected_rows in zip(loaded_members, member_row_lists):
            got = member["input_trace"]["input_translation"]["mappings"]
            expected = sorted(
                expected_rows,
                key=lambda r: (repr(r["shared_path"]), repr(r["native_path"])))
            assert got == expected
        # Hash-oracle equivalence: the RESOLVED, fully expanded payload
        # hashes back to the same payload_hash `make_pair` computed over the
        # ORIGINAL, pre-compaction document.
        assert content_hash(loaded_payload) == oracle_payload_hash
    finally:
        capture._cleanup_trace_spill()


def test_a_translation_table_writer_prefers_the_best_overlapping_table(tmp_path) -> None:
    """Given three already-written tables, a new occurrence references
    whichever one shares the most rows with it -- not merely the first or
    the most recent -- so a corpus with several distinct dominant row sets
    (e.g. two different chooser folds) still gets a small delta for each
    new occurrence of either one.
    """
    writer = capture._TranslationTableWriter(tmp_path / "shared")
    pool_a = sorted((_translation_row(["a", i], ["a", i], i) for i in range(100)),
                    key=capture._translation_row_sort_key)
    pool_b = sorted((_translation_row(["b", i], ["b", i], i) for i in range(100)),
                    key=capture._translation_row_sort_key)
    ref_a = writer.reference(pool_a)
    ref_b = writer.reference(pool_b)
    assert ref_a["omit"] == ref_a["extra"] == []
    assert ref_b["omit"] == ref_b["extra"] == []
    assert ref_a[capture.ROWS_REF_KEY] != ref_b[capture.ROWS_REF_KEY]

    # A third occurrence differs from pool_b by 3 rows and from pool_a by
    # every row: it must reference pool_b's table, with a 3-row delta each
    # way, not pool_a's (which would need a ~100-row delta).
    near_b = list(pool_b)
    for i in (1, 2, 3):
        near_b[i] = _translation_row(
            near_b[i]["shared_path"], near_b[i]["native_path"], f"near-b-{i}")
    ref_near_b = writer.reference(near_b)
    assert ref_near_b[capture.ROWS_REF_KEY] == ref_b[capture.ROWS_REF_KEY]
    assert len(ref_near_b["omit"]) == 3
    assert len(ref_near_b["extra"]) == 3
    assert len(writer.digests()) == 2  # no third table written


def test_an_unsorted_mappings_list_is_a_hard_refusal(tmp_path) -> None:
    """Reconstruction always re-sorts by ``(repr(shared_path),
    repr(native_path))``; compacting a list that is not ALREADY in that
    order would silently change the array's order (and so its content
    hash) on the round trip. Caught here rather than trusted.
    """
    rows = [_translation_row(["a", 1], ["a", 1], 1),
            _translation_row(["a", 0], ["a", 0], 0)]  # out of order
    writer = capture._TranslationTableWriter(tmp_path / "shared")
    with pytest.raises(ValueError):
        writer.reference(rows)


def test_a_duplicate_row_in_one_mappings_list_is_a_hard_refusal(tmp_path) -> None:
    """Two distinct leaf paths can never legitimately collapse to the same
    (native_path, shared_path, value_hash) triple within one member (the
    upstream translation builder already rejects a duplicate path) -- a
    duplicate row reaching the writer is a data-integrity bug, refused
    loudly rather than silently dropped (which would corrupt `omit`/`extra`
    bookkeeping for every later reference against this table).
    """
    writer = capture._TranslationTableWriter(tmp_path / "shared")
    row = _translation_row(["a", 0], ["a", 0], 0)
    with pytest.raises(ValueError):
        writer.reference([row, dict(row)])
