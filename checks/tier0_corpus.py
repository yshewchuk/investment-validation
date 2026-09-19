#!/usr/bin/env python3
"""Replay the tier-0 corpus — seconds, no panel, no network, no fitting.

    python3 checks/tier0_corpus.py
    python3 checks/tier0_corpus.py --corpus fixtures/tier0 --json

The corpus is the migration's **oracle**: frozen `(request, record)` pairs
captured once by ``tools/capture_tier0_corpus.py`` through the real public
entry points. Every later phase states its exit gate as a comparison against
it, so this check's job is to prove the oracle is well-formed, addressable,
covering and able to catch the defects it exists for — fast enough to run on
every edit.

**What tier 0 proves.**

1. **manifest membership** — the files on disk are EXACTLY the pairs the index
   declares, each file's hashes, kind and covers match the manifest row, and
   the corpus hash re-computes from the survivors. The expected population
   comes from the DECLARED manifest, never from what happened to load.
2. **addressing** — every record is reachable from the content hash of its own
   full-precision request. A rounded request hashes differently and stops
   resolving, which is what makes `b33036c` structurally impossible to hide.
3. **digest** — the payload on disk re-hashes to the ``payload_hash`` beside
   it. `6b9d5cf` was a file disagreeing with its own digest.
4. **serialized round trip** — each record written to a real file and read
   back compares equal.
5. **coverage, re-derived** — every axis claim is recomputed from the surviving
   records by :func:`derive_covers`, the single definition the capture also
   uses. The index's claims are compared against it, never trusted.
6. **pinned counterparts** — every pinned fixture names the selector-resolved
   fixture it was pinned from, that fixture is present, and the two agree on
   the forecast block and the legs: the `e845f3e` regression, on real data.
7. **seeded negative controls** — contracts §15.3: the four seedable
   2026-09-11 causes planted into four distinct REAL frozen pairs, one pass,
   each localized to its stage with nothing else moving. The spec lives in
   :mod:`checks.replay_identity`.
8. **loader determinism** — batch versus single, and a fresh process.

**What tier 0 does not prove.** It does not re-score: §11 lists "a corpus that
fits or fetches" as a failure mode, and re-scoring is ``tools/replay_tier1.py``.
Nor does it run the engine-level parity matrix — reordered inputs, batching and
restart through the scorer — which is rearchitecture phase 1's acceptance test
O30. An earlier version ran a "reordered inputs" case here that iterated sorted
ids whichever way the files were loaded; it could not fail, so it is gone
rather than kept as a claim.

Stdlib plus ``engine/v2/diagnosis`` only. It must run in a bare checkout with
no pandas, no store and no models, or the "every edit" commitment is not real.
"""
from __future__ import annotations

import argparse
import copy
import json
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.replay_identity import (  # noqa: E402
    FORECAST_BLOCK,
    SEEDED_CONTROLS,
    check_control,
    pick_seed_targets,
)
from engine.v2.diagnosis import (  # noqa: E402
    AGREE,
    DIFFER,
    INCOMPARABLE,
    ComparisonReceipt,
    compare_records,
    content_hash,
    flatten,
    merge_receipts,
    problem,
)

__all__ = ["Corpus", "load", "run", "main", "derive_covers", "seeded_controls",
           "derived_uncovered", "TIME_BUDGET_SECONDS", "CorpusFormatError",
           "SHARED_REF_KEY", "SHARED_DOCUMENT_SCHEMA_VERSION"]

DEFAULT_CORPUS = ROOT / "fixtures" / "tier0"

#: §7.3: total runtime under ten seconds, network disabled.
TIME_BUDGET_SECONDS = 10.0

#: The marker ``capture_tier0_corpus.jsonable`` freezes a NaN/Infinity under.
NONFINITE = "__nonfinite__"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


@dataclass
class Corpus:
    root: Path
    index: dict[str, Any]
    pairs: dict[str, dict] = field(default_factory=dict)

    @property
    def ordered_ids(self) -> list[str]:
        return sorted(self.pairs)

    def record_of(self, fixture_id: str) -> dict:
        return self.pairs[fixture_id]["payload"]["record"]

    def request_of(self, fixture_id: str) -> dict:
        return self.pairs[fixture_id]["payload"]["request"]


#: A pair's payload may embed a frozen document `tools/capture_tier0_corpus.py`
#: identified as shared (a served fold pool, residual pool or payoff fit) by
#: reference instead of in full: `{SHARED_REF_KEY: "sha256:<64 hex>"}` and
#: NOTHING else in that dict. This module carries the SAME two literals as
#: `tools/capture_tier0_corpus.py` independently rather than importing them
#: -- this module must run in a bare checkout with no pandas, no store and no
#: models (module docstring), and the writer needs all three. One
#: convention, not two (see `engine/v2/foundation/canonical.py`'s
#: `NONFINITE_KEY` docstring for the same argument); kept in sync by
#: `tests/test_tier0_shared_documents.py`.
SHARED_REF_KEY = "$shared"
SHARED_DOCUMENT_SCHEMA_VERSION = "tier0_shared_document.v1.0"


class CorpusFormatError(ValueError):
    """A pair's shared-document reference, or the shared document itself,
    does not match the format this loader understands. Raised loudly --
    never silently skipped or replaced with a partial value."""


def _resolve_shared(node: Any, shared_dir: Path, cache: dict[str, Any],
                     resolving: set[str]) -> Any:
    """Walk ``node``, replacing every ``{SHARED_REF_KEY: digest}`` reference
    with the ONE Python object ``_load_shared_document`` reads for that
    digest -- the same object at every occurrence, restoring the capture's
    own in-memory sharing (``cache`` is keyed by digest and shared across
    every pair one :func:`load` call reads). A dict carrying ``SHARED_REF_KEY``
    alongside any other key is an unknown reference shape: refused loudly,
    never silently treated as ordinary data. An old-format document with no
    reference nodes at all passes through unchanged.
    """
    if isinstance(node, dict):
        if SHARED_REF_KEY in node:
            if set(node) != {SHARED_REF_KEY}:
                raise CorpusFormatError(
                    "unknown shared-reference shape (expected only "
                    f"{SHARED_REF_KEY!r}): {sorted(node)}")
            digest = node[SHARED_REF_KEY]
            if not isinstance(digest, str) or not digest.startswith("sha256:"):
                raise CorpusFormatError(f"malformed shared digest: {digest!r}")
            resolved = cache.get(digest)
            if resolved is None:
                if digest in resolving:
                    raise CorpusFormatError(f"cyclic shared-document reference: {digest}")
                resolving.add(digest)
                try:
                    resolved = _load_shared_document(shared_dir, digest, cache, resolving)
                finally:
                    resolving.discard(digest)
                cache[digest] = resolved
            return resolved
        return {k: _resolve_shared(v, shared_dir, cache, resolving) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_shared(v, shared_dir, cache, resolving) for v in node]
    return node


def _load_shared_document(shared_dir: Path, digest: str, cache: dict[str, Any],
                           resolving: set[str]) -> Any:
    """Read, verify and resolve one ``shared/<hex>.json`` document.

    A missing file, a malformed body, an unsupported schema version, a
    ``digest`` field that disagrees with the requested digest, or content
    that does not hash back to it are all hard refusals -- never a silent
    skip or a partial/best-effort value.
    """
    hexpart = digest.split(":", 1)[-1]
    path = shared_dir / f"{hexpart}.json"
    if not path.is_file():
        raise CorpusFormatError(f"missing shared document for {digest}: {path}")
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise CorpusFormatError(f"unreadable shared document {digest}: {exc}") from exc
    if not isinstance(doc, dict) or set(doc) != {"schema_version", "digest", "value"}:
        raise CorpusFormatError(f"malformed shared document {digest}: {path}")
    if doc.get("schema_version") != SHARED_DOCUMENT_SCHEMA_VERSION:
        raise CorpusFormatError(
            f"unsupported shared document schema {doc.get('schema_version')!r} "
            f"for {digest}")
    if doc.get("digest") != digest:
        raise CorpusFormatError(
            f"shared document {path} declares digest {doc.get('digest')!r}, "
            f"expected {digest}")
    # A shared document may itself reference another (nested sharing):
    # resolve those FIRST, so the hash below is taken over the same fully
    # expanded logical value the writer computed `digest` from.
    value = _resolve_shared(doc["value"], shared_dir, cache, resolving)
    actual = content_hash(value)
    if actual != digest:
        raise CorpusFormatError(
            f"shared document {path} content does not match its digest: "
            f"{actual} != {digest}")
    return value


def resolve_corpus(root: Path) -> Path:
    """The version directory a corpus root points at.

    A bare ``root/INDEX.json`` is itself the corpus (the layout tests build).
    Otherwise ``root/CURRENT`` names the published version directory — the
    pointer ``capture_tier0_corpus.py`` flips atomically after a versioned write.
    """
    if (root / "INDEX.json").is_file():
        return root
    current = root / "CURRENT"
    if current.is_file():
        try:
            version = str(json.loads(current.read_text()).get("version") or "")
        except (ValueError, OSError):
            return root
        candidate = root / version
        if (candidate / "INDEX.json").is_file():
            return candidate
    return root


def load(root: Path) -> Corpus:
    """Read the corpus: the index, and every pair file under ``pairs/``.

    A pair's payload may embed a shared frozen document by reference
    (``{SHARED_REF_KEY: "sha256:..."}``) instead of in full -- see
    ``_resolve_shared``. Every reference in the pairs this ONE call loads
    resolves to a SINGLE Python object per digest, read once from
    ``root/shared/<hex>.json`` and cached across every pair, so repeated
    occurrences (a chooser's 11 ranked members, or two pairs served by the
    same fold) share that object by identity again -- restoring the
    capture's own in-memory sharing instead of duplicating it per pair. A
    missing or digest-mismatched shared file, or an unrecognized reference
    shape, is a hard refusal (``CorpusFormatError``), never a silent skip.
    An OLD-format corpus with no ``shared/`` directory and no reference
    nodes loads unchanged: nothing in it is reference-shaped, so nothing
    here does anything but pass values through.
    """
    index = json.loads((root / "INDEX.json").read_text())
    shared_dir = root / "shared"
    shared_cache: dict[str, Any] = {}
    pairs = {}
    for path in sorted((root / "pairs").glob("*.json")):
        pair = json.loads(path.read_text())
        pair = _resolve_shared(pair, shared_dir, shared_cache, set())
        pairs[pair["fixture_id"]] = pair
    return Corpus(root=root, index=index, pairs=pairs)


# --------------------------------------------------------------------------
# case 0 — manifest membership: the files ARE the corpus the index declares
# --------------------------------------------------------------------------


def case_manifest(corpus: Corpus) -> ComparisonReceipt:
    """Exact membership and per-file agreement with the manifest.

    Files the index does not declare, declared pairs with no file, and files
    whose stored hashes/kind/covers disagree with the manifest row are each a
    finding. The corpus hash is recomputed from the SURVIVORS, so a deletion
    changes it.
    """
    declared = corpus.index.get("pairs", {})
    left: dict[str, Any] = {
        "pair_ids": sorted(declared),
        "corpus_hash": corpus.index.get("corpus_hash"),
        "pairs": {},
    }
    right: dict[str, Any] = {
        "pair_ids": sorted(corpus.pairs),
        "corpus_hash": content_hash(
            {fid: corpus.pairs[fid].get("payload_hash")
             for fid in sorted(corpus.pairs)}),
        "pairs": {},
    }
    for fid in sorted(set(declared) | set(corpus.pairs)):
        row = declared.get(fid) or {}
        pair = corpus.pairs.get(fid) or {}
        payload = pair.get("payload") or {}
        left["pairs"][fid] = {
            "payload_hash": row.get("payload_hash"),
            "request_hash": row.get("request_hash"),
            "record_kind": row.get("record_kind"),
            "covers": row.get("covers"),
        }
        right["pairs"][fid] = {
            "payload_hash": pair.get("payload_hash"),
            "request_hash": pair.get("request_hash"),
            "record_kind": payload.get("record_kind"),
            "covers": pair.get("covers"),
        }
    return compare_records(
        left, right, comparison_kind="tier0_manifest",
        left_ref="INDEX.json", right_ref="files-on-disk",
    )


# --------------------------------------------------------------------------
# cases 1-3 — addressing, digest, serialized round trip
# --------------------------------------------------------------------------


def case_addressing(corpus: Corpus) -> list[ComparisonReceipt]:
    """Resolve every record from the hash of its own frozen request."""
    by_request: dict[str, list[str]] = {}
    for fixture_id in corpus.ordered_ids:
        by_request.setdefault(content_hash(corpus.request_of(fixture_id)),
                              []).append(fixture_id)

    out: list[ComparisonReceipt] = []
    for fixture_id in corpus.ordered_ids:
        pair = corpus.pairs[fixture_id]
        digest = content_hash(pair["payload"]["request"])
        resolved = by_request.get(pair["request_hash"], [])
        left = {"request_hash": pair["request_hash"], "resolves_to": [fixture_id]}
        right = {"request_hash": digest, "resolves_to": resolved}
        out.append(compare_records(
            left, right, comparison_kind="tier0_request_addressing",
            left_ref=f"{fixture_id}#declared", right_ref=f"{fixture_id}#recomputed",
        ))
    return out


def case_digest(corpus: Corpus) -> list[ComparisonReceipt]:
    out: list[ComparisonReceipt] = []
    for fixture_id in corpus.ordered_ids:
        pair = corpus.pairs[fixture_id]
        out.append(compare_records(
            {"payload_hash": pair["payload_hash"]},
            {"payload_hash": content_hash(pair["payload"])},
            comparison_kind="tier0_payload_digest",
            left_ref=f"{fixture_id}#stored", right_ref=f"{fixture_id}#recomputed",
        ))
    return out


def case_round_trip(corpus: Corpus, scratch: Path) -> list[ComparisonReceipt]:
    """Write each record to an actual file and read it back before comparing."""
    scratch.mkdir(parents=True, exist_ok=True)
    out: list[ComparisonReceipt] = []
    for fixture_id in corpus.ordered_ids:
        record = corpus.record_of(fixture_id)
        path = scratch / f"{fixture_id}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        out.append(compare_records(
            record, json.loads(path.read_text()),
            comparison_kind="tier0_serialized_round_trip",
            left_ref=f"{fixture_id}#memory", right_ref=f"{fixture_id}#disk",
        ))
        path.unlink()
    return out


def corpus_case_list(corpus: Corpus, scratch: Path) -> list[ComparisonReceipt]:
    return (case_addressing(corpus) + case_digest(corpus)
            + case_round_trip(corpus, scratch))


def corpus_verdict(corpus: Corpus, scratch: Path) -> ComparisonReceipt:
    """One receipt over every pair and every per-pair case.

    The expected population comes from the DECLARED manifest, not from what
    loaded: a corpus that silently lost fifteen of sixteen files compares three
    receipts against an expectation of forty-eight and is ``incomparable``.
    """
    receipts = corpus_case_list(corpus, scratch)
    declared = len(corpus.index.get("pairs") or {}) or len(corpus.pairs)
    return merge_receipts(receipts, comparison_kind="tier0_corpus_replay",
                          tier=0, expected=3 * declared)


# --------------------------------------------------------------------------
# loader determinism — batch/single and a fresh process
# --------------------------------------------------------------------------


def _one(corpus: Corpus, fixture_id: str) -> Corpus:
    return Corpus(root=corpus.root, index=corpus.index,
                  pairs={fixture_id: corpus.pairs[fixture_id]})


def case_batch_and_single(corpus: Corpus, scratch: Path) -> ComparisonReceipt:
    """The corpus verdict folded pair by pair equals the batch verdict.

    Determinism of THIS check, not of the scorer: the engine-level batch/single
    parity is rearchitecture phase 1's O30.
    """
    singles = [
        merge_receipts(corpus_case_list(_one(corpus, fid), scratch),
                       comparison_kind="tier0_single", tier=0, expected=3)
        for fid in corpus.ordered_ids
    ]
    batch = corpus_verdict(corpus, scratch)
    folded = merge_receipts(singles, comparison_kind="tier0_corpus_replay",
                            tier=0, expected=len(corpus.pairs))
    return compare_records(
        {"verdict": batch.verdict,
         "findings": sorted(f.finding_id for f in batch.findings)},
        {"verdict": folded.verdict,
         "findings": sorted(f.finding_id for f in folded.findings)},
        comparison_kind="tier0_batch_vs_single",
        left_ref="batch", right_ref="single",
    )


def case_fresh_process(corpus_root: Path, this_verdict: str) -> ComparisonReceipt:
    """Run the corpus again in a subprocess that shares no memory with this one."""
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()),
         "--corpus", str(corpus_root), "--emit-verdict"],
        capture_output=True, text=True, cwd=str(ROOT), check=False,
    )
    other = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return compare_records(
        {"verdict": this_verdict},
        {"verdict": other or None},
        comparison_kind="tier0_fresh_process",
        left_ref="this-process", right_ref="subprocess",
    )


# --------------------------------------------------------------------------
# coverage — one definition, re-derived from the survivors
# --------------------------------------------------------------------------


def priced(record: dict) -> bool:
    """A row that resolved legs and a cost — not a refusal."""
    return bool(record.get("legs")) and record.get("entry_cost") is not None


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, dict) and NONFINITE in value)


def _flags(record: dict) -> list:
    return record.get("flags") or []


def _roles(record: dict, roles: list[str]) -> set[str]:
    """Which registered model roles this record shows running."""
    out: set[str] = set()
    versions = record.get("model_versions") or {}
    for role in roles:
        if role in versions or any(role in str(k) for k in versions):
            out.add(role)
    signals = (
        ("forecast_model", "size"), ("driver_prediction", "size"),
        ("runup_move_prediction", "runup_move"),
        ("implied_move_at_entry", "implied_t1"), ("exp_pnl_sim", "iv_crush"),
        ("gate_score", "gate"), ("gate_pass", "gate"),
        ("chooser_score", "chooser"),
    )
    out |= {role for key, role in signals if not _missing(record.get(key))}
    return out


def _boundaries(record: dict) -> set[str]:
    entry, exit_ = record.get("entry_date"), record.get("exit_date")
    if not entry or not exit_:
        return set()
    out = set()
    if entry[:4] != exit_[:4]:
        out.add("boundary:year")
    if entry[:7] != exit_[:7]:
        out.add("boundary:month")
    return out


def _leg_strikes(record: dict) -> list[float]:
    return sorted({float(leg["strike"]) for leg in record.get("legs") or []
                   if isinstance(leg, dict) and isinstance(leg.get("strike"), (int, float))})


def _geometry(record: dict, request: dict, relations: dict) -> set[str]:
    """Geometry axes. Every one but the coarse-ladder refusal needs a PRICED row.

    A refusal has no legs, so it demonstrates nothing about how a shape was
    chosen: the previous definition let a NO_CHAIN row whose requested strike
    was a computed 14.7615 cover "round listed strike".
    """
    out: set[str] = set()
    if "COARSE_LADDER" in _flags(record):
        out.add("geometry:coarse_ladder")
    if not priced(record):
        return out
    params = record.get("structure_params")
    if request.get("structure_params"):
        if relations.get("pinned_from"):
            out.add("geometry:pinned")
    elif isinstance(params, dict) and params:
        out.add("geometry:selector")
    if isinstance(params, dict) and isinstance(params.get("width_moneyness"), float):
        out.add("geometry:computed_width")
    strikes = _leg_strikes(record)
    requested = request.get("strike")
    if isinstance(requested, (int, float)) and float(requested) in strikes:
        # The requested strike IS a listed strike the legs resolved to — not a
        # computed moneyness the chain snapped away from.
        out.add("geometry:round_listed_strike")
    if len(strikes) >= 3 and len({round(b - a, 6) for a, b in zip(strikes, strikes[1:])}) == 1:
        out.add("geometry:exact_mirror")
    return out


def _dyn_sv(record: dict, request: dict, menu: list[str]) -> set[str]:
    """DYN-SV axes, only for a choice made over a BOARD-SHAPED frame.

    Board-shaped means one row per menu structure per event, as the board's
    ``engine.score.score_calendar`` loop produces at its ATM pass. A frame carrying a structure twice — a pinned
    copy beside its selector row — can "tie" a structure with itself, which is
    what the first corpus's only tie fixture turned out to be.
    """
    rows = request.get("frame_rows") or []
    in_menu = [(r.get("record") or {}).get("strategy") for r in rows]
    in_menu = [s for s in in_menu if s in menu]
    if not rows or len(in_menu) != len(set(in_menu)):
        return set()
    out: set[str] = set()
    size = record.get("menu_size")
    size = size if isinstance(size, int) and not isinstance(size, bool) else 0
    out.add("dyn_sv:full_menu" if size >= len(menu) else "dyn_sv:partial_menu")
    margin = record.get("chosen_margin")
    if (isinstance(margin, (int, float)) and not isinstance(margin, bool)
            and float(margin) == 0.0 and size >= 2):
        out.add("dyn_sv:tie")
    if _missing(record.get("chooser_score")):
        out.add("dyn_sv:fallback")
    return out


def derive_covers(record: dict, request: dict, record_kind: str | None,
                  axis_inputs: dict, relations: dict | None = None) -> list[str]:
    """The coverage axes one frozen pair demonstrates. THE definition.

    ``tools/capture_tier0_corpus.py`` selects fixtures with this function and
    this check re-derives with it, so there is exactly one meaning of each
    axis. What protects the index is not a second implementation — two copies
    agreeing proves only that they were copied — but that the claims written
    into the index are recomputed from the surviving records on every run, and
    that each axis has a crafted positive and negative case in
    ``tests/test_tier0_corpus.py``.
    """
    relations = relations or {}
    strategy = record.get("strategy")
    out = {f"strategy:{strategy}"}
    if priced(record):
        out.add(f"priced:{strategy}")
    if record.get("session"):
        out.add(f"session:{record['session']}")
    mapping = axis_inputs.get("refusal_code_mapping") or {}
    out |= {f"refusal:{code}" for flag in _flags(record)
            for code, emitted in mapping.items() if emitted == flag}
    out |= {f"model_role:{r}"
            for r in _roles(record, axis_inputs.get("model_roles") or [])}
    out |= _boundaries(record)
    if record_kind == "score_result":
        out |= _geometry(record, request, relations)
    if (strategy in (axis_inputs.get("disabled") or [])
            and "UNVALIDATED_STRUCTURE" in _flags(record)):
        out.add(f"disabled:{strategy}:refused")
    if record_kind == "research_replay":
        out.add(f"disabled:{strategy}:research_replay")
    if record_kind == "dyn_sv_choice":
        out |= _dyn_sv(record, request, axis_inputs.get("menu") or [])
    return sorted(out)


def _axis_inputs(corpus: Corpus) -> dict:
    axis_inputs = dict(corpus.index.get("axis_inputs") or {})
    axis_inputs.setdefault("refusal_code_mapping",
                           corpus.index.get("refusal_code_mapping") or {})
    return axis_inputs


def _derived(corpus: Corpus) -> dict[str, list[str]]:
    """``{fixture_id: covers}``, re-derived from each surviving pair."""
    axis_inputs = _axis_inputs(corpus)
    out = {}
    for fid in corpus.ordered_ids:
        payload = corpus.pairs[fid].get("payload") or {}
        out[fid] = derive_covers(payload.get("record") or {},
                                 payload.get("request") or {},
                                 payload.get("record_kind"), axis_inputs,
                                 payload.get("relations"))
    return out


def derived_uncovered(corpus: Corpus) -> list[str]:
    """Required axes no surviving pair demonstrates — never the index's claim."""
    covered = {axis for covers in _derived(corpus).values() for axis in covers}
    return sorted(set(corpus.index.get("required_axes", [])) - covered)


def case_coverage(corpus: Corpus) -> ComparisonReceipt:
    """Every axis claim recomputed from the surviving files (§12.2).

    Each pair's stored ``covers`` against the re-derivation from its own frozen
    content, and the index's coverage table against the coverage the survivors
    actually provide. A deleted fixture fails through its axes even if every
    hash in the manifest still passes, and an inflated ``covers`` list fails
    even though nothing was deleted.
    """
    left: dict[str, Any] = {"pairs": {}, "axes": {}}
    right: dict[str, Any] = {"pairs": {}, "axes": {}}
    derived_coverage: dict[str, list[str]] = {}
    for fid, derived in _derived(corpus).items():
        left["pairs"][fid] = {"covers": corpus.pairs[fid].get("covers")}
        right["pairs"][fid] = {"covers": derived}
        for axis in derived:
            derived_coverage.setdefault(axis, []).append(fid)
    claimed = corpus.index.get("coverage", {})
    for axis in corpus.index.get("required_axes", []):
        left["axes"][axis] = sorted(claimed.get(axis) or [])
        right["axes"][axis] = sorted(derived_coverage.get(axis) or [])
    return compare_records(
        left, right, comparison_kind="tier0_coverage",
        left_ref="index-claims", right_ref="derived-from-survivors",
    )


# --------------------------------------------------------------------------
# pinned counterparts — `e845f3e` on real data
# --------------------------------------------------------------------------

#: What a pinned re-score must reproduce from the selector row it was pinned
#: from: the forecast that chose the shape, and the contract that was priced.
PINNED_FIELDS = FORECAST_BLOCK + ("legs",)


def case_pinned_counterparts(corpus: Corpus) -> ComparisonReceipt:
    """Every pinned fixture's source is present and agrees on forecast and legs.

    A pinned fixture alone cannot show `e845f3e` — there is nothing to compare
    its forecast against. So the capture records which selector-resolved pair
    each pinned request was pinned FROM, keeps that pair in the corpus, and this
    case compares the two.
    """
    by_request = {pair.get("request_hash"): fid for fid, pair in corpus.pairs.items()}
    left: dict[str, Any] = {}
    right: dict[str, Any] = {}
    for fid in corpus.ordered_ids:
        payload = corpus.pairs[fid].get("payload") or {}
        source_hash = (payload.get("relations") or {}).get("pinned_from")
        if not source_hash:
            continue
        source = by_request.get(source_hash)
        source_record = corpus.record_of(source) if source else {}
        record = payload.get("record") or {}
        left[fid] = {"source_request_hash": source_hash,
                     **{f: source_record.get(f) for f in PINNED_FIELDS}}
        right[fid] = {"source_request_hash": (corpus.pairs[source]["request_hash"]
                                              if source else None),
                      **{f: record.get(f) for f in PINNED_FIELDS}}
    if not left:
        # The coverage case owns the absence of a pinned axis; there is simply
        # nothing to pair here.
        left = right = {"pinned_fixtures": 0}
    return compare_records(
        left, right, comparison_kind="tier0_pinned_counterparts",
        left_ref="selector-source", right_ref="pinned-rescore",
    )


# --------------------------------------------------------------------------
# seeded negative controls — contracts §15.3, over the REAL corpus
# --------------------------------------------------------------------------

#: Tier 0 writes no replayed record, so it has no round-trip receipt.
TIER0_RECEIPT_KINDS = ("record", "integrity")


def _nudged(value: float) -> float:
    return value + max(abs(value) * 1e-3, 1e-9)


def round_params(record: dict) -> dict:
    """``structure_params`` rounded to six places — the `json_safe` defect."""
    out = copy.deepcopy(record)
    params = out.get("structure_params")
    if isinstance(params, dict):
        out["structure_params"] = {k: round(v, 6) if isinstance(v, float) else v
                                   for k, v in params.items()}
    return out


def _seeded_record(cause: str | None, record: dict) -> dict:
    """The frozen record as the seeded defect leaves it at its first stage."""
    out = copy.deepcopy(record)
    if cause == "forecast_suppressed":
        out.update({key: None for key in FORECAST_BLOCK})
    elif cause == "analog_bootstrap_reseeded":
        out["ci_low"], out["ci_high"] = _nudged(out["ci_low"]), _nudged(out["ci_high"])
    elif cause == "replay_input_rounded":
        out = round_params(out)
    return out


def finding_dicts(receipt: ComparisonReceipt) -> list[dict]:
    return [{"first_differing_stage": f.first_differing_stage,
             "field_path": f.field_path, "kind": f.kind}
            for f in receipt.findings]


def _seed_pair(fid: str, payload: dict, cause: str | None) -> tuple[ComparisonReceipt, ComparisonReceipt]:
    frozen = payload.get("record") or {}
    seeded = _seeded_record(cause, frozen)
    record_receipt = compare_records(
        frozen, seeded, comparison_kind="tier0_seeded_record",
        left_ref=f"{fid}#frozen", right_ref=f"{fid}#seeded")
    written = {**payload, "record": seeded}
    stored = content_hash(written)
    if cause == "rounded_after_digest":
        # `6b9d5cf`: the digest was taken, THEN the bytes were re-rounded.
        written = {**written, "record": round_params(seeded)}
    integrity = compare_records(
        {"payload_hash": stored}, {"payload_hash": content_hash(written)},
        comparison_kind="tier0_seeded_integrity",
        left_ref=f"{fid}#stored-digest", right_ref=f"{fid}#written-bytes")
    return record_receipt, integrity


def seeded_controls(corpus: Corpus) -> dict[str, Any]:
    """Plant every seedable cause into its own real pair, and judge one pass.

    Distinct target pairs make each cause's findings attributable, so the one
    pass is also an ablation: every control must produce exactly its specified
    findings, every untargeted pair must stay clean, and every pair's compared
    population must equal its own leaves — the `28cf8b1` property.
    """
    targets, missing = pick_seed_targets(corpus.pairs)
    cause_of = {fid: cause for cause, fid in targets.items()}
    receipts: list[ComparisonReceipt] = []
    found_by_cause: dict[str, dict[str, list[dict]]] = {}
    untargeted: list[dict] = []
    field_set_mismatches: list[str] = []
    for fid in corpus.ordered_ids:
        payload = corpus.pairs[fid].get("payload") or {}
        cause = cause_of.get(fid)
        record_receipt, integrity = _seed_pair(fid, payload, cause)
        receipts += [record_receipt, integrity]
        frozen = payload.get("record") or {}
        leaves = set(flatten(frozen)) | set(flatten(_seeded_record(cause, frozen)))
        if leaves and record_receipt.population.compared != len(leaves):
            field_set_mismatches.append(fid)
        found = {"record": finding_dicts(record_receipt),
                 "integrity": finding_dicts(integrity)}
        if cause:
            found_by_cause[cause] = found
        else:
            untargeted += [dict(f, fixture_id=fid) for f in found["record"] + found["integrity"]]
    one_pass = merge_receipts(receipts, comparison_kind="tier0_seeded_one_pass",
                              expected=len(receipts))
    controls = {}
    for cause, spec in SEEDED_CONTROLS.items():
        found = found_by_cause.get(cause, {})
        problems = (["no frozen pair can carry this control"] if cause in missing
                    else check_control(cause, found, kinds=TIER0_RECEIPT_KINDS))
        controls[cause] = {
            "commit": spec["commit"], "target": targets.get(cause),
            "observed": {kind: sorted({f"{f['first_differing_stage']}: {f['field_path']}"
                                       for f in rows}) for kind, rows in found.items()},
            "problems": problems,
        }
    return {"one_pass_verdict": one_pass.verdict,
            "one_pass_findings": len(one_pass.findings),
            "controls": controls, "untargeted_findings": untargeted[:10],
            "field_set_mismatches": field_set_mismatches}


def case_seeded_controls(corpus: Corpus) -> ComparisonReceipt:
    """``agree`` exactly when every seeded control behaved as specified."""
    summary = seeded_controls(corpus)
    expected: dict[str, Any] = {cause: [] for cause in SEEDED_CONTROLS}
    observed: dict[str, Any] = {cause: summary["controls"][cause]["problems"]
                                for cause in SEEDED_CONTROLS}
    expected.update(one_pass_verdict=DIFFER, untargeted_findings=0,
                    field_set_mismatches=[])
    observed.update(one_pass_verdict=summary["one_pass_verdict"],
                    untargeted_findings=len(summary["untargeted_findings"]),
                    field_set_mismatches=summary["field_set_mismatches"])
    return compare_records(
        expected, observed, comparison_kind="tier0_seeded_controls",
        left_ref="control-spec", right_ref="one-seeded-pass",
    )


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------


class _NetworkUsed(RuntimeError):
    pass


def _forbid_network() -> None:
    """Any socket call from here on is an error, not a slow test."""
    def refuse(*_args, **_kwargs):  # pragma: no cover - it must never fire
        raise _NetworkUsed(
            "a tier-0 check opened a socket; the corpus replays from frozen "
            "fixtures and must not fetch"
        )
    socket.socket = refuse            # type: ignore[assignment]
    socket.create_connection = refuse  # type: ignore[assignment]


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def run(corpus_root: Path) -> tuple[ComparisonReceipt, dict[str, ComparisonReceipt]]:
    with tempfile.TemporaryDirectory(prefix="tier0-") as tmp:
        return _run(corpus_root, Path(tmp))


def _run(corpus_root: Path,
         scratch: Path) -> tuple[ComparisonReceipt, dict[str, ComparisonReceipt]]:
    corpus = load(corpus_root)
    if not corpus.pairs:
        empty = merge_receipts([], comparison_kind="tier0_corpus_replay",
                               tier=0, expected=1)
        return empty, {}
    verdict = corpus_verdict(corpus, scratch)
    cases = {
        "manifest": case_manifest(corpus),
        "corpus_replay": verdict,
        "coverage": case_coverage(corpus),
        "pinned_counterparts": case_pinned_counterparts(corpus),
        "seeded_controls": case_seeded_controls(corpus),
        "batch_vs_single": case_batch_and_single(corpus, scratch),
        "fresh_process": case_fresh_process(corpus_root, verdict.verdict),
    }
    merged = merge_receipts(list(cases.values()),
                            comparison_kind="tier0_corpus", tier=0,
                            expected=len(cases))
    return merged, cases


def _json_report(root: Path, merged: ComparisonReceipt,
                 cases: dict[str, ComparisonReceipt]) -> dict[str, Any]:
    corpus = load(root)
    return {
        "verdict": merged.verdict,
        "cases": {name: r.verdict for name, r in cases.items()},
        "findings": [f.describe() for f in merged.findings],
        "population": {"expected": merged.population.expected,
                       "compared": merged.population.compared},
        "pairs": len(corpus.pairs),
        "declared_pairs": len(corpus.index.get("pairs") or {}),
        "corpus_hash": corpus.index.get("corpus_hash"),
        "uncovered_axes": derived_uncovered(corpus),
        "seeded_controls": seeded_controls(corpus) if corpus.pairs else {},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--emit-verdict", action="store_true",
                    help="print only the corpus verdict (the fresh-process case)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    root = resolve_corpus(Path(args.corpus))
    if not (root / "INDEX.json").exists():
        prob = problem(
            "CORPUS_MISSING",
            f"no tier-0 corpus at {root}; run tools/capture_tier0_corpus.py",
            category="dependency", stage="resolve_context",
        )
        print(f"TIER0 INCOMPARABLE — {prob['message']}", file=sys.stderr)
        return 1

    _forbid_network()

    if args.emit_verdict:
        with tempfile.TemporaryDirectory(prefix="tier0-child-") as tmp:
            print(corpus_verdict(load(root), Path(tmp)).verdict)
        return 0

    merged, cases = run(root)
    if args.json:
        print(json.dumps(_json_report(root, merged, cases), indent=2, sort_keys=True))
        return 0 if merged.verdict == AGREE else 1

    if not args.quiet:
        for name, receipt in cases.items():
            print(f"  {receipt.verdict:12s}  {name}")
        print(merged.summary())
    if merged.verdict == AGREE:
        print("TIER0 OK")
        return 0
    label = "INCOMPARABLE" if merged.verdict == INCOMPARABLE else "FAILED"
    print(f"\nTIER0 {label}", file=sys.stderr)
    for finding in merged.findings:
        print(f"  {finding.describe()}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
