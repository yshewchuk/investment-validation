#!/usr/bin/env python3
"""Replay the tier-0 corpus — seconds, no panel, no network, no fitting.

    python3 checks/tier0_corpus.py
    python3 checks/tier0_corpus.py --corpus fixtures/tier0 --json

The corpus is the migration's **oracle**: frozen `(request, record)` pairs
captured once by ``tools/capture_tier0_corpus.py`` through the real public
entry points. Every later phase states its exit gate as a comparison against
it, so this check's job is to prove the oracle is well-formed, addressable and
replay-identical — fast enough to run on every edit.

What it deliberately does **not** do is re-score. §11 lists "a corpus that fits
or fetches" as a failure mode: it stops being seconds, stops running on every
edit, and becomes a tier-2 check nobody waits for. Re-computing these records
through a *different* implementation is what the corpus is **for**, and it is
the parity comparison of phases 2-8, not this.

`component_contracts.md` §9.5: "Running twice in the same process is
insufficient: include fresh process, reordered inputs, batch/single and
serialized round-trip cases." That sentence is the acceptance criterion for the
whole corpus, and it is these five cases:

1. **addressing** — every record is reachable from the content hash of its own
   full-precision request. A rounded request hashes differently and stops
   resolving, which is what makes `b33036c` structurally impossible to hide.
2. **digest** — the payload on disk re-hashes to the ``payload_hash`` beside
   it. `6b9d5cf` was a file disagreeing with its own digest.
3. **round trip** — serialized to disk and read back, every record compares
   equal through the staged comparator.
4. **batch and single** — the merged receipt over all pairs equals the
   per-pair receipts folded together.
5. **reordered, and a fresh process** — the corpus verdict is identical when
   the pairs are loaded backwards, and identical again in a subprocess that
   shares no memory with this one.

Stdlib plus ``engine/v2/diagnosis`` only. It must run in a bare checkout with
no pandas, no store and no models, or the "every edit" commitment is not real.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import tempfile
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.diagnosis import (  # noqa: E402
    AGREE,
    INCOMPARABLE,
    ComparisonReceipt,
    compare_records,
    content_hash,
    merge_receipts,
    problem,
)

__all__ = ["Corpus", "load", "run", "main", "TIME_BUDGET_SECONDS"]

DEFAULT_CORPUS = ROOT / "fixtures" / "tier0"

#: §7.3: total runtime under ten seconds, network disabled.
TIME_BUDGET_SECONDS = 10.0


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


def load(root: Path, *, reverse: bool = False) -> Corpus:
    """Read the corpus. ``reverse`` is the reordered-inputs case, not a option."""
    index = json.loads((root / "INDEX.json").read_text())
    names = sorted((root / "pairs").glob("*.json"), key=lambda p: p.name,
                   reverse=reverse)
    pairs = {}
    for path in names:
        pair = json.loads(path.read_text())
        pairs[pair["fixture_id"]] = pair
    return Corpus(root=root, index=index, pairs=pairs)


# --------------------------------------------------------------------------
# case 1 — addressing
# --------------------------------------------------------------------------


def case_addressing(corpus: Corpus) -> list[ComparisonReceipt]:
    """Resolve every record from the hash of its own frozen request.

    This is what "reproduces from its frozen request" means for a frozen
    corpus: the key is the content hash of the full-precision request, so a
    client that copied a rounded value out of a table addresses nothing.
    """
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


# --------------------------------------------------------------------------
# case 2 — the digest beside the payload
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# case 3 — serialized round trip
# --------------------------------------------------------------------------


def case_round_trip(corpus: Corpus, scratch: Path) -> list[ComparisonReceipt]:
    """Write each record to an actual file and read it back before comparing.

    A round-trip loss through serialization is invisible to any check that
    keeps the object in memory, which is the whole reason contracts §9.5 names
    this case separately.
    """
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


# --------------------------------------------------------------------------
# cases 4 and 5 — batch/single, reordered, fresh process
# --------------------------------------------------------------------------


def corpus_case_list(corpus: Corpus, scratch: Path) -> list[ComparisonReceipt]:
    return (case_addressing(corpus) + case_digest(corpus)
            + case_round_trip(corpus, scratch))


def corpus_verdict(corpus: Corpus, scratch: Path) -> ComparisonReceipt:
    """One receipt over every pair and every case. The corpus's answer."""
    receipts = corpus_case_list(corpus, scratch)
    return merge_receipts(receipts, comparison_kind="tier0_corpus_replay",
                          tier=0, expected=3 * len(corpus.pairs))


def _one(corpus: Corpus, fixture_id: str) -> Corpus:
    return Corpus(root=corpus.root, index=corpus.index,
                  pairs={fixture_id: corpus.pairs[fixture_id]})


def case_batch_and_single(corpus: Corpus, scratch: Path) -> ComparisonReceipt:
    """The merged receipt must equal the per-pair receipts folded together.

    A batch that reuses common work is allowed to be faster and is not allowed
    to be different (contracts §9.2: batching reuses common feature work
    without changing results).
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


def case_reordered(corpus_root: Path, scratch: Path) -> ComparisonReceipt:
    forward = corpus_verdict(load(corpus_root), scratch)
    backward = corpus_verdict(load(corpus_root, reverse=True), scratch)
    return compare_records(
        {"verdict": forward.verdict, "findings": sorted(
            f.finding_id for f in forward.findings)},
        {"verdict": backward.verdict, "findings": sorted(
            f.finding_id for f in backward.findings)},
        comparison_kind="tier0_reordered_inputs",
        left_ref="in-order", right_ref="reversed",
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
# coverage
# --------------------------------------------------------------------------


def case_coverage(corpus: Corpus) -> ComparisonReceipt:
    """Every strategy and every refusal code has a frozen pair (§12.2)."""
    required = [a for a in corpus.index.get("required_axes", [])
                if a.startswith(("strategy:", "refusal:"))]
    covered = corpus.index.get("coverage", {})
    left = {axis: True for axis in required}
    right = {axis: bool(covered.get(axis)) for axis in required}
    return compare_records(
        left, right, comparison_kind="tier0_coverage",
        left_ref="required", right_ref="corpus",
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
        "corpus_replay": verdict,
        "coverage": case_coverage(corpus),
        "batch_vs_single": case_batch_and_single(corpus, scratch),
        "reordered_inputs": case_reordered(corpus_root, scratch),
        "fresh_process": case_fresh_process(corpus_root, verdict.verdict),
    }
    merged = merge_receipts(list(cases.values()),
                            comparison_kind="tier0_corpus", tier=0,
                            expected=len(cases))
    return merged, cases


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--emit-verdict", action="store_true",
                    help="print only the corpus verdict (the fresh-process case)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.corpus)
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
        print(json.dumps({
            "verdict": merged.verdict,
            "cases": {name: r.verdict for name, r in cases.items()},
            "findings": [f.describe() for f in merged.findings],
            "population": {
                "expected": merged.population.expected,
                "compared": merged.population.compared,
            },
        }, indent=2, sort_keys=True))
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
