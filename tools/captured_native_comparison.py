"""Export one verified saved STR-THRU replay; never publish a current board."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from checks import phase4_real, tier0_corpus
from engine.v2.foundation import content_hash
from engine.v2.ops.native_parity_report import PARITY_DIMENSIONS, compare_native_vs_legacy
from tools.phase4_targeted_replay import (
    _ProgressReporter,
    _atomic_write,
    _reject_output_inside_corpus,
    _replay_regular_member,
    _verify_manifest_fields,
)


class ComparisonRefused(ValueError):
    """A selected capture cannot produce a verified comparison artifact."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonRefused(message)


def _selected_pair(root: Path, fixture_id: str) -> tuple[dict, dict]:
    """Hydrate only the declared selection and verify its content addressing."""
    _require(bool(fixture_id) and Path(fixture_id).name == fixture_id
             and fixture_id not in {".", ".."}, "invalid fixture selection")
    index = json.loads((root / "INDEX.json").read_text())
    declared = index["pairs"]
    _require(fixture_id in declared, "fixture is not manifest-declared")
    manifest_hash = content_hash({key: row["payload_hash"]
                                  for key, row in sorted(declared.items())})
    _require(manifest_hash == index["corpus_hash"], "corpus declaration hash mismatch")
    pair = json.loads((root / "pairs" / f"{fixture_id}.json").read_text())
    pair = tier0_corpus._resolve_shared(pair, root / "shared", {}, set(), {})
    _require(not _verify_manifest_fields(declared, fixture_id, pair),
             "selected manifest metadata mismatch")
    payload = pair["payload"]
    _require(content_hash(payload) == pair["payload_hash"], "selected payload hash mismatch")
    _require(content_hash(payload["request"]) == pair["request_hash"],
             "selected request hash mismatch")
    _require(payload["record_kind"] == "score_result"
             and payload["record"]["strategy"] == "STR-THRU",
             "selection must be a STR-THRU score_result")
    return index, pair


def _comparison(index: dict, pair: dict, root: Path, fixture_id: str) -> dict:
    """Replay verified inputs and project both answers without reconciliation."""
    verified = phase4_real._verified_trace_bundle(pair, root)
    _require(verified["frozen_replay"] is not None, "frozen inference is required")
    payload = pair["payload"]
    legacy = payload["record"]
    member = _replay_regular_member(legacy, verified)
    native = member["native"]
    identity_fields = ("ticker", "strategy", "event_date", "session", "as_of",
                       "entry_date", "exit_date")
    identity = {name: legacy.get(name) for name in identity_fields}
    _require(all(isinstance(value, str) and value for value in identity.values()),
             "paired decision identity is incomplete")
    _require(identity == {name: native.resolved_request.get(name) for name in identity_fields},
             "paired decision identity mismatch")
    expected, actual = phase4_real._numeric_views(legacy, native)
    flatten = lambda groups: {key: value for group in groups.values() for key, value in group.items()}
    report = compare_native_vs_legacy(
        {fixture_id: flatten(expected)}, {fixture_id: flatten(actual)}, PARITY_DIMENSIONS)
    trace = payload["input_trace"]
    release_id = trace["metadata"]["frozen_inference"]["release_resource_id"]
    _require(isinstance(release_id, str) and bool(release_id), "missing frozen release identity")
    report["captured_comparison"] = {
        "schema_version": "captured_native_comparison.v1.0",
        "scope": "selected_saved_replay",
        "full_population_verified": False,
        "cutover_qualified": False,
        "current_board": False,
        "identity": identity,
        "clocks": {
            "corpus_as_of": index.get("as_of"),
            "requested_decision_at": trace["request"]["requested_decision_at"],
            "decision_as_of": identity["as_of"],
            "quote_as_of": verified["inputs"].context.get("chain_as_of"),
            "event_date": identity["event_date"],
            "session": identity["session"],
        },
        "provenance": {
            "corpus_hash": index["corpus_hash"],
            "fixture_id": fixture_id,
            "payload_hash": pair["payload_hash"],
            "legacy_request_hash": pair["request_hash"],
            "native_request_hash": trace["request_hash"],
            "trace_hash": verified["trace_hash"],
            "same_input_receipt": verified["same_input_receipt"],
            "frozen_release_id": release_id,
            "native_snapshot_ref": native.snapshot_ref,
        },
        "legacy": expected,
        "native": actual,
        "checks": member["checks"],
        "numeric_comparisons": member["numeric"],
        "runtime_stage_count": member["runtime_stages"],
    }
    return report


def build_captured_comparison(corpus_root: str | Path, fixture_id: str) -> dict:
    """Verify and replay one declared capture; refuse with path-free errors.

    Only the selected payload/resources are read. The declared corpus digest
    binds the selection but does not assert verification of all other files.
    """
    try:
        root = tier0_corpus.resolve_corpus(Path(corpus_root))
        with _ProgressReporter(1, sys.stderr) as progress:
            progress.begin_load()
            index, pair = _selected_pair(root, fixture_id)
            progress.end_load()
            started = time.perf_counter()
            progress.begin_row(fixture_id)
            report = _comparison(index, pair, root, fixture_id)
            progress.end_row(fixture_id, time.perf_counter() - started)
        # Strict JSON also refuses non-finite or non-serializable captured values.
        json.dumps(report, allow_nan=False)
        return report
    except ComparisonRefused:
        raise
    except Exception:
        raise ComparisonRefused("capture verification or native replay refused") from None


def main(argv: list[str] | None = None) -> int:
    """Publish atomically outside the retained corpus after strict replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        root = tier0_corpus.resolve_corpus(args.corpus)
        _reject_output_inside_corpus(args.corpus, root, args.output)
        report = build_captured_comparison(root, args.fixture_id)
        _atomic_write(args.output, json.dumps(report, sort_keys=True, indent=2,
                                            allow_nan=False) + "\n")
    except Exception:
        print("captured comparison refused; no comparison published", file=sys.stderr)
        return 1
    print("captured comparison published", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
