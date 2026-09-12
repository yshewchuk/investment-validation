#!/usr/bin/env python3
"""Tier-1 real replay: re-score the frozen corpus through the production engine.

    python3 tools/replay_tier1.py            # verify deps, replay, write receipt
    python3 tools/replay_tier1.py --json     # receipt payload on stdout too
    python3 tools/replay_tier1.py --skip-deps-verify   # diagnostics only

``checks/tier0_corpus.py`` is an ARTIFACT-INTEGRITY check: seconds-fast, no
engine, proving the frozen answers are addressable, digest-consistent and
round-trip-stable. It cannot prove the engine still PRODUCES those answers —
a changed pricing formula would leave it green. This tool is the other half of
the compatibility claim: it verifies the hash-frozen dependencies from the
baseline package, builds a real ``Scorer``, re-scores every frozen request
through the same public entry point it was captured through, and compares the
fresh record against the frozen one with the staged comparator.

It refuses rather than false-passes:

* any dependency sha256 drift           -> ``incomparable`` (DEPENDENCY_DRIFT)
* registry-declared hash != disk hash   -> ``incomparable`` (REGISTRY_DRIFT)
* corpus snapshot != store snapshot     -> ``incomparable`` (SNAPSHOT_DRIFT)
* the receipt binds to ``corpus_hash``  -> a stale receipt fails the gate

Scope, stated in the receipt population rather than silently narrowed:
``score_result`` pairs replay through ``Scorer.score`` (refusals included, via
``unscorable_result`` exactly as capture did); ``research_replay`` pairs
replay through ``engine.replay.replay_one`` from the frozen plan row, with the
structure REBUILT FROM CURRENT CODE — a drifted structure definition is a
finding, which is the point. ``dyn_sv_choice`` pairs are skipped with a named
reason: the chooser resolves over its sibling menu rows, and the menu frame is
not frozen inside the pair; freezing it is phase-1 corpus work. Skips appear in
``population.skipped_with_reasons`` and the expected count, never vanish.

Heavy by nature (~3G, minutes): this is the nightly/manual tier, not the
per-edit tier. Run it under ``tools/bounded_run.py --max-rss-gb 5.5``.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.tier0_corpus import load, resolve_corpus  # noqa: E402
from engine.v2.diagnosis import (  # noqa: E402
    AGREE,
    INCOMPARABLE,
    ComparisonReceipt,
    merge_receipts,
    problem,
)

RECEIPTS = ROOT / "fixtures" / "tier0" / "receipts"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_capture_module():
    """The capture's own ``jsonable`` — identical serialization, not a copy."""
    spec = importlib.util.spec_from_file_location(
        "capture_tier0_corpus", ROOT / "tools" / "capture_tier0_corpus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def latest_baseline() -> Path | None:
    versions = sorted(p for p in (ROOT / "baseline").glob("*")
                      if (p / "MANIFEST.json").is_file())
    return versions[-1] if versions else None


def verify_dependencies(skip: bool = False) -> tuple[list[dict], dict]:
    """Hash-verify every frozen dependency. Returns (drift rows, deps part)."""
    baseline = latest_baseline()
    if baseline is None:
        return [{"path": "baseline", "issue": "no baseline package found"}], {}
    deps_path = baseline / "artifacts" / "dependencies.json"
    if not deps_path.is_file():
        return [{"path": str(deps_path), "issue": "no dependencies part in "
                 f"baseline {baseline.name}"}], {}
    deps = json.loads(deps_path.read_text())["payload"]
    if skip:
        return [], deps
    drift = []
    for row in deps.get("artifacts", []):
        path = ROOT / row["path"]
        if not path.is_file():
            drift.append({"path": row["path"], "issue": "missing"})
        elif path.stat().st_size != row["bytes"]:
            drift.append({"path": row["path"], "issue": "size differs",
                          "frozen_bytes": row["bytes"],
                          "current_bytes": path.stat().st_size})
        elif _sha256_file(path) != row["sha256"]:
            drift.append({"path": row["path"], "issue": "sha256 differs"})
    drift.extend(deps.get("registry_sha_drift") or [])
    return drift, deps


def request_from_dict(data: dict, score_mod, pd):
    """Inverse of the capture's ``request_to_dict``, field by field."""
    date_fields = {"as_of", "event_date", "expiry", "chain_as_of"}
    kwargs = {}
    for f in dataclasses.fields(score_mod.ScoreRequest):
        if f.name not in data:
            continue
        value = data[f.name]
        if f.name == "fill":
            value = score_mod.FillModel(alpha=float(data["fill"]["alpha"]))
        elif f.name in date_fields and isinstance(value, str):
            value = pd.Timestamp(value)
        kwargs[f.name] = value
    return score_mod.ScoreRequest(**kwargs)


def _refusal_receipts(reasons: dict[str, str]) -> list:
    out = []
    for code, message in reasons.items():
        out.append(ComparisonReceipt(
            receipt_id=code.lower(),
            comparison_kind="tier1_real_replay",
            tier=1,
            left_ref="frozen", right_ref="engine",
            stage_plan_ref="scorer.v1",
            tolerance_policy_ref="score_record.exact.v1",
            verdict=INCOMPARABLE,
            problems=(problem(code, message, category="dependency",
                              stage="resolve_context"),),
        ))
    return out


def run(*, skip_deps: bool = False, limit: int | None = None) -> dict:
    import pandas as pd
    from engine import replay as replay_mod
    from engine import score as score_mod
    from engine.structures import STRUCTURES

    cap = _load_capture_module()
    jsonable = cap.jsonable

    started = time.time()
    corpus_root = resolve_corpus(ROOT / "fixtures" / "tier0")
    if not (corpus_root / "INDEX.json").is_file():
        raise SystemExit("no tier-0 corpus; run tools/capture_tier0_corpus.py")
    corpus = load(corpus_root)
    index = corpus.index
    version = corpus_root.name if corpus_root != ROOT / "fixtures" / "tier0" else "inline"

    drift, deps = verify_dependencies(skip_deps)
    snapshot_now = score_mod._snapshot_hash()
    snapshot_frozen = (deps.get("tier2_snapshot") or {}).get("snapshot")
    refusals: dict[str, str] = {}
    if drift:
        refusals["DEPENDENCY_DRIFT"] = (
            f"{len(drift)} frozen dependencies differ on disk: "
            + ", ".join(d["path"] for d in drift[:6]))
    if snapshot_frozen and snapshot_now and snapshot_frozen != snapshot_now:
        refusals["SNAPSHOT_DRIFT"] = (
            f"corpus was captured against store snapshot {snapshot_frozen[:12]}, "
            f"current store is {snapshot_now[:12]}; the Tier-2 store was rebuilt "
            "since capture — re-capture the corpus or restore the snapshot")
    if index.get("snapshot") and snapshot_now and index["snapshot"] != snapshot_now:
        refusals.setdefault("SNAPSHOT_DRIFT",
                            f"INDEX snapshot {index['snapshot'][:12]} != current "
                            f"{snapshot_now[:12]}")
    if refusals:
        merged = merge_receipts(_refusal_receipts(refusals),
                                comparison_kind="tier1_real_replay", tier=1,
                                expected=0)
        return _receipt_payload(merged, index, version, drift, deps,
                                snapshot_now, started, n_replayed=0, n_skipped=0)

    print(f"[tier1] dependencies verified: {deps.get('count')} artifacts, "
          f"snapshot {snapshot_now[:12]}", flush=True)
    print("[tier1] building the scorer...", flush=True)
    scorer = score_mod.Scorer()
    print(f"[tier1] scorer ready in {time.time()-started:.0f}s", flush=True)

    from engine.v2.diagnosis import compare_records

    receipts: list[ComparisonReceipt] = []
    skipped: list[ComparisonReceipt] = []
    ids = corpus.ordered_ids[:limit] if limit else corpus.ordered_ids
    n_replayed = n_skipped = 0
    for i, fid in enumerate(ids):
        pair = corpus.pairs[fid]
        payload = pair["payload"]
        kind = payload.get("record_kind")
        frozen = payload["record"]
        request_dict = payload["request"]
        if kind == "score_result":
            request = request_from_dict(request_dict, score_mod, pd)
            as_of = request.as_of if request.as_of is not None else (
                pd.Timestamp(frozen["as_of"]) if frozen.get("as_of") else None)
            try:
                result = scorer.score(request)
            except score_mod.UNSCORABLE as exc:
                result = score_mod.unscorable_result(
                    request, as_of=as_of, snapshot=scorer.snapshot, exc=exc)
            fresh = jsonable(result.as_dict())
            receipts.append(compare_records(
                frozen, fresh, comparison_kind="tier1_score_replay", tier=1,
                left_ref=f"{fid}#frozen", right_ref=f"{fid}#replayed"))
            n_replayed += 1
        elif kind == "research_replay":
            strategy = frozen.get("strategy")
            structure = STRUCTURES[strategy]()
            plan_row = dict(request_dict.get("plan_row") or {})
            plan_row["event_date"] = pd.Timestamp(plan_row["event_date"])
            plan = replay_mod.plan_events(
                structure, pd.DataFrame([plan_row]), calendar=scorer.calendar)
            chain_index = replay_mod.load_chain_index(plan.chain_keys,
                                                      progress_every=0)
            rows, skip_reason = replay_mod.replay_one(
                structure, plan.frame.to_dict("records")[0], chain_index,
                include_legs=True)
            fresh = {"rows": jsonable(rows), "skip_reason": skip_reason,
                     "strategy": strategy}
            receipts.append(compare_records(
                frozen, fresh, comparison_kind="tier1_research_replay", tier=1,
                left_ref=f"{fid}#frozen", right_ref=f"{fid}#replayed"))
            n_replayed += 1
        else:
            skipped.append(ComparisonReceipt(
                receipt_id=f"skip-{fid}",
                comparison_kind="tier1_real_replay", tier=1,
                left_ref=f"{fid}#frozen", right_ref=f"{fid}#not-replayed",
                stage_plan_ref="scorer.v1",
                tolerance_policy_ref="score_record.exact.v1",
                verdict=INCOMPARABLE,
                problems=(problem(
                    "TIER1_OUT_OF_SCOPE",
                    f"{fid}: record_kind={kind!r} — the DYN-SV chooser "
                    "resolves over sibling menu rows that the pair does not "
                    "freeze; freezing the menu frame is phase-1 corpus work",
                    category="validation", stage="chooser"),),
            ))
            n_skipped += 1
        print(f"[tier1] {i+1}/{len(ids)} {fid} ({kind}) "
              f"{'AGREE' if receipts and receipts[-1].verdict == AGREE else 'CHECK'}",
              flush=True)

    merged = merge_receipts(receipts + skipped,
                            comparison_kind="tier1_real_replay", tier=1,
                            expected=n_replayed)
    return _receipt_payload(merged, index, version, drift, deps, snapshot_now,
                            started, n_replayed, n_skipped)


def _receipt_payload(merged: ComparisonReceipt, index: dict, version: str,
                     drift: list, deps: dict, snapshot: str, started: float,
                     n_replayed: int, n_skipped: int) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True,
            text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    payload = merged.payload()
    payload["bindings"] = {
        "corpus_version": version,
        "corpus_hash": index.get("corpus_hash"),
        "index_snapshot": index.get("snapshot"),
        "store_snapshot_at_replay": snapshot,
        "baseline": (latest_baseline() or Path("none")).name,
        "engine_commit": commit,
        "dependencies_verified": deps.get("count"),
        "dependency_drift": drift,
        "replayed": n_replayed,
        "skipped": n_skipped,
    }
    return {
        "schema_version": "tier1_replay_receipt.v1.0",
        "payload": payload,
        "envelope": {
            "ran_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "duration_seconds": round(time.time() - started, 1),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--skip-deps-verify", action="store_true",
                    help="diagnostics only; the receipt records that it ran "
                         "unverified and the gate will not accept it")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    doc = run(skip_deps=args.skip_deps_verify, limit=args.limit)
    verdict = doc["payload"]["verdict"]
    if args.skip_deps_verify:
        doc["payload"]["bindings"]["deps_unverified"] = True

    RECEIPTS.mkdir(parents=True, exist_ok=True)
    out = RECEIPTS / f"{doc['payload']['bindings']['corpus_version']}.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    bindings = doc["payload"]["bindings"]
    print(f"[tier1] verdict {verdict.upper()} — replayed "
          f"{bindings['replayed']}, skipped {bindings['skipped']}, "
          f"{doc['envelope']['duration_seconds']}s")
    print(f"[tier1] receipt -> {out}")
    if args.json:
        print(json.dumps(doc, indent=2, sort_keys=True))
    for finding in doc["payload"].get("findings", [])[:20]:
        print(f"  - {finding['first_differing_stage']}: {finding['field_path']} "
              f"({finding['kind']})")
    return 0 if verdict == AGREE else 1


if __name__ == "__main__":
    raise SystemExit(main())
