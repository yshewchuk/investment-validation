#!/usr/bin/env python3
"""Tier-1 real replay: re-score the frozen corpus through the production engine.

    python3 tools/replay_tier1.py                 # verify deps, replay, write receipt
    python3 tools/replay_tier1.py --seed-defects  # the seeded controls, through real stages
    python3 tools/replay_tier1.py --json          # receipt payload on stdout too

``checks/tier0_corpus.py`` is ARTIFACT INTEGRITY: seconds-fast, no engine. It
cannot prove the engine still PRODUCES the frozen answers — a changed pricing
formula leaves it green. This is the other half of the compatibility claim. It
verifies the hash-frozen dependencies named by the current baseline package,
builds a real ``Scorer`` in this fresh process, re-scores every frozen request
through the entry point it was captured through, writes each fresh record to a
real file and reads it back, and compares with the staged comparator.

Per pair, three receipts:

* ``record`` — the frozen record against the freshly computed one;
* ``round_trip`` — the fresh record against what a write and read-back of it
  returned (contracts §15.3: tier 1 must write to and read from a real file);
* ``integrity`` — the digest stored beside the written record against the
  digest of the record actually read back.

Every record kind replays, and the population is the DECLARED corpus:

* ``score_result`` through ``Scorer.score``, refusals included;
* ``research_replay`` through ``engine.replay.replay_one``, with the structure
  REBUILT FROM CURRENT CODE, so a drifted definition is a finding;
* ``dyn_sv_choice`` by re-scoring every frozen frame row, in frozen order, and
  re-running ``engine.score.dynamic_short_vol`` over them. Order matters: a
  tie is broken by input-row order.

The receipt binds to the corpus hash, the content hash of the code
(:func:`checks.replay_identity.code_hash`), the frozen dependency hash, the
current baseline and the store snapshot. The gate re-computes all of them and
refuses a receipt that no longer matches, a partial ``--limit`` run, or one
with any pair unreplayed.

``--seed-defects`` is the negative control through REAL stages, not a record
edit: it patches the legacy engine in this process only — the forecast
recording skipped for a pinned request (`e845f3e`), the analog bootstrap run
over rows in arrival order (`b9aa1fd`) — and the serialization path —
structure params rounded before the digest (`b33036c`) and after it
(`6b9d5cf`) — each armed for one distinct fixture. It writes a separate
receipt whose verdict is ``agree`` exactly when every control produced its
specified findings, localized to its stage, with every other pair clean.

Heavy by nature (~3G, minutes): the nightly/manual tier, not the per-edit
tier. Run it under ``tools/bounded_run.py --max-rss-gb 5.5``.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.replay_identity import (  # noqa: E402
    SEEDED_CONTROLS,
    check_control,
    code_hash,
    current_baseline,
    dependency_identity,
    pick_seed_targets,
)
from checks.tier0_corpus import (  # noqa: E402
    finding_dicts,
    load,
    resolve_corpus,
    round_params,
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

CORPUS = ROOT / "fixtures" / "tier0"
RECEIPTS = CORPUS / "receipts"
RECEIPT_KINDS = ("record", "round_trip", "integrity")


def _load_capture_module():
    """The capture's own serialization and scoring path — identical, not a copy."""
    spec = importlib.util.spec_from_file_location(
        "capture_tier0_corpus", ROOT / "tools" / "capture_tier0_corpus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# seeded defects — real stages, one fixture each
# --------------------------------------------------------------------------


class Seeds:
    """Which fixture each seeded defect is armed for, and which is replaying."""

    def __init__(self, targets: dict[str, str]):
        self.targets = targets
        self.active: str | None = None

    def fires(self, cause: str) -> bool:
        return self.active is not None and self.targets.get(cause) == self.active


class _UnsortedNumpy:
    """``numpy`` with ``sort`` as the identity — the pre-`b9aa1fd` bootstrap."""

    def __init__(self, np):
        self._np = np

    def __getattr__(self, name):
        return getattr(self._np, name)

    def sort(self, a, *args, **kwargs):
        return self._np.asarray(a)


@contextlib.contextmanager
def seeded_engine(seeds: Seeds, score_mod, analogs_mod, np):
    """Patch the two engine stages the historical defects lived in.

    Process-local and reverted on exit; ``engine/`` is never edited. Each patch
    delegates to the real method unless its cause is armed for the fixture
    being replayed right now.
    """
    real_size = score_mod.Scorer._size_from_forecast
    real_summarize = analogs_mod.AnalogMatcher._summarize

    def size_from_forecast(self, request, result, structure, *, size=True):
        if not size and seeds.fires("forecast_suppressed"):
            # `e845f3e`: the call was skipped wholesale whenever params were
            # supplied, so a pinned replay recorded no forecast at all.
            return request, structure
        return real_size(self, request, result, structure, size=size)

    def summarize(self, matched, *args, **kwargs):
        if not seeds.fires("analog_bootstrap_reseeded"):
            return real_summarize(self, matched, *args, **kwargs)
        # `b9aa1fd`: the bootstrap sampled by INDEX over rows in whatever order
        # they arrived. Reverse the arrival order and skip the sort.
        saved = analogs_mod.np
        analogs_mod.np = _UnsortedNumpy(np)
        try:
            return real_summarize(self, matched.iloc[::-1], *args, **kwargs)
        finally:
            analogs_mod.np = saved

    score_mod.Scorer._size_from_forecast = size_from_forecast
    analogs_mod.AnalogMatcher._summarize = summarize
    try:
        yield
    finally:
        score_mod.Scorer._size_from_forecast = real_size
        analogs_mod.AnalogMatcher._summarize = real_summarize


# --------------------------------------------------------------------------
# replaying one pair
# --------------------------------------------------------------------------


class Context:
    """Everything a pair replay needs, built once per process."""

    def __init__(self, scorer, cap, score_mod, replay_mod, structures, pd,
                 scratch: Path, seeds: Seeds):
        self.scorer, self.cap, self.score_mod = scorer, cap, score_mod
        self.replay_mod, self.structures, self.pd = replay_mod, structures, pd
        self.scratch, self.seeds = scratch, seeds


def _score(ctx: Context, request_dict: dict) -> tuple[dict, dict]:
    raw, record, _ = ctx.cap._score(ctx.scorer, ctx.cap.request_from_dict(request_dict))
    return raw, record


def _research(ctx: Context, request: dict, frozen: dict) -> dict:
    strategy = frozen.get("strategy")
    structure = ctx.structures[strategy]()
    plan_row = dict(request.get("plan_row") or {})
    plan_row["event_date"] = ctx.pd.Timestamp(plan_row["event_date"])
    plan = ctx.replay_mod.plan_events(structure, ctx.pd.DataFrame([plan_row]),
                                      calendar=ctx.scorer.calendar)
    index = ctx.replay_mod.load_chain_index(plan.chain_keys, progress_every=0)
    rows, skip = ctx.replay_mod.replay_one(
        structure, plan.frame.to_dict("records")[0], index, include_legs=True)
    return {"rows": ctx.cap.jsonable(rows), "skip_reason": skip, "strategy": strategy}


def _dyn_sv(ctx: Context, fid: str, request: dict) -> tuple[dict, list[ComparisonReceipt]]:
    """Re-score the frozen frame rows in order, then re-run the chooser."""
    raws, receipts = [], []
    for i, row in enumerate(request.get("frame_rows") or []):
        raw, record = _score(ctx, row["request"])
        raws.append(raw)
        receipts.append(compare_records(
            row["record"], record, comparison_kind="tier1_dyn_sv_frame_row", tier=1,
            left_ref=f"{fid}#frame[{i}]#frozen", right_ref=f"{fid}#frame[{i}]#replayed"))
    frame = ctx.pd.DataFrame([raw | {"strike_offset": None} for raw in raws])
    chosen = ctx.score_mod.dynamic_short_vol(frame)
    fresh = ctx.cap.jsonable(chosen.iloc[0].to_dict()) if not chosen.empty else {}
    return fresh, receipts


def _serialize(ctx: Context, fid: str, fresh: dict) -> tuple[dict, ComparisonReceipt, ComparisonReceipt]:
    """Write the fresh record to a real file and read it back.

    The two serialization seeds live here, because that is where the
    historical defects lived: one rounded the replay input before the digest
    was taken, the other re-rounded the bytes after it.
    """
    record = round_params(fresh) if ctx.seeds.fires("replay_input_rounded") else fresh
    stored = content_hash(record)
    written = round_params(record) if ctx.seeds.fires("rounded_after_digest") else record
    path = ctx.scratch / f"{fid}.json"
    path.write_text(json.dumps({"payload_hash": stored, "record": written},
                               indent=2, sort_keys=True) + "\n")
    doc = json.loads(path.read_text())
    path.unlink()
    round_trip = compare_records(
        record, doc["record"], comparison_kind="tier1_round_trip", tier=1,
        left_ref=f"{fid}#memory", right_ref=f"{fid}#disk")
    integrity = compare_records(
        {"payload_hash": doc["payload_hash"]},
        {"payload_hash": content_hash(doc["record"])},
        comparison_kind="tier1_integrity", tier=1,
        left_ref=f"{fid}#stored-digest", right_ref=f"{fid}#read-back")
    return record, round_trip, integrity


def replay_pair(ctx: Context, fid: str, pair: dict) -> dict[str, Any] | None:
    """The receipts for one pair, or None for a record kind nobody can replay."""
    payload = pair["payload"]
    kind, frozen, request = payload.get("record_kind"), payload["record"], payload["request"]
    extra: list[ComparisonReceipt] = []
    ctx.seeds.active = fid
    try:
        if kind == "score_result":
            _, fresh = _score(ctx, request)
        elif kind == "research_replay":
            fresh = _research(ctx, request, frozen)
        elif kind == "dyn_sv_choice":
            fresh, extra = _dyn_sv(ctx, fid, request)
        else:
            return None
        record, round_trip, integrity = _serialize(ctx, fid, fresh)
    finally:
        ctx.seeds.active = None
    compared = compare_records(
        frozen, record, comparison_kind=f"tier1_{kind}", tier=1,
        left_ref=f"{fid}#frozen", right_ref=f"{fid}#replayed")
    leaves = set(flatten(frozen)) | set(flatten(record))
    return {"record": compared, "round_trip": round_trip, "integrity": integrity,
            "frame_rows": extra,
            "field_set_ok": not leaves or compared.population.compared == len(leaves)}


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def _refusal_receipts(reasons: dict[str, str]) -> list[ComparisonReceipt]:
    return [ComparisonReceipt(
        receipt_id=code.lower(), comparison_kind="tier1_real_replay", tier=1,
        left_ref="frozen", right_ref="engine", stage_plan_ref="scorer.v1",
        tolerance_policy_ref="score_record.exact.v1", verdict=INCOMPARABLE,
        problems=(problem(code, message, category="dependency",
                          stage="resolve_context"),),
    ) for code, message in reasons.items()]


def _unreplayable(fid: str, kind: Any) -> ComparisonReceipt:
    return ComparisonReceipt(
        receipt_id=f"unreplayable-{fid}", comparison_kind="tier1_real_replay", tier=1,
        left_ref=f"{fid}#frozen", right_ref=f"{fid}#not-replayed",
        stage_plan_ref="scorer.v1", tolerance_policy_ref="score_record.exact.v1",
        verdict=INCOMPARABLE,
        problems=(problem("UNREPLAYABLE_RECORD_KIND",
                          f"{fid}: record_kind={kind!r} has no replay path",
                          category="validation", stage="resolve_context"),),
    )


def _git_head() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                          capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def _preflight(corpus, deps_drift: list, deps_payload_snapshot: str | None,
               snapshot_now: str | None) -> dict[str, str]:
    refusals: dict[str, str] = {}
    if deps_drift:
        refusals["DEPENDENCY_DRIFT"] = (
            f"{len(deps_drift)} frozen dependencies differ on disk: "
            + ", ".join(str(d.get("path")) for d in deps_drift[:6]))
    frozen = corpus.index.get("snapshot") or deps_payload_snapshot
    if frozen and snapshot_now and frozen != snapshot_now:
        refusals["SNAPSHOT_DRIFT"] = (
            f"corpus was captured against store snapshot {frozen[:12]}, current "
            f"store is {snapshot_now[:12]}; re-capture or restore the snapshot")
    return refusals


def _controls(results: dict[str, dict], targets: dict[str, str],
              missing: list[str]) -> tuple[dict, list[dict]]:
    """Judge every seeded control, and collect findings on untargeted pairs."""
    controls = {}
    for cause, spec in SEEDED_CONTROLS.items():
        fid = targets.get(cause)
        if fid is None or fid not in results:
            problems = ["no frozen pair can carry this control" if cause in missing
                        else "target pair was not replayed"]
            found: dict[str, list[dict]] = {}
        else:
            found = {kind: finding_dicts(results[fid][kind]) for kind in RECEIPT_KINDS}
            problems = check_control(cause, found)
        controls[cause] = {
            "commit": spec["commit"], "target": fid,
            "observed": {kind: sorted({f"{f['first_differing_stage']}: {f['field_path']}"
                                       for f in rows}) for kind, rows in found.items()},
            "problems": problems,
        }
    target_ids = set(targets.values())
    untargeted = [
        dict(f, fixture_id=fid)
        for fid, res in results.items() if fid not in target_ids
        for receipt in [*(res[k] for k in RECEIPT_KINDS), *res["frame_rows"]]
        for f in finding_dicts(receipt)
    ]
    return controls, untargeted


def run(*, skip_deps: bool = False, limit: int | None = None,
        seed_defects: bool = False) -> dict:
    import numpy as np
    import pandas as pd
    from engine import analogs as analogs_mod
    from engine import replay as replay_mod
    from engine import score as score_mod
    from engine.structures import STRUCTURES

    started = time.time()
    corpus_root = resolve_corpus(CORPUS)
    if not (corpus_root / "INDEX.json").is_file():
        raise SystemExit("no tier-0 corpus; run tools/capture_tier0_corpus.py")
    corpus = load(corpus_root)
    declared = len(corpus.index.get("pairs") or {})
    baseline = current_baseline(ROOT)
    deps_hash, drift = dependency_identity(baseline)
    deps_payload = {}
    if baseline is not None and (baseline / "artifacts" / "dependencies.json").is_file():
        deps_payload = json.loads((baseline / "artifacts" / "dependencies.json").read_text())["payload"]
    snapshot_now = score_mod._snapshot_hash()
    bindings: dict[str, Any] = {
        "corpus_version": corpus_root.name if corpus_root != CORPUS else "inline",
        "corpus_hash": corpus.index.get("corpus_hash"),
        "index_snapshot": corpus.index.get("snapshot"),
        "store_snapshot_at_replay": snapshot_now,
        "baseline": baseline.name if baseline is not None else None,
        "code_hash": code_hash(ROOT),
        "dependencies_hash": deps_hash,
        "dependency_drift": [] if skip_deps else drift,
        "deps_unverified": skip_deps,
        "engine_commit": _git_head(),
        "declared": declared, "limit": limit, "seeded": seed_defects,
        "replayed": 0, "skipped": 0,
    }
    refusals = _preflight(corpus, [] if skip_deps else drift,
                          (deps_payload.get("tier2_snapshot") or {}).get("snapshot"),
                          snapshot_now)
    if refusals:
        merged = merge_receipts(_refusal_receipts(refusals),
                                comparison_kind="tier1_real_replay", tier=1,
                                expected=3 * declared)
        return _document(merged.payload(), bindings, started)

    print(f"[tier1] dependencies verified: {len(deps_payload.get('artifacts', []))} "
          f"artifacts; snapshot {snapshot_now[:12]}; building the scorer...", flush=True)
    scorer = score_mod.Scorer()
    print(f"[tier1] scorer ready in {time.time()-started:.0f}s", flush=True)

    targets, missing = pick_seed_targets(corpus.pairs) if seed_defects else ({}, [])
    seeds = Seeds(targets)
    ids = corpus.ordered_ids[:limit] if limit else corpus.ordered_ids
    results: dict[str, dict] = {}
    receipts: list[ComparisonReceipt] = []
    with tempfile.TemporaryDirectory(prefix="tier1-") as tmp, (
            seeded_engine(seeds, score_mod, analogs_mod, np) if seed_defects
            else contextlib.nullcontext()):
        ctx = Context(scorer, _load_capture_module(), score_mod, replay_mod,
                      STRUCTURES, pd, Path(tmp), seeds)
        for i, fid in enumerate(ids):
            result = replay_pair(ctx, fid, corpus.pairs[fid])
            if result is None:
                receipts.append(_unreplayable(fid, corpus.pairs[fid]["payload"].get("record_kind")))
                bindings["skipped"] += 1
                continue
            results[fid] = result
            receipts += [result[k] for k in RECEIPT_KINDS] + result["frame_rows"]
            bindings["replayed"] += 1
            state = "AGREE" if all(result[k].verdict == AGREE for k in RECEIPT_KINDS) else "DIFFER"
            print(f"[tier1] {i+1}/{len(ids)} {fid} {state}", flush=True)

    merged = merge_receipts(receipts, comparison_kind="tier1_real_replay", tier=1,
                            expected=3 * declared)
    payload = merged.payload()
    bindings["field_set_mismatches"] = sorted(f for f, r in results.items() if not r["field_set_ok"])
    if seed_defects:
        controls, untargeted = _controls(results, targets, missing)
        failed = (any(c["problems"] for c in controls.values()) or untargeted
                  or bindings["field_set_mismatches"])
        payload.update(
            comparison_kind="tier1_seeded_controls",
            verdict=(INCOMPARABLE if merged.verdict == INCOMPARABLE
                     else DIFFER if failed else AGREE),
            controls=controls, untargeted_findings=untargeted[:20],
            seeded_pass_verdict=merged.verdict)
    elif bindings["field_set_mismatches"] and payload["verdict"] == AGREE:
        payload["verdict"] = DIFFER
    return _document(payload, bindings, started)


def _document(payload: dict, bindings: dict, started: float) -> dict:
    payload["bindings"] = bindings
    return {
        "schema_version": "tier1_replay_receipt.v1.1",
        "payload": payload,
        "envelope": {
            "ran_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "duration_seconds": round(time.time() - started, 1),
        },
    }


def receipt_path(bindings: dict) -> Path:
    suffix = ".seeded" if bindings.get("seeded") else ""
    return RECEIPTS / f"{bindings['corpus_version']}{suffix}.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--skip-deps-verify", action="store_true",
                    help="diagnostics only; the receipt records that it ran "
                         "unverified and the gate will not accept it")
    ap.add_argument("--limit", type=int, default=None,
                    help="diagnostics only; a partial receipt never satisfies the gate")
    ap.add_argument("--seed-defects", action="store_true",
                    help="run the seeded negative controls through real stages")
    args = ap.parse_args(argv)

    doc = run(skip_deps=args.skip_deps_verify, limit=args.limit,
              seed_defects=args.seed_defects)
    payload, bindings = doc["payload"], doc["payload"]["bindings"]
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    out = receipt_path(bindings)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(f"[tier1] verdict {payload['verdict'].upper()} — replayed "
          f"{bindings['replayed']} of {bindings['declared']}, skipped "
          f"{bindings['skipped']}, {doc['envelope']['duration_seconds']}s")
    print(f"[tier1] receipt -> {out}")
    for cause, control in (payload.get("controls") or {}).items():
        state = "ok" if not control["problems"] else "; ".join(control["problems"])
        print(f"  control {cause} [{control['target']}]: {state}")
    if args.json:
        print(json.dumps(doc, indent=2, sort_keys=True))
    for finding in payload.get("findings", [])[:20]:
        print(f"  - {finding['first_differing_stage']}: {finding['field_path']} "
              f"({finding['kind']})")
    return 0 if payload["verdict"] == AGREE else 1


if __name__ == "__main__":
    raise SystemExit(main())
