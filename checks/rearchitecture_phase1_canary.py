#!/usr/bin/env python3
"""Validation-only private shadow canary for legacy/adaptor parity.

The command consumes outputs produced by two sequential bounded runs. It never
opens the production ledger, refreshes providers, or chooses a worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.diagnosis import compare_records  # noqa: E402


def _load(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing canary artifact: {path}")
    value = json.loads(path.read_text())
    if not value:
        raise RuntimeError(f"empty canary artifact: {path}")
    return value


def _rows(value: dict) -> dict[str, dict]:
    rows = value if isinstance(value, list) else value.get("rows", [])
    result = {}
    for row in rows:
        if isinstance(row, dict) and row.get("request_id") and isinstance(row.get("record"), dict):
            key, record = str(row["request_id"]), row["record"]
        elif isinstance(row, dict) and row.get("row_id"):
            key, record = str(row["row_id"]), row
        else:
            raise RuntimeError("canary row lacks both request_id/record and row_id identity")
        if key in result:
            raise RuntimeError("canary score population contains duplicate or invalid IDs")
        result[key] = record
    if not result:
        raise RuntimeError("canary score population is empty")
    return result


def run(reference: Path, adapter: Path, selfcheck: Path) -> dict:
    left, right = _rows(_load(reference)), _rows(_load(adapter))
    if set(left) != set(right):
        raise RuntimeError("canary score populations differ")
    receipts = [compare_records(left[key], right[key], tier=1,
                                left_ref="legacy", right_ref="adapter")
                for key in sorted(left)]
    findings = [item for receipt in receipts for item in receipt.findings]
    check = _load(selfcheck)
    if check.get("ok") is not True:
        raise RuntimeError("serialized bundle selfcheck did not pass")
    return {"schema_version": "phase1_canary.v1.0", "population": len(left),
            "comparisons": len(receipts), "finding_count": len(findings),
            "comparison_receipts": [receipt.payload() for receipt in receipts],
            "selfcheck": True,
            "status": "pass" if not findings else "fail"}


def prepare(root: Path, baseline: Path, corpus: Path, output: Path) -> dict:
    """Copy the declared baseline package and complete tier-0 corpus privately."""
    output.mkdir(parents=True, exist_ok=True)
    files = []
    def copy_one(source, relative):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            raise RuntimeError(f"indirect frozen input: {source}")
        shutil.copyfile(source, target)
        target.chmod(0o444)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        files.append({"artifact_id": "sha256:" + digest,
                      "path": str(relative), "bytes": target.stat().st_size})
    for source_root in (baseline, corpus):
        if not source_root.is_dir():
            raise RuntimeError(f"missing frozen source root: {source_root}")
        destination = output / source_root.name
        for source in sorted(source_root.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(source_root)
            copy_one(source, destination.relative_to(output) / relative)
    dependency_file = baseline / "artifacts" / "dependencies.json"
    if dependency_file.is_file():
        payload = json.loads(dependency_file.read_text()).get("payload", {})
        for item in payload.get("artifacts", []):
            source = root / item["path"]
            if not source.is_file():
                raise RuntimeError(f"declared dependency missing: {source}")
            copy_one(source, Path(item["path"]))
    for relative in ("engine/models/registry.json", "data/MANIFEST.md"):
        source = root / relative
        if source.is_file():
            copy_one(source, Path(relative))
    for directory in ("data/curated", "engine", "tools", "checks"):
        source_root = root / directory
        if source_root.is_dir():
            for source in source_root.rglob("*"):
                if source.is_file() and not source.is_symlink() and (
                        source.suffix in (".py", ".json", ".yaml", ".yml", ".toml")
                        or directory == "data/curated"):
                    copy_one(source, source.relative_to(root))
    manifest = {"schema_version": "phase1_canary_inputs.v1.0",
                "root": str(root.resolve()), "files": files,
                "file_count": len(files)}
    (output / "INPUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2,
                                                            sort_keys=True))
    from checks.tier0_corpus import load

    requests = []
    corpus_data = load(corpus)
    for fixture_id in corpus_data.ordered_ids:
        payload = corpus_data.pairs[fixture_id]["payload"]
        if payload.get("record_kind") == "score_result":
            requests.append((fixture_id, payload["request"]))
        for index, frame in enumerate(payload.get("request", {}).get("frame_rows", ())):
            requests.append((fixture_id + "#frame-" + str(index), frame["request"]))
    rows = [{"canary_id": canary_id, "request": request}
            for canary_id, request in requests]
    (output / "score_requests.json").write_text(json.dumps(rows,
                                                            indent=2, sort_keys=True))
    manifest["score_request_count"] = len(rows)
    (output / "INPUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2,
                                                            sort_keys=True))
    return manifest


def _run_fixed(root: Path, mode: str, output: Path) -> int:
    request_file = root / "score_requests.json"
    if not request_file.is_file():
        raise RuntimeError("prepared private root lacks score_requests.json")
    script = root / "checks" / ("rearchitecture_phase1_reference.py"
                                 if mode == "reference" else
                                 "rearchitecture_phase1_adapter.py")
    if not script.is_file():
        raise RuntimeError("prepared private root lacks reference runner")
    command = ["/usr/bin/python3", "-u", str(script),
               "--requests", str(request_file), "--output", str(output)]
    started = time.monotonic()
    environment = os.environ.copy()
    environment["INVESTING_PLAN_ROOT"] = str(root)
    process = subprocess.Popen(command, cwd=root, env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True)
    lines = []
    for line in process.stdout or ():
        print(f"[{mode}] {line.rstrip()}", flush=True)
        lines.append(line)
    result_code = process.wait()
    # The runner's own output file is the evidence; the run log is a SEPARATE
    # file, because clobbering the output here would compare run logs, not scores.
    log_path = output.with_name(output.stem + ".runlog.json")
    log_path.write_text(json.dumps({"mode": mode, "returncode": result_code,
                                    "elapsed_s": round(time.monotonic() - started, 1),
                                    "output": str(output),
                                    "stdout_tail": "".join(lines)[-2000:]}, indent=2))
    return result_code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "reference", "adapted", "compare"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--selfcheck", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if not args.baseline or not args.corpus or not args.output_dir:
            parser.error("prepare requires --baseline, --corpus and --output-dir")
        print(json.dumps(prepare(args.root, args.baseline, args.corpus,
                                 args.output_dir), sort_keys=True))
        return 0
    if args.command in ("reference", "adapted"):
        if not args.output_dir:
            parser.error("run command requires --output-dir")
        return _run_fixed(args.root, args.command,
                          args.output_dir / (args.command + ".json"))
    if not args.reference or not args.adapter or not args.selfcheck:
        parser.error("compare requires --reference, --adapter and --selfcheck")
    result = run(args.reference, args.adapter, args.selfcheck)
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
