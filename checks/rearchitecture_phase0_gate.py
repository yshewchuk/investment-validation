#!/usr/bin/env python3
"""The phase-0 exit gate, as one runnable check.

    python3 checks/rearchitecture_phase0_gate.py            # human-readable
    python3 checks/rearchitecture_phase0_gate.py --json     # for the nightly

`system_rearchitecture.md` §12 states the phase-0 gate as three claims, and the
guide's §12 adds more. This runs all eight and answers each with a fact:

| Claim | Answered by |
|---|---|
| Every strategy and critical refusal reproducible | `checks/tier0_corpus.py` |
| Every score fixture re-scored by the REAL engine against frozen dependencies | `tools/replay_tier1.py` receipt |
| Five seeded defects yield five stage-named findings in one pass | `tests/test_phase0_negative_controls.py` |
| The layer check runs green with an empty adapter ledger | `checks/import_layers.py` |
| v2 budgets green at zero exemptions | `checks/code_budgets.py` |
| Every package README present, consumers matching the graph | `checks/package_readmes.py` |
| The hook is installed and versioned | `checks/install_hooks.py --check` |
| The baseline package is frozen on disk, intact, and its lock matches the repo | `baseline/<latest>/MANIFEST.json` + root `requirements.txt` |

Named for the rearchitecture rather than for "phase 0": ``checks/phase0_*.py``
already belong to the DATA phase 0 (``guides/phase0_data_foundations.md``),
which is a different programme at the same number.

**This is the nightly's entry point.** §4.6 makes the nightly the control and
the hook the convenience, and §4.7 makes a failure here refuse publication
while ingestion, scoring, settlement and backup advance normally. Wiring that
watermark split into `engine/dashboard/nightly.py` is phase 1's Operations
deliverable — phase 0 is explicitly forbidden from editing legacy `engine/`
(guide §10) — so what lands now is the machine-readable result the nightly will
read, shaped for the `code_budgets` class §4.7 adds to `health.json`.

The negative-control row needs pytest, which may not be installed everywhere.
It degrades to ``skipped`` with a reason rather than to ``ok``: a check that
reports success when it did not run is the failure mode this whole phase is
written against.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import install_hooks  # noqa: E402
from checks import tier0_corpus as tier0_check  # noqa: E402

CORPUS = ROOT / "fixtures" / "tier0"
BASELINE = ROOT / "baseline"

#: ``(name, argv)`` for each subprocess check. Run as subprocesses rather than
#: imported so one check's import side effects cannot change another's answer.
_CHECKS = (
    ("import_layers", ["checks/import_layers.py", "--all", "--quiet"]),
    ("code_budgets", ["checks/code_budgets.py", "--all", "--quiet"]),
    ("package_readmes", ["checks/package_readmes.py", "--all", "--quiet"]),
)


def _run(argv: list[str]) -> dict:
    started = time.monotonic()
    proc = subprocess.run([sys.executable, *argv], cwd=str(ROOT),
                          capture_output=True, text=True, check=False)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "seconds": round(time.monotonic() - started, 2),
        "detail": (proc.stderr or proc.stdout).strip()[-2000:],
    }


def _tier0() -> dict:
    corpus_dir = tier0_check.resolve_corpus(CORPUS)
    if not (corpus_dir / "INDEX.json").exists():
        return {"ok": False, "skipped": True, "seconds": 0.0,
                "detail": f"no tier-0 corpus at {CORPUS}; run "
                          "tools/capture_tier0_corpus.py"}
    result = _run(["checks/tier0_corpus.py", "--corpus", str(CORPUS), "--quiet"])
    index = json.loads((corpus_dir / "INDEX.json").read_text())
    result["pairs"] = len(index.get("pairs", {}))
    result["uncovered_axes"] = index.get("uncovered_axes", [])
    result["corpus_hash"] = index.get("corpus_hash")
    result["corpus_version"] = (corpus_dir.name
                                if corpus_dir != CORPUS else None)
    if result["uncovered_axes"]:
        result["ok"] = False
        result["detail"] = (
            f"{len(result['uncovered_axes'])} required axes uncovered: "
            + ", ".join(result["uncovered_axes"][:8])
        )
    return result


def _negative_controls() -> dict:
    try:
        import pytest  # noqa: F401
    except ImportError:
        return {"ok": False, "skipped": True, "seconds": 0.0,
                "detail": "pytest is not installed; the negative controls did "
                          "NOT run. This is not a pass."}
    return _run(["-m", "pytest", "-q",
                 "tests/test_phase0_negative_controls.py"])


def _baseline_package() -> dict:
    """DoD #1, the cheap half: the frozen package EXISTS and is intact.

    Two copies of the environment lock exist by design — the root
    ``requirements.txt`` (the repo's only lock, per guide §6.1) and the one
    inside each dated export. They are generated from the same builder, so a
    byte difference means one of the two was edited by hand: that is drift, and
    drift in the lock is the one ambiguity the lock exists to remove.

    Full byte-reproducibility (``baseline_export.py --verify``, the §6
    acceptance) re-imports the engine and is deliberately NOT run here; this
    check stays seconds-fast by verifying what is on disk against its own
    manifest.
    """
    started = time.monotonic()
    versions = sorted(p for p in BASELINE.glob("*")
                      if (p / "MANIFEST.json").is_file())
    if not versions:
        return {"ok": False, "seconds": round(time.monotonic() - started, 2),
                "detail": f"no baseline package under {BASELINE}; run "
                          "tools/baseline_export.py"}
    latest = versions[-1]
    from engine.v2.diagnosis import content_hash  # stdlib-only, safe to import

    manifest = json.loads((latest / "MANIFEST.json").read_text())
    parts = manifest.get("parts", {})
    bad = []
    for name, digest in sorted(parts.items()):
        path = latest / name
        if not path.is_file():
            bad.append(f"missing: {name}")
        elif content_hash(path.read_text()) != digest:
            bad.append(f"hash mismatch: {name}")
    extra = sorted(str(p.relative_to(latest)) for p in latest.rglob("*")
                   if p.is_file() and p.name != "MANIFEST.json"
                   and str(p.relative_to(latest)) not in parts)
    if extra:
        bad.extend(f"undeclared: {name}" for name in extra)

    drift = None
    root_req = ROOT / "requirements.txt"
    frozen_req = latest / "requirements.txt"
    if root_req.is_file() and frozen_req.is_file():
        drift = root_req.read_text() != frozen_req.read_text()
        if drift:
            bad.append("root requirements.txt differs from the frozen lock in "
                       f"baseline/{latest.name}/requirements.txt")
    elif root_req.is_file() or frozen_req.is_file():
        drift = True
        bad.append("one of the two requirement locks is missing")

    return {
        "ok": not bad,
        "seconds": round(time.monotonic() - started, 2),
        "detail": "; ".join(bad[:8]) if bad else (
            f"baseline/{latest.name}: {len(parts)} parts, manifest hashes "
            "intact, lock matches the repo"),
        "version": latest.name,
        "versions": [p.name for p in versions],
        "parts": len(parts),
        "requirements_drift": drift,
    }


def _tier1_receipt() -> dict:
    """The REAL-replay receipt — the strategy-compatibility claim itself.

    The gate stays seconds-fast by READING the receipt
    ``tools/replay_tier1.py`` writes after re-scoring every fixture through
    the production entry points against hash-verified frozen dependencies.
    The receipt must bind to the CURRENT corpus (version + corpus_hash) and
    must not have run with dependency verification skipped. Missing or stale
    is a failure: tier-0 alone is artifact integrity, and a compatibility
    claim resting on it is the overclaim the 2026-09-12 review named.
    """
    corpus_dir = tier0_check.resolve_corpus(CORPUS)
    index_path = corpus_dir / "INDEX.json"
    if not index_path.is_file():
        return {"ok": False, "skipped": True, "seconds": 0.0,
                "detail": f"no tier-0 corpus at {CORPUS}"}
    index = json.loads(index_path.read_text())
    version = corpus_dir.name if corpus_dir != CORPUS else "inline"
    receipt_path = CORPUS / "receipts" / f"{version}.json"
    if not receipt_path.is_file():
        return {"ok": False, "seconds": 0.0,
                "detail": f"no tier-1 receipt for corpus version {version}; "
                          "run tools/replay_tier1.py (bounded, ~5 min)"}
    doc = json.loads(receipt_path.read_text())
    payload = doc.get("payload", {})
    bindings = payload.get("bindings", {})
    problems = []
    if bindings.get("corpus_hash") != index.get("corpus_hash"):
        problems.append("receipt binds a different corpus_hash (stale)")
    if payload.get("verdict") != "agree":
        problems.append(f"replay verdict is {payload.get('verdict')!r}")
    if bindings.get("deps_unverified"):
        problems.append("ran with --skip-deps-verify")
    if bindings.get("dependency_drift"):
        problems.append(f"{len(bindings['dependency_drift'])} dependency drift(s)")
    detail = "; ".join(problems) or (
        f"agree — {bindings.get('replayed')} pairs re-scored through the "
        f"production entry points against {bindings.get('dependencies_verified')} "
        f"hash-verified artifacts; {bindings.get('skipped')} skipped with reasons")
    return {"ok": not problems, "seconds": 0.0, "detail": detail,
            "pairs_replayed": bindings.get("replayed"),
            "pairs_skipped": bindings.get("skipped"),
            "receipt": str(receipt_path)}


def gate() -> dict:
    started = time.monotonic()
    results = {name: _run(argv) for name, argv in _CHECKS}
    results["tier0_corpus"] = _tier0()
    results["tier1_real_replay"] = _tier1_receipt()
    results["negative_controls"] = _negative_controls()
    results["baseline_package"] = _baseline_package()

    hook = install_hooks.state(ROOT)
    results["pre_commit_hook"] = {
        "ok": hook["installed"] and hook["current"] and hook["executable"],
        "seconds": 0.0,
        "detail": hook.get("reason", ""),
        **hook,
    }

    failed = sorted(name for name, r in results.items() if not r["ok"])
    return {
        "schema_version": "phase0_gate.v1.0",
        "ok": not failed,
        "failed": failed,
        # §4.7: a failure refuses PUBLICATION and nothing else. Ingestion,
        # scoring, prediction commit, settlement and backup advance normally,
        # and predictions still commit — they are correct regardless of the
        # complexity of the function that produced them.
        "refuses_publication": bool(failed),
        "watermarks_unaffected": ["ingestion", "scoring", "prediction_commit",
                                  "settlement", "backup"],
        "seconds": round(time.monotonic() - started, 2),
        "checks": results,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    result = gate()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1

    for name, row in result["checks"].items():
        mark = "ok " if row["ok"] else ("SKIP" if row.get("skipped") else "FAIL")
        print(f"  {mark:4s} {name:20s} {row['seconds']:6.2f}s")
        if not row["ok"] and row.get("detail"):
            for line in row["detail"].splitlines()[-6:]:
                print(f"         {line}")
    print(f"\nphase 0 gate: {'GREEN' if result['ok'] else 'RED'} "
          f"in {result['seconds']:.1f}s")
    if not result["ok"]:
        print(f"  publication refused; {', '.join(result['failed'])} failed",
              file=sys.stderr)
        print(f"  {', '.join(result['watermarks_unaffected'])} advance normally",
              file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
