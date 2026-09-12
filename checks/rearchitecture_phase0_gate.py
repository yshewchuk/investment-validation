#!/usr/bin/env python3
"""The phase-0 exit gate, as one runnable check.

    python3 checks/rearchitecture_phase0_gate.py                   # human-readable
    python3 checks/rearchitecture_phase0_gate.py --json            # for the nightly
    python3 checks/rearchitecture_phase0_gate.py --json --previous last.json

`system_rearchitecture.md` §12 states the phase-0 gate as three claims, and the
guide's §12 adds more. Every row below is answered by a check that re-derives
its answer now, never by a stored claim alone:

| Row | Answered by |
|---|---|
| ``tier0_corpus`` | ``checks/tier0_corpus.py --json``: integrity, coverage re-derived, pinned counterparts, seeded controls over the real corpus |
| ``tier1_real_replay`` | the ``tools/replay_tier1.py`` receipt, re-validated against the CURRENT corpus, code, dependencies, baseline and snapshot |
| ``tier1_seeded_controls`` | the ``--seed-defects`` receipt: the causes planted into real engine stages |
| ``negative_controls`` | the phase-0 pytest suite |
| ``import_layers`` / ``code_budgets`` / ``package_readmes`` | the three structural checks |
| ``baseline_package`` | ``baseline/CURRENT``: manifest intact, lock matches the repo, byte-identical re-export receipt |
| ``pre_commit_hook`` | ``checks/install_hooks.py`` state |

**Red stays red until each finding is addressed** — by a fix, or by a recorded
decision and a new baseline version (§3.2). There is no acceptance list. What
``--previous`` adds is information only: which rows' answers CHANGED since the
last run, so a gate that is red for a new reason is distinguishable from
yesterday's red. It never turns a row green.

Named for the rearchitecture rather than for "phase 0": ``checks/phase0_*.py``
already belong to the DATA phase 0, a different programme at the same number.

This is the nightly's entry point, but wiring it into ``engine/dashboard/
nightly.py`` is rearchitecture phase 1 — and phase 1 §10.4 forbids feeding its
aggregated exit code into a publication override: correctness failures block
the affected decisions and budget failures withhold publication, separately.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import install_hooks  # noqa: E402
from checks import replay_identity as identity  # noqa: E402
from checks.tier0_corpus import resolve_corpus  # noqa: E402
from engine.v2.diagnosis import content_hash  # noqa: E402

CORPUS = ROOT / "fixtures" / "tier0"

#: ``(name, argv)`` for each structural check. Subprocesses, so one check's
#: import side effects cannot change another's answer.
_CHECKS = (
    ("import_layers", ["checks/import_layers.py", "--all", "--quiet"]),
    ("code_budgets", ["checks/code_budgets.py", "--all", "--quiet"]),
    ("package_readmes", ["checks/package_readmes.py", "--all", "--quiet"]),
)

#: The phase-0 test suite: every instrument's own negative controls.
PHASE0_TESTS = (
    "tests/test_diagnosis_comparator.py",
    "tests/test_phase0_negative_controls.py",
    "tests/test_tier0_corpus.py",
    "tests/test_import_layers.py",
    "tests/test_code_budgets.py",
    "tests/test_phase0_gate.py",
    "tests/test_baseline_export.py",
    "tests/test_replay_tier1.py",
    "tests/test_phase0_report.py",
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
        "stdout": proc.stdout,
    }


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# tier 0
# --------------------------------------------------------------------------


def _tier0() -> dict:
    corpus_dir = resolve_corpus(CORPUS)
    if not (corpus_dir / "INDEX.json").exists():
        return {"ok": False, "skipped": True, "seconds": 0.0,
                "detail": f"no tier-0 corpus at {CORPUS}; run "
                          "tools/capture_tier0_corpus.py"}
    result = _run(["checks/tier0_corpus.py", "--corpus", str(CORPUS), "--json"])
    try:
        report = json.loads(result.pop("stdout"))
    except ValueError:
        result["ok"] = False
        return result
    result.update(
        corpus_version=corpus_dir.name if corpus_dir != CORPUS else None,
        pairs=report["pairs"], declared_pairs=report["declared_pairs"],
        corpus_hash=report["corpus_hash"], cases=report["cases"],
        uncovered_axes=report["uncovered_axes"],
        seeded_controls=report["seeded_controls"],
        findings=report["findings"][:20],
    )
    problems = [f"case {name} is {verdict}" for name, verdict in report["cases"].items()
                if verdict != "agree"]
    if report["uncovered_axes"]:
        problems.append(f"{len(report['uncovered_axes'])} required axes uncovered: "
                        + ", ".join(report["uncovered_axes"][:8]))
    result["ok"] = result["ok"] and not problems
    result["detail"] = "; ".join(problems) or (
        f"TIER0 OK — {report['pairs']} pairs, every case agrees, every required "
        "axis covered, seeded controls behaved")
    return result


# --------------------------------------------------------------------------
# tier 1 receipts
# --------------------------------------------------------------------------


def expectations(root: Path = ROOT) -> tuple[dict[str, Any], str | None]:
    """What a valid receipt must bind to, computed from the CURRENT state."""
    corpus_dir = resolve_corpus(CORPUS)
    index_path = corpus_dir / "INDEX.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {}
    baseline = identity.current_baseline(root)
    deps_hash, drift = identity.dependency_identity(baseline, root)
    version = (corpus_dir.name if corpus_dir != CORPUS else "inline") if index else None
    return {
        "corpus_hash": index.get("corpus_hash"),
        "declared_pairs": len(index.get("pairs") or {}),
        "code": identity.code_hash(root),
        "dependencies": deps_hash,
        "drift": drift,
        "baseline_version": baseline.name if baseline is not None else None,
        "snapshot": identity.store_snapshot(),
    }, version


def _receipt_row(expect: dict[str, Any], version: str | None, *, seeded: bool) -> dict:
    """Validate one tier-1 receipt against the current state."""
    started = time.monotonic()
    suffix, command = (".seeded", "tools/replay_tier1.py --seed-defects") if seeded \
        else ("", "tools/replay_tier1.py")
    if version is None:
        return {"ok": False, "skipped": True, "seconds": 0.0,
                "detail": f"no tier-0 corpus at {CORPUS}"}
    path = CORPUS / "receipts" / f"{version}{suffix}.json"
    if not path.is_file():
        return {"ok": False, "seconds": 0.0,
                "detail": f"no receipt for corpus version {version}; run {command} "
                          "(bounded, minutes)"}
    doc = json.loads(path.read_text())
    payload = doc.get("payload") or {}
    bind = payload.get("bindings") or {}
    problems = identity.evaluate_receipt(doc, **expect)
    if bool(bind.get("seeded")) != seeded:
        problems.append("receipt is a " + ("compatibility" if seeded else "seeded-control")
                        + " run, not the one this row needs")
    controls = payload.get("controls") or {}
    for cause, control in controls.items():
        problems.extend(f"control {cause}: {p}" for p in control.get("problems", []))
    replayed = (f"{bind.get('replayed')}/{expect['declared_pairs']} pairs re-scored in a "
                "fresh process, written and read back, bound to the current code, "
                "dependencies, baseline and snapshot")
    success = (f"agree — all {len(controls)} seeded controls produced exactly their "
               f"specified findings through real engine stages, every other pair "
               f"clean; {replayed}") if seeded else f"agree — {replayed}"
    return {
        "ok": not problems,
        "seconds": round(time.monotonic() - started, 2),
        "detail": "; ".join(problems[:8]) or success,
        "receipt": _display(path),
        "pairs_replayed": bind.get("replayed"),
        "controls": payload.get("controls"),
        "findings": [] if seeded else [
            f"{f['first_differing_stage']}: {f['field_path']}"
            for f in (payload.get("findings") or [])[:20]],
    }


# --------------------------------------------------------------------------
# the rest
# --------------------------------------------------------------------------


def _negative_controls() -> dict:
    try:
        import pytest  # noqa: F401
    except ImportError:
        return {"ok": False, "skipped": True, "seconds": 0.0,
                "detail": "pytest is not installed; the negative controls did "
                          "NOT run. This is not a pass."}
    present = [t for t in PHASE0_TESTS if (ROOT / t).is_file()]
    missing = sorted(set(PHASE0_TESTS) - set(present))
    result = _run(["-m", "pytest", "-q", "-p", "no:cacheprovider", *present])
    result.pop("stdout", None)
    if missing:
        result["ok"] = False
        result["detail"] = f"missing test files: {missing}; " + result["detail"]
    return result


def _manifest_problems(package: Path, manifest: dict) -> list[str]:
    parts = manifest.get("parts", {})
    bad = []
    for name, digest in sorted(parts.items()):
        path = package / name
        if not path.is_file():
            bad.append(f"missing: {name}")
        elif content_hash(path.read_text()) != digest:
            bad.append(f"hash mismatch: {name}")
    bad.extend(f"undeclared: {p.relative_to(package)}" for p in sorted(package.rglob("*"))
               if p.is_file() and p.name != "MANIFEST.json"
               and str(p.relative_to(package)) not in parts)
    if content_hash(dict(sorted(parts.items()))) != manifest.get("package_hash"):
        bad.append("package_hash does not recompute from the parts")
    return bad


def _verify_receipt_problems(root: Path, package: Path, manifest: dict) -> list[str]:
    path = root / "baseline" / "receipts" / f"{package.name}.json"
    if not path.is_file():
        return ["no byte-identical re-export receipt; run tools/baseline_export.py "
                f"--verify baseline/{package.name}"]
    payload = json.loads(path.read_text()).get("payload") or {}
    problems = []
    if payload.get("package_hash") != manifest.get("package_hash"):
        problems.append("re-export receipt names a different package_hash")
    if not payload.get("byte_identical"):
        problems.append("the recorded re-export was NOT byte-identical: "
                        f"{payload.get('differing')}")
    return problems


def _baseline_package(root: Path = ROOT) -> dict:
    """DoD #1: the CURRENT package is intact, reproducible, and its lock is the repo's.

    Full byte-reproducibility re-imports the engine, so it runs in
    ``baseline_export.py --verify`` and leaves a receipt; this check reads that
    receipt and verifies what is on disk against its own manifest.
    """
    started = time.monotonic()
    package = identity.current_baseline(root)
    if package is None:
        return {"ok": False, "seconds": 0.0,
                "detail": "no baseline/CURRENT pointer naming a package; run "
                          "tools/baseline_export.py"}
    manifest = json.loads((package / "MANIFEST.json").read_text())
    bad = _manifest_problems(package, manifest)
    root_req, frozen_req = root / "requirements.txt", package / "requirements.txt"
    if not (root_req.is_file() and frozen_req.is_file()):
        bad.append("one of the two requirement locks is missing")
    elif root_req.read_text() != frozen_req.read_text():
        bad.append(f"root requirements.txt differs from baseline/{package.name}/requirements.txt")
    bad.extend(_verify_receipt_problems(root, package, manifest))
    return {
        "ok": not bad,
        "seconds": round(time.monotonic() - started, 2),
        "detail": "; ".join(bad[:8]) or (
            f"baseline/{package.name}: {len(manifest.get('parts', {}))} parts intact, "
            "re-export byte-identical, lock matches the repo"),
        "version": package.name,
        "package_hash": manifest.get("package_hash"),
    }


#: Elapsed times inside a detail line ("224 passed in 4.17s", "7.0 min").
_DURATION = re.compile(r"\b\d+(?:\.\d+)?\s*(?:s|sec|seconds|min)\b")


def fingerprint(row: dict) -> str:
    """What a row SAID — its verdict and reasons, not how long it took.

    Durations are stripped from the detail: pytest reports "N passed in 4.17s",
    and a fingerprint that moved with the clock would report every row as
    changed on every run, which is the same as reporting nothing.
    """
    detail = _DURATION.sub("<t>", str(row.get("detail", "")))
    return content_hash({"ok": row.get("ok"), "detail": detail,
                         "findings": row.get("findings", [])})


def changed_since(previous: dict, fingerprints: dict[str, str]) -> list[str]:
    """Rows whose answer differs from a previous run. Information, not a verdict."""
    before = previous.get("fingerprints") or {}
    return sorted(name for name, fp in fingerprints.items() if before.get(name) != fp)


def gate(previous: dict | None = None) -> dict:
    started = time.monotonic()
    results = {name: _run(argv) for name, argv in _CHECKS}
    for row in results.values():
        row.pop("stdout", None)
    expect, version = expectations()
    results["tier0_corpus"] = _tier0()
    results["tier1_real_replay"] = _receipt_row(expect, version, seeded=False)
    results["tier1_seeded_controls"] = _receipt_row(expect, version, seeded=True)
    results["negative_controls"] = _negative_controls()
    results["baseline_package"] = _baseline_package()
    hook = install_hooks.state(ROOT)
    results["pre_commit_hook"] = {
        "ok": hook["installed"] and hook["current"] and hook["executable"],
        "seconds": 0.0, "detail": hook.get("reason", ""), **hook,
    }

    failed = sorted(name for name, r in results.items() if not r["ok"])
    fingerprints = {name: fingerprint(row) for name, row in results.items()}
    out: dict[str, Any] = {
        "schema_version": "phase0_gate.v1.1",
        "ok": not failed,
        "failed": failed,
        "code_hash": expect["code"],
        "fingerprints": fingerprints,
        "seconds": round(time.monotonic() - started, 2),
        "checks": results,
    }
    if previous is not None:
        out["changed_since_previous"] = changed_since(previous, fingerprints)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--previous", default=None,
                    help="a previous --json output, to report which rows changed")
    args = ap.parse_args(argv)

    previous = None
    if args.previous and Path(args.previous).is_file():
        previous = json.loads(Path(args.previous).read_text())
    result = gate(previous)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result["ok"] else 1

    for name, row in result["checks"].items():
        mark = "ok " if row["ok"] else ("SKIP" if row.get("skipped") else "FAIL")
        print(f"  {mark:4s} {name:22s} {row['seconds']:6.2f}s")
        if not row["ok"] and row.get("detail"):
            for line in row["detail"].splitlines()[-6:]:
                print(f"         {line}")
    if "changed_since_previous" in result:
        print(f"\n  changed since previous run: {result['changed_since_previous'] or 'none'}")
    print(f"\nphase 0 gate: {'GREEN' if result['ok'] else 'RED'} "
          f"in {result['seconds']:.1f}s")
    if not result["ok"]:
        print(f"  failed: {', '.join(result['failed'])}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
