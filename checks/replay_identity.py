"""What a replay receipt is a receipt OF, and whether it still holds.

A tier-1 receipt says "the engine re-scored these pairs and they agreed". That
sentence is only true of a particular engine, particular frozen dependencies,
a particular corpus and a particular baseline. If any of those has moved since
the replay ran, the receipt is about something that no longer exists, and a
gate that reads it anyway is green over code nobody replayed. The 2026-09-12
review planted a receipt that replayed 1 of 18 pairs from another commit and the
gate accepted it; this module is the fix.

Identity is by CONTENT, not by git commit. Binding to a commit would turn the
gate red on every documentation commit, and would still miss an uncommitted
edit to ``engine/score.py``. The code hash covers every file the replay's
answer is a function of; the dependency hash covers the frozen artifacts named
in the baseline package, re-verified against the bytes on disk.

It also holds the seeded negative controls' SPECIFICATION — which cause must
produce which findings in which stage — so the tier-0 check, the tier-1 replay
and the gate judge a control by one definition.

Stdlib plus ``engine/v2/diagnosis`` only. The gate reads this in seconds, and
tests drive it without a scorer.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.diagnosis import content_hash  # noqa: E402

__all__ = [
    "CODE_FILES",
    "SEEDED_CONTROLS",
    "code_hash",
    "current_pointer",
    "current_baseline",
    "dependency_identity",
    "store_snapshot",
    "evaluate_receipt",
    "pick_seed_targets",
    "check_control",
    "FORECAST_BLOCK",
]

#: Everything a replay's answer is a function of, beyond the frozen artifacts:
#: the legacy engine that re-scores, the v2 packages the harness imports to
#: compare and hash, and the harness itself.
#:
#: ``engine/v2/**`` outside :data:`REPLAY_V2_PACKAGES` is deliberately NOT in
#: the hash (rearchitecture phase 1 decision D6). Legacy may not import v2
#: (§4.2 rule 3, enforced), and the harness (tools/replay_tier1.py,
#: tools/capture_tier0_corpus.py, checks/tier0_corpus.py, checks/replay_identity.py)
#: imports diagnosis and related packages whose own dependency closures include
#: scoring, models, features, registry, domain/generation, domain/valuation, and
#: foundation/contracts. An edit to ``engine/v2/ops`` therefore cannot change a
#: replay's answer, and binding the receipt to it would turn the phase-0 gate red on
#: every operations commit — the same reasoning that rejected binding to a git commit.
#: ``tests/test_v2_ops_replay_scope.py`` re-derives that closure from the
#: import graph, so a harness that starts importing another v2 package fails
#: until this tuple grows with it.
CODE_ROOTS = (("engine", "**/*.py"),)
REPLAY_V2_PACKAGES = (
    "engine/v2/contracts",
    "engine/v2/diagnosis",
    "engine/v2/domain/generation",
    "engine/v2/domain/valuation",
    "engine/v2/features",
    "engine/v2/foundation",
    "engine/v2/models",
    "engine/v2/models/training",
    "engine/v2/registry",
    "engine/v2/scoring",
)
CODE_FILES = (
    "tools/replay_tier1.py",
    "tools/capture_tier0_corpus.py",
    "checks/tier0_corpus.py",
    "checks/replay_identity.py",
)

#: The fields `e845f3e` blanked: the forecast block a pinned replay dropped.
FORECAST_BLOCK = ("forecast_abs_move", "forecast_p10", "forecast_p90",
                  "forecast_sd", "forecast_model", "forecast_fold")


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------


def in_replay_scope(rel: str) -> bool:
    """Whether a repo-relative engine file is part of what a replay executes."""
    if not rel.startswith("engine/v2/"):
        return True
    return rel == "engine/v2/__init__.py" or any(
        rel.startswith(package + "/") for package in REPLAY_V2_PACKAGES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_hash(root: Path = ROOT) -> str:
    """Content hash of every code file a replay's answer depends on."""
    files: dict[str, str | None] = {}
    for rel_dir, pattern in CODE_ROOTS:
        for path in sorted((root / rel_dir).glob(pattern)):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if in_replay_scope(rel):
                files[rel] = sha256_file(path)
    for rel in CODE_FILES:
        path = root / rel
        files[rel] = sha256_file(path) if path.is_file() else None
    return content_hash(files)


def current_pointer(directory: Path) -> str | None:
    """The version a ``CURRENT`` pointer names, or None."""
    pointer = directory / "CURRENT"
    if not pointer.is_file():
        return None
    try:
        version = json.loads(pointer.read_text()).get("version")
    except (ValueError, OSError):
        return None
    return str(version) if version else None


def current_baseline(root: Path = ROOT) -> Path | None:
    """The baseline package ``baseline/CURRENT`` names — never "latest by name".

    Sorting directory names picks ``2026-09-12b`` over ``2026-09-12`` only by
    accident of spelling; an explicit pointer is the published choice.
    """
    version = current_pointer(root / "baseline")
    if version is None:
        return None
    candidate = root / "baseline" / version
    return candidate if (candidate / "MANIFEST.json").is_file() else None


def dependency_identity(baseline: Path | None,
                        root: Path = ROOT) -> tuple[str | None, list[dict]]:
    """``(frozen dependency hash, drift rows)`` for a baseline package.

    The hash is over the DECLARED artifact hashes and the frozen store
    snapshot, so it names what the answers depend on; the drift rows compare
    those declarations with the bytes on disk now.
    """
    if baseline is None:
        return None, [{"path": "baseline/CURRENT", "issue": "no current baseline"}]
    deps_path = baseline / "artifacts" / "dependencies.json"
    if not deps_path.is_file():
        return None, [{"path": str(deps_path), "issue": "no dependencies part"}]
    payload = json.loads(deps_path.read_text())["payload"]
    declared: dict[str, str] = {}
    drift: list[dict] = []
    for row in payload.get("artifacts", []):
        declared[row["path"]] = row["sha256"]
        path = root / row["path"]
        if not path.is_file():
            drift.append({"path": row["path"], "issue": "missing"})
        elif path.stat().st_size != row["bytes"]:
            drift.append({"path": row["path"], "issue": "size differs"})
        elif sha256_file(path) != row["sha256"]:
            drift.append({"path": row["path"], "issue": "sha256 differs"})
    drift.extend(payload.get("registry_sha_drift") or [])
    snapshot = (payload.get("tier2_snapshot") or {}).get("snapshot")
    return content_hash({"artifacts": declared, "tier2_snapshot": snapshot}), drift


def store_snapshot() -> str | None:
    """The data snapshot id the engine reads, from its snapshot file."""
    from engine import paths  # stdlib-only; honours INVESTING_PLAN_ROOT

    try:
        return json.loads(paths.SNAPSHOT_FILE.read_text()).get("snapshot")
    except (ValueError, OSError):
        return None


# --------------------------------------------------------------------------
# judging a receipt
# --------------------------------------------------------------------------


def evaluate_receipt(doc: dict[str, Any], *, corpus_hash: str | None,
                     declared_pairs: int, code: str, dependencies: str | None,
                     drift: list[dict], baseline_version: str | None,
                     snapshot: str | None) -> list[str]:
    """Every reason a replay receipt does not certify the CURRENT state.

    Empty means the receipt binds to this corpus, this code, these frozen
    dependencies and this baseline, replayed every declared pair in full, and
    agreed. Each problem is a sentence an operator can act on.
    """
    payload = doc.get("payload") or {}
    bind = payload.get("bindings") or {}
    problems: list[str] = []
    checks: tuple[tuple[bool, str], ...] = (
        (bind.get("corpus_hash") != corpus_hash,
         "receipt binds a different corpus_hash (stale corpus)"),
        (bind.get("code_hash") != code,
         "code changed since the replay ran (code_hash differs)"),
        (dependencies is None, "no frozen dependency identity in the baseline"),
        (dependencies is not None and bind.get("dependencies_hash") != dependencies,
         "receipt binds different frozen dependencies"),
        (bool(drift), f"{len(drift)} frozen dependencies differ on disk now"),
        (bind.get("baseline") != baseline_version,
         f"receipt binds baseline {bind.get('baseline')!r}, "
         f"current is {baseline_version!r}"),
        (snapshot is not None and bind.get("store_snapshot_at_replay") != snapshot,
         "the Tier-2 store snapshot changed since the replay"),
        (bool(bind.get("deps_unverified")), "ran with --skip-deps-verify"),
        (bind.get("limit") is not None, f"partial run (--limit {bind.get('limit')})"),
        (bind.get("declared") != declared_pairs,
         f"receipt declares {bind.get('declared')} pairs, corpus declares "
         f"{declared_pairs}"),
        (bind.get("replayed") != declared_pairs,
         f"replayed {bind.get('replayed')} of {declared_pairs} declared pairs"),
        (bool(bind.get("skipped")), f"{bind.get('skipped')} pairs skipped"),
        (payload.get("verdict") != "agree",
         f"verdict is {payload.get('verdict')!r}"),
    )
    problems.extend(message for failed, message in checks if failed)
    return problems


# --------------------------------------------------------------------------
# the seeded negative controls
# --------------------------------------------------------------------------

#: cause -> the commit that fixed it, and per receipt kind the findings the
#: seeded defect must produce. A receipt kind not named for a cause must stay
#: CLEAN for that cause's target: a control that also moves something it should
#: not is a control that proves less than it claims.
#:
#: Receipt kinds: ``record`` compares the frozen record with the replayed one;
#: ``round_trip`` compares the replayed record with what a write-and-read-back
#: of it returned (tier 1 only); ``integrity`` compares the hash stored beside
#: a written artifact with the hash of the bytes actually written.
#:
#: The fifth 2026-09-11 cause, `28cf8b1` (a field dropped from the compared
#: set), cannot be seeded without editing the comparator. It is checked by
#: construction in every run instead: each pair's compared population must
#: equal the leaves of its own records — see ``field_set_mismatches``.
SEEDED_CONTROLS: dict[str, dict[str, Any]] = {
    "forecast_suppressed": {
        "commit": "e845f3e",
        "findings": {"record": {"stages": ["forecast"],
                                "includes": ["forecast_abs_move"]}},
    },
    "analog_bootstrap_reseeded": {
        "commit": "b9aa1fd",
        "findings": {"record": {"stages": ["analogs"],
                                "includes": ["ci_low", "ci_high"],
                                "excludes": ["exp_pnl_analog", "n_analogs",
                                             "win_analog", "selected_row_ids",
                                             "contributing_row_ids"]}},
    },
    "replay_input_rounded": {
        "commit": "b33036c",
        "findings": {"record": {"stages": ["serialization"],
                                "prefix": "structure_params."}},
    },
    "rounded_after_digest": {
        "commit": "6b9d5cf",
        "findings": {"integrity": {"stages": ["serialization"],
                                   "includes": ["payload_hash"]},
                     "round_trip": {"stages": ["serialization"],
                                    "prefix": "structure_params."}},
    },
}


def _float(value: Any) -> bool:
    return isinstance(value, float)


def _pinned_with_forecast(pair: dict) -> bool:
    payload = pair.get("payload") or {}
    return (payload.get("record_kind") == "score_result"
            and bool((payload.get("request") or {}).get("structure_params"))
            and _float((payload.get("record") or {}).get("forecast_abs_move")))


def _has_interval(pair: dict) -> bool:
    payload = pair.get("payload") or {}
    record = payload.get("record") or {}
    return (payload.get("record_kind") == "score_result"
            and _float(record.get("ci_low")) and _float(record.get("ci_high")))


def _rounding_sensitive(pair: dict) -> bool:
    payload = pair.get("payload") or {}
    params = (payload.get("record") or {}).get("structure_params")
    return (payload.get("record_kind") == "score_result"
            and isinstance(params, dict)
            and any(_float(v) and round(v, 6) != v for v in params.values()))


#: Scarcest first, so a common predicate cannot take the only pair a rare one
#: can use.
_TARGET_ORDER: tuple[tuple[str, Callable[[dict], bool]], ...] = (
    ("forecast_suppressed", _pinned_with_forecast),
    ("analog_bootstrap_reseeded", _has_interval),
    ("replay_input_rounded", _rounding_sensitive),
    ("rounded_after_digest", _rounding_sensitive),
)


def pick_seed_targets(pairs: dict[str, dict]) -> tuple[dict[str, str], list[str]]:
    """One DISTINCT frozen pair per cause, and the causes that found none.

    Distinct pairs are what make the controls separable: each cause's findings
    are attributable to its own pair, so one pass over all four is also four
    ablations at once. The forecast target must be a PINNED request, because
    the `e845f3e` defect only fired when ``structure_params`` were supplied.
    """
    used: set[str] = set()
    targets: dict[str, str] = {}
    missing: list[str] = []
    for cause, predicate in _TARGET_ORDER:
        choice = next((fid for fid in sorted(pairs)
                       if fid not in used and predicate(pairs[fid])), None)
        if choice is None:
            missing.append(cause)
            continue
        used.add(choice)
        targets[cause] = choice
    return targets, missing


def _spec_problems(kind: str, spec: dict, found: list[dict]) -> list[str]:
    if not found:
        return [f"{kind}: no finding — the seeded defect went undetected"]
    problems: list[str] = []
    stages = sorted({f["first_differing_stage"] for f in found})
    if stages != spec["stages"]:
        problems.append(f"{kind}: localized to {stages}, expected {spec['stages']}")
    paths = {f["field_path"] for f in found}
    missing = [p for p in spec.get("includes", ()) if p not in paths]
    if missing:
        problems.append(f"{kind}: no finding at {missing}")
    moved = [p for p in spec.get("excludes", ()) if p in paths]
    if moved:
        problems.append(f"{kind}: moved fields the defect leaves alone: {moved}")
    prefix = spec.get("prefix")
    stray = sorted(p for p in paths if prefix and not p.startswith(prefix))
    if stray:
        problems.append(f"{kind}: findings outside {prefix}*: {stray[:5]}")
    return problems


def check_control(cause: str, findings_by_kind: dict[str, list[dict]],
                  kinds: tuple[str, ...] | None = None) -> list[str]:
    """Every way one seeded control failed to produce exactly its findings.

    ``findings_by_kind`` holds, for each receipt kind the run produced, the
    finding dicts on the cause's target pair. ``kinds`` limits the judgement
    to the receipt kinds a tier actually produces — tier 0 writes no replayed
    record, so it has no round-trip receipt to hold a finding.
    """
    expected = {kind: spec for kind, spec in SEEDED_CONTROLS[cause]["findings"].items()
                if kinds is None or kind in kinds}
    problems: list[str] = []
    for kind, found in sorted(findings_by_kind.items()):
        spec = expected.get(kind)
        if spec is None:
            if found:
                problems.append(f"{kind}: {len(found)} finding(s) where the "
                                f"defect should leave this receipt clean, e.g. "
                                f"{found[0]['field_path']}")
            continue
        problems.extend(_spec_problems(kind, spec, found))
    absent = sorted(set(expected) - set(findings_by_kind))
    problems.extend(f"{kind}: this run produced no {kind} receipt" for kind in absent)
    return problems
