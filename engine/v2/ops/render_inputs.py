"""Pure logic for the supervised ``legacy_render`` action (P2-5, D19).

Nothing here imports a legacy (``engine.*``) symbol — that would open a
second adapter module inside the ``engine.v2.ops`` package, which
``checks/import_layers.py`` §4.2 forbids (one adapter module per package).
Legacy calls (``render_bundle``, ``build_meta``, ``build_health``, ...) stay
in ``engine.v2.ops.legacy_adapter``; this module only assembles and compares
their inputs and outputs.
"""
from __future__ import annotations

import json
import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath

import pandas as pd

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail

__all__ = [
    "ABSENT_STAGES",
    "EXECUTION_METADATA_FIELDS",
    "absent_stage_flags",
    "assemble_scores",
    "bundle_content_hash",
    "diff_bundles",
    "normalized_bundle_entries",
    "stage_ledger_generation",
    "stage_model_evidence",
]

#: guides/rearchitecture_phase2_data_access.md §9.4: "Stages not carried:
#: refresh, the validate-refresh battery, the Tier 3/4 rebuild, missed-night
#: backfill, and calibration flags stay outside the shadow graph. Every
#: shadow receipt lists them as absent so a shadow board is never mistaken
#: for a production one."
ABSENT_STAGES = ("refresh", "validate_refresh", "tier34_rebuild", "backfill",
                  "calibration_flags")

#: Top-level keys of ``meta.json`` / ``health.json`` that carry wall-clock or
#: host state rather than board content: ``generated_at`` stamps the instant
#: ``build_meta``/``build_health`` ran; ``freshness`` embeds per-source ages
#: and the last network-call timestamp off disk; ``quota`` embeds the quota
#: ledger's own ``ts``; ``cron`` names the host's schedule entry (legacy
#: nightly interpolates ``paths.ROOT.name`` into it). None of the four is
#: reproducible from (scores, ladder, model evidence, ledger, finality)
#: alone, so D19 parity excludes them rather than freezing a clock.
EXECUTION_METADATA_FIELDS = {
    "meta": ("generated_at", "freshness", "quota", "cron"),
    "health": ("generated_at",),
}


def absent_stage_flags() -> list[dict]:
    """One board flag per :data:`ABSENT_STAGES` entry — the shadow disclosure."""
    return [{"kind": "shadow_stage_absent", "stage": stage,
             "detail": f"{stage} does not run in the supervised shadow render"}
            for stage in ABSENT_STAGES]


def assemble_scores(score_document: dict) -> pd.DataFrame:
    """Reproduce the legacy nightly's board+ladder frame from ``score.json``.

    ``engine/dashboard/nightly.py`` (quoted, its own line numbers)::

        1412  scores = score_calendar(
        1413      as_of, horizon_days=horizon_days, alt_strikes=0,
        1414      scorer=engine, tickers=tickers,
        1415      progress_every=10,
        1416  )
        1422  board_scores = scores
        ...
        1456  ladder = strike_ladder(
        1457      board_scores, scorer=engine, alt_strikes=alt_strikes, as_of=as_of
        1458  )
        1459  if ladder:
        1460      scores = pd.concat([scores, pd.DataFrame(ladder)], ignore_index=True)

    ``_action_score`` already ran ``score_calendar`` and ``strike_ladder`` and
    wrote both frames into the score artifact (``rows`` is ``board_scores``,
    ``ladder`` is the strike-ladder rows) — so this performs the *same*
    ``pd.concat([scores, pd.DataFrame(ladder)], ignore_index=True)`` over
    those two frames rather than re-deriving either one.
    """
    board = pd.DataFrame(score_document.get("rows") or [])
    ladder_rows = score_document.get("ladder") or []
    if not ladder_rows:
        return board
    return pd.concat([board, pd.DataFrame(ladder_rows)], ignore_index=True)


def stage_model_evidence(evidence_path: Path, legacy_root: Path) -> Path:
    """Place the bound model-evidence artifact at its legacy on-disk path.

    ``render_bundle`` reads it back through
    ``engine.dashboard.model_evidence.load_model_evidence()``, which is
    ``paths.FEATURES / "model_evidence.json"`` — i.e.
    ``<legacy_root>/data/features/model_evidence.json`` once
    ``INVESTING_PLAN_ROOT`` is ``legacy_root``.
    """
    destination = Path(legacy_root) / "data" / "features" / "model_evidence.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    shutil.copyfile(Path(evidence_path), destination)
    return destination


def stage_ledger_generation(tar_path: Path, legacy_root: Path) -> Path:
    """Replace ``<legacy_root>/ledger`` with exactly one export generation.

    Any staged copy of the *mutable* ledger (e.g. a stray general read-set
    copy) is removed first — the book view and the live size-model MAE must
    read the bound, immutable generation, never a mutable ledger file that
    happened to be sitting in staging.
    """
    ledger_dir = Path(legacy_root) / "ledger"
    if ledger_dir.exists() or ledger_dir.is_symlink():
        shutil.rmtree(ledger_dir)
    ledger_dir.mkdir(parents=True)
    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise fail("INTEGRITY_FAILED",
                           "ledger generation archive member is unsafe",
                           details={"member": member.name})
            if not member.isfile() and not member.isdir():
                raise fail("INTEGRITY_FAILED",
                           "ledger generation archive member is not a plain file",
                           details={"member": member.name})
            if name.parts and name.parts[0] not in ("predictions", "outcomes"):
                raise fail("INTEGRITY_FAILED",
                           "ledger generation archive has an unexpected member",
                           details={"member": member.name})
        archive.extractall(ledger_dir)
    return ledger_dir


_JS_ASSIGNMENT = re.compile(r"^window\.[A-Za-z0-9_]+ = ")


def _strip_metadata(stem: str, payload):
    fields = EXECUTION_METADATA_FIELDS.get(stem)
    if not fields or not isinstance(payload, dict):
        return payload
    return {key: value for key, value in payload.items() if key not in fields}


def _normalized_js(stem: str, text: str):
    match = _JS_ASSIGNMENT.match(text)
    if not match:
        return text
    body = text[match.end():]
    if body.endswith(";\n"):
        body = body[:-2]
    try:
        payload = json.loads(body)
    except ValueError:
        return text
    stripped = _strip_metadata(stem, payload)
    return match.group(0) + json.dumps(stripped, sort_keys=True, default=str) + ";\n"


def normalized_bundle_entries(bundle_dir: Path) -> dict:
    """Every file in a rendered bundle, keyed by relative path.

    A declared metadata file (``meta.json``/``meta.js``, ``health.json``/
    ``health.js``) is parsed and stripped of :data:`EXECUTION_METADATA_FIELDS`
    before it enters the map; every other file is a hash of its raw bytes.
    Two bundles compare equal under this map exactly when D19 requires:
    identical everywhere except declared execution-metadata fields.
    """
    bundle_dir = Path(bundle_dir)
    entries: dict[str, object] = {}
    for path in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(bundle_dir).as_posix()
        stem = path.stem
        if path.suffix == ".json" and stem in EXECUTION_METADATA_FIELDS:
            entries[relative] = _strip_metadata(stem, json.loads(path.read_text()))
        elif path.suffix == ".js" and stem in EXECUTION_METADATA_FIELDS:
            entries[relative] = _normalized_js(stem, path.read_text())
        else:
            import hashlib

            entries[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return entries


def bundle_content_hash(bundle_dir: Path) -> str:
    """The D19 identity hash: every bundle file, metadata fields removed."""
    return content_hash(normalized_bundle_entries(bundle_dir))


def diff_bundles(dir_a: Path, dir_b: Path) -> list[str]:
    """Name every file (and, for meta/health, every field) that differs."""
    left = normalized_bundle_entries(dir_a)
    right = normalized_bundle_entries(dir_b)
    diffs = []
    for name in sorted(set(left) | set(right)):
        a, b = left.get(name), right.get(name)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            changed = sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))
            diffs.append(f"{name}: {', '.join(changed) or 'missing'}")
        else:
            diffs.append(name)
    return diffs
