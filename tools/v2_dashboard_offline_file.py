#!/usr/bin/env python3
"""P6-5 export-offline-file: a single-file offline export of a PINNED v2 release.

Not a scrape of a running dashboard: it operates directly on the v2
publication tree (``<release-root>/releases/<release-id>/``), which already
holds the real ``engine.dashboard.render.render_bundle`` output (P3-4's
compatibility bundle format -- see ``tools/v2_dashboard_project.py``). It
reuses ``write_single_file`` byte-for-byte, the same function the legacy
nightly uses, over that real release directory -- never a second renderer,
never an HTTP fetch of a live page.

Guarantees checked here, not merely assumed from ``write_single_file``:

* the release id is a single, non-traversing path segment (no ``..``, no
  absolute path, no embedded separator) and the release directory is not
  reached through a symlink -- the same discipline
  ``engine.v2.serving.operations`` applies to every path it resolves;
* the built file carries no credential-shaped text and no ``.env`` value
  (``engine.dashboard.publish.secret_scan``, the same scan the nightly
  publish path runs before anything ships);
* the built file references no external network resource (every
  ``src``/``href`` is either an anchor, a ``data:`` URI, or already inlined
  by ``write_single_file``) -- the acceptance's "opens without network".

Usage::

    python3 tools/v2_dashboard_offline_file.py \\
        --release-root /path/to/release-root --release-id r_2026... \\
        [--out earnings-board-r_2026....html] [--artifact-root evidence/]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.dashboard.publish import secret_scan  # noqa: E402
from engine.dashboard.render import write_single_file  # noqa: E402
from engine.v2.foundation import ArtifactError, fsync_directory, safe_relative_path  # noqa: E402

__all__ = ["OfflineFileError", "export_offline_file", "main"]

SCHEMA_VERSION = "offline_file_receipt.v1.0"

#: An external resource reference in the built page: a real network fetch,
#: not an anchor (``href="#..."``) and not an already-inlined ``data:`` URI.
_EXTERNAL_REF_RE = re.compile(r'(?:src|href)="(https?://[^"]*)"')


class OfflineFileError(RuntimeError):
    """The release id was unsafe, or the built file failed a safety check."""


def _release_directory(release_root: Path, release_id: str) -> Path:
    """The one pinned release directory this export is allowed to read.

    ``release_id`` must resolve to exactly one path segment (P2's
    ``safe_relative_path``): no ``..``, no absolute path, no embedded ``/``.
    Neither the release directory nor its parent may be a symlink -- a
    pinned release is never reached indirectly.
    """
    try:
        parts = safe_relative_path(release_id)
    except ArtifactError as exc:
        raise OfflineFileError(f"unsafe release id: {exc}") from exc
    if len(parts) != 1:
        raise OfflineFileError("release id must be a single path segment")
    releases_dir = release_root / "releases"
    if releases_dir.is_symlink():
        raise OfflineFileError("releases directory is indirect")
    release_dir = releases_dir / release_id
    if release_dir.is_symlink():
        raise OfflineFileError("release directory is indirect")
    if not release_dir.is_dir():
        raise OfflineFileError(f"no such release: {release_id}")
    return release_dir


def _external_refs(html: str) -> list[str]:
    return sorted(set(_EXTERNAL_REF_RE.findall(html)))


def export_offline_file(release_root: Path | str, release_id: str, *,
                        out: Path | str | None = None,
                        artifact_root: Path | str | None = None) -> dict:
    """Build the single-file offline export for ``release_id`` and return its receipt.

    Builds into a private scratch directory first so ``secret_scan`` (which
    walks a directory) and the external-reference check both run BEFORE
    anything lands at ``out`` -- a build that fails either check leaves no
    file behind at the destination.
    """
    release_root = Path(release_root)
    release_dir = _release_directory(release_root, release_id)
    out = Path(out) if out is not None else release_root / f"earnings-board-{release_id}.html"
    if out.is_symlink():
        raise OfflineFileError("output path is indirect")

    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)
        built = write_single_file(release_dir, scratch_dir / out.name)
        html = built.read_text()

        hits = secret_scan(scratch_dir)
        if hits:
            raise OfflineFileError(
                f"offline export carries {len(hits)} credential-shaped match(es); refusing to ship it")

        external = _external_refs(html)
        if external:
            raise OfflineFileError(
                f"offline export references {len(external)} external resource(s); it must open with no network")

        data = built.read_bytes()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        fsync_directory(out.parent)

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "path": str(out),
        "byte_size": len(data),
        "content_hash": "sha256:" + hashlib.sha256(data).hexdigest(),
        "secret_scan_hits": 0,
        "external_refs": 0,
    }
    if artifact_root is not None:
        artifact_root = Path(artifact_root)
        artifact_root.mkdir(parents=True, exist_ok=True)
        receipt_path = artifact_root / f"offline_file_{release_id}.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        receipt = export_offline_file(args.release_root, args.release_id,
                                      out=args.out, artifact_root=args.artifact_root)
    except OfflineFileError as exc:
        print(f"offline-file: refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
