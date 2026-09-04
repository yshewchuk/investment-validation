"""``data/MANIFEST.md`` and the snapshot hash — generated, never hand-written.

The old manifest was a document someone remembered to update. This one is
output: it is regenerated on every rebuild from what is actually on disk, so it
cannot drift from the store it describes.

The **snapshot hash** is a sha256 over the Tier-2 table content hashes plus the
Tier-3 panel digest. It is the single identifier a report's provenance block
pins, and it answers the only question that matters when a number is disputed
later: *was this produced from the same data?*
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from engine import paths
from engine.data import store
from engine.data.schemas import CONVENTIONS, SCHEMAS, SOURCE_PRIORITY

__all__ = ["snapshot_hash", "collect_stats", "write_manifest", "write_snapshot", "read_snapshot"]


def collect_stats() -> dict:
    """Row counts, year coverage, and content hashes for every Tier-2 table."""
    return {name: store.table_stats(name).as_dict() for name in sorted(SCHEMAS)}


def _panel_digest() -> str | None:
    return store.file_sha256(paths.PANEL) if paths.PANEL.exists() else None


def _tier4_digest() -> str | None:
    """The Tier-4 hash — recorded beside the snapshot, deliberately not inside it.

    Tier 4 holds model forecasts, so it moves whenever a champion is promoted.
    Folding it into :func:`snapshot_hash` would make every such promotion
    invalidate the provenance of experiments that never read a forecast, which
    is the exact coupling Tier 4 exists to avoid. An experiment that reads
    forecasts pins this hash *as well*; one that does not pins the snapshot
    alone. Reports must say which — see ``guides/tier4_feature_models.md`` §10.
    """
    return store.file_sha256(paths.TIER4) if paths.TIER4.exists() else None


def snapshot_hash(stats: dict | None = None) -> str:
    """Deterministic identifier for the current state of the whole store."""
    stats = stats if stats is not None else collect_stats()
    parts = [f"{name}:{stats[name]['content_hash']}" for name in sorted(stats)]
    panel = _panel_digest()
    if panel:
        parts.append(f"panel:{panel}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def write_snapshot(stats: dict | None = None, path: Path | None = None) -> str:
    """Write ``data/features/SNAPSHOT`` and return the hash."""
    stats = stats if stats is not None else collect_stats()
    digest = snapshot_hash(stats)
    path = paths.assert_writable(path or paths.SNAPSHOT_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "snapshot": digest,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "format": store.table_format(),
                "tables": stats,
                "panel_sha256": _panel_digest(),
                "tier4_sha256": _tier4_digest(),
            },
            indent=1,
            sort_keys=True,
        )
    )
    return digest


def read_snapshot(path: Path | None = None) -> dict | None:
    path = path or paths.SNAPSHOT_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def _table_rows(stats: dict) -> str:
    header = (
        "| Table | Rows | Years | Partitions | Files | Bytes | Content hash |\n"
        "|---|---:|---|---:|---:|---:|---|\n"
    )
    lines = []
    for name in sorted(stats):
        s = stats[name]
        lines.append(
            f"| `{name}` | {s['rows']:,} | {s['years']} | {s['partitions']} | "
            f"{s['files']} | {s['bytes']:,} | `{s['content_hash'][:16]}…` |"
        )
    return header + "\n".join(lines)


def _quarantine_summary() -> str:
    root = paths.QUARANTINE
    flags = sorted(root.glob("*.flag.json")) if root.exists() else []
    if not flags:
        return "No quarantine flags. Every normalized row passed validation.\n"
    lines = [
        f"**{len(flags)} quarantine flag(s).** Raw files are kept in place; the "
        "listed rows were excluded from Tier 2.\n",
        "| Raw file | Reason |",
        "|---|---|",
    ]
    for path in flags[:50]:
        try:
            entries = json.loads(path.read_text())
            entry = entries[-1] if isinstance(entries, list) else entries
            lines.append(f"| `{entry.get('source_file')}` | {entry.get('reason')} |")
        except (ValueError, OSError):
            lines.append(f"| `{path.name}` | (unreadable flag file) |")
    if len(flags) > 50:
        lines.append(f"| … | {len(flags) - 50} more |")
    return "\n".join(lines) + "\n"


def write_manifest(
    stats: dict | None = None,
    *,
    path: Path | None = None,
    extra_sections: dict[str, str] | None = None,
) -> Path:
    """Render ``data/MANIFEST.md`` from the current store."""
    stats = stats if stats is not None else collect_stats()
    digest = snapshot_hash(stats)
    path = paths.assert_writable(path or paths.MANIFEST)
    path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = sum(s["rows"] for s in stats.values())
    total_bytes = sum(s["bytes"] for s in stats.values())

    sections = [
        "# Data Manifest",
        "",
        "**Generated output — do not edit by hand.** Regenerated by "
        "`python3 -m engine.data.rebuild`; any manual change is lost on the next "
        "rebuild and, worse, would make this file disagree with the store it "
        "describes.",
        "",
        f"- Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Snapshot hash: `{digest}`",
        f"- Storage format: `{store.table_format()}`",
        f"- Totals: **{total_rows:,} rows**, {total_bytes:,} bytes across "
        f"{len(stats)} tables",
        "",
        "## Tier 2 — normalized store",
        "",
        _table_rows(stats),
        "",
        "## Tier 3 — features",
        "",
        (
            f"- `{paths.PANEL.relative_to(paths.ROOT)}` — sha256 "
            f"`{_panel_digest()}`"
            if _panel_digest()
            else "- panel not built"
        ),
        "",
        "## Source priority on conflict",
        "",
        "| Domain | Winner |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(SOURCE_PRIORITY.items())],
        "",
        "## Unit and convention traps (fixed in `engine/data/normalize/`)",
        "",
        *[f"- **{k}** — {v}" for k, v in sorted(CONVENTIONS.items())],
        "",
        "## Ingestion quarantine",
        "",
        _quarantine_summary(),
    ]
    for title, body in (extra_sections or {}).items():
        sections += [f"## {title}", "", body, ""]

    path.write_text("\n".join(sections))
    return path
