"""Bundle → target, atomic, access-checked.

Two guarantees the guide makes non-negotiable:

* **Atomicity.** A publish either replaces the served snapshot completely or
  not at all. The local publisher stages each release under
  ``{target}/releases/{stamp}/`` and flips a ``current`` symlink with a single
  atomic ``os.replace``; a process killed mid-copy leaves the previous release
  serving.
* **Nothing ships unauthenticated.** The board discloses position intent and
  redistributes ORATS-derived quotes, so :func:`publish_bundle` refuses any target
  whose access probe returns an unauthenticated ``200`` for ``meta.json``.
  The probe result is recorded either way.

The remote channel is deliberately pluggable: ``LocalPublisher`` is the
reference implementation (and the acceptance tests' target); a shell-command
publisher covers Cloudflare Pages / R2 / S3 setups once the user creates the
project and drops credentials in ``.env`` — see ``dashboard/README.md`` for
the checklist of steps only the user can do.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from engine import paths

__all__ = [
    "PublishError",
    "LocalPublisher",
    "CommandPublisher",
    "publish_bundle",
    "secret_scan",
    "access_probe",
    "KEEP_RELEASES",
    "SECRET_PATTERNS",
]

#: Text extensions scanned for secrets. Binaries are skipped by extension;
#: the bundle carries nothing else by construction.
_TEXT_SUFFIXES = {
    ".html", ".js", ".css", ".json", ".md", ".txt", ".csv", ".map", ".svg",
}

#: The guide's grep list (``token``, ``key``, ``/root/``) at secret-grade
#: specificity: the bare word "key" cannot be banned from JavaScript
#: (``Object.keys``) without breaking the client, so the patterns name what a
#: leaked credential actually looks like, plus any literal value from ``.env``.
SECRET_PATTERNS = (
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"api[_-]?key", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"passw(or)?d", re.IGNORECASE),
    re.compile(r"private[_ -]?key", re.IGNORECASE),
    re.compile(r"/root/", re.IGNORECASE),
    re.compile(r"\.env\b"),
)

#: Releases kept alongside ``current``; older ones are pruned after the flip.
KEEP_RELEASES = 5


class PublishError(RuntimeError):
    """The bundle was refused, or the target could not be staged safely."""


# --------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------


def secret_scan(bundle: Path | str) -> list[dict]:
    """Grep the bundle for credentials and internal paths.

    Returns a list of hits; empty means clean. Two pattern classes: generic
    credential words (:data:`SECRET_PATTERNS`) and the literal values of the
    local ``.env`` — a staged secret is caught even when it is an unknown
    word, because the exact bytes are known locally.
    """
    bundle = Path(bundle)
    hits: list[dict] = []

    env_values: list[str] = []
    if paths.ENV_FILE.exists():
        try:
            for line in paths.ENV_FILE.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if len(value) >= 8:
                    env_values.append(value)
        except OSError:
            pass

    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                hits.append(
                    {
                        "file": str(path.relative_to(bundle)),
                        "pattern": pattern.pattern,
                        "snippet": text[max(0, match.start() - 20): match.end() + 20],
                    }
                )
        for value in env_values:
            if value in text:
                hits.append(
                    {
                        "file": str(path.relative_to(bundle)),
                        "pattern": "<.env value>",
                        "snippet": "(redacted)",
                    }
                )
    return hits


def access_probe(url: str, *, timeout: float = 15.0) -> dict:
    """Unauthenticated GET of ``meta.json`` on the published target.

    Returns ``{"status": int|None, "public": bool, "error": str|None}``.
    ``public`` is True only on a definitive unauthenticated 200; a redirect
    (302 to an Access login) or any error is NOT proof of publicness.
    """
    probe = url.rstrip("/")
    if not probe.endswith("meta.json"):
        probe = f"{probe}/data/meta.json"
    request = urllib.request.Request(probe, headers={"User-Agent": "phase3-access-probe"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
        return {"status": status, "public": status == 200, "error": None}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "public": exc.code == 200, "error": None}
    except Exception as exc:
        return {"status": None, "public": False, "error": f"{type(exc).__name__}: {exc}"[:200]}


def _validate_bundle(bundle: Path) -> None:
    for required in ("index.html", "data/board.json", "data/meta.json"):
        if not (bundle / required).exists():
            raise PublishError(f"bundle {bundle} is incomplete: missing {required}")


# --------------------------------------------------------------------------
# publishers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishResult:
    target: str
    release: str
    as_of: str
    probe: dict | None
    pruned: int

    def as_dict(self) -> dict:
        return {
            "target": self.target,
            "release": self.release,
            "as_of": self.as_of,
            "probe": self.probe,
            "pruned": self.pruned,
        }


class LocalPublisher:
    """Staging + atomic symlink flip. The reference publisher.

    Layout::

        {target}/releases/{as_of}-{epoch}/   full copy of the bundle
        {target}/current                     symlink -> one release

    A crash anywhere before the flip leaves ``current`` untouched; after the
    flip the old release is still on disk until pruning. ``copy_hook`` is a
    test seam: it runs after each staged file and may raise to simulate a
    kill mid-upload.
    """

    def __init__(self, target: Path | str, *, copy_hook: Callable[[Path], None] | None = None):
        self.target = paths.assert_writable(Path(target))
        self.copy_hook = copy_hook

    def _read_as_of(self, bundle: Path) -> str:
        import json

        meta = json.loads((bundle / "data" / "meta.json").read_text())
        return str(meta.get("as_of", "unknown"))

    def publish(self, bundle: Path | str) -> PublishResult:
        bundle = Path(bundle)
        _validate_bundle(bundle)
        hits = secret_scan(bundle)
        if hits:
            files = sorted({h["file"] for h in hits})
            raise PublishError(
                f"secret scan found {len(hits)} hit(s) in {len(files)} file(s), "
                f"e.g. {files[:3]} — the bundle never ships until it is clean"
            )

        as_of = self._read_as_of(bundle)
        releases = self.target / "releases"
        releases.mkdir(parents=True, exist_ok=True)
        release = releases / f"{as_of}-{int(time.time())}"

        # Stage completely BEFORE touching `current`. Dotfiles (the bundle's
        # private state) never ship.
        for src in sorted(bundle.rglob("*")):
            if not src.is_file():
                continue
            relative = src.relative_to(bundle)
            if any(part.startswith(".") for part in relative.parts):
                continue
            dst = release / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            if self.copy_hook is not None:
                self.copy_hook(dst)
        _validate_bundle(release)

        current = self.target / "current"
        tmp_link = self.target / f".current-tmp-{os.getpid()}"
        if tmp_link.is_symlink() or tmp_link.exists():
            tmp_link.unlink()
        # Relative so a published tree stays valid when its parent moves.
        os.symlink(str(Path("releases") / release.name), tmp_link)
        os.replace(tmp_link, current)  # the atomic flip

        pruned = 0
        kept = sorted(
            (p for p in releases.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        for stale in kept[:-KEEP_RELEASES]:
            shutil.rmtree(stale, ignore_errors=True)
            pruned += 1
        return PublishResult(
            target=str(self.target), release=str(release), as_of=as_of,
            probe=None, pruned=pruned,
        )


class CommandPublisher:
    """Ship the bundle with an external command (wrangler / rclone / aws…).

    The command template receives the bundle path; a typical value from
    ``.env``::

        DASHBOARD_PUBLISH_CMD="wrangler pages deploy {bundle} --project-name=earnings-board"

    When ``probe_url`` is set, an unauthenticated 200 on ``meta.json`` REFUSES
    the publish (the hard rule): Access must be in front of the target before
    anything is pushed.
    """

    def __init__(self, command: str, *, probe_url: str | None = None):
        if "{bundle}" not in command:
            raise PublishError("command template must contain {bundle}")
        self.command = command
        self.probe_url = probe_url

    def publish(self, bundle: Path | str) -> PublishResult:
        bundle = Path(bundle)
        _validate_bundle(bundle)
        hits = secret_scan(bundle)
        if hits:
            raise PublishError(f"secret scan found {len(hits)} hit(s) — refusing to ship")

        probe = None
        if self.probe_url:
            probe = access_probe(self.probe_url)
            if probe["public"]:
                raise PublishError(
                    f"access probe got an unauthenticated 200 from {self.probe_url} — "
                    "the target is publicly readable. Put Cloudflare Access in "
                    "front of it before publishing positions and licensed data."
                )

        command = self.command.format(bundle=str(bundle))
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            raise PublishError(
                f"publish command failed ({proc.returncode}): "
                f"{(proc.stderr or proc.stdout)[-500:]}"
            )
        import json

        meta = json.loads((bundle / "data" / "meta.json").read_text())
        return PublishResult(
            target=self.command, release=str(meta.get("as_of")), as_of=str(meta.get("as_of")),
            probe=probe, pruned=0,
        )


def publish_bundle(
    bundle: Path | str, target: Path | str | None = None, *, probe_url: str | None = None
) -> PublishResult:
    """Publish ``bundle`` to ``target`` (default ``dashboard/published``).

    Named ``publish_bundle`` rather than ``publish`` on purpose: a package-level
    export called ``publish`` shadows this very module, so
    ``from engine.dashboard import publish`` would hand a caller the function
    where it asked for the module — and ``publish.PublishError`` would then
    fail at the moment the publish step needed to catch an error.

    Directory targets use :class:`LocalPublisher`; a string containing shell
    syntax or ``{bundle}`` is treated as a :class:`CommandPublisher` command.
    """
    bundle = Path(bundle)
    if target is None:
        target = paths.ROOT / "dashboard" / "published"
    target_str = str(target)
    if "{" in target_str or "$" in target_str or target_str.startswith(("wrangler ", "rclone ", "aws ")):
        return CommandPublisher(target_str, probe_url=probe_url).publish(bundle)
    return LocalPublisher(target_str).publish(bundle)
