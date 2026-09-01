"""Load ``.env`` into the process environment, once, at engine import.

Credentials live in ``.env`` and every adapter reads them from
``os.environ`` — but nothing ever put them there. The gap was invisible for as
long as pulls were launched from a shell that happened to have sourced the file,
and it surfaced the first time the nightly ran on its own: it stopped at the
refresh step with ``ORATS_API_KEY is unset``. The documented cron entry
(``cd investing-plan && python3 -m engine.dashboard.nightly``) does not source
anything either, so the scheduled job would have failed on a missing credential
every night, and the only symptom would have been a line in a log nobody reads.

Two rules, both of which matter more than the convenience:

**The real environment always wins.** A variable already set is never
overwritten. CI, a container's own secrets, and ``ORATS_API_KEY=... python3 ...``
on the command line must all beat a file on disk — otherwise this module becomes
a way to silently run against the wrong credential.

**A missing or unreadable ``.env`` is not an error.** It is the normal state
wherever secrets arrive by another route. The adapters already raise a clear
``CredentialRotated`` when a key they need is absent; failing at import instead
would turn a specific, actionable message into a stack trace during startup.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["load_env", "loaded_keys", "parse_env"]

def parse_env(text: str) -> dict[str, str]:
    """Parse a shell-style ``.env`` body into a mapping.

    Handles ``export K=V``, ``K=V``, quoted values, comments and blank lines.
    Deliberately tolerant: a line it cannot parse is skipped rather than
    raising, because a hygiene check must never fail open on a formatting
    surprise in an unrelated line.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key:
            out[key] = val
    return out



#: Names loaded from ``.env`` in this process, for diagnostics. Names only —
#: this module must never make a secret's VALUE easier to reach than
#: ``os.environ`` already does.
loaded_keys: tuple[str, ...] = ()


def load_env(path: Path | None = None, *, override: bool = False) -> tuple[str, ...]:
    """Put ``.env`` values into ``os.environ`` and return the names that landed.

    Idempotent: with ``override`` false, a second call is a no-op because the
    first call's values are already present.
    """
    global loaded_keys
    from engine import paths

    env_path = Path(path) if path is not None else paths.ENV_FILE
    try:
        text = env_path.read_text()
    except (OSError, UnicodeDecodeError):
        return ()

    landed = []
    for key, value in parse_env(text).items():
        if override or key not in os.environ:
            os.environ[key] = value
            landed.append(key)
    loaded_keys = tuple(landed)
    return loaded_keys
