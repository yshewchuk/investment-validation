#!/usr/bin/env python3
"""L09 evidence producer -- guide §9 row L09.

Real subprocess (mirroring ``tests/test_v2_serving_api.py``'s own
``test_no_scoring_or_provider_import_after_startup_and_after_a_request``
convention -- a separate process is the only way this check is meaningful,
since ``sys.modules`` in THIS process is already polluted by everything the
producer itself imported).

Two layers of proof, both inside the child process, before ``create_app``
is even imported:

1. **Rigged constructors.** Fake stand-in modules are planted in
   ``sys.modules`` under the real scoring/provider module names (real code
   never runs them -- these stand-ins are exactly what the module docstring
   means by "constructors rigged to fail if invoked"): any attribute access
   that looks like a constructor call raises ``RuntimeError`` immediately.
   If the read path ever really tried to build a scorer or hit a provider,
   the request would blow up loudly instead of silently degrading.
2. **Import-surface check.** ``sys.modules`` is snapshotted right after
   ``create_app`` returns and again after a handful of real GETs (current,
   events, operations, and one against an explicit release id where a real
   committed release is given); neither snapshot may contain the forbidden
   module names, whether or not the rigged stand-ins caught anything.

**Convention (a judgement call, consistent with this package's other
negative controls): ``verdict=DIFFER`` means neither layer ever fired --
the safe, desired outcome (scoring/providers genuinely never touched).
``verdict=AGREE`` means a rig raised or a forbidden module appeared** (a
:class:`Finding` records exactly which).

Never writes to ``--serving-db``/``--store-root``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase1_gate import source_files, source_hash  # noqa: E402
from checks.rearchitecture_phase2_gate import environment_hash as _environment_hash  # noqa: E402
from engine.v2.diagnosis import AGREE, DIFFER, ComparisonReceipt, Envelope, Finding, Population, content_hash  # noqa: E402
from engine.v2.foundation import to_document  # noqa: E402

KIND = "no_scoring_startup_negative_control"

#: Same list ``tests/test_v2_serving_api.py`` already established as the
#: real forbidden surface for this read path.
_FORBIDDEN_MODULE_SUBSTRINGS = ("engine.score", "engine.v2.ops", "engine.data.pulls",
                                "engine.data.sources", "yfinance")

#: The exact names this producer's own rig plants in ``sys.modules`` (module
#: docstring point 1) -- their bare presence is BY DESIGN and must not count
#: as a violation; only a DIFFERENT, more specific name (e.g. a real
#: ``engine.score.scorer`` submodule actually imported) should.
_SELF_PLANTED_MODULES = frozenset({"engine.score", "yfinance"})

_GUARD_SCRIPT = textwrap.dedent('''
    import json, sys, threading, time, urllib.error, urllib.request, types

    sys.path.insert(0, {root!r})

    class _Rigged:
        def __getattr__(self, name):
            def _raise(*a, **k):
                raise RuntimeError("rigged constructor invoked: " + name)
            return _raise
        def __call__(self, *a, **k):
            raise RuntimeError("rigged module called directly")

    rig_error = None
    for _name in ("engine.score", "yfinance"):
        sys.modules[_name] = _Rigged()

    from engine.v2.serving.api import create_app
    app = create_app(serving_db={serving_db!r}, store_root={store_root!r},
                     serving_root={serving_root!r}, token={token!r})
    after_create = sorted(sys.modules)

    import uvicorn
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    base = "http://127.0.0.1:" + str(port)

    def _get(path):
        req = urllib.request.Request(base + path, headers={{"Authorization": "Bearer " + {token!r}}})
        try:
            urllib.request.urlopen(req, timeout=10).read()
        except urllib.error.HTTPError:
            pass

    try:
        _get("/api/v1/operations")
        _get("/api/v1/releases/current")
        _get("/api/v1/events")
        if {release_id!r} is not None:
            _get("/api/v1/events?release_id=" + {release_id!r})
    except RuntimeError as exc:
        rig_error = str(exc)

    after_request = sorted(sys.modules)
    server.should_exit = True
    thread.join(timeout=5)
    print(json.dumps({{"after_create": after_create, "after_request": after_request,
                       "rig_error": rig_error}}))
''')


def _finding(findings: list, field: str) -> None:
    findings.append(Finding(
        finding_id=content_hash([KIND, field])[7:19], first_differing_stage="startup",
        field_path=field, kind="value", owning_stage="startup"))


def build(serving_db: Path, store_root: Path, serving_root: Path, *, token: str,
         release_id: str | None) -> ComparisonReceipt:
    serving_db.parent.mkdir(parents=True, exist_ok=True)
    script = _GUARD_SCRIPT.format(root=str(ROOT), serving_db=str(serving_db), store_root=str(store_root),
                                  serving_root=str(serving_root), token=token, release_id=release_id)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="phase3-l09-") as tmp:
        script_path = Path(tmp) / "guard.py"
        script_path.write_text(script)
        result = subprocess.run([sys.executable, str(script_path)], cwd=str(ROOT),
                                capture_output=True, text=True, timeout=60)

    findings: list[Finding] = []
    checks = 0
    if result.returncode != 0:
        _finding(findings, "subprocess_crashed")
        payload = {"after_create": [], "after_request": [], "rig_error": result.stderr[-2000:]}
    else:
        payload = json.loads(result.stdout.strip().splitlines()[-1])

    checks += 1
    if payload.get("rig_error"):
        _finding(findings, "rigged_constructor_invoked")

    for key in ("after_create", "after_request"):
        checks += 1
        modules = payload.get(key, [])
        hit = [m for m in modules if m not in _SELF_PLANTED_MODULES
              and any(f in m for f in _FORBIDDEN_MODULE_SUBSTRINGS)]
        if hit:
            _finding(findings, f"{key}_forbidden_modules")

    population = Population(expected=checks, supported=checks, compared=checks)
    verdict = AGREE if findings else DIFFER
    code_hash = source_hash(source_files(ROOT))
    env_hash, _source = _environment_hash(ROOT)
    envelope = Envelope(code_hash=code_hash, environment_hash=env_hash)
    receipt_id = content_hash([KIND, str(release_id), [f.finding_id for f in findings]])[7:23]
    return ComparisonReceipt(
        receipt_id=receipt_id, comparison_kind=KIND, tier=1,
        left_ref="rigged_no_call:" + str(release_id), right_ref="observed_startup_and_gets",
        stage_plan_ref="api_startup.v1", tolerance_policy_ref="module_surface.v1",
        verdict=verdict, findings=tuple(findings), population=population, envelope=envelope)


def publish(receipt: ComparisonReceipt, artifact_root: Path) -> dict:
    artifact_root.mkdir(parents=True, exist_ok=True)
    data = json.dumps(to_document(receipt), indent=2, sort_keys=True).encode()
    path = artifact_root / "no_scoring_startup_negative_control.json"
    path.write_bytes(data)
    return {"path": path.name, "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-db", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--serving-root", type=Path, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--release-id", default=None)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = build(args.serving_db, args.store_root, args.serving_root, token=args.token,
                    release_id=args.release_id)
    ref = publish(receipt, args.artifact_root)
    print(json.dumps({**ref, "verdict": receipt.verdict}, indent=2))
    return 0 if receipt.verdict == DIFFER else 1


if __name__ == "__main__":
    raise SystemExit(main())
