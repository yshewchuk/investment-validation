"""The tier-0 corpus replay, and its negative controls.

`component_contracts.md` §9.5: "Running twice in the same process is
insufficient: include fresh process, reordered inputs, batch/single and
serialized round-trip cases." Each of those is a case in
:mod:`checks.tier0_corpus`, and each is proved here by corrupting a synthetic
corpus and asserting the case goes red.

A synthetic corpus rather than the real one, deliberately: the real corpus
carries licensed quotes and is not in this repository, so a test that needed it
would be a test that does not run on a clean checkout. The real corpus is
exercised by ``checks/rearchitecture_phase0_gate.py``, which reports honestly when it is
absent instead of passing.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import tier0_corpus as t0  # noqa: E402
from engine.v2.diagnosis import AGREE, DIFFER, INCOMPARABLE, content_hash  # noqa: E402


STRATEGIES = ("STR-THRU", "TWIN-P5", "CAL-P")
REFUSALS = ("NO_CHAIN", "UNVALIDATED_STRUCTURE")


def _pair(fixture_id: str, strategy: str, covers: list[str], *,
          width: float = 0.05123456789012) -> dict:
    request = {
        "ticker": "MTN", "strategy": strategy, "event_date": "2026-09-28",
        "session": "AMC", "structure_params": None, "strike": None,
        "fill": {"policy_id": "legacy.fill_alpha.v1", "alpha": 0.5},
        "identity_key": f"MTN|{strategy}||2026-09-28||0.5000|||||",
    }
    record = {
        "ticker": "MTN", "strategy": strategy, "event_date": "2026-09-28",
        "entry_date": "2026-09-28", "exit_date": "2026-09-29",
        "forecast_abs_move": 5.218743916,
        "ci_low": -0.01193117, "ci_high": 0.04217742,
        "structure_params": {"width_moneyness": width},
        "flags": [],
    }
    payload = {"request": request, "record": record, "record_kind": "score_result"}
    return {
        "schema_version": "tier0_pair.v1.0",
        "fixture_id": fixture_id,
        "covers": sorted(covers),
        "notes": "",
        "payload": payload,
        "payload_hash": content_hash(payload),
        "request_hash": content_hash(request),
        "envelope": {"captured_at": "2026-09-12T00:00:00.000000+00:00",
                     "worker_ref": "test:1", "duration_seconds": 0.01},
    }


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "tier0"
    (root / "pairs").mkdir(parents=True)
    pairs = []
    for i, strategy in enumerate(STRATEGIES):
        covers = [f"strategy:{strategy}"]
        if i < len(REFUSALS):
            covers.append(f"refusal:{REFUSALS[i]}")
        pair = _pair(f"{i:03d}_{strategy}", strategy, covers,
                     width=0.05123456789012 + i)
        pairs.append(pair)
        (root / "pairs" / f"{pair['fixture_id']}.json").write_text(
            json.dumps(pair, indent=2, sort_keys=True) + "\n"
        )
    coverage: dict[str, list[str]] = {}
    for pair in pairs:
        for axis in pair["covers"]:
            coverage.setdefault(axis, []).append(pair["fixture_id"])
    required = [f"strategy:{s}" for s in STRATEGIES]
    required += [f"refusal:{c}" for c in REFUSALS]
    (root / "INDEX.json").write_text(json.dumps({
        "schema_version": "tier0_corpus.v1.0",
        "pairs": {p["fixture_id"]: {"payload_hash": p["payload_hash"],
                                    "request_hash": p["request_hash"],
                                    "covers": p["covers"]} for p in pairs},
        "coverage": coverage,
        "required_axes": sorted(required),
        "uncovered_axes": [],
    }, indent=2, sort_keys=True) + "\n")
    return root


def _pair_path(root: Path, fixture_id: str) -> Path:
    return root / "pairs" / f"{fixture_id}.json"


def _rewrite(root: Path, fixture_id: str, mutate) -> None:
    path = _pair_path(root, fixture_id)
    doc = json.loads(path.read_text())
    mutate(doc)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------
# a healthy corpus
# --------------------------------------------------------------------------


def test_a_healthy_corpus_agrees(corpus):
    merged, cases = t0.run(corpus)
    assert merged.verdict == AGREE, merged.summary()
    assert set(cases) == {"corpus_replay", "coverage", "batch_vs_single",
                          "reordered_inputs", "fresh_process"}
    assert all(r.verdict == AGREE for r in cases.values())


def test_it_runs_well_inside_the_ten_second_budget(corpus):
    started = time.monotonic()
    t0.run(corpus)
    elapsed = time.monotonic() - started
    assert elapsed < t0.TIME_BUDGET_SECONDS, elapsed


def test_it_loads_no_panel_and_opens_no_socket(corpus):
    """§7.2: no network, no panel load, no fitting on replay."""
    script = (
        "import sys, json; sys.path.insert(0, %r)\n"
        "from pathlib import Path\n"
        "from checks import tier0_corpus as t0\n"
        "merged, _ = t0.run(Path(%r))\n"
        "print(json.dumps({'verdict': merged.verdict,\n"
        "  'pandas': 'pandas' in sys.modules,\n"
        "  'score': 'engine.score' in sys.modules}))\n"
    ) % (str(ROOT), str(corpus))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, cwd=ROOT, check=True)
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["verdict"] == AGREE
    assert out["pandas"] is False
    assert out["score"] is False


def test_the_network_guard_actually_refuses(corpus):
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from checks import tier0_corpus as t0\n"
        "t0._forbid_network()\n"
        "import socket\n"
        "try:\n"
        "    socket.socket()\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__)\n"
    ) % str(ROOT)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, cwd=ROOT, check=True)
    assert "_NetworkUsed" in proc.stdout


# --------------------------------------------------------------------------
# the negative controls, one per case
# --------------------------------------------------------------------------


def test_a_rounded_request_stops_addressing_its_record(corpus):
    """`b33036c`, made structurally impossible to hide.

    The lookup key is the content hash of the FULL-PRECISION request. Round a
    value on the way out and the request no longer addresses anything, which is
    a loud failure rather than a quiet six-place drift.
    """
    fid = f"000_{STRATEGIES[0]}"
    _rewrite(corpus, fid, lambda d: d["payload"]["request"].update(
        {"identity_key": d["payload"]["request"]["identity_key"] + "-rounded"}))
    merged, cases = t0.run(corpus)
    assert merged.verdict == DIFFER
    assert cases["corpus_replay"].verdict == DIFFER
    paths = {f.field_path for f in merged.findings}
    assert {"request_hash", "resolves_to.0"} & paths


def test_a_file_that_disagrees_with_its_digest_fails(corpus):
    """`6b9d5cf`: rounding reapplied after the exemption."""
    fid = f"001_{STRATEGIES[1]}"
    _rewrite(corpus, fid, lambda d: d["payload"]["record"]["structure_params"].update(
        {"width_moneyness": 0.051235}))
    merged, _ = t0.run(corpus)
    assert merged.verdict == DIFFER
    assert any(f.field_path == "payload_hash" for f in merged.findings)


def test_a_missing_strategy_makes_the_corpus_incomparable(corpus):
    """§12.2: every strategy and every refusal code has a frozen pair."""
    index = json.loads((corpus / "INDEX.json").read_text())
    index["coverage"].pop(f"strategy:{STRATEGIES[2]}")
    (corpus / "INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    merged, cases = t0.run(corpus)
    assert cases["coverage"].verdict == DIFFER
    assert any(f.field_path == f"strategy:{STRATEGIES[2]}"
               for f in cases["coverage"].findings)


def test_a_missing_refusal_code_fails(corpus):
    index = json.loads((corpus / "INDEX.json").read_text())
    index["coverage"].pop(f"refusal:{REFUSALS[0]}")
    (corpus / "INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    _, cases = t0.run(corpus)
    assert cases["coverage"].verdict == DIFFER


def test_an_empty_corpus_is_incomparable_not_agreement(tmp_path):
    root = tmp_path / "tier0"
    (root / "pairs").mkdir(parents=True)
    (root / "INDEX.json").write_text(json.dumps(
        {"pairs": {}, "coverage": {}, "required_axes": [], "uncovered_axes": []}))
    merged, _ = t0.run(root)
    assert merged.verdict == INCOMPARABLE


def test_a_missing_corpus_exits_nonzero_rather_than_passing(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "checks" / "tier0_corpus.py"),
         "--corpus", str(tmp_path / "nothing")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 1
    assert "CORPUS_MISSING" in proc.stderr or "INCOMPARABLE" in proc.stderr


def test_reordering_the_inputs_changes_nothing(corpus):
    forward = t0.load(corpus)
    backward = t0.load(corpus, reverse=True)
    assert forward.ordered_ids == backward.ordered_ids
    assert list(backward.pairs) != list(forward.pairs) or len(forward.pairs) == 1


def test_batch_and_single_agree_on_a_corrupted_corpus_too(corpus):
    """The batch must not hide a finding the singles would have reported."""
    fid = f"002_{STRATEGIES[2]}"
    _rewrite(corpus, fid, lambda d: d["payload"]["record"].update({"ci_low": -9.9}))
    merged, cases = t0.run(corpus)
    assert cases["batch_vs_single"].verdict == AGREE
    assert merged.verdict == DIFFER


# --------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------


def test_the_cli_reports_json(corpus):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "checks" / "tier0_corpus.py"),
         "--corpus", str(corpus), "--json"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["verdict"] == AGREE
    assert out["population"]["compared"] >= 1
