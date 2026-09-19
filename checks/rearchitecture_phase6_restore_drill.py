#!/usr/bin/env python3
"""P6-5 export-restore-drill acceptance: the drill actually catches a broken restore.

Builds one small, entirely synthetic deployment (a committed decision, one
``ScoreRequest``/``NativeScoreInputs`` pair backed up as artifacts, and one
exported generation) using only real production code
(``engine.v2.ledger.decisions.insert``, ``engine.v2.ops.backup.run_backup``,
``engine.v2.ledger.export.export_generation``, ``engine.v2.ops.cli.rescore_command``
-- the same functions ``tools/v2_restore_drill.py`` composes for an operator),
then runs the drill three times:

1. against the real backup -- must be ``PASS``: the restored deployment
   replays the SAME score byte-for-byte with no provider pulls and no
   fitting, and its ledger export matches the pre-backup export exactly;
2. with a wrong expected score hash -- must be ``FAIL`` (a real regression
   the drill is meant to catch, not silently accepted);
3. with a wrong expected decision count -- must be ``FAIL`` for the same
   reason on the ledger side.

A drill that cannot fail these negative controls would be a rubber stamp,
not a check. No network, no real provider call, no capture/corpus/mutmut --
every byte here is built in this process.

Usage::

    python3 checks/rearchitecture_phase6_restore_drill.py [--artifact-root evidence/]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts import ScoreRequest  # noqa: E402
from engine.v2.domain.generation import Pricing, generate, price  # noqa: E402
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash, to_document  # noqa: E402
from engine.v2.ledger.decisions import insert, set_authority  # noqa: E402
from engine.v2.ledger.export import export_generation  # noqa: E402
from engine.v2.ops.backup import prepare_backup, run_backup  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.ops.catalog import transaction  # noqa: E402
from engine.v2.ops.cli import rescore_command  # noqa: E402
from engine.v2.scoring.stages import NativeScoreInputs, STAGE_NAMES, StageReceipt  # noqa: E402
from tools.v2_restore_drill import run_drill  # noqa: E402

STAMP = "2026-09-19T00:00:00.000000Z"


def _receipts():
    return tuple(StageReceipt(stage, "declared-input", "declared-output")
                 for stage in STAGE_NAMES if stage != "diagnostics")


def _write_file(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    """Real deployment bytes: one committed decision, one score fixture, one
    export generation, one backup. Returns (backup_dir, original_export_dir,
    expected_score_hash)."""
    clock = SystemClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    store = ArtifactStore(tmp_path / "objects")

    request = ScoreRequest(
        event_id="evt-drill", calendar_revision="cal-1", strategy_version="STR-THRU",
        deployment_id="dep-1", decision_clock_id="entry-close",
        requested_decision_at="2026-09-19", snapshot_id="snap-1",
        mode="replay", fill_model={"alpha": 0.5},
    )
    context = {"ticker": "AAA", "strategy": "STR-THRU", "event_date": "2026-09-19",
              "entry_date": "2026-09-19", "exit_date": "2026-09-20", "expiry": "2026-09-21",
              "spot": 100.0, "strike": 95.0}
    features = {"strike": 100.0, "model_inputs": {"strike": 100.0}}
    forecast = {"models": {"driver_prediction": {"intercept": 7.0, "coefficients": {}}}}
    stale_geometry = generate("STR-THRU", {**context, "strike": 95.0})
    priced_legs = []
    for strike, bid, ask in ((95.0, 0.5, 1.5), (100.0, 1.0, 2.0), (105.0, 2.0, 4.0), (110.0, 3.0, 6.0)):
        geometry = generate("STR-THRU", {**context, "strike": strike})
        quotes = {(leg.right, leg.strike, leg.expiry): {"bid": bid, "ask": ask} for leg in geometry.legs}
        priced_legs.extend(price(geometry, quotes, 0.5).legs)
    pricing = Pricing("STR-THRU", 100.0, 0.0, tuple(priced_legs))
    inputs = NativeScoreInputs(
        context=context, features=features, forecast=forecast, geometry=stale_geometry,
        pricing=pricing, analogs={}, simulation={}, gate={}, chooser={}, diagnostics={},
        source_ref="typed-native-fixture", stage_receipts=_receipts(),
    )
    request_bytes = json.dumps(to_document(request)).encode()
    native_bytes = json.dumps(to_document(inputs)).encode()
    request_ref = store.publish_bytes(request_bytes, schema_ref="score_request.v1.0")
    native_ref = store.publish_bytes(native_bytes, schema_ref="native_score_inputs.v1.0")

    ground_truth = rescore_command(SimpleNamespace(
        request=_write_file(tmp_path / "gt_request.json", request_bytes),
        native_inputs=_write_file(tmp_path / "gt_native.json", native_bytes)))
    expected_hash = content_hash(to_document(ground_truth))

    with transaction(conn):
        set_authority(conn, None, "drill", STAMP)
        insert(conn, logical_key="k1", decision_id="prediction:p1",
              payload={"row_id": "p1", "event_id": "evt-drill", "as_of": "2026-09-19"},
              purpose="shadow", kind="prediction", validations={}, created_at=STAMP, owner="drill")

    original_export_dir = export_generation(conn, tmp_path / "live-export", generation="g1")

    prepare_backup(conn, "drill-1",
                   {"request.json": request_ref, "native_inputs.json": native_ref}, clock=clock)
    backup_dir = tmp_path / "backup"
    run_backup(conn, key="drill-1", owner="operator", target=backup_dir, clock=clock, store=store)
    conn.close()
    return backup_dir, original_export_dir, expected_hash


def run_acceptance(artifact_root: Path | None = None) -> dict:
    with tempfile.TemporaryDirectory() as scratch:
        tmp_path = Path(scratch)
        backup_dir, original_export_dir, expected_hash = _build_fixture(tmp_path)

        real = run_drill(
            backup_dir=backup_dir, restore_root=tmp_path / "restored",
            score_request_name="request.json", native_inputs_name="native_inputs.json",
            expected_score_hash=expected_hash, original_export_dir=original_export_dir,
            generation="g1", expected_decisions_count=1,
            artifact_root=artifact_root)

        wrong_hash = run_drill(
            backup_dir=backup_dir, restore_root=tmp_path / "restored-wrong-hash",
            score_request_name="request.json", native_inputs_name="native_inputs.json",
            expected_score_hash="sha256:" + "0" * 64, original_export_dir=original_export_dir,
            generation="g1", expected_decisions_count=1)

        wrong_count = run_drill(
            backup_dir=backup_dir, restore_root=tmp_path / "restored-wrong-count",
            score_request_name="request.json", native_inputs_name="native_inputs.json",
            expected_score_hash=expected_hash, original_export_dir=original_export_dir,
            generation="g1", expected_decisions_count=99)

    findings = []
    if real["verdict"] != "PASS":
        findings.append({"code": "DRILL_DID_NOT_PASS_ON_A_GOOD_RESTORE", "detail": json.dumps(real)})
    if wrong_hash["verdict"] != "FAIL":
        findings.append({"code": "NEGATIVE_CONTROL_NOT_TRIGGERED",
                         "detail": "a wrong expected score hash did not fail the drill"})
    if wrong_count["verdict"] != "FAIL":
        findings.append({"code": "NEGATIVE_CONTROL_NOT_TRIGGERED",
                         "detail": "a wrong expected decision count did not fail the drill"})
    return {"real": real, "findings": findings}


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    result = run_acceptance(args.artifact_root)
    for finding in result["findings"]:
        print(f"{finding['code']}: {finding['detail']}")
    verdict = "FAIL" if result["findings"] else "PASS"
    print(f"restore_drill_acceptance: {verdict} ({len(result['findings'])} finding(s))")
    return 1 if result["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
