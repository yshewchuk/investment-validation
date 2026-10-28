"""Planning/submission checks: no workers or host resource admission waits."""
import json
import os
import sqlite3
from types import SimpleNamespace

import pytest

from engine.v2.contracts.data import SnapshotRef
from engine.v2.data.legacy_nightly_read_plan import NIGHTLY_CAPTURE_IMPLEMENTATION_REF
from engine.v2.foundation import SystemClock, to_document
from engine.v2.ops import cli
from engine.v2.ops.workflows import commands
from tests.ops_support import sample

SESSION = "2026-09-12"


def _stub_shadow_snapshot_head(monkeypatch):
    def resolve(conn, store, scope, *, clock):
        assert scope == "shadow"
        document = to_document(SnapshotRef(
            snapshot_id="test-shadow-snapshot",
            manifest_hash="a" * 64,
            table_versions={},
            calendar_version="calendar-v1",
            source_priority_version="source-priority-v1",
            finality_receipt_refs=(),
            knowledge_mode_by_table={},
        ))
        return store.publish_bytes(
            json.dumps(document).encode(),
            schema_ref="snapshot_ref.v1.0")
    monkeypatch.setattr("engine.v2.ops.snapshot_planning.resolve_snapshot_head", resolve)


@pytest.mark.parametrize("host_cpu_count", [8, 12])
def test_plan_nightly_pins_decision_clock_and_resubmission_reuses_it(
        tmp_path, capsys, monkeypatch, host_cpu_count):
    """Clock identity is independent of the host running this submission-only test."""
    # isolated os seen only by commands; parametrization stays effective
    monkeypatch.setattr(commands, "os", SimpleNamespace(**{**vars(os), "cpu_count": lambda: host_cpu_count}))
    capacity = sample(SystemClock())
    monkeypatch.setattr(commands, "sample_capacity", lambda *args, **kwargs: capacity)
    root = tmp_path / "ops"
    _stub_shadow_snapshot_head(monkeypatch)
    population_file = tmp_path / "population.json"
    population_file.write_text(json.dumps(["FAKE|TWIN-P|" + SESSION]))
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({
        "manifest_id": "m1",
        "file_refs": [{"path": "data/curated/daily_market/year=2024/part-0000.parquet",
                      "content_hash": "sha256:" + "0" * 64, "byte_size": 1},
                     {"path": "data/curated/option_chains/year=2024/part-0000.parquet",
                      "content_hash": "sha256:" + "0" * 64, "byte_size": 1},
                     {"path": "data/curated/earnings_events/year=2024/part-0000.parquet",
                      "content_hash": "sha256:" + "0" * 64, "byte_size": 1},
                     {"path": "data/curated/trades/year=2024/part-0000.parquet",
                      "content_hash": "sha256:" + "0" * 64, "byte_size": 1},
                     {"path": "data/raw/fetch/orats/ab/placeholder.meta.json",
                      "content_hash": "sha256:" + "0" * 64, "byte_size": 1},
                     {"path": "earnings_predictions/data/raw/oquants/moves/moves_AAA.json",
                      "content_hash": "sha256:" + "0" * 64, "byte_size": 1}],
        "registry_and_model_refs": ["placeholder::sha256:" + "0" * 64],
        "calendar_ref": "placeholder::sha256:" + "0" * 64,
        "capture_implementation_ref": NIGHTLY_CAPTURE_IMPLEMENTATION_REF}))
    assert cli.main(["--root", str(root), "init"]) == 0
    capsys.readouterr()
    assert cli.main(["--root", str(root), "plan", "nightly", "--as-of", SESSION,
                     "--tickers", "FAKE", "--input-manifest", str(manifest_file),
                     "--expected-population", str(population_file)]) == 0
    plan_doc = json.loads(capsys.readouterr().out)
    assert plan_doc["plan"]["snapshot_id"] == "test-shadow-snapshot"
    decision_clock = plan_doc["plan"]["decision_clock"]
    assert decision_clock
    plan_ref = plan_doc["plan_ref"]

    assert cli.main(["--root", str(root), "submit", "--plan", plan_ref,
                     "--idempotency-key", "s1"]) == 0
    capsys.readouterr()
    # A second submission of the SAME plan artifact is the "retry" case: the
    # decision_evidence job it names is unchanged (job ids are keyed off the
    # plan's own session/scope, not the CLI's --idempotency-key), so its
    # decision_clock parameter is read back off the ORIGINAL submission.
    assert cli.main(["--root", str(root), "submit", "--plan", plan_ref,
                     "--idempotency-key", "s2"]) == 0
    capsys.readouterr()

    raw = sqlite3.connect(root / "catalog.sqlite")
    try:
        rows = raw.execute("SELECT spec_json FROM jobs WHERE kind='decision_evidence'").fetchall()
        assert len(rows) == 1
        assert json.loads(rows[0][0])["parameters"]["decision_clock"] == decision_clock
    finally:
        raw.close()
