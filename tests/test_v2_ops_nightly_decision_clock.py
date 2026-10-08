"""Planning/submission checks: no workers or host resource admission waits."""
import json
import sqlite3
from types import SimpleNamespace

import pytest

from engine.v2.data.legacy_nightly_read_plan import NIGHTLY_CAPTURE_IMPLEMENTATION_REF
from engine.v2.foundation import SystemClock
from engine.v2.ops import cli
from tests.ops_support import sample

SESSION = "2026-09-12"


@pytest.mark.parametrize("host_cpu_count", [4, 12])
def test_plan_nightly_pins_decision_clock_and_resubmission_reuses_it(
        tmp_path, capsys, monkeypatch, host_cpu_count):
    """Clock identity is independent of the host running this submission-only test."""
    monkeypatch.setattr("os.cpu_count", lambda: host_cpu_count)
    capacity = sample(SystemClock())
    monkeypatch.setattr(cli, "sample_capacity", lambda *args, **kwargs: capacity)
    # Keep DEFAULT_POLICY and its admission checks; supply a fitting fake host
    # at both discovery boundaries, without changing other modules' CPU view.
    monkeypatch.setattr(cli, "os", SimpleNamespace(cpu_count=lambda: len(capacity.allowed_cpu_ids)))
    root = tmp_path / "ops"
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
