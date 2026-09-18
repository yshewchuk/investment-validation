"""P5-4 frozen residual pools: type, builder, loader, release pinning, rebuild.

Parity is judged against legacy's own construction, not a restatement of it:
the paired pool against ``engine/score.py::Scorer._residual_pool``'s three
merges and ``engine.pnl_sim.ResidualPool`` (reproduced here in pandas on the
same universe frames, because the real method reads the real panel), and the
driver pool against ``native_payoff.driver_residual_pool`` (itself pinned to
legacy ``registry.bucket_residuals`` in test_v2_scoring_native_payoff.py).
"""
from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import pandas as pd
import pytest

from engine import pnl_sim
from engine.v2.contracts.data import DatasetVersionRef, TableContractRef, TimeInterval
from engine.v2.contracts.incremental import ChangeSet, RowChange
from engine.v2.models.adapters import RuntimeFitForbidden
from engine.v2.models.contracts import ArtifactMember, ModelBinding, ModelRelease
from engine.v2.models.deployment import StagingRefused, resolve_release, stage_release
from engine.v2.models.frozen_release import inventory_member, release_member
from engine.v2.models.frozen_state import (
    FrozenStateError,
    FrozenStateLoader,
    FrozenStateRef,
    serialize_frozen_state,
)
from engine.v2.models.releases import (
    ArtifactInventoryMember,
    ModelArtifactInventory,
    ModelReleaseInventory,
    ReleaseBinding,
    ReleaseRequirement,
)
from engine.v2.models.lineage import (
    DataDependency,
    Lineage,
    propagate_corrections,
    rebuild_order,
    state_node,
)
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.models.residual_artifact import (
    PairedResidualPoolArtifact,
    ResidualArtifactError,
    make_paired_residual_pool_artifact,
    paired_residual_pool_key,
)
from engine.v2.models.training.residuals import (
    build_driver_residual_pool_artifact,
    build_paired_residual_pool_artifact,
)
from engine.v2.scoring import native_payoff

TICKERS = ("AAA", "BBB", "CCC", "DDD", "EEE")
CUTOFF = "2024-06-01"
LINEAGE = Lineage(data=(
    DataDependency(table="tier4.forecasts", end_exclusive=CUTOFF),
    DataDependency(table="tier3.panel", end_exclusive=CUTOFF),
    DataDependency(table="tier2.crush", end_exclusive=CUTOFF),
), upstream=("fold:size:2024-05-01", "fold:iv_crush:2024-05-01"))


def _universe(seed: int = 3, days: int = 200):
    """Forecast/outcome/crush rows for five tickers, one event per ticker-day
    offset so every event date is distinct (a total date order)."""
    rng = np.random.default_rng(seed)
    forecasts, outcomes, crush = [], [], []
    start = pd.Timestamp("2024-01-01")
    for index in range(days):
        ticker = TICKERS[index % len(TICKERS)]
        day = str((start + pd.Timedelta(days=index)).date())
        forecasts.append({"ticker": ticker, "event_date": day,
                          "pred_abs_move": float(rng.uniform(2, 9)),
                          "pred_iv_crush_30": float(rng.uniform(-40, -5))})
        outcomes.append({"ticker": ticker, "event_date": day,
                         "abs_move": float(rng.uniform(0, 14))})
        crush.append({"ticker": ticker, "event_date": day,
                      "crush_pct_iv30": float(rng.uniform(-60, 5))})
    # Rows the join must drop: an outcome with no forecast, a NaN forecast.
    outcomes.append({"ticker": "ZZZ", "event_date": "2024-02-02", "abs_move": 3.0})
    forecasts.append({"ticker": "YYY", "event_date": "2024-02-03",
                      "pred_abs_move": float("nan"), "pred_iv_crush_30": -10.0})
    outcomes.append({"ticker": "YYY", "event_date": "2024-02-03", "abs_move": 3.0})
    crush.append({"ticker": "YYY", "event_date": "2024-02-03", "crush_pct_iv30": -9.0})
    return forecasts, outcomes, crush


def _build(forecasts, outcomes, crush, cutoff=CUTOFF, lineage=LINEAGE):
    return build_paired_residual_pool_artifact(
        forecasts, outcomes, crush, move_model_id="size_v1_4",
        crush_model_id="iv_crush_v1_gbm", cutoff=cutoff, lineage=lineage,
    )


def _legacy_history(forecasts, outcomes, crush) -> pd.DataFrame:
    """engine/score.py Scorer._residual_pool, lines verbatim over frames."""
    panel = pd.DataFrame(outcomes)[["ticker", "event_date", "abs_move"]]
    frame = pd.DataFrame(forecasts)[["ticker", "event_date", "pred_abs_move",
                                     "pred_iv_crush_30"]]
    realized = pd.DataFrame(crush)[["ticker", "event_date", "crush_pct_iv30"]]
    for part in (panel, frame, realized):
        part["event_date"] = pd.to_datetime(part["event_date"])
    h = frame.merge(panel, on=["ticker", "event_date"], how="inner")
    h = h.merge(realized, on=["ticker", "event_date"], how="inner")
    h["err_move"] = h["abs_move"] - h["pred_abs_move"]
    h["err_crush"] = h["crush_pct_iv30"] - h["pred_iv_crush_30"]
    return h.dropna(subset=["err_move", "err_crush"])


# ---------------------------------------------------------------------------
# paired pool: parity with legacy's construction
# ---------------------------------------------------------------------------


def test_paired_pool_matches_legacy_residual_pool_construction():
    forecasts, outcomes, crush = _universe()
    artifact = _build(forecasts, outcomes, crush, cutoff=None)
    legacy = pnl_sim.ResidualPool(_legacy_history(forecasts, outcomes, crush))

    assert len(artifact.rows) == len(legacy)
    assert np.array_equal(
        np.asarray([row[0] for row in artifact.rows], dtype="datetime64[D]"),
        legacy._dates.astype("datetime64[D]"),
    )
    assert np.array_equal([row[2] for row in artifact.rows], legacy._pred)
    assert np.array_equal([row[3] for row in artifact.rows], legacy._move)
    assert np.array_equal([row[4] for row in artifact.rows], legacy._crush)


def test_paired_pool_cutoff_excludes_events_on_or_after_it():
    forecasts, outcomes, crush = _universe()
    artifact = _build(forecasts, outcomes, crush)
    everything = _build(forecasts, outcomes, crush, cutoff=None)

    assert artifact.rows and all(row[0] < CUTOFF for row in artifact.rows)
    assert artifact.rows == tuple(row for row in everything.rows if row[0] < CUTOFF)
    assert artifact.key == paired_residual_pool_key("size_v1_4", "iv_crush_v1_gbm", CUTOFF)
    with pytest.raises(ResidualArtifactError):
        make_paired_residual_pool_artifact(
            move_model_id="m", crush_model_id="c", cutoff="2024-01-01",
            rows=[("2024-01-01", "AAA", 1.0, 0.0, 0.0)], lineage=LINEAGE)


# ---------------------------------------------------------------------------
# request context cannot change frozen state; equivalent rebuilds agree
# ---------------------------------------------------------------------------


def test_equivalent_rebuild_agrees_byte_for_byte_whatever_the_input_order():
    forecasts, outcomes, crush = _universe()
    first = serialize_frozen_state(_build(forecasts, outcomes, crush))
    rng = random.Random(11)
    for _ in range(3):
        shuffled = [list(part) for part in (forecasts, outcomes, crush)]
        for part in shuffled:
            rng.shuffle(part)
        assert serialize_frozen_state(_build(*shuffled)) == first


def test_the_builder_has_no_context_and_a_context_scoped_rebuild_is_a_different_state():
    """Legacy scoped the crush table to the Scorer's loaded tickers, so the
    pool moved with context (memory: residual-pool-context-dependence). The
    builder takes the universe explicitly; the subset a bounded scorer would
    have produced is visibly a different artifact, not the same key with a
    silently different population."""
    forecasts, outcomes, crush = _universe()
    universe = _build(forecasts, outcomes, crush)
    loaded = {"AAA", "BBB"}
    scoped = _build(forecasts, outcomes, [row for row in crush if row["ticker"] in loaded])

    assert scoped.key == universe.key
    assert scoped.content_hash != universe.content_hash
    assert len(scoped.rows) < len(universe.rows)


def test_frozen_state_is_immutable():
    artifact = _build(*_universe())
    with pytest.raises(AttributeError):
        artifact.rows = ()  # type: ignore[misc]
    assert isinstance(artifact.rows, tuple) and isinstance(artifact.rows[0], tuple)
    rebuilt = make_paired_residual_pool_artifact(
        move_model_id=artifact.move_model_id, crush_model_id=artifact.crush_model_id,
        cutoff=artifact.cutoff, rows=artifact.rows, lineage=artifact.lineage)
    assert rebuilt.content_hash == artifact.content_hash


def test_builders_refuse_under_the_no_fit_guard_and_without_lineage():
    forecasts, outcomes, crush = _universe()
    with no_fit_guard():
        with pytest.raises(RuntimeFitForbidden):
            _build(forecasts, outcomes, crush)
        with pytest.raises(RuntimeFitForbidden):
            build_driver_residual_pool_artifact(
                [{"prediction": 1.0, "residual": 0.0}], role="size",
                model_id="m", fold=None, lineage=LINEAGE)
    with pytest.raises(ValueError):
        _build(forecasts, outcomes, crush, lineage=Lineage())


# ---------------------------------------------------------------------------
# driver pool: lookup equals the per-request rebuild it replaces
# ---------------------------------------------------------------------------


def _driver_rows(n: int, seed: int = 5) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = [{"prediction": float(p), "residual": float(r)}
            for p, r in zip(rng.uniform(0, 10, n), rng.normal(0, 2, n))]
    rows.insert(7, {"prediction": float("nan"), "residual": 1.0})
    rows.insert(9, {"prediction": 1.0})
    return rows


@pytest.mark.parametrize("n", [40, 3000])  # flat only, and decile-bucketed
def test_driver_pool_lookup_matches_the_rows_path_bit_for_bit(n):
    from engine.v2.scoring.native_residuals import driver_pool_from_artifact

    rows = _driver_rows(n)
    artifact = build_driver_residual_pool_artifact(
        rows, role="size", model_id="size_v1_4", fold="2024-05-01", lineage=LINEAGE)
    assert (artifact.bucket_edges is None) == (n < 10 * native_payoff.MIN_POOL)
    block = {"model_residual_artifacts": {"driver": artifact},
             "model_residual_artifact_recipe": {"driver": {
                 "role": "size", "model_id": "size_v1_4", "fold": "2024-05-01"}}}
    for prediction in (-1.0, 0.3, 2.5, 5.0, 7.7, 9.99, 50.0):
        expected = native_payoff.driver_residual_pool(rows, prediction)
        actual, flag = driver_pool_from_artifact(block, "driver", prediction)
        assert flag is None
        assert np.array_equal(actual, expected)


def test_driver_pool_without_valid_rows_is_none():
    assert build_driver_residual_pool_artifact(
        [{"prediction": 1.0}], role="size", model_id="m", fold=None, lineage=LINEAGE) is None


# ---------------------------------------------------------------------------
# verified loader and release pinning
# ---------------------------------------------------------------------------


def _write(tmp_path, state, name="state.json") -> FrozenStateRef:
    (tmp_path / name).write_bytes(serialize_frozen_state(state))
    return FrozenStateRef(path=name, content_hash=state.content_hash)


def test_loader_round_trips_both_pool_kinds_including_infinite_bucket_edges(tmp_path):
    paired = _build(*_universe())
    driver = build_driver_residual_pool_artifact(
        _driver_rows(3000), role="size", model_id="m", fold="2024-05-01", lineage=LINEAGE)
    assert driver.bucket_edges[0] == -np.inf
    loader = FrozenStateLoader(tmp_path)
    for index, state in enumerate((paired, driver)):
        loaded = loader.load(_write(tmp_path, state, f"s{index}.json"))
        assert loaded == state
    assert loader.cache_size == 2


def test_loader_refuses_tampered_missing_and_escaping_files(tmp_path):
    artifact = _build(*_universe())
    ref = _write(tmp_path, artifact)
    raw = (tmp_path / "state.json").read_bytes()
    (tmp_path / "state.json").write_bytes(raw.replace(b'"AAA"', b'"AAB"', 1))
    loader = FrozenStateLoader(tmp_path)
    with pytest.raises(FrozenStateError, match="hash mismatch"):
        loader.load(ref)
    with pytest.raises(FrozenStateError, match="missing"):
        loader.load(FrozenStateRef(path="absent.json", content_hash=artifact.content_hash))
    with pytest.raises(FrozenStateError, match="escapes"):
        loader.load(FrozenStateRef(path="../x.json", content_hash=artifact.content_hash))


def _estimator_payload() -> bytes:
    return json.dumps({
        "schema_version": "linear_estimator.v1.0", "feature_order": ["x"],
        "outputs": [{"name": "prediction", "intercept": 1.0, "coefficients": [2.0]}],
    }, sort_keys=True).encode()


def _pinned_release(release_id: str, states: dict):
    """A size binding whose release pins an estimator plus frozen states,
    each member named for the kind the completeness inventory requires."""
    estimator = _estimator_payload()
    estimator_hash = "sha256:" + hashlib.sha256(estimator).hexdigest()
    members = (ArtifactMember(name="estimator", path="", content_hash=estimator_hash),) + tuple(
        release_member(state, kind) for kind, state in states.items())
    binding = ModelBinding(
        binding_id="size", model_id="size_v1_4", role="size", strategy_id="*",
        decision_clock_id="entry-close", adapter="json-linear.v1",
        feature_order=("x",), output_names=("prediction",), members=members,
    )
    inventory_members = (ArtifactInventoryMember(
        member_id="size:estimator", kind="estimator", artifact_ref="artifact://size",
        content_hash=estimator_hash),) + tuple(
        inventory_member(state, f"size:{kind}", f"artifact://size/{kind}")
        for kind, state in states.items())
    inventory = ModelReleaseInventory(
        release_id=release_id, deployment_id="d1", known_clock_ids=("entry-close",),
        artifacts=(ModelArtifactInventory(
            artifact_id="size_v1_4", role="size", strategy_ids=("*",),
            compatible_clock_ids=("entry-close",), target_contract_ref="abs_move.v1",
            ordered_features=("x",), members=inventory_members),),
        bindings=(ReleaseBinding(
            role="size", strategy_id="*", clock_id="entry-close", artifact_id="size_v1_4",
            ordered_features=("x",),
            required_member_kinds=("estimator", *states)),),
        requirements=(ReleaseRequirement(role="size", strategy_id="*", clock_id="entry-close"),),
        artifact_manifest_ref="manifest://r", evidence_refs=("evidence://r",),
    )
    release = ModelRelease(release_id=release_id, deployment_id="d1", bindings=(binding,))
    payloads = {estimator_hash: estimator}
    payloads.update({state.content_hash: serialize_frozen_state(state)
                     for state in states.values()})
    return release, inventory, payloads


def test_a_release_pins_frozen_pools_and_the_loader_reads_them_back(tmp_path):
    paired = _build(*_universe())
    driver = build_driver_residual_pool_artifact(
        _driver_rows(3000), role="size", model_id="size_v1_4", fold="2024-05-01",
        lineage=LINEAGE)
    states = {"paired_simulation": paired, "residual_bucket": driver}
    release, inventory, payloads = _pinned_release("rel-1", states)

    stage_release(tmp_path, release, inventory, payloads)
    staged = {member.name: member for member in resolve_release(tmp_path, "rel-1").bindings[0].members}
    loader = FrozenStateLoader(tmp_path)
    for kind, state in states.items():
        ref = FrozenStateRef(path=staged[kind].path, content_hash=staged[kind].content_hash)
        assert loader.load(ref) == state


def test_a_release_refuses_a_context_scoped_pool_under_the_pinned_hash(tmp_path):
    forecasts, outcomes, crush = _universe()
    universe = _build(forecasts, outcomes, crush)
    scoped = _build(forecasts, outcomes, [row for row in crush if row["ticker"] == "AAA"])
    release, inventory, payloads = _pinned_release("rel-2", {"paired_simulation": universe})
    payloads[universe.content_hash] = serialize_frozen_state(scoped)

    with pytest.raises(StagingRefused) as refused:
        stage_release(tmp_path, release, inventory, payloads)
    assert refused.value.issues[0].code == "PAYLOAD_HASH_MISMATCH"


# ---------------------------------------------------------------------------
# correction -> invalidation -> rebuild, end to end on real artifacts
# ---------------------------------------------------------------------------


def _panel_correction(key: str, day: str) -> ChangeSet:
    contract = TableContractRef(contract_id="tier3.panel", definition_hash="sha256:" + "0" * 64)
    version = lambda name: DatasetVersionRef(  # noqa: E731
        dataset_version_id=name, table_contract_ref=contract, manifest_hash="sha256:" + "1" * 64)
    change = RowChange(
        logical_key=key, partition_key=day, columns=("abs_move",),
        time_range=TimeInterval(column="event_date", start_inclusive=day),
        old_hash=None, new_hash=None, revision_kind="correction", revision_id="r1")
    return ChangeSet(
        changeset_id="cs-panel", table_contract_ref=contract,
        base_dataset_version_ref=version("v1"), result_dataset_version_ref=version("v2"),
        acquisition_receipt_refs=(), coverage_receipt_refs=(), changes=(change,),
        changed_partitions=(), dependency_impacts=(), unknown_dependencies=(),
        dependency_disposition="exact", outcome="changed",
        normalized_payloads=1, rewritten_partitions=0)


def test_correction_invalidates_later_pools_and_the_rebuild_is_exact():
    forecasts, outcomes, crush = _universe()
    cutoffs = ("2024-03-01", "2024-05-01", "2024-07-01")

    def pools(outcome_rows):
        return {
            f"paired:{cutoff}": _build(
                forecasts, outcome_rows, crush, cutoff=cutoff,
                lineage=Lineage(data=(DataDependency(table="tier3.panel", end_exclusive=cutoff),
                                      DataDependency(table="tier2.crush", end_exclusive=cutoff))))
            for cutoff in cutoffs
        }

    before = pools(outcomes)
    corrected_day = "2024-04-10"
    corrected = [dict(row) for row in outcomes]
    target = next(row for row in corrected if row["event_date"] == corrected_day)
    target["abs_move"] += 1.0

    report = propagate_corrections(
        [state_node(state_id, state) for state_id, state in before.items()],
        [_panel_correction(f"{target['ticker']}|{corrected_day}", corrected_day)],
    )
    assert report.invalid_ids == {"paired:2024-05-01", "paired:2024-07-01"}
    assert report.valid == ("paired:2024-03-01",)

    after = pools(corrected)
    for state_id in rebuild_order(
            [state_node(k, v) for k, v in before.items()], report.invalid_ids):
        assert after[state_id].content_hash != before[state_id].content_hash
    # The state the correction could not reach rebuilds to the same bytes, and
    # rebuilding from the uncorrected data reproduces every original exactly.
    assert serialize_frozen_state(after["paired:2024-03-01"]) == \
        serialize_frozen_state(before["paired:2024-03-01"])
    again = pools(outcomes)
    assert all(serialize_frozen_state(again[k]) == serialize_frozen_state(v)
               for k, v in before.items())


def test_paired_artifact_rejects_non_finite_values():
    with pytest.raises(ResidualArtifactError):
        make_paired_residual_pool_artifact(
            move_model_id="m", crush_model_id="c", cutoff=None,
            rows=[("2024-01-01", "AAA", 1.0, float("inf"), 0.0)], lineage=LINEAGE)
    assert isinstance(_build(*_universe()), PairedResidualPoolArtifact)
