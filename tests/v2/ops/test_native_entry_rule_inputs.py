"""entry_rule_declarations: one entry-rule row's frozen inputs, synthetic only."""
import dataclasses
import engine.v2.ops.native_score_batch as nsb
from types import SimpleNamespace

import pytest

from engine.v2.models.lineage import DataDependency, Lineage
from engine.v2.models.residual_artifact import make_paired_residual_pool_artifact
from engine.v2.models.trailing_cutoff_artifact import make_trailing_cutoff_artifact
from engine.v2.ops.native_entry_rule_inputs import entry_rule_declarations
from engine.v2.scoring.nightly_source_bundle import NightlySourceBundleRefusal
from tests.v2.ops.test_native_score_batch_driver_alias import (
    SIZE,
    _assemble,
    _event_inputs,
    _stage,
)
from tests.v2.ops.test_native_score_batch_twin_registry import IV_CRUSH

SIZE_MODEL = "m-move"
CRUSH_MODEL = "m-crush"
POOL_CUTOFF = "2026-02-01"
EVENT_DATE = "2026-03-16"
MCAP = 2e10


def _paired(*, move_model_id=SIZE_MODEL, crush_model_id=CRUSH_MODEL,
            cutoff=POOL_CUTOFF, value=1.0):
    return make_paired_residual_pool_artifact(
        move_model_id=move_model_id, crush_model_id=crush_model_id, cutoff=cutoff,
        rows=[["2026-01-01", "AAA", 1.0, value, 0.2]], lineage=Lineage())


def _trailing(*, history_id="features.pnl_sim_history", month="2026-03-01", cutoff=0.5):
    return make_trailing_cutoff_artifact(
        month=month, cutoff=cutoff, history_id=history_id,
        lineage=Lineage(data=(DataDependency(
            table="synthetic.pnl-history", end_exclusive=month),)))


def _binding(paired, trailing):
    return SimpleNamespace(paired_residual_artifacts=paired,
                           trailing_cutoff_artifacts=trailing)


def _identity(model_id):
    return SimpleNamespace(model_id=model_id)


def _ready_binding():
    return _binding((_paired(),), (_trailing(),))


def _declare(binding, **overrides):
    values = dict(
        strategy="TWIN-P",
        size_identity=_identity(SIZE_MODEL),
        crush_identity=_identity(CRUSH_MODEL),
        cutoff=POOL_CUTOFF,
        event_date=EVENT_DATE,
        mcap_usd=MCAP,
    )
    values.update(overrides)
    return entry_rule_declarations(binding, **values)


def _refused(binding, detail, **overrides):
    with pytest.raises(NightlySourceBundleRefusal) as error:
        _declare(binding, **overrides)
    assert error.value.code == "MODEL_NOT_READY"
    assert error.value.detail == detail


def test_happy_path_returns_the_exact_pool_recipe_and_gate_recipe():
    pool = _paired()
    bar = _trailing()
    result = _declare(_binding((pool,), (bar,)))
    assert result == {
        "paired_residual_artifact": pool,
        "paired_residual_recipe": {
            "move_model_id": SIZE_MODEL,
            "crush_model_id": CRUSH_MODEL,
            "cutoff": POOL_CUTOFF,
            "content_hash": pool.content_hash,
        },
        "trailing_cutoff_artifact": bar,
        "gate_recipe": {"mode": "entry_rule", "mcap_usd": MCAP},
    }


def test_mcap_none_is_allowed():
    result = _declare(_ready_binding(), mcap_usd=None)
    assert result["gate_recipe"] == {"mode": "entry_rule", "mcap_usd": None}


@pytest.mark.parametrize("strategy", ["STR-THRU", "CAL-P"])
def test_a_strategy_without_an_entry_rule_is_refused(strategy):
    _refused(_ready_binding(), "entry rule strategy is not defined",
             strategy=strategy)


def test_zero_paired_pools_is_refused():
    _refused(_binding((), (_trailing(),)),
             "one exact paired residual pool is required")


def test_two_matching_paired_pools_is_refused():
    _refused(_binding((_paired(value=1.0), _paired(value=9.0)), (_trailing(),)),
             "more than one exact paired residual pool")


def test_a_pool_of_another_crush_model_is_refused():
    _refused(_binding((_paired(crush_model_id="m-other"),), (_trailing(),)),
             "one exact paired residual pool is required")


def test_a_pool_of_another_move_model_is_refused():
    _refused(_binding((_paired(move_model_id="m-other"),), (_trailing(),)),
             "one exact paired residual pool is required")


def test_a_pool_frozen_at_another_cutoff_is_refused():
    _refused(_binding((_paired(cutoff="2026-03-01"),), (_trailing(),)),
             "one exact paired residual pool is required")


def test_no_trailing_cutoff_for_the_event_month_is_refused():
    _refused(_binding((_paired(),), (_trailing(month="2026-04-01"),)),
             "one exact trailing cutoff artifact is required")


def test_a_trailing_cutoff_of_another_history_is_refused():
    _refused(_binding((_paired(),),
                      (_trailing(history_id="features.other_history"),)),
             "one exact trailing cutoff artifact is required")


def test_two_matching_trailing_cutoffs_is_refused():
    _refused(_binding((_paired(),), (_trailing(cutoff=0.5), _trailing(cutoff=9.0))),
             "more than one exact trailing cutoff artifact")


def test_an_unparsable_event_date_is_refused():
    _refused(_ready_binding(),
             "event date is not a valid trailing cutoff month",
             event_date="not-a-date")


@pytest.mark.parametrize("mcap_usd", [True, float("nan"), float("inf"), "5", b"5"],
                         ids=["bool", "nan", "inf", "str", "bytes"])
def test_an_unusable_mcap_is_refused(mcap_usd):
    _refused(_ready_binding(),
             "entry rule mcap_usd must be a finite number or None",
             mcap_usd=mcap_usd)


_TWIN_SIZE_MODEL = "m-size-*"
_TWIN_CRUSH_MODEL = "m-iv_crush-*"
_TWIN_ROW_CUTOFF = "2026-01-10"  # min(as_of 2026-01-10, entry 2026-01-15)
_TWIN_EVENT_MONTH = "2026-01-01"  # cutoff_month(event_date 2026-01-15)


def _twin_pool(**overrides):
    return _paired(move_model_id=_TWIN_SIZE_MODEL, crush_model_id=_TWIN_CRUSH_MODEL,
                   cutoff=_TWIN_ROW_CUTOFF, **overrides)


def _twin_cutoff(**overrides):
    return _trailing(month=_TWIN_EVENT_MONTH, **overrides)


@pytest.mark.parametrize("paired,trailing,expected_detail", [
    ((), (_twin_cutoff(),),
     "nightly_source_bundle refused: MODEL_NOT_READY: paired residual pool missing"),
    ((_twin_pool(value=1.0), _twin_pool(value=9.0)), (_twin_cutoff(),),
     "nightly_source_bundle refused: MODEL_NOT_READY: duplicate paired residual pool"),
    ((_twin_pool(),), (),
     "nightly_source_bundle refused: MODEL_NOT_READY: trailing cutoff artifact missing"),
    ((_twin_pool(),), (_twin_cutoff(cutoff=0.5), _twin_cutoff(cutoff=9.0)),
     "nightly_source_bundle refused: MODEL_NOT_READY: duplicate trailing cutoff artifact"),
], ids=["pool-missing", "pool-duplicate", "cutoff-missing", "cutoff-duplicate"])
def test_row_assembly_publishes_each_missing_and_duplicate_label(
        tmp_path, paired, trailing, expected_detail, monkeypatch):
    # End-to-end through native_score_batch row assembly on a real promoted
    # release: the families are replaced post-resolution because the release
    # resolver itself refuses duplicate semantic keys release-wide. The
    # published detail strings are typed here, never read from production.
    # Seed a valid chooser state so projection/admission passes and the test
    # reaches the targeted source-bundle refusals instead of the chooser refusal.
    event = _event_inputs(strategy="TWIN-P")
    event.panel_row.update({name: 1.0 for name in nsb._CHOOSER_PRIMITIVES})
    event.panel_row["x"] = 1.0
    state = SimpleNamespace(
        producer_bindings={
            "pred_im_t1_d14": SimpleNamespace(binding_id="chooser_implied", feature_order=("x",)),
            "pred_runup_abs_move_d14": SimpleNamespace(binding_id="chooser_runup", feature_order=("x",)),
        },
        artifact_refs=(),
        fold_provenance={},
        analog_pool=SimpleNamespace(pool_id="p", cutoff="2026-01-15", content_hash="h"),
        admissible_table=SimpleNamespace(table_id="t", version=1, content_hash="h"),
        fold_pools={},
    )
    monkeypatch.setattr(nsb, "select_chooser_state", lambda *_a, **_k: state)
    binding = dataclasses.replace(
        _stage(tmp_path, [SIZE, IV_CRUSH]),
        paired_residual_artifacts=paired, trailing_cutoff_artifacts=trailing)
    assembled, refusals = _assemble(binding, [event])
    assert assembled == {}
    assert [(item.key, item.code, item.detail) for item in refusals] == [
        (event.key, "MODEL_NOT_READY", expected_detail)]