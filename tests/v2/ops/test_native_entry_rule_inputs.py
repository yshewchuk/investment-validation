"""entry_rule_declarations: one entry-rule row's frozen inputs, synthetic only."""
from types import SimpleNamespace

import pytest

from engine.v2.models.lineage import DataDependency, Lineage
from engine.v2.models.residual_artifact import make_paired_residual_pool_artifact
from engine.v2.models.trailing_cutoff_artifact import make_trailing_cutoff_artifact
from engine.v2.ops.native_entry_rule_inputs import entry_rule_declarations
from engine.v2.scoring.nightly_source_bundle import NightlySourceBundleRefusal

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
             "one exact paired residual pool is required")


def test_a_pool_of_another_crush_model_is_refused():
    _refused(_binding((_paired(crush_model_id="m-other"),), (_trailing(),)),
             "one exact paired residual pool is required")


def test_a_pool_frozen_at_another_cutoff_is_refused():
    _refused(_binding((_paired(cutoff="2026-03-01"),), (_trailing(),)),
             "one exact paired residual pool is required")


def test_no_trailing_cutoff_for_the_event_month_is_refused():
    _refused(_binding((_paired(),), (_trailing(month="2026-04-01"),)),
             "one exact trailing cutoff artifact is required")


def test_two_matching_trailing_cutoffs_is_refused():
    _refused(_binding((_paired(),), (_trailing(cutoff=0.5), _trailing(cutoff=9.0))),
             "one exact trailing cutoff artifact is required")


def test_an_unparsable_event_date_is_refused():
    _refused(_ready_binding(),
             "event date is not a valid trailing cutoff month",
             event_date="not-a-date")


@pytest.mark.parametrize("mcap_usd", [True, float("nan"), float("inf"), "2e10"],
                         ids=["bool", "nan", "inf", "str"])
def test_an_unusable_mcap_is_refused(mcap_usd):
    _refused(_ready_binding(),
             "entry rule mcap_usd must be a finite number or None",
             mcap_usd=mcap_usd)