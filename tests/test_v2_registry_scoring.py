from engine.v2.registry import DYNAMIC_MENU, STRATEGY_IDS, default_registry


def test_inventory_contains_all_factories_and_dynamic_menu():
    registry = default_registry()
    assert len(STRATEGY_IDS) == 11
    assert tuple(spec.strategy_id for spec in registry.strategies[:-1]) == STRATEGY_IDS
    assert registry.strategy("DYN-SV").structure_parameters["menu"] == DYNAMIC_MENU
    assert registry.strategy("CAL-P").validation_status == "disabled"
    assert registry.strategy("CND-P").validation_status == "disabled"


def test_deployment_has_six_roles_and_is_read_only():
    deployment = default_registry().deployment("legacy-phase4-deployment.v1")
    assert set(deployment.model_role_bindings) == {
        "size", "implied_t1", "runup_move", "iv_crush",
        "gate:STR-THRU", "gate:STR-RUNUP", "chooser:DYN-SV",
    }
    assert deployment.mode == "historical"
