"""Tests for planner plugin registry."""

import pytest

from asset_optimization import registry as plugin_registry
from asset_optimization.effects import RuleBasedEffectModel
from asset_optimization.models import ProportionalHazardsModel, WeibullModel
from asset_optimization.optimization import Optimizer
from asset_optimization.simulation import Simulator


@pytest.fixture(autouse=True)
def restore_registry_state() -> None:
    """Preserve global registry state across tests."""
    snapshots = {
        "RISK_MODELS": dict(plugin_registry.RISK_MODELS),
        "EFFECT_MODELS": dict(plugin_registry.EFFECT_MODELS),
        "SIMULATORS": dict(plugin_registry.SIMULATORS),
        "OPTIMIZERS": dict(plugin_registry.OPTIMIZERS),
    }
    try:
        yield
    finally:
        for name, snapshot in snapshots.items():
            bucket = getattr(plugin_registry, name)
            bucket.clear()
            bucket.update(snapshot)


def test_builtins_are_registered_at_import() -> None:
    """Step 7 builtins are available without manual registration."""
    assert plugin_registry.get("risk_models", "weibull") is WeibullModel
    assert (
        plugin_registry.get("risk_models", "proportional_hazards")
        is ProportionalHazardsModel
    )
    assert plugin_registry.get("effect_models", "rule_based") is RuleBasedEffectModel
    assert plugin_registry.get("simulators", "basic") is Simulator
    assert plugin_registry.get("optimizers", "greedy") is Optimizer


def test_register_and_get_custom_plugin() -> None:
    """Custom plugin registration stores and resolves by key."""

    class CustomRiskModel:
        pass

    plugin_registry.register("risk_models", "custom", CustomRiskModel)

    assert plugin_registry.get("risk_models", "custom") is CustomRiskModel
    assert "custom" in plugin_registry.list_registered("risk_models")


def test_alias_categories_are_supported() -> None:
    """Singular category names map to canonical registries."""

    class CustomOptimizer:
        pass

    plugin_registry.register("optimizer", "custom", CustomOptimizer)
    assert plugin_registry.get("optimizers", "custom") is CustomOptimizer


def test_clear_single_category_does_not_clear_others() -> None:
    """clear(category) should only reset one bucket."""
    plugin_registry.clear("effect_models")

    assert plugin_registry.list_registered("effect_models") == []
    assert "weibull" in plugin_registry.list_registered("risk_models")


def test_clear_without_category_clears_all() -> None:
    """clear() with no category should clear every bucket."""
    plugin_registry.clear()

    assert plugin_registry.list_registered("risk_models") == []
    assert plugin_registry.list_registered("effect_models") == []
    assert plugin_registry.list_registered("simulators") == []
    assert plugin_registry.list_registered("optimizers") == []


def test_unknown_category_raises_value_error() -> None:
    """Unknown categories should fail with an explicit message."""
    with pytest.raises(ValueError, match="Unknown category"):
        plugin_registry.list_registered("unknown")


def test_missing_key_raises_key_error() -> None:
    """Lookup of a missing plugin key should raise KeyError."""
    with pytest.raises(KeyError, match="No plugin registered"):
        plugin_registry.get("risk_models", "missing")


def test_register_requires_class_type() -> None:
    """Registry only accepts class objects for plugin values."""
    with pytest.raises(TypeError, match="class type"):
        plugin_registry.register("risk_models", "bad", object())
