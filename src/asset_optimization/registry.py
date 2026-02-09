"""In-memory plugin registry for planner service implementations."""

from __future__ import annotations

from typing import Any

from asset_optimization.effects.rule_based import RuleBasedEffectModel
from asset_optimization.models.proportional_hazards import ProportionalHazardsModel
from asset_optimization.models.weibull import WeibullModel
from asset_optimization.optimization.optimizer import Optimizer
from asset_optimization.simulators.basic import BasicNetworkSimulator

RISK_MODELS: dict[str, type[Any]] = {}
EFFECT_MODELS: dict[str, type[Any]] = {}
SIMULATORS: dict[str, type[Any]] = {}
OPTIMIZERS: dict[str, type[Any]] = {}

_REGISTRIES: dict[str, dict[str, type[Any]]] = {
    "risk_models": RISK_MODELS,
    "effect_models": EFFECT_MODELS,
    "simulators": SIMULATORS,
    "optimizers": OPTIMIZERS,
}

_CATEGORY_ALIASES = {
    "risk": "risk_models",
    "risk_model": "risk_models",
    "effect": "effect_models",
    "effect_model": "effect_models",
    "simulator": "simulators",
    "optimizer": "optimizers",
}


def _resolve_category(category: str) -> str:
    if not isinstance(category, str) or not category.strip():
        raise ValueError("category must be a non-empty string")

    normalized = category.strip().lower()
    normalized = _CATEGORY_ALIASES.get(normalized, normalized)
    if normalized not in _REGISTRIES:
        valid = ", ".join(sorted(_REGISTRIES))
        raise ValueError(f"Unknown category '{category}'. Valid categories: {valid}")
    return normalized


def register(category: str, key: str, cls: type[Any]) -> type[Any]:
    """Register a plugin class under a category and key."""
    resolved = _resolve_category(category)
    if not isinstance(key, str) or not key.strip():
        raise ValueError("key must be a non-empty string")
    if not isinstance(cls, type):
        raise TypeError("cls must be a class type")

    _REGISTRIES[resolved][key.strip().lower()] = cls
    return cls


def get(category: str, key: str) -> type[Any]:
    """Return a registered plugin class by category and key."""
    resolved = _resolve_category(category)
    normalized_key = key.strip().lower()
    try:
        return _REGISTRIES[resolved][normalized_key]
    except KeyError as exc:
        raise KeyError(
            f"No plugin registered for category='{resolved}' and key='{normalized_key}'"
        ) from exc


def list_registered(category: str) -> list[str]:
    """List registered keys for a category."""
    resolved = _resolve_category(category)
    return sorted(_REGISTRIES[resolved])


def clear(category: str | None = None) -> None:
    """Clear one category or all categories."""
    if category is None:
        for registry in _REGISTRIES.values():
            registry.clear()
        return

    resolved = _resolve_category(category)
    _REGISTRIES[resolved].clear()


def _register_builtins() -> None:
    register("risk_models", "weibull", WeibullModel)
    register("risk_models", "proportional_hazards", ProportionalHazardsModel)
    register("effect_models", "rule_based", RuleBasedEffectModel)
    register("simulators", "basic", BasicNetworkSimulator)
    register("optimizers", "greedy", Optimizer)


_register_builtins()


__all__ = [
    "RISK_MODELS",
    "EFFECT_MODELS",
    "SIMULATORS",
    "OPTIMIZERS",
    "register",
    "get",
    "list_registered",
    "clear",
]
