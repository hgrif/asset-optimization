"""Tests for root-package public API exports."""

import types

import asset_optimization as ao
from asset_optimization.constraints import ConstraintSet
from asset_optimization.effects import RuleBasedEffectModel
from asset_optimization.objective import ObjectiveBuilder
from asset_optimization.planner import Planner
from asset_optimization.protocols import RiskModel
from asset_optimization.repositories import DataFrameRepository
from asset_optimization.types import PlanResult, PlanningHorizon


def test_root_package_exports_proposal_a_symbols() -> None:
    """Step 8 keeps planner-facing Proposal A symbols at package root."""
    assert ao.ConstraintSet is ConstraintSet
    assert ao.ObjectiveBuilder is ObjectiveBuilder
    assert ao.Planner is Planner
    assert ao.DataFrameRepository is DataFrameRepository
    assert ao.RuleBasedEffectModel is RuleBasedEffectModel
    assert ao.PlanningHorizon is PlanningHorizon
    assert ao.PlanResult is PlanResult
    assert ao.RiskModel is RiskModel


def test_root_package_exports_registry_module_and_helpers() -> None:
    """Registry module and helper functions are available from package root."""
    assert isinstance(ao.registry, types.ModuleType)
    assert callable(ao.register)
    assert callable(ao.get)
    assert callable(ao.list_registered)
    assert callable(ao.clear)
    assert "weibull" in ao.list_registered("risk_models")
