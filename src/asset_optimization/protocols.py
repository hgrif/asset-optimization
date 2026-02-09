"""Service protocol contracts for the Proposal A planner architecture."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from asset_optimization.constraints import ConstraintSet
from asset_optimization.objective import Objective
from asset_optimization.types import (
    DataFrameLike,
    PlanResult,
    PlanningHorizon,
    ScenarioSet,
)


@runtime_checkable
class AssetRepository(Protocol):
    """Data access interface for planner orchestration."""

    def load_assets(self) -> DataFrameLike: ...

    def load_events(self, event_type: str | None = None) -> DataFrameLike: ...

    def load_interventions(self) -> DataFrameLike: ...

    def load_outcomes(self) -> DataFrameLike: ...

    def load_covariates(self) -> DataFrameLike: ...

    def load_topology(self) -> DataFrameLike: ...


@runtime_checkable
class RiskModel(Protocol):
    """Risk estimation service interface."""

    def fit(
        self,
        assets: DataFrameLike,
        events: DataFrameLike,
        covariates: DataFrameLike | None = None,
    ) -> "RiskModel": ...

    def predict_distribution(
        self,
        assets: DataFrameLike,
        horizon: PlanningHorizon,
        scenarios: ScenarioSet | None = None,
    ) -> DataFrameLike: ...

    def describe(self) -> dict[str, Any]: ...


@runtime_checkable
class InterventionEffectModel(Protocol):
    """Intervention effect estimation service interface."""

    def fit(
        self,
        interventions: DataFrameLike,
        outcomes: DataFrameLike,
    ) -> "InterventionEffectModel": ...

    def estimate_effect(
        self,
        candidate_actions: DataFrameLike,
        horizon: PlanningHorizon,
        scenarios: ScenarioSet | None = None,
    ) -> DataFrameLike: ...

    def describe(self) -> dict[str, Any]: ...


@runtime_checkable
class NetworkSimulator(Protocol):
    """Network consequence simulation service interface."""

    def simulate(
        self,
        topology: DataFrameLike,
        failures: DataFrameLike,
        actions: DataFrameLike,
        scenarios: ScenarioSet | None = None,
    ) -> DataFrameLike: ...


@runtime_checkable
class PlanOptimizer(Protocol):
    """Portfolio plan optimization service interface."""

    def solve(
        self,
        objective: Objective,
        constraints: ConstraintSet,
        candidates: DataFrameLike,
        risk_measure: str = "expected_value",
    ) -> PlanResult: ...


__all__ = [
    "AssetRepository",
    "RiskModel",
    "InterventionEffectModel",
    "NetworkSimulator",
    "PlanOptimizer",
]
