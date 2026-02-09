"""Basic network simulator implementation."""

from __future__ import annotations

from asset_optimization.types import DataFrameLike, ScenarioSet


class BasicNetworkSimulator:
    """Pass-through network simulator for planner orchestration.

    This simulator provides a minimal implementation of the NetworkSimulator
    protocol. It returns the candidate actions unchanged and exists as a
    default plugin when no domain-specific network model is available.
    """

    def simulate(
        self,
        topology: DataFrameLike,
        failures: DataFrameLike,
        actions: DataFrameLike,
        scenarios: ScenarioSet | None = None,
    ) -> DataFrameLike:
        return actions.copy(deep=True)


__all__ = ["BasicNetworkSimulator"]
