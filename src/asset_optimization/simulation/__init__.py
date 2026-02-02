"""Simulation engine for asset portfolio optimization.

This module provides:
- SimulationConfig: Immutable configuration for simulation runs
- SimulationResult: Structured output with DataFrames for analysis
- InterventionType: Dataclass for intervention type definitions
- Predefined interventions: DO_NOTHING, INSPECT, REPAIR, REPLACE
"""

from asset_optimization.simulation.config import SimulationConfig
from asset_optimization.simulation.interventions import (
    DO_NOTHING,
    INSPECT,
    REPAIR,
    REPLACE,
    InterventionType,
)
from asset_optimization.simulation.result import SimulationResult

__all__ = [
    "SimulationConfig",
    "SimulationResult",
    "InterventionType",
    "DO_NOTHING",
    "INSPECT",
    "REPAIR",
    "REPLACE",
]
