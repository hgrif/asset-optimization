"""Simulation engine for asset portfolio optimization.

This module provides:
- SimulationConfig: Immutable configuration for simulation runs
- SimulationResult: Structured output with DataFrames for analysis
"""

from asset_optimization.simulation.config import SimulationConfig

__all__ = [
    'SimulationConfig',
]
