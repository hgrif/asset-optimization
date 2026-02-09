"""Core data contracts for Proposal A planner APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

_VALID_HORIZON_STEPS = {"monthly", "quarterly", "yearly"}
_SCENARIO_REQUIRED_COLUMNS = {
    "scenario_id",
    "variable",
    "timestamp",
    "value",
    "probability",
}

DataFrameLike = pd.DataFrame


@dataclass(frozen=True)
class PlanningHorizon:
    """Planning window definition for optimization/simulation calls.

    Parameters
    ----------
    start_date : str or pd.Timestamp
        First date in the planning window.
    end_date : str or pd.Timestamp
        Last date in the planning window (must be after ``start_date``).
    step : str
        Temporal resolution for the horizon. Must be one of
        ``monthly``, ``quarterly``, or ``yearly``.
    """

    start_date: str | pd.Timestamp
    end_date: str | pd.Timestamp
    step: str

    def __post_init__(self) -> None:
        step = self.step.lower()
        if step not in _VALID_HORIZON_STEPS:
            options = ", ".join(sorted(_VALID_HORIZON_STEPS))
            raise ValueError(f"step must be one of: {options}")

        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)
        if pd.isna(start) or pd.isna(end):
            raise ValueError("start_date and end_date must be valid timestamps")
        if end <= start:
            raise ValueError("end_date must be after start_date")

        object.__setattr__(self, "start_date", start)
        object.__setattr__(self, "end_date", end)
        object.__setattr__(self, "step", step)


@dataclass(frozen=True)
class PlanResult:
    """Optimization output contract used by planner-facing APIs.

    Parameters
    ----------
    selected_actions : pd.DataFrame
        Chosen interventions/actions.
    objective_breakdown : dict[str, float]
        Contribution by objective term.
    constraint_shadow_prices : dict[str, float]
        Constraint dual values (if available).
    metadata : dict[str, Any], optional
        Additional implementation-specific context.
    """

    selected_actions: pd.DataFrame
    objective_breakdown: dict[str, float]
    constraint_shadow_prices: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.selected_actions, pd.DataFrame):
            raise TypeError("selected_actions must be a pandas DataFrame")


@dataclass(frozen=True)
class ScenarioSet:
    """Stochastic scenario container for uncertain inputs.

    Parameters
    ----------
    scenarios : pd.DataFrame
        Long-format scenario table with required columns:
        ``scenario_id``, ``variable``, ``timestamp``, ``value``, ``probability``.
    """

    scenarios: pd.DataFrame

    def __post_init__(self) -> None:
        if not isinstance(self.scenarios, pd.DataFrame):
            raise TypeError("scenarios must be a pandas DataFrame")

        missing = _SCENARIO_REQUIRED_COLUMNS.difference(self.scenarios.columns)
        if missing:
            raise ValueError(
                "scenarios is missing required columns: " + ", ".join(sorted(missing))
            )

        probabilities = pd.to_numeric(self.scenarios["probability"], errors="coerce")
        if probabilities.isna().any():
            raise ValueError("scenarios.probability must be numeric")
        if ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise ValueError("scenarios.probability values must be between 0 and 1")

        scenarios_copy = self.scenarios.copy()
        scenarios_copy["timestamp"] = pd.to_datetime(
            scenarios_copy["timestamp"], errors="coerce"
        )
        if scenarios_copy["timestamp"].isna().any():
            raise ValueError("scenarios.timestamp must contain valid timestamps")

        object.__setattr__(self, "scenarios", scenarios_copy)


@dataclass(frozen=True)
class ValidationReport:
    """Input validation status for planner lifecycle.

    Parameters
    ----------
    passed : bool
        Overall validation outcome.
    checks : dict[str, bool]
        Named checks and their pass/fail status.
    warnings : list[str], optional
        Non-blocking validation messages.
    """

    passed: bool
    checks: dict[str, bool]
    warnings: list[str] = field(default_factory=list)
