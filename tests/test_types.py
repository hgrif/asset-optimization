"""Tests for Proposal A foundational types and model errors."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from asset_optimization.exceptions import ModelError
from asset_optimization.types import (
    PlanningHorizon,
    PlanResult,
    ScenarioSet,
    ValidationReport,
)


def test_planning_horizon_normalizes_dates_and_step() -> None:
    """PlanningHorizon stores timestamps and normalizes step to lowercase."""
    horizon = PlanningHorizon("2026-01-01", "2026-12-31", "YEARLY")

    assert isinstance(horizon.start_date, pd.Timestamp)
    assert isinstance(horizon.end_date, pd.Timestamp)
    assert horizon.step == "yearly"


def test_planning_horizon_rejects_invalid_step() -> None:
    """PlanningHorizon rejects unknown step values."""
    with pytest.raises(ValueError, match="step must be one of"):
        PlanningHorizon("2026-01-01", "2026-12-31", "weekly")


def test_planning_horizon_requires_end_after_start() -> None:
    """PlanningHorizon requires end_date > start_date."""
    with pytest.raises(ValueError, match="end_date must be after start_date"):
        PlanningHorizon("2026-01-01", "2025-12-31", "yearly")


def test_plan_result_is_frozen() -> None:
    """PlanResult dataclass is immutable."""
    result = PlanResult(
        selected_actions=pd.DataFrame({"asset_id": ["A1"]}),
        objective_breakdown={"risk_reduction": 1.0},
        constraint_shadow_prices={"budget_limit": 0.0},
    )

    with pytest.raises(FrozenInstanceError):
        result.metadata = {"changed": True}


def test_scenario_set_validates_required_columns_and_probability_range() -> None:
    """ScenarioSet rejects missing columns and invalid probabilities."""
    missing_cols_df = pd.DataFrame(
        {
            "scenario_id": [1],
            "variable": ["demand"],
            "timestamp": ["2026-01-01"],
            "value": [12.0],
        }
    )
    with pytest.raises(ValueError, match="missing required columns"):
        ScenarioSet(missing_cols_df)

    invalid_prob_df = pd.DataFrame(
        {
            "scenario_id": [1],
            "variable": ["demand"],
            "timestamp": ["2026-01-01"],
            "value": [12.0],
            "probability": [1.2],
        }
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        ScenarioSet(invalid_prob_df)


def test_validation_report_defaults_warnings_to_empty_list() -> None:
    """ValidationReport defaults warnings to an empty list."""
    report = ValidationReport(passed=True, checks={"required_columns": True})
    assert report.warnings == []


def test_model_error_includes_details_in_message() -> None:
    """ModelError message includes optional details."""
    err = ModelError("fit failed", details={"model": "WeibullModel", "phase": "fit"})

    assert "Model error: fit failed" in str(err)
    assert "model=WeibullModel" in str(err)
    assert "phase=fit" in str(err)
