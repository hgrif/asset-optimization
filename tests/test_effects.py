"""Tests for rule-based intervention effect model."""

import pandas as pd
import pytest

from asset_optimization.effects import RuleBasedEffectModel
from asset_optimization.exceptions import ModelError
from asset_optimization.protocols import InterventionEffectModel
from asset_optimization.types import PlanningHorizon


def test_rule_based_effect_model_matches_protocol() -> None:
    """RuleBasedEffectModel structurally satisfies effect protocol."""
    model = RuleBasedEffectModel({"replace": 1.0, "repair": 0.5})
    assert isinstance(model, InterventionEffectModel)


def test_fit_is_noop_and_returns_self() -> None:
    """fit keeps scikit-learn style contract."""
    model = RuleBasedEffectModel({"replace": 1.0})
    interventions = pd.DataFrame({"asset_id": ["A1"], "action_type": ["replace"]})
    outcomes = pd.DataFrame({"asset_id": ["A1"], "loss": [10.0]})

    result = model.fit(interventions=interventions, outcomes=outcomes)

    assert result is model


def test_estimate_effect_adds_expected_columns_without_mutating_input() -> None:
    """Model adds expected columns from rule map and candidate risk/cost."""
    model = RuleBasedEffectModel({"replace": 1.0, "repair": 0.5})
    candidates = pd.DataFrame(
        {
            "asset_id": ["A1", "A2", "A3"],
            "action_type": ["replace", "repair", "inspect"],
            "failure_prob": [0.8, 0.4, 0.6],
            "consequence_cost": [1000.0, 500.0, 400.0],
        }
    )
    horizon = PlanningHorizon("2026-01-01", "2026-12-31", "yearly")

    result = model.estimate_effect(candidates, horizon)

    assert "expected_risk_reduction" in result.columns
    assert "expected_benefit" in result.columns
    assert "expected_risk_reduction" not in candidates.columns
    assert result["expected_risk_reduction"].tolist() == pytest.approx([0.8, 0.2, 0.0])
    assert result["expected_benefit"].tolist() == pytest.approx([800.0, 100.0, 0.0])


def test_estimate_effect_falls_back_to_failure_probability_column() -> None:
    """Model uses failure_probability if failure_prob is absent."""
    model = RuleBasedEffectModel({"repair": 0.5})
    candidates = pd.DataFrame(
        {
            "asset_id": ["A1"],
            "action_type": ["repair"],
            "failure_probability": [0.6],
            "consequence_cost": [100.0],
        }
    )
    horizon = PlanningHorizon("2026-01-01", "2026-12-31", "yearly")

    result = model.estimate_effect(candidates, horizon)

    assert result["expected_risk_reduction"].iloc[0] == pytest.approx(0.3)
    assert result["expected_benefit"].iloc[0] == pytest.approx(30.0)


def test_estimate_effect_requires_action_type() -> None:
    """Candidate actions must include action_type for rule lookup."""
    model = RuleBasedEffectModel({"replace": 1.0})
    candidates = pd.DataFrame({"asset_id": ["A1"], "failure_prob": [0.8]})
    horizon = PlanningHorizon("2026-01-01", "2026-12-31", "yearly")

    with pytest.raises(ModelError, match="action_type"):
        model.estimate_effect(candidates, horizon)


def test_describe_returns_effect_rules() -> None:
    """describe includes serialized rule mapping."""
    model = RuleBasedEffectModel({"replace": 1.0, "repair": 0.5})

    description = model.describe()

    assert description["model_type"] == "RuleBasedEffectModel"
    assert description["effect_rules"] == {"replace": 1.0, "repair": 0.5}
