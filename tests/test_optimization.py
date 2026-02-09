"""Tests for optimization module."""

import pytest
import pandas as pd

from asset_optimization.constraints import ConstraintSet
from asset_optimization.exceptions import OptimizationError
from asset_optimization.objective import ObjectiveBuilder
from asset_optimization.optimization import Optimizer
from asset_optimization.protocols import PlanOptimizer
from asset_optimization.types import PlanResult


class TestOptimizerInit:
    """Tests for Optimizer initialization."""

    def test_default_strategy(self):
        """Default strategy is 'greedy'."""
        opt = Optimizer()
        assert opt.strategy == "greedy"

    def test_custom_strategy(self):
        """Can set custom strategy."""
        opt = Optimizer(strategy="milp")
        assert opt.strategy == "milp"

    def test_repr(self):
        """Repr shows strategy."""
        opt = Optimizer()
        assert "greedy" in repr(opt)


class TestPlannerOptimizationCompatibility:
    """Planner-oriented PlanOptimizer compatibility checks."""

    def test_optimizer_matches_plan_optimizer_protocol(self):
        opt = Optimizer()
        assert isinstance(opt, PlanOptimizer)

    def test_solve_selects_within_budget_and_returns_plan_result(self):
        opt = Optimizer()
        candidates = pd.DataFrame(
            {
                "asset_id": ["A1", "A2", "A3"],
                "action_type": ["replace", "repair", "inspect"],
                "direct_cost": [50000.0, 5000.0, 500.0],
                "expected_benefit": [12000.0, 4000.0, 200.0],
                "expected_risk_reduction": [0.25, 0.15, 0.01],
            }
        )
        objective = (
            ObjectiveBuilder()
            .add_expected_risk_reduction()
            .add_total_cost(weight=-1.0)
            .build()
        )
        constraints = ConstraintSet().add_budget_limit(5500.0)

        result = opt.solve(objective, constraints, candidates)

        assert isinstance(result, PlanResult)
        assert result.selected_actions["asset_id"].tolist() == ["A2", "A3"]
        assert result.selected_actions["direct_cost"].sum() <= 5500.0
        assert result.metadata["budget_limit"] == 5500.0
        assert result.metadata["selected_count"] == 2
        assert "expected_risk_reduction" in result.objective_breakdown
        assert "total_cost" in result.objective_breakdown

    def test_solve_uses_unbounded_budget_when_constraint_absent(self):
        opt = Optimizer()
        candidates = pd.DataFrame(
            {
                "asset_id": ["A1", "A2"],
                "action_type": ["replace", "repair"],
                "direct_cost": [50000.0, 5000.0],
                "expected_benefit": [12000.0, 4000.0],
            }
        )
        objective = ObjectiveBuilder().add_expected_risk_reduction().build()
        constraints = ConstraintSet()

        result = opt.solve(objective, constraints, candidates)

        assert result.metadata["budget_limit"] is None
        assert result.metadata["selected_count"] == 2

    def test_solve_requires_required_candidate_columns(self):
        opt = Optimizer()
        objective = ObjectiveBuilder().add_expected_risk_reduction().build()
        constraints = ConstraintSet().add_budget_limit(1000.0)
        missing_cost = pd.DataFrame({"asset_id": ["A1"], "expected_benefit": [1.0]})

        with pytest.raises(OptimizationError, match="required columns"):
            opt.solve(objective, constraints, missing_cost)


class TestSolveEdgeCases:
    """Edge-case tests for the solve() method."""

    def test_solve_empty_candidates(self):
        """Empty candidates DataFrame returns empty PlanResult."""
        opt = Optimizer()
        candidates = pd.DataFrame(
            columns=["asset_id", "direct_cost", "expected_benefit"]
        )
        objective = ObjectiveBuilder().add_expected_risk_reduction().build()
        constraints = ConstraintSet().add_budget_limit(10000.0)

        result = opt.solve(objective, constraints, candidates)

        assert isinstance(result, PlanResult)
        assert len(result.selected_actions) == 0
        assert result.metadata["selected_count"] == 0

    def test_solve_zero_budget(self):
        """Zero budget selects nothing."""
        opt = Optimizer()
        candidates = pd.DataFrame(
            {
                "asset_id": ["A1", "A2"],
                "direct_cost": [5000.0, 1000.0],
                "expected_benefit": [3000.0, 500.0],
            }
        )
        objective = ObjectiveBuilder().add_expected_risk_reduction().build()
        constraints = ConstraintSet().add_budget_limit(0.0)

        result = opt.solve(objective, constraints, candidates)

        assert len(result.selected_actions) == 0
        assert result.metadata["budget_spent"] == 0.0

    def test_solve_missing_columns_raises(self):
        """Missing required columns raise OptimizationError."""
        opt = Optimizer()
        objective = ObjectiveBuilder().add_expected_risk_reduction().build()
        constraints = ConstraintSet()

        bad_df = pd.DataFrame({"asset_id": ["A1"], "direct_cost": [100.0]})
        with pytest.raises(OptimizationError, match="required columns"):
            opt.solve(objective, constraints, bad_df)
