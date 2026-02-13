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


class TestGroupCoherence:
    """Tests for group coherence constraint in optimizer."""

    def test_group_coherence_selects_complete_groups(self):
        """Optimizer selects all assets in a group together."""
        opt = Optimizer()
        candidates = pd.DataFrame(
            {
                "asset_id": ["A1", "A2", "A3", "A4"],
                "action_type": ["repair", "repair", "replace", "replace"],
                "direct_cost": [1000.0, 1000.0, 2000.0, 2000.0],
                "expected_benefit": [500.0, 500.0, 1500.0, 1500.0],
                "group_id": ["group_1", "group_1", "group_2", "group_2"],
            }
        )
        objective = (
            ObjectiveBuilder()
            .add_expected_risk_reduction()
            .add_total_cost(weight=-1.0)
            .build()
        )
        constraints = ConstraintSet().add_budget_limit(2500.0).add_group_coherence()

        result = opt.solve(objective, constraints, candidates)

        # Budget allows group_1 (2000 total) but not group_2 (4000 total)
        assert len(result.selected_actions) == 2
        assert set(result.selected_actions["asset_id"]) == {"A1", "A2"}
        assert result.selected_actions["direct_cost"].sum() == 2000.0

    def test_group_coherence_skips_unaffordable_group(self):
        """Optimizer skips entire group if budget insufficient."""
        opt = Optimizer()
        candidates = pd.DataFrame(
            {
                "asset_id": ["A1", "A2", "A3"],
                "action_type": ["repair", "repair", "replace"],
                "direct_cost": [1000.0, 1000.0, 500.0],
                "expected_benefit": [800.0, 800.0, 300.0],
                "group_id": ["group_1", "group_1", None],
            }
        )
        objective = (
            ObjectiveBuilder()
            .add_expected_risk_reduction()
            .add_total_cost(weight=-1.0)
            .build()
        )
        constraints = ConstraintSet().add_budget_limit(1500.0).add_group_coherence()

        result = opt.solve(objective, constraints, candidates)

        # Budget can't afford group_1 (2000), but can afford singleton A3
        assert len(result.selected_actions) == 1
        assert result.selected_actions["asset_id"].tolist() == ["A3"]

    def test_group_coherence_null_group_ids_as_singletons(self):
        """Assets with null group_id are treated as singletons."""
        opt = Optimizer()
        candidates = pd.DataFrame(
            {
                "asset_id": ["A1", "A2", "A3"],
                "action_type": ["repair", "repair", "inspect"],
                "direct_cost": [1000.0, 1000.0, 200.0],
                "expected_benefit": [600.0, 500.0, 100.0],
                "group_id": [None, None, None],
            }
        )
        objective = (
            ObjectiveBuilder()
            .add_expected_risk_reduction()
            .add_total_cost(weight=-1.0)
            .build()
        )
        constraints = ConstraintSet().add_budget_limit(1500.0).add_group_coherence()

        result = opt.solve(objective, constraints, candidates)

        # All null group_id → treated as individual singletons
        assert len(result.selected_actions) == 2
        assert set(result.selected_actions["asset_id"]) == {"A1", "A3"}

    def test_group_coherence_no_group_column(self):
        """Group coherence constraint is no-op when group_id column missing."""
        opt = Optimizer()
        candidates = pd.DataFrame(
            {
                "asset_id": ["A1", "A2"],
                "action_type": ["repair", "replace"],
                "direct_cost": [1000.0, 2000.0],
                "expected_benefit": [800.0, 1500.0],
            }
        )
        objective = (
            ObjectiveBuilder()
            .add_expected_risk_reduction()
            .add_total_cost(weight=-1.0)
            .build()
        )
        constraints = ConstraintSet().add_budget_limit(3500.0).add_group_coherence()

        result = opt.solve(objective, constraints, candidates)

        # No group_id column → behaves like normal budget selection
        # A1 ratio=0.8, A2 ratio=0.75 → both selected within budget
        assert len(result.selected_actions) == 2
        assert set(result.selected_actions["asset_id"]) == {"A1", "A2"}

    def test_group_coherence_with_budget_limit(self):
        """Group coherence respects budget by selecting groups in priority order."""
        opt = Optimizer()
        candidates = pd.DataFrame(
            {
                "asset_id": ["A1", "A2", "A3", "A4", "A5"],
                "action_type": ["repair"] * 5,
                "direct_cost": [500.0, 500.0, 1000.0, 1000.0, 300.0],
                "expected_benefit": [400.0, 400.0, 1200.0, 1200.0, 150.0],
                "group_id": ["grp_A", "grp_A", "grp_B", "grp_B", None],
            }
        )
        objective = (
            ObjectiveBuilder()
            .add_expected_risk_reduction()
            .add_total_cost(weight=-1.0)
            .build()
        )
        constraints = ConstraintSet().add_budget_limit(2500.0).add_group_coherence()

        result = opt.solve(objective, constraints, candidates)

        # grp_B has best group ratio (1200*2 / 2000 = 1.2)
        # grp_A has ratio (400*2 / 1000 = 0.8)
        # A5 singleton ratio = 0.5
        # Budget 2500 allows grp_B (2000) + A5 (300)
        assert len(result.selected_actions) == 3
        assert set(result.selected_actions["asset_id"]) == {"A3", "A4", "A5"}

    def test_optimizer_without_group_coherence_unchanged(self):
        """Optimizer without group coherence constraint works as before."""
        opt = Optimizer()
        candidates = pd.DataFrame(
            {
                "asset_id": ["A1", "A2", "A3"],
                "action_type": ["repair", "repair", "replace"],
                "direct_cost": [1000.0, 1000.0, 2000.0],
                "expected_benefit": [800.0, 600.0, 1500.0],
                "group_id": ["group_1", "group_1", "group_2"],
            }
        )
        objective = (
            ObjectiveBuilder()
            .add_expected_risk_reduction()
            .add_total_cost(weight=-1.0)
            .build()
        )
        # No group coherence constraint
        constraints = ConstraintSet().add_budget_limit(2500.0)

        result = opt.solve(objective, constraints, candidates)

        # Without group coherence, optimizer picks best individual assets
        # A1 ratio=0.8, A2 ratio=0.6, A3 ratio=0.75
        # Picks A1, A3 (total 3000) but budget is 2500, so only A1, A2
        assert len(result.selected_actions) == 2
        assert set(result.selected_actions["asset_id"]) == {"A1", "A2"}
