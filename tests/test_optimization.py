"""Tests for optimization module."""

import pytest
import pandas as pd
import numpy as np

from asset_optimization.optimization import Optimizer, OptimizationResult


class TestOptimizerInit:
    """Tests for Optimizer initialization."""

    def test_default_strategy(self):
        """Default strategy is 'greedy'."""
        opt = Optimizer()
        assert opt.strategy == "greedy"

    def test_default_min_risk_threshold(self):
        """Default min_risk_threshold is 0.0."""
        opt = Optimizer()
        assert opt.min_risk_threshold == 0.0

    def test_custom_strategy(self):
        """Can set custom strategy."""
        opt = Optimizer(strategy="milp")
        assert opt.strategy == "milp"

    def test_custom_threshold(self):
        """Can set custom min_risk_threshold."""
        opt = Optimizer(min_risk_threshold=0.1)
        assert opt.min_risk_threshold == 0.1


class TestOptimizerFit:
    """Tests for Optimizer.fit() method."""

    def test_fit_returns_self(self, optimization_portfolio, weibull_model):
        """fit() returns self for method chaining."""
        portfolio = optimization_portfolio

        opt = Optimizer()
        result = opt.fit(portfolio, weibull_model, budget=100000.0)
        assert result is opt

    def test_fit_sets_result_attribute(self, optimization_portfolio, weibull_model):
        """fit() sets result_ attribute."""
        portfolio = optimization_portfolio

        opt = Optimizer()
        opt.fit(portfolio, weibull_model, budget=100000.0)
        assert hasattr(opt, "result_")
        assert isinstance(opt.result_, OptimizationResult)

    def test_result_property_before_fit_raises(self):
        """Accessing result before fit() raises AttributeError."""
        opt = Optimizer()
        with pytest.raises(AttributeError, match="not been fitted"):
            _ = opt.result

    def test_result_property_after_fit(self, optimization_portfolio, weibull_model):
        """result property returns result_ after fit()."""
        portfolio = optimization_portfolio

        opt = Optimizer()
        opt.fit(portfolio, weibull_model, budget=100000.0)
        assert opt.result is opt.result_

    def test_milp_strategy_raises(self, optimization_portfolio, weibull_model):
        """MILP strategy raises NotImplementedError."""
        portfolio = optimization_portfolio

        opt = Optimizer(strategy="milp")
        with pytest.raises(NotImplementedError, match="MILP"):
            opt.fit(portfolio, weibull_model, budget=100000.0)

    def test_unknown_strategy_raises(self, optimization_portfolio, weibull_model):
        """Unknown strategy raises ValueError."""
        portfolio = optimization_portfolio

        opt = Optimizer(strategy="unknown")
        with pytest.raises(ValueError, match="Unknown strategy"):
            opt.fit(portfolio, weibull_model, budget=100000.0)


class TestBudgetConstraint:
    """Tests for budget constraint enforcement."""

    def test_budget_never_exceeded(self, optimization_portfolio, weibull_model):
        """Total spent never exceeds budget."""
        portfolio = optimization_portfolio

        for budget in [1000, 5000, 10000, 50000, 100000, 500000]:
            opt = Optimizer()
            opt.fit(portfolio, weibull_model, budget=float(budget))
            assert opt.result.total_spent <= budget

    def test_zero_budget_empty_selections(self, optimization_portfolio, weibull_model):
        """Zero budget results in empty selections."""
        portfolio = optimization_portfolio

        opt = Optimizer()
        opt.fit(portfolio, weibull_model, budget=0.0)
        assert len(opt.result.selections) == 0
        assert opt.result.total_spent == 0.0

    def test_budget_utilization_calculated(self, optimization_portfolio, weibull_model):
        """Budget utilization percentage is correctly calculated."""
        portfolio = optimization_portfolio

        opt = Optimizer()
        opt.fit(portfolio, weibull_model, budget=100000.0)

        expected_util = (opt.result.total_spent / 100000.0) * 100
        assert abs(opt.result.utilization_pct - expected_util) < 0.01


class TestGreedyRanking:
    """Tests for greedy selection algorithm."""

    def test_highest_risk_selected_first(self, optimization_portfolio, weibull_model):
        """Highest risk-to-cost ratio asset selected first (rank=1)."""
        portfolio = optimization_portfolio

        opt = Optimizer()
        opt.fit(portfolio, weibull_model, budget=100000.0)

        if len(opt.result.selections) > 0:
            # Oldest asset (A1) should have highest risk, should be rank 1
            first_selected = opt.result.selections[opt.result.selections["rank"] == 1]
            assert len(first_selected) == 1
            # A1 is oldest (1980), should be first
            assert first_selected.iloc[0]["asset_id"] == "A1"

    def test_selections_ordered_by_rank(self, optimization_portfolio, weibull_model):
        """Selections have sequential rank values."""
        portfolio = optimization_portfolio

        opt = Optimizer()
        opt.fit(portfolio, weibull_model, budget=200000.0)

        if len(opt.result.selections) > 1:
            ranks = opt.result.selections["rank"].tolist()
            assert ranks == list(range(1, len(ranks) + 1))

    def test_tie_breaking_oldest_first(self, weibull_model):
        """When risk-to-cost ratio is equal, oldest asset selected first."""
        # Create portfolio with same-age assets for tie scenario
        df = pd.DataFrame(
            {
                "asset_id": ["A1", "A2"],
                "asset_type": ["pipe", "pipe"],
                "material": ["PVC", "PVC"],
                "install_date": pd.to_datetime(["1990-01-01", "2000-01-01"]),
                "diameter_mm": [100, 100],
                "length_m": [100.0, 100.0],
            }
        )
        portfolio = df.copy()
        portfolio["age"] = (
            pd.Timestamp.now() - portfolio["install_date"]
        ).dt.days / 365.25

        opt = Optimizer()
        opt.fit(portfolio, weibull_model, budget=100000.0)

        # A1 is older (1990 vs 2000), should be first
        if len(opt.result.selections) >= 2:
            first = opt.result.selections[opt.result.selections["rank"] == 1].iloc[0]
            assert first["asset_id"] == "A1"


class TestMinRiskThreshold:
    """Tests for minimum risk threshold filtering."""

    def test_low_risk_assets_excluded(self, optimization_portfolio, weibull_model):
        """Assets below min_risk_threshold are not selected."""
        portfolio = optimization_portfolio
        risk_df = weibull_model.transform(portfolio)

        # Get risk of youngest asset (A5, ~5 years old)
        a5_risk = risk_df[risk_df["asset_id"] == "A5"]["failure_probability"].iloc[0]

        # Set threshold above A5's risk
        opt = Optimizer(min_risk_threshold=a5_risk + 0.01)
        opt.fit(portfolio, weibull_model, budget=500000.0)

        selected_ids = opt.result.selections["asset_id"].tolist()
        assert "A5" not in selected_ids

    def test_threshold_zero_includes_all(self, optimization_portfolio, weibull_model):
        """Threshold of 0 includes all assets (if budget allows)."""
        portfolio = optimization_portfolio

        opt = Optimizer(min_risk_threshold=0.0)
        opt.fit(portfolio, weibull_model, budget=500000.0)  # Large budget

        # Should select something
        assert len(opt.result.selections) > 0


class TestExclusionList:
    """Tests for asset exclusion functionality."""

    def test_excluded_assets_not_selected(self, optimization_portfolio, weibull_model):
        """Excluded asset IDs are not in selections."""
        portfolio = optimization_portfolio

        opt = Optimizer()
        opt.fit(portfolio, weibull_model, budget=500000.0, exclusions=["A1", "A2"])

        selected_ids = opt.result.selections["asset_id"].tolist()
        assert "A1" not in selected_ids
        assert "A2" not in selected_ids

    def test_empty_exclusion_list(self, optimization_portfolio, weibull_model):
        """Empty exclusion list excludes nothing."""
        portfolio = optimization_portfolio

        opt1 = Optimizer()
        opt1.fit(portfolio, weibull_model, budget=100000.0, exclusions=[])

        opt2 = Optimizer()
        opt2.fit(portfolio, weibull_model, budget=100000.0, exclusions=None)

        assert len(opt1.result.selections) == len(opt2.result.selections)


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_selections_dataframe_columns(self, optimization_portfolio, weibull_model):
        """Selections DataFrame has required columns."""
        portfolio = optimization_portfolio

        opt = Optimizer()
        opt.fit(portfolio, weibull_model, budget=100000.0)

        required_cols = ["asset_id", "intervention_type", "cost", "risk_score", "rank"]
        for col in required_cols:
            assert col in opt.result.selections.columns

    def test_selections_include_risk_details(
        self, optimization_portfolio, weibull_model
    ):
        """Selections include risk_before, risk_after, and risk_reduction."""
        portfolio = optimization_portfolio

        opt = Optimizer()
        opt.fit(portfolio, weibull_model, budget=100000.0)

        selections = opt.result.selections
        for col in ["risk_before", "risk_after", "risk_reduction"]:
            assert col in selections.columns

        if len(selections) > 0:
            row = selections.iloc[0]
            assert np.isclose(
                row["risk_reduction"],
                row["risk_before"] - row["risk_after"],
            )

    def test_budget_summary_columns(self, optimization_portfolio, weibull_model):
        """Budget summary DataFrame has required columns."""
        portfolio = optimization_portfolio

        opt = Optimizer()
        opt.fit(portfolio, weibull_model, budget=100000.0)

        required_cols = ["budget", "spent", "remaining", "utilization_pct"]
        for col in required_cols:
            assert col in opt.result.budget_summary.columns

    def test_strategy_recorded(self, optimization_portfolio, weibull_model):
        """Strategy is recorded in result."""
        portfolio = optimization_portfolio

        opt = Optimizer(strategy="greedy")
        opt.fit(portfolio, weibull_model, budget=100000.0)
        assert opt.result.strategy == "greedy"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_portfolio(self, weibull_model):
        """Empty portfolio returns empty selections."""
        portfolio = pd.DataFrame(
            {
                "asset_id": pd.Series([], dtype=str),
                "asset_type": pd.Series([], dtype=str),
                "material": pd.Series([], dtype=str),
                "install_date": pd.Series([], dtype="datetime64[ns]"),
                "diameter_mm": pd.Series([], dtype="Int64"),
                "length_m": pd.Series([], dtype=float),
                "condition_score": pd.Series([], dtype=float),
            }
        )

        opt = Optimizer()
        opt.fit(portfolio, weibull_model, budget=100000.0)

        assert len(opt.result.selections) == 0
        assert opt.result.total_spent == 0.0

    def test_single_asset(self, weibull_model):
        """Single asset portfolio works correctly."""
        df = pd.DataFrame(
            {
                "asset_id": ["A1"],
                "asset_type": ["pipe"],
                "material": ["PVC"],
                "install_date": pd.to_datetime(["1990-01-01"]),
                "diameter_mm": [100],
                "length_m": [100.0],
            }
        )
        portfolio = df.copy()
        portfolio["age"] = (
            pd.Timestamp.now() - portfolio["install_date"]
        ).dt.days / 365.25

        opt = Optimizer()
        opt.fit(portfolio, weibull_model, budget=100000.0)

        # Should select the single asset
        assert len(opt.result.selections) == 1

    def test_budget_too_small_for_any(self, optimization_portfolio, weibull_model):
        """Budget smaller than any intervention cost returns empty."""
        portfolio = optimization_portfolio

        # Inspect is cheapest at $500
        opt = Optimizer()
        opt.fit(portfolio, weibull_model, budget=100.0)  # Less than Inspect cost

        assert len(opt.result.selections) == 0
