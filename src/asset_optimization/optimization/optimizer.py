"""Budget-constrained optimizer for asset intervention selection.

This module provides the Optimizer class that implements budget-constrained
intervention selection using a two-stage greedy algorithm.
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import weibull_min

from asset_optimization.optimization.result import OptimizationResult
from asset_optimization.simulation import DO_NOTHING, INSPECT, REPAIR, REPLACE
from asset_optimization.exceptions import OptimizationError
from asset_optimization.portfolio import validate_portfolio

if TYPE_CHECKING:
    from asset_optimization.models.weibull import WeibullModel


class Optimizer:
    """Budget-constrained optimizer for asset intervention selection.

    Implements a scikit-learn-style API with fit() returning self.
    Supports 'greedy' strategy (default) and 'milp' strategy (planned).

    Parameters
    ----------
    strategy : str, default='greedy'
        Optimization strategy. Valid values: 'greedy', 'milp'.
        - 'greedy': Two-stage greedy algorithm (fast, good approximation)
        - 'milp': Mixed-integer linear programming (planned, raises NotImplementedError)
    min_risk_threshold : float, default=0.0
        Minimum failure probability to consider for intervention.
        Assets below this threshold are excluded from optimization.

    Attributes
    ----------
    strategy : str
        Optimization strategy used.
    min_risk_threshold : float
        Minimum risk threshold.
    result_ : OptimizationResult
        Results from optimization (only available after fit()).

    Examples
    --------
    >>> import pandas as pd
    >>> from asset_optimization import WeibullModel, Optimizer
    >>> portfolio = pd.read_csv('assets.csv', parse_dates=['install_date'])
    >>> model = WeibullModel({'PVC': (2.5, 50.0)})
    >>> opt = Optimizer(strategy='greedy', min_risk_threshold=0.1)
    >>> opt.fit(portfolio, model, budget=100000.0)
    >>> print(opt.result.selections)
    """

    # Define available interventions (excluding DoNothing which is not an active choice)
    _INTERVENTIONS = [REPLACE, REPAIR, INSPECT]

    def __init__(self, strategy: str = "greedy", min_risk_threshold: float = 0.0):
        """Initialize optimizer with strategy and threshold.

        Parameters
        ----------
        strategy : str, default='greedy'
            Optimization strategy.
        min_risk_threshold : float, default=0.0
            Minimum failure probability for intervention consideration.
        """
        self.strategy = strategy
        self.min_risk_threshold = min_risk_threshold

    def fit(
        self,
        portfolio: pd.DataFrame,
        model: "WeibullModel",
        budget: float,
        exclusions: list[str] | None = None,
    ) -> "Optimizer":
        """Select interventions within budget.

        Parameters
        ----------
        portfolio : pd.DataFrame
            Asset portfolio data.
        model : WeibullModel
            Fitted deterioration model (used for risk_after calculation).
        budget : float
            Annual budget (strict upper bound, never exceeded).
        exclusions : list[str], optional
            Asset IDs to skip entirely.

        Returns
        -------
        self
            Fitted optimizer. Access results via result_ attribute.

        Raises
        ------
        ValueError
            If strategy is unknown or budget is negative.
        NotImplementedError
            If strategy is 'milp' (planned feature).
        OptimizationError
            If portfolio data is invalid.
        """
        # Validate strategy
        valid_strategies = ["greedy", "milp"]
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. "
                f"Valid strategies: {valid_strategies}"
            )

        # Handle MILP strategy
        if self.strategy == "milp":
            raise NotImplementedError(
                "MILP strategy is planned but not yet implemented. "
                "Use 'greedy' strategy for now."
            )

        # Validate budget
        if budget < 0:
            raise ValueError(f"Budget must be non-negative, got {budget}")

        validated = validate_portfolio(portfolio)

        # Dispatch to strategy implementation
        if self.strategy == "greedy":
            self._fit_greedy(validated, model, budget, exclusions)

        return self

    def _fit_greedy(
        self,
        portfolio: pd.DataFrame,
        model: "WeibullModel",
        budget: float,
        exclusions: list[str] | None = None,
    ) -> None:
        """Implement two-stage greedy algorithm.

        Stage 1: For each asset, find the best intervention (highest cost-effectiveness).
        Stage 2: Rank all candidates by risk-to-cost ratio and greedily fill budget.

        Parameters
        ----------
        portfolio : pd.DataFrame
            Asset portfolio data.
        model : WeibullModel
            Deterioration model for risk calculations.
        budget : float
            Budget constraint.
        exclusions : list[str], optional
            Asset IDs to exclude.
        """
        exclusions = exclusions or []

        # Handle empty portfolio
        if len(portfolio) == 0:
            self.result_ = self._build_empty_result(budget)
            return

        # Handle zero budget
        if budget <= 0:
            self.result_ = self._build_empty_result(budget)
            return

        # ============================================================
        # Stage 1: Best intervention per asset
        # ============================================================

        # Copy portfolio data to working DataFrame
        df = portfolio.copy()

        # CRITICAL: Ensure 'age' column exists BEFORE calling model.transform()
        # The transform() method validates 'age' exists
        if "age" not in df.columns:
            df["age"] = (pd.Timestamp.now() - df["install_date"]).dt.days / 365.25

        # Now call model.transform() to add failure_probability
        df = model.transform(df)

        # Validate required column exists
        if "failure_probability" not in df.columns:
            raise OptimizationError(
                "Portfolio data must have 'failure_probability' column after transform",
                details={"columns": df.columns.tolist()},
            )

        # Apply exclusion filter
        if exclusions:
            df = df[~df["asset_id"].isin(exclusions)].copy()

        # Handle case where all assets are excluded
        if df.empty:
            self.result_ = self._build_empty_result(budget)
            return

        # Find best intervention for each asset
        candidates = []

        for idx, row in df.iterrows():
            asset_id = row["asset_id"]
            current_age = row["age"]
            risk_before = row["failure_probability"]
            material = row[model.type_column]

            # Get Weibull parameters for this asset type
            shape, scale = model.params[material]

            best_intervention = None
            best_cost_effectiveness = -np.inf

            for intervention in self._INTERVENTIONS:
                # Calculate new age after intervention
                new_age = intervention.apply_age_effect(current_age)

                # Calculate risk after intervention
                risk_after = weibull_min.cdf(new_age, c=shape, scale=scale)

                # Calculate cost effectiveness (risk reduction per dollar)
                risk_reduction = risk_before - risk_after
                cost = intervention.cost

                # Skip DoNothing implicitly (it's not in _INTERVENTIONS)
                # Inspect has cost_effectiveness=0 (no risk reduction)
                if cost > 0:
                    cost_effectiveness = risk_reduction / cost
                else:
                    # Edge case: free intervention (shouldn't happen with standard interventions)
                    cost_effectiveness = (
                        risk_reduction * 1e10 if risk_reduction > 0 else 0
                    )

                if cost_effectiveness > best_cost_effectiveness:
                    best_cost_effectiveness = cost_effectiveness
                    best_intervention = intervention
                    best_risk_after = risk_after

            # Store candidate with best intervention
            if best_intervention is not None:
                candidates.append(
                    {
                        "asset_id": asset_id,
                        "intervention_type": best_intervention.name,
                        "cost": best_intervention.cost,
                        "risk_before": risk_before,
                        "risk_after": best_risk_after,
                        "cost_effectiveness": best_cost_effectiveness,
                        "install_date": row["install_date"],
                    }
                )

        # Handle case where no candidates found
        if not candidates:
            self.result_ = self._build_empty_result(budget)
            return

        candidates_df = pd.DataFrame(candidates)

        # ============================================================
        # Stage 2: Rank and fill budget
        # ============================================================

        # Filter: remove assets below min_risk_threshold
        if self.min_risk_threshold > 0:
            candidates_df = candidates_df[
                candidates_df["risk_before"] >= self.min_risk_threshold
            ].copy()

        # Handle case where all assets below threshold
        if candidates_df.empty:
            self.result_ = self._build_empty_result(budget)
            return

        # Filter: remove DoNothing (shouldn't happen, but safety)
        candidates_df = candidates_df[
            candidates_df["intervention_type"] != DO_NOTHING.name
        ].copy()

        # Compute risk-to-cost ratio for ranking
        candidates_df["risk_to_cost_ratio"] = (
            candidates_df["risk_before"] / candidates_df["cost"]
        )

        # Sort: risk_to_cost_ratio DESC, then install_date ASC (oldest first tie-breaker)
        candidates_df = candidates_df.sort_values(
            by=["risk_to_cost_ratio", "install_date"], ascending=[False, True]
        ).reset_index(drop=True)

        # Greedy fill: iterate and add if cost <= remaining budget
        selections = []
        remaining_budget = budget
        rank = 0

        for _, row in candidates_df.iterrows():
            cost = row["cost"]
            if cost <= remaining_budget:
                rank += 1
                selections.append(
                    {
                        "asset_id": row["asset_id"],
                        "intervention_type": row["intervention_type"],
                        "cost": cost,
                        "risk_score": row["risk_before"],
                        "risk_before": row["risk_before"],
                        "risk_after": row["risk_after"],
                        "risk_reduction": row["risk_before"] - row["risk_after"],
                        "rank": rank,
                    }
                )
                remaining_budget -= cost

        # Build result
        if selections:
            selections_df = pd.DataFrame(selections)
        else:
            selections_df = pd.DataFrame(
                columns=[
                    "asset_id",
                    "intervention_type",
                    "cost",
                    "risk_score",
                    "risk_before",
                    "risk_after",
                    "risk_reduction",
                    "rank",
                ]
            )

        spent = budget - remaining_budget
        utilization_pct = (spent / budget * 100) if budget > 0 else 0.0

        budget_summary = pd.DataFrame(
            {
                "budget": [budget],
                "spent": [spent],
                "remaining": [remaining_budget],
                "utilization_pct": [utilization_pct],
            }
        )

        self.result_ = OptimizationResult(
            selections=selections_df,
            budget_summary=budget_summary,
            strategy="greedy",
        )

    def _build_empty_result(self, budget: float) -> OptimizationResult:
        """Build an empty result for edge cases.

        Parameters
        ----------
        budget : float
            Budget amount (for summary).

        Returns
        -------
        OptimizationResult
            Result with empty selections.
        """
        selections_df = pd.DataFrame(
            columns=[
                "asset_id",
                "intervention_type",
                "cost",
                "risk_score",
                "risk_before",
                "risk_after",
                "risk_reduction",
                "rank",
            ]
        )

        budget_summary = pd.DataFrame(
            {
                "budget": [budget],
                "spent": [0.0],
                "remaining": [budget],
                "utilization_pct": [0.0],
            }
        )

        return OptimizationResult(
            selections=selections_df,
            budget_summary=budget_summary,
            strategy=self.strategy,
        )

    @property
    def result(self) -> OptimizationResult:
        """Access optimization result.

        Returns
        -------
        OptimizationResult
            Results from optimization.

        Raises
        ------
        AttributeError
            If optimizer has not been fitted.
        """
        if not hasattr(self, "result_"):
            raise AttributeError("Optimizer has not been fitted. Call fit() first.")
        return self.result_

    def __repr__(self) -> str:
        """Return informative string representation."""
        fitted = hasattr(self, "result_")
        return (
            f"Optimizer(strategy='{self.strategy}', "
            f"min_risk_threshold={self.min_risk_threshold}, "
            f"fitted={fitted})"
        )
