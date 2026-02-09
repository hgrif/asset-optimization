"""Budget-constrained optimizer for asset intervention selection.

This module provides the Optimizer class that implements budget-constrained
intervention selection using a greedy algorithm via the solve() method.
"""

import math

import numpy as np
import pandas as pd

from asset_optimization.constraints import ConstraintSet
from asset_optimization.objective import Objective
from asset_optimization.exceptions import OptimizationError
from asset_optimization.types import PlanResult


class Optimizer:
    """Budget-constrained optimizer for asset intervention selection.

    Implements the PlanOptimizer protocol with a greedy budget-fill strategy.

    Parameters
    ----------
    strategy : str, default='greedy'
        Optimization strategy. Valid values: 'greedy', 'milp'.

    Examples
    --------
    >>> from asset_optimization import Optimizer, ObjectiveBuilder, ConstraintSet
    >>> opt = Optimizer()
    >>> result = opt.solve(objective, constraints, candidates)
    """

    def __init__(self, strategy: str = "greedy"):
        self.strategy = strategy

    def solve(
        self,
        objective: Objective,
        constraints: ConstraintSet,
        candidates: pd.DataFrame,
        risk_measure: str = "expected_value",
    ) -> PlanResult:
        """Solve planner candidate selection with a greedy budget rule.

        Parameters
        ----------
        objective : Objective
            Weighted objective terms used for reporting contribution totals.
        constraints : ConstraintSet
            Planner constraints. ``budget_limit`` is honored when present.
        candidates : pd.DataFrame
            Candidate actions with at least ``asset_id``, ``direct_cost``,
            and ``expected_benefit`` columns.
        risk_measure : str, default='expected_value'
            Risk aggregation hint for metadata.

        Returns
        -------
        PlanResult
            Planner-compatible optimization result.
        """
        if not isinstance(candidates, pd.DataFrame):
            raise TypeError("candidates must be a pandas DataFrame")

        required_columns = {"asset_id", "direct_cost", "expected_benefit"}
        missing = sorted(required_columns.difference(candidates.columns))
        if missing:
            raise OptimizationError(
                "candidates missing required columns",
                details={"missing_columns": missing},
            )

        budget_limit = self._extract_budget_limit(constraints)
        ranked = self._prepare_ranked_candidates(candidates)
        selected_actions, remaining_budget = self._select_with_budget(
            ranked, budget_limit
        )
        objective_breakdown = self._compute_objective_breakdown(
            objective, selected_actions
        )

        budget_spent = float(selected_actions["direct_cost"].sum())
        if math.isfinite(budget_limit):
            constraint_shadow_prices = {"budget_limit": 0.0}
            budget_limit_meta = float(budget_limit)
            budget_remaining_meta = float(max(remaining_budget, 0.0))
        else:
            constraint_shadow_prices = {}
            budget_limit_meta = None
            budget_remaining_meta = None

        metadata = {
            "risk_measure": risk_measure,
            "candidate_count": int(len(candidates)),
            "selected_count": int(len(selected_actions)),
            "budget_limit": budget_limit_meta,
            "budget_spent": budget_spent,
            "budget_remaining": budget_remaining_meta,
        }

        return PlanResult(
            selected_actions=selected_actions,
            objective_breakdown=objective_breakdown,
            constraint_shadow_prices=constraint_shadow_prices,
            metadata=metadata,
        )

    @staticmethod
    def _extract_budget_limit(constraints: ConstraintSet) -> float:
        budget_constraints = constraints.find("budget_limit")
        if not budget_constraints:
            return float("inf")

        annual_capex = budget_constraints[-1].params.get("annual_capex")
        try:
            budget = float(annual_capex)
        except (TypeError, ValueError) as exc:
            raise OptimizationError(
                "budget_limit must define numeric annual_capex",
                details={"annual_capex": annual_capex},
            ) from exc

        if not math.isfinite(budget) or budget < 0:
            raise OptimizationError(
                "budget_limit annual_capex must be finite and non-negative",
                details={"annual_capex": annual_capex},
            )

        return budget

    @staticmethod
    def _prepare_ranked_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
        working = candidates.copy(deep=True)
        numeric_columns = ["direct_cost", "expected_benefit"]
        for column in numeric_columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

        if working[numeric_columns].isna().any().any():
            raise OptimizationError(
                "candidates direct_cost and expected_benefit must be numeric",
                details={"columns": numeric_columns},
            )

        if (working["direct_cost"] < 0).any():
            raise OptimizationError("candidates direct_cost must be non-negative")

        cost = working["direct_cost"].to_numpy(dtype=float, copy=False)
        benefit = working["expected_benefit"].to_numpy(dtype=float, copy=False)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(
                benefit,
                cost,
                out=np.zeros(len(working), dtype=float),
                where=cost > 0,
            )
        ratio = np.where((cost <= 0) & (benefit > 0), np.inf, ratio)
        ratio = np.where((cost <= 0) & (benefit <= 0), 0.0, ratio)

        working["benefit_cost_ratio"] = ratio

        if "action_type" not in working.columns:
            if "intervention_type" in working.columns:
                working["action_type"] = working["intervention_type"]
            else:
                working["action_type"] = "action"

        ranked = working.sort_values(
            by=["benefit_cost_ratio", "expected_benefit", "direct_cost"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        return ranked

    @staticmethod
    def _select_with_budget(
        ranked_candidates: pd.DataFrame, budget_limit: float
    ) -> tuple[pd.DataFrame, float]:
        if ranked_candidates.empty:
            return ranked_candidates.copy(), budget_limit

        selected_rows: list[pd.Series] = []
        remaining_budget = budget_limit
        unbounded_budget = not math.isfinite(budget_limit)

        for _, row in ranked_candidates.iterrows():
            cost = float(row["direct_cost"])
            if unbounded_budget or cost <= remaining_budget:
                selected_rows.append(row)
                if not unbounded_budget:
                    remaining_budget -= cost

        selected_actions = (
            pd.DataFrame(selected_rows).reset_index(drop=True)
            if selected_rows
            else ranked_candidates.iloc[0:0].copy().reset_index(drop=True)
        )
        selected_actions["rank"] = np.arange(1, len(selected_actions) + 1, dtype=int)

        ordered_columns = [
            "asset_id",
            "action_type",
            "direct_cost",
            "expected_benefit",
            "benefit_cost_ratio",
            "rank",
        ]
        remaining_columns = [
            column
            for column in selected_actions.columns
            if column not in set(ordered_columns)
        ]
        return selected_actions[ordered_columns + remaining_columns], remaining_budget

    @staticmethod
    def _compute_objective_breakdown(
        objective: Objective, selected_actions: pd.DataFrame
    ) -> dict[str, float]:
        breakdown: dict[str, float] = {}
        for term in objective.terms:
            if term.kind == "expected_risk_reduction":
                value = (
                    selected_actions["expected_risk_reduction"].sum()
                    if "expected_risk_reduction" in selected_actions.columns
                    else selected_actions["expected_benefit"].sum()
                )
            elif term.kind == "total_cost":
                value = selected_actions["direct_cost"].sum()
            elif term.kind == "resilience_gain":
                value = (
                    selected_actions["resilience_gain"].sum()
                    if "resilience_gain" in selected_actions.columns
                    else 0.0
                )
            elif term.kind == "equity_term":
                if "equity_term" in selected_actions.columns:
                    value = selected_actions["equity_term"].sum()
                elif "equity_score" in selected_actions.columns:
                    value = selected_actions["equity_score"].sum()
                else:
                    value = 0.0
            elif term.kind in selected_actions.columns:
                value = (
                    pd.to_numeric(selected_actions[term.kind], errors="coerce")
                    .fillna(0.0)
                    .sum()
                )
            else:
                value = 0.0

            breakdown[term.kind] = float(term.weight * float(value))

        breakdown["objective_total"] = float(sum(breakdown.values()))
        return breakdown

    def __repr__(self) -> str:
        return f"Optimizer(strategy='{self.strategy}')"
