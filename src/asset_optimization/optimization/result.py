"""Optimization result dataclass."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class OptimizationResult:
    """Results from optimization run.

    Attributes
    ----------
    selections : pd.DataFrame
        Selected interventions with columns:
        - asset_id: str
        - intervention_type: str ('Replace', 'Repair', 'Inspect')
        - cost: float
        - risk_score: float (P(failure) at current age)
        - rank: int (position in greedy selection order)
    budget_summary : pd.DataFrame
        Single-row DataFrame with columns:
        - budget: float (total annual budget)
        - spent: float (total cost of selected interventions)
        - remaining: float (budget - spent)
        - utilization_pct: float (spent / budget * 100)
    strategy : str
        Strategy used ('greedy', 'milp')

    Examples
    --------
    >>> import pandas as pd
    >>> from asset_optimization.optimization.result import OptimizationResult
    >>> selections = pd.DataFrame({
    ...     'asset_id': ['A1', 'A2'],
    ...     'intervention_type': ['Replace', 'Repair'],
    ...     'cost': [5000.0, 1000.0],
    ...     'risk_score': [0.8, 0.3],
    ...     'rank': [1, 2],
    ... })
    >>> budget_summary = pd.DataFrame({
    ...     'budget': [10000.0],
    ...     'spent': [6000.0],
    ...     'remaining': [4000.0],
    ...     'utilization_pct': [60.0],
    ... })
    >>> result = OptimizationResult(
    ...     selections=selections,
    ...     budget_summary=budget_summary,
    ...     strategy='greedy',
    ... )
    >>> result.total_spent
    6000.0
    >>> result.utilization_pct
    60.0
    """

    selections: pd.DataFrame
    budget_summary: pd.DataFrame
    strategy: str

    @property
    def total_spent(self) -> float:
        """Get total amount spent from budget summary.

        Returns
        -------
        float
            Total cost of selected interventions.
            Returns 0.0 if budget_summary is empty or column missing.
        """
        if self.budget_summary.empty or 'spent' not in self.budget_summary.columns:
            return 0.0
        return float(self.budget_summary['spent'].iloc[0])

    @property
    def utilization_pct(self) -> float:
        """Get budget utilization percentage.

        Returns
        -------
        float
            Percentage of budget used (spent / budget * 100).
            Returns 0.0 if budget_summary is empty or column missing.
        """
        if self.budget_summary.empty or 'utilization_pct' not in self.budget_summary.columns:
            return 0.0
        return float(self.budget_summary['utilization_pct'].iloc[0])

    def __repr__(self) -> str:
        """Rich representation showing key metrics."""
        n_selected = len(self.selections)
        spent = self.total_spent
        utilization = self.utilization_pct

        return (
            f"OptimizationResult("
            f"strategy='{self.strategy}', "
            f"selected={n_selected} assets, "
            f"spent=${spent:,.0f}, "
            f"utilization={utilization:.1f}%)"
        )
