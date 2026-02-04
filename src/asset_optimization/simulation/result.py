"""Simulation result dataclass."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import pandas as pd

if TYPE_CHECKING:
    from asset_optimization.simulation.config import SimulationConfig


@dataclass
class SimulationResult:
    """Results from simulation run.

    Attributes
    ----------
    summary : pd.DataFrame
        Summary statistics per year with columns:
        - year, total_cost, failure_count, intervention_count, avg_age
    cost_breakdown : pd.DataFrame
        Cost breakdown by intervention type and asset type
    failure_log : pd.DataFrame
        Event log of all failures with columns:
        - year, asset_id, age_at_failure, material, direct_cost, consequence_cost
    config : SimulationConfig
        Configuration used for this run (for reproducibility)
    asset_history : pd.DataFrame, optional
        Full asset-level traces (only if config.track_asset_history=True)

    Examples
    --------
    >>> import pandas as pd
    >>> from asset_optimization.simulation import SimulationConfig, SimulationResult
    >>> config = SimulationConfig(n_years=10)
    >>> summary = pd.DataFrame({
    ...     'year': [2026, 2027],
    ...     'total_cost': [100000.0, 120000.0],
    ...     'failure_count': [5, 7],
    ... })
    >>> result = SimulationResult(
    ...     summary=summary,
    ...     cost_breakdown=pd.DataFrame(),
    ...     failure_log=pd.DataFrame(),
    ...     config=config,
    ... )
    >>> result.total_cost()
    220000.0
    >>> result.total_failures()
    12
    """

    summary: pd.DataFrame
    cost_breakdown: pd.DataFrame
    failure_log: pd.DataFrame
    config: 'SimulationConfig'
    asset_history: Optional[pd.DataFrame] = None

    def total_cost(self) -> float:
        """Calculate total cost across all simulation years.

        Returns
        -------
        float
            Sum of total_cost column in summary DataFrame.
            Returns 0.0 if summary is empty or column missing.
        """
        if self.summary.empty or 'total_cost' not in self.summary.columns:
            return 0.0
        return float(self.summary['total_cost'].sum())

    def total_failures(self) -> int:
        """Calculate total failures across all simulation years.

        Returns
        -------
        int
            Sum of failure_count column in summary DataFrame.
            Returns 0 if summary is empty or column missing.
        """
        if self.summary.empty or 'failure_count' not in self.summary.columns:
            return 0
        return int(self.summary['failure_count'].sum())

    def to_parquet(self, path: Union[str, Path], format: str = 'summary') -> None:
        """Export simulation results to parquet.

        Parameters
        ----------
        path : str or Path
            Output path for parquet file
        format : str, default 'summary'
            Export format:
            - 'summary': Year-by-year summary with cost, failures, interventions
            - 'cost_projections': Long format for plotting (year, metric, value)
            - 'failure_log': Detailed failure event log

        Raises
        ------
        ValueError
            If format is not one of the supported options.

        Examples
        --------
        >>> result.to_parquet('simulation_summary.parquet')
        >>> result.to_parquet('projections.parquet', format='cost_projections')
        """
        from asset_optimization.exports import export_cost_projections

        if format == 'summary':
            self.summary.to_parquet(path, index=False)
        elif format == 'cost_projections':
            export_cost_projections(self.summary, path)
        elif format == 'failure_log':
            self.failure_log.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'summary', 'cost_projections', or 'failure_log'")

    def __repr__(self) -> str:
        """Rich representation showing key metrics."""
        n_years = len(self.summary)
        if n_years > 0 and 'year' in self.summary.columns:
            start_year = self.summary['year'].min()
            end_year = self.summary['year'].max()
            year_range = f"{start_year}-{end_year}"
        else:
            year_range = "no data"

        total_cost = self.total_cost()
        total_failures = self.total_failures()

        return (
            f"SimulationResult("
            f"years={year_range}, "
            f"total_cost=${total_cost:,.0f}, "
            f"failures={total_failures})"
        )
