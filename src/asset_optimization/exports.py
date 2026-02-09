"""Export utilities for parquet format."""

from pathlib import Path
from typing import Union

import pandas as pd


def export_schedule_minimal(
    selections: pd.DataFrame,
    path: Union[str, Path],
    year: int = 2024,
) -> None:
    """Export intervention schedule in minimal format.

    Parameters
    ----------
    selections : pd.DataFrame
        Selected actions from PlanResult with columns:
        asset_id, action_type, direct_cost, rank
    path : str or Path
        Output path for parquet file
    year : int, default 2024
        Year for the intervention schedule

    Notes
    -----
    Minimal format columns: asset_id, year, action_type, direct_cost
    """
    minimal = pd.DataFrame(
        {
            "asset_id": selections["asset_id"],
            "year": year,
            "action_type": selections["action_type"],
            "direct_cost": selections["direct_cost"],
        }
    )
    minimal.to_parquet(path, index=False)


def export_schedule_detailed(
    selections: pd.DataFrame,
    path: Union[str, Path],
    year: int = 2024,
    portfolio: pd.DataFrame = None,
) -> None:
    """Export intervention schedule in detailed format.

    Parameters
    ----------
    selections : pd.DataFrame
        Selected actions from PlanResult.
    path : str or Path
        Output path for parquet file
    year : int, default 2024
        Year for the intervention schedule
    portfolio : pd.DataFrame, optional
        Portfolio DataFrame to join for material and age columns.
        If None, material and age columns will be omitted.

    Notes
    -----
    Detailed format columns: asset_id, year, action_type, direct_cost,
    expected_benefit, rank, material, age
    """
    detailed = selections[["asset_id", "action_type", "direct_cost", "rank"]].copy()
    detailed["year"] = year

    if "expected_benefit" in selections.columns:
        detailed["expected_benefit"] = selections["expected_benefit"]

    # Join portfolio for material and age if provided
    if portfolio is not None:
        portfolio_cols = portfolio[["asset_id"]].copy()
        if "material" in portfolio.columns:
            portfolio_cols["material"] = portfolio["material"]
        elif "asset_type" in portfolio.columns:
            portfolio_cols["material"] = portfolio["asset_type"]
        if "age" in portfolio.columns:
            portfolio_cols["age"] = portfolio["age"]
        detailed = detailed.merge(portfolio_cols, on="asset_id", how="left")

    # Reorder columns for consistency
    col_order = ["asset_id", "year", "action_type", "direct_cost", "rank"]
    optional_cols = ["expected_benefit", "material", "age"]
    for col in optional_cols:
        if col in detailed.columns:
            col_order.append(col)
    detailed = detailed[col_order]

    detailed.to_parquet(path, index=False)


def export_cost_projections(
    summary: pd.DataFrame,
    path: Union[str, Path],
) -> None:
    """Export cost projections by year in long format.

    Parameters
    ----------
    summary : pd.DataFrame
        Summary DataFrame from SimulationResult with columns:
        year, total_cost, failure_count, intervention_count, avg_age
    path : str or Path
        Output path for parquet file

    Notes
    -----
    Output is in long format with columns: year, metric, value
    This format is easier for plotting with seaborn/matplotlib.
    Includes failure_count metric (OUTP-03: expected failure metrics by year).
    """
    # Melt to long format
    metrics = ["total_cost", "failure_count", "intervention_count", "avg_age"]
    available_metrics = [m for m in metrics if m in summary.columns]

    long_df = summary.melt(
        id_vars=["year"],
        value_vars=available_metrics,
        var_name="metric",
        value_name="value",
    )
    long_df.to_parquet(path, index=False)
