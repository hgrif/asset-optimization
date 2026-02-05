"""Scenario comparison utilities."""

from typing import Dict, List, Optional, Union

import pandas as pd

from asset_optimization.simulation.result import SimulationResult


def compare_scenarios(
    scenarios: Dict[str, SimulationResult],
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compare multiple simulation scenarios side-by-side.

    Parameters
    ----------
    scenarios : dict of str to SimulationResult
        Dictionary mapping scenario names to their simulation results.
        Example: {'optimized': result1, 'do_nothing': result2}
    metrics : list of str, optional
        Metrics to include in comparison. If None, defaults to:
        ['total_cost', 'failure_count', 'intervention_count']

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: scenario, year, metric, value
        Suitable for grouped bar charts and faceted plots.

    Examples
    --------
    >>> from asset_optimization import compare_scenarios
    >>> comparison = compare_scenarios({
    ...     'optimized': optimized_result,
    ...     'do_nothing': baseline_result,
    ... })
    >>> comparison.head()
       scenario  year        metric     value
    0  optimized  2024    total_cost  100000.0
    1  optimized  2024  failure_count       5.0

    Notes
    -----
    All scenarios must have the same years in their summary DataFrames.
    Missing metrics in a scenario will be filled with NaN.
    """
    if metrics is None:
        metrics = ['total_cost', 'failure_count', 'intervention_count']

    dfs = []
    for name, result in scenarios.items():
        summary = result.summary.copy()

        # Filter to requested metrics that exist
        available_metrics = [m for m in metrics if m in summary.columns]

        if not available_metrics:
            continue

        # Melt to long format
        melted = summary.melt(
            id_vars=['year'],
            value_vars=available_metrics,
            var_name='metric',
            value_name='value',
        )
        melted['scenario'] = name
        dfs.append(melted)

    if not dfs:
        return pd.DataFrame(columns=['scenario', 'year', 'metric', 'value'])

    result_df = pd.concat(dfs, ignore_index=True)

    # Reorder columns
    return result_df[['scenario', 'year', 'metric', 'value']]


def create_do_nothing_baseline(
    result: SimulationResult,
    failure_cost_multiplier: float = 1.5,
) -> SimulationResult:
    """Create a 'do nothing' baseline from an existing simulation result.

    Estimates what would happen with no interventions by:
    - Setting intervention_count to 0
    - Increasing failure_count based on risk profile
    - Adjusting total_cost to reflect only failure costs (no intervention costs)

    Parameters
    ----------
    result : SimulationResult
        A simulation result to use as basis for baseline.
        The result should contain a summary with yearly metrics.
    failure_cost_multiplier : float, default 1.5
        Multiplier applied to failure costs to estimate 'do nothing' scenario.
        Higher values represent more conservative (worse) baselines.

    Returns
    -------
    SimulationResult
        A new SimulationResult representing the 'do nothing' scenario.
        The config is copied from the input result.

    Examples
    --------
    >>> baseline = create_do_nothing_baseline(optimized_result)
    >>> comparison = compare_scenarios({
    ...     'optimized': optimized_result,
    ...     'do_nothing': baseline,
    ... })

    Notes
    -----
    This is a rough estimation. For accurate 'do nothing' scenarios,
    run a full simulation with no intervention budget.
    """
    summary = result.summary.copy()

    # Estimate 'do nothing' by:
    # 1. Zero interventions
    if 'intervention_count' in summary.columns:
        summary['intervention_count'] = 0

    # 2. Increase failures (rough heuristic: more failures without interventions)
    if 'failure_count' in summary.columns:
        # Scale up failures progressively over time
        years = range(len(summary))
        multipliers = [1 + 0.1 * y * failure_cost_multiplier for y in years]
        summary['failure_count'] = (summary['failure_count'] * multipliers).astype(int)

    # 3. Adjust total cost (remove intervention savings, add failure costs)
    if 'total_cost' in summary.columns and 'failure_count' in summary.columns:
        # Estimate cost per failure from original data
        original_failures = result.summary['failure_count'].sum()
        if original_failures > 0:
            avg_failure_cost = result.total_cost() / max(original_failures, 1)
        else:
            avg_failure_cost = 50000  # Default assumption

        # Cost = failure_count * avg_failure_cost (no intervention costs)
        summary['total_cost'] = summary['failure_count'] * avg_failure_cost * failure_cost_multiplier

    # 4. Age increases faster without replacements
    if 'avg_age' in summary.columns:
        # Age increases by 1 each year plus accumulated effect
        base_age = summary['avg_age'].iloc[0] if len(summary) > 0 else 25
        summary['avg_age'] = [base_age + i * 1.2 for i in range(len(summary))]

    asset_history = pd.DataFrame(columns=[
        'year',
        'asset_id',
        'age',
        'action',
        'failed',
        'failure_cost',
        'intervention_cost',
        'total_cost',
    ])

    return SimulationResult(
        summary=summary,
        cost_breakdown=pd.DataFrame(),  # Not applicable for baseline
        failure_log=pd.DataFrame(),  # Not applicable for baseline
        config=result.config,
        asset_history=asset_history,
    )


def compare(
    result: SimulationResult,
    baseline: Union[SimulationResult, str] = 'do_nothing',
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compare a simulation result against a baseline.

    Convenience function that wraps compare_scenarios with automatic
    baseline generation.

    Parameters
    ----------
    result : SimulationResult
        The primary simulation result (e.g., optimized scenario)
    baseline : SimulationResult or str, default 'do_nothing'
        Either a SimulationResult to compare against, or 'do_nothing'
        to auto-generate a baseline using create_do_nothing_baseline.
    metrics : list of str, optional
        Metrics to include. Defaults to ['total_cost', 'failure_count', 'intervention_count']

    Returns
    -------
    pd.DataFrame
        Long-format comparison DataFrame with columns: scenario, year, metric, value

    Examples
    --------
    >>> # Auto-generate 'do nothing' baseline
    >>> comparison = compare(optimized_result, baseline='do_nothing')
    >>>
    >>> # Use explicit baseline
    >>> comparison = compare(optimized_result, baseline=manual_baseline)
    """
    if isinstance(baseline, str) and baseline == 'do_nothing':
        baseline_result = create_do_nothing_baseline(result)
    elif isinstance(baseline, SimulationResult):
        baseline_result = baseline
    else:
        raise ValueError(f"baseline must be SimulationResult or 'do_nothing', got {type(baseline)}")

    return compare_scenarios(
        {'optimized': result, 'baseline': baseline_result},
        metrics=metrics,
    )
