"""Visualization utilities with SDK-specific theme."""

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from asset_optimization.simulation.result import SimulationResult

# SDK Theme Colors - Professional blue palette
SDK_COLORS = {
    'primary': '#2563eb',      # Blue 600
    'secondary': '#64748b',    # Slate 500
    'success': '#16a34a',      # Green 600
    'warning': '#ea580c',      # Orange 600
    'danger': '#dc2626',       # Red 600
    'background': '#f8fafc',   # Slate 50
    'grid': '#e2e8f0',         # Slate 200
    'text': '#1e293b',         # Slate 800
}

SDK_PALETTE = [
    SDK_COLORS['primary'],
    SDK_COLORS['warning'],
    SDK_COLORS['success'],
    SDK_COLORS['danger'],
    SDK_COLORS['secondary'],
]


def set_sdk_theme() -> None:
    """Apply SDK theme to matplotlib and seaborn.

    Sets consistent styling across all plots:
    - Clean white background with subtle grid
    - Professional blue color palette
    - Readable fonts and sizes

    Examples
    --------
    >>> from asset_optimization.visualization import set_sdk_theme
    >>> set_sdk_theme()  # Call once at start of notebook
    """
    # Set seaborn style
    sns.set_theme(style='whitegrid')

    # Custom matplotlib params
    plt.rcParams.update({
        'figure.facecolor': SDK_COLORS['background'],
        'axes.facecolor': 'white',
        'axes.edgecolor': SDK_COLORS['grid'],
        'axes.labelcolor': SDK_COLORS['text'],
        'axes.grid': True,
        'grid.color': SDK_COLORS['grid'],
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'xtick.color': SDK_COLORS['text'],
        'ytick.color': SDK_COLORS['text'],
        'text.color': SDK_COLORS['text'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
    })

    # Set default color palette
    sns.set_palette(SDK_PALETTE)


def plot_cost_over_time(
    result: SimulationResult,
    ax: Optional[plt.Axes] = None,
    title: str = 'Total Cost Over Time',
    figsize: tuple = (10, 6),
) -> plt.Axes:
    """Plot total cost over time as a line chart.

    Parameters
    ----------
    result : SimulationResult
        Simulation result containing summary with year and total_cost columns
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    title : str, default 'Total Cost Over Time'
        Chart title
    figsize : tuple, default (10, 6)
        Figure size if creating new figure

    Returns
    -------
    matplotlib.axes.Axes
        The axes object for further customization

    Examples
    --------
    >>> ax = plot_cost_over_time(result)
    >>> ax.set_ylim(0, 200000)  # Customize
    >>> plt.savefig('cost_chart.png')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    data = result.summary

    sns.lineplot(
        data=data,
        x='year',
        y='total_cost',
        ax=ax,
        marker='o',
        markersize=8,
        linewidth=2,
        color=SDK_COLORS['primary'],
    )

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Cost ($)')

    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
    )

    # Ensure integer years on x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    return ax


def plot_failures_by_year(
    result: SimulationResult,
    ax: Optional[plt.Axes] = None,
    title: str = 'Failures by Year',
    figsize: tuple = (10, 6),
) -> plt.Axes:
    """Plot failure count by year as a bar chart.

    Parameters
    ----------
    result : SimulationResult
        Simulation result containing summary with year and failure_count columns
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    title : str, default 'Failures by Year'
        Chart title
    figsize : tuple, default (10, 6)
        Figure size if creating new figure

    Returns
    -------
    matplotlib.axes.Axes
        The axes object for further customization

    Examples
    --------
    >>> ax = plot_failures_by_year(result)
    >>> plt.savefig('failures_chart.png')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    data = result.summary

    sns.barplot(
        data=data,
        x='year',
        y='failure_count',
        ax=ax,
        color=SDK_COLORS['danger'],
        edgecolor='white',
    )

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Failures')

    # Ensure integer y-axis
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    return ax


def plot_risk_distribution(
    data: pd.DataFrame,
    risk_column: str = 'risk_score',
    ax: Optional[plt.Axes] = None,
    title: str = 'Risk Score Distribution',
    figsize: tuple = (10, 6),
    bins: int = 20,
) -> plt.Axes:
    """Plot distribution of risk scores as a histogram.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing risk scores. Can be:
        - OptimizationResult.selections (with risk_score column)
        - Portfolio data with failure_probability
    risk_column : str, default 'risk_score'
        Name of the column containing risk values
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    title : str, default 'Risk Score Distribution'
        Chart title
    figsize : tuple, default (10, 6)
        Figure size if creating new figure
    bins : int, default 20
        Number of histogram bins

    Returns
    -------
    matplotlib.axes.Axes
        The axes object for further customization

    Examples
    --------
    >>> # From optimization result
    >>> ax = plot_risk_distribution(opt_result.selections)
    >>>
    >>> # From portfolio with failure probability
    >>> ax = plot_risk_distribution(portfolio.data, risk_column='failure_probability')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if risk_column not in data.columns:
        raise ValueError(f"Column '{risk_column}' not found. Available: {list(data.columns)}")

    sns.histplot(
        data=data,
        x=risk_column,
        ax=ax,
        bins=bins,
        color=SDK_COLORS['warning'],
        edgecolor='white',
        kde=True,
    )

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Risk Score')
    ax.set_ylabel('Count')

    # Add vertical line for mean
    mean_risk = data[risk_column].mean()
    ax.axvline(mean_risk, color=SDK_COLORS['danger'], linestyle='--', linewidth=2, label=f'Mean: {mean_risk:.3f}')
    ax.legend()

    plt.tight_layout()
    return ax


def plot_scenario_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'total_cost',
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Axes:
    """Plot scenario comparison as a grouped bar chart.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Long-format DataFrame from compare_scenarios() with columns:
        scenario, year, metric, value
    metric : str, default 'total_cost'
        Which metric to plot. Common options:
        'total_cost', 'failure_count', 'intervention_count'
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    title : str, optional
        Chart title. If None, auto-generated from metric name.
    figsize : tuple, default (10, 6)
        Figure size if creating new figure

    Returns
    -------
    matplotlib.axes.Axes
        The axes object for further customization

    Examples
    --------
    >>> from asset_optimization import compare_scenarios
    >>> comparison = compare_scenarios({'optimized': result1, 'baseline': result2})
    >>> ax = plot_scenario_comparison(comparison, metric='total_cost')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Filter to requested metric
    plot_data = comparison_df[comparison_df['metric'] == metric].copy()

    if plot_data.empty:
        available = comparison_df['metric'].unique().tolist()
        raise ValueError(f"Metric '{metric}' not found. Available: {available}")

    sns.barplot(
        data=plot_data,
        x='year',
        y='value',
        hue='scenario',
        ax=ax,
        palette=SDK_PALETTE[:len(plot_data['scenario'].unique())],
        edgecolor='white',
    )

    if title is None:
        # Convert metric name to title case
        title = metric.replace('_', ' ').title() + ' by Scenario'

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Year')

    # Format y-axis based on metric
    if 'cost' in metric.lower():
        ax.set_ylabel('Cost ($)')
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
        )
    elif 'count' in metric.lower():
        ax.set_ylabel('Count')
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    else:
        ax.set_ylabel(metric.replace('_', ' ').title())

    ax.legend(title='Scenario')
    plt.tight_layout()
    return ax
