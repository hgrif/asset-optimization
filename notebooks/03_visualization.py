# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Visualization and Export
#
# Learn how to create publication-quality charts and export results.
#
# This notebook demonstrates:
# 1. SDK theme for consistent styling
# 2. Cost over time line charts
# 3. Failures by year bar charts
# 4. Risk distribution histograms
# 5. Scenario comparison charts
# 6. Asset action heatmap
# 7. Customizing and combining charts
#

# %% [markdown]
# ## Setup

# %%
# Core imports
import os
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# SDK imports
from asset_optimization import (
    WeibullModel,
    Simulator,
    SimulationConfig,
    Optimizer,
    set_sdk_theme,
    plot_cost_over_time,
    plot_failures_by_year,
    plot_risk_distribution,
    plot_scenario_comparison,
    plot_asset_action_heatmap,
    compare,
)

# %% [markdown]
# ## 1. Create Sample Data
#
# First, we'll create simulation and optimization results to visualize.

# %%
# Generate synthetic portfolio
np.random.seed(42)

n_assets = 500
materials = ["Cast Iron", "PVC", "Ductile Iron"]

base_date = date(2024, 1, 1)
install_dates = [
    base_date - timedelta(days=int(np.random.uniform(20 * 365, 80 * 365)))
    for _ in range(n_assets)
]

data = pd.DataFrame(
    {
        "asset_id": [f"PIPE-{i:04d}" for i in range(n_assets)],
        "install_date": pd.to_datetime(install_dates),
        "asset_type": "pipe",
        "material": np.random.choice(materials, n_assets, p=[0.4, 0.35, 0.25]),
        "diameter_mm": np.random.choice([150, 200, 300, 400], n_assets),
        "length_m": np.random.uniform(50, 500, n_assets).round(0),
    }
)

portfolio = data
print(portfolio.head())

# %%
# Create deterioration model
params = {
    "Cast Iron": (3.0, 60),
    "PVC": (2.5, 80),
    "Ductile Iron": (2.8, 70),
}
model = WeibullModel(params)

# %%
# Run simulation
config = SimulationConfig(
    n_years=10,
    start_year=2024,
    random_seed=42,
    failure_response="replace",
)

sim = Simulator(model, config)
sim_result = sim.run(portfolio)
print(sim_result)

# %%
sim_result.asset_history

# %%
# Run optimization
optimizer = Optimizer(strategy="greedy", min_risk_threshold=0.1)
optimizer.fit(portfolio, model, budget=500_000)
opt_result = optimizer.result
print(opt_result)

# %%

# %% [markdown]
# ## 2. SDK Theme
#
# All plots use a consistent professional theme. Call `set_sdk_theme()` once at the start of your notebook.
#
# The theme provides:
# - Clean white background with subtle grid
# - Professional blue color palette
# - Readable fonts and sizes

# %%
# Apply SDK theme (call once at notebook start)
set_sdk_theme()
print("SDK theme applied")

# %%
sim_result.asset_history

# %% [markdown]
# ## 3. Cost Over Time
#
# Line chart showing total cost trajectory over the simulation period.

# %%
# Basic cost over time chart
ax = plot_cost_over_time(sim_result)
plt.show()

# %%
# With custom title
ax = plot_cost_over_time(sim_result, title="Projected Maintenance Costs (2024-2033)")
plt.show()

# %% [markdown]
# ## 4. Failures by Year
#
# Bar chart showing failure counts per year.

# %%
# Basic failures chart
ax = plot_failures_by_year(sim_result)
plt.show()

# %%
# With custom title
ax = plot_failures_by_year(sim_result, title="Expected Asset Failures (2024-2033)")
plt.show()

# %% [markdown]
# ## 5. Risk Distribution
#
# Histogram showing the distribution of risk scores for selected interventions.

# %%
# Risk distribution of selected assets
ax = plot_risk_distribution(opt_result.selections)
plt.show()

# %%
# You can also plot failure probability from portfolio data
# First enrich the portfolio with failure probabilities
portfolio_with_risk = portfolio.copy()
portfolio_with_risk["age"] = (
    pd.Timestamp.now() - portfolio_with_risk["install_date"]
).dt.days / 365.25
portfolio_enriched = model.transform(portfolio_with_risk)

# Plot with different column name
ax = plot_risk_distribution(
    portfolio_enriched,
    risk_column="failure_probability",
    title="Portfolio-Wide Failure Probability Distribution",
    bins=30,
)
plt.show()

# %% [markdown]
# ## 6. Scenario Comparison
#
# Compare the optimized scenario against a 'do nothing' baseline.

# %%
# Compare simulation result against auto-generated baseline
comparison = compare(sim_result, baseline="do_nothing")

print("Comparison DataFrame:")
comparison.head(10)

# %%
# Plot total cost comparison
ax = plot_scenario_comparison(comparison, metric="total_cost")
plt.show()

# %%
# Plot failure count comparison
ax = plot_scenario_comparison(comparison, metric="failure_count")
plt.show()

# %% [markdown]
# ## 7. Asset Action Heatmap
#
# Visualize asset actions across years using the asset history captured during simulation.
#

# %%
# Plot asset actions over time
ax = plot_asset_action_heatmap(sim_result, max_assets=50)
plt.show()

# %% [markdown]
# ## 8. Customizing Charts
#
# All plot functions return `matplotlib.axes.Axes` objects for further customization.

# %%
# Get axes and customize
ax = plot_cost_over_time(sim_result)

# Add annotations
ax.set_ylim(0, ax.get_ylim()[1] * 1.1)  # Add 10% headroom
ax.axhline(
    y=sim_result.summary["total_cost"].mean(),
    color="orange",
    linestyle="--",
    alpha=0.7,
    label="Average",
)
ax.legend()

plt.show()

# %%
# Provide your own figure size
ax = plot_failures_by_year(sim_result, figsize=(12, 4))
ax.set_title("Wide Format Chart")
plt.show()

# %% [markdown]
# ## 9. Exporting Results
#
# ### To Parquet

# %%
# Export simulation results
sim_result.to_parquet("sim_summary.parquet", format="summary")
sim_result.to_parquet("sim_projections.parquet", format="cost_projections")

# Export optimization results
opt_result.to_parquet("opt_schedule.parquet", format="minimal", year=2024)

print("Files exported:")
print("  - sim_summary.parquet")
print("  - sim_projections.parquet")
print("  - opt_schedule.parquet")

# %% [markdown]
# ### Reading Exports

# %%
# Read back parquet files
summary = pd.read_parquet("sim_summary.parquet")
print("Simulation Summary:")
summary

# %%
# Long format is ready for seaborn/matplotlib
projections = pd.read_parquet("sim_projections.parquet")
print("Cost Projections (long format):")
projections.head(12)

# %% [markdown]
# ### Saving Charts

# %%
# Save chart to file
ax = plot_cost_over_time(sim_result)
plt.savefig("cost_chart.png", dpi=150, bbox_inches="tight")
print("Saved: cost_chart.png")
plt.close()

# %% [markdown]
# ## 10. Creating Multi-Panel Figures
#
# Combine multiple charts into a single figure for reports or dashboards.

# %%
# Create 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top left: Cost over time
plot_cost_over_time(sim_result, ax=axes[0, 0], title="Total Cost Over Time")

# Top right: Failures by year
plot_failures_by_year(sim_result, ax=axes[0, 1], title="Failures by Year")

# Bottom left: Risk distribution
plot_risk_distribution(
    opt_result.selections, ax=axes[1, 0], title="Selected Assets Risk Distribution"
)

# Bottom right: Scenario comparison
plot_scenario_comparison(comparison, metric="total_cost", ax=axes[1, 1])

# Add overall title
fig.suptitle("Asset Optimization Dashboard", fontsize=16, fontweight="bold", y=1.02)

plt.tight_layout()
plt.show()

# %%
# Save multi-panel figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot_cost_over_time(sim_result, ax=axes[0, 0])
plot_failures_by_year(sim_result, ax=axes[0, 1])
plot_risk_distribution(opt_result.selections, ax=axes[1, 0])
plot_scenario_comparison(comparison, metric="total_cost", ax=axes[1, 1])

fig.suptitle("Asset Optimization Dashboard", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()

plt.savefig("dashboard.png", dpi=150, bbox_inches="tight")
print("Saved: dashboard.png")
plt.close()

# %% [markdown]
# ## Summary
#
# This notebook covered the visualization and export capabilities:
#
# 1. **SDK Theme**: `set_sdk_theme()` for consistent styling
# 2. **Five Chart Types**:
#    - `plot_cost_over_time()` - Line chart of costs
#    - `plot_failures_by_year()` - Bar chart of failures
#    - `plot_risk_distribution()` - Histogram of risk scores
#    - `plot_scenario_comparison()` - Grouped bar chart for scenarios
#    - `plot_asset_action_heatmap()` - Categorical heatmap of asset actions
# 3. **Customization**: All functions return axes for further customization
# 4. **Export**: Parquet format for data, PNG/PDF for charts
# 5. **Multi-Panel Figures**: Combine charts into dashboards

# %%
# Clean up temporary files
for f in [
    "sim_summary.parquet",
    "sim_projections.parquet",
    "opt_schedule.parquet",
    "cost_chart.png",
    "dashboard.png",
]:
    if os.path.exists(f):
        os.remove(f)
        print(f"Cleaned up: {f}")
