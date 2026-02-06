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
# # Optimization Workflow
#
# Learn how to select optimal interventions under budget constraints.
#
# This notebook demonstrates:
# 1. Setting up a portfolio and deterioration model
# 2. Configuring and running the optimizer
# 3. Examining intervention selections
# 4. Understanding the greedy algorithm
# 5. Comparing different budget scenarios

# %% [markdown]
# ## Setup

# %%
# Core imports
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

# SDK imports
from asset_optimization import (
    WeibullModel,
    Optimizer,
    Simulator,
    SimulationConfig,
)

# %% [markdown]
# ## 1. Create Portfolio Data and Model
#
# First, we create a synthetic portfolio of water pipes with different materials and ages.

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
# Configure deterioration model
params = {
    "Cast Iron": (3.0, 60),
    "PVC": (2.5, 80),
    "Ductile Iron": (2.8, 70),
}

model = WeibullModel(params)
print(model)

# %% [markdown]
# ## 2. Configure Optimizer
#
# The **Optimizer** uses a two-stage greedy algorithm to select interventions:
#
# 1. **Stage 1**: For each asset, find the best intervention (highest cost-effectiveness)
# 2. **Stage 2**: Rank all candidates by risk-to-cost ratio and greedily fill the budget
#
# Parameters:
# - **strategy**: 'greedy' (default) or 'milp' (planned)
# - **min_risk_threshold**: Only consider assets above this failure probability

# %%
# Create optimizer with risk threshold
optimizer = Optimizer(
    strategy="greedy",
    min_risk_threshold=0.1,  # Only consider assets with >10% failure risk
)

print(optimizer)

# %% [markdown]
# ## 3. Run Optimization
#
# The `fit()` method follows the scikit-learn pattern, returning self with a `result_` attribute.

# %%
# Run optimization with $500,000 budget
budget = 500_000

optimizer.fit(portfolio, model, budget=budget)

# Access result via .result property
result = optimizer.result
print(result)

# %% [markdown]
# ## 4. Examine Selections
#
# The result contains:
# - **selections**: DataFrame of selected interventions
# - **budget_summary**: Budget utilization statistics

# %%
# Budget summary
print("Budget Summary:")
print(f"  Total budget: ${budget:,.0f}")
print(f"  Total spent: ${result.total_spent:,.0f}")
print(f"  Utilization: {result.utilization_pct:.1f}%")
print(f"\nSelected {len(result.selections)} interventions")

# %%
# View top selections (highest priority first)
print("Top 15 Selected Interventions:")
result.selections.head(15)

# %%
# Intervention type breakdown
type_counts = result.selections["intervention_type"].value_counts()
print("Interventions by Type:")
for itype, count in type_counts.items():
    total_cost = result.selections[result.selections["intervention_type"] == itype][
        "cost"
    ].sum()
    print(f"  {itype}: {count} assets (${total_cost:,.0f})")

# %% [markdown]
# ## 5. Understand the Algorithm
#
# The greedy algorithm prioritizes assets based on their **risk-to-cost ratio**:
#
# ```
# priority = risk_score / intervention_cost
# ```
#
# This means:
# - High-risk assets with low intervention costs are selected first
# - Assets just above the risk threshold may not be selected if budget is limited

# %%
# Look at the risk distribution of selected assets
selections = result.selections

print("Risk Score Distribution of Selected Assets:")
print(f"  Min risk: {selections['risk_score'].min():.3f}")
print(f"  Max risk: {selections['risk_score'].max():.3f}")
print(f"  Mean risk: {selections['risk_score'].mean():.3f}")
print(f"  Median risk: {selections['risk_score'].median():.3f}")

# %%
# Show why certain assets were selected
# Add age information for context
portfolio_with_age = portfolio.copy()
portfolio_with_age["age"] = (
    pd.Timestamp.now() - portfolio_with_age["install_date"]
).dt.days / 365.25

# Join selections with portfolio data
analysis = selections.merge(
    portfolio_with_age[["asset_id", "material", "age"]], on="asset_id", how="left"
)

print("Selection Analysis (first 10):")
analysis[
    ["rank", "asset_id", "material", "age", "risk_score", "intervention_type", "cost"]
].head(10)

# %% [markdown]
# ## 6. Compare Scenarios
#
# What if we had different budget levels? Let's compare:
# - **Low budget**: $250,000
# - **Medium budget**: $500,000 (current)
# - **High budget**: $1,000,000

# %%
# Run optimization at different budget levels
budgets = {
    "low": 250_000,
    "medium": 500_000,
    "high": 1_000_000,
}

results = {}
for name, budget_amount in budgets.items():
    opt = Optimizer(strategy="greedy", min_risk_threshold=0.1)
    opt.fit(portfolio, model, budget=budget_amount)
    results[name] = opt.result

# Compare results
print("Budget Comparison:")
print("-" * 60)
print(f"{'Scenario':<10} {'Budget':>12} {'Spent':>12} {'Assets':>8} {'Util%':>8}")
print("-" * 60)
for name, res in results.items():
    budget_amount = budgets[name]
    print(
        f"{name:<10} ${budget_amount:>10,} ${res.total_spent:>10,.0f} {len(res.selections):>8} {res.utilization_pct:>7.1f}%"
    )

# %%
# Run simulations for each budget scenario
# to see impact on costs and failures

config = SimulationConfig(
    n_years=10,
    start_year=2024,
    random_seed=42,
    failure_response="replace",
)

sim = Simulator(model, config)

# Run baseline simulation (no optimization context, just for comparison)
sim_result = sim.run(portfolio)

print("\n10-Year Simulation Results (baseline):")
print(f"  Total cost: ${sim_result.total_cost():,.0f}")
print(f"  Total failures: {sim_result.total_failures()}")

# %% [markdown]
# ## 7. Export Intervention Schedule
#
# Export results in different formats:
# - **minimal**: Just asset_id, year, intervention_type, cost
# - **detailed**: Includes risk scores, rankings, and optional portfolio data

# %%
# Export minimal format
result.to_parquet("schedule_minimal.parquet", format="minimal", year=2024)
print("Exported: schedule_minimal.parquet")

# Export detailed format with portfolio data
result.to_parquet(
    "schedule_detailed.parquet",
    format="detailed",
    year=2024,
    portfolio=portfolio_with_age,
)
print("Exported: schedule_detailed.parquet")

# %%
# Read back and verify
minimal = pd.read_parquet("schedule_minimal.parquet")
print("Minimal format columns:", list(minimal.columns))
minimal.head()

# %%
detailed = pd.read_parquet("schedule_detailed.parquet")
print("Detailed format columns:", list(detailed.columns))
detailed.head()

# %% [markdown]
# ## Summary
#
# In this notebook, we covered:
#
# 1. **Portfolio Data and Model Setup**: Creating a realistic asset portfolio with deterioration parameters
# 2. **Optimizer Configuration**: Using the greedy strategy with risk thresholds
# 3. **Running Optimization**: The `fit()` method returns self with results in `result_`
# 4. **Examining Selections**: Understanding which assets were selected and why
# 5. **Budget Comparison**: Seeing how different budgets affect intervention counts
# 6. **Export Formats**: Saving schedules in minimal or detailed parquet format
#
# Next: See **`visualization.ipynb`** for charts and scenario comparisons.

# %%
# Clean up temporary files
for f in ["schedule_minimal.parquet", "schedule_detailed.parquet"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"Cleaned up: {f}")
