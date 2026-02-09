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
# Learn how to select optimal interventions under budget constraints using
# the Planner API.
#
# This notebook demonstrates:
# 1. Setting up a portfolio, model, and Planner
# 2. Defining objectives and constraints
# 3. Running optimization and examining results
# 4. Understanding the greedy algorithm
# 5. Comparing different budget scenarios
# 6. Exporting intervention schedules

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
    ConstraintSet,
    DataFrameRepository,
    ObjectiveBuilder,
    Optimizer,
    Planner,
    PlanningHorizon,
    RuleBasedEffectModel,
    Simulator,
    SimulationConfig,
    WeibullModel,
    export_schedule_detailed,
    export_schedule_minimal,
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

portfolio = pd.DataFrame(
    {
        "asset_id": [f"PIPE-{i:04d}" for i in range(n_assets)],
        "install_date": pd.to_datetime(install_dates),
        "asset_type": "pipe",
        "material": np.random.choice(materials, n_assets, p=[0.4, 0.35, 0.25]),
        "diameter_mm": np.random.choice([150, 200, 300, 400], n_assets),
        "length_m": np.random.uniform(50, 500, n_assets).round(0),
    }
)

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
# ## 2. Build the Planner
#
# The **Planner** orchestrates the full optimization workflow:
# 1. Load assets from a **DataFrameRepository**
# 2. Predict failure distributions using a **risk model** (WeibullModel)
# 3. Estimate intervention effects using an **effect model**
# 4. Select optimal interventions using the **Optimizer** with a greedy strategy
#
# We also need to add an `age` column so the model can calculate failure rates.

# %%
# Add age column for the deterioration model
portfolio["age"] = (pd.Timestamp.now() - portfolio["install_date"]).dt.days / 365.25

# Build the planner
repository = DataFrameRepository(assets=portfolio)

planner = Planner(
    repository=repository,
    risk_model=model,
    effect_model=RuleBasedEffectModel(),
    simulator=Simulator(model, SimulationConfig(n_years=1)),
    optimizer=Optimizer(),
)

# Validate inputs
report = planner.validate_inputs()
print(f"Validation passed: {report.passed}")

# Fit the planner (trains risk and effect models)
planner.fit()
print("Planner fitted successfully")

# %% [markdown]
# ## 3. Run Optimization
#
# Define an **objective** (what to maximize) and **constraints** (budget limit),
# then call `optimize_plan()`.

# %%
# Define objective: maximize risk reduction, penalize cost
objective = (
    ObjectiveBuilder()
    .add_expected_risk_reduction(weight=1.0)
    .add_total_cost(weight=-0.001)  # Small negative weight to prefer lower cost
    .build()
)

# Define constraint: $500,000 annual budget
budget = 500_000
constraints = ConstraintSet().add_budget_limit(budget)

# Run optimization
horizon = PlanningHorizon("2024-01-01", "2024-12-31", "yearly")
result = planner.optimize_plan(
    horizon=horizon,
    scenarios=None,
    objective=objective,
    constraints=constraints,
)

print(f"PlanResult: {result.metadata['selected_count']} actions selected")

# %% [markdown]
# ## 4. Examine Selections
#
# The `PlanResult` contains:
# - **selected_actions**: DataFrame of selected interventions
# - **metadata**: Budget utilization statistics
# - **objective_breakdown**: How each objective term contributes

# %%
# Budget summary
budget_spent = result.metadata["budget_spent"]
budget_remaining = result.metadata["budget_remaining"]
utilization = (budget_spent / budget) * 100

print("Budget Summary:")
print(f"  Total budget: ${budget:,.0f}")
print(f"  Total spent: ${budget_spent:,.0f}")
print(f"  Remaining: ${budget_remaining:,.0f}")
print(f"  Utilization: {utilization:.1f}%")
print(f"\nSelected {len(result.selected_actions)} interventions")

# %%
# View top selections (highest benefit-cost ratio first)
print("Top 15 Selected Interventions:")
result.selected_actions.head(15)

# %%
# Action type breakdown
selections = result.selected_actions
type_counts = selections["action_type"].value_counts()
print("Interventions by Type:")
for atype, count in type_counts.items():
    total_cost = selections[selections["action_type"] == atype]["direct_cost"].sum()
    print(f"  {atype}: {count} assets (${total_cost:,.0f})")

# %% [markdown]
# ## 5. Understand the Algorithm
#
# The greedy algorithm prioritizes assets based on their **benefit-to-cost ratio**:
#
# ```
# priority = expected_benefit / direct_cost
# ```
#
# This means:
# - High-benefit assets with low intervention costs are selected first
# - Assets are greedily added until the budget is exhausted

# %%
# Look at the benefit distribution of selected assets
print("Benefit-Cost Analysis of Selected Assets:")
print(f"  Min benefit: {selections['expected_benefit'].min():.3f}")
print(f"  Max benefit: {selections['expected_benefit'].max():.3f}")
print(f"  Mean benefit: {selections['expected_benefit'].mean():.3f}")

if "benefit_cost_ratio" in selections.columns:
    print(f"  Mean ratio: {selections['benefit_cost_ratio'].mean():.3f}")

# %%
# Join selections with portfolio data for context
analysis = selections.merge(
    portfolio[["asset_id", "material", "age"]], on="asset_id", how="left"
)

display_cols = ["rank", "asset_id", "material", "age", "action_type", "direct_cost"]
if "expected_benefit" in analysis.columns:
    display_cols.append("expected_benefit")

print("Selection Analysis (first 10):")
analysis[display_cols].head(10)

# %% [markdown]
# ## 6. Compare Budget Scenarios
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
    c = ConstraintSet().add_budget_limit(budget_amount)
    r = planner.optimize_plan(
        horizon=horizon,
        scenarios=None,
        objective=objective,
        constraints=c,
    )
    results[name] = r

# Compare results
print("Budget Comparison:")
print("-" * 60)
print(f"{'Scenario':<10} {'Budget':>12} {'Spent':>12} {'Assets':>8}")
print("-" * 60)
for name, res in results.items():
    budget_amount = budgets[name]
    spent = res.metadata["budget_spent"]
    n_selected = res.metadata["selected_count"]
    print(f"{name:<10} ${budget_amount:>10,} ${spent:>10,.0f} {n_selected:>8}")

# %%
# Objective breakdown comparison
print("\nObjective Breakdown by Scenario:")
for name, res in results.items():
    print(f"\n  {name}:")
    for term, value in res.objective_breakdown.items():
        print(f"    {term}: {value:.2f}")

# %%
# Run baseline simulation for cost context
config = SimulationConfig(
    n_years=10,
    start_year=2024,
    random_seed=42,
    failure_response="replace",
)

sim = Simulator(model, config)
sim_result = sim.run(portfolio)

print("\n10-Year Simulation Results (baseline):")
print(f"  Total cost: ${sim_result.total_cost():,.0f}")
print(f"  Total failures: {sim_result.total_failures()}")

# %% [markdown]
# ## 7. Export Intervention Schedule
#
# Export results using the standalone export functions:
# - **minimal**: asset_id, year, action_type, direct_cost
# - **detailed**: Includes expected_benefit, rank, and optional portfolio data

# %%
# Export minimal format
export_schedule_minimal(result.selected_actions, "schedule_minimal.parquet", year=2024)
print("Exported: schedule_minimal.parquet")

# Export detailed format with portfolio data
export_schedule_detailed(
    result.selected_actions,
    "schedule_detailed.parquet",
    year=2024,
    portfolio=portfolio,
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
# 1. **Planner Setup**: Connecting repository, model, effect model, and optimizer
# 2. **Objectives and Constraints**: Defining what to optimize and budget limits
# 3. **Running Optimization**: `optimize_plan()` returns a `PlanResult`
# 4. **Examining Selections**: Understanding which assets were selected and why
# 5. **Budget Comparison**: Seeing how different budgets affect intervention counts
# 6. **Export Formats**: Saving schedules in minimal or detailed parquet format
#
# Next: See **`03_visualization.ipynb`** for charts and scenario comparisons.

# %%
# Clean up temporary files
for f in ["schedule_minimal.parquet", "schedule_detailed.parquet"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"Cleaned up: {f}")
