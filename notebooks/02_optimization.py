# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Optimization Deep Dive (Planner API)

This notebook focuses on objective and constraint tuning with the planner.
"""

# %%
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from asset_optimization import (  # noqa: E402
    BasicNetworkSimulator,
    ConstraintSet,
    DataFrameRepository,
    ObjectiveBuilder,
    Optimizer,
    Planner,
    PlanningHorizon,
    RuleBasedEffectModel,
    WeibullModel,
)

# %% [markdown]
"""
## 1. Sample Asset Data
"""

# %%
np.random.seed(21)

n_assets = 20
base_date = pd.Timestamp("2026-01-01")
install_dates = base_date - pd.to_timedelta(
    np.random.randint(5 * 365, 80 * 365, size=n_assets), unit="D"
)

assets = pd.DataFrame(
    {
        "asset_id": [f"ASSET-{i:03d}" for i in range(n_assets)],
        "asset_type": "pipe",
        "install_date": install_dates,
        "material": np.random.choice(["PVC", "Cast Iron"], size=n_assets),
    }
)
assets["age"] = (base_date - assets["install_date"]).dt.days / 365.25

interventions = pd.DataFrame(
    {
        "action_type": ["inspect", "repair", "replace"],
        "direct_cost": [600.0, 6000.0, 42000.0],
    }
)

# %% [markdown]
"""
## 2. Planner Setup
"""

# %%
risk_model = WeibullModel(
    {"PVC": (2.0, 75.0), "Cast Iron": (3.1, 40.0)},
    type_column="material",
    age_column="age",
)

effect_model = RuleBasedEffectModel(
    {
        "inspect": 0.05,
        "repair": 0.4,
        "replace": 0.95,
    }
)

planner = Planner(
    repository=DataFrameRepository(assets=assets, interventions=interventions),
    risk_model=risk_model,
    effect_model=effect_model,
    simulator=BasicNetworkSimulator(),
    optimizer=Optimizer(),
)

planner.fit()

horizon = PlanningHorizon("2026-01-01", "2026-12-31", "quarterly")
objective = (
    ObjectiveBuilder()
    .add_expected_risk_reduction(weight=1.0)
    .add_total_cost(weight=-0.1)
    .build()
)

# %% [markdown]
"""
## 3. Compare Budget Scenarios
"""

# %%
budgets = [20000.0, 60000.0, 120000.0]
rows = []

for budget in budgets:
    constraints = ConstraintSet().add_budget_limit(budget)
    result = planner.optimize_plan(
        horizon=horizon,
        scenarios=None,
        objective=objective,
        constraints=constraints,
    )
    selected = result.selected_actions
    rows.append(
        {
            "budget_limit": budget,
            "selected_count": len(selected),
            "budget_spent": float(selected["direct_cost"].sum()),
            "total_benefit": float(selected["expected_benefit"].sum()),
        }
    )

summary = pd.DataFrame(rows)
summary

# %% [markdown]
"""
## 4. Inspect the Largest Budget Plan
"""

# %%
constraints = ConstraintSet().add_budget_limit(budgets[-1])
plan = planner.optimize_plan(
    horizon=horizon,
    scenarios=None,
    objective=objective,
    constraints=constraints,
)

plan.selected_actions.sort_values("expected_benefit", ascending=False).head(10)
