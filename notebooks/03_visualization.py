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
# Basic Visualization (Planner API)

The legacy visualization utilities are removed. This notebook shows how to
visualize planner outputs using plain pandas + matplotlib.
"""

# %%
import matplotlib.pyplot as plt  # noqa: E402
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
## 1. Planner Setup
"""

# %%
np.random.seed(11)

n_assets = 18
base_date = pd.Timestamp("2026-01-01")
install_dates = base_date - pd.to_timedelta(
    np.random.randint(8 * 365, 60 * 365, size=n_assets), unit="D"
)

assets = pd.DataFrame(
    {
        "asset_id": [f"PIPE-{i:03d}" for i in range(n_assets)],
        "asset_type": "pipe",
        "install_date": install_dates,
        "material": np.random.choice(["PVC", "Cast Iron"], size=n_assets),
    }
)
assets["age"] = (base_date - assets["install_date"]).dt.days / 365.25

interventions = pd.DataFrame(
    {
        "action_type": ["inspect", "repair", "replace"],
        "direct_cost": [750.0, 7000.0, 50000.0],
    }
)

risk_model = WeibullModel(
    {"PVC": (2.3, 70.0), "Cast Iron": (3.0, 45.0)},
    type_column="material",
    age_column="age",
)

effect_model = RuleBasedEffectModel({"inspect": 0.05, "repair": 0.45, "replace": 0.95})

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
    ObjectiveBuilder().add_expected_risk_reduction().add_total_cost(weight=-0.1).build()
)
constraints = ConstraintSet().add_budget_limit(90000.0)

candidates = planner.propose_actions(horizon=horizon, scenarios=None)
plan = planner.optimize_plan(
    horizon=horizon,
    scenarios=None,
    objective=objective,
    constraints=constraints,
)

# %% [markdown]
"""
## 2. Scatter Plot: Cost vs Benefit
"""

# %%
selected_ids = set(plan.selected_actions["asset_id"])
plot_df = candidates.copy()
plot_df["selected"] = plot_df["asset_id"].isin(selected_ids)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(
    plot_df.loc[~plot_df["selected"], "direct_cost"],
    plot_df.loc[~plot_df["selected"], "expected_benefit"],
    label="Not selected",
    alpha=0.6,
)
ax.scatter(
    plot_df.loc[plot_df["selected"], "direct_cost"],
    plot_df.loc[plot_df["selected"], "expected_benefit"],
    label="Selected",
    color="#2563eb",
)

ax.set_xlabel("Direct cost")
ax.set_ylabel("Expected benefit")
ax.set_title("Candidate actions")
ax.legend()
plt.tight_layout()

# %% [markdown]
"""
## 3. Selected Actions by Type
"""

# %%
summary = (
    plan.selected_actions.groupby("action_type")["direct_cost"]
    .sum()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(6, 4))
summary.plot(kind="bar", ax=ax)
ax.set_ylabel("Total cost")
ax.set_title("Budget allocated by action type")
plt.tight_layout()
