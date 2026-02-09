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
# Proportional Hazards Example (Planner API)

This notebook shows how to use `ProportionalHazardsModel` with covariates.
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
    ProportionalHazardsModel,
    RuleBasedEffectModel,
    WeibullModel,
)

# %% [markdown]
"""
## 1. Build Asset Data with Covariates
"""

# %%
np.random.seed(17)

n_assets = 14
base_date = pd.Timestamp("2026-01-01")
install_dates = base_date - pd.to_timedelta(
    np.random.randint(5 * 365, 70 * 365, size=n_assets), unit="D"
)

assets = pd.DataFrame(
    {
        "asset_id": [f"ASSET-{i:03d}" for i in range(n_assets)],
        "asset_type": "pipe",
        "install_date": install_dates,
        "material": np.random.choice(["PVC", "Cast Iron"], size=n_assets),
        "traffic_index": np.random.uniform(0.2, 1.5, size=n_assets).round(2),
        "soil_corrosion": np.random.uniform(0.0, 1.0, size=n_assets).round(2),
    }
)
assets["age"] = (base_date - assets["install_date"]).dt.days / 365.25
assets.head()

# %% [markdown]
"""
## 2. Configure the Proportional Hazards Model
"""

# %%
baseline = WeibullModel(
    {"PVC": (2.0, 75.0), "Cast Iron": (3.0, 45.0)},
    type_column="material",
    age_column="age",
)

ph_model = ProportionalHazardsModel(
    baseline=baseline,
    covariates=["traffic_index", "soil_corrosion"],
    coefficients={"traffic_index": 0.6, "soil_corrosion": 0.9},
)

# %% [markdown]
"""
## 3. Plan with the PH Risk Model
"""

# %%
interventions = pd.DataFrame(
    {
        "action_type": ["repair", "replace"],
        "direct_cost": [7000.0, 48000.0],
    }
)

planner = Planner(
    repository=DataFrameRepository(assets=assets, interventions=interventions),
    risk_model=ph_model,
    effect_model=RuleBasedEffectModel({"repair": 0.4, "replace": 0.9}),
    simulator=BasicNetworkSimulator(),
    optimizer=Optimizer(),
)

planner.fit()

horizon = PlanningHorizon("2026-01-01", "2026-12-31", "quarterly")
objective = ObjectiveBuilder().add_expected_risk_reduction().build()
constraints = ConstraintSet().add_budget_limit(120000.0)

plan = planner.optimize_plan(
    horizon=horizon,
    scenarios=None,
    objective=objective,
    constraints=constraints,
)

plan.selected_actions.head()
