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
# Roads Example (Planner API)

The legacy `RoadDomain` helper has been removed. This notebook shows how to
model road assets directly with the Proposal A planner APIs.
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
## 1. Road Asset Data
"""

# %%
np.random.seed(31)

n_segments = 10
base_date = pd.Timestamp("2026-01-01")
install_dates = base_date - pd.to_timedelta(
    np.random.randint(6 * 365, 35 * 365, size=n_segments), unit="D"
)

assets = pd.DataFrame(
    {
        "asset_id": [f"ROAD-{i:03d}" for i in range(n_segments)],
        "asset_type": "road_segment",
        "install_date": install_dates,
        "surface_type": np.random.choice(["asphalt", "concrete"], size=n_segments),
        "traffic_load": np.random.uniform(0.5, 1.5, size=n_segments).round(2),
    }
)
assets["age"] = (base_date - assets["install_date"]).dt.days / 365.25
assets.head()

# %% [markdown]
"""
## 2. Models and Planner

We use `surface_type` as the Weibull type column and add simple action rules.
"""

# %%
weibull_params = {
    "asphalt": (2.4, 20.0),
    "concrete": (2.0, 30.0),
}

risk_model = WeibullModel(
    weibull_params,
    type_column="surface_type",
    age_column="age",
)

effect_model = RuleBasedEffectModel(
    {
        "patch": 0.2,
        "resurface": 0.6,
        "reconstruct": 0.95,
    }
)

interventions = pd.DataFrame(
    {
        "action_type": ["patch", "resurface", "reconstruct"],
        "direct_cost": [12000.0, 60000.0, 180000.0],
        "crew_hours": [16.0, 80.0, 160.0],
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

# %% [markdown]
"""
## 3. Optimize a Plan
"""

# %%
horizon = PlanningHorizon("2026-01-01", "2027-12-31", "yearly")
objective = (
    ObjectiveBuilder()
    .add_expected_risk_reduction(weight=1.0)
    .add_total_cost(weight=-0.15)
    .build()
)
constraints = ConstraintSet().add_budget_limit(250000.0)

plan = planner.optimize_plan(
    horizon=horizon,
    scenarios=None,
    objective=objective,
    constraints=constraints,
)

plan.selected_actions
