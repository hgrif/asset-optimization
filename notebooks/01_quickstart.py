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
# Asset Optimization Quickstart (Planner API)

This notebook walks through the new Proposal A planner flow:

- Create a simple asset table
- Configure a risk model and effect model
- Build a planner and validate inputs
- Generate candidate actions and optimize a plan
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
## 1. Build a Sample Asset Table

The planner expects an asset table with at least:
- `asset_id`
- `asset_type`
- `install_date`

The Weibull model also needs an `age` column and a type column (here `material`).
"""

# %%
np.random.seed(7)

n_assets = 12
materials = ["PVC", "Cast Iron"]
base_date = pd.Timestamp("2026-01-01")
install_dates = base_date - pd.to_timedelta(
    np.random.randint(10 * 365, 70 * 365, size=n_assets), unit="D"
)

assets = pd.DataFrame(
    {
        "asset_id": [f"PIPE-{i:03d}" for i in range(n_assets)],
        "asset_type": "pipe",
        "install_date": install_dates,
        "material": np.random.choice(materials, size=n_assets, p=[0.6, 0.4]),
    }
)
assets["age"] = (base_date - assets["install_date"]).dt.days / 365.25
assets.head()

# %% [markdown]
"""
## 2. Define Intervention Templates

Interventions can be provided as templates (no `asset_id`). The planner will
cross-join them to assets to build candidate actions.
"""

# %%
interventions = pd.DataFrame(
    {
        "action_type": ["repair", "replace"],
        "direct_cost": [8000.0, 45000.0],
        "crew_hours": [24.0, 60.0],
    }
)
interventions

# %% [markdown]
"""
## 3. Configure Services

We use a Weibull risk model and a simple rule-based effect model.
"""

# %%
weibull_params = {
    "PVC": (2.2, 70.0),
    "Cast Iron": (3.0, 45.0),
}

risk_model = WeibullModel(weibull_params, type_column="material", age_column="age")

# Repair restores ~40% of risk, replace restores ~90%
effect_model = RuleBasedEffectModel({"repair": 0.4, "replace": 0.9})

simulator = BasicNetworkSimulator()
optimizer = Optimizer()

# %% [markdown]
"""
## 4. Build the Planner
"""

# %%
repository = DataFrameRepository(assets=assets, interventions=interventions)
planner = Planner(
    repository=repository,
    risk_model=risk_model,
    effect_model=effect_model,
    simulator=simulator,
    optimizer=optimizer,
)

validation = planner.validate_inputs()
validation

# %% [markdown]
"""
## 5. Fit and Optimize
"""

# %%
planner.fit()

horizon = PlanningHorizon("2026-01-01", "2026-12-31", "quarterly")
objective = (
    ObjectiveBuilder()
    .add_expected_risk_reduction(weight=1.0)
    .add_total_cost(weight=-0.2)
    .build()
)
constraints = ConstraintSet().add_budget_limit(150000.0)

result = planner.optimize_plan(
    horizon=horizon,
    scenarios=None,
    objective=objective,
    constraints=constraints,
)

result.selected_actions.head()

# %% [markdown]
"""
## 6. Inspect Results
"""

# %%
summary = result.selected_actions.groupby("action_type").agg(
    selected_count=("asset_id", "count"),
    total_cost=("direct_cost", "sum"),
    total_benefit=("expected_benefit", "sum"),
)
summary
