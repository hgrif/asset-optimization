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

# %%
# ruff: noqa: E402

# %% [markdown]
"""
# Road Domain Configuration and Simulation

This notebook demonstrates how to use `RoadDomain` to validate a road portfolio,
configure road-specific interventions, and run a multi-year simulation.

**You will learn how to:**
- Build and validate a road portfolio DataFrame
- Inspect road-specific default interventions and deterioration model settings
- Encode covariates for proportional hazards modeling
- Run a multi-year simulation and visualize outcomes
- Compare surface types using the same workflow
"""

# %% [markdown]
"""
## Setup

We will use a small synthetic portfolio to keep the example fast and easy to
follow. The workflow mirrors what you would do with real road data.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from asset_optimization import (
    RoadDomain,
    SimulationConfig,
    Simulator,
    plot_cost_over_time,
    plot_failures_by_year,
    set_sdk_theme,
)

set_sdk_theme()

# %% [markdown]
"""
## 1. Road Domain Overview

`RoadDomain` provides:
- **Schema validation** for road portfolio data
- **Default interventions** with surface-specific costs
- **A proportional hazards model** with a Weibull baseline
"""

# %%
# Create the domain
road_domain = RoadDomain()

# Summarize default interventions by surface type
surface_types = ["asphalt", "concrete", "gravel"]


def summarize_interventions(
    surface_type: str, age_example: float = 20.0
) -> pd.DataFrame:
    interventions = road_domain.default_interventions(surface_type=surface_type)
    rows = []
    for key, intervention in interventions.items():
        rows.append(
            {
                "intervention": key,
                "cost": intervention.cost,
                "age_after_20": intervention.apply_age_effect(age_example),
                "upgrade_type": intervention.upgrade_type,
            }
        )
    return pd.DataFrame(rows).set_index("intervention")


for surface in surface_types:
    print(f"\nDefault interventions for {surface}:")
    print(summarize_interventions(surface))

# Show default deterioration model configuration
model = road_domain.default_model()
print("\nDefault road model:")
print(model)
print("Baseline model:")
print(model.baseline)

# %% [markdown]
"""
## 2. Create a Road Portfolio

We'll create a small portfolio with mixed surface types, traffic loads, and
climate zones. Install dates span roughly 5-30 years in the past.
"""

# %%
np.random.seed(42)

n_assets = 24
base_date = pd.Timestamp("2024-01-01")
install_offsets = np.random.uniform(5 * 365, 30 * 365, n_assets).astype(int)
install_dates = base_date - pd.to_timedelta(install_offsets, unit="D")

portfolio = pd.DataFrame(
    {
        "asset_id": [f"ROAD-{i:03d}" for i in range(n_assets)],
        "install_date": pd.to_datetime(install_dates),
        "surface_type": np.random.choice(surface_types, n_assets, p=[0.5, 0.3, 0.2]),
        "traffic_load": np.random.choice(
            ["low", "medium", "high"], n_assets, p=[0.4, 0.4, 0.2]
        ),
        "climate_zone": np.random.choice(
            ["temperate", "cold", "hot_dry", "hot_humid"], n_assets
        ),
        "length_km": np.random.uniform(0.5, 5.0, n_assets).round(2),
        "condition_score": np.random.uniform(40, 95, n_assets).round(1),
    }
)

print("Sample portfolio (first 5 rows):")
print(portfolio.head())

# Validate and coerce types
validated = road_domain.validate(portfolio)
print("\nValidated portfolio summary:")
print(validated["surface_type"].value_counts())

# %% [markdown]
"""
## 3. Encode Covariates

`RoadDomain.encode_covariates` adds numeric columns for traffic load and climate
zone. These are required by the proportional hazards model.
"""

# %%
encoded = RoadDomain.encode_covariates(validated)
print(
    encoded[
        ["traffic_load", "traffic_load_encoded", "climate_zone", "climate_zone_encoded"]
    ].head()
)

# %% [markdown]
"""
## 4. Run Simulation

The simulator currently validates data using the **pipe portfolio schema**.
To run a road simulation, we add `asset_type="road"` and `material=surface_type`
columns to satisfy that schema, while keeping `surface_type` for the road model.

We also pass the **encoded** DataFrame so the proportional hazards model can
use numeric covariates.
"""

# %%
simulation_df = encoded.assign(
    asset_type="road",
    material=encoded["surface_type"],
)

config = SimulationConfig(n_years=10, random_seed=42)

# For mixed surface portfolios, choose a representative intervention set.
representative_surface = validated["surface_type"].mode().iloc[0]
interventions = road_domain.default_interventions(surface_type=representative_surface)

simulator = Simulator(model, config, interventions=interventions)
result = simulator.run(simulation_df)

print(result)
print(f"Total cost: ${result.total_cost():,.0f}")
print(f"Total failures: {result.total_failures()}")

# %% [markdown]
"""
## 5. Visualize Results
"""

# %%
plot_cost_over_time(result)
plot_failures_by_year(result)

# %% [markdown]
"""
## 6. Compare Surface Types

To highlight differences in deterioration, we can run the same portfolio as
all-asphalt vs all-gravel and compare total costs.
"""


# %%
def run_surface_scenario(base_df: pd.DataFrame, surface_type: str) -> float:
    scenario = base_df.copy()
    scenario["surface_type"] = surface_type
    encoded_scenario = RoadDomain.encode_covariates(scenario)
    simulation_scenario = encoded_scenario.assign(
        asset_type="road",
        material=encoded_scenario["surface_type"],
    )

    scenario_model = road_domain.default_model()
    scenario_interventions = road_domain.default_interventions(
        surface_type=surface_type
    )
    scenario_sim = Simulator(
        scenario_model, config, interventions=scenario_interventions
    )
    scenario_result = scenario_sim.run(simulation_scenario)
    return scenario_result.total_cost()


asphalt_cost = run_surface_scenario(validated, "asphalt")
gravel_cost = run_surface_scenario(validated, "gravel")

comparison = pd.DataFrame(
    {
        "surface_type": ["asphalt", "gravel"],
        "total_cost": [asphalt_cost, gravel_cost],
    }
)

print(comparison)

# %% [markdown]
"""
## Cleanup
"""

# %%
plt.close("all")
