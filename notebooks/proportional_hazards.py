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
# # Proportional Hazards Modeling
#
# This notebook demonstrates how to use `ProportionalHazardsModel` to model failure rates
# that depend on asset properties (covariates) beyond just age and type.
#
# ## What is Proportional Hazards?
#
# The proportional hazards model scales the baseline hazard rate by a risk score:
#
# h(t|x) = h_baseline(t) * exp(beta_1 * x_1 + beta_2 * x_2 + ...)
#
# Where:
# - h_baseline(t) is the baseline hazard (e.g., from WeibullModel)
# - x_i are covariate values (asset properties like diameter, length, etc.)
# - beta_i are coefficients that control how each covariate affects risk
#
# **Use cases:**
# - Larger diameter pipes may have different failure characteristics
# - Longer pipe segments may have higher failure risk
# - Environmental factors (soil type, traffic load) affecting failure
#

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from asset_optimization import (
    WeibullModel,
    Simulator,
    SimulationConfig,
    set_sdk_theme,
)
from asset_optimization.models import ProportionalHazardsModel
from asset_optimization.portfolio import validate_portfolio

set_sdk_theme()

# %% [markdown]
# ## Creating a Portfolio with Covariates
#
# Our portfolio includes pipe diameter as a covariate. We'll model the hypothesis
# that larger diameter pipes have slightly higher failure rates due to increased
# stress and maintenance complexity.
#

# %%
# Create portfolio with varying diameters
np.random.seed(42)
n_assets = 50

portfolio_data = pd.DataFrame(
    {
        "asset_id": [f"PIPE-{i:03d}" for i in range(n_assets)],
        "asset_type": ["pipe"] * n_assets,
        "material": ["PVC"] * n_assets,
        "install_date": pd.to_datetime("2000-01-01")
        + pd.to_timedelta(np.random.randint(0, 3650, n_assets), unit="D"),
        "diameter_mm": np.random.choice([100, 150, 200, 250, 300], n_assets),
        "length_m": np.random.uniform(50, 200, n_assets).round(1),
    }
)

portfolio = validate_portfolio(portfolio_data)
print(f"Portfolio: {len(portfolio)} assets")
print(f"Diameter distribution:{portfolio['diameter_mm'].value_counts().sort_index()}")

# %% [markdown]
# ## Baseline Weibull Model
#
# First, let's create a baseline Weibull model without covariate effects.
#

# %%
baseline = WeibullModel({"PVC": (2.5, 50.0)})
print(baseline)

# %% [markdown]
# ## Proportional Hazards Model
#
# Now we wrap the baseline with a ProportionalHazardsModel. We'll use pipe diameter
# as a covariate with a positive coefficient, meaning larger pipes have higher risk.
#
# The coefficient 0.005 means that each additional mm of diameter multiplies the
# hazard rate by exp(0.005) ≈ 1.005 (0.5% increase per mm).
#

# %%
ph_model = ProportionalHazardsModel(
    baseline=baseline, covariates=["diameter_mm"], coefficients={"diameter_mm": 0.005}
)

print(ph_model)
print(f"Risk multiplier for 100mm pipe: {np.exp(0.005 * 100):.2f}x baseline")
print(f"Risk multiplier for 300mm pipe: {np.exp(0.005 * 300):.2f}x baseline")

# %% [markdown]
# ## Comparing Failure Rates
#
# Let's see how the covariate affects failure rates for different pipe sizes.
#

# %%
# Create test DataFrame with same age, different diameters
test_df = pd.DataFrame(
    {"material": ["PVC"] * 5, "age": [25] * 5, "diameter_mm": [100, 150, 200, 250, 300]}
)

# Get baseline failure rates
baseline_result = baseline.transform(test_df)

# Get PH failure rates
ph_result = ph_model.transform(test_df)

# Compare
comparison = pd.DataFrame(
    {
        "Diameter (mm)": test_df["diameter_mm"],
        "Baseline Rate": baseline_result["failure_rate"],
        "PH Rate": ph_result["failure_rate"],
        "Risk Multiplier": ph_result["failure_rate"] / baseline_result["failure_rate"],
    }
)
print(comparison.to_string(index=False))

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Failure rate by diameter
ax1 = axes[0]
ax1.bar(
    comparison["Diameter (mm)"].astype(str),
    comparison["Baseline Rate"],
    alpha=0.7,
    label="Baseline",
)
ax1.bar(
    comparison["Diameter (mm)"].astype(str),
    comparison["PH Rate"],
    alpha=0.7,
    label="With Covariates",
)
ax1.set_xlabel("Diameter (mm)")
ax1.set_ylabel("Failure Rate")
ax1.set_title("Failure Rate by Pipe Diameter (Age 25)")
ax1.legend()

# Right: Risk multiplier
ax2 = axes[1]
ax2.plot(
    comparison["Diameter (mm)"],
    comparison["Risk Multiplier"],
    "o-",
    linewidth=2,
    markersize=8,
)
ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline (1.0)")
ax2.set_xlabel("Diameter (mm)")
ax2.set_ylabel("Risk Multiplier")
ax2.set_title("Risk Multiplier vs Baseline")
ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Running Simulation with Covariates
#
# Now let's run a simulation to see how covariates affect long-term outcomes.
#

# %%
config = SimulationConfig(n_years=20, random_seed=42)

# Simulation with baseline model
sim_baseline = Simulator(baseline, config)
result_baseline = sim_baseline.run(portfolio)

# Simulation with PH model
sim_ph = Simulator(ph_model, config)
result_ph = sim_ph.run(portfolio)

# Note: total_cost() and total_failures() are methods, not attributes
print("Baseline Model Results:")
print(f"  Total failures: {result_baseline.total_failures()}")
print(f"  Total cost: ${result_baseline.total_cost():,.0f}")

print("Proportional Hazards Model Results:")
print(f"  Total failures: {result_ph.total_failures()}")
print(f"  Total cost: ${result_ph.total_cost():,.0f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Failures over time comparison
# Note: result.summary is a DataFrame with columns: year, total_cost, failure_count, etc.
years = result_baseline.summary["year"].tolist()
baseline_failures = result_baseline.summary["failure_count"].tolist()
ph_failures = result_ph.summary["failure_count"].tolist()

ax1 = axes[0]
ax1.plot(years, baseline_failures, "o-", label="Baseline", alpha=0.7)
ax1.plot(years, ph_failures, "s-", label="With Covariates", alpha=0.7)
ax1.set_xlabel("Year")
ax1.set_ylabel("Failures")
ax1.set_title("Annual Failures: Baseline vs Proportional Hazards")
ax1.legend()

# Cumulative cost comparison
baseline_cumcost = result_baseline.summary["total_cost"].cumsum()
ph_cumcost = result_ph.summary["total_cost"].cumsum()

ax2 = axes[1]
ax2.plot(years, baseline_cumcost / 1000, "o-", label="Baseline", alpha=0.7)
ax2.plot(years, ph_cumcost / 1000, "s-", label="With Covariates", alpha=0.7)
ax2.set_xlabel("Year")
ax2.set_ylabel("Cumulative Cost ($K)")
ax2.set_title("Cumulative Cost Over Time")
ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Using Multiple Covariates
#
# ProportionalHazardsModel supports multiple covariates. Here we add pipe length
# as a second covariate - longer pipes have more potential failure points.
#

# %%
ph_multi = ProportionalHazardsModel(
    baseline=baseline,
    covariates=["diameter_mm", "length_m"],
    coefficients={
        "diameter_mm": 0.003,  # Each mm adds 0.3% risk
        "length_m": 0.002,  # Each meter adds 0.2% risk
    },
)

# Show risk for different combinations
examples = pd.DataFrame(
    {
        "material": ["PVC"] * 4,
        "age": [25] * 4,
        "diameter_mm": [100, 100, 300, 300],
        "length_m": [50, 150, 50, 150],
    }
)

result_multi = ph_multi.transform(examples)
examples["risk_score"] = (
    result_multi["failure_rate"] / baseline.transform(examples)["failure_rate"]
)

print("Risk scores for different pipe configurations:")
print(examples[["diameter_mm", "length_m", "risk_score"]].to_string(index=False))

# %% [markdown]
# ## Covariate Requirements
#
# ProportionalHazardsModel requires all covariate columns to be present and non-null.
# If a covariate column is missing or contains NaNs, the model raises a ValueError.
# Check covariate completeness before running transforms or simulations.
#

# %%
# Check covariate completeness before modeling
covariate_cols = ["diameter_mm"]
missing_cols = [col for col in covariate_cols if col not in portfolio.columns]
if missing_cols:
    raise ValueError(f"Missing required covariates: {missing_cols}")

nan_counts = portfolio[covariate_cols].isna().sum()
if (nan_counts > 0).any():
    raise ValueError(f"Covariates contain NaNs: {nan_counts[nan_counts > 0].to_dict()}")

print("Covariates look complete for modeling.")

# %% [markdown]
# ## Summary
#
# The `ProportionalHazardsModel` enables sophisticated failure rate modeling:
#
# 1. **Wrap any baseline model** - Works with WeibullModel or any DeteriorationModel
# 2. **Flexible covariates** - Use any numeric DataFrame columns as risk factors
# 3. **Interpretable coefficients** - exp(beta) gives the risk multiplier per unit
# 4. **Simulation-ready** - Works seamlessly with Simulator for long-term projections
# 5. **Strict covariates** - Missing or NaN covariates raise errors
#
# ### When to use Proportional Hazards
#
# - When asset properties beyond age/type affect failure risk
# - When you have domain knowledge about risk factors (e.g., "larger pipes fail more")
# - When you want to model heterogeneous risk across your portfolio
# - When comparing scenarios with different asset characteristics
#
