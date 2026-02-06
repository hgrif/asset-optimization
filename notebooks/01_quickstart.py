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
# Asset Optimization Quickstart

This notebook is a guided walkthrough of the core SDK workflow. We'll go from a
synthetic portfolio to a multi-year simulation and export the results.

**You will learn how to:**
- Build a portfolio DataFrame with realistic asset attributes
- Validate the portfolio and inspect data quality
- Configure a Weibull deterioration model
- Run a multi-year simulation and read the results
- Export outputs for downstream analysis
"""

# %% [markdown]
"""
## Setup

If you have not installed the SDK yet, install it first:

```bash
pip install asset-optimization
```
"""


# %%
# Core imports
import os  # noqa: E402
from datetime import date, timedelta  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from IPython.display import display  # noqa: E402

# SDK imports
from asset_optimization import SimulationConfig, Simulator, WeibullModel  # noqa: E402
from asset_optimization.portfolio import (  # noqa: E402
    compute_quality_metrics,
    validate_portfolio,
)


# %% [markdown]
"""
## 1. Generate Synthetic Portfolio

We will create a sample water pipe portfolio with realistic characteristics.
This gives us a consistent, reproducible dataset to explore the workflow.

The portfolio includes:
- **500 pipes** of different materials
- **Three materials**: Cast Iron (oldest), PVC (modern), Ductile Iron
- **Install dates**: Ranging from 20-80 years ago
- **Diameters and lengths**: Typical for water distribution systems
"""


# %%
# Set seed for reproducibility
np.random.seed(42)

n_assets = 500
materials = ["Cast Iron", "PVC", "Ductile Iron"]

# Generate realistic install dates (20-80 years old)
base_date = date(2024, 1, 1)
install_dates = [
    base_date - timedelta(days=int(np.random.uniform(20 * 365, 80 * 365)))
    for _ in range(n_assets)
]

# Create portfolio DataFrame
data = pd.DataFrame(
    {
        "asset_id": [f"PIPE-{i:04d}" for i in range(n_assets)],
        "install_date": pd.to_datetime(install_dates),
        "asset_type": "pipe",  # All are pipes in this example
        "material": np.random.choice(materials, n_assets, p=[0.4, 0.35, 0.25]),
        "diameter_mm": np.random.choice([150, 200, 300, 400], n_assets),
        "length_m": np.random.uniform(50, 500, n_assets).round(0),
    }
)

print(f"Generated {len(data)} assets")
data.head(10)


# %% [markdown]
"""
## 2. Validate Portfolio Data

Validation is handled via a DataFrame-first helper. The simulator will validate
inputs automatically, but running validation early gives you fast feedback and
makes it easier to troubleshoot data issues.
"""


# %%
# Validate portfolio DataFrame (optional helper)
portfolio = validate_portfolio(data)

# Display portfolio summary
print(portfolio.head())
print(f"\nAsset types: {sorted(portfolio['asset_type'].unique())}")

age_years = (pd.Timestamp.now() - portfolio["install_date"]).dt.days / 365.25
print(f"Mean age: {age_years.mean():.1f} years")


# %%
# Check data quality metrics
quality = compute_quality_metrics(portfolio)
print("Data Quality Metrics:")
print(quality)


# %%
# Access individual assets
oldest_idx = portfolio["install_date"].idxmin()
oldest = portfolio.loc[oldest_idx]
print(f"Oldest asset: {oldest['asset_id']}")
print(f"  Material: {oldest['material']}")
print(f"  Installed: {oldest['install_date'].date()}")


# %% [markdown]
"""
## 3. Configure Deterioration Model

We use a **Weibull model** where each material type has different parameters:

- **shape (k)**: Controls failure rate behavior
  - k > 1 means increasing failure rate (typical for aging infrastructure)
- **scale (lambda)**: Characteristic life in years

Typical values for water pipes:
- Cast Iron: Older technology, shorter expected life
- PVC: Modern material, longer expected life
- Ductile Iron: Good durability, moderate expected life
"""


# %%
# Define Weibull parameters for each material type
# Format: 'material': (shape, scale)
params = {
    "Cast Iron": (3.0, 60),  # Older, shape=3 (increasing failures)
    "PVC": (2.5, 80),  # Modern, longer life
    "Ductile Iron": (2.8, 70),  # Good durability
}

model = WeibullModel(params)
print(model)


# %%
# The model can transform portfolio data to add failure probabilities
# First, we need to add an 'age' column
portfolio_with_age = portfolio.copy()
portfolio_with_age["age"] = (
    pd.Timestamp.now() - portfolio_with_age["install_date"]
).dt.days / 365.25

# Transform adds failure_rate and failure_probability columns
enriched = model.transform(portfolio_with_age)
enriched[["asset_id", "material", "age", "failure_rate", "failure_probability"]].head(
    10
)


# %% [markdown]
"""
## 4. Run Simulation

Run a **10-year simulation** that tracks:
- Costs (failure costs + intervention costs)
- Failures (sampled based on deterioration model)
- Asset aging

The simulation uses **conditional probability** to sample failures:
- P(fail in year t | survived to t) = (S(t) - S(t+1)) / S(t)
- Failed assets are automatically replaced (default behavior)
"""


# %%
# Configure simulation
config = SimulationConfig(
    n_years=10,
    start_year=2024,
    random_seed=42,  # For reproducibility
    failure_response="replace",  # Replace failed assets
)

print(config)


# %%
# Create simulator and run
sim = Simulator(model, config)
result = sim.run(portfolio)

print(result)


# %% [markdown]
"""
## 5. Examine Results

The `SimulationResult` contains:
- **summary**: Year-by-year metrics
- **cost_breakdown**: Detailed cost allocation
- **failure_log**: Individual failure events
"""


# %%
# Summary statistics
print(f"Total cost over {config.n_years} years: ${result.total_cost():,.0f}")
print(f"Total failures: {result.total_failures()}")
print(f"Average failures per year: {result.total_failures() / config.n_years:.1f}")


# %%
# Year-by-year summary
result.summary


# %%
# Cost breakdown by year
result.cost_breakdown


# %%
# Individual failure events
if not result.failure_log.empty:
    print("\nSample failures (first 10):")
    display(result.failure_log.head(10))
else:
    print("No failures recorded (lucky run!)")


# %% [markdown]
"""
## 6. Export Results

Results can be exported to **Parquet format** for further analysis or reporting.

Supported formats:
- `summary`: Year-by-year metrics (default)
- `cost_projections`: Long format for plotting
- `failure_log`: Detailed failure events
"""


# %%
# Export summary (default format)
result.to_parquet("simulation_summary.parquet")
print("Exported: simulation_summary.parquet")

# Export in long format (good for plotting)
result.to_parquet("cost_projections.parquet", format="cost_projections")
print("Exported: cost_projections.parquet")


# %%
# Read back and verify
df = pd.read_parquet("simulation_summary.parquet")
print("Read back simulation_summary.parquet:")
df.head()


# %% [markdown]
"""
## Next Steps

- See **`optimization.ipynb`** for budget-constrained intervention selection
- See **`visualization.ipynb`** for charts and scenario comparisons
"""


# %%
# Clean up temporary files
for f in ["simulation_summary.parquet", "cost_projections.parquet"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"Cleaned up: {f}")
