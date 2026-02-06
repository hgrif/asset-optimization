"""Simulator class for multi-timestep asset simulation.

This module provides the core simulation engine that runs multi-year simulations,
updates asset states, samples failures using conditional probabilities, applies
intervention responses, and tracks costs/metrics.
"""

from typing import Optional

import numpy as np
import pandas as pd

from asset_optimization.models.base import DeteriorationModel
from asset_optimization.portfolio import validate_portfolio
from asset_optimization.simulation.config import SimulationConfig
from asset_optimization.simulation.interventions import (
    DO_NOTHING,
    INSPECT,
    REPAIR,
    REPLACE,
    InterventionType,
)
from asset_optimization.simulation.result import SimulationResult


# Default costs for failures
DEFAULT_FAILURE_DIRECT_COST = 10000.0
DEFAULT_FAILURE_CONSEQUENCE_COST = 5000.0


class Simulator:
    """Multi-timestep asset simulation engine.

    Runs multi-year simulations that step through time, update asset states,
    sample failures using conditional probability, apply intervention responses,
    and track costs/metrics.

    Parameters
    ----------
    deterioration_model : DeteriorationModel
        Model for calculating failure probabilities (e.g., WeibullModel)
    config : SimulationConfig
        Simulation configuration
    interventions : dict[str, InterventionType], optional
        Custom interventions keyed by name. If None, uses defaults.

    Attributes
    ----------
    model : DeteriorationModel
        The deterioration model used for failure probability calculations.
    config : SimulationConfig
        Configuration for the simulation run.
    interventions : dict[str, InterventionType]
        Available intervention types.
    rng : np.random.Generator
        Random number generator for reproducible sampling.

    Examples
    --------
    >>> import pandas as pd
    >>> from asset_optimization import WeibullModel, Simulator, SimulationConfig
    >>> portfolio = pd.read_csv('assets.csv', parse_dates=['install_date'])
    >>> model = WeibullModel({'PVC': (2.5, 50), 'Cast Iron': (3.0, 40)})
    >>> config = SimulationConfig(n_years=10, random_seed=42)
    >>> sim = Simulator(model, config)
    >>> result = sim.run(portfolio)
    >>> print(result)

    Reproducibility
    ---------------
    The same portfolio + same config (with random_seed) produces identical results:

    >>> config = SimulationConfig(n_years=10, random_seed=42)
    >>> result1 = Simulator(model, config).run(portfolio)
    >>> result2 = Simulator(model, config).run(portfolio)
    >>> assert result1.total_cost() == result2.total_cost()  # Always True

    Notes
    -----
    Timestep order: Age -> Failures -> Interventions

    Conditional failure probability is calculated as:
    P(fail in [t, t+1) | survived to t) = (S(t) - S(t+1)) / S(t)
    where S(t) is the survival function.
    """

    def __init__(
        self,
        deterioration_model: DeteriorationModel,
        config: SimulationConfig,
        interventions: Optional[dict[str, InterventionType]] = None,
    ):
        """Initialize Simulator with model, config, and optional custom interventions.

        Parameters
        ----------
        deterioration_model : DeteriorationModel
            Model for calculating failure probabilities.
        config : SimulationConfig
            Simulation configuration.
        interventions : dict[str, InterventionType], optional
            Custom interventions keyed by name. If None, uses default
            DO_NOTHING, INSPECT, REPAIR, REPLACE.
        """
        self.model = deterioration_model
        self.config = config

        # Create isolated RNG for reproducibility
        self.rng = np.random.default_rng(config.random_seed)

        # Set up interventions
        if interventions is None:
            self.interventions = {
                "do_nothing": DO_NOTHING,
                "inspect": INSPECT,
                "repair": REPAIR,
                "replace": REPLACE,
            }
        else:
            self.interventions = interventions

    def run(self, portfolio: pd.DataFrame) -> SimulationResult:
        """Run multi-timestep simulation on portfolio.

        Parameters
        ----------
        portfolio : pd.DataFrame
            Asset portfolio data to simulate.

        Returns
        -------
        SimulationResult
            Results containing summary stats, cost breakdown, and failure log.

        Examples
        --------
        >>> result = sim.run(portfolio)
        >>> print(f"Total cost: ${result.total_cost():,.0f}")
        >>> print(f"Total failures: {result.total_failures()}")
        """
        validated = validate_portfolio(portfolio)
        state = validated.copy()

        # Calculate initial ages from install_date relative to start_year
        start_date = pd.Timestamp(year=self.config.start_year, month=1, day=1)
        state["age"] = (start_date - state["install_date"]).dt.days / 365.25
        state["age"] = state["age"].clip(lower=0)  # No negative ages

        # Initialize tracking structures
        summary_rows = []
        failure_log_rows = []
        cost_breakdown_rows = []
        asset_history_rows = []

        # Run simulation loop
        for year_offset in range(self.config.n_years):
            year = self.config.start_year + year_offset

            # Simulate one timestep
            state, failures_mask, costs, asset_events = self._simulate_timestep(
                state, year
            )

            # Count failures and interventions
            n_failures = failures_mask.sum()
            n_interventions = (
                n_failures if self.config.failure_response != "record_only" else 0
            )

            # Record summary row
            summary_rows.append(
                {
                    "year": year,
                    "total_cost": costs["total"],
                    "failure_count": n_failures,
                    "intervention_count": n_interventions,
                    "avg_age": state["age"].mean(),
                }
            )

            # Record cost breakdown
            cost_breakdown_rows.append(
                {
                    "year": year,
                    "failure_direct_cost": costs["failure_direct"],
                    "failure_consequence_cost": costs["failure_consequence"],
                    "intervention_cost": costs["intervention"],
                }
            )

            # Record failure log entries
            if n_failures > 0:
                failed_assets = state[failures_mask]
                for _, asset in failed_assets.iterrows():
                    failure_log_rows.append(
                        {
                            "year": year,
                            "asset_id": asset["asset_id"],
                            "age_at_failure": asset.get("age_at_failure", asset["age"]),
                            "material": asset["material"],
                            "direct_cost": DEFAULT_FAILURE_DIRECT_COST,
                            "consequence_cost": DEFAULT_FAILURE_CONSEQUENCE_COST,
                        }
                    )

            # Track asset history if enabled
            asset_history_rows.append(
                pd.DataFrame(
                    {
                        "year": year,
                        "asset_id": state["asset_id"].values,
                        "age": state["age"].values,
                        "action": asset_events["action"].values,
                        "failed": asset_events["failed"].values,
                        "failure_cost": asset_events["failure_cost"].values,
                        "intervention_cost": asset_events["intervention_cost"].values,
                        "total_cost": asset_events["total_cost"].values,
                    }
                )
            )

        # Build result DataFrames
        summary = pd.DataFrame(summary_rows)
        cost_breakdown = pd.DataFrame(cost_breakdown_rows)
        failure_log = (
            pd.DataFrame(failure_log_rows)
            if failure_log_rows
            else pd.DataFrame(
                columns=[
                    "year",
                    "asset_id",
                    "age_at_failure",
                    "material",
                    "direct_cost",
                    "consequence_cost",
                ]
            )
        )
        if asset_history_rows:
            asset_history = pd.concat(asset_history_rows, ignore_index=True)
        else:
            asset_history = pd.DataFrame(
                columns=[
                    "year",
                    "asset_id",
                    "age",
                    "action",
                    "failed",
                    "failure_cost",
                    "intervention_cost",
                    "total_cost",
                ]
            )

        return SimulationResult(
            summary=summary,
            cost_breakdown=cost_breakdown,
            failure_log=failure_log,
            config=self.config,
            asset_history=asset_history,
        )

    def _simulate_timestep(
        self, state: pd.DataFrame, year: int
    ) -> tuple[pd.DataFrame, pd.Series, dict, dict]:
        """Simulate a single timestep.

        Order: Age -> Failures -> Interventions

        Parameters
        ----------
        state : pd.DataFrame
            Current asset state.
        year : int
            Current simulation year.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series, dict, dict]
            Updated state, boolean mask of failures, cost dictionary,
            and per-asset event data.
        """
        state = state.copy()

        # Step 1: Increment age by 1 year
        state["age"] = state["age"] + 1

        # Step 2: Calculate conditional failure probability
        probs = self._calculate_conditional_probability(state)

        # Step 3: Sample failures using RNG
        random_draws = self.rng.random(len(state))
        failures_mask = pd.Series(random_draws < probs, index=state.index)

        # Record age at failure before interventions modify it
        state.loc[failures_mask, "age_at_failure"] = state.loc[failures_mask, "age"]

        # Step 4: Apply failure_response intervention to failed assets
        costs = {
            "failure_direct": 0.0,
            "failure_consequence": 0.0,
            "intervention": 0.0,
            "total": 0.0,
        }
        actions = np.full(len(state), "none", dtype=object)
        failure_costs = np.zeros(len(state), dtype=float)
        intervention_costs = np.zeros(len(state), dtype=float)

        n_failures = failures_mask.sum()
        if n_failures > 0:
            # Failure costs
            costs["failure_direct"] = n_failures * DEFAULT_FAILURE_DIRECT_COST
            costs["failure_consequence"] = n_failures * DEFAULT_FAILURE_CONSEQUENCE_COST
            failure_costs[failures_mask.to_numpy()] = (
                DEFAULT_FAILURE_DIRECT_COST + DEFAULT_FAILURE_CONSEQUENCE_COST
            )
            actions[failures_mask.to_numpy()] = self.config.failure_response

            # Apply intervention based on failure_response config
            if self.config.failure_response == "replace":
                intervention = self.interventions.get("replace", REPLACE)
                state.loc[failures_mask, "age"] = state.loc[failures_mask, "age"].apply(
                    intervention.apply_age_effect
                )
                costs["intervention"] = n_failures * intervention.cost
                intervention_costs[failures_mask.to_numpy()] = intervention.cost

            elif self.config.failure_response == "repair":
                intervention = self.interventions.get("repair", REPAIR)
                state.loc[failures_mask, "age"] = state.loc[failures_mask, "age"].apply(
                    intervention.apply_age_effect
                )
                costs["intervention"] = n_failures * intervention.cost
                intervention_costs[failures_mask.to_numpy()] = intervention.cost

            # 'record_only' doesn't apply intervention or add intervention cost

        costs["total"] = (
            costs["failure_direct"]
            + costs["failure_consequence"]
            + costs["intervention"]
        )

        asset_events = {
            "action": pd.Series(actions, index=state.index),
            "failed": pd.Series(failures_mask.to_numpy(), index=state.index),
            "failure_cost": pd.Series(failure_costs, index=state.index),
            "intervention_cost": pd.Series(intervention_costs, index=state.index),
            "total_cost": pd.Series(
                failure_costs + intervention_costs, index=state.index
            ),
        }

        return state, failures_mask, costs, asset_events

    def _calculate_conditional_probability(self, state: pd.DataFrame) -> np.ndarray:
        """Calculate conditional failure probability for each asset.

        Delegates to the configured deterioration model interface.

        Parameters
        ----------
        state : pd.DataFrame
            Current asset state.

        Returns
        -------
        np.ndarray
            Conditional failure probabilities for each asset.
        """
        return self.model.calculate_conditional_probability(state)

    def get_intervention_options(self, state: pd.DataFrame, year: int) -> pd.DataFrame:
        """Generate available intervention options for each asset.

        This method exposes what interventions *could* be done at the current
        timestep. Phase 4 optimization will use this to select optimal actions.

        Parameters
        ----------
        state : pd.DataFrame
            Current asset state with 'asset_id' and 'age' columns.
        year : int
            Current simulation year (for context).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - asset_id: Asset identifier
            - intervention_type: Name of intervention
            - cost: Direct cost of intervention
            - age_effect: Description of age effect

        Examples
        --------
        >>> state = portfolio.data.copy()
        >>> state['age'] = 26.0
        >>> options = sim.get_intervention_options(state, 2026)
        >>> print(options.head())
           asset_id intervention_type     cost    age_effect
        0  PIPE-001         DoNothing      0.0     no change
        1  PIPE-001           Inspect    500.0     no change
        2  PIPE-001            Repair   5000.0     age - 5
        3  PIPE-001           Replace  50000.0     age = 0
        """
        rows = []

        for _, asset in state.iterrows():
            asset_id = asset["asset_id"]
            current_age = asset.get("age", 0.0)

            for intervention in self.interventions.values():
                # Describe age effect
                new_age = intervention.apply_age_effect(current_age)
                if new_age == current_age:
                    age_effect_desc = "no change"
                elif new_age == 0:
                    age_effect_desc = "age = 0"
                else:
                    age_diff = current_age - new_age
                    if age_diff > 0:
                        age_effect_desc = f"age - {age_diff:.0f}"
                    else:
                        age_effect_desc = f"age + {-age_diff:.0f}"

                rows.append(
                    {
                        "asset_id": asset_id,
                        "intervention_type": intervention.name,
                        "cost": intervention.cost,
                        "age_effect": age_effect_desc,
                    }
                )

        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        """Return informative string representation."""
        return (
            f"Simulator(model={self.model.__class__.__name__}, "
            f"n_years={self.config.n_years}, "
            f"seed={self.config.random_seed})"
        )
