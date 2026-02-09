"""Abstract base class for deterioration models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from asset_optimization.types import PlanningHorizon, ScenarioSet


class DeteriorationModel(ABC):
    """Abstract base for deterioration models.

    All deterioration models must implement:
    - failure_rate(): Calculate instantaneous hazard function
    - transform(): Add failure metrics to portfolio DataFrame
    - calculate_conditional_probability(): Conditional one-step failure probability

    Models accept full portfolio DataFrames and operate vectorized per asset type.
    The transform() method returns a copy with new columns added (immutable pattern).

    Subclasses should:
    1. Accept parameters at __init__
    2. Validate parameters in __init__ (fail fast)
    3. Implement vectorized calculations for performance

    Examples
    --------
    >>> class CustomModel(DeteriorationModel):
    ...     def __init__(self, params):
    ...         self.params = params
    ...
    ...     def failure_rate(self, age, **kwargs):
    ...         return np.zeros_like(age)  # Custom calculation
    ...
    ...     def transform(self, df):
    ...         result = df.copy()
    ...         result['failure_rate'] = self.failure_rate(df['age'].values)
    ...         result['failure_probability'] = 0.0  # Custom calculation
    ...         return result
    """

    @abstractmethod
    def failure_rate(self, age: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate failure rate (hazard function) at given ages.

        Parameters
        ----------
        age : np.ndarray
            Asset ages in years (vectorized input).
        **kwargs : dict
            Model-specific parameters (e.g., shape, scale for Weibull).

        Returns
        -------
        rates : np.ndarray
            Failure rates (instantaneous hazard function values).
            Same shape as input age array.

        Notes
        -----
        Hazard function h(t) represents instantaneous failure rate.
        For Weibull: h(t) = (k/lambda) * (t/lambda)^(k-1)
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add failure rate and probability columns to portfolio DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Portfolio DataFrame with at minimum an age column and
            a type column for parameter lookup.

        Returns
        -------
        enriched : pd.DataFrame
            Copy of input with two new columns added:
            - failure_rate: instantaneous hazard h(t)
            - failure_probability: cumulative probability F(t)

        Raises
        ------
        ValueError
            If required columns missing or asset types not in parameters.
        TypeError
            If age column is not numeric.

        Notes
        -----
        This method MUST return a copy of the input DataFrame.
        The original DataFrame must not be modified (immutable pattern).
        This aligns with pandas copy-on-write behavior in pandas 2.0+.
        """
        pass

    @abstractmethod
    def calculate_conditional_probability(self, state: pd.DataFrame) -> np.ndarray:
        """Calculate one-step conditional failure probability.

        Parameters
        ----------
        state : pd.DataFrame
            Current asset state with the columns needed by the model.

        Returns
        -------
        np.ndarray
            Conditional probabilities:
            P(fail in [t, t+1) | survived to t)
            for each row in ``state``.
        """
        pass

    def fit(
        self,
        assets: pd.DataFrame,
        events: pd.DataFrame,
        covariates: pd.DataFrame | None = None,
    ) -> "DeteriorationModel":
        """Fit model parameters from historical data.

        Default implementation is a no-op because current deterioration
        models are parameterized at initialization time.
        """
        del assets, events, covariates
        return self

    def predict_distribution(
        self,
        assets: pd.DataFrame,
        horizon: PlanningHorizon,
        scenarios: ScenarioSet | None = None,
    ) -> pd.DataFrame:
        """Predict failure distributions in planner-compatible schema.

        Parameters
        ----------
        assets : pd.DataFrame
            Asset records. Must include ``asset_id`` and model-required columns.
        horizon : PlanningHorizon
            Planning window used to determine horizon steps.
        scenarios : ScenarioSet, optional
            Scenario definition. Skeleton implementation ignores scenario-specific
            dynamics and repeats baseline values per scenario.

        Returns
        -------
        pd.DataFrame
            Long-format table with columns:
            ``asset_id``, ``scenario_id``, ``horizon_step``, ``failure_prob``,
            and ``loss_mean``.
        """
        enriched = self.transform(assets)
        if "asset_id" not in enriched.columns:
            raise ValueError("assets must include an 'asset_id' column")
        if "failure_probability" not in enriched.columns:
            raise ValueError("transform() output must include 'failure_probability'")

        columns = [
            "asset_id",
            "scenario_id",
            "horizon_step",
            "failure_prob",
            "loss_mean",
        ]
        if enriched.empty:
            return pd.DataFrame(columns=columns)

        failure_prob = pd.to_numeric(
            enriched["failure_probability"], errors="coerce"
        ).to_numpy(dtype=float)
        failure_prob = np.nan_to_num(failure_prob, nan=0.0)
        failure_prob = np.clip(failure_prob, 0.0, 1.0)
        step_count = self._count_horizon_steps(horizon)
        scenario_ids = self._extract_scenario_ids(scenarios)

        rows = []
        for scenario_id in scenario_ids:
            for horizon_step in range(step_count):
                rows.append(
                    pd.DataFrame(
                        {
                            "asset_id": enriched["asset_id"].to_numpy(),
                            "scenario_id": scenario_id,
                            "horizon_step": horizon_step,
                            "failure_prob": failure_prob,
                            "loss_mean": np.zeros(len(enriched), dtype=float),
                        }
                    )
                )

        return pd.concat(rows, ignore_index=True)

    def describe(self) -> dict[str, Any]:
        """Return model metadata for planner orchestration."""
        return {"model_type": self.__class__.__name__}

    @staticmethod
    def _count_horizon_steps(horizon: PlanningHorizon) -> int:
        freq_map = {"monthly": "M", "quarterly": "Q", "yearly": "Y"}
        freq = freq_map[horizon.step]
        periods = pd.period_range(
            start=horizon.start_date,
            end=horizon.end_date,
            freq=freq,
        )
        return max(len(periods), 1)

    @staticmethod
    def _extract_scenario_ids(scenarios: ScenarioSet | None) -> list[str]:
        if scenarios is None or scenarios.scenarios.empty:
            return ["baseline"]

        unique_ids = scenarios.scenarios["scenario_id"].drop_duplicates()
        scenario_ids = [str(value) for value in unique_ids.tolist()]
        return scenario_ids or ["baseline"]
