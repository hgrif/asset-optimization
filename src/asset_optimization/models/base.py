"""Abstract base class for deterioration models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from asset_optimization.types import PlanningHorizon, ScenarioSet


class DeteriorationModel(ABC):
    """Abstract base for deterioration models.

    All deterioration models must implement:
    - calculate_conditional_probability(): Conditional one-step failure probability
    - _enrich_portfolio(): Add failure metrics to portfolio DataFrame (private)
    - _failure_rate(): Calculate instantaneous hazard function (private)

    Public API (RiskModel protocol):
    - fit(): Fit model from historical data (default no-op)
    - predict_distribution(): Predict failure distributions for planner
    - describe(): Return model metadata

    Subclasses should:
    1. Accept parameters at __init__
    2. Validate parameters in __init__ (fail fast)
    3. Implement vectorized calculations for performance
    """

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
        enriched = self._enrich_portfolio(assets)
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
