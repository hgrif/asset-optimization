"""Group failure propagation wrapper for RiskModel implementations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from asset_optimization.protocols import RiskModel
from asset_optimization.types import DataFrameLike, PlanningHorizon, ScenarioSet


class GroupPropagationRiskModel:
    """Wrapper that increases failure probabilities for grouped assets.

    This model wraps an existing RiskModel and applies deterministic failure
    propagation for assets that share a group identifier. Assets in the same
    group experience correlated risk: when one asset has high failure probability,
    all assets in the group receive a proportional increase.

    The propagation formula for each group g:
        P_group = 1 - Π(1 - P_i)  for all i in group g
        P_i_new = min(P_i * (1 + propagation_factor * P_group), 1.0)

    This mean-field approximation increases failure probability without requiring
    a simulation loop.

    Parameters
    ----------
    base_model : RiskModel
        The underlying risk model to wrap. Must implement the RiskModel protocol.
    propagation_factor : float, default=0.5
        Multiplier for group-level risk propagation. Must be non-negative and finite.
        A value of 0 disables propagation (returns baseline predictions).
    group_column : str, default="group_id"
        Column name in the assets DataFrame containing group identifiers.
        Must be a non-empty string.
    min_group_size : int, default=2
        Minimum group size for propagation to apply. Groups with fewer assets
        are treated as singletons (no propagation). Must be at least 2.

    Examples
    --------
    >>> from asset_optimization.models import WeibullModel, GroupPropagationRiskModel
    >>> base_model = WeibullModel({"PVC": (2.5, 50)})
    >>> model = GroupPropagationRiskModel(
    ...     base_model=base_model,
    ...     propagation_factor=0.5,
    ...     group_column="group_id",
    ... )
    >>> assets = pd.DataFrame({
    ...     "asset_id": ["A", "B", "C"],
    ...     "asset_type": ["PVC", "PVC", "PVC"],
    ...     "age": [10, 20, 30],
    ...     "group_id": ["group1", "group1", None],
    ... })
    >>> horizon = PlanningHorizon(start_date="2026-01-01", end_date="2027-01-01", step="yearly")
    >>> predictions = model.predict_distribution(assets, horizon)
    """

    def __init__(
        self,
        base_model: RiskModel,
        propagation_factor: float = 0.5,
        group_column: str = "group_id",
        min_group_size: int = 2,
    ):
        # Validate base_model implements RiskModel protocol
        if not isinstance(base_model, RiskModel):
            raise TypeError(
                f"base_model must implement RiskModel protocol, got {type(base_model).__name__}"
            )

        # Validate propagation_factor
        if not math.isfinite(propagation_factor):
            raise ValueError(
                f"propagation_factor must be finite, got {propagation_factor}"
            )
        if propagation_factor < 0:
            raise ValueError(
                f"propagation_factor must be non-negative, got {propagation_factor}"
            )

        # Validate group_column
        if not isinstance(group_column, str) or len(group_column) == 0:
            raise ValueError(
                f"group_column must be a non-empty string, got {group_column!r}"
            )

        # Validate min_group_size
        if not isinstance(min_group_size, int) or min_group_size < 2:
            raise ValueError(
                f"min_group_size must be an integer >= 2, got {min_group_size}"
            )

        self.base_model = base_model
        self.propagation_factor = propagation_factor
        self.group_column = group_column
        self.min_group_size = min_group_size

    def fit(
        self,
        assets: DataFrameLike,
        events: DataFrameLike,
        covariates: DataFrameLike | None = None,
    ) -> "GroupPropagationRiskModel":
        """Fit the base model to historical data.

        Delegates to the base model's fit method.

        Parameters
        ----------
        assets : DataFrameLike
            Asset records with historical state.
        events : DataFrameLike
            Historical event data.
        covariates : DataFrameLike, optional
            Additional covariate data.

        Returns
        -------
        GroupPropagationRiskModel
            Self, for method chaining.
        """
        self.base_model.fit(assets, events, covariates)
        return self

    def predict_distribution(
        self,
        assets: DataFrameLike,
        horizon: PlanningHorizon,
        scenarios: ScenarioSet | None = None,
    ) -> DataFrameLike:
        """Predict failure distributions with group propagation.

        Parameters
        ----------
        assets : DataFrameLike
            Asset records. Must include ``asset_id`` and columns required by
            the base model. May optionally include the group column.
        horizon : PlanningHorizon
            Planning window for predictions.
        scenarios : ScenarioSet, optional
            Scenario definitions for multi-scenario predictions.

        Returns
        -------
        DataFrameLike
            Long-format table with columns: ``asset_id``, ``scenario_id``,
            ``horizon_step``, ``failure_prob``, ``loss_mean``.
            For grouped assets, ``failure_prob`` is increased based on group risk.
        """
        # Get baseline predictions from base model
        baseline = self.base_model.predict_distribution(assets, horizon, scenarios)

        # If no group column in assets or propagation disabled, return baseline
        if self.group_column not in assets.columns or self.propagation_factor == 0:
            return baseline

        # Convert to DataFrame if needed
        if not isinstance(baseline, pd.DataFrame):
            baseline = pd.DataFrame(baseline)
        if not isinstance(assets, pd.DataFrame):
            assets = pd.DataFrame(assets)

        # Join group_id onto baseline predictions
        group_mapping = assets[["asset_id", self.group_column]].copy()
        merged = baseline.merge(group_mapping, on="asset_id", how="left")

        # Compute group sizes and filter to eligible groups (size >= min_group_size, non-null)
        group_sizes = (
            group_mapping[self.group_column].value_counts(dropna=True).to_dict()
        )
        eligible_groups = {
            g for g, size in group_sizes.items() if size >= self.min_group_size
        }

        if not eligible_groups:
            # No groups to propagate, return baseline
            return baseline

        # Mark eligible rows (in groups that meet size threshold)
        merged["_eligible"] = merged[self.group_column].isin(eligible_groups)

        # Compute group-level failure probability: P_group = 1 - Π(1 - P_i)
        def compute_group_failure_prob(group_df: pd.DataFrame) -> float:
            """Compute joint failure probability for a group."""
            probs = group_df["failure_prob"].to_numpy()
            # P_group = 1 - Π(1 - P_i)
            return 1.0 - np.prod(1.0 - probs)

        # Group by (scenario_id, horizon_step, group_id) and compute P_group
        eligible_rows = merged[merged["_eligible"]].copy()
        if not eligible_rows.empty:
            group_agg = (
                eligible_rows.groupby(
                    ["scenario_id", "horizon_step", self.group_column], dropna=False
                )
                .apply(compute_group_failure_prob, include_groups=False)
                .rename("_p_group")
                .reset_index()
            )

            # Join P_group back to merged
            merged = merged.merge(
                group_agg,
                on=["scenario_id", "horizon_step", self.group_column],
                how="left",
            )

            # Apply propagation formula: P_i_new = min(P_i * (1 + factor * P_group), 1.0)
            has_propagation = merged["_p_group"].notna()
            merged.loc[has_propagation, "failure_prob"] = np.minimum(
                merged.loc[has_propagation, "failure_prob"]
                * (
                    1.0
                    + self.propagation_factor * merged.loc[has_propagation, "_p_group"]
                ),
                1.0,
            )

        # Drop helper columns and return
        result = merged.drop(
            columns=[self.group_column, "_eligible"]
            + (["_p_group"] if "_p_group" in merged.columns else []),
            errors="ignore",
        )

        # Ensure column order matches baseline
        return result[baseline.columns]

    def describe(self) -> dict[str, Any]:
        """Return model metadata including base model and propagation settings.

        Returns
        -------
        dict[str, Any]
            Metadata dictionary including base model description and propagation
            parameters.
        """
        base_description = self.base_model.describe()
        return {
            **base_description,
            "wrapper": "GroupPropagationRiskModel",
            "propagation_factor": self.propagation_factor,
            "group_column": self.group_column,
            "min_group_size": self.min_group_size,
        }


__all__ = ["GroupPropagationRiskModel"]
