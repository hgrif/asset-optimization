"""In-memory DataFrame repository for planner orchestration."""

from __future__ import annotations

import pandas as pd

from asset_optimization.types import DataFrameLike


class DataFrameRepository:
    """AssetRepository implementation backed by in-memory DataFrames.

    Parameters
    ----------
    assets : pd.DataFrame
        Required asset table.
    events : pd.DataFrame, optional
        Historical event log.
    interventions : pd.DataFrame, optional
        Intervention records.
    outcomes : pd.DataFrame, optional
        Intervention outcome records.
    covariates : pd.DataFrame, optional
        Additional model covariates.
    topology : pd.DataFrame, optional
        Optional network topology data.
    """

    def __init__(
        self,
        assets: DataFrameLike,
        events: DataFrameLike | None = None,
        interventions: DataFrameLike | None = None,
        outcomes: DataFrameLike | None = None,
        covariates: DataFrameLike | None = None,
        topology: DataFrameLike | None = None,
    ) -> None:
        self._assets = self._validate_required(assets, name="assets")
        self._events = self._validate_optional(events, name="events")
        self._interventions = self._validate_optional(
            interventions, name="interventions"
        )
        self._outcomes = self._validate_optional(outcomes, name="outcomes")
        self._covariates = self._validate_optional(covariates, name="covariates")
        self._topology = self._validate_optional(topology, name="topology")

    def load_assets(self) -> DataFrameLike:
        """Return a defensive copy of assets."""
        return self._assets.copy(deep=True)

    def load_events(self, event_type: str | None = None) -> DataFrameLike:
        """Return events; optionally filter by event_type when available."""
        events = self._events.copy(deep=True)
        if event_type is None or events.empty or "event_type" not in events.columns:
            return events
        return events.loc[events["event_type"] == event_type].copy(deep=True)

    def load_interventions(self) -> DataFrameLike:
        """Return a defensive copy of interventions."""
        return self._interventions.copy(deep=True)

    def load_outcomes(self) -> DataFrameLike:
        """Return a defensive copy of outcomes."""
        return self._outcomes.copy(deep=True)

    def load_covariates(self) -> DataFrameLike:
        """Return a defensive copy of covariates."""
        return self._covariates.copy(deep=True)

    def load_topology(self) -> DataFrameLike:
        """Return a defensive copy of topology."""
        return self._topology.copy(deep=True)

    @staticmethod
    def _validate_required(data: DataFrameLike, name: str) -> DataFrameLike:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        return data.copy(deep=True)

    @staticmethod
    def _validate_optional(
        data: DataFrameLike | None,
        name: str,
    ) -> DataFrameLike:
        if data is None:
            return pd.DataFrame()
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        return data.copy(deep=True)
