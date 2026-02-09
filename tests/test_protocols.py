"""Tests for Proposal A service protocol contracts."""

import inspect

import pandas as pd

from asset_optimization.protocols import (
    AssetRepository,
    InterventionEffectModel,
    NetworkSimulator,
    PlanOptimizer,
    RiskModel,
)
from asset_optimization.types import PlanResult


class DummyRepository:
    """Minimal structural implementation of AssetRepository."""

    def load_assets(self) -> pd.DataFrame:
        return pd.DataFrame()

    def load_events(self, event_type: str | None = None) -> pd.DataFrame:
        return pd.DataFrame()

    def load_interventions(self) -> pd.DataFrame:
        return pd.DataFrame()

    def load_outcomes(self) -> pd.DataFrame:
        return pd.DataFrame()

    def load_covariates(self) -> pd.DataFrame:
        return pd.DataFrame()

    def load_topology(self) -> pd.DataFrame:
        return pd.DataFrame()


class DummyRiskModel:
    """Minimal structural implementation of RiskModel."""

    def fit(
        self,
        assets: pd.DataFrame,
        events: pd.DataFrame,
        covariates: pd.DataFrame | None = None,
    ) -> "DummyRiskModel":
        return self

    def predict_distribution(
        self,
        assets: pd.DataFrame,
        horizon: object,
        scenarios: object | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def describe(self) -> dict[str, str]:
        return {"model_type": "dummy"}


class DummyEffectModel:
    """Minimal structural implementation of InterventionEffectModel."""

    def fit(
        self,
        interventions: pd.DataFrame,
        outcomes: pd.DataFrame,
    ) -> "DummyEffectModel":
        return self

    def estimate_effect(
        self,
        candidate_actions: pd.DataFrame,
        horizon: object,
        scenarios: object | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def describe(self) -> dict[str, str]:
        return {"model_type": "dummy_effect"}


class DummySimulator:
    """Minimal structural implementation of NetworkSimulator."""

    def simulate(
        self,
        topology: pd.DataFrame,
        failures: pd.DataFrame,
        actions: pd.DataFrame,
        scenarios: object | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()


class DummyOptimizer:
    """Minimal structural implementation of PlanOptimizer."""

    def solve(
        self,
        objective: object,
        constraints: object,
        candidates: pd.DataFrame,
        risk_measure: str = "expected_value",
    ) -> PlanResult:
        return PlanResult(
            selected_actions=pd.DataFrame(),
            objective_breakdown={},
            constraint_shadow_prices={},
        )


class IncompleteRepository:
    """Missing required protocol methods on purpose."""

    def load_assets(self) -> pd.DataFrame:
        return pd.DataFrame()


def test_protocols_are_runtime_checkable() -> None:
    """Protocols should support isinstance structural checks at runtime."""
    assert isinstance(DummyRepository(), AssetRepository)
    assert isinstance(DummyRiskModel(), RiskModel)
    assert isinstance(DummyEffectModel(), InterventionEffectModel)
    assert isinstance(DummySimulator(), NetworkSimulator)
    assert isinstance(DummyOptimizer(), PlanOptimizer)


def test_structural_check_fails_when_methods_are_missing() -> None:
    """Runtime protocol checks fail for incomplete implementations."""
    assert not isinstance(IncompleteRepository(), AssetRepository)


def test_optional_defaults_for_scenarios_and_event_type() -> None:
    """Step 2 contract keeps scenarios/event_type optional in signatures."""
    events_sig = inspect.signature(AssetRepository.load_events)
    risk_sig = inspect.signature(RiskModel.predict_distribution)
    effect_sig = inspect.signature(InterventionEffectModel.estimate_effect)
    simulator_sig = inspect.signature(NetworkSimulator.simulate)

    assert events_sig.parameters["event_type"].default is None
    assert risk_sig.parameters["scenarios"].default is None
    assert effect_sig.parameters["scenarios"].default is None
    assert simulator_sig.parameters["scenarios"].default is None
