"""Tests for Planner orchestrator lifecycle."""

from __future__ import annotations

import pandas as pd
import pytest

from asset_optimization.constraints import ConstraintSet
from asset_optimization.exceptions import ModelError
from asset_optimization.objective import ObjectiveBuilder
from asset_optimization.planner import Planner
from asset_optimization.repositories import DataFrameRepository
from asset_optimization.types import PlanResult, PlanningHorizon, ScenarioSet


class SpyRiskModel:
    """Risk model spy for planner lifecycle tests."""

    def __init__(self, fail_on_fit: bool = False) -> None:
        self.fail_on_fit = fail_on_fit
        self.fit_calls: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
        self.predict_calls: list[
            tuple[pd.DataFrame, PlanningHorizon, ScenarioSet | None]
        ] = []

    def fit(
        self,
        assets: pd.DataFrame,
        events: pd.DataFrame,
        covariates: pd.DataFrame | None = None,
    ) -> "SpyRiskModel":
        if self.fail_on_fit:
            raise ValueError("risk fit exploded")
        if covariates is None:
            covariates = pd.DataFrame()
        self.fit_calls.append(
            (assets.copy(deep=True), events.copy(deep=True), covariates.copy(deep=True))
        )
        return self

    def predict_distribution(
        self,
        assets: pd.DataFrame,
        horizon: PlanningHorizon,
        scenarios: ScenarioSet | None = None,
    ) -> pd.DataFrame:
        self.predict_calls.append((assets.copy(deep=True), horizon, scenarios))
        return pd.DataFrame(
            {
                "asset_id": assets["asset_id"].tolist(),
                "scenario_id": ["baseline"] * len(assets),
                "horizon_step": [0] * len(assets),
                "failure_prob": [0.6, 0.2][: len(assets)],
                "loss_mean": [200.0, 80.0][: len(assets)],
            }
        )

    def describe(self) -> dict[str, str]:
        return {"model_type": "spy_risk"}


class SpyEffectModel:
    """Effect model spy that computes simple expected benefit columns."""

    def __init__(self) -> None:
        self.fit_calls: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        self.estimate_calls: list[
            tuple[pd.DataFrame, PlanningHorizon, ScenarioSet | None]
        ] = []

    def fit(
        self, interventions: pd.DataFrame, outcomes: pd.DataFrame
    ) -> "SpyEffectModel":
        self.fit_calls.append((interventions.copy(deep=True), outcomes.copy(deep=True)))
        return self

    def estimate_effect(
        self,
        candidate_actions: pd.DataFrame,
        horizon: PlanningHorizon,
        scenarios: ScenarioSet | None = None,
    ) -> pd.DataFrame:
        self.estimate_calls.append(
            (candidate_actions.copy(deep=True), horizon, scenarios)
        )
        result = candidate_actions.copy(deep=True)
        result["expected_risk_reduction"] = result["failure_prob"] * 0.5
        result["expected_benefit"] = (
            result["expected_risk_reduction"] * result["consequence_cost"]
        )
        return result

    def describe(self) -> dict[str, str]:
        return {"model_type": "spy_effect"}


class SpySimulator:
    """Simulator spy used to verify optional network simulation call."""

    def __init__(self) -> None:
        self.calls: list[
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ScenarioSet | None]
        ] = []

    def simulate(
        self,
        topology: pd.DataFrame,
        failures: pd.DataFrame,
        actions: pd.DataFrame,
        scenarios: ScenarioSet | None = None,
    ) -> pd.DataFrame:
        self.calls.append(
            (
                topology.copy(deep=True),
                failures.copy(deep=True),
                actions.copy(deep=True),
                scenarios,
            )
        )
        result = actions.copy(deep=True)
        result["consequence_cost"] = result["consequence_cost"] + 10.0
        return result


class SpyOptimizer:
    """Optimizer spy returning a deterministic PlanResult."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, object, pd.DataFrame, str]] = []

    def solve(
        self,
        objective: object,
        constraints: object,
        candidates: pd.DataFrame,
        risk_measure: str = "expected_value",
    ) -> PlanResult:
        self.calls.append(
            (objective, constraints, candidates.copy(deep=True), risk_measure)
        )
        selected = (
            candidates.sort_values("expected_benefit", ascending=False)
            .head(1)
            .reset_index(drop=True)
        )
        return PlanResult(
            selected_actions=selected,
            objective_breakdown={
                "objective_total": float(selected["expected_benefit"].sum())
            },
            constraint_shadow_prices={},
            metadata={"risk_measure": risk_measure},
        )


def _build_repository(include_topology: bool = False) -> DataFrameRepository:
    assets = pd.DataFrame(
        {
            "asset_id": ["A1", "A2"],
            "asset_type": ["pipe", "pipe"],
            "install_date": pd.to_datetime(["2000-01-01", "2010-01-01"]),
        }
    )
    events = pd.DataFrame({"asset_id": ["A1"], "event_type": ["break"]})
    interventions = pd.DataFrame(
        {
            "asset_id": ["A1", "A2"],
            "action_type": ["replace", "repair"],
            "direct_cost": [50000.0, 8000.0],
        }
    )
    outcomes = pd.DataFrame({"asset_id": ["A1"], "observed_loss": [1000.0]})
    covariates = pd.DataFrame({"asset_id": ["A1"], "traffic_index": [0.9]})
    topology = (
        pd.DataFrame({"from_asset_id": ["A1"], "to_asset_id": ["A2"]})
        if include_topology
        else pd.DataFrame()
    )
    return DataFrameRepository(
        assets=assets,
        events=events,
        interventions=interventions,
        outcomes=outcomes,
        covariates=covariates,
        topology=topology,
    )


def test_validate_inputs_passes_with_required_columns_and_unique_ids() -> None:
    planner = Planner(
        repository=_build_repository(),
        risk_model=SpyRiskModel(),
        effect_model=SpyEffectModel(),
        simulator=SpySimulator(),
        optimizer=SpyOptimizer(),
    )

    report = planner.validate_inputs()

    assert report.passed is True
    assert report.checks["required_columns"] is True
    assert report.checks["unique_asset_id"] is True
    assert report.warnings == []


def test_validate_inputs_flags_missing_columns_and_duplicate_asset_ids() -> None:
    repository = DataFrameRepository(
        assets=pd.DataFrame({"asset_id": ["A1", "A1"], "asset_type": ["pipe", "pipe"]})
    )
    planner = Planner(
        repository=repository,
        risk_model=SpyRiskModel(),
        effect_model=SpyEffectModel(),
        simulator=SpySimulator(),
        optimizer=SpyOptimizer(),
    )

    report = planner.validate_inputs()

    assert report.passed is False
    assert report.checks["required_columns"] is False
    assert report.checks["unique_asset_id"] is False
    assert any("install_date" in warning for warning in report.warnings)
    assert any("duplicate asset_id" in warning for warning in report.warnings)


def test_fit_calls_risk_and_effect_model_with_repository_data() -> None:
    risk_model = SpyRiskModel()
    effect_model = SpyEffectModel()
    planner = Planner(
        repository=_build_repository(),
        risk_model=risk_model,
        effect_model=effect_model,
        simulator=SpySimulator(),
        optimizer=SpyOptimizer(),
    )

    fitted = planner.fit()

    assert fitted is planner
    assert len(risk_model.fit_calls) == 1
    assert len(effect_model.fit_calls) == 1
    risk_assets, risk_events, risk_covariates = risk_model.fit_calls[0]
    assert set(risk_assets.columns) >= {"asset_id", "asset_type", "install_date"}
    assert set(risk_events.columns) >= {"asset_id", "event_type"}
    assert set(risk_covariates.columns) >= {"asset_id", "traffic_index"}


def test_fit_wraps_model_failures_in_model_error() -> None:
    planner = Planner(
        repository=_build_repository(),
        risk_model=SpyRiskModel(fail_on_fit=True),
        effect_model=SpyEffectModel(),
        simulator=SpySimulator(),
        optimizer=SpyOptimizer(),
    )

    with pytest.raises(ModelError, match="risk_model.fit"):
        planner.fit()


def test_propose_actions_requires_fit_before_inference() -> None:
    planner = Planner(
        repository=_build_repository(),
        risk_model=SpyRiskModel(),
        effect_model=SpyEffectModel(),
        simulator=SpySimulator(),
        optimizer=SpyOptimizer(),
    )
    horizon = PlanningHorizon("2026-01-01", "2026-12-31", "yearly")

    with pytest.raises(RuntimeError, match="must be fitted"):
        planner.propose_actions(horizon=horizon)


def test_propose_actions_scores_and_simulates_when_topology_is_present() -> None:
    risk_model = SpyRiskModel()
    effect_model = SpyEffectModel()
    simulator = SpySimulator()
    planner = Planner(
        repository=_build_repository(include_topology=True),
        risk_model=risk_model,
        effect_model=effect_model,
        simulator=simulator,
        optimizer=SpyOptimizer(),
    )
    horizon = PlanningHorizon("2026-01-01", "2026-12-31", "yearly")
    scenarios = ScenarioSet(
        pd.DataFrame(
            {
                "scenario_id": ["s1"],
                "variable": ["demand"],
                "timestamp": ["2026-01-01"],
                "value": [1.0],
                "probability": [1.0],
            }
        )
    )
    planner.fit()

    candidates = planner.propose_actions(horizon=horizon, scenarios=scenarios)

    assert len(risk_model.predict_calls) == 1
    assert len(effect_model.estimate_calls) == 1
    assert len(simulator.calls) == 1
    assert candidates["consequence_cost"].tolist() == pytest.approx([210.0, 90.0])
    assert "expected_benefit" in candidates.columns
    assert effect_model.estimate_calls[0][2] is scenarios
    assert simulator.calls[0][3] is scenarios


def test_optimize_plan_passes_candidates_to_optimizer() -> None:
    optimizer = SpyOptimizer()
    planner = Planner(
        repository=_build_repository(),
        risk_model=SpyRiskModel(),
        effect_model=SpyEffectModel(),
        simulator=SpySimulator(),
        optimizer=optimizer,
    )
    horizon = PlanningHorizon("2026-01-01", "2026-12-31", "yearly")
    objective = ObjectiveBuilder().add_expected_risk_reduction().build()
    constraints = ConstraintSet().add_budget_limit(100000.0)
    planner.fit()

    result = planner.optimize_plan(
        horizon=horizon,
        scenarios=None,
        objective=objective,
        constraints=constraints,
        risk_measure="cvar_95",
    )

    assert isinstance(result, PlanResult)
    assert len(optimizer.calls) == 1
    _, _, candidates, risk_measure = optimizer.calls[0]
    assert risk_measure == "cvar_95"
    assert {"asset_id", "action_type", "direct_cost", "expected_benefit"}.issubset(
        candidates.columns
    )
