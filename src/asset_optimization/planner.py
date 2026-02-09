"""Planner orchestrator for Proposal A service composition."""

from __future__ import annotations

from typing import Any

import pandas as pd

from asset_optimization.constraints import ConstraintSet
from asset_optimization.exceptions import ModelError
from asset_optimization.objective import Objective
from asset_optimization.protocols import (
    AssetRepository,
    InterventionEffectModel,
    NetworkSimulator,
    PlanOptimizer,
    RiskModel,
)
from asset_optimization.types import (
    DataFrameLike,
    PlanResult,
    PlanningHorizon,
    ScenarioSet,
    ValidationReport,
)

_REQUIRED_ASSET_COLUMNS = {"asset_id", "asset_type", "install_date"}
_DEFAULT_ACTION_TYPE = "replace"
_DEFAULT_DIRECT_COST = 1000.0


class Planner:
    """Coordinate repository, model, simulation, and optimization services.

    Parameters
    ----------
    repository : AssetRepository
        Data access service for assets/events/interventions.
    risk_model : RiskModel
        Risk service used to predict failure distributions.
    effect_model : InterventionEffectModel
        Effect service used to estimate intervention outcomes.
    simulator : NetworkSimulator
        Optional network consequence simulator.
    optimizer : PlanOptimizer
        Portfolio optimization engine.
    registry : Any, optional
        Optional plugin registry reference for future extension points.
    """

    def __init__(
        self,
        repository: AssetRepository,
        risk_model: RiskModel,
        effect_model: InterventionEffectModel,
        simulator: NetworkSimulator,
        optimizer: PlanOptimizer,
        registry: Any | None = None,
    ) -> None:
        self.repository = repository
        self.risk_model = risk_model
        self.effect_model = effect_model
        self.simulator = simulator
        self.optimizer = optimizer
        self.registry = registry
        self._is_fitted = False

    def validate_inputs(self) -> ValidationReport:
        """Validate required planner input columns on repository assets."""
        checks = {
            "assets_loaded": False,
            "required_columns": False,
            "unique_asset_id": False,
        }
        warnings: list[str] = []

        try:
            assets = self.repository.load_assets()
        except Exception as exc:  # pragma: no cover - defensive guard
            warnings.append(f"failed to load assets: {exc}")
            return ValidationReport(passed=False, checks=checks, warnings=warnings)

        checks["assets_loaded"] = isinstance(assets, pd.DataFrame)
        if not checks["assets_loaded"]:
            warnings.append("repository.load_assets() must return a pandas DataFrame")
            return ValidationReport(passed=False, checks=checks, warnings=warnings)

        missing_columns = sorted(_REQUIRED_ASSET_COLUMNS.difference(assets.columns))
        checks["required_columns"] = not missing_columns
        if missing_columns:
            warnings.append(
                "assets missing required columns: " + ", ".join(missing_columns)
            )

        if "asset_id" in assets.columns:
            duplicated = assets["asset_id"].duplicated(keep=False)
            duplicate_ids = (
                assets.loc[duplicated, "asset_id"]
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            checks["unique_asset_id"] = not duplicate_ids
            if duplicate_ids:
                warnings.append(
                    "duplicate asset_id values found: " + ", ".join(duplicate_ids)
                )

        return ValidationReport(
            passed=all(checks.values()),
            checks=checks,
            warnings=warnings,
        )

    def fit(self) -> "Planner":
        """Fit planner services from repository data."""
        assets = self.repository.load_assets()
        events = self.repository.load_events()
        interventions = self.repository.load_interventions()
        outcomes = self.repository.load_outcomes()
        covariates = self.repository.load_covariates()

        try:
            self.risk_model.fit(assets=assets, events=events, covariates=covariates)
        except Exception as exc:
            raise self._to_model_error(
                phase="risk_model.fit",
                service=self.risk_model,
                original_error=exc,
            ) from exc

        try:
            self.effect_model.fit(interventions=interventions, outcomes=outcomes)
        except Exception as exc:
            raise self._to_model_error(
                phase="effect_model.fit",
                service=self.effect_model,
                original_error=exc,
            ) from exc

        self._is_fitted = True
        return self

    def propose_actions(
        self,
        horizon: PlanningHorizon,
        scenarios: ScenarioSet | None = None,
    ) -> DataFrameLike:
        """Build and score candidate actions for optimization."""
        if not self._is_fitted:
            raise RuntimeError(
                "Planner must be fitted before calling propose_actions()"
            )

        assets = self.repository.load_assets()
        failures = self._predict_failures(
            assets=assets, horizon=horizon, scenarios=scenarios
        )
        interventions = self.repository.load_interventions()
        candidates = self._build_candidate_actions(
            assets=assets,
            failures=failures,
            interventions=interventions,
        )

        try:
            scored_candidates = self.effect_model.estimate_effect(
                candidate_actions=candidates,
                horizon=horizon,
                scenarios=scenarios,
            )
        except Exception as exc:
            raise self._to_model_error(
                phase="effect_model.estimate_effect",
                service=self.effect_model,
                original_error=exc,
            ) from exc

        topology = self.repository.load_topology()
        if topology.empty:
            return scored_candidates

        failures_for_simulation = failures.copy(deep=True)
        if "consequence_cost" not in failures_for_simulation.columns:
            if "loss_mean" in failures_for_simulation.columns:
                failures_for_simulation["consequence_cost"] = pd.to_numeric(
                    failures_for_simulation["loss_mean"],
                    errors="coerce",
                ).fillna(0.0)
            else:
                failures_for_simulation["consequence_cost"] = 0.0

        try:
            return self.simulator.simulate(
                topology=topology,
                failures=failures_for_simulation,
                actions=scored_candidates,
                scenarios=scenarios,
            )
        except Exception as exc:
            raise self._to_model_error(
                phase="simulator.simulate",
                service=self.simulator,
                original_error=exc,
            ) from exc

    def optimize_plan(
        self,
        horizon: PlanningHorizon,
        scenarios: ScenarioSet | None,
        objective: Objective,
        constraints: ConstraintSet,
        risk_measure: str = "expected_value",
    ) -> PlanResult:
        """Generate and optimize a plan for the provided objective/constraints."""
        candidates = self.propose_actions(horizon=horizon, scenarios=scenarios)
        return self.optimizer.solve(
            objective=objective,
            constraints=constraints,
            candidates=candidates,
            risk_measure=risk_measure,
        )

    def _predict_failures(
        self,
        assets: DataFrameLike,
        horizon: PlanningHorizon,
        scenarios: ScenarioSet | None,
    ) -> DataFrameLike:
        try:
            failures = self.risk_model.predict_distribution(
                assets=assets,
                horizon=horizon,
                scenarios=scenarios,
            )
        except Exception as exc:
            raise self._to_model_error(
                phase="risk_model.predict_distribution",
                service=self.risk_model,
                original_error=exc,
            ) from exc

        if not isinstance(failures, pd.DataFrame):
            raise ModelError(
                "risk_model.predict_distribution must return a pandas DataFrame",
                details={"phase": "risk_model.predict_distribution"},
            )
        if "asset_id" not in failures.columns:
            raise ModelError(
                "risk_model output must include 'asset_id'",
                details={"phase": "risk_model.predict_distribution"},
            )
        return failures

    def _build_candidate_actions(
        self,
        assets: DataFrameLike,
        failures: DataFrameLike,
        interventions: DataFrameLike,
    ) -> DataFrameLike:
        failure_summary = self._summarize_failures(assets=assets, failures=failures)
        asset_context = self._extract_asset_context(assets)
        candidates = failure_summary.merge(asset_context, on="asset_id", how="left")

        if interventions.empty:
            candidates["action_type"] = _DEFAULT_ACTION_TYPE
            candidates["direct_cost"] = self._default_direct_costs(assets=assets)
        elif "asset_id" in interventions.columns:
            candidates = candidates.merge(
                interventions.copy(deep=True), on="asset_id", how="left"
            )
        else:
            templates = interventions.copy(deep=True)
            templates["_template_key"] = 1
            candidates["_template_key"] = 1
            candidates = candidates.merge(
                templates, on="_template_key", how="left"
            ).drop(columns=["_template_key"])

        if "action_type" not in candidates.columns:
            candidates["action_type"] = _DEFAULT_ACTION_TYPE
        candidates["action_type"] = candidates["action_type"].fillna(
            _DEFAULT_ACTION_TYPE
        )

        if "direct_cost" not in candidates.columns:
            candidates["direct_cost"] = _DEFAULT_DIRECT_COST
        candidates["direct_cost"] = pd.to_numeric(
            candidates["direct_cost"],
            errors="coerce",
        ).fillna(_DEFAULT_DIRECT_COST)

        if "consequence_cost" in candidates.columns:
            candidates["consequence_cost"] = pd.to_numeric(
                candidates["consequence_cost"],
                errors="coerce",
            ).fillna(candidates["loss_mean"])
        else:
            candidates["consequence_cost"] = candidates["loss_mean"]

        candidates["failure_prob"] = pd.to_numeric(
            candidates["failure_prob"],
            errors="coerce",
        ).fillna(0.0)
        candidates["loss_mean"] = pd.to_numeric(
            candidates["loss_mean"],
            errors="coerce",
        ).fillna(0.0)

        ordered_columns = [
            "asset_id",
            "asset_type",
            "action_type",
            "failure_prob",
            "loss_mean",
            "consequence_cost",
            "direct_cost",
        ]
        remaining_columns = [
            column
            for column in candidates.columns
            if column not in set(ordered_columns)
        ]
        return candidates[ordered_columns + remaining_columns]

    @staticmethod
    def _summarize_failures(
        assets: DataFrameLike, failures: DataFrameLike
    ) -> DataFrameLike:
        working = failures.copy(deep=True)
        if "failure_prob" in working.columns:
            probability_source = working["failure_prob"]
        elif "failure_probability" in working.columns:
            probability_source = working["failure_probability"]
        else:
            probability_source = pd.Series(0.0, index=working.index, dtype=float)
        working["failure_prob"] = pd.to_numeric(
            probability_source, errors="coerce"
        ).fillna(0.0)

        if "loss_mean" in working.columns:
            working["loss_mean"] = pd.to_numeric(
                working["loss_mean"], errors="coerce"
            ).fillna(0.0)
        else:
            working["loss_mean"] = 0.0

        aggregated = (
            working.groupby("asset_id", as_index=False)
            .agg(
                failure_prob=("failure_prob", "mean"),
                loss_mean=("loss_mean", "mean"),
            )
            .sort_values("asset_id")
        )

        all_assets = assets[["asset_id"]].drop_duplicates()
        summary = all_assets.merge(aggregated, on="asset_id", how="left")
        summary["failure_prob"] = summary["failure_prob"].fillna(0.0)
        summary["loss_mean"] = summary["loss_mean"].fillna(0.0)
        return summary

    @staticmethod
    def _extract_asset_context(assets: DataFrameLike) -> DataFrameLike:
        columns = ["asset_id"]
        if "asset_type" in assets.columns:
            columns.append("asset_type")
        context = assets[columns].drop_duplicates(subset=["asset_id"]).copy(deep=True)
        if "asset_type" not in context.columns:
            context["asset_type"] = pd.NA
        return context

    @staticmethod
    def _default_direct_costs(assets: DataFrameLike) -> pd.Series:
        cost_column_candidates = ("direct_cost", "replacement_cost", "estimated_cost")
        for column in cost_column_candidates:
            if column in assets.columns:
                return pd.to_numeric(assets[column], errors="coerce").fillna(
                    _DEFAULT_DIRECT_COST
                )
        return pd.Series(_DEFAULT_DIRECT_COST, index=assets.index, dtype=float)

    @staticmethod
    def _to_model_error(
        phase: str,
        service: Any,
        original_error: Exception,
    ) -> ModelError:
        model_type = service.__class__.__name__
        return ModelError(
            f"{phase} failed: {original_error}",
            details={"phase": phase, "model_type": model_type},
        )


__all__ = ["Planner"]
