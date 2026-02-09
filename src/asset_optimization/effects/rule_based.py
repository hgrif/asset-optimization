"""Rule-based intervention effect model for planner candidate scoring."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from asset_optimization.exceptions import ModelError
from asset_optimization.types import DataFrameLike, PlanningHorizon, ScenarioSet


class RuleBasedEffectModel:
    """Estimate intervention effects from simple action-type rules.

    Parameters
    ----------
    effect_rules : dict[str, float], optional
        Mapping of ``action_type`` to a life-restoration fraction in ``[0, 1]``.
    """

    def __init__(self, effect_rules: Mapping[str, float] | None = None) -> None:
        raw_rules = {} if effect_rules is None else dict(effect_rules)
        self.effect_rules = self._validate_rules(raw_rules)

    def fit(
        self,
        interventions: DataFrameLike,
        outcomes: DataFrameLike,
    ) -> "RuleBasedEffectModel":
        """No-op fit for API compatibility."""
        del interventions, outcomes
        return self

    def estimate_effect(
        self,
        candidate_actions: DataFrameLike,
        horizon: PlanningHorizon,
        scenarios: ScenarioSet | None = None,
    ) -> DataFrameLike:
        """Estimate expected risk reduction and benefit for candidates."""
        del horizon, scenarios
        if not isinstance(candidate_actions, pd.DataFrame):
            raise TypeError("candidate_actions must be a pandas DataFrame")
        if "action_type" not in candidate_actions.columns:
            raise ModelError(
                "candidate actions must include 'action_type'",
                details={"missing_columns": ["action_type"]},
            )

        result = candidate_actions.copy(deep=True)
        if result.empty:
            result["expected_risk_reduction"] = pd.Series(dtype=float)
            result["expected_benefit"] = pd.Series(dtype=float)
            return result

        base_risk = self._resolve_base_risk(result)
        restoration = (
            result["action_type"].map(self.effect_rules).fillna(0.0).astype(float)
        )
        expected_risk_reduction = (base_risk * restoration).clip(lower=0.0, upper=1.0)

        if "consequence_cost" in result.columns:
            consequence_cost = pd.to_numeric(
                result["consequence_cost"], errors="coerce"
            ).fillna(0.0)
        else:
            consequence_cost = pd.Series(0.0, index=result.index, dtype=float)

        result["expected_risk_reduction"] = expected_risk_reduction.astype(float)
        result["expected_benefit"] = (
            expected_risk_reduction * consequence_cost
        ).astype(float)
        return result

    def describe(self) -> dict[str, Any]:
        """Return model metadata for planner orchestration."""
        return {
            "model_type": self.__class__.__name__,
            "effect_rules": dict(self.effect_rules),
        }

    @staticmethod
    def _validate_rules(rules: dict[str, float]) -> dict[str, float]:
        validated: dict[str, float] = {}
        for action_type, value in rules.items():
            if not isinstance(action_type, str) or not action_type.strip():
                raise ValueError("effect_rules keys must be non-empty strings")
            numeric_value = float(value)
            if numeric_value < 0.0 or numeric_value > 1.0:
                raise ValueError("effect_rules values must be between 0 and 1")
            validated[action_type] = numeric_value
        return validated

    @staticmethod
    def _resolve_base_risk(candidate_actions: DataFrameLike) -> pd.Series:
        if "failure_prob" in candidate_actions.columns:
            source = candidate_actions["failure_prob"]
        elif "failure_probability" in candidate_actions.columns:
            source = candidate_actions["failure_probability"]
        else:
            source = pd.Series(0.0, index=candidate_actions.index, dtype=float)

        return pd.to_numeric(source, errors="coerce").fillna(0.0).clip(0.0, 1.0)
