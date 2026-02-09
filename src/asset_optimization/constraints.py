"""Constraint DSL for planner optimization requests."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable


@dataclass(frozen=True)
class Constraint:
    """Single optimization constraint.

    Parameters
    ----------
    kind : str
        Constraint identifier.
    params : dict[str, Any], optional
        Constraint-specific parameters.
    """

    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = self.kind.strip().lower()
        if not kind:
            raise ValueError("kind must be a non-empty string")
        if not isinstance(self.params, dict):
            raise TypeError("params must be a dictionary")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "params", self.params.copy())


class ConstraintSet:
    """Fluent accumulator for optimizer constraints."""

    def __init__(self, constraints: Iterable[Constraint] | None = None) -> None:
        constraints_list = list(constraints or [])
        if not all(isinstance(item, Constraint) for item in constraints_list):
            raise TypeError("constraints must contain Constraint instances")
        self._constraints = constraints_list

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        """Return constraints as an immutable tuple."""
        return tuple(self._constraints)

    def add_budget_limit(self, annual_capex: float) -> "ConstraintSet":
        """Add annual capital budget limit."""
        value = self._validate_non_negative(annual_capex, "annual_capex")
        return self._add("budget_limit", annual_capex=value)

    def add_crew_hours_limit(self, crew_hours: float) -> "ConstraintSet":
        """Add crew-hour resource limit."""
        value = self._validate_non_negative(crew_hours, "crew_hours")
        return self._add("crew_hours_limit", crew_hours=value)

    def add_outage_windows(self, windows: Iterable[tuple[str, str]]) -> "ConstraintSet":
        """Add allowed outage windows as ``(start, end)`` tuples."""
        windows_list = list(windows)
        return self._add("outage_windows", windows=windows_list)

    def add_policy_rule(self, rule_name: str, **params: Any) -> "ConstraintSet":
        """Add policy constraint with named rule and optional parameters."""
        normalized_rule = rule_name.strip()
        if not normalized_rule:
            raise ValueError("rule_name must be a non-empty string")
        return self._add("policy_rule", rule_name=normalized_rule, **params)

    def add_minimum_service_level(
        self, minimum_service_level: float
    ) -> "ConstraintSet":
        """Add minimum service level floor (0 to 1)."""
        value = float(minimum_service_level)
        if not math.isfinite(value):
            raise ValueError("minimum_service_level must be finite")
        if not 0.0 <= value <= 1.0:
            raise ValueError("minimum_service_level must be between 0 and 1")
        return self._add("minimum_service_level", minimum_service_level=value)

    def find(self, kind: str) -> tuple[Constraint, ...]:
        """Return all constraints matching ``kind``."""
        normalized_kind = kind.strip().lower()
        return tuple(
            constraint
            for constraint in self._constraints
            if constraint.kind == normalized_kind
        )

    def __len__(self) -> int:
        return len(self._constraints)

    def __iter__(self):
        return iter(self._constraints)

    def _add(self, kind: str, **params: Any) -> "ConstraintSet":
        self._constraints.append(Constraint(kind=kind, params=params))
        return self

    @staticmethod
    def _validate_non_negative(value: float, name: str) -> float:
        try:
            validated = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be numeric") from exc
        if not math.isfinite(validated):
            raise ValueError(f"{name} must be finite")
        if validated < 0:
            raise ValueError(f"{name} must be non-negative")
        return validated
