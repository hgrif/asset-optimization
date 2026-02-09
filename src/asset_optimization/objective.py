"""Objective DSL for planner optimization requests."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass(frozen=True)
class ObjectiveTerm:
    """Single weighted objective term.

    Parameters
    ----------
    kind : str
        Objective term identifier.
    weight : float, default=1.0
        Relative weight of this term in aggregate scoring.
    params : dict[str, Any], optional
        Optional term-specific configuration.
    """

    kind: str
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = self.kind.strip().lower()
        if not kind:
            raise ValueError("kind must be a non-empty string")

        try:
            weight = float(self.weight)
        except (TypeError, ValueError) as exc:
            raise TypeError("weight must be numeric") from exc
        if not math.isfinite(weight):
            raise ValueError("weight must be finite")

        if not isinstance(self.params, dict):
            raise TypeError("params must be a dictionary")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "params", self.params.copy())


@dataclass(frozen=True)
class Objective:
    """Immutable objective made of one or more terms.

    Parameters
    ----------
    terms : tuple[ObjectiveTerm, ...]
        Ordered objective terms.
    """

    terms: tuple[ObjectiveTerm, ...]

    def __post_init__(self) -> None:
        terms = tuple(self.terms)
        if not terms:
            raise ValueError("objective must contain at least one term")
        if not all(isinstance(term, ObjectiveTerm) for term in terms):
            raise TypeError("terms must contain ObjectiveTerm instances")
        object.__setattr__(self, "terms", terms)


class ObjectiveBuilder:
    """Fluent builder for :class:`Objective`."""

    def __init__(self) -> None:
        self._terms: list[ObjectiveTerm] = []

    def add_expected_risk_reduction(
        self, weight: float = 1.0, **params: Any
    ) -> "ObjectiveBuilder":
        """Add expected risk-reduction objective term."""
        return self._add_term("expected_risk_reduction", weight, params)

    def add_total_cost(self, weight: float = 1.0, **params: Any) -> "ObjectiveBuilder":
        """Add total-cost objective term."""
        return self._add_term("total_cost", weight, params)

    def add_resilience_gain(
        self, weight: float = 1.0, **params: Any
    ) -> "ObjectiveBuilder":
        """Add resilience-gain objective term."""
        return self._add_term("resilience_gain", weight, params)

    def add_equity_term(self, weight: float = 1.0, **params: Any) -> "ObjectiveBuilder":
        """Add equity objective term."""
        return self._add_term("equity_term", weight, params)

    def build(self) -> Objective:
        """Create immutable objective from accumulated terms."""
        return Objective(terms=tuple(self._terms))

    def _add_term(
        self, kind: str, weight: float, params: dict[str, Any]
    ) -> "ObjectiveBuilder":
        self._terms.append(ObjectiveTerm(kind=kind, weight=weight, params=params))
        return self
