"""Tests for objective DSL contracts."""

from dataclasses import FrozenInstanceError

import pytest

from asset_optimization.objective import ObjectiveBuilder, ObjectiveTerm


def test_objective_builder_creates_expected_terms() -> None:
    """Builder creates terms in insertion order with provided params."""
    objective = (
        ObjectiveBuilder()
        .add_expected_risk_reduction(weight=2.0, horizon_years=10)
        .add_total_cost(weight=0.5)
        .add_resilience_gain(region="north")
        .add_equity_term(group_column="district")
        .build()
    )

    assert [term.kind for term in objective.terms] == [
        "expected_risk_reduction",
        "total_cost",
        "resilience_gain",
        "equity_term",
    ]
    assert objective.terms[0].weight == 2.0
    assert objective.terms[0].params["horizon_years"] == 10
    assert objective.terms[1].weight == 0.5
    assert objective.terms[3].params["group_column"] == "district"


def test_objective_builder_rejects_empty_objective() -> None:
    """Objective requires at least one term."""
    with pytest.raises(ValueError, match="at least one term"):
        ObjectiveBuilder().build()


def test_objective_term_normalizes_and_copies_params() -> None:
    """ObjectiveTerm normalizes kind and stores defensive params copy."""
    source = {"metric": "capex"}
    term = ObjectiveTerm(kind=" Total_Cost ", weight=1, params=source)

    source["metric"] = "opex"

    assert term.kind == "total_cost"
    assert term.weight == 1.0
    assert term.params == {"metric": "capex"}


def test_objective_term_is_frozen() -> None:
    """ObjectiveTerm dataclass is immutable."""
    term = ObjectiveTerm(kind="total_cost")
    with pytest.raises(FrozenInstanceError):
        term.weight = 5.0
