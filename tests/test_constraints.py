"""Tests for constraint DSL contracts."""

from dataclasses import FrozenInstanceError

import pytest

from asset_optimization.constraints import Constraint, ConstraintSet


def test_constraint_set_fluent_builder_adds_expected_constraints() -> None:
    """ConstraintSet accumulates constraints in insertion order."""
    constraints = (
        ConstraintSet()
        .add_budget_limit(100_000)
        .add_crew_hours_limit(40)
        .add_outage_windows([("2026-01-01", "2026-01-03")])
        .add_policy_rule("historic_district", applies_to="district_7")
        .add_minimum_service_level(0.95)
    )

    assert len(constraints) == 5
    assert constraints.constraints[0].kind == "budget_limit"
    assert constraints.constraints[0].params["annual_capex"] == 100000.0
    assert constraints.constraints[1].kind == "crew_hours_limit"
    assert constraints.constraints[2].params["windows"] == [
        ("2026-01-01", "2026-01-03")
    ]
    assert constraints.constraints[3].params["rule_name"] == "historic_district"
    assert constraints.constraints[4].params["minimum_service_level"] == 0.95


def test_constraint_set_find_returns_kind_matches() -> None:
    """find() returns all constraints of a given kind."""
    constraints = (
        ConstraintSet()
        .add_budget_limit(10_000)
        .add_policy_rule("policy_a")
        .add_policy_rule("policy_b")
    )

    matched = constraints.find("policy_rule")
    assert len(matched) == 2
    assert [item.params["rule_name"] for item in matched] == ["policy_a", "policy_b"]


def test_constraint_set_validates_inputs() -> None:
    """Builder methods enforce numeric and range validations."""
    with pytest.raises(ValueError, match="non-negative"):
        ConstraintSet().add_budget_limit(-1)

    with pytest.raises(ValueError, match="between 0 and 1"):
        ConstraintSet().add_minimum_service_level(1.5)

    with pytest.raises(ValueError, match="non-empty string"):
        ConstraintSet().add_policy_rule("   ")


def test_constraint_is_frozen() -> None:
    """Constraint dataclass is immutable."""
    constraint = Constraint(kind="budget_limit", params={"annual_capex": 5})
    with pytest.raises(FrozenInstanceError):
        constraint.params = {}
