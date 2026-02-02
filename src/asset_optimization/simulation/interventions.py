"""Intervention type definitions for asset optimization simulation.

This module provides the InterventionType dataclass and predefined intervention
constants for use in simulation scenarios.
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class InterventionType:
    """Configuration for an intervention type.

    Attributes
    ----------
    name : str
        Intervention name (e.g., 'Replace', 'Repair')
    cost : float
        Direct cost to perform intervention
    age_effect : Callable[[float], float]
        Function: old_age -> new_age
        Examples:
        - Replace: lambda age: 0.0
        - Repair: lambda age: max(0.0, age - 5.0)
        - DoNothing: lambda age: age
    consequence_cost : float
        Additional cost for failures (service disruption)
    upgrade_type : str, optional
        New asset type after intervention (for Replace with upgrade)

    Examples
    --------
    >>> # Create a custom heavy repair intervention
    >>> heavy_repair = InterventionType(
    ...     name='HeavyRepair',
    ...     cost=15000.0,
    ...     age_effect=lambda age: max(0.0, age - 10.0)
    ... )
    >>> heavy_repair.apply_age_effect(25.0)
    15.0
    """

    name: str
    cost: float
    age_effect: Callable[[float], float]
    consequence_cost: float = 0.0
    upgrade_type: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate intervention type attributes."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("name must be a non-empty string")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")
        if self.consequence_cost < 0:
            raise ValueError("consequence_cost must be non-negative")

    def apply_age_effect(self, age: float) -> float:
        """Apply age effect and return new age.

        Parameters
        ----------
        age : float
            Current age of the asset

        Returns
        -------
        float
            New age after applying the intervention effect
        """
        return self.age_effect(age)

    def total_cost(self) -> float:
        """Total cost including consequence cost.

        Returns
        -------
        float
            Sum of direct cost and consequence cost
        """
        return self.cost + self.consequence_cost


# Predefined intervention constants
# These are defaults; users create custom InterventionType instances for their specific costs

DO_NOTHING: InterventionType = InterventionType(
    name="DoNothing",
    cost=0.0,
    age_effect=lambda age: age,
)

INSPECT: InterventionType = InterventionType(
    name="Inspect",
    cost=500.0,
    age_effect=lambda age: age,  # v1: no follow-up logic, age unchanged
)

REPAIR: InterventionType = InterventionType(
    name="Repair",
    cost=5000.0,
    age_effect=lambda age: max(0.0, age - 5.0),  # Reduce age by 5, clamped to 0
)

REPLACE: InterventionType = InterventionType(
    name="Replace",
    cost=50000.0,
    age_effect=lambda age: 0.0,  # Reset age to 0
)
