"""Simulation configuration dataclass."""

from dataclasses import dataclass
from typing import Optional


# Valid failure response options
FAILURE_RESPONSES = ('replace', 'repair', 'record_only')


@dataclass(frozen=True)
class SimulationConfig:
    """Immutable simulation configuration.

    Parameters
    ----------
    n_years : int
        Number of years to simulate (e.g., 10, 20, 30)
    start_year : int
        Calendar year to start simulation (default: 2026)
    random_seed : int, optional
        Seed for reproducible results (None = non-deterministic)
    track_asset_history : bool
        Whether to save full asset-level traces (memory-intensive)
    failure_response : str
        How to handle failures: 'replace', 'repair', 'record_only'

    Examples
    --------
    >>> config = SimulationConfig(n_years=10, random_seed=42)
    >>> config.n_years
    10

    >>> config = SimulationConfig(n_years=20, failure_response='repair')
    >>> config.failure_response
    'repair'
    """

    n_years: int
    start_year: int = 2026
    random_seed: Optional[int] = None
    track_asset_history: bool = False
    failure_response: str = 'replace'

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_years <= 0:
            raise ValueError(
                f"n_years must be positive, got {self.n_years}"
            )

        if self.failure_response not in FAILURE_RESPONSES:
            raise ValueError(
                f"failure_response must be one of {FAILURE_RESPONSES}, "
                f"got '{self.failure_response}'"
            )
