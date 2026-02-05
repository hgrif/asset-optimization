"""Asset portfolio optimization for infrastructure management."""

__version__ = "0.1.0"

from .exceptions import (
    ValidationError,
    MissingFieldError,
    DataQualityError,
    OptimizationError,
)
from .models import DeteriorationModel, WeibullModel
from .quality import QualityMetrics
from .simulation import (
    Simulator,
    SimulationConfig,
    SimulationResult,
    InterventionType,
    DO_NOTHING,
    INSPECT,
    REPAIR,
    REPLACE,
)
from .optimization import Optimizer, OptimizationResult
from .exports import (
    export_schedule_minimal,
    export_schedule_detailed,
    export_cost_projections,
)
from .scenarios import compare_scenarios, create_do_nothing_baseline, compare
from .visualization import (
    set_sdk_theme,
    plot_cost_over_time,
    plot_failures_by_year,
    plot_risk_distribution,
    plot_scenario_comparison,
)

__all__ = [
    "__version__",
    "ValidationError",
    "MissingFieldError",
    "DataQualityError",
    "OptimizationError",
    "DeteriorationModel",
    "WeibullModel",
    "QualityMetrics",
    "Simulator",
    "SimulationConfig",
    "SimulationResult",
    "InterventionType",
    "DO_NOTHING",
    "INSPECT",
    "REPAIR",
    "REPLACE",
    "Optimizer",
    "OptimizationResult",
    "export_schedule_minimal",
    "export_schedule_detailed",
    "export_cost_projections",
    "compare_scenarios",
    "create_do_nothing_baseline",
    "compare",
    "set_sdk_theme",
    "plot_cost_over_time",
    "plot_failures_by_year",
    "plot_risk_distribution",
    "plot_scenario_comparison",
]
