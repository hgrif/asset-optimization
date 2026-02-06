"""Asset portfolio optimization for infrastructure management."""

__version__ = "0.1.0"

from asset_optimization.exceptions import (
    ValidationError,
    MissingFieldError,
    DataQualityError,
    OptimizationError,
)
from asset_optimization.models import DeteriorationModel, WeibullModel
from asset_optimization.quality import QualityMetrics
from asset_optimization.simulation import (
    Simulator,
    SimulationConfig,
    SimulationResult,
    InterventionType,
    DO_NOTHING,
    INSPECT,
    REPAIR,
    REPLACE,
)
from asset_optimization.domains import Domain, PipeDomain
from asset_optimization.optimization import Optimizer, OptimizationResult
from asset_optimization.exports import (
    export_schedule_minimal,
    export_schedule_detailed,
    export_cost_projections,
)
from asset_optimization.scenarios import (
    compare_scenarios,
    create_do_nothing_baseline,
    compare,
)
from asset_optimization.visualization import (
    set_sdk_theme,
    plot_cost_over_time,
    plot_failures_by_year,
    plot_risk_distribution,
    plot_scenario_comparison,
    plot_asset_action_heatmap,
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
    "Domain",
    "PipeDomain",
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
    "plot_asset_action_heatmap",
]
