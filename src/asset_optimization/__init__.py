"""Asset portfolio optimization for infrastructure management."""

__version__ = "0.1.0"

from asset_optimization import registry
from asset_optimization.constraints import Constraint, ConstraintSet
from asset_optimization.effects import RuleBasedEffectModel
from asset_optimization.exceptions import (
    DataQualityError,
    MissingFieldError,
    ModelError,
    OptimizationError,
    ValidationError,
)
from asset_optimization.exports import (
    export_cost_projections,
    export_schedule_detailed,
    export_schedule_minimal,
)
from asset_optimization.models import (
    DeteriorationModel,
    ProportionalHazardsModel,
    WeibullModel,
)
from asset_optimization.objective import Objective, ObjectiveBuilder, ObjectiveTerm
from asset_optimization.optimization import Optimizer
from asset_optimization.planner import Planner
from asset_optimization.protocols import (
    AssetRepository,
    InterventionEffectModel,
    NetworkSimulator,
    PlanOptimizer,
    RiskModel,
)
from asset_optimization.quality import QualityMetrics
from asset_optimization.registry import clear, get, list_registered, register
from asset_optimization.repositories import DataFrameRepository
from asset_optimization.scenarios import (
    compare,
    create_do_nothing_baseline,
    compare_scenarios,
)
from asset_optimization.simulation import (
    DO_NOTHING,
    INSPECT,
    REPAIR,
    REPLACE,
    InterventionType,
    SimulationConfig,
    SimulationResult,
    Simulator,
)
from asset_optimization.domains import Domain, PipeDomain, RoadDomain
from asset_optimization.types import (
    DataFrameLike,
    PlanResult,
    PlanningHorizon,
    ScenarioSet,
    ValidationReport,
)
from asset_optimization.visualization import (
    plot_asset_action_heatmap,
    plot_cost_over_time,
    plot_failures_by_year,
    plot_risk_distribution,
    plot_scenario_comparison,
    set_sdk_theme,
)

__all__ = [
    "__version__",
    "registry",
    "Constraint",
    "ConstraintSet",
    "RuleBasedEffectModel",
    "DataQualityError",
    "MissingFieldError",
    "ModelError",
    "OptimizationError",
    "ValidationError",
    "export_cost_projections",
    "export_schedule_detailed",
    "export_schedule_minimal",
    "DeteriorationModel",
    "ProportionalHazardsModel",
    "WeibullModel",
    "Objective",
    "ObjectiveBuilder",
    "ObjectiveTerm",
    "Optimizer",
    "Planner",
    "AssetRepository",
    "InterventionEffectModel",
    "NetworkSimulator",
    "PlanOptimizer",
    "RiskModel",
    "QualityMetrics",
    "register",
    "get",
    "list_registered",
    "clear",
    "DataFrameRepository",
    "compare",
    "create_do_nothing_baseline",
    "compare_scenarios",
    "Domain",
    "PipeDomain",
    "RoadDomain",
    "DataFrameLike",
    "PlanResult",
    "PlanningHorizon",
    "ScenarioSet",
    "ValidationReport",
    "InterventionType",
    "SimulationConfig",
    "SimulationResult",
    "Simulator",
    "DO_NOTHING",
    "INSPECT",
    "REPAIR",
    "REPLACE",
    "plot_asset_action_heatmap",
    "plot_cost_over_time",
    "plot_failures_by_year",
    "plot_risk_distribution",
    "plot_scenario_comparison",
    "set_sdk_theme",
]
