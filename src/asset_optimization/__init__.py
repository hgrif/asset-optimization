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
from asset_optimization.models import (
    DeteriorationModel,
    GroupPropagationRiskModel,
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
from asset_optimization.registry import clear, get, list_registered, register
from asset_optimization.repositories import DataFrameRepository
from asset_optimization.simulators import BasicNetworkSimulator
from asset_optimization.types import (
    DataFrameLike,
    PlanResult,
    PlanningHorizon,
    ScenarioSet,
    ValidationReport,
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
    "DeteriorationModel",
    "GroupPropagationRiskModel",
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
    "register",
    "get",
    "list_registered",
    "clear",
    "DataFrameRepository",
    "DataFrameLike",
    "PlanResult",
    "PlanningHorizon",
    "ScenarioSet",
    "ValidationReport",
    "BasicNetworkSimulator",
]
