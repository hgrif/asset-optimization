# Architecture Research: v2 Extended Asset Modeling

**Domain:** Asset Optimization SDK
**Researched:** 2026-02-05
**Confidence:** HIGH

## Current Architecture (Reference)

The v1 SDK has these core components:

```
src/asset_optimization/
├── data/                # Portfolio loading, validation
│   ├── loader.py       # CSV/Excel loading
│   └── schema.py       # Pandera schemas
├── models/             # Deterioration models
│   ├── base.py         # DeteriorationModel interface
│   └── weibull.py      # WeibullModel implementation
├── simulation/         # Multi-timestep simulation
│   └── simulator.py    # Simulator class
├── optimization/       # Intervention optimization
│   ├── base.py         # Optimizer interface
│   └── greedy.py       # GreedyOptimizer implementation
├── results/            # Results and export
│   └── results.py      # SimulationResults class
└── visualization/      # Plotting
    └── plots.py        # Matplotlib visualizations
```

**Key interfaces:**
- `DeteriorationModel.evaluate(portfolio: DataFrame) -> DataFrame` with failure rates
- `Optimizer.optimize(portfolio: DataFrame, options: DataFrame, budget: float) -> DataFrame`
- `Simulator.run(portfolio: DataFrame, model: DeteriorationModel, optimizer: Optimizer, years: int) -> SimulationResults`

## 1. Proportional Hazards Integration

### Integration Points

- **DeteriorationModel interface**: ProportionalHazardsModel implements the same interface
- **Portfolio DataFrame**: Needs additional covariate columns (diameter, soil_type, installation_quality)
- **Simulator**: No changes needed — already accepts any DeteriorationModel

### New Components

```python
# src/asset_optimization/models/proportional_hazards.py
class ProportionalHazardsModel(DeteriorationModel):
    """
    Cox-style proportional hazards model.

    h(t|x) = h₀(t) × exp(β₁×x₁ + β₂×x₂ + ...)

    Where h₀(t) is baseline hazard (Weibull) and x are covariates.
    """
    def __init__(
        self,
        baseline_model: DeteriorationModel,  # e.g., WeibullModel
        covariates: list[str],               # column names
        coefficients: dict[str, float],      # β values per covariate
    ):
        ...

    def evaluate(self, portfolio: DataFrame) -> DataFrame:
        # 1. Get baseline hazard from baseline_model
        # 2. Compute exp(Σ βᵢ × xᵢ) for each asset
        # 3. Multiply baseline × exp term
        ...
```

### Data Flow Changes

```
Before (v1):
portfolio → WeibullModel.evaluate() → failure_rates

After (v2):
portfolio (with covariates) → ProportionalHazardsModel.evaluate()
    → baseline_model.evaluate() → baseline_hazard
    → compute_hazard_multiplier(covariates, coefficients)
    → baseline_hazard × multiplier → failure_rates
```

### Schema Changes

Portfolio DataFrame needs optional covariate columns:
- `diameter` (float): pipe diameter in mm
- `soil_type` (categorical): corrosive, normal, favorable
- `installation_quality` (categorical): poor, average, good

Use Pandera coerce/nullable to make these optional for backward compatibility.

## 2. Multi-Domain Support (Roads)

### Integration Points

- **Data loading**: Domain-agnostic CSV/Excel loading (already exists)
- **Validation**: Domain-specific schemas (new)
- **Deterioration models**: Domain-specific parameters (already pluggable)
- **Interventions**: Domain-specific types (needs extension)

### New Components

```python
# src/asset_optimization/domains/base.py
class DomainConfig:
    """Configuration for an asset domain."""
    name: str
    schema: pa.DataFrameSchema          # Validation schema
    intervention_types: list[str]        # Available interventions
    default_deterioration_params: dict   # Domain-specific defaults

# src/asset_optimization/domains/water_pipes.py
WATER_PIPES = DomainConfig(
    name="water_pipes",
    schema=WaterPipeSchema,
    intervention_types=["do_nothing", "inspect", "repair", "replace"],
    default_deterioration_params={...}
)

# src/asset_optimization/domains/roads.py
ROADS = DomainConfig(
    name="roads",
    schema=RoadSchema,
    intervention_types=["do_nothing", "inspect", "patch", "resurface", "reconstruct"],
    default_deterioration_params={
        "traffic_load_coefficient": ...,
        "climate_coefficient": ...,
    }
)
```

### Data Flow Changes

```
Before (v1):
load_portfolio(path) → validate_water_pipe_schema() → DataFrame

After (v2):
load_portfolio(path, domain="roads")
    → get_domain_config("roads")
    → validate_domain_schema(config.schema)
    → DataFrame
```

### Road-Specific Schema

| Column | Type | Description |
|--------|------|-------------|
| asset_id | str | Unique identifier |
| length_m | float | Road segment length |
| surface_type | category | asphalt, concrete, gravel |
| traffic_load | float | Annual vehicle count or AADT |
| climate_zone | category | freeze_thaw, temperate, hot |
| age_years | int | Years since last resurface |
| condition | float | 0-100 condition index |
| location | str | Geographic identifier |

## 3. Asset Groupings

### Integration Points

- **Portfolio DataFrame**: Add `group_id` column for membership
- **Simulator**: Group-aware failure propagation
- **Optimizer**: Group constraint handling

### New Components

```python
# src/asset_optimization/relationships/groups.py
class AssetGroup:
    """Represents a group of connected/related assets."""
    group_id: str
    members: list[str]  # asset_ids
    relationship_type: Literal["connected", "constraint", "shared_intervention"]

class GroupConfig:
    """Configuration for group behavior."""
    failure_propagation: bool = True      # failures affect neighbors
    propagation_factor: float = 0.3       # how much risk increases
    shared_intervention: bool = False     # repair one benefits others
    intervention_constraint: bool = False # must intervene together

# src/asset_optimization/simulation/group_effects.py
def apply_group_failure_propagation(
    portfolio: DataFrame,
    groups: list[AssetGroup],
    config: GroupConfig,
) -> DataFrame:
    """
    When asset fails, increase failure risk of group members.
    """
    ...

def apply_shared_intervention_effects(
    portfolio: DataFrame,
    interventions: DataFrame,
    groups: list[AssetGroup],
    config: GroupConfig,
) -> DataFrame:
    """
    When asset is repaired, nearby assets get condition improvement.
    """
    ...
```

### Data Flow Changes

```
Before (v1):
simulate_year() → for each asset: evaluate, decide, apply

After (v2):
simulate_year()
    → for each asset: evaluate (with group-adjusted rates)
    → apply_group_constraints()  # can't split groups
    → decide
    → apply_shared_intervention_effects()  # neighbors benefit
    → check_failure_propagation()  # cascades
```

### Schema Changes

Portfolio DataFrame additions:
- `group_id` (str, optional): Which group asset belongs to
- `group_position` (str, optional): Role in group (source, sink, intermediate)

Group definition DataFrame (new input):
| Column | Type | Description |
|--------|------|-------------|
| group_id | str | Unique group identifier |
| asset_id | str | Member asset ID |
| relationship_type | category | connected, constraint, shared |

## 4. Asset Hierarchy

### Integration Points

- **Portfolio DataFrame**: Add `parent_id` column
- **Simulator**: Hierarchy-aware failure effects
- **Optimizer**: Hierarchy constraint handling

### New Components

```python
# src/asset_optimization/relationships/hierarchy.py
class HierarchyConfig:
    """Configuration for hierarchy behavior."""
    dependency_failure: bool = True       # parent fails → children fail
    cost_sharing: bool = False            # parent intervention reduces child cost
    cost_sharing_factor: float = 0.2      # how much cost reduces
    condition_propagation: bool = False   # parent condition affects child rates
    propagation_factor: float = 0.1       # strength of propagation

def build_hierarchy_tree(portfolio: DataFrame) -> dict:
    """
    Build tree from parent_id relationships.
    Returns {asset_id: [child_ids]}
    """
    ...

def apply_hierarchy_failures(
    portfolio: DataFrame,
    failed_assets: list[str],
    config: HierarchyConfig,
) -> list[str]:
    """
    Propagate failures down hierarchy.
    If pump fails, all connected pipes fail.
    """
    ...

def apply_hierarchy_cost_effects(
    portfolio: DataFrame,
    planned_interventions: DataFrame,
    config: HierarchyConfig,
) -> DataFrame:
    """
    If parent is being replaced, children get cost discount.
    """
    ...
```

### Data Flow Changes

```
Before (v1):
calculate_failure_risk(asset) → independent calculation

After (v2):
calculate_failure_risk(asset)
    → base_risk = model.evaluate(asset)
    → parent = get_parent(asset)
    → if parent.condition < threshold:
        → risk *= (1 + config.propagation_factor)
    → return adjusted_risk

apply_interventions()
    → for each intervention:
        → apply to asset
        → if is_parent(asset):
            → apply_cost_sharing_to_children()
```

### Schema Changes

Portfolio DataFrame additions:
- `parent_id` (str, optional): ID of parent asset
- `asset_class` (category): pump, pipe, valve, etc.

## Suggested Build Order

1. **Phase 7: Proportional Hazards** (foundation for multi-property)
   - Implement ProportionalHazardsModel
   - Update schemas for covariates
   - Add tests, notebook examples
   - *Low integration risk — new model, existing interface*

2. **Phase 8: Multi-Domain Foundation** (roads)
   - Implement DomainConfig abstraction
   - Create roads domain with schema
   - Road-specific deterioration parameters
   - *Medium integration risk — refactors loading/validation*

3. **Phase 9: Asset Groupings**
   - Implement AssetGroup and GroupConfig
   - Add group failure propagation
   - Add shared intervention effects
   - Add group constraints to optimizer
   - *High integration risk — changes simulation loop*

4. **Phase 10: Asset Hierarchy**
   - Implement HierarchyConfig
   - Add hierarchy tree building
   - Add dependency failures
   - Add cost sharing
   - Add condition propagation
   - *High integration risk — changes simulation and optimization*

5. **Phase 11: Documentation & Examples**
   - Comprehensive notebooks for all new features
   - API documentation updates
   - Migration guide from v1

## Key Design Decisions

### Why Composition for ProportionalHazards?

ProportionalHazardsModel wraps a baseline model (composition) rather than extending WeibullModel (inheritance):
- Can use any baseline hazard (Weibull, exponential, custom)
- Cleaner separation of concerns
- Easier testing — baseline and hazard multiplier testable independently

### Why DomainConfig Pattern?

A DomainConfig object bundles domain-specific behavior:
- Explicit documentation of domain requirements
- Easy to add new domains without code changes
- Type-safe configuration

### Why Separate Groups from Hierarchy?

Groups and hierarchy are orthogonal concepts:
- Groups = lateral relationships (connected pipes at same level)
- Hierarchy = vertical relationships (pump → pipes)
- An asset can be in a group AND have a parent
- Different failure/cost propagation rules

### Backward Compatibility

All v2 features use optional columns and parameters:
- Existing v1 portfolios work unchanged
- ProportionalHazards needs covariates but WeibullModel still works
- Groups/hierarchy disabled when columns missing

---
*Architecture research for: v2 Extended Asset Modeling*
*Researched: 2026-02-05*
