# Proposal A: Core API Skeleton Implementation Plan

## Context

The research in `research/7_proposal_a_spec.md` defines a domain-oriented service API for asset optimization with explicit service boundaries (repository, risk, effect, simulation, optimization) and uncertainty-aware outputs. The gap analysis (`research/7_proposal_a_gap_analysis.md`) identifies 11 gaps between the current codebase and the spec.

This plan implements the **core API skeleton**: Planner orchestrator, service protocols, core data contracts, ObjectiveBuilder/ConstraintSet DSL, and in-memory plugin registry. Backtesting, governance, and multi-domain are deferred.

**Strategy**: Add protocol-conforming methods to existing classes (additive refactoring). Existing `run()`/`fit()` methods stay intact so current tests and notebooks continue working. New `solve()`/`simulate()`/`predict_distribution()` methods satisfy the Proposal A protocols.

## New Files

| File | Purpose |
|------|---------|
| `src/asset_optimization/types.py` | `DataFrameLike`, `PlanningHorizon`, `PlanResult`, `ScenarioSet`, `ValidationReport` |
| `src/asset_optimization/protocols.py` | `AssetRepository`, `RiskModel`, `InterventionEffectModel`, `NetworkSimulator`, `PlanOptimizer` protocols |
| `src/asset_optimization/registry.py` | In-memory plugin registry with `register()`/`get()`/`list_registered()` |
| `src/asset_optimization/objective.py` | `ObjectiveBuilder`, `ObjectiveTerm`, `Objective` |
| `src/asset_optimization/constraints.py` | `ConstraintSet`, `Constraint` |
| `src/asset_optimization/planner.py` | `Planner` orchestrator |
| `src/asset_optimization/repositories/__init__.py` | Subpackage exports |
| `src/asset_optimization/repositories/dataframe.py` | `DataFrameRepository` (in-memory `AssetRepository` impl) |
| `src/asset_optimization/effects/__init__.py` | Subpackage exports |
| `src/asset_optimization/effects/rule_based.py` | `RuleBasedEffectModel` (rule-based `InterventionEffectModel` impl) |
| `tests/test_types.py` | Data contract tests |
| `tests/test_protocols.py` | Protocol conformance tests |
| `tests/test_registry.py` | Plugin registry tests |
| `tests/test_objective.py` | ObjectiveBuilder tests |
| `tests/test_constraints.py` | ConstraintSet tests |
| `tests/test_planner.py` | Planner lifecycle + integration tests |
| `tests/test_repositories.py` | DataFrameRepository tests |
| `tests/test_effects.py` | RuleBasedEffectModel tests |

## Modified Files

| File | Changes |
|------|---------|
| `src/asset_optimization/exceptions.py` | Add `ModelError` class |
| `src/asset_optimization/models/base.py` | Add `fit()`, `predict_distribution()`, `describe()` concrete methods to `DeteriorationModel` |
| `src/asset_optimization/models/weibull.py` | Override `describe()` to include Weibull params |
| `src/asset_optimization/models/proportional_hazards.py` | Override `describe()` to include covariates/coefficients |
| `src/asset_optimization/simulation/simulator.py` | Add `simulate()` method for `NetworkSimulator` protocol |
| `src/asset_optimization/optimization/optimizer.py` | Add `solve()` method for `PlanOptimizer` protocol |
| `src/asset_optimization/__init__.py` | Add new exports |
| Existing test files | Add protocol conformance assertions (additive only) |

## Implementation Steps

### Step 1: Foundation types and exceptions (no dependencies) ✅ Completed 2026-02-09

**`types.py`** — Frozen dataclasses:
- `PlanningHorizon(start_date, end_date, step)` with validation (step in monthly/quarterly/yearly, end > start)
- `PlanResult(selected_actions, objective_breakdown, constraint_shadow_prices, metadata)`
- `ScenarioSet(scenarios)` — DataFrame with scenario_id, variable, timestamp, value, probability
- `ValidationReport(passed, checks, warnings)`
- `DataFrameLike = pd.DataFrame` type alias

**`exceptions.py`** — Add `ModelError(AssetOptimizationError)` with message + details dict.

### Step 2: Service protocols (depends on Step 1) ✅ Completed 2026-02-09

**`protocols.py`** — Five `@runtime_checkable` Protocol classes matching the spec signatures. Make `ScenarioSet` optional (`| None = None`) in all signatures since the skeleton doesn't implement scenario logic yet.

### Step 3: DSL classes (no dependencies) ✅ Completed 2026-02-09

**`objective.py`**:
- `ObjectiveTerm(frozen=True)`: kind, weight, params
- `Objective(frozen=True)`: tuple of terms
- `ObjectiveBuilder`: fluent builder with `add_expected_risk_reduction()`, `add_total_cost()`, `add_resilience_gain()`, `add_equity_term()`, `build()`

**`constraints.py`**:
- `Constraint(frozen=True)`: kind, params
- `ConstraintSet`: fluent accumulator with `add_budget_limit()`, `add_crew_hours_limit()`, `add_outage_windows()`, `add_policy_rule()`, `add_minimum_service_level()`

### Step 4: Refactor existing classes (depends on Steps 1-2) ✅ Completed 2026-02-09

**`models/base.py`** — Add three concrete methods to `DeteriorationModel`:
- `fit(assets, events, covariates=None)` — no-op default (params set at init), returns self
- `predict_distribution(assets, horizon, scenarios=None)` — default impl that calls existing `transform()` to get `failure_probability`, then reshapes to Proposal A schema (asset_id, scenario_id, horizon_step, failure_prob, loss_mean)
- `describe()` — returns `{"model_type": class_name}`

**`models/weibull.py`** — Override `describe()` to include `params`, `type_column`, `age_column`.

**`models/proportional_hazards.py`** — Override `describe()` to include baseline description, covariates, coefficients.

**`simulation/simulator.py`** — Add `simulate(topology, failures, actions, scenarios=None)` method. Basic implementation: returns actions DataFrame with a `consequence_cost` column added (pass-through for non-network domains).

**`optimization/optimizer.py`** — Add `solve(objective, constraints, candidates, risk_measure="expected_value")` method:
- Extract `annual_capex` from `ConstraintSet` budget_limit constraint
- Rank candidates by `expected_benefit / direct_cost` ratio (greedy)
- Select within budget
- Return `PlanResult` with selected_actions (Proposal A schema), objective_breakdown, constraint_shadow_prices, metadata

### Step 5: New implementations (depends on Steps 1-2) ✅ Completed 2026-02-09

**`repositories/dataframe.py`** — `DataFrameRepository`: constructor accepts DataFrames for assets/events/interventions/outcomes/covariates/topology (all optional except assets). Each `load_*()` returns a copy.

**`effects/rule_based.py`** — `RuleBasedEffectModel`:
- Constructor accepts `effect_rules: dict[str, float]` mapping action_type to life-restoration fraction
- `fit()` is a no-op (returns self)
- `estimate_effect(candidates, horizon, scenarios=None)` adds `expected_risk_reduction` and `expected_benefit` columns
- `describe()` returns rules dict

### Step 6: Planner orchestrator (depends on Steps 1-5) ✅ Completed 2026-02-09

**`planner.py`** — `Planner` class:
- Constructor takes repository, risk_model, effect_model, simulator, optimizer, registry(optional)
- `validate_inputs()`: loads assets, checks required columns (asset_id, asset_type, install_date), checks duplicates, returns `ValidationReport`
- `fit()`: loads all data from repository, calls `risk_model.fit()` and `effect_model.fit()`, wraps failures in `ModelError`
- `propose_actions(horizon, scenarios=None)`: calls `predict_distribution()`, builds candidate actions, calls `estimate_effect()`, optionally runs simulator
- `optimize_plan(horizon, scenarios, objective, constraints, risk_measure)`: calls `propose_actions()` then `optimizer.solve()`

### Step 7: Plugin registry (depends on Steps 4-5) ✅ Completed 2026-02-09

**`registry.py`**:
- Module-level dicts: `RISK_MODELS`, `EFFECT_MODELS`, `SIMULATORS`, `OPTIMIZERS`
- `register(category, key, cls)`, `get(category, key)`, `list_registered(category)`, `clear(category=None)`
- Auto-register builtins at import: weibull, proportional_hazards, rule_based, basic simulator, greedy optimizer

### Step 8: Public API and tests (depends on all above) ✅ Completed 2026-02-09

**`__init__.py`** — Add exports for all new public names.

**Tests** — Write test files listed above. Add protocol conformance checks to existing test files (e.g., `assert isinstance(WeibullModel(...), RiskModel)` in `test_deterioration.py`).

## Verification

1. `make test` — all existing tests pass unchanged + new test files pass
2. `make lint` — new code passes ruff
3. `make docs` — notebooks still execute (they use existing interfaces which are preserved)
4. Manual check: end-to-end Planner lifecycle works:
   ```python
   from asset_optimization import *
   repo = DataFrameRepository(assets=sample_df)
   planner = Planner(repo, WeibullModel({...}), RuleBasedEffectModel(), Simulator(...), Optimizer())
   report = planner.validate_inputs()
   planner.fit()
   result = planner.optimize_plan(
       horizon=PlanningHorizon("2027-01-01", "2027-12-31", "yearly"),
       objective=ObjectiveBuilder().add_expected_risk_reduction().build(),
       constraints=ConstraintSet().add_budget_limit(100_000),
   )
   ```
