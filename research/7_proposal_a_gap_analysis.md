# Proposal A Gap Analysis vs Current Implementation

Date: 2026-02-08
Spec: `/Users/henkgriffioen/code/asset-optimization/research/7_proposal_a_spec.md`

## Executive Summary
Current code implements a **single-domain, portfolio-first simulation + greedy budget optimizer** stack. Proposal A specifies a **service-oriented planning platform** with explicit boundaries (repository, risk, effect, network simulation, optimizer, backtesting, registry), uncertainty-aware outputs, and multi-domain orchestration.

Most Proposal A components are currently missing or only partially represented. The largest deltas are:
1. No `Planner` orchestration layer.
2. No `AssetRepository`, `InterventionEffectModel`, `Backtester`, or `ModelRegistry` APIs.
3. No objective/constraint DSL (`ObjectiveBuilder`, `ConstraintSet`).
4. No domain/use-case catalog (`DomainProfile`, `UseCaseTemplate`, `DomainCatalog`).
5. Output/data contracts do not match Proposal A (`PlanResult`, scenario-weighted risk distributions, action schemas).

## What Exists Today (Baseline)
- Deterioration model abstractions exist (`DeteriorationModel`, `WeibullModel`, `ProportionalHazardsModel`):
  - `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/models/base.py`
  - `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/models/weibull.py`
  - `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/models/proportional_hazards.py`
- Simulation engine exists (`Simulator`, `SimulationConfig`, `SimulationResult`):
  - `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/simulation/simulator.py`
- Budget optimizer exists, but only greedy/MILP placeholder (`Optimizer`, `OptimizationResult`):
  - `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/optimization/optimizer.py`
- Domain-specific helpers exist for pipes/roads, but not Proposal A catalog/presets:
  - `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/domains/`
- Validation and custom exceptions exist in a narrow form:
  - `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/schema.py`
  - `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/exceptions.py`

## Gap Matrix (Spec -> Current -> Needed)

### 1) System Architecture and Orchestration
- Spec requires: `Planner` coordinating repository + risk + effect + simulator + optimizer + registry.
- Current: No planner orchestration object.
- Needed change:
  - Add `planning/planner.py` with `validate_inputs()`, `fit()`, `propose_actions()`, `optimize_plan()`.
  - Planner should be the entrypoint; current `Optimizer` and `Simulator` become pluggable components.

### 2) Core Contracts / Dataclasses
- Spec requires: `PlanningHorizon`, `PlanResult`, `ScenarioSet`, `ValidationReport`.
- Current: `SimulationConfig`/`SimulationResult` and `OptimizationResult`, but not matching Proposal A contract.
- Needed change:
  - Add `contracts.py` (or `planning/contracts.py`) with frozen dataclasses for the spec contract.
  - Keep `OptimizationResult` internal or migrate it to `PlanResult` shape.

### 3) Service Protocols
- Spec requires `AssetRepository`, `RiskModel`, `InterventionEffectModel`, `NetworkSimulator`, `PlanOptimizer`, `Backtester` protocols.
- Current:
  - Partial analogs: deterioration model, simulator, optimizer.
  - Missing repository/effect/backtester protocols entirely.
- Needed change:
  - Add protocol definitions in `services/interfaces.py`.
  - Add adapters:
    - Wrap current deterioration models into `RiskModel` (`fit`, `predict_distribution`, `describe`).
    - Wrap current optimizer into `PlanOptimizer.solve(...)`.
    - Wrap current simulator into `NetworkSimulator.simulate(...)` (or separate hazard simulator from network simulator).

### 4) Objective / Constraint DSL
- Spec requires: `ObjectiveBuilder`, `ConstraintSet` with composable terms/rules.
- Current: Hardcoded greedy ranking and budget cutoff.
- Needed change:
  - Add `planning/objective.py` and `planning/constraints.py`.
  - Move budget/resource/policy checks from optimizer internals into `ConstraintSet` evaluation.

### 5) Data Contracts and Validation Scope
- Spec requires required-column contracts for assets/events/actions/risk output/plan output; scenario probability validation.
- Current: single `portfolio_schema` focused on portfolio table.
- Needed change:
  - Create table-specific schemas (assets/events/interventions/outcomes/covariates/topology/scenarios/candidate_actions).
  - Implement planner-level `validate_inputs()` returning `ValidationReport`.
  - Add rules for unknown asset references, negative costs, invalid schedules, missing scenario probabilities.

### 6) Output Contracts
- Spec requires `PlanResult` with:
  - `selected_actions`
  - `objective_breakdown`
  - `constraint_shadow_prices`
  - `metadata`
- Current: `OptimizationResult` with selections + budget summary only.
- Needed change:
  - Redesign optimizer output to populate full `PlanResult` contract.
  - Include explainability fields in `selected_actions` (`selection_reason`, etc.).

### 7) Plugin Registry
- Spec requires plugin dictionaries and swappable implementations for risk/effect/sim/optimizer.
- Current: no plugin registry abstraction.
- Needed change:
  - Add `registry/plugins.py` with registries and registration API.
  - Seed with existing classes (`WeibullModel`, `ProportionalHazardsModel`, current `Optimizer` adapter, etc.).

### 8) Multi-Domain Support
- Spec requires `DomainProfile`, `UseCaseTemplate`, `DomainCatalog`, planner factory, isolated/shared-budget modes.
- Current: only `PipeDomain` and `RoadDomain` helpers.
- Needed change:
  - Add `domains/catalog.py` and `domains/profiles.py`.
  - Implement `build_planner_for(domain, repository)` and `MultiDomainPlanner`.

### 9) Governance and Model Registry
- Spec requires model/version metadata logging on `fit()` and `optimize_plan()` and registry interface.
- Current: no model registry; no fit/optimize audit logs.
- Needed change:
  - Add `governance/model_registry.py` protocol + basic implementation (e.g., in-memory or sqlite-backed).
  - Emit structured metadata from planner operations.

### 10) Exception Model Alignment
- Spec requires mapping schema -> `ValidationError`, plugin incompatibility -> `ModelError`, infeasible optimization -> `OptimizationError`.
- Current: has `ValidationError` and `OptimizationError`; no `ModelError`.
- Needed change:
  - Add `ModelError` in `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/exceptions.py`.
  - Normalize conversion points where adapter outputs violate expected contracts.

### 11) Backtesting
- Spec requires rolling-origin walk-forward backtesting.
- Current: no backtesting module.
- Needed change:
  - Add `backtesting/backtester.py` with `walk_forward(history, planner, window)` and out-of-sample metrics.

## Recommended Implementation Order (How)

### Phase 1: Contract-first MVP
1. Add Proposal A contracts/protocols (`PlanningHorizon`, `PlanResult`, `ScenarioSet`, `ValidationReport`, service protocols).
2. Implement `Planner` skeleton with dependency injection and `validate_inputs()`.
3. Implement adapters around current `Weibull/PH`, `Simulator`, and `Optimizer`.
4. Implement minimal `ObjectiveBuilder` + `ConstraintSet` with budget-only logic.
5. Add tests for planner orchestration and contract conformance.

### Phase 2: Scenarios + Risk Measure
1. Extend risk outputs to scenario-step distributions (`failure_prob`, `loss_mean`, optional tails).
2. Add scenario probability validation and weighted expectation support.
3. Implement `risk_measure` plumbing (`expected_value` first, then `cvar`).

### Phase 3: Domain Catalog + Multi-Domain
1. Add `DomainProfile`, `UseCaseTemplate`, `DomainCatalog`, `build_planner_for(...)`.
2. Implement `MultiDomainPlanner.optimize_isolated` and shared budget mode.
3. Add domain presets and template validation tests.

### Phase 4: Governance + Backtesting
1. Add `ModelRegistry` + metadata logging for `fit/optimize` calls.
2. Add rolling walk-forward `Backtester` and evaluation outputs.
3. Add end-to-end tests for governance artifacts and backtest reproducibility.

## Suggested File Additions
- `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/planning/contracts.py`
- `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/planning/planner.py`
- `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/planning/objective.py`
- `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/planning/constraints.py`
- `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/services/interfaces.py`
- `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/registry/plugins.py`
- `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/governance/model_registry.py`
- `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/backtesting/backtester.py`
- `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/domains/catalog.py`
- `/Users/henkgriffioen/code/asset-optimization/src/asset_optimization/domains/profiles.py`

## Test Changes Required
- Add contract tests for all service protocols.
- Add planner integration tests:
  - `fit -> optimize_plan` flow with scenario set.
  - Validation failure matrix for missing columns/invalid references.
- Add optimizer output-schema tests for Proposal A `PlanResult` columns.
- Add multi-domain tests (isolated/shared budget).
- Add backtesting walk-forward tests and registry logging assertions.

## Breaking Changes to Accept
Given repo rules explicitly allow API breakage (pre-release SDK), it is cleaner to:
- Replace the current top-level workflow (`Simulator` + `Optimizer`) with a planner-led API.
- Keep legacy classes as internal adapters or remove them after migration.
