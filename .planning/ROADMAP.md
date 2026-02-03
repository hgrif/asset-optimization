# Roadmap: Asset Optimization

## Overview

Build a Python SDK for simulating and optimizing infrastructure asset portfolios. Start with data loading and Weibull deterioration models, implement multi-timestep simulation with intervention logic, add optimization to select interventions under budget constraints, and finish with outputs, visualizations, and documentation. Focused on water pipe networks for v1.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation** - Project structure, data loading, validation
- [x] **Phase 2: Deterioration Models** - Weibull failure rates and pluggable interface
- [x] **Phase 3: Simulation Core** - Multi-timestep simulation with interventions
- [ ] **Phase 4: Optimization** - Constraint-based intervention selection
- [ ] **Phase 5: Results & Polish** - Outputs, visualization, documentation

## Phase Details

### Phase 1: Foundation
**Goal**: Users can load and validate asset portfolio data through a well-structured Python package

**Depends on**: Nothing (first phase)

**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DEVX-01

**Success Criteria** (what must be TRUE):
  1. User can install package via `pip install asset-optimization`
  2. User can load asset portfolio from CSV file with 1000+ pipes and see validation report
  3. User can load asset portfolio from Excel file with required fields validated
  4. System reports data quality metrics (completeness percentages, missing value counts)
  5. User can query and filter assets by age, type, condition, location

**Plans**: 3 plans

Plans:
- [x] 01-01-PLAN.md — Package setup, exceptions, and Pandera schema
- [x] 01-02-PLAN.md — Portfolio class with loading and quality metrics
- [x] 01-03-PLAN.md — Test suite for all functionality

### Phase 2: Deterioration Models
**Goal**: System calculates failure rates using Weibull deterioration model with pluggable architecture

**Depends on**: Phase 1

**Requirements**: DTRN-01, DTRN-02, DTRN-03, DTRN-04

**Success Criteria** (what must be TRUE):
  1. System calculates failure rates for 1000+ pipes using Weibull 2-parameter model efficiently (<1 second)
  2. User can configure different Weibull shape and scale parameters per asset type (e.g., cast iron vs PVC)
  3. System evaluates failure probabilities across entire portfolio in vectorized operations
  4. User can swap Weibull model for custom deterioration model via pluggable interface

**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md — Abstract base class and scipy dependency
- [x] 02-02-PLAN.md — Weibull deterioration model implementation
- [x] 02-03-PLAN.md — Test suite for deterioration models

### Phase 3: Simulation Core
**Goal**: Users can run multi-timestep simulations with intervention effects on asset states

**Depends on**: Phase 2

**Requirements**: SIMU-01, SIMU-02, SIMU-03, SIMU-04, SIMU-05, INTV-01, INTV-02, INTV-03, INTV-04

**Success Criteria** (what must be TRUE):
  1. User can run 10-year simulation with configurable timesteps
  2. System updates asset ages and conditions after each timestep
  3. System applies intervention effects (age reset on Replace, condition improvement on Repair)
  4. User can configure costs and state effects for 4 intervention types (DoNothing, Inspect, Repair, Replace)
  5. Simulation produces deterministic results with random seed control
  6. System tracks cumulative cost and failure counts across timesteps

**Plans**: 4 plans

Plans:
- [x] 03-01-PLAN.md — Simulation configuration and result dataclasses
- [x] 03-02-PLAN.md — Intervention types with costs and effects
- [x] 03-03-PLAN.md — Simulator core with multi-timestep loop
- [x] 03-04-PLAN.md — Test suite for simulation module

### Phase 4: Optimization
**Goal**: System selects optimal interventions within budget constraints using pluggable optimizer

**Depends on**: Phase 3

**Requirements**: OPTM-01, OPTM-02, OPTM-03, OPTM-04

**Success Criteria** (what must be TRUE):
  1. System selects interventions that stay within annual budget constraint
  2. Greedy heuristic prioritizes interventions by risk-to-cost ratio
  3. User can swap greedy optimizer for MILP solver via pluggable interface
  4. System reports which interventions were selected and why (e.g., "selected due to high failure probability + low cost")

**Plans**: TBD

Plans:
- [ ] TBD

### Phase 5: Results & Polish
**Goal**: Users can export results, visualize outcomes, and understand the SDK through documentation

**Depends on**: Phase 4

**Requirements**: OUTP-01, OUTP-02, OUTP-03, OUTP-04, OUTP-05, OUTP-06, DEVX-02, DEVX-03, DEVX-04

**Success Criteria** (what must be TRUE):
  1. User can export intervention schedule to CSV (asset ID, year, action, cost)
  2. User can export cost projections and failure metrics by year to CSV
  3. User can compare 2-3 scenarios side-by-side (e.g., "do nothing" vs "optimized")
  4. User can generate basic visualizations (cost over time, failures avoided, risk reduction)
  5. All public API functions have type hints
  6. User can run Jupyter notebook examples demonstrating end-to-end workflow
  7. Documentation covers API reference and usage patterns

**Plans**: TBD

Plans:
- [ ] TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 3/3 | ✓ Complete | 2026-01-30 |
| 2. Deterioration Models | 3/3 | ✓ Complete | 2026-01-31 |
| 3. Simulation Core | 4/4 | ✓ Complete | 2026-02-03 |
| 4. Optimization | 0/TBD | Not started | - |
| 5. Results & Polish | 0/TBD | Not started | - |
