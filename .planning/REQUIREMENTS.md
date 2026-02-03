# Requirements: Asset Optimization

**Defined:** 2026-01-30
**Core Value:** Enable data-driven intervention decisions that minimize cost and risk across asset portfolios

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Data & Portfolio

- [x] **DATA-01**: User can load asset portfolio from CSV file
- [x] **DATA-02**: User can load asset portfolio from Excel file
- [x] **DATA-03**: System validates required fields on ingestion (age, type, condition)
- [x] **DATA-04**: System reports data quality metrics (completeness, missing values)
- [x] **DATA-05**: User can query and filter assets by attributes

### Deterioration Modeling

- [x] **DTRN-01**: System calculates failure rates using Weibull 2-parameter model
- [x] **DTRN-02**: User can configure Weibull shape and scale per asset type
- [x] **DTRN-03**: System evaluates failure probabilities across entire portfolio efficiently
- [x] **DTRN-04**: Model interface is pluggable (can swap Weibull for other models later)

### Simulation

- [x] **SIMU-01**: User can run multi-timestep simulation (configurable years, e.g., 10-30)
- [x] **SIMU-02**: System updates asset states after each timestep (age increments)
- [x] **SIMU-03**: System applies intervention effects to asset state (age reset on replace, condition improvement on repair)
- [x] **SIMU-04**: Simulation is deterministic with random seed control
- [x] **SIMU-05**: System tracks cumulative metrics across timesteps (total cost, total failures)

### Interventions

- [x] **INTV-01**: System supports 4 intervention types: DoNothing, Inspect, Repair, Replace
- [x] **INTV-02**: User can configure cost per intervention type
- [x] **INTV-03**: User can configure state effects per intervention type
- [x] **INTV-04**: System generates intervention options per asset per timestep

### Optimization

- [ ] **OPTM-01**: System selects interventions within budget constraint
- [ ] **OPTM-02**: System uses greedy heuristic (prioritize by risk/cost ratio)
- [ ] **OPTM-03**: Optimizer interface is pluggable (can swap heuristic for MILP later)
- [ ] **OPTM-04**: System reports which interventions were selected and why

### Outputs

- [ ] **OUTP-01**: System generates intervention schedule (asset ID, year, action)
- [ ] **OUTP-02**: System generates cost projection by year
- [ ] **OUTP-03**: System generates expected failure metrics by year
- [ ] **OUTP-04**: User can compare multiple scenarios side-by-side
- [ ] **OUTP-05**: User can export results to CSV
- [ ] **OUTP-06**: System generates basic visualizations (cost over time, failures avoided)

### Developer Experience

- [x] **DEVX-01**: SDK installable via pip
- [ ] **DEVX-02**: API has type hints throughout
- [ ] **DEVX-03**: Jupyter notebook examples demonstrate end-to-end workflow
- [ ] **DEVX-04**: Documentation covers API and usage patterns

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Extended Modeling

- **DTRN-05**: ML-based deterioration models (survival analysis, gradient boosting)
- **DTRN-06**: Cohort-based rate models (group-level vs per-asset)
- **DTRN-07**: Multiple failure modes per asset type

### Advanced Optimization

- **OPTM-05**: MILP solver integration (PuLP with CBC/Gurobi)
- **OPTM-06**: Multiple constraint types (crew capacity, seasonal restrictions)
- **OPTM-07**: Risk-weighted optimization objective

### Simulation Extensions

- **SIMU-06**: Monte Carlo simulation (multiple stochastic runs)
- **SIMU-07**: State rollback and scenario branching
- **SIMU-08**: Batch scenario execution with parameter sweeps

### Additional Domains

- **DOMN-01**: Data center server management domain
- **DOMN-02**: Rail infrastructure domain
- **DOMN-03**: Generic asset type framework

### Integration

- **INTG-01**: Excel export for stakeholder reports
- **INTG-02**: Parquet persistence for large portfolios
- **INTG-03**: Geographic visualization (lat/lon on maps)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Web UI | SDK first; consultants build custom UIs; add templated UI in v2 if patterns emerge |
| Real-time data integration | Users have batch exports; focus on CSV/Excel import |
| Auto-tuning deterioration models | Statistical complexity; users want control over assumptions |
| Multi-objective optimization | Single objective + constraints sufficient; Pareto fronts add complexity |
| Geographic mapping (interactive) | Heavy dependencies; users have GIS tools |
| Database storage | Users want files for version control; pickle/parquet sufficient |
| Network topology modeling | Asset interdependencies are v2; validate core loop first |
| Automatic report generation | Opinionated formatting; export structured data instead |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| DATA-04 | Phase 1 | Complete |
| DATA-05 | Phase 1 | Complete |
| DTRN-01 | Phase 2 | Complete |
| DTRN-02 | Phase 2 | Complete |
| DTRN-03 | Phase 2 | Complete |
| DTRN-04 | Phase 2 | Complete |
| SIMU-01 | Phase 3 | Complete |
| SIMU-02 | Phase 3 | Complete |
| SIMU-03 | Phase 3 | Complete |
| SIMU-04 | Phase 3 | Complete |
| SIMU-05 | Phase 3 | Complete |
| INTV-01 | Phase 3 | Complete |
| INTV-02 | Phase 3 | Complete |
| INTV-03 | Phase 3 | Complete |
| INTV-04 | Phase 3 | Complete |
| OPTM-01 | Phase 4 | Pending |
| OPTM-02 | Phase 4 | Pending |
| OPTM-03 | Phase 4 | Pending |
| OPTM-04 | Phase 4 | Pending |
| OUTP-01 | Phase 5 | Pending |
| OUTP-02 | Phase 5 | Pending |
| OUTP-03 | Phase 5 | Pending |
| OUTP-04 | Phase 5 | Pending |
| OUTP-05 | Phase 5 | Pending |
| OUTP-06 | Phase 5 | Pending |
| DEVX-01 | Phase 1 | Complete |
| DEVX-02 | Phase 5 | Pending |
| DEVX-03 | Phase 5 | Pending |
| DEVX-04 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 32 total
- Mapped to phases: 32
- Unmapped: 0 ✓

---
*Requirements defined: 2026-01-30*
*Last updated: 2026-02-03 after Phase 3 completion*
