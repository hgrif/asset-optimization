# Requirements: Asset Optimization

**Defined:** 2026-01-30
**Core Value:** Enable data-driven intervention decisions that minimize cost and risk across asset portfolios

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Data & Portfolio

- [ ] **DATA-01**: User can load asset portfolio from CSV file
- [ ] **DATA-02**: User can load asset portfolio from Excel file
- [ ] **DATA-03**: System validates required fields on ingestion (age, type, condition)
- [ ] **DATA-04**: System reports data quality metrics (completeness, missing values)
- [ ] **DATA-05**: User can query and filter assets by attributes

### Deterioration Modeling

- [ ] **DTRN-01**: System calculates failure rates using Weibull 2-parameter model
- [ ] **DTRN-02**: User can configure Weibull shape and scale per asset type
- [ ] **DTRN-03**: System evaluates failure probabilities across entire portfolio efficiently
- [ ] **DTRN-04**: Model interface is pluggable (can swap Weibull for other models later)

### Simulation

- [ ] **SIMU-01**: User can run multi-timestep simulation (configurable years, e.g., 10-30)
- [ ] **SIMU-02**: System updates asset states after each timestep (age increments)
- [ ] **SIMU-03**: System applies intervention effects to asset state (age reset on replace, condition improvement on repair)
- [ ] **SIMU-04**: Simulation is deterministic with random seed control
- [ ] **SIMU-05**: System tracks cumulative metrics across timesteps (total cost, total failures)

### Interventions

- [ ] **INTV-01**: System supports 4 intervention types: DoNothing, Inspect, Repair, Replace
- [ ] **INTV-02**: User can configure cost per intervention type
- [ ] **INTV-03**: User can configure state effects per intervention type
- [ ] **INTV-04**: System generates intervention options per asset per timestep

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

- [ ] **DEVX-01**: SDK installable via pip
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
| DATA-01 | TBD | Pending |
| DATA-02 | TBD | Pending |
| DATA-03 | TBD | Pending |
| DATA-04 | TBD | Pending |
| DATA-05 | TBD | Pending |
| DTRN-01 | TBD | Pending |
| DTRN-02 | TBD | Pending |
| DTRN-03 | TBD | Pending |
| DTRN-04 | TBD | Pending |
| SIMU-01 | TBD | Pending |
| SIMU-02 | TBD | Pending |
| SIMU-03 | TBD | Pending |
| SIMU-04 | TBD | Pending |
| SIMU-05 | TBD | Pending |
| INTV-01 | TBD | Pending |
| INTV-02 | TBD | Pending |
| INTV-03 | TBD | Pending |
| INTV-04 | TBD | Pending |
| OPTM-01 | TBD | Pending |
| OPTM-02 | TBD | Pending |
| OPTM-03 | TBD | Pending |
| OPTM-04 | TBD | Pending |
| OUTP-01 | TBD | Pending |
| OUTP-02 | TBD | Pending |
| OUTP-03 | TBD | Pending |
| OUTP-04 | TBD | Pending |
| OUTP-05 | TBD | Pending |
| OUTP-06 | TBD | Pending |
| DEVX-01 | TBD | Pending |
| DEVX-02 | TBD | Pending |
| DEVX-03 | TBD | Pending |
| DEVX-04 | TBD | Pending |

**Coverage:**
- v1 requirements: 32 total
- Mapped to phases: 0
- Unmapped: 32 ⚠️

---
*Requirements defined: 2026-01-30*
*Last updated: 2026-01-30 after initial definition*
