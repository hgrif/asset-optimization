# Requirements Archive: v1 MVP

**Archived:** 2026-02-05
**Status:** SHIPPED

This is the archived requirements specification for v1.
For current requirements, see `.planning/REQUIREMENTS.md` (created for next milestone).

---

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

- [x] **OPTM-01**: System selects interventions within budget constraint
- [x] **OPTM-02**: System uses greedy heuristic (prioritize by risk/cost ratio)
- [x] **OPTM-03**: Optimizer interface is pluggable (can swap heuristic for MILP later)
- [x] **OPTM-04**: System reports which interventions were selected and why

### Outputs

- [x] **OUTP-01**: System generates intervention schedule (asset ID, year, action)
- [x] **OUTP-02**: System generates cost projection by year
- [x] **OUTP-03**: System generates expected failure metrics by year
- [x] **OUTP-04**: User can compare multiple scenarios side-by-side
- [x] **OUTP-05**: User can export results to parquet
- [x] **OUTP-06**: System generates basic visualizations (cost over time, failures avoided)

### Developer Experience

- [x] **DEVX-01**: SDK installable via pip
- [x] **DEVX-02**: API has type hints throughout
- [x] **DEVX-03**: Jupyter notebook examples demonstrate end-to-end workflow
- [x] **DEVX-04**: Documentation covers API and usage patterns
- [x] **DEVX-05**: End-to-end test validates deterministic simulation results from portfolio data

### API & Data Handling

- [x] **API-01**: Portfolio is a DataFrame interface (no public Portfolio class); validation happens in consumers

### Traceability

- [x] **TRACE-01**: Simulation returns asset-level event history per year (action, failure flag, costs, age)

### Visualization Enhancements

- [x] **VIS-01**: Heatmap visualizes asset actions over years with categorical colors

## Traceability

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
| OPTM-01 | Phase 4 | Complete |
| OPTM-02 | Phase 4 | Complete |
| OPTM-03 | Phase 4 | Complete |
| OPTM-04 | Phase 4 | Complete |
| OUTP-01 | Phase 5 | Complete |
| OUTP-02 | Phase 5 | Complete |
| OUTP-03 | Phase 5 | Complete |
| OUTP-04 | Phase 5 | Complete |
| OUTP-05 | Phase 5 | Complete |
| OUTP-06 | Phase 5 | Complete |
| DEVX-01 | Phase 1 | Complete |
| DEVX-02 | Phase 5 | Complete |
| DEVX-03 | Phase 5 | Complete |
| DEVX-04 | Phase 5 | Complete |
| DEVX-05 | Phase 6 | Complete |
| API-01 | Phase 6 | Complete |
| TRACE-01 | Phase 6 | Complete |
| VIS-01 | Phase 6 | Complete |

**Coverage:**
- v1 requirements: 36 total
- Shipped: 36
- Adjusted: 0
- Dropped: 0

---

## Milestone Summary

**Shipped:** 36 of 36 v1 requirements
**Adjusted:** None
**Dropped:** None

All requirements were implemented as specified. No scope changes during milestone.

---

*Archived: 2026-02-05 as part of v1 milestone completion*
