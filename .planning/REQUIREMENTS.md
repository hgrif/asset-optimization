# Requirements: Asset Optimization v2

**Defined:** 2026-02-05
**Core Value:** Enable data-driven intervention decisions that minimize cost and risk across asset portfolios

## v2 Requirements

Requirements for Extended Asset Modeling milestone. Each maps to roadmap phases.

### Proportional Hazards

- [ ] **HAZD-01**: User can create ProportionalHazardsModel with a baseline deterioration model
- [ ] **HAZD-02**: User can specify which DataFrame columns are covariates affecting hazard rate
- [ ] **HAZD-03**: User can provide coefficient (β) values for each covariate
- [ ] **HAZD-04**: ProportionalHazardsModel implements DeteriorationModel interface (works with Simulator)
- [ ] **HAZD-05**: Portfolios without covariate columns use baseline hazard only (backward compatible)

### Roads Domain

- [ ] **ROAD-01**: User can define domain configuration (schema, intervention types, parameters)
- [ ] **ROAD-02**: User can load road portfolio with domain-specific validation schema
- [ ] **ROAD-03**: Road schema supports surface_type, traffic_load, climate_zone columns
- [ ] **ROAD-04**: User can specify road intervention types (do_nothing, inspect, patch, resurface, reconstruct)
- [ ] **ROAD-05**: User can configure road-specific deterioration parameters

### Asset Groupings

- [ ] **GRUP-01**: User can define asset groups via group_id column in portfolio
- [ ] **GRUP-02**: User can enable failure propagation (failed asset increases group members' risk)
- [ ] **GRUP-03**: User can configure propagation factor (how much risk increases)
- [ ] **GRUP-04**: Optimizer respects group constraints (must intervene on entire group together)

### Asset Hierarchy

- [ ] **HIER-01**: User can define asset hierarchy via parent_id column in portfolio
- [ ] **HIER-02**: User can enable dependency failures (parent fails → children fail)
- [ ] **HIER-03**: Simulator propagates failures down hierarchy tree

### Documentation

- [ ] **DOCS-01**: Notebook example demonstrating proportional hazards with covariates
- [ ] **DOCS-02**: Notebook example demonstrating road domain simulation
- [ ] **DOCS-03**: Notebook example demonstrating asset groupings
- [ ] **DOCS-04**: Notebook example demonstrating asset hierarchy
- [ ] **DOCS-05**: API documentation updated for all new classes and functions

## Future Requirements

Deferred to future release. Tracked but not in current roadmap.

### Proportional Hazards (v3)

- **HAZD-F01**: Coefficient validation (warn on extreme values)
- **HAZD-F02**: Auto-fitting coefficients from historical failure data

### Asset Groupings (v3)

- **GRUP-F01**: Shared interventions (repair one asset, neighbors benefit)
- **GRUP-F02**: Multiple relationship types in single portfolio

### Asset Hierarchy (v3)

- **HIER-F01**: Cost sharing (parent intervention reduces child cost)
- **HIER-F02**: Condition propagation (parent condition affects child deterioration rates)
- **HIER-F03**: Configurable propagation factors per hierarchy level

### Additional Domains (v3+)

- **DOMN-F01**: Data centers domain
- **DOMN-F02**: Rail infrastructure domain
- **DOMN-F03**: Cross-domain portfolios (mixed asset types)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Auto-fitting coefficients | Requires survival analysis expertise; users provide coefficients |
| Traffic simulation integration | Different product domain |
| Pavement condition index (PCI) calculation | Sensor data processing, not asset optimization |
| Full graph topology (shortest paths, flow) | Network analysis, not asset management |
| Real-time network flow simulation | Hydraulic modeling, not asset optimization |
| Monte Carlo simulation | Deferred to v3; architecture supports it |
| ML-based deterioration models | Interface is pluggable; users can implement |
| Web UI | SDK-first; UI layer comes later |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| HAZD-01 | TBD | Pending |
| HAZD-02 | TBD | Pending |
| HAZD-03 | TBD | Pending |
| HAZD-04 | TBD | Pending |
| HAZD-05 | TBD | Pending |
| ROAD-01 | TBD | Pending |
| ROAD-02 | TBD | Pending |
| ROAD-03 | TBD | Pending |
| ROAD-04 | TBD | Pending |
| ROAD-05 | TBD | Pending |
| GRUP-01 | TBD | Pending |
| GRUP-02 | TBD | Pending |
| GRUP-03 | TBD | Pending |
| GRUP-04 | TBD | Pending |
| HIER-01 | TBD | Pending |
| HIER-02 | TBD | Pending |
| HIER-03 | TBD | Pending |
| DOCS-01 | TBD | Pending |
| DOCS-02 | TBD | Pending |
| DOCS-03 | TBD | Pending |
| DOCS-04 | TBD | Pending |
| DOCS-05 | TBD | Pending |

**Coverage:**
- v2 requirements: 22 total
- Mapped to phases: 0
- Unmapped: 22 (pending roadmap creation)

---
*Requirements defined: 2026-02-05*
*Last updated: 2026-02-05 after initial definition*
