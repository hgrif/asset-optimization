# Roadmap: Asset Optimization v2

## Milestones

- v1.0 MVP - Phases 1-6 (shipped 2026-02-05)
- **v2.0 Extended Asset Modeling** - Phases 7-11 (in progress)

## Overview

v2 extends the asset optimization SDK with advanced deterioration modeling (proportional hazards with covariates), domain-specific configurations (roads), and asset relationship features (groupings and hierarchy). The build order follows dependency flow: proportional hazards provides the modeling foundation, roads demonstrates domain configuration, groupings and hierarchy add relationship modeling, and documentation captures all new capabilities.

## Phases

<details>
<summary>v1.0 MVP (Phases 1-6) - SHIPPED 2026-02-05</summary>

See `.planning/milestones/v1-ROADMAP.md` for full details.

</details>

### v2.0 Extended Asset Modeling

**Milestone Goal:** Expand modeling capabilities with multi-property hazards, additional asset domains, and asset relationships.

- [ ] **Phase 7: Proportional Hazards** - Covariate-based failure rate modeling with notebook
- [ ] **Phase 8: Roads Domain** - Domain-specific configuration and validation with notebook
- [ ] **Phase 9: Asset Groupings** - Group-level constraints and failure propagation with notebook
- [ ] **Phase 10: Asset Hierarchy** - Parent-child dependency failures with notebook and API docs

## Phase Details

### Phase 7: Proportional Hazards
**Goal**: Users can model failure rates that depend on asset properties (covariates) beyond just age and type
**Depends on**: v1 complete (DeteriorationModel interface exists)
**Requirements**: HAZD-01, HAZD-02, HAZD-03, HAZD-04, HAZD-05, DOCS-01
**Success Criteria** (what must be TRUE):
  1. User can create a ProportionalHazardsModel with any existing deterioration model as baseline
  2. User can specify DataFrame columns as covariates and provide coefficient values for each
  3. ProportionalHazardsModel works with Simulator (produces failure rates, integrates with simulation loop)
  4. Existing portfolios without covariate columns continue to work (baseline hazard only)
  5. Notebook demonstrates proportional hazards with covariates affecting failure rates
**Plans**: TBD

Plans:
- [ ] 07-01: TBD
- [ ] 07-02: TBD

### Phase 8: Roads Domain
**Goal**: Users can configure and simulate road asset portfolios with domain-specific schema and interventions
**Depends on**: Phase 7 (proportional hazards enables road covariates)
**Requirements**: ROAD-01, ROAD-02, ROAD-03, ROAD-04, ROAD-05, DOCS-02
**Success Criteria** (what must be TRUE):
  1. User can define a road domain configuration specifying schema, intervention types, and parameters
  2. User can load a road portfolio with validation of road-specific columns (surface_type, traffic_load, climate_zone)
  3. User can specify road intervention types (do_nothing, inspect, patch, resurface, reconstruct) with costs and effects
  4. User can run simulation with road-specific deterioration parameters
  5. Notebook demonstrates road domain configuration and simulation
**Plans**: TBD

Plans:
- [ ] 08-01: TBD
- [ ] 08-02: TBD

### Phase 9: Asset Groupings
**Goal**: Users can model related assets that share risk and require coordinated interventions
**Depends on**: Phase 8 (domain patterns established)
**Requirements**: GRUP-01, GRUP-02, GRUP-03, GRUP-04, DOCS-03
**Success Criteria** (what must be TRUE):
  1. User can define asset groups via group_id column in portfolio DataFrame
  2. User can enable failure propagation where a failed asset increases risk for other group members
  3. User can configure propagation factor controlling how much risk increases
  4. Optimizer respects group constraints (intervening on one asset requires intervening on all in group)
  5. Notebook demonstrates asset groupings with failure propagation
**Plans**: TBD

Plans:
- [ ] 09-01: TBD
- [ ] 09-02: TBD

### Phase 10: Asset Hierarchy
**Goal**: Users can model parent-child asset relationships where parent failures cascade to children
**Depends on**: Phase 9 (relationship modeling patterns established)
**Requirements**: HIER-01, HIER-02, HIER-03, DOCS-04, DOCS-05
**Success Criteria** (what must be TRUE):
  1. User can define asset hierarchy via parent_id column in portfolio DataFrame
  2. User can enable dependency failures where parent failure causes child failures
  3. Simulator correctly propagates failures down the hierarchy tree during simulation
  4. Notebook demonstrates asset hierarchy with dependency failures
  5. API documentation covers all new v2 classes and functions
**Plans**: TBD

Plans:
- [ ] 10-01: TBD
- [ ] 10-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 7 -> 8 -> 9 -> 10

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 7. Proportional Hazards | v2.0 | 0/TBD | Not started | - |
| 8. Roads Domain | v2.0 | 0/TBD | Not started | - |
| 9. Asset Groupings | v2.0 | 0/TBD | Not started | - |
| 10. Asset Hierarchy | v2.0 | 0/TBD | Not started | - |

---
*Roadmap created: 2026-02-05*
*Milestone: v2.0 Extended Asset Modeling*
