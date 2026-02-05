# Research Summary: v2 Extended Asset Modeling

**Synthesized:** 2026-02-05
**Sources:** STACK.md (v1), FEATURES.md (v1), PITFALLS.md (v1), ARCHITECTURE.md (v2)

## Executive Summary

v2 extends the asset optimization SDK with three major capabilities:
1. **Multi-property hazard modeling** — Proportional hazards (Cox-style) with covariates
2. **Additional asset domains** — Roads beyond water pipes
3. **Asset relationships** — Groupings (connected assets) and hierarchy (parent-child)

The existing pluggable architecture supports these extensions well. Key risks are integration complexity for groupings/hierarchy and ensuring backward compatibility.

## Stack Additions for v2

**Required additions:** None — existing stack (NumPy, Pandas, SciPy, Pandera) sufficient.

**Stack remains:**
- Python 3.11+, NumPy, Pandas 3.0, SciPy, Pandera
- Weibull functions from `scipy.stats` work for baseline hazard
- No graph library needed — asset relationships stored in DataFrame columns (`group_id`, `parent_id`)

**Rationale:** Keep dependencies minimal. Graph libraries (networkx) add complexity without clear benefit for simple parent/group relationships queryable via Pandas.

## Feature Summary by Area

### 1. Proportional Hazards (Multi-Property)

**Table Stakes:**
- ProportionalHazardsModel implementing DeteriorationModel interface — medium complexity
- Configurable covariate columns (user specifies which columns affect hazard) — low complexity
- Configurable coefficients per covariate (β values) — low complexity
- Backward-compatible with existing portfolios (covariates optional) — low complexity

**Differentiators:**
- Composition pattern (wrap any baseline model, not just Weibull) — medium complexity
- Coefficient validation (warn if extreme values) — low complexity

**Anti-features:**
- Auto-fitting coefficients from data — requires survival analysis expertise, defer to v3
- Time-varying covariates — significant complexity, not needed for water/roads

### 2. Roads Domain

**Table Stakes:**
- DomainConfig abstraction for domain-specific behavior — medium complexity
- Road schema (surface_type, traffic_load, climate_zone) — low complexity
- Road intervention types (patch, resurface, reconstruct) — low complexity
- Road-specific deterioration parameters — medium complexity

**Differentiators:**
- Domain registry pattern (add domains without code changes) — medium complexity
- Cross-domain portfolios (mixed pipes + roads in one simulation) — high complexity, defer

**Anti-features:**
- Pavement condition index (PCI) calculation from sensor data — different product
- Traffic simulation integration — out of scope

### 3. Asset Groupings

**Table Stakes:**
- Group membership via `group_id` column — low complexity
- Failure propagation (failed asset increases neighbor risk) — medium complexity
- Shared interventions (repair one, neighbors benefit) — medium complexity
- Group constraints (must intervene on entire group) — high complexity (optimizer changes)

**Differentiators:**
- Configurable propagation factors — low complexity
- Multiple relationship types (connected vs constraint vs shared) — medium complexity

**Anti-features:**
- Full graph topology (shortest paths, flow analysis) — different problem domain
- Real-time network flow simulation — hydraulic modeling, not asset management

### 4. Asset Hierarchy

**Table Stakes:**
- Parent-child relationship via `parent_id` column — low complexity
- Dependency failures (parent fails → children fail) — medium complexity
- Cost sharing (parent intervention reduces child cost) — medium complexity
- Condition propagation (parent condition affects child rates) — medium complexity

**Differentiators:**
- Multi-level hierarchies (pump → main → branches) — medium complexity
- Configurable propagation factors per level — low complexity

**Anti-features:**
- Automatic hierarchy inference from spatial data — GIS functionality
- Criticality scoring based on downstream assets — separate analysis, not core simulation

## Key Pitfalls to Avoid

From PITFALLS.md, these are most relevant to v2:

### For Proportional Hazards
- **Invalid model assumptions:** Validate that covariates actually affect failure rates
- **Prevention:** Document expected coefficient signs (larger diameter = lower risk?), provide coefficient validation

### For Multi-Domain
- **Over-generalization:** Making everything configurable slows development
- **Prevention:** Build roads domain concretely first, then extract patterns

### For Groupings/Hierarchy
- **Cascade complexity:** Failure propagation can cause exponential cascades
- **Prevention:** Cap propagation depth, test with realistic network topologies
- **Performance trap:** O(n²) group lookups
- **Prevention:** Use efficient indexing (group_id index on DataFrame)

## Suggested Build Order

1. **Phase 7: Proportional Hazards** — Lowest risk, validates multi-property concept
2. **Phase 8: Roads Domain** — Medium risk, validates domain abstraction
3. **Phase 9: Asset Groupings** — Higher risk, changes simulation loop
4. **Phase 10: Asset Hierarchy** — Higher risk, changes simulation and optimization
5. **Phase 11: Documentation** — Consolidate all new features

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Composition for ProportionalHazards | Wrap any baseline model, cleaner testing |
| DomainConfig pattern | Explicit domain behavior, easy to add domains |
| Relationships in DataFrame columns | No new dependencies, familiar Pandas patterns |
| Separate Groups from Hierarchy | Orthogonal concepts with different propagation rules |
| Optional covariate columns | Backward compatibility with v1 portfolios |

## Requirements Derivation

Based on research, v2 requirements should include:

**Proportional Hazards:**
- User can create ProportionalHazardsModel with baseline model and coefficients
- User can specify which DataFrame columns are covariates
- Existing portfolios work without covariates (uses baseline hazard only)

**Roads Domain:**
- User can load road portfolio with domain-specific schema
- User can run simulation with road-appropriate intervention types
- User can configure road deterioration parameters

**Asset Groupings:**
- User can define groups via group_id column
- User can enable failure propagation between group members
- User can enable shared intervention effects
- Optimizer respects group constraints (if configured)

**Asset Hierarchy:**
- User can define hierarchy via parent_id column
- User can enable dependency failures (parent fails → children fail)
- User can enable cost sharing (parent intervention reduces child cost)
- User can enable condition propagation

---
*Research summary for: v2 Extended Asset Modeling*
*Synthesized: 2026-02-05*
