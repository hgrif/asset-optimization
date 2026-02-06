# Phase 8: Roads Domain - Context

**Gathered:** 2026-02-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Add road asset portfolios as a use case for the SDK. Roads is not a new system — it's a new domain with its own schema, interventions, and deterioration parameters that uses the existing modeling and simulation infrastructure. This phase also introduces a `domains/` package with a shared Domain interface, and refactors existing water pipe functionality into the same pattern.

</domain>

<decisions>
## Implementation Decisions

### Domain philosophy
- Roads is a use case, not a new abstraction system
- No formal "domain config" object that changes how simulation works
- Domains are config providers — they produce schema validators, default interventions, and default model params
- Users assemble domain outputs and pass them to the existing Simulator/Optimizer API (unchanged)

### Domain classes pattern
- Create `src/asset_optimization/domains/` package with a shared Domain interface (protocol or ABC)
- Each domain is a class: `PipeDomain`, `RoadDomain`
- Domain classes provide: `.validate(df)`, `.default_interventions()`, `.default_model()` (or similar)
- Refactor existing pipe schema and interventions into `PipeDomain` as part of this phase
- Proves the pattern works for both domains

### Road schema
- Road schema is standalone — does NOT extend the base pipe schema
- Required columns: `surface_type`, `traffic_load`, `climate_zone` (plus `asset_id`, `install_date`)
- Valid values are a fixed set with defaults (e.g., surface_type in [asphalt, concrete, gravel])
- Exact valid values to be determined by research into real road performance modeling

### Road interventions
- Five intervention types: do_nothing, inspect, patch, resurface, reconstruct
- Costs should be realistic defaults based on industry research — usable out of the box
- Costs vary by surface_type (e.g., resurfacing concrete costs more than asphalt)
- Reconstruct can change surface type (e.g., gravel -> asphalt), using the existing upgrade_type pattern
- Claude's discretion on whether to create road-specific InterventionType constants or reuse/extend existing ones

### Covariates and deterioration
- traffic_load and climate_zone affect deterioration rates only (via ProportionalHazards covariates), not intervention costs
- Deterioration parameters should be research-backed with realistic defaults per surface type
- Research phase should investigate actual road performance modeling to ground parameters

### Claude's Discretion
- Whether road interventions reuse existing DO_NOTHING/INSPECT constants or create road-specific ones
- Exact column names and valid value sets (guided by research)
- How model parameter lookup adapts from `material` column to road-specific columns
- Internal organization of the domains package

</decisions>

<specifics>
## Specific Ideas

- "I only want to add more use cases — a new system is not required"
- Domains and performance models are two separate concepts: domains are use cases, models (Weibull, ProportionalHazards) can apply to any domain
- Research should drive the road-specific requirements (surface types, intervention costs, deterioration parameters)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 08-roads-domain*
*Context gathered: 2026-02-06*
