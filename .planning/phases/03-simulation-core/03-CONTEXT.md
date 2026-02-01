# Phase 3: Simulation Core - Context

**Gathered:** 2026-02-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Multi-timestep simulation with intervention effects on asset states. Users can run simulations that step through time, update asset conditions, apply interventions, and track costs/failures. This phase builds the core simulation engine; optimization logic is Phase 4.

</domain>

<decisions>
## Implementation Decisions

### Timestep Mechanics
- Annual timesteps only (one step = one year)
- Expected value calculations (deterministic), but architecture should allow probabilistic sampling later
- Use conditional probability P(fail this year | survived to year t) for failure calculations
- Track both calendar year and effective age (interventions reset effective age, not calendar)
- Order within timestep: Age → Failures → Interventions
- User specifies simulation start year; ages calculated from install_date; default to current year
- No warm-start/resume — each simulation starts fresh from portfolio + start year

### Intervention Behavior
- **Replace**: Resets effective age to 0, with option to upgrade asset type (e.g., cast iron → PVC)
- **Repair**: Configurable age reduction per repair type (user defines effectiveness parameter)
- **Inspect**: Triggers follow-up actions via user-defined rules (inspection → condition check → action mapping)
- **DoNothing**: No cost, no state change (baseline comparison)

### Simulation Outputs
- Summary stats always returned (costs, failures, risk per year)
- Asset-level traces optional (full history per asset — can be memory-heavy for large portfolios)
- Cost breakdowns by intervention type AND by asset type
- Comparison support deferred to Phase 5 (Results & Polish)

### Failure Handling
- Configurable failure response policy (auto-replace, auto-repair, or record-only)
- Separate cost components: direct cost (emergency repair) + consequence cost (service disruption)
- Uniform consequence costs per asset type (no criticality weighting in v1)
- Full event log of failures with context (asset ID, year, age at failure, type, costs)

### Claude's Discretion
- Result object structure (recommend: dataclass with DataFrames to match SDK patterns)
- Internal state management during simulation loop
- Memory optimization for large portfolio traces
- Exact parameter names and defaults

</decisions>

<specifics>
## Specific Ideas

- Expected value is primary mode, but "keep the door open" for probabilistic sampling via random seed later
- Inspection follow-up rules should feel like configuring a simple policy, not writing code

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-simulation-core*
*Context gathered: 2026-02-01*
