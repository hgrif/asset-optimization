# Phase 6: Asset Traceability - Context

**Gathered:** 2026-02-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Improve traceability and usability post-v1: add a deterministic end-to-end test, remove the Portfolio class from the public API in favor of DataFrame inputs, move validation into consumers, add asset-level event tracking to simulations with a default return, and introduce a heatmap visualization.

</domain>

<decisions>
## Implementation Decisions

### End-to-end test
- Covers portfolio data → simulation results only
- Asserts deterministic outputs across repeated runs with the same seed
- Does NOT include visualization

### Portfolio interface
- Portfolio becomes an internal DataFrame alias (no user-facing class in __init__)
- Validation occurs inside Simulator/Optimizer using shared validator (schema + QualityMetrics)
- CSV/Excel loading helpers (if kept) remain internal and return DataFrames

### Asset event history
- Asset history is collected by default; optional opt-out for memory via config flag
- One row per asset per year
- Columns: year, asset_id, age, action, failed, failure_cost, intervention_cost, total_cost
- Action values align to failure_response: none, record_only, repair, replace (stable ordering for plots)

### Visualization
- Add action heatmap (assets x years) with categorical colors
- Provide action palette override and action order to control legend

### Testing
- Add a true end-to-end test to cover data -> model -> simulation determinism

### Claude's Discretion
- Exact naming of the asset history DataFrame field (asset_history vs asset_events)
- How to represent action/costs when no failure occurs
- Whether to expose lightweight helper for sampling assets in heatmap

</decisions>

<specifics>
## Specific Ideas

- Heatmap should handle large portfolios by sampling or allowing a max_assets parameter
- Asset history should be returned on SimulationResult for downstream exports and plots

</specifics>

<deferred>
## Deferred Ideas

- Interactive heatmaps (Altair/Plotly)
- Map-based asset visualization

</deferred>

---

*Phase: 06-asset-traceability*
*Context gathered: 2026-02-05*
