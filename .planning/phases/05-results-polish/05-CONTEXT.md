# Phase 5: Results & Polish - Context

**Gathered:** 2026-02-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can export results, visualize outcomes, and understand the SDK through documentation. This includes intervention schedule exports, cost projections, scenario comparison, basic charts, and Jupyter notebook examples.

</domain>

<decisions>
## Implementation Decisions

### Export formats & structure
- Two export levels: **minimal** (asset_id, year, intervention_type, cost) and **detailed** (adds risk_score, rank, material, age, risk_before, risk_after, risk_reduction)
- **Parquet format only** — no CSV exports
- Separate files: `schedule_minimal.parquet` and `schedule_detailed.parquet`
- Long format for cost projections (rows for each year-metric combo — easier for plotting)
- Export methods on result objects: `result.to_parquet('path')` following pandas pattern
- No metadata in exports — users track provenance themselves
- Detailed export includes computed fields (risk_before, risk_after, risk_reduction)

### Scenario comparison
- Output as DataFrame with columns: scenario, year, metric, value
- Support auto-generation of "do nothing" baseline: `compare(result, baseline='do_nothing')`

### Visualization style
- Library preference order: **seaborn > pandas plotting > OOP matplotlib**
- Standard chart set for v1:
  - Cost over time (line chart)
  - Failures by year (bar chart)
  - Risk distribution (histogram)
  - Scenario comparison (grouped bar)
- Return axes objects for user customization: `ax = result.plot_costs()`
- SDK-specific theme for consistent styling across all plots

### Documentation
- **Docstrings only** for API reference (NumPy-style, accessible via help())
- No generated documentation site for v1
- **Multiple focused notebooks**: quickstart, optimization, visualization
- Include synthetic data so notebooks run out-of-the-box
- Tutorial-style with explanatory markdown between code cells

### Claude's Discretion
- Scenario creation API design (named objects vs dict of results)
- Which metrics to include in comparison by default
- Combined vs separate export methods structure

</decisions>

<specifics>
## Specific Ideas

- Plots should have a consistent SDK theme — not just seaborn defaults
- "Do nothing" baseline auto-generation for easy before/after comparison
- Notebooks should be educational, not just code dumps

</specifics>

<deferred>
## Deferred Ideas

- **Altair for interactive plots** — add to backlog for future consideration
- Generated documentation site (Sphinx/MkDocs) — defer until SDK matures

</deferred>

---

*Phase: 05-results-polish*
*Context gathered: 2026-02-03*
