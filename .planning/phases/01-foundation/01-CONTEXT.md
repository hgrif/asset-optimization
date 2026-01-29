# Phase 1: Foundation - Context

**Gathered:** 2026-01-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Project structure, data loading, and validation for asset portfolios. Users can load CSV/Excel files, validate data quality, and query assets. This phase delivers the Portfolio class and loading infrastructure — deterioration models, simulation, and optimization are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Data Loading API
- Class constructor pattern: `Portfolio.from_csv()`, `Portfolio.from_excel()`, `Portfolio.from_dataframe()`
- Accept both file paths and pandas DataFrames as input
- Exact column names required (no mapping, no auto-detection)
- Core fields required: asset_id, install_date, asset_type, material
- Date parsing via pandas auto-inference
- Single file per load (no multi-file merging)
- Warn about unused columns, then ignore them
- Return Portfolio directly (not a result wrapper)
- Informative repr when displaying Portfolio in REPL/notebook

### Validation Behavior
- Fail fast on first validation error (no partial loads)
- Basic sanity checks: install_date not in future, no duplicate IDs, reasonable ranges
- Custom exception class with structured error details
- Validation always runs, no override option

### Quality Reporting
- Property access pattern: `portfolio.quality.completeness`, `portfolio.quality.missing_counts`
- Metrics: completeness percentages per column + absolute counts (rows, nulls)
- Computed eagerly at load time (always available)
- Rich display in notebooks (formatted table)

### Portfolio Object Design
- Dict-like ID lookup: `portfolio['PIPE-001']`
- Expose underlying DataFrame via `portfolio.data` property for filtering
- Age-related computed properties: count, asset_types, mean_age, age_distribution, oldest, newest
- Mutable after loading
- `len(portfolio)` returns asset count
- Informative repr showing key summary stats

### Claude's Discretion
- Exact exception class hierarchy
- Internal data storage structure
- Package directory layout
- Test file organization

</decisions>

<specifics>
## Specific Ideas

- "I want the display string to be informative" — repr should show asset count, date range, asset types at a glance
- scikit-learn style API (from PROJECT.md) — OOP for classes, familiar to data scientists

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-foundation*
*Context gathered: 2026-01-30*
