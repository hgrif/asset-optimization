# Phase 4: Optimization - Context

**Gathered:** 2026-02-03
**Status:** Ready for planning

<domain>
## Phase Boundary

System selects optimal interventions within budget constraints using pluggable optimizer. Greedy heuristic prioritizes by risk-to-cost ratio. MILP solver is interface-only in v1. Reports which interventions were selected with metrics.

</domain>

<decisions>
## Implementation Decisions

### Prioritization Logic
- Risk-to-cost ratio: P(failure) ÷ intervention cost
- Tie-breaking: oldest asset first
- Configurable minimum risk threshold (min P(failure) to consider)
- Optimizer picks best intervention type per asset (Replace/Repair/Inspect/DoNothing)
- Cost-effectiveness for intervention comparison: (risk_before - risk_after) ÷ cost
- Single year optimization (no multi-year lookahead)
- All four intervention types considered by optimizer
- No grouping constraints — each asset evaluated independently
- Inspect recommended when cost trade-off makes sense (cheaper than Replace, borderline risk)
- No maximum interventions limit — budget is only constraint
- Exclusion list supported — user can pass asset IDs to skip

### Budget Constraints
- Single annual budget (one total number)
- Strict constraint — never exceed budget, even if close
- No minimum spend requirement — can recommend $0 if nothing meets threshold
- Report utilization: budget, spent, remaining, % utilized

### Selection Explanation
- Minimal detail: just metrics (risk score, cost, rank)
- Selected assets only — no explanation for skipped assets
- No comparison metrics (no before/after risk reduction)
- Results returned as DataFrame — users can sort/filter themselves

### Optimizer Interface
- Constructor parameter: `Optimizer(strategy='greedy')` or `Optimizer(strategy='milp')`
- Greedy only in v1 — MILP interface documented for future
- Raise `NotImplementedError` for unavailable strategies (no silent fallback)
- No custom optimizer subclassing in v1 — built-in strategies only

### Claude's Discretion
- Budget utilization logic (whether to use remaining budget for medium-risk assets)
- Internal data structures for optimization
- Exact implementation of cost-effectiveness calculations

</decisions>

<specifics>
## Specific Ideas

- API should feel like scikit-learn: `optimizer.fit(portfolio, budget)` or similar
- Risk-to-cost ratio aligns with standard asset management practice
- Greedy heuristic is sufficient for v1 since portfolios are ~1000 assets

</specifics>

<deferred>
## Deferred Ideas

- **Consequence/criticality field** — Add to portfolio schema for risk-weighted prioritization (risk = P(failure) × consequence)
- **Multi-year lookahead** — Consider future failure probabilities when making current decisions
- **MILP implementation** — Interface exists, actual solver deferred
- **Custom optimizer subclassing** — ABC pattern for user-defined strategies

</deferred>

---

*Phase: 04-optimization*
*Context gathered: 2026-02-03*
