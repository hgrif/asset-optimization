# Asset Optimization

## What This Is

A Python SDK for simulating and optimizing infrastructure asset portfolios. Consultants use the library to model asset deterioration over time, run multi-year simulations, and generate optimized intervention plans (repair, replace, inspect, do nothing) under budget and resource constraints. Initially focused on water pipe networks with Weibull-based failure modeling.

## Core Value

Enable data-driven intervention decisions that minimize cost and risk across asset portfolios — answering "which assets should we fix, when, and why?"

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Load asset portfolio data (age, type, condition, location per asset)
- [ ] Configure Weibull deterioration model with parameters per asset type
- [ ] Evaluate failure rates/probabilities across portfolio
- [ ] Generate intervention options per asset (DoNothing, Inspect, Repair, Replace)
- [ ] Define costs and state effects for each intervention type
- [ ] Apply constraints (budget, crew capacity)
- [ ] Optimize intervention selection via heuristic solver
- [ ] Support pluggable optimizer interface (swap heuristic for MILP later)
- [ ] Run multi-timestep simulation (e.g., 10 years, optimize each year)
- [ ] Update asset states after decisions (age resets, condition changes)
- [ ] Output intervention schedules (which assets, which action, which year)
- [ ] Output cost projections and expected failure metrics
- [ ] Compare scenarios (e.g., "what if we double replacement budget?")

### Out of Scope

- Monte Carlo simulation (multiple stochastic runs) — future capability, architecture should allow it
- Other asset domains (data centers, rail) — v1 focuses on water pipes only
- ML-based performance models (survival models, boosted trees) — future, model interface should be pluggable
- Web frontend/backend — SDK first, UI layer comes later
- Real-time data integration — batch processing for now

## Context

- Informed by Core and Core 2, earlier systems in this domain with overlapping functionality
- Target users: consultants building custom models, asset managers at utilities
- Typical portfolio size: thousands of assets
- Performance models need to be pluggable (statistical now, ML later)
- The simulation loop: load → evaluate rates → generate options → optimize → apply decisions → advance time → repeat

## Constraints

- **Language**: Pure Python with NumPy/Pandas for performance — can refactor to Rust later if needed
- **Scale**: Must handle thousands of assets efficiently
- **Optimizer**: Heuristic first, but interface must allow MILP solvers (OR-Tools, Gurobi) to plug in

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Python SDK first, UI later | Consultants need flexibility; templated UI can wrap SDK | — Pending |
| scikit-learn style API | Balance of OOP (classes for models, optimizers) and functional (helper functions). Familiar pattern for data scientists. | — Pending |
| Water pipes for v1 | Concrete domain to validate architecture before generalizing | — Pending |
| Weibull deterioration model | Well-understood statistical model, good starting point | — Pending |
| Pluggable optimizer interface | Start with heuristics, swap in MILP when needed | — Pending |
| Single deterministic run (no Monte Carlo) | Simpler v1, but architecture should allow multiple runs later | — Pending |

---
*Last updated: 2025-01-29 after initialization*
