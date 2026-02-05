# Asset Optimization

## What This Is

A Python SDK for simulating and optimizing infrastructure asset portfolios. Consultants use the library to model asset deterioration over time, run multi-year simulations, and generate optimized intervention plans (repair, replace, inspect, do nothing) under budget and resource constraints. Focused on water pipe networks with Weibull-based failure modeling, pluggable architecture for future extensions.

## Core Value

Enable data-driven intervention decisions that minimize cost and risk across asset portfolios — answering "which assets should we fix, when, and why?"

## Current State

**Version:** v1 MVP (shipped 2026-02-05)
**Codebase:** 2,874 LOC Python (src) + 2,573 LOC tests = 5,447 total
**Tests:** 164 passing
**Tech stack:** Python, NumPy, Pandas, Pandera, SciPy, Matplotlib, Seaborn

**Capabilities:**
- Load portfolios from CSV/Excel with validation and quality metrics
- Weibull deterioration model with per-type parameters
- Multi-timestep simulation with 4 intervention types
- Budget-constrained greedy optimization
- Parquet export, scenario comparison, visualization
- Asset-level event history and action heatmap

## Requirements

### Validated

- Load asset portfolio data (age, type, condition, location per asset) — v1
- Configure Weibull deterioration model with parameters per asset type — v1
- Evaluate failure rates/probabilities across portfolio — v1
- Generate intervention options per asset (DoNothing, Inspect, Repair, Replace) — v1
- Define costs and state effects for each intervention type — v1
- Apply constraints (budget) — v1
- Optimize intervention selection via heuristic solver — v1
- Support pluggable optimizer interface (swap heuristic for MILP later) — v1
- Run multi-timestep simulation (e.g., 10 years, optimize each year) — v1
- Update asset states after decisions (age resets, condition changes) — v1
- Output intervention schedules (which assets, which action, which year) — v1
- Output cost projections and expected failure metrics — v1
- Compare scenarios (e.g., "what if we double replacement budget?") — v1
- DataFrame-first API (no Portfolio class in public API) — v1
- Asset-level event history per year — v1
- Action heatmap visualization — v1
- End-to-end determinism test — v1

### Active

(None — v1 complete, v2 scope TBD)

### Out of Scope

- Monte Carlo simulation (multiple stochastic runs) — future capability, architecture allows it
- Other asset domains (data centers, rail) — v1 focuses on water pipes only
- ML-based performance models (survival models, boosted trees) — future, model interface is pluggable
- Web frontend/backend — SDK first, UI layer comes later
- Real-time data integration — batch processing for now
- Crew capacity constraints — only budget constraints in v1

## Context

- Informed by Core and Core 2, earlier systems in this domain with overlapping functionality
- Target users: consultants building custom models, asset managers at utilities
- Typical portfolio size: thousands of assets
- Performance models are pluggable (statistical now, ML later)
- The simulation loop: load → evaluate rates → generate options → optimize → apply decisions → advance time → repeat

## Constraints

- **Language**: Pure Python with NumPy/Pandas for performance — can refactor to Rust later if needed
- **Scale**: Must handle thousands of assets efficiently (achieved: <1s for 1000+ assets)
- **Optimizer**: Heuristic first, but interface allows MILP solvers (OR-Tools, Gurobi) to plug in

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Python SDK first, UI later | Consultants need flexibility; templated UI can wrap SDK | ✓ Good — SDK provides full flexibility |
| scikit-learn style API | Balance of OOP (classes for models, optimizers) and functional (helper functions). Familiar pattern for data scientists. | ✓ Good — consistent API patterns |
| Water pipes for v1 | Concrete domain to validate architecture before generalizing | ✓ Good — validated core loop |
| Weibull deterioration model | Well-understood statistical model, good starting point | ✓ Good — pluggable for future models |
| Pluggable optimizer interface | Start with heuristics, swap in MILP when needed | ✓ Good — interface ready |
| Single deterministic run (no Monte Carlo) | Simpler v1, but architecture should allow multiple runs later | ✓ Good — seed control works |
| DataFrame-first portfolio API | Remove Portfolio class from public API, treat portfolios as DataFrames | ✓ Good — simpler interface |
| Always-on asset history | Track per-asset events by default for traceability | ✓ Good — enables debugging |

---
*Last updated: 2026-02-05 after v1 milestone*
