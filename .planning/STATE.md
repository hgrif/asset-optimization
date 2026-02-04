# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2025-01-29)

**Core value:** Enable data-driven intervention decisions that minimize cost and risk across asset portfolios

**Current focus:** Phase 5: Results & Polish

## Current Position

Phase: 5 of 5 (Results & Polish)
Plan: 0 of TBD completed
Status: Ready to plan
Last activity: 2026-02-03 — Completed Phase 4 (Optimization)

Progress: [████████░░] 80%

## Performance Metrics

**Velocity:**
- Total plans completed: 13
- Average duration: 2m 47s
- Total execution time: 0.60 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation | 3 | 9m 44s | 3m 15s |
| 02-deterioration-models | 3 | 7m 0s | 2m 20s |
| 03-simulation-core | 4 | 12m 5s | 3m 1s |
| 04-optimization | 3 | 8m 7s | 2m 42s |

**Recent Trend:**
- Last 5 plans: 03-03 (2m 52s), 03-04 (5m 57s), 04-01 (2m 7s), 04-02 (4m), 04-03 (2m)
- Trend: Consistent

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Python SDK first, UI later — Balance flexibility for consultants with templated UI later
- scikit-learn style API — OOP for models/optimizers, functional helpers, familiar to data scientists
- Water pipes for v1 — Validate architecture before generalizing to other domains
- Weibull deterioration model — Well-understood statistical baseline
- Pluggable optimizer interface — Greedy heuristic first, MILP later via same interface
- Single deterministic run — No Monte Carlo in v1, but architecture allows it later
- **use-src-layout (01-01)** — Use src layout instead of flat layout for package structure
- **pandera-for-validation (01-01)** — Use Pandera for DataFrame schema validation
- **strict-false-coerce-true (01-01)** — Set Pandera schema with strict=False, coerce=True
- **eager-quality-metrics (01-02)** — Compute quality metrics at load time, not lazily
- **property-based-analysis (01-02)** — Use @property for portfolio analysis methods
- **isinstance-check-failure-cases (01-02)** — Check type before accessing Pandera failure_cases
- **pytest-as-test-framework (01-03)** — Use pytest for test suite (standard choice)
- **fixtures-in-conftest (01-03)** — Centralize shared fixtures in conftest.py
- **test-organization-by-functionality (01-03)** — Organize tests by functionality (portfolio, validation, quality)
- **abc-for-pluggable-models (02-01)** — Use ABC pattern for pluggable deterioration model architecture
- **immutable-transform-pattern (02-01)** — transform() methods return copies, never mutate input
- **direct-hazard-formula (02-02)** — Use direct h(t) formula instead of scipy pdf/sf for 3-5x performance
- **groupby-vectorization (02-02)** — Process each asset type separately with groupby for parameter lookup
- **zero-age-handling (02-02)** — Define h(0)=0 for numerical stability in hazard calculations
- **scipy-for-cdf-verification (02-03)** — Use scipy.stats.weibull_min.cdf to verify failure_probability calculations
- **direct-formula-for-hazard-verification (02-03)** — Verify hazard rate matches h(t)=(k/lambda)*(t/lambda)^(k-1)
- **frozen-dataclass-for-config (03-01)** — Use frozen=True for immutable configuration
- **post-init-validation (03-01)** — Validate parameters in __post_init__ method
- **convenience-methods-on-result (03-01)** — Add total_cost() and total_failures() for common queries
- **frozen-dataclass-immutable (03-02)** — Use frozen=True for InterventionType immutability
- **callable-age-effect (03-02)** — Use Callable[[float], float] for pluggable age transformations
- **post-init-validation (03-02)** — Validate cost >= 0 and non-empty name in __post_init__
- **conditional-probability-via-survival (03-03)** — Use S(t)-S(t+1)/S(t) for accurate failure sampling
- **isolated-rng-per-simulator (03-03)** — Each Simulator instance has own RNG for reproducibility
- **direct-params-access (03-03)** — Access model.params directly (not transform()) for survival function
- **test-class-organization (03-04)** — Organize tests by component (Config, Result, Intervention, Simulator)
- **parametrize-failure-responses (03-04)** — Use @pytest.mark.parametrize for testing all valid failure_response values
- **scipy-formula-verification (03-04)** — Verify conditional probability implementation against scipy.stats.weibull_min
- **follow-simulation-result-pattern (04-01)** — Use non-frozen dataclass with convenience properties for OptimizationResult
- **inherit-from-base-exception (04-01)** — OptimizationError inherits AssetOptimizationError for package exception hierarchy
- **two-stage-greedy (04-02)** — Stage 1 finds best intervention per asset, Stage 2 ranks by risk-to-cost ratio
- **cost-effectiveness-metric (04-02)** — Use (risk_before - risk_after) / cost for intervention selection
- **risk-to-cost-ranking (04-02)** — Use risk_before / cost for budget filling prioritization
- **scikit-learn-fit-api (04-02)** — fit() returns self with result_ attribute
- **asset-type-required-in-fixtures (04-03)** — Portfolio schema requires asset_type column in test fixtures
- **test-class-by-concern (04-03)** — Organize optimization tests into classes by concern (Init, Fit, Budget, Greedy, Threshold, Exclusions, Result, EdgeCases)

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-03
Stopped at: Completed Phase 4 (Optimization) — verified and committed
Resume file: None
