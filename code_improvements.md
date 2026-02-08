# Code Review Action Items (Prioritized)

1) Fix simulator/model interface mismatch (Impact: High, Effort: Medium)
   - Simulator assumes `WeibullModel` internals (`params`, `type_column`) and uses `weibull_min` directly, which breaks for any other `DeteriorationModel` implementation. Add a required interface method (e.g., `conditional_failure_probability(state_df)` or `conditional_failure_probability(ages, types)`), or update `DeteriorationModel` to expose the needed params in a consistent way. Update `Simulator._calculate_conditional_probability` to use only the abstract interface and raise if unsupported.

2) Align simulation cost breakdown with docstring or update docs (Impact: High, Effort: Low)
   - `SimulationResult.cost_breakdown` docstring claims “by intervention type and asset type,” but the implementation stores yearly aggregates. Either change the implementation to match the docstring or update the docstring (and tests) to reflect actual contents.

3) Prevent `read_csv`/`read_excel` dtype errors when optional columns are missing (Impact: High, Effort: Medium)
   - `from_csv`/`from_excel` pass a full dtype map including optional columns. Pandas can error if those columns are absent, causing a pandas error instead of a `ValidationError`. Consider reading without dtype constraints, or applying dtype only to columns that exist (e.g., read first, then `df = df.astype(existing_dtypes)`), then rely on schema coercion.

4) Make date validation deterministic and resilient (Impact: Medium, Effort: Low)
   - `portfolio_schema` uses `Check.less_than_or_equal_to(pd.Timestamp.now())`, which is evaluated at import time. Use a callable check (e.g., `Check(lambda s: s <= pd.Timestamp.now())`) or inject a “reference date” parameter so validation uses the actual current time when validating.

5) Reduce time-based test flakiness (Impact: Medium, Effort: Medium)
   - Tests assert age ranges based on the wall clock and fixtures with fixed dates (e.g., `invalid_future_date.csv` uses 2030). These will eventually fail as “future” becomes past or the mean age drifts. Introduce a time-freezing fixture, pass a reference date into Portfolio methods, or generate dates relative to “today” inside tests.

6) Add validation for negative/invalid ages in WeibullModel (Impact: Medium, Effort: Low)
   - `WeibullModel` only checks numeric age. Negative ages can silently produce odd hazards and probabilities. Either validate ages are >= 0 (raise) or clamp to 0, and add tests.

7) Add simulator tests for core behaviors (Impact: Medium, Effort: Medium)
   - No tests cover `Simulator`, `SimulationConfig`, `InterventionType`, or `SimulationResult`. Add coverage for:
     - deterministic outcomes with `random_seed`
     - cost breakdown totals and failure counts
     - failure_response behavior (`replace`/`repair`/`record_only`)
     - `SimulationResult.total_cost()`/`total_failures()` for empty/partial summaries

8) Handle unknown asset types consistently in simulation (Impact: Medium, Effort: Low)
   - `_calculate_conditional_probability` silently skips unknown types, which yields zero failure probability. Prefer raising an explicit error or logging/collecting missing types to avoid silently “immortal” assets.

9) Mark performance tests as optional/slow (Impact: Low, Effort: Low)
   - Tests enforce sub-second/5-second bounds that can be flaky on CI or slower machines. Mark them with `@pytest.mark.slow` and document how to run them, or relax thresholds based on environment.
