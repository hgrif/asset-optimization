# Phase 2: Deterioration Models - Context

**Gathered:** 2026-01-30
**Status:** Ready for planning

<domain>
## Phase Boundary

System calculates failure rates using Weibull deterioration model with pluggable architecture. Users provide their own Weibull parameters (shape, scale) per asset type. The model produces failure rates and probabilities for the entire portfolio. Custom deterioration models can be swapped in via abstract base class.

</domain>

<decisions>
## Implementation Decisions

### Parameter Configuration
- Parameters provided via code only (dict), no external config files
- Parameters set at model creation time (immutable)
- If asset type not in params dict, raise error (fail fast, no silent defaults)
- User brings their own Weibull parameters from literature/expertise

### Model Interface Design
- Abstract base class pattern: `class DeteriorationModel(ABC)`
- Single required method: `failure_rate()` (validation happens in `__init__`)
- Model accepts full portfolio DataFrame, not individual assets (vectorized)
- Transformer-style API: `transform()` returns input DataFrame with new columns added
- `transform()` returns a copy, does not modify input in-place

### Output Format
- Returns DataFrame with original columns + `failure_rate` + `failure_probability`
- Both hazard rate and probability included (simulation uses probability)
- Age column not duplicated in output (already in input)

### Asset Type Mapping
- User specifies which column identifies asset type: `type_column='material'`
- Age column defaults to 'age' (error if missing)
- Composite keys supported by user creating combined column beforehand

### Claude's Discretion
- Exact validation logic in `__init__`
- Helper methods on the model class
- Column naming conventions (snake_case assumed)
- Error message formatting

</decisions>

<specifics>
## Specific Ideas

- "I want it to feel familiar to data scientists" — scikit-learn familiarity but transformer output style
- Model should work across domains (pipes, roads, etc.) by letting user specify type column

</specifics>

<deferred>
## Deferred Ideas

- Parameter estimation from historical failure data — high priority for post-v1 backlog
- Confidence intervals on failure probabilities — future enhancement

</deferred>

---

*Phase: 02-deterioration-models*
*Context gathered: 2026-01-30*
