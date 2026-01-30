---
phase: 01-foundation
plan: 01
subsystem: package-infrastructure
tags: [python, packaging, validation, pandera, exceptions]
dependency:
  requires: []
  provides:
    - pip-installable package structure
    - custom exception hierarchy
    - pandera validation schema
  affects:
    - 01-02 (Portfolio class implementation)
    - 01-03 (Quality metrics computation)
tech-stack:
  added:
    - setuptools: package building (pyproject.toml)
    - pandas: ">=2.0.0" (CSV/Excel data handling)
    - openpyxl: ">=3.1.0" (Excel file support)
    - pandera: ">=0.18.0" (DataFrame schema validation)
    - pytest: ">=7.1.0" (testing framework)
  patterns:
    - src layout for pip-installable packages
    - custom exception hierarchy with structured error details
    - declarative schema validation with Pandera
key-files:
  created:
    - pyproject.toml
    - README.md
    - src/asset_optimization/__init__.py
    - src/asset_optimization/exceptions.py
    - src/asset_optimization/schema.py
  modified: []
decisions:
  - id: use-src-layout
    what: Use src layout instead of flat layout
    why: Prevents import issues during development, forces proper installation
    impact: All imports require package installation (editable mode for dev)
  - id: pandera-for-validation
    what: Use Pandera for DataFrame schema validation
    why: Lightweight, pandas-native API, declarative schemas, better than manual validation
    impact: Schema definition is single source of truth for portfolio structure
  - id: strict-false-coerce-true
    what: Set Pandera schema with strict=False, coerce=True
    why: Allow extra columns (flexibility), auto-coerce types before validation
    impact: Users can include additional columns without validation failure
metrics:
  duration: 2m 18s
  completed: 2026-01-30
---

# Phase 1 Plan 1: Package Structure and Validation

Pip-installable Python package with custom exceptions and Pandera validation schema.

## What Was Built

Established the foundational package structure for asset-optimization:

1. **Package Infrastructure**
   - Created src layout with `asset_optimization` package
   - Configured pyproject.toml with build system and dependencies
   - Package installable via `uv pip install -e .`
   - Version 0.1.0 accessible via import

2. **Exception Hierarchy**
   - `AssetOptimizationError` - base exception
   - `ValidationError` - with field, message, details attributes
   - `MissingFieldError` - for missing required columns
   - `DataQualityError` - for quality threshold failures
   - Structured error formatting with context

3. **Pandera Validation Schema**
   - `portfolio_schema` with required columns:
     - `asset_id`: string, unique, not null
     - `install_date`: timestamp, not null, not future
     - `asset_type`: string, not null
     - `material`: string, not null
   - Optional columns with range checks:
     - `diameter_mm`: nullable Int64, > 0
     - `length_m`: nullable float, > 0
     - `condition_score`: nullable float, 0-100 range
   - Validates against missing fields, future dates, duplicate IDs

## Decisions Made

**1. Src Layout for Package Structure**
- **Context:** Choosing between flat layout (simple) and src layout (recommended)
- **Decision:** Use src layout with package in `src/asset_optimization/`
- **Reasoning:** Prevents import issues during development, forces proper installation testing, PyPA recommendation
- **Impact:** All imports require package installation (editable mode for dev), cleaner separation

**2. Pandera for Schema Validation**
- **Context:** Need DataFrame validation (alternatives: manual checks, Great Expectations, Pydantic)
- **Decision:** Use Pandera with declarative schema definition
- **Reasoning:** Lightweight (vs GE), pandas-native API, detailed error reports, statistical testing support
- **Impact:** Schema is single source of truth, validation is declarative, easy to extend

**3. Schema Configuration (strict=False, coerce=True)**
- **Context:** How strict should schema be about extra columns and type coercion
- **Decision:** Set `strict=False` (allow extra columns), `coerce=True` (auto-coerce types)
- **Reasoning:** Flexibility for users to include additional columns, pandas compatibility
- **Impact:** Extra columns don't cause validation failure, type mismatches auto-coerce before validation

## Implementation Notes

**Package Structure:**
- Followed PEP 621 standard with `[project]` table in pyproject.toml
- Dependencies: pandas>=2.0.0, openpyxl>=3.1.0, pandera>=0.18.0
- Dev dependencies: pytest>=7.1.0
- setuptools as build backend (most widely documented, standard)

**Exception Design:**
- ValidationError stores structured details (field, message, details dict)
- Formatted messages include context: `"Validation failed for 'field': message (key=value)"`
- Inheritance hierarchy allows catching specific error types

**Schema Validation:**
- Pandera schema validates structure, types, and constraints in single pass
- Future date check uses `pd.Timestamp.now()` for install_date
- Unique constraint on asset_id prevents duplicates
- Range checks ensure positive values for physical measurements

**Verification Results:**
- ✓ Package installs via `uv pip install -e .`
- ✓ Version 0.1.0 accessible via import
- ✓ Exceptions importable from `asset_optimization`
- ✓ Schema validates correct DataFrames
- ✓ Schema rejects invalid data (missing fields, future dates, duplicates)

## Deviations from Plan

None - plan executed exactly as written.

## Blockers & Challenges

None encountered. Package structure and validation setup completed without issues.

## Next Phase Readiness

**Ready for:**
- Plan 01-02: Portfolio class implementation
- Plan 01-03: Quality metrics computation

**Provides:**
- Installable package structure
- Validation primitives (exceptions, schema)
- Foundation for all Phase 1 work

**Blockers:** None

## Testing Evidence

```python
# Valid DataFrame passes
valid_df = pd.DataFrame({
    'asset_id': ['PIPE-001', 'PIPE-002'],
    'install_date': pd.to_datetime(['2010-01-15', '2015-06-20']),
    'asset_type': ['pipe', 'pipe'],
    'material': ['PVC', 'Cast Iron'],
})
portfolio_schema.validate(valid_df)  # ✓ Passes

# Invalid data rejected
invalid_df = pd.DataFrame({
    'install_date': pd.to_datetime(['2010-01-15']),  # Missing asset_id
    'asset_type': ['pipe'],
    'material': ['PVC'],
})
portfolio_schema.validate(invalid_df)  # ✗ Raises SchemaError

# Future dates rejected
future_df = pd.DataFrame({
    'asset_id': ['PIPE-001'],
    'install_date': pd.to_datetime(['2030-01-15']),  # Future date
    'asset_type': ['pipe'],
    'material': ['PVC'],
})
portfolio_schema.validate(future_df)  # ✗ Raises SchemaError

# Duplicate IDs rejected
duplicate_df = pd.DataFrame({
    'asset_id': ['PIPE-001', 'PIPE-001'],  # Duplicate
    'install_date': pd.to_datetime(['2010-01-15', '2015-06-20']),
    'asset_type': ['pipe', 'pipe'],
    'material': ['PVC', 'Cast Iron'],
})
portfolio_schema.validate(duplicate_df)  # ✗ Raises SchemaError
```

## Links

**Related plans:**
- Next: 01-02 (Portfolio class with factory methods)
- Next: 01-03 (Quality metrics computation)

**Key files:**
- `pyproject.toml` - Package metadata and dependencies
- `src/asset_optimization/__init__.py` - Package entry point
- `src/asset_optimization/exceptions.py` - Custom exception hierarchy
- `src/asset_optimization/schema.py` - Pandera validation schema
