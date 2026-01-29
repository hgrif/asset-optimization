# Stack Research

**Domain:** Asset simulation and optimization Python SDK
**Researched:** 2026-01-29
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| NumPy | 2.4.1 | N-dimensional array processing for asset data | Industry standard for numerical computing. Free-threaded Python support in 2.x enables better performance. Essential foundation for vectorized operations on thousands of assets. |
| Pandas | 3.0.0 | Time-series data management and portfolio analysis | Released Jan 2026 with Apache Arrow support for 30-60% memory reduction. Familiar API, massive ecosystem. Best for <1M assets with extensive analytical functions. |
| SciPy | 1.17.0 | Statistical distributions (Weibull, Poisson) and optimization | Reference implementation for scientific algorithms. `scipy.stats.weibull_min` and `scipy.stats.poisson` provide battle-tested distribution functions. Includes optimization algorithms (though not MILP-specific). |
| Pydantic | 2.12.5 | Data validation and SDK input/output schema | Core validation written in Rust, 10-100x faster than pure Python. Essential for SDK data contracts. Type-safe configuration and API boundaries. Industry standard for Python SDKs. |

### Data Processing (Choose Based on Scale)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Pandas | 3.0.0 | Standard dataframe operations | <1M assets, prototype development, extensive ecosystem integration needed |
| Polars | 1.37.1 | High-performance dataframe operations | >1M assets, 3-10x faster than Pandas, 30-60% less memory. Use for production ETL and large portfolio analysis. |
| PyArrow | 23.0.0 | Zero-copy I/O and columnar data | Parquet file I/O (4x size reduction vs CSV), 10-100x faster reads. Always use for data persistence. |

### Optimization

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| PuLP | 3.3.0 | MILP modeling and solving | **Recommended for this SDK**. Pure Python, solver-agnostic (CBC, GLPK, Gurobi), lightweight. Perfect for budget-constrained intervention optimization. |
| OR-Tools | 9.11+ | Google's optimization suite | Alternative for complex routing/scheduling. Heavier dependency but includes CP-SAT solver. |
| Gurobi | 12.0+ | Commercial MILP solver | Optional premium solver. 10-100x faster on large problems but requires expensive license (academia-free). PuLP can interface with it. |

### Simulation

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| SimPy | 4.1.1 | Discrete-event simulation | **Recommended**. Process-based DES for modeling asset lifecycles and failure events. Supports both real-time and fast-forward simulation. |
| Mesa | 3.0+ | Agent-based modeling with discrete events | Alternative if agent-based approach needed. Mesa 3 (2025) adds DiscreteEventSimulator for hybrid ABM/DES. Likely overkill for asset portfolio. |

### Reliability Analysis

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| SciPy (scipy.stats) | 1.17.0 | Basic Weibull/Poisson distributions | Core statistical functions sufficient for most needs |
| reliability | 0.9.0 | Specialized reliability engineering | Optional. Adds `Fit_Weibull_2P`, right-censored data handling, MRR/MLE fitting. Only if advanced reliability analysis needed. |

### Development Tools

| Tool | Version | Purpose | Notes |
|------|---------|---------|-------|
| Poetry | 2.3.1 | Dependency management and packaging | Modern standard for Python SDKs. Lock files ensure reproducible builds. PEP 621 compliant. |
| pytest | 9.0.2 | Testing framework | Industry standard. Fast, plugin ecosystem (pytest-cov, pytest-benchmark). |
| Hypothesis | 6.151.4 | Property-based testing | Essential for SDK testing. Finds edge cases in distribution/optimization code. Seamless pytest integration. |
| mypy | 1.19.1 | Type checking (CI/CD) | Reference implementation. Run in CI for strict type enforcement. |
| Pyright | 1.1.400+ | Type checking (IDE) | 3-5x faster than mypy, powers VS Code Pylance. Use for development, mypy for CI. |

### Data I/O

| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| PyArrow | 23.0.0 | Parquet read/write | **Always use Parquet for persistence**. 4x smaller than CSV, 5-10x faster reads with column pruning. |
| openpyxl | 3.1+ | Excel I/O | Only if Excel compatibility required for consultants. Significantly slower than Parquet. |
| pandas/polars | - | CSV fallback | Use for legacy data import only. Always convert to Parquet internally. |

## Installation

```bash
# Core dependencies (SDK production)
pip install numpy>=2.4.1 pandas>=3.0.0 scipy>=1.17.0 pydantic>=2.12.5

# Optimization
pip install pulp>=3.3.0

# Simulation
pip install simpy>=4.1.2

# High-performance data (for large portfolios)
pip install polars>=1.37.1 pyarrow>=23.0.0

# Development dependencies
pip install -D poetry>=2.3.1 pytest>=9.0.2 hypothesis>=6.151.4 mypy>=1.19.1

# Optional: Advanced reliability analysis
pip install reliability>=0.9.0

# Optional: Premium solver (if budget allows)
# pip install gurobipy>=12.0  # Requires license
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Pandas | Polars | **Switch to Polars** when portfolio size >1M assets or Pandas becomes performance bottleneck (3-10x speedup) |
| Pandas | Dask | Only if distributed computing across clusters needed. Adds complexity. Polars single-machine performance often sufficient. |
| PuLP | Pyomo | PuLP simpler and sufficient for MILP. Pyomo better for complex nonlinear models not needed here. |
| PuLP | cvxpy | cvxpy better for convex optimization. This SDK needs MILP for discrete intervention decisions. |
| SimPy | Custom timestep loop | SimPy only if event-based logic needed. For fixed timesteps, NumPy/Pandas vectorization simpler. |
| Poetry | setuptools + pip-tools | Legacy projects. Poetry provides superior dependency resolution and lock files. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Pandas <2.0 | Missing Arrow integration, 30-60% more memory usage | Pandas >=3.0.0 with Arrow backend |
| NumPy <2.0 | Lacks free-threading support, slower | NumPy >=2.4.1 |
| CSV for storage | 4x larger files, 5-10x slower reads, no column pruning | Parquet via PyArrow |
| Excel for large datasets | Extremely slow (100-1000x vs Parquet), row limits (1M rows) | Parquet (convert Excel to Parquet on import) |
| pickle for data | Not portable, version-dependent, security risk | Parquet (portable, fast, type-safe) |
| CPLEX/Gurobi initially | Expensive licenses, overkill for prototyping | PuLP with open-source solvers (CBC/GLPK). Add Gurobi later if needed. |
| Anaconda | License restrictions for commercial use, slow resolver | Poetry + pip for dependency management |

## Stack Patterns by Variant

**For small-scale prototypes (<10K assets):**
- Use Pandas for data processing
- Use basic SciPy distributions
- Fixed timestep loops instead of SimPy
- CSV I/O acceptable for iteration speed

**For production SDK (10K-1M assets):**
- Use Pandas with Arrow backend
- Use PuLP + CBC solver
- Use SimPy for event-based simulation
- Use Parquet for all persistence

**For large-scale deployments (>1M assets):**
- **Switch to Polars** for 3-10x speedup and 30-60% memory reduction
- Consider Gurobi license if optimization becomes bottleneck
- Use partitioned Parquet files for distributed storage
- Profile and optimize hot paths with Numba JIT compilation

**For consultant-facing SDK:**
- Support Excel import (convert to Parquet internally)
- Export results to Excel/CSV for compatibility
- Use Pydantic for clear error messages on malformed input
- Provide example datasets in multiple formats

## Version Compatibility

| Core Stack | Compatible Versions | Notes |
|-----------|---------------------|-------|
| NumPy 2.4.1 | Pandas 3.0.0 | Pandas 3.0 requires NumPy >=1.26.4, compatible with 2.x |
| NumPy 2.4.1 | SciPy 1.17.0 | SciPy 1.17 requires NumPy >=1.26.4, compatible with 2.x |
| NumPy 2.4.1 | Polars 1.37.1 | Polars has optional NumPy backend, compatible |
| Pydantic 2.12.5 | Python >=3.9 | Requires Python 3.9+. Recommend 3.11+ for performance. |
| Pandas 3.0.0 | Python >=3.11 | **Breaking change**: Pandas 3.0 drops Python <3.11 support |
| Poetry 2.3.1 | Python >=3.10 | Poetry requires Python 3.10+. Use 3.11+ for consistency. |

**Recommended Python version:** 3.11 or 3.12
- Pandas 3.0 requires >=3.11
- Python 3.11+ has significant performance improvements (~25% faster)
- Python 3.13 has free-threading but library support still maturing

## Performance Considerations

### Memory Management (Thousands of Assets)

**Problem:** Pandas requires 5-10x dataset size in RAM for operations.

**Solutions:**
1. **Use Polars for >1M assets** - 2-4x dataset size requirement vs Pandas 5-10x
2. **Use Pandas categorical dtypes** - 10x memory reduction for repeated strings
3. **Downcast numeric types** - int64 → int32, float64 → float32 where possible
4. **Enable Pandas Arrow backend** - 30-60% memory reduction
5. **Process in chunks** - Use chunking for portfolio >RAM size

### Computational Performance

**Problem:** Multi-timestep simulation over thousands of assets can be slow.

**Solutions:**
1. **Vectorize with NumPy** - 10-100x faster than Python loops
2. **Use Polars lazy evaluation** - Optimizes query plans automatically
3. **Profile hot paths** - Use `pytest-benchmark` and `line_profiler`
4. **JIT compile bottlenecks** - Use Numba `@jit` decorator for numerical loops
5. **Parallelize simulations** - Use `multiprocessing` for independent scenarios

### I/O Performance

**Problem:** Loading/saving large portfolios can bottleneck workflows.

**Solutions:**
1. **Always use Parquet** - 5-10x faster than CSV, 4x smaller files
2. **Use PyArrow engine** - Faster than fastparquet
3. **Enable compression** - Snappy (fast) or ZSTD (smaller)
4. **Use column pruning** - Only read needed columns
5. **Partition large datasets** - Split by year, region, or asset type

## Confidence Assessment

| Component | Confidence | Source | Notes |
|-----------|------------|--------|-------|
| NumPy/Pandas/SciPy versions | **HIGH** | PyPI official pages (Jan 2026) | Verified current stable releases |
| Pydantic | **HIGH** | PyPI + recent docs | v2.12.5 stable, v3 announced but not released yet |
| Polars performance claims | **HIGH** | Multiple 2025 benchmarks + official docs | 3-10x speedup consistently reported |
| PuLP for MILP | **HIGH** | Community consensus + official docs | Standard choice for open-source MILP in Python |
| PyArrow/Parquet I/O | **HIGH** | Apache docs + benchmarks | 5-10x speedup vs CSV well-documented |
| SimPy for DES | **MEDIUM** | Official docs + community usage | Mature library but limited recent innovation |
| Mesa 3 discrete events | **MEDIUM** | JOSS paper (2025) + docs | New feature, experimental status |
| Hypothesis best practices | **MEDIUM** | Recent tutorials + docs | Good library but testing strategies domain-specific |
| Poetry vs alternatives | **MEDIUM** | 2026 blog post + survey data | Good consensus but packaging ecosystem still evolving |
| mypy vs Pyright | **MEDIUM** | Multiple 2025 comparisons | Both viable, team preference matters |

## Rationale: Key Technology Decisions

### Why NumPy/Pandas/SciPy (Not Alternative Scientific Stack)?

**Decision:** Standard scientific Python stack (NumPy/Pandas/SciPy) over alternatives (JAX, Julia, R).

**Rationale:**
- **Ecosystem maturity:** Pandas has 15+ years of battle-testing, extensive documentation
- **Consultant familiarity:** Target users (infrastructure consultants) likely know Pandas
- **Library compatibility:** Optimization (PuLP), reliability, visualization all Pandas-native
- **JAX overkill:** GPU acceleration not needed for thousands (not millions) of assets
- **Julia/R friction:** Python SDK calling Julia/R adds deployment complexity

**Confidence:** HIGH - This is a Python SDK, standard stack is the right choice.

### Why Polars as Performance Escape Hatch (Not Dask)?

**Decision:** Recommend Polars for large portfolios over Dask or PySpark.

**Rationale:**
- **Single-machine performance:** 3-10x Pandas speedup without cluster complexity
- **Pandas-like API:** Low migration cost (though not 100% compatible)
- **Memory efficiency:** 30-60% reduction critical for >1M asset portfolios
- **Dask overhead:** Dask better for multi-machine, adds scheduler complexity
- **PySpark overkill:** Requires JVM, cluster setup. Polars single-binary simpler.

**Confidence:** HIGH - Multiple 2025 benchmarks + community momentum.

### Why PuLP for Optimization (Not Gurobi/Pyomo)?

**Decision:** PuLP with open-source solvers (CBC/GLPK) as primary, Gurobi optional.

**Rationale:**
- **Cost:** Gurobi requires expensive commercial license (~$20K+/year)
- **Simplicity:** PuLP pure Python, lightweight, easy to install
- **Solver agnostic:** PuLP can switch to Gurobi later if needed (same model code)
- **Good enough:** CBC/GLPK handle small-medium MILP (hundreds of assets per timestep)
- **Pyomo complexity:** Pyomo better for nonlinear. This SDK needs MILP for discrete decisions.

**Confidence:** HIGH - PuLP is standard for open-source MILP, Gurobi upgrade path exists.

### Why Parquet (Not CSV/Excel/HDF5)?

**Decision:** Parquet via PyArrow as primary persistence format.

**Rationale:**
- **Performance:** 5-10x faster reads than CSV, 4x smaller files
- **Columnar:** Only read needed columns (critical for wide asset tables)
- **Typed:** Schema enforcement prevents data corruption
- **Portable:** Cross-language (R, Julia, Spark can read)
- **Compression:** Snappy/ZSTD built-in
- **HDF5 vs Parquet:** HDF5 good for arrays, Parquet better for dataframes

**Confidence:** HIGH - Industry standard for analytics data (2025).

### Why Pydantic (Not dataclasses/attrs)?

**Decision:** Pydantic for SDK data validation and schemas.

**Rationale:**
- **Validation:** Automatic runtime validation (dataclasses don't validate)
- **Performance:** Rust core, 10-100x faster than pure Python validation
- **JSON schema:** Auto-generate API docs and client libraries
- **Error messages:** Clear validation errors for consultant users
- **Industry standard:** Most Python SDKs use Pydantic (FastAPI, Prefect, etc.)

**Confidence:** HIGH - Pydantic is the obvious choice for SDK data contracts.

### Why SimPy for Simulation (Not Custom Loop)?

**Decision:** SimPy for discrete-event simulation (with caveat).

**Rationale:**
- **When to use:** If failure events are stochastic and event-driven (random Weibull failures)
- **When NOT to use:** If fixed timesteps suffice (annual evaluations), NumPy loops simpler
- **SimPy advantage:** Process-based modeling matches conceptual "asset lifecycle" model
- **SimPy lightweight:** Pure Python, minimal dependency

**Caveat:** For simple "advance time → evaluate → optimize" loops, SimPy may be overkill. Start with NumPy vectorized timesteps, add SimPy if event-based logic emerges.

**Confidence:** MEDIUM - Depends on final simulation architecture (fixed timestep vs event-driven).

### Why Poetry (Not setuptools/pip-tools)?

**Decision:** Poetry for packaging and dependency management.

**Rationale:**
- **Lock files:** Reproducible builds critical for SDK users
- **Dependency resolution:** Poetry resolver better than pip
- **Modern:** PEP 517/518/621 compliant, future-proof
- **Developer experience:** Single `pyproject.toml`, no `setup.py`
- **SDK best practice:** Modern Python SDKs overwhelmingly use Poetry

**Confidence:** HIGH - Industry momentum strongly toward Poetry for new projects.

### Why pytest + Hypothesis (Not unittest)?

**Decision:** pytest as test framework, Hypothesis for property-based testing.

**Rationale:**
- **pytest standard:** Cleaner syntax than unittest, massive plugin ecosystem
- **Hypothesis critical:** Statistical/optimization code has edge cases (division by zero, negative values, etc.)
- **SDK quality:** Property-based tests catch bugs example-based tests miss
- **Integration:** Hypothesis seamless with pytest

**Confidence:** HIGH - pytest + Hypothesis is best practice for scientific Python libraries.

## Sources

**Version Verification (HIGH Confidence):**
- [NumPy 2.4.1 - PyPI](https://pypi.org/project/numpy/) - January 2026
- [Pandas 3.0.0 - PyPI](https://pypi.org/project/pandas/) - January 2026
- [SciPy 1.17.0 - PyPI](https://pypi.org/project/scipy/) - January 2026
- [Pydantic 2.12.5 - PyPI](https://pypi.org/project/pydantic/) - November 2025
- [Polars 1.37.1 - PyPI](https://pypi.org/project/polars/) - January 2026
- [PyArrow 23.0.0 - PyPI](https://pypi.org/project/pyarrow/) - January 2026
- [PuLP 3.3.0 - PyPI](https://pypi.org/project/PuLP/) - September 2025
- [pytest 9.0.2 - PyPI](https://pypi.org/project/pytest/) - December 2025
- [Hypothesis 6.151.4 - PyPI](https://pypi.org/project/hypothesis/) - January 2026
- [Poetry 2.3.1 - PyPI](https://pypi.org/project/poetry/) - January 2026
- [mypy 1.19.1 - PyPI](https://pypi.org/project/mypy/) - December 2025

**Performance Benchmarks (HIGH Confidence):**
- [Polars vs Pandas Performance](https://www.shuttle.dev/blog/2025/09/24/pandas-vs-polars) - Sept 2025
- [PyArrow High-Performance Data Processing](https://www.pythoncentral.io/pyarrow-high-performance-data-processing/) - 2025
- [Pandas Scaling Documentation](https://pandas.pydata.org/docs/user_guide/scale.html) - Official docs

**Optimization Libraries (HIGH Confidence):**
- [MIP Solvers Guide: PuLP, CPLEX, Gurobi, OR-Tools, Pyomo](https://medium.com/operations-research-bit/mip-solvers-unleashed-a-beginners-guide-to-pulp-cplex-gurobi-google-or-tools-and-pyomo-0150d4bd3999) - Medium

**Simulation (MEDIUM Confidence):**
- [SimPy Documentation](https://simpy.readthedocs.io/) - Official docs
- [Mesa 3: Agent-based modeling with Python in 2025](https://joss.theoj.org/papers/10.21105/joss.07668) - JOSS 2025

**Reliability Analysis (HIGH Confidence):**
- [SciPy Weibull Distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html) - Official docs
- [reliability package documentation](https://reliability.readthedocs.io/) - Official docs

**Data Validation (HIGH Confidence):**
- [Pydantic v3 Announcement](https://codemagnet.in/2025/12/15/pydantic-v3-the-new-standard-for-data-validation-in-python-why-everything-changed-in-2025/) - Dec 2025
- [Pydantic Best Practices](https://medium.com/algomart/working-with-pydantic-v2-the-best-practices-i-wish-i-had-known-earlier-83da3aa4d17a) - Dec 2025

**Packaging (MEDIUM Confidence):**
- [Python Packaging Best Practices 2026](https://dasroot.net/posts/2026/01/python-packaging-best-practices-setuptools-poetry-hatch/) - Jan 2026

**Type Checking (MEDIUM Confidence):**
- [mypy vs Pyright Performance Battle](https://medium.com/@asma.shaikh_19478/python-type-checking-mypy-vs-pyright-performance-battle-fce38c8cb874) - Nov 2025
- [Pyright vs Mypy Comparison](https://pyseek.com/2025/05/pyright-vs-mypy-static-type-checking-in-python/) - May 2025

---
*Stack research for: Asset simulation and optimization Python SDK*
*Researched: 2026-01-29*
