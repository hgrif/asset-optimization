# Feature Research

**Domain:** Asset Optimization/Simulation SDK for Infrastructure
**Researched:** 2026-01-29
**Confidence:** MEDIUM-HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Portfolio Data Loading** | Every asset tool needs to ingest existing data (CSV, Excel, database) | LOW | Pandas-based import; support common formats. Users have legacy data |
| **Asset State Representation** | Core data structure (age, type, condition, location) | LOW | Foundation for all features; must be queryable and filterable |
| **Deterioration Modeling** | Users expect statistical failure models (Weibull standard for infrastructure) | MEDIUM | Weibull 2P is baseline; shape/scale parameters per asset type |
| **Failure Rate Calculation** | Calculate probability of failure at current state | MEDIUM | Core to decision-making; must handle thousands of assets efficiently |
| **Multi-Timestep Simulation** | Consultants model 10-50 year horizons, not single snapshots | MEDIUM | Time loop with state updates; discrete annual timesteps typical |
| **Intervention Types** | DoNothing, Inspect, Repair, Replace — standard asset management actions | LOW | Enum or class hierarchy; each has cost and state effects |
| **Cost Tracking** | Total cost per scenario (capital, O&M) | LOW | Aggregation across assets and years; output to DataFrame |
| **Basic Optimization** | Select interventions within budget constraints | HIGH | Heuristic acceptable for v1; users expect "optimized" schedules |
| **Scenario Comparison** | Run multiple "what-if" scenarios and compare outcomes | MEDIUM | Users never run one scenario; need side-by-side comparison of cost/risk |
| **Tabular Export** | Output to CSV/Excel for reporting to stakeholders | LOW | Pandas `.to_csv()` / `.to_excel()` — non-negotiable for consultants |
| **Simple Visualization** | Basic plots (failure rates over time, cost breakdown) | MEDIUM | Matplotlib/Seaborn sufficient; consultants build custom dashboards later |
| **Documentation & Examples** | Clear API docs with Jupyter notebook examples | MEDIUM | Sphinx docs + example notebooks in repo; consultants learn by example |

### Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Pluggable Model Interface** | Users can swap Weibull for ML models (survival analysis, XGBoost) without rewriting simulation | MEDIUM | Abstract base class for deterioration models; huge for consultants building custom models |
| **Pluggable Optimizer Interface** | Start with heuristic, swap in MILP (OR-Tools, Gurobi) when needed | HIGH | Separation of concerns: simulation vs optimization logic; enables academic/commercial solvers |
| **Constraint Flexibility** | Support multiple constraint types (budget, crew capacity, regulatory deadlines) | MEDIUM | Users face complex real-world constraints; configurable constraint objects |
| **Intervention Chaining** | Model dependencies (must inspect before repair, can't replace frozen ground in winter) | MEDIUM | Domain-specific logic; differentiates from generic optimization tools |
| **Risk Metrics** | Calculate expected failures, consequence-weighted risk (not just cost) | MEDIUM | Infrastructure failures have social/safety costs; users value risk-adjusted optimization |
| **State Rollback / Branching** | Save simulation state, fork scenarios from midpoint | MEDIUM | Consultants want "what if we changed year 5 decisions?" without re-running years 1-4 |
| **Performance Profiling** | Built-in timing for data loading, model evaluation, optimization steps | LOW | Consultants optimize models over thousands of runs; performance visibility = trust |
| **Batch Scenario Execution** | Run 100 scenarios with parameter sweeps (e.g., budget from $1M-$10M) | MEDIUM | Enables sensitivity analysis; parallelize with multiprocessing |
| **Asset Grouping / Hierarchies** | Model portfolios as networks (pipes connect) or hierarchies (district → zone → asset) | HIGH | Water networks have topology; enables network-aware optimization (future) |
| **Intervention Schedule Export** | Generate implementation-ready work orders (asset ID, year, action, crew assignment) | LOW | Output format consultants hand to asset managers; increases perceived utility |
| **Python-Native API** | Pythonic design (context managers, method chaining, type hints) | LOW | Consultants are Python users; feels native vs clunky wrapper around C++ |
| **Deterministic by Default** | Reproducible results with random seed control | LOW | Critical for consultant reports; stochastic Monte Carlo comes later |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Web UI in v1** | "Users want dashboards" | Locks in opinionated workflow before validating SDK; slows iteration | Ship SDK first; consultants build custom UIs; add templated UI in v2 if pattern emerges |
| **Real-Time Data Integration** | "Connect to SCADA/IoT sensors" | Adds authentication, streaming, error handling complexity; most users have batch exports | Focus on batch CSV/Excel import; document integration patterns; build connectors in v2+ |
| **Multi-Asset Domains (v1)** | "Support pipes, roads, bridges, data centers" | Domain-specific logic (Weibull params, intervention types, constraints differ); dilutes focus | Water pipes for v1 validates architecture; abstract later with clear extension points |
| **Auto-Tuning Deterioration Models** | "Fit Weibull to historical failure data automatically" | Statistical modeling pitfalls (censored data, small samples); users want control over assumptions | Accept model parameters as input; document how to fit externally (statsmodels, reliability); v2 could add `.fit()` methods |
| **Built-In Geographic Mapping** | "Show assets on interactive map" | Dependencies (folium, geopandas), bloated SDK, most users have GIS tools | Export lat/lon with results; users join to GIS; document integration with QGIS/ArcGIS |
| **Multi-Objective Optimization** | "Minimize cost AND risk AND equity" | Pareto fronts, user-defined weights, complex UX; most users optimize one objective + constraints | Single-objective with risk/equity as constraints or penalty terms; document multi-objective patterns for advanced users |
| **Embedded Database** | "Store all scenarios in SQLite/Postgres" | Users want files (CSV, pickle) for version control and sharing; database = config overhead | Lightweight serialization (pickle, joblib, parquet); users manage storage |
| **Automatic Report Generation** | "Generate PDF reports with charts/tables" | Opinionated formatting, localization, branding; every consultant has house style | Export structured data (JSON, CSV); document report generation with Jinja2/Quarto |
| **Blockchain for Audit Trail** | "Immutable decision history" | Buzzword-driven; git + serialized files provide audit trail; blockchain = complexity with no clear benefit | Version control simulation inputs/outputs with git; document audit patterns |

## Feature Dependencies

```
Portfolio Data Loading
    └──requires──> Asset State Representation
                       └──requires──> Deterioration Modeling
                                          └──requires──> Failure Rate Calculation
                                                             └──requires──> Intervention Types
                                                                                └──requires──> Basic Optimization
                                                                                                   └──requires──> Multi-Timestep Simulation

Scenario Comparison ──enhances──> Multi-Timestep Simulation
State Rollback ──enhances──> Scenario Comparison
Batch Scenario Execution ──requires──> Scenario Comparison
Risk Metrics ──enhances──> Failure Rate Calculation
Constraint Flexibility ──enhances──> Basic Optimization
Pluggable Optimizer Interface ──enhances──> Basic Optimization
Pluggable Model Interface ──enhances──> Deterioration Modeling
Intervention Chaining ──requires──> Intervention Types

Tabular Export ──requires──> [any simulation output]
Simple Visualization ──requires──> [any simulation output]
```

### Dependency Notes

- **Core Pipeline**: Portfolio → State → Model → Rates → Interventions → Optimization → Simulation is linear dependency chain
- **Pluggable Interfaces**: Must be designed early (hard to retrofit); enable all advanced use cases
- **Scenario Comparison**: No dependencies but amplifies value of every other feature
- **Export/Viz**: Terminal features consuming outputs; implement last
- **State Rollback**: Requires careful state management architecture from day one

## MVP Definition

### Launch With (v1)

Minimum viable product — what's needed to validate the concept.

- [x] **Portfolio Data Loading** — Without data, SDK is useless
- [x] **Asset State Representation** — Core data structure
- [x] **Weibull Deterioration Model (hardcoded)** — Specific implementation, abstract later
- [x] **Failure Rate Calculation** — Core decision input
- [x] **4 Intervention Types** — DoNothing, Inspect, Repair, Replace
- [x] **Budget Constraint** — Most common constraint
- [x] **Greedy Heuristic Optimizer** — Simple, deterministic, "good enough"
- [x] **Multi-Timestep Simulation (10 years)** — Proves time-based modeling works
- [x] **Scenario Comparison (2-3 scenarios manually)** — Side-by-side cost/failure comparison
- [x] **CSV Export** — Intervention schedules + cost summary
- [x] **Basic Matplotlib Charts** — Cost over time, failures avoided
- [x] **Jupyter Notebook Examples** — 2-3 scenarios demonstrating API

**MVP Success Criteria**: Consultant can load water pipe data, run 10-year simulation with budget constraint, compare "do nothing" vs "optimized" scenarios, export results to CSV, present findings to client.

### Add After Validation (v1.x)

Features to add once core is working.

- [ ] **Pluggable Deterioration Model Interface** — Trigger: User requests custom model (e.g., Weibull 3P, log-normal)
- [ ] **Pluggable Optimizer Interface** — Trigger: User needs exact optimization (MILP) for high-stakes decisions
- [ ] **Multiple Constraint Types** — Trigger: User has crew capacity or regulatory constraints beyond budget
- [ ] **Risk Metrics** — Trigger: User values failure avoidance over cost minimization
- [ ] **State Rollback** — Trigger: User wants to branch scenarios mid-simulation
- [ ] **Batch Scenario Execution** — Trigger: User runs 50+ scenarios for sensitivity analysis
- [ ] **Excel Export** — Trigger: Client deliverable format (not internal analysis)
- [ ] **Performance Profiling Tools** — Trigger: User models 10K+ assets and needs bottleneck visibility
- [ ] **Intervention Chaining** — Trigger: User models seasonal constraints or inspection-before-repair rules
- [ ] **Asset Grouping** — Trigger: User needs to model network topology (connected pipes)

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] **Monte Carlo Simulation** — Why defer: Architecture complexity; most users start deterministic
- [ ] **ML-Based Models (survival analysis, gradient boosting)** — Why defer: Pluggable interface enables later; requires training data UX
- [ ] **Multi-Asset Domains (beyond water pipes)** — Why defer: Proves generalization after water pipe validation
- [ ] **Templated Web UI** — Why defer: Wait for common usage patterns to emerge from SDK users
- [ ] **Real-Time Data Connectors** — Why defer: Users currently work with batch exports
- [ ] **Auto-Tuning / Model Fitting** — Why defer: Statistical complexity; users want control initially
- [ ] **Multi-Objective Optimization (Pareto fronts)** — Why defer: Advanced use case; single-objective + constraints sufficient for most
- [ ] **Cloud Execution (AWS Lambda, Dask distributed)** — Why defer: Premature scaling; local execution fine for v1 portfolio sizes
- [] **Geographic Visualization (interactive maps)** — Why defer: Heavy dependencies; users have GIS tools

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Portfolio Data Loading | HIGH | LOW | P1 |
| Asset State Representation | HIGH | LOW | P1 |
| Deterioration Modeling (Weibull hardcoded) | HIGH | MEDIUM | P1 |
| Failure Rate Calculation | HIGH | MEDIUM | P1 |
| Intervention Types (4 basic) | HIGH | LOW | P1 |
| Basic Optimization (greedy heuristic) | HIGH | MEDIUM | P1 |
| Multi-Timestep Simulation | HIGH | MEDIUM | P1 |
| Scenario Comparison (manual) | HIGH | LOW | P1 |
| CSV Export | HIGH | LOW | P1 |
| Jupyter Examples | HIGH | MEDIUM | P1 |
| Basic Visualization | MEDIUM | MEDIUM | P1 |
| Pluggable Model Interface | HIGH | MEDIUM | P2 |
| Pluggable Optimizer Interface | HIGH | HIGH | P2 |
| Constraint Flexibility | HIGH | MEDIUM | P2 |
| Risk Metrics | MEDIUM | MEDIUM | P2 |
| State Rollback | MEDIUM | MEDIUM | P2 |
| Batch Scenario Execution | MEDIUM | LOW | P2 |
| Excel Export | MEDIUM | LOW | P2 |
| Performance Profiling | MEDIUM | LOW | P2 |
| Intervention Chaining | MEDIUM | MEDIUM | P2 |
| Asset Grouping | MEDIUM | HIGH | P2 |
| Monte Carlo Simulation | HIGH | HIGH | P3 |
| ML-Based Models | MEDIUM | HIGH | P3 |
| Multi-Asset Domains | HIGH | HIGH | P3 |
| Web UI | LOW | HIGH | P3 |
| Real-Time Integration | LOW | HIGH | P3 |
| Auto-Tuning Models | LOW | HIGH | P3 |
| Multi-Objective Optimization | LOW | HIGH | P3 |
| Geographic Visualization | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for launch (MVP)
- P2: Should have, add after validation (v1.x)
- P3: Nice to have, future consideration (v2+)

## Competitor Feature Analysis

| Feature | Enterprise Tools (InfoWater Pro, IBM Maximo) | Academic Tools (InfraRisk, smif) | Our Approach |
|---------|--------------|--------------|--------------|
| **Portfolio Management** | Full-featured GIS integration, database-backed, multi-user | CSV import, single-user scripts | Pandas-based import (CSV/Excel), lightweight, no DB required |
| **Deterioration Models** | Proprietary models, limited customization | Weibull, exponential, configurable | Weibull v1, pluggable interface for custom models v1.x |
| **Optimization** | Built-in MILP/genetic algorithms, black box | Greedy/heuristic, transparent | Greedy v1, pluggable optimizer (bring your own MILP) v1.x |
| **Multi-Timestep Simulation** | Yes, with hydraulic coupling (slow) | Yes, simplified physics | Yes, statistical deterioration (fast), no hydraulics v1 |
| **Scenario Comparison** | UI-driven dashboards, limited export | Manual scripting, no built-in comparison | Programmatic comparison, DataFrame output, Jupyter-friendly |
| **Constraint Handling** | Fixed constraint types (budget, crew) | Custom constraints in code | Budget v1, extensible constraint framework v1.x |
| **Risk Metrics** | Consequence modeling, monetary valuation | Basic failure counts | Expected failures v1, risk-weighted optimization v1.x |
| **Visualization** | Rich dashboards, proprietary | Matplotlib in notebooks | Matplotlib v1, extensible (users bring Plotly/Dash) |
| **Extensibility** | Closed APIs, plugin systems complex | Open source, fork required | Python SDK, pluggable interfaces, encourage extension |
| **Deployment** | Windows desktop apps, enterprise licensing | Python scripts, manual setup | `pip install`, pure Python, open source (or commercial) |
| **Documentation** | Manuals + training courses ($$$) | Academic papers + code comments | Sphinx docs + Jupyter tutorials (free) |
| **Performance** | Optimized C++, handles 100K+ assets | Python, slow on large portfolios | NumPy/Pandas, targets 10K assets efficiently, document scaling limits |

**Key Differentiators**:
1. **Pluggable Architecture**: Users bring their own models/optimizers (vs black box commercial or hardcoded academic)
2. **Consultant-Friendly**: Pythonic, Jupyter-first, lightweight (vs enterprise complexity or academic one-offs)
3. **Balance**: Middle ground between proprietary full-featured tools and academic proof-of-concepts

## Sources

**Asset Management & Optimization Features:**
- [Top Asset Management Software Features Businesses Can't Ignore in 2026](https://www.webkorps.com/blog/top-asset-management-software-features/)
- [Essential asset management software features for 2026](https://monday.com/blog/service/asset-management-software-features/)
- [IBM Maximo Asset Performance Management](https://www.ibm.com/products/maximo/asset-performance-management)

**Infrastructure Simulation Tools:**
- [InfraRisk: An Open-Source Simulation Platform for Asset-Level Resilience Analysis](https://arxiv.org/abs/2205.04717)
- [A Software Framework for the Integration of Infrastructure (smif)](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.265)
- [Eclypse: a Python Framework for Simulation and Emulation](https://arxiv.org/html/2501.17126v1)

**Water Infrastructure Modeling:**
- [Autodesk InfoWater Pro 2026 Features](https://www.autodesk.com/products/infowater-pro/features)
- [InfoWater Pro 2026: New Leak Detection & Modeling Tools](https://microcad3d.com/infowater-pro-2026-leakage-modeling/)
- [MIKE+ Water Distribution Modelling Software](https://www.dhigroup.com/technologies/mikepoweredbydhi/mikeplus-water-distribution)

**Deterioration Modeling:**
- [Weibull Distributions primer for Asset Modelling](https://www.modla.co/blog/weibull-distributions-in-asset-modelling)
- [Using Weibull Models to Analyze Asset Failures in IBM Maximo](https://maximomastery.com/using-weibull-models-to-analyze-asset-failures-in-ibm-maximo/)
- [Simplified Deterioration Modeling for Highway Sign Support Systems](https://www.mdpi.com/2673-4591/36/1/44)

**Optimization & Constraints:**
- [Optimization (scipy.optimize) — SciPy v1.17.0 Manual](https://docs.scipy.org/doc/scipy/tutorial/optimize.html)
- [Get Started with OR-Tools for Python](https://developers.google.com/optimization/introduction/python)

**Scenario Planning & Analysis:**
- [pyam: Analysis & visualization of energy & climate scenarios](https://github.com/IAMconsortium/pyam)
- [Top 6 Scenario Planning Software & Tools in 2026](https://productive.io/blog/scenario-planning-software/)

**API Design Best Practices:**
- [General Guidelines: API Design | Azure SDKs](https://azure.github.io/azure-sdk/general_design.html)
- [API Design Best Practices | Secure API Architecture 2026](https://eluminoustechnologies.com/blog/api-design/)

**Multi-Timestep Simulation:**
- [SimPy Documentation](https://simpy.readthedocs.io/)
- [Modeling and Simulation in Python](https://greenteapress.com/modsimpy/ModSimPy3.pdf)

**Maintenance Planning & Optimization:**
- [Asset Lifecycle Management Checklist 2026 Guide](https://www.assetinfinity.com/blog/asset-lifecycle-management-checklist-2026)
- [Asset Management Predictions for 2026](https://www.clickmaint.com/blog/asset-management-predictions)
- [CMMS Roadmap 2026: Reactive to proactive maintenance](https://www.clickmaint.com/blog/cmms-roadmap-2026)

**Infrastructure SDK Common Mistakes:**
- [Why Infrastructure as Code Feels Broken in 2025](https://weirdion.com/posts/2025-07-01-why-does-iac-feels-broken-in-2025/)
- [Avoid These 10 Common Mistakes When Implementing Infrastructure as Code](https://moldstud.com/articles/p-top-10-mistakes-to-avoid-in-infrastructure-as-code)

**Data Interoperability:**
- [Simulation Interoperability Standards Organization](https://www.sisostandards.org/)
- [pandas — Python Data Analysis Library](https://pandas.pydata.org/)

---
*Feature research for: Asset Optimization/Simulation SDK for Infrastructure*
*Researched: 2026-01-29*
