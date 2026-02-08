# Physical Asset Optimization Domains and Use Cases

## Executive Summary
Physical asset optimization across infrastructure sectors is fundamentally the same decision problem: allocate scarce budget, crews, and outage windows across a portfolio of assets under uncertainty to maximize service reliability and safety at minimum lifecycle cost. The domain differences are mostly in failure consequences, observability, network effects, and intervention lead times.

A unified software framework can therefore model each domain as: (1) assets and hierarchies, (2) condition and risk evolution models, (3) intervention actions with costs/effects, (4) network/system coupling, and (5) decision policies under constraints.

## Top-Down View

### 1. Shared Decision Pattern
All domains can be framed as:
- State estimation: what condition/risk is each asset in now?
- Forecasting: how will state evolve if we do nothing?
- Intervention impact: what changes if we inspect/repair/replace/operate differently?
- Portfolio optimization: what action set gives best system outcomes within budget and operational constraints?
- Re-planning: how does the plan update as new data arrives?

### 2. Why Domains Differ
Key differentiators that affect modeling choices:
- Failure dynamics: gradual deterioration (roads) vs abrupt failures (pumps, transformers, UPS batteries).
- Network coupling: weak in isolated assets, strong in water/power/transport networks.
- Consequence asymmetry: some failures are expensive but tolerable; others are safety- or service-critical.
- Sensing maturity: dense telemetry (data centers, power grids) vs sparse inspections (roads, bridges).

## Detailed Domain and Use-Case Catalog

### Roads, Bridges, and Tunnels
Typical assets:
- Pavement segments, bridges, culverts, retaining walls, tunnels.

Common decisions:
- Preventive maintenance vs rehabilitation vs full replacement.
- Corridor-level sequencing to minimize user disruption.

Representative use cases:
- Optimize treatment timing for pavement sections to minimize lifecycle cost while maintaining service level.
- Bundle nearby interventions to reduce mobilization and traffic disruption costs.
- Prioritize bridge rehabilitation under safety, resilience, and budget constraints.

Modeling notes:
- Strong use of condition states and deterioration curves.
- Spatial and network detour impacts matter for consequence modeling.

### Water Distribution, Stormwater, and Wastewater
Typical assets:
- Water mains, valves, pumps, storage tanks, storm drains, culverts, lift stations.

Common decisions:
- Replace/line pipes, replace pumps, add redundancy, perform targeted inspections.

Representative use cases:
- Prioritize pump replacements to reduce flood probability in rainfall events.
- Pipe replacement planning balancing break risk, leakage, and disruption.
- Stormwater bottleneck upgrades based on event-based hydraulic risk.

Modeling notes:
- Hydraulic network effects are central (pressure/flow dependencies).
- Event-driven risk (e.g., storms) often dominates expected-loss calculations.

### Electricity Transmission and Distribution
Typical assets:
- Transformers, breakers, cables, overhead lines, pylons, substations, protection devices.

Common decisions:
- Asset renewal and uprating; condition-based maintenance; vegetation and hardening programs.

Representative use cases:
- Risk-based transformer replacement under reliability and outage constraints.
- Grid hardening investments for extreme weather resilience.
- Portfolio planning that co-optimizes reliability metrics and capex/opex.

Modeling notes:
- Power flow and contingency constraints are strict.
- Reliability standards and outage penalties drive objective terms.

### Data Centers and Digital Infrastructure
Typical assets:
- UPS systems, chillers, CRAC/CRAH units, switchgear, generators, PDUs, batteries.

Common decisions:
- Preventive replacement intervals, redundancy strategy, load rebalancing, maintenance windows.

Representative use cases:
- Optimize UPS battery replacement policy to reduce unplanned downtime risk.
- Joint optimization of cooling equipment maintenance and energy efficiency.
- Capacity-expansion timing under demand growth uncertainty.

Modeling notes:
- High-frequency sensor data enables predictive maintenance.
- Reliability block diagrams and critical path analysis are useful abstractions.

### Rail, Metro, and Transit Systems
Typical assets:
- Track segments, points/switches, rolling stock components, signaling assets.

Common decisions:
- Maintenance possession scheduling; renewal timing; fleet overhaul strategy.

Representative use cases:
- Optimize track tamping/renewal cycles with possession windows.
- Prioritize switch replacements based on delay and safety risk.

Modeling notes:
- Timetable and access-window constraints are often as important as budget.
- Delay cost is a major consequence term.

### Ports, Airports, and Logistics Hubs
Typical assets:
- Runways, aprons, jet bridges, cranes, berths, pavement, baggage systems.

Common decisions:
- Renewal scheduling with minimal operational disruption.

Representative use cases:
- Stage runway rehabilitation over seasons to control capacity impact.
- Crane replacement prioritization based on reliability and throughput impact.

Modeling notes:
- Throughput/queueing impacts should be modeled with operational KPIs.

### Buildings and Campuses (Hospitals, Universities, Municipal Estates)
Typical assets:
- HVAC, roofing, elevators, fire suppression, electrical systems.

Common decisions:
- Deferred maintenance planning and retrofit programs.

Representative use cases:
- Multi-year capital plan balancing backlog reduction and criticality.
- Energy-retrofit timing integrated with asset renewal cycles.

Modeling notes:
- Mixed objective space: reliability, compliance, occupant comfort, energy.

### Industrial Facilities (Process Plants, Mining, Manufacturing)
Typical assets:
- Rotating equipment, conveyors, pumps/compressors, instrumentation.

Common decisions:
- Condition-based maintenance and spare strategy.

Representative use cases:
- Optimize maintenance intervals by production loss risk.
- Replacement planning for high-consequence single-point failures.

Modeling notes:
- Production-loss economics and spare lead times dominate policy quality.

## Cross-Domain Decision Archetypes
Across domains, most practical optimization programs reduce to one or more of these archetypes:
- Risk-based renewal ranking: choose which assets to replace this year.
- Intervention timing: choose when to intervene on each asset.
- Route/network-aware scheduling: sequence jobs considering access and network effects.
- Resilience investment: allocate budget to reduce tail risk from extreme events.
- Inspection planning: choose where to collect more information before committing capex.

## Data Inputs Needed by Nearly Every Domain
- Asset registry and hierarchy (asset -> subsystem -> system).
- Condition history and inspections.
- Failure/outage/incident history.
- Cost data (capex, opex, disruption, societal/safety proxy costs).
- Spatial and network topology.
- Exogenous covariates (weather, load, demand growth, soil/traffic).

## Implications for a Unified Framework
A single framework should not be domain-specific at the core. It should expose domain plugins for:
- Topology simulators (hydraulic/power/transport).
- Asset-type-specific deterioration/failure models.
- Domain consequence functions.
- Intervention libraries with effect priors.

The core engine can remain domain-agnostic if it standardizes:
- State representation.
- Uncertainty representation.
- Objective/constraint expression.
- Optimization and policy evaluation loop.

## Selected References
- [FHWA Pavement Management](https://www.fhwa.dot.gov/pavement/management/)
- [EPA Water Utility Asset Management](https://www.epa.gov/waterfinancecenter/asset-management-water-and-wastewater-utilities)
- [ISO 55000 Asset Management Family](https://www.iso.org/standard/55088.html)
