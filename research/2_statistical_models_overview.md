# Statistical Models for Physical Asset Optimization

## Executive Summary
No single statistical model is sufficient for physical asset optimization. In practice, strong solutions combine three model layers: (1) asset-level deterioration/failure models, (2) system/network impact models, and (3) decision/optimization models. The right mix depends on data density, network coupling, and decision horizon.

For a unified framework, treat models as composable modules with a shared contract: estimate risk and uncertainty for assets and systems under candidate interventions.

## Top-Down View

### 1. Modeling Stack
- Layer A: asset dynamics (failure/degradation/remaining life).
- Layer B: system propagation (how local failures affect network service).
- Layer C: decision model (which intervention set optimizes objective under constraints).

### 2. Core Tradeoff
- Simpler models are easier to calibrate and explain but can miss interactions.
- Richer hierarchical/network models capture reality better but require more data and computation.

## Detailed Model Families

### 1. Survival and Hazard Models
What they do:
- Model time-to-failure or hazard as a function of age, condition, and covariates.

Typical formulations:
- Parametric Weibull/log-logistic survival models.
- Semi-parametric Cox proportional hazards.

Best-fit domains/use cases:
- Pumps, transformers, UPS batteries, bridge components.
- Replace-now vs wait-one-year decisions.

Strengths:
- Interpretable hazard ratios.
- Works with censored data.

Limitations:
- Limited direct representation of network effects unless coupled with other models.

### 2. Degradation-State Models (Markov / Hidden Markov / State-Space)
What they do:
- Model transitions between condition states or latent degradation levels.

Typical formulations:
- Discrete-time Markov chains for condition ratings.
- Hidden Markov models when true state is not directly observed.
- Bayesian state-space models for noisy sensor streams.

Best-fit domains/use cases:
- Pavements, bridges, rail track, buildings.
- Condition-based maintenance planning over multi-year horizons.

Strengths:
- Natural for inspection-driven asset management.
- Handles incomplete observability.

Limitations:
- Transition probabilities may drift when operations/environment change.

### 3. Hierarchical (Multilevel) Bayesian Models
What they do:
- Pool information across assets/subsystems while allowing asset-type or location-specific variation.

Typical formulations:
- Asset-level parameters drawn from group-level distributions.
- Hierarchical priors by material, manufacturer, soil type, climate zone.

Best-fit domains/use cases:
- Pipe networks with sparse failures per segment.
- Multi-site data centers with similar equipment families.

Strengths:
- Robust with sparse, imbalanced data.
- Provides full uncertainty intervals for downstream optimization.

Limitations:
- Higher computational and modeling complexity.

### 4. Spatiotemporal Models
What they do:
- Capture spatial correlation and temporal evolution in risk.

Typical formulations:
- Gaussian process or autoregressive spatiotemporal models.
- Bayesian hierarchical spatial effects.

Best-fit domains/use cases:
- Road deterioration by corridor; flood risk across catchments; vegetation-related power-line failures.

Strengths:
- Improves predictions where neighboring assets share environment/load.

Limitations:
- Can become computationally heavy for large networks.

### 5. Network Reliability and Flow Models
What they do:
- Quantify system-level service impact from component failures/interventions.

Typical formulations:
- Graph connectivity and flow-feasibility checks.
- Hydraulic simulation (water) and load-flow/contingency analysis (power).

Best-fit domains/use cases:
- Water pump replacement to reduce flood or pressure-loss events.
- Substation and line investments under N-1 reliability criteria.
- Road closure impacts on accessibility/travel time.

Strengths:
- Explicitly models cascading/system effects.

Limitations:
- Requires topology quality and domain simulators.

### 6. Extreme-Event and Tail-Risk Models
What they do:
- Focus on low-frequency, high-consequence outcomes.

Typical formulations:
- Peaks-over-threshold / generalized Pareto methods.
- Scenario-based stress testing with weather or demand extremes.

Best-fit domains/use cases:
- Flood defense and stormwater pumping; grid hardening; coastal infrastructure.

Strengths:
- Better alignment with resilience objectives.

Limitations:
- Data scarcity in tails; high scenario uncertainty.

### 7. Causal/Uplift Models for Intervention Effectiveness
What they do:
- Estimate treatment effects: what risk reduction is attributable to intervention.

Typical formulations:
- Causal forests / doubly robust learners / Bayesian structural models.

Best-fit domains/use cases:
- Validate whether a maintenance action actually changes failure/degradation trajectory.

Strengths:
- Avoids confounding-driven policy mistakes.

Limitations:
- Requires careful design and assumptions on treatment assignment.

### 8. Simulation-Optimization (Monte Carlo + Optimization)
What they do:
- Simulate uncertain futures and optimize expected or risk-adjusted outcomes.

Typical formulations:
- Sample Average Approximation (SAA).
- Chance-constrained or CVaR optimization.

Best-fit domains/use cases:
- Multi-year capex plans with uncertain failures, costs, and weather.

Strengths:
- Directly integrates uncertainty into decision-making.

Limitations:
- Computationally intensive for large portfolios.

## Example Model Compositions by Domain

### Water Networks (Pipes + Pumps)
- Hierarchical survival model for pump failure.
- Spatiotemporal break-risk model for pipes.
- Hydraulic network simulation to map failures to service impact.
- Stochastic optimization for replacement schedule under budget constraints.

### Roads and Bridges
- Markov deterioration model for condition states.
- Traffic-based consequence model for service impact.
- Network-aware intervention bundling to reduce user-delay costs.

### Data Centers
- Survival model for UPS/chiller failure.
- Queueing or reliability-block model for redundancy impact.
- Risk-constrained maintenance scheduling around demand windows.

### Electricity Networks
- Bayesian hazard/degradation models for key assets.
- Power-flow contingency simulation for system consequences.
- Portfolio optimization with reliability and resilience constraints.

## What a Unified Framework Should Standardize
- `predict_state(asset, horizon, scenario)`
- `predict_failure(asset, horizon, scenario)`
- `simulate_system(network, failures, interventions)`
- `evaluate_policy(policy, scenarios)`
- `optimize(actions, constraints, objective)`

This API allows mixing model families without locking into one statistical ideology.

## Selected References
- [survival (R) package](https://cran.r-project.org/web/packages/survival/index.html)
- [lifelines documentation](https://lifelines.readthedocs.io/)
- [scikit-survival documentation](https://scikit-survival.readthedocs.io/)
- [PyMC documentation](https://www.pymc.io/projects/docs/en/stable/)
- [Stan User Guides](https://mc-stan.org/users/documentation/)
- [WNTR documentation](https://usepa.github.io/WNTR/)
- [pandapower documentation](https://pandapower.readthedocs.io/)
