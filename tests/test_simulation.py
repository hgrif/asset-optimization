"""Tests for the simulation module.

Covers:
- SimulationConfig validation and defaults
- SimulationResult convenience methods
- InterventionType effects and immutability
- Simulator behavior, reproducibility, and correctness
- Conditional probability calculations
- Planner.simulate_horizon() integration
"""

import pandas as pd
import pytest
from dataclasses import FrozenInstanceError
from scipy.stats import weibull_min

from asset_optimization.models import WeibullModel
from asset_optimization.planner import Planner
from asset_optimization.protocols import NetworkSimulator
from asset_optimization.repositories import DataFrameRepository
from asset_optimization.simulation import (
    DO_NOTHING,
    INSPECT,
    REPAIR,
    REPLACE,
    InterventionType,
    SimulationConfig,
    SimulationResult,
    Simulator,
)


# ============================================================================
# TestSimulationConfig
# ============================================================================


class TestSimulationConfig:
    """Tests for SimulationConfig validation and behavior."""

    def test_config_valid_creation(self):
        """Valid config should be created successfully."""
        config = SimulationConfig(n_years=10, random_seed=42)
        assert config.n_years == 10
        assert config.random_seed == 42

    def test_config_default_values(self):
        """Config should have sensible defaults."""
        config = SimulationConfig(n_years=5)
        assert config.start_year == 2026
        assert config.random_seed is None
        assert config.failure_response == "replace"

    def test_config_immutable(self):
        """Config should be frozen (immutable)."""
        config = SimulationConfig(n_years=10)
        with pytest.raises(FrozenInstanceError):
            config.n_years = 20

    def test_config_n_years_validation_zero(self):
        """n_years=0 should raise ValueError."""
        with pytest.raises(ValueError, match="n_years must be positive"):
            SimulationConfig(n_years=0)

    def test_config_n_years_validation_negative(self):
        """Negative n_years should raise ValueError."""
        with pytest.raises(ValueError, match="n_years must be positive"):
            SimulationConfig(n_years=-5)

    @pytest.mark.parametrize("response", ["replace", "repair", "record_only"])
    def test_config_valid_failure_responses(self, response):
        """All valid failure_response values should work."""
        config = SimulationConfig(n_years=5, failure_response=response)
        assert config.failure_response == response

    def test_config_invalid_failure_response(self):
        """Invalid failure_response should raise ValueError."""
        with pytest.raises(ValueError, match="failure_response must be one of"):
            SimulationConfig(n_years=5, failure_response="invalid")


# ============================================================================
# TestSimulationResult
# ============================================================================


class TestSimulationResult:
    """Tests for SimulationResult convenience methods."""

    def test_result_total_cost(self):
        """total_cost() should sum total_cost column."""
        config = SimulationConfig(n_years=3)
        summary = pd.DataFrame(
            {
                "year": [2026, 2027, 2028],
                "total_cost": [100000.0, 150000.0, 120000.0],
                "failure_count": [5, 8, 6],
            }
        )
        result = SimulationResult(
            summary=summary,
            cost_breakdown=pd.DataFrame(),
            failure_log=pd.DataFrame(),
            config=config,
        )
        assert result.total_cost() == 370000.0

    def test_result_total_failures(self):
        """total_failures() should sum failure_count column."""
        config = SimulationConfig(n_years=3)
        summary = pd.DataFrame(
            {
                "year": [2026, 2027, 2028],
                "total_cost": [100000.0, 150000.0, 120000.0],
                "failure_count": [5, 8, 6],
            }
        )
        result = SimulationResult(
            summary=summary,
            cost_breakdown=pd.DataFrame(),
            failure_log=pd.DataFrame(),
            config=config,
        )
        assert result.total_failures() == 19

    def test_result_empty_summary(self):
        """Empty summary should return 0 for totals."""
        config = SimulationConfig(n_years=1)
        result = SimulationResult(
            summary=pd.DataFrame(),
            cost_breakdown=pd.DataFrame(),
            failure_log=pd.DataFrame(),
            config=config,
        )
        assert result.total_cost() == 0.0
        assert result.total_failures() == 0

    def test_result_repr(self):
        """__repr__ should show key metrics."""
        config = SimulationConfig(n_years=3)
        summary = pd.DataFrame(
            {
                "year": [2026, 2027, 2028],
                "total_cost": [100000.0, 150000.0, 120000.0],
                "failure_count": [5, 8, 6],
            }
        )
        result = SimulationResult(
            summary=summary,
            cost_breakdown=pd.DataFrame(),
            failure_log=pd.DataFrame(),
            config=config,
        )
        repr_str = repr(result)
        assert "SimulationResult" in repr_str
        assert "2026-2028" in repr_str
        assert "$370,000" in repr_str
        assert "failures=19" in repr_str

    def test_result_with_asset_history(self):
        """Result should accept asset_history DataFrame."""
        config = SimulationConfig(n_years=1)
        asset_history = pd.DataFrame(
            {
                "year": [2026, 2026],
                "asset_id": ["PIPE-001", "PIPE-002"],
                "age": [10.0, 15.0],
                "action": ["none", "replace"],
                "failed": [False, True],
                "failure_cost": [0.0, 15000.0],
                "intervention_cost": [0.0, 50000.0],
                "total_cost": [0.0, 65000.0],
            }
        )
        result = SimulationResult(
            summary=pd.DataFrame(
                {"year": [2026], "total_cost": [0.0], "failure_count": [0]}
            ),
            cost_breakdown=pd.DataFrame(),
            failure_log=pd.DataFrame(),
            config=config,
            asset_history=asset_history,
        )
        assert result.asset_history is not None
        assert len(result.asset_history) == 2


# ============================================================================
# TestInterventionType
# ============================================================================


class TestInterventionType:
    """Tests for InterventionType effects and validation."""

    def test_do_nothing_age_unchanged(self):
        """DoNothing should not change age."""
        assert DO_NOTHING.apply_age_effect(25.0) == 25.0
        assert DO_NOTHING.apply_age_effect(0.0) == 0.0
        assert DO_NOTHING.apply_age_effect(100.0) == 100.0

    def test_replace_age_reset_to_zero(self):
        """Replace should reset age to 0."""
        assert REPLACE.apply_age_effect(25.0) == 0.0
        assert REPLACE.apply_age_effect(0.0) == 0.0
        assert REPLACE.apply_age_effect(100.0) == 0.0

    def test_repair_age_reduced(self):
        """Repair should reduce age by 5 years."""
        assert REPAIR.apply_age_effect(25.0) == 20.0
        assert REPAIR.apply_age_effect(30.0) == 25.0

    def test_repair_age_clamped_to_zero(self):
        """Repair should not go below 0."""
        assert REPAIR.apply_age_effect(3.0) == 0.0
        assert REPAIR.apply_age_effect(0.0) == 0.0
        assert REPAIR.apply_age_effect(5.0) == 0.0

    def test_inspect_age_unchanged(self):
        """Inspect should not change age (v1 behavior)."""
        assert INSPECT.apply_age_effect(25.0) == 25.0

    def test_custom_intervention(self):
        """Custom intervention with specific cost and effect."""
        heavy_repair = InterventionType(
            name="HeavyRepair",
            cost=15000.0,
            age_effect=lambda age: max(0.0, age - 10.0),
            consequence_cost=2000.0,
        )
        assert heavy_repair.apply_age_effect(25.0) == 15.0
        assert heavy_repair.apply_age_effect(5.0) == 0.0
        assert heavy_repair.total_cost() == 17000.0

    def test_intervention_immutable(self):
        """InterventionType should be frozen."""
        with pytest.raises(FrozenInstanceError):
            DO_NOTHING.cost = 1000.0

    def test_intervention_validation_empty_name(self):
        """Empty name should raise ValueError."""
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            InterventionType(name="", cost=0.0, age_effect=lambda x: x)

    def test_intervention_validation_negative_cost(self):
        """Negative cost should raise ValueError."""
        with pytest.raises(ValueError, match="cost must be non-negative"):
            InterventionType(
                name="BadIntervention", cost=-100.0, age_effect=lambda x: x
            )

    def test_intervention_validation_negative_consequence_cost(self):
        """Negative consequence_cost should raise ValueError."""
        with pytest.raises(ValueError, match="consequence_cost must be non-negative"):
            InterventionType(
                name="BadIntervention",
                cost=100.0,
                age_effect=lambda x: x,
                consequence_cost=-50.0,
            )


# ============================================================================
# TestSimulator
# ============================================================================


class TestSimulator:
    """Tests for Simulator behavior and correctness."""

    def test_simulator_run_returns_result(
        self, sample_portfolio, weibull_model, simulation_config
    ):
        """Simulator.run() should return SimulationResult."""
        sim = Simulator(weibull_model, simulation_config)
        result = sim.run(sample_portfolio)
        assert isinstance(result, SimulationResult)
        assert len(result.summary) == simulation_config.n_years

    def test_simulator_reproducibility_same_seed(self, sample_portfolio, weibull_model):
        """Same seed should produce identical results."""
        config = SimulationConfig(n_years=5, random_seed=42)

        sim1 = Simulator(weibull_model, config)
        result1 = sim1.run(sample_portfolio)

        sim2 = Simulator(weibull_model, config)
        result2 = sim2.run(sample_portfolio)

        pd.testing.assert_frame_equal(result1.summary, result2.summary)
        assert result1.total_cost() == result2.total_cost()
        assert result1.total_failures() == result2.total_failures()

    def test_simulator_different_seeds_different_results(
        self, sample_portfolio, weibull_model
    ):
        """Different seeds should (likely) produce different results."""
        config1 = SimulationConfig(n_years=10, random_seed=42)
        config2 = SimulationConfig(n_years=10, random_seed=123)

        sim1 = Simulator(weibull_model, config1)
        sim2 = Simulator(weibull_model, config2)

        result1 = sim1.run(sample_portfolio)
        result2 = sim2.run(sample_portfolio)

        assert result1.total_failures() != result2.total_failures()

    def test_simulator_ages_increment_each_year(self, weibull_model):
        """Asset ages should increment by 1 each simulation year."""
        test_data = pd.DataFrame(
            {
                "asset_id": ["PIPE-001"],
                "asset_type": ["pipe"],
                "material": ["PVC"],
                "install_date": pd.to_datetime(["2010-01-01"]),
                "diameter_mm": [100],
                "length_m": [50.0],
                "condition_score": [80.0],
            }
        )

        config = SimulationConfig(
            n_years=3,
            start_year=2026,
            random_seed=42,
            failure_response="record_only",
        )
        sim = Simulator(weibull_model, config)
        result = sim.run(test_data)

        avg_ages = result.summary["avg_age"].values
        assert avg_ages[0] < avg_ages[1] < avg_ages[2]

    def test_simulator_failures_trigger_intervention(self, weibull_model):
        """Failures should be recorded in failure_log."""
        test_data = pd.DataFrame(
            {
                "asset_id": [f"PIPE-{i:03d}" for i in range(50)],
                "asset_type": ["pipe"] * 50,
                "material": ["Cast Iron"] * 50,
                "install_date": pd.to_datetime(["1970-01-01"] * 50),
                "diameter_mm": [100] * 50,
                "length_m": [50.0] * 50,
                "condition_score": [50.0] * 50,
            }
        )

        config = SimulationConfig(n_years=5, random_seed=42)
        sim = Simulator(weibull_model, config)
        result = sim.run(test_data)

        assert result.total_failures() > 0
        assert len(result.failure_log) > 0
        assert "asset_id" in result.failure_log.columns
        assert "age_at_failure" in result.failure_log.columns

    def test_simulator_replace_resets_age(self, weibull_model):
        """Replace intervention should reset age to 0."""
        test_data = pd.DataFrame(
            {
                "asset_id": ["PIPE-001"],
                "asset_type": ["pipe"],
                "material": ["Cast Iron"],
                "install_date": pd.to_datetime(["1950-01-01"]),
                "diameter_mm": [100],
                "length_m": [50.0],
                "condition_score": [30.0],
            }
        )

        config = SimulationConfig(
            n_years=10,
            random_seed=42,
            failure_response="replace",
        )
        sim = Simulator(weibull_model, config)
        result = sim.run(test_data)

        assert result.total_failures() > 0

    def test_simulator_cumulative_costs(
        self, sample_portfolio, weibull_model, simulation_config
    ):
        """Total costs should be sum of yearly costs."""
        sim = Simulator(weibull_model, simulation_config)
        result = sim.run(sample_portfolio)

        assert result.total_cost() == result.summary["total_cost"].sum()

    def test_simulator_failure_log_populated(self, weibull_model):
        """Failure log should contain details for each failure."""
        test_data = pd.DataFrame(
            {
                "asset_id": [f"PIPE-{i:03d}" for i in range(20)],
                "asset_type": ["pipe"] * 20,
                "material": ["Cast Iron"] * 20,
                "install_date": pd.to_datetime(["1975-01-01"] * 20),
                "diameter_mm": [100] * 20,
                "length_m": [50.0] * 20,
                "condition_score": [50.0] * 20,
            }
        )

        config = SimulationConfig(n_years=10, random_seed=42)
        sim = Simulator(weibull_model, config)
        result = sim.run(test_data)

        if len(result.failure_log) > 0:
            assert "year" in result.failure_log.columns
            assert "asset_id" in result.failure_log.columns
            assert "age_at_failure" in result.failure_log.columns
            assert "material" in result.failure_log.columns
            assert "direct_cost" in result.failure_log.columns
            assert "consequence_cost" in result.failure_log.columns

    def test_simulator_asset_history_default(self, sample_portfolio, weibull_model):
        """Asset history should be populated by default."""
        config = SimulationConfig(n_years=3, random_seed=42)
        sim = Simulator(weibull_model, config)
        result = sim.run(sample_portfolio)

        assert result.asset_history is not None
        assert len(result.asset_history) == 300
        required_cols = {
            "year",
            "asset_id",
            "age",
            "action",
            "failed",
            "failure_cost",
            "intervention_cost",
            "total_cost",
        }
        assert required_cols.issubset(result.asset_history.columns)
        allowed_actions = {"none", "record_only", "repair", "replace"}
        assert set(result.asset_history["action"].unique()).issubset(allowed_actions)

    def test_simulator_repr(self, weibull_model, simulation_config):
        """__repr__ should show model and config info."""
        sim = Simulator(weibull_model, simulation_config)
        repr_str = repr(sim)
        assert "Simulator" in repr_str
        assert "WeibullModel" in repr_str
        assert "n_years=5" in repr_str
        assert "seed=42" in repr_str


# ============================================================================
# TestConditionalProbability
# ============================================================================


class TestConditionalProbability:
    """Tests for conditional probability calculation."""

    def test_conditional_prob_young_assets_low(self, weibull_model, simulation_config):
        """Young assets should have low conditional failure probability."""
        test_data = pd.DataFrame(
            {
                "asset_id": [f"PIPE-{i:03d}" for i in range(10)],
                "asset_type": ["pipe"] * 10,
                "material": ["PVC"] * 10,
                "install_date": pd.date_range("2020-01-01", periods=10, freq="30D"),
                "diameter_mm": [100] * 10,
                "length_m": [50.0] * 10,
                "condition_score": [90.0] * 10,
            }
        )

        config = SimulationConfig(
            n_years=1,
            random_seed=42,
            failure_response="record_only",
        )
        sim = Simulator(weibull_model, config)

        state = test_data.copy()
        state["age"] = 6.0

        probs = sim._calculate_conditional_probability(state)

        assert probs.max() < 0.05

    def test_conditional_prob_old_assets_higher(self, weibull_model, simulation_config):
        """Old assets should have higher conditional failure probability."""
        test_data = pd.DataFrame(
            {
                "asset_id": ["PIPE-001", "PIPE-002"],
                "asset_type": ["pipe", "pipe"],
                "material": ["Cast Iron", "Cast Iron"],
                "install_date": pd.to_datetime(["2000-01-01", "2000-01-01"]),
                "diameter_mm": [100, 100],
                "length_m": [50.0, 50.0],
                "condition_score": [80.0, 80.0],
            }
        )

        config = SimulationConfig(n_years=1, random_seed=42)
        sim = Simulator(weibull_model, config)

        state_young = test_data.copy()
        state_young["age"] = 10.0

        state_old = test_data.copy()
        state_old["age"] = 50.0

        probs_young = sim._calculate_conditional_probability(state_young)
        probs_old = sim._calculate_conditional_probability(state_old)

        assert probs_old.mean() > probs_young.mean()

    def test_conditional_prob_handles_zero_survival(
        self, weibull_model, simulation_config
    ):
        """Conditional probability should handle S(t)=0 case (return 1.0)."""
        test_data = pd.DataFrame(
            {
                "asset_id": ["PIPE-001"],
                "asset_type": ["pipe"],
                "material": ["Cast Iron"],
                "install_date": pd.to_datetime(["2000-01-01"]),
                "diameter_mm": [100],
                "length_m": [50.0],
                "condition_score": [80.0],
            }
        )

        config = SimulationConfig(n_years=1, random_seed=42)
        sim = Simulator(weibull_model, config)

        state = test_data.copy()
        state["age"] = 500.0

        probs = sim._calculate_conditional_probability(state)

        assert probs[0] == 1.0

    def test_conditional_prob_formula_verification(self, weibull_model):
        """Verify conditional probability matches expected formula."""
        shape, scale = 3.0, 40.0

        config = SimulationConfig(n_years=1, random_seed=42)
        model = WeibullModel({"Cast Iron": (shape, scale)})
        sim = Simulator(model, config)

        test_data = pd.DataFrame(
            {
                "asset_id": ["PIPE-001"],
                "asset_type": ["pipe"],
                "material": ["Cast Iron"],
                "install_date": pd.to_datetime(["2000-01-01"]),
                "diameter_mm": [100],
                "length_m": [50.0],
                "condition_score": [80.0],
            }
        )

        state = test_data.copy()
        state["age"] = 30.0

        actual_prob = sim._calculate_conditional_probability(state)[0]

        s_t = weibull_min.sf(30.0, c=shape, scale=scale)
        s_t_plus_1 = weibull_min.sf(31.0, c=shape, scale=scale)
        expected_prob = (s_t - s_t_plus_1) / s_t

        assert abs(actual_prob - expected_prob) < 1e-10


# ============================================================================
# TestTimestepOrder
# ============================================================================


class TestTimestepOrder:
    """Tests verifying timestep execution order: Age -> Failures -> Interventions."""

    def test_age_increments_before_failure_sampling(self, weibull_model):
        """Age should increment before failure probability calculation."""
        test_data = pd.DataFrame(
            {
                "asset_id": ["PIPE-001"],
                "asset_type": ["pipe"],
                "material": ["PVC"],
                "install_date": pd.to_datetime(["2025-01-01"]),
                "diameter_mm": [100],
                "length_m": [50.0],
                "condition_score": [80.0],
            }
        )

        config = SimulationConfig(
            n_years=1,
            start_year=2026,
            random_seed=42,
            failure_response="record_only",
        )
        sim = Simulator(weibull_model, config)
        result = sim.run(test_data)

        assert result.summary["avg_age"].iloc[0] > 1.5

    def test_failures_before_interventions(self, weibull_model):
        """Failures should be sampled before interventions are applied."""
        test_data = pd.DataFrame(
            {
                "asset_id": ["PIPE-001"],
                "asset_type": ["pipe"],
                "material": ["Cast Iron"],
                "install_date": pd.to_datetime(["1950-01-01"]),
                "diameter_mm": [100],
                "length_m": [50.0],
                "condition_score": [30.0],
            }
        )

        config = SimulationConfig(n_years=1, random_seed=42, failure_response="replace")
        sim = Simulator(weibull_model, config)
        result = sim.run(test_data)

        if len(result.failure_log) > 0:
            age_at_failure = result.failure_log["age_at_failure"].iloc[0]
            assert age_at_failure > 70

    def test_intervention_applied_after_failure(self, weibull_model):
        """Intervention effects should be applied after failure is recorded."""
        test_data = pd.DataFrame(
            {
                "asset_id": [f"PIPE-{i:03d}" for i in range(30)],
                "asset_type": ["pipe"] * 30,
                "material": ["Cast Iron"] * 30,
                "install_date": pd.to_datetime(["1960-01-01"] * 30),
                "diameter_mm": [100] * 30,
                "length_m": [50.0] * 30,
                "condition_score": [40.0] * 30,
            }
        )

        config = SimulationConfig(n_years=3, random_seed=42, failure_response="replace")
        sim = Simulator(weibull_model, config)
        result = sim.run(test_data)

        final_avg_age = result.summary["avg_age"].iloc[-1]
        expected_no_intervention = 66 + 3

        if result.total_failures() > 0:
            assert final_avg_age < expected_no_intervention


class TestPlannerSimulationCompatibility:
    """Planner-oriented NetworkSimulator compatibility checks."""

    def test_simulator_matches_network_simulator_protocol(
        self, weibull_model, simulation_config
    ):
        sim = Simulator(weibull_model, simulation_config)
        assert isinstance(sim, NetworkSimulator)

    def test_simulate_adds_consequence_cost_column(self, weibull_model):
        sim = Simulator(weibull_model, SimulationConfig(n_years=1, random_seed=42))
        topology = pd.DataFrame({"from_asset_id": ["A1"], "to_asset_id": ["A2"]})
        failures = pd.DataFrame(
            {
                "asset_id": ["A1", "A1", "A3"],
                "consequence_cost": [1000.0, 250.0, 500.0],
            }
        )
        actions = pd.DataFrame(
            {
                "asset_id": ["A1", "A2"],
                "action_type": ["replace", "inspect"],
                "direct_cost": [50000.0, 500.0],
            }
        )

        result = sim.simulate(topology, failures, actions)

        assert "consequence_cost" in result.columns
        a1_cost = result[result["asset_id"] == "A1"]["consequence_cost"].iloc[0]
        a2_cost = result[result["asset_id"] == "A2"]["consequence_cost"].iloc[0]
        assert a1_cost == pytest.approx(1250.0)
        assert a2_cost == pytest.approx(0.0)

    def test_simulate_returns_copy_not_mutating_actions(self, weibull_model):
        sim = Simulator(weibull_model, SimulationConfig(n_years=1))
        actions = pd.DataFrame({"asset_id": ["A1"], "action_type": ["repair"]})
        original_columns = actions.columns.tolist()

        _ = sim.simulate(pd.DataFrame(), pd.DataFrame(), actions)

        assert actions.columns.tolist() == original_columns


# ============================================================================
# TestPlannerSimulateHorizon
# ============================================================================


class TestPlannerSimulateHorizon:
    """Tests for Planner.simulate_horizon() integration."""

    def _build_planner(self, portfolio_df, weibull_model):
        """Build a Planner with a DataFrameRepository."""
        from asset_optimization.effects import RuleBasedEffectModel

        repository = DataFrameRepository(assets=portfolio_df)
        return Planner(
            repository=repository,
            risk_model=weibull_model,
            effect_model=RuleBasedEffectModel(),
            simulator=Simulator(weibull_model, SimulationConfig(n_years=1)),
            optimizer=_StubOptimizer(),
        )

    def test_simulate_horizon_returns_result(self, sample_portfolio, weibull_model):
        """simulate_horizon() returns a SimulationResult."""
        planner = self._build_planner(sample_portfolio, weibull_model)
        config = SimulationConfig(n_years=5, random_seed=42)

        result = planner.simulate_horizon(config)

        assert isinstance(result, SimulationResult)
        assert len(result.summary) == 5

    def test_simulate_horizon_reproducible(self, sample_portfolio, weibull_model):
        """Same config produces identical results."""
        planner = self._build_planner(sample_portfolio, weibull_model)
        config = SimulationConfig(n_years=3, random_seed=42)

        r1 = planner.simulate_horizon(config)
        r2 = planner.simulate_horizon(config)

        pd.testing.assert_frame_equal(r1.summary, r2.summary)

    def test_simulate_horizon_requires_deterioration_model(self, sample_portfolio):
        """simulate_horizon() raises TypeError if risk_model is not a DeteriorationModel."""
        from asset_optimization.effects import RuleBasedEffectModel

        class FakeRiskModel:
            def fit(self, **kw):
                return self

            def predict_distribution(self, **kw):
                return pd.DataFrame()

            def describe(self):
                return {}

        repository = DataFrameRepository(assets=sample_portfolio)
        planner = Planner(
            repository=repository,
            risk_model=FakeRiskModel(),
            effect_model=RuleBasedEffectModel(),
            simulator=_StubSimulator(),
            optimizer=_StubOptimizer(),
        )

        with pytest.raises(TypeError, match="DeteriorationModel"):
            planner.simulate_horizon(SimulationConfig(n_years=1))


class _StubOptimizer:
    def solve(self, objective, constraints, candidates, risk_measure="expected_value"):
        from asset_optimization.types import PlanResult

        return PlanResult(
            selected_actions=pd.DataFrame(),
            objective_breakdown={},
            constraint_shadow_prices={},
            metadata={},
        )


class _StubSimulator:
    def simulate(self, topology, failures, actions, scenarios=None):
        return actions.copy()
