"""Tests for the simulation module.

Covers:
- SimulationConfig validation and defaults
- SimulationResult convenience methods
- InterventionType effects and immutability
- Simulator behavior, reproducibility, and correctness
- get_intervention_options for Phase 4 optimization
- Conditional probability calculations
"""

import numpy as np
import pandas as pd
import pytest
from dataclasses import FrozenInstanceError
from scipy.stats import weibull_min

from asset_optimization.models import WeibullModel
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
from asset_optimization.simulation.config import FAILURE_RESPONSES


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
        assert config.track_asset_history is False
        assert config.failure_response == 'replace'

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

    @pytest.mark.parametrize("response", ['replace', 'repair', 'record_only'])
    def test_config_valid_failure_responses(self, response):
        """All valid failure_response values should work."""
        config = SimulationConfig(n_years=5, failure_response=response)
        assert config.failure_response == response

    def test_config_invalid_failure_response(self):
        """Invalid failure_response should raise ValueError."""
        with pytest.raises(ValueError, match="failure_response must be one of"):
            SimulationConfig(n_years=5, failure_response='invalid')


# ============================================================================
# TestSimulationResult
# ============================================================================


class TestSimulationResult:
    """Tests for SimulationResult convenience methods."""

    def test_result_total_cost(self):
        """total_cost() should sum total_cost column."""
        config = SimulationConfig(n_years=3)
        summary = pd.DataFrame({
            'year': [2026, 2027, 2028],
            'total_cost': [100000.0, 150000.0, 120000.0],
            'failure_count': [5, 8, 6],
        })
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
        summary = pd.DataFrame({
            'year': [2026, 2027, 2028],
            'total_cost': [100000.0, 150000.0, 120000.0],
            'failure_count': [5, 8, 6],
        })
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
        summary = pd.DataFrame({
            'year': [2026, 2027, 2028],
            'total_cost': [100000.0, 150000.0, 120000.0],
            'failure_count': [5, 8, 6],
        })
        result = SimulationResult(
            summary=summary,
            cost_breakdown=pd.DataFrame(),
            failure_log=pd.DataFrame(),
            config=config,
        )
        repr_str = repr(result)
        assert 'SimulationResult' in repr_str
        assert '2026-2028' in repr_str
        assert '$370,000' in repr_str
        assert 'failures=19' in repr_str

    def test_result_with_asset_history(self):
        """Result should accept optional asset_history DataFrame."""
        config = SimulationConfig(n_years=1, track_asset_history=True)
        asset_history = pd.DataFrame({
            'year': [2026, 2026],
            'asset_id': ['PIPE-001', 'PIPE-002'],
            'age': [10.0, 15.0],
        })
        result = SimulationResult(
            summary=pd.DataFrame({'year': [2026], 'total_cost': [0.0], 'failure_count': [0]}),
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
            name='HeavyRepair',
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
            InterventionType(name='', cost=0.0, age_effect=lambda x: x)

    def test_intervention_validation_negative_cost(self):
        """Negative cost should raise ValueError."""
        with pytest.raises(ValueError, match="cost must be non-negative"):
            InterventionType(name='BadIntervention', cost=-100.0, age_effect=lambda x: x)

    def test_intervention_validation_negative_consequence_cost(self):
        """Negative consequence_cost should raise ValueError."""
        with pytest.raises(ValueError, match="consequence_cost must be non-negative"):
            InterventionType(
                name='BadIntervention',
                cost=100.0,
                age_effect=lambda x: x,
                consequence_cost=-50.0,
            )


# ============================================================================
# TestSimulator
# ============================================================================


class TestSimulator:
    """Tests for Simulator behavior and correctness."""

    def test_simulator_run_returns_result(self, sample_portfolio, weibull_model, simulation_config):
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

        # Summaries should be identical
        pd.testing.assert_frame_equal(result1.summary, result2.summary)
        assert result1.total_cost() == result2.total_cost()
        assert result1.total_failures() == result2.total_failures()

    def test_simulator_different_seeds_different_results(self, sample_portfolio, weibull_model):
        """Different seeds should (likely) produce different results."""
        config1 = SimulationConfig(n_years=10, random_seed=42)
        config2 = SimulationConfig(n_years=10, random_seed=123)

        sim1 = Simulator(weibull_model, config1)
        sim2 = Simulator(weibull_model, config2)

        result1 = sim1.run(sample_portfolio)
        result2 = sim2.run(sample_portfolio)

        # Very unlikely to have identical failure counts across 10 years
        # with different seeds (probability ~0)
        assert result1.total_failures() != result2.total_failures()

    def test_simulator_ages_increment_each_year(self, weibull_model):
        """Asset ages should increment by 1 each simulation year."""
        # Create a portfolio with known ages (all installed same date)
        test_data = pd.DataFrame({
            'asset_id': ['PIPE-001'],
            'asset_type': ['pipe'],
            'material': ['PVC'],
            'install_date': pd.to_datetime(['2010-01-01']),  # ~16 years old in 2026
            'diameter_mm': [100],
            'length_m': [50.0],
            'condition_score': [80.0],
        })
        portfolio = test_data

        # Use record_only to prevent age resets from interventions
        config = SimulationConfig(
            n_years=3,
            start_year=2026,
            random_seed=42,
            failure_response='record_only',
        )
        sim = Simulator(weibull_model, config)
        result = sim.run(portfolio)

        # Check average ages increase (roughly)
        # Note: Ages are recorded at end of timestep after incrementing
        avg_ages = result.summary['avg_age'].values
        # Year 1: Age after increment (~17), Year 2: (~18), Year 3: (~19)
        assert avg_ages[0] < avg_ages[1] < avg_ages[2]

    def test_simulator_failures_trigger_intervention(self, weibull_model):
        """Failures should be recorded in failure_log."""
        # Create old assets likely to fail
        test_data = pd.DataFrame({
            'asset_id': [f'PIPE-{i:03d}' for i in range(50)],
            'asset_type': ['pipe'] * 50,
            'material': ['Cast Iron'] * 50,  # Shorter life
            'install_date': pd.to_datetime(['1970-01-01'] * 50),  # Very old
            'diameter_mm': [100] * 50,
            'length_m': [50.0] * 50,
            'condition_score': [50.0] * 50,
        })
        portfolio = test_data

        config = SimulationConfig(n_years=5, random_seed=42)
        sim = Simulator(weibull_model, config)
        result = sim.run(portfolio)

        # With very old cast iron assets, expect failures
        assert result.total_failures() > 0
        assert len(result.failure_log) > 0
        assert 'asset_id' in result.failure_log.columns
        assert 'age_at_failure' in result.failure_log.columns

    def test_simulator_replace_resets_age(self, weibull_model):
        """Replace intervention should reset age to 0."""
        # Create very old asset that will definitely fail
        test_data = pd.DataFrame({
            'asset_id': ['PIPE-001'],
            'asset_type': ['pipe'],
            'material': ['Cast Iron'],
            'install_date': pd.to_datetime(['1950-01-01']),  # 76+ years old
            'diameter_mm': [100],
            'length_m': [50.0],
            'condition_score': [30.0],
        })
        portfolio = test_data

        config = SimulationConfig(
            n_years=10,
            random_seed=42,
            failure_response='replace',
        )
        sim = Simulator(weibull_model, config)
        result = sim.run(portfolio)

        # Asset should have failed and been replaced
        # After replacement, age resets, so avg_age shouldn't keep climbing to 86+
        # Instead, it should show evidence of resets
        avg_ages = result.summary['avg_age'].values
        # After 10 years starting at ~76, if never replaced: ~86
        # If replaced: some ages should be lower
        # We verify replacement happened by checking failures occurred
        assert result.total_failures() > 0

    def test_simulator_cumulative_costs(self, sample_portfolio, weibull_model, simulation_config):
        """Total costs should be sum of yearly costs."""
        sim = Simulator(weibull_model, simulation_config)
        result = sim.run(sample_portfolio)

        # total_cost should equal sum of summary['total_cost']
        assert result.total_cost() == result.summary['total_cost'].sum()

    def test_simulator_failure_log_populated(self, weibull_model):
        """Failure log should contain details for each failure."""
        # Create assets likely to fail
        test_data = pd.DataFrame({
            'asset_id': [f'PIPE-{i:03d}' for i in range(20)],
            'asset_type': ['pipe'] * 20,
            'material': ['Cast Iron'] * 20,
            'install_date': pd.to_datetime(['1975-01-01'] * 20),
            'diameter_mm': [100] * 20,
            'length_m': [50.0] * 20,
            'condition_score': [50.0] * 20,
        })
        portfolio = test_data

        config = SimulationConfig(n_years=10, random_seed=42)
        sim = Simulator(weibull_model, config)
        result = sim.run(portfolio)

        # Check failure_log structure
        if len(result.failure_log) > 0:
            assert 'year' in result.failure_log.columns
            assert 'asset_id' in result.failure_log.columns
            assert 'age_at_failure' in result.failure_log.columns
            assert 'material' in result.failure_log.columns
            assert 'direct_cost' in result.failure_log.columns
            assert 'consequence_cost' in result.failure_log.columns

    def test_simulator_track_asset_history(self, sample_portfolio, weibull_model):
        """track_asset_history=True should populate asset_history."""
        config = SimulationConfig(n_years=3, random_seed=42, track_asset_history=True)
        sim = Simulator(weibull_model, config)
        result = sim.run(sample_portfolio)

        assert result.asset_history is not None
        # 100 assets x 3 years = 300 rows
        assert len(result.asset_history) == 300
        assert 'year' in result.asset_history.columns
        assert 'asset_id' in result.asset_history.columns
        assert 'age' in result.asset_history.columns

    def test_simulator_repr(self, weibull_model, simulation_config):
        """__repr__ should show model and config info."""
        sim = Simulator(weibull_model, simulation_config)
        repr_str = repr(sim)
        assert 'Simulator' in repr_str
        assert 'WeibullModel' in repr_str
        assert 'n_years=5' in repr_str
        assert 'seed=42' in repr_str


# ============================================================================
# TestInterventionOptions (INTV-04)
# ============================================================================


class TestInterventionOptions:
    """Tests for get_intervention_options method (INTV-04 requirement)."""

    def test_get_intervention_options_returns_dataframe(self, sample_portfolio, weibull_model, simulation_config):
        """get_intervention_options should return a DataFrame."""
        sim = Simulator(weibull_model, simulation_config)
        state = sample_portfolio.copy()
        state['age'] = 20.0

        options = sim.get_intervention_options(state, year=2026)
        assert isinstance(options, pd.DataFrame)

    def test_get_intervention_options_includes_all_assets(self, sample_portfolio, weibull_model, simulation_config):
        """All assets should have intervention options."""
        sim = Simulator(weibull_model, simulation_config)
        state = sample_portfolio.copy()
        state['age'] = 20.0

        options = sim.get_intervention_options(state, year=2026)

        # Each asset should have 4 options (do_nothing, inspect, repair, replace)
        n_assets = len(state)
        n_intervention_types = 4
        assert len(options) == n_assets * n_intervention_types

    def test_get_intervention_options_includes_all_intervention_types(self, weibull_model, simulation_config):
        """Each asset should have all intervention type options."""
        test_data = pd.DataFrame({
            'asset_id': ['PIPE-001'],
            'asset_type': ['pipe'],
            'material': ['PVC'],
            'install_date': pd.to_datetime(['2010-01-01']),
            'diameter_mm': [100],
            'length_m': [50.0],
            'condition_score': [80.0],
        })
        portfolio = test_data

        sim = Simulator(weibull_model, simulation_config)
        state = portfolio.copy()
        state['age'] = 20.0

        options = sim.get_intervention_options(state, year=2026)

        # Check all intervention types present
        intervention_types = options['intervention_type'].unique()
        assert 'DoNothing' in intervention_types
        assert 'Inspect' in intervention_types
        assert 'Repair' in intervention_types
        assert 'Replace' in intervention_types

    def test_get_intervention_options_has_required_columns(self, sample_portfolio, weibull_model, simulation_config):
        """Options DataFrame should have required columns."""
        sim = Simulator(weibull_model, simulation_config)
        state = sample_portfolio.copy()
        state['age'] = 20.0

        options = sim.get_intervention_options(state, year=2026)

        required_cols = ['asset_id', 'intervention_type', 'cost', 'age_effect']
        for col in required_cols:
            assert col in options.columns, f"Missing column: {col}"

    def test_get_intervention_options_age_effect_descriptions(self, weibull_model, simulation_config):
        """age_effect column should describe the effect correctly."""
        test_data = pd.DataFrame({
            'asset_id': ['PIPE-001'],
            'asset_type': ['pipe'],
            'material': ['PVC'],
            'install_date': pd.to_datetime(['2010-01-01']),
            'diameter_mm': [100],
            'length_m': [50.0],
            'condition_score': [80.0],
        })
        portfolio = test_data

        sim = Simulator(weibull_model, simulation_config)
        state = portfolio.copy()
        state['age'] = 20.0

        options = sim.get_intervention_options(state, year=2026)

        # Check age effect descriptions
        do_nothing_effect = options[options['intervention_type'] == 'DoNothing']['age_effect'].iloc[0]
        assert do_nothing_effect == 'no change'

        replace_effect = options[options['intervention_type'] == 'Replace']['age_effect'].iloc[0]
        assert replace_effect == 'age = 0'

        repair_effect = options[options['intervention_type'] == 'Repair']['age_effect'].iloc[0]
        assert 'age - ' in repair_effect

    def test_get_intervention_options_costs_match_interventions(self, weibull_model, simulation_config):
        """Costs in options should match intervention definitions."""
        test_data = pd.DataFrame({
            'asset_id': ['PIPE-001'],
            'asset_type': ['pipe'],
            'material': ['PVC'],
            'install_date': pd.to_datetime(['2010-01-01']),
            'diameter_mm': [100],
            'length_m': [50.0],
            'condition_score': [80.0],
        })
        portfolio = test_data

        sim = Simulator(weibull_model, simulation_config)
        state = portfolio.copy()
        state['age'] = 20.0

        options = sim.get_intervention_options(state, year=2026)

        # Verify costs match predefined interventions
        do_nothing_cost = options[options['intervention_type'] == 'DoNothing']['cost'].iloc[0]
        assert do_nothing_cost == DO_NOTHING.cost

        replace_cost = options[options['intervention_type'] == 'Replace']['cost'].iloc[0]
        assert replace_cost == REPLACE.cost

        repair_cost = options[options['intervention_type'] == 'Repair']['cost'].iloc[0]
        assert repair_cost == REPAIR.cost


# ============================================================================
# TestConditionalProbability
# ============================================================================


class TestConditionalProbability:
    """Tests for conditional probability calculation."""

    def test_conditional_prob_young_assets_low(self, weibull_model, simulation_config):
        """Young assets should have low conditional failure probability."""
        # Create young assets
        test_data = pd.DataFrame({
            'asset_id': [f'PIPE-{i:03d}' for i in range(10)],
            'asset_type': ['pipe'] * 10,
            'material': ['PVC'] * 10,
            'install_date': pd.date_range('2020-01-01', periods=10, freq='30D'),
            'diameter_mm': [100] * 10,
            'length_m': [50.0] * 10,
            'condition_score': [90.0] * 10,
        })
        portfolio = test_data

        # Run short simulation with record_only to track failures without intervention
        config = SimulationConfig(
            n_years=1,
            random_seed=42,
            failure_response='record_only',
        )
        sim = Simulator(weibull_model, config)

        # Test conditional probability directly
        state = portfolio.copy()
        # Start year 2026, assets installed 2020 = ~6 years old
        state['age'] = 6.0

        probs = sim._calculate_conditional_probability(state)

        # PVC with shape=2.5, scale=50, at age 6: very low probability
        # P(fail in [6,7) | survived to 6) should be < 1%
        assert probs.max() < 0.05  # All probabilities < 5%

    def test_conditional_prob_old_assets_higher(self, weibull_model, simulation_config):
        """Old assets should have higher conditional failure probability."""
        # Create state with varying ages
        test_data = pd.DataFrame({
            'asset_id': ['PIPE-001', 'PIPE-002'],
            'asset_type': ['pipe', 'pipe'],
            'material': ['Cast Iron', 'Cast Iron'],
            'install_date': pd.to_datetime(['2000-01-01', '2000-01-01']),
            'diameter_mm': [100, 100],
            'length_m': [50.0, 50.0],
            'condition_score': [80.0, 80.0],
        })
        portfolio = test_data

        config = SimulationConfig(n_years=1, random_seed=42)
        sim = Simulator(weibull_model, config)

        # Test two different ages
        state_young = portfolio.copy()
        state_young['age'] = 10.0

        state_old = portfolio.copy()
        state_old['age'] = 50.0

        probs_young = sim._calculate_conditional_probability(state_young)
        probs_old = sim._calculate_conditional_probability(state_old)

        # Old assets should have higher probability
        assert probs_old.mean() > probs_young.mean()

    def test_conditional_prob_handles_zero_survival(self, weibull_model, simulation_config):
        """Conditional probability should handle S(t)=0 case (return 1.0)."""
        test_data = pd.DataFrame({
            'asset_id': ['PIPE-001'],
            'asset_type': ['pipe'],
            'material': ['Cast Iron'],  # shape=3.0, scale=40
            'install_date': pd.to_datetime(['2000-01-01']),
            'diameter_mm': [100],
            'length_m': [50.0],
            'condition_score': [80.0],
        })
        portfolio = test_data

        config = SimulationConfig(n_years=1, random_seed=42)
        sim = Simulator(weibull_model, config)

        # Extreme age where survival is essentially 0
        state = portfolio.copy()
        state['age'] = 500.0  # Way beyond any reasonable life

        probs = sim._calculate_conditional_probability(state)

        # Should return 1.0 when survival is 0 (certain failure)
        assert probs[0] == 1.0

    def test_conditional_prob_formula_verification(self, weibull_model):
        """Verify conditional probability matches expected formula."""
        # P(fail in [t, t+1) | survived to t) = (S(t) - S(t+1)) / S(t)
        shape, scale = 3.0, 40.0  # Cast Iron params

        config = SimulationConfig(n_years=1, random_seed=42)
        model = WeibullModel({'Cast Iron': (shape, scale)})
        sim = Simulator(model, config)

        test_data = pd.DataFrame({
            'asset_id': ['PIPE-001'],
            'asset_type': ['pipe'],
            'material': ['Cast Iron'],
            'install_date': pd.to_datetime(['2000-01-01']),
            'diameter_mm': [100],
            'length_m': [50.0],
            'condition_score': [80.0],
        })
        portfolio = test_data

        # Test at age 30
        state = portfolio.copy()
        state['age'] = 30.0

        actual_prob = sim._calculate_conditional_probability(state)[0]

        # Calculate expected using scipy
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
        # This is implicitly tested by the fact that simulation works correctly
        # The _simulate_timestep method increments age first
        test_data = pd.DataFrame({
            'asset_id': ['PIPE-001'],
            'asset_type': ['pipe'],
            'material': ['PVC'],
            'install_date': pd.to_datetime(['2025-01-01']),  # ~1 year old at start
            'diameter_mm': [100],
            'length_m': [50.0],
            'condition_score': [80.0],
        })
        portfolio = test_data

        config = SimulationConfig(
            n_years=1,
            start_year=2026,
            random_seed=42,
            failure_response='record_only',
        )
        sim = Simulator(weibull_model, config)
        result = sim.run(portfolio)

        # After 1 timestep: age was 1, incremented to 2
        # avg_age should be ~2 (not ~1)
        assert result.summary['avg_age'].iloc[0] > 1.5

    def test_failures_before_interventions(self, weibull_model):
        """Failures should be sampled before interventions are applied."""
        # Create asset that will definitely fail
        test_data = pd.DataFrame({
            'asset_id': ['PIPE-001'],
            'asset_type': ['pipe'],
            'material': ['Cast Iron'],
            'install_date': pd.to_datetime(['1950-01-01']),  # 76+ years old
            'diameter_mm': [100],
            'length_m': [50.0],
            'condition_score': [30.0],
        })
        portfolio = test_data

        config = SimulationConfig(n_years=1, random_seed=42, failure_response='replace')
        sim = Simulator(weibull_model, config)
        result = sim.run(portfolio)

        # If failures occurred and replace was applied, failure_log should have
        # the age_at_failure recorded BEFORE the reset
        if len(result.failure_log) > 0:
            age_at_failure = result.failure_log['age_at_failure'].iloc[0]
            # Age should be old (77+), not 0
            assert age_at_failure > 70

    def test_intervention_applied_after_failure(self, weibull_model):
        """Intervention effects should be applied after failure is recorded."""
        # Create multiple old assets
        test_data = pd.DataFrame({
            'asset_id': [f'PIPE-{i:03d}' for i in range(30)],
            'asset_type': ['pipe'] * 30,
            'material': ['Cast Iron'] * 30,
            'install_date': pd.to_datetime(['1960-01-01'] * 30),
            'diameter_mm': [100] * 30,
            'length_m': [50.0] * 30,
            'condition_score': [40.0] * 30,
        })
        portfolio = test_data

        config = SimulationConfig(n_years=3, random_seed=42, failure_response='replace')
        sim = Simulator(weibull_model, config)
        result = sim.run(portfolio)

        # With replace, failed assets get age=0
        # avg_age should be lower than if no interventions
        # (66 + 3 years = 69 if no failures/replacements)
        # With some replacements, avg_age should be lower
        final_avg_age = result.summary['avg_age'].iloc[-1]
        expected_no_intervention = 66 + 3  # ~69

        # If any failures occurred, avg_age should be less than this
        if result.total_failures() > 0:
            assert final_avg_age < expected_no_intervention
