"""Tests for scenario comparison functions."""

import pandas as pd
import pytest

from asset_optimization import (
    SimulationResult,
    SimulationConfig,
    compare_scenarios,
    create_do_nothing_baseline,
    compare,
)


class TestCompareScenarios:
    """Tests for compare_scenarios function."""

    @pytest.fixture
    def sample_result(self) -> SimulationResult:
        """Create sample SimulationResult."""
        config = SimulationConfig(n_years=3)
        summary = pd.DataFrame(
            {
                "year": [2024, 2025, 2026],
                "total_cost": [100000.0, 120000.0, 110000.0],
                "failure_count": [5, 7, 6],
                "intervention_count": [10, 12, 11],
            }
        )
        return SimulationResult(
            summary=summary,
            cost_breakdown=pd.DataFrame(),
            failure_log=pd.DataFrame(),
            config=config,
        )

    def test_output_has_correct_columns(self, sample_result: SimulationResult) -> None:
        """Output DataFrame has scenario, year, metric, value columns."""
        result = compare_scenarios({"test": sample_result})

        assert list(result.columns) == ["scenario", "year", "metric", "value"]

    def test_multiple_scenarios(self, sample_result: SimulationResult) -> None:
        """Multiple scenarios produce multiple scenario values."""
        result = compare_scenarios(
            {
                "scenario_a": sample_result,
                "scenario_b": sample_result,
            }
        )

        assert set(result["scenario"].unique()) == {"scenario_a", "scenario_b"}

    def test_custom_metrics(self, sample_result: SimulationResult) -> None:
        """Custom metrics filter to specified columns."""
        result = compare_scenarios(
            {"test": sample_result},
            metrics=["total_cost"],
        )

        assert set(result["metric"].unique()) == {"total_cost"}

    def test_empty_scenarios_returns_empty_df(self) -> None:
        """Empty scenarios dict returns empty DataFrame with correct columns."""
        result = compare_scenarios({})

        assert list(result.columns) == ["scenario", "year", "metric", "value"]
        assert len(result) == 0


class TestCreateDoNothingBaseline:
    """Tests for create_do_nothing_baseline function."""

    @pytest.fixture
    def sample_result(self) -> SimulationResult:
        """Create sample SimulationResult."""
        config = SimulationConfig(n_years=3)
        summary = pd.DataFrame(
            {
                "year": [2024, 2025, 2026],
                "total_cost": [100000.0, 120000.0, 110000.0],
                "failure_count": [5, 7, 6],
                "intervention_count": [10, 12, 11],
                "avg_age": [25.0, 26.0, 27.0],
            }
        )
        return SimulationResult(
            summary=summary,
            cost_breakdown=pd.DataFrame(),
            failure_log=pd.DataFrame(),
            config=config,
        )

    def test_returns_simulation_result(self, sample_result: SimulationResult) -> None:
        """Returns a SimulationResult object."""
        baseline = create_do_nothing_baseline(sample_result)

        assert isinstance(baseline, SimulationResult)

    def test_intervention_count_zero(self, sample_result: SimulationResult) -> None:
        """Baseline has zero interventions."""
        baseline = create_do_nothing_baseline(sample_result)

        assert all(baseline.summary["intervention_count"] == 0)

    def test_failures_increase_over_time(self, sample_result: SimulationResult) -> None:
        """Failure count tends to increase over time."""
        baseline = create_do_nothing_baseline(sample_result)

        # Later years should have more failures (general trend)
        first_year = baseline.summary["failure_count"].iloc[0]
        last_year = baseline.summary["failure_count"].iloc[-1]
        assert last_year >= first_year

    def test_config_preserved(self, sample_result: SimulationResult) -> None:
        """Config is preserved from original result."""
        baseline = create_do_nothing_baseline(sample_result)

        assert baseline.config.n_years == sample_result.config.n_years


class TestCompare:
    """Tests for compare convenience function."""

    @pytest.fixture
    def sample_result(self) -> SimulationResult:
        """Create sample SimulationResult."""
        config = SimulationConfig(n_years=3)
        summary = pd.DataFrame(
            {
                "year": [2024, 2025, 2026],
                "total_cost": [100000.0, 120000.0, 110000.0],
                "failure_count": [5, 7, 6],
                "intervention_count": [10, 12, 11],
            }
        )
        return SimulationResult(
            summary=summary,
            cost_breakdown=pd.DataFrame(),
            failure_log=pd.DataFrame(),
            config=config,
        )

    def test_auto_baseline_do_nothing(self, sample_result: SimulationResult) -> None:
        """baseline='do_nothing' auto-generates baseline."""
        result = compare(sample_result, baseline="do_nothing")

        assert "optimized" in result["scenario"].unique()
        assert "baseline" in result["scenario"].unique()

    def test_explicit_baseline(self, sample_result: SimulationResult) -> None:
        """Explicit SimulationResult baseline works."""
        result = compare(sample_result, baseline=sample_result)

        assert len(result["scenario"].unique()) == 2

    def test_invalid_baseline_raises(self, sample_result: SimulationResult) -> None:
        """Invalid baseline string raises ValueError."""
        with pytest.raises(ValueError):
            compare(sample_result, baseline="invalid_string")
