"""Tests for visualization functions."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import pandas as pd
import pytest

from asset_optimization import (
    SimulationResult,
    SimulationConfig,
    MissingFieldError,
    set_sdk_theme,
    plot_cost_over_time,
    plot_failures_by_year,
    plot_risk_distribution,
    plot_scenario_comparison,
    plot_asset_action_heatmap,
    compare,
)


class TestSetSdkTheme:
    """Tests for set_sdk_theme function."""

    def test_sets_seaborn_style(self) -> None:
        """Theme can be applied without error."""
        # Just verify it runs without exception
        set_sdk_theme()


class TestPlotCostOverTime:
    """Tests for plot_cost_over_time function."""

    @pytest.fixture
    def sample_result(self) -> SimulationResult:
        """Create sample SimulationResult."""
        config = SimulationConfig(n_years=3)
        summary = pd.DataFrame({
            'year': [2024, 2025, 2026],
            'total_cost': [100000.0, 120000.0, 110000.0],
            'failure_count': [5, 7, 6],
        })
        return SimulationResult(
            summary=summary,
            cost_breakdown=pd.DataFrame(),
            failure_log=pd.DataFrame(),
            config=config,
        )

    def test_returns_axes(self, sample_result: SimulationResult) -> None:
        """Function returns matplotlib Axes object."""
        ax = plot_cost_over_time(sample_result)

        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_custom_title(self, sample_result: SimulationResult) -> None:
        """Custom title is applied."""
        ax = plot_cost_over_time(sample_result, title='Custom Title')

        assert ax.get_title() == 'Custom Title'
        import matplotlib.pyplot as plt
        plt.close('all')


class TestPlotFailuresByYear:
    """Tests for plot_failures_by_year function."""

    @pytest.fixture
    def sample_result(self) -> SimulationResult:
        """Create sample SimulationResult."""
        config = SimulationConfig(n_years=3)
        summary = pd.DataFrame({
            'year': [2024, 2025, 2026],
            'total_cost': [100000.0, 120000.0, 110000.0],
            'failure_count': [5, 7, 6],
        })
        return SimulationResult(
            summary=summary,
            cost_breakdown=pd.DataFrame(),
            failure_log=pd.DataFrame(),
            config=config,
        )

    def test_returns_axes(self, sample_result: SimulationResult) -> None:
        """Function returns matplotlib Axes object."""
        ax = plot_failures_by_year(sample_result)

        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close('all')


class TestPlotRiskDistribution:
    """Tests for plot_risk_distribution function."""

    def test_returns_axes(self) -> None:
        """Function returns matplotlib Axes object."""
        data = pd.DataFrame({'risk_score': [0.1, 0.3, 0.5, 0.7, 0.9]})
        ax = plot_risk_distribution(data)

        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_custom_column_name(self) -> None:
        """Custom risk column name works."""
        data = pd.DataFrame({'failure_probability': [0.1, 0.3, 0.5, 0.7, 0.9]})
        ax = plot_risk_distribution(data, risk_column='failure_probability')

        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_missing_column_raises(self) -> None:
        """Missing column raises ValueError."""
        data = pd.DataFrame({'other_column': [0.1, 0.3, 0.5]})

        with pytest.raises(ValueError, match="not found"):
            plot_risk_distribution(data, risk_column='risk_score')


class TestPlotScenarioComparison:
    """Tests for plot_scenario_comparison function."""

    @pytest.fixture
    def comparison_df(self) -> pd.DataFrame:
        """Create sample comparison DataFrame."""
        config = SimulationConfig(n_years=3)
        summary = pd.DataFrame({
            'year': [2024, 2025, 2026],
            'total_cost': [100000.0, 120000.0, 110000.0],
            'failure_count': [5, 7, 6],
        })
        result = SimulationResult(
            summary=summary,
            cost_breakdown=pd.DataFrame(),
            failure_log=pd.DataFrame(),
            config=config,
        )
        return compare(result, baseline='do_nothing')

    def test_returns_axes(self, comparison_df: pd.DataFrame) -> None:
        """Function returns matplotlib Axes object."""
        ax = plot_scenario_comparison(comparison_df, metric='total_cost')

        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_invalid_metric_raises(self, comparison_df: pd.DataFrame) -> None:
        """Invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            plot_scenario_comparison(comparison_df, metric='nonexistent_metric')


class TestPlotAssetActionHeatmap:
    """Tests for plot_asset_action_heatmap function."""

    @pytest.fixture
    def asset_history(self) -> pd.DataFrame:
        """Create sample asset history data."""
        return pd.DataFrame({
            'asset_id': [101, 101, 101, 202, 202, 202],
            'year': [2024, 2025, 2026, 2024, 2025, 2026],
            'action': ['none', 'repair', 'none', 'record_only', 'replace', 'none'],
        })

    def test_returns_axes(self, asset_history: pd.DataFrame) -> None:
        """Function returns matplotlib Axes object."""
        ax = plot_asset_action_heatmap(asset_history, max_assets=2)

        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_missing_columns_raise(self) -> None:
        """Missing required columns raises MissingFieldError."""
        data = pd.DataFrame({'asset_id': [1], 'year': [2024]})

        with pytest.raises(MissingFieldError, match="Missing required columns"):
            plot_asset_action_heatmap(data)
