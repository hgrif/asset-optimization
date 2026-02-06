"""Tests for export functions."""

from pathlib import Path

import pandas as pd
import pytest

from asset_optimization import (
    SimulationResult,
    SimulationConfig,
    OptimizationResult,
    export_schedule_minimal,
    export_schedule_detailed,
    export_cost_projections,
)


class TestExportScheduleMinimal:
    """Tests for export_schedule_minimal function."""

    def test_exports_correct_columns(self, tmp_path: Path) -> None:
        """Minimal export has exactly asset_id, year, intervention_type, cost."""
        selections = pd.DataFrame(
            {
                "asset_id": ["A1", "A2"],
                "intervention_type": ["Replace", "Repair"],
                "cost": [5000.0, 1000.0],
                "risk_score": [0.8, 0.3],
                "rank": [1, 2],
            }
        )
        output_path = tmp_path / "schedule.parquet"

        export_schedule_minimal(selections, output_path, year=2024)

        result = pd.read_parquet(output_path)
        assert list(result.columns) == ["asset_id", "year", "intervention_type", "cost"]

    def test_year_column_added(self, tmp_path: Path) -> None:
        """Year column is added with specified value."""
        selections = pd.DataFrame(
            {
                "asset_id": ["A1"],
                "intervention_type": ["Replace"],
                "cost": [5000.0],
                "risk_score": [0.8],
                "rank": [1],
            }
        )
        output_path = tmp_path / "schedule.parquet"

        export_schedule_minimal(selections, output_path, year=2030)

        result = pd.read_parquet(output_path)
        assert result["year"].iloc[0] == 2030

    def test_empty_selections(self, tmp_path: Path) -> None:
        """Empty selections produce empty parquet with correct schema."""
        selections = pd.DataFrame(
            {
                "asset_id": [],
                "intervention_type": [],
                "cost": [],
                "risk_score": [],
                "rank": [],
            }
        )
        output_path = tmp_path / "schedule.parquet"

        export_schedule_minimal(selections, output_path, year=2024)

        result = pd.read_parquet(output_path)
        assert len(result) == 0
        assert list(result.columns) == ["asset_id", "year", "intervention_type", "cost"]


class TestExportScheduleDetailed:
    """Tests for export_schedule_detailed function."""

    def test_includes_risk_columns(self, tmp_path: Path) -> None:
        """Detailed export includes risk_before, risk_after, risk_reduction."""
        selections = pd.DataFrame(
            {
                "asset_id": ["A1"],
                "intervention_type": ["Replace"],
                "cost": [5000.0],
                "risk_score": [0.8],
                "rank": [1],
                "risk_before": [0.8],
                "risk_after": [0.0],
            }
        )
        output_path = tmp_path / "schedule.parquet"

        export_schedule_detailed(selections, output_path, year=2024)

        result = pd.read_parquet(output_path)
        assert "risk_before" in result.columns
        assert "risk_after" in result.columns
        assert "risk_reduction" in result.columns
        assert result["risk_reduction"].iloc[0] == 0.8

    def test_portfolio_join_adds_material(self, tmp_path: Path) -> None:
        """Portfolio join adds material column."""
        selections = pd.DataFrame(
            {
                "asset_id": ["A1", "A2"],
                "intervention_type": ["Replace", "Repair"],
                "cost": [5000.0, 1000.0],
                "risk_score": [0.8, 0.3],
                "rank": [1, 2],
            }
        )
        portfolio = pd.DataFrame(
            {
                "asset_id": ["A1", "A2"],
                "material": ["Cast Iron", "PVC"],
                "age": [50, 20],
            }
        )
        output_path = tmp_path / "schedule.parquet"

        export_schedule_detailed(
            selections, output_path, year=2024, portfolio=portfolio
        )

        result = pd.read_parquet(output_path)
        assert "material" in result.columns
        assert "age" in result.columns
        assert result[result["asset_id"] == "A1"]["material"].iloc[0] == "Cast Iron"

    def test_asset_type_fallback_to_material(self, tmp_path: Path) -> None:
        """asset_type column is used as material if material not present."""
        selections = pd.DataFrame(
            {
                "asset_id": ["A1"],
                "intervention_type": ["Replace"],
                "cost": [5000.0],
                "risk_score": [0.8],
                "rank": [1],
            }
        )
        portfolio = pd.DataFrame(
            {
                "asset_id": ["A1"],
                "asset_type": ["Water Main"],
            }
        )
        output_path = tmp_path / "schedule.parquet"

        export_schedule_detailed(
            selections, output_path, year=2024, portfolio=portfolio
        )

        result = pd.read_parquet(output_path)
        assert "material" in result.columns
        assert result["material"].iloc[0] == "Water Main"


class TestExportCostProjections:
    """Tests for export_cost_projections function."""

    def test_long_format_output(self, tmp_path: Path) -> None:
        """Cost projections are in long format (year, metric, value)."""
        summary = pd.DataFrame(
            {
                "year": [2024, 2025],
                "total_cost": [100000.0, 120000.0],
                "failure_count": [5, 7],
            }
        )
        output_path = tmp_path / "projections.parquet"

        export_cost_projections(summary, output_path)

        result = pd.read_parquet(output_path)
        assert "year" in result.columns
        assert "metric" in result.columns
        assert "value" in result.columns
        assert set(result["metric"].unique()) == {"total_cost", "failure_count"}

    def test_multiple_years(self, tmp_path: Path) -> None:
        """Multiple years produce multiple rows per metric."""
        summary = pd.DataFrame(
            {
                "year": [2024, 2025, 2026],
                "total_cost": [100000.0, 120000.0, 110000.0],
                "failure_count": [5, 7, 6],
            }
        )
        output_path = tmp_path / "projections.parquet"

        export_cost_projections(summary, output_path)

        result = pd.read_parquet(output_path)
        # 3 years * 2 metrics = 6 rows
        assert len(result) == 6


class TestSimulationResultToParquet:
    """Tests for SimulationResult.to_parquet method."""

    @pytest.fixture
    def sim_result(self) -> SimulationResult:
        """Create test SimulationResult."""
        config = SimulationConfig(n_years=3)
        summary = pd.DataFrame(
            {
                "year": [2024, 2025, 2026],
                "total_cost": [100000.0, 120000.0, 110000.0],
                "failure_count": [5, 7, 6],
                "intervention_count": [10, 12, 11],
            }
        )
        asset_history = pd.DataFrame(
            {
                "year": [2024, 2024],
                "asset_id": ["A1", "A2"],
                "age": [10.0, 20.0],
                "action": ["none", "replace"],
                "failed": [False, True],
                "failure_cost": [0.0, 15000.0],
                "intervention_cost": [0.0, 50000.0],
                "total_cost": [0.0, 65000.0],
            }
        )
        return SimulationResult(
            summary=summary,
            cost_breakdown=pd.DataFrame(),
            failure_log=pd.DataFrame({"year": [2024], "asset_id": ["A1"]}),
            config=config,
            asset_history=asset_history,
        )

    def test_summary_format(self, sim_result: SimulationResult, tmp_path: Path) -> None:
        """Summary format exports full summary DataFrame."""
        output_path = tmp_path / "summary.parquet"
        sim_result.to_parquet(output_path, format="summary")

        result = pd.read_parquet(output_path)
        assert "year" in result.columns
        assert "total_cost" in result.columns
        assert len(result) == 3

    def test_cost_projections_format(
        self, sim_result: SimulationResult, tmp_path: Path
    ) -> None:
        """Cost projections format exports long format."""
        output_path = tmp_path / "projections.parquet"
        sim_result.to_parquet(output_path, format="cost_projections")

        result = pd.read_parquet(output_path)
        assert "metric" in result.columns
        assert "value" in result.columns

    def test_failure_log_format(
        self, sim_result: SimulationResult, tmp_path: Path
    ) -> None:
        """Failure log format exports failure_log DataFrame."""
        output_path = tmp_path / "failures.parquet"
        sim_result.to_parquet(output_path, format="failure_log")

        result = pd.read_parquet(output_path)
        assert "asset_id" in result.columns

    def test_asset_history_format(
        self, sim_result: SimulationResult, tmp_path: Path
    ) -> None:
        """Asset history format exports asset_history DataFrame."""
        output_path = tmp_path / "asset_history.parquet"
        sim_result.to_parquet(output_path, format="asset_history")

        result = pd.read_parquet(output_path)
        assert "asset_id" in result.columns
        assert "action" in result.columns
        assert "total_cost" in result.columns

    def test_invalid_format_raises(
        self, sim_result: SimulationResult, tmp_path: Path
    ) -> None:
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unknown format"):
            sim_result.to_parquet(tmp_path / "out.parquet", format="invalid")


class TestOptimizationResultToParquet:
    """Tests for OptimizationResult.to_parquet method."""

    @pytest.fixture
    def opt_result(self) -> OptimizationResult:
        """Create test OptimizationResult."""
        selections = pd.DataFrame(
            {
                "asset_id": ["A1", "A2"],
                "intervention_type": ["Replace", "Repair"],
                "cost": [5000.0, 1000.0],
                "risk_score": [0.8, 0.3],
                "rank": [1, 2],
                "risk_before": [0.8, 0.3],
                "risk_after": [0.0, 0.2],
            }
        )
        budget_summary = pd.DataFrame(
            {
                "budget": [10000.0],
                "spent": [6000.0],
                "remaining": [4000.0],
                "utilization_pct": [60.0],
            }
        )
        return OptimizationResult(
            selections=selections,
            budget_summary=budget_summary,
            strategy="greedy",
        )

    def test_minimal_format(
        self, opt_result: OptimizationResult, tmp_path: Path
    ) -> None:
        """Minimal format has 4 columns."""
        output_path = tmp_path / "minimal.parquet"
        opt_result.to_parquet(output_path, format="minimal", year=2024)

        result = pd.read_parquet(output_path)
        assert list(result.columns) == ["asset_id", "year", "intervention_type", "cost"]

    def test_detailed_format(
        self, opt_result: OptimizationResult, tmp_path: Path
    ) -> None:
        """Detailed format includes risk columns."""
        output_path = tmp_path / "detailed.parquet"
        opt_result.to_parquet(output_path, format="detailed", year=2024)

        result = pd.read_parquet(output_path)
        assert "risk_reduction" in result.columns

    def test_invalid_format_raises(
        self, opt_result: OptimizationResult, tmp_path: Path
    ) -> None:
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unknown format"):
            opt_result.to_parquet(tmp_path / "out.parquet", format="invalid")
