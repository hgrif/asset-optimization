"""Tests for GroupPropagationRiskModel (Phase 9)."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from asset_optimization.models.group_propagation import GroupPropagationRiskModel
from asset_optimization.types import DataFrameLike, PlanningHorizon, ScenarioSet


class DummyRiskModel:
    """Dummy RiskModel that returns deterministic failure_prob from assets."""

    def fit(
        self,
        assets: DataFrameLike,
        events: DataFrameLike,
        covariates: DataFrameLike | None = None,
    ) -> "DummyRiskModel":
        """No-op fit for testing."""
        return self

    def predict_distribution(
        self,
        assets: DataFrameLike,
        horizon: PlanningHorizon,
        scenarios: ScenarioSet | None = None,
    ) -> DataFrameLike:
        """Return deterministic failure_prob from assets."""
        if not isinstance(assets, pd.DataFrame):
            assets = pd.DataFrame(assets)

        # Use failure_prob from assets if available, otherwise default 0.1
        if "failure_prob" in assets.columns:
            base_probs = assets["failure_prob"].to_numpy()
        else:
            base_probs = np.full(len(assets), 0.1)

        # Simple single-step, single-scenario prediction
        return pd.DataFrame(
            {
                "asset_id": assets["asset_id"].to_numpy(),
                "scenario_id": "baseline",
                "horizon_step": 0,
                "failure_prob": base_probs,
                "loss_mean": 0.0,
            }
        )

    def describe(self) -> dict[str, Any]:
        """Return dummy metadata."""
        return {"model_type": "DummyRiskModel"}


@pytest.fixture
def dummy_base_model():
    """Create a dummy base model for testing."""
    return DummyRiskModel()


@pytest.fixture
def planning_horizon():
    """Create a short planning horizon for testing."""
    return PlanningHorizon(
        start_date="2026-01-01", end_date="2026-12-31", step="yearly"
    )


class TestGroupPropagationModelInit:
    """Initialization and validation behavior."""

    def test_instantiation_with_valid_params(self, dummy_base_model):
        model = GroupPropagationRiskModel(
            base_model=dummy_base_model,
            propagation_factor=0.5,
            group_column="group_id",
            min_group_size=2,
        )
        assert model.base_model is dummy_base_model
        assert model.propagation_factor == 0.5
        assert model.group_column == "group_id"
        assert model.min_group_size == 2

    def test_base_model_must_implement_risk_model_protocol(self):
        with pytest.raises(TypeError, match="must implement RiskModel protocol"):
            GroupPropagationRiskModel(base_model="not a model")

    def test_propagation_factor_must_be_finite(self, dummy_base_model):
        with pytest.raises(ValueError, match="must be finite"):
            GroupPropagationRiskModel(
                base_model=dummy_base_model, propagation_factor=float("inf")
            )

    def test_propagation_factor_must_be_non_negative(self, dummy_base_model):
        with pytest.raises(ValueError, match="must be non-negative"):
            GroupPropagationRiskModel(
                base_model=dummy_base_model, propagation_factor=-0.1
            )

    def test_group_column_must_be_non_empty_string(self, dummy_base_model):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            GroupPropagationRiskModel(base_model=dummy_base_model, group_column="")

    def test_min_group_size_must_be_at_least_2(self, dummy_base_model):
        with pytest.raises(ValueError, match="must be an integer >= 2"):
            GroupPropagationRiskModel(base_model=dummy_base_model, min_group_size=1)

    def test_defaults(self, dummy_base_model):
        model = GroupPropagationRiskModel(base_model=dummy_base_model)
        assert model.propagation_factor == 0.5
        assert model.group_column == "group_id"
        assert model.min_group_size == 2


class TestGroupPropagationModelFit:
    """Fit method delegates to base model."""

    def test_fit_delegates_to_base_model(self, dummy_base_model):
        model = GroupPropagationRiskModel(base_model=dummy_base_model)
        assets = pd.DataFrame({"asset_id": ["A"]})
        events = pd.DataFrame()

        result = model.fit(assets, events)
        assert result is model  # Should return self


class TestGroupPropagationModelPredictDistribution:
    """Prediction with group propagation."""

    def test_propagation_no_group_column(self, dummy_base_model, planning_horizon):
        """Assets without group_id return unchanged predictions."""
        model = GroupPropagationRiskModel(
            base_model=dummy_base_model, propagation_factor=0.5
        )
        assets = pd.DataFrame(
            {
                "asset_id": ["A", "B", "C"],
                "failure_prob": [0.1, 0.2, 0.3],
            }
        )

        predictions = model.predict_distribution(assets, planning_horizon)

        # Should return baseline predictions unchanged
        assert len(predictions) == 3
        expected_probs = [0.1, 0.2, 0.3]
        actual_probs = predictions["failure_prob"].tolist()
        assert actual_probs == expected_probs

    def test_propagation_increases_grouped_assets(
        self, dummy_base_model, planning_horizon
    ):
        """Grouped assets increase when factor > 0; ungrouped unchanged."""
        model = GroupPropagationRiskModel(
            base_model=dummy_base_model, propagation_factor=0.5
        )
        assets = pd.DataFrame(
            {
                "asset_id": ["A", "B", "C"],
                "failure_prob": [0.1, 0.2, 0.3],
                "group_id": ["group1", "group1", None],
            }
        )

        predictions = model.predict_distribution(assets, planning_horizon)

        # Assets A and B are in group1, should have increased probs
        # Asset C has no group, should be unchanged
        result_df = predictions.set_index("asset_id")

        # Group failure probability: P_group = 1 - (1-0.1)*(1-0.2) = 1 - 0.72 = 0.28
        # A new: min(0.1 * (1 + 0.5 * 0.28), 1.0) = 0.1 * 1.14 = 0.114
        # B new: min(0.2 * (1 + 0.5 * 0.28), 1.0) = 0.2 * 1.14 = 0.228
        # C unchanged: 0.3
        assert result_df.loc["A", "failure_prob"] == pytest.approx(0.114, abs=1e-6)
        assert result_df.loc["B", "failure_prob"] == pytest.approx(0.228, abs=1e-6)
        assert result_df.loc["C", "failure_prob"] == pytest.approx(0.3, abs=1e-6)

    def test_propagation_skips_singletons(self, dummy_base_model, planning_horizon):
        """Group size 1 does not change."""
        model = GroupPropagationRiskModel(
            base_model=dummy_base_model, propagation_factor=0.5, min_group_size=2
        )
        assets = pd.DataFrame(
            {
                "asset_id": ["A", "B"],
                "failure_prob": [0.1, 0.2],
                "group_id": ["group1", "group2"],  # Two singleton groups
            }
        )

        predictions = model.predict_distribution(assets, planning_horizon)

        # Both groups are singletons, should be unchanged
        result_df = predictions.set_index("asset_id")
        assert result_df.loc["A", "failure_prob"] == pytest.approx(0.1, abs=1e-6)
        assert result_df.loc["B", "failure_prob"] == pytest.approx(0.2, abs=1e-6)

    def test_propagation_factor_zero_returns_baseline(
        self, dummy_base_model, planning_horizon
    ):
        """propagation_factor=0 leaves predictions unchanged."""
        model = GroupPropagationRiskModel(
            base_model=dummy_base_model, propagation_factor=0.0
        )
        assets = pd.DataFrame(
            {
                "asset_id": ["A", "B"],
                "failure_prob": [0.1, 0.2],
                "group_id": ["group1", "group1"],
            }
        )

        predictions = model.predict_distribution(assets, planning_horizon)

        # Even though grouped, propagation_factor=0 means no change
        result_df = predictions.set_index("asset_id")
        assert result_df.loc["A", "failure_prob"] == pytest.approx(0.1, abs=1e-6)
        assert result_df.loc["B", "failure_prob"] == pytest.approx(0.2, abs=1e-6)

    def test_propagation_clips_to_1(self, dummy_base_model, planning_horizon):
        """Propagation does not exceed 1.0."""
        model = GroupPropagationRiskModel(
            base_model=dummy_base_model,
            propagation_factor=5.0,  # Large factor
        )
        assets = pd.DataFrame(
            {
                "asset_id": ["A", "B"],
                "failure_prob": [0.8, 0.9],
                "group_id": ["group1", "group1"],
            }
        )

        predictions = model.predict_distribution(assets, planning_horizon)

        # Both should be clipped to 1.0
        result_df = predictions.set_index("asset_id")
        assert result_df.loc["A", "failure_prob"] <= 1.0
        assert result_df.loc["B", "failure_prob"] <= 1.0


class TestGroupPropagationModelDescribe:
    """Describe method includes propagation metadata."""

    def test_describe_includes_propagation_metadata(self, dummy_base_model):
        """describe() includes propagation settings."""
        model = GroupPropagationRiskModel(
            base_model=dummy_base_model,
            propagation_factor=0.5,
            group_column="group_id",
            min_group_size=2,
        )

        description = model.describe()

        # Should include base model metadata
        assert description["model_type"] == "DummyRiskModel"

        # Should include wrapper metadata
        assert description["wrapper"] == "GroupPropagationRiskModel"
        assert description["propagation_factor"] == 0.5
        assert description["group_column"] == "group_id"
        assert description["min_group_size"] == 2
