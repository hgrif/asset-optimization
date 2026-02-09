"""Tests for deterioration models (Phase 2)."""

import time
import pytest
import numpy as np
import pandas as pd
from scipy.stats import weibull_min

from asset_optimization import WeibullModel
from asset_optimization.models import DeteriorationModel
from asset_optimization.protocols import RiskModel
from asset_optimization.types import PlanningHorizon


class TestDeteriorationModelInterface:
    """Test abstract base class interface."""

    def test_cannot_instantiate_abc(self):
        """Verify abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            DeteriorationModel()

    def test_interface_defines_failure_rate(self):
        """Verify failure_rate is an abstract method."""
        assert hasattr(DeteriorationModel, "failure_rate")
        assert getattr(DeteriorationModel.failure_rate, "__isabstractmethod__", False)

    def test_interface_defines_transform(self):
        """Verify transform is an abstract method."""
        assert hasattr(DeteriorationModel, "transform")
        assert getattr(DeteriorationModel.transform, "__isabstractmethod__", False)

    def test_interface_defines_calculate_conditional_probability(self):
        """Verify conditional probability is an abstract method."""
        assert hasattr(DeteriorationModel, "calculate_conditional_probability")
        assert getattr(
            DeteriorationModel.calculate_conditional_probability,
            "__isabstractmethod__",
            False,
        )

    def test_weibull_is_subclass(self):
        """Verify WeibullModel inherits from DeteriorationModel."""
        assert issubclass(WeibullModel, DeteriorationModel)


class TestWeibullModelInit:
    """Test WeibullModel initialization and parameter validation."""

    def test_instantiation_with_valid_params(self):
        """Model instantiates with valid parameters."""
        params = {"PVC": (2.5, 50), "Cast Iron": (3.0, 40)}
        model = WeibullModel(params)
        assert model.params == params
        assert model.type_column == "material"
        assert model.age_column == "age"

    def test_custom_column_names(self):
        """Model accepts custom column names."""
        params = {"PVC": (2.5, 50)}
        model = WeibullModel(params, type_column="pipe_material", age_column="pipe_age")
        assert model.type_column == "pipe_material"
        assert model.age_column == "pipe_age"

    def test_empty_params_raises_error(self):
        """Empty params dict raises ValueError."""
        with pytest.raises(ValueError, match="params dict cannot be empty"):
            WeibullModel({})

    def test_invalid_shape_zero_raises_error(self):
        """Zero shape parameter raises ValueError."""
        with pytest.raises(ValueError, match="Shape parameter must be > 0"):
            WeibullModel({"PVC": (0, 50)})

    def test_invalid_shape_negative_raises_error(self):
        """Negative shape parameter raises ValueError."""
        with pytest.raises(ValueError, match="Shape parameter must be > 0"):
            WeibullModel({"PVC": (-1.5, 50)})

    def test_invalid_scale_zero_raises_error(self):
        """Zero scale parameter raises ValueError."""
        with pytest.raises(ValueError, match="Scale parameter must be > 0"):
            WeibullModel({"PVC": (2.5, 0)})

    def test_invalid_scale_negative_raises_error(self):
        """Negative scale parameter raises ValueError."""
        with pytest.raises(ValueError, match="Scale parameter must be > 0"):
            WeibullModel({"PVC": (2.5, -40)})

    def test_invalid_param_format_raises_error(self):
        """Non-tuple parameters raise ValueError."""
        with pytest.raises(ValueError, match="must be.*tuple"):
            WeibullModel({"PVC": [2.5, 50]})

    def test_invalid_param_length_raises_error(self):
        """Wrong-length tuple raises ValueError."""
        with pytest.raises(ValueError, match="must be.*tuple"):
            WeibullModel({"PVC": (2.5, 50, 10)})

    def test_repr(self):
        """Model has informative repr."""
        model = WeibullModel({"PVC": (2.5, 50)})
        repr_str = repr(model)
        assert "WeibullModel" in repr_str
        assert "PVC" in repr_str


class TestWeibullModelTransform:
    """Test WeibullModel transform method."""

    @pytest.fixture
    def params(self):
        """Standard Weibull parameters for tests."""
        return {"PVC": (2.5, 50), "Cast Iron": (3.0, 40)}

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "asset_id": ["P1", "P2", "P3", "P4"],
                "material": ["PVC", "PVC", "Cast Iron", "Cast Iron"],
                "age": [10, 25, 15, 35],
            }
        )

    def test_transform_adds_columns(self, params, sample_df):
        """Transform adds failure_rate and failure_probability columns."""
        model = WeibullModel(params)
        result = model.transform(sample_df)

        assert "failure_rate" in result.columns
        assert "failure_probability" in result.columns

    def test_transform_returns_copy(self, params, sample_df):
        """Transform returns a new DataFrame, doesn't modify original."""
        model = WeibullModel(params)
        original_columns = sample_df.columns.tolist()

        result = model.transform(sample_df)

        # Original should be unchanged
        assert sample_df.columns.tolist() == original_columns
        assert "failure_rate" not in sample_df.columns

        # Result should be different object
        assert result is not sample_df

    def test_transform_preserves_original_columns(self, params, sample_df):
        """Transform preserves all original columns in output."""
        model = WeibullModel(params)
        result = model.transform(sample_df)

        for col in sample_df.columns:
            assert col in result.columns

    def test_transform_correct_row_count(self, params, sample_df):
        """Transform output has same number of rows as input."""
        model = WeibullModel(params)
        result = model.transform(sample_df)

        assert len(result) == len(sample_df)

    def test_transform_failure_rate_positive(self, params, sample_df):
        """Failure rates are non-negative."""
        model = WeibullModel(params)
        result = model.transform(sample_df)

        assert (result["failure_rate"] >= 0).all()

    def test_transform_failure_probability_range(self, params, sample_df):
        """Failure probabilities are between 0 and 1."""
        model = WeibullModel(params)
        result = model.transform(sample_df)

        assert (result["failure_probability"] >= 0).all()
        assert (result["failure_probability"] <= 1).all()

    def test_transform_missing_type_column_raises(self, params):
        """Missing type column raises ValueError."""
        model = WeibullModel(params)
        df = pd.DataFrame({"age": [10, 20]})

        with pytest.raises(ValueError, match="Required columns missing"):
            model.transform(df)

    def test_transform_missing_age_column_raises(self, params):
        """Missing age column raises ValueError."""
        model = WeibullModel(params)
        df = pd.DataFrame({"material": ["PVC", "Cast Iron"]})

        with pytest.raises(ValueError, match="Required columns missing"):
            model.transform(df)

    def test_transform_unknown_asset_type_raises(self, params):
        """Unknown asset type raises ValueError."""
        model = WeibullModel(params)
        df = pd.DataFrame(
            {
                "material": ["HDPE"],  # Not in params
                "age": [10],
            }
        )

        with pytest.raises(ValueError, match="Asset types in data missing from params"):
            model.transform(df)

    def test_transform_non_numeric_age_raises(self, params):
        """Non-numeric age column raises TypeError."""
        model = WeibullModel(params)
        df = pd.DataFrame(
            {
                "material": ["PVC"],
                "age": ["ten years"],  # String, not numeric
            }
        )

        with pytest.raises(TypeError, match="must be numeric"):
            model.transform(df)


class TestWeibullModelMathematical:
    """Test mathematical correctness of Weibull calculations."""

    def test_failure_rate_increases_with_age_shape_gt_1(self):
        """For shape > 1, failure rate increases with age."""
        params = {"PVC": (2.5, 50)}  # shape=2.5 > 1
        model = WeibullModel(params)

        df = pd.DataFrame(
            {
                "material": ["PVC"] * 5,
                "age": [10, 20, 30, 40, 50],
            }
        )

        result = model.transform(df)
        rates = result["failure_rate"].values

        # Each rate should be higher than the previous
        for i in range(1, len(rates)):
            assert rates[i] > rates[i - 1], (
                f"Rate at age {df['age'].iloc[i]} should be higher than at age {df['age'].iloc[i - 1]}"
            )

    def test_failure_probability_increases_with_age(self):
        """Failure probability (CDF) always increases with age."""
        params = {"PVC": (2.5, 50)}
        model = WeibullModel(params)

        df = pd.DataFrame(
            {
                "material": ["PVC"] * 5,
                "age": [10, 20, 30, 40, 50],
            }
        )

        result = model.transform(df)
        probs = result["failure_probability"].values

        # Each probability should be higher than the previous
        for i in range(1, len(probs)):
            assert probs[i] > probs[i - 1]

    def test_failure_rate_matches_weibull_formula(self):
        """Failure rate matches direct Weibull hazard formula."""
        shape, scale = 3.0, 40
        params = {"Test": (shape, scale)}
        model = WeibullModel(params)

        ages = np.array([10, 20, 30, 40, 50])
        df = pd.DataFrame({"material": ["Test"] * len(ages), "age": ages})

        result = model.transform(df)

        # Expected hazard: h(t) = (k/lambda) * (t/lambda)^(k-1)
        expected = (shape / scale) * np.power(ages / scale, shape - 1)

        np.testing.assert_allclose(result["failure_rate"].values, expected, rtol=1e-10)

    def test_failure_probability_matches_scipy(self):
        """Failure probability matches scipy.stats.weibull_min.cdf."""
        shape, scale = 2.5, 50
        params = {"Test": (shape, scale)}
        model = WeibullModel(params)

        ages = np.array([10, 20, 30, 40, 50])
        df = pd.DataFrame({"material": ["Test"] * len(ages), "age": ages})

        result = model.transform(df)

        # Expected CDF from scipy
        expected = weibull_min.cdf(ages, c=shape, scale=scale)

        np.testing.assert_allclose(
            result["failure_probability"].values, expected, rtol=1e-10
        )

    def test_age_zero_handled(self):
        """Age=0 produces defined result (not NaN or infinity)."""
        params = {"PVC": (2.5, 50)}
        model = WeibullModel(params)

        df = pd.DataFrame(
            {
                "material": ["PVC"],
                "age": [0],
            }
        )

        result = model.transform(df)

        # Should not be NaN
        assert not np.isnan(result["failure_rate"].iloc[0])
        assert not np.isnan(result["failure_probability"].iloc[0])

        # Age 0 should have very low failure probability
        assert result["failure_probability"].iloc[0] == pytest.approx(0.0, abs=1e-10)


class TestWeibullModelPerformance:
    """Test performance requirements from success criteria."""

    def test_performance_1000_assets_under_1_second(self):
        """Transform 1000+ assets in under 1 second (DTRN-03)."""
        # Generate 1000 assets with mixed types
        n_assets = 1000
        params = {"PVC": (2.5, 50), "Cast Iron": (3.0, 40)}
        model = WeibullModel(params)

        df = pd.DataFrame(
            {
                "asset_id": [f"P{i}" for i in range(n_assets)],
                "material": ["PVC"] * (n_assets // 2) + ["Cast Iron"] * (n_assets // 2),
                "age": np.random.uniform(1, 60, n_assets),
            }
        )

        # Time the transform
        start = time.time()
        result = model.transform(df)
        elapsed = time.time() - start

        # Must be under 1 second (success criterion)
        assert elapsed < 1.0, f"Transform took {elapsed:.2f}s, must be <1s"
        assert len(result) == n_assets

    def test_performance_10000_assets(self):
        """Verify scalability to larger portfolios."""
        n_assets = 10000
        params = {"PVC": (2.5, 50), "Cast Iron": (3.0, 40)}
        model = WeibullModel(params)

        df = pd.DataFrame(
            {
                "asset_id": [f"P{i}" for i in range(n_assets)],
                "material": ["PVC"] * (n_assets // 2) + ["Cast Iron"] * (n_assets // 2),
                "age": np.random.uniform(1, 60, n_assets),
            }
        )

        start = time.time()
        result = model.transform(df)
        elapsed = time.time() - start

        # Should still be fast for 10K assets
        assert elapsed < 5.0, f"Transform took {elapsed:.2f}s for 10K assets"
        assert len(result) == n_assets


class TestWeibullModelConditionalProbability:
    """Test conditional one-step failure probabilities."""

    def test_returns_array_same_length(self):
        model = WeibullModel({"PVC": (2.5, 50), "Cast Iron": (3.0, 40)})
        state = pd.DataFrame(
            {
                "material": ["PVC", "Cast Iron", "PVC"],
                "age": [10, 20, 30],
            }
        )
        probs = model.calculate_conditional_probability(state)
        assert len(probs) == len(state)

    def test_values_are_within_probability_bounds(self):
        model = WeibullModel({"PVC": (2.5, 50)})
        state = pd.DataFrame(
            {
                "material": ["PVC"] * 5,
                "age": [0, 10, 20, 40, 80],
            }
        )
        probs = model.calculate_conditional_probability(state)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_conditional_probability_increases_with_age_for_shape_gt_1(self):
        model = WeibullModel({"PVC": (2.5, 50)})
        state = pd.DataFrame(
            {
                "material": ["PVC", "PVC"],
                "age": [10, 40],
            }
        )
        probs = model.calculate_conditional_probability(state)
        assert probs[1] > probs[0]


class TestWeibullModelMultipleTypes:
    """Test with multiple asset types (DTRN-02)."""

    def test_different_params_per_type(self):
        """Different types use their respective parameters."""
        # PVC with shape=2 (gradual wear)
        # Cast Iron with shape=4 (steep wear)
        params = {
            "PVC": (2.0, 50),
            "Cast Iron": (4.0, 40),
        }
        model = WeibullModel(params)

        # Same age, different types
        df = pd.DataFrame(
            {
                "material": ["PVC", "Cast Iron"],
                "age": [30, 30],
            }
        )

        result = model.transform(df)

        # Failure rates should be different
        pvc_rate = result[result["material"] == "PVC"]["failure_rate"].iloc[0]
        ci_rate = result[result["material"] == "Cast Iron"]["failure_rate"].iloc[0]

        assert pvc_rate != ci_rate

        # With shape=4 vs shape=2 at same age, higher shape has higher hazard at later ages
        # At age 30, shape=4, scale=40: h = (4/40) * (30/40)^3 = 0.1 * 0.4219 = 0.0422
        # At age 30, shape=2, scale=50: h = (2/50) * (30/50)^1 = 0.04 * 0.6 = 0.024
        assert ci_rate > pvc_rate

    def test_five_asset_types(self):
        """Model handles many asset types."""
        params = {
            "PVC": (2.0, 50),
            "Cast Iron": (3.0, 40),
            "Ductile Iron": (2.5, 60),
            "Steel": (3.5, 45),
            "Concrete": (2.2, 70),
        }
        model = WeibullModel(params)

        df = pd.DataFrame(
            {
                "material": list(params.keys()),
                "age": [20] * len(params),
            }
        )

        result = model.transform(df)

        # All rows should have valid values
        assert not result["failure_rate"].isna().any()
        assert not result["failure_probability"].isna().any()


class TestPlannerRiskModelCompatibility:
    """Planner-oriented RiskModel compatibility checks."""

    def test_weibull_matches_risk_model_protocol(self):
        """WeibullModel structurally satisfies the RiskModel protocol."""
        model = WeibullModel({"PVC": (2.5, 50.0)})
        assert isinstance(model, RiskModel)

    def test_fit_returns_self_noop(self):
        """Default fit implementation is a no-op that returns self."""
        model = WeibullModel({"PVC": (2.5, 50.0)})
        assets = pd.DataFrame({"asset_id": ["A1"], "material": ["PVC"], "age": [20]})
        events = pd.DataFrame({"asset_id": ["A1"], "event_type": ["break"]})
        result = model.fit(assets, events)
        assert result is model

    def test_predict_distribution_returns_proposal_a_schema(self):
        """predict_distribution returns planner-compatible output columns."""
        model = WeibullModel({"PVC": (2.5, 50.0)})
        assets = pd.DataFrame(
            {"asset_id": ["A1", "A2"], "material": ["PVC", "PVC"], "age": [10, 25]}
        )
        horizon = PlanningHorizon("2026-01-01", "2026-12-31", "yearly")

        result = model.predict_distribution(assets, horizon)

        assert {
            "asset_id",
            "scenario_id",
            "horizon_step",
            "failure_prob",
            "loss_mean",
        }.issubset(result.columns)
        assert len(result) == len(assets)
        assert (result["scenario_id"] == "baseline").all()
        assert (result["horizon_step"] == 0).all()
        assert (result["failure_prob"] >= 0.0).all()
        assert (result["failure_prob"] <= 1.0).all()

    def test_weibull_describe_includes_parameters(self):
        """Weibull describe() reports model metadata used by planner."""
        model = WeibullModel(
            {"PVC": (2.5, 50.0)},
            type_column="material",
            age_column="age",
        )
        description = model.describe()

        assert description["model_type"] == "WeibullModel"
        assert description["params"] == {"PVC": (2.5, 50.0)}
        assert description["type_column"] == "material"
        assert description["age_column"] == "age"
