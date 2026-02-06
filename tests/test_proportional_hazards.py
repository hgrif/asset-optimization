"""Tests for ProportionalHazardsModel (Phase 7)."""

import numpy as np
import pandas as pd
import pytest

from asset_optimization.models import ProportionalHazardsModel, WeibullModel


@pytest.fixture
def weibull_baseline():
    """Baseline Weibull model for tests."""
    return WeibullModel({'PVC': (2.5, 50)})


@pytest.fixture
def df_with_covariates():
    """Sample DataFrame with covariate columns."""
    return pd.DataFrame({
        'material': ['PVC', 'PVC', 'PVC'],
        'age': [10, 20, 30],
        'diameter_mm': [100.0, 150.0, 200.0],
    })


@pytest.fixture
def df_without_covariates():
    """Sample DataFrame without covariate columns."""
    return pd.DataFrame({
        'material': ['PVC', 'PVC'],
        'age': [10, 20],
    })


@pytest.fixture
def ph_model(weibull_baseline):
    """ProportionalHazardsModel with a single covariate."""
    return ProportionalHazardsModel(
        weibull_baseline,
        covariates=['diameter_mm'],
        coefficients={'diameter_mm': 0.01},
    )


class TestProportionalHazardsModelInit:
    """Initialization and validation behavior."""

    def test_instantiation_with_valid_params(self, weibull_baseline):
        model = ProportionalHazardsModel(
            weibull_baseline,
            covariates=['diameter_mm'],
            coefficients={'diameter_mm': 0.01},
        )
        assert model.baseline is weibull_baseline
        assert model.covariates == ['diameter_mm']
        assert model.coefficients['diameter_mm'] == 0.01

    def test_covariates_must_match_coefficients(self, weibull_baseline):
        with pytest.raises(ValueError, match="covariates must match coefficient keys"):
            ProportionalHazardsModel(
                weibull_baseline,
                covariates=['diameter_mm', 'length_m'],
                coefficients={'diameter_mm': 0.01},
            )

    def test_empty_covariates_allowed(self, weibull_baseline):
        model = ProportionalHazardsModel(
            weibull_baseline,
            covariates=[],
            coefficients={},
        )
        assert model.covariates == []
        assert model.coefficients == {}

    def test_repr(self, ph_model):
        repr_str = repr(ph_model)
        assert 'ProportionalHazardsModel' in repr_str
        assert 'baseline=' in repr_str
        assert 'covariates=' in repr_str


class TestProportionalHazardsModelDelegation:
    """Delegation to baseline model."""

    def test_params_delegates_to_baseline(self, ph_model, weibull_baseline):
        assert ph_model.params == weibull_baseline.params

    def test_type_column_delegates_to_baseline(self, ph_model, weibull_baseline):
        assert ph_model.type_column == weibull_baseline.type_column

    def test_age_column_delegates_to_baseline(self, ph_model, weibull_baseline):
        assert ph_model.age_column == weibull_baseline.age_column


class TestProportionalHazardsModelTransform:
    """Transform behavior and covariate effects."""

    def test_transform_adds_columns(self, ph_model, df_with_covariates):
        result = ph_model.transform(df_with_covariates)
        assert 'failure_rate' in result.columns
        assert 'failure_probability' in result.columns

    def test_transform_returns_copy(self, ph_model, df_with_covariates):
        original = df_with_covariates.copy(deep=True)
        result = ph_model.transform(df_with_covariates)
        assert result is not df_with_covariates
        pd.testing.assert_frame_equal(df_with_covariates, original)

    def test_transform_preserves_original_columns(self, ph_model, df_with_covariates):
        result = ph_model.transform(df_with_covariates)
        for col in df_with_covariates.columns:
            assert col in result.columns

    def test_transform_increases_failure_rate_with_positive_covariate(
        self, weibull_baseline
    ):
        ph = ProportionalHazardsModel(
            weibull_baseline,
            covariates=['diameter_mm'],
            coefficients={'diameter_mm': 0.01},
        )
        df = pd.DataFrame({
            'material': ['PVC', 'PVC'],
            'age': [20, 20],
            'diameter_mm': [100.0, 200.0],
        })
        result = ph.transform(df)
        assert result.loc[1, 'failure_rate'] > result.loc[0, 'failure_rate']

    def test_transform_decreases_failure_rate_with_negative_coefficient(
        self, weibull_baseline
    ):
        ph = ProportionalHazardsModel(
            weibull_baseline,
            covariates=['diameter_mm'],
            coefficients={'diameter_mm': -0.01},
        )
        df = pd.DataFrame({
            'material': ['PVC', 'PVC'],
            'age': [20, 20],
            'diameter_mm': [100.0, 200.0],
        })
        result = ph.transform(df)
        assert result.loc[1, 'failure_rate'] < result.loc[0, 'failure_rate']


class TestProportionalHazardsModelBackwardCompat:
    """Missing covariates and NaN handling (HAZD-05)."""

    def test_missing_covariate_column_uses_baseline_only(
        self, weibull_baseline, df_without_covariates
    ):
        ph = ProportionalHazardsModel(
            weibull_baseline,
            covariates=['diameter_mm'],
            coefficients={'diameter_mm': 0.01},
        )
        baseline_result = weibull_baseline.transform(df_without_covariates)
        ph_result = ph.transform(df_without_covariates)
        np.testing.assert_allclose(
            ph_result['failure_rate'].values,
            baseline_result['failure_rate'].values,
        )
        np.testing.assert_allclose(
            ph_result['failure_probability'].values,
            baseline_result['failure_probability'].values,
        )

    def test_nan_covariate_uses_baseline_for_that_row(self, ph_model, weibull_baseline):
        df = pd.DataFrame({
            'material': ['PVC', 'PVC'],
            'age': [10, 20],
            'diameter_mm': [np.nan, 150.0],
        })
        baseline_result = weibull_baseline.transform(df)
        ph_result = ph_model.transform(df)
        assert ph_result.loc[0, 'failure_rate'] == baseline_result.loc[0, 'failure_rate']
        assert np.isclose(
            ph_result.loc[0, 'failure_probability'],
            baseline_result.loc[0, 'failure_probability'],
        )

    def test_partial_nan_handles_correctly(self, ph_model, weibull_baseline):
        df = pd.DataFrame({
            'material': ['PVC', 'PVC', 'PVC'],
            'age': [10, 20, 30],
            'diameter_mm': [100.0, np.nan, 200.0],
        })
        baseline_result = weibull_baseline.transform(df)
        ph_result = ph_model.transform(df)
        assert ph_result.loc[1, 'failure_rate'] == baseline_result.loc[1, 'failure_rate']
        assert ph_result.loc[2, 'failure_rate'] != baseline_result.loc[2, 'failure_rate']


class TestProportionalHazardsModelMathematical:
    """Mathematical correctness checks."""

    def test_risk_score_exp_of_linear_combination(self, ph_model, df_with_covariates):
        expected = np.exp(0.01 * df_with_covariates['diameter_mm'].values)
        risk = ph_model._risk_score(df_with_covariates)
        np.testing.assert_allclose(risk, expected)

    def test_failure_probability_uses_survival_power_formula(
        self, weibull_baseline, df_with_covariates
    ):
        ph = ProportionalHazardsModel(
            weibull_baseline,
            covariates=['diameter_mm'],
            coefficients={'diameter_mm': 0.02},
        )
        baseline_result = weibull_baseline.transform(df_with_covariates)
        ph_result = ph.transform(df_with_covariates)
        risk = ph._risk_score(df_with_covariates)

        survival_baseline = 1.0 - baseline_result['failure_probability'].values
        expected = 1.0 - np.power(survival_baseline, risk)
        np.testing.assert_allclose(ph_result['failure_probability'].values, expected)

    def test_zero_covariates_equals_baseline(self, weibull_baseline, df_with_covariates):
        ph = ProportionalHazardsModel(
            weibull_baseline,
            covariates=[],
            coefficients={},
        )
        baseline_result = weibull_baseline.transform(df_with_covariates)
        ph_result = ph.transform(df_with_covariates)
        np.testing.assert_allclose(
            ph_result['failure_rate'].values,
            baseline_result['failure_rate'].values,
        )
        np.testing.assert_allclose(
            ph_result['failure_probability'].values,
            baseline_result['failure_probability'].values,
        )

    def test_overflow_protection(self, weibull_baseline):
        ph = ProportionalHazardsModel(
            weibull_baseline,
            covariates=['x'],
            coefficients={'x': 5.0},
        )
        df = pd.DataFrame({'material': ['PVC'], 'age': [10], 'x': [1000.0]})
        risk = ph._risk_score(df)
        assert np.isfinite(risk).all()
        assert (risk > 0).all()


class TestProportionalHazardsConditionalProbability:
    """Conditional probability behavior for PH model."""

    def test_returns_array_same_length_as_input(self, ph_model):
        state = pd.DataFrame({
            'material': ['PVC', 'PVC', 'PVC'],
            'age': [10, 20, 30],
            'diameter_mm': [100.0, 150.0, 200.0],
        })
        probs = ph_model.calculate_conditional_probability(state)
        assert len(probs) == len(state)

    def test_probabilities_in_valid_range(self, ph_model):
        state = pd.DataFrame({
            'material': ['PVC'] * 5,
            'age': [5, 10, 20, 40, 60],
            'diameter_mm': [100.0, 120.0, 140.0, 160.0, 180.0],
        })
        probs = ph_model.calculate_conditional_probability(state)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_higher_covariate_increases_conditional_probability(self, ph_model):
        state = pd.DataFrame({
            'material': ['PVC', 'PVC'],
            'age': [25, 25],
            'diameter_mm': [100.0, 200.0],
        })
        probs = ph_model.calculate_conditional_probability(state)
        assert probs[1] > probs[0]
