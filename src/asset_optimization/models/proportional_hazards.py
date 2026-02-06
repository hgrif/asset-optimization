"""Proportional hazards deterioration model wrapper."""

import numpy as np
import pandas as pd

from .base import DeteriorationModel


class ProportionalHazardsModel(DeteriorationModel):
    """Proportional hazards wrapper around any baseline DeteriorationModel.

    h(t|x) = h_baseline(t) * exp(sum(beta_i * x_i))

    Parameters
    ----------
    baseline : DeteriorationModel
        The baseline deterioration model (e.g., WeibullModel).
    covariates : list[str]
        Column names in the portfolio DataFrame to use as covariates.
    coefficients : dict[str, float]
        Maps covariate column name to beta coefficient value.
        Must have exactly one entry per covariate in the covariates list.

    Notes
    -----
    Optimizer support currently relies on delegated baseline attributes
    (``params`` and ``type_column``). Optimizer's risk-after calculation
    remains baseline-only in this phase.
    """

    def __init__(
        self,
        baseline: DeteriorationModel,
        covariates: list[str],
        coefficients: dict[str, float],
    ) -> None:
        self.baseline = baseline
        self.covariates = list(covariates)
        self.coefficients = dict(coefficients)
        self._validate()

    @property
    def params(self):
        """Delegate baseline parameters."""
        return self.baseline.params

    @property
    def type_column(self) -> str:
        """Delegate baseline type column name."""
        return self.baseline.type_column

    @property
    def age_column(self) -> str:
        """Delegate baseline age column name."""
        return self.baseline.age_column

    def _validate(self) -> None:
        """Validate baseline and covariate configuration."""
        if not isinstance(self.baseline, DeteriorationModel):
            raise TypeError(
                "baseline must be a DeteriorationModel instance"
            )

        if len(self.covariates) != len(set(self.covariates)):
            raise ValueError("covariates must be unique")

        covariate_set = set(self.covariates)
        coeff_set = set(self.coefficients.keys())
        if covariate_set != coeff_set:
            missing = sorted(covariate_set - coeff_set)
            extra = sorted(coeff_set - covariate_set)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            detail_str = ", ".join(details) if details else ""
            raise ValueError(
                "covariates must match coefficient keys" + (f" ({detail_str})" if detail_str else "")
            )

        for key, value in self.coefficients.items():
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Coefficient for '{key}' must be numeric, got {type(value).__name__}"
                )

    def _risk_score(self, df: pd.DataFrame) -> np.ndarray:
        """Compute exp(sum(beta_i * x_i)) for each row.

        Raises if any required covariate columns are missing, contain NaNs,
        or are non-numeric.
        """
        if not self.covariates:
            return np.ones(len(df))

        missing_columns = [col for col in self.covariates if col not in df.columns]
        if missing_columns:
            raise ValueError(
                "Missing covariate columns: "
                f"{missing_columns}. Available columns: {df.columns.tolist()}"
            )

        non_numeric = [
            col for col in self.covariates
            if not pd.api.types.is_numeric_dtype(df[col])
        ]
        if non_numeric:
            dtypes = {col: str(df[col].dtype) for col in non_numeric}
            raise TypeError(
                "Covariate columns must be numeric. "
                f"Non-numeric columns: {dtypes}"
            )

        nan_counts = df[self.covariates].isna().sum()
        if (nan_counts > 0).any():
            missing = nan_counts[nan_counts > 0]
            details = ", ".join(f"{col}={int(count)}" for col, count in missing.items())
            raise ValueError(
                "Covariate columns contain NaN values: "
                f"{details}"
            )

        linear_pred = np.zeros(len(df), dtype=float)
        for col in self.covariates:
            beta = self.coefficients[col]
            linear_pred += beta * df[col].to_numpy(dtype=float, copy=False)

        linear_pred = np.clip(linear_pred, -500.0, 500.0)
        risk = np.exp(linear_pred)

        return risk

    def failure_rate(self, age: np.ndarray, **kwargs) -> np.ndarray:
        """Delegate failure rate to baseline model.

        Scaling by covariates occurs in transform().
        """
        return self.baseline.failure_rate(age, **kwargs)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add covariate-scaled failure rate and probability columns.

        Parameters
        ----------
        df : pd.DataFrame
            Portfolio DataFrame with age, type, and covariate columns.

        Returns
        -------
        pd.DataFrame
            Copy of input with updated failure_rate and failure_probability.
        """
        result = self.baseline.transform(df)
        risk = self._risk_score(df)

        result['failure_rate'] = result['failure_rate'].to_numpy() * risk

        survival_baseline = 1.0 - result['failure_probability'].to_numpy()
        survival_adjusted = np.power(survival_baseline, risk)
        result['failure_probability'] = 1.0 - survival_adjusted
        result['failure_probability'] = np.clip(result['failure_probability'], 0.0, 1.0)

        return result

    def calculate_conditional_probability(self, state: pd.DataFrame) -> np.ndarray:
        """Calculate conditional one-step failure probabilities.

        For Weibull baselines this is exact:
        ``P = 1 - (S0(t+1)/S0(t)) ** risk``.

        For non-Weibull baselines this applies the same survival-ratio power
        form using baseline conditional probabilities as an approximation.

        Parameters
        ----------
        state : pd.DataFrame
            Current asset state with baseline-required columns and covariates.

        Returns
        -------
        np.ndarray
            Conditional failure probabilities for each asset.
        """
        risk = self._risk_score(state)
        baseline_probs = np.asarray(
            self.baseline.calculate_conditional_probability(state),
            dtype=float,
        )
        baseline_probs = np.clip(baseline_probs, 0.0, 1.0)

        survival_ratio = 1.0 - baseline_probs
        probs = 1.0 - np.power(survival_ratio, risk)
        return np.clip(probs, 0.0, 1.0)

    def __repr__(self) -> str:
        """Return informative string representation."""
        return (
            "ProportionalHazardsModel("
            f"baseline={self.baseline.__class__.__name__}, "
            f"covariates={self.covariates})"
        )
