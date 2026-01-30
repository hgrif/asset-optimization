"""Abstract base class for deterioration models."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class DeteriorationModel(ABC):
    """Abstract base for deterioration models.

    All deterioration models must implement:
    - failure_rate(): Calculate instantaneous hazard function
    - transform(): Add failure metrics to portfolio DataFrame

    Models accept full portfolio DataFrames and operate vectorized per asset type.
    The transform() method returns a copy with new columns added (immutable pattern).

    Subclasses should:
    1. Accept parameters at __init__
    2. Validate parameters in __init__ (fail fast)
    3. Implement vectorized calculations for performance

    Examples
    --------
    >>> class CustomModel(DeteriorationModel):
    ...     def __init__(self, params):
    ...         self.params = params
    ...
    ...     def failure_rate(self, age, **kwargs):
    ...         return np.zeros_like(age)  # Custom calculation
    ...
    ...     def transform(self, df):
    ...         result = df.copy()
    ...         result['failure_rate'] = self.failure_rate(df['age'].values)
    ...         result['failure_probability'] = 0.0  # Custom calculation
    ...         return result
    """

    @abstractmethod
    def failure_rate(self, age: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate failure rate (hazard function) at given ages.

        Parameters
        ----------
        age : np.ndarray
            Asset ages in years (vectorized input).
        **kwargs : dict
            Model-specific parameters (e.g., shape, scale for Weibull).

        Returns
        -------
        rates : np.ndarray
            Failure rates (instantaneous hazard function values).
            Same shape as input age array.

        Notes
        -----
        Hazard function h(t) represents instantaneous failure rate.
        For Weibull: h(t) = (k/lambda) * (t/lambda)^(k-1)
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add failure rate and probability columns to portfolio DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Portfolio DataFrame with at minimum an age column and
            a type column for parameter lookup.

        Returns
        -------
        enriched : pd.DataFrame
            Copy of input with two new columns added:
            - failure_rate: instantaneous hazard h(t)
            - failure_probability: cumulative probability F(t)

        Raises
        ------
        ValueError
            If required columns missing or asset types not in parameters.
        TypeError
            If age column is not numeric.

        Notes
        -----
        This method MUST return a copy of the input DataFrame.
        The original DataFrame must not be modified (immutable pattern).
        This aligns with pandas copy-on-write behavior in pandas 2.0+.
        """
        pass
