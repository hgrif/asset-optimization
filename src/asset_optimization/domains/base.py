"""Domain protocol definitions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from asset_optimization.models import DeteriorationModel
from asset_optimization.simulation import InterventionType


@runtime_checkable
class Domain(Protocol):
    """Protocol for asset domains.

    Domains encapsulate validation, default interventions, and default
    deterioration model behavior for a specific asset class.
    """

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate a domain-specific DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input data to validate.

        Returns
        -------
        pd.DataFrame
            Validated DataFrame (copy) with coerced types.

        Raises
        ------
        ValidationError
            If validation fails for the provided data.
        """

    def default_interventions(self) -> dict[str, InterventionType]:
        """Return default interventions for the domain.

        Returns
        -------
        dict[str, InterventionType]
            Mapping of intervention keys to definitions.
        """

    def default_model(self) -> DeteriorationModel:
        """Return the default deterioration model for the domain.

        Returns
        -------
        DeteriorationModel
            Configured deterioration model instance.
        """
