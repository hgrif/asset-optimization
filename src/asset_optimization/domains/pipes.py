"""Pipe domain implementation."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from asset_optimization.exceptions import ValidationError
from asset_optimization.models import DeteriorationModel, WeibullModel
from asset_optimization.schema import portfolio_schema
from asset_optimization.simulation import (
    InterventionType,
    DO_NOTHING,
    INSPECT,
    REPAIR,
    REPLACE,
)


class PipeDomain:
    """Domain implementation for pipe assets."""

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate pipe portfolio data using the shared schema.

        Parameters
        ----------
        df : pd.DataFrame
            Portfolio data to validate.

        Returns
        -------
        pd.DataFrame
            Validated DataFrame (copy) with coerced types.

        Raises
        ------
        ValidationError
            If data fails schema validation.
        """
        try:
            validated_df = portfolio_schema.validate(df.copy(), lazy=False)
        except pa.errors.SchemaError as exc:
            raise _schema_error_to_validation_error(exc) from exc
        return validated_df

    def default_interventions(self) -> dict[str, InterventionType]:
        """Return the default pipe interventions.

        Returns
        -------
        dict[str, InterventionType]
            Mapping of intervention keys to definitions.
        """
        return {
            "do_nothing": DO_NOTHING,
            "inspect": INSPECT,
            "repair": REPAIR,
            "replace": REPLACE,
        }

    def default_model(self) -> DeteriorationModel:
        """Return the default pipe deterioration model.

        Returns
        -------
        DeteriorationModel
            Weibull model configured with pipe parameters.
        """
        return WeibullModel({"PVC": (2.5, 50.0), "Cast Iron": (3.0, 40.0)})


def _schema_error_to_validation_error(exc: pa.errors.SchemaError) -> ValidationError:
    """Convert a Pandera SchemaError to a ValidationError."""
    failure_cases = getattr(exc, "failure_cases", None)

    if isinstance(failure_cases, pd.DataFrame) and not failure_cases.empty:
        first_failure = failure_cases.iloc[0]
        field = first_failure["column"] if "column" in first_failure else "schema"
        message = first_failure["check"] if "check" in first_failure else str(exc)

        details = {}
        if "failure_case" in failure_cases.columns:
            details["failing_values"] = failure_cases["failure_case"].tolist()[:5]
            details["total_failures"] = len(failure_cases)
        else:
            details["total_failures"] = len(failure_cases)

        return ValidationError(field=field, message=message, details=details)

    return ValidationError(field="schema", message=str(exc), details={})
