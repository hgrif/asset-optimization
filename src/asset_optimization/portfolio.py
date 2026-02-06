"""Portfolio helpers for validating asset data."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from asset_optimization.exceptions import ValidationError
from asset_optimization.quality import QualityMetrics
from asset_optimization.schema import portfolio_schema


def validate_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """Validate portfolio data with the Pandera schema.

    Parameters
    ----------
    df : pd.DataFrame
        Portfolio data.

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


def compute_quality_metrics(df: pd.DataFrame) -> QualityMetrics:
    """Compute quality metrics for portfolio data.

    Parameters
    ----------
    df : pd.DataFrame
        Portfolio data.

    Returns
    -------
    QualityMetrics
        Data quality metrics.
    """
    completeness = df.notna().mean()
    missing_counts = df.isna().sum()
    return QualityMetrics(
        completeness=completeness,
        missing_counts=missing_counts,
        total_rows=len(df),
    )


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
