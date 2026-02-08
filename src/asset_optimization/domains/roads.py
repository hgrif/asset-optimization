"""Road domain implementation."""

from __future__ import annotations

from typing import Callable

import pandas as pd
import pandera.pandas as pa

from asset_optimization.exceptions import ValidationError
from asset_optimization.models import (
    DeteriorationModel,
    ProportionalHazardsModel,
    WeibullModel,
)
from asset_optimization.simulation import InterventionType

SURFACE_TYPES = ("asphalt", "concrete", "gravel")
TRAFFIC_LOAD_LEVELS = ("low", "medium", "high")
CLIMATE_ZONES = ("temperate", "cold", "hot_dry", "hot_humid")

_TRAFFIC_LOAD_ENCODING = {"low": 1, "medium": 2, "high": 3}
_CLIMATE_ZONE_ENCODING = {
    "temperate": 1,
    "cold": 2,
    "hot_dry": 3,
    "hot_humid": 4,
}

_ROAD_SCHEMA = pa.DataFrameSchema(
    columns={
        "asset_id": pa.Column(
            str,
            unique=True,
            nullable=False,
        ),
        "install_date": pa.Column(
            pd.Timestamp,
            nullable=False,
        ),
        "surface_type": pa.Column(
            str,
            nullable=False,
            checks=pa.Check.isin(SURFACE_TYPES),
        ),
        "traffic_load": pa.Column(
            str,
            nullable=False,
            checks=pa.Check.isin(TRAFFIC_LOAD_LEVELS),
        ),
        "climate_zone": pa.Column(
            str,
            nullable=False,
            checks=pa.Check.isin(CLIMATE_ZONES),
        ),
        "length_km": pa.Column(
            float,
            nullable=True,
            checks=pa.Check.greater_than(0),
            required=False,
        ),
        "condition_score": pa.Column(
            float,
            nullable=True,
            checks=pa.Check.in_range(0, 100),
            required=False,
        ),
    },
    strict=False,
    coerce=True,
)


class RoadDomain:
    """Domain implementation for road assets."""

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate road portfolio data using the road schema.

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
            validated_df = _ROAD_SCHEMA.validate(df.copy(), lazy=False)
        except pa.errors.SchemaError as exc:
            raise _schema_error_to_validation_error(exc) from exc
        return validated_df

    def default_interventions(
        self, surface_type: str = "asphalt"
    ) -> dict[str, InterventionType]:
        """Return default interventions for the specified surface type.

        Parameters
        ----------
        surface_type : str, default="asphalt"
            Road surface type to use for cost assumptions.

        Returns
        -------
        dict[str, InterventionType]
            Mapping of intervention keys to definitions.

        Raises
        ------
        ValueError
            If surface_type is not a supported road surface type.
        """
        if surface_type not in SURFACE_TYPES:
            raise ValueError(
                f"Unknown surface_type '{surface_type}'. Supported: {SURFACE_TYPES}"
            )

        if surface_type == "asphalt":
            inspect_cost = 500
            patch_cost = 15_000
            resurface_cost = 80_000
            reconstruct_cost = 250_000
            patch_reduction = 3.0
            resurface_reduction = 12.0
            upgrade_type = "asphalt"
        elif surface_type == "concrete":
            inspect_cost = 500
            patch_cost = 20_000
            resurface_cost = 120_000
            reconstruct_cost = 400_000
            patch_reduction = 3.0
            resurface_reduction = 15.0
            upgrade_type = "concrete"
        else:
            inspect_cost = 300
            patch_cost = 8_000
            resurface_cost = 40_000
            reconstruct_cost = 200_000
            patch_reduction = 2.0
            resurface_reduction = 8.0
            upgrade_type = "asphalt"

        def reduce_age(reduction: float) -> Callable[[float], float]:
            return lambda age: max(0.0, age - reduction)

        return {
            "do_nothing": InterventionType(
                name="DoNothing",
                cost=0.0,
                age_effect=lambda age: age,
            ),
            "inspect": InterventionType(
                name="Inspect",
                cost=float(inspect_cost),
                age_effect=lambda age: age,
            ),
            "patch": InterventionType(
                name="Patch",
                cost=float(patch_cost),
                age_effect=reduce_age(patch_reduction),
            ),
            "resurface": InterventionType(
                name="Resurface",
                cost=float(resurface_cost),
                age_effect=reduce_age(resurface_reduction),
            ),
            "reconstruct": InterventionType(
                name="Reconstruct",
                cost=float(reconstruct_cost),
                age_effect=lambda age: 0.0,
                upgrade_type=upgrade_type,
            ),
        }

    def default_model(self) -> DeteriorationModel:
        """Return the default road deterioration model.

        Returns
        -------
        DeteriorationModel
            Proportional hazards model configured for roads.
        """
        road_params = {
            "asphalt": (3.5, 20.0),
            "concrete": (2.8, 35.0),
            "gravel": (4.0, 12.0),
        }
        baseline = WeibullModel(road_params, type_column="surface_type")
        return ProportionalHazardsModel(
            baseline=baseline,
            covariates=["traffic_load_encoded", "climate_zone_encoded"],
            coefficients={
                "traffic_load_encoded": 0.35,
                "climate_zone_encoded": 0.15,
            },
        )

    @staticmethod
    def encode_covariates(df: pd.DataFrame) -> pd.DataFrame:
        """Encode traffic load and climate zone into numeric covariates.

        Parameters
        ----------
        df : pd.DataFrame
            Road portfolio data with traffic_load and climate_zone columns.

        Returns
        -------
        pd.DataFrame
            Copy of input with encoded covariate columns.

        Raises
        ------
        ValueError
            If unknown traffic_load or climate_zone values are encountered.
        """
        result = df.copy(deep=True)
        result["traffic_load_encoded"] = result["traffic_load"].map(
            _TRAFFIC_LOAD_ENCODING
        )
        result["climate_zone_encoded"] = result["climate_zone"].map(
            _CLIMATE_ZONE_ENCODING
        )

        _validate_encoded_values(result, "traffic_load", "traffic_load_encoded")
        _validate_encoded_values(result, "climate_zone", "climate_zone_encoded")

        return result


def _validate_encoded_values(df: pd.DataFrame, source: str, encoded: str) -> None:
    """Validate encoded covariate values for unknown categories."""
    if df[encoded].isna().any():
        unknown = df.loc[df[encoded].isna(), source].unique().tolist()
        raise ValueError(f"Unknown {source} values: {unknown}")


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
