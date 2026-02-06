"""Pandera schema for portfolio validation."""

import pandas as pd
import pandera.pandas as pa


# Define portfolio schema with required and optional columns
portfolio_schema = pa.DataFrameSchema(
    columns={
        # Required columns
        "asset_id": pa.Column(
            str,
            unique=True,
            nullable=False,
        ),
        "install_date": pa.Column(
            pd.Timestamp,
            nullable=False,
            checks=pa.Check.less_than_or_equal_to(pd.Timestamp.now()),
        ),
        "asset_type": pa.Column(
            str,
            nullable=False,
        ),
        "material": pa.Column(
            str,
            nullable=False,
        ),
        # Optional columns
        "diameter_mm": pa.Column(
            "Int64",
            nullable=True,
            checks=pa.Check.greater_than(0),
            required=False,
        ),
        "length_m": pa.Column(
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
    strict=False,  # Allow extra columns
    coerce=True,  # Try to coerce types before validation
)
