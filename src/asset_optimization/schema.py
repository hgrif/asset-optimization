"""Pandera schema for portfolio validation."""

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check


# Define portfolio schema with required and optional columns
portfolio_schema = DataFrameSchema(
    columns={
        # Required columns
        'asset_id': Column(
            str,
            unique=True,
            nullable=False,
        ),
        'install_date': Column(
            pd.Timestamp,
            nullable=False,
            checks=Check.less_than_or_equal_to(pd.Timestamp.now()),
        ),
        'asset_type': Column(
            str,
            nullable=False,
        ),
        'material': Column(
            str,
            nullable=False,
        ),
        # Optional columns
        'diameter_mm': Column(
            'Int64',
            nullable=True,
            checks=Check.greater_than(0),
            required=False,
        ),
        'length_m': Column(
            float,
            nullable=True,
            checks=Check.greater_than(0),
            required=False,
        ),
        'condition_score': Column(
            float,
            nullable=True,
            checks=Check.in_range(0, 100),
            required=False,
        ),
    },
    strict=False,  # Allow extra columns
    coerce=True,   # Try to coerce types before validation
)
