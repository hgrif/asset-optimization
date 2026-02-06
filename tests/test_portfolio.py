"""Tests for portfolio helper functions."""

import pandas as pd
import pytest

from asset_optimization import ValidationError
from asset_optimization.portfolio import validate_portfolio, compute_quality_metrics


def test_load_from_csv(valid_csv_path):
    """CSV read + validation returns DataFrame."""
    df = pd.read_csv(valid_csv_path, parse_dates=["install_date"])
    df = validate_portfolio(df)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 12
    assert "asset_id" in df.columns


def test_load_from_excel(valid_excel_path):
    """Excel read + validation returns DataFrame."""
    df = pd.read_excel(
        valid_excel_path, engine="openpyxl", parse_dates=["install_date"]
    )
    df = validate_portfolio(df)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 12
    assert "asset_id" in df.columns


def test_validate_dataframe(sample_dataframe):
    """validate_portfolio returns validated DataFrame."""
    df = validate_portfolio(sample_dataframe)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "install_date" in df.columns


def test_validate_missing_field_raises(invalid_missing_field_path):
    """Missing required columns raise ValidationError."""
    df = pd.read_csv(invalid_missing_field_path, parse_dates=["install_date"])
    with pytest.raises(ValidationError):
        validate_portfolio(df)


def test_compute_quality_metrics(valid_csv_path):
    """Quality metrics computed from validated data."""
    df = pd.read_csv(valid_csv_path, parse_dates=["install_date"])
    df = validate_portfolio(df)
    metrics = compute_quality_metrics(df)
    assert hasattr(metrics, "completeness")
    assert hasattr(metrics, "missing_counts")
    assert hasattr(metrics, "total_rows")
    assert metrics.total_rows == len(df)
