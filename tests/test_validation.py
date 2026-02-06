"""Tests for Portfolio validation errors."""

import pandas as pd
import pytest
from asset_optimization import ValidationError
from asset_optimization.portfolio import validate_portfolio


def test_missing_required_field_raises_validation_error(invalid_missing_field_path):
    """Test that missing asset_id column raises ValidationError."""
    df = pd.read_csv(invalid_missing_field_path, parse_dates=["install_date"])
    with pytest.raises(ValidationError) as exc_info:
        validate_portfolio(df)

    error = exc_info.value
    assert error.field is not None, "ValidationError should have field attribute"
    assert error.message is not None, "ValidationError should have message attribute"


def test_future_date_raises_validation_error(invalid_future_date_path):
    """Test that install_date in future raises ValidationError."""
    df = pd.read_csv(invalid_future_date_path, parse_dates=["install_date"])
    with pytest.raises(ValidationError) as exc_info:
        validate_portfolio(df)

    error = exc_info.value
    assert "install_date" in str(error).lower() or "date" in error.field.lower(), (
        "Error should mention install_date field"
    )


def test_duplicate_id_raises_validation_error(invalid_duplicate_id_path):
    """Test that duplicate asset_id values raise ValidationError."""
    df = pd.read_csv(invalid_duplicate_id_path, parse_dates=["install_date"])
    with pytest.raises(ValidationError) as exc_info:
        validate_portfolio(df)

    error = exc_info.value
    assert "asset_id" in str(error).lower() or "asset_id" in error.field.lower(), (
        "Error should mention asset_id field"
    )


def test_validation_error_has_field_attribute(invalid_missing_field_path):
    """Test that ValidationError has field attribute set."""
    df = pd.read_csv(invalid_missing_field_path, parse_dates=["install_date"])
    with pytest.raises(ValidationError) as exc_info:
        validate_portfolio(df)

    error = exc_info.value
    assert hasattr(error, "field"), "ValidationError should have 'field' attribute"
    assert isinstance(error.field, str), "field attribute should be string"


def test_validation_error_has_message_attribute(invalid_missing_field_path):
    """Test that ValidationError has message attribute set."""
    df = pd.read_csv(invalid_missing_field_path, parse_dates=["install_date"])
    with pytest.raises(ValidationError) as exc_info:
        validate_portfolio(df)

    error = exc_info.value
    assert hasattr(error, "message"), "ValidationError should have 'message' attribute"
    assert isinstance(error.message, str), "message attribute should be string"
    assert len(error.message) > 0, "message should not be empty"


def test_validation_error_has_details_attribute(invalid_missing_field_path):
    """Test that ValidationError has details attribute set."""
    df = pd.read_csv(invalid_missing_field_path, parse_dates=["install_date"])
    with pytest.raises(ValidationError) as exc_info:
        validate_portfolio(df)

    error = exc_info.value
    assert hasattr(error, "details"), "ValidationError should have 'details' attribute"
    assert isinstance(error.details, dict), "details attribute should be dict"
