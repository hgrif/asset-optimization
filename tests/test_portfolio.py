"""Tests for Portfolio class loading and querying."""

import pytest
import pandas as pd
from asset_optimization import Portfolio


def test_load_from_csv(valid_csv_path):
    """Test loading portfolio from CSV file."""
    portfolio = Portfolio.from_csv(valid_csv_path)
    assert len(portfolio) > 0, "Portfolio should have assets"
    assert len(portfolio) == 12, "Portfolio should have 12 assets from fixture"


def test_load_from_excel(valid_excel_path):
    """Test loading portfolio from Excel file."""
    portfolio = Portfolio.from_excel(valid_excel_path)
    assert len(portfolio) > 0, "Portfolio should have assets"
    assert len(portfolio) == 12, "Portfolio should have 12 assets from fixture"


def test_load_from_dataframe(sample_dataframe):
    """Test loading portfolio from in-memory DataFrame."""
    portfolio = Portfolio.from_dataframe(sample_dataframe)
    assert len(portfolio) > 0, "Portfolio should have assets"
    assert len(portfolio) == 3, "Portfolio should have 3 assets from sample DataFrame"


def test_len_returns_asset_count(valid_csv_path):
    """Test that len(portfolio) returns asset count."""
    portfolio = Portfolio.from_csv(valid_csv_path)
    assert len(portfolio) == 12, "len() should match row count"
    assert isinstance(len(portfolio), int), "len() should return int"


def test_getitem_returns_asset(valid_csv_path):
    """Test dict-like access with portfolio['PIPE-001']."""
    portfolio = Portfolio.from_csv(valid_csv_path)
    asset = portfolio['PIPE-001']
    assert isinstance(asset, pd.Series), "Should return pandas Series"
    assert asset['asset_id'] == 'PIPE-001', "Should return correct asset"
    assert asset['asset_type'] == 'pipe', "Should have asset_type field"
    assert asset['material'] == 'PVC', "Should have material field"


def test_getitem_missing_raises_keyerror(valid_csv_path):
    """Test that accessing non-existent asset raises KeyError."""
    portfolio = Portfolio.from_csv(valid_csv_path)
    with pytest.raises(KeyError, match="NONEXISTENT"):
        _ = portfolio['NONEXISTENT']


def test_repr_is_informative(valid_csv_path):
    """Test that repr contains useful information."""
    portfolio = Portfolio.from_csv(valid_csv_path)
    repr_str = repr(portfolio)
    assert 'Portfolio' in repr_str, "repr should contain 'Portfolio'"
    assert '12' in repr_str, "repr should contain asset count"
    assert 'pipe' in repr_str or 'valve' in repr_str, "repr should mention asset types"


def test_data_property_returns_dataframe(valid_csv_path):
    """Test that .data property returns DataFrame."""
    portfolio = Portfolio.from_csv(valid_csv_path)
    df = portfolio.data
    assert isinstance(df, pd.DataFrame), "data property should return DataFrame"
    assert len(df) == len(portfolio), "DataFrame should have same length as portfolio"
    assert 'asset_id' in df.columns, "DataFrame should have asset_id column"


def test_asset_types_property(valid_csv_path):
    """Test that asset_types returns list of unique types."""
    portfolio = Portfolio.from_csv(valid_csv_path)
    types = portfolio.asset_types
    assert isinstance(types, list), "asset_types should return list"
    assert len(types) > 0, "Should have at least one asset type"
    assert 'pipe' in types, "Should include 'pipe' type"
    assert 'valve' in types, "Should include 'valve' type"
    assert len(types) == len(set(types)), "Types should be unique"


def test_mean_age_property(valid_csv_path):
    """Test that mean_age returns float greater than 0."""
    portfolio = Portfolio.from_csv(valid_csv_path)
    mean_age = portfolio.mean_age
    assert isinstance(mean_age, float), "mean_age should return float"
    assert mean_age > 0, "mean_age should be positive"
    # Assets from 2010-2023, so mean should be roughly 3-13 years
    assert 1 < mean_age < 20, "mean_age should be reasonable for test data"


def test_age_distribution_property(valid_csv_path):
    """Test that age_distribution returns Series with same length."""
    portfolio = Portfolio.from_csv(valid_csv_path)
    ages = portfolio.age_distribution
    assert isinstance(ages, pd.Series), "age_distribution should return Series"
    assert len(ages) == len(portfolio), "Should have one age per asset"
    assert (ages > 0).all(), "All ages should be positive"


def test_oldest_property(valid_csv_path):
    """Test that oldest returns Series with oldest asset."""
    portfolio = Portfolio.from_csv(valid_csv_path)
    oldest = portfolio.oldest
    assert isinstance(oldest, pd.Series), "oldest should return Series"
    assert 'asset_id' in oldest.index, "Should have asset_id field"
    assert oldest['asset_id'] == 'PIPE-001', "PIPE-001 is oldest (2010-03-15)"


def test_newest_property(valid_csv_path):
    """Test that newest returns Series with newest asset."""
    portfolio = Portfolio.from_csv(valid_csv_path)
    newest = portfolio.newest
    assert isinstance(newest, pd.Series), "newest should return Series"
    assert 'asset_id' in newest.index, "Should have asset_id field"
    assert newest['asset_id'] == 'PIPE-007', "PIPE-007 is newest (2023-05-12)"
