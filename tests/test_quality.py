"""Tests for Portfolio quality metrics."""

import pytest
import pandas as pd
from asset_optimization import Portfolio


def test_quality_property_returns_metrics(valid_csv_path):
    """Test that portfolio.quality returns QualityMetrics instance."""
    portfolio = Portfolio.from_csv(valid_csv_path)
    quality = portfolio.quality
    # Check it has the expected attributes
    assert hasattr(quality, 'completeness'), "QualityMetrics should have completeness"
    assert hasattr(quality, 'missing_counts'), "QualityMetrics should have missing_counts"
    assert hasattr(quality, 'total_rows'), "QualityMetrics should have total_rows"


def test_completeness_is_series(valid_csv_path):
    """Test that quality.completeness is pd.Series."""
    portfolio = Portfolio.from_csv(valid_csv_path)
    completeness = portfolio.quality.completeness
    assert isinstance(completeness, pd.Series), "completeness should be pd.Series"
    assert len(completeness) > 0, "completeness should have entries"


def test_missing_counts_is_series(valid_csv_path):
    """Test that quality.missing_counts is pd.Series."""
    portfolio = Portfolio.from_csv(valid_csv_path)
    missing_counts = portfolio.quality.missing_counts
    assert isinstance(missing_counts, pd.Series), "missing_counts should be pd.Series"
    assert len(missing_counts) > 0, "missing_counts should have entries"


def test_total_rows_matches_len(valid_csv_path):
    """Test that quality.total_rows matches len(portfolio)."""
    portfolio = Portfolio.from_csv(valid_csv_path)
    assert portfolio.quality.total_rows == len(portfolio), \
        "total_rows should match portfolio length"


def test_completeness_values_between_0_and_1(valid_csv_path):
    """Test that completeness values are between 0 and 1."""
    portfolio = Portfolio.from_csv(valid_csv_path)
    completeness = portfolio.quality.completeness
    assert (completeness >= 0).all(), "completeness should be >= 0"
    assert (completeness <= 1).all(), "completeness should be <= 1"


def test_quality_repr_not_empty(valid_csv_path):
    """Test that repr(quality) is non-empty string."""
    portfolio = Portfolio.from_csv(valid_csv_path)
    quality = portfolio.quality
    repr_str = repr(quality)
    assert isinstance(repr_str, str), "repr should return string"
    assert len(repr_str) > 0, "repr should not be empty"


def test_quality_repr_html_for_notebook(valid_csv_path):
    """Test that _repr_html_() returns HTML string for notebooks."""
    portfolio = Portfolio.from_csv(valid_csv_path)
    quality = portfolio.quality
    html = quality._repr_html_()
    assert isinstance(html, str), "_repr_html_() should return string"
    assert len(html) > 0, "_repr_html_() should not be empty"
    # Basic check that it looks like HTML
    assert '<' in html and '>' in html, "_repr_html_() should contain HTML tags"
