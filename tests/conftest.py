"""Shared pytest fixtures for test suite."""

import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def fixtures_dir():
    """Return path to fixtures directory."""
    return Path(__file__).parent / 'fixtures'


@pytest.fixture
def valid_csv_path(fixtures_dir):
    """Path to valid portfolio CSV file."""
    return fixtures_dir / 'valid_portfolio.csv'


@pytest.fixture
def valid_excel_path(fixtures_dir):
    """Path to valid portfolio Excel file."""
    return fixtures_dir / 'valid_portfolio.xlsx'


@pytest.fixture
def invalid_missing_field_path(fixtures_dir):
    """Path to CSV missing required asset_id column."""
    return fixtures_dir / 'invalid_missing_field.csv'


@pytest.fixture
def invalid_future_date_path(fixtures_dir):
    """Path to CSV with future install_date."""
    return fixtures_dir / 'invalid_future_date.csv'


@pytest.fixture
def invalid_duplicate_id_path(fixtures_dir):
    """Path to CSV with duplicate asset_id values."""
    return fixtures_dir / 'invalid_duplicate_id.csv'


@pytest.fixture
def sample_dataframe():
    """In-memory DataFrame with valid portfolio data."""
    return pd.DataFrame({
        'asset_id': ['PIPE-001', 'PIPE-002', 'VALVE-001'],
        'install_date': pd.to_datetime(['2010-01-15', '2015-06-20', '2018-03-10']),
        'asset_type': ['pipe', 'pipe', 'valve'],
        'material': ['PVC', 'Cast Iron', 'Brass'],
        'diameter_mm': pd.array([150, 200, 100], dtype='Int64'),
        'length_m': [100.5, 50.3, 1.2],
        'condition_score': [85.0, 72.5, 90.0],
    })
