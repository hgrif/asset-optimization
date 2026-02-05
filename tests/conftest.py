"""Shared pytest fixtures for test suite."""

import pandas as pd
import pytest
from pathlib import Path

from asset_optimization.portfolio import validate_portfolio
from asset_optimization.models import WeibullModel
from asset_optimization.simulation import SimulationConfig


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


@pytest.fixture
def end_to_end_dataframe():
    """Small DataFrame for end-to-end deterministic simulation tests."""
    return pd.DataFrame({
        'asset_id': [
            'PIPE-001', 'PIPE-002', 'PIPE-003', 'PIPE-004',
            'VALVE-001', 'VALVE-002', 'VALVE-003', 'VALVE-004',
        ],
        'install_date': pd.to_datetime([
            '2000-01-01', '2005-06-15', '2010-03-10', '2012-07-20',
            '2014-09-01', '2016-11-05', '2018-02-28', '2020-05-12',
        ]),
        'asset_type': [
            'pipe', 'pipe', 'pipe', 'pipe',
            'valve', 'valve', 'valve', 'valve',
        ],
        'material': [
            'PVC', 'PVC', 'Cast Iron', 'Cast Iron',
            'PVC', 'Cast Iron', 'PVC', 'Cast Iron',
        ],
        'diameter_mm': pd.array([150, 200, 175, 225, 80, 90, 110, 120], dtype='Int64'),
        'length_m': [120.0, 80.0, 95.0, 110.0, 2.5, 3.0, 2.0, 2.8],
        'condition_score': [82.0, 75.5, 68.0, 70.0, 90.0, 88.0, 92.0, 85.0],
    })


@pytest.fixture
def sample_portfolio():
    """Create sample portfolio DataFrame for simulation tests."""
    n_assets = 100
    test_data = pd.DataFrame({
        'asset_id': [f'PIPE-{i:03d}' for i in range(n_assets)],
        'asset_type': ['pipe'] * n_assets,
        'material': ['PVC'] * 50 + ['Cast Iron'] * 50,
        'install_date': pd.date_range('2000-01-01', periods=n_assets, freq='30D'),
        'diameter_mm': [100] * n_assets,
        'length_m': [50.0] * n_assets,
        'condition_score': [80.0] * n_assets,
    })
    return validate_portfolio(test_data)


@pytest.fixture
def weibull_model():
    """Create WeibullModel for simulation tests."""
    return WeibullModel({
        'PVC': (2.5, 50.0),
        'Cast Iron': (3.0, 40.0),
    })


@pytest.fixture
def simulation_config():
    """Create default simulation config."""
    return SimulationConfig(n_years=5, random_seed=42)


@pytest.fixture
def optimization_portfolio():
    """Portfolio DataFrame with known ages for optimization testing."""
    df = pd.DataFrame({
        'asset_id': ['A1', 'A2', 'A3', 'A4', 'A5'],
        'asset_type': ['pipe', 'pipe', 'pipe', 'pipe', 'pipe'],
        'material': ['PVC', 'PVC', 'PVC', 'PVC', 'PVC'],
        'install_date': pd.to_datetime([
            '1980-01-01',  # ~45 years old, high risk
            '1990-01-01',  # ~35 years old, medium-high risk
            '2000-01-01',  # ~25 years old, medium risk
            '2010-01-01',  # ~15 years old, low risk
            '2020-01-01',  # ~5 years old, very low risk
        ]),
        'diameter_mm': [100, 100, 100, 100, 100],
        'length_m': [100.0, 100.0, 100.0, 100.0, 100.0],
    })
    validated = validate_portfolio(df)
    validated['age'] = (pd.Timestamp.now() - validated['install_date']).dt.days / 365.25
    return validated
