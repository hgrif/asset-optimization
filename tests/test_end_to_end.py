"""End-to-end tests for simulation snapshot stability."""

import pandas as pd

from asset_optimization import Portfolio
from asset_optimization.models import WeibullModel
from asset_optimization.simulation import SimulationConfig, Simulator


# Snapshot values captured on 2026-02-05 from a seeded run.
# If simulation logic intentionally changes, update these snapshots.
EXPECTED_SUMMARY = pd.DataFrame(
    [
        {
            'year': 2026,
            'total_cost': 0.0,
            'failure_count': 0,
            'intervention_count': 0,
            'avg_age': 14.723134839151266,
        },
        {
            'year': 2027,
            'total_cost': 0.0,
            'failure_count': 0,
            'intervention_count': 0,
            'avg_age': 15.723134839151266,
        },
        {
            'year': 2028,
            'total_cost': 0.0,
            'failure_count': 0,
            'intervention_count': 0,
            'avg_age': 16.723134839151264,
        },
        {
            'year': 2029,
            'total_cost': 65000.0,
            'failure_count': 1,
            'intervention_count': 1,
            'avg_age': 14.654688569472965,
        },
        {
            'year': 2030,
            'total_cost': 0.0,
            'failure_count': 0,
            'intervention_count': 0,
            'avg_age': 15.654688569472965,
        },
        {
            'year': 2031,
            'total_cost': 0.0,
            'failure_count': 0,
            'intervention_count': 0,
            'avg_age': 16.654688569472967,
        },
    ],
    columns=['year', 'total_cost', 'failure_count', 'intervention_count', 'avg_age'],
)

EXPECTED_FAILURE_LOG = pd.DataFrame(
    [
        {
            'year': 2029,
            'asset_id': 'PIPE-002',
            'age_at_failure': 24.54757015742642,
            'material': 'PVC',
            'direct_cost': 10000.0,
            'consequence_cost': 5000.0,
        },
    ],
    columns=[
        'year',
        'asset_id',
        'age_at_failure',
        'material',
        'direct_cost',
        'consequence_cost',
    ],
)


def test_end_to_end_simulation_snapshot(end_to_end_dataframe):
    """Snapshot a specific run to guard against refactor regressions."""
    portfolio = Portfolio.from_dataframe(end_to_end_dataframe)
    model = WeibullModel({
        'PVC': (2.5, 50.0),
        'Cast Iron': (3.0, 40.0),
    })
    config = SimulationConfig(n_years=6, random_seed=123)

    result = Simulator(model, config).run(portfolio)

    pd.testing.assert_frame_equal(result.summary, EXPECTED_SUMMARY)
    pd.testing.assert_frame_equal(result.failure_log, EXPECTED_FAILURE_LOG)
    assert result.total_cost() == float(EXPECTED_SUMMARY['total_cost'].sum())
    assert result.total_failures() == int(EXPECTED_SUMMARY['failure_count'].sum())
