"""Tests for DataFrameRepository planner data access."""

import pandas as pd
import pandas.testing as pdt
import pytest

from asset_optimization.protocols import AssetRepository
from asset_optimization.repositories import DataFrameRepository


def test_dataframe_repository_matches_asset_repository_protocol() -> None:
    """DataFrameRepository structurally satisfies AssetRepository."""
    repository = DataFrameRepository(assets=pd.DataFrame({"asset_id": ["A1"]}))
    assert isinstance(repository, AssetRepository)


def test_dataframe_repository_requires_assets_dataframe() -> None:
    """assets is the only required constructor input."""
    with pytest.raises(TypeError, match="assets must be a pandas DataFrame"):
        DataFrameRepository(assets=None)  # type: ignore[arg-type]


def test_dataframe_repository_returns_defensive_copies() -> None:
    """Load methods return copies and do not mutate stored data."""
    assets = pd.DataFrame({"asset_id": ["A1"], "asset_type": ["pipe"]})
    events = pd.DataFrame(
        {
            "asset_id": ["A1", "A1"],
            "event_type": ["break", "inspection"],
        }
    )
    interventions = pd.DataFrame({"asset_id": ["A1"], "action_type": ["repair"]})
    outcomes = pd.DataFrame({"asset_id": ["A1"], "observed_loss": [1000.0]})
    covariates = pd.DataFrame({"asset_id": ["A1"], "traffic_index": [0.9]})
    topology = pd.DataFrame({"from_asset_id": ["A1"], "to_asset_id": ["A2"]})

    repository = DataFrameRepository(
        assets=assets,
        events=events,
        interventions=interventions,
        outcomes=outcomes,
        covariates=covariates,
        topology=topology,
    )

    loaded_assets = repository.load_assets()
    loaded_assets.loc[0, "asset_type"] = "road"

    loaded_interventions = repository.load_interventions()
    loaded_interventions.loc[0, "action_type"] = "replace"

    pdt.assert_frame_equal(
        repository.load_assets(),
        pd.DataFrame({"asset_id": ["A1"], "asset_type": ["pipe"]}),
    )
    pdt.assert_frame_equal(
        repository.load_interventions(),
        pd.DataFrame({"asset_id": ["A1"], "action_type": ["repair"]}),
    )


def test_load_events_filters_on_event_type_when_column_exists() -> None:
    """load_events applies optional event_type filter."""
    events = pd.DataFrame(
        {
            "asset_id": ["A1", "A2", "A3"],
            "event_type": ["break", "inspection", "break"],
        }
    )
    repository = DataFrameRepository(
        assets=pd.DataFrame({"asset_id": ["A1"]}), events=events
    )

    filtered = repository.load_events(event_type="break")

    pdt.assert_frame_equal(
        filtered.reset_index(drop=True),
        pd.DataFrame(
            {
                "asset_id": ["A1", "A3"],
                "event_type": ["break", "break"],
            }
        ),
    )


def test_optional_tables_default_to_empty_dataframes() -> None:
    """Optional repository tables default to empty DataFrames."""
    repository = DataFrameRepository(assets=pd.DataFrame({"asset_id": ["A1"]}))

    assert repository.load_events().empty
    assert repository.load_interventions().empty
    assert repository.load_outcomes().empty
    assert repository.load_covariates().empty
    assert repository.load_topology().empty
