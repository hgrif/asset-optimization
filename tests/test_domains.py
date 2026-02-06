"""Tests for domain protocol and pipe domain implementation."""

import pandas as pd
import pytest

from asset_optimization import ValidationError
from asset_optimization.domains import Domain, PipeDomain
from asset_optimization.models import WeibullModel
from asset_optimization.simulation import (
    DO_NOTHING,
    INSPECT,
    REPAIR,
    REPLACE,
    InterventionType,
)


class TestDomainProtocol:
    """Tests for Domain protocol compliance."""

    def test_pipe_domain_satisfies_protocol(self):
        """PipeDomain should satisfy the Domain protocol."""
        assert isinstance(PipeDomain(), Domain)


class TestPipeDomainValidate:
    """Tests for PipeDomain validation."""

    @pytest.fixture
    def valid_df(self):
        """Valid pipe DataFrame fixture."""
        return pd.DataFrame(
            {
                "asset_id": ["P1", "P2"],
                "install_date": [
                    pd.Timestamp("2010-01-01"),
                    pd.Timestamp("2012-06-15"),
                ],
                "asset_type": ["pipe", "pipe"],
                "material": ["PVC", "Cast Iron"],
            }
        )

    def test_validate_accepts_valid_df(self, valid_df):
        """Validate should accept valid pipe data."""
        validated = PipeDomain().validate(valid_df)
        assert isinstance(validated, pd.DataFrame)
        assert set(validated.columns) >= {
            "asset_id",
            "install_date",
            "asset_type",
            "material",
        }

    def test_validate_missing_required_column_raises(self, valid_df):
        """Missing required columns should raise ValidationError."""
        df = valid_df.drop(columns=["asset_id"])
        with pytest.raises(ValidationError):
            PipeDomain().validate(df)

    def test_validate_allows_extra_columns(self, valid_df):
        """Extra columns should be allowed when strict=False."""
        df = valid_df.assign(extra_column=123)
        validated = PipeDomain().validate(df)
        assert "extra_column" in validated.columns


class TestPipeDomainDefaultInterventions:
    """Tests for PipeDomain default interventions."""

    def test_default_interventions_keys(self):
        """Default interventions include expected keys."""
        interventions = PipeDomain().default_interventions()
        assert set(interventions.keys()) == {
            "do_nothing",
            "inspect",
            "repair",
            "replace",
        }

    def test_default_interventions_types(self):
        """Default interventions are InterventionType instances."""
        interventions = PipeDomain().default_interventions()
        for intervention in interventions.values():
            assert isinstance(intervention, InterventionType)

    def test_default_interventions_costs(self):
        """Default interventions have expected costs."""
        interventions = PipeDomain().default_interventions()
        assert interventions["do_nothing"].cost == 0.0
        assert interventions["inspect"].cost == 500.0
        assert interventions["repair"].cost == 5000.0
        assert interventions["replace"].cost == 50000.0

    def test_default_interventions_are_constants(self):
        """Default interventions should return the existing constants."""
        interventions = PipeDomain().default_interventions()
        assert interventions["do_nothing"] is DO_NOTHING
        assert interventions["inspect"] is INSPECT
        assert interventions["repair"] is REPAIR
        assert interventions["replace"] is REPLACE


class TestPipeDomainDefaultModel:
    """Tests for PipeDomain default model."""

    def test_default_model_instance(self):
        """Default model should be a WeibullModel."""
        model = PipeDomain().default_model()
        assert isinstance(model, WeibullModel)

    def test_default_model_params(self):
        """Default model uses standard pipe parameters."""
        model = PipeDomain().default_model()
        assert model.params["PVC"] == (2.5, 50.0)
        assert model.params["Cast Iron"] == (3.0, 40.0)

    def test_default_model_type_column(self):
        """Default model should use material as the type column."""
        model = PipeDomain().default_model()
        assert model.type_column == "material"
