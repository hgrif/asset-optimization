"""Tests for RoadDomain implementation."""

import pandas as pd
import pytest

from asset_optimization import ValidationError
from asset_optimization.domains import Domain, RoadDomain
from asset_optimization.models import ProportionalHazardsModel, WeibullModel
from asset_optimization.simulation import InterventionType


class TestRoadDomainProtocol:
    """Tests for RoadDomain protocol compliance."""

    def test_road_domain_satisfies_protocol(self):
        """RoadDomain should satisfy the Domain protocol."""
        assert isinstance(RoadDomain(), Domain)


class TestRoadDomainValidate:
    """Tests for RoadDomain validation."""

    @pytest.fixture
    def valid_df(self):
        """Valid road DataFrame fixture."""
        return pd.DataFrame(
            {
                "asset_id": ["R1", "R2"],
                "install_date": ["2010-01-01", "2015-06-15"],
                "surface_type": ["asphalt", "concrete"],
                "traffic_load": ["low", "high"],
                "climate_zone": ["temperate", "cold"],
            }
        )

    def test_validate_accepts_valid_df(self, valid_df):
        """Validate should accept valid road data."""
        validated = RoadDomain().validate(valid_df)
        assert isinstance(validated, pd.DataFrame)
        assert set(validated.columns) >= {
            "asset_id",
            "install_date",
            "surface_type",
            "traffic_load",
            "climate_zone",
        }

    def test_validate_missing_required_column_raises(self, valid_df):
        """Missing required columns should raise ValidationError."""
        df = valid_df.drop(columns=["surface_type"])
        with pytest.raises(ValidationError):
            RoadDomain().validate(df)

    def test_validate_invalid_surface_type_raises(self, valid_df):
        """Invalid surface_type values should raise ValidationError."""
        df = valid_df.copy()
        df.loc[0, "surface_type"] = "rubber"
        with pytest.raises(ValidationError):
            RoadDomain().validate(df)

    def test_validate_invalid_traffic_load_raises(self, valid_df):
        """Invalid traffic_load values should raise ValidationError."""
        df = valid_df.copy()
        df.loc[0, "traffic_load"] = "extreme"
        with pytest.raises(ValidationError):
            RoadDomain().validate(df)

    def test_validate_invalid_climate_zone_raises(self, valid_df):
        """Invalid climate_zone values should raise ValidationError."""
        df = valid_df.copy()
        df.loc[0, "climate_zone"] = "tropical"
        with pytest.raises(ValidationError):
            RoadDomain().validate(df)

    def test_validate_allows_extra_columns(self, valid_df):
        """Extra columns should be allowed when strict=False."""
        df = valid_df.assign(extra_column=123)
        validated = RoadDomain().validate(df)
        assert "extra_column" in validated.columns

    def test_validate_coerces_install_date(self, valid_df):
        """Validate should coerce install_date to Timestamp."""
        assert not pd.api.types.is_datetime64_any_dtype(valid_df["install_date"])
        validated = RoadDomain().validate(valid_df)
        assert pd.api.types.is_datetime64_any_dtype(validated["install_date"])

    def test_validate_returns_copy(self, valid_df):
        """Validate should return a copy and not mutate input."""
        validated = RoadDomain().validate(valid_df)
        validated.loc[0, "surface_type"] = "gravel"
        assert valid_df.loc[0, "surface_type"] == "asphalt"


class TestRoadDomainDefaultInterventions:
    """Tests for RoadDomain default interventions."""

    def test_default_interventions_keys(self):
        """Default interventions include expected keys."""
        interventions = RoadDomain().default_interventions()
        assert set(interventions.keys()) == {
            "do_nothing",
            "inspect",
            "patch",
            "resurface",
            "reconstruct",
        }

    def test_default_interventions_types(self):
        """Default interventions are InterventionType instances."""
        interventions = RoadDomain().default_interventions()
        for intervention in interventions.values():
            assert isinstance(intervention, InterventionType)

    def test_do_nothing_cost_is_zero(self):
        """Do nothing intervention should be zero cost."""
        interventions = RoadDomain().default_interventions()
        assert interventions["do_nothing"].cost == 0.0

    def test_reconstruct_has_upgrade_type(self):
        """Reconstruct intervention should have upgrade_type set."""
        interventions = RoadDomain().default_interventions()
        assert interventions["reconstruct"].upgrade_type is not None

    def test_surface_type_costs_vary(self):
        """Resurface and reconstruct costs should vary by surface type."""
        asphalt = RoadDomain().default_interventions(surface_type="asphalt")
        concrete = RoadDomain().default_interventions(surface_type="concrete")
        assert asphalt["resurface"].cost != concrete["resurface"].cost
        assert asphalt["reconstruct"].cost != concrete["reconstruct"].cost

    def test_patch_age_effect_reduces_age(self):
        """Patch should reduce age but not reset to zero."""
        interventions = RoadDomain().default_interventions(surface_type="asphalt")
        new_age = interventions["patch"].apply_age_effect(20.0)
        assert 0.0 < new_age < 20.0

    def test_resurface_age_effect_reduces_age(self):
        """Resurface should reduce age significantly but not reset to zero."""
        interventions = RoadDomain().default_interventions(surface_type="asphalt")
        patch_age = interventions["patch"].apply_age_effect(20.0)
        resurface_age = interventions["resurface"].apply_age_effect(20.0)
        assert 0.0 < resurface_age < 20.0
        assert resurface_age < patch_age

    def test_reconstruct_age_effect_resets_age(self):
        """Reconstruct should reset age to zero."""
        interventions = RoadDomain().default_interventions(surface_type="asphalt")
        assert interventions["reconstruct"].apply_age_effect(20.0) == 0.0

    def test_encode_covariates_returns_copy(self):
        """encode_covariates should return a copy without mutating input."""
        df = pd.DataFrame(
            {
                "traffic_load": ["low", "medium"],
                "climate_zone": ["temperate", "hot_dry"],
            }
        )
        encoded = RoadDomain.encode_covariates(df)
        assert "traffic_load_encoded" in encoded.columns
        assert "climate_zone_encoded" in encoded.columns
        assert "traffic_load_encoded" not in df.columns
        assert "climate_zone_encoded" not in df.columns


class TestRoadDomainDefaultModel:
    """Tests for RoadDomain default model."""

    def test_default_model_instance(self):
        """Default model should be a ProportionalHazardsModel."""
        model = RoadDomain().default_model()
        assert isinstance(model, ProportionalHazardsModel)

    def test_default_model_baseline(self):
        """Default model should use a Weibull baseline."""
        model = RoadDomain().default_model()
        assert isinstance(model.baseline, WeibullModel)

    def test_default_model_covariates(self):
        """Default model should include traffic and climate covariates."""
        model = RoadDomain().default_model()
        assert "traffic_load_encoded" in model.covariates
        assert "climate_zone_encoded" in model.covariates

    def test_default_model_params(self):
        """Default model should include parameters for each surface type."""
        model = RoadDomain().default_model()
        assert set(model.params.keys()) == {"asphalt", "concrete", "gravel"}

    def test_default_model_type_column(self):
        """Default model should use surface_type as type column."""
        model = RoadDomain().default_model()
        assert model.type_column == "surface_type"
