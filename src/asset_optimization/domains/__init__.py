"""Domain abstractions for asset classes."""

from asset_optimization.domains.base import Domain
from asset_optimization.domains.pipes import PipeDomain
from asset_optimization.domains.roads import RoadDomain

__all__ = ["Domain", "PipeDomain", "RoadDomain"]
