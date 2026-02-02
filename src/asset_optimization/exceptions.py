"""Custom exceptions for asset portfolio validation."""


class AssetOptimizationError(Exception):
    """Base exception for asset-optimization package."""
    pass


class ValidationError(AssetOptimizationError):
    """Raised when portfolio data fails validation.

    Attributes
    ----------
    field : str
        Name of field that failed validation.
    message : str
        Human-readable error description.
    details : dict
        Additional error context (failed values, expected range, etc).
    """

    def __init__(self, field, message, details=None):
        self.field = field
        self.message = message
        self.details = details or {}
        super().__init__(self._format_message())

    def _format_message(self):
        """Format error message with field and details."""
        msg = f"Validation failed for '{self.field}': {self.message}"
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            msg += f" ({detail_str})"
        return msg


class MissingFieldError(ValidationError):
    """Raised when required field is missing from data."""
    pass


class DataQualityError(ValidationError):
    """Raised when data quality is below acceptable threshold."""
    pass


class SimulationError(AssetOptimizationError):
    """Raised when simulation encounters an error.

    Attributes
    ----------
    message : str
        Human-readable error description.
    year : int, optional
        Simulation year when error occurred.
    details : dict
        Additional error context.
    """

    def __init__(self, message, year=None, details=None):
        self.message = message
        self.year = year
        self.details = details or {}
        super().__init__(self._format_message())

    def _format_message(self):
        """Format error message with year and details."""
        if self.year is not None:
            msg = f"Simulation error at year {self.year}: {self.message}"
        else:
            msg = f"Simulation error: {self.message}"
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            msg += f" ({detail_str})"
        return msg
