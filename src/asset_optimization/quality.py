"""Quality metrics for portfolio data."""

import pandas as pd
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Data quality metrics for portfolio.

    Attributes
    ----------
    completeness : pd.Series
        Percentage of non-null values per column (0.0-1.0).
    missing_counts : pd.Series
        Count of missing values per column.
    total_rows : int
        Total number of assets in portfolio.
    """

    completeness: pd.Series
    missing_counts: pd.Series
    total_rows: int

    def _repr_html_(self):
        """Rich HTML display for Jupyter notebooks."""
        summary = pd.DataFrame({
            'Completeness (%)': self.completeness * 100,
            'Missing Count': self.missing_counts,
        })
        return summary._repr_html_()

    def __repr__(self):
        """Text display for terminal/REPL."""
        summary = pd.DataFrame({
            'Completeness (%)': self.completeness * 100,
            'Missing Count': self.missing_counts,
        })
        return str(summary)
