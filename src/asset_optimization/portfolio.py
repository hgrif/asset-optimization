"""Portfolio class for loading and querying asset data."""

import pandas as pd
import pandera as pa
from pathlib import Path
from .schema import portfolio_schema
from .quality import QualityMetrics
from .exceptions import ValidationError


class Portfolio:
    """Asset portfolio with validation and quality metrics.

    Use factory methods to load portfolio data:
    - Portfolio.from_csv(path)
    - Portfolio.from_excel(path)
    - Portfolio.from_dataframe(df)

    Examples
    --------
    >>> portfolio = Portfolio.from_csv('assets.csv')
    >>> print(len(portfolio))
    1000
    >>> print(portfolio['PIPE-001']['material'])
    'PVC'
    >>> print(portfolio.quality)
    """

    def __init__(self):
        """Initialize empty Portfolio.

        Use factory methods to load data:
        - Portfolio.from_csv(path)
        - Portfolio.from_excel(path)
        - Portfolio.from_dataframe(df)
        """
        pass

    @classmethod
    def from_csv(cls, path, **kwargs):
        """Load portfolio from CSV file.

        Parameters
        ----------
        path : str or Path
            Path to CSV file.
        **kwargs : dict
            Additional arguments passed to pd.read_csv().

        Returns
        -------
        portfolio : Portfolio
            Loaded portfolio instance.

        Raises
        ------
        ValidationError
            If data fails schema validation.

        Examples
        --------
        >>> portfolio = Portfolio.from_csv('assets.csv')
        >>> portfolio = Portfolio.from_csv('assets.csv', sep=';')
        """
        instance = cls()

        # Explicit dtypes for performance and type safety
        dtypes = {
            'asset_id': str,
            'asset_type': str,
            'material': str,
            'diameter_mm': 'Int64',  # Nullable integer
            'length_m': float,
            'condition_score': float,
        }

        df = pd.read_csv(
            path,
            dtype=dtypes,
            parse_dates=['install_date'],
            **kwargs
        )

        instance._load_data(df)
        return instance

    @classmethod
    def from_excel(cls, path, **kwargs):
        """Load portfolio from Excel file.

        Parameters
        ----------
        path : str or Path
            Path to Excel file (.xlsx).
        **kwargs : dict
            Additional arguments passed to pd.read_excel().

        Returns
        -------
        portfolio : Portfolio
            Loaded portfolio instance.

        Raises
        ------
        ValidationError
            If data fails schema validation.

        Examples
        --------
        >>> portfolio = Portfolio.from_excel('assets.xlsx')
        >>> portfolio = Portfolio.from_excel('assets.xlsx', sheet_name='Sheet2')
        """
        instance = cls()

        # Explicit dtypes for performance and type safety
        dtypes = {
            'asset_id': str,
            'asset_type': str,
            'material': str,
            'diameter_mm': 'Int64',  # Nullable integer
            'length_m': float,
            'condition_score': float,
        }

        df = pd.read_excel(
            path,
            dtype=dtypes,
            parse_dates=['install_date'],
            engine='openpyxl',
            **kwargs
        )

        instance._load_data(df)
        return instance

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        """Load portfolio from DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with portfolio data.

        Returns
        -------
        portfolio : Portfolio
            Loaded portfolio instance.

        Raises
        ------
        ValidationError
            If data fails schema validation.

        Examples
        --------
        >>> df = pd.DataFrame({'asset_id': ['PIPE-001'], ...})
        >>> portfolio = Portfolio.from_dataframe(df)
        """
        instance = cls()
        instance._load_data(df)
        return instance

    def _load_data(self, df: pd.DataFrame):
        """Validate and store portfolio data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame to validate and store.

        Raises
        ------
        ValidationError
            If data fails schema validation.
        """
        # Validate using Pandera schema
        try:
            validated_df = portfolio_schema.validate(df, lazy=False)
        except pa.errors.SchemaError as exc:
            # Extract first failure for user-friendly message
            # failure_cases can be a DataFrame or string depending on error type
            if isinstance(exc.failure_cases, pd.DataFrame) and not exc.failure_cases.empty:
                first_failure = exc.failure_cases.iloc[0]
                raise ValidationError(
                    field=first_failure['column'] if 'column' in first_failure else 'schema',
                    message=first_failure['check'] if 'check' in first_failure else str(exc),
                    details={
                        'failing_values': exc.failure_cases['failure_case'].tolist()[:5] if 'failure_case' in exc.failure_cases.columns else [],
                        'total_failures': len(exc.failure_cases),
                    }
                ) from exc
            else:
                # Schema error without failure cases (e.g., missing column)
                raise ValidationError(
                    field='schema',
                    message=str(exc),
                    details={}
                ) from exc

        # Store validated data
        self._data = validated_df
        self._n_assets = len(validated_df)
        self._quality = self._compute_quality(validated_df)

    def _compute_quality(self, df: pd.DataFrame) -> QualityMetrics:
        """Compute quality metrics for portfolio data.

        Parameters
        ----------
        df : pd.DataFrame
            Portfolio data.

        Returns
        -------
        metrics : QualityMetrics
            Data quality metrics.
        """
        completeness = df.notna().mean()
        missing_counts = df.isna().sum()
        return QualityMetrics(
            completeness=completeness,
            missing_counts=missing_counts,
            total_rows=len(df),
        )

    @property
    def data(self) -> pd.DataFrame:
        """Access portfolio data for filtering.

        Returns
        -------
        df : pd.DataFrame
            Portfolio DataFrame.

        Examples
        --------
        >>> pipes = portfolio.data[portfolio.data['asset_type'] == 'pipe']
        """
        if not hasattr(self, '_data'):
            raise AttributeError("No data loaded")
        return self._data

    @property
    def quality(self) -> QualityMetrics:
        """Access quality metrics.

        Returns
        -------
        metrics : QualityMetrics
            Data quality metrics with rich display.

        Examples
        --------
        >>> print(portfolio.quality)
        """
        if not hasattr(self, '_quality'):
            raise AttributeError("Quality metrics not computed (no data loaded)")
        return self._quality

    @property
    def asset_types(self) -> list[str]:
        """Get unique asset types in portfolio.

        Returns
        -------
        types : list[str]
            List of unique asset types.

        Examples
        --------
        >>> portfolio.asset_types
        ['pipe', 'valve', 'hydrant']
        """
        if not hasattr(self, '_data'):
            raise AttributeError("No data loaded")
        return self._data['asset_type'].unique().tolist()

    @property
    def mean_age(self) -> float:
        """Calculate mean age of assets in years.

        Returns
        -------
        age : float
            Mean age in years.

        Examples
        --------
        >>> portfolio.mean_age
        12.5
        """
        if not hasattr(self, '_data'):
            raise AttributeError("No data loaded")
        ages = (pd.Timestamp.now() - self._data['install_date']).dt.days / 365.25
        return ages.mean()

    @property
    def age_distribution(self) -> pd.Series:
        """Get age distribution of all assets in years.

        Returns
        -------
        ages : pd.Series
            Series of ages in years for all assets.

        Examples
        --------
        >>> portfolio.age_distribution.describe()
        """
        if not hasattr(self, '_data'):
            raise AttributeError("No data loaded")
        return (pd.Timestamp.now() - self._data['install_date']).dt.days / 365.25

    @property
    def oldest(self) -> pd.Series:
        """Get the oldest asset (earliest install_date).

        Returns
        -------
        asset : pd.Series
            Row data for the oldest asset.

        Examples
        --------
        >>> portfolio.oldest['asset_id']
        'PIPE-001'
        """
        if not hasattr(self, '_data'):
            raise AttributeError("No data loaded")
        idx = self._data['install_date'].idxmin()
        return self._data.loc[idx]

    @property
    def newest(self) -> pd.Series:
        """Get the newest asset (latest install_date).

        Returns
        -------
        asset : pd.Series
            Row data for the newest asset.

        Examples
        --------
        >>> portfolio.newest['asset_id']
        'VALVE-100'
        """
        if not hasattr(self, '_data'):
            raise AttributeError("No data loaded")
        idx = self._data['install_date'].idxmax()
        return self._data.loc[idx]

    def __len__(self) -> int:
        """Get number of assets in portfolio.

        Returns
        -------
        n : int
            Number of assets.

        Examples
        --------
        >>> len(portfolio)
        1000
        """
        if not hasattr(self, '_n_assets'):
            return 0
        return self._n_assets

    def __getitem__(self, asset_id: str) -> pd.Series:
        """Get asset by ID (dict-like access).

        Parameters
        ----------
        asset_id : str
            Asset identifier.

        Returns
        -------
        asset : pd.Series
            Row data for the asset.

        Raises
        ------
        KeyError
            If asset_id not found.

        Examples
        --------
        >>> portfolio['PIPE-001']['material']
        'PVC'
        """
        if not hasattr(self, '_data'):
            raise AttributeError("No data loaded")

        # Find row with matching asset_id
        mask = self._data['asset_id'] == asset_id
        if not mask.any():
            raise KeyError(f"Asset ID '{asset_id}' not found")

        return self._data[mask].iloc[0]

    def __repr__(self) -> str:
        """Informative repr for REPL/notebook display.

        Returns
        -------
        repr : str
            String representation.

        Examples
        --------
        >>> portfolio
        Portfolio(n_assets=1000, types=['pipe', 'valve'], date_range='2010-2020')
        """
        if not hasattr(self, '_data'):
            return "Portfolio(empty)"

        # Get date range
        min_date = self._data['install_date'].min()
        max_date = self._data['install_date'].max()
        date_range = f"{min_date.year}-{max_date.year}"

        return (
            f"Portfolio(n_assets={self._n_assets}, "
            f"types={self.asset_types}, "
            f"date_range='{date_range}')"
        )
