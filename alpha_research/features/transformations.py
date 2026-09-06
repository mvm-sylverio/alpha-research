import pandas as pd
import polars as pl
import numpy as np

from alpha_research._utils import _validate_df, _validate_positive_integer

__all__ = [
    'cross_sectional_rank',
    'cross_sectional_zscore',
    'cross_sectional_quantile_rank',
    'rolling_rank',
    'rolling_zscore',
    'rolling_quantile_rank',
]


def _normalize_feature_cols(feature_cols: str | list[str]) -> list[str]:
    """
    Normalize one or more feature-column names to a non-empty list.

    Parameters
    ----------
    feature_cols : str | list[str]
        One feature-column name or a non-empty list of feature-column names.

    Returns
    -------
    list[str]
        Feature-column names in their original order.

    Raises
    ------
    TypeError
        If feature_cols is neither a string nor a list.
    ValueError
        If feature_cols is an empty list.
    """
    if isinstance(feature_cols, str):
        return [feature_cols]

    if not isinstance(feature_cols, list):
        raise TypeError('feature_cols must be str or list[str].')

    if not feature_cols:
        raise ValueError('feature_cols must not be an empty list.')

    return feature_cols


def _validate_temporal_order_by_symbol(
        df: pd.DataFrame | pl.DataFrame,
        symbol_col: str,
        time_col: str,
) -> None:
    """
    Validate unique increasing times independently within every asset.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame containing the asset and temporal key columns.
    symbol_col : str
        Asset identifier used to separate individual time series.
    time_col : str
        Temporal key that must be uniquely and increasingly ordered per asset.

    Returns
    -------
    None
        Returns None when every asset has valid temporal order.

    Raises
    ------
    ValueError
        If any asset contains missing, duplicate, or decreasing times.
    """
    if isinstance(df, pd.DataFrame):
        asset_frames = (
            asset_frame
            for _, asset_frame in df.groupby(symbol_col, sort=False, dropna=False)
        )
    else:
        asset_frames = iter(df.partition_by(symbol_col, maintain_order=True))

    for asset_frame in asset_frames:
        times = asset_frame[time_col]

        if isinstance(df, pd.DataFrame):
            has_missing = times.isna().any()
            has_duplicates = times.duplicated().any()
            is_ordered = times.is_monotonic_increasing
        else:
            has_missing = times.is_null().any()
            has_duplicates = times.is_duplicated().any()
            is_ordered = times.is_sorted()

        if has_missing:
            raise ValueError(f'{time_col} must not contain missing values.')

        if has_duplicates:
            raise ValueError(
                f'{time_col} must contain unique values within each {symbol_col}.',
            )

        if not is_ordered:
            raise ValueError(
                f'{time_col} must be increasingly ordered within each {symbol_col}.',
            )


def cross_sectional_rank(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Compute the cross-sectional assets rank for all the features in feature_cols.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Wide DataFrame with the time_col, symbol_col and all features contained in feature_cols.
    feature_cols : str | list[str]
        All the columns in df which will be cross-sectionally ranked.
    symbol_col : str
        Column which contains the asset names of the dataset.
    time_col : str
        Time column of the dataset.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        DataFrame with the time_col, symbol_col, all the original feature_cols and their respective
        cross-sectional ranks columns.

    Raises
    ------
    KeyError
        If time_col, symbol_col and all the columns in feature_cols are not columns of the df.
    TypeError
        If df is not pandas or polars type.
        If feature_cols is not str or list[str].
    ValueError
        If feature_cols is an empty list.

    Notes
    -----
    Designed to operate on a merged feature DataFrame - typically the output
    of joining multiple feature DataFrames on ['time', 'symbol'] before
    applying cross-sectional transformations.

    Example workflow:
        df = simple_return(ohlcv, horizon=5)
            .merge(fwd_return(ohlcv, horizon=10), on=['time', 'symbol'])
        df_ranked = cross_sectional_rank(df, feature_cols=['simple_ret_5'])

    Ties are broken using the average method.

    Cross-sectional operation: rank is computed per date across assets,
    not along the time axis. Not meaningful for single-asset datasets.
    """
    feature_cols = _normalize_feature_cols(feature_cols)

    _validate_df(df, [symbol_col, time_col] + feature_cols)

    if isinstance(df, pd.DataFrame):
        result = df.copy()
        for feature in feature_cols:
            result[f'{feature}_cs_rank'] = result.groupby(time_col)[feature].rank(method='average')
        return result

    else:  # polars DataFrame
        return df.with_columns([
            pl.col(feature)
            .rank(method='average')
            .over(time_col)
            .alias(f'{feature}_cs_rank')
            for feature in feature_cols
        ])

def rolling_rank(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
        window: int,
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Rank each feature value within its own trailing per-asset window.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Wide DataFrame with time_col, symbol_col, and the requested features.
        Rows must be ordered chronologically within each asset.
    feature_cols : str | list[str]
        Feature columns ranked within every trailing window.
    window : int
        Positive number of observations in each trailing causal window.
    symbol_col : str, default 'symbol'
        Asset identifier used to isolate rolling calculations.
    time_col : str, default 'time'
        Time column validated within each asset.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Input columns plus ``{feature}_rolling_rank_{window}`` columns.

    Raises
    ------
    KeyError
        If a required column is absent.
    TypeError
        If df is not a Pandas or Polars DataFrame, or feature_cols is invalid.
    ValueError
        If df is empty, a required column is entirely missing, feature_cols is
        empty, window is invalid, or times are invalid within an asset.

    Notes
    -----
    The current value is included in its trailing window. A full window of
    non-missing values is required, and ties use the average-rank method.
    """
    feature_cols = _normalize_feature_cols(feature_cols)
    _validate_positive_integer(window, 'window')
    _validate_df(df, [symbol_col, time_col] + feature_cols)
    _validate_temporal_order_by_symbol(df, symbol_col, time_col)

    if isinstance(df, pd.DataFrame):
        result = df.copy()
        for feature in feature_cols:
            result[f'{feature}_rolling_rank_{window}'] = result.groupby(
                symbol_col,
                sort=False,
            )[feature].transform(
                lambda values: values.rolling(window, min_periods=window).apply(
                    lambda window_values: window_values.rank(method='average').iloc[-1],
                    raw=False,
                ),
            )
        return result

    return df.with_columns([
        pl.col(feature)
        .fill_nan(None)
        .rolling_map(
            lambda values: values.rank(method='average')[-1],
            window_size=window,
            min_samples=window,
        )
        .over(symbol_col)
        .alias(f'{feature}_rolling_rank_{window}')
        for feature in feature_cols
    ])
