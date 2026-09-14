from numbers import Real

import numpy as np
import pandas as pd
import polars as pl

from alpha_research._utils import (
    _validate_df,
    _validate_finite_number,
    _validate_positive_integer,
)

__all__ = [
    'absolute',
    'clip',
    'cross_sectional_quantile_rank',
    'cross_sectional_rank',
    'cross_sectional_zscore',
    'log1p',
    'power',
    'rolling_quantile_rank',
    'rolling_rank',
    'rolling_zscore',
    'sign',
    'signed_log1p',
    'signed_power',
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


def _validate_elementwise_features(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
) -> list[str]:
    """Validate numeric finite-or-missing inputs for elementwise transforms.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame containing the requested feature columns.
    feature_cols : str | list[str]
        One feature column or a non-empty feature-column list.

    Returns
    -------
    list[str]
        Normalized feature names in caller order.

    Raises
    ------
    KeyError
        If a feature column is absent.
    TypeError
        If df or feature_cols has an unsupported type.
    ValueError
        If a feature is non-numeric, entirely missing, or contains infinity.
    """
    normalized = _normalize_feature_cols(feature_cols)
    if any(not isinstance(feature, str) or not feature for feature in normalized):
        raise TypeError('every feature in feature_cols must be a non-empty string.')
    if len(set(normalized)) != len(normalized):
        raise ValueError('feature_cols must not contain duplicate names.')
    if isinstance(df, pd.DataFrame) and not df.columns.is_unique:
        raise ValueError('DataFrame must not contain duplicate column names.')
    _validate_df(df, normalized)
    for feature in normalized:
        series = df[feature]
        if isinstance(df, pd.DataFrame):
            if (
                    not pd.api.types.is_numeric_dtype(series)
                    or pd.api.types.is_bool_dtype(series)
                    or pd.api.types.is_complex_dtype(series)
            ):
                raise ValueError(f'{feature} must be real numeric and non-boolean.')
            if np.isinf(series.dropna().to_numpy(dtype=float)).any():
                raise ValueError(f'{feature} must contain only finite or missing values.')
        else:
            if not series.dtype.is_numeric():
                raise ValueError(f'{feature} must be numeric.')
            non_missing = series.drop_nulls()
            if non_missing.dtype.is_float() and non_missing.is_infinite().any():
                raise ValueError(f'{feature} must contain only finite or missing values.')
    return normalized


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


def cross_sectional_zscore(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Standardize feature values across assets independently at every time based on z-score.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Wide DataFrame with time_col, symbol_col, and the requested features.
    feature_cols : str | list[str]
        Feature columns standardized within every cross-section.
    symbol_col : str, default 'symbol'
        Asset identifier column required by the feature-frame contract.
    time_col : str, default 'time'
        Time column defining each cross-section.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Input columns plus ``{feature}_cs_zscore`` columns. Each valid
        cross-section uses its population standard deviation (``ddof=0``).

    Raises
    ------
    KeyError
        If a required column is absent.
    TypeError
        If df is not a Pandas or Polars DataFrame, or feature_cols is invalid.
    ValueError
        If df is empty, a required column is entirely missing, or feature_cols
        is an empty list.

    Notes
    -----
    Cross-sections with fewer than two valid values or zero dispersion produce
    missing transformed values because their z-score is undefined.
    """
    feature_cols = _normalize_feature_cols(feature_cols)
    _validate_df(df, [symbol_col, time_col] + feature_cols)

    if isinstance(df, pd.DataFrame):
        result = df.copy()
        for feature in feature_cols:
            grouped = result.groupby(time_col)[feature]
            mean = grouped.transform('mean')
            std = grouped.transform(lambda values: values.std(ddof=0))
            result[f'{feature}_cs_zscore'] = ((result[feature] - mean) / std).where(std != 0)
        return result

    expressions = []
    for feature in feature_cols:
        values = pl.col(feature).fill_nan(None)
        mean = values.mean().over(time_col)
        std = values.std(ddof=0).over(time_col)
        expressions.append(
            pl.when(std > 0)
            .then((values - mean) / std)
            .otherwise(None)
            .alias(f'{feature}_cs_zscore'),
        )

    return df.with_columns(expressions)


def cross_sectional_quantile_rank(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
        n_quantiles: int,
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Assign cross-sectional feature values to rank-based quantile groups.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Wide DataFrame with time_col, symbol_col, and the requested features.
    feature_cols : str | list[str]
        Feature columns ranked and assigned to quantile groups per time.
    n_quantiles : int
        Positive number of quantile groups. Every cross-section must have at
        least this many valid feature values to receive group labels.
    symbol_col : str, default 'symbol'
        Asset identifier column required by the feature-frame contract.
    time_col : str, default 'time'
        Time column defining each cross-section.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Input columns plus ``{feature}_cs_quantile_rank_{n_quantiles}``
        columns containing group labels from 1 (lowest) to n_quantiles
        (highest).

    Raises
    ------
    KeyError
        If a required column is absent.
    TypeError
        If df is not a Pandas or Polars DataFrame, or feature_cols is invalid.
    ValueError
        If df is empty, a required column is entirely missing, feature_cols is
        empty, or n_quantiles is not a positive integer.

    Notes
    -----
    Average ranks define ties, so tied observations are never split across
    groups. This can leave some quantile labels unused. Cross-sections with
    fewer than n_quantiles valid observations return missing labels.
    """
    feature_cols = _normalize_feature_cols(feature_cols)
    _validate_positive_integer(n_quantiles, 'n_quantiles')
    _validate_df(df, [symbol_col, time_col] + feature_cols)

    if isinstance(df, pd.DataFrame):
        result = df.copy()
        for feature in feature_cols:
            grouped = result.groupby(time_col)[feature]
            ranks = grouped.rank(method='average')
            counts = grouped.transform('count')
            quantile_rank = np.ceil(ranks * n_quantiles / counts)
            result[f'{feature}_cs_quantile_rank_{n_quantiles}'] = quantile_rank.where(
                counts >= n_quantiles,
            )
        return result

    expressions = []
    for feature in feature_cols:
        values = pl.col(feature).fill_nan(None)
        ranks = values.rank(method='average').over(time_col)
        counts = values.count().over(time_col)
        expressions.append(
            pl.when(counts >= n_quantiles)
            .then((ranks * n_quantiles / counts).ceil())
            .otherwise(None)
            .alias(f'{feature}_cs_quantile_rank_{n_quantiles}'),
        )

    return df.with_columns(expressions)


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


def rolling_zscore(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
        window: int,
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Standardize feature values within trailing per-asset windows based on z-score.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Wide DataFrame with time_col, symbol_col, and the requested features.
        Rows must be ordered chronologically within each asset.
    feature_cols : str | list[str]
        Feature columns standardized within every trailing window.
    window : int
        Positive number of observations in each trailing causal window.
    symbol_col : str, default 'symbol'
        Asset identifier used to isolate rolling calculations.
    time_col : str, default 'time'
        Time column validated within each asset.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Input columns plus ``{feature}_rolling_zscore_{window}`` columns.
        Every valid window uses its population standard deviation (``ddof=0``).

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
    non-missing values is required. Constant windows return missing values.
    """
    feature_cols = _normalize_feature_cols(feature_cols)
    _validate_positive_integer(window, 'window')
    _validate_df(df, [symbol_col, time_col] + feature_cols)
    _validate_temporal_order_by_symbol(df, symbol_col, time_col)

    if isinstance(df, pd.DataFrame):
        result = df.copy()
        for feature in feature_cols:
            grouped = result.groupby(symbol_col, sort=False)[feature]
            mean = grouped.transform(
                lambda values: values.rolling(window, min_periods=window).mean(),
            )
            std = grouped.transform(
                lambda values: values.rolling(window, min_periods=window).std(ddof=0),
            )
            result[f'{feature}_rolling_zscore_{window}'] = (
                (result[feature] - mean) / std
            ).where(std != 0)
        return result

    expressions = []
    for feature in feature_cols:
        values = pl.col(feature).fill_nan(None)
        mean = values.rolling_mean(
            window_size=window,
            min_samples=window,
        ).over(symbol_col)
        std = values.rolling_std(
            window_size=window,
            min_samples=window,
            ddof=0,
        ).over(symbol_col)
        expressions.append(
            pl.when(std > 0)
            .then((values - mean) / std)
            .otherwise(None)
            .alias(f'{feature}_rolling_zscore_{window}'),
        )

    return df.with_columns(expressions)


def rolling_quantile_rank(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
        window: int,
        n_quantiles: int,
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Assign feature values to trailing per-asset rank-based quantile groups.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Wide DataFrame with time_col, symbol_col, and the requested features.
        Rows must be ordered chronologically within each asset.
    feature_cols : str | list[str]
        Feature columns assigned to quantile groups within each trailing window.
    window : int
        Positive number of observations in each trailing causal window.
    n_quantiles : int
        Positive number of quantile groups. It must not exceed window.
    symbol_col : str, default 'symbol'
        Asset identifier used to isolate rolling calculations.
    time_col : str, default 'time'
        Time column validated within each asset.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Input columns plus ``{feature}_rolling_quantile_rank_{window}_{n_quantiles}``
        columns containing group labels from 1 (lowest) to n_quantiles
        (highest).

    Raises
    ------
    KeyError
        If a required column is absent.
    TypeError
        If df is not a Pandas or Polars DataFrame, or feature_cols is invalid.
    ValueError
        If df is empty, a required column is entirely missing, feature_cols is
        empty, window or n_quantiles is invalid, n_quantiles exceeds window,
        or times are invalid within an asset.

    Notes
    -----
    The current value is included in its trailing window. A full window of
    non-missing values is required. Average ranks define ties, so ties are not
    split across groups and some group labels can be unused.
    """
    feature_cols = _normalize_feature_cols(feature_cols)
    _validate_positive_integer(window, 'window')
    _validate_positive_integer(n_quantiles, 'n_quantiles')
    if n_quantiles > window:
        raise ValueError('n_quantiles must not exceed window.')

    _validate_df(df, [symbol_col, time_col] + feature_cols)
    _validate_temporal_order_by_symbol(df, symbol_col, time_col)

    if isinstance(df, pd.DataFrame):
        result = df.copy()
        for feature in feature_cols:
            result[f'{feature}_rolling_quantile_rank_{window}_{n_quantiles}'] = (
                result.groupby(symbol_col, sort=False)[feature].transform(
                    lambda values: values.rolling(window, min_periods=window).apply(
                        lambda window_values: np.ceil(
                            window_values.rank(method='average').iloc[-1]
                            * n_quantiles
                            / window,
                        ),
                        raw=False,
                    ),
                )
            )
        return result

    return df.with_columns([
        pl.col(feature)
        .fill_nan(None)
        .rolling_map(
            lambda values: np.ceil(
                values.rank(method='average')[-1] * n_quantiles / window,
            ),
            window_size=window,
            min_samples=window,
        )
        .over(symbol_col)
        .alias(f'{feature}_rolling_quantile_rank_{window}_{n_quantiles}')
        for feature in feature_cols
    ])


def sign(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
) -> pd.DataFrame | pl.DataFrame:
    """Add the elementwise sign of every requested feature.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame containing numeric feature columns.
    feature_cols : str | list[str]
        Features transformed independently at each observation.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Input columns plus ``{feature}_sign`` columns containing -1, 0, or 1.

    Raises
    ------
    KeyError
        If a requested feature is absent.
    TypeError
        If df or feature_cols has an unsupported type.
    ValueError
        If a feature is non-numeric, entirely missing, or infinite.
    """
    features = _validate_elementwise_features(df, feature_cols)
    if isinstance(df, pd.DataFrame):
        result = df.copy()
        for feature in features:
            result[f'{feature}_sign'] = np.sign(result[feature])
        return result
    return df.with_columns([
        pl.col(feature).fill_nan(None).sign().alias(f'{feature}_sign')
        for feature in features
    ])


def absolute(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
) -> pd.DataFrame | pl.DataFrame:
    """Add the elementwise absolute value of every requested feature.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame containing numeric feature columns.
    feature_cols : str | list[str]
        Features transformed independently at each observation.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Input columns plus ``{feature}_abs`` columns.

    Raises
    ------
    KeyError
        If a requested feature is absent.
    TypeError
        If df or feature_cols has an unsupported type.
    ValueError
        If a feature is non-numeric, entirely missing, or infinite.
    """
    features = _validate_elementwise_features(df, feature_cols)
    if isinstance(df, pd.DataFrame):
        result = df.copy()
        for feature in features:
            result[f'{feature}_abs'] = result[feature].abs()
        return result
    return df.with_columns([
        pl.col(feature).fill_nan(None).abs().alias(f'{feature}_abs')
        for feature in features
    ])


def power(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
        exponent: Real,
) -> pd.DataFrame | pl.DataFrame:
    """Raise each feature to one fixed real-valued exponent elementwise.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame containing numeric feature columns.
    feature_cols : str | list[str]
        Features transformed independently at each observation.
    exponent : numbers.Real
        Finite exponent. Fractional exponents require non-negative inputs;
        negative exponents require non-zero inputs.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Input columns plus ``{feature}_power_{exponent}`` columns.

    Raises
    ------
    KeyError
        If a requested feature is absent.
    TypeError
        If df, feature_cols, or exponent has an unsupported type.
    ValueError
        If inputs are invalid or outside the real-valued power domain.
    """
    features = _validate_elementwise_features(df, feature_cols)
    normalized_exponent = _validate_finite_number(exponent, 'exponent')
    exponent_label = str(exponent)
    is_integer = normalized_exponent.is_integer()
    for feature in features:
        series = df[feature]
        if isinstance(df, pd.DataFrame):
            valid = series.dropna()
            has_negative = (valid < 0).any()
            has_zero = (valid == 0).any()
        else:
            valid = series.drop_nulls()
            if valid.dtype.is_float():
                valid = valid.drop_nans()
            has_negative = bool((valid < 0).any())
            has_zero = bool((valid == 0).any())
        if not is_integer and has_negative:
            raise ValueError(
                'fractional exponent requires non-negative feature values.',
            )
        if normalized_exponent < 0 and has_zero:
            raise ValueError('negative exponent requires non-zero feature values.')

    if isinstance(df, pd.DataFrame):
        result = df.copy()
        for feature in features:
            result[f'{feature}_power_{exponent_label}'] = (
                result[feature]
                .pow(normalized_exponent)
                .where(result[feature].notna())
            )
        return result
    return df.with_columns([
        pl.col(feature)
        .fill_nan(None)
        .pow(normalized_exponent)
        .alias(f'{feature}_power_{exponent_label}')
        for feature in features
    ])


def signed_power(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
        exponent: Real,
) -> pd.DataFrame | pl.DataFrame:
    """Apply ``sign(x) * abs(x) ** exponent`` to each feature.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame containing numeric feature columns.
    feature_cols : str | list[str]
        Features transformed independently at each observation.
    exponent : numbers.Real
        Finite exponent. Negative exponents require non-zero inputs.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Input columns plus ``{feature}_signed_power_{exponent}`` columns.

    Raises
    ------
    KeyError
        If a requested feature is absent.
    TypeError
        If df, feature_cols, or exponent has an unsupported type.
    ValueError
        If inputs are invalid or a negative exponent receives zero.
    """
    features = _validate_elementwise_features(df, feature_cols)
    normalized_exponent = _validate_finite_number(exponent, 'exponent')
    exponent_label = str(exponent)
    if normalized_exponent < 0:
        for feature in features:
            series = df[feature]
            valid = series.dropna() if isinstance(df, pd.DataFrame) else series.drop_nulls()
            if not isinstance(df, pd.DataFrame) and valid.dtype.is_float():
                valid = valid.drop_nans()
            if bool((valid == 0).any()):
                raise ValueError(
                    'negative exponent requires non-zero feature values.',
                )

    if isinstance(df, pd.DataFrame):
        result = df.copy()
        for feature in features:
            result[f'{feature}_signed_power_{exponent_label}'] = (
                np.sign(result[feature])
                * result[feature].abs().pow(normalized_exponent)
            )
        return result
    return df.with_columns([
        (
            pl.col(feature).fill_nan(None).sign()
            * pl.col(feature).fill_nan(None).abs().pow(normalized_exponent)
        ).alias(f'{feature}_signed_power_{exponent_label}')
        for feature in features
    ])


def log1p(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
) -> pd.DataFrame | pl.DataFrame:
    """Apply the real-valued ``log(1 + x)`` transformation elementwise.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame containing numeric feature columns strictly greater than -1.
    feature_cols : str | list[str]
        Features transformed independently at each observation.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Input columns plus ``{feature}_log1p`` columns.

    Raises
    ------
    KeyError
        If a requested feature is absent.
    TypeError
        If df or feature_cols has an unsupported type.
    ValueError
        If inputs are invalid or contain a finite value at or below -1.
    """
    features = _validate_elementwise_features(df, feature_cols)
    for feature in features:
        series = df[feature]
        valid = series.dropna() if isinstance(df, pd.DataFrame) else series.drop_nulls()
        if not isinstance(df, pd.DataFrame) and valid.dtype.is_float():
            valid = valid.drop_nans()
        if bool((valid <= -1).any()):
            raise ValueError('log1p requires feature values greater than -1.')
    if isinstance(df, pd.DataFrame):
        result = df.copy()
        for feature in features:
            result[f'{feature}_log1p'] = np.log1p(result[feature])
        return result
    return df.with_columns([
        pl.col(feature).fill_nan(None).log1p().alias(f'{feature}_log1p')
        for feature in features
    ])


def signed_log1p(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
) -> pd.DataFrame | pl.DataFrame:
    """Apply ``sign(x) * log(1 + abs(x))`` to each feature.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame containing numeric feature columns.
    feature_cols : str | list[str]
        Features transformed independently at each observation.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Input columns plus ``{feature}_signed_log1p`` columns.

    Raises
    ------
    KeyError
        If a requested feature is absent.
    TypeError
        If df or feature_cols has an unsupported type.
    ValueError
        If a feature is non-numeric, entirely missing, or infinite.
    """
    features = _validate_elementwise_features(df, feature_cols)
    if isinstance(df, pd.DataFrame):
        result = df.copy()
        for feature in features:
            result[f'{feature}_signed_log1p'] = (
                np.sign(result[feature]) * np.log1p(result[feature].abs())
            )
        return result
    return df.with_columns([
        (
            pl.col(feature).fill_nan(None).sign()
            * pl.col(feature).fill_nan(None).abs().log1p()
        ).alias(f'{feature}_signed_log1p')
        for feature in features
    ])


def clip(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
        lower: Real | None = None,
        upper: Real | None = None,
) -> pd.DataFrame | pl.DataFrame:
    """Clip each feature to fixed researcher-supplied bounds.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame containing numeric feature columns.
    feature_cols : str | list[str]
        Features transformed independently at each observation.
    lower, upper : numbers.Real | None, default None
        Inclusive fixed bounds. At least one bound must be supplied.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Input columns plus ``{feature}_clip_{lower}_{upper}`` columns.

    Raises
    ------
    KeyError
        If a requested feature is absent.
    TypeError
        If df, feature_cols, or a bound has an unsupported type.
    ValueError
        If inputs or bounds are invalid.
    """
    features = _validate_elementwise_features(df, feature_cols)
    if lower is None and upper is None:
        raise ValueError('at least one of lower or upper must be provided.')
    normalized_lower = (
        None if lower is None else _validate_finite_number(lower, 'lower')
    )
    normalized_upper = (
        None if upper is None else _validate_finite_number(upper, 'upper')
    )
    if (
            normalized_lower is not None
            and normalized_upper is not None
            and normalized_lower > normalized_upper
    ):
        raise ValueError('lower must not exceed upper.')
    label = f'{lower}_{upper}'
    if isinstance(df, pd.DataFrame):
        result = df.copy()
        for feature in features:
            result[f'{feature}_clip_{label}'] = result[feature].clip(
                lower=normalized_lower,
                upper=normalized_upper,
            )
        return result
    return df.with_columns([
        pl.col(feature)
        .fill_nan(None)
        .clip(lower_bound=normalized_lower, upper_bound=normalized_upper)
        .alias(f'{feature}_clip_{label}')
        for feature in features
    ])
