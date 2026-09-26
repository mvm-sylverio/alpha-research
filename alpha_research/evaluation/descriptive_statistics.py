from inspect import signature

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import kurtosis, skew

from alpha_research._utils import (
    _validate_df,
    _validate_positive_integer,
    _validate_time_order,
    _validate_unique_keys,
)
from alpha_research.features.schema import get_feature_name

__all__ = ['distribution_summary', 'rolling_distribution_summary']

_QUANTILES = (
    ('q01', 0.01),
    ('q05', 0.05),
    ('q25', 0.25),
    ('median', 0.50),
    ('q75', 0.75),
    ('q95', 0.95),
    ('q99', 0.99),
)
_ROLLING_MINIMUM_KEYWORD = (
    'min_samples'
    if 'min_samples' in signature(pl.Expr.rolling_mean).parameters
    else 'min_periods'
)


def _prepare_value_frame(
        value_frame: pd.DataFrame | pl.DataFrame,
        value_col: str | None,
        time_col: str,
        symbol_col: str,
) -> tuple[pd.DataFrame | pl.DataFrame, str]:
    """Validate a feature or target frame and select its value and key columns.

    Parameters
    ----------
    value_frame : pd.DataFrame | pl.DataFrame
        Frame containing temporal and asset keys and a numeric value column.
    value_col : str | None
        Value column, inferred when the frame has exactly one non-key column.
    time_col, symbol_col : str
        Names of the temporal and asset key columns.

    Returns
    -------
    tuple[pd.DataFrame | pl.DataFrame, str]
        Validated three-column frame in its original backend and its value
        column name.

    Raises
    ------
    TypeError
        If the input is not Pandas or Polars or its value column is not numeric.
    KeyError
        If a required column is absent.
    ValueError
        If the frame is empty, keys are missing or duplicated, or column names
        overlap.
    """
    _validate_df(value_frame, [time_col, symbol_col], check_all_missing=False)
    if time_col == symbol_col:
        raise ValueError('time_col and symbol_col must be distinct.')

    if value_col is None:
        value_col = get_feature_name(
            value_frame,
            time_col=time_col,
            symbol_col=symbol_col,
        )
    if value_col in (time_col, symbol_col):
        raise ValueError('value_col must differ from the key columns.')
    _validate_df(
        value_frame,
        [time_col, symbol_col, value_col],
        check_all_missing=False,
    )

    if isinstance(value_frame, pd.DataFrame):
        is_numeric = pd.api.types.is_numeric_dtype(value_frame[value_col])
        is_boolean = pd.api.types.is_bool_dtype(value_frame[value_col])
        all_missing = value_frame[value_col].isna().all()
    else:
        is_numeric = value_frame.schema[value_col].is_numeric()
        is_boolean = value_frame.schema[value_col] == pl.Boolean
        all_missing = value_frame[value_col].is_null().all()
    if is_boolean or (not is_numeric and not all_missing):
        raise TypeError(f'{value_col} must be numeric and non-boolean.')

    for key in (time_col, symbol_col):
        if isinstance(value_frame, pd.DataFrame):
            has_missing = value_frame[key].isna().any()
        else:
            has_missing = value_frame[key].is_null().any()
            if value_frame.schema[key].is_float():
                has_missing = has_missing or value_frame[key].is_nan().any()
        if has_missing:
            raise ValueError(f'{key} must not contain missing values.')
    _validate_unique_keys(value_frame, [time_col, symbol_col], 'value_frame')
    if isinstance(value_frame, pd.DataFrame):
        return value_frame[[time_col, symbol_col, value_col]].copy(), value_col
    return value_frame.select([time_col, symbol_col, value_col]), value_col


def _summarize_values(values: pl.Series) -> dict[str, int | float]:
    """Calculate descriptive statistics for one numeric group of observations.

    Parameters
    ----------
    values : pl.Series
        Numeric observations, including possible missing or infinite values.

    Returns
    -------
    dict[str, int | float]
        Counts and finite-value statistics. Sample standard deviation uses
        ddof=1; skewness and Fisher excess kurtosis are bias-corrected.

    Raises
    ------
    TypeError
        If values is not a numeric, non-boolean Polars Series.
    """
    if not isinstance(values, pl.Series):
        raise TypeError('values must be a numeric, non-boolean Polars Series.')
    all_missing = values.is_null().all()
    if values.dtype == pl.Boolean or (not values.dtype.is_numeric() and not all_missing):
        raise TypeError('values must be a numeric, non-boolean Polars Series.')
    numeric = (
        np.full(len(values), np.nan)
        if all_missing else values.cast(pl.Float64).to_numpy()
    )

    finite = numeric[np.isfinite(numeric)]
    n_valid = len(finite)
    result = {
        'n_total': len(values),
        'n_valid': n_valid,
        'n_missing': int(np.isnan(numeric).sum()),
        'n_infinite': int(np.isinf(numeric).sum()),
        'mean': np.nan,
        'std': np.nan,
        'min': np.nan,
        'q01': np.nan,
        'q05': np.nan,
        'q25': np.nan,
        'median': np.nan,
        'q75': np.nan,
        'q95': np.nan,
        'q99': np.nan,
        'max': np.nan,
        'iqr': np.nan,
        'skewness': np.nan,
        'excess_kurtosis': np.nan,
    }
    if n_valid == 0:
        return result

    quantile_values = np.quantile(finite, [level for _, level in _QUANTILES])
    quantiles = {
        name: float(value)
        for (name, _), value in zip(_QUANTILES, quantile_values)
    }
    result.update({
        'mean': float(np.mean(finite)),
        'std': float(np.std(finite, ddof=1)) if n_valid >= 2 else np.nan,
        'min': float(np.min(finite)),
        **quantiles,
        'max': float(np.max(finite)),
        'iqr': float(quantiles['q75'] - quantiles['q25']),
    })
    if np.min(finite) != np.max(finite):
        if n_valid >= 3:
            result['skewness'] = float(skew(finite, bias=False))
        if n_valid >= 4:
            result['excess_kurtosis'] = float(kurtosis(
                finite,
                fisher=True,
                bias=False,
            ))
    return result


def distribution_summary(
        value_frame: pd.DataFrame | pl.DataFrame,
        group_by: str | None = None,
        value_col: str | None = None,
        time_col: str = 'time',
        symbol_col: str = 'symbol',
) -> pd.DataFrame | pl.DataFrame:
    """Summarize a feature or target overall, by asset, or cross-sectionally.

    Parameters
    ----------
    value_frame : pd.DataFrame | pl.DataFrame
        Feature or target observations with time, symbol, and numeric value.
        An explicit value_col also permits a wider input frame.
    group_by : str | None, default None
        Use symbol_col for one row per asset, time_col for one row per date,
        or None for one overall row.
    value_col : str | None, default None
        Numeric value column, inferred from a three-column frame when omitted.
    time_col, symbol_col : str
        Names of the temporal and asset key columns.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Input-backend table with one row per requested group and fixed count,
        quantile, dispersion, skewness, and excess-kurtosis columns.

    Raises
    ------
    TypeError
        If the frame or value column has an unsupported type.
    KeyError
        If a required column is absent.
    ValueError
        If the frame or grouping specification is invalid.
    """
    if group_by not in (None, symbol_col, time_col):
        raise ValueError('group_by must be None, symbol_col, or time_col.')
    selected_frame, value_col = _prepare_value_frame(
        value_frame, value_col, time_col, symbol_col,
    )
    polars_frame = (
        pl.from_pandas(selected_frame)
        if isinstance(selected_frame, pd.DataFrame) else selected_frame
    )
    groups = (
        [polars_frame]
        if group_by is None
        else (group for _, group in polars_frame.group_by(group_by, maintain_order=True))
    )
    rows = []
    for group in groups:
        row = {'value_name': value_col, **_summarize_values(group[value_col])}
        if group_by is not None:
            row = {group_by: group[group_by][0], **row}
        rows.append(row)

    if isinstance(value_frame, pd.DataFrame):
        summary = pd.DataFrame(rows)
        return summary.sort_values(time_col).reset_index(drop=True) if group_by == time_col else summary
    summary = pl.DataFrame(rows, infer_schema_length=None)
    return summary.sort(time_col) if group_by == time_col else summary


def rolling_distribution_summary(
        value_frame: pd.DataFrame | pl.DataFrame,
        window_size: int,
        window_step: int = 1,
        value_col: str | None = None,
        time_col: str = 'time',
        symbol_col: str = 'symbol',
) -> pd.DataFrame | pl.DataFrame:
    """Summarize one asset's value distribution in consecutive time windows.

    Parameters
    ----------
    value_frame : pd.DataFrame | pl.DataFrame
        One asset's chronologically ordered feature or target observations.
        An explicit value_col also permits a paired feature-target frame.
    window_size : int
        Number of input rows in each full window.
    window_step : int, default 1
        Number of rows between consecutive window endpoints.
    value_col : str | None, default None
        Numeric value column, inferred from a three-column frame when omitted.
    time_col, symbol_col : str
        Names of the temporal and asset key columns.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Input-backend table with one row per full window, its start and end
        times, and the same metrics as distribution_summary().

    Raises
    ------
    TypeError
        If the frame or value column has an unsupported type.
    KeyError
        If a required column is absent.
    ValueError
        If the keys, ordering, or window parameters are invalid.

    Notes
    -----
    Unlike rolling temporal association, distribution statistics use finite
    feature values even when other rows in the window are missing. Counts make
    this distinction visible without changing the association methodology.
    """
    selected_frame, value_col = _prepare_value_frame(
        value_frame, value_col, time_col, symbol_col,
    )
    _validate_positive_integer(window_size, 'window_size')
    _validate_positive_integer(window_step, 'window_step')
    if (
            selected_frame[symbol_col].nunique()
            if isinstance(selected_frame, pd.DataFrame)
            else selected_frame[symbol_col].n_unique()
    ) != 1:
        raise ValueError('rolling distribution requires exactly one symbol.')
    _validate_time_order(selected_frame, time_col)
    if window_size > len(selected_frame):
        raise ValueError('window_size must not exceed the number of observations.')

    polars_frame = (
        pl.from_pandas(selected_frame)
        if isinstance(selected_frame, pd.DataFrame) else selected_frame
    )
    raw = pl.col(value_col).cast(pl.Float64)
    finite = pl.when(raw.is_finite()).then(raw).otherwise(None)
    full_window = {_ROLLING_MINIMUM_KEYWORD: window_size}
    one_valid = {_ROLLING_MINIMUM_KEYWORD: 1}
    minimum = finite.rolling_min(window_size, **one_valid)
    maximum = finite.rolling_max(window_size, **one_valid)
    quantiles = {
        name: finite.rolling_quantile(
            level, interpolation='linear', window_size=window_size, **one_valid,
        )
        for name, level in _QUANTILES
    }
    std = (
        finite.rolling_std(window_size, ddof=1, **{_ROLLING_MINIMUM_KEYWORD: 2})
        if window_size >= 2 else pl.lit(None, dtype=pl.Float64)
    )
    skewness = (
        finite.rolling_skew(window_size, bias=False, **{_ROLLING_MINIMUM_KEYWORD: 3})
        if window_size >= 3 else pl.lit(None, dtype=pl.Float64)
    )
    excess_kurtosis = (
        finite.rolling_kurtosis(
            window_size, fisher=True, bias=False, **{_ROLLING_MINIMUM_KEYWORD: 4},
        )
        if window_size >= 4 else pl.lit(None, dtype=pl.Float64)
    )

    summary = polars_frame.select(
        pl.col(symbol_col),
        pl.lit(value_col).alias('value_name'),
        pl.col(time_col).shift(window_size - 1).alias('window_start'),
        pl.col(time_col).alias('window_end'),
        pl.lit(window_size).alias('n_total'),
        finite.is_not_null().cast(pl.Int64).rolling_sum(
            window_size, **full_window,
        ).alias('n_valid'),
        (raw.is_null() | raw.is_nan().fill_null(False)).cast(pl.Int64).rolling_sum(
            window_size, **full_window,
        ).alias('n_missing'),
        raw.is_infinite().fill_null(False).cast(pl.Int64).rolling_sum(
            window_size, **full_window,
        ).alias('n_infinite'),
        finite.rolling_mean(window_size, **one_valid).alias('mean'),
        std.alias('std'),
        minimum.alias('min'),
        *(expression.alias(name) for name, expression in quantiles.items()),
        maximum.alias('max'),
        (quantiles['q75'] - quantiles['q25']).alias('iqr'),
        pl.when(minimum != maximum).then(skewness).otherwise(None).alias('skewness'),
        pl.when(minimum != maximum).then(excess_kurtosis).otherwise(None).alias('excess_kurtosis'),
    )
    position = pl.int_range(pl.len())
    summary = summary.filter(
        (position >= window_size - 1)
        & ((position - (window_size - 1)) % window_step == 0)
    )
    return summary.to_pandas() if isinstance(value_frame, pd.DataFrame) else summary
