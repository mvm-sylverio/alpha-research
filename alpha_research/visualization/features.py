from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from alpha_research._utils import _validate_df
from alpha_research.features.schema import get_feature_name

__all__ = [
    'plot_time_series_value',
    'plot_cross_sectional_value_summary',
]


def plot_time_series_value(
        value_frame: pd.DataFrame | pl.DataFrame,
        symbol: str | None = None,
        value_col: str | None = None,
        ax: Any = None,
        color: str = 'C0',
        label: str | None = None,
        title: str | None = None,
        time_col: str = 'time',
        symbol_col: str = 'symbol',
) -> Any:
    """
    Plot one feature or target value through time for a single asset.

    Parameters
    ----------
    value_frame : pd.DataFrame | pl.DataFrame
        A feature or target frame with time_col, symbol_col, and exactly one
        value column when value_col is omitted.
    symbol : str | None, default None
        Asset to plot. It is required when value_frame contains more than one
        non-missing symbol; otherwise the single available symbol is used.
    value_col : str | None, default None
        Value column to plot. When omitted, the single non-key value column is
        resolved from the standard feature/target frame contract.
    ax : matplotlib.axes.Axes | None, default None
        Axis to draw into. When omitted, a simple new figure and axis are
        created.
    color : str, default 'C0'
        Matplotlib color for the plotted value line.
    label : str | None, default None
        Optional line label. The value column name is used when omitted.
    title : str | None, default None
        Optional axis title.
    time_col, symbol_col : str
        Key-column names used by the feature or target frame.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the single selected asset's value line.

    Raises
    ------
    ImportError
        If Matplotlib is not installed. Install the optional ``viz`` extra.
    KeyError
        If a required key or value column is absent.
    TypeError
        If value_frame is not a Pandas or Polars DataFrame.
    ValueError
        If the frame has multiple value columns without value_col, symbol is
        ambiguous or absent, or the selected series has no finite values.

    Notes
    -----
    This function is deliberately agnostic to whether the value is a feature
    or a target. It preserves input row order and does not sort the time axis;
    callers should provide chronologically ordered feature or target frames.
    """
    required_columns = [time_col, symbol_col]
    if value_col is not None:
        required_columns.append(value_col)
    _validate_df(value_frame, required_columns, check_all_missing=False)

    if value_col is None:
        value_col = get_feature_name(
            value_frame,
            time_col=time_col,
            symbol_col=symbol_col,
        )

    pandas_frame = (
        value_frame.copy()
        if isinstance(value_frame, pd.DataFrame)
        else value_frame.to_pandas()
    )
    available_symbols = pandas_frame[symbol_col].dropna().unique().tolist()

    if symbol is None:
        if len(available_symbols) != 1:
            raise ValueError(
                'symbol must be provided when value_frame contains more than '
                'one symbol.',
            )
        symbol = available_symbols[0]

    selected = pandas_frame.loc[
        pandas_frame[symbol_col] == symbol,
        [time_col, value_col],
    ]
    if selected.empty:
        raise ValueError(f'symbol {symbol!r} is not present in value_frame.')

    values = selected[value_col].to_numpy(dtype=float)
    if not np.isfinite(values).any():
        raise ValueError('selected value series must contain at least one finite value.')

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            'Matplotlib is required for plotting. Install alpha-research[viz].',
        ) from error

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(
        selected[time_col].to_numpy(),
        values,
        color=color,
        label=value_col if label is None else label,
    )
    ax.set_xlabel(time_col)
    ax.set_ylabel(value_col)
    if title is not None:
        ax.set_title(title)

    return ax


def plot_cross_sectional_value_summary(
        value_frame: pd.DataFrame | pl.DataFrame,
        value_col: str | None = None,
        lower_quantile: float = 0.25,
        upper_quantile: float = 0.75,
        ax: Any = None,
        color: str = 'C0',
        band_alpha: float = 0.20,
        label: str | None = None,
        title: str | None = None,
        time_col: str = 'time',
        symbol_col: str = 'symbol',
) -> Any:
    """
    Plot a per-date cross-sectional value median and quantile band.

    Parameters
    ----------
    value_frame : pd.DataFrame | pl.DataFrame
        A feature or target frame with time_col, symbol_col, and exactly one
        value column when value_col is omitted.
    value_col : str | None, default None
        Value column summarized across assets. When omitted, the single
        non-key value column is resolved from the standard frame contract.
    lower_quantile, upper_quantile : float, default 0.25, 0.75
        Inclusive quantile bounds for the shaded cross-sectional band. The
        lower value must be strictly smaller than the upper value.
    ax : matplotlib.axes.Axes | None, default None
        Axis to draw into. When omitted, a simple new figure and axis are
        created.
    color : str, default 'C0'
        Matplotlib color for the median line and quantile band.
    band_alpha : float, default 0.20
        Opacity of the quantile band, constrained to [0, 1].
    label : str | None, default None
        Optional median-line label. A value-column-based label is used when
        omitted.
    title : str | None, default None
        Optional axis title.
    time_col, symbol_col : str
        Key-column names used by the feature or target frame. symbol_col is
        validated to preserve the cross-sectional data contract.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the cross-sectional median line and quantile band.

    Raises
    ------
    ImportError
        If Matplotlib is not installed. Install the optional ``viz`` extra.
    KeyError
        If a required key or value column is absent.
    TypeError
        If value_frame is not a Pandas or Polars DataFrame or a plotting
        parameter has an invalid type.
    ValueError
        If the frame has multiple value columns without value_col, quantile or
        band_alpha parameters are invalid, or no finite values are available.

    Notes
    -----
    This function is deliberately agnostic to whether the value is a feature
    or a target. Each displayed date summarizes the values across all available
    symbols at that date. It is not a temporal association calculation.
    """
    for name, value in [
        ('lower_quantile', lower_quantile),
        ('upper_quantile', upper_quantile),
        ('band_alpha', band_alpha),
    ]:
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise TypeError(f'{name} must be a number.')

    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError(
            'lower_quantile and upper_quantile must satisfy '
            '0 <= lower_quantile < upper_quantile <= 1.',
        )
    if not 0 <= band_alpha <= 1:
        raise ValueError('band_alpha must be a number between zero and one.')

    required_columns = [time_col, symbol_col]
    if value_col is not None:
        required_columns.append(value_col)
    _validate_df(value_frame, required_columns, check_all_missing=False)

    if value_col is None:
        value_col = get_feature_name(
            value_frame,
            time_col=time_col,
            symbol_col=symbol_col,
        )

    pandas_frame = (
        value_frame[[time_col, symbol_col, value_col]].copy()
        if isinstance(value_frame, pd.DataFrame)
        else value_frame.select([time_col, symbol_col, value_col]).to_pandas()
    )
    pandas_frame[value_col] = pd.to_numeric(pandas_frame[value_col], errors='coerce')
    summary = pandas_frame.groupby(time_col, sort=False)[value_col].agg(
        median='median',
        lower=lambda values: values.quantile(lower_quantile),
        upper=lambda values: values.quantile(upper_quantile),
    ).reset_index()

    if not np.isfinite(summary['median'].to_numpy(dtype=float)).any():
        raise ValueError('value_frame must contain at least one finite value.')

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            'Matplotlib is required for plotting. Install alpha-research[viz].',
        ) from error

    if ax is None:
        _, ax = plt.subplots()

    times = summary[time_col].to_numpy()
    lower = summary['lower'].to_numpy(dtype=float)
    upper = summary['upper'].to_numpy(dtype=float)
    median = summary['median'].to_numpy(dtype=float)
    ax.fill_between(times, lower, upper, color=color, alpha=band_alpha)
    ax.plot(
        times,
        median,
        color=color,
        label=f'{value_col} median' if label is None else label,
    )
    ax.set_xlabel(time_col)
    ax.set_ylabel(value_col)
    if title is not None:
        ax.set_title(title)

    return ax
