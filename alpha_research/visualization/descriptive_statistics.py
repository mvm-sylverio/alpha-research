from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import gaussian_kde, probplot

from alpha_research._utils import _validate_df, _validate_time_order
from alpha_research.evaluation.descriptive_statistics import _prepare_value_frame

__all__ = [
    'plot_distribution_histogram',
    'plot_distribution_qq',
    'plot_distribution_quantiles',
]


def _select_plot_values(
        value_frame: pd.DataFrame | pl.DataFrame,
        symbol: str | None,
        value_col: str | None,
        time_col: str,
        symbol_col: str,
) -> tuple[np.ndarray, str]:
    """Select finite observations of one asset for distribution plots.

    Parameters
    ----------
    value_frame : pd.DataFrame | pl.DataFrame
        Feature or target observations.
    symbol : str | None
        Asset to plot; optional only for a single-asset frame.
    value_col : str | None
        Value column, inferred from a three-column frame when omitted.
    time_col, symbol_col : str
        Names of the key columns.

    Returns
    -------
    tuple[np.ndarray, str]
        Finite values of the selected asset and their column name.

    Raises
    ------
    TypeError
        If the input or value column type is unsupported.
    KeyError
        If a required column is absent.
    ValueError
        If no unique asset can be selected or it has no finite values.
    """
    selected_frame, value_col = _prepare_value_frame(
        value_frame, value_col, time_col, symbol_col,
    )
    if symbol is None:
        n_symbols = (
            selected_frame[symbol_col].nunique()
            if isinstance(selected_frame, pd.DataFrame)
            else selected_frame[symbol_col].n_unique()
        )
        if n_symbols != 1:
            raise ValueError('symbol must be provided for a multi-asset frame.')
        symbol = (
            selected_frame[symbol_col].iloc[0]
            if isinstance(selected_frame, pd.DataFrame)
            else selected_frame[symbol_col][0]
        )
    if isinstance(selected_frame, pd.DataFrame):
        selected = selected_frame.loc[selected_frame[symbol_col] == symbol, value_col]
        numeric = selected.to_numpy(dtype=float, na_value=np.nan)
    else:
        selected = selected_frame.filter(pl.col(symbol_col) == symbol)[value_col]
        numeric = selected.cast(pl.Float64).to_numpy()
    if len(selected) == 0:
        raise ValueError(f'symbol {symbol!r} is not present in value_frame.')
    finite = numeric[np.isfinite(numeric)]
    if len(finite) == 0:
        raise ValueError('selected symbol must contain a finite value.')
    return finite, value_col


def plot_distribution_histogram(
        value_frame: pd.DataFrame | pl.DataFrame,
        symbol: str | None = None,
        value_col: str | None = None,
        ax: Any = None,
        bins: int | str = 'auto',
        show_density: bool = True,
        show_quantiles: bool = True,
        color: str = 'C0',
        time_col: str = 'time',
        symbol_col: str = 'symbol',
        density_color: str | None = None,
) -> Any:
    """Plot a histogram and optional kernel density for one asset's values.

    Parameters
    ----------
    value_frame : pd.DataFrame | pl.DataFrame
        Feature or target observations, optionally filtered to a time period.
    symbol : str | None, default None
        Asset to plot, required if the frame contains multiple assets.
    value_col : str | None, default None
        Numeric value column, inferred from a three-column frame when omitted.
    ax : matplotlib.axes.Axes | None, default None
        Axis to draw into; a new axis is created if omitted.
    bins : int | str, default 'auto'
        Histogram bin specification passed to Matplotlib.
    show_density : bool, default True
        Draw a Gaussian kernel density if at least two distinct values exist.
    show_quantiles : bool, default True
        Mark q05, median, and q95; mark only the median for constant values.
    color : str, default 'C0'
        Histogram and density color.
    time_col, symbol_col : str
        Names of the key columns.
    density_color : str | None, default None
        Optional color for the density curve; None uses ``color``.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the histogram and optional density line.

    Raises
    ------
    ImportError
        If Matplotlib is not installed.
    TypeError
        If the input or a boolean plotting parameter is invalid.
    ValueError
        If the selection or histogram bin specification is invalid.
    """
    for name, value in [('show_density', show_density), ('show_quantiles', show_quantiles)]:
        if not isinstance(value, bool):
            raise TypeError(f'{name} must be a boolean.')
    values, value_col = _select_plot_values(
        value_frame, symbol, value_col, time_col, symbol_col,
    )
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            'Matplotlib is required for plotting. Install alpha-research[viz].',
        ) from error
    if ax is None:
        _, ax = plt.subplots()

    ax.hist(values, bins=bins, density=True, color=color, alpha=0.35)
    if show_density and len(values) >= 2 and np.min(values) != np.max(values):
        grid = np.linspace(np.min(values), np.max(values), 200)
        ax.plot(grid, gaussian_kde(values)(grid), color=density_color or color)
    if show_quantiles:
        for quantile in ([0.5] if np.min(values) == np.max(values) else [0.05, 0.5, 0.95]):
            ax.axvline(
                float(np.quantile(values, quantile)),
                color=color,
                linestyle='--' if quantile == 0.5 else ':',
                linewidth=1.0,
            )
    ax.set_xlabel(value_col)
    ax.set_ylabel('Density')
    return ax


def plot_distribution_qq(
        value_frame: pd.DataFrame | pl.DataFrame,
        symbol: str | None = None,
        value_col: str | None = None,
        ax: Any = None,
        color: str = 'C0',
        time_col: str = 'time',
        symbol_col: str = 'symbol',
        reference_color: str | None = None,
) -> Any:
    """Plot one asset's empirical quantiles against Normal quantiles.

    Parameters
    ----------
    value_frame : pd.DataFrame | pl.DataFrame
        Feature or target observations, optionally filtered to a time period.
    symbol : str | None, default None
        Asset to plot, required if the frame contains multiple assets.
    value_col : str | None, default None
        Numeric value column, inferred from a three-column frame when omitted.
    ax : matplotlib.axes.Axes | None, default None
        Axis to draw into; a new axis is created if omitted.
    color : str, default 'C0'
        Point and reference-line color.
    time_col, symbol_col : str
        Names of the key columns.
    reference_color : str | None, default None
        Optional color for the fitted reference line; None uses ``color``.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing Normal theoretical quantiles and observed values.

    Raises
    ------
    ImportError
        If Matplotlib is not installed.
    TypeError
        If the frame or its value column type is unsupported.
    ValueError
        If the selection has fewer than two finite observations.

    Notes
    -----
    The fitted reference line is visual guidance, not a normality test.
    """
    values, value_col = _select_plot_values(
        value_frame, symbol, value_col, time_col, symbol_col,
    )
    if len(values) < 2:
        raise ValueError('Q-Q plot requires at least two finite values.')
    theoretical_and_observed, fit = probplot(values, dist='norm', fit=True)
    theoretical, observed = theoretical_and_observed
    slope, intercept, _ = fit
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            'Matplotlib is required for plotting. Install alpha-research[viz].',
        ) from error
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(theoretical, observed, '.', color=color)
    ax.plot(theoretical, slope * theoretical + intercept, '--', color=reference_color or color)
    ax.set_xlabel('Normal theoretical quantiles')
    ax.set_ylabel(f'Observed {value_col} quantiles')
    return ax


def plot_distribution_quantiles(
        summary_frame: pd.DataFrame | pl.DataFrame,
        time_col: str = 'time',
        outer_lower_col: str = 'q05',
        outer_upper_col: str = 'q95',
        ax: Any = None,
        color: str = 'C0',
) -> Any:
    """Plot temporal median and quantile bands from grouped or rolling results.

    Parameters
    ----------
    summary_frame : pd.DataFrame | pl.DataFrame
        Per-date distribution summary or single-asset rolling summary.
    time_col : str, default 'time'
        Temporal key; use 'window_end' for rolling results.
    outer_lower_col, outer_upper_col : str
        Outer quantile columns, defaulting to q05 and q95.
    ax : matplotlib.axes.Axes | None, default None
        Axis to draw into; a new axis is created if omitted.
    color : str, default 'C0'
        Median line and band color.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the median, q25-q75, and outer quantile bands.

    Raises
    ------
    ImportError
        If Matplotlib is not installed.
    TypeError
        If the summary frame is not Pandas or Polars or metrics are nonnumeric.
    KeyError
        If a required result column is absent.
    ValueError
        If temporal keys are invalid, no finite median exists, or bands are
        inconsistent.
    """
    columns = [time_col, outer_lower_col, 'q25', 'median', 'q75', outer_upper_col]
    _validate_df(summary_frame, columns, check_all_missing=False)
    _validate_time_order(summary_frame, time_col)
    try:
        quantiles = (
            summary_frame[columns[1:]].to_numpy(dtype=float)
            if isinstance(summary_frame, pd.DataFrame)
            else summary_frame.select(columns[1:]).to_numpy().astype(float)
        )
    except (TypeError, ValueError) as error:
        raise TypeError('quantile columns must be numeric.') from error
    if not np.isfinite(quantiles[:, 2]).any():
        raise ValueError('summary_frame must contain a finite median.')
    complete = np.isfinite(quantiles).all(axis=1)
    if np.isfinite(quantiles).any(axis=1).sum() != complete.sum():
        raise ValueError('quantile rows must be entirely finite or missing.')
    if np.any(np.diff(quantiles[complete], axis=1) < 0):
        raise ValueError('quantile bands must be ordered from low to high.')
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            'Matplotlib is required for plotting. Install alpha-research[viz].',
        ) from error
    if ax is None:
        _, ax = plt.subplots()

    times = summary_frame[time_col].to_numpy()
    outer_label = f'{outer_lower_col.upper()}–{outer_upper_col.upper()}'
    ax.fill_between(times, quantiles[:, 0], quantiles[:, 4], color=color, alpha=0.12, label=outer_label)
    ax.fill_between(times, quantiles[:, 1], quantiles[:, 3], color=color, alpha=0.25, label='Q25–Q75')
    ax.plot(times, quantiles[:, 2], color=color, label='Median')
    ax.set_xlabel(time_col)
    ax.set_ylabel('Value')
    return ax
