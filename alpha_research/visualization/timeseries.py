from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from alpha_research._utils import _validate_df, _validate_time_order
from alpha_research.visualization._utils import (
    _close_on_plot_error,
    _plot_numeric_values,
    _valid_plot_intervals,
)

__all__ = ['plot_rolling_temporal_association']


def plot_rolling_temporal_association(
        rolling_frame: pd.DataFrame | pl.DataFrame,
        ax: Any = None,
        time_col: str = 'window_end',
        association_col: str = 'association',
        ci_lower_col: str = 'bootstrap_ci_lower',
        ci_upper_col: str = 'bootstrap_ci_upper',
        color: str = 'C0',
        band_alpha: float = 0.20,
        label: str = 'Rolling temporal association',
        title: str | None = None,
) -> Any:
    """
    Plot rolling temporal association with a percentile bootstrap band.

    Parameters
    ----------
    rolling_frame : pd.DataFrame | pl.DataFrame
        Rolling result frame returned by rolling_temporal_association().
    ax : matplotlib.axes.Axes | None, default None
        Axis to draw into. When omitted, a simple new Matplotlib figure and
        axis are created. Supplying an axis lets callers compose this panel
        with application-specific subplots.
    time_col, association_col, ci_lower_col, ci_upper_col : str
        Column names used for the temporal coordinate, observed association,
        and percentile bootstrap band.
    color : str, default 'C0'
        Matplotlib color for the observed-association line.
    band_alpha : float, default 0.20
        Opacity of the bootstrap interval fill. It must be in [0, 1].
    label : str, default 'Rolling temporal association'
        Line label passed to Matplotlib.
    title : str | None, default None
        Optional axis title.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the association line, bootstrap band, and zero
        reference line.

    Raises
    ------
    ImportError
        If Matplotlib is not installed. Install the optional ``viz`` extra.
    ValueError
        If band_alpha, numeric values, or interval order is invalid, no finite
        association exists, or times are missing, duplicated, or decreasing.

    Notes
    -----
    This is intentionally a narrow, static diagnostic plot. It does not
    create contextual market metrics or application-specific layouts; callers
    can pass an existing axis to compose it with their own shared-time panels.
    Times must be unique and increasing. Non-finite values produce gaps; a band
    is shown only where the estimate and both bounds are finite. Finite bounds
    must be ordered, but a percentile interval may exclude the point estimate.
    """
    if not isinstance(band_alpha, (int, float, np.integer, np.floating)) or isinstance(band_alpha, (bool, np.timedelta64)):
        raise ValueError('band_alpha must be a number between zero and one.')

    if not 0.0 <= band_alpha <= 1.0:
        raise ValueError('band_alpha must be a number between zero and one.')

    _validate_df(
        rolling_frame,
        [time_col, association_col, ci_lower_col, ci_upper_col],
        check_all_missing=False,
    )
    _validate_time_order(rolling_frame, time_col)

    pandas_frame = (
        rolling_frame
        if isinstance(rolling_frame, pd.DataFrame)
        else rolling_frame.to_pandas()
    )
    association = _plot_numeric_values(pandas_frame[association_col], association_col)
    ci_lower = _plot_numeric_values(pandas_frame[ci_lower_col], ci_lower_col)
    ci_upper = _plot_numeric_values(pandas_frame[ci_upper_col], ci_upper_col)
    valid_intervals = _valid_plot_intervals(association, ci_lower, ci_upper)
    ci_lower[~valid_intervals] = np.nan
    ci_upper[~valid_intervals] = np.nan

    if not np.isfinite(association).any():
        raise ValueError('rolling_frame must contain at least one finite association.')

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            'Matplotlib is required for plotting. Install alpha-research[viz].',
        ) from error

    created_axis = ax is None
    if created_axis:
        _, ax = plt.subplots()

    times = pandas_frame[time_col].to_numpy()
    with _close_on_plot_error(ax, created_axis):
        ax.fill_between(times, ci_lower, ci_upper, color=color, alpha=band_alpha)
        ax.plot(times, association, color=color, label=label)
        ax.axhline(0.0, color='black', linewidth=1.0, linestyle='--')
        ax.set_xlabel(time_col)
        ax.set_ylabel('Temporal association')

        if title is not None:
            ax.set_title(title)

    return ax
