"""Shared data and figure-lifetime contracts for diagnostic plots."""

from contextlib import contextmanager

import numpy as np
import pandas as pd
import polars as pl


def _plot_numeric_values(series: pd.Series | pl.Series, name: str) -> np.ndarray:
    """Convert numeric plotting data and preserve gaps as NaN.

    Parameters
    ----------
    series : pd.Series | pl.Series
        Real numeric values or numeric strings, optionally missing.
    name : str
        Column name used in validation errors.

    Returns
    -------
    np.ndarray
        Float values in input order, with non-finite observations set to NaN.

    Raises
    ------
    TypeError
        If the input is not a supported Series.
    ValueError
        If values are boolean, complex, temporal, or cannot be converted.
    """
    if isinstance(series, pl.Series):
        series = series.to_pandas()
    if not isinstance(series, pd.Series):
        raise TypeError('plot values must be a Pandas or Polars Series.')
    if (pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_complex_dtype(series.dtype)
            or pd.api.types.is_datetime64_any_dtype(series.dtype)
            or pd.api.types.is_timedelta64_dtype(series.dtype)
            or any(isinstance(value, (bool, np.bool_)) for value in series.dropna())):
        raise ValueError(f'{name} must contain real numeric values.')
    try:
        numeric = pd.to_numeric(series, errors='raise')
        if pd.api.types.is_complex_dtype(numeric.dtype):
            raise ValueError('complex plotting values are unsupported.')
        values = numeric.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must contain real numeric values.') from error
    values = values.copy()
    values[~np.isfinite(values)] = np.nan
    return values


def _plot_decisions(series: pd.Series | pl.Series, name: str) -> np.ndarray:
    """Validate boolean significance decisions without truth-value coercion.

    Parameters
    ----------
    series : pd.Series | pl.Series
        Boolean decisions. Missing decisions use the neutral, false style.
    name : str
        Column name used in validation errors.

    Returns
    -------
    np.ndarray
        Boolean decisions in input order.

    Raises
    ------
    TypeError
        If the input is not a supported Series.
    ValueError
        If a non-missing decision is not boolean.
    """
    if isinstance(series, pl.Series):
        series = series.to_pandas()
    if not isinstance(series, pd.Series):
        raise TypeError('plot decisions must be a Pandas or Polars Series.')
    missing = series.isna().to_numpy()
    values = series.to_list()
    if any(not absent and not isinstance(value, (bool, np.bool_))
           for value, absent in zip(values, missing)):
        raise ValueError(f'{name} must contain boolean decisions or missing values.')
    return np.asarray([False if absent else value for value, absent in zip(values, missing)], dtype=bool)


def _valid_plot_intervals(values: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                          *, contain_estimate: bool = False) -> np.ndarray:
    """Validate finite interval geometry and identify drawable intervals.

    Parameters
    ----------
    values, lower, upper : np.ndarray
        Aligned one-dimensional float arrays. Non-finite data marks a gap.
    contain_estimate : bool, default False
        Require containment for centered error bars, such as Wald intervals.
        Percentile bootstrap bands do not require containment.

    Returns
    -------
    np.ndarray
        Mask selecting finite estimates with two finite, valid bounds.

    Raises
    ------
    TypeError
        If inputs are not numeric arrays or contain_estimate is not boolean.
    ValueError
        If dimensions differ, bounds are reversed, or required containment fails.
    """
    if type(contain_estimate) is not bool or any(
            not isinstance(array, np.ndarray) or array.dtype.kind not in 'iuf'
            for array in (values, lower, upper)):
        raise TypeError('interval inputs must be numeric arrays and contain_estimate must be boolean.')
    if any(array.ndim != 1 or array.shape != values.shape for array in (values, lower, upper)):
        raise ValueError('interval inputs must be aligned one-dimensional arrays.')
    finite_limits = np.isfinite(lower) & np.isfinite(upper)
    if np.any(lower[finite_limits] > upper[finite_limits]):
        raise ValueError('lower confidence limits must not exceed upper confidence limits.')
    valid = finite_limits & np.isfinite(values)
    if contain_estimate and np.any((lower[valid] > values[valid]) | (upper[valid] < values[valid])):
        raise ValueError('confidence limits must contain the plotted values.')
    return valid


@contextmanager
def _close_on_plot_error(ax, created: bool):
    """Release an internally created figure if drawing fails.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis owned by the caller or created for this plotting operation.
    created : bool
        Whether this operation owns the figure.

    Returns
    -------
    contextlib.AbstractContextManager
        Context yielding the axis and retaining successful figures for the caller.

    Raises
    ------
    TypeError
        If created is not boolean.
    BaseException
        Reraises a drawing failure after closing an owned figure.
    """
    if type(created) is not bool:
        raise TypeError('created must be boolean.')
    try:
        yield ax
    except BaseException:
        if created:
            import matplotlib.pyplot as plt

            plt.close(ax.figure)
        raise
