from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from alpha_research._utils import _validate_df

__all__ = ['plot_decay_curves']


def plot_decay_curves(
        decay_tables: Mapping[str, pd.DataFrame | pl.DataFrame],
        value_col: str,
        ax: Any,
        horizon_col: str = 'horizon',
        ci_lower_col: str | None = None,
        ci_upper_col: str | None = None,
        significance_col: str | None = 'fdr_rejected',
) -> Any:
    """
    Plot one or more signed decay curves on an existing Matplotlib axis.

    Each mapping entry supplies one labelled curve. The same function applies
    to IC decay tables, using ``value_col='mean'``, and temporal-association
    decay tables, using ``value_col='association'``. Optional confidence
    limits are rendered as discrete error bars at each horizon.

    Parameters
    ----------
    decay_tables : Mapping[str, pd.DataFrame | pl.DataFrame]
        Non-empty mapping from a curve label to its detailed decay table. Each
        table must contain horizon_col and value_col, with exactly one row per
        horizon. Tables may independently use Pandas or Polars.
    value_col : str
        Signed decay metric column to plot on the y-axis, such as ``'mean'``
        for IC decay or ``'association'`` for temporal-association decay.
    ax : matplotlib.axes.Axes
        Existing axis to draw into. This function never creates a figure or
        arranges a layout, allowing callers to compose the plot externally.
    horizon_col : str, default 'horizon'
        Numeric horizon column plotted on the x-axis.
    ci_lower_col : str | None, default None
        Optional lower confidence-limit column. It must be provided together
        with ci_upper_col. Error bars are only drawn for finite triplets of
        value, lower limit, and upper limit.
    ci_upper_col : str | None, default None
        Optional upper confidence-limit column. It must be provided together
        with ci_lower_col.
    significance_col : str | None, default 'fdr_rejected'
        Optional boolean significance column. Finite points with a true value
        receive an outlined marker over the curve marker. Pass None when the
        supplied tables do not contain an FDR result.

    Returns
    -------
    matplotlib.axes.Axes
        The supplied axis after all curves, optional error bars, significant
        point outlines, and the zero reference line have been drawn.

    Raises
    ------
    TypeError
        If decay_tables is not a mapping or a mapped table is not a Pandas or
        Polars DataFrame.
    ValueError
        If decay_tables is empty, a table has duplicate horizons, no finite
        plotted values, non-numeric horizon or value data, invalid confidence
        limits, or only one confidence-limit column is supplied.
    KeyError
        If a required plot column is absent from a mapped table.
    ImportError
        If Matplotlib is not installed. Install the optional ``viz`` extra.

    Notes
    -----
    The line connects the observed signed metric at the requested discrete
    horizons. It does not imply that untested horizons were estimated.
    Error bars are appropriate only when the input explicitly contains
    interval bounds, such as the Wald limits from temporal-association decay.
    IC-decay ``std`` is not interpreted as an interval by this function.
    """
    if not isinstance(decay_tables, Mapping):
        raise TypeError('decay_tables must be a mapping from labels to DataFrames.')

    if not decay_tables:
        raise ValueError('decay_tables must not be empty.')

    if (ci_lower_col is None) != (ci_upper_col is None):
        raise ValueError(
            'ci_lower_col and ci_upper_col must be provided together.',
        )

    required_columns = [horizon_col, value_col]
    if ci_lower_col is not None:
        required_columns.extend([ci_lower_col, ci_upper_col])
    if significance_col is not None:
        required_columns.append(significance_col)

    prepared_curves = []
    for label, decay_table in decay_tables.items():
        _validate_df(
            decay_table,
            required_columns,
            check_all_missing=False,
        )
        pandas_table = (
            decay_table
            if isinstance(decay_table, pd.DataFrame)
            else decay_table.to_pandas()
        ).sort_values(horizon_col)

        try:
            horizons = pandas_table[horizon_col].to_numpy(dtype=float)
            values = pandas_table[value_col].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'{horizon_col} and {value_col} must contain numeric values.',
            ) from error

        if len(np.unique(horizons)) != len(horizons):
            raise ValueError('each decay table must contain one row per horizon.')

        if not np.isfinite(horizons).all():
            raise ValueError(f'{horizon_col} must contain only finite values.')

        if not np.isfinite(values).any():
            raise ValueError(
                f"decay table '{label}' must contain at least one finite {value_col}.",
            )

        if ci_lower_col is not None:
            try:
                ci_lower = pandas_table[ci_lower_col].to_numpy(dtype=float)
                ci_upper = pandas_table[ci_upper_col].to_numpy(dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError('confidence-limit columns must contain numeric values.') from error

            finite_limits = np.isfinite(ci_lower) & np.isfinite(ci_upper)
            if (ci_lower[finite_limits] > ci_upper[finite_limits]).any():
                raise ValueError('ci_lower_col must not exceed ci_upper_col.')

            finite_intervals = finite_limits & np.isfinite(values)
            if (
                    (ci_lower[finite_intervals] > values[finite_intervals]).any()
                    or (ci_upper[finite_intervals] < values[finite_intervals]).any()
            ):
                raise ValueError(
                    'confidence limits must contain the plotted values.',
                )
        else:
            ci_lower = None
            ci_upper = None

        if significance_col is None:
            significant = None
        else:
            significant = pandas_table[significance_col].fillna(False).to_numpy(
                dtype=bool,
            )

        prepared_curves.append((
            str(label),
            horizons,
            values,
            ci_lower,
            ci_upper,
            significant,
        ))

    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError as error:
        raise ImportError(
            'Matplotlib is required for plotting. Install alpha-research[viz].',
        ) from error

    for label, horizons, values, ci_lower, ci_upper, significant in prepared_curves:
        line = ax.plot(horizons, values, marker='o', label=label)[0]
        color = line.get_color()

        if ci_lower is not None:
            finite_interval = (
                np.isfinite(values)
                & np.isfinite(ci_lower)
                & np.isfinite(ci_upper)
            )
            if finite_interval.any():
                ax.errorbar(
                    horizons[finite_interval],
                    values[finite_interval],
                    yerr=np.vstack([
                        values[finite_interval] - ci_lower[finite_interval],
                        ci_upper[finite_interval] - values[finite_interval],
                    ]),
                    fmt='none',
                    ecolor=color,
                    capsize=3,
                    alpha=0.75,
                    zorder=line.get_zorder() - 1,
                )

        if significant is not None:
            significant_points = significant & np.isfinite(values)
            if significant_points.any():
                ax.scatter(
                    horizons[significant_points],
                    values[significant_points],
                    facecolors='none',
                    edgecolors=color,
                    linewidths=1.5,
                    s=65,
                    zorder=line.get_zorder() + 1,
                )

    ax.axhline(0.0, color='black', linewidth=1.0, linestyle='--')

    return ax
