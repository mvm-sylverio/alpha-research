"""Compare named numeric distributions without assuming their origin."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
import polars as pl

from alpha_research._utils import _validate_finite_number, _validate_positive_integer
from alpha_research.visualization._utils import (
    _close_on_plot_error,
    _plot_numeric_values,
)

__all__ = [
    'plot_boxplot_comparison',
    'plot_ecdf_comparison',
    'plot_exceedance_comparison',
    'plot_histogram_comparison',
]

DistributionValues: TypeAlias = Sequence[float] | np.ndarray | pd.Series | pl.Series


def _prepare_distributions(distributions: Mapping[str, DistributionValues]) -> dict[str, np.ndarray]:
    """Copy named samples into finite numeric arrays in mapping order.

    Parameters
    ----------
    distributions : Mapping[str, DistributionValues]
        Non-empty mapping of non-blank names to one-dimensional samples.
        Numeric strings follow the existing plotting conversion contract.

    Returns
    -------
    dict[str, np.ndarray]
        Independent Float64 arrays, excluding missing and infinite values.

    Raises
    ------
    TypeError
        If the mapping or a sample container has an unsupported type.
    ValueError
        If names are invalid, samples are not one-dimensional real numeric
        data, or any sample contains no finite observations.
    """
    if not isinstance(distributions, Mapping):
        raise TypeError('distributions must be a mapping of names to numeric samples.')
    if not distributions:
        raise ValueError('distributions must not be empty.')
    prepared = {}
    for name, sample in distributions.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError('distribution names must be non-blank strings.')
        if isinstance(sample, np.ndarray) and sample.ndim != 1:
            raise ValueError(f'{name} must be one-dimensional.')
        if not isinstance(sample, (pd.Series, pl.Series)):
            if isinstance(sample, (str, bytes)) or not isinstance(sample, (Sequence, np.ndarray)):
                raise TypeError(f'{name} must be a sequence, NumPy array, or Pandas or Polars Series.')
            try:
                sample = pd.Series(sample)
            except (ValueError, OverflowError) as error:
                raise ValueError(f'{name} must contain one-dimensional representable numeric values.') from error
        values = _plot_numeric_values(sample, name)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f'{name} must contain at least one finite observation.')
        prepared[name] = finite
    return prepared


def _comparison_histogram_edges(
        distributions: Mapping[str, np.ndarray],
        bins: int | str | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Resolve shared histogram edges covering every prepared observation.

    Parameters
    ----------
    distributions : Mapping[str, np.ndarray]
        Non-empty finite arrays returned by _prepare_distributions().
    bins : int | str | Sequence[float] | np.ndarray
        Positive bin count, NumPy bin estimator, or strictly increasing finite
        edges covering all samples. Automatic edges use the pooled samples.

    Returns
    -------
    np.ndarray
        Shared finite edges with positive, representable bin widths.

    Raises
    ------
    TypeError
        If bins has an unsupported type or explicit edges are not real numeric.
    ValueError
        If counts, estimator, edge geometry, coverage, or numeric range is invalid.
    """
    pooled = np.concatenate(list(distributions.values()))
    if isinstance(bins, (int, np.integer)):
        _validate_positive_integer(bins, 'bins')
        bins = int(bins)
    elif not isinstance(bins, str):
        if isinstance(bins, (bytes, bool, np.bool_)) or not isinstance(bins, (Sequence, np.ndarray)):
            raise TypeError('bins must be a positive integer, estimator name, or numeric edges.')
        edges = np.asarray(bins)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError('bin edges must be one-dimensional with at least two values.')
        if edges.dtype.kind not in 'iuf':
            raise TypeError('bin edges must be real numeric values.')
        bins = edges
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            edges = np.asarray(np.histogram_bin_edges(pooled, bins=bins), dtype=float)
            widths = np.diff(edges)
            if not np.isfinite(edges).all() or not np.isfinite(widths).all() or np.any(widths <= 0):
                raise ValueError('bin edges must be finite and strictly increasing.')
    except (FloatingPointError, OverflowError) as error:
        raise ValueError('histogram range must have representable finite bin widths.') from error
    if edges[0] > pooled.min() or edges[-1] < pooled.max():
        raise ValueError('bin edges must cover every finite observation.')
    return edges


@contextmanager
def _comparison_plot(
        labels: list[str],
        ax: Any,
        *,
        colors: Mapping[str, Any] | None,
        value_label: str,
        title: str | None,
        note: str | None,
        reference_value: float | None,
        reference_label: str,
        value_on_y: bool = False,
) -> Iterator[tuple[Any, list[Any]]]:
    """Validate shared presentation options and manage a comparison figure.

    Parameters
    ----------
    labels : list[str]
        Validated distribution names in input order.
    ax : matplotlib.axes.Axes | None
        Caller-owned axis, or None to create one lazily.
    colors : Mapping[str, Any] | None
        Optional colors keyed by distribution name; omitted names use C0, C1, etc.
    value_label : str
        Label of the numeric value axis.
    title, note : str | None
        Optional title and small annotation inside the lower-left of the axis.
    reference_value : float | None
        Optional finite reference on the numeric value axis.
    reference_label : str
        Non-blank legend label for the reference.
    value_on_y : bool, default False
        Whether values are on the vertical axis, as in box plots.

    Returns
    -------
    contextlib.AbstractContextManager
        Context yielding (axis, colors). Successful figures stay open; failed
        internally created figures close. Caller-owned figures remain open.

    Raises
    ------
    ImportError
        If Matplotlib is unavailable.
    TypeError
        If options or the supplied axis have unsupported types.
    ValueError
        If colors, reference values, or numeric drawing ranges are invalid.
    BaseException
        Propagates other drawing errors after figure cleanup.
    """
    if colors is not None and not isinstance(colors, Mapping):
        raise TypeError('colors must be a mapping or None.')
    if colors is not None and any(name not in labels for name in colors):
        raise ValueError('colors contains unknown distribution names.')
    for name, value in [('value_label', value_label), ('reference_label', reference_label)]:
        if not isinstance(value, str):
            raise TypeError(f'{name} must be a string.')
    if not reference_label.strip():
        raise ValueError('reference_label must not be blank.')
    for name, value in [('title', title), ('note', note)]:
        if value is not None and not isinstance(value, str):
            raise TypeError(f'{name} must be a string or None.')
    if type(value_on_y) is not bool:
        raise TypeError('value_on_y must be boolean.')
    if reference_value is not None:
        reference_value = _validate_finite_number(reference_value, 'reference_value')
    try:
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes
        from matplotlib.colors import is_color_like
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise ImportError('Matplotlib is required for plotting. Install alpha-research[viz].') from error
    if ax is not None and not isinstance(ax, Axes):
        raise TypeError('ax must be a Matplotlib Axes or None.')
    palette = [(colors or {}).get(name, f'C{index % 10}') for index, name in enumerate(labels)]
    if any(not is_color_like(color) for color in palette):
        raise ValueError('colors must contain valid Matplotlib colors.')
    created = ax is None
    if created:
        _, ax = plt.subplots()
    with _close_on_plot_error(ax, created):
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                yield ax, palette
                legend_handles = [] if value_on_y else [Line2D([], [], color=color) for color in palette]
                legend_labels = [] if value_on_y else list(labels)
                if reference_value is not None:
                    draw_reference = ax.axhline if value_on_y else ax.axvline
                    reference_line = draw_reference(reference_value, color='black', linestyle='--', label=reference_label)
                    legend_handles.append(reference_line)
                    legend_labels.append(reference_label)
                if value_on_y:
                    ax.set_ylabel(value_label)
                else:
                    ax.set_xlabel(value_label)
                if title is not None:
                    ax.set_title(title)
                if note is not None:
                    ax.text(.01, .01, note, transform=ax.transAxes, fontsize='small',
                            ha='left', va='bottom',
                            bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
                if not value_on_y or reference_value is not None:
                    # Safe placeholders preserve names starting with underscores,
                    # which older Matplotlib legends otherwise silently omit.
                    legend = ax.legend(legend_handles, [f'Sample {index}' for index in range(len(legend_labels))])
                    for text, label in zip(legend.texts, legend_labels):
                        text.set_text(label)
        except (FloatingPointError, OverflowError) as error:
            raise ValueError('distribution values exceed the representable plotting range.') from error


def plot_histogram_comparison(
        distributions: Mapping[str, DistributionValues],
        ax: Any = None,
        bins: int | str | Sequence[float] | np.ndarray = 'auto',
        stat: Literal['count', 'probability', 'density'] = 'probability',
        *,
        colors: Mapping[str, Any] | None = None,
        value_label: str = 'Value',
        title: str | None = None,
        note: str | None = None,
        reference_value: float | None = None,
        reference_label: str = 'Reference',
) -> Any:
    """Compare named samples using step histograms with shared bin edges.

    Parameters
    ----------
    distributions : Mapping[str, sequence | np.ndarray | pd.Series | pl.Series]
        Named one-dimensional numeric samples, in display order. Numeric
        strings are accepted. Missing and infinite values are excluded per
        sample; each sample must retain at least one finite observation.
    ax : matplotlib.axes.Axes | None, default None
        Existing subplot to use; None creates a new figure.
    bins : int | str | sequence | np.ndarray, default 'auto'
        Shared bin count, NumPy estimator, or increasing finite edges covering
        every sample. The final bin includes its right edge.
    stat : {'count', 'probability', 'density'}, default 'probability'
        Bin count, count divided by that sample's finite size, or density with
        unit area. Probability supports comparison of unequal sample sizes.
    colors : Mapping[str, color] | None, default None
        Colors by sample name; omitted names use the Matplotlib cycle colors.
    value_label : str, default 'Value'
        Horizontal-axis label.
    title, note : str | None, default None
        Optional title and small lower-left annotation describing sample origins.
    reference_value : float | None, default None
        Optional finite value marked by a dashed vertical line.
    reference_label : str, default 'Reference'
        Non-blank legend label for that line.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the overlaid histograms, legend, and optional reference.

    Raises
    ------
    ImportError
        If Matplotlib is unavailable.
    TypeError
        If samples, bins, colors, axis, or presentation options have invalid types.
    ValueError
        If data, names, bins, normalization, colors, or reference are invalid.

    Notes
    -----
    Samples are copied and never combined into a single distribution. Pooling
    is used only to choose common bin edges. This plot performs no inference.
    """
    prepared = _prepare_distributions(distributions)
    if stat not in ('count', 'probability', 'density'):
        raise ValueError('stat must be count, probability, or density.')
    edges = _comparison_histogram_edges(prepared, bins)
    with _comparison_plot(list(prepared), ax, colors=colors, value_label=value_label,
                          title=title, note=note, reference_value=reference_value,
                          reference_label=reference_label) as (axis, palette):
        for (name, values), color in zip(prepared.items(), palette):
            weights = np.full(values.size, 1 / values.size) if stat == 'probability' else None
            axis.hist(values, bins=edges, weights=weights, density=stat == 'density',
                    histtype='step', linewidth=1.5, color=color, label=name)
        axis.set_ylabel({'count': 'Count', 'probability': 'Proportion', 'density': 'Density'}[stat])
    return axis


def plot_ecdf_comparison(
        distributions: Mapping[str, DistributionValues],
        ax: Any = None,
        *,
        colors: Mapping[str, Any] | None = None,
        value_label: str = 'Value',
        title: str | None = None,
        note: str | None = None,
        reference_value: float | None = None,
        reference_label: str = 'Reference',
) -> Any:
    """Compare empirical cumulative distributions using P(value <= threshold).

    Parameters
    ----------
    distributions : Mapping[str, sequence | np.ndarray | pd.Series | pl.Series]
        Named one-dimensional numeric samples in display order. Numeric strings
        are accepted; missing and infinite values are excluded per sample.
        Every sample must retain at least one finite observation.
    ax : matplotlib.axes.Axes | None, default None
        Existing subplot to use; None creates a new figure.
    colors : Mapping[str, color] | None, default None
        Colors by sample name; omitted names use cycle colors.
    value_label : str, default 'Value'
        Horizontal-axis label. Transform samples before calling to compare
        absolute values instead of signed values.
    title, note : str | None, default None
        Optional title and small lower-left annotation.
    reference_value : float | None, default None
        Optional finite threshold marked by a dashed vertical line.
    reference_label : str, default 'Reference'
        Non-blank legend label for that line.

    Returns
    -------
    matplotlib.axes.Axes
        Axis with right-continuous empirical step curves on a zero-to-one scale.

    Raises
    ------
    ImportError
        If Matplotlib is unavailable.
    TypeError
        If sample containers, axis, or presentation options have invalid types.
    ValueError
        If data, sample names, colors, or reference are invalid.

    Notes
    -----
    Ties contribute their full frequency at each threshold. Samples are sorted
    independently without mutating inputs. Curves describe samples, not tests.
    """
    prepared = _prepare_distributions(distributions)
    with _comparison_plot(list(prepared), ax, colors=colors, value_label=value_label,
                          title=title, note=note, reference_value=reference_value,
                          reference_label=reference_label) as (axis, palette):
        for (name, values), color in zip(prepared.items(), palette):
            thresholds, counts = np.unique(values, return_counts=True)
            cumulative = np.cumsum(counts) / values.size
            axis.step(np.r_[thresholds[0], thresholds], np.r_[0., cumulative],
                    where='post', label=name, color=color)
        axis.set_ylabel('Cumulative proportion (value <= threshold)')
        axis.set_ylim(0., 1.)
    return axis


def plot_exceedance_comparison(
        distributions: Mapping[str, DistributionValues],
        ax: Any = None,
        *,
        colors: Mapping[str, Any] | None = None,
        value_label: str = 'Value',
        title: str | None = None,
        note: str | None = None,
        reference_value: float | None = None,
        reference_label: str = 'Reference',
) -> Any:
    """Compare inclusive empirical exceedance curves, P(value >= threshold).

    Parameters
    ----------
    distributions : Mapping[str, sequence | np.ndarray | pd.Series | pl.Series]
        Named one-dimensional numeric samples in display order. Numeric strings
        are accepted; missing and infinite values are excluded per sample.
        Every sample must retain at least one finite observation.
    ax : matplotlib.axes.Axes | None, default None
        Existing subplot to use; None creates a new figure.
    colors : Mapping[str, color] | None, default None
        Colors by sample name; omitted names use cycle colors.
    value_label : str, default 'Value'
        Horizontal-axis label. Pass absolute-valued samples explicitly when
        the comparison concerns magnitude rather than direction.
    title, note : str | None, default None
        Optional title and small lower-left annotation.
    reference_value : float | None, default None
        Optional finite threshold marked by a dashed vertical line.
    reference_label : str, default 'Reference'
        Non-blank legend label for that line.

    Returns
    -------
    matplotlib.axes.Axes
        Axis with left-continuous inclusive exceedance curves on a zero-to-one
        scale. The repeated final x coordinate shows the right-hand limit zero.

    Raises
    ------
    ImportError
        If Matplotlib is unavailable.
    TypeError
        If sample containers, axis, or presentation options have invalid types.
    ValueError
        If data, sample names, colors, or reference are invalid.

    Notes
    -----
    Equality counts as exceedance, including ties at the maximum. This is
    1 - F(threshold-), not 1 - F(threshold). These are empirical frequencies,
    not automatically calibrated p-values or confidence intervals.
    """
    prepared = _prepare_distributions(distributions)
    with _comparison_plot(list(prepared), ax, colors=colors, value_label=value_label,
                          title=title, note=note, reference_value=reference_value,
                          reference_label=reference_label) as (axis, palette):
        for (name, values), color in zip(prepared.items(), palette):
            thresholds, counts = np.unique(values, return_counts=True)
            exceedance = (values.size - np.r_[0, np.cumsum(counts)[:-1]]) / values.size
            axis.step(np.r_[thresholds, thresholds[-1]], np.r_[exceedance, 0.],
                    where='pre', label=name, color=color)
        axis.set_ylabel('Exceedance proportion (value >= threshold)')
        axis.set_ylim(0., 1.)
    return axis


def plot_boxplot_comparison(
        distributions: Mapping[str, DistributionValues],
        ax: Any = None,
        whis: float = 1.5,
        show_fliers: bool = True,
        *,
        colors: Mapping[str, Any] | None = None,
        value_label: str = 'Value',
        title: str | None = None,
        note: str | None = None,
        reference_value: float | None = None,
        reference_label: str = 'Reference',
) -> Any:
    """Compare named samples with side-by-side vertical box plots.

    Parameters
    ----------
    distributions : Mapping[str, sequence | np.ndarray | pd.Series | pl.Series]
        Named one-dimensional numeric samples in display order. Numeric strings
        are accepted; missing and infinite values are excluded per sample.
        Every sample must retain at least one finite observation.
    ax : matplotlib.axes.Axes | None, default None
        Existing subplot to use; None creates a new figure.
    whis : float, default 1.5
        Finite, strictly positive IQR multiplier for Tukey whiskers. Whiskers
        end at observed values within those fences; they are not intervals.
    show_fliers : bool, default True
        Whether to display observations beyond the whiskers.
    colors : Mapping[str, color] | None, default None
        Box colors by sample name; omitted names use cycle colors.
    value_label : str, default 'Value'
        Vertical-axis label.
    title, note : str | None, default None
        Optional title and small lower-left annotation. A note can distinguish
        bootstrap uncertainty from variation across generated samples.
    reference_value : float | None, default None
        Optional finite value marked by a dashed horizontal line.
    reference_label : str, default 'Reference'
        Non-blank legend label for that line.

    Returns
    -------
    matplotlib.axes.Axes
        Axis with one box per sample in mapping order. Boxes show quartiles,
        with a median line and no notches or bootstrap confidence intervals.

    Raises
    ------
    ImportError
        If Matplotlib is unavailable.
    TypeError
        If samples, axis, flags, or presentation options have invalid types.
    ValueError
        If data, names, whisker multiplier, colors, or reference are invalid.

    Notes
    -----
    Quartiles use Matplotlib's linear-percentile convention. Singleton and
    constant samples collapse to a line. Visual box overlap is not a test.
    """
    prepared = _prepare_distributions(distributions)
    whis = _validate_finite_number(whis, 'whis')
    if whis <= 0:
        raise ValueError('whis must be strictly positive.')
    if type(show_fliers) is not bool:
        raise TypeError('show_fliers must be boolean.')
    with _comparison_plot(list(prepared), ax, colors=colors, value_label=value_label,
                          title=title, note=note, reference_value=reference_value,
                          reference_label=reference_label, value_on_y=True) as (axis, palette):
        artists = axis.boxplot(list(prepared.values()), whis=whis, showfliers=show_fliers, patch_artist=True)
        for box, color in zip(artists['boxes'], palette):
            box.set_facecolor(color)
            box.set_alpha(.4)
        axis.set_xticks(range(1, len(prepared) + 1), list(prepared))
        axis.set_xlabel('Distribution')
    return axis
