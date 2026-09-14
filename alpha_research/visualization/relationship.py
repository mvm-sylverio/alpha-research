from numbers import Real
from typing import Any, Literal

import numpy as np
import pandas as pd

from alpha_research.evaluation.relationship import FeatureTargetRelationshipResult

__all__ = [
    'plot_feature_target_bins',
    'plot_feature_target_scatter',
]


def plot_feature_target_scatter(
        result: FeatureTargetRelationshipResult,
        ax: Any = None,
        max_points: int | None = None,
        random_state: int | None = 0,
        color: str = 'C0',
        alpha: float = 0.30,
        title: str | None = None,
) -> Any:
    """Plot every cleaned X-Y pair without bin aggregation.

    The horizontal coordinate is the feature X and the vertical coordinate is
    target Y. This is the noisy, observation-level view used to inspect
    outliers, asymmetry, heteroscedasticity, and visible nonlinear structure.

    Parameters
    ----------
    result : FeatureTargetRelationshipResult
        Output produced by feature_target_relationship.
    ax : matplotlib.axes.Axes | None, default None
        Axis to draw into. A new axis is created when omitted.
    max_points : int | None, default None
        Optional positive cap for deterministic random scatter sampling.
    random_state : int | None, default 0
        Sampling seed used only when max_points is below the pair count.
    color : str, default 'C0'
        Matplotlib point color.
    alpha : float, default 0.30
        Point opacity constrained to [0, 1].
    title : str | None, default None
        Optional title. A descriptive default is used when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the scatter plot.

    Raises
    ------
    ImportError
        If Matplotlib is not installed. Install the optional ``viz`` extra.
    TypeError
        If result or a plotting parameter has an unsupported type.
    ValueError
        If a numeric plotting parameter is outside its accepted range.

    Notes
    -----
    Sampling affects visualization only and never changes the relationship
    summaries stored in result. This function performs no new statistical
    calculation; it only visualizes ``result.pairs``.
    """
    if not isinstance(result, FeatureTargetRelationshipResult):
        raise TypeError('result must be a FeatureTargetRelationshipResult.')
    if max_points is not None and (
            not isinstance(max_points, (int, np.integer))
            or isinstance(max_points, bool)
            or max_points <= 0
    ):
        raise ValueError('max_points must be a positive integer or None.')
    if random_state is not None and (
            not isinstance(random_state, (int, np.integer))
            or isinstance(random_state, bool)
    ):
        raise TypeError('random_state must be an integer or None.')
    if not isinstance(alpha, Real) or isinstance(alpha, bool):
        raise TypeError('alpha must be numeric.')
    if not np.isfinite(float(alpha)) or not 0 <= float(alpha) <= 1:
        raise ValueError('alpha must be finite and between zero and one.')

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            'Matplotlib is required for plotting. Install alpha-research[viz].',
        ) from error

    pairs = (
        result.pairs.copy()
        if isinstance(result.pairs, pd.DataFrame)
        else result.pairs.to_pandas()
    )
    if max_points is not None and len(pairs) > max_points:
        pairs = pairs.sample(n=max_points, random_state=random_state)
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(
        pairs[result.feature].to_numpy(dtype=float),
        pairs[result.target].to_numpy(dtype=float),
        color=color,
        alpha=float(alpha),
    )
    ax.axhline(0, color='0.6', linewidth=0.8, linestyle='--')
    ax.set_xlabel(result.feature)
    ax.set_ylabel(result.target)
    ax.set_title(
        title if title is not None else f'{result.feature} vs {result.target}',
    )
    return ax


def plot_feature_target_bins(
        result: FeatureTargetRelationshipResult,
        target_statistic: Literal['mean', 'median'] = 'mean',
        feature_statistic: Literal['mean', 'median'] = 'median',
        ax: Any = None,
        color: str = 'C1',
        title: str | None = None,
) -> Any:
    """Plot mean or median Y against representative values of binned X.

    Bins come from feature X only and were already calculated by
    feature_target_relationship. For every X bin, this plot places a point at
    the bin's mean or median X and its corresponding mean or median Y. It is a
    less noisy empirical response curve, not a plot of Y quantiles.

    Parameters
    ----------
    result : FeatureTargetRelationshipResult
        Output produced by feature_target_relationship.
    target_statistic : {'mean', 'median'}, default 'mean'
        Target summary displayed on the vertical axis.
    feature_statistic : {'mean', 'median'}, default 'median'
        Representative feature value displayed on the horizontal axis.
    ax : matplotlib.axes.Axes | None, default None
        Axis to draw into. A new axis is created when omitted.
    color : str, default 'C1'
        Matplotlib line and point color.
    title : str | None, default None
        Optional title. A descriptive default is used when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the binned relationship plot.

    Raises
    ------
    ImportError
        If Matplotlib is not installed. Install the optional ``viz`` extra.
    TypeError
        If result has an unsupported type.
    ValueError
        If a statistic name is invalid.

    Notes
    -----
    With 10 quantile bins, the first point represents the observations with
    the lowest X ranks and the tenth represents the highest X ranks. Reading
    the points from left to right shows how the typical target changes as the
    feature increases.
    """
    if not isinstance(result, FeatureTargetRelationshipResult):
        raise TypeError('result must be a FeatureTargetRelationshipResult.')
    if target_statistic not in {'mean', 'median'}:
        raise ValueError("target_statistic must be 'mean' or 'median'.")
    if feature_statistic not in {'mean', 'median'}:
        raise ValueError("feature_statistic must be 'mean' or 'median'.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            'Matplotlib is required for plotting. Install alpha-research[viz].',
        ) from error

    summary = (
        result.bin_summary.copy()
        if isinstance(result.bin_summary, pd.DataFrame)
        else result.bin_summary.to_pandas()
    )
    x_col = f'feature_{feature_statistic}'
    y_col = f'target_{target_statistic}'
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(
        summary[x_col].to_numpy(dtype=float),
        summary[y_col].to_numpy(dtype=float),
        color=color,
        marker='o',
    )
    ax.axhline(0, color='0.6', linewidth=0.8, linestyle='--')
    ax.set_xlabel(f'{result.feature} bin {feature_statistic}')
    ax.set_ylabel(f'{result.target} bin {target_statistic}')
    ax.set_title(
        title
        if title is not None
        else f'{result.target} by {result.feature} {result.binning} bin',
    )
    return ax
