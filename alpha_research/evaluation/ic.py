from dataclasses import dataclass

import polars as pl
import pandas as pd
import numpy as np
from typing import Literal, Callable

from scipy.stats import pearsonr, spearmanr

from alpha_research.evaluation.statistical_tests import newey_west_tstat, fdr_correction
from alpha_research._utils import (
    _is_constant_series,
    _validate_df,
    _validate_same_backend,
)
from alpha_research.features.schema import get_feature_name, join_feature_target_frames

__all__ = ['information_coefficient', 'compute_ic', 'ICMetrics', 'compute_ic_metrics', 'ic_summary_table', 'ic_decay',
           'ic_decay_summary', 'ic_decay_summary_table']

DataFrame = pd.DataFrame | pl.DataFrame
TargetFn = Callable[[DataFrame, int], DataFrame]


def information_coefficient(
        x: pd.Series | pl.Series,
        y: pd.Series | pl.Series,
        corr_method: Literal['pearson', 'spearman'] = 'spearman',
) -> float:
    """
    Compute the Information Coefficient (IC) between two vectors.

    IC measures the rank correlation between a signal (x) and
    a target (y) within a single cross-section.

    Parameters
    ----------
    x : pd.Series | pl.Series
        Signal vector (e.g. factor values across assets).
    y : pd.Series | pl.Series
        Target vector (e.g. forward returns across assets).
    corr_method : {'spearman', 'pearson'}
        Spearman captures monotonic relationships (default).
        Pearson assumes linearity.

    Returns
    -------
    float
        Correlation coefficient in [-1, 1].

    Raises
    ------
    ValueError
        If len(x) != len(y) or corr_method is invalid.

    Notes
    -----
    Returns nan if either input is constant (undefined correlation).
    Constant inputs are detected before calling scipy.
    """

    if len(x) != len(y):
        raise ValueError(f'len(x): {len(x)} != len(y): {len(y)}.')

    if _is_constant_series(x) or _is_constant_series(y):
        return np.nan

    if corr_method == 'pearson':
        return pearsonr(x, y)[0]  # [0] is the correlation coefficient
    elif corr_method == 'spearman':
        return spearmanr(x, y)[0]  # [0] is the correlation coefficient
    else:
        raise ValueError(f"corr_method must be 'spearman' or 'pearson'.")


def _compute_ic_pandas(
        df: pd.DataFrame,
        feature: str,
        target: str,
        corr_method: Literal['pearson', 'spearman'] = 'spearman',
        date_column: str = 'time',
        ic_column='ic',
) -> pd.DataFrame:
    """
    Core pandas implementation of compute_ic.
    See compute_ic() for full documentation.
    """
    sub = df[[date_column, feature, target]].dropna()

    ic_by_date = (sub.groupby(date_column)[[feature, target]]
                  .apply(lambda v: information_coefficient(v[feature], v[target], corr_method))
                  .rename(ic_column)
                  .reset_index()
                  .dropna()
                  .sort_values(by=[date_column])
    )

    return ic_by_date


def _compute_ic_polars(
        df: pl.DataFrame,
        feature: str,
        target: str,
        corr_method: Literal['pearson', 'spearman'] = 'spearman',
        date_column: str = 'time',
        ic_column='ic',
) -> pl.DataFrame:
    """
    Core polars implementation of compute_ic.
    See compute_ic() for full documentation.
    """
    df_ic = df[[date_column, feature, target]].drop_nulls()

    # group by time
    df_ic = (
        df_ic
        .group_by(date_column)
        .agg(
            pl.corr(feature, target, method=corr_method)
            .alias(ic_column)
        )
        .drop_nulls()
        .sort(date_column)
    )

    return df_ic


def compute_ic(
        df: pd.DataFrame | pl.DataFrame,
        feature: str,
        target: str,
        corr_method: Literal['pearson', 'spearman'] = 'spearman',
        date_column: str = 'time',
        ic_column: str = 'ic'
) -> pd.DataFrame | pl.DataFrame:
    """
    Compute the Information Coefficient (IC) between two columns of a pandas or polars DataFrame
    cross-sectionally for each date in the DataFrame.

    IC measures the correlation between a signal feature and a target.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        df where the ic computation will be applied, which contains the 'feature', 'target', 'date_column' columns.
    feature : str
        Column name of the Signal vector Series in the df (e.g. 'factor_A').
    target : str
        Column name of the Target vector Series in the df (e.g. 'fwd_ret_5').
    corr_method : {'spearman', 'pearson'}
        Spearman captures monotonic relationships (default).
        Pearson assumes linearity.
    date_column : str
        String name of the date column in df (default = 'time').
    ic_column : str
        String name of the new column in df with the computed ic values.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Clean DataFrame with columns [date_column, ic_column] with values sorted by date_column.

    Raises
    ------
    KeyError
        If date_column, feature, target are not columns of the df.
    TypeError
        If df is not pandas or polars type.
    """
    _validate_df(df, [date_column, feature, target])

    if isinstance(df, pd.DataFrame):
        return _compute_ic_pandas(df, feature, target, corr_method, date_column, ic_column)
    else: # pl.Dataframe type
        return _compute_ic_polars(df, feature, target, corr_method, date_column, ic_column)


@dataclass(frozen=True, slots=True)  # unchangable object results and attributes
class ICMetrics:
    """
    Immutable container for Information Coefficient general metrics.

    All scalar metrics are read-only after creation.

    Attributes
    ----------
    mean : float
        Mean IC over all dates (preserves sign).
    abs_mean : float
        Absolute mean IC. Magnitude of the signal regardless of direction.
    sign : int
        Direction of the signal: +1 if mean IC >= 0, -1 otherwise. Aligns signal direction of the feature.
    std : float
        Standard deviation of IC (ddof=1).
    stability : float
        IC Information Ratio: abs_mean / std.
        Higher values indicate more consistent signal. nan if std == 0.
    pct_positive : float
        Fraction of dates where the adjusted IC > 0, i.e. signal was
        on the correct side. Computed on adjusted_series.
    quantiles : dict
        Quartiles of the adjusted IC series: q25, q50, q75.
    original_series : pd.Series | pl.Series
        Raw IC time series as returned by compute_ic().
    adjusted_series : pd.Series | pl.Series
        IC series multiplied by sign — always oriented positively. Useful for plotting and further statistical analysis.
    """
    mean: float
    abs_mean: float
    sign: int
    std: float
    stability: float
    pct_positive: float
    quantiles: dict
    original_series: pd.Series | pl.Series
    adjusted_series: pd.Series | pl.Series


def compute_ic_metrics(ic_series: pd.Series | pl.Series) -> ICMetrics:
    """
    Compute the general metrics for a time series of IC values.

    Handles inverse signals automatically, flipping metrics to represent the magnitude of the signal.

    Parameters
    ----------
    ic_series : pd.Series | pl.Series
        Time series by date of IC values, as returned by compute_ic().

    Returns
    -------
    ICMetrics
        Dataclass with scalar metrics and both original and
        sign-adjusted IC series.

    Notes
    -----
    pct_positive and quantiles are computed on the sign adjusted series.
    """

    ic_arr = ic_series.to_numpy()  # to_numpy unifies treatment of the code below

    # guard for empty or all-nan series
    if len(ic_arr) == 0 or np.all(np.isnan(ic_arr)):
        return ICMetrics(
            mean=np.nan,
            abs_mean=np.nan,
            sign=1,
            std=np.nan,
            stability=np.nan,
            pct_positive=np.nan,
            quantiles={'q25': np.nan, 'q50': np.nan, 'q75': np.nan},
            original_series=ic_series,
            adjusted_series=ic_series,
        )

    ic_mean = np.mean(ic_arr)
    ic_std = np.std(ic_arr, ddof=1)  # ddof=1 -> sample

    # corrections for floating-point precision that will not return 0 when should
    ic_mean_is_zero = np.isclose(ic_mean, 0, 1e-8)
    ic_std_is_zero = np.isclose(ic_std, 0, atol=1e-8)

    # sign adjustments
    ic_sign = np.sign(ic_mean) if not ic_mean_is_zero else 1
    adjusted_arr = ic_sign * ic_arr

    return ICMetrics(
        mean=float(ic_mean),
        abs_mean=float(np.mean(np.abs(ic_arr))),
        sign=int(ic_sign),
        std=float(ic_std),
        stability=float(abs(ic_mean) / ic_std) if not ic_std_is_zero else np.nan,
        pct_positive=float(np.mean(adjusted_arr > 0)),
        quantiles={
            'q25': np.quantile(adjusted_arr, 0.25),
            'q50': np.quantile(adjusted_arr, 0.5),
            'q75': np.quantile(adjusted_arr, 0.75)
        },
        original_series=ic_series,
        adjusted_series=ic_series * ic_sign
    )


@dataclass(frozen=True, slots=True)
class ICSummaryResult:
    """
    Result of ic_summary_table computation.

    Attributes
    ----------
    table : pd.DataFrame | pl.DataFrame
        Summary table with IC metrics and t-stats per feature.
    ic_frames : dict[str, pd.DataFrame | pl.DataFrame]
        Raw IC time series per feature — {feature_name: df[[time, ic]]}.
        As returned by compute_ic(). Use for rolling analysis and plots.
    """
    table: pd.DataFrame | pl.DataFrame
    ic_frames: dict[str, pd.DataFrame | pl.DataFrame]


def ic_summary_table(
        df: pd.DataFrame | pl.DataFrame,
        feature_list: list[str],
        target: str,
        corr_method: Literal['pearson', 'spearman'] = 'spearman',
        date_column: str = 'time',
        feature_groups: dict[str, str] | None = None,

) -> ICSummaryResult:
    """
    Compute the information coefficient (IC) between every feature in feature_list and the target.
    Compute the Newey-West t-statistic and metrics for the IC of every feature in feature_list.
    Builds a table with the summarized results compilated and the classification of each feature
    as defined by the user.

    Adopts the Newey West t-statistics because the ic time series usually violate the i.i.d assumption.
    For more explanations, see newey_west_tstat docstring.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        df where the ic computation will be applied, which contains all the columns in 'feature_list',
        the 'target' and 'date_column' columns.
    feature_list : list[str]
        Column names of the Signals vector Series in the df (e.g. ['factor_A', 'factor_B', ...]).
    target : str
        Column name of the Target vector Series in the df (e.g. 'fwd_ret_5').
    corr_method : {'spearman', 'pearson'}
        Spearman captures monotonic relationships (default).
        Pearson assumes linearity.
    date_column : str
        String name of the date column in df (default = 'time').
    feature_groups : dict[str, str] | None
        Dict defined by the user with the group name of each feature.
        Useful for the application of the Benjamini-Hochberg test for the correction of
        multiple features and control of false positives.
        All features are labeled 'ungrouped' if feature_groups is not provided.
        A feature is also labeled 'ungrouped' if its group is not found on feature_groups.

    Returns
    -------
    ICSummaryResult
        IC summary table and dict containing the DataFrames of the ic time-series per feature.
    """
    # Guard against empty list
    if not feature_list:
        raise ValueError("feature_list must not be empty.")

    rows = []
    ic_dfs_dict = {}

    for feature in feature_list:
        # the existence of the column on the df is already checked on compute_ic
        df_ic = compute_ic(df, feature, target, corr_method, date_column)

        # Save df_ic to dict for return clause
        ic_dfs_dict[feature] = df_ic

        ic_series = df_ic['ic']  # internal usage adopts default ic_column name

        # ic metrics
        metrics = compute_ic_metrics(ic_series)  # default option for ic column name

        # t stat of the ic
        nw_test = newey_west_tstat(ic_series)

        # parsing feature_group of the feature
        if feature_groups is not None:
            feature_group = feature_groups.get(feature, 'ungrouped')
        else:
            feature_group = 'ungrouped'

        rows.append({
            'feature': feature,
            'mean': metrics.mean,
            'abs_mean': metrics.abs_mean,
            'sign': metrics.sign,
            'std': metrics.std,
            'stability': metrics.stability,
            'pct_positive': metrics.pct_positive,
            'quantile25': metrics.quantiles['q25'],
            'quantile50': metrics.quantiles['q50'],
            'quantile75': metrics.quantiles['q75'],
            't_stat': nw_test.t_stat,
            'p_value':nw_test.p_value,
            'feature_group': feature_group,
            'n_obs': len(ic_series),
        })

    if isinstance(df, pd.DataFrame):
        return ICSummaryResult(pd.DataFrame(rows), ic_dfs_dict)
    else:  # type already checked when calling compute_ic, else means pl.DataFrame
        return ICSummaryResult(pl.DataFrame(rows), ic_dfs_dict)


@dataclass(frozen=True, slots=True)
class ICDecayResult:
    """
    Result of ic_decay computation.

    Attributes
    ----------
    table : pd.DataFrame | pl.DataFrame
        IC metrics, Newey-West statistics and FDR results for each horizon.
    ic_frames : dict[int, pd.DataFrame | pl.DataFrame]
        Raw IC time series for each horizon:
        {horizon: df[[time, ic]]}.
    """
    table: pd.DataFrame | pl.DataFrame
    ic_frames: dict[int, pd.DataFrame | pl.DataFrame]

def ic_decay(
        df_feature: pd.DataFrame | pl.DataFrame,
        feature: str,
        target_data: pd.DataFrame | pl.DataFrame,
        horizons: list[int],
        target_fn: TargetFn,
        corr_method: Literal['pearson', 'spearman'] = 'spearman',
        date_column: str = 'time',
        symbol_column: str = 'symbol',
        feature_groups: dict[str, str] | None = None,
        fdr: float = 0.05,
        fdr_method: Literal['bh', 'by'] = 'bh',
        progress_callback: Callable[[], None] | None = None,
) -> ICDecayResult:
    """
    Compute the Information Coefficient decay curve of one feature.

    df_feature is a wide feature DataFrame. The selected feature is evaluated against
    targets generated from target_data at multiple forward horizons. IC
    metrics, Newey-West statistics and FDR-corrected significance are computed
    independently for every horizon.

    Multiple-testing correction is applied jointly across all horizons of the
    feature, treating them as one family of hypotheses.

    Parameters
    ----------
    df_feature : pd.DataFrame or pl.DataFrame
        Wide feature DataFrame containing date_column, symbol_column, and
        feature.
    feature : str
        Name of the feature column in df_feature to evaluate.
    target_data : pd.DataFrame or pl.DataFrame
        DataFrame passed to target_fn to generate each target. It must use the
        same DataFrame backend as df.
    horizons : list[int]
        Forward horizons to evaluate. Must contain unique positive integers.
    target_fn : Callable[[pd.DataFrame | pl.DataFrame, int], pd.DataFrame | pl.DataFrame]
        Function used to generate targets. It must accept:

            target_fn(target_data, horizon=horizon)

        Fixed target configuration may be supplied by a wrapper function or
        functools.partial when necessary.
    corr_method : {'spearman', 'pearson'}
        Correlation method used by compute_ic.
    date_column : str
        Date/time column.
    symbol_column : str
        Asset identifier column.
    feature_groups : dict[str, str] | None
        Optional mapping between feature name and its semantic feature group.
        Used as metadata. FDR correction still occurs only across horizons
        of this feature.
    fdr : float
        False discovery rate.
    fdr_method : {'bh', 'by'}
        Multiple-testing correction method.
    progress_callback : Callable[[], None] | None, default None
        Optional callback invoked after each completed horizon estimate.

    Returns
    -------
    ICDecayResult
        table:
            One row per horizon containing IC metrics, Newey-West statistics
            and FDR results.

        ic_frames:
            Raw IC time series indexed by horizon.

    Raises
    ------
    ValueError
        If df_feature or target_data is invalid, feature is missing, or horizons is
        empty, contains duplicates, or contains non-positive values.
    TypeError
        If df_feature, target_data, or a target returned by target_fn use different
        DataFrame backends.
    """
    _validate_df(df_feature, [date_column, symbol_column, feature])

    target_frames = _generate_target_frames(
        df_feature=df_feature,
        target_data=target_data,
        horizons=horizons,
        target_fn=target_fn,
    )
    return _ic_decay_from_target_frames(
        df_feature=df_feature,
        feature=feature,
        target_frames=target_frames,
        corr_method=corr_method,
        date_column=date_column,
        symbol_column=symbol_column,
        feature_groups=feature_groups,
        fdr=fdr,
        fdr_method=fdr_method,
        progress_callback=progress_callback,
    )


def _generate_target_frames(
        df_feature: pd.DataFrame | pl.DataFrame,
        target_data: pd.DataFrame | pl.DataFrame,
        horizons: list[int],
        target_fn: TargetFn,
) -> dict[int, pd.DataFrame | pl.DataFrame]:
    """
    Generate one target DataFrame for each requested horizon.

    Parameters
    ----------
    df_feature : pd.DataFrame | pl.DataFrame
        Feature DataFrame used to validate the backend of target_data and the
        generated target DataFrames.
    target_data : pd.DataFrame | pl.DataFrame
        DataFrame passed to target_fn to generate each target.
    horizons : list[int]
        Unique positive forward horizons to generate.
    target_fn : Callable[[pd.DataFrame | pl.DataFrame, int], pd.DataFrame | pl.DataFrame]
        Function called as target_fn(target_data, horizon=horizon).

    Returns
    -------
    dict[int, pd.DataFrame | pl.DataFrame]
        Target DataFrames indexed by their sorted horizon values.

    Raises
    ------
    ValueError
        If target_data is invalid, horizons is empty, contains duplicates, or
        contains non-positive values.
    TypeError
        If df, target_data, or a target returned by target_fn use different
        DataFrame backends.

    Notes
    -----
    This helper computes every target once. The resulting mapping can be
    reused across multiple features without calling target_fn again.
    """
    _validate_df(target_data, [])
    _validate_same_backend(df_feature, target_data)

    if not horizons:
        raise ValueError("horizons must not be empty.")

    if any(not isinstance(h, int) or isinstance(h, bool) or h <= 0 for h in horizons):
        raise ValueError("horizons must contain positive integers.")

    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must not contain duplicates.")

    target_frames = {}

    for horizon in sorted(horizons):
        target_df = target_fn(target_data, horizon=horizon)

        _validate_same_backend(df_feature, target_df)

        target_frames[horizon] = target_df

    return target_frames


def _ic_decay_from_target_frames(
        df_feature: pd.DataFrame | pl.DataFrame,
        feature: str,
        target_frames: dict[int, pd.DataFrame | pl.DataFrame],
        corr_method: Literal['pearson', 'spearman'],
        date_column: str,
        symbol_column: str,
        feature_groups: dict[str, str] | None,
        fdr: float,
        fdr_method: Literal['bh', 'by'],
        progress_callback: Callable[[], None] | None = None,
) -> ICDecayResult:
    """
    Compute one feature's IC decay from pre-generated target DataFrames.

    Parameters
    ----------
    df_feature : pd.DataFrame | pl.DataFrame
        Wide feature DataFrame containing date_column, symbol_column, and
        feature.
    feature : str
        Name of the feature column in df_feature to evaluate.
    target_frames : dict[int, pd.DataFrame | pl.DataFrame]
        Target DataFrames indexed by horizon. Each target DataFrame must
        contain date_column, symbol_column, and exactly one target column.
    corr_method : {'spearman', 'pearson'}
        Correlation method used to calculate IC.
    date_column : str
        Date/time column.
    symbol_column : str
        Asset identifier column.
    feature_groups : dict[str, str] | None
        Optional mapping between feature names and semantic feature groups.
    fdr : float
        False discovery rate.
    fdr_method : {'bh', 'by'}
        Multiple-testing correction method.
    progress_callback : Callable[[], None] | None, default None
        Optional callback invoked after each completed horizon estimate.

    Returns
    -------
    ICDecayResult
        IC metrics, FDR results, and raw IC time series for every horizon.

    Raises
    ------
    ValueError
        If the feature or target DataFrames fail the join validation.
    TypeError
        If feature and target DataFrames use different backends.

    Notes
    -----
    This helper does not call target_fn. It is shared by ic_decay() and
    ic_decay_summary_table() so that batch evaluation can reuse targets
    generated once per horizon.
    """
    feature_group = (
        feature_groups.get(feature, 'ungrouped')
        if feature_groups is not None
        else 'ungrouped'
    )

    rows = []
    ic_frames = {}

    for horizon, target_df in target_frames.items():
        target_col = get_feature_name(target_df, date_column, symbol_column)

        joined_df = join_feature_target_frames(
            feature_df=df_feature,
            target_df=target_df,
            feature_col=feature,
            target_col=target_col,
            time_col=date_column,
            symbol_col=symbol_column,
        )

        df_ic = compute_ic(
            joined_df,
            feature=feature,
            target=target_col,
            corr_method=corr_method,
            date_column=date_column,
        )

        ic_frames[horizon] = df_ic

        ic_series = df_ic['ic']

        metrics = compute_ic_metrics(ic_series)
        nw_test = newey_west_tstat(ic_series)

        rows.append({
            'feature': feature,
            'target': target_col,
            'horizon': horizon,

            'mean': metrics.mean,
            'abs_mean': metrics.abs_mean,
            'sign': metrics.sign,
            'std': metrics.std,
            'stability': metrics.stability,
            'pct_positive': metrics.pct_positive,

            'quantile25': metrics.quantiles['q25'],
            'quantile50': metrics.quantiles['q50'],
            'quantile75': metrics.quantiles['q75'],

            't_stat': nw_test.t_stat,
            'p_value': nw_test.p_value,

            'feature_group': feature_group,

            'n_obs': len(ic_series),
        })
        if progress_callback is not None:
            progress_callback()

    if isinstance(df_feature, pd.DataFrame):
        result_table = pd.DataFrame(rows)
    else:
        result_table = pl.DataFrame(rows)

    # One correction across all horizons of this feature.
    result_table = fdr_correction(
        result_table,
        fdr=fdr,
        method=fdr_method,
    )

    if isinstance(result_table, pd.DataFrame):
        result_table = (
            result_table
            .sort_values('horizon')
            .reset_index(drop=True)
        )
    else:
        result_table = result_table.sort('horizon')

    return ICDecayResult(
        table=result_table,
        ic_frames=ic_frames,
    )


@dataclass(frozen=True, slots=True)
class DecaySummaryResult:
    """
    Scalar summary of an IC decay curve.

    Attributes
    ----------
    peak_horizon : int
        Horizon with the highest absolute mean IC.
    peak_abs_ic : float
        Absolute mean IC at peak_horizon.
    halflife_horizon : int | None
        First tested horizon at or after peak_horizon where absolute
        mean IC falls to 50% or less of peak_abs_ic.
    last_significant_horizon : int | None
        Largest horizon passing the FDR correction.
    auc : float
        Area under the absolute mean IC curve versus horizon.
    """
    peak_horizon: int
    peak_abs_ic: float
    halflife_horizon: int | None
    last_significant_horizon: int | None
    auc: float

def ic_decay_summary(
        decay_curve: pd.DataFrame | pl.DataFrame,
) -> DecaySummaryResult:
    """
    Summarize an IC decay curve into scalar diagnostics.

    The decay magnitude is defined as abs(mean IC) at each horizon,
    not mean(abs(IC)).

    Parameters
    ----------
    decay_curve : pd.DataFrame or pl.DataFrame
        Table returned by ic_decay().table.

        Required columns:
        ['horizon', 'mean', 'fdr_rejected'].

    Returns
    -------
    DecaySummaryResult
        Scalar diagnostics describing signal strength and persistence.

    Notes
    -----
    peak_horizon
        Horizon with maximum abs(mean IC).

    halflife_horizon
        First tested horizon at or after the peak where abs(mean IC)
        is <= 50% of the peak. This is a discrete diagnostic; no
        interpolation is performed.

        None if all tested horizons stay above 50% of peak IC.

    last_significant_horizon
        Largest horizon that passes FDR correction.

    auc
        Trapezoidal area under abs(mean IC) versus horizon.
        Comparisons between features are meaningful when the same horizon
        grid is used.
    """
    _validate_df(
        decay_curve,
        ['horizon', 'mean', 'fdr_rejected'],
    )

    if isinstance(decay_curve, pd.DataFrame):
        curve = decay_curve.sort_values('horizon')

        horizons = curve['horizon'].to_numpy(dtype=int)
        mean_ic = curve['mean'].to_numpy(dtype=float)
        rejected = curve['fdr_rejected'].to_numpy(dtype=bool)

    else:
        curve = decay_curve.sort('horizon')

        horizons = curve['horizon'].to_numpy()
        mean_ic = curve['mean'].to_numpy()
        rejected = curve['fdr_rejected'].to_numpy()

    if len(horizons) == 0:
        raise ValueError("decay_curve must not be empty.")

    if len(np.unique(horizons)) != len(horizons):
        raise ValueError(
            "decay_curve must contain exactly one row per horizon."
        )

    # For valid horizons and ic values
    finite_mask = np.isfinite(mean_ic)

    if not finite_mask.any():
        raise ValueError(
            "decay_curve contains no finite mean IC values."
        )

    valid_horizons = horizons[finite_mask]
    valid_abs_ic = np.abs(mean_ic[finite_mask])

    # ----------------------------------
    # Peak
    # ----------------------------------
    peak_idx = int(np.argmax(valid_abs_ic))

    peak_horizon = int(valid_horizons[peak_idx])
    peak_abs_ic = float(valid_abs_ic[peak_idx])

    # ----------------------------------
    # Half-life
    # ----------------------------------
    halflife_horizon = None

    if not np.isclose(peak_abs_ic, 0.0, atol=1e-12):
        half_peak = peak_abs_ic / 2

        post_peak_mask = valid_horizons >= peak_horizon

        post_peak_horizons = valid_horizons[post_peak_mask]
        post_peak_abs_ic = valid_abs_ic[post_peak_mask]

        half_mask = post_peak_abs_ic <= half_peak

        if half_mask.any():
            halflife_horizon = int(
                post_peak_horizons[
                    np.flatnonzero(half_mask)[0]
                ]
            )

    # ----------------------------------
    # Last significant horizon
    # ----------------------------------
    significant_horizons = horizons[rejected]

    last_significant_horizon = (
        int(np.max(significant_horizons))
        if len(significant_horizons) > 0
        else None
    )

    # ----------------------------------
    # Area under |mean IC|
    # ----------------------------------
    if len(valid_horizons) >= 2:
        auc = float(
            np.trapezoid(
                y=valid_abs_ic,
                x=valid_horizons,
            )
        )
    else:
        auc = 0.0

    return DecaySummaryResult(
        peak_horizon=peak_horizon,
        peak_abs_ic=peak_abs_ic,
        halflife_horizon=halflife_horizon,
        last_significant_horizon=last_significant_horizon,
        auc=auc,
    )


@dataclass(frozen=True, slots=True)
class ICDecaySummaryTableResult:
    """
    Result of ic_decay_summary_table computation.

    Attributes
    ----------
    table : pd.DataFrame | pl.DataFrame
        Scalar decay diagnostics for every feature.
    decay_results : dict[str, ICDecayResult]
        Complete decay result for every feature.
    """
    table: pd.DataFrame | pl.DataFrame
    decay_results: dict[str, ICDecayResult]


def ic_decay_summary_table(
        df_features: pd.DataFrame | pl.DataFrame,
        feature_list: list[str],
        target_data: pd.DataFrame | pl.DataFrame,
        horizons: list[int],
        target_fn: TargetFn,
        corr_method: Literal['pearson', 'spearman'] = 'spearman',
        date_column: str = 'time',
        symbol_column: str = 'symbol',
        feature_groups: dict[str, str] | None = None,
        fdr: float = 0.05,
        fdr_method: Literal['bh', 'by'] = 'bh',
        progress_callback: Callable[[], None] | None = None,
) -> ICDecaySummaryTableResult:
    """
    Compute IC decay diagnostics for multiple columns in a wide DataFrame.

    Each target is generated once per horizon from target_data and reused for
    every feature. For every feature, the function:
        1. compute IC at every horizon;
        2. compute Newey-West statistics;
        3. apply FDR correction across its horizons;
        4. summarize the resulting decay curve.

    Parameters
    ----------
    df_features : pd.DataFrame or pl.DataFrame
        Wide feature DataFrame containing date_column, symbol_column, and all
        feature columns in feature_list.
    feature_list : list[str]
        Names of the feature columns in df_features to evaluate. Must not be empty or
        contain duplicates.
    target_data : pd.DataFrame or pl.DataFrame
        DataFrame passed to target_fn to generate each target. It must use the
        same DataFrame backend as df.
    horizons : list[int]
        Forward horizons evaluated for every feature.
    target_fn : Callable[[pd.DataFrame | pl.DataFrame, int], pd.DataFrame | pl.DataFrame]
        Target function accepting target_fn(target_data, horizon=horizon).
    corr_method : {'spearman', 'pearson'}
        IC correlation method.
    date_column : str
        Date/time column.
    symbol_column : str
        Asset identifier column.
    feature_groups : dict[str, str] | None
        Optional mapping between features and semantic groups.
    fdr : float
        False discovery rate.
    fdr_method : {'bh', 'by'}
        Multiple-testing correction applied independently to the horizons
        of each feature.
    progress_callback : Callable[[], None] | None, default None
        Optional callback invoked after every completed feature-horizon
        estimate.

    Returns
    -------
    ICDecaySummaryTableResult
        table:
            One row per feature with decay summary diagnostics.

        decay_results:
            Complete ICDecayResult for every feature.

    Raises
    ------
    ValueError
        If feature_list is empty or contains duplicates, df_features or target_data is
        invalid, or horizons are invalid.
    TypeError
        If df, target_data, or a target returned by target_fn use different
        DataFrame backends.
    """
    if not feature_list:
        raise ValueError("feature_list must not be empty.")

    if len(set(feature_list)) != len(feature_list):
        raise ValueError("feature_list must not contain duplicates.")

    _validate_df(df_features, [date_column, symbol_column] + feature_list)

    target_frames = _generate_target_frames(
        df_feature=df_features,
        target_data=target_data,
        horizons=horizons,
        target_fn=target_fn,
    )

    rows = []
    decay_results = {}

    for feature in feature_list:
        decay_result = _ic_decay_from_target_frames(
            df_feature=df_features,
            feature=feature,
            target_frames=target_frames,
            corr_method=corr_method,
            date_column=date_column,
            symbol_column=symbol_column,
            feature_groups=feature_groups,
            fdr=fdr,
            fdr_method=fdr_method,
            progress_callback=progress_callback,
        )

        decay_table = decay_result.table

        if isinstance(decay_table, pd.DataFrame):
            feature = decay_table['feature'].iloc[0]
            feature_group = decay_table['feature_group'].iloc[0]
        else:
            feature = decay_table['feature'][0]
            feature_group = decay_table['feature_group'][0]

        summary = ic_decay_summary(decay_table)

        decay_results[feature] = decay_result

        rows.append({
            'feature': feature,

            'peak_horizon': summary.peak_horizon,
            'peak_abs_ic': summary.peak_abs_ic,
            'halflife_horizon': summary.halflife_horizon,
            'last_significant_horizon': (
                summary.last_significant_horizon
            ),
            'auc': summary.auc,

            'feature_group': feature_group,
        })

    if isinstance(df_features, pd.DataFrame):
        table = pd.DataFrame(rows)
    else:
        table = pl.DataFrame(rows)

    return ICDecaySummaryTableResult(
        table=table,
        decay_results=decay_results,
    )
