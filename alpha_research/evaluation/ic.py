from dataclasses import dataclass

import polars as pl
import pandas as pd
import numpy as np
from typing import Literal

from scipy.stats import pearsonr, spearmanr


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
    """

    if len(x) != len(y):
        raise ValueError(f'len(x): {len(x)} != len(y): {len(y)}.')

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
    Compute the Information Coefficient (IC) between two columns of a pandas DataFrame.

    Uses the information_coefficient function.

    Parameters
    ----------
    df : pd.DataFrame
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
    pd.DataFrame
        Clean DataFrame with columns [date_column, ic_column], sorted by date_column.
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
    Compute the Information Coefficient (IC) between two columns of a polars DataFrame.

    Uses the internal polars corr function.

    Parameters
    ----------
    df : pl.DataFrame
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
    pl.DataFrame
        Clean DataFrame with columns [date_column, ic_column], sorted by date_column.
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
    Compute the Information Coefficient (IC) between two columns of a pandas or polars DataFrame.

    Dispatches to _compute_ic_pandas or _compute_ic_polars based on df type.

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
        Clean df[[date_column, feature, target, ic_column]] with values sorted by date_column.

    Raises
    ------
    KeyError
        If date_column, feature, target are not columns of the df.
    ValueError
        If df is not pandas or polars type.
    """
    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        raise ValueError('df must be Pandas or Polars type.')

    # Initial checks - already checked for DataFrame format before
    if date_column not in df.columns:
        raise KeyError(f'{date_column} is not a column of df.')
    if feature not in df.columns:
        raise KeyError(f'{feature} is not a column of df.')
    if target not in df.columns:
        raise KeyError(f'{target} is not a column of df.')

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


# --------------------------
# Plots
# --------------------------
