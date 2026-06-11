from dataclasses import dataclass
from typing import Literal

import polars as pl
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.multitest import multipletests
import math

__all__ = ['newey_west_tstat', 'adf_test', 'stationarity_test', 'NWTestResult', 'ADFTestResult', 'fdr_correction',
           'benjamini_hochberg', 'benjamini_yekutieli']


# ------------------------------------------------------
# Newey-West test
# ------------------------------------------------------
@dataclass(frozen=True, slots=True)
class NWTestResult:
    """
    Result of Newey-West t-test for mean IC significance.

    Attributes
    ----------
    t_stat : float
        HAC-adjusted t-statistic for the null hypothesis mean(IC) = 0.
    p_value : float
        Two-sided p-value associated with t_stat.
    """
    t_stat: float
    p_value: float

def newey_west_tstat(ic_series: pd.Series | pl.Series) -> NWTestResult:
    """
    Compute the Newey-West t-statistic for an Information Coefficient (IC) time series.

    Uses statsmodels module for the computation.

    The IC series is regressed on a constant vector of ones. In a constant-only OLS regression,
    the estimated intercept equals the sample mean of the IC series. The resulting model is used
    to compute the t-statistic and p-value of the mean IC.

    Why this is necessary?
    ----------------------
    A standard t-statistic assumes independent and identically distributed samples,
    which leads to overstated statistical significance because the IC time series
    typically violates the indepentent and identically distributed (i.i.d.) assumption due to:
        1) Serial correlation,
        2) Overlapping observations,
        3) Time-varying volatility (heteroskedasticity).

    The Newey-West estimator (HAC) corrects the standard errors for both autocorrelation and
    heteroskedasticity, providing a more reliable estimate of the statistical significance
    of the mean IC.

    Lag Selection
    -------------
    The number of lags is selected automatically using the Andrews (1991) rule:
        lags = floor(4 * (T / 100) ^ (2/9))
    where T is the length of the IC series. This provides a data-driven
    bandwidth that balances bias and variance in the HAC estimator.

    Parameters
    ----------
    ic_series : pd.Series | pl.Series
        Time series by date of IC values, as returned by ic.compute_ic().

    Returns
    -------
    NWTestResult
        t-statistic for the mean IC and associated pvalue.

    Notes
    -----
    The null hypothesis is: mean(IC) = 0.
    p-value < 0.05 (5% significance)
        Reject the null hypothesis (mean !=0, result is significant).

    As a rule of thumb:
    |t-stat| > 2 -> moderate signal
    |t-stat| > 3 -> strong signal
    |t-stat| > 5 -> very strong signal

    Example
    -------
    A tstat of 3.1 and a p-value of 0.002 indicate strong evidence to reject the
    null hypothesis that the mean IC equals zero. Therefore, we assume that the
    signal is significant.
    """

    ic = ic_series.to_numpy()  # more appropriate for statsmodels

    # guard for empty or all-nan series
    if len(ic) == 0 or np.all(np.isnan(ic)):
        return NWTestResult(t_stat=np.nan, p_value=np.nan)

    # Constant IC series has zero uncertainty, making the t-stat undefined for practical purposes
    if np.isclose(np.std(ic, ddof=1), 0, atol=1e-8):
        return NWTestResult(t_stat=np.nan, p_value=np.nan)

    T = len(ic)
    andrews_lags = math.floor(4 * (T / 100) ** (2 / 9))

    X = np.ones(T)  # Constant-only regression used to estimate the mean IC

    model = sm.OLS(ic, X)
    results = model.fit(cov_type='HAC',
                        cov_kwds={'maxlags': andrews_lags})

    return NWTestResult(
        t_stat=float(results.tvalues[0]),
        p_value=float(results.pvalues[0])
    )


# ------------------------------------------------------
# ADF test - Stationarity test
# ------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ADFTestResult:
    """
    Result of ADF test for stationarity of a series.

    Attributes
    ----------
    t_stat : float
        t-statistic for the null hypothesis that there is a unit root = True = non-stationary.
    p_value : float
        p-value associated with the t_stat.
    is_stationary : bool | None
        p-value tested against the significance_level chosen by the user in adf_test function.
        True if p_value < significance_level, False otherwise.
        None if the test could not be performed (constant or short series).
    used_lags : int | None
        Effective number of lags used by adfuller.
    """
    t_stat: float
    p_value: float
    is_stationary: bool | None
    used_lags: int | None

def adf_test(
        series: pd.Series | pl.Series,
        significance_level: float = 0.05,
        minimum_observations: int = 20
) -> ADFTestResult:
    """
    Compute the ADF test to check the stationarity of a series.

    Assumes the default values of maxlag and autolag for the adfuller statsmodels function.

    Why is this necessary?
    ----------------------
    Feature research usually improves with stationary series. Feature usually need to be tested to garantee
    minimum stationarity.

    Parameters
    ----------
    series : pd.Series | pl.Series
        Series which will be tested for stationarity.
    significance_level : float
        Significance level in which the resulted p-value will be tested against.
    minimum_observations : int
        Minimum number of non-nan observations to garantee test robustness.

    Returns
    -------
    ADFTestResult
        t-statistic for the stationarity of the series, associated p-value, the test result
        based on the significance_level adopted and the number of lags used by adfuller.

    Notes
    -----
    The null hypothesis is that there is a unit root (non-stationary series).

    The ADF test is a left-tailed unit root test.

    The statsmodels adfuller function returns additional data, in different order, when store=True.
    This function always uses the default store=False. Under this configuration, adfuller always return
    t-stat, p-value and used_lags in consistent order.
    """
    np_series = series.to_numpy()  # more appropriate for statsmodels
    np_series = np_series[~np.isnan(np_series)]  # clean array

    # inital checks - minimum length and constant series
    if len(np_series) < minimum_observations:
        raise ValueError(f'Series is too short for a valid ADF test: {len(series)} < {minimum_observations}')

    if np.isclose(np.std(np_series, ddof=1), 0, atol=1e-8):
        return ADFTestResult(t_stat=np.nan, p_value=np.nan, is_stationary=None, used_lags=None)

    t_stat, p_value, used_lags, *_ = adfuller(np_series)  # only first three returns are required in this function

    return ADFTestResult(
        t_stat=float(t_stat),
        p_value=float(p_value),
        is_stationary=bool(p_value < significance_level),
        used_lags=int(used_lags)
    )

def stationarity_test(
        series: pd.Series | pl.Series,
        significance_level: float = 0.05,
        minimum_observations: int = 20
) -> ADFTestResult:
    """
    Alias for function adf_test.
    See adf_test() for full documentation.
    """
    return adf_test(series, significance_level, minimum_observations)


# ------------------------------------------------------
# FDR correction - BH and BY
# ------------------------------------------------------
def _fdr_correction_pandas(
        df: pd.DataFrame,
        fdr: float = 0.05,
        method: Literal['bh', 'by'] = 'bh'
) -> pd.DataFrame:
    """
    Core pandas implementation of fdr_correction.
    See fdr_correction() for full documentation.
    """
    df = df.copy()  # mutable array in pandas - requires copy.

    # get all group names in df['feature_group']
    groups = np.unique(df['feature_group'].to_numpy())

    for group in groups:
        mask = df['feature_group'] == group

        p_values = np.asarray(df.loc[mask, 'p_value'])

        rejected, corrected_p_values, *_ = multipletests(pvals=p_values, alpha=fdr, method=f'fdr_{method}')

        df.loc[mask, 'fdr_rejected'] = rejected  # True = passed fdr correction = signal is significant
        df.loc[mask, 'fdr_corrected_p_value'] = corrected_p_values

    return df

def fdr_correction(
        df: pd.DataFrame | pl.DataFrame,
        fdr: float = 0.05,
        method: Literal['bh', 'by'] = 'bh'
) -> pd.DataFrame | pl.DataFrame:
    """
    Apply the FDR correction (Benjamini-Hochberg - BH or Benjamini-Yekutieli - BY)
    on the result pandas DataFrame of ic.ic_summary_table.

    FDR correction is applied independently per feature_group.
    If all features share the same group (e.g. 'ungrouped', the default in
    ic_summary_table), correction runs once across all features, equivalent
    to a global FDR correction.

    Why is this necessary?
    ----------------------
    Multiple tries in feature generation may produce significant signals by chance.
    FDR correction removes most of the false positives while maximizing the gain of
    information with the features.

    BH and BY adopts a different philosophy than the more usual Benferroni, which tries
    to null the false positives, but possibly losing valuable information on the process.

    Parameters
    ----------
    df : pd.DataFrame
        df returned by ic.ic_summary_table with columns ['p_value', 'feature_group'].
    fdr : float
        significance level in which the corrected p-values will be tested against.
    method : {'bh', 'by'}
        bh = Benjamini-Hochberg correction - for independent or positive
        correlated tries.
        by = Benjamini-Yekutieli correction - for arbitrary dependence, including
        negative correlation. More conservative than BH.

    Returns
    -------
    pd.DataFrame
        DataFrame with all the columns given by ic.ic_summary_table and additional
        ['fdr_rejected', 'fdr_corrected_p_values'] columns with the results from the
        FDR correction.

    Raises
    ------
    KeyError
        If p_value, feature_group are not columns of the df.
    ValueError
        If df is not pandas or polars type.

    Notes
    -----
    Summary tables should not have different run performance in polars
    and pandas. Therefore, everything is transformed to pandas for code simplicity.
    """
    # Initial checks on columns
    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        raise TypeError('df must be Pandas or Polars DataFrame.')

    if any(col not in df.columns for col in ['p_value', 'feature_group']):
        raise KeyError(f"df must have both 'p_value' and 'feature_group' columns.")

    if isinstance(df, pd.DataFrame):
        return _fdr_correction_pandas(df, fdr, method)
    else:  # pl.Dataframe type
        return pl.from_pandas(_fdr_correction_pandas(df.to_pandas(), fdr, method))

def benjamini_hochberg(
        df: pd.DataFrame | pl.DataFrame,
        fdr: float = 0.05,
) -> pd.DataFrame | pl.DataFrame:
    """
    Apply Benjamini-Hochberg FDR correction per feature group.

    Alias for fdr_correction() with method='bh'. Assumes independence or positive
    correlation between tests within the same group.

    For arbitrary dependence, use benjamini_yekutieli().

    See fdr_correction() for full documentation.
    """
    return fdr_correction(df, fdr, 'bh')

def benjamini_yekutieli(
        df: pd.DataFrame | pl.DataFrame,
        fdr: float = 0.05,
) -> pd.DataFrame | pl.DataFrame:
    """
    Apply Benjamini-Yekutieli FDR correction per feature group.

    Alias for fdr_correction() with method='by'. Valid under arbitrary
    dependence between tests, including negative correlation.

    More conservative than benjamini_hochberg().

    See fdr_correction() for full documentation.
    """
    return fdr_correction(df, fdr, 'by')
