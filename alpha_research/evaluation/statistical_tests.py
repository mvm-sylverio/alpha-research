from dataclasses import dataclass

import polars as pl
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import math

__all__ = ['newey_west_tstat', 'adf_test', 'stationarity_test', 'NWTestResult', 'ADFTestResult']


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
    Feature research requires stationary series. Feature usually need to be tested to garantee
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
    """Alias for function adf_test. See its documentation for full details."""
    return adf_test(series, significance_level, minimum_observations)
