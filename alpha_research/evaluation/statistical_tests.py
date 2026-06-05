from dataclasses import dataclass

import polars as pl
import pandas as pd
import numpy as np
import statsmodels.api as sm
import math

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
