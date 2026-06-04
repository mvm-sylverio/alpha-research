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

def compute_newey_west_tstat(
        df_ic: pl.DataFrame,
        target_col: str,
        fwd_periods: int):
    Attributes
    ----------
    t_stat : float
        HAC-adjusted t-statistic for the null hypothesis mean(IC) = 0.
    p_value : float
        Two-sided p-value associated with t_stat.
    """
    t_stat: float
    p_value: float

    """
    Compute the Newey-West (HAC-adjusted) t-statistic for the mean of an IC time series.

    Why this is necessary:
    ----------------------
    The Information Coefficient (IC) time series typically violates the i.i.d. assumption
    due to:
        1) Serial correlation (especially when using multi-period forward returns, e.g. 5, 10 bars),
        2) Overlapping observations,
        3) Time-varying volatility (heteroskedasticity).

    A standard t-statistic assumes independent and identically distributed samples,
    which leads to overstated statistical significance in the presence of these issues.

    The Newey-West estimator corrects the standard errors for both autocorrelation and
    heteroskedasticity, providing a more reliable estimate of the statistical significance
    of the mean IC.

    :param df_ic:
    :param target_col:
    :param fwd_periods:
    :return: float HAC-adjusted t-statistic for the mean IC, float pvalue.
    """

    ic = df_ic[f'IC_{target_col}'].to_numpy()

    # Constant IC series has zero uncertainty, making the t-stat undefined for practical purposes
    if np.isclose(np.std(ic, ddof=1), 0, atol=1e-8):
        return NWTestResult(t_stat=np.nan, p_value=np.nan)

    T = len(ic)
    andrews_lags = math.floor(4 * (T / 100) ** (2 / 9))

    X = np.ones(len(ic))
    model = sm.OLS(ic, X)
    results = model.fit(cov_type='HAC',
                        cov_kwds={'maxlags': andrews_lags})

    return NWTestResult(
        t_stat=float(results.tvalues[0]),
        p_value=float(results.pvalues[0])
    )
