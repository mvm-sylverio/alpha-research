import polars as pl
import numpy as np
import statsmodels.api as sm


def compute_newey_west_tstat(
        df_ic: pl.DataFrame,
        target_col: str,
        fwd_periods: int):
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

    # Andrew_Lags
    T = len(ic)
    andrews_lags = int(4 * (T / 100) ** (2 / 9))

    X = np.ones(len(ic))
    model = sm.OLS(ic, X)
    results = model.fit(cov_type='HAC',
                        cov_kwds={'maxlags': max(fwd_periods - 1, andrews_lags)})

    return results.tvalues[0], results.pvalues[0]
