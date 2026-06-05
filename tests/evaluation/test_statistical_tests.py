import pytest
import pandas as pd
import polars as pl
import numpy as np

from alpha_research.evaluation.statistical_tests import newey_west_tstat, NWTestResult


# ------------------------------------------------------
# fixtures
# ------------------------------------------------------
@pytest.fixture
def constant_ic_series():
    """
    Constant IC series -> std=0, t-stat undefined.
    All values equal to 0.05.
    """
    return pd.Series([0.05] * 100)

@pytest.fixture
def strong_positive_ic_series():
    """
    Strong positive IC series with 100 observations: mean=0.10.
    Expected: |t-stat| >> 2, p-value << 0.05.

    Constructed as arithmetic progression.
    """
    base = np.linspace(0.08, 0.12, 100)  # determinístico, variância controlada
    return pd.Series(base)

@pytest.fixture
def strong_negative_ic_series():
    """
    Strong negative IC series with 100 observations: mean=-0.10.
    Expected: |t-stat| >> 2, p-value << 0.05.

    Constructed as arithmetic progression.
    """
    base = np.linspace(-0.08, -0.12, 100)  # determinístico, variância controlada
    return pd.Series(base)

@pytest.fixture
def zero_mean_ic_series():
    """
    Symmetric IC series around zero - mean exactly 0 by construction.
    Expected: p-value high, fail to reject null.
    """
    base = np.linspace(-0.05, 0.05, 100)  # simétrico, mean=0 exato
    return pd.Series(base)


# ------------------------------------------------------
# NWTestResult
# ------------------------------------------------------
def test_nw_returns_nwtestresult(strong_positive_ic_series):
    """Should return NWTestResult instance and fields."""
    result = newey_west_tstat(strong_positive_ic_series)
    assert isinstance(result, NWTestResult)
    assert hasattr(result, 't_stat')
    assert hasattr(result, 'p_value')


# ------------------------------------------------------
# newey_west_tstat
# ------------------------------------------------------
def test_nw_constant_series_returns_nan(constant_ic_series):
    """Should return nan t_stat and p_value for constant IC series."""
    result = newey_west_tstat(constant_ic_series)
    assert np.isnan(result.t_stat)
    assert np.isnan(result.p_value)

def test_nw_high_ic_tstat_significant(strong_positive_ic_series, strong_negative_ic_series):
    """Strong IC series should produce |t-stat| > 2."""
    assert abs(newey_west_tstat(strong_positive_ic_series).t_stat) > 2
    assert abs(newey_west_tstat(strong_negative_ic_series).t_stat) > 2

def test_nw_high_ic_pvalue_significant(strong_positive_ic_series, strong_negative_ic_series):
    """Strong IC series should produce p-value < 0.05."""
    assert newey_west_tstat(strong_positive_ic_series).p_value < 0.05
    assert newey_west_tstat(strong_negative_ic_series).p_value < 0.05

def test_nw_zero_mean_pvalue_not_significant(zero_mean_ic_series):
    """Zero mean IC should fail to reject null - p-value > 0.05."""
    assert newey_west_tstat(zero_mean_ic_series).p_value > 0.05

def test_nw_tstat_sign_matches_ic_mean_sign(strong_positive_ic_series, strong_negative_ic_series):
    """t-stat sign should match IC mean sign."""
    assert newey_west_tstat(strong_positive_ic_series).t_stat > 0
    assert newey_west_tstat(strong_negative_ic_series).t_stat < 0

def test_nw_pvalue_in_valid_range(strong_positive_ic_series, zero_mean_ic_series, strong_negative_ic_series):
    """p-value should always be in [0, 1]."""
    for series in [strong_positive_ic_series, zero_mean_ic_series, strong_negative_ic_series]:
        p = newey_west_tstat(series).p_value
        assert 0.0 <= p <= 1.0

def test_nw_pandas_polars_consistency(strong_positive_ic_series):
    """Should return identical results for pandas and polars input."""
    pl_series = pl.Series(strong_positive_ic_series.to_numpy())
    res_pd = newey_west_tstat(strong_positive_ic_series)
    res_pl = newey_west_tstat(pl_series)
    assert res_pd.t_stat == pytest.approx(res_pl.t_stat, rel=1e-6)
    assert res_pd.p_value == pytest.approx(res_pl.p_value, rel=1e-6)

# tests for empty or all nan ic series
def test_nw_empty_series_returns_nan():
    """Should return nan for empty series."""
    result = newey_west_tstat(pd.Series([], dtype=float))
    assert np.isnan(result.t_stat)
    assert np.isnan(result.p_value)

def test_nw_all_nan_series_returns_nan():
    """Should return nan for all-nan series."""
    result = newey_west_tstat(pd.Series([np.nan, np.nan, np.nan]))
    assert np.isnan(result.t_stat)
    assert np.isnan(result.p_value)
