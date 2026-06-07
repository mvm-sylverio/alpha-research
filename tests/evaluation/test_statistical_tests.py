import pytest
import pandas as pd
import polars as pl
import numpy as np

from alpha_research.evaluation.statistical_tests import (newey_west_tstat, NWTestResult, adf_test, ADFTestResult,
                                                         stationarity_test)


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

@pytest.fixture
def stationary_series():
    """
    White-noise series - clearly stationary. Mean = 0, no trend, no unit root.
    Expected: is_stationary=True, p_value < 0.05
    """
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0, 1, 200))

@pytest.fixture
def nonstationary_series():
    """
    Random walk - clearly non-stationary (unit root).
    Constructed as cumulative sum of randomness.
    Expected: is_stationary=False, p_value > 0.05
    """
    rng = np.random.default_rng(42)
    return pd.Series(np.cumsum(rng.normal(0, 1, 200)))

@pytest.fixture
def constant_series():
    """Constant series - std=0, ADF undefined."""
    return pd.Series([1.0] * 100)

@pytest.fixture
def short_series():
    """Series shorter than minimum_observations."""
    return pd.Series([1.0, 2.0, 3.0])

@pytest.fixture
def series_with_many_nans():
    """
    Series with 100 observations but only 10 valid after NaN removal.
    Should raise ValueError - effective length < minimum_observations.
    """
    arr = np.full(100, np.nan)
    arr[:10] = np.linspace(0, 1, 10)
    return pd.Series(arr)


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

# pandas-polars consistency
def test_nw_pandas_polars_consistency(strong_positive_ic_series):
    """Should return identical results for pandas and polars input."""
    pl_series = pl.Series(strong_positive_ic_series.to_numpy())
    res_pd = newey_west_tstat(strong_positive_ic_series)
    res_pl = newey_west_tstat(pl_series)
    assert res_pd.t_stat == pytest.approx(res_pl.t_stat, rel=1e-6)
    assert res_pd.p_value == pytest.approx(res_pl.p_value, rel=1e-6)

# tests for empty or all nan ic series - edge cases
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


# ------------------------------------------------------
# ADFTestResult
# ------------------------------------------------------
def test_adf_returns_fields_adftestresult(stationary_series):
    """Should return ADFTestResult instance and expected fields."""
    result = adf_test(stationary_series)
    assert isinstance(result, ADFTestResult)
    assert hasattr(result, 't_stat')
    assert hasattr(result, 'p_value')
    assert hasattr(result, 'is_stationary')
    assert hasattr(result, 'used_lags')


# ------------------------------------------------------
# adf_test and stationarity_test (alias)
# ------------------------------------------------------
def test_adf_stationary_series(stationary_series):
    """Stationary_series should be detected as stationary."""
    result = adf_test(stationary_series)
    assert result.is_stationary is True
    assert result.p_value < 0.05

def test_adf_nonstationary_series(nonstationary_series):
    """Non-stationary series should be detected as non-stationary."""
    result = adf_test(nonstationary_series)
    assert result.is_stationary is False
    assert result.p_value > 0.05

def test_adf_used_lags_is_positive_integer(stationary_series, nonstationary_series):
    """used_lags should be a non-negative integer for all valid series."""
    result_stationary = adf_test(stationary_series)
    result_non_stationary = adf_test(nonstationary_series)
    assert isinstance(result_stationary.used_lags, int)
    assert result_stationary.used_lags >= 0
    assert isinstance(result_non_stationary.used_lags, int)
    assert result_non_stationary.used_lags >= 0


# significance level
def test_adf_significance_level(nonstationary_series):
    """is_stationary should reflect comparison of p_value against significance_level."""
    result = adf_test(nonstationary_series)

    above = adf_test(nonstationary_series, significance_level=result.p_value + 0.001)
    below = adf_test(nonstationary_series, significance_level=result.p_value - 0.001)

    assert above.is_stationary is True
    assert below.is_stationary is False

# edge cases
def test_adf_constant_series_returns_nan(constant_series):
    """Constant series should return nan and is_stationary=None."""
    result = adf_test(constant_series)
    assert np.isnan(result.t_stat)
    assert np.isnan(result.p_value)
    assert result.is_stationary is None
    assert result.used_lags is None

def test_adf_short_series_raises(short_series):
    """Series shorter than minimum_observations should raise ValueError."""
    with pytest.raises(ValueError, match="Series is too short"):
        adf_test(short_series)

def test_adf_series_with_many_nans_raises(series_with_many_nans):
    """Series with effective length < minimum_observations after NaN removal should raise ValueError."""
    with pytest.raises(ValueError, match="Series is too short"):
        adf_test(series_with_many_nans)

def test_adf_pvalue_in_valid_range(stationary_series, nonstationary_series):
    """p-value should always be in [0, 1]."""
    for series in [stationary_series, nonstationary_series]:
        p = adf_test(series).p_value
        assert 0.0 <= p <= 1.0

# pandas-polars consistency
def test_adf_pandas_polars_consistency(stationary_series):
    """Should return identical results for pandas and polars input."""
    pl_series = pl.Series(stationary_series.to_numpy())
    res_pd = adf_test(stationary_series)
    res_pl = adf_test(pl_series)
    assert res_pd.t_stat == pytest.approx(res_pl.t_stat, rel=1e-6)
    assert res_pd.p_value == pytest.approx(res_pl.p_value, rel=1e-6)
    assert res_pd.is_stationary == res_pl.is_stationary
    assert res_pd.used_lags == res_pl.used_lags

# alias
def test_stationarity_test_is_alias(stationary_series):
    """stationarity_test should return identical results to adf_test."""
    res_adf = adf_test(stationary_series)
    res_alias = stationarity_test(stationary_series)
    assert res_adf.t_stat == pytest.approx(res_alias.t_stat)
    assert res_adf.p_value == pytest.approx(res_alias.p_value)
    assert res_adf.is_stationary == res_alias.is_stationary
    assert res_adf.used_lags == res_alias.used_lags
