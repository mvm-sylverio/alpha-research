import pytest
import pandas as pd
import polars as pl
import numpy as np
from scipy.stats import norm

from alpha_research.evaluation.statistical_tests import (
    ADFTestResult,
    NWTestResult,
    WaldTemporalAssociationTestResult,
    adf_test,
    benjamini_hochberg,
    benjamini_yekutieli,
    fdr_correction,
    newey_west_tstat,
    stationarity_test,
    wald_temporal_association_test,
)


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

@pytest.fixture
def single_group_table():
    """
    ic_summary_table output with single group.
    p_values chosen so that some pass and some fail at fdr=0.05.
    """
    return pd.DataFrame({
        'feature':       ['f1', 'f2', 'f3', 'f4', 'f5'],
        'p_value':       [0.001, 0.008, 0.039, 0.041, 0.200],
        'feature_group': ['momentum'] * 5,
        'mean':          [0.05, 0.04, 0.03, 0.02, 0.01],
    })

@pytest.fixture
def multi_group_table():
    """
    ic_summary_table output with two groups.
    BH applied independently per group.
    """
    return pd.DataFrame({
        'feature':       ['f1', 'f2', 'f3', 'f4', 'f5', 'f6'],
        'p_value':       [0.001, 0.008, 0.200, 0.001, 0.008, 0.200],
        'feature_group': ['momentum', 'momentum', 'momentum',
                          'reversal', 'reversal', 'reversal'],
        'mean':          [0.05, 0.04, 0.01, 0.05, 0.04, 0.01],
    })

@pytest.fixture
def all_significant_table():
    """All p_values clearly below threshold."""
    return pd.DataFrame({
        'feature':       ['f1', 'f2', 'f3'],
        'p_value':       [0.001, 0.002, 0.003],
        'feature_group': ['momentum'] * 3,
        'mean':          [0.05, 0.04, 0.03],
    })

@pytest.fixture
def none_significant_table():
    """All p_values clearly above threshold."""
    return pd.DataFrame({
        'feature':       ['f1', 'f2', 'f3'],
        'p_value':       [0.300, 0.400, 0.500],
        'feature_group': ['momentum'] * 3,
        'mean':          [0.01, 0.01, 0.01],
    })

@pytest.fixture
def non_finite_pvalue_table():
    """Table with finite and non-finite p-values in the same group."""
    return pd.DataFrame({
        'feature':       ['f1', 'f2', 'f3', 'f4'],
        'p_value':       [0.01, np.nan, np.inf, -np.inf],
        'feature_group': ['momentum'] * 4,
        'mean':          [0.05, 0.04, 0.03, 0.02],
    })

@pytest.fixture
def all_non_finite_pvalue_table():
    """Table containing no finite p-values."""
    return pd.DataFrame({
        'feature':       ['f1', 'f2', 'f3'],
        'p_value':       [np.nan, np.inf, -np.inf],
        'feature_group': ['momentum'] * 3,
        'mean':          [0.05, 0.04, 0.03],
    })


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
# wald_temporal_association_test
# ------------------------------------------------------
def test_wald_temporal_association_test_returns_result_type():
    """Should return a WaldTemporalAssociationTestResult instance."""
    result = wald_temporal_association_test(0.20, 0.05)

    assert isinstance(result, WaldTemporalAssociationTestResult)


def test_wald_temporal_association_test_calculates_statistic():
    """Should standardize the observed association around the null value."""
    result = wald_temporal_association_test(0.20, 0.05)

    assert result.test_statistic == pytest.approx(4.0)


def test_wald_temporal_association_test_calculates_normal_two_sided_p_value():
    """Should calculate the two-sided p-value from the standard Normal tail."""
    result = wald_temporal_association_test(0.10, 0.05)

    assert result.p_value == pytest.approx(2 * norm.sf(2.0))


@pytest.mark.parametrize(
    'irrelevant_argument',
    [
        'bootstrap_ci_lower',
        'bootstrap_ci_upper',
        'n_non_positive',
        'n_non_negative',
        'n_bootstraps',
    ],
)
def test_wald_temporal_association_test_does_not_accept_bootstrap_distribution_metrics(
        irrelevant_argument,
):
    """Should depend on the bootstrap standard error, not other bootstrap metrics."""
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        wald_temporal_association_test(
            0.10,
            0.05,
            **{irrelevant_argument: 0.0},
        )


def test_wald_temporal_association_test_rejects_positive_association():
    """Should reject H0 for a sufficiently positive Wald statistic."""
    result = wald_temporal_association_test(0.20, 0.05)

    assert result.reject_h0 is True
    assert result.test_statistic == pytest.approx(4.0)


def test_wald_temporal_association_test_rejects_negative_association():
    """Should reject H0 for a sufficiently negative Wald statistic."""
    result = wald_temporal_association_test(-0.20, 0.05)

    assert result.reject_h0 is True
    assert result.test_statistic == pytest.approx(-4.0)


def test_wald_temporal_association_test_does_not_reject_non_significant_association():
    """Should not reject H0 when the Normal p-value is large."""
    result = wald_temporal_association_test(0.02, 0.05)

    assert result.reject_h0 is False
    assert result.p_value > 0.05


def test_wald_temporal_association_test_calculates_wald_confidence_interval():
    """Should calculate Wald interval bounds from the Normal critical value."""
    result = wald_temporal_association_test(0.10, 0.05)
    z_critical = norm.ppf(0.975)

    assert result.wald_ci_lower == pytest.approx(0.10 - z_critical * 0.05)
    assert result.wald_ci_upper == pytest.approx(0.10 + z_critical * 0.05)


@pytest.mark.parametrize('confidence_level', [0.95, 0.90])
def test_wald_temporal_association_test_uses_confidence_level_for_interval(
        confidence_level,
):
    """Should use the chosen confidence level for alpha and the Wald interval."""
    result = wald_temporal_association_test(
        0.10,
        0.05,
        confidence_level=confidence_level,
    )
    alpha = 1 - confidence_level
    z_critical = norm.ppf(1 - alpha / 2)

    assert result.alpha == pytest.approx(alpha)
    assert result.wald_ci_lower == pytest.approx(0.10 - z_critical * 0.05)
    assert result.wald_ci_upper == pytest.approx(0.10 + z_critical * 0.05)


@pytest.mark.parametrize('observed_association, bootstrap_standard_error', [
    (0.20, 0.05),
    (-0.20, 0.05),
    (0.02, 0.05),
])
def test_wald_temporal_association_test_p_value_matches_wald_interval_decision(
        observed_association,
        bootstrap_standard_error,
):
    """Should reject exactly when zero is outside the Wald confidence interval."""
    result = wald_temporal_association_test(
        observed_association,
        bootstrap_standard_error,
    )
    null_outside_ci = not (
        result.wald_ci_lower <= 0.0 <= result.wald_ci_upper
    )

    assert result.reject_h0 is null_outside_ci


def test_wald_temporal_association_test_wald_interval_contains_zero():
    """Should retain H0 when the Wald interval contains zero."""
    result = wald_temporal_association_test(0.02, 0.05)

    assert result.wald_ci_lower <= 0.0 <= result.wald_ci_upper
    assert result.reject_h0 is False


def test_wald_temporal_association_test_wald_interval_excludes_zero():
    """Should reject H0 when the Wald interval excludes zero."""
    result = wald_temporal_association_test(0.20, 0.05)

    assert not result.wald_ci_lower <= 0.0 <= result.wald_ci_upper
    assert result.reject_h0 is True


def test_wald_temporal_association_test_uses_null_value():
    """Should center the statistic and Wald interval around a non-zero null value."""
    result = wald_temporal_association_test(0.30, 0.05, null_value=0.20)

    assert result.test_statistic == pytest.approx(2.0)
    assert result.p_value == pytest.approx(2 * norm.sf(2.0))
    assert result.reject_h0 is True


@pytest.mark.parametrize('observed_association', [np.nan, np.inf, -np.inf])
def test_wald_temporal_association_test_rejects_non_finite_observed_association(
        observed_association,
):
    """Should reject non-finite associations from the original sample."""
    with pytest.raises(ValueError, match='must be finite'):
        wald_temporal_association_test(observed_association, 0.05)


@pytest.mark.parametrize('bootstrap_standard_error', [np.nan, np.inf, -np.inf])
def test_wald_temporal_association_test_rejects_non_finite_standard_error(
        bootstrap_standard_error,
):
    """Should reject non-finite bootstrap standard errors."""
    with pytest.raises(ValueError, match='must be finite'):
        wald_temporal_association_test(0.10, bootstrap_standard_error)


@pytest.mark.parametrize('null_value', [np.nan, np.inf, -np.inf])
def test_wald_temporal_association_test_rejects_non_finite_null_value(null_value):
    """Should reject non-finite null-hypothesis values."""
    with pytest.raises(ValueError, match='must be finite'):
        wald_temporal_association_test(0.10, 0.05, null_value=null_value)


@pytest.mark.parametrize(
    'argument',
    ['observed_association', 'bootstrap_standard_error', 'null_value'],
)
def test_wald_temporal_association_test_rejects_non_numeric_values(argument):
    """Should reject non-numeric association, standard error, and null values."""
    kwargs = {
        'observed_association': 0.10,
        'bootstrap_standard_error': 0.05,
        'null_value': 0.0,
    }
    kwargs[argument] = 'invalid'

    with pytest.raises(ValueError, match='must be numeric'):
        wald_temporal_association_test(**kwargs)


@pytest.mark.parametrize('confidence_level', [0.0, 1.0, -0.1, 1.1, True])
def test_wald_temporal_association_test_rejects_invalid_confidence_level(confidence_level):
    """Should reject confidence levels outside the open interval from zero to one."""
    with pytest.raises(ValueError, match='confidence_level'):
        wald_temporal_association_test(0.10, 0.05, confidence_level)


@pytest.mark.parametrize('bootstrap_standard_error', [0.0, -0.01])
def test_wald_temporal_association_test_rejects_non_positive_standard_error(
        bootstrap_standard_error,
):
    """Should reject zero or negative standard errors that make Wald inference invalid."""
    with pytest.raises(ValueError, match='greater than zero'):
        wald_temporal_association_test(0.10, bootstrap_standard_error)


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


# ------------------------------------------------------
# FDR correction and aliases
# ------------------------------------------------------
# structure
def test_fdr_returns_dataframe_and_output_columns(single_group_table):
    """Should return pandas DataFrame and fdr_rejected and fdr_corrected_p_value columns."""
    result = fdr_correction(single_group_table)
    assert isinstance(result, pd.DataFrame)
    assert 'fdr_rejected' in result.columns
    assert 'fdr_corrected_p_value' in result.columns

def test_fdr_preserves_input_columns_and_row_count(single_group_table):
    """Should preserve all original columns of df and same number of rows as input."""
    result = fdr_correction(single_group_table)
    for col in single_group_table.columns:
        assert col in result.columns
    assert len(result) == len(single_group_table)

def test_fdr_does_not_mutate_input(single_group_table):
    """Should not modify the original DataFrame - columns and values."""
    original = single_group_table.copy()
    fdr_correction(single_group_table)
    pd.testing.assert_frame_equal(single_group_table, original)

# results
def test_fdr_all_significant(all_significant_table):
    """All clearly significant p_values should all be rejected=True."""
    result = fdr_correction(all_significant_table)
    assert result['fdr_rejected'].all()

def test_fdr_none_significant(none_significant_table):
    """All clearly non-significant p_values should be rejected=False."""
    result = fdr_correction(none_significant_table)
    assert not result['fdr_rejected'].any()

def test_fdr_corrected_pvalues_geq_original(single_group_table):
    """Corrected p_values should always be >= original p_values."""
    result = fdr_correction(single_group_table)
    assert (result['fdr_corrected_p_value'] >= result['p_value']).all()

def test_fdr_corrected_pvalues_in_valid_range(single_group_table):
    """Corrected p_values should be in [0, 1]."""
    result = fdr_correction(single_group_table)
    assert (result['fdr_corrected_p_value'] >= 0).all()
    assert (result['fdr_corrected_p_value'] <= 1).all()


def test_fdr_excludes_non_finite_pvalues(non_finite_pvalue_table):
    """Should leave non-finite p-values unrejected and uncorrected."""
    result = fdr_correction(non_finite_pvalue_table)

    assert bool(result.loc[0, 'fdr_rejected']) is True
    assert result.loc[0, 'fdr_corrected_p_value'] == pytest.approx(0.01)
    assert not result.loc[1:, 'fdr_rejected'].any()
    assert result.loc[1:, 'fdr_corrected_p_value'].isna().all()


def test_fdr_handles_group_with_only_non_finite_pvalues(all_non_finite_pvalue_table):
    """Should return default FDR results when a group has no finite p-values."""
    result = fdr_correction(all_non_finite_pvalue_table)

    assert not result['fdr_rejected'].any()
    assert result['fdr_corrected_p_value'].isna().all()

# groups
def test_fdr_applied_per_group(multi_group_table):
    """BH should be applied independently per group — same p_values in different groups should yield same results."""
    result = fdr_correction(multi_group_table)
    momentum = result[result['feature_group'] == 'momentum']
    reversal = result[result['feature_group'] == 'reversal']
    assert list(momentum['fdr_rejected']) == list(reversal['fdr_rejected'])
    assert momentum['fdr_corrected_p_value'].values == pytest.approx(
        reversal['fdr_corrected_p_value'].values, abs=1e-8)


def test_fdr_accepts_custom_column_names(single_group_table):
    """Should use the configured p-value and feature-group columns."""
    custom_table = single_group_table.rename(
        columns={'p_value': 'probability', 'feature_group': 'family'}
    )

    result = fdr_correction(
        custom_table,
        p_value_column='probability',
        feature_group_column='family',
    )
    expected = fdr_correction(single_group_table)

    pd.testing.assert_series_equal(
        result['fdr_rejected'], expected['fdr_rejected'], check_names=False
    )
    pd.testing.assert_series_equal(
        result['fdr_corrected_p_value'],
        expected['fdr_corrected_p_value'],
        check_names=False,
    )
    assert 'probability' in result.columns
    assert 'family' in result.columns


def test_fdr_accepts_custom_column_names_for_polars(single_group_table):
    """Should support custom column names for Polars input as well."""
    custom_table = pl.from_pandas(
        single_group_table.rename(
            columns={'p_value': 'probability', 'feature_group': 'family'}
        )
    )

    result = fdr_correction(
        custom_table,
        p_value_column='probability',
        feature_group_column='family',
    )
    expected = fdr_correction(
        custom_table.to_pandas(),
        p_value_column='probability',
        feature_group_column='family',
    )

    assert isinstance(result, pl.DataFrame)
    pd.testing.assert_frame_equal(
        result.to_pandas().reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    ('alias', 'method'),
    [(benjamini_hochberg, 'bh'), (benjamini_yekutieli, 'by')],
)
def test_fdr_aliases_accept_custom_column_names(single_group_table, alias, method):
    """BH and BY aliases should forward custom column names."""
    custom_table = single_group_table.rename(
        columns={'p_value': 'probability', 'feature_group': 'family'}
    )

    result = alias(
        custom_table,
        p_value_column='probability',
        feature_group_column='family',
    )
    expected = fdr_correction(
        custom_table,
        method=method,
        p_value_column='probability',
        feature_group_column='family',
    )

    pd.testing.assert_frame_equal(result, expected)


def test_fdr_custom_column_names_are_validated(single_group_table):
    """Should raise KeyError when a configured column is missing."""
    with pytest.raises(KeyError):
        fdr_correction(
            single_group_table,
            p_value_column='probability',
            feature_group_column='family',
        )

# raises
def test_fdr_missing_column_raises(single_group_table):
    """Should raise KeyError if required columns are missing."""
    df = single_group_table.drop(columns=['p_value'])
    with pytest.raises(KeyError):
        fdr_correction(df)

def test_fdr_invalid_type_raises():
    """Should raise TypeError for unsupported input types."""
    with pytest.raises(TypeError):
        fdr_correction([[1, 2], [3, 4]])

# BH vs BY
def test_fdr_by_more_conservative_than_bh(single_group_table):
    """
    BY should reject fewer or equal features than BH.
    BY should have corrected p-values equal or higher than BH.
    """
    result_bh = fdr_correction(single_group_table, method='bh')
    result_by = fdr_correction(single_group_table, method='by')
    assert result_by['fdr_rejected'].sum() <= result_bh['fdr_rejected'].sum()
    assert (result_by['fdr_corrected_p_value'] >= result_bh['fdr_corrected_p_value']).all()

# aliases
def test_benjamini_hochberg_alias(single_group_table):
    """benjamini_hochberg should return identical results to fdr_correction with method='bh'."""
    res_alias = benjamini_hochberg(single_group_table)
    res_direct = fdr_correction(single_group_table, method='bh')
    pd.testing.assert_frame_equal(res_alias, res_direct)

def test_benjamini_yekutieli_alias(single_group_table):
    """benjamini_yekutieli should return identical results to fdr_correction with method='by'."""
    res_alias = benjamini_yekutieli(single_group_table)
    res_direct = fdr_correction(single_group_table, method='by')
    pd.testing.assert_frame_equal(res_alias, res_direct)

# pandas / polars consistency
def test_fdr_pandas_polars_consistency(single_group_table):
    """Should return identical results for pandas and polars input."""
    pl_table = pl.from_pandas(single_group_table)
    res_pd = fdr_correction(single_group_table)
    res_pl = fdr_correction(pl_table).to_pandas()
    pd.testing.assert_frame_equal(
        res_pd.reset_index(drop=True),
        res_pl.reset_index(drop=True),
        check_dtype=False
    )


def test_fdr_non_finite_pvalues_pandas_polars_consistency(non_finite_pvalue_table):
    """Should handle non-finite p-values identically for both backends."""
    polars_table = pl.from_pandas(non_finite_pvalue_table)

    result_pandas = fdr_correction(non_finite_pvalue_table)
    result_polars = fdr_correction(polars_table).to_pandas()

    pd.testing.assert_frame_equal(
        result_pandas.reset_index(drop=True),
        result_polars.reset_index(drop=True),
        check_dtype=False,
    )
