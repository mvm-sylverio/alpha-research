import pytest
import numpy as np
import pandas as pd
import polars as pl
from alpha_research.evaluation.ic import information_coefficient, compute_ic, compute_ic_metrics, ic_summary_table


# ------------------------------------------------------
# fixtures
# ------------------------------------------------------
@pytest.fixture
def perfect_corr():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    return x, x.copy()

@pytest.fixture
def perfect_inverse():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
    return x, y

@pytest.fixture
def cross_section_df_pandas():
    """
    Two dates, with 5 assets each.
    date 1: perfect correlation
    date 2: perfect inverse correlation
    """
    return pd.DataFrame({
        'time': ['2024-01-01'] * 5 + ['2024-01-02'] * 5,
        'feature': [1.0, 2.0, 3.0, 4.0, 5.0,
                    1.0, 2.0, 3.0, 4.0, 5.0],
        'target':  [1.0, 2.0, 3.0, 4.0, 5.0,
                    5.0, 4.0, 3.0, 2.0, 1.0],
    })

@pytest.fixture
def cross_section_df_polars():
    return pl.DataFrame({
        'time': [
            '2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01',
            '2024-01-02', '2024-01-02', '2024-01-02', '2024-01-02', '2024-01-02',
        ],
        'feature': [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        'target':  [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    }).with_columns(pl.col('time').str.to_datetime())

# values were computed manually and checked
@pytest.fixture
def positive_ic_series():
    return pd.Series([0.04, 0.08, 0.06, 0.10, 0.02])
    # mean    = 0.060
    # abs_mean = mean(abs) = 0.060
    # std     = 0.03162  (ddof=1)
    # stability = abs(mean)/std = 0.060/0.03162 = approx 1.897
    # sign    = 1
    # pct_pos = 1.0

@pytest.fixture
def negative_ic_series():
    return pd.Series([-0.04, -0.08, -0.06, -0.10, -0.02])
    # mean     = -0.060
    # abs_mean = 0.060
    # sign     = -1
    # pct_pos  = 1.0  (should be all positive)

@pytest.fixture
def mixed_ic_series():
    return pd.Series([0.10, -0.10, 0.06, -0.06, 0.04])
    # mean     = 0.008  → sign = 1
    # abs_mean = mean([0.10, 0.10, 0.06, 0.06, 0.04]) = 0.072
    # pct_pos on adjusted series: [0.10, -0.10, 0.06, -0.06, 0.04] = 0.6

@pytest.fixture
def multi_feature_df_pandas():
    """
    DataFrame with 3 features and 2 dates, 5 assets each.
    feature_a: perfect positive correlation on both dates
    feature_b: perfect inverse correlation on both dates
    feature_c: constant values

    Precomputed:
        feature_a: mean IC ≈ 1.0
        feature_b: mean IC ≈ -1.0
        feature_c: mean IC = nan (constant)
    """
    return pd.DataFrame({
        'time':      ['2024-01-01'] * 5 + ['2024-01-02'] * 5,
        'feature_a': [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
        'feature_b': [5.0, 4.0, 3.0, 2.0, 1.0] * 2,
        'feature_c': [1.0, 1.0, 1.0, 1.0, 1.0] * 2,
        'target':    [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
    })


# ------------------------------------------------------
# information_coefficient
# ------------------------------------------------------
def test_ic_perfect_positive(perfect_corr):
    """Should return 1.0 for perfect positive correlation with spearman corr_method."""
    x, y = perfect_corr
    assert information_coefficient(x, y) == pytest.approx(1.0)

def test_ic_perfect_inverse(perfect_inverse):
    """Should return -1.0 for perfect inverse correlation with spearman corr_method."""
    x, y = perfect_inverse
    assert information_coefficient(x, y) == pytest.approx(-1.0)

def test_ic_pearson(perfect_corr):
    """Should return 1.0 for perfect positive correlation with pearson corr_method."""
    x, y = perfect_corr
    assert information_coefficient(x, y, corr_method='pearson') == pytest.approx(1.0)

def test_ic_length_mismatch_raises():
    """Should raise ValueError on length mismatch."""
    with pytest.raises(ValueError, match="len"):
        information_coefficient(pd.Series([1, 2]), pd.Series([1]))

def test_ic_invalid_method_raises():
    """Should raise ValueError for unsupported corr_method."""
    x = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="corr_method"):
        information_coefficient(x, x, corr_method='kendall')


# ------------------------------------------------------
# compute_ic
# ------------------------------------------------------
def test_compute_ic_pandas_shape(cross_section_df_pandas):
    """Should return one IC value per date and ic should be one of the two columns."""
    result = compute_ic(cross_section_df_pandas, 'feature', 'target')
    assert result.shape == (2, 2)
    assert 'ic' in result.columns

def test_compute_ic_pandas_values(cross_section_df_pandas):
    """Should return perfect positive correlation in first date and perfect inverse correlation in second date."""
    result = compute_ic(cross_section_df_pandas, 'feature', 'target')
    ic_values = result.set_index('time')['ic']
    assert ic_values['2024-01-01'] == pytest.approx(1.0)
    assert ic_values['2024-01-02'] == pytest.approx(-1.0)

def test_compute_ic_polars_shape(cross_section_df_polars):
    """Should return shape (2,2) because cross_section_df_polars has two dates."""
    result = compute_ic(cross_section_df_polars, 'feature', 'target')
    assert result.shape == (2, 2)

def test_compute_ic_polars_values(cross_section_df_polars):
    """Should return perfect positive correlation in first date and perfect inverse correlation in second date."""
    result = compute_ic(cross_section_df_polars, 'feature', 'target')
    result_sorted = result.sort('time')
    assert result_sorted['ic'][0] == pytest.approx(1.0)
    assert result_sorted['ic'][1] == pytest.approx(-1.0)

def test_compute_ic_pandas_polars_consistency(cross_section_df_pandas, cross_section_df_polars):
    """Backends should return numerically identical IC values."""
    res_pd = compute_ic(cross_section_df_pandas, 'feature', 'target').sort_values('time')
    res_pl = compute_ic(cross_section_df_polars, 'feature', 'target').sort('time')
    np.testing.assert_allclose(
        res_pd['ic'].values,
        res_pl['ic'].to_numpy(),
        rtol=1e-6
    )

def test_compute_ic_missing_column_raises(cross_section_df_pandas):
    """Should raise KeyError for missing columns."""
    with pytest.raises(KeyError):
        compute_ic(cross_section_df_pandas, 'nonexistent', 'target')

def test_compute_ic_invalid_df_raises():
    """Should raise ValueError with unsupported input types."""
    with pytest.raises(TypeError, match="Pandas or Polars"):
        compute_ic([[1, 2], [3, 4]], 'feature', 'target')

def test_compute_ic_custom_ic_column(cross_section_df_pandas):
    """Should return custom ic column name correctly."""
    result = compute_ic(cross_section_df_pandas, 'feature', 'target', ic_column='my_ic')
    assert 'my_ic' in result.columns


# ------------------------------------------------------
# ICMetrics and compute_ic_metrics
# ------------------------------------------------------
def test_ic_metrics_mean_preserves_sign(positive_ic_series, negative_ic_series):
    """Should preserve sign in mean."""
    assert compute_ic_metrics(positive_ic_series).mean > 0
    assert compute_ic_metrics(negative_ic_series).mean < 0

def test_ic_metrics_abs_mean_always_positive(negative_ic_series):
    """Should return positive abs_mean for negative IC series."""
    assert compute_ic_metrics(negative_ic_series).abs_mean > 0

def test_ic_metrics_sign_positive_series(positive_ic_series):
    """Should return sign=1 for positive IC series."""
    assert compute_ic_metrics(positive_ic_series).sign == 1

def test_ic_metrics_sign_negative_series(negative_ic_series):
    """Should return sign=-1 for negative IC series."""
    assert compute_ic_metrics(negative_ic_series).sign == -1

def test_ic_metrics_sign_zero_mean():
    """Should return sign=1 when mean IC is exactly zero."""
    series = pd.Series([0.1, -0.1, 0.1, -0.1])
    assert compute_ic_metrics(series).sign == 1

def test_ic_metrics_stability_is_nan_when_std_zero():
    """Should return nan stability when all IC values are identical."""
    series = pd.Series([0.05, 0.05, 0.05])
    assert np.isnan(compute_ic_metrics(series).stability)

def test_ic_metrics_pct_positive_on_adjusted_series(negative_ic_series):
    """pct_positive should be high for consistently negative IC (adjusted series)."""
    metrics = compute_ic_metrics(negative_ic_series)
    assert metrics.pct_positive == pytest.approx(1.0)

def test_ic_metrics_pct_positive_mixed(mixed_ic_series):
    """Should return 0.6 for mixed series."""
    assert compute_ic_metrics(mixed_ic_series).pct_positive == pytest.approx(0.6)

def test_ic_metrics_quantiles_ordered(negative_ic_series):
    """Should return q25 <= q50 <= q75 for negative IC (adjusted series)."""
    q = compute_ic_metrics(negative_ic_series).quantiles
    assert q['q25'] <= q['q50'] <= q['q75']

def test_ic_metrics_adjusted_series_positive_for_negative_input(negative_ic_series):
    """Adjusted series should be all positive for consistently negative IC."""
    metrics = compute_ic_metrics(negative_ic_series)
    assert (metrics.adjusted_series > 0).all()

def test_ic_metrics_original_series_unchanged(negative_ic_series):
    """Original series should stay negative for consistently negative IC."""
    metrics = compute_ic_metrics(negative_ic_series)
    assert (metrics.original_series < 0).all()

# Specific values tests as presented on fixtures
def test_ic_metrics_mean_value_on_positive_series(positive_ic_series):
    """Should compute mean correctly."""
    assert compute_ic_metrics(positive_ic_series).mean == pytest.approx(0.060)

def test_ic_metrics_mean_value_on_negative_series(negative_ic_series):
    """Should compute mean correctly."""
    assert compute_ic_metrics(negative_ic_series).mean == pytest.approx(-0.060)

def test_ic_metrics_abs_mean_is_mean_of_abs(mixed_ic_series):
    """abs_mean should be mean(abs(IC)), not abs(mean(IC))."""
    assert compute_ic_metrics(mixed_ic_series).abs_mean == pytest.approx(0.072)

def test_ic_metrics_abs_mean_differs_from_abs_of_mean(mixed_ic_series):
    """abs_mean and abs(mean) should differ for mixed series."""
    metrics = compute_ic_metrics(mixed_ic_series)
    assert metrics.abs_mean != pytest.approx(abs(metrics.mean))

def test_ic_metrics_std_value(positive_ic_series):
    """Should compute sample std correctly (ddof=1)."""
    assert compute_ic_metrics(positive_ic_series).std == pytest.approx(0.03162, rel=1e-3)

def test_ic_metrics_stability_formula(positive_ic_series):
    """Should compute stability metric correctly."""
    metrics = compute_ic_metrics(positive_ic_series)
    assert metrics.stability == pytest.approx(1.897, rel=1e-3)


# ------------------------------------------------------
# ic_summary_table
# ------------------------------------------------------
def test_ic_summary_table_shape(cross_section_df_pandas):
    """Should return one row per feature."""
    result = ic_summary_table(cross_section_df_pandas, ['feature'], 'target')
    assert result.shape[0] == 1

def test_ic_summary_table_columns(cross_section_df_pandas):
    """Should return expected columns."""
    result = ic_summary_table(cross_section_df_pandas, ['feature'], 'target')
    expected = {'feature', 'mean', 'abs_mean', 'sign', 'std', 'stability',
                'pct_positive', 'quantile25', 'quantile50', 'quantile75',
                't_stat', 'p_value', 'feature_group'}
    assert set(result.columns) == expected

def test_ic_summary_table_feature_group_default(cross_section_df_pandas):
    """Should label all features as ungrouped when feature_groups is None."""
    result = ic_summary_table(cross_section_df_pandas, ['feature'], 'target')
    assert result['feature_group'].iloc[0] == 'ungrouped'

def test_ic_summary_table_feature_group_assigned(cross_section_df_pandas):
    """Should assign correct group when feature_groups is provided."""
    result = ic_summary_table(
        cross_section_df_pandas, ['feature'], 'target',
        feature_groups={'feature': 'momentum'}
    )
    assert result['feature_group'].iloc[0] == 'momentum'

def test_ic_summary_table_feature_group_unknown(cross_section_df_pandas):
    """Should label as ungrouped if feature not in feature_groups."""
    result = ic_summary_table(
        cross_section_df_pandas, ['feature'], 'target',
        feature_groups={'other_feature': 'momentum'}
    )
    assert result['feature_group'].iloc[0] == 'ungrouped'

def test_ic_summary_table_empty_list_raises(cross_section_df_pandas):
    """Should raise ValueError for empty feature_list."""
    with pytest.raises(ValueError, match="empty"):
        ic_summary_table(cross_section_df_pandas, [], 'target')

def test_ic_summary_table_polars(cross_section_df_polars):
    """Should return pl.DataFrame for polars input."""
    result = ic_summary_table(cross_section_df_polars, ['feature'], 'target')
    assert isinstance(result, pl.DataFrame)

def test_ic_summary_table_pandas_polars_consistency(cross_section_df_pandas, cross_section_df_polars):
    """Should return numerically identical results for pandas and polars input."""
    res_pd = ic_summary_table(cross_section_df_pandas, ['feature'], 'target')
    res_pl = ic_summary_table(cross_section_df_polars, ['feature'], 'target')
    assert res_pd['mean'].iloc[0] == pytest.approx(res_pl['mean'][0], rel=1e-6)
    assert res_pd['t_stat'].iloc[0] == pytest.approx(res_pl['t_stat'][0], rel=1e-6)

# Multiple features tests
def test_ic_summary_table_multiple_features_shape(multi_feature_df_pandas):
    """Should return one row per feature: 3 features = 3 rows."""
    result = ic_summary_table(
        multi_feature_df_pandas,
        ['feature_a', 'feature_b', 'feature_c'],
        'target'
    )
    assert result.shape[0] == 3

def test_ic_summary_table_multiple_features_values(multi_feature_df_pandas):
    """Should compute correct mean IC per feature."""
    result = ic_summary_table(
        multi_feature_df_pandas,
        ['feature_a', 'feature_b', 'feature_c'],
        'target'
    )

    means = dict(zip(result["feature"], result["mean"]))

    assert means["feature_a"] == pytest.approx(1.0)
    assert means["feature_b"] == pytest.approx(-1.0)
    assert np.isnan(means["feature_c"])

def test_ic_summary_table_multiple_feature_groups(multi_feature_df_pandas):
    """Should assign correct groups to multiple features."""
    groups = {
        'feature_a': 'momentum',
        'feature_b': 'momentum',
        'feature_c': 'volatility',
    }
    result = ic_summary_table(
        multi_feature_df_pandas,
        ['feature_a', 'feature_b', 'feature_c'],
        'target',
        feature_groups=groups
    )

    feature_groups_result = dict(zip(result["feature"], result["feature_group"]))

    assert feature_groups_result["feature_a"] == "momentum"
    assert feature_groups_result["feature_b"] == "momentum"
    assert feature_groups_result["feature_c"] == "volatility"


def test_ic_summary_table_feature_order_preserved(multi_feature_df_pandas):
    """Should preserve feature order from input list."""
    features = ['feature_c', 'feature_a', 'feature_b']
    result = ic_summary_table(multi_feature_df_pandas, features, 'target')
    assert list(result['feature']) == features
