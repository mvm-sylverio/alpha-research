import pytest
import numpy as np
import pandas as pd
import polars as pl
from alpha_research.evaluation.ic import (
    compute_ic,
    compute_ic_metrics,
    _generate_target_frames,
    _ic_decay_from_target_frames,
    ic_decay,
    ic_decay_summary,
    ic_decay_summary_table,
    ic_summary_table,
    information_coefficient,
)


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


@pytest.fixture
def decay_features_df_pandas():
    return pd.DataFrame({
        'time': ['2024-01-01'] * 5 + ['2024-01-02'] * 5 + ['2024-01-03'] * 5,
        'symbol': ['A', 'B', 'C', 'D', 'E'] * 3,
        'feature_a': [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        'feature_b': [5.0, 4.0, 3.0, 2.0, 1.0] * 3,
    })


@pytest.fixture
def decay_target_data_pandas(decay_features_df_pandas):
    return pd.DataFrame({
        'time': decay_features_df_pandas['time'],
        'symbol': decay_features_df_pandas['symbol'],
        'target_1': [
            1.0, 2.0, 3.0, 4.0, 5.0,
            1.0, 2.0, 3.0, 5.0, 4.0,
            5.0, 4.0, 3.0, 2.0, 1.0,
        ],
        'target_2': [
            5.0, 4.0, 3.0, 2.0, 1.0,
            4.0, 5.0, 3.0, 2.0, 1.0,
            1.0, 2.0, 3.0, 4.0, 5.0,
        ],
    })


@pytest.fixture
def decay_features_df_polars(decay_features_df_pandas):
    return pl.from_pandas(decay_features_df_pandas)


@pytest.fixture
def decay_target_data_polars(decay_target_data_pandas):
    return pl.from_pandas(decay_target_data_pandas)


@pytest.fixture
def decay_summary_curve_pandas():
    return pd.DataFrame({
        'horizon': [4, 1, 3, 2],
        'mean': [0.04, 0.10, 0.08, 0.20],
        'fdr_rejected': [False, False, True, True],
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
# ic_summary_table.table
def test_ic_summary_table_shape(cross_section_df_pandas):
    """table should return one row per feature."""
    result = ic_summary_table(cross_section_df_pandas, ['feature'], 'target').table
    assert result.shape[0] == 1

def test_ic_summary_table_columns(cross_section_df_pandas):
    """table should return expected columns."""
    result = ic_summary_table(cross_section_df_pandas, ['feature'], 'target').table
    expected = {'feature', 'mean', 'abs_mean', 'sign', 'std', 'stability',
                'pct_positive', 'quantile25', 'quantile50', 'quantile75',
                't_stat', 'p_value', 'feature_group', 'n_obs'}
    assert set(result.columns) == expected


def test_ic_summary_table_n_obs_matches_number_of_ic_observations(
        cross_section_df_pandas,
):
    """n_obs should equal the number of cross-sectional IC observations."""
    result = ic_summary_table(
        cross_section_df_pandas,
        ['feature'],
        'target',
    )

    assert result.table['n_obs'].iloc[0] == 2
    assert result.table['n_obs'].iloc[0] == len(result.ic_frames['feature'])


def test_ic_summary_table_feature_group_default(cross_section_df_pandas):
    """table should label all features as ungrouped when feature_groups is None."""
    result = ic_summary_table(cross_section_df_pandas, ['feature'], 'target').table
    assert result['feature_group'].iloc[0] == 'ungrouped'

def test_ic_summary_table_feature_group_assigned(cross_section_df_pandas):
    """table should assign correct group when feature_groups is provided."""
    result = ic_summary_table(
        cross_section_df_pandas, ['feature'], 'target',
        feature_groups={'feature': 'momentum'}
    ).table
    assert result['feature_group'].iloc[0] == 'momentum'

def test_ic_summary_table_feature_group_unknown(cross_section_df_pandas):
    """table should label as ungrouped if feature not in feature_groups."""
    result = ic_summary_table(
        cross_section_df_pandas, ['feature'], 'target',
        feature_groups={'other_feature': 'momentum'}
    ).table
    assert result['feature_group'].iloc[0] == 'ungrouped'

def test_ic_summary_table_empty_list_raises(cross_section_df_pandas):
    """table should raise ValueError for empty feature_list."""
    with pytest.raises(ValueError, match="empty"):
        ic_summary_table(cross_section_df_pandas, [], 'target').table

def test_ic_summary_table_polars(cross_section_df_polars):
    """table should return pl.DataFrame for polars input."""
    result = ic_summary_table(cross_section_df_polars, ['feature'], 'target').table
    assert isinstance(result, pl.DataFrame)

def test_ic_summary_table_pandas_polars_consistency(cross_section_df_pandas, cross_section_df_polars):
    """table should return numerically identical results for pandas and polars input."""
    res_pd = ic_summary_table(cross_section_df_pandas, ['feature'], 'target').table
    res_pl = ic_summary_table(cross_section_df_polars, ['feature'], 'target').table
    assert res_pd['mean'].iloc[0] == pytest.approx(res_pl['mean'][0], rel=1e-6)
    assert res_pd['t_stat'].iloc[0] == pytest.approx(res_pl['t_stat'][0], rel=1e-6)

# Multiple features tests
def test_ic_summary_table_multiple_features_shape(multi_feature_df_pandas):
    """table should return one row per feature: 3 features = 3 rows."""
    result = ic_summary_table(
        multi_feature_df_pandas,
        ['feature_a', 'feature_b', 'feature_c'],
        'target'
    ).table
    assert result.shape[0] == 3

def test_ic_summary_table_multiple_features_values(multi_feature_df_pandas):
    """table should compute correct mean IC per feature."""
    result = ic_summary_table(
        multi_feature_df_pandas,
        ['feature_a', 'feature_b', 'feature_c'],
        'target'
    ).table

    means = dict(zip(result["feature"], result["mean"]))

    assert means["feature_a"] == pytest.approx(1.0)
    assert means["feature_b"] == pytest.approx(-1.0)
    assert np.isnan(means["feature_c"])

def test_ic_summary_table_multiple_feature_groups(multi_feature_df_pandas):
    """table should assign correct groups to multiple features."""
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
    ).table

    feature_groups_result = dict(zip(result["feature"], result["feature_group"]))

    assert feature_groups_result["feature_a"] == "momentum"
    assert feature_groups_result["feature_b"] == "momentum"
    assert feature_groups_result["feature_c"] == "volatility"


def test_ic_summary_table_feature_order_preserved(multi_feature_df_pandas):
    """table should preserve feature order from input list."""
    features = ['feature_c', 'feature_a', 'feature_b']
    result = ic_summary_table(multi_feature_df_pandas, features, 'target').table
    assert list(result['feature']) == features


# ic_summary_table.ic_frames
def test_ic_summary_table_ic_frames_keys(cross_section_df_pandas):
    """ic_frames should have one key per feature in feature_list."""
    result = ic_summary_table(cross_section_df_pandas, ['feature'], 'target')
    assert set(result.ic_frames.keys()) == {'feature'}

def test_ic_summary_table_ic_frames_columns(cross_section_df_pandas):
    """Each ic_frames DataFrame should have time and ic columns."""
    result = ic_summary_table(cross_section_df_pandas, ['feature'], 'target')
    assert 'time' in result.ic_frames['feature'].columns
    assert 'ic' in result.ic_frames['feature'].columns

def test_ic_summary_table_ic_frames_multiple_features(multi_feature_df_pandas):
    """ic_frames should have one key per feature."""
    features = ['feature_a', 'feature_b', 'feature_c']
    result = ic_summary_table(multi_feature_df_pandas, features, 'target')
    assert set(result.ic_frames.keys()) == set(features)

def test_ic_summary_table_ic_frames_mean_consistent_with_table(cross_section_df_pandas):
    """Mean of ic_frames series should match mean in summary table."""
    result = ic_summary_table(cross_section_df_pandas, ['feature'], 'target')
    ic_mean_from_series = result.ic_frames['feature']['ic'].mean()
    ic_mean_from_table = result.table['mean'].iloc[0]
    assert ic_mean_from_series == pytest.approx(ic_mean_from_table, rel=1e-6)

def test_ic_summary_table_ic_frames_polars(cross_section_df_polars):
    """ic_frames should contain pl.DataFrame for polars input."""
    result = ic_summary_table(cross_section_df_polars, ['feature'], 'target')
    assert isinstance(result.ic_frames['feature'], pl.DataFrame)


# ------------------------------------------------------
# _generate_target_frames
# ------------------------------------------------------
def test_generate_target_frames_calls_target_fn_once_per_sorted_horizon(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should generate one target DataFrame for each sorted horizon."""
    calls = []

    def target_fn(target_data, horizon):
        calls.append(horizon)
        return target_data[['time', 'symbol', f'target_{horizon}']]

    result = _generate_target_frames(
        df_feature=decay_features_df_pandas,
        target_data=decay_target_data_pandas,
        horizons=[2, 1],
        target_fn=target_fn,
    )

    assert calls == [1, 2]
    assert list(result) == [1, 2]
    assert list(result[1].columns) == ['time', 'symbol', 'target_1']
    assert list(result[2].columns) == ['time', 'symbol', 'target_2']


def test_generate_target_frames_rejects_duplicate_horizons(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject duplicate horizons before calling target_fn."""
    calls = []

    def target_fn(target_data, horizon):
        calls.append(horizon)
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='horizons must not contain duplicates'):
        _generate_target_frames(
            df_feature=decay_features_df_pandas,
            target_data=decay_target_data_pandas,
            horizons=[1, 1],
            target_fn=target_fn,
        )

    assert calls == []


def test_generate_target_frames_rejects_empty_horizons(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject an empty horizon list before calling target_fn."""
    calls = []

    def target_fn(target_data, horizon):
        calls.append(horizon)
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='horizons must not be empty'):
        _generate_target_frames(
            df_feature=decay_features_df_pandas,
            target_data=decay_target_data_pandas,
            horizons=[],
            target_fn=target_fn,
        )

    assert calls == []


def test_generate_target_frames_rejects_none_horizon(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject None as a horizon value."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='positive integers'):
        _generate_target_frames(
            df_feature=decay_features_df_pandas,
            target_data=decay_target_data_pandas,
            horizons=[None],
            target_fn=target_fn,
        )


def test_generate_target_frames_rejects_nan_horizon(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject NaN as a horizon value."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='positive integers'):
        _generate_target_frames(
            df_feature=decay_features_df_pandas,
            target_data=decay_target_data_pandas,
            horizons=[np.nan],
            target_fn=target_fn,
        )


def test_generate_target_frames_rejects_float_horizon(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject a floating-point horizon value."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='positive integers'):
        _generate_target_frames(
            df_feature=decay_features_df_pandas,
            target_data=decay_target_data_pandas,
            horizons=[1.0],
            target_fn=target_fn,
        )


def test_generate_target_frames_rejects_string_horizon(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject a string horizon value."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='positive integers'):
        _generate_target_frames(
            df_feature=decay_features_df_pandas,
            target_data=decay_target_data_pandas,
            horizons=['1'],
            target_fn=target_fn,
        )


def test_generate_target_frames_rejects_boolean_horizon(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject a boolean horizon value."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='positive integers'):
        _generate_target_frames(
            df_feature=decay_features_df_pandas,
            target_data=decay_target_data_pandas,
            horizons=[True],
            target_fn=target_fn,
        )


def test_generate_target_frames_rejects_non_positive_horizons(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject zero and negative horizon values."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='positive integers'):
        _generate_target_frames(
            df_feature=decay_features_df_pandas,
            target_data=decay_target_data_pandas,
            horizons=[0],
            target_fn=target_fn,
        )

    with pytest.raises(ValueError, match='positive integers'):
        _generate_target_frames(
            df_feature=decay_features_df_pandas,
            target_data=decay_target_data_pandas,
            horizons=[-1],
            target_fn=target_fn,
        )


def test_generate_target_frames_pandas_polars_consistency(
        decay_features_df_pandas,
        decay_target_data_pandas,
        decay_features_df_polars,
        decay_target_data_polars,
):
    """Should generate equivalent target mappings for both backends."""
    def pandas_target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    def polars_target_fn(target_data, horizon):
        return target_data.select(['time', 'symbol', f'target_{horizon}'])

    pandas_result = _generate_target_frames(
        df_feature=decay_features_df_pandas,
        target_data=decay_target_data_pandas,
        horizons=[2, 1],
        target_fn=pandas_target_fn,
    )
    polars_result = _generate_target_frames(
        df_feature=decay_features_df_polars,
        target_data=decay_target_data_polars,
        horizons=[2, 1],
        target_fn=polars_target_fn,
    )

    assert list(pandas_result) == list(polars_result) == [1, 2]

    for horizon in pandas_result:
        pd.testing.assert_frame_equal(
            pandas_result[horizon].reset_index(drop=True),
            polars_result[horizon].to_pandas().reset_index(drop=True),
            check_dtype=False,
        )


# ------------------------------------------------------
# _ic_decay_from_target_frames
# ------------------------------------------------------
def test_ic_decay_from_target_frames_uses_pre_generated_targets(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should compute decay directly from the supplied target mapping."""
    target_frames = {
        1: decay_target_data_pandas[['time', 'symbol', 'target_1']],
        2: decay_target_data_pandas[['time', 'symbol', 'target_2']],
    }

    result = _ic_decay_from_target_frames(
        df_feature=decay_features_df_pandas,
        feature='feature_a',
        target_frames=target_frames,
        corr_method='spearman',
        date_column='time',
        symbol_column='symbol',
        feature_groups={'feature_a': 'momentum'},
        fdr=0.05,
        fdr_method='bh',
    )

    assert list(result.table['horizon']) == [1, 2]
    assert list(result.table['feature_group']) == ['momentum', 'momentum']
    assert set(result.ic_frames) == {1, 2}


def test_ic_decay_from_target_frames_sorts_horizons_in_result_table(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should sort the result table even when target frames are unordered."""
    target_frames = {
        2: decay_target_data_pandas[['time', 'symbol', 'target_2']],
        1: decay_target_data_pandas[['time', 'symbol', 'target_1']],
    }

    result = _ic_decay_from_target_frames(
        df_feature=decay_features_df_pandas,
        feature='feature_a',
        target_frames=target_frames,
        corr_method='spearman',
        date_column='time',
        symbol_column='symbol',
        feature_groups=None,
        fdr=0.05,
        fdr_method='bh',
    )

    assert list(result.table['horizon']) == [1, 2]


def test_ic_decay_from_target_frames_pandas_polars_consistency(
        decay_features_df_pandas,
        decay_target_data_pandas,
        decay_features_df_polars,
        decay_target_data_polars,
):
    """Should return equivalent decay results for Pandas and Polars inputs."""
    pandas_target_frames = {
        1: decay_target_data_pandas[['time', 'symbol', 'target_1']],
        2: decay_target_data_pandas[['time', 'symbol', 'target_2']],
    }
    polars_target_frames = {
        1: decay_target_data_polars.select(['time', 'symbol', 'target_1']),
        2: decay_target_data_polars.select(['time', 'symbol', 'target_2']),
    }

    pandas_result = _ic_decay_from_target_frames(
        df_feature=decay_features_df_pandas,
        feature='feature_a',
        target_frames=pandas_target_frames,
        corr_method='spearman',
        date_column='time',
        symbol_column='symbol',
        feature_groups=None,
        fdr=0.05,
        fdr_method='bh',
    )
    polars_result = _ic_decay_from_target_frames(
        df_feature=decay_features_df_polars,
        feature='feature_a',
        target_frames=polars_target_frames,
        corr_method='spearman',
        date_column='time',
        symbol_column='symbol',
        feature_groups=None,
        fdr=0.05,
        fdr_method='bh',
    )

    pd.testing.assert_frame_equal(
        pandas_result.table.reset_index(drop=True),
        polars_result.table.to_pandas().reset_index(drop=True),
        check_dtype=False,
    )
    for horizon in pandas_result.ic_frames:
        pd.testing.assert_frame_equal(
            pandas_result.ic_frames[horizon].reset_index(drop=True),
            polars_result.ic_frames[horizon].to_pandas().reset_index(drop=True),
            check_dtype=False,
        )


# ------------------------------------------------------
# ic_decay
# ------------------------------------------------------
def test_ic_decay_preserves_polars_backend(
        decay_features_df_polars,
        decay_target_data_polars,
):
    """Should return Polars results when features and targets use Polars."""
    def target_fn(target_data, horizon):
        return target_data.select(['time', 'symbol', f'target_{horizon}'])

    result = ic_decay(
        df_feature=decay_features_df_polars,
        feature='feature_a',
        target_data=decay_target_data_polars,
        horizons=[1, 2],
        target_fn=target_fn,
    )

    assert isinstance(result.table, pl.DataFrame)
    assert all(isinstance(frame, pl.DataFrame) for frame in result.ic_frames.values())


def test_ic_decay_pandas_polars_consistency(
        decay_features_df_pandas,
        decay_target_data_pandas,
        decay_features_df_polars,
        decay_target_data_polars,
):
    """Should return equivalent decay results for Pandas and Polars inputs."""
    def pandas_target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    def polars_target_fn(target_data, horizon):
        return target_data.select(['time', 'symbol', f'target_{horizon}'])

    pandas_result = ic_decay(
        df_feature=decay_features_df_pandas,
        feature='feature_a',
        target_data=decay_target_data_pandas,
        horizons=[1, 2],
        target_fn=pandas_target_fn,
    )
    polars_result = ic_decay(
        df_feature=decay_features_df_polars,
        feature='feature_a',
        target_data=decay_target_data_polars,
        horizons=[1, 2],
        target_fn=polars_target_fn,
    )

    pd.testing.assert_frame_equal(
        pandas_result.table.reset_index(drop=True),
        polars_result.table.to_pandas().reset_index(drop=True),
        check_dtype=False,
    )
    for horizon in pandas_result.ic_frames:
        pd.testing.assert_frame_equal(
            pandas_result.ic_frames[horizon].reset_index(drop=True),
            polars_result.ic_frames[horizon].to_pandas().reset_index(drop=True),
            check_dtype=False,
        )


# ------------------------------------------------------
# ic_decay_summary
# ------------------------------------------------------
def test_ic_decay_summary_identifies_peak_halflife_and_significance(
        decay_summary_curve_pandas,
):
    """Should derive diagnostics from a curve sorted by horizon."""
    result = ic_decay_summary(decay_summary_curve_pandas)

    assert result.peak_horizon == 2
    assert result.peak_abs_ic == pytest.approx(0.20)
    assert result.halflife_horizon == 3
    assert result.last_significant_horizon == 3


def test_ic_decay_summary_returns_none_when_no_horizon_is_significant(
        decay_summary_curve_pandas,
):
    """Should return None when no horizon passes FDR correction."""
    decay_curve = decay_summary_curve_pandas.assign(fdr_rejected=False)

    result = ic_decay_summary(decay_curve)

    assert result.last_significant_horizon is None


def test_ic_decay_summary_returns_none_when_signal_never_reaches_half_peak():
    """Should return None when no post-peak value reaches half the peak."""
    decay_curve = pd.DataFrame({
        'horizon': [1, 2, 3],
        'mean': [0.20, 0.16, 0.11],
        'fdr_rejected': [True, True, False],
    })

    result = ic_decay_summary(decay_curve)

    assert result.peak_horizon == 1
    assert result.halflife_horizon is None


def test_ic_decay_summary_uses_first_peak_when_peak_values_are_equal():
    """Should use the earliest horizon when absolute peak IC values tie."""
    decay_curve = pd.DataFrame({
        'horizon': [1, 2, 3],
        'mean': [0.10, -0.10, 0.00],
        'fdr_rejected': [True, True, False],
    })

    result = ic_decay_summary(decay_curve)

    assert result.peak_horizon == 1
    assert result.peak_abs_ic == pytest.approx(0.10)
    assert result.halflife_horizon == 3


def test_ic_decay_summary_calculates_trapezoidal_auc():
    """Should calculate AUC using the horizon values as trapezoid spacing."""
    decay_curve = pd.DataFrame({
        'horizon': [1, 2, 4],
        'mean': [0.40, -0.20, 0.10],
        'fdr_rejected': [True, True, False],
    })

    result = ic_decay_summary(decay_curve)

    assert result.auc == pytest.approx(0.60)


def test_ic_decay_summary_rejects_empty_curve():
    """Should reject an empty decay curve."""
    decay_curve = pd.DataFrame({
        'horizon': [],
        'mean': [],
        'fdr_rejected': [],
    })

    with pytest.raises(ValueError, match='must not be empty'):
        ic_decay_summary(decay_curve)


def test_ic_decay_summary_rejects_duplicate_horizons():
    """Should require exactly one row for each horizon."""
    decay_curve = pd.DataFrame({
        'horizon': [1, 1],
        'mean': [0.10, 0.05],
        'fdr_rejected': [True, False],
    })

    with pytest.raises(ValueError, match='exactly one row per horizon'):
        ic_decay_summary(decay_curve)


def test_ic_decay_summary_rejects_curve_without_finite_mean_ic_values():
    """Should reject a curve that contains no usable mean IC values."""
    decay_curve = pd.DataFrame({
        'horizon': [1, 2, 3],
        'mean': [np.nan, np.inf, -np.inf],
        'fdr_rejected': [False, False, False],
    })

    with pytest.raises(ValueError, match='no finite mean IC values'):
        ic_decay_summary(decay_curve)


# ------------------------------------------------------
# ic_decay_summary_table
# ------------------------------------------------------
def test_ic_decay_summary_table_reuses_targets_for_all_features(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should generate each target once and reuse it for every feature."""
    calls = []

    def target_fn(target_data, horizon):
        calls.append(horizon)
        return target_data[['time', 'symbol', f'target_{horizon}']]

    result = ic_decay_summary_table(
        df_features=decay_features_df_pandas,
        feature_list=['feature_a', 'feature_b'],
        target_data=decay_target_data_pandas,
        horizons=[1, 2],
        target_fn=target_fn,
    )

    assert calls == [1, 2]


def test_ic_decay_summary_table_rejects_empty_feature_list(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject an empty feature list."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='feature_list must not be empty'):
        ic_decay_summary_table(
            df_features=decay_features_df_pandas,
            feature_list=[],
            target_data=decay_target_data_pandas,
            horizons=[1, 2],
            target_fn=target_fn,
        )


def test_ic_decay_summary_table_returns_expected_columns_and_feature_groups(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should return one summary row per feature with its assigned group."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    result = ic_decay_summary_table(
        df_features=decay_features_df_pandas,
        feature_list=['feature_a', 'feature_b'],
        target_data=decay_target_data_pandas,
        horizons=[1, 2],
        target_fn=target_fn,
        feature_groups={
            'feature_a': 'momentum',
            'feature_b': 'reversal',
        },
    )

    expected_columns = {
        'feature',
        'peak_horizon',
        'peak_abs_ic',
        'halflife_horizon',
        'last_significant_horizon',
        'auc',
        'feature_group',
    }

    assert set(result.table.columns) == expected_columns
    assert list(result.table['feature']) == ['feature_a', 'feature_b']
    assert list(result.table['feature_group']) == ['momentum', 'reversal']


def test_ic_decay_summary_table_preserves_individual_decay_results(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Stored decay results should match individual ic_decay computations."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    summary_result = ic_decay_summary_table(
        df_features=decay_features_df_pandas,
        feature_list=['feature_a', 'feature_b'],
        target_data=decay_target_data_pandas,
        horizons=[1, 2],
        target_fn=target_fn,
    )

    for feature in ['feature_a', 'feature_b']:
        expected_result = ic_decay(
            df_feature=decay_features_df_pandas,
            feature=feature,
            target_data=decay_target_data_pandas,
            horizons=[1, 2],
            target_fn=target_fn,
        )
        actual_result = summary_result.decay_results[feature]

        pd.testing.assert_frame_equal(
            actual_result.table,
            expected_result.table,
        )
        for horizon in expected_result.ic_frames:
            pd.testing.assert_frame_equal(
                actual_result.ic_frames[horizon],
                expected_result.ic_frames[horizon],
            )


def test_ic_decay_summary_table_pandas_polars_consistency(
        decay_features_df_pandas,
        decay_target_data_pandas,
        decay_features_df_polars,
        decay_target_data_polars,
):
    """Should return equivalent summary tables for Pandas and Polars inputs."""
    def pandas_target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    def polars_target_fn(target_data, horizon):
        return target_data.select(['time', 'symbol', f'target_{horizon}'])

    pandas_result = ic_decay_summary_table(
        df_features=decay_features_df_pandas,
        feature_list=['feature_a', 'feature_b'],
        target_data=decay_target_data_pandas,
        horizons=[1, 2],
        target_fn=pandas_target_fn,
    )
    polars_result = ic_decay_summary_table(
        df_features=decay_features_df_polars,
        feature_list=['feature_a', 'feature_b'],
        target_data=decay_target_data_polars,
        horizons=[1, 2],
        target_fn=polars_target_fn,
    )

    pd.testing.assert_frame_equal(
        pandas_result.table.reset_index(drop=True),
        polars_result.table.to_pandas().reset_index(drop=True),
        check_dtype=False,
    )
    assert set(pandas_result.decay_results) == set(polars_result.decay_results)


def test_ic_decay_summary_table_duplicate_feature_list_raises(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject duplicate feature names."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='feature_list must not contain duplicates'):
        ic_decay_summary_table(
            df_features=decay_features_df_pandas,
            feature_list=['feature_a', 'feature_a'],
            target_data=decay_target_data_pandas,
            horizons=[1, 2],
            target_fn=target_fn,
        )
