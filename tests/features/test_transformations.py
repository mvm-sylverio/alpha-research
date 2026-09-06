import pytest
import pandas as pd
import polars as pl
import numpy as np

from alpha_research.features.transformations import (
    _normalize_feature_cols,
    _validate_temporal_order_by_symbol,
    cross_sectional_quantile_rank,
    cross_sectional_rank,
    cross_sectional_zscore,
    rolling_quantile_rank,
    rolling_rank,
    rolling_zscore,
)

# ------------------------------------------------------
# fixtures
# ------------------------------------------------------
@pytest.fixture
def multi_asset_features_pandas():
    """
    2 assets, 3 dates, 1 feature.

    simple_ret_1 by date:
        2024-01-01: AAPL=0.10, MSFT=0.05  → AAPL rank=2, MSFT rank=1
        2024-01-02: AAPL=0.08, MSFT=0.12  → AAPL rank=1, MSFT rank=2
        2024-01-03: AAPL=0.15, MSFT=0.03  → AAPL rank=2, MSFT rank=1
    """
    return pd.DataFrame({
        'time': ['2024-01-01', '2024-01-01',
                 '2024-01-02', '2024-01-02',
                 '2024-01-03', '2024-01-03'],
        'symbol': ['AAPL', 'MSFT', 'AAPL', 'MSFT', 'AAPL', 'MSFT'],
        'simple_ret_1': [0.10, 0.05, 0.08, 0.12, 0.15, 0.03],
    })


@pytest.fixture
def temporal_multi_asset_features_pandas():
    """Two ordered assets with opposite four-observation feature paths."""
    return pd.DataFrame({
        'time': [
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
        ],
        'symbol': ['AAPL'] * 4 + ['MSFT'] * 4,
        'momentum': [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
    })


# ------------------------------------------------------
# cross_sectional_rank
# ------------------------------------------------------
# output structure
def test_rank_output_preserves_original_columns(multi_asset_features_pandas):
    """Should preserve all original columns and add rank column."""
    result = cross_sectional_rank(multi_asset_features_pandas, feature_cols='simple_ret_1')
    assert 'simple_ret_1' in result.columns
    assert 'simple_ret_1_cs_rank' in result.columns

def test_rank_output_shape(multi_asset_features_pandas):
    """Should return same number of rows as input."""
    result = cross_sectional_rank(multi_asset_features_pandas, feature_cols='simple_ret_1')
    assert len(result) == len(multi_asset_features_pandas)

def test_rank_single_str_input(multi_asset_features_pandas):
    """Should accept str input for single feature."""
    result = cross_sectional_rank(multi_asset_features_pandas, feature_cols='simple_ret_1')
    assert 'simple_ret_1_cs_rank' in result.columns

def test_rank_list_input_batch(multi_asset_features_pandas):
    """Should accept list of features and rank all features in list."""
    df = multi_asset_features_pandas.copy()
    df['log_ret_1'] = [0.09, 0.05, 0.08, 0.11, 0.14, 0.03]
    result = cross_sectional_rank(df, feature_cols=['simple_ret_1', 'log_ret_1'])
    assert 'simple_ret_1_cs_rank' in result.columns
    assert 'log_ret_1_cs_rank' in result.columns

# correctness
def test_rank_cross_sectional_values(multi_asset_features_pandas):
    """Rank should be correctly computed per date across assets."""
    result = cross_sectional_rank(multi_asset_features_pandas, feature_cols='simple_ret_1')
    result = result.sort_values(['time', 'symbol']).reset_index(drop=True)

    # 2024-01-01: AAPL=0.10 > MSFT=0.05 → AAPL rank=2, MSFT rank=1
    date1 = result[result['time'] == '2024-01-01']
    assert date1[date1['symbol'] == 'AAPL']['simple_ret_1_cs_rank'].values[0] == 2.0
    assert date1[date1['symbol'] == 'MSFT']['simple_ret_1_cs_rank'].values[0] == 1.0

    # 2024-01-02: MSFT=0.12 > AAPL=0.08 → MSFT rank=2, AAPL rank=1
    date2 = result[result['time'] == '2024-01-02']
    assert date2[date2['symbol'] == 'AAPL']['simple_ret_1_cs_rank'].values[0] == 1.0
    assert date2[date2['symbol'] == 'MSFT']['simple_ret_1_cs_rank'].values[0] == 2.0

    # 2024-01-03: AAPL=0.25 > MSFT=0.03 → AAPL rank=2, MSFT rank=1
    date2 = result[result['time'] == '2024-01-03']
    assert date2[date2['symbol'] == 'AAPL']['simple_ret_1_cs_rank'].values[0] == 2.0
    assert date2[date2['symbol'] == 'MSFT']['simple_ret_1_cs_rank'].values[0] == 1.0

def test_rank_does_not_mutate_input(multi_asset_features_pandas):
    """Should not modify the original DataFrame."""
    original = multi_asset_features_pandas.copy()
    cross_sectional_rank(multi_asset_features_pandas, feature_cols='simple_ret_1')
    pd.testing.assert_frame_equal(multi_asset_features_pandas, original)

# pandas / polars consistency
def test_rank_pandas_polars_consistency(multi_asset_features_pandas):
    """Should return identical results for pandas and polars input."""
    pl_df = pl.from_pandas(multi_asset_features_pandas)
    res_pd = cross_sectional_rank(multi_asset_features_pandas, feature_cols='simple_ret_1')
    res_pl = cross_sectional_rank(pl_df, feature_cols='simple_ret_1').to_pandas()
    res_pd = res_pd.sort_values(['time', 'symbol']).reset_index(drop=True)
    res_pl = res_pl.sort_values(['time', 'symbol']).reset_index(drop=True)
    np.testing.assert_allclose(
        res_pd['simple_ret_1_cs_rank'].values,
        res_pl['simple_ret_1_cs_rank'].values,
        rtol=1e-6
    )

# raises
def test_rank_empty_list_raises(multi_asset_features_pandas):
    """Should raise ValueError for empty feature_cols list."""
    with pytest.raises(ValueError, match="empty"):
        cross_sectional_rank(multi_asset_features_pandas, feature_cols=[])

def test_rank_invalid_feature_cols_type_raises(multi_asset_features_pandas):
    """Should raise TypeError for invalid feature_cols type."""
    with pytest.raises(TypeError):
        cross_sectional_rank(multi_asset_features_pandas, feature_cols=123)

def test_rank_missing_feature_column_raises(multi_asset_features_pandas):
    """Should raise KeyError for missing feature column."""
    with pytest.raises(KeyError):
        cross_sectional_rank(multi_asset_features_pandas, feature_cols='nonexistent')

def test_rank_invalid_df_type_raises(multi_asset_features_pandas):
    """Should raise TypeError for invalid df type."""
    with pytest.raises(TypeError):
        cross_sectional_rank([[1, 2, 3]], feature_cols='simple_ret_1')


# ------------------------------------------------------
# transformation helpers
# ------------------------------------------------------
def test_normalize_feature_cols_accepts_string_and_list():
    """Should normalize a single feature without altering a feature list."""
    assert _normalize_feature_cols('feature') == ['feature']
    assert _normalize_feature_cols(['feature_a', 'feature_b']) == ['feature_a', 'feature_b']


@pytest.mark.parametrize('feature_cols, exception', [
    ([], ValueError),
    (123, TypeError),
])
def test_normalize_feature_cols_rejects_invalid_input(feature_cols, exception):
    """Should validate feature-column input independently of public callers."""
    with pytest.raises(exception):
        _normalize_feature_cols(feature_cols)


@pytest.mark.parametrize('backend', [pd.DataFrame, pl.DataFrame])
def test_validate_temporal_order_by_symbol_accepts_ordered_asset_series(backend):
    """Should accept unique increasing times for every asset."""
    df = backend({
        'time': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02'],
        'symbol': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
    })
    assert _validate_temporal_order_by_symbol(df, 'symbol', 'time') is None


@pytest.mark.parametrize('time_values, error_match', [
    (['2024-01-02', '2024-01-01'], 'increasingly ordered'),
    (['2024-01-01', '2024-01-01'], 'unique values'),
    (['2024-01-01', None], 'missing values'),
])
def test_validate_temporal_order_by_symbol_rejects_invalid_times(time_values, error_match):
    """Should reject invalid time-series keys before rolling calculations."""
    df = pd.DataFrame({'time': time_values, 'symbol': ['AAPL', 'AAPL']})
    with pytest.raises(ValueError, match=error_match):
        _validate_temporal_order_by_symbol(df, 'symbol', 'time')


# ------------------------------------------------------
# cross_sectional_zscore
# ------------------------------------------------------
def test_cross_sectional_zscore_calculates_population_standardization(
        multi_asset_features_pandas,
):
    """Should standardize each two-asset date to z-scores of minus and plus one."""
    result = cross_sectional_zscore(multi_asset_features_pandas, 'simple_ret_1')

    assert 'simple_ret_1_cs_zscore' in result.columns
    np.testing.assert_allclose(
        result['simple_ret_1_cs_zscore'].to_numpy(),
        [1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
    )


def test_cross_sectional_zscore_returns_missing_for_constant_cross_section():
    """Should not assign an arbitrary z-score to a zero-dispersion date."""
    df = pd.DataFrame({
        'time': ['2024-01-01', '2024-01-01'],
        'symbol': ['AAPL', 'MSFT'],
        'feature': [2.0, 2.0],
    })
    result = cross_sectional_zscore(df, 'feature')
    assert result['feature_cs_zscore'].isna().all()


def test_cross_sectional_zscore_rejects_missing_feature_column(
        multi_asset_features_pandas,
):
    """Should retain the shared required-column validation contract."""
    with pytest.raises(KeyError, match='missing required columns'):
        cross_sectional_zscore(multi_asset_features_pandas, 'missing_feature')


# ------------------------------------------------------
# cross_sectional_quantile_ranks
# ------------------------------------------------------
def test_cross_sectional_quantile_rank_assigns_rank_based_groups(
        multi_asset_features_pandas,
):
    """Should label the lower and upper member of each pair as groups one and two."""
    result = cross_sectional_quantile_rank(
        multi_asset_features_pandas,
        'simple_ret_1',
        n_quantiles=2,
    )
    np.testing.assert_allclose(
        result['simple_ret_1_cs_quantile_rank_2'].to_numpy(),
        [2.0, 1.0, 1.0, 2.0, 2.0, 1.0],
    )


def test_cross_sectional_quantile_rank_preserves_ties_and_skips_small_sections():
    """Should not split tied values and should require enough values for all groups."""
    df = pd.DataFrame({
        'time': ['2024-01-01'] * 3 + ['2024-01-02'] * 2,
        'symbol': ['AAPL', 'MSFT', 'GOOG', 'AAPL', 'MSFT'],
        'feature': [1.0, 1.0, 3.0, 1.0, 2.0],
    })
    result = cross_sectional_quantile_rank(df, 'feature', n_quantiles=3)

    assert result.loc[:1, 'feature_cs_quantile_rank_3'].tolist() == [2.0, 2.0]
    assert result.loc[2, 'feature_cs_quantile_rank_3'] == 3.0
    assert result.loc[3:, 'feature_cs_quantile_rank_3'].isna().all()


@pytest.mark.parametrize('n_quantiles', [0, True, 1.5])
def test_cross_sectional_quantile_rank_validates_quantile_count(
        multi_asset_features_pandas,
        n_quantiles,
):
    """Should require a positive integer number of quantile groups."""
    with pytest.raises(ValueError, match='n_quantiles'):
        cross_sectional_quantile_rank(
            multi_asset_features_pandas,
            'simple_ret_1',
            n_quantiles=n_quantiles,
        )


# ------------------------------------------------------
# rolling_rank
# ------------------------------------------------------
def test_rolling_rank_calculates_each_asset_independently(
        temporal_multi_asset_features_pandas,
):
    """Should rank the current value only against the asset's trailing values."""
    result = rolling_rank(temporal_multi_asset_features_pandas, 'momentum', window=3)

    np.testing.assert_allclose(
        result['momentum_rolling_rank_3'].to_numpy(),
        [np.nan, np.nan, 3.0, 3.0, np.nan, np.nan, 1.0, 1.0],
        equal_nan=True,
    )


# ------------------------------------------------------
# rolling_zscore
# ------------------------------------------------------
def test_rolling_zscore_calculates_causal_population_standardization(
        temporal_multi_asset_features_pandas,
):
    """Should include the current observation and use only its trailing window."""
    result = rolling_zscore(temporal_multi_asset_features_pandas, 'momentum', window=3)
    expected_extreme = 1.224744871391589

    np.testing.assert_allclose(
        result['momentum_rolling_zscore_3'].to_numpy(),
        [
            np.nan, np.nan, expected_extreme, expected_extreme,
            np.nan, np.nan, -expected_extreme, -expected_extreme,
        ],
        equal_nan=True,
    )


def test_rolling_zscore_returns_missing_for_constant_window():
    """Should leave the z-score undefined when a full rolling window is constant."""
    df = pd.DataFrame({
        'time': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'symbol': ['AAPL'] * 3,
        'feature': [2.0, 2.0, 2.0],
    })
    result = rolling_zscore(df, 'feature', window=3)
    assert result['feature_rolling_zscore_3'].isna().all()


# ------------------------------------------------------
# rolling_quantile_rank
# ------------------------------------------------------
def test_rolling_quantile_rank_assigns_groups_from_trailing_rank(
        temporal_multi_asset_features_pandas,
):
    """Should assign the current observation to its rank-based trailing group."""
    result = rolling_quantile_rank(
        temporal_multi_asset_features_pandas,
        'momentum',
        window=3,
        n_quantiles=2,
    )

    np.testing.assert_allclose(
        result['momentum_rolling_quantile_rank_3_2'].to_numpy(),
        [np.nan, np.nan, 2.0, 2.0, np.nan, np.nan, 1.0, 1.0],
        equal_nan=True,
    )


def test_rolling_transformations_do_not_use_future_values(
        temporal_multi_asset_features_pandas,
):
    """Should leave an already-computed rolling value unchanged after future edits."""
    original = rolling_zscore(temporal_multi_asset_features_pandas, 'momentum', window=3)
    changed_future = temporal_multi_asset_features_pandas.copy()
    changed_future.loc[3, 'momentum'] = 10_000.0
    changed = rolling_zscore(changed_future, 'momentum', window=3)

    assert (
        original.loc[2, 'momentum_rolling_zscore_3']
        == changed.loc[2, 'momentum_rolling_zscore_3']
    )


@pytest.mark.parametrize('function, kwargs', [
    (rolling_rank, {'window': 0}),
    (rolling_zscore, {'window': True}),
    (rolling_quantile_rank, {'window': 2, 'n_quantiles': 3}),
])
def test_rolling_transformations_validate_window_arguments(
        temporal_multi_asset_features_pandas,
        function,
        kwargs,
):
    """Should reject invalid rolling-window and quantile configurations."""
    with pytest.raises(ValueError):
        function(temporal_multi_asset_features_pandas, 'momentum', **kwargs)


def test_rolling_transformations_reject_unordered_asset_time(
        temporal_multi_asset_features_pandas,
):
    """Should reject rows that would make the rolling calculation non-causal."""
    unordered = temporal_multi_asset_features_pandas.copy()
    unordered.loc[1, 'time'] = '2023-12-31'
    with pytest.raises(ValueError, match='increasingly ordered'):
        rolling_rank(unordered, 'momentum', window=3)


@pytest.mark.parametrize('function, kwargs, output_col', [
    (cross_sectional_zscore, {}, 'simple_ret_1_cs_zscore'),
    (cross_sectional_quantile_rank, {'n_quantiles': 2}, 'simple_ret_1_cs_quantile_rank_2'),
])
def test_cross_sectional_transformations_match_polars(
        multi_asset_features_pandas,
        function,
        kwargs,
        output_col,
):
    """Should produce equivalent cross-sectional values in both DataFrame backends."""
    pandas_result = function(multi_asset_features_pandas, 'simple_ret_1', **kwargs)
    polars_result = function(pl.from_pandas(multi_asset_features_pandas), 'simple_ret_1', **kwargs)

    np.testing.assert_allclose(
        pandas_result[output_col].to_numpy(),
        polars_result[output_col].to_numpy(),
        equal_nan=True,
    )


@pytest.mark.parametrize('function, kwargs, output_col', [
    (rolling_rank, {'window': 3}, 'momentum_rolling_rank_3'),
    (rolling_zscore, {'window': 3}, 'momentum_rolling_zscore_3'),
    (rolling_quantile_rank, {'window': 3, 'n_quantiles': 2}, 'momentum_rolling_quantile_rank_3_2'),
])
def test_rolling_transformations_match_polars(
        temporal_multi_asset_features_pandas,
        function,
        kwargs,
        output_col,
):
    """Should produce equivalent per-asset rolling values in both DataFrame backends."""
    pandas_result = function(temporal_multi_asset_features_pandas, 'momentum', **kwargs)
    polars_result = function(
        pl.from_pandas(temporal_multi_asset_features_pandas),
        'momentum',
        **kwargs,
    ).to_pandas()

    np.testing.assert_allclose(
        pandas_result[output_col].to_numpy(),
        polars_result[output_col].to_numpy(),
        equal_nan=True,
    )
