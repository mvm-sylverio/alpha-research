import pytest
import pandas as pd
import polars as pl
import numpy as np

from alpha_research.features.transformations import cross_sectional_rank

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


# ------------------------------------------------------
# cross_sectional_rank
# ------------------------------------------------------
# output structure
def test_rank_output_preserves_original_columns(multi_asset_features_pandas):
    """Should preserve all original columns and add rank column."""
    result = cross_sectional_rank(multi_asset_features_pandas, feature_cols='simple_ret_1')
    assert 'simple_ret_1' in result.columns
    assert 'simple_ret_1_rank' in result.columns

def test_rank_output_shape(multi_asset_features_pandas):
    """Should return same number of rows as input."""
    result = cross_sectional_rank(multi_asset_features_pandas, feature_cols='simple_ret_1')
    assert len(result) == len(multi_asset_features_pandas)

def test_rank_single_str_input(multi_asset_features_pandas):
    """Should accept str input for single feature."""
    result = cross_sectional_rank(multi_asset_features_pandas, feature_cols='simple_ret_1')
    assert 'simple_ret_1_rank' in result.columns

def test_rank_list_input_batch(multi_asset_features_pandas):
    """Should accept list of features and rank all features in list."""
    df = multi_asset_features_pandas.copy()
    df['log_ret_1'] = [0.09, 0.05, 0.08, 0.11, 0.14, 0.03]
    result = cross_sectional_rank(df, feature_cols=['simple_ret_1', 'log_ret_1'])
    assert 'simple_ret_1_rank' in result.columns
    assert 'log_ret_1_rank' in result.columns

# correctness
def test_rank_cross_sectional_values(multi_asset_features_pandas):
    """Rank should be correctly computed per date across assets."""
    result = cross_sectional_rank(multi_asset_features_pandas, feature_cols='simple_ret_1')
    result = result.sort_values(['time', 'symbol']).reset_index(drop=True)

    # 2024-01-01: AAPL=0.10 > MSFT=0.05 → AAPL rank=2, MSFT rank=1
    date1 = result[result['time'] == '2024-01-01']
    assert date1[date1['symbol'] == 'AAPL']['simple_ret_1_rank'].values[0] == 2.0
    assert date1[date1['symbol'] == 'MSFT']['simple_ret_1_rank'].values[0] == 1.0

    # 2024-01-02: MSFT=0.12 > AAPL=0.08 → MSFT rank=2, AAPL rank=1
    date2 = result[result['time'] == '2024-01-02']
    assert date2[date2['symbol'] == 'AAPL']['simple_ret_1_rank'].values[0] == 1.0
    assert date2[date2['symbol'] == 'MSFT']['simple_ret_1_rank'].values[0] == 2.0

    # 2024-01-03: AAPL=0.25 > MSFT=0.03 → AAPL rank=2, MSFT rank=1
    date2 = result[result['time'] == '2024-01-03']
    assert date2[date2['symbol'] == 'AAPL']['simple_ret_1_rank'].values[0] == 2.0
    assert date2[date2['symbol'] == 'MSFT']['simple_ret_1_rank'].values[0] == 1.0

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
        res_pd['simple_ret_1_rank'].values,
        res_pl['simple_ret_1_rank'].values,
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
