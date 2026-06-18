import pytest
import polars as pl
import numpy as np

from alpha_research.features.returns import simple_returns, log_returns


# ------------------------------------------------------
# simple_returns
# ------------------------------------------------------
# output structure
def test_compute_returns_output_columns(single_asset_ohlcv_pandas):
    """Should return only time, symbol and simple_ret column."""
    result = simple_returns(single_asset_ohlcv_pandas, horizon=1)
    assert set(result.columns) == {'time', 'symbol', 'simple_ret_1'}

def test_compute_returns_output_shape(single_asset_ohlcv_pandas):
    """Should return same number of rows as input."""
    result = simple_returns(single_asset_ohlcv_pandas, horizon=1)
    assert len(result) == len(single_asset_ohlcv_pandas)

# correctness - single asset
def test_compute_returns_values_horizon_1(single_asset_ohlcv_pandas):
    """Should compute correct simple returns for horizon=1."""
    result = simple_returns(single_asset_ohlcv_pandas, horizon=1)
    expected = [np.nan, 0.10, 0.10, 0.10, 0.10]
    assert np.isnan(result['simple_ret_1'].iloc[0])
    np.testing.assert_allclose(
        result['simple_ret_1'].values[1:],
        expected[1:],
        rtol=1e-4
    )

def test_compute_returns_values_horizon_2(single_asset_ohlcv_pandas):
    """Should compute correct simple returns for horizon=2."""
    result = simple_returns(single_asset_ohlcv_pandas, horizon=2)
    expected = [np.nan, np.nan, 0.21, 0.21, 0.21]
    assert np.isnan(result['simple_ret_2'].iloc[0]) and np.isnan(result['simple_ret_2'].iloc[1])
    np.testing.assert_allclose(
        result['simple_ret_2'].values[2:],
        expected[2:],
        rtol=1e-4
    )

# correctness - multi asset
def test_compute_returns_multi_asset_no_mixing(multi_asset_ohlcv_pandas):
    """groupby/over must not mix assets - each asset should be computed independently."""
    result = simple_returns(multi_asset_ohlcv_pandas, horizon=1)
    aapl = result[result['symbol'] == 'AAPL']['simple_ret_1'].values
    msft = result[result['symbol'] == 'MSFT']['simple_ret_1'].values
    assert np.isnan(aapl[0])
    assert np.isnan(msft[0])
    np.testing.assert_allclose(aapl[1:], [0.10, 0.10], rtol=1e-4)
    np.testing.assert_allclose(msft[1:], [0.05, 10/210], rtol=1e-4)

def test_compute_returns_multi_asset_horizon_2(multi_asset_ohlcv_pandas):
    """Horizon=2 should use price 2 bars before per asset independently."""
    result = simple_returns(multi_asset_ohlcv_pandas, horizon=2)
    aapl = result[result['symbol'] == 'AAPL']['simple_ret_2'].values
    msft = result[result['symbol'] == 'MSFT']['simple_ret_2'].values
    assert np.isnan(aapl[0]) and np.isnan(aapl[1])
    assert np.isnan(msft[0]) and np.isnan(msft[1])
    assert aapl[2] == pytest.approx(0.21, rel=1e-4)
    assert msft[2] == pytest.approx(0.10, rel=1e-4)

# pandas / polars consistency
def test_compute_returns_pandas_polars_consistency(single_asset_ohlcv_pandas):
    """Should return identical results for pandas and polars input."""
    pl_df = pl.from_pandas(single_asset_ohlcv_pandas)
    res_pd = simple_returns(single_asset_ohlcv_pandas, horizon=1)
    res_pl = simple_returns(pl_df, horizon=1).to_pandas()
    np.testing.assert_allclose(
        res_pd['simple_ret_1'].values,
        res_pl['simple_ret_1'].values,
        rtol=1e-6,
        equal_nan=True
    )

def test_compute_returns_pandas_polars_consistency_multi_asset(multi_asset_ohlcv_pandas):
    """Should return identical results for pandas and polars multi-asset input."""
    pl_df = pl.from_pandas(multi_asset_ohlcv_pandas)
    res_pd = simple_returns(multi_asset_ohlcv_pandas, horizon=1)
    res_pl = simple_returns(pl_df, horizon=1).to_pandas()
    np.testing.assert_allclose(
        res_pd['simple_ret_1'].values,
        res_pl['simple_ret_1'].values,
        rtol=1e-6,
        equal_nan=True
    )

# raises
def test_compute_returns_missing_column_raises(single_asset_ohlcv_pandas):
    """Should raise KeyError for missing required columns."""
    with pytest.raises(KeyError):
        simple_returns(single_asset_ohlcv_pandas.drop(columns=['close']), horizon=1)

def test_compute_returns_invalid_type_raises():
    """Should raise TypeError for unsupported input types."""
    with pytest.raises(TypeError):
        simple_returns([[1, 2, 3]], horizon=1)


# ------------------------------------------------------
# log_returns
# ------------------------------------------------------
# output structure
def test_compute_log_returns_output_columns(single_asset_ohlcv_pandas):
    """Should return only time, symbol and log_ret column."""
    result = log_returns(single_asset_ohlcv_pandas, horizon=1)
    assert set(result.columns) == {'time', 'symbol', 'log_ret_1'}

def test_compute_log_returns_output_shape(single_asset_ohlcv_pandas):
    """Should return same number of rows as input."""
    result = log_returns(single_asset_ohlcv_pandas, horizon=1)
    assert len(result) == len(single_asset_ohlcv_pandas)

# correctness - single asset
def test_compute_log_returns_values_horizon_1(single_asset_ohlcv_pandas):
    """Should compute correct log returns for horizon=1."""
    result = log_returns(single_asset_ohlcv_pandas, horizon=1)
    assert np.isnan(result['log_ret_1'].iloc[0])
    np.testing.assert_allclose(
        result['log_ret_1'].values[1:],
        [0.09531, 0.09531, 0.09531, 0.09531],
        rtol=1e-4
    )

def test_compute_log_returns_values_horizon_2(single_asset_ohlcv_pandas):
    """Should compute correct log returns for horizon=2."""
    result = log_returns(single_asset_ohlcv_pandas, horizon=2)
    assert np.isnan(result['log_ret_2'].iloc[0]) and np.isnan(result['log_ret_2'].iloc[1])
    np.testing.assert_allclose(
        result['log_ret_2'].values[2:],
        [0.19062, 0.19062, 0.19062],
        rtol=1e-4
    )

# correctness - multi asset
def test_compute_log_returns_multi_asset_no_mixing(multi_asset_ohlcv_pandas):
    """Groupby must not mix assets - each asset should be computed independently."""
    result = log_returns(multi_asset_ohlcv_pandas, horizon=1)
    aapl = result[result['symbol'] == 'AAPL']['log_ret_1'].values
    msft = result[result['symbol'] == 'MSFT']['log_ret_1'].values
    assert np.isnan(aapl[0])
    assert np.isnan(msft[0])
    np.testing.assert_allclose(aapl[1:], [0.09531, 0.09531], rtol=1e-4)
    np.testing.assert_allclose(msft[1:], [0.04879, 0.04652], rtol=1e-4)

def test_compute_log_returns_multi_asset_horizon_2(multi_asset_ohlcv_pandas):
    """Horizon=2 should use price 2 bars before per asset independently."""
    result = log_returns(multi_asset_ohlcv_pandas, horizon=2)
    aapl = result[result['symbol'] == 'AAPL']['log_ret_2'].values
    msft = result[result['symbol'] == 'MSFT']['log_ret_2'].values
    assert np.isnan(aapl[0]) and np.isnan(aapl[1])
    assert np.isnan(msft[0]) and np.isnan(msft[1])
    assert aapl[2] == pytest.approx(0.19062, rel=1e-4)
    assert msft[2] == pytest.approx(0.09531, rel=1e-4)

# log vs simple return relationship
def test_log_returns_less_than_simple_return(single_asset_ohlcv_pandas):
    """Log return should always be less than simple return for positive returns."""
    log_ret = log_returns(single_asset_ohlcv_pandas, horizon=1)
    simple_ret = simple_returns(single_asset_ohlcv_pandas, horizon=1)
    assert (log_ret['log_ret_1'].dropna() < simple_ret['simple_ret_1'].dropna()).all()

# pandas / polars consistency
def test_compute_log_returns_pandas_polars_consistency(single_asset_ohlcv_pandas):
    """Should return identical results for pandas and polars input."""
    pl_df = pl.from_pandas(single_asset_ohlcv_pandas)
    res_pd = log_returns(single_asset_ohlcv_pandas, horizon=1)
    res_pl = log_returns(pl_df, horizon=1).to_pandas()
    np.testing.assert_allclose(
        res_pd['log_ret_1'].values[1:],
        res_pl['log_ret_1'].values[1:],
        rtol=1e-6
    )

def test_compute_log_returns_pandas_polars_consistency_multi_asset(multi_asset_ohlcv_pandas):
    """Should return identical results for pandas and polars multi-asset input."""
    pl_df = pl.from_pandas(multi_asset_ohlcv_pandas)
    res_pd = log_returns(multi_asset_ohlcv_pandas, horizon=1)
    res_pl = log_returns(pl_df, horizon=1).to_pandas()
    np.testing.assert_allclose(
        res_pd['log_ret_1'].values,
        res_pl['log_ret_1'].values,
        rtol=1e-6,
        equal_nan=True
    )

# raises
def test_compute_log_returns_missing_column_raises(single_asset_ohlcv_pandas):
    """Should raise KeyError for missing required columns."""
    with pytest.raises(KeyError):
        log_returns(
            single_asset_ohlcv_pandas.drop(columns=['close']), horizon=1
        )

def test_compute_log_returns_invalid_type_raises():
    """Should raise TypeError for unsupported input types."""
    with pytest.raises(TypeError):
        log_returns([[1, 2, 3]], horizon=1)