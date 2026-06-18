import pytest
import numpy as np
import polars as pl

from alpha_research.features.targets import fwd_returns
from alpha_research.features.returns import simple_returns

# ------------------------------------------------------
# fwd_returns
# ------------------------------------------------------
# output structure
def test_fwd_return_output_columns(single_asset_ohlcv_pandas):
    """Should return only time, symbol and fwd_ret column."""
    result = fwd_returns(single_asset_ohlcv_pandas, horizon=1)
    assert set(result.columns) == {'time', 'symbol', 'fwd_ret_1'}

def test_fwd_return_output_shape(single_asset_ohlcv_pandas):
    """Should return same number of rows as input."""
    result = fwd_returns(single_asset_ohlcv_pandas, horizon=1)
    assert len(result) == len(single_asset_ohlcv_pandas)

# correctness - single asset
def test_fwd_return_values_horizon_1(single_asset_ohlcv_pandas):
    """Should compute correct forward returns for horizon=1."""
    result = fwd_returns(single_asset_ohlcv_pandas, horizon=1)
    assert np.isnan(result['fwd_ret_1'].iloc[-1])
    np.testing.assert_allclose(
        result['fwd_ret_1'].values[:-1],
        [0.10, 0.10, 0.10, 0.10],
        rtol=1e-6
    )

def test_fwd_return_values_horizon_2(single_asset_ohlcv_pandas):
    """Should compute correct forward returns for horizon=2."""
    result = fwd_returns(single_asset_ohlcv_pandas, horizon=2)
    assert np.isnan(result['fwd_ret_2'].iloc[-1])
    assert np.isnan(result['fwd_ret_2'].iloc[-2])
    np.testing.assert_allclose(
        result['fwd_ret_2'].values[:-2],
        [0.21, 0.21, 0.21],
        rtol=1e-6
    )

# correctness - multi asset
def test_fwd_return_multi_asset_no_mixing(multi_asset_ohlcv_pandas):
    """Groupby must not mix assets — last row per asset should be NaN."""
    result = fwd_returns(multi_asset_ohlcv_pandas, horizon=1)
    aapl = result[result['symbol'] == 'AAPL']['fwd_ret_1'].values
    msft = result[result['symbol'] == 'MSFT']['fwd_ret_1'].values
    np.testing.assert_allclose(aapl[:-1], [0.10, 0.10], rtol=1e-6)
    np.testing.assert_allclose(msft[:-1], [0.05, 0.047619], rtol=1e-4)
    assert np.isnan(aapl[-1])
    assert np.isnan(msft[-1])

# relationship with simple_return
def test_fwd_return_equals_future_simple_return(single_asset_ohlcv_pandas):
    """fwd_ret_1 at t should equal simple_ret_1 at t+1."""
    fwd = fwd_returns(single_asset_ohlcv_pandas, horizon=1)
    ret = simple_returns(single_asset_ohlcv_pandas, horizon=1)
    np.testing.assert_allclose(
        fwd['fwd_ret_1'].values[:-1],
        ret['simple_ret_1'].values[1:],
        rtol=1e-6
    )

# pandas / polars consistency
def test_fwd_return_pandas_polars_consistency(single_asset_ohlcv_pandas):
    """Should return identical results for pandas and polars input."""
    pl_df = pl.from_pandas(single_asset_ohlcv_pandas)
    res_pd = fwd_returns(single_asset_ohlcv_pandas, horizon=1)
    res_pl = fwd_returns(pl_df, horizon=1).to_pandas()
    np.testing.assert_allclose(
        res_pd['fwd_ret_1'].values[:-1],
        res_pl['fwd_ret_1'].values[:-1],
        rtol=1e-6
    )

def test_fwd_return_pandas_polars_consistency_multi_asset(multi_asset_ohlcv_pandas):
    """Multi asset consistency - groupby must behave identically."""
    pl_df = pl.from_pandas(multi_asset_ohlcv_pandas)
    res_pd = fwd_returns(multi_asset_ohlcv_pandas, horizon=1)
    res_pl = fwd_returns(pl_df, horizon=1).to_pandas()
    res_pd = res_pd.sort_values(['time', 'symbol']).reset_index(drop=True)
    res_pl = res_pl.sort_values(['time', 'symbol']).reset_index(drop=True)
    np.testing.assert_allclose(
        res_pd['fwd_ret_1'].values,
        res_pl['fwd_ret_1'].values,
        rtol=1e-6,
        equal_nan=True
    )

# raises
def test_fwd_return_missing_column_raises(single_asset_ohlcv_pandas):
    """Should raise KeyError for missing required columns."""
    with pytest.raises(KeyError):
        fwd_returns(single_asset_ohlcv_pandas.drop(columns=['close']), horizon=1)

def test_fwd_return_invalid_type_raises():
    """Should raise TypeError for unsupported input types."""
    with pytest.raises(TypeError):
        fwd_returns([[1, 2, 3]], horizon=1)
