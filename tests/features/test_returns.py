import pytest
import pandas as pd
import polars as pl
import numpy as np

from alpha_research.features.returns import compute_simple_returns

# ------------------------------------------------------
# fixtures
# ------------------------------------------------------
@pytest.fixture
def single_asset_ohlcv_pandas():
    """
    Single asset, 5 bars.
    close = [100, 110, 121, 133.1, 146.41]

    Precomputed simple_ret_1:
        bar 0: NaN (no previous)
        bar 1: 110/100 - 1 = 0.10
        bar 2: 121/110 - 1 = 0.10
        bar 3: 133.1/121 - 1 = 0.10
        bar 4: 146.41/133.1 - 1 = 0.10

    Precomputed simple_ret_2:
        bar 0: NaN
        bar 1: NaN
        bar 2: 121/100 - 1 = 0.21
        bar 3: 133.1/110 - 1 = 0.21
        bar 4: 146.41/121 - 1 = 0.21
    """
    return pd.DataFrame({
        'time': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'symbol': ['AAPL'] * 5,
        'close': [100.0, 110.0, 121.0, 133.1, 146.41],
    })


@pytest.fixture
def multi_asset_ohlcv_pandas():
    """
    Two assets, 3 bars each — interleaved.
    AAPL close = [100, 110, 121]
    MSFT close = [200, 210, 220]

    Precomputed ret_1:
        AAPL: [NaN, 0.10, 0.10]
        MSFT: [NaN, 0.05, 0.04762]

    Precomputed ret_2:
        AAPL: [NaN, NaN, 0.21]
        MSFT: [NaN, NaN, 0.10]

    Critical: groupby must not mix assets.
    """
    return pd.DataFrame({
        'time': ['2024-01-01', '2024-01-01',
                 '2024-01-02', '2024-01-02',
                 '2024-01-03', '2024-01-03'],
        'symbol': ['AAPL', 'MSFT', 'AAPL', 'MSFT', 'AAPL', 'MSFT'],
        'close': [100.0, 200.0, 110.0, 210.0, 121.0, 220.0],
    })


# ------------------------------------------------------
# compute_simple_returns
# ------------------------------------------------------
# output structure
def test_compute_return_output_columns(single_asset_ohlcv_pandas):
    """Should return only time, symbol and simple_ret column."""
    result = compute_simple_returns(single_asset_ohlcv_pandas, horizon=1)
    assert set(result.columns) == {'time', 'symbol', 'simple_ret_1'}

def test_compute_return_output_shape(single_asset_ohlcv_pandas):
    """Should return same number of rows as input."""
    result = compute_simple_returns(single_asset_ohlcv_pandas, horizon=1)
    assert len(result) == len(single_asset_ohlcv_pandas)

# correctness - single asset
def test_compute_return_values_horizon_1(single_asset_ohlcv_pandas):
    """Should compute correct simple returns for horizon=1."""
    result = compute_simple_returns(single_asset_ohlcv_pandas, horizon=1)
    expected = [np.nan, 0.10, 0.10, 0.10, 0.10]
    assert np.isnan(result['simple_ret_1'].iloc[0])
    np.testing.assert_allclose(
        result['simple_ret_1'].values[1:],
        expected[1:],
        rtol=1e-4
    )

def test_compute_return_values_horizon_2(single_asset_ohlcv_pandas):
    """Should compute correct simple returns for horizon=2."""
    result = compute_simple_returns(single_asset_ohlcv_pandas, horizon=2)
    expected = [np.nan, np.nan, 0.21, 0.21, 0.21]
    assert np.isnan(result['simple_ret_2'].iloc[0]) and np.isnan(result['simple_ret_2'].iloc[1])
    np.testing.assert_allclose(
        result['simple_ret_2'].values[2:],
        expected[2:],
        rtol=1e-4
    )

# correctness - multi asset
def test_compute_return_multi_asset_no_mixing(multi_asset_ohlcv_pandas):
    """groupby/over must not mix assets - each asset should be computed independently."""
    result = compute_simple_returns(multi_asset_ohlcv_pandas, horizon=1)
    aapl = result[result['symbol'] == 'AAPL']['simple_ret_1'].values
    msft = result[result['symbol'] == 'MSFT']['simple_ret_1'].values
    np.testing.assert_allclose(aapl[1:], [0.10, 0.10], rtol=1e-4)
    np.testing.assert_allclose(msft[1:], [0.05, 10/210], rtol=1e-4)

def test_compute_return_multi_asset_horizon_2(multi_asset_ohlcv_pandas):
    """Horizon=2 should use price 2 bars before per asset independently."""
    result = compute_simple_returns(multi_asset_ohlcv_pandas, horizon=2)
    aapl = result[result['symbol'] == 'AAPL']['simple_ret_2'].values
    msft = result[result['symbol'] == 'MSFT']['simple_ret_2'].values
    assert np.isnan(aapl[0]) and np.isnan(aapl[1])
    assert np.isnan(msft[0]) and np.isnan(msft[1])
    assert aapl[2] == pytest.approx(0.21, rel=1e-4)
    assert msft[2] == pytest.approx(0.10, rel=1e-4)

# pandas / polars consistency
def test_compute_return_pandas_polars_consistency(single_asset_ohlcv_pandas):
    """Should return identical results for pandas and polars input."""
    pl_df = pl.from_pandas(single_asset_ohlcv_pandas)
    res_pd = compute_simple_returns(single_asset_ohlcv_pandas, horizon=1)
    res_pl = compute_simple_returns(pl_df, horizon=1).to_pandas()
    np.testing.assert_allclose(
        res_pd['simple_ret_1'].values[1:],
        res_pl['simple_ret_1'].values[1:],
        rtol=1e-6
    )

# raises
def test_compute_return_missing_column_raises(single_asset_ohlcv_pandas):
    """Should raise KeyError for missing required columns."""
    with pytest.raises(KeyError):
        compute_simple_returns(single_asset_ohlcv_pandas.drop(columns=['close']), horizon=1)

def test_compute_return_invalid_type_raises():
    """Should raise TypeError for unsupported input types."""
    with pytest.raises(TypeError):
        compute_simple_returns([[1, 2, 3]], horizon=1)
