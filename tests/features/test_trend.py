import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.features.trend import (
    adx,
    average_directional_index,
    price_to_sma_ratio,
    sma_crossover,
)

# -------------------------------------------------------
# fixtures
#  ------------------------------------------------------
@pytest.fixture
def adx_ohlcv_pandas():
    """Create a seven-bar series with positive and negative directional moves."""
    return pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=7, freq='D'),
        'symbol': ['AAPL'] * 7,
        'high': [10.0, 12.0, 13.0, 14.0, 15.0, 14.0, 16.0],
        'low': [8.0, 9.0, 10.0, 11.0, 12.0, 11.0, 13.0],
        'close': [9.0, 11.0, 12.0, 13.0, 13.0, 12.0, 15.0],
    })


# ------------------------------------------------------
# price_to_sma_ratio
# ------------------------------------------------------
# output structure
def test_price_to_sma_ratio_output_columns(single_asset_ohlcv_pandas):
    """Should return only time, symbol and price_to_sma_ratio column."""
    result = price_to_sma_ratio(single_asset_ohlcv_pandas, window=3)
    assert set(result.columns) == {'time', 'symbol', 'price_to_sma_ratio_3'}

def test_price_to_sma_ratio_output_shape(single_asset_ohlcv_pandas):
    """Should return same number of rows as input."""
    result = price_to_sma_ratio(single_asset_ohlcv_pandas, window=3)
    assert len(result) == len(single_asset_ohlcv_pandas)

# correctness - single asset
def test_price_to_sma_ratio_values(single_asset_ohlcv_pandas):
    """Should compute correct price to SMA ratio."""
    result = price_to_sma_ratio(single_asset_ohlcv_pandas, window=3)
    assert np.isnan(result['price_to_sma_ratio_3'].iloc[0])
    assert np.isnan(result['price_to_sma_ratio_3'].iloc[1])
    np.testing.assert_allclose(
        result['price_to_sma_ratio_3'].values[2:],
        [0.09667, 0.09667, 0.09667],
        rtol=1e-4
    )

def test_price_to_sma_ratio_positive_when_price_above_sma(single_asset_ohlcv_pandas):
    """Ratio should be positive when price is above SMA — uptrend."""
    result = price_to_sma_ratio(single_asset_ohlcv_pandas, window=3)
    assert (result['price_to_sma_ratio_3'].dropna() > 0).all()

# correctness - multi asset
def test_price_to_sma_ratio_multi_asset_no_mixing(multi_asset_ohlcv_pandas):
    """Groupby must not mix assets - each asset should be computed independently."""
    result = price_to_sma_ratio(multi_asset_ohlcv_pandas, window=2)
    aapl = result[result['symbol'] == 'AAPL']['price_to_sma_ratio_2'].values
    msft = result[result['symbol'] == 'MSFT']['price_to_sma_ratio_2'].values
    assert np.isnan(aapl[0])
    assert np.isnan(msft[0])
    np.testing.assert_allclose(
        aapl[1:],
        [0.047619, 0.047619],
        rtol=1e-4
    )
    np.testing.assert_allclose(
        msft[1:],
        [0.024390, 0.023256],
        rtol=1e-4
    )

# pandas / polars consistency
def test_price_to_sma_ratio_pandas_polars_consistency(single_asset_ohlcv_pandas):
    """Should return identical results for pandas and polars input."""
    pl_df = pl.from_pandas(single_asset_ohlcv_pandas)
    res_pd = price_to_sma_ratio(single_asset_ohlcv_pandas, window=3)
    res_pl = price_to_sma_ratio(pl_df, window=3).to_pandas()
    np.testing.assert_allclose(
        res_pd['price_to_sma_ratio_3'].values[2:],
        res_pl['price_to_sma_ratio_3'].values[2:],
        rtol=1e-6
    )

def test_price_to_sma_ratio_pandas_polars_consistency_multi_asset(multi_asset_ohlcv_pandas):
    """Should return identical results for pandas and polars multi-asset input."""
    pl_df = pl.from_pandas(multi_asset_ohlcv_pandas)
    res_pd = price_to_sma_ratio(multi_asset_ohlcv_pandas, window=2)
    res_pl = price_to_sma_ratio(pl_df, window=2).to_pandas()

    # sort both by time and symbol before comparing
    res_pd = res_pd.sort_values(['time', 'symbol']).reset_index(drop=True)
    res_pl = res_pl.sort_values(['time', 'symbol']).reset_index(drop=True)

    np.testing.assert_allclose(
        res_pd['price_to_sma_ratio_2'].values,
        res_pl['price_to_sma_ratio_2'].values,
        rtol=1e-6,
        equal_nan=True
    )

# raises
def test_price_to_sma_ratio_missing_column_raises(single_asset_ohlcv_pandas):
    """Should raise KeyError for missing required columns."""
    with pytest.raises(KeyError):
        price_to_sma_ratio(single_asset_ohlcv_pandas.drop(columns=['close']), window=3)

def test_price_to_sma_ratio_invalid_type_raises():
    """Should raise TypeError for unsupported input types."""
    with pytest.raises(TypeError):
        price_to_sma_ratio([[1, 2, 3]], window=3)


# ------------------------------------------------------
# sma_crossover
# ------------------------------------------------------
# windows validations
def test_sma_crossover_fast_gte_slow_raises(single_asset_ohlcv_pandas):
    """Should raise ValueError when fast_window >= slow_window."""
    with pytest.raises(ValueError, match="fast_window"):
        sma_crossover(single_asset_ohlcv_pandas, fast_window=20, slow_window=5)

def test_sma_crossover_fast_equal_slow_raises(single_asset_ohlcv_pandas):
    """Should raise ValueError when fast_window == slow_window."""
    with pytest.raises(ValueError, match="fast_window"):
        sma_crossover(single_asset_ohlcv_pandas, fast_window=5, slow_window=5)

# output structure
def test_sma_crossover_output_columns(single_asset_ohlcv_pandas):
    """Should return only time, symbol and crossover column."""
    result = sma_crossover(single_asset_ohlcv_pandas, fast_window=2, slow_window=3)
    assert set(result.columns) == {'time', 'symbol', 'sma_2_crossover_sma_3'}

def test_sma_crossover_output_shape(single_asset_ohlcv_pandas):
    """Should return same number of rows as input."""
    result = sma_crossover(single_asset_ohlcv_pandas, fast_window=2, slow_window=3)
    assert len(result) == len(single_asset_ohlcv_pandas)

# correctness
def test_sma_crossover_values(single_asset_ohlcv_pandas):
    """Should compute correct crossover ratio."""
    result = sma_crossover(single_asset_ohlcv_pandas, fast_window=2, slow_window=3)
    assert np.isnan(result['sma_2_crossover_sma_3'].iloc[0])
    assert np.isnan(result['sma_2_crossover_sma_3'].iloc[1])
    np.testing.assert_allclose(
        result['sma_2_crossover_sma_3'].values[2:],
        [0.046827, 0.046827, 0.046827],
        rtol=1e-4
    )

def test_sma_crossover_positive_in_uptrend(single_asset_ohlcv_pandas):
    """Fast SMA > slow SMA in uptrend — crossover should be positive."""
    result = sma_crossover(single_asset_ohlcv_pandas, fast_window=2, slow_window=3)
    assert (result['sma_2_crossover_sma_3'].dropna() > 0).all()

# multi asset
def test_sma_crossover_multi_asset_no_mixing(multi_asset_ohlcv_pandas):
    """Groupby must not mix assets - each asset should be computed independently."""
    result = sma_crossover(multi_asset_ohlcv_pandas, fast_window=2, slow_window=3)
    aapl = result[result['symbol'] == 'AAPL']['sma_2_crossover_sma_3'].values
    msft = result[result['symbol'] == 'MSFT']['sma_2_crossover_sma_3'].values
    assert np.isnan(aapl[0]) and np.isnan(aapl[1])
    assert np.isnan(msft[0]) and np.isnan(msft[1])
    assert aapl[2] == pytest.approx(0.046827, rel=1e-4)
    assert msft[2] == pytest.approx(0.023810, rel=1e-4)

# pandas / polars consistency
def test_sma_crossover_pandas_polars_consistency(single_asset_ohlcv_pandas):
    """Should return identical results for pandas and polars input."""
    pl_df = pl.from_pandas(single_asset_ohlcv_pandas)
    res_pd = sma_crossover(single_asset_ohlcv_pandas, fast_window=2, slow_window=3)
    res_pl = sma_crossover(pl_df, fast_window=2, slow_window=3).to_pandas()
    np.testing.assert_allclose(
        res_pd['sma_2_crossover_sma_3'].values[2:],
        res_pl['sma_2_crossover_sma_3'].values[2:],
        rtol=1e-6
    )

def test_sma_crossover_pandas_polars_consistency_multi_asset(multi_asset_ohlcv_pandas):
    """Should return identical results for pandas and polars multi-asset input."""
    pl_df = pl.from_pandas(multi_asset_ohlcv_pandas)
    res_pd = sma_crossover(multi_asset_ohlcv_pandas, fast_window=2, slow_window=3)
    res_pl = sma_crossover(pl_df, fast_window=2, slow_window=3).to_pandas()
    res_pd = res_pd.sort_values(['time', 'symbol']).reset_index(drop=True)
    res_pl = res_pl.sort_values(['time', 'symbol']).reset_index(drop=True)
    np.testing.assert_allclose(
        res_pd['sma_2_crossover_sma_3'].values,
        res_pl['sma_2_crossover_sma_3'].values,
        rtol=1e-6,
        equal_nan=True
    )

# raises
def test_sma_crossover_missing_column_raises(single_asset_ohlcv_pandas):
    """Should raise KeyError for missing required columns."""
    with pytest.raises(KeyError):
        sma_crossover(
            single_asset_ohlcv_pandas.drop(columns=['close']),
            fast_window=2, slow_window=3
        )

def test_sma_crossover_invalid_type_raises():
    """Should raise TypeError for unsupported input types."""
    with pytest.raises(TypeError):
        sma_crossover([[1, 2, 3]], fast_window=2, slow_window=3)


# ------------------------------------------------------
# average_directional_index
# ------------------------------------------------------
def test_average_directional_index_returns_wilder_adx(adx_ohlcv_pandas):
    """Should apply two Wilder smoothing stages to Directional Index values."""
    result = average_directional_index(adx_ohlcv_pandas, window=3)

    assert list(result.columns) == ['time', 'symbol', 'adx_3']
    assert result['adx_3'].iloc[:5].isna().all()
    np.testing.assert_allclose(
        result['adx_3'].iloc[5:].to_numpy(),
        [80.64516129032258, 76.75194660734148],
        rtol=1e-12,
    )


def test_average_directional_index_does_not_mix_assets(adx_ohlcv_pandas):
    """Should maintain separate directional histories for interleaved symbols."""
    msft = adx_ohlcv_pandas.copy()
    msft['symbol'] = 'MSFT'
    msft['high'] = [20.0, 21.0, 24.0, 24.0, 25.0, 23.0, 27.0]
    msft['low'] = [18.0, 19.0, 20.0, 21.0, 22.0, 20.0, 23.0]
    msft['close'] = [19.0, 20.0, 22.0, 22.0, 23.0, 21.0, 26.0]
    multi_asset = pd.concat([adx_ohlcv_pandas, msft]).sort_values(
        ['time', 'symbol'],
        ignore_index=True,
    )

    result = average_directional_index(multi_asset, window=3)
    for symbol in ['AAPL', 'MSFT']:
        expected = average_directional_index(
            multi_asset.loc[multi_asset['symbol'] == symbol],
            window=3,
        )['adx_3'].to_numpy()
        actual = result.loc[result['symbol'] == symbol, 'adx_3'].to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, equal_nan=True)


def test_average_directional_index_matches_polars(adx_ohlcv_pandas):
    """Should preserve ADX values when the input backend is Polars."""
    pandas_result = average_directional_index(adx_ohlcv_pandas, window=3)
    polars_result = average_directional_index(
        pl.from_pandas(adx_ohlcv_pandas),
        window=3,
    ).to_pandas()

    np.testing.assert_allclose(
        pandas_result['adx_3'].to_numpy(),
        polars_result['adx_3'].to_numpy(),
        rtol=1e-12,
        equal_nan=True,
    )


def test_adx_alias_matches_average_directional_index(adx_ohlcv_pandas):
    """Should provide ADX shorthand alias without changing the calculation."""
    expected = average_directional_index(adx_ohlcv_pandas, window=3)
    result = adx(adx_ohlcv_pandas, window=3)

    np.testing.assert_allclose(
        result['adx_3'].to_numpy(),
        expected['adx_3'].to_numpy(),
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.parametrize('window', [0, -1, True, 1.5])
def test_average_directional_index_validates_window(adx_ohlcv_pandas, window):
    """Should require a positive integer Wilder period."""
    with pytest.raises(ValueError, match='window'):
        average_directional_index(adx_ohlcv_pandas, window=window)


def test_average_directional_index_validates_input(adx_ohlcv_pandas):
    """Should reject missing OHLC inputs and unsupported DataFrames."""
    with pytest.raises(KeyError):
        average_directional_index(adx_ohlcv_pandas.drop(columns='low'), window=3)

    with pytest.raises(TypeError):
        average_directional_index([[1, 2, 3]], window=3)
