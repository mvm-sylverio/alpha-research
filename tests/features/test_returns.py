import pytest
import polars as pl
import numpy as np
import pandas as pd

from alpha_research.features.returns import simple_returns, log_returns
from alpha_research.features.targets import (
    _valid_price_expression, _validate_return_input, fwd_returns,
)


def test_valid_price_expression_masks_invalid_values():
    """Should preserve positive finite prices and mask invalid values."""
    frame = pl.DataFrame({'close': [2., 0., -1., None, float('inf')]})
    assert frame.select(_valid_price_expression('close')).to_series().to_list() == [
        2., None, None, None, None,
    ]
    with pytest.raises(TypeError, match='price_col'):
        _valid_price_expression(1)


def test_validate_return_input_checks_horizon_and_keys():
    """Should validate a complete frame and reject an invalid horizon and keys."""
    frame = pd.DataFrame({'time': [1, 2], 'symbol': ['A', 'A'], 'close': [1., 2.]})
    assert _validate_return_input(frame, 1, 'close', 'symbol', 'time') is None
    with pytest.raises(ValueError, match='horizon'):
        _validate_return_input(frame, 0, 'close', 'symbol', 'time')
    with pytest.raises(KeyError):
        _validate_return_input(frame, 1, 'missing', 'symbol', 'time')


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('function,column,expected', [
    (simple_returns, 'simple_ret_1', [np.nan, np.nan, 1., -0.5, 0.5]),
    (log_returns, 'log_ret_1', [np.nan, np.nan, np.log(2), np.log(0.5), np.log(1.5)]),
    (fwd_returns, 'fwd_ret_1', [1., -0.5, 0.5, np.nan, np.nan]),
])
def test_returns_interleaved_assets_match_hand_calculated_pairs(backend, function, column, expected):
    """Should shift within each interleaved asset and retain original row order."""
    frame = pd.DataFrame({'time': [1, 1, 2, 2, 3], 'symbol': ['A', 'B', 'A', 'B', 'A'],
                          'close': [10., 100., 20., 50., 30.]})
    data = frame if backend == 'pandas' else pl.from_pandas(frame)
    result = function(data, 1)
    np.testing.assert_allclose(result[column].to_numpy().astype(float), expected, equal_nan=True)
    assert result['symbol'].to_list() == frame['symbol'].tolist()


@pytest.mark.parametrize('function', [simple_returns, log_returns, fwd_returns])
@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('key', ['time', 'symbol'])
def test_returns_reject_missing_keys(function, backend, key):
    """Should reject missing temporal keys instead of silently dropping rows."""
    frame = pd.DataFrame({'time': [1, 2], 'symbol': ['A', 'A'], 'close': [1., 2.]})
    frame.loc[1, key] = None
    data = frame if backend == 'pandas' else pl.from_pandas(frame)
    with pytest.raises(ValueError, match=key):
        function(data, 1)


@pytest.mark.parametrize('function', [simple_returns, log_returns, fwd_returns])
@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_returns_reject_non_numeric_prices(function, backend):
    """Should reject text that cannot be interpreted as a numeric price."""
    frame = pd.DataFrame({'time': [1, 2], 'symbol': ['A', 'A'], 'close': ['x', 'y']})
    data = frame if backend == 'pandas' else pl.from_pandas(frame)
    with pytest.raises((ValueError, TypeError, pl.exceptions.InvalidOperationError)):
        function(data, 1)


@pytest.mark.parametrize('function', [simple_returns, log_returns, fwd_returns])
@pytest.mark.parametrize('horizon', [0, -1, True, 1.5])
@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_returns_reject_invalid_horizons(function, horizon, backend):
    """Should reject horizons that cannot represent a positive bar count."""
    frame = pd.DataFrame({'time': [1, 2], 'symbol': ['A', 'A'], 'close': [1., 2.]})
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    with pytest.raises(ValueError, match='horizon'):
        function(frame, horizon)


@pytest.mark.parametrize('function', [simple_returns, log_returns, fwd_returns])
@pytest.mark.parametrize('times', [[2, 1], [1, 1]])
@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_returns_reject_unordered_or_duplicate_times(function, times, backend):
    """Should reject ambiguous temporal ordering before shifting prices."""
    frame = pd.DataFrame({'time': times, 'symbol': ['A', 'A'], 'close': [1., 2.]})
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    with pytest.raises(ValueError):
        function(frame, 1)


@pytest.mark.parametrize('function,column', [
    (simple_returns, 'simple_ret_1'), (log_returns, 'log_ret_1'),
    (fwd_returns, 'fwd_ret_1'),
])
@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_returns_invalid_prices_only_invalidate_affected_pairs(function, column, backend):
    """Should keep valid pairs while masking zero, negative and non-finite prices."""
    frame = pd.DataFrame({'time': range(9), 'symbol': ['A'] * 9,
                          'close': [10., 20., 0., 30., -1., np.inf, np.nan, 40., 80.]})
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    result = function(frame, 1)
    values = result[column].to_numpy().astype(float)
    expected = np.full(9, np.nan)
    positions = [0, 7] if function is fwd_returns else [1, 8]
    expected[positions] = np.log(2) if function is log_returns else 1.0
    np.testing.assert_allclose(values, expected, equal_nan=True)


@pytest.mark.parametrize('function', [simple_returns, log_returns, fwd_returns])
def test_returns_long_horizon_produces_missing_values(function):
    """Should return missing values when no symbol has enough history."""
    frame = pd.DataFrame({'time': [1, 2], 'symbol': ['A', 'A'], 'close': [1., 2.]})
    assert function(frame, 2).iloc[:, -1].isna().all()


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('function,column,expected', [
    (simple_returns, 'simple_ret_2', [np.nan, np.nan, np.nan, np.nan, 2.]),
    (log_returns, 'log_ret_2', [np.nan, np.nan, np.nan, np.nan, np.log(3)]),
    (fwd_returns, 'fwd_ret_2', [2., np.nan, np.nan, np.nan, np.nan]),
])
def test_returns_horizon_on_different_asset_history_lengths(backend, function, column, expected):
    """Should calculate only assets with enough observations for the horizon."""
    frame = pd.DataFrame({'time': [1, 1, 2, 2, 3], 'symbol': ['A', 'B', 'A', 'B', 'A'],
                          'close': [10., 100., 20., 50., 30.]})
    data = frame if backend == 'pandas' else pl.from_pandas(frame)
    np.testing.assert_allclose(function(data, 2)[column].to_numpy().astype(float), expected, equal_nan=True)


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
