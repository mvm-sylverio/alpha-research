import pytest
import numpy as np
import pandas as pd
import polars as pl

from alpha_research.features.targets import (
    _as_float_values,
    _as_numpy_values,
    _barrier_values,
    _triple_barrier_batch_labels,
    _triple_barrier_group_labels,
    _validate_and_group_times,
    _validate_barrier_spec,
    fwd_returns,
    ohlc_triple_barrier_labels,
)
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


# ------------------------------------------------------
# ohlc_triple_barrier_labels
# ------------------------------------------------------
def _ohlc_frame(close, high, low, **columns):
    """Build a single-symbol OHLC frame for Triple-Barrier tests."""
    frame = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=len(close), freq='D'),
        'symbol': 'AAPL',
        'close': close,
        'high': high,
        'low': low,
    })

    for name, values in columns.items():
        frame[name] = values

    return frame


def test_ohlc_triple_barrier_uses_first_horizontal_touch():
    """High and low should determine the first horizontal barrier touched."""
    df = _ohlc_frame(
        close=[100.0, 101.0, 99.0, 100.0, 100.0],
        high=[100.0, 103.0, 100.0, 100.0, 100.5],
        low=[100.0, 99.0, 96.0, 99.0, 99.5],
    )

    result = ohlc_triple_barrier_labels(
        df, horizon=2, upper_barrier=0.02, lower_barrier=0.02,
    )

    np.testing.assert_allclose(
        result['triple_barrier_2'].values[:3], [1.0, -1.0, 1.0],
    )
    assert result['triple_barrier_2'].iloc[3:].isna().all()


@pytest.mark.parametrize(
    ('tie_break', 'expected'),
    [
        ('neutral', 0.0),
        ('take_profit', 1.0),
        ('stop_loss', -1.0),
    ],
)
def test_ohlc_triple_barrier_resolves_same_bar_touches(tie_break, expected):
    """The configured tie-break should resolve an unknowable intrabar order."""
    df = _ohlc_frame(
        close=[100.0, 100.0],
        high=[100.0, 102.0],
        low=[100.0, 98.0],
    )

    result = ohlc_triple_barrier_labels(
        df,
        horizon=1,
        upper_barrier=0.02,
        lower_barrier=0.02,
        tie_break=tie_break,
    )

    assert result['triple_barrier_1'].iloc[0] == expected


def test_ohlc_triple_barrier_supports_both_vertical_label_conventions():
    """Vertical labels should support return signs and always-neutral outcomes."""
    df = _ohlc_frame(
        close=[100.0, 101.0, 99.0, 100.0],
        high=[100.0, 101.5, 99.5, 100.5],
        low=[100.0, 100.5, 98.5, 99.5],
    )

    sign_result = ohlc_triple_barrier_labels(
        df, horizon=2, upper_barrier=0.05, lower_barrier=0.05,
    )
    neutral_result = ohlc_triple_barrier_labels(
        df,
        horizon=2,
        upper_barrier=0.05,
        lower_barrier=0.05,
        vertical_label='neutral',
    )

    np.testing.assert_allclose(sign_result['triple_barrier_2'].values[:2], [-1.0, -1.0])
    np.testing.assert_allclose(neutral_result['triple_barrier_2'].values[:2], [0.0, 0.0])


def test_ohlc_triple_barrier_return_sign_preserves_positive_zero_and_negative_outcomes():
    """A vertical barrier should preserve each possible realized-return sign."""
    df = _ohlc_frame(
        close=[100.0, 101.0, 101.0, 100.0],
        high=[100.0, 101.0, 101.0, 100.0],
        low=[100.0, 101.0, 101.0, 100.0],
    )

    result = ohlc_triple_barrier_labels(
        df, horizon=1, upper_barrier=0.05, lower_barrier=0.05,
    )

    np.testing.assert_allclose(result['triple_barrier_1'].values[:3], [1.0, 0.0, -1.0])


def test_ohlc_triple_barrier_gives_horizontal_touch_precedence_at_horizon():
    """A touch in the final eligible bar should resolve before the vertical label."""
    df = _ohlc_frame(
        close=[100.0, 100.0, 100.0],
        high=[100.0, 101.0, 102.0],
        low=[100.0, 99.0, 99.0],
    )

    result = ohlc_triple_barrier_labels(
        df, horizon=2, upper_barrier=0.02, lower_barrier=0.02,
    )

    assert result['triple_barrier_2'].iloc[0] == 1.0


def test_ohlc_triple_barrier_handles_missing_path_and_vertical_close_data():
    """Missing path data should be indeterminate, while neutral expiry needs no close."""
    interrupted_path = _ohlc_frame(
        close=[100.0, 100.0, 100.0],
        high=[100.0, np.nan, 103.0],
        low=[100.0, 99.0, 99.0],
    )
    missing_close = _ohlc_frame(
        close=[100.0, np.nan],
        high=[100.0, 101.0],
        low=[100.0, 99.0],
    )

    interrupted_result = ohlc_triple_barrier_labels(
        interrupted_path, horizon=2, upper_barrier=0.02, lower_barrier=0.02,
    )
    sign_result = ohlc_triple_barrier_labels(
        missing_close, horizon=1, upper_barrier=0.05, lower_barrier=0.05,
    )
    neutral_result = ohlc_triple_barrier_labels(
        missing_close,
        horizon=1,
        upper_barrier=0.05,
        lower_barrier=0.05,
        vertical_label='neutral',
    )

    assert np.isnan(interrupted_result['triple_barrier_2'].iloc[0])
    assert np.isnan(sign_result['triple_barrier_1'].iloc[0])
    assert neutral_result['triple_barrier_1'].iloc[0] == 0.0


def test_ohlc_triple_barrier_uses_event_specific_barrier_columns():
    """Barrier columns should be read at entry and fixed for that event."""
    df = _ohlc_frame(
        close=[100.0, 100.0, 100.0],
        high=[100.0, 101.5, 100.0],
        low=[100.0, 99.5, 100.0],
        pt_distance=[0.01, 0.02, 0.01],
        sl_distance=[0.02, 0.02, 0.02],
    )

    result = ohlc_triple_barrier_labels(
        df,
        horizon=1,
        upper_barrier='pt_distance',
        lower_barrier='sl_distance',
        vertical_label='neutral',
    )

    np.testing.assert_allclose(result['triple_barrier_1'].values[:2], [1.0, 0.0])
    assert np.isnan(result['triple_barrier_1'].iloc[2])


def test_ohlc_triple_barrier_does_not_mix_interleaved_symbols():
    """Barrier paths must remain independent for interleaved assets."""
    df = pd.DataFrame({
        'time': pd.to_datetime([
            '2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02',
        ]),
        'symbol': ['AAPL', 'MSFT', 'AAPL', 'MSFT'],
        'close': [100.0, 100.0, 100.0, 100.0],
        'high': [100.0, 100.0, 103.0, 101.0],
        'low': [100.0, 100.0, 99.0, 97.0],
    })

    result = ohlc_triple_barrier_labels(
        df, horizon=1, upper_barrier=0.02, lower_barrier=0.02,
    )

    np.testing.assert_allclose(result['triple_barrier_1'].values[:2], [1.0, -1.0])
    assert result['triple_barrier_1'].iloc[2:].isna().all()


def test_ohlc_triple_barrier_pandas_polars_consistency():
    """Pandas and Polars inputs should produce the same labels."""
    df = _ohlc_frame(
        close=[100.0, 100.0, 99.0, 100.0],
        high=[100.0, 103.0, 100.0, 100.5],
        low=[100.0, 99.0, 96.0, 99.5],
    )
    kwargs = {
        'horizon': 2,
        'upper_barrier': 0.02,
        'lower_barrier': 0.02,
    }

    pandas_result = ohlc_triple_barrier_labels(df, **kwargs)
    polars_result = ohlc_triple_barrier_labels(pl.from_pandas(df), **kwargs).to_pandas()

    np.testing.assert_allclose(
        pandas_result['triple_barrier_2'].values,
        polars_result['triple_barrier_2'].values,
        equal_nan=True,
    )


def test_ohlc_triple_barrier_rejects_invalid_tie_break():
    """Unknown intrabar policies should fail explicitly."""
    df = _ohlc_frame(
        close=[100.0, 100.0], high=[100.0, 101.0], low=[100.0, 99.0],
    )

    with pytest.raises(ValueError, match='tie_break'):
        ohlc_triple_barrier_labels(
            df,
            horizon=1,
            upper_barrier=0.02,
            lower_barrier=0.02,
            tie_break='unknown',
        )


# ------------------------------------------------------
# Triple-Barrier helpers
# ------------------------------------------------------
def test_validate_barrier_spec_accepts_scalars_and_column_names():
    """Barrier specifications should accept positive scalars or column names."""
    assert _validate_barrier_spec(0.02, 'barrier') is None
    assert _validate_barrier_spec(np.float64(0.02), 'barrier') is None
    assert _validate_barrier_spec('volatility_scaled_barrier', 'barrier') is None


@pytest.mark.parametrize('barrier', [0, -0.01, np.nan, np.inf])
def test_validate_barrier_spec_rejects_non_positive_or_non_finite_scalars(barrier):
    """Scalar barriers should define positive finite relative distances."""
    with pytest.raises(ValueError, match='barrier'):
        _validate_barrier_spec(barrier, 'barrier')


@pytest.mark.parametrize('barrier', [True, None, [0.02]])
def test_validate_barrier_spec_rejects_invalid_types(barrier):
    """Barrier specifications should not accept ambiguous input types."""
    with pytest.raises(TypeError, match='barrier'):
        _validate_barrier_spec(barrier, 'barrier')


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_as_numpy_values_preserves_the_selected_column(backend):
    """Raw extraction should preserve values for both supported backends."""
    pandas_df = pd.DataFrame({'value': [1, 2, 3]})
    df = pandas_df if backend == 'pandas' else pl.from_pandas(pandas_df)

    np.testing.assert_array_equal(_as_numpy_values(df, 'value'), [1, 2, 3])


def test_as_float_values_converts_numeric_values_and_rejects_text():
    """Numeric extraction should yield floats and fail clearly for text columns."""
    numeric = pd.DataFrame({'value': [1, None, 3]})
    text = pd.DataFrame({'value': ['one', 'two']})

    np.testing.assert_allclose(
        _as_float_values(numeric, 'value'), [1.0, np.nan, 3.0], equal_nan=True,
    )
    with pytest.raises(TypeError, match='value'):
        _as_float_values(text, 'value')


def test_validate_and_group_times_preserves_interleaved_symbol_order():
    """Grouping should retain each symbol's input positions after validation."""
    groups = _validate_and_group_times(
        symbols=np.array(['AAPL', 'MSFT', 'AAPL', 'MSFT']),
        times=np.array([1, 1, 2, 2]),
        symbol_col='symbol',
        time_col='time',
    )

    assert [group.tolist() for group in groups] == [[0, 2], [1, 3]]


@pytest.mark.parametrize(
    ('symbols', 'times', 'message'),
    [
        (np.array(['AAPL', None]), np.array([1, 2]), 'symbol'),
        (np.array(['AAPL', 'AAPL']), np.array([1, np.nan]), 'time'),
        (np.array(['AAPL', 'AAPL']), np.array([1, 1]), 'unique'),
        (np.array(['AAPL', 'AAPL']), np.array([2, 1]), 'increasingly'),
    ],
)
def test_validate_and_group_times_rejects_invalid_time_keys(symbols, times, message):
    """Time keys should be complete, unique, and ordered within each symbol."""
    with pytest.raises(ValueError, match=message):
        _validate_and_group_times(symbols, times, 'symbol', 'time')


def test_barrier_values_resolves_scalar_and_column_distances():
    """Barrier resolution should broadcast scalars and preserve column missingness."""
    df = pd.DataFrame({'distance': [0.01, np.nan, 0.03]})

    np.testing.assert_allclose(
        _barrier_values(df, 0.02, 'barrier'), [0.02, 0.02, 0.02],
    )
    np.testing.assert_allclose(
        _barrier_values(df, 'distance', 'barrier'), [0.01, np.nan, 0.03],
        equal_nan=True,
    )


@pytest.mark.parametrize('values', [[0.01, 0.0], [0.01, -0.01], [0.01, np.inf]])
def test_barrier_values_rejects_invalid_column_distances(values):
    """Known barrier distances in a column must be positive and finite."""
    df = pd.DataFrame({'distance': values})

    with pytest.raises(ValueError, match='barrier'):
        _barrier_values(df, 'distance', 'barrier')


def test_triple_barrier_group_labels_preserves_incomplete_horizon_as_missing():
    """The group helper should label complete events and leave trailing rows missing."""
    labels = _triple_barrier_group_labels(
        entry=np.array([100.0, 100.0, 100.0]),
        high=np.array([100.0, 102.0, 100.0]),
        low=np.array([100.0, 99.0, 98.0]),
        close=np.array([100.0, 100.0, 100.0]),
        upper=np.array([0.02, 0.02, 0.02]),
        lower=np.array([0.02, 0.02, 0.02]),
        horizon=1,
        vertical_label='return_sign',
        tie_break='neutral',
    )

    np.testing.assert_allclose(labels[:2], [1.0, -1.0])
    assert np.isnan(labels[2])


def test_triple_barrier_batch_labels_handles_missing_paths_and_vertical_labels():
    """The batch helper should not label paths interrupted by missing OHLC data."""
    labels = _triple_barrier_batch_labels(
        high=np.array([[102.0], [np.nan], [101.0]]),
        low=np.array([[99.0], [99.0], [99.0]]),
        entry=np.array([100.0, 100.0, 100.0]),
        upper_price=np.array([102.0, 102.0, 102.0]),
        lower_price=np.array([98.0, 98.0, 98.0]),
        close_at_horizon=np.array([100.0, 100.0, 101.0]),
        valid_event=np.array([True, True, True]),
        vertical_label='return_sign',
        tie_break='neutral',
    )

    np.testing.assert_allclose(labels[[0, 2]], [1.0, 1.0])
    assert np.isnan(labels[1])
