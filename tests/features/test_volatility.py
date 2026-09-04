import numpy as np
import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.features.volatility import (
    _asset_positions,
    _average_true_range_values,
    _feature_result_frame,
    _format_annualization_factor,
    _true_range_by_symbol,
    _validate_annualization_factor,
    _wilder_smoothing_by_symbol,
    average_true_range,
    atr,
    realized_volatility,
)


@pytest.fixture
def volatility_ohlcv_pandas():
    """Create one OHLC series with ranges, gaps, and a later down move."""
    return pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=7, freq='D'),
        'symbol': ['AAPL'] * 7,
        'high': [10.0, 12.0, 13.0, 14.0, 15.0, 14.0, 16.0],
        'low': [8.0, 9.0, 10.0, 11.0, 12.0, 11.0, 13.0],
        'close': [9.0, 11.0, 12.0, 13.0, 13.0, 12.0, 15.0],
    })


@pytest.fixture
def multi_asset_volatility_ohlcv_pandas(volatility_ohlcv_pandas):
    """Create interleaved assets to verify all calculations remain isolated."""
    msft = volatility_ohlcv_pandas.copy()
    msft['symbol'] = 'MSFT'
    msft['high'] = [20.0, 21.0, 24.0, 24.0, 25.0, 23.0, 27.0]
    msft['low'] = [18.0, 19.0, 20.0, 21.0, 22.0, 20.0, 23.0]
    msft['close'] = [19.0, 20.0, 22.0, 22.0, 23.0, 21.0, 26.0]

    return pd.concat([volatility_ohlcv_pandas, msft]).sort_values(
        ['time', 'symbol'],
        ignore_index=True,
    )


# ------------------------------------------------------
# _validate_annualization_factor
# ------------------------------------------------------
def test_validate_annualization_factor_accepts_valid_values():
    """Should accept None and finite positive numeric annualization factors."""
    for factor in [None, 252, 252.5, np.int64(365), np.float64(24.0)]:
        assert _validate_annualization_factor(factor) is None


@pytest.mark.parametrize('factor', [0, -1, True, '252', np.nan, np.inf])
def test_validate_annualization_factor_rejects_invalid_values(factor):
    """Should reject values that cannot define annualized volatility units."""
    with pytest.raises(ValueError, match='annualization_factor'):
        _validate_annualization_factor(factor)


# ------------------------------------------------------
# _format_annualization_factor
# ------------------------------------------------------
def test_format_annualization_factor_preserves_column_name_units():
    """Should format integer and fractional factors without decimal separators."""
    assert _format_annualization_factor(252.0) == '252'
    assert _format_annualization_factor(365.25) == '365_25'


# ------------------------------------------------------
# _asset_positions
# ------------------------------------------------------
def test_asset_positions_groups_interleaved_symbols_in_input_order():
    """Should retain each asset's temporal row order without sorting the frame."""
    result = _asset_positions(['AAPL', 'MSFT', 'AAPL', 'GOOGL', 'MSFT'])

    assert result == [[0, 2], [1, 4], [3]]


# ------------------------------------------------------
# _true_range_by_symbol
# ------------------------------------------------------
def test_true_range_by_symbol_uses_only_the_previous_close_of_each_asset():
    """Should compute intrabar range first and gaps from the same asset only."""
    result = _true_range_by_symbol(
        high=np.array([10.0, 20.0, 14.0, 24.0]),
        low=np.array([8.0, 18.0, 10.0, 21.0]),
        close=np.array([9.0, 19.0, 11.0, 23.0]),
        symbols=['AAPL', 'MSFT', 'AAPL', 'MSFT'],
    )

    np.testing.assert_allclose(result, [2.0, 2.0, 5.0, 5.0])


# ------------------------------------------------------
# _wilder_smoothing_by_symbol
# ------------------------------------------------------
def test_wilder_smoothing_by_symbol_seeds_recurses_and_resets_after_missing():
    """Should isolate symbols, use the first mean, and restart after missing data."""
    result = _wilder_smoothing_by_symbol(
        values=np.array([1.0, 10.0, 2.0, 20.0, 3.0, np.nan, 5.0, 30.0]),
        symbols=['AAPL', 'MSFT', 'AAPL', 'MSFT', 'AAPL', 'MSFT', 'AAPL', 'MSFT'],
        window=3,
    )

    np.testing.assert_allclose(
        result,
        [np.nan, np.nan, np.nan, np.nan, 2.0, np.nan, 3.0, np.nan],
        equal_nan=True,
    )


# ------------------------------------------------------
# _average_true_range_values
# ------------------------------------------------------
def test_average_true_range_values_integrates_true_range_and_wilder_smoothing(
        volatility_ohlcv_pandas,
):
    """Should return the ATR sequence produced by the two underlying helpers."""
    result = _average_true_range_values(
        high=volatility_ohlcv_pandas['high'].to_numpy(),
        low=volatility_ohlcv_pandas['low'].to_numpy(),
        close=volatility_ohlcv_pandas['close'].to_numpy(),
        symbols=volatility_ohlcv_pandas['symbol'].to_list(),
        window=3,
    )

    np.testing.assert_allclose(
        result,
        [np.nan, np.nan, 8 / 3, 25 / 9, 77 / 27, 235 / 81, 794 / 243],
        rtol=1e-12,
        equal_nan=True,
    )


# ------------------------------------------------------
# _feature_result_frame
# ------------------------------------------------------
@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_feature_result_frame_preserves_backend_and_feature_contract(
        volatility_ohlcv_pandas,
        backend,
):
    """Should preserve keys, backend, row order, and one supplied value column."""
    source = (
        volatility_ohlcv_pandas
        if backend == 'pandas'
        else pl.from_pandas(volatility_ohlcv_pandas)
    )
    result = _feature_result_frame(
        source,
        values=np.array([0.1] * len(source)),
        feature_name='test_feature',
        time_col='time',
        symbol_col='symbol',
    )

    assert isinstance(result, type(source))
    assert list(result.columns) == ['time', 'symbol', 'test_feature']
    assert len(result) == len(source)


# ------------------------------------------------------
# realized_volatility
# ------------------------------------------------------
def test_realized_volatility_returns_one_feature_column(volatility_ohlcv_pandas):
    """Should preserve keys and return a named realized-volatility feature."""
    result = realized_volatility(volatility_ohlcv_pandas, window=3)

    assert list(result.columns) == ['time', 'symbol', 'realized_vol_3']
    assert len(result) == len(volatility_ohlcv_pandas)


def test_realized_volatility_uses_past_log_returns(volatility_ohlcv_pandas):
    """Should equal sample volatility of the trailing known log returns."""
    result = realized_volatility(volatility_ohlcv_pandas, window=3)
    log_returns = np.log(
        volatility_ohlcv_pandas['close'].to_numpy()[1:]
        / volatility_ohlcv_pandas['close'].to_numpy()[:-1],
    )
    expected = [
        np.std(log_returns[index - 2:index + 1], ddof=1)
        for index in range(2, len(log_returns))
    ]

    assert result['realized_vol_3'].iloc[:3].isna().all()
    np.testing.assert_allclose(
        result['realized_vol_3'].to_numpy()[3:],
        expected,
        rtol=1e-12,
    )


def test_realized_volatility_reuses_log_return_feature(
        monkeypatch,
        volatility_ohlcv_pandas,
):
    """Should delegate one-bar return construction to log_returns()."""
    captured_kwargs = {}

    def fake_log_returns(**kwargs):
        captured_kwargs.update(kwargs)
        source = kwargs['df']
        return pd.DataFrame({
            'time': source['time'],
            'symbol': source['symbol'],
            'log_ret_1': [np.nan, 0.10, 0.20, -0.10, 0.10, 0.20, -0.10],
        })

    monkeypatch.setattr(
        'alpha_research.features.volatility.log_returns',
        fake_log_returns,
    )

    result = realized_volatility(volatility_ohlcv_pandas, window=3)

    assert captured_kwargs['horizon'] == 1
    assert captured_kwargs['price_col'] == 'close'
    assert captured_kwargs['symbol_col'] == 'symbol'
    assert captured_kwargs['time_col'] == 'time'
    assert result['realized_vol_3'].iloc[:3].isna().all()
    assert result['realized_vol_3'].iloc[3] == pytest.approx(
        np.std([0.10, 0.20, -0.10], ddof=1),
    )


def test_realized_volatility_annualizes_and_names_units(
        volatility_ohlcv_pandas,
):
    """Should apply sqrt(factor) and encode annualization in the feature name."""
    raw = realized_volatility(volatility_ohlcv_pandas, window=3)
    annualized = realized_volatility(
        volatility_ohlcv_pandas,
        window=3,
        annualization_factor=252,
    )

    assert list(annualized.columns) == [
        'time', 'symbol', 'realized_vol_3_annualized_252',
    ]
    np.testing.assert_allclose(
        annualized['realized_vol_3_annualized_252'].to_numpy(),
        raw['realized_vol_3'].to_numpy() * np.sqrt(252),
        rtol=1e-12,
        equal_nan=True,
    )


def test_realized_volatility_does_not_mix_assets(
        multi_asset_volatility_ohlcv_pandas,
):
    """Should use trailing returns only from the current symbol."""
    result = realized_volatility(multi_asset_volatility_ohlcv_pandas, window=2)

    for symbol in ['AAPL', 'MSFT']:
        expected = realized_volatility(
            multi_asset_volatility_ohlcv_pandas.loc[
                multi_asset_volatility_ohlcv_pandas['symbol'] == symbol,
            ],
            window=2,
        )['realized_vol_2'].to_numpy()
        actual = result.loc[result['symbol'] == symbol, 'realized_vol_2'].to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize('window', [0, 1, -1, True, 1.5])
def test_realized_volatility_validates_window(volatility_ohlcv_pandas, window):
    """Should require an integer window capable of sample dispersion."""
    with pytest.raises(ValueError, match='window'):
        realized_volatility(volatility_ohlcv_pandas, window=window)


@pytest.mark.parametrize(
    'annualization_factor',
    [0, -252, True, '252', np.inf, np.nan],
)
def test_realized_volatility_validates_annualization_factor(
        volatility_ohlcv_pandas,
        annualization_factor,
):
    """Should reject invalid annualization factors instead of changing units silently."""
    with pytest.raises(ValueError, match='annualization_factor'):
        realized_volatility(
            volatility_ohlcv_pandas,
            window=3,
            annualization_factor=annualization_factor,
        )


def test_realized_volatility_validates_input(volatility_ohlcv_pandas):
    """Should reject missing prices and unsupported input backends."""
    with pytest.raises(KeyError):
        realized_volatility(volatility_ohlcv_pandas.drop(columns='close'), window=3)

    with pytest.raises(TypeError):
        realized_volatility([[1, 2, 3]], window=3)


# ------------------------------------------------------
# ATR
# ------------------------------------------------------
def test_average_true_range_uses_wilder_true_range(volatility_ohlcv_pandas):
    """Should include gaps in True Range and smooth it by Wilder's recursion."""
    result = average_true_range(
        volatility_ohlcv_pandas,
        window=3,
        normalize=False,
    )

    expected = [
        np.nan,
        np.nan,
        8 / 3,
        25 / 9,
        77 / 27,
        235 / 81,
        794 / 243,
    ]
    assert list(result.columns) == ['time', 'symbol', 'atr_3']
    np.testing.assert_allclose(
        result['atr_3'].to_numpy(),
        expected,
        rtol=1e-12,
        equal_nan=True,
    )


def test_average_true_range_normalizes_by_default(
        volatility_ohlcv_pandas,
):
    """Should return the same Wilder ATR divided by contemporaneous close."""
    raw_atr = average_true_range(
        volatility_ohlcv_pandas,
        window=3,
        normalize=False,
    )
    result = average_true_range(volatility_ohlcv_pandas, window=3)

    assert list(result.columns) == ['time', 'symbol', 'natr_3']
    np.testing.assert_allclose(
        result['natr_3'].to_numpy(),
        raw_atr['atr_3'].to_numpy() / volatility_ohlcv_pandas['close'].to_numpy(),
        rtol=1e-12,
        equal_nan=True,
    )


def test_average_true_range_returns_missing_for_zero_close_when_normalized(
        volatility_ohlcv_pandas,
):
    """Should not emit an infinite normalized range when the close is zero."""
    zero_close = volatility_ohlcv_pandas.copy()
    zero_close.loc[4, 'close'] = 0.0

    result = average_true_range(zero_close, window=3)

    assert np.isnan(result['natr_3'].iloc[4])


def test_atr_features_do_not_mix_assets(
        multi_asset_volatility_ohlcv_pandas,
):
    """Should calculate normalized range features independently within every symbol."""
    result = average_true_range(multi_asset_volatility_ohlcv_pandas, window=3)
    feature_col = result.columns[-1]

    for symbol in ['AAPL', 'MSFT']:
        expected = average_true_range(
            multi_asset_volatility_ohlcv_pandas.loc[
                multi_asset_volatility_ohlcv_pandas['symbol'] == symbol,
            ],
            window=3,
        )[feature_col].to_numpy()
        actual = result.loc[result['symbol'] == symbol, feature_col].to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, equal_nan=True)


def test_average_true_range_validates_window_input_and_normalize(
        volatility_ohlcv_pandas,
):
    """Should validate all required OHLC columns and a positive window."""
    with pytest.raises(ValueError, match='window'):
        average_true_range(volatility_ohlcv_pandas, window=0)

    with pytest.raises(KeyError):
        average_true_range(volatility_ohlcv_pandas.drop(columns='high'), window=3)

    with pytest.raises(TypeError):
        average_true_range([[1, 2, 3]], window=3)

    with pytest.raises(TypeError, match='normalize'):
        average_true_range(volatility_ohlcv_pandas, window=3, normalize='yes')


@pytest.mark.parametrize('normalize', [True, False])
def test_atr_alias_matches_average_true_range(
        volatility_ohlcv_pandas,
        normalize,
):
    """Should forward the selected normalization behavior without alteration."""
    expected = average_true_range(
        volatility_ohlcv_pandas,
        window=3,
        normalize=normalize,
    )
    result = atr(
        volatility_ohlcv_pandas,
        window=3,
        normalize=normalize,
    )
    feature_col = expected.columns[-1]

    np.testing.assert_allclose(
        result[feature_col].to_numpy(),
        expected[feature_col].to_numpy(),
        rtol=1e-12,
        equal_nan=True,
    )


# ------------------------------------------------------
# Pandas / Polars consistency
# ------------------------------------------------------
@pytest.mark.parametrize(
    'feature_function, kwargs',
    [
        (realized_volatility, {'window': 3}),
        (realized_volatility, {'window': 3, 'annualization_factor': 252}),
        (average_true_range, {'window': 3}),
        (average_true_range, {'window': 3, 'normalize': False}),
    ],
)
def test_volatility_features_match_between_pandas_and_polars(
        multi_asset_volatility_ohlcv_pandas,
        feature_function,
        kwargs,
):
    """Should preserve numerical feature values across supported backends."""
    pandas_result = feature_function(multi_asset_volatility_ohlcv_pandas, **kwargs)
    polars_result = feature_function(
        pl.from_pandas(multi_asset_volatility_ohlcv_pandas),
        **kwargs,
    ).to_pandas()
    feature_col = pandas_result.columns[-1]

    pandas_result = pandas_result.sort_values(['time', 'symbol']).reset_index(drop=True)
    polars_result = polars_result.sort_values(['time', 'symbol']).reset_index(drop=True)
    np.testing.assert_allclose(
        pandas_result[feature_col].to_numpy(),
        polars_result[feature_col].to_numpy(),
        rtol=1e-12,
        equal_nan=True,
    )
