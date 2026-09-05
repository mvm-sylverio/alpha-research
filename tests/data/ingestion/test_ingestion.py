from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import polars as pl
import pytest

from alpha_research.data.ingestion import mt5 as ingestion
from alpha_research.data.market.constants import str_tf_to_mt5_tf


# ------------------------------------------------------
# fixtures
# ------------------------------------------------------
@pytest.fixture
def rates():
    """Two MT5 rates with the fields used by the ingestion function."""
    return [
        {
            'time': 1704067200,
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'tick_volume': 10,
            'real_volume': 8,
            'spread': 2,
        },
        {
            'time': 1704153600,
            'open': 100.5,
            'high': 102.0,
            'low': 100.0,
            'close': 101.5,
            'tick_volume': 12,
            'real_volume': 9,
            'spread': 3,
        },
    ]


@pytest.fixture
def configured_mt5(monkeypatch):
    """Configure the MT5 module with deterministic mocked responses."""
    mock_mt5 = SimpleNamespace(
        TIMEFRAME_M1=1,
        TIMEFRAME_M3=3,
        TIMEFRAME_M5=5,
        TIMEFRAME_M15=15,
        TIMEFRAME_M30=30,
        TIMEFRAME_H1=16385,
        TIMEFRAME_H4=16388,
        TIMEFRAME_D1=16408,
        initialize=Mock(return_value=True),
        shutdown=Mock(),
        symbol_info=Mock(return_value=SimpleNamespace(point=0.01)),
        copy_rates_range=Mock(),
    )
    monkeypatch.setattr(ingestion, 'mt5', mock_mt5)
    return mock_mt5


# ------------------------------------------------------
# _ensure_datetime
# ------------------------------------------------------
def test_ensure_datetime_normalizes_string_and_preserves_datetime():
    """Date strings should be parsed while datetime values remain unchanged."""
    date = datetime(2024, 1, 1)

    assert ingestion._ensure_datetime('2024-01-01') == date
    assert ingestion._ensure_datetime(date) is date


def test_ensure_datetime_rejects_invalid_string():
    """Invalid date strings should raise the parser's ValueError."""
    with pytest.raises(ValueError):
        ingestion._ensure_datetime('2024/01/01')


# ------------------------------------------------------
# fetch_mt_data_prices
# ------------------------------------------------------
@pytest.mark.parametrize('backend, expected_type', [
    ('polars', pl.DataFrame),
    ('pandas', pd.DataFrame),
])
def test_fetch_mt_data_prices_supports_both_backends(
    configured_mt5,
    rates,
    backend,
    expected_type,
):
    """Fetched data should be transformed correctly for both backends."""
    configured_mt5.copy_rates_range.return_value = rates

    result = ingestion.fetch_mt_data_prices(
        symbol='EURUSD',
        timeframe='H1',
        start_date='2024-01-01',
        end_date='2024-01-03',
        backend=backend,
    )

    assert isinstance(result, expected_type)
    assert list(result.columns) == [
        'time', 'open', 'high', 'low', 'close', 'volume',
        'spread', 'spread_pts',
    ]
    volume = result['volume'].to_list() if backend == 'polars' else result['volume'].tolist()
    spread = result['spread'].to_list() if backend == 'polars' else result['spread'].tolist()
    assert volume == [10, 12]
    assert spread == [0.02, 0.03]
    configured_mt5.initialize.assert_called_once_with()
    configured_mt5.shutdown.assert_called_once_with()


def test_fetch_mt_data_prices_applies_warmup_and_chunks_requests(
    configured_mt5,
    rates,
):
    """The warm-up period should shift the first request and split long ranges."""
    configured_mt5.copy_rates_range.side_effect = [rates, rates]

    result = ingestion.fetch_mt_data_prices(
        symbol='EURUSD',
        timeframe='M1',
        start_date=datetime(2024, 1, 10),
        end_date=datetime(2024, 1, 18),
        days_before=2,
    )

    first_call = configured_mt5.copy_rates_range.call_args_list[0]
    second_call = configured_mt5.copy_rates_range.call_args_list[1]
    assert first_call.args == ('EURUSD', configured_mt5.TIMEFRAME_M1, datetime(2024, 1, 8), datetime(2024, 1, 13))
    assert second_call.args == ('EURUSD', configured_mt5.TIMEFRAME_M1, datetime(2024, 1, 13, 0, 0, 1), datetime(2024, 1, 18))
    assert configured_mt5.copy_rates_range.call_count == 2
    assert len(result) == 2


def test_fetch_mt_data_prices_rejects_unknown_backend(configured_mt5):
    """Unknown backend names should fail before contacting MT5."""
    with pytest.raises(ValueError, match='backend'):
        ingestion.fetch_mt_data_prices(
            'EURUSD', 'H1', '2024-01-01', '2024-01-02', backend='numpy'
        )

    configured_mt5.initialize.assert_not_called()


def test_fetch_mt_data_prices_raises_when_mt5_initialization_fails(configured_mt5):
    """MT5 initialization failures should be reported to the caller."""
    configured_mt5.initialize.return_value = False

    with pytest.raises(RuntimeError, match='initialization failed'):
        ingestion.fetch_mt_data_prices(
            'EURUSD', 'H1', '2024-01-01', '2024-01-02'
        )


def test_fetch_mt_data_prices_raises_for_unknown_symbol(configured_mt5):
    """An unknown symbol should close MT5 and raise a clear error."""
    configured_mt5.symbol_info.return_value = None

    with pytest.raises(RuntimeError, match='Symbol not found: EURUSD'):
        ingestion.fetch_mt_data_prices(
            'EURUSD', 'H1', '2024-01-01', '2024-01-02'
        )

    configured_mt5.shutdown.assert_called_once_with()


def test_fetch_mt_data_prices_raises_when_no_rates_are_retrieved(configured_mt5):
    """An empty MT5 response should close MT5 and raise a clear error."""
    configured_mt5.copy_rates_range.return_value = None

    with pytest.raises(RuntimeError, match='No data retrieved for EURUSD'):
        ingestion.fetch_mt_data_prices(
            'EURUSD', 'H1', '2024-01-01', '2024-01-02'
        )

    configured_mt5.shutdown.assert_called_once_with()


def test_timeframe_helper_contains_standard_mt5_timeframes():
    """The timeframe helper should expose the supported string mappings."""
    assert str_tf_to_mt5_tf['M1'] == ingestion.mt5.TIMEFRAME_M1
    assert str_tf_to_mt5_tf['H1'] == ingestion.mt5.TIMEFRAME_H1
