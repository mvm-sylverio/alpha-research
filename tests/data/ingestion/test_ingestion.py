import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import polars as pl
import pytest

from alpha_research.data.ingestion import mt5 as ingestion
from alpha_research.data.market.constants import str_tf_to_mt5_tf


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
    """Replace the optional MT5 client with a deterministic test double."""
    timeframe_attributes = {
        f'TIMEFRAME_{name}': value
        for name, value in str_tf_to_mt5_tf.items()
    }
    mock_mt5 = SimpleNamespace(
        **timeframe_attributes,
        initialize=Mock(return_value=True),
        shutdown=Mock(),
        symbol_info=Mock(return_value=SimpleNamespace(point=0.01)),
        copy_rates_range=Mock(),
    )
    monkeypatch.setattr(ingestion, 'mt5', mock_mt5)
    return mock_mt5


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('2024-01-01', datetime(2024, 1, 1, tzinfo=UTC)),
        (datetime.fromisoformat('2024-01-01'), datetime(2024, 1, 1, tzinfo=UTC)),
        (
            datetime(2024, 1, 1, tzinfo=timezone(timedelta(hours=3))),
            datetime(2023, 12, 31, 21, tzinfo=UTC),
        ),
    ],
)
def test_ensure_datetime_returns_utc_aware_values(value, expected):
    """Should normalize date strings and naive or aware datetimes to UTC."""
    assert ingestion._ensure_datetime(value, 'date') == expected
    assert ingestion._ensure_datetime(value, 'date').tzinfo is UTC


@pytest.mark.parametrize('value', ['2024/01/01', 'not-a-date'])
def test_ensure_datetime_rejects_invalid_string(value):
    """Should reject date strings that do not match the documented format."""
    with pytest.raises(ValueError, match='YYYY-MM-DD'):
        ingestion._ensure_datetime(value, 'date')


@pytest.mark.parametrize('value', [None, 20240101, object()])
def test_ensure_datetime_rejects_unsupported_types(value):
    """Should reject values that are neither supported strings nor datetimes."""
    with pytest.raises(TypeError, match='string or datetime'):
        ingestion._ensure_datetime(value, 'date')


@pytest.mark.parametrize('backend, expected_type', [
    ('polars', pl.DataFrame),
    ('pandas', pd.DataFrame),
])
def test_fetch_returns_expected_schema_values_and_utc_timezone(
    configured_mt5,
    rates,
    backend,
    expected_type,
):
    """Should preserve the schema, conversions, and UTC timezone for each backend."""
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
        'time', 'open', 'high', 'low', 'close', 'volume', 'spread', 'spread_pts',
    ]
    if backend == 'polars':
        assert result.schema['time'] == pl.Datetime(time_zone='UTC')
        assert result['time'].to_list()[0] == datetime(2024, 1, 1, tzinfo=UTC)
        assert result['volume'].to_list() == [10, 12]
        assert result['spread'].to_list() == [0.02, 0.03]
    else:
        assert isinstance(result['time'].dtype, pd.DatetimeTZDtype)
        assert str(result['time'].dtype.tz) == 'UTC'
        assert result['time'].iloc[0].to_pydatetime() == datetime(2024, 1, 1, tzinfo=UTC)
        assert result['volume'].tolist() == [10, 12]
        assert result['spread'].tolist() == [0.02, 0.03]

    configured_mt5.initialize.assert_called_once_with()
    configured_mt5.shutdown.assert_called_once_with()
    args = configured_mt5.copy_rates_range.call_args.args
    assert args[2:] == (
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
    )


def test_fetch_normalizes_aware_input_and_applies_utc_warmup(configured_mt5, rates):
    """Should normalize request bounds to UTC before warmup and chunking."""
    configured_mt5.copy_rates_range.side_effect = [rates, rates]
    plus_three = timezone(timedelta(hours=3))

    result = ingestion.fetch_mt_data_prices(
        symbol='EURUSD',
        timeframe='M1',
        start_date=datetime(2024, 1, 10, tzinfo=plus_three),
        end_date=datetime(2024, 1, 18, tzinfo=plus_three),
        days_before=2,
    )

    first_call, second_call = configured_mt5.copy_rates_range.call_args_list
    assert first_call.args == (
        'EURUSD', str_tf_to_mt5_tf['M1'],
        datetime(2024, 1, 7, 21, tzinfo=UTC),
        datetime(2024, 1, 12, 21, tzinfo=UTC),
    )
    assert second_call.args == (
        'EURUSD', str_tf_to_mt5_tf['M1'],
        datetime(2024, 1, 12, 21, 0, 1, tzinfo=UTC),
        datetime(2024, 1, 17, 21, tzinfo=UTC),
    )
    assert len(result) == 2


@pytest.mark.parametrize('timeframe', ['BAD', 'MN2', 999999])
def test_fetch_rejects_unknown_timeframe_before_initializing(configured_mt5, timeframe):
    """Should reject unknown timeframe labels and constants before opening MT5."""
    with pytest.raises(ValueError, match='timeframe|Unknown'):
        ingestion.fetch_mt_data_prices(
            'EURUSD', timeframe, '2024-01-01', '2024-01-02',
        )
    configured_mt5.initialize.assert_not_called()


@pytest.mark.parametrize('timeframe', [None, 1.5, True])
def test_fetch_rejects_invalid_timeframe_type(configured_mt5, timeframe):
    """Should reject timeframe values with unsupported Python types."""
    with pytest.raises(TypeError, match='timeframe'):
        ingestion.fetch_mt_data_prices(
            'EURUSD', timeframe, '2024-01-01', '2024-01-02',
        )
    configured_mt5.initialize.assert_not_called()


def test_fetch_accepts_raw_mt5_integer_timeframe(configured_mt5, rates):
    """Should accept a supported MT5 timeframe enum passed as an integer."""
    configured_mt5.copy_rates_range.return_value = rates
    result = ingestion.fetch_mt_data_prices(
        'EURUSD', str_tf_to_mt5_tf['H1'], '2024-01-01', '2024-01-02',
    )
    assert len(result) == 2
    assert configured_mt5.copy_rates_range.call_args.args[1] == str_tf_to_mt5_tf['H1']


@pytest.mark.parametrize(
    ('start', 'end'),
    [
        ('2024-01-02', '2024-01-01'),
        ('2024-01-01', '2024-01-01'),
    ],
)
def test_fetch_rejects_reversed_or_equal_dates_before_initializing(
    configured_mt5,
    start,
    end,
):
    """Should reject empty or reversed ranges before initializing MT5."""
    with pytest.raises(ValueError, match='end_date'):
        ingestion.fetch_mt_data_prices('EURUSD', 'H1', start, end)
    configured_mt5.initialize.assert_not_called()


@pytest.mark.parametrize('bad_date', [None, 20240101, object(), '2024/01/01'])
def test_fetch_rejects_invalid_date_inputs_before_initializing(configured_mt5, bad_date):
    """Should reject malformed or unsupported dates before initializing MT5."""
    with pytest.raises((TypeError, ValueError)):
        ingestion.fetch_mt_data_prices(
            'EURUSD', 'H1', bad_date, '2024-01-02',
        )
    configured_mt5.initialize.assert_not_called()


@pytest.mark.parametrize(
    ('days_before', 'error'),
    [(-1, ValueError), (1.5, TypeError), (True, TypeError), ('2', TypeError)],
)
def test_fetch_validates_days_before(configured_mt5, days_before, error):
    """Should require the warmup duration to be a non-negative integer."""
    with pytest.raises(error, match='days_before'):
        ingestion.fetch_mt_data_prices(
            'EURUSD', 'H1', '2024-01-01', '2024-01-02', days_before,
        )
    configured_mt5.initialize.assert_not_called()


def test_fetch_rejects_unknown_backend_before_initializing(configured_mt5):
    """Should reject unsupported dataframe backends before initializing MT5."""
    with pytest.raises(ValueError, match='backend'):
        ingestion.fetch_mt_data_prices(
            'EURUSD', 'H1', '2024-01-01', '2024-01-02', backend='numpy',
        )
    configured_mt5.initialize.assert_not_called()


def test_fetch_raises_and_shuts_down_when_initialization_fails(configured_mt5):
    """Should shut down MT5 when initialization reports failure."""
    configured_mt5.initialize.return_value = False
    with pytest.raises(RuntimeError, match='initialization failed'):
        ingestion.fetch_mt_data_prices('EURUSD', 'H1', '2024-01-01', '2024-01-02')
    configured_mt5.shutdown.assert_called_once_with()


def test_fetch_shuts_down_when_initialization_raises(configured_mt5):
    """Should shut down MT5 when initialization itself raises."""
    configured_mt5.initialize.side_effect = OSError('terminal unavailable')
    with pytest.raises(OSError, match='terminal unavailable'):
        ingestion.fetch_mt_data_prices('EURUSD', 'H1', '2024-01-01', '2024-01-02')
    configured_mt5.shutdown.assert_called_once_with()


def test_fetch_shuts_down_when_symbol_is_unknown(configured_mt5):
    """Should shut down MT5 when the requested symbol has no metadata."""
    configured_mt5.symbol_info.return_value = None
    with pytest.raises(RuntimeError, match='Symbol not found: EURUSD'):
        ingestion.fetch_mt_data_prices('EURUSD', 'H1', '2024-01-01', '2024-01-02')
    configured_mt5.shutdown.assert_called_once_with()


def test_fetch_shuts_down_when_symbol_lookup_raises(configured_mt5):
    """Should shut down MT5 when symbol metadata retrieval raises."""
    configured_mt5.symbol_info.side_effect = OSError('metadata unavailable')
    with pytest.raises(OSError, match='metadata unavailable'):
        ingestion.fetch_mt_data_prices('EURUSD', 'H1', '2024-01-01', '2024-01-02')
    configured_mt5.shutdown.assert_called_once_with()


def test_fetch_skips_empty_early_chunk_and_returns_later_data(
    configured_mt5,
    rates,
):
    """Should continue after an empty chunk when a later chunk contains bars."""
    configured_mt5.copy_rates_range.side_effect = [None, rates]
    result = ingestion.fetch_mt_data_prices(
        'EURUSD', 'M1', '2024-01-01', '2024-01-09',
    )
    assert len(result) == 2
    assert configured_mt5.copy_rates_range.call_count == 2
    configured_mt5.shutdown.assert_called_once_with()


def test_fetch_raises_only_after_all_chunks_are_empty(configured_mt5):
    """Should raise the no-data error only after checking the complete range."""
    configured_mt5.copy_rates_range.return_value = None
    with pytest.raises(RuntimeError, match='No data retrieved'):
        ingestion.fetch_mt_data_prices(
            'EURUSD', 'M1', '2024-01-01', '2024-01-09',
        )
    assert configured_mt5.copy_rates_range.call_count == 2
    configured_mt5.shutdown.assert_called_once_with()


def test_fetch_shuts_down_when_retrieval_raises(configured_mt5):
    """Should shut down MT5 when a chunk retrieval call raises."""
    configured_mt5.copy_rates_range.side_effect = OSError('connection lost')
    with pytest.raises(OSError, match='connection lost'):
        ingestion.fetch_mt_data_prices('EURUSD', 'H1', '2024-01-01', '2024-01-02')
    configured_mt5.shutdown.assert_called_once_with()


def test_fetch_rejects_malformed_rate_schema_and_shuts_down(configured_mt5):
    """Should reject records missing required fields and still shut down MT5."""
    configured_mt5.copy_rates_range.return_value = [{'time': 1704067200, 'open': 1.0}]
    with pytest.raises(ValueError, match='missing required columns'):
        ingestion.fetch_mt_data_prices('EURUSD', 'H1', '2024-01-01', '2024-01-02')
    configured_mt5.shutdown.assert_called_once_with()


def test_fetch_deduplicates_overlap_and_sorts_rates(configured_mt5, rates):
    """Should sort returned bars and keep one record for overlapping timestamps."""
    duplicate = {**rates[0], 'close': 999.0}
    configured_mt5.copy_rates_range.side_effect = [rates, [duplicate, rates[1]]]
    result = ingestion.fetch_mt_data_prices(
        'EURUSD', 'M1', '2024-01-01', '2024-01-09',
    )
    assert result['time'].to_list() == sorted(result['time'].to_list())
    assert result['time'].n_unique() == 2
    assert result['close'].to_list()[0] == rates[0]['close']
    configured_mt5.shutdown.assert_called_once_with()


def test_timeframe_constants_cover_all_supported_mt5_periods():
    """Should expose all supported timeframes with their MT5 enum values."""
    expected = {
        'M1': 1,
        'M2': 2,
        'M3': 3,
        'M4': 4,
        'M5': 5,
        'M6': 6,
        'M10': 10,
        'M12': 12,
        'M15': 15,
        'M20': 20,
        'M30': 30,
        'H1': 16385,
        'H2': 16386,
        'H3': 16387,
        'H4': 16388,
        'H6': 16390,
        'H8': 16392,
        'H12': 16396,
        'D1': 16408,
        'W1': 32769,
        'MN1': 49153,
    }
    assert str_tf_to_mt5_tf == expected
    if ingestion.mt5 is not None:
        for name, value in expected.items():
            assert getattr(ingestion.mt5, f'TIMEFRAME_{name}') == value


def test_ingestion_module_imports_without_optional_mt5_dependency():
    """Should allow importing without MT5 but guard actual collection."""
    script = '''
import sys

class BlockMetaTrader5:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "MetaTrader5":
            raise ModuleNotFoundError("simulated minimal install")
        return None

sys.meta_path.insert(0, BlockMetaTrader5())
from alpha_research.data.ingestion.mt5 import fetch_mt_data_prices
assert callable(fetch_mt_data_prices)
try:
    fetch_mt_data_prices("EURUSD", "H1", "2024-01-01", "2024-01-02")
except ImportError as exc:
    assert "alpha-research[mt5]" in str(exc)
else:
    raise AssertionError("MT5 call should require its optional dependency")
'''
    completed = subprocess.run(
        [sys.executable, '-c', script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
