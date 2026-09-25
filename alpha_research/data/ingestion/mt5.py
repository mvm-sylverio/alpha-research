"""MetaTrader 5 market-data ingestion utilities."""

from datetime import UTC, datetime, timedelta
from typing import Literal

import pandas as pd
import polars as pl

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - exercised in a subprocess
    mt5 = None

from alpha_research.data.market.constants import str_tf_to_mt5_tf

__all__ = ['fetch_mt_data_prices']


def _ensure_datetime(value: str | datetime, name: str) -> datetime:
    """Return a UTC-aware datetime; naive inputs are explicitly interpreted as UTC."""
    if isinstance(value, str):
        try:
            value = datetime.strptime(value, '%Y-%m-%d').replace(tzinfo=UTC)
        except ValueError as exc:
            raise ValueError(f'{name} must use the YYYY-MM-DD format.') from exc
    elif not isinstance(value, datetime):
        raise TypeError(f'{name} must be a YYYY-MM-DD string or datetime.')

    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=UTC)
    else:
        value = value.astimezone(UTC)
    return value


def _require_mt5():
    if mt5 is None:
        raise ImportError(
            'MT5 ingestion requires the optional dependency. '
            'Install alpha-research[mt5].',
        )
    return mt5


def _resolve_timeframe(timeframe: str | int) -> tuple[int, str]:
    if isinstance(timeframe, str):
        try:
            return str_tf_to_mt5_tf[timeframe], timeframe
        except KeyError as exc:
            supported = ', '.join(str_tf_to_mt5_tf)
            raise ValueError(
                f'Unknown timeframe {timeframe!r}. Supported values: {supported}.',
            ) from exc

    if isinstance(timeframe, bool) or not isinstance(timeframe, int):
        raise TypeError('timeframe must be a supported string or MT5 integer constant.')

    for name, value in str_tf_to_mt5_tf.items():
        if value == timeframe:
            return timeframe, name
    raise ValueError(f'Unknown MT5 timeframe constant: {timeframe!r}.')


def _chunk_days(timeframe_name: str) -> int:
    if timeframe_name in {'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12'}:
        return 5
    if timeframe_name in {'M15', 'M20', 'M30'}:
        return 15
    if timeframe_name in {'H1', 'H2', 'H3'}:
        return 30
    if timeframe_name in {'H4', 'H6', 'H8', 'H12'}:
        return 90
    if timeframe_name == 'D1':
        return 365
    return 365


def _rates_to_polars(given_rates, point: float) -> pl.DataFrame | None:
    if given_rates is None or len(given_rates) == 0:
        return None

    try:
        rates = pl.DataFrame(given_rates)
    except Exception as exc:
        raise ValueError('MT5 rates must be a tabular record array.') from exc

    required = {'time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread'}
    missing = required.difference(rates.columns)
    if missing:
        raise ValueError(f'MT5 rates are missing required columns: {sorted(missing)}.')

    time_expr = pl.from_epoch(pl.col('time'), time_unit='s').dt.replace_time_zone('UTC')
    return (
        rates.rename({'tick_volume': 'volume', 'spread': 'spread_pts'})
        .with_columns(
            time_expr.alias('time'),
            (pl.col('spread_pts') * point).alias('spread'),
        )
        .select('time', 'open', 'high', 'low', 'close', 'volume', 'spread', 'spread_pts')
        .sort('time')
    )


def fetch_mt_data_prices(
        symbol: str,
        timeframe: str | int,
        start_date: str | datetime,
        end_date: str | datetime,
        days_before: int = 0,
        backend: Literal['pandas', 'polars'] = 'polars',
) -> pd.DataFrame | pl.DataFrame:
    """Fetch UTC-aware OHLCV bars from MetaTrader 5.

    Date-only strings and naive datetimes are interpreted as UTC. Aware
    datetimes are converted to UTC before they are passed to MT5. MT5 returns
    candle times as UTC epoch seconds; the returned ``time`` column retains
    the UTC timezone for both supported DataFrame backends.

    Empty chunks are skipped, and the request fails only if the complete range
    contains no bars. MT5 is shut down after every attempted initialization.
    """
    if backend not in ('pandas', 'polars'):
        raise ValueError("backend must be either 'pandas' or 'polars'.")
    if not isinstance(days_before, int) or isinstance(days_before, bool):
        raise TypeError('days_before must be a non-negative integer.')
    if days_before < 0:
        raise ValueError('days_before must be a non-negative integer.')

    mt5_module = _require_mt5()
    timeframe_value, timeframe_name = _resolve_timeframe(timeframe)
    start_utc = _ensure_datetime(start_date, 'start_date')
    end_utc = _ensure_datetime(end_date, 'end_date')
    if end_utc <= start_utc:
        raise ValueError('end_date must be later than start_date.')

    start_utc -= timedelta(days=days_before)
    chunk_size = _chunk_days(timeframe_name)

    try:
        if not mt5_module.initialize():
            raise RuntimeError('MetaTrader5 initialization failed.')

        info = mt5_module.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f'Symbol not found: {symbol}')

        chunks: list[pl.DataFrame] = []
        current_start = start_utc
        while current_start < end_utc:
            chunk_end = min(current_start + timedelta(days=chunk_size), end_utc)
            rates = mt5_module.copy_rates_range(
                symbol,
                timeframe_value,
                current_start,
                chunk_end,
            )
            chunk = _rates_to_polars(rates, info.point)
            if chunk is not None:
                chunks.append(chunk)
            current_start = chunk_end + timedelta(seconds=1)

        if not chunks:
            raise RuntimeError(f'No data retrieved for {symbol}')

        result = (
            pl.concat(chunks)
            .unique(subset=['time'], keep='first')
            .sort('time')
        )
        if backend == 'pandas':
            return result.to_pandas()
        return result
    finally:
        mt5_module.shutdown()
