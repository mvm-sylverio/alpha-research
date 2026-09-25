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
    """
    Normalize a supported date value to a timezone-aware UTC datetime.

    Parameters
    ----------
    value : str or datetime
        Date in ``YYYY-MM-DD`` format or a datetime. Naive values are
        interpreted as UTC; timezone-aware values are converted to UTC.
    name : str
        Name of the input parameter, used in validation error messages.

    Returns
    -------
    datetime
        The date represented as a timezone-aware UTC datetime.

    Raises
    ------
    TypeError
        If ``value`` is neither a supported string nor a datetime.
    ValueError
        If a string does not match the ``YYYY-MM-DD`` format.
    """
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
    """
    Return the optional MetaTrader 5 client.

    Parameters
    ----------
    None

    Returns
    -------
    module
        The imported ``MetaTrader5`` module.

    Raises
    ------
    ImportError
        If the optional MT5 dependency is not installed or cannot be imported.
    """
    if mt5 is None:
        raise ImportError(
            'MT5 ingestion requires the optional dependency. '
            'Install alpha-research[mt5].',
        )
    return mt5


def _resolve_timeframe(timeframe: str | int) -> tuple[int, str]:
    """
    Resolve a timeframe label or MT5 enum to its enum value and public name.

    Parameters
    ----------
    timeframe : str or int
        Supported timeframe label, such as ``'H1'``, or its MT5 integer enum.

    Returns
    -------
    tuple[int, str]
        The MT5 enum value and its corresponding public timeframe label.

    Raises
    ------
    TypeError
        If ``timeframe`` is not a string or integer enum.
    ValueError
        If the label or integer enum is not supported.
    """
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
    """
    Choose the request chunk size for a timeframe.

    Parameters
    ----------
    timeframe_name : str
        Public MT5 timeframe label.

    Returns
    -------
    int
        Number of calendar days per MT5 request chunk.

    Raises
    ------
    None
    """
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
    """
    Convert MT5 rate records to the normalized UTC Polars schema.

    MT5 candle times are epoch seconds in UTC. The conversion attaches the
    UTC timezone without shifting those instants.

    Parameters
    ----------
    given_rates : tabular records or None
        Rate records returned by the MT5 client. Empty input is allowed.
    point : float
        Price-point size from the selected symbol metadata, used to convert
        ``spread_pts`` into price units.

    Returns
    -------
    pl.DataFrame or None
        Sorted bars with UTC-aware ``time`` and columns ``time``, ``open``,
        ``high``, ``low``, ``close``, ``volume``, ``spread`` and
        ``spread_pts``; ``None`` for an empty chunk.

    Raises
    ------
    ValueError
        If the records cannot be converted to a table or omit required MT5
        fields.
    """
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
    """
    Fetch OHLCV price data from MetaTrader 5 (MT5).

    Data is fetched in chunks to accommodate MT5 request limits. Empty chunks
    are skipped, overlapping timestamps are deduplicated, and the final data
    is sorted chronologically. Date-only strings and naive datetimes are
    interpreted as UTC; timezone-aware datetimes are converted to UTC before
    being passed to MT5. MT5 returns candle times as UTC epoch seconds, and
    the output ``time`` column retains the UTC timezone for both backends.
    The MT5 client is shut down after every attempted initialization.

    Parameters
    ----------
    symbol : str
        MT5 symbol from which prices are requested.
    timeframe : str or int
        Supported timeframe label, such as ``'H1'``, or its MT5 integer enum.
    start_date, end_date : str or datetime
        Request bounds. Strings must use ``YYYY-MM-DD`` and denote midnight
        UTC. Naive datetimes are interpreted as UTC; aware datetimes are
        converted to UTC. MT5 returns bars whose open times are greater than
        or equal to ``start_date`` and less than or equal to ``end_date``.
    days_before : int, default 0
        Number of calendar days to prepend to ``start_date`` for warmup.
    backend : {'pandas', 'polars'}, default 'polars'
        DataFrame backend for the returned data.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Chronologically sorted price data with UTC-aware ``time`` and columns
        ``time``, ``open``, ``high``, ``low``, ``close``, ``volume``,
        ``spread`` and ``spread_pts``. ``volume`` is MT5 tick volume.

    Raises
    ------
    ImportError
        If the optional MT5 dependency is unavailable.
    TypeError
        If a date, timeframe or ``days_before`` has an unsupported type.
    ValueError
        If the backend, timeframe, date format or date range is invalid, or
        if ``days_before`` is negative.
    RuntimeError
        If MT5 initialization fails, symbol metadata is unavailable, or no
        bars are returned for the complete requested range.
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
