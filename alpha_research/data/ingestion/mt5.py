"""MetaTrader 5 market-data ingestion utilities."""

from datetime import datetime, timedelta
from typing import Literal

import MetaTrader5 as mt5
import pandas as pd
import polars as pl

from alpha_research.data.market.constants import str_tf_to_mt5_tf


def ensure_datetime(value: str | datetime) -> datetime:
    """
    Convert a supported date value to a ``datetime`` object.

    Parameters
    ----------
    value : str or datetime
        Date in ``YYYY-MM-DD`` format or an existing ``datetime`` object.

    Returns
    -------
    datetime
        Normalized date value.
    """
    if isinstance(value, str):
        return datetime.strptime(value, "%Y-%m-%d")
    return value


def fetch_mt_data_prices(
        symbol: str,
        timeframe: str,
        start_date: str | datetime,
        end_date: str | datetime,
        days_before: int = 0,
        backend: Literal['pandas', 'polars'] = 'polars',
) -> pd.DataFrame | pl.DataFrame:
    """
    Fetch OHLCV price data from MetaTrader 5 (MT5).

    Data is fetched in chunks to avoid MT5's range-size limits. Chunk results
    are deduplicated by timestamp and returned in chronological order.

    Parameters
    ----------
    symbol : str
        Symbol from which data is imported.
    timeframe : str
        MT5 timeframe name, such as ``'H1'`` or ``'D1'``.
    start_date, end_date : str or datetime
        Inclusive start and exclusive end of the requested range. Strings
        must use the ``YYYY-MM-DD`` format.
    days_before : int, default 0
        Number of days to prepend to ``start_date``. This is useful for
        computing indicators that need a warm-up period.
    backend : {'pandas', 'polars'}, default 'polars'
        DataFrame backend for the returned data. The default preserves the
        original Polars return type.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Data with columns ``time``, ``open``, ``high``, ``low``, ``close``,
        ``volume``, ``spread``, and ``spread_pts``.

    Raises
    ------
    ValueError
        If ``backend`` is not ``'pandas'`` or ``'polars'``.
    RuntimeError
        If MT5 cannot be initialized, the symbol does not exist, or no data
        is retrieved.
    """

    if backend not in ('pandas', 'polars'):
        raise ValueError("backend must be either 'pandas' or 'polars'.")

    # Initialize MT5 app
    if not mt5.initialize():
        raise RuntimeError("MetaTrader5 initialization failed")

    # ---------------------------------------------------------------
    # 1. Normalize inputs
    # ---------------------------------------------------------------
    # Convert timeframe string into MT5 constant
    if isinstance(timeframe, str):
        timeframe = str_tf_to_mt5_tf[timeframe]

    # Convert date strings into datetime objects
    start_date = ensure_datetime(start_date)
    end_date = ensure_datetime(end_date)

    # Rolls back start_date. Ensures rolling indicators are fully defined at the true start date
    start_date = start_date - timedelta(days=days_before)

    # ---------------------------------------------------------------
    # 2. Retrieve symbol metadata
    # ---------------------------------------------------------------
    info = mt5.symbol_info(symbol)
    if info is None:
        mt5.shutdown()
        raise RuntimeError(f"Symbol not found: {symbol}")

    # point is required to convert spread points into price units
    point = info.point

    # ------------------------------------------------------------------
    # 3. Helper: Convert MT5 rates to Polars df
    # ------------------------------------------------------------------
    # Inner function, only used here
    def _rates_to_polars(given_rates):

        if given_rates is None or len(given_rates) == 0:
            return None

        return_df = pl.DataFrame(given_rates)

        # Epoch to datetime conversion. MT5 returns 'time' in seconds since epoch (UTC)
        # epoch = number of seconds since 01/01/1970 - 00:00:00 UTC
        t = pl.from_epoch(pl.col('time'), time_unit='s')

        # Rename columns and compute spread in price units
        return_df = (
            return_df.rename({
                "tick_volume": "volume",
                "real_volume": "real_vol",
                "spread": "spread_pts"
            })
            .with_columns([
                t.dt.replace_time_zone(None).alias("time"),
                (pl.col("spread_pts") * point).alias("spread")
            ])
            .select("time", "open", "high", "low", "close", "volume", "spread", "spread_pts")
            .sort("time")
        )

        return return_df

    # ------------------------------------------------------------------
    # 4. Data retrieval, chunked if necessary
    # ------------------------------------------------------------------
    # MT5 limits the size of data provided
    def _get_chunk_days(tf):
        # safe defaults
        if tf in [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M3, mt5.TIMEFRAME_M5]:
            return 5
        elif tf in [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M30]:
            return 15
        elif tf in [mt5.TIMEFRAME_H1]:
            return 30
        elif tf in [mt5.TIMEFRAME_H4]:
            return 90
        elif tf in [mt5.TIMEFRAME_D1]:
            return 365
        else:
            return 30


    chunk_days = _get_chunk_days(timeframe)

    current_start = start_date
    chunks = []
    df = None

    while current_start < end_date:

        chunk_end = min(
            current_start + timedelta(days=chunk_days),
            end_date,
        )

        rates = mt5.copy_rates_range(symbol, timeframe, current_start, chunk_end)

        df_chunk = _rates_to_polars(rates)

        if df_chunk is not None:
            chunks.append(df_chunk)

        # advances without overlapping candles
        current_start = chunk_end + timedelta(seconds=1)

        if not chunks:
            mt5.shutdown()
            raise RuntimeError(f"No data retrieved for {symbol}")

        # concatenate chunks and remove potential overlapping timestamps
        df = pl.concat(chunks)
        df = df.unique(subset=["time"], keep="first").sort("time")

    # ------------------------------------------------------------------
    # 5. Close MT5 session and logging
    # ------------------------------------------------------------------
    mt5.shutdown()

    if backend == 'pandas':
        return df.to_pandas()

    return df
