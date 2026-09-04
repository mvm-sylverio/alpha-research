import pandas as pd
import polars as pl
import numpy as np

from alpha_research._utils import _validate_df, _validate_positive_integer
from alpha_research.features.volatility import (
    _asset_positions,
    _average_true_range_values,
    _feature_result_frame,
    _wilder_smoothing_by_symbol,
)

__all__ = [
    'price_to_sma_ratio',
    'sma_crossover',
    'average_directional_index',
    'adx',
]


def price_to_sma_ratio(
        df: pd.DataFrame | pl.DataFrame,
        window: int,
        price_col: str = 'close',
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Compute the price to SMA ratio on all symbols.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Wide DataFrame (usually a full OHLCV) with at least the columns price_col,
        symbol_col and time_col.
    window : int
        Window of the SMA which will be used to compute the price_to_sma ratio.
    price_col : str
        Price column which is used to compute the sma and the price_to_sma ratio.
        Usually the close price column.
    symbol_col : str
        Column which contains the asset names of the dataset.
    time_col : str
        Time column of the dataset.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        DataFrame with the time_col, symbol_col and the price_to_sma_{window} with
        the ratio of the price column to the sma.

    Raises
    ------
    KeyError
        If time_col, symbol_col, price_col are not columns of the df.
    TypeError
        If df is not pandas or polars type.
    """
    _validate_df(df, [time_col, symbol_col, price_col])

    if isinstance(df, pd.DataFrame):
        sma = df.groupby(symbol_col)[price_col].transform(lambda x: x.rolling(window).mean())
        result = df[[time_col, symbol_col]].copy()
        result[f'price_to_sma_ratio_{window}'] = df[price_col] / sma - 1
        return result

    else:  # polars DataFrame
        return df.select([
            time_col,
            symbol_col,
            (pl.col(price_col) / pl.col(price_col).rolling_mean(window).over(symbol_col) - 1)
            .alias(f'price_to_sma_ratio_{window}')
        ])


def sma_crossover(
        df: pd.DataFrame | pl.DataFrame,
        fast_window: int,
        slow_window: int,
        price_col: str = 'close',
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Compute the ratio of the fast SMA to the slow SMA on all symbols.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Wide DataFrame (usually a full OHLCV) with at least the columns price_col,
        symbol_col and time_col.
    fast_window : int
        Window of the fast SMA.
    slow_window : int
        Window of the slow SMA.
    price_col : str
        Price column which is used to compute the fast and the slow SMA.
        Usually the close price column.
    symbol_col : str
        Column which contains the asset names of the dataset.
    time_col : str
        Time column of the dataset.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        DataFrame with the time_col, symbol_col and the
        sma_{fast_window}_crossover_sma_{slow_window} with
        the ratio of the fast SMA to the slow SMA.

    Raises
    ------
    KeyError
        If time_col, symbol_col, price_col are not columns of the df.
    TypeError
        If df is not pandas or polars type.
    """
    _validate_df(df, [time_col, symbol_col, price_col])

    if fast_window >= slow_window:
        raise ValueError(f'fast_window ({fast_window}) must be smaller than slow_window ({slow_window}).')

    if isinstance(df, pd.DataFrame):
        fast_sma = df.groupby(symbol_col)[price_col].transform(lambda x: x.rolling(fast_window).mean())
        slow_sma = df.groupby(symbol_col)[price_col].transform(lambda x: x.rolling(slow_window).mean())
        result = df[[time_col, symbol_col]].copy()
        result[f'sma_{fast_window}_crossover_sma_{slow_window}'] = fast_sma / slow_sma - 1
        return result
    else:  # polars DataFrame
        return df.select([
            time_col,
            symbol_col,
            (pl.col(price_col).rolling_mean(fast_window).over(symbol_col) /
             pl.col(price_col).rolling_mean(slow_window).over(symbol_col) - 1)
            .alias(f'sma_{fast_window}_crossover_sma_{slow_window}')
        ])


def average_directional_index(
        df: pd.DataFrame | pl.DataFrame,
        window: int,
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Compute Wilder's Average Directional Index (ADX) independently for every asset.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        OHLCV-like DataFrame containing high_col, low_col, close_col, and the
        time and symbol keys. Rows must be ordered chronologically within each
        asset; this function does not sort them.
    window : int
        Wilder smoothing period used for directional movement, True Range, and
        the final Directional Index average.
    high_col, low_col, close_col : str
        OHLC column names used to calculate directional movement and ATR.
    symbol_col : str, default 'symbol'
        Asset identifier column used to isolate calculations.
    time_col : str, default 'time'
        Temporal key preserved in the result.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        DataFrame with time_col, symbol_col, and ``adx_{window}``. ADX is a
        directionless trend-strength measure on an approximate 0 to 100 scale.

    Raises
    ------
    KeyError
        If a required column is absent.
    TypeError
        If df is not a Pandas or Polars DataFrame.
    ValueError
        If window is not a positive integer.

    Notes
    -----
    Positive and negative directional movements are calculated separately and
    normalized by Wilder ATR. ADX then applies a second Wilder smoothing to
    the Directional Index. Its first value requires two full smoothing periods
    after the beginning of an asset's valid OHLC history. ADX measures trend
    strength only; price-to-SMA and SMA-crossover features can supply direction.
    """
    _validate_df(df, [time_col, symbol_col, high_col, low_col, close_col])
    _validate_positive_integer(window, 'window')

    high = np.asarray(df[high_col].to_numpy(), dtype=float)
    low = np.asarray(df[low_col].to_numpy(), dtype=float)
    close = np.asarray(df[close_col].to_numpy(), dtype=float)
    symbols = df[symbol_col].to_list()
    plus_directional_movement = np.full(len(df), np.nan, dtype=float)
    minus_directional_movement = np.full(len(df), np.nan, dtype=float)

    for positions in _asset_positions(symbols):
        previous_high = np.nan
        previous_low = np.nan

        for position in positions:
            current_high = high[position]
            current_low = low[position]

            if (
                    np.isfinite(current_high)
                    and np.isfinite(current_low)
                    and np.isfinite(previous_high)
                    and np.isfinite(previous_low)
            ):
                upward_move = current_high - previous_high
                downward_move = previous_low - current_low
                plus_directional_movement[position] = (
                    upward_move
                    if upward_move > downward_move and upward_move > 0
                    else 0.0
                )
                minus_directional_movement[position] = (
                    downward_move
                    if downward_move > upward_move and downward_move > 0
                    else 0.0
                )

            previous_high = current_high
            previous_low = current_low

    atr = _average_true_range_values(high, low, close, symbols, window)
    smoothed_plus_dm = _wilder_smoothing_by_symbol(
        plus_directional_movement,
        symbols,
        window,
    )
    smoothed_minus_dm = _wilder_smoothing_by_symbol(
        minus_directional_movement,
        symbols,
        window,
    )
    plus_di = np.divide(
        100 * smoothed_plus_dm,
        atr,
        out=np.full(len(df), np.nan, dtype=float),
        where=np.isfinite(atr) & (atr != 0),
    )
    minus_di = np.divide(
        100 * smoothed_minus_dm,
        atr,
        out=np.full(len(df), np.nan, dtype=float),
        where=np.isfinite(atr) & (atr != 0),
    )
    directional_index = np.divide(
        100 * np.abs(plus_di - minus_di),
        plus_di + minus_di,
        out=np.full(len(df), np.nan, dtype=float),
        where=(plus_di + minus_di) != 0,
    )
    adx = _wilder_smoothing_by_symbol(directional_index, symbols, window)

    return _feature_result_frame(
        df,
        adx,
        f'adx_{window}',
        time_col,
        symbol_col,
    )


def adx(
        df: pd.DataFrame | pl.DataFrame,
        window: int,
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Alias for function average_directional_index.
    See average_directional_index() for full documentation.
    """
    return average_directional_index(
        df=df,
        window=window,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        symbol_col=symbol_col,
        time_col=time_col,
    )
