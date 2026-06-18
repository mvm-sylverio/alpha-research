import pandas as pd
import polars as pl

from alpha_research._utils import _validate_df

__all__ = ['price_to_sma_ratio', 'sma_crossover']


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
