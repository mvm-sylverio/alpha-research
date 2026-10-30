import pandas as pd
import polars as pl
import numpy as np

from alpha_research.features.targets import _valid_price_expression, _validate_return_input

__all__ = ['simple_returns', 'log_returns']


def simple_returns(
        df: pd.DataFrame | pl.DataFrame,
        horizon: int,
        price_col: str = 'close',
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Compute the backwards simple returns on all symbols.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Wide DataFrame (usually a full OHLCV) with at least the columns price_col,
        symbol_col and time_col.
    horizon : int
        Positive number of previous observations per symbol.
    price_col : str
        Price column in which the simple return will be computed. Usually the close
        price column.
    symbol_col : str
        Column which contains the asset names of the dataset.
    time_col : str
        Time column of the dataset.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        DataFrame with the time_col, symbol_col and the simple_ret_{horizon} with
        the computed backwards simple cross-sectional returns.

    Raises
    ------
    KeyError
        If time_col, symbol_col, price_col are not columns of the df.
    TypeError
        If df is not pandas or polars type.
    ValueError
        If horizon is not a positive integer or keys are missing, duplicated,
        or out of chronological order within a symbol.

    Notes
    -----
    Non-positive or non-finite prices yield missing returns for affected pairs.
    A horizon at least as long as a symbol's history yields missing values.
    """
    _validate_return_input(df, horizon, price_col, symbol_col, time_col)

    if isinstance(df, pd.DataFrame):
        result = df[[time_col, symbol_col]].copy()
        price = pd.to_numeric(df[price_col], errors='raise').astype(float)
        price = price.where(np.isfinite(price) & (price > 0))
        previous_price = price.groupby(df[symbol_col]).shift(horizon)
        result[f'simple_ret_{horizon}'] = (price / previous_price - 1).replace([np.inf, -np.inf], np.nan)
        return result

    else:  # pl.Dataframe type
        price = _valid_price_expression(price_col)
        previous_price = price.shift(horizon).over(symbol_col)
        return df.select([
            time_col,
            symbol_col,
            (price / previous_price - 1)
            .replace([float('inf'), float('-inf')], None)
            .alias(f'simple_ret_{horizon}')])


def log_returns(
        df: pd.DataFrame | pl.DataFrame,
        horizon: int,
        price_col: str = 'close',
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Compute the backwards natural logarithm returns on all symbols.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Wide DataFrame (usually a full OHLCV) with at least the columns price_col,
        symbol_col and time_col.
    horizon : int
        Positive number of previous observations per symbol.
    price_col : str
        Price column in which the logarithm return will be computed. Usually the close
        price column.
    symbol_col : str
        Column which contains the asset names of the dataset.
    time_col : str
        Time column of the dataset.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        DataFrame with the time_col, symbol_col and the log_ret_{horizon} with
        the computed backwards logarithm returns.

    Raises
    ------
    KeyError
        If time_col, symbol_col, price_col are not columns of the df.
    TypeError
        If df is not pandas or polars type.
    ValueError
        If horizon is not a positive integer or keys are missing, duplicated,
        or out of chronological order within a symbol.

    Notes
    -----
    Non-positive or non-finite prices yield missing returns for affected pairs.
    A horizon at least as long as a symbol's history yields missing values.
    """
    _validate_return_input(df, horizon, price_col, symbol_col, time_col)

    if isinstance(df, pd.DataFrame):
        result = df[[time_col, symbol_col]].copy()
        price = pd.to_numeric(df[price_col], errors='raise').astype(float)
        price = price.where(np.isfinite(price) & (price > 0))
        previous_price = price.groupby(df[symbol_col]).shift(horizon)
        result[f'log_ret_{horizon}'] = np.log(price) - np.log(previous_price)
        return result

    else:  # pl.Dataframe type
        log_price = _valid_price_expression(price_col).log()
        previous_log_price = log_price.shift(horizon).over(symbol_col)
        return df.select([
            time_col,
            symbol_col,
            (log_price - previous_log_price)
            .alias(f'log_ret_{horizon}')])
