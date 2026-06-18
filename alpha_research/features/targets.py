import polars as pl
import pandas as pd

from alpha_research._utils import _validate_df

__all__ = ['fwd_returns']


def fwd_returns(
        df: pd.DataFrame | pl.DataFrame,
        horizon: int,
        price_col: str = 'close',
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Compute the forward returns on all symbols.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Wide DataFrame (usually a full OHLCV) with at least the columns price_col,
        symbol_col and time_col.
    horizon : int
        Forward horizon in which the forward returns will be computed.
    price_col : str
        Price column in which the forward returns will be computed. Usually the close
        price column.
    symbol_col : str
        Column which contains the asset names of the dataset.
    time_col : str
        Time column of the dataset.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        DataFrame with the time_col, symbol_col and the fwd_ret_{horizon} with
        the computed forward cross-sectional returns.

    Raises
    ------
    KeyError
        If time_col, symbol_col, price_col are not columns of the df.
    TypeError
        If df is not pandas or polars type.
    """
    _validate_df(df, [time_col, symbol_col, price_col])

    if isinstance(df, pd.DataFrame):
        result = df[[time_col, symbol_col]].copy()
        fwd_price = df.groupby(symbol_col)[price_col].shift(-horizon)
        result[f'fwd_ret_{horizon}'] = fwd_price / df[price_col] - 1
        return result

    else:  # pl.Dataframe type
        return df.select([
            time_col,
            symbol_col,
            (pl.col(price_col).shift(-horizon) / pl.col(price_col) - 1).over(symbol_col)
            .alias(f'fwd_ret_{horizon}')])
