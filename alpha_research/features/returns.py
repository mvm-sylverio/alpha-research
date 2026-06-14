import pandas as pd
import polars as pl

from alpha_research._utils import _validate_df

def compute_simple_returns(
        df: pd.DataFrame | pl.DataFrame,
        horizon: int,
        price_col: str = 'close',
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Compute the backwards simple cross-sectional returns.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Wide DataFrame (usually a full OHLCV) with at least the columns price_col,
        symbol_col and time_col.
    horizon : int
        Backwards horizon in which the simple return will be computed.
    price_col : str
        Price column in which the simple return will be computed. Usually the close
        price column.
    symbol_col : str
        Column which contains the cross-sectional asset names of the dataset.
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
    """
    _validate_df(df, [time_col, symbol_col, price_col])

    if isinstance(df, pd.DataFrame):
        result = df[[time_col, symbol_col]].copy()
        previous_price = df.groupby(symbol_col)[price_col].shift(horizon)
        result[f'simple_ret_{horizon}'] = df[price_col] / previous_price - 1
        return result

    else:  # pl.Dataframe type
        return df.select([
            time_col,
            symbol_col,
            (pl.col(price_col) / pl.col(price_col).shift(horizon) - 1)
            .over(symbol_col)
            .alias(f'simple_ret_{horizon}')])

def compute_log_return(df_prices: pl.DataFrame, horizon: int):

    assert horizon > 0, 'The horizon needs to be positive.'

    df = df_prices.with_columns([
        (pl.col("close").log() - pl.col("close").shift(horizon).log()).over('symbol').alias("value")
    ])

    df = df.select([
        "symbol",
        "timeframe",
        "time",
        pl.lit("returns").alias("feature_group"),
        pl.lit(f"logret_{horizon}").alias("feature_name"),
        "value"
    ])

    return df.drop_nulls()
