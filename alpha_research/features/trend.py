import polars as pl


def compute_sma_feature(df_prices: pl.DataFrame, window: int, collumn='close'):
    """
    Already receives the df_prices with the correct symbol and timeframe.
    Computes the simple moving average (SMA) in the chosen 'window' and 'collumn'.
    """
    df = df_prices.with_columns([
        pl.col(collumn).rolling_mean(window).alias("value")
    ])

    df = df.select([
        "symbol",
        "timeframe",
        "time",
        pl.lit("trend").alias("feature_group"),
        pl.lit(f"sma_{window}").alias("feature_name"),
        "value"
    ])

    return df.drop_nulls()
