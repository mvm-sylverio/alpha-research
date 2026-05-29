import polars as pl


def compute_normal_return(df_prices: pl.DataFrame, horizon: int):

    df = df_prices.with_columns([
        (pl.col("close") / pl.col("close").shift(horizon) - 1).alias("value")
    ])

    df = df.select([
        "symbol",
        "timeframe",
        "time",
        pl.lit("returns").alias("feature_group"),
        pl.lit(f"ret_{horizon}").alias("feature_name"),
        "value"
    ])

    return df.drop_nulls()


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
