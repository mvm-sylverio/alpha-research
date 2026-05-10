import polars as pl


def compute_fwd_return(df_prices: pl.DataFrame, horizon: int):
    assert horizon > 0, 'The horizon needs to be positive.'

    df = df_prices.with_columns([
        (-pl.col("close").log() + pl.col("close").shift(-horizon).log()).over('symbol').alias("value")
    ])

    df = df.select([
        "symbol",
        "timeframe",
        "time",
        pl.lit("fwd_ret").alias("feature_group"),
        pl.lit(f"fwd_ret_{horizon}").alias("feature_name"),
        "value"
    ])

    return df.drop_nulls()
