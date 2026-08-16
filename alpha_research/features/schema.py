import pandas as pd
import polars as pl

from alpha_research._utils import (
    _validate_df,
    _validate_df_type,
    _validate_same_backend,
    _validate_unique_keys
)


__all__ = ['get_feature_name', 'join_feature_target_frames']


def get_feature_name(
        df: pd.DataFrame | pl.DataFrame,
        time_col: str = 'time',
        symbol_col: str = 'symbol',
) -> str:
    """
    Return the single value column name from a feature or target DataFrame.

    A feature or target DataFrame is expected to contain the time and symbol
    key columns plus exactly one additional value column. The key column names
    can be customized for DataFrames that use a different schema.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Feature or target DataFrame.
    time_col : str, default 'time'
        Name of the time key column.
    symbol_col : str, default 'symbol'
        Name of the symbol key column.

    Returns
    -------
    str
        Name of the intended single column other than time_col and symbol_col.

    Raises
    ------
    TypeError
        If df is not a Pandas or Polars DataFrame.
    ValueError
        If df has zero or more than one value column.

    Notes
    -----
    This function identifies the value column from the DataFrame schema. It
    does not validate the presence or contents of the key columns.
    """
    _validate_df_type(df)

    columns = [column for column in df.columns if column not in (time_col, symbol_col)]

    if len(columns) != 1:
        raise ValueError(
            f"Expected exactly one feature or target column, found {columns}."
        )

    return columns[0]


def join_feature_target_frames(
        feature_df: pd.DataFrame | pl.DataFrame,
        target_df: pd.DataFrame | pl.DataFrame,
        feature_col: str,
        target_col: str,
        time_col: str,
        symbol_col: str,
) -> pd.DataFrame | pl.DataFrame:
    """
    Join a feature and target DataFrame on time and symbol keys.

    The function selects only the key and requested value columns from each
    input, applies an inner one-to-one join, then removes observations with a
    null or NaN feature or target value. The result preserves the backend of
    the input DataFrames.

    Parameters
    ----------
    feature_df : pd.DataFrame | pl.DataFrame
        DataFrame containing feature observations.
    target_df : pd.DataFrame | pl.DataFrame
        DataFrame containing target observations.
    feature_col : str
        Name of the feature value column in feature_df.
    target_col : str
        Name of the target value column in target_df.
    time_col : str
        Name of the time join key in both DataFrames.
    symbol_col : str
        Name of the symbol join key in both DataFrames.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Joined DataFrame with columns [time_col, symbol_col, feature_col,
        target_col], excluding unmatched and missing observations.

    Raises
    ------
    TypeError
        If either input is not a Pandas or Polars DataFrame, or if the inputs
        use different backends.
    KeyError
        If a requested key or value column is missing.
    ValueError
        If an input DataFrame is empty, a value column contains only missing
        values, or an input contains duplicate [time_col, symbol_col] pairs.

    Notes
    -----
    The one-to-one constraint prevents duplicate observations from silently
    multiplying rows during the join. Columns outside the requested keys and
    value columns are intentionally excluded from the result.
    """
    keys = [time_col, symbol_col]

    # check DataFrames validity - general, same backend and uniqueness of keys
    _validate_df(feature_df, keys + [feature_col])
    _validate_df(target_df, keys + [target_col])

    _validate_same_backend(feature_df, target_df)

    _validate_unique_keys(feature_df, keys, 'feature_df')
    _validate_unique_keys(target_df, keys, 'target_df')

    if isinstance(feature_df, pd.DataFrame):
        return (
            feature_df[keys + [feature_col]]
            .merge(
                target_df[keys + [target_col]],
                on=keys,
                how='inner',
                validate='one_to_one',
            )
            .dropna(subset=[feature_col, target_col])
        )

    return (
        feature_df
        .select(keys + [feature_col])
        .join(
            target_df.select(keys + [target_col]),
            on=keys,
            how='inner',
            validate='1:1',
        )
        .drop_nulls([feature_col, target_col])
        .drop_nans([feature_col, target_col])
    )
