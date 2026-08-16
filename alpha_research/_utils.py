import pandas as pd
import polars as pl


def _validate_df(
        df: pd.DataFrame | pl.DataFrame,
        required_cols: list[str],
) -> None:
    """
    Validate a Dataframe before usage. Includes variable type check
    and columns existence check.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        df which will be validated.
    required_cols : list[str]
        Columns required to exist in the df

    Raises
    ------
    TypeError
        If df is not pandas or polars type.
    KeyError
        If columns in required_cols are not columns of the df.
    """
    # Check DataFrame type
    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        raise TypeError('DataFrame must be Pandas or Polars type.')

    # Check for empty DataFrame
    if len(df) == 0:
        raise ValueError("DataFrame must not be empty.")

    # Check missing columns
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise KeyError(f'missing required columns in DataFrame: {missing_cols}.')

    # Check None/NaN values on all columns
    cols_with_all_missing = [col for col in required_cols if _is_all_missing(df[col])]

    if cols_with_all_missing:
        raise ValueError(f'required columns in DataFrame missing all values: {cols_with_all_missing}.')


def _is_all_missing(series) -> bool:
    """
    Check if a Series of a DataFrame has only NaN or Null values.

    Parameters
    ----------
    series : pd.Series | pl.Series
        Series which will be checked for NaN or Null values.

    Returns
    -------
    bool
        True if the series has only NaN or Null values, False if not.

    Raises
    ------
    TypeError
        if Series is not Polars or Pandas type.

    Notes
    -----
    Assumes non-empty series — empty check should be performed upstream
    in _validate_df before calling this function.
    Polars treats NaN and null as distinct missing values; both are
    considered missing here. NaN only applies to floating-point columns.
    """
    if isinstance(series, pl.Series):
        if series.dtype.is_float():
            return (series.is_null() | series.is_nan()).all()

        return bool(series.is_null().all())

    if isinstance(series, pd.Series):
        return bool(series.isna().all())

    raise TypeError("Unsupported Series type.")


def _select_columns(
        df: pd.DataFrame | pl.DataFrame,
        columns: list[str],
) -> pd.DataFrame | pl.DataFrame:
    """
    Select columns from a Pandas or Polars DataFrame.

    Raises
    ------
    TypeError
        If df is not a Pandas or Polars DataFrame.
    """
    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        raise TypeError('DataFrame must be Pandas or Polars type.')

    if isinstance(df, pd.DataFrame):
        return df[columns]

    return df.select(columns)


def _validate_same_backend(
        left: pd.DataFrame | pl.DataFrame,
        right: pd.DataFrame | pl.DataFrame,
) -> None:
    """
    Validate that two DataFrames use the same supported backend.

    Raises
    ------
    TypeError
        If an input is not a Pandas or Polars DataFrame, or if the
        DataFrames use different backends.
    """
    if not isinstance(left, (pd.DataFrame, pl.DataFrame)):
        raise TypeError('left must be a Pandas or Polars DataFrame.')

    if not isinstance(right, (pd.DataFrame, pl.DataFrame)):
        raise TypeError('right must be a Pandas or Polars DataFrame.')

    if isinstance(left, pd.DataFrame) != isinstance(right, pd.DataFrame):
        raise TypeError(
            "feature_fn and target_fn must return the same DataFrame "
            "backend as the input df."
        )
