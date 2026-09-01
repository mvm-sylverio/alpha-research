import pandas as pd
import polars as pl
import numpy as np


def _validate_df_type(df: pd.DataFrame | pl.DataFrame):
    """
    Validate a DataFrame type before usage. Only accepts Polars or Pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        df which will be validated.

    Raises
    ------
    TypeError
        If df is not pandas or polars type.
    """
    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        raise TypeError('DataFrame must be Pandas or Polars type.')


def _validate_df(
        df: pd.DataFrame | pl.DataFrame,
        required_cols: list[str],
        check_all_missing: bool = True,
) -> None:
    """
    Validate a Dataframe before usage.

    Includes variable type check, columns existence check, empty DataFrame
    check, and optionally an all-NaN/Nulls required-columns check.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        df which will be validated.
    required_cols : list[str]
        Columns required to exist in the df.
    check_all_missing : bool, default True
        Whether every required column must contain at least one non-missing
        value. Set to False for result frames where an all-missing diagnostic
        column is a valid, explicit outcome.

    Raises
    ------
    TypeError
        If df is not pandas or polars type.
    KeyError
        If columns in required_cols are not columns of the df.
    ValueError
        If DataFrame is empty or, when check_all_missing is True, required
        columns are missing all values.
    """
    # Check DataFrame type
    _validate_df_type(df)

    # Check for empty DataFrame
    if len(df) == 0:
        raise ValueError("DataFrame must not be empty.")

    # Check missing columns
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise KeyError(f'missing required columns in DataFrame: {missing_cols}.')

    if check_all_missing:
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


def _is_constant_series(series: pd.Series | pl.Series) -> bool:
    """
    Check whether a Series has zero or one distinct non-missing values.

    Parameters
    ----------
    series : pd.Series | pl.Series
        Series which will be checked for constant values.

    Returns
    -------
    bool
        True if the Series has at most one distinct non-missing value,
        False otherwise.

    Raises
    ------
    TypeError
        If series is not a Pandas or Polars Series.

    Notes
    -----
    NaN and null values are excluded before evaluating the number of
    distinct values. A Series with no non-missing values is considered
    constant.
    """
    if isinstance(series, pd.Series):
        return series.dropna().nunique() <= 1

    if isinstance(series, pl.Series):
        non_missing = series.drop_nulls()

        if non_missing.dtype.is_float():
            non_missing = non_missing.drop_nans()

        return non_missing.n_unique() <= 1

    raise TypeError("Unsupported Series type.")


def _select_columns(
        df: pd.DataFrame | pl.DataFrame,
        columns: list[str],
) -> pd.DataFrame | pl.DataFrame:
    """
    Select columns from a Pandas or Polars DataFrame.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame from which columns will be selected.
    columns : list[str]
        Names of the columns to select, in the desired output order.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        DataFrame containing only the requested columns and preserving the
        input backend.

    Raises
    ------
    TypeError
        If df is not a Pandas or Polars DataFrame.
    """
    _validate_df_type(df)

    if isinstance(df, pd.DataFrame):
        return df[columns]

    return df.select(columns)


def _validate_same_backend(
        left: pd.DataFrame | pl.DataFrame,
        right: pd.DataFrame | pl.DataFrame,
) -> None:
    """
    Validate that two DataFrames use the same supported backend.

    Parameters
    ----------
    left : pd.DataFrame | pl.DataFrame
        First DataFrame to validate.
    right : pd.DataFrame | pl.DataFrame
        Second DataFrame to validate.

    Returns
    -------
    None
        This function returns None when both inputs use the same backend.

    Raises
    ------
    TypeError
        If an input is not a Pandas or Polars DataFrame, or if the
        DataFrames use different backends.
    """
    # validate left and right DataFrame backends
    _validate_df_type(left)
    _validate_df_type(right)

    if isinstance(left, pd.DataFrame) != isinstance(right, pd.DataFrame):
        raise TypeError(
            "left and right arguments must use the same DataFrame backend."
        )


def _validate_unique_keys(
        df: pd.DataFrame | pl.DataFrame,
        keys: list[str],
        df_name: str,
) -> None:
    """
    Validate that a DataFrame has not duplicate key combinations.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame whose key columns will be checked for duplicates.
    keys : list[str]
        Columns that jointly identify an observation.
    df_name : str
        DataFrame name used in the error message.

    Returns
    -------
    None
        This function returns None when every key combination is unique.

    Raises
    ------
    ValueError
        If two or more rows share the same key combination.

    Notes
    -----
    This function does not check for duplicate values in
    each key column independently. It assumes df is a supported DataFrame and
    that keys exist; those validations should be performed upstream.
    """
    if isinstance(df, pd.DataFrame):
        has_duplicates = df.duplicated(subset=keys).any()
    else:
        has_duplicates = df.select(keys).is_duplicated().any()

    if has_duplicates:
        raise ValueError(
            f'{df_name} must contain unique {keys} key combinations.'
        )


def _validate_time_order(
        df: pd.DataFrame | pl.DataFrame,
        time_col: str,
) -> None:
    """
    Validate that a DataFrame has unique, increasingly ordered times.

    This validator does not sort the input. Time-series calculations must
    receive observations in chronological order so that causal operations do
    not silently use an invalid row order.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame containing the time column.
    time_col : str
        Name of the time column.

    Raises
    ------
    TypeError
        If df is not a Pandas or Polars DataFrame.
    KeyError
        If time_col is not a column of df.
    ValueError
        If time_col contains missing values, duplicates, or is not increasingly
        ordered.

    Notes
    -----
    This function validates the single-series temporal contract. Multi-asset
    callers should validate time order independently within each asset.
    """
    _validate_df(df, [time_col])
    _validate_unique_keys(df, [time_col], 'df')

    time_series = df[time_col]

    if isinstance(df, pd.DataFrame):
        if time_series.isna().any():
            raise ValueError(f'{time_col} must not contain missing values.')

        is_ordered = time_series.is_monotonic_increasing
    else:
        if time_series.is_null().any():
            raise ValueError(f'{time_col} must not contain missing values.')

        is_ordered = time_series.is_sorted()

    if not is_ordered:
        raise ValueError(f'{time_col} must be increasingly ordered.')


def _validate_positive_integer(value: int, name: str) -> None:
    """
    Validate that a named argument is a positive integer.

    Parameters
    ----------
    value : int
        Value to validate. Boolean values are rejected even though bool is an
        integer subclass in Python.
    name : str
        Argument name included in the error message.

    Returns
    -------
    None
        This function returns None when value is a valid positive integer.

    Raises
    ------
    ValueError
        If value is not an integer, is a boolean, or is less than one.
    """
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f'{name} must be a positive integer.')
