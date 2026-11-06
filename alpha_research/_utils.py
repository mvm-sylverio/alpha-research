import math
from numbers import Real

import numpy as np
import pandas as pd
import polars as pl


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
        Non-empty column names required to exist in the df. An empty list is
        allowed when only frame-level validation is needed.
    check_all_missing : bool, default True
        Whether every required column must contain at least one non-missing
        value. Set to False for result frames where an all-missing diagnostic
        column is a valid, explicit outcome.

    Raises
    ------
    TypeError
        If df is not pandas or polars type, required_cols is not a list of
        non-empty strings, or check_all_missing is not boolean.
    KeyError
        If columns in required_cols are not columns of the df.
    ValueError
        If DataFrame is empty or, when check_all_missing is True, required
        columns are missing all values, or a required column name is ambiguous.
    """
    if type(check_all_missing) is not bool:
        raise TypeError('check_all_missing must be boolean.')

    # Check DataFrame type
    _validate_df_type(df)
    if not isinstance(required_cols, list) or any(
            not isinstance(column, str) or not column for column in required_cols
    ):
        raise TypeError('required_cols must be a list of non-empty column-name strings.')

    # Check for empty DataFrame
    if len(df) == 0:
        raise ValueError("DataFrame must not be empty.")

    # Check missing columns
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise KeyError(f'missing required columns in DataFrame: {missing_cols}.')

    if isinstance(df, pd.DataFrame):
        ambiguous = df.columns[df.columns.duplicated()].intersection(required_cols).to_list()
        if ambiguous:
            raise ValueError(f'required column names must be unique in DataFrame: {ambiguous}.')

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


def _validate_key_values(
        df: pd.DataFrame | pl.DataFrame,
        keys: list[str],
) -> None:
    """Validate that observation keys contain no missing or infinite values.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Non-empty observations containing the requested keys.
    keys : list[str]
        Key-column names. Uniqueness and temporal order are separate checks.

    Returns
    -------
    None
        Returns None when all key values are present and finite where numeric.

    Raises
    ------
    TypeError
        If df or the column specification has an unsupported type.
    KeyError
        If a key column is absent.
    ValueError
        If the frame is empty, a selected column is ambiguous, or a key
        contains missing or infinite values.
    """
    _validate_df(df, keys, check_all_missing=False)
    for key in keys:
        series = df[key]
        if isinstance(series, pd.Series):
            has_missing = series.isna().any()
            # Object and categorical keys may also contain numeric scalars.
            values = series.to_numpy()
            if values.dtype.kind in 'fc':
                has_infinite = np.isinf(values).any()
            elif values.dtype.kind == 'O' and not pd.api.types.is_datetime64_any_dtype(series.dtype):
                has_infinite = any(
                    isinstance(value, (float, complex, np.inexact)) and not np.isfinite(value)
                    for value in values
                )
            else:
                has_infinite = False
        else:
            has_missing = series.is_null().any()
            has_infinite = False
            if series.dtype.is_float():
                has_missing = has_missing or series.is_nan().any()
                has_infinite = series.is_infinite().any()
        if has_missing:
            raise ValueError(f'{key} must not contain missing values.')
        if has_infinite:
            raise ValueError(f'{key} must contain only finite values.')


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
        If time_col contains missing or infinite values, duplicates, or is not
        increasingly ordered.

    Notes
    -----
    This function validates the single-series temporal contract. Multi-asset
    callers should validate time order independently within each asset.
    """
    _validate_key_values(df, [time_col])
    _validate_unique_keys(df, [time_col], 'df')

    time_series = df[time_col]

    if isinstance(df, pd.DataFrame):
        is_ordered = time_series.is_monotonic_increasing
    else:
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
    if (not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_, np.timedelta64)) or value <= 0):
        raise ValueError(f'{name} must be a positive integer.')


def _validate_finite_number(value: Real, name: str) -> float:
    """Return one validated finite real-valued parameter.

    Parameters
    ----------
    value : numbers.Real
        Candidate numeric value.
    name : str
        Parameter name used in validation errors.

    Returns
    -------
    float
        Finite value converted to float.

    Raises
    ------
    TypeError
        If value is not a real number or is a boolean.
    ValueError
        If value is NaN, infinite, or not representable as a finite float.
    """
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_, np.timedelta64)):
        raise TypeError(f'{name} must be numeric.')
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(f'{name} must be finite and representable as a float.') from error
    if not math.isfinite(normalized):
        raise ValueError(f'{name} must be finite.')
    return normalized
