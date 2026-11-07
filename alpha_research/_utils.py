import math
from collections.abc import Iterable
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


def _has_missing_values(series: pd.Series | pl.Series) -> bool:
    """Check a series for nulls or native floating-point NaNs.

    Parameters
    ----------
    series : pd.Series | pl.Series
        Column to inspect, possibly empty.

    Returns
    -------
    bool
        Whether the series contains any missing values.

    Raises
    ------
    TypeError
        If the series backend is unsupported.
    """
    if isinstance(series, pd.Series):
        return bool(series.isna().any())
    if isinstance(series, pl.Series):
        return bool(series.is_null().any() or (series.dtype.is_float() and series.is_nan().any()))
    raise TypeError('Unsupported Series type.')


def _validate_key_values(
        df: pd.DataFrame | pl.DataFrame,
        keys: list[str],
        *,
        check_infinite: bool = True,
) -> None:
    """Validate that observation keys contain no missing or infinite values.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Non-empty observations containing the requested keys.
    keys : list[str]
        Key-column names. Uniqueness and temporal order are separate checks.
    check_infinite : bool, default True
        Whether to reject infinities as well as missing keys.

    Returns
    -------
    None
        Returns None when all key values are present and finite where numeric.

    Raises
    ------
    TypeError
        If df or the column specification has an unsupported type, or
        check_infinite is not boolean.
    KeyError
        If a key column is absent.
    ValueError
        If the frame is empty, a selected column is ambiguous, or a key
        contains missing or infinite values.
    """
    if type(check_infinite) is not bool:
        raise TypeError('check_infinite must be boolean.')
    _validate_df(df, keys, check_all_missing=False)
    for key in keys:
        series = df[key]
        if _has_missing_values(series):
            raise ValueError(f'{key} must not contain missing values.')
        if not check_infinite:
            continue
        if isinstance(series, pd.Series):
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
            has_infinite = series.dtype.is_float() and series.is_infinite().any()
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


def _validate_single_symbol(
        df: pd.DataFrame | pl.DataFrame,
        symbol_col: str,
) -> None:
    """Require exactly one non-missing asset in a temporal analysis.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Frame with a previously validated column schema.
    symbol_col : str
        Column identifying the asset.

    Returns
    -------
    None
        Returns None when all rows identify one asset.

    Raises
    ------
    ValueError
        If the column contains missing keys or has fewer or more than one asset.
    """
    symbols = df[symbol_col]
    n_symbols = symbols.nunique() if isinstance(df, pd.DataFrame) else symbols.n_unique()
    if _has_missing_values(symbols) or n_symbols != 1:
        raise ValueError(
            f'{symbol_col} must contain exactly one non-missing symbol for '
            'temporal association.'
        )


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
    normalized = _as_real_float(
        value,
        type_error=f'{name} must be numeric.',
        overflow_error=f'{name} must be finite and representable as a float.',
    )
    if not math.isfinite(normalized):
        raise ValueError(f'{name} must be finite.')
    return normalized


def _as_real_float(value: object, *, type_error: str, overflow_error: str) -> float:
    """Convert a real scalar to float while leaving NaN and infinity intact.

    Parameters
    ----------
    value : object
        Scalar to convert.
    type_error, overflow_error : str
        Messages used for an unsupported scalar or unrepresentable magnitude.

    Returns
    -------
    float
        The converted scalar. The caller decides how to handle non-finite data.

    Raises
    ------
    TypeError
        If value is not a real numeric scalar, including booleans and durations.
    ValueError
        If a finite value overflows the float representation.
    """
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_, np.timedelta64)):
        raise TypeError(type_error)
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(overflow_error) from error
    if isinstance(value, np.floating) and not math.isfinite(normalized) and np.isfinite(value):
        raise ValueError(overflow_error)
    return normalized


def _validate_probability(value: float, name: str) -> None:
    """Validate a probability in the open interval (0, 1).

    Parameters
    ----------
    value : float
        Python or NumPy integer or floating-point scalar to validate.
    name : str
        Name included in the validation error.

    Returns
    -------
    None
        Returns None when the probability is valid.

    Raises
    ------
    ValueError
        If value is not a supported scalar strictly between zero and one.
    """
    if (
            not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, (bool, np.bool_, np.timedelta64))
            or not 0 < value < 1
    ):
        raise ValueError(f'{name} must be strictly between 0 and 1.')


def _validate_random_state(random_state: int | None) -> None:
    """Validate an optional seed without drawing from or changing an RNG.

    Parameters
    ----------
    random_state : int | None
        Non-negative Python or NumPy integer seed, or None.

    Returns
    -------
    None
        Returns None when the seed is valid.

    Raises
    ------
    TypeError
        If random_state is neither an integer nor None.
    ValueError
        If the seed is negative.
    """
    if random_state is None:
        return
    if not isinstance(random_state, (int, np.integer)) or isinstance(random_state, (bool, np.bool_, np.timedelta64)):
        raise TypeError('random_state must be an integer or None.')
    if random_state < 0:
        raise ValueError('expected non-negative integer')


def _as_numpy_values(
        df: pd.DataFrame | pl.DataFrame,
        column: str,
) -> np.ndarray:
    """
    Return a column as a NumPy array without changing its logical values.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Source DataFrame containing column.
    column : str
        Name of the column to extract.

    Returns
    -------
    np.ndarray
        Values from column in input row order.

    Raises
    ------
    KeyError
        If column is not present in df.
    """
    if isinstance(df, pd.DataFrame):
        return df[column].to_numpy()

    return df.get_column(column).to_numpy()


def _asset_positions(symbols: Iterable[object] | pl.Series) -> list[list[int]]:
    """
    Return input-row positions grouped by symbol while preserving row order.

    Parameters
    ----------
    symbols : Iterable or pl.Series
        Asset identifiers in input-row order. Polars series are grouped
        natively to preserve identities that cannot round-trip through NumPy.

    Returns
    -------
    list[list[int]]
        One list of row positions for each distinct symbol. Positions within
        every asset list retain their original temporal order.

    Raises
    ------
    TypeError
        If an identifier in a Python iterable is not hashable.
    """
    if isinstance(symbols, pl.Series):
        return (
            pl.DataFrame({'symbol': symbols})
            .with_row_index('position')
            .group_by('symbol', maintain_order=True)
            .agg('position')['position']
            .to_list()
        )

    positions_by_symbol: dict[object, list[int]] = {}

    for position, symbol in enumerate(symbols):
        positions_by_symbol.setdefault(symbol, []).append(position)

    return list(positions_by_symbol.values())


def _validate_and_group_times(
        symbols: np.ndarray | pd.Series | pl.Series,
        times: np.ndarray | pd.Series | pl.Series,
        symbol_col: str,
        time_col: str,
) -> list[np.ndarray]:
    """
    Validate the per-symbol time contract and return input-order indices.

    Parameters
    ----------
    symbols : np.ndarray | pd.Series | pl.Series
        Asset identifiers in input row order.
    times : np.ndarray | pd.Series | pl.Series
        Observation times from the same backend. Series retain their native
        ordering semantics, including categorical order.
    symbol_col : str
        Symbol-column name used in validation error messages.
    time_col : str
        Time-column name used in validation error messages.

    Returns
    -------
    list[np.ndarray]
        One positional-index array per symbol, preserving input symbol order.

    Raises
    ------
    ValueError
        If arrays are not one-dimensional and aligned, symbols or times are
        missing, keys are duplicated, or times decrease within a symbol.
    TypeError
        If inputs do not share a supported backend or symbols are not hashable.
    """
    if isinstance(symbols, np.ndarray) and isinstance(times, np.ndarray):
        if symbols.ndim != 1 or times.ndim != 1:
            raise ValueError('symbols and times must be aligned one-dimensional arrays.')
        symbols, times = pd.Series(symbols), pd.Series(times)
    elif not (
            isinstance(symbols, pd.Series) and isinstance(times, pd.Series)
            or isinstance(symbols, pl.Series) and isinstance(times, pl.Series)
    ):
        raise TypeError('symbols and times must be arrays or series from the same backend.')
    if len(symbols) != len(times):
        raise ValueError('symbols and times must be aligned one-dimensional arrays.')
    if _has_missing_values(symbols):
        raise ValueError(f'{symbol_col} must not contain missing values.')

    if _has_missing_values(times):
        raise ValueError(f'{time_col} must not contain missing values.')

    groups = [np.asarray(indices, dtype=np.intp) for indices in _asset_positions(symbols)]

    # Check duplicates across all assets before reporting decreasing times.
    if isinstance(times, pd.Series):
        group_times = [times.iloc[indices] for indices in groups]
        has_duplicates = any(not series.is_unique for series in group_times)
        is_ordered = all(series.is_monotonic_increasing for series in group_times)
    else:
        group_times = [times.gather(indices) for indices in groups]
        has_duplicates = any(series.is_duplicated().any() for series in group_times)
        is_ordered = all(series.is_sorted() for series in group_times)
    if has_duplicates:
        raise ValueError(f'{time_col} must contain unique values within each {symbol_col}.')
    if not is_ordered:
        raise ValueError(f'{time_col} must be increasingly ordered within each {symbol_col}.')

    return groups


def _normalize_column_names(columns: str | list[str], *, type_error: str, empty_error: str) -> list[str]:
    """Normalize a column name or a non-empty list without changing its contents.

    Parameters
    ----------
    columns : str | list[str]
        Name or list of names in caller order.
    type_error, empty_error : str
        Messages for an invalid container and an empty list, respectively.

    Returns
    -------
    list[str]
        A one-element list for a string, or the original list.

    Raises
    ------
    TypeError
        If columns is neither a string nor a list.
    ValueError
        If columns is an empty list.
    """
    if isinstance(columns, str):
        return [columns]
    if not isinstance(columns, list):
        raise TypeError(type_error)
    if not columns:
        raise ValueError(empty_error)
    return columns


def _normalize_feature_cols(feature_cols: str | list[str]) -> list[str]:
    """
    Normalize one or more feature-column names to a non-empty list.

    Parameters
    ----------
    feature_cols : str | list[str]
        One feature-column name or a non-empty list of feature-column names.

    Returns
    -------
    list[str]
        Feature-column names in their original order.

    Raises
    ------
    TypeError
        If feature_cols is not a string or list of non-empty strings.
    ValueError
        If feature_cols is an empty list or contains duplicate names.
    """
    normalized = _normalize_column_names(
        feature_cols,
        type_error='feature_cols must be str or list[str].',
        empty_error='feature_cols must not be an empty list.',
    )
    if any(not isinstance(feature, str) or not feature for feature in normalized):
        raise TypeError('every feature in feature_cols must be a non-empty string.')
    if len(set(normalized)) != len(normalized):
        raise ValueError('feature_cols must not contain duplicate names.')
    return normalized


def _validate_elementwise_features(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
) -> list[str]:
    """Validate numeric finite-or-missing inputs for elementwise transforms.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame containing the requested feature columns.
    feature_cols : str | list[str]
        One feature column or a non-empty feature-column list.

    Returns
    -------
    list[str]
        Normalized feature names in caller order.

    Raises
    ------
    KeyError
        If a feature column is absent.
    TypeError
        If df or feature_cols has an unsupported type.
    ValueError
        If a non-missing feature is non-numeric or contains infinity.
    """
    normalized = _normalize_feature_cols(feature_cols)
    if isinstance(df, pd.DataFrame) and not df.columns.is_unique:
        raise ValueError('DataFrame must not contain duplicate column names.')
    _validate_df(df, normalized, check_all_missing=False)
    for feature in normalized:
        series = df[feature]
        if isinstance(df, pd.DataFrame):
            if not series.isna().all() and (
                    not pd.api.types.is_numeric_dtype(series)
                    or pd.api.types.is_bool_dtype(series)
                    or pd.api.types.is_complex_dtype(series)
            ):
                raise ValueError(f'{feature} must be real numeric and non-boolean.')
            if np.isinf(series.dropna().to_numpy(dtype=float)).any():
                raise ValueError(f'{feature} must contain only finite or missing values.')
        else:
            if not series.dtype.is_numeric() and series.dtype != pl.Null:
                raise ValueError(f'{feature} must be numeric.')
            non_missing = series.drop_nulls()
            if non_missing.dtype.is_float() and non_missing.is_infinite().any():
                raise ValueError(f'{feature} must contain only finite or missing values.')
    return normalized


def _polars_feature_values(df: pl.DataFrame, feature: str) -> pl.Expr:
    """Normalize missing Polars feature values for calculation.

    Parameters
    ----------
    df : pl.DataFrame
        Source frame.
    feature : str
        Selected numeric feature.

    Returns
    -------
    pl.Expr
        Expression converting NaN to null and Null dtype to Float64.

    Raises
    ------
    KeyError
        If the feature is missing.
    """
    if feature not in df.columns:
        raise KeyError(feature)
    value = pl.col(feature)
    if df.schema[feature] == pl.Null:
        value = value.cast(pl.Float64)
    return value.fill_nan(None)


def _validate_grouped_features(df, feature_cols, symbol_col, time_col):
    """Validate feature values and observation keys for grouped transforms.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Input observations.
    feature_cols : str | list[str]
        Feature names to transform.
    symbol_col, time_col : str
        Observation key columns.

    Returns
    -------
    list[str]
        Validated feature names in input order.

    Raises
    ------
    TypeError, KeyError, ValueError
        If the input, feature values, or observation keys are invalid.
    """
    features = _validate_elementwise_features(df, feature_cols)
    _validate_key_values(df, [symbol_col, time_col], check_infinite=False)
    _validate_unique_keys(df, [time_col, symbol_col], 'df')
    return features


def _validate_output_columns(df, features: list[str], suffix: str) -> None:
    """Reject generated column names that already exist in the input.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Validated input frame.
    features : list[str]
        Selected feature names.
    suffix : str
        Suffix appended to each feature name.

    Returns
    -------
    None
        Returns None when output names are available.

    Raises
    ------
    ValueError
        If a generated output column is already present or repeated.
    """
    outputs = [f'{feature}_{suffix}' for feature in features]
    collisions = [name for name in outputs if name in df.columns]
    if collisions or len(outputs) != len(set(outputs)):
        raise ValueError(f'transformation output columns already exist or repeat: {collisions}.')


def _normalize_covariates(covariates: str | list[str]) -> list[str]:
    """
    Normalize one or more covariate names into a validated list.

    Parameters
    ----------
    covariates : str | list[str]
        One covariate name or a non-empty list of unique covariate names.

    Returns
    -------
    list[str]
        Covariate names in the order supplied by the caller.

    Raises
    ------
    TypeError
        If covariates is neither a string nor a list of strings.
    ValueError
        If covariates is empty or contains duplicate names.
    """
    covariates = _normalize_column_names(
        covariates,
        type_error='covariates must be a string or a list of strings.',
        empty_error='covariates must not be empty.',
    )
    if not all(isinstance(covariate, str) for covariate in covariates):
        raise TypeError('covariates must be a string or a list of strings.')
    if any(not covariate.strip() for covariate in covariates):
        raise ValueError('covariate names must not be empty.')
    if len(set(covariates)) != len(covariates):
        raise ValueError('covariates must not contain duplicates.')

    return covariates
