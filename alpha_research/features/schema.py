import numpy as np
import pandas as pd
import polars as pl

from alpha_research._utils import (
    _validate_df,
    _validate_df_type,
    _validate_key_values,
    _validate_same_backend,
    _validate_unique_keys,
)

__all__ = ['get_feature_name', 'join_feature_target_frames']


def _join_key_kind(series: pd.Series | pl.Series) -> str:
    """Identify a supported scalar identity domain without coercing values.

    Parameters
    ----------
    series : pd.Series | pl.Series
        Non-empty key column without missing values.

    Returns
    -------
    str
        Semantic key domain, independent of integer width or string storage.

    Raises
    ------
    TypeError
        If the series has an unsupported, mixed, or non-scalar key type.
    """
    if isinstance(series, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            return 'datetime'
        if pd.api.types.is_timedelta64_dtype(series.dtype):
            return 'duration'
        values = series.astype(object) if isinstance(series.dtype, pd.CategoricalDtype) else series
        kind = pd.api.types.infer_dtype(values, skipna=False)
        if kind in ('integer', 'floating', 'boolean', 'string', 'bytes', 'date', 'time'):
            return kind
    elif isinstance(series, pl.Series):
        dtype = series.dtype
        if dtype.is_integer():
            return 'integer'
        if dtype.is_float():
            return 'floating'
        if dtype == pl.Boolean:
            return 'boolean'
        if dtype == pl.String or dtype == pl.Categorical or isinstance(dtype, pl.Enum):
            return 'string'
        if dtype == pl.Binary:
            return 'bytes'
        if dtype == pl.Date:
            return 'date'
        if dtype == pl.Time:
            return 'time'
        if isinstance(dtype, pl.Datetime):
            return 'datetime'
        if isinstance(dtype, pl.Duration):
            return 'duration'
    raise TypeError('join keys must contain homogeneous supported scalar values.')


def _align_join_keys(
        feature_df: pd.DataFrame | pl.DataFrame,
        target_df: pd.DataFrame | pl.DataFrame,
        keys: list[str],
) -> tuple[pd.DataFrame | pl.DataFrame, pd.DataFrame | pl.DataFrame]:
    """Build losslessly compatible join keys without modifying either input.

    Parameters
    ----------
    feature_df, target_df : pd.DataFrame | pl.DataFrame
        Non-empty frames using the same backend and containing unique columns.
    keys : list[str]
        Non-empty, distinct join-key columns. Numeric domains must match;
        integer and float widths may differ if a lossless common representation
        exists. Aware datetimes are compared as instants in the feature timezone,
        at the finer precision.

    Returns
    -------
    tuple[pd.DataFrame | pl.DataFrame, pd.DataFrame | pl.DataFrame]
        Independent key frames with matching join types. Original key dtypes
        are restored by the caller when constructing the public result.

    Raises
    ------
    TypeError
        If backends or key domains differ, a key dtype is unsupported, naive
        and aware datetimes are mixed, or normalization cannot be lossless.
    KeyError
        If a key column is absent.
    ValueError
        If frames are empty, columns are ambiguous, the key list is empty or
        repeated, or key values are missing or infinite.
    """
    _validate_same_backend(feature_df, target_df)
    _validate_key_values(feature_df, keys)
    _validate_key_values(target_df, keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('join keys must be non-empty and distinct.')
    is_pandas = isinstance(feature_df, pd.DataFrame)
    left = feature_df[keys].copy() if is_pandas else feature_df.select(keys)
    right = target_df[keys].copy() if is_pandas else target_df.select(keys)
    for key in keys:
        left_kind = _join_key_kind(left[key])
        if left_kind != _join_key_kind(right[key]):
            raise TypeError(f'incompatible join-key types for {key}.')
        try:
            if left_kind in ('integer', 'floating'):
                if left[key].dtype == right[key].dtype:
                    continue
                common = np.promote_types(left[key].to_numpy().dtype, right[key].to_numpy().dtype)
                if left_kind == 'integer' and (common.kind not in 'iuO' or (not is_pandas and common.kind == 'O')):
                    raise TypeError('integer keys require a lossless common integer dtype.')
                if is_pandas:
                    left[key], right[key] = left[key].astype(common), right[key].astype(common)
                else:
                    prefix = 'Float' if common.kind == 'f' else 'Int' if common.kind == 'i' else 'UInt'
                    dtype = getattr(pl, f'{prefix}{common.itemsize * 8}')
                    left, right = (frame.with_columns(pl.col(key).cast(dtype)) for frame in (left, right))
            elif left_kind in ('datetime', 'duration'):
                originals = left[key], right[key]
                left_dtype, right_dtype = left[key].dtype, right[key].dtype
                if left_kind == 'datetime':
                    left_tz = getattr(left_dtype, 'tz', None) if is_pandas else left_dtype.time_zone
                    right_tz = getattr(right_dtype, 'tz', None) if is_pandas else right_dtype.time_zone
                    if (left_tz is None) != (right_tz is None):
                        raise TypeError('naive and timezone-aware keys cannot be mixed.')
                    if left_tz is not None:
                        if is_pandas:
                            right[key] = right[key].dt.tz_convert(left_tz)
                        else:
                            right = right.with_columns(pl.col(key).dt.convert_time_zone(left_tz))
                if is_pandas:
                    left_unit = left_dtype.unit if isinstance(left_dtype, pd.DatetimeTZDtype) else np.datetime_data(left_dtype)[0]
                    right_unit = right_dtype.unit if isinstance(right_dtype, pd.DatetimeTZDtype) else np.datetime_data(right_dtype)[0]
                else:
                    left_unit, right_unit = left_dtype.time_unit, right_dtype.time_unit
                unit = max((left_unit, right_unit), key=('s', 'ms', 'us', 'ns').index)
                if is_pandas:
                    dtype = (pd.DatetimeTZDtype(unit=unit, tz=left_tz) if left_kind == 'datetime' and left_tz is not None
                             else f'{"datetime64" if left_kind == "datetime" else "timedelta64"}[{unit}]')
                    left[key], right[key] = left[key].astype(dtype), right[key].astype(dtype)
                    if any(not frame[key].astype(original.dtype).equals(original)
                           for frame, original in zip((left, right), originals)):
                        raise ValueError('temporal normalization changed key values.')
                else:
                    dtype = pl.Datetime(unit, left_tz) if left_kind == 'datetime' else pl.Duration(unit)
                    left, right = (frame.with_columns(pl.col(key).cast(dtype)) for frame in (left, right))
                    if any(not frame[key].cast(original.dtype).equals(original)
                           for frame, original in zip((left, right), originals)):
                        raise ValueError('temporal normalization changed key values.')
            elif is_pandas:
                left[key], right[key] = left[key].astype(object), right[key].astype(object)
            elif left_kind == 'string':
                left, right = (frame.with_columns(pl.col(key).cast(pl.String)) for frame in (left, right))
        except (TypeError, ValueError, OverflowError, pl.exceptions.PolarsError) as error:
            raise TypeError(f'incompatible join-key types for {key}; normalization must be lossless.') from error
    return left, right


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
        *,
        check_all_missing: bool = True,
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
    check_all_missing : bool, default True
        Whether to reject value columns containing only missing values.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Joined DataFrame with columns [time_col, symbol_col, feature_col,
        target_col], excluding unmatched and missing observations. Rows follow
        feature_df order, with a fresh RangeIndex for Pandas. Key and value
        dtypes are preserved from their respective input columns, including
        when no usable pairs remain.

    Raises
    ------
    TypeError
        If either input is not a Pandas or Polars DataFrame, or if the inputs
        use different backends, names are invalid, or key types are incompatible.
    KeyError
        If a requested key or value column is missing.
    ValueError
        If an input DataFrame is empty, a value column contains only missing
        values, column roles overlap, key values are missing or infinite, or an
        input contains duplicate [time_col, symbol_col] pairs.

    Notes
    -----
    The one-to-one constraint prevents duplicate observations from silently
    multiplying rows during the join. Columns outside the requested keys and
    value columns are intentionally excluded from the result.
    Keys must be present even when check_all_missing is False. Integers and
    floats, dates and datetimes, and naive and aware datetimes are distinct key
    domains. Supported scalar keys also include strings, binary values,
    booleans, clock times, and durations. Categorical keys match by label.
    Numeric widths may differ if their common dtype is lossless; signed/unsigned
    combinations requiring float promotion are rejected. Aware datetimes in
    different timezones match by instant; timestamp precision is never reduced
    for matching. Infinities in value columns are retained; callers determine
    their own numeric-value policy.
    """
    columns = [time_col, symbol_col, feature_col, target_col]
    if any(not isinstance(column, str) or not column for column in columns):
        raise TypeError('join column names must be non-empty strings.')
    if len(set(columns)) != len(columns):
        raise ValueError('time, symbol, feature, and target columns must have distinct names.')
    keys = [time_col, symbol_col]

    # check DataFrames validity - general, same backend and uniqueness of keys
    _validate_df(feature_df, keys + [feature_col], check_all_missing=check_all_missing)
    _validate_df(target_df, keys + [target_col], check_all_missing=check_all_missing)

    left, right = _align_join_keys(feature_df, target_df, keys)

    _validate_unique_keys(feature_df, keys, 'feature_df')
    _validate_unique_keys(target_df, keys, 'target_df')

    position_col = '__alpha_research_position'
    while position_col in columns:
        position_col += '_'
    if isinstance(feature_df, pd.DataFrame):
        left[position_col] = np.arange(len(left))
        right[target_col] = target_df[target_col].reset_index(drop=True).array
        joined = left.merge(right, on=keys, how='inner', validate='one_to_one', sort=False)
        joined = joined.sort_values(position_col).reset_index(drop=True)
        result = feature_df.iloc[joined[position_col].to_numpy()][keys + [feature_col]].reset_index(drop=True)
        result[target_col] = joined[target_col]
        return result.dropna(subset=[feature_col, target_col]).reset_index(drop=True)

    joined = (
        left.with_row_index(position_col)
        .join(
            right.with_columns(target_df[target_col]),
            on=keys,
            how='inner',
            validate='1:1',
        )
        .sort(position_col)
    )
    result = feature_df.select([
        pl.col(column).gather(joined[position_col]) for column in keys + [feature_col]
    ]).with_columns(joined[target_col])
    result = result.drop_nulls([feature_col, target_col])
    float_values = [column for column in (feature_col, target_col) if result.schema[column].is_float()]
    return result.drop_nans(float_values) if float_values else result
