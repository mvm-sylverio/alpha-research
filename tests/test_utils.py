import numpy as np
import pandas as pd
import polars as pl
import pytest

# Imports that should not be inspected
# noinspection PyProtectedMember
from alpha_research._utils import (
    _is_all_missing,
    _is_constant_series,
    _select_columns,
    _validate_df,
    _validate_finite_number,
    _validate_key_values,
    _validate_positive_integer,
    _validate_same_backend,
    _validate_time_order,
    _validate_unique_keys,
)


# ------------------------------------------------------
# fixtures
# ------------------------------------------------------
@pytest.fixture
def pd_all_nan():
    return pd.Series([np.nan, np.nan, np.nan])

@pytest.fixture
def pd_some_nan():
    return pd.Series([1.0, np.nan, 3.0])

@pytest.fixture
def pd_no_nan():
    return pd.Series([1.0, 2.0, 3.0])

@pytest.fixture
def pd_all_none():
    return pd.Series([None, None, None])

@pytest.fixture
def pl_float_all_nan():
    return pl.Series([float('nan'), float('nan')], dtype=pl.Float64)

@pytest.fixture
def pl_float_all_null():
    return pl.Series([None, None], dtype=pl.Float64)

@pytest.fixture
def pl_float_mixed_nan_null():
    return pl.Series([float('nan'), None], dtype=pl.Float64)

@pytest.fixture
def pl_float_some_valid():
    return pl.Series([1.0, float('nan'), None], dtype=pl.Float64)

@pytest.fixture
def pl_str_all_null():
    return pl.Series([None, None], dtype=pl.String)

@pytest.fixture
def pl_str_some_valid():
    return pl.Series(['a', None], dtype=pl.String)


# ------------------------------------------------------
# _validate_df_type
# ------------------------------------------------------
def test_validate_dataframe_accepts_pandas():
    """Should not raise for valid pandas DataFrame."""
    df = pd.DataFrame({'a': [1], 'b': [2]})
    _validate_df(df, required_cols=['a', 'b'])

def test_validate_dataframe_accepts_polars():
    """Should not raise for valid polars DataFrame."""
    df = pl.DataFrame({'a': [1], 'b': [2]})
    _validate_df(df, required_cols=['a', 'b'])

def test_validate_dataframe_invalid_type_raises():
    """Should raise TypeError for unsupported types."""
    with pytest.raises(TypeError):
        _validate_df([[1, 2]], required_cols=['a'])


# ------------------------------------------------------
# _validate_df
# ------------------------------------------------------
def test_validate_dataframe_missing_column_raises():
    """Should raise KeyError for missing required columns."""
    df = pd.DataFrame({'a': [1]})
    with pytest.raises(KeyError, match="missing required"):
        _validate_df(df, required_cols=['a', 'b'])

# all missing values in columns
def test_validate_df_all_missing_non_required_col_does_not_raise():
    """Should not raise when all-missing column is not in required_cols."""
    df = pd.DataFrame({'a': [1.0, 2.0], 'b': [np.nan, np.nan]})
    _validate_df(df, required_cols=['a'])

def test_validate_df_all_missing_reports_correct_column():
    """Error message should name the column with all missing values."""
    df = pd.DataFrame({'a': [np.nan, np.nan], 'b': [1.0, 2.0]})
    with pytest.raises(ValueError, match="'a'"):
        _validate_df(df, required_cols=['a', 'b'])

def test_validate_df_multiple_all_missing_reports_all():
    """Error message should name all columns with all missing values."""
    df = pd.DataFrame({'a': [np.nan, np.nan], 'b': [np.nan, np.nan], 'c': [1.0, 2.0]})
    with pytest.raises(ValueError, match="'a'"):
        _validate_df(df, required_cols=['a', 'b', 'c'])


@pytest.mark.parametrize('df', [
    pd.DataFrame({'a': [np.nan, np.nan]}),
    pl.DataFrame({'a': [float('nan'), None]}),
])
def test_validate_df_can_allow_all_missing_required_columns(df):
    """Should still validate type and schema when result diagnostics are all missing."""
    _validate_df(df, required_cols=['a'], check_all_missing=False)


@pytest.mark.parametrize('check_all_missing', [None, 0, 1, 'false'])
def test_validate_df_rejects_non_boolean_missing_flag(check_all_missing):
    """Should reject ambiguous values for the all-missing validation flag."""
    df = pd.DataFrame({'a': [1.0]})

    with pytest.raises(TypeError, match='check_all_missing must be boolean'):
        _validate_df(
            df,
            required_cols=['a'],
            check_all_missing=check_all_missing,
        )


# ------------------------------------------------------
# _is_all_missing
# ------------------------------------------------------
# pandas
def test_is_all_missing_pandas_all_nan(pd_all_nan):
    """Should return True for all-NaN pandas Series."""
    assert _is_all_missing(pd_all_nan) is True

def test_is_all_missing_pandas_some_nan(pd_some_nan):
    """Should return False when pandas Series has valid values."""
    assert _is_all_missing(pd_some_nan) is False

def test_is_all_missing_pandas_no_nan(pd_no_nan):
    """Should return False for pandas Series with no missing values."""
    assert _is_all_missing(pd_no_nan) is False

def test_is_all_missing_pandas_all_none(pd_all_none):
    """Should return True for pandas Series with all None."""
    assert _is_all_missing(pd_all_none) is True

# polars float
def test_is_all_missing_polars_float_all_nan(pl_float_all_nan):
    """Should return True for polars float Series with all NaN."""
    assert _is_all_missing(pl_float_all_nan) is True

def test_is_all_missing_polars_float_all_null(pl_float_all_null):
    """Should return True for polars float Series with all null."""
    assert _is_all_missing(pl_float_all_null) is True

def test_is_all_missing_polars_float_mixed(pl_float_mixed_nan_null):
    """Should return True for polars float Series with NaN and null mixed."""
    assert _is_all_missing(pl_float_mixed_nan_null) is True

def test_is_all_missing_polars_float_some_valid(pl_float_some_valid):
    """Should return False for polars float Series with at least one valid value."""
    assert _is_all_missing(pl_float_some_valid) is False

# polars non-float
def test_is_all_missing_polars_str_all_null(pl_str_all_null):
    """Should return True for polars non-float Series with all null."""
    assert _is_all_missing(pl_str_all_null) is True

def test_is_all_missing_polars_str_some_valid(pl_str_some_valid):
    """Should return False for polars non-float Series with valid values."""
    assert _is_all_missing(pl_str_some_valid) is False

# edge cases
def test_is_all_missing_invalid_type_raises():
    """Should raise TypeError for unsupported types."""
    with pytest.raises(TypeError):
        _is_all_missing([1, 2, 3])


# ------------------------------------------------------
# _is_constant_series
# ------------------------------------------------------
def test_is_constant_series_pandas_constant_values():
    """Should return True for a Pandas Series with constant values."""
    series = pd.Series([1.0, 1.0, 1.0])

    assert _is_constant_series(series) is True


def test_is_constant_series_pandas_non_constant_values():
    """Should return False for a Pandas Series with distinct values."""
    series = pd.Series([1.0, 2.0, 1.0])

    assert _is_constant_series(series) is False


def test_is_constant_series_polars_ignores_nan_and_null_values():
    """Should ignore NaN and null values when checking a Polars Series."""
    series = pl.Series([1.0, float('nan'), None, 1.0])

    assert _is_constant_series(series) is True


def test_is_constant_series_polars_non_constant_values():
    """Should return False for a Polars Series with distinct values."""
    series = pl.Series([1.0, 2.0, None])

    assert _is_constant_series(series) is False


def test_is_constant_series_invalid_type_raises():
    """Should raise TypeError for an unsupported Series type."""
    with pytest.raises(TypeError, match='Unsupported Series type'):
        _is_constant_series([1.0, 1.0])


# ------------------------------------------------------
# _select_columns
# ------------------------------------------------------
def test_select_columns_pandas_returns_requested_columns():
    """Should select the requested columns from a pandas DataFrame."""
    df = pd.DataFrame({'time': [1], 'symbol': ['A'], 'feature': [0.1]})

    result = _select_columns(df, ['symbol', 'feature'])

    assert list(result.columns) == ['symbol', 'feature']
    assert result.to_dict(orient='list') == {'symbol': ['A'], 'feature': [0.1]}


def test_select_columns_polars_returns_requested_columns():
    """Should select the requested columns from a polars DataFrame."""
    df = pl.DataFrame({'time': [1], 'symbol': ['A'], 'feature': [0.1]})

    result = _select_columns(df, ['symbol', 'feature'])

    assert result.columns == ['symbol', 'feature']
    assert result.to_dict(as_series=False) == {'symbol': ['A'], 'feature': [0.1]}


def test_select_columns_invalid_type_raises():
    """Should raise TypeError for an unsupported DataFrame type."""
    with pytest.raises(TypeError, match='Pandas or Polars'):
        _select_columns([[1, 2]], ['a'])


# ------------------------------------------------------
# _validate_same_backend
# ------------------------------------------------------
def test_validate_same_backend_accepts_pandas_dataframes():
    """Should not raise when both DataFrames use pandas."""
    left = pd.DataFrame({'a': [1]})
    right = pd.DataFrame({'b': [2]})

    _validate_same_backend(left, right)


def test_validate_same_backend_accepts_polars_dataframes():
    """Should not raise when both DataFrames use polars."""
    left = pl.DataFrame({'a': [1]})
    right = pl.DataFrame({'b': [2]})

    _validate_same_backend(left, right)


def test_validate_same_backend_rejects_mixed_dataframes():
    """Should raise when DataFrames use different backends."""
    left = pd.DataFrame({'a': [1]})
    right = pl.DataFrame({'b': [2]})

    with pytest.raises(TypeError, match='same DataFrame backend'):
        _validate_same_backend(left, right)


def test_validate_same_backend_invalid_left_type_raises():
    """Should raise TypeError when left is not a supported DataFrame."""
    right = pd.DataFrame({'a': [1]})

    with pytest.raises(TypeError, match='Pandas or Polars'):
        _validate_same_backend([[1]], right)


def test_validate_same_backend_invalid_right_type_raises():
    """Should raise TypeError when right is not a supported DataFrame."""
    left = pl.DataFrame({'a': [1]})

    with pytest.raises(TypeError, match='Pandas or Polars'):
        _validate_same_backend(left, [[1]])


# ------------------------------------------------------
# _validate_unique_keys
# ------------------------------------------------------
def test_validate_unique_keys_accepts_unique_pandas_keys():
    """Should not raise when pandas key pairs are unique."""
    df = pd.DataFrame({
        'time': [1, 1, 2],
        'symbol': ['A', 'B', 'A'],
    })

    _validate_unique_keys(df, ['time', 'symbol'], 'feature_df')


def test_validate_unique_keys_accepts_unique_polars_keys():
    """Should not raise when polars key pairs are unique."""
    df = pl.DataFrame({
        'time': [1, 1, 2],
        'symbol': ['A', 'B', 'A'],
    })

    _validate_unique_keys(df, ['time', 'symbol'], 'feature_df')


def test_validate_unique_keys_rejects_duplicate_pandas_keys():
    """Should raise when pandas contains a duplicated key pair."""
    df = pd.DataFrame({
        'time': [1, 1],
        'symbol': ['A', 'A'],
    })

    with pytest.raises(ValueError, match='feature_df must contain unique'):
        _validate_unique_keys(df, ['time', 'symbol'], 'feature_df')


def test_validate_unique_keys_rejects_duplicate_polars_keys():
    """Should raise when polars contains a duplicated key pair."""
    df = pl.DataFrame({
        'time': [1, 1],
        'symbol': ['A', 'A'],
    })

    with pytest.raises(ValueError, match='feature_df must contain unique'):
        _validate_unique_keys(df, ['time', 'symbol'], 'feature_df')


# ------------------------------------------------------
# _validate_time_order
# ------------------------------------------------------
def test_validate_time_order_accepts_unique_increasing_pandas_times():
    """Should accept unique, increasing pandas times."""
    df = pd.DataFrame({
        'time': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
    })

    _validate_time_order(df, 'time')


def test_validate_time_order_accepts_unique_increasing_polars_times():
    """Should accept unique, increasing Polars times."""
    df = pl.DataFrame({
        'time': [
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2024-01-02'),
            pd.Timestamp('2024-01-03'),
        ],
    })

    _validate_time_order(df, 'time')


def test_validate_time_order_accepts_increasing_intraday_times():
    """Should accept timestamps with increasing hours and minutes."""
    df = pd.DataFrame({
        'time': pd.to_datetime([
            '2024-01-01 09:30:00',
            '2024-01-01 09:45:00',
            '2024-01-01 10:00:00',
        ]),
    })

    _validate_time_order(df, 'time')


def test_validate_time_order_rejects_decreasing_intraday_times():
    """Should reject timestamps that move backwards within the same day."""
    df = pd.DataFrame({
        'time': pd.to_datetime([
            '2024-01-01 09:30:00',
            '2024-01-01 10:00:00',
            '2024-01-01 09:45:00',
        ]),
    })

    with pytest.raises(ValueError, match='increasingly ordered'):
        _validate_time_order(df, 'time')


def test_validate_time_order_rejects_duplicate_pandas_times():
    """Should reject repeated pandas times."""
    df = pd.DataFrame({
        'time': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02']),
    })

    with pytest.raises(ValueError, match='unique'):
        _validate_time_order(df, 'time')


def test_validate_time_order_rejects_duplicate_polars_times():
    """Should reject repeated Polars times."""
    df = pl.DataFrame({
        'time': [
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2024-01-02'),
        ],
    })

    with pytest.raises(ValueError, match='unique'):
        _validate_time_order(df, 'time')


def test_validate_time_order_rejects_decreasing_pandas_times():
    """Should reject pandas times that move backwards."""
    df = pd.DataFrame({
        'time': pd.to_datetime(['2024-01-01', '2024-01-03', '2024-01-02']),
    })

    with pytest.raises(ValueError, match='increasingly ordered'):
        _validate_time_order(df, 'time')


def test_validate_time_order_rejects_decreasing_polars_times():
    """Should reject Polars times that move backwards."""
    df = pl.DataFrame({
        'time': [
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2024-01-03'),
            pd.Timestamp('2024-01-02'),
        ],
    })

    with pytest.raises(ValueError, match='increasingly ordered'):
        _validate_time_order(df, 'time')


def test_validate_time_order_rejects_partially_missing_pandas_times():
    """Should reject missing times even when the column has valid values."""
    df = pd.DataFrame({
        'time': pd.to_datetime(['2024-01-01', None, '2024-01-02']),
    })

    with pytest.raises(ValueError, match='must not contain missing values'):
        _validate_time_order(df, 'time')


# ------------------------------------------------------
# _validate_positive_integer
# ------------------------------------------------------
def test_validate_positive_integer_accepts_positive_integer():
    """Should accept Python and NumPy positive integers."""
    _validate_positive_integer(1, 'value')
    _validate_positive_integer(np.int64(2), 'value')


@pytest.mark.parametrize('value', [0, -1, True, 1.0, '1'])
def test_validate_positive_integer_rejects_invalid_values(value):
    """Should reject non-positive, boolean, and noninteger values."""
    with pytest.raises(ValueError, match='value must be a positive integer'):
        _validate_positive_integer(value, 'value')


# ------------------------------------------------------
# _validate_finite_number
# ------------------------------------------------------
@pytest.mark.parametrize('value', [0, -1, 1.5, np.int64(2), np.float64(2.5)])
def test_validate_finite_number_accepts_real_finite_values(value):
    """Should normalize Python and NumPy finite real numbers to float."""
    assert _validate_finite_number(value, 'value') == float(value)


@pytest.mark.parametrize('value', [True, '1', None, 1 + 2j])
def test_validate_finite_number_rejects_non_real_values(value):
    """Should reject booleans and values outside the real-number contract."""
    with pytest.raises(TypeError, match='value must be numeric'):
        _validate_finite_number(value, 'value')


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_validate_finite_number_rejects_non_finite_values(value):
    """Should reject NaN and both infinity signs."""
    with pytest.raises(ValueError, match='value must be finite'):
        _validate_finite_number(value, 'value')


# ------------------------------------------------------
# Validation boundary contracts
# ------------------------------------------------------
@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('check_all_missing', [True, False])
def test_validate_df_rejects_empty_frames_directly(constructor, check_all_missing):
    """Should reject empty frames even when all-missing columns are allowed."""
    frame = constructor({'x': []})
    with pytest.raises(ValueError, match='must not be empty'):
        _validate_df(frame, ['x'], check_all_missing=check_all_missing)


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('columns', ['x', ('x',), None, [1], [None], [''], [['x']]])
def test_validate_df_rejects_malformed_required_columns(constructor, columns):
    """Should reject ambiguous column specifications before selecting data."""
    with pytest.raises(TypeError, match='required_cols must be a list'):
        _validate_df(constructor({'x': [1]}), columns)


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_validate_df_allows_empty_required_columns(constructor):
    """Should retain frame-only validation without rejecting unused missing data."""
    _validate_df(constructor({'unused': [None]}), [], check_all_missing=False)


@pytest.mark.parametrize('check_all_missing', [True, False])
def test_validate_df_rejects_ambiguous_selected_columns(check_all_missing):
    """Should report duplicated requested labels independently of missing policy."""
    frame = pd.DataFrame([[1, 2]], columns=['x', 'x'])
    with pytest.raises(ValueError, match='required column names must be unique.*x'):
        _validate_df(frame, ['x'], check_all_missing=check_all_missing)


def test_validate_df_ignores_unused_duplicate_columns():
    """Should validate unambiguous requested columns without inspecting unused labels."""
    frame = pd.DataFrame([[1, 2, 3]], columns=['x', 'unused', 'unused'])
    _validate_df(frame, ['x'])


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('key', ['time', 'symbol'])
@pytest.mark.parametrize('missing', [None, np.nan])
@pytest.mark.parametrize('position', [0, 1, 2])
def test_validate_key_values_rejects_partial_missing_keys(constructor, key, missing, position):
    """Should recognize native null and NaN keys at every row position."""
    data = {'time': [1., 2., 3.], 'symbol': [10., 20., 30.]}
    data[key][position] = missing
    with pytest.raises(ValueError, match=f'{key} must not contain missing values'):
        _validate_key_values(constructor(data), ['time', 'symbol'])


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('key', ['time', 'symbol'])
@pytest.mark.parametrize('infinite', [np.inf, -np.inf])
def test_validate_key_values_rejects_infinite_keys(constructor, key, infinite):
    """Should reject both signs of infinity in either numeric key."""
    data = {'time': [1., 2.], 'symbol': [10., 20.]}
    data[key][1] = infinite
    with pytest.raises(ValueError, match=f'{key} must contain only finite values'):
        _validate_key_values(constructor(data), ['time', 'symbol'])


@pytest.mark.parametrize('dtype', [object, 'category', 'Float64'])
def test_validate_key_values_rejects_infinity_in_pandas_storage_types(dtype):
    """Should detect infinite scalars in object, categorical, and nullable keys."""
    frame = pd.DataFrame({'time': pd.Series([1., np.inf], dtype=dtype)})
    with pytest.raises(ValueError, match='only finite values'):
        _validate_key_values(frame, ['time'])


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_validate_key_values_preserves_values_and_does_not_require_unique_keys(constructor):
    """Should validate key contents separately from ordering and uniqueness."""
    frame = constructor({'time': [2, 1, 1], 'symbol': ['B', 'A', 'A'], 'unused': [None] * 3})
    before = frame.copy(deep=True) if isinstance(frame, pd.DataFrame) else frame.clone()
    _validate_key_values(frame, ['time', 'symbol'])
    if isinstance(frame, pd.DataFrame):
        pd.testing.assert_frame_equal(frame, before)
    else:
        assert frame.equals(before)


def test_validate_key_values_rejects_unsupported_frame():
    """Should reject unsupported containers before accessing their keys."""
    with pytest.raises(TypeError, match='Pandas or Polars'):
        _validate_key_values({'time': [1]}, ['time'])


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_validate_key_values_checks_schema_and_empty_frames(constructor):
    """Should expose missing-column and empty-frame failures consistently."""
    with pytest.raises(KeyError, match='time'):
        _validate_key_values(constructor({'x': [1]}), ['time'])
    with pytest.raises(ValueError, match='must not be empty'):
        _validate_key_values(constructor({'time': []}), ['time'])
    with pytest.raises(TypeError, match='required_cols'):
        _validate_key_values(constructor({'time': [1]}), 'time')


def test_validate_key_values_rejects_ambiguous_columns():
    """Should reject duplicate selected key labels with an actionable error."""
    with pytest.raises(ValueError, match='unique.*time'):
        _validate_key_values(pd.DataFrame([[1, 2]], columns=['time', 'time']), ['time'])


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('values', [[np.nan, 1., 2.], [1., np.nan, 2.], [1., 2., np.nan],
                                   [None, 1., 2.], [1., None, 2.], [1., 2., None]])
def test_validate_time_order_rejects_all_missing_positions(constructor, values):
    """Should reject missing time keys even when the backend considers them sorted."""
    with pytest.raises(ValueError, match='must not contain missing values'):
        _validate_time_order(constructor({'time': values}), 'time')


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('values', [[-np.inf, 1.], [1., np.inf]])
def test_validate_time_order_rejects_sorted_infinities(constructor, values):
    """Should reject infinite time keys despite otherwise increasing order."""
    with pytest.raises(ValueError, match='only finite values'):
        _validate_time_order(constructor({'time': values}), 'time')


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_validate_time_order_accepts_singleton_and_aware_times(constructor):
    """Should accept one observation and chronological timezone-aware timestamps."""
    _validate_time_order(constructor({'time': [1]}), 'time')
    dates = pd.date_range('2024-11-03 00:30', periods=4, freq='h', tz='America/New_York')
    _validate_time_order(constructor({'time': dates}), 'time')


@pytest.mark.parametrize('value', [np.bool_(True), np.timedelta64(1, 'D'), np.timedelta64('NaT'),
                                 np.datetime64('2024-01-01'), None, np.nan, np.inf, 1 + 0j])
def test_validate_positive_integer_rejects_other_scalar_types(value):
    """Should reject temporal and noninteger scalars instead of accepting durations."""
    with pytest.raises(ValueError, match='value must be a positive integer'):
        _validate_positive_integer(value, 'value')


@pytest.mark.parametrize('value', [np.int8(1), np.uint64(2), 10 ** 400])
def test_validate_positive_integer_does_not_impose_an_unrelated_size_limit(value):
    """Should leave operation-specific size limits to the caller."""
    _validate_positive_integer(value, 'value')


@pytest.mark.parametrize('value', [np.bool_(True), np.timedelta64(1, 'D'), np.timedelta64('NaT'),
                                 np.datetime64('2024-01-01'), np.array([1.]), np.array(1.)])
def test_validate_finite_number_rejects_temporal_and_array_values(value):
    """Should reject unsupported scalar types with the parameter-specific message."""
    with pytest.raises(TypeError, match='value must be numeric'):
        _validate_finite_number(value, 'value')


@pytest.mark.parametrize('value', [10 ** 400, -(10 ** 400)])
def test_validate_finite_number_normalizes_overflow_errors(value):
    """Should reject unrepresentable real parameters with a documented ValueError."""
    with pytest.raises(ValueError, match='value must be finite'):
        _validate_finite_number(value, 'value')


# ------------------------------------------------------
# Public consumers of shared validators
# ------------------------------------------------------
@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_positive_integer_hardening_reaches_public_consumers(constructor):
    """Should reject durations before return, trend, and block-generation calculations."""
    from alpha_research.features.targets import fwd_returns
    from alpha_research.features.trend import price_to_sma_ratio
    from alpha_research.resampling.block_bootstrap import generate_moving_blocks

    frame = constructor({'time': [1, 2, 3], 'symbol': ['A'] * 3, 'close': [10., 11., 12.]})
    invalid = np.timedelta64(1, 'D')
    with pytest.raises(ValueError, match='horizon must be a positive integer'):
        fwd_returns(frame, invalid)
    with pytest.raises(ValueError, match='window must be a positive integer'):
        price_to_sma_ratio(frame, invalid)
    with pytest.raises(ValueError, match='block_length must be a positive integer'):
        generate_moving_blocks(frame, invalid)


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_time_key_hardening_reaches_temporal_association(constructor):
    """Should reject a trailing NaN time instead of computing an apparently valid association."""
    from alpha_research.evaluation.timeseries import temporal_association

    frame = constructor({'time': [1., 2., np.nan], 'symbol': ['A'] * 3,
                         'x': [1., 2., 3.], 'y': [2., 1., 3.]})
    with pytest.raises(ValueError, match='time must not contain missing values'):
        temporal_association(frame, 'x', 'y')
