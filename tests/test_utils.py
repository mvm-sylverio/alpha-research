import numpy as np
import pandas as pd
import polars as pl
import pytest

# Imports that should not be inspected
# noinspection PyProtectedMember
from alpha_research._utils import (
    _as_numpy_values,
    _as_real_float,
    _asset_positions,
    _has_missing_values,
    _is_all_missing,
    _is_constant_series,
    _normalize_column_names,
    _normalize_covariates,
    _normalize_feature_cols,
    _polars_feature_values,
    _select_columns,
    _validate_and_group_times,
    _validate_df,
    _validate_elementwise_features,
    _validate_finite_number,
    _validate_grouped_features,
    _validate_key_values,
    _validate_output_columns,
    _validate_positive_integer,
    _validate_probability,
    _validate_random_state,
    _validate_same_backend,
    _validate_single_symbol,
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


# ------------------------------------------------------
# Shared validation contracts
# ------------------------------------------------------
@pytest.mark.parametrize('constructor', [pd.Series, pl.Series])
@pytest.mark.parametrize('values, expected', [
    ([], False), (['A', 'B'], False), ([np.inf, 1.], False),
    ([1., None], True), ([1., np.nan], True), (['A', None], True),
])
def test_has_missing_values_distinguishes_missing_from_infinite(constructor, values, expected):
    """Should recognize native missing values without classifying infinity as missing."""
    assert _has_missing_values(constructor(values)) is expected


def test_has_missing_values_requires_a_supported_series():
    """Should reject containers outside the Pandas and Polars series contract."""
    with pytest.raises(TypeError, match='Unsupported Series type'):
        _has_missing_values([1, None])


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_validate_key_values_can_preserve_a_missing_only_policy(constructor):
    """Should permit infinities only when explicitly requested and still reject NaN."""
    _validate_key_values(constructor({'time': [1., np.inf]}), ['time'], check_infinite=False)
    with pytest.raises(ValueError, match='missing values'):
        _validate_key_values(constructor({'time': [1., np.nan]}), ['time'], check_infinite=False)
    with pytest.raises(TypeError, match='check_infinite must be boolean'):
        _validate_key_values(constructor({'time': [1.]}), ['time'], check_infinite=0)


@pytest.mark.parametrize('columns, expected', [('x', ['x']), (['z', 'x'], ['z', 'x'])])
def test_normalize_column_names_preserves_name_order(columns, expected):
    """Should preserve the caller's name order when normalizing a container."""
    assert _normalize_column_names(columns, type_error='invalid names', empty_error='no names') == expected


@pytest.mark.parametrize('columns, exception, message', [
    (('x',), TypeError, 'invalid names'), (None, TypeError, 'invalid names'), ([], ValueError, 'no names'),
])
def test_normalize_column_names_preserves_caller_errors(columns, exception, message):
    """Should report an invalid or empty container with the requested message."""
    with pytest.raises(exception, match=message):
        _normalize_column_names(columns, type_error='invalid names', empty_error='no names')


@pytest.mark.parametrize('value, expected', [(3, 3.), (np.float32(0.5), 0.5), (np.inf, np.inf), (-np.inf, -np.inf)])
def test_as_real_float_preserves_real_values(value, expected):
    """Should convert supported scalars and preserve the sign of infinity."""
    assert _as_real_float(value, type_error='invalid scalar', overflow_error='float overflow') == expected


def test_as_real_float_leaves_nan_for_callers():
    """Should leave a NaN intact so each caller can apply its missing-value policy."""
    assert np.isnan(_as_real_float(np.nan, type_error='invalid scalar', overflow_error='float overflow'))


@pytest.mark.parametrize('value', [True, np.bool_(False), '1', 1j, None, np.timedelta64(1, 'D')])
def test_as_real_float_rejects_coercion_of_unsupported_scalars(value):
    """Should reject numeric-looking non-real values with the caller's error."""
    with pytest.raises(TypeError, match='invalid scalar'):
        _as_real_float(value, type_error='invalid scalar', overflow_error='float overflow')


@pytest.mark.parametrize('value', [10 ** 400, -(10 ** 400)])
def test_as_real_float_rejects_unrepresentable_magnitudes(value):
    """Should distinguish overflow of a finite scalar from an input infinity."""
    with pytest.raises(ValueError, match='float overflow'):
        _as_real_float(value, type_error='invalid scalar', overflow_error='float overflow')


@pytest.mark.parametrize('value', [0.05, np.float32(0.5), np.nextafter(1., 0.)])
def test_validate_probability_accepts_open_interval(value):
    """Should accept representable probabilities strictly inside the interval."""
    assert _validate_probability(value, 'level') is None


@pytest.mark.parametrize('value', [0, 1, -0.5, 1.5, True, np.bool_(False), np.nan, np.inf, '0.5',
                                 np.timedelta64(1, 'D')])
def test_validate_probability_rejects_invalid_scalars(value):
    """Should reject endpoints and invalid scalar types with a named ValueError."""
    with pytest.raises(ValueError, match='level must be strictly between 0 and 1'):
        _validate_probability(value, 'level')


@pytest.mark.parametrize('seed', [None, 0, np.int64(4), np.uint64(2 ** 64 - 1), 2 ** 100])
def test_validate_random_state_accepts_supported_seeds(seed):
    """Should accept optional non-negative integer seeds without narrowing their range."""
    assert _validate_random_state(seed) is None


@pytest.mark.parametrize('seed', [True, np.bool_(True), 1., '1', np.timedelta64(1, 'D')])
def test_validate_random_state_rejects_non_integer_seeds(seed):
    """Should reject booleans, durations, and coercible noninteger seeds."""
    with pytest.raises(TypeError, match='random_state must be an integer or None'):
        _validate_random_state(seed)


def test_validate_random_state_rejects_negative_seeds():
    """Should reject a negative seed before a consumer initializes its RNG."""
    with pytest.raises(ValueError, match='expected non-negative integer'):
        _validate_random_state(np.int64(-1))


def test_asset_positions_groups_interleaved_symbols_in_input_order():
    """Should retain each asset's temporal row order without sorting the frame."""
    result = _asset_positions(['AAPL', 'MSFT', 'AAPL', 'GOOGL', 'MSFT'])

    assert result == [[0, 2], [1, 4], [3]]



def test_asset_positions_requires_hashable_identities():
    """Should reject nested unhashable asset identifiers."""
    with pytest.raises(TypeError, match='unhashable'):
        _asset_positions([['A']])


@pytest.mark.parametrize('kind', [
    'list',
    pytest.param('wide_integer', marks=pytest.mark.skipif(not hasattr(pl, 'Int128'), reason='Polars has no Int128')),
])
def test_grouping_preserves_native_polars_asset_keys(kind):
    """Should preserve asset identities that cannot be grouped through a NumPy array."""
    from alpha_research.features.transformations import rolling_rank

    if kind == 'list':
        symbols = pl.Series([[1], [2], [1], [2]])
    else:
        symbols = pl.Series([2 ** 80, 2 ** 80 + 1, 2 ** 80, 2 ** 80 + 1], dtype=pl.Int128)
    assert _asset_positions(symbols) == [[0, 2], [1, 3]]

    frame = pl.DataFrame({'symbol': symbols, 'time': [3, 1, 4, 2], 'x': [8., 5., 9., 4.]})
    result = rolling_rank(frame, 'x', window=2)
    assert result['symbol'].equals(frame['symbol'])
    assert result['x_rolling_rank_2'].to_list() == [None, None, 2., 1.]


@pytest.mark.parametrize('constructor', [np.asarray, pd.Series, pl.Series])
def test_validate_and_group_times_preserves_interleaved_positions(constructor):
    """Should accept overlapping times across assets and preserve first-seen order."""
    symbols = constructor(['B', 'A', 'B', 'A'])
    times = constructor([4, 1, 5, 2])
    groups = _validate_and_group_times(symbols, times, 'asset', 'date')
    assert [group.tolist() for group in groups] == [[0, 2], [1, 3]]


def test_validate_and_group_times_preserves_pandas_category_order():
    """Should use the configured temporal category order rather than lexical order."""
    symbols = pd.Series(['A', 'A'], index=[50, 20])
    times = pd.Series(pd.Categorical(['early', 'late'], categories=['early', 'late'], ordered=True))
    times = times.cat.rename_categories(['z_first', 'a_second'])
    assert [group.tolist() for group in _validate_and_group_times(symbols, times, 'asset', 'date')] == [[0, 1]]
    with pytest.raises(ValueError, match='date must be increasingly ordered within each asset'):
        _validate_and_group_times(symbols, times.iloc[::-1], 'asset', 'date')


@pytest.mark.parametrize('symbols, times, exception, message', [
    (pd.Series(['A']), pl.Series([1]), TypeError, 'same backend'),
    (np.array([['A']]), np.array([1]), ValueError, 'one-dimensional arrays'),
    (pl.Series(['A', 'B']), pl.Series([1]), ValueError, 'aligned'),
    (pl.Series([1., np.nan]), pl.Series([1, 2]), ValueError, 'asset must not contain missing values'),
    (pl.Series(['A', 'A']), pl.Series([1., np.nan]), ValueError, 'date must not contain missing values'),
    (pl.Series(['A', 'A']), pl.Series([1, 1]), ValueError, 'unique values within each asset'),
    (pl.Series(['A', 'A']), pl.Series([2, 1]), ValueError, 'increasingly ordered'),
])
def test_validate_and_group_times_checks_input_contract(symbols, times, exception, message):
    """Should reject mismatched data, missing identities, and invalid within-asset ordering."""
    with pytest.raises(exception, match=message):
        _validate_and_group_times(symbols, times, 'asset', 'date')


def test_validate_and_group_times_preserves_interleaved_symbol_order():
    """Should retain each symbol's input positions after validation."""
    groups = _validate_and_group_times(
        symbols=np.array(['AAPL', 'MSFT', 'AAPL', 'MSFT']),
        times=np.array([1, 1, 2, 2]),
        symbol_col='symbol',
        time_col='time',
    )

    assert [group.tolist() for group in groups] == [[0, 2], [1, 3]]


@pytest.mark.parametrize(
    ('symbols', 'times', 'message'),
    [
        (np.array(['AAPL', None]), np.array([1, 2]), 'symbol'),
        (np.array(['AAPL', 'AAPL']), np.array([1, np.nan]), 'time'),
        (np.array(['AAPL', 'AAPL']), np.array([1, 1]), 'unique'),
        (np.array(['AAPL', 'AAPL']), np.array([2, 1]), 'increasingly'),
    ],
)
def test_validate_and_group_times_rejects_invalid_time_keys(symbols, times, message):
    """Should require complete, unique, ordered time keys within each symbol."""
    with pytest.raises(ValueError, match=message):
        _validate_and_group_times(symbols, times, 'symbol', 'time')


# ------------------------------------------------------
# Single-asset validation
# ------------------------------------------------------
def test_validate_single_symbol_accepts_one_pandas_symbol():
    """Should accept a pandas DataFrame containing one non-missing symbol."""
    frame = pd.DataFrame({'symbol': ['AAPL', 'AAPL']})
    _validate_single_symbol(frame, 'symbol')


def test_validate_single_symbol_accepts_one_polars_symbol():
    """Should accept a Polars DataFrame containing one non-missing symbol."""
    frame = pl.DataFrame({'symbol': ['AAPL', 'AAPL']})
    _validate_single_symbol(frame, 'symbol')


def test_validate_single_symbol_rejects_multiple_pandas_symbols():
    """Should reject a pandas DataFrame containing multiple symbols."""
    frame = pd.DataFrame({'symbol': ['AAPL', 'MSFT']})
    with pytest.raises(ValueError, match='exactly one non-missing symbol'):
        _validate_single_symbol(frame, 'symbol')


def test_validate_single_symbol_rejects_multiple_polars_symbols():
    """Should reject a Polars DataFrame containing multiple symbols."""
    frame = pl.DataFrame({'symbol': ['AAPL', 'MSFT']})
    with pytest.raises(ValueError, match='exactly one non-missing symbol'):
        _validate_single_symbol(frame, 'symbol')


def test_validate_single_symbol_rejects_missing_pandas_symbols():
    """Should reject a pandas DataFrame containing missing symbols."""
    frame = pd.DataFrame({'symbol': ['AAPL', None]})
    with pytest.raises(ValueError, match='exactly one non-missing symbol'):
        _validate_single_symbol(frame, 'symbol')


def test_validate_single_symbol_rejects_missing_polars_symbols():
    """Should reject a Polars DataFrame containing missing symbols."""
    frame = pl.DataFrame({'symbol': ['AAPL', None]})
    with pytest.raises(ValueError, match='exactly one non-missing symbol'):
        _validate_single_symbol(frame, 'symbol')


# ------------------------------------------------------
# Column extraction and feature validation
# ------------------------------------------------------
@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_as_numpy_values_preserves_the_selected_column(backend):
    """Should preserve extracted values for both supported backends."""
    pandas_df = pd.DataFrame({'value': [1, 2, 3]})
    df = pandas_df if backend == 'pandas' else pl.from_pandas(pandas_df)

    np.testing.assert_array_equal(_as_numpy_values(df, 'value'), [1, 2, 3])


def test_polars_feature_values_handles_null_dtype_and_missing_column():
    """Should normalize Null dtype without modifying source or accepting absent columns."""
    frame = pl.DataFrame({'x': [None, None]})
    assert frame.select(_polars_feature_values(frame, 'x')).schema['x'] == pl.Float64
    assert frame.schema['x'] == pl.Null
    with pytest.raises(KeyError):
        _polars_feature_values(frame, 'missing')


@pytest.mark.parametrize(
    'frame, features, error_type, message',
    [
        (pd.DataFrame({'x': [1.0]}), [], ValueError, 'empty'),
        (pd.DataFrame({'x': [1.0]}), 'missing', KeyError, 'missing required'),
        (pd.DataFrame({'x': ['a']}), 'x', ValueError, 'numeric'),
        (pd.DataFrame({'x': [np.inf]}), 'x', ValueError, 'finite'),
    ],
)
def test_validate_elementwise_features_rejects_invalid_inputs(
        frame,
        features,
        error_type,
        message,
):
    """Should enforce the direct helper contract for pointwise inputs."""
    with pytest.raises(error_type, match=message):
        _validate_elementwise_features(frame, features)


def test_validate_elementwise_features_returns_normalized_names():
    """Should normalize a valid single feature into a list."""
    assert _validate_elementwise_features(pd.DataFrame({'x': [1.0]}), 'x') == ['x']


def test_validate_grouped_features_contract():
    """Should validate grouped feature names and keys without rejecting missing values."""
    frame = pd.DataFrame({'time': [1, 2], 'symbol': ['A', 'A'], 'x': [np.nan, np.nan]})
    assert _validate_grouped_features(frame, 'x', 'symbol', 'time') == ['x']
    with pytest.raises(KeyError):
        _validate_grouped_features(frame, 'x', 'symbol', 'missing')


def test_validate_output_columns_rejects_conflicts():
    """Should accept unused names and reject existing or repeated outputs."""
    frame = pd.DataFrame({'x': [1.], 'x_sign': [9.]})
    assert _validate_output_columns(frame, ['x'], 'abs') is None
    with pytest.raises(ValueError, match='x_sign'):
        _validate_output_columns(frame, ['x'], 'sign')
    with pytest.raises(ValueError, match='repeat'):
        _validate_output_columns(frame, ['x', 'x'], 'abs')


# ------------------------------------------------------
# Feature and covariate names
# ------------------------------------------------------
def test_normalize_feature_cols_accepts_string_and_list():
    """Should normalize a single feature without altering a feature list."""
    assert _normalize_feature_cols('feature') == ['feature']
    assert _normalize_feature_cols(['feature_a', 'feature_b']) == ['feature_a', 'feature_b']


@pytest.mark.parametrize('feature_cols, exception', [
    ([], ValueError),
    (123, TypeError),
])
def test_normalize_feature_cols_rejects_invalid_input(feature_cols, exception):
    """Should validate feature-column input independently of public callers."""
    with pytest.raises(exception):
        _normalize_feature_cols(feature_cols)


def test_normalize_covariates_accepts_one_or_many_covariates():
    """Should normalize covariate names without changing order."""
    assert _normalize_covariates('covariate_a') == ['covariate_a']
    assert _normalize_covariates(['covariate_a', 'covariate_b']) == [
        'covariate_a',
        'covariate_b',
    ]


@pytest.mark.parametrize(
    ('covariates', 'error_type', 'message'),
    [
        ((), TypeError, 'string or a list'),
        ([], ValueError, 'must not be empty'),
        (['covariate_a', 1], TypeError, 'string or a list'),
        (['covariate_a', 'covariate_a'], ValueError, 'duplicates'),
    ],
)
def test_normalize_covariates_rejects_invalid_specifications(
        covariates,
        error_type,
        message,
):
    """Should report invalid covariate specifications."""
    with pytest.raises(error_type, match=message):
        _normalize_covariates(covariates)


@pytest.mark.parametrize('controls', ['', [''], ['z', ' ']])
def test_covariates_rejects_empty_names(controls):
    """Should reject empty control names without stripping legitimate surrounding spaces."""
    with pytest.raises(ValueError, match='empty'):
        _normalize_covariates(controls)
    assert _normalize_covariates(' z ') == [' z ']
