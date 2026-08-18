import pytest
import pandas as pd
import polars as pl
import numpy as np

# Imports that should not be inspected
# noinspection PyProtectedMember
from alpha_research._utils import (
    _is_all_missing,
    _is_constant_series,
    _select_columns,
    _validate_df,
    _validate_same_backend,
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
