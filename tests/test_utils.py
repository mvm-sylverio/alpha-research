import pytest
import pandas as pd
import polars as pl
import numpy as np

# Imports that should not be inspected
# noinspection PyProtectedMember
from alpha_research._utils import _validate_df, _is_all_missing

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
# _validate_df
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
