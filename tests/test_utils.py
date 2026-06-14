import pytest
import pandas as pd
import polars as pl

# Imports that should not be inspected
# noinspection PyProtectedMember
from alpha_research._utils import _validate_df


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
