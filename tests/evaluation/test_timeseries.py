import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.evaluation.timeseries import (
    _validate_single_symbol,
    temporal_association,
)


# ------------------------------------------------------
# fixtures
# ------------------------------------------------------
@pytest.fixture
def temporal_df_pandas():
    return pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=5, freq='D'),
        'symbol': ['AAPL'] * 5,
        'feature': [1.0, 2.0, 3.0, 4.0, 5.0],
        'target': [2.0, 4.0, 6.0, 8.0, 10.0],
    })


@pytest.fixture
def temporal_df_polars(temporal_df_pandas):
    return pl.from_pandas(temporal_df_pandas)


# ------------------------------------------------------
# _validate_single_symbol
# ------------------------------------------------------
def test_validate_single_symbol_accepts_one_pandas_symbol(temporal_df_pandas):
    """Should accept a pandas DataFrame containing one non-missing symbol."""
    _validate_single_symbol(temporal_df_pandas, 'symbol')


def test_validate_single_symbol_accepts_one_polars_symbol(temporal_df_polars):
    """Should accept a Polars DataFrame containing one non-missing symbol."""
    _validate_single_symbol(temporal_df_polars, 'symbol')


def test_validate_single_symbol_rejects_multiple_pandas_symbols(temporal_df_pandas):
    """Should reject a pandas DataFrame containing multiple symbols."""
    df = temporal_df_pandas.copy()
    df.loc[4, 'symbol'] = 'MSFT'

    with pytest.raises(ValueError, match='exactly one non-missing symbol'):
        _validate_single_symbol(df, 'symbol')


def test_validate_single_symbol_rejects_multiple_polars_symbols(temporal_df_polars):
    """Should reject a Polars DataFrame containing multiple symbols."""
    df = temporal_df_polars.with_columns(
        pl.when(pl.int_range(pl.len()) == 4)
        .then(pl.lit('MSFT'))
        .otherwise(pl.col('symbol'))
        .alias('symbol')
    )

    with pytest.raises(ValueError, match='exactly one non-missing symbol'):
        _validate_single_symbol(df, 'symbol')


def test_validate_single_symbol_rejects_missing_pandas_symbols(temporal_df_pandas):
    """Should reject a pandas DataFrame containing missing symbols."""
    df = temporal_df_pandas.copy()
    df.loc[2, 'symbol'] = None

    with pytest.raises(ValueError, match='exactly one non-missing symbol'):
        _validate_single_symbol(df, 'symbol')


def test_validate_single_symbol_rejects_missing_polars_symbols(temporal_df_polars):
    """Should reject a Polars DataFrame containing missing symbols."""
    df = temporal_df_polars.with_columns(
        pl.when(pl.int_range(pl.len()) == 2)
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.col('symbol'))
        .alias('symbol')
    )

    with pytest.raises(ValueError, match='exactly one non-missing symbol'):
        _validate_single_symbol(df, 'symbol')

# ------------------------------------------------------
# temporal_association
# ------------------------------------------------------
def test_temporal_association_returns_perfect_spearman_correlation(temporal_df_pandas):
    """Should return a perfect Spearman association for aligned monotonic series."""
    result = temporal_association(temporal_df_pandas, 'feature', 'target')

    assert result == pytest.approx(1.0)


def test_temporal_association_returns_perfect_pearson_correlation(temporal_df_pandas):
    """Should return a perfect Pearson association for linearly related series."""
    result = temporal_association(
        temporal_df_pandas,
        'feature',
        'target',
        corr_method='pearson',
    )

    assert result == pytest.approx(1.0)


def test_temporal_association_drops_missing_aligned_values(temporal_df_pandas):
    """Should calculate the association from aligned non-missing observations."""
    df = temporal_df_pandas.copy()
    df.loc[2, 'target'] = np.nan

    result = temporal_association(df, 'feature', 'target')

    assert result == pytest.approx(1.0)


def test_temporal_association_rejects_duplicate_times(temporal_df_pandas):
    """Should reject duplicate observation times through the temporal validator."""
    df = temporal_df_pandas.copy()
    df.loc[4, 'time'] = df.loc[3, 'time']

    with pytest.raises(ValueError, match='unique'):
        temporal_association(df, 'feature', 'target')


def test_temporal_association_rejects_unordered_times(temporal_df_pandas):
    """Should reject unordered observation times through the temporal validator."""
    df = temporal_df_pandas.iloc[[0, 2, 1, 3, 4]].reset_index(drop=True)

    with pytest.raises(ValueError, match='increasingly ordered'):
        temporal_association(df, 'feature', 'target')


def test_temporal_association_rejects_missing_time(temporal_df_pandas):
    """Should reject missing observation times through the temporal validator."""
    df = temporal_df_pandas.copy()
    df.loc[2, 'time'] = pd.NaT

    with pytest.raises(ValueError, match='time must not contain missing values'):
        temporal_association(df, 'feature', 'target')


def test_temporal_association_returns_nan_for_constant_series(temporal_df_pandas):
    """Should return nan when the feature series is constant."""
    df = temporal_df_pandas.copy()
    df['feature'] = 1.0

    result = temporal_association(df, 'feature', 'target')

    assert np.isnan(result)


def test_temporal_association_pandas_polars_consistency(
        temporal_df_pandas,
        temporal_df_polars,
):
    """Should return equivalent associations for pandas and Polars inputs."""
    pandas_result = temporal_association(temporal_df_pandas, 'feature', 'target')
    polars_result = temporal_association(temporal_df_polars, 'feature', 'target')

    assert pandas_result == pytest.approx(polars_result)
