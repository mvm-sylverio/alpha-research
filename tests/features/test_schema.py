import pandas as pd
import polars as pl
import pytest

from alpha_research.features.schema import get_feature_name, join_feature_target_frames


# ------------------------------------------------------
# fixtures
# ------------------------------------------------------
@pytest.fixture
def pandas_feature_df():
    return pd.DataFrame({
        'time': [1, 2, 3, 4],
        'symbol': ['A', 'A', 'A', 'A'],
        'feature': [1.0, None, 3.0, 4.0],
        'feature_metadata': ['a', 'b', 'c', 'd'],
    })


@pytest.fixture
def pandas_target_df():
    return pd.DataFrame({
        'time': [1, 2, 3, 5],
        'symbol': ['A', 'A', 'A', 'A'],
        'target': [0.1, 0.2, None, 0.5],
        'target_metadata': ['a', 'b', 'c', 'e'],
    })


@pytest.fixture
def polars_feature_df():
    return pl.DataFrame({
        'time': [1, 2, 3, 4],
        'symbol': ['A', 'A', 'A', 'A'],
        'feature': [1.0, None, 3.0, 4.0],
        'feature_metadata': ['a', 'b', 'c', 'd'],
    })


@pytest.fixture
def polars_target_df():
    return pl.DataFrame({
        'time': [1, 2, 3, 5],
        'symbol': ['A', 'A', 'A', 'A'],
        'target': [0.1, 0.2, None, 0.5],
        'target_metadata': ['a', 'b', 'c', 'e'],
    })


# ------------------------------------------------------
# get_feature_name
# ------------------------------------------------------
def test_get_feature_name_pandas_returns_single_value_column():
    """Should return the single value column from a pandas DataFrame."""
    df = pd.DataFrame({'time': [1], 'symbol': ['A'], 'feature': [1.0]})

    assert get_feature_name(df) == 'feature'


def test_get_feature_name_polars_returns_single_value_column():
    """Should return the single value column from a polars DataFrame."""
    df = pl.DataFrame({'time': [1], 'symbol': ['A'], 'feature': [1.0]})

    assert get_feature_name(df) == 'feature'


def test_get_feature_name_requires_one_value_column():
    """Should raise when a DataFrame has more than one value column."""
    df = pd.DataFrame({
        'time': [1],
        'symbol': ['A'],
        'first': [1.0],
        'second': [2.0],
    })

    with pytest.raises(ValueError, match='exactly one'):
        get_feature_name(df)


def test_get_feature_name_invalid_type_raises():
    """Should raise TypeError for an unsupported DataFrame type."""
    with pytest.raises(TypeError, match='Pandas or Polars'):
        get_feature_name([[1, 2]])


# ------------------------------------------------------
# join_feature_target_frames
# ------------------------------------------------------
def test_join_feature_target_frames_pandas_keeps_only_matched_valid_rows(
        pandas_feature_df,
        pandas_target_df,
):
    """Should apply inner join and remove missing feature or target values."""
    result = join_feature_target_frames(
        pandas_feature_df,
        pandas_target_df,
        'feature',
        'target',
        'time',
        'symbol',
    )

    expected = pd.DataFrame({
        'time': [1],
        'symbol': ['A'],
        'feature': [1.0],
        'target': [0.1],
    })

    pd.testing.assert_frame_equal(result, expected)


def test_join_feature_target_frames_polars_keeps_only_matched_valid_rows(
        polars_feature_df,
        polars_target_df,
):
    """Should apply inner join and remove missing feature or target values."""
    result = join_feature_target_frames(
        polars_feature_df,
        polars_target_df,
        'feature',
        'target',
        'time',
        'symbol',
    )

    expected = pl.DataFrame({
        'time': [1],
        'symbol': ['A'],
        'feature': [1.0],
        'target': [0.1],
    })

    assert result.equals(expected)


def test_join_feature_target_frames_polars_removes_nan_values():
    """Should remove NaN feature and target values from polars DataFrames."""
    feature_df = pl.DataFrame({
        'time': [1, 2, 3],
        'symbol': ['A', 'A', 'A'],
        'feature': [float('nan'), 2.0, 3.0],
    })
    target_df = pl.DataFrame({
        'time': [1, 2, 3],
        'symbol': ['A', 'A', 'A'],
        'target': [0.1, float('nan'), 0.3],
    })

    result = join_feature_target_frames(
        feature_df,
        target_df,
        'feature',
        'target',
        'time',
        'symbol',
    )

    expected = pl.DataFrame({
        'time': [3],
        'symbol': ['A'],
        'feature': [3.0],
        'target': [0.3],
    })

    assert result.equals(expected)


def test_join_feature_target_frames_polars_removes_null_values():
    """Should remove null feature and target values from polars DataFrames."""
    feature_df = pl.DataFrame({
        'time': [1, 2, 3],
        'symbol': ['A', 'A', 'A'],
        'feature': [None, 2.0, 3.0],
    })
    target_df = pl.DataFrame({
        'time': [1, 2, 3],
        'symbol': ['A', 'A', 'A'],
        'target': [0.1, None, 0.3],
    })

    result = join_feature_target_frames(
        feature_df,
        target_df,
        'feature',
        'target',
        'time',
        'symbol',
    )

    expected = pl.DataFrame({
        'time': [3],
        'symbol': ['A'],
        'feature': [3.0],
        'target': [0.3],
    })

    assert result.equals(expected)


def test_join_feature_target_frames_pandas_rejects_duplicate_feature_keys():
    """Should reject duplicated time and symbol pairs in the feature DataFrame."""
    feature_df = pd.DataFrame({
        'time': [1, 1],
        'symbol': ['A', 'A'],
        'feature': [1.0, 2.0],
    })
    target_df = pd.DataFrame({
        'time': [1],
        'symbol': ['A'],
        'target': [0.1],
    })

    with pytest.raises(ValueError, match='feature_df must contain unique'):
        join_feature_target_frames(
            feature_df,
            target_df,
            'feature',
            'target',
            'time',
            'symbol',
        )


def test_join_feature_target_frames_pandas_rejects_duplicate_target_keys():
    """Should reject duplicated time and symbol pairs in the target DataFrame."""
    feature_df = pd.DataFrame({
        'time': [1],
        'symbol': ['A'],
        'feature': [1.0],
    })
    target_df = pd.DataFrame({
        'time': [1, 1],
        'symbol': ['A', 'A'],
        'target': [0.1, 0.2],
    })

    with pytest.raises(ValueError, match='target_df must contain unique'):
        join_feature_target_frames(
            feature_df,
            target_df,
            'feature',
            'target',
            'time',
            'symbol',
        )


def test_join_feature_target_frames_polars_rejects_duplicate_feature_keys():
    """Should reject duplicated time and symbol pairs in the feature DataFrame."""
    feature_df = pl.DataFrame({
        'time': [1, 1],
        'symbol': ['A', 'A'],
        'feature': [1.0, 2.0],
    })
    target_df = pl.DataFrame({
        'time': [1],
        'symbol': ['A'],
        'target': [0.1],
    })

    with pytest.raises(ValueError, match='feature_df must contain unique'):
        join_feature_target_frames(
            feature_df,
            target_df,
            'feature',
            'target',
            'time',
            'symbol',
        )


def test_join_feature_target_frames_polars_rejects_duplicate_target_keys():
    """Should reject duplicated time and symbol pairs in the target DataFrame."""
    feature_df = pl.DataFrame({
        'time': [1],
        'symbol': ['A'],
        'feature': [1.0],
    })
    target_df = pl.DataFrame({
        'time': [1, 1],
        'symbol': ['A', 'A'],
        'target': [0.1, 0.2],
    })

    with pytest.raises(ValueError, match='target_df must contain unique'):
        join_feature_target_frames(
            feature_df,
            target_df,
            'feature',
            'target',
            'time',
            'symbol',
        )


def test_join_feature_target_frames_rejects_mixed_backends():
    """Should raise when feature and target use different backends."""
    feature_df = pd.DataFrame({'time': [1], 'symbol': ['A'], 'feature': [1.0]})
    target_df = pl.DataFrame({'time': [1], 'symbol': ['A'], 'target': [0.1]})

    with pytest.raises(TypeError, match='same DataFrame backend'):
        join_feature_target_frames(
            feature_df,
            target_df,
            'feature',
            'target',
            'time',
            'symbol',
        )


def test_join_feature_target_frames_rejects_empty_dataframes():
    """Should raise when a required input DataFrame is empty."""
    feature_df = pd.DataFrame({
        'time': [],
        'symbol': [],
        'feature': [],
    })
    target_df = pd.DataFrame({
        'time': [1],
        'symbol': ['A'],
        'target': [0.1],
    })

    with pytest.raises(ValueError, match='must not be empty'):
        join_feature_target_frames(
            feature_df,
            target_df,
            'feature',
            'target',
            'time',
            'symbol',
        )


def test_join_feature_target_frames_rejects_all_missing_target_values():
    """Should raise when all target values are missing."""
    feature_df = pl.DataFrame({
        'time': [1, 2],
        'symbol': ['A', 'A'],
        'feature': [1.0, 2.0],
    })
    target_df = pl.DataFrame({
        'time': [1, 2],
        'symbol': ['A', 'A'],
        'target': [None, None],
    })

    with pytest.raises(ValueError, match='missing all values'):
        join_feature_target_frames(
            feature_df,
            target_df,
            'feature',
            'target',
            'time',
            'symbol',
        )
