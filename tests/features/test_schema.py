from datetime import date, time

import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.features.schema import (
    _align_join_keys,
    _join_key_kind,
    get_feature_name,
    join_feature_target_frames,
)


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
@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_join_can_allow_all_missing_values(backend):
    """Should allow an entirely missing value column when explicitly requested."""
    constructor = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    features = constructor({'time': [1], 'symbol': ['A'], 'x': [None]})
    targets = constructor({'time': [1], 'symbol': ['A'], 'y': [1.]})
    result = join_feature_target_frames(features, targets, 'x', 'y', 'time', 'symbol',
                                        check_all_missing=False)
    assert len(result) == 0


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


# ------------------------------------------------------
# Schema-only inference and internal key normalization
# ------------------------------------------------------
@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('data', [{}, {'time': [1], 'symbol': ['A']}, {'time': [], 'symbol': []}])
def test_get_feature_name_rejects_zero_value_columns(constructor, data):
    """Should reject schemas without a value column regardless of row count."""
    with pytest.raises(ValueError, match='exactly one'):
        get_feature_name(constructor(data))


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('data', [{'x': []}, {'x': [None]}, {'time': [], 'symbol': [], 'x': []}])
def test_get_feature_name_only_inspects_schema(constructor, data):
    """Should infer a value name without requiring rows, valid keys, or observed values."""
    assert get_feature_name(constructor(data)) == 'x'


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_get_feature_name_respects_custom_keys(constructor):
    """Should exclude only the customized key names when inferring the value."""
    frame = constructor({'date': [None], 'asset': [None], 'signal': [1.]})
    assert get_feature_name(frame, 'date', 'asset') == 'signal'


@pytest.mark.parametrize('series, expected', [
    (pd.Series([1], dtype='Int64'), 'integer'),
    (pd.Series([1.], dtype='Float64'), 'floating'),
    (pd.Series([True]), 'boolean'),
    (pd.Series(['A'], dtype='string'), 'string'),
    (pd.Series(['A'], dtype='category'), 'string'),
    (pd.Series([b'A']), 'bytes'),
    (pd.Series([date(2024, 1, 1)]), 'date'),
    (pd.Series([time(9, 30)]), 'time'),
    (pd.Series(pd.date_range('2024-01-01', periods=1, tz='UTC')), 'datetime'),
    (pd.Series(pd.to_timedelta([1], unit='D')), 'duration'),
    (pl.Series([1], dtype=pl.Int32), 'integer'),
    (pl.Series([1.], dtype=pl.Float32), 'floating'),
    (pl.Series([True]), 'boolean'),
    (pl.Series(['A']), 'string'),
    (pl.Series(['A']).cast(pl.Categorical), 'string'),
    (pl.Series(['A']).cast(pl.Enum(['B', 'A'])), 'string'),
    (pl.Series([b'A']), 'bytes'),
    (pl.Series([date(2024, 1, 1)]), 'date'),
    (pl.Series([time(9, 30)]), 'time'),
    (pl.Series([pd.Timestamp('2024-01-01', tz='UTC')]), 'datetime'),
    (pl.Series([1], dtype=pl.Duration('us')), 'duration'),
])
def test_join_key_kind_identifies_semantic_domains(series, expected):
    """Should distinguish identity domains independently of storage widths."""
    assert _join_key_kind(series) == expected


@pytest.mark.parametrize('series', [
    [1], pd.Series([1 + 2j]), pd.Series(['A', 1], dtype=object),
    pd.Series([[1], [2]]), pl.Series([[1], [2]]), pl.Series([{'a': 1}]),
    pl.Series('key', [object()], dtype=pl.Object),
])
def test_join_key_kind_rejects_unsupported_domains(series):
    """Should reject mixed, complex, nested, and unsupported key containers."""
    with pytest.raises(TypeError, match='homogeneous supported scalar'):
        _join_key_kind(series)


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_align_join_keys_matches_aware_instants_without_mutation(constructor):
    """Should normalize timezone metadata while retaining exact instants and input order."""
    # 09:30 in Sao Paulo and 12:30 UTC are the same instant on this date.
    left = constructor({'time': [pd.Timestamp('2024-01-02 09:30', tz='America/Sao_Paulo')], 'symbol': ['A']})
    right = constructor({'time': [pd.Timestamp('2024-01-02 12:30', tz='UTC')], 'symbol': ['A']})
    left_before = left.copy(deep=True) if isinstance(left, pd.DataFrame) else left.clone()
    right_before = right.copy(deep=True) if isinstance(right, pd.DataFrame) else right.clone()
    aligned_left, aligned_right = _align_join_keys(left, right, ['time', 'symbol'])
    assert aligned_left['time'].to_list() == aligned_right['time'].to_list()
    if isinstance(left, pd.DataFrame):
        pd.testing.assert_frame_equal(left, left_before)
        pd.testing.assert_frame_equal(right, right_before)
    else:
        assert left.equals(left_before)
        assert right.equals(right_before)


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_align_join_keys_validates_its_inputs(constructor):
    """Should directly validate backend, schema, key contents, and input size."""
    frame = constructor({'time': [1], 'symbol': ['A']})
    with pytest.raises(KeyError, match='missing'):
        _align_join_keys(frame, frame, ['missing'])
    with pytest.raises(ValueError, match='must not be empty'):
        _align_join_keys(constructor({'time': []}), frame, ['time'])
    with pytest.raises(ValueError, match='missing values'):
        _align_join_keys(constructor({'time': [None]}), frame, ['time'])
    with pytest.raises(ValueError, match='only finite values'):
        _align_join_keys(constructor({'time': [np.inf]}), frame, ['time'])
    with pytest.raises(TypeError, match='required_cols'):
        _align_join_keys(frame, frame, 'time')
    with pytest.raises(TypeError, match='Pandas or Polars'):
        _align_join_keys(frame, {}, ['time'])
    for keys in ([], ['time', 'time']):
        with pytest.raises(ValueError, match='join keys must be non-empty and distinct'):
            _align_join_keys(frame, frame, keys)


def test_align_join_keys_rejects_mixed_backends_directly():
    """Should reject mixed DataFrame backends before normalizing their keys."""
    with pytest.raises(TypeError, match='same DataFrame backend'):
        _align_join_keys(pd.DataFrame({'time': [1]}), pl.DataFrame({'time': [1]}), ['time'])


# ------------------------------------------------------
# Join identity and missing-value contracts
# ------------------------------------------------------
@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('side', ['feature', 'target'])
@pytest.mark.parametrize('key', ['time', 'symbol'])
@pytest.mark.parametrize('missing', [None, np.nan])
@pytest.mark.parametrize('check_all_missing', [True, False])
def test_join_rejects_partially_missing_keys(constructor, side, key, missing, check_all_missing):
    """Should reject invalid identities on either side before attempting a join."""
    left = {'time': [1., 2.], 'symbol': [10., 20.], 'x': [1., 2.]}
    right = {'time': [1., 2.], 'symbol': [10., 20.], 'y': [3., 4.]}
    (left if side == 'feature' else right)[key][1] = missing
    with pytest.raises(ValueError, match=f'{key} must not contain missing values'):
        join_feature_target_frames(constructor(left), constructor(right), 'x', 'y', 'time', 'symbol',
                                   check_all_missing=check_all_missing)


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('side', ['feature', 'target'])
@pytest.mark.parametrize('key', ['time', 'symbol'])
@pytest.mark.parametrize('infinite', [np.inf, -np.inf])
def test_join_rejects_infinite_keys(constructor, side, key, infinite):
    """Should reject infinite identities while leaving value policies to consumers."""
    left = {'time': [1., 2.], 'symbol': [10., 20.], 'x': [1., 2.]}
    right = {'time': [1., 2.], 'symbol': [10., 20.], 'y': [3., 4.]}
    (left if side == 'feature' else right)[key][1] = infinite
    with pytest.raises(ValueError, match=f'{key} must contain only finite values'):
        join_feature_target_frames(constructor(left), constructor(right), 'x', 'y', 'time', 'symbol')


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('side', ['feature', 'target'])
@pytest.mark.parametrize('key', ['time', 'symbol'])
def test_join_all_missing_opt_out_never_allows_missing_keys(constructor, side, key):
    """Should apply the all-missing opt-out only to values and never to identities."""
    left = {'time': [1.], 'symbol': [10.], 'x': [1.]}
    right = {'time': [1.], 'symbol': [10.], 'y': [2.]}
    (left if side == 'feature' else right)[key] = [None]
    with pytest.raises(ValueError, match=f'{key} must not contain missing values'):
        join_feature_target_frames(constructor(left), constructor(right), 'x', 'y', 'time', 'symbol',
                                   check_all_missing=False)


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('side', ['feature', 'target'])
def test_join_rejects_nat_keys(constructor, side):
    """Should reject missing datetime keys rather than matching or discarding them."""
    dates = [pd.Timestamp('2024-01-01'), None]
    left = constructor({'time': dates if side == 'feature' else dates[:1] * 2,
                        'symbol': ['A', 'B'], 'x': [1., 2.]})
    right = constructor({'time': dates if side == 'target' else dates[:1] * 2,
                         'symbol': ['A', 'B'], 'y': [3., 4.]})
    with pytest.raises(ValueError, match='time must not contain missing values'):
        join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('left_keys, right_keys', [
    ([1, 2], [1., 2.]), ([1, 2], ['1', '2']),
    ([True, False], [1, 0]), ([b'A', b'B'], ['A', 'B']),
    ([date(2024, 1, 1)], [pd.Timestamp('2024-01-01')]),
])
def test_join_rejects_incompatible_identity_domains(constructor, left_keys, right_keys):
    """Should never silently coerce identifiers across distinct semantic domains."""
    left = constructor({'time': left_keys, 'symbol': ['A'] * len(left_keys), 'x': [1.] * len(left_keys)})
    right = constructor({'time': right_keys, 'symbol': ['A'] * len(right_keys), 'y': [2.] * len(right_keys)})
    with pytest.raises(TypeError, match='incompatible join-key types for time'):
        join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('left_dtype, right_dtype', [('int8', 'int64'), ('uint16', 'int32'), ('float32', 'float64')])
def test_join_accepts_lossless_numeric_width_changes(constructor, left_dtype, right_dtype):
    """Should match compatible numeric widths and restore the feature key dtype."""
    left = constructor({'time': np.array([2, 1], dtype=left_dtype), 'symbol': ['A', 'A'], 'x': [20., 10.]})
    right = constructor({'time': np.array([1, 2], dtype=right_dtype), 'symbol': ['A', 'A'], 'y': [100., 200.]})
    result = join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')
    assert result['time'].to_list() == [2, 1]
    assert result['y'].to_list() == [200., 100.]
    assert result['time'].dtype == left['time'].dtype


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_join_rejects_integer_promotion_through_float(constructor):
    """Should reject signed/unsigned combinations without an exact common integer dtype."""
    left = constructor({'time': np.array([1], dtype=np.int64), 'symbol': ['A'], 'x': [1.]})
    right = constructor({'time': np.array([1], dtype=np.uint64), 'symbol': ['A'], 'y': [2.]})
    with pytest.raises(TypeError, match='normalization must be lossless'):
        join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_join_keeps_large_integer_identities_distinct(constructor):
    """Should distinguish adjacent integer identities above the exact Float64 range."""
    left = constructor({'time': [2 ** 53, 2 ** 53 + 1], 'symbol': ['A', 'A'], 'x': [1., 2.]})
    right = constructor({'time': [2 ** 53 + 1, 2 ** 53], 'symbol': ['A', 'A'], 'y': [10., 20.]})
    result = join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')
    assert result['time'].to_list() == [2 ** 53, 2 ** 53 + 1]
    assert result['y'].to_list() == [20., 10.]


# ------------------------------------------------------
# Temporal precision, timezone, and storage contracts
# ------------------------------------------------------
@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('aware_side', ['feature', 'target'])
def test_join_rejects_naive_aware_mixtures(constructor, aware_side):
    """Should reject ambiguous timezone interpretation on either join side."""
    naive = [pd.Timestamp('2024-01-01 12:00')]
    aware = [pd.Timestamp('2024-01-01 12:00', tz='UTC')]
    left = constructor({'time': aware if aware_side == 'feature' else naive, 'symbol': ['A'], 'x': [1.]})
    right = constructor({'time': aware if aware_side == 'target' else naive, 'symbol': ['A'], 'y': [2.]})
    with pytest.raises(TypeError, match='incompatible join-key types for time'):
        join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_join_matches_timezones_across_dst_by_instant(constructor):
    """Should distinguish repeated DST wall times and preserve the feature timezone."""
    # The two 01:30 wall times have UTC offsets -04:00 and -05:00.
    utc = pd.date_range('2024-11-03 05:30', periods=2, freq='h', tz='UTC')
    left = constructor({'time': utc.tz_convert('America/New_York'), 'symbol': ['A', 'A'], 'x': [1., 2.]})
    right = constructor({'time': utc[::-1].tz_convert('America/Sao_Paulo'), 'symbol': ['A', 'A'], 'y': [20., 10.]})
    left_before = left.copy(deep=True) if isinstance(left, pd.DataFrame) else left.clone()
    right_before = right.copy(deep=True) if isinstance(right, pd.DataFrame) else right.clone()
    result = join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')
    assert result['time'].to_list() == left['time'].to_list()
    assert result['time'].dtype == left['time'].dtype
    assert result['y'].to_list() == [10., 20.]
    if isinstance(left, pd.DataFrame):
        pd.testing.assert_frame_equal(left, left_before)
        pd.testing.assert_frame_equal(right, right_before)
    else:
        assert left.equals(left_before)
        assert right.equals(right_before)


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('kind', ['datetime64', 'timedelta64'])
def test_join_uses_finer_temporal_precision_without_false_matches(constructor, kind):
    """Should not round a target timestamp or duration into an existing feature key."""
    if kind == 'datetime64':
        left_values = np.array(['2024-01-01T00:00:00.000', '2024-01-01T00:00:00.001'], dtype=f'{kind}[ms]')
        right_values = np.array(['2024-01-01T00:00:00.001000001', '2024-01-01T00:00:00.000000000'], dtype=f'{kind}[ns]')
    else:
        left_values = np.array([0, 1], dtype=f'{kind}[ms]')
        right_values = np.array([1000001, 0], dtype=f'{kind}[ns]')
    left = constructor({'time': left_values, 'symbol': ['A', 'A'], 'x': [1., 2.]})
    right = constructor({'time': right_values, 'symbol': ['A', 'A'], 'y': [10., 20.]})
    result = join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')
    assert len(result) == 1
    assert result['x'].to_list() == [1.]
    assert result['y'].to_list() == [20.]
    assert result['time'].dtype == left['time'].dtype


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('kind', ['datetime64', 'timedelta64'])
def test_align_join_keys_rejects_temporal_conversion_overflow(constructor, kind):
    """Should reject finer-unit conversion that would wrap finite temporal keys."""
    # 20 billion seconds exceeds the approximately 292-year Int64-ns range.
    left_values = (np.array(['2300-01-01'], dtype='datetime64[us]') if kind == 'datetime64'
                   else np.array([20_000_000_000_000_000], dtype='timedelta64[us]'))
    right_values = (np.array(['2024-01-01'], dtype='datetime64[ns]') if kind == 'datetime64'
                    else np.array([0], dtype='timedelta64[ns]'))
    left = constructor({'time': left_values})
    right = constructor({'time': right_values})
    with pytest.raises(TypeError, match='normalization must be lossless'):
        _align_join_keys(left, right, ['time'])


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_join_preserves_categorical_feature_keys(constructor):
    """Should compare category labels rather than unrelated category codes."""
    if constructor is pd.DataFrame:
        left_symbols = pd.Categorical(['B', 'A'], categories=['A', 'B', 'unused'], ordered=True)
        right_symbols = pd.Categorical(['A', 'B'], categories=['B', 'A'], ordered=False)
    else:
        left_symbols = pl.Series(['B', 'A']).cast(pl.Enum(['A', 'B', 'unused']))
        right_symbols = pl.Series(['A', 'B']).cast(pl.Categorical)
    left = constructor({'time': [1, 1], 'symbol': left_symbols, 'x': [2., 1.]})
    right = constructor({'time': [1, 1], 'symbol': right_symbols, 'y': [10., 20.]})
    result = join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')
    assert result['symbol'].to_list() == ['B', 'A']
    assert result['symbol'].dtype == left['symbol'].dtype
    assert result['y'].to_list() == [20., 10.]


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_join_float_keys_are_not_rounded_for_matching(constructor):
    """Should compare floating identities at their represented precision."""
    left = constructor({'time': np.array([0.1], dtype=np.float32), 'symbol': ['A'], 'x': [1.]})
    right = constructor({'time': np.array([0.1], dtype=np.float64), 'symbol': ['A'], 'y': [2.]})
    # Float32(0.1) and Float64(0.1) are different exact binary numbers.
    assert len(join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')) == 0


# ------------------------------------------------------
# Output schema, row order, indexes, and non-mutation
# ------------------------------------------------------
@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_join_preserves_feature_order_with_interleaved_assets(constructor):
    """Should return explicitly paired rows in feature order without sorting time."""
    left = constructor({'time': [3, 1, 2, 4, 1], 'symbol': ['A', 'B', 'A', 'A', 'A'],
                        'x': [30., 10., np.nan, 40., 11.], 'metadata': ['left'] * 5})
    right = constructor({'time': [4, 1, 2, 1, 3], 'symbol': ['A', 'A', 'A', 'B', 'A'],
                         'y': [400., 110., 200., 100., 300.], 'metadata': ['right'] * 5})
    left_before = left.copy(deep=True) if isinstance(left, pd.DataFrame) else left.clone()
    right_before = right.copy(deep=True) if isinstance(right, pd.DataFrame) else right.clone()
    result = join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')
    expected = {'time': [3, 1, 4, 1], 'symbol': ['A', 'B', 'A', 'A'],
                'x': [30., 10., 40., 11.], 'y': [300., 100., 400., 110.]}
    if isinstance(result, pd.DataFrame):
        pd.testing.assert_frame_equal(result, pd.DataFrame(expected))
        pd.testing.assert_frame_equal(left, left_before)
        pd.testing.assert_frame_equal(right, right_before)
    else:
        assert result.equals(pl.DataFrame(expected))
        assert left.equals(left_before)
        assert right.equals(right_before)


@pytest.mark.parametrize('index', [pd.Index([90, 10, 90], name='original'),
                                 pd.MultiIndex.from_tuples([('B', 1), ('A', 2), ('B', 1)])])
def test_join_uses_key_values_instead_of_pandas_indexes(index):
    """Should pair by keys and return RangeIndex regardless of input index labels."""
    left = pd.DataFrame({'time': [3, 1, 2], 'symbol': ['A'] * 3,
                         'x': pd.array([30, None, 20], dtype='Float64')}, index=index)
    right = pd.DataFrame({'time': [2, 3, 1], 'symbol': ['A'] * 3,
                          'y': pd.array([200, 300, 100], dtype='Int64')}, index=['z', 'x', 'y'])
    result = join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')
    expected = pd.DataFrame({'time': [3, 2], 'symbol': ['A', 'A'],
                             'x': pd.array([30, 20], dtype='Float64'), 'y': pd.array([300, 200], dtype='Int64')})
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('reason', ['no_overlap', 'no_valid_pairs', 'missing_feature', 'missing_target'])
def test_join_empty_results_preserve_exact_schema(constructor, reason):
    """Should retain all four input column dtypes when no usable pair remains."""
    x = [None, None] if reason == 'missing_feature' else [None, 2.]
    y = [None, None] if reason == 'missing_target' else [3., None]
    left = constructor({'time': np.array([1, 2], dtype=np.int32), 'symbol': ['A', 'B'], 'x': x})
    right = constructor({'time': np.array([3, 4] if reason == 'no_overlap' else [1, 2], dtype=np.int64),
                         'symbol': ['A', 'B'], 'y': y})
    result = join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol', check_all_missing=False)
    assert result.shape == (0, 4)
    assert list(result.columns) == ['time', 'symbol', 'x', 'y']
    assert [result[column].dtype for column in result.columns] == [
        left['time'].dtype, left['symbol'].dtype, left['x'].dtype, right['y'].dtype,
    ]
    if isinstance(result, pd.DataFrame):
        assert isinstance(result.index, pd.RangeIndex)


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('side', ['feature', 'target'])
@pytest.mark.parametrize('missing', [None, np.nan])
def test_join_default_rejects_all_missing_value_columns(constructor, side, missing):
    """Should retain default rejection for wholly absent feature or target values."""
    left = constructor({'time': [1, 2], 'symbol': ['A', 'A'], 'x': [missing] * 2 if side == 'feature' else [1., 2.]})
    right = constructor({'time': [1, 2], 'symbol': ['A', 'A'], 'y': [missing] * 2 if side == 'target' else [3., 4.]})
    with pytest.raises(ValueError, match='missing all values'):
        join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_join_leaves_infinite_value_policy_to_consumers(constructor):
    """Should retain infinite values while removing only missing feature-target pairs."""
    left = constructor({'time': [1, 2, 3], 'symbol': ['A'] * 3, 'x': [np.inf, -np.inf, np.nan]})
    right = constructor({'time': [1, 2, 3], 'symbol': ['A'] * 3, 'y': [1., 2., 3.]})
    result = join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')
    assert result['time'].to_list() == [1, 2]
    assert result['x'].to_list() == [np.inf, -np.inf]


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_join_allows_same_frame_for_distinct_value_columns(constructor):
    """Should select distinct values from an aliased input without mutating it."""
    frame = constructor({'time': [2, 1], 'symbol': ['A', 'A'], 'x': [2., 1.], 'y': [20., 10.]})
    result = join_feature_target_frames(frame, frame, 'x', 'y', 'time', 'symbol')
    assert result['y'].to_list() == [20., 10.]
    assert list(frame.columns) == ['time', 'symbol', 'x', 'y']


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('role', ['time', 'symbol', 'x', 'y'])
def test_join_internal_position_name_cannot_collide_with_user_columns(constructor, role):
    """Should preserve user-defined names that resemble the temporary position column."""
    names = {name: '__alpha_research_position' if name == role else name for name in ('time', 'symbol', 'x', 'y')}
    left = constructor({names['time']: [2, 1], names['symbol']: ['A', 'A'], names['x']: [2., 1.]})
    right = constructor({names['time']: [1, 2], names['symbol']: ['A', 'A'], names['y']: [10., 20.]})
    result = join_feature_target_frames(left, right, names['x'], names['y'], names['time'], names['symbol'])
    assert list(result.columns) == [names[name] for name in ('time', 'symbol', 'x', 'y')]
    assert result[names['y']].to_list() == [20., 10.]


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('names', [('x', 'x', 'time', 'symbol'), ('time', 'y', 'time', 'symbol'),
                                 ('x', 'symbol', 'time', 'symbol'), ('x', 'y', 'time', 'time')])
def test_join_rejects_overlapping_column_roles(constructor, names):
    """Should reject role collisions explicitly rather than suffixing or duplicating columns."""
    frame = constructor({'time': [1], 'symbol': ['A'], 'x': [1.], 'y': [2.]})
    with pytest.raises(ValueError, match='distinct names'):
        join_feature_target_frames(frame, frame, *names)


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('position', range(4))
@pytest.mark.parametrize('invalid', ['', None, 1, ['x']])
def test_join_validates_each_column_name(constructor, position, invalid):
    """Should reject malformed names for every public column-name parameter."""
    frame = constructor({'time': [1], 'symbol': ['A'], 'x': [1.], 'y': [2.]})
    names = ['x', 'y', 'time', 'symbol']
    names[position] = invalid
    with pytest.raises(TypeError, match='non-empty strings'):
        join_feature_target_frames(frame, frame, *names)


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('side, column', [('feature', 'time'), ('target', 'time'),
                                       ('feature', 'symbol'), ('target', 'symbol'),
                                       ('feature', 'x'), ('target', 'y')])
def test_join_checks_required_columns_on_both_inputs(constructor, side, column):
    """Should identify a missing required key or value on either side."""
    left = {'time': [1], 'symbol': ['A'], 'x': [1.]}
    right = {'time': [1], 'symbol': ['A'], 'y': [2.]}
    del (left if side == 'feature' else right)[column]
    with pytest.raises(KeyError, match=column):
        join_feature_target_frames(constructor(left), constructor(right), 'x', 'y', 'time', 'symbol')


@pytest.mark.parametrize('side', ['feature', 'target'])
def test_join_rejects_ambiguous_pandas_key_columns(side):
    """Should report duplicate selected labels before Pandas attempts a merge."""
    left = pd.DataFrame([[1, 'A', 1.]], columns=['time', 'symbol', 'x'])
    right = pd.DataFrame([[1, 'A', 2.]], columns=['time', 'symbol', 'y'])
    if side == 'feature':
        left = pd.concat([left, left[['time']]], axis=1)
    else:
        right = pd.concat([right, right[['time']]], axis=1)
    with pytest.raises(ValueError, match='required column names must be unique.*time'):
        join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_join_disjoint_valid_frames_return_typed_empty_output(constructor):
    """Should allow an empty intersection under the default validation policy."""
    left = constructor({'time': [1], 'symbol': ['A'], 'x': ['observed']})
    right = constructor({'time': [2], 'symbol': ['A'], 'y': [True]})
    result = join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')
    assert result.shape == (0, 4)
    assert list(result.columns) == ['time', 'symbol', 'x', 'y']
    assert result['x'].dtype == left['x'].dtype
    assert result['y'].dtype == right['y'].dtype


def test_join_preserves_pandas_nullable_and_categorical_value_dtypes():
    """Should preserve nullable keys and categorical values after pairing and filtering."""
    left = pd.DataFrame({'time': pd.array([2, 1, 3], dtype='Int32'),
                         'symbol': pd.array(['B', 'A', 'C'], dtype='string'),
                         'x': pd.Categorical(['high', 'low', None], categories=['low', 'high'], ordered=True)})
    right = pd.DataFrame({'time': pd.array([1, 3, 2], dtype='Int64'),
                          'symbol': ['A', 'C', 'B'], 'y': pd.array([10, 30, 20], dtype='Int64')})
    expected = pd.DataFrame({'time': pd.array([2, 1], dtype='Int32'),
                             'symbol': pd.array(['B', 'A'], dtype='string'),
                             'x': pd.Categorical(['high', 'low'], categories=['low', 'high'], ordered=True),
                             'y': pd.array([20, 10], dtype='Int64')})
    pd.testing.assert_frame_equal(join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol'), expected)


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('side', ['feature', 'target'])
def test_join_validates_keys_on_unmatched_rows_too(constructor, side):
    """Should reject invalid input identities even if an inner join would discard them."""
    left = constructor({'time': [1., None] if side == 'feature' else [1., 2.],
                        'symbol': ['A', 'B'], 'x': [1., 2.]})
    right = constructor({'time': [1., None] if side == 'target' else [1., 3.],
                         'symbol': ['A', 'C'], 'y': [3., 4.]})
    with pytest.raises(ValueError, match='time must not contain missing values'):
        join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('kind', ['date', 'time', 'boolean', 'bytes'])
def test_join_matches_other_supported_scalar_domains(constructor, kind):
    """Should pair supported date, clock-time, boolean, and binary identities exactly."""
    values = {'date': [date(2024, 1, 2), date(2024, 1, 1)],
              'time': [time(10), time(9)], 'boolean': [True, False], 'bytes': [b'B', b'A']}[kind]
    left = constructor({'time': values, 'symbol': ['A', 'A'], 'x': [2., 1.]})
    right = constructor({'time': values[::-1], 'symbol': ['A', 'A'], 'y': [10., 20.]})
    result = join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')
    assert result['time'].to_list() == values
    assert result['y'].to_list() == [20., 10.]
    assert result['time'].dtype == left['time'].dtype


def test_align_join_keys_rejects_ambiguous_selected_labels_directly():
    """Should reject ambiguous key columns independently of the public join."""
    ambiguous = pd.DataFrame([[1, 2]], columns=['time', 'time'])
    with pytest.raises(ValueError, match='required column names must be unique'):
        _align_join_keys(ambiguous, pd.DataFrame({'time': [1]}), ['time'])


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_join_combines_timezone_and_precision_normalization(constructor):
    """Should match equal aware instants without rounding nearby nanosecond keys."""
    local = pd.Series(pd.date_range('2024-01-02 09:30', periods=2, freq='h', tz='America/Sao_Paulo'))
    local = local.astype(pd.DatetimeTZDtype(unit='ms', tz='America/Sao_Paulo'))
    utc = pd.Series(pd.to_datetime(['2024-01-02T12:30:00.000000001Z', '2024-01-02T13:30:00Z'], format='ISO8601'))
    utc = utc.astype(pd.DatetimeTZDtype(unit='ns', tz='UTC'))
    left = constructor({'time': local, 'symbol': ['A', 'A'], 'x': [1., 2.]})
    right = constructor({'time': utc, 'symbol': ['A', 'A'], 'y': [10., 20.]})
    result = join_feature_target_frames(left, right, 'x', 'y', 'time', 'symbol')
    assert result['x'].to_list() == [2.]
    assert result['y'].to_list() == [20.]
    assert result['time'].dtype == left['time'].dtype
