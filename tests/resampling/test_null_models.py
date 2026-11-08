from unittest.mock import Mock

import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.resampling import circular_shift, generate_random_walk


# ------------------------------------------------------
# fixtures
# ------------------------------------------------------
@pytest.fixture(params=['pandas', 'polars'])
def feature_series(request):
    """Should provide the same named feature in both supported backends."""
    values = [11, 23, 37, 41, 59, 61]
    if request.param == 'pandas':
        return pd.Series(values, name='feature', index=pd.Index([9, 4, 8, 2, 7, 1], name='row'))
    return pl.Series('feature', values)


# ------------------------------------------------------
# generate_random_walk
# ------------------------------------------------------
def test_random_walk_uses_hand_calculated_recurrence(monkeypatch):
    """Should include the starting value and accumulate independently specified Gaussian steps."""
    # These prescribed increments make the recurrence hand-verifiable;
    # the oracle does not recreate NumPy's sampling or cumulative-sum algorithm.
    rng = Mock()
    rng.normal.return_value = np.array([2., -1., 0., 3.])
    factory = Mock(return_value=rng)
    monkeypatch.setattr('alpha_research.resampling.null_models.np.random.default_rng', factory)

    result = generate_random_walk(5, 17, initial_value=10., step_std=2.5)

    np.testing.assert_array_equal(result, [10., 12., 11., 11., 14.])
    assert result.shape == (5,) and result.dtype == np.dtype('float64')
    factory.assert_called_once_with(17)
    rng.normal.assert_called_once_with(0., 2.5, size=4)


def test_random_walk_defaults_follow_zero_mean_unit_steps(monkeypatch):
    """Should start at zero and request exactly one standard Gaussian step per transition."""
    rng = Mock()
    rng.normal.return_value = np.array([-2., 3.])
    factory = Mock(return_value=rng)
    monkeypatch.setattr('alpha_research.resampling.null_models.np.random.default_rng', factory)

    np.testing.assert_array_equal(generate_random_walk(3), [0., -2., 1.])
    factory.assert_called_once_with(None)
    rng.normal.assert_called_once_with(0., 1., size=2)


def test_random_walk_singleton_draws_no_increments(monkeypatch):
    """Should return only the initial value without creating a generator for a singleton."""
    factory = Mock(side_effect=AssertionError('no random draws expected'))
    monkeypatch.setattr('alpha_research.resampling.null_models.np.random.default_rng', factory)

    np.testing.assert_array_equal(generate_random_walk(1, 9, initial_value=-7.), [-7.])
    factory.assert_not_called()


@pytest.mark.parametrize('seed', [0, 19, np.int64(23), np.uint64(31), 10 ** 100])
def test_random_walk_reproducibility_and_prefix(seed):
    """Should reproduce each seeded path and retain its prefix when requesting more observations."""
    short = generate_random_walk(8, seed)
    same = generate_random_walk(8, seed)
    longer = generate_random_walk(20, seed)

    np.testing.assert_array_equal(short, same)
    np.testing.assert_array_equal(short, longer[:8])
    assert np.isfinite(longer).all()
    assert np.any(short[1:] != short[:-1])
    short[0] = 99.
    assert same[0] == longer[0] == 0.


def test_random_walk_uses_local_rng_without_touching_legacy_state():
    """Should leave NumPy's global legacy RNG untouched for seeded and unseeded paths."""
    before = np.random.get_state()

    generate_random_walk(12, 7)
    generate_random_walk(12)

    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_random_walk_distinct_seeds_and_backend_conversion():
    """Should produce distinct seeded paths and allow identical Pandas and Polars wrapping."""
    values = generate_random_walk(np.uint64(12), np.int32(3), initial_value=np.float32(2), step_std=np.float64(.5))

    assert not np.array_equal(values, generate_random_walk(12, 4, initial_value=2., step_std=.5))
    assert pd.Series(values).to_list() == pl.Series(values).to_list()
    assert values[0] == 2.


@pytest.mark.parametrize('sample_size', [0, -1, 2.5, '3', True, np.bool_(False),
                                       np.timedelta64(2, 'D'), None, np.array(3), [3]])
def test_random_walk_rejects_invalid_sample_size_before_rng(sample_size, monkeypatch):
    """Should reject invalid observation counts before any random work."""
    factory = Mock(side_effect=AssertionError('invalid input must not draw'))
    monkeypatch.setattr('alpha_research.resampling.null_models.np.random.default_rng', factory)

    with pytest.raises(ValueError, match='sample_size must be a positive integer'):
        generate_random_walk(sample_size)
    factory.assert_not_called()


@pytest.mark.parametrize('seed, error', [(-1, ValueError), (np.int64(-2), ValueError),
                                       (True, TypeError), (np.bool_(False), TypeError),
                                       (1., TypeError), ('1', TypeError), ([], TypeError),
                                       (np.timedelta64(1, 'D'), TypeError),
                                       (np.array(1), TypeError), (np.random.default_rng(1), TypeError)])
def test_random_walk_rejects_invalid_seeds_even_for_singleton(seed, error, monkeypatch):
    """Should validate seeds even when no innovations are required."""
    factory = Mock(side_effect=AssertionError('invalid input must not draw'))
    monkeypatch.setattr('alpha_research.resampling.null_models.np.random.default_rng', factory)

    with pytest.raises(error):
        generate_random_walk(1, seed)
    factory.assert_not_called()


@pytest.mark.parametrize('parameter', ['initial_value', 'step_std'])
@pytest.mark.parametrize('value, error', [
    (None, TypeError), ('1', TypeError), (True, TypeError), (np.bool_(True), TypeError),
    (1j, TypeError), ([1.], TypeError), (np.array(1.), TypeError),
    (np.timedelta64(1, 'D'), TypeError), (np.datetime64('2024-01-01'), TypeError),
    (np.nan, ValueError), (np.inf, ValueError), (-np.inf, ValueError), (10 ** 400, ValueError),
])
def test_random_walk_rejects_invalid_numeric_parameters(parameter, value, error, monkeypatch):
    """Should reject non-real, missing, infinite, and unrepresentable parameters before sampling."""
    factory = Mock(side_effect=AssertionError('invalid input must not draw'))
    monkeypatch.setattr('alpha_research.resampling.null_models.np.random.default_rng', factory)

    with pytest.raises(error, match=parameter):
        generate_random_walk(3, **{parameter: value})
    factory.assert_not_called()


@pytest.mark.parametrize('step_std', [0, -0., -1., np.float32(-.5)])
def test_random_walk_rejects_non_positive_step_deviation(step_std):
    """Should reject a non-positive innovation deviation even for singleton paths."""
    with pytest.raises(ValueError, match='step_std must be strictly positive'):
        generate_random_walk(1, step_std=step_std)


@pytest.mark.parametrize('initial, steps', [
    (0., [1e308, 1e308]), (1e308, [1e308]),
    (0., [np.inf]), (0., [-np.inf]), (0., [np.nan]),
])
def test_random_walk_rejects_non_finite_generated_paths(initial, steps, monkeypatch):
    """Should report unrepresentable paths explicitly and restore the caller's NumPy error settings."""
    rng = Mock()
    rng.normal.return_value = np.array(steps)
    monkeypatch.setattr('alpha_research.resampling.null_models.np.random.default_rng', Mock(return_value=rng))

    with np.errstate(over='raise', invalid='raise'):
        before = np.geterr().copy()
        with pytest.raises(ValueError, match='random walk must contain only finite'):
            generate_random_walk(len(steps) + 1, initial_value=initial)
        assert np.geterr() == before


def test_random_walk_includes_initial_value_before_large_steps(monkeypatch):
    """Should keep a finite recurrence when summing the innovations separately would overflow."""
    rng = Mock()
    rng.normal.return_value = np.array([1e308, 1e308])
    monkeypatch.setattr('alpha_research.resampling.null_models.np.random.default_rng', Mock(return_value=rng))

    with np.errstate(over='raise', invalid='raise'):
        result = generate_random_walk(3, initial_value=-1e308)

    np.testing.assert_array_equal(result, [-1e308, 0., 1e308])


# ------------------------------------------------------
# circular_shift
# ------------------------------------------------------
@pytest.mark.parametrize('shift, min_shift, expected', [
    (1, 1, [61, 11, 23, 37, 41, 59]),
    (2, 2, [59, 61, 11, 23, 37, 41]),
    (3, 3, [41, 59, 61, 11, 23, 37]),
    (4, 2, [37, 41, 59, 61, 11, 23]),
    (5, 1, [23, 37, 41, 59, 61, 11]),
])
def test_circular_shift_matches_hand_written_rotations(feature_series, shift, min_shift, expected):
    """Should rotate values right by the specified offset while preserving backend, name, and dtype."""
    result = circular_shift(feature_series, shift, min_shift)

    assert type(result) is type(feature_series)
    assert result.to_list() == expected
    assert result.name == feature_series.name and result.dtype == feature_series.dtype
    assert feature_series.to_list() == [11, 23, 37, 41, 59, 61]
    if isinstance(feature_series, pd.Series):
        pd.testing.assert_index_equal(result.index, feature_series.index, exact=True)


@pytest.mark.parametrize('index', [
    pd.date_range('2024-01-01', periods=4, tz='UTC', name='time'),
    pd.Index(['b', 'a', 'b', 'c'], name='duplicate_labels'),
    pd.MultiIndex.from_tuples([('b', 2), ('a', 1), ('b', 1), ('a', 2)], names=['asset', 'row']),
])
def test_circular_shift_keeps_pandas_destination_labels_on_frame_assignment(index):
    """Should preserve target and time alignment when assigning a shifted feature back to its frame."""
    frame = pd.DataFrame({'feature': [10, 20, 30, 40], 'target': [1, 2, 3, 4]}, index=index)
    before = frame.copy(deep=True)

    frame['placebo'] = circular_shift(frame['feature'], 2, min_shift=2)

    assert frame['placebo'].to_list() == [30, 40, 10, 20]
    pd.testing.assert_frame_equal(frame[['feature', 'target']], before)
    pd.testing.assert_index_equal(frame.index, before.index, exact=True)


@pytest.mark.parametrize('data, expected', [
    (pd.Series([1, None, 3, 4], dtype='Int64', name='nullable'), [4, 1, pd.NA, 3]),
    (pd.Series(['a', 'b', 'a', None], dtype='category', name='category'), [np.nan, 'a', 'b', 'a']),
    (pd.Series(['a', None, 'c', 'd'], dtype='string'), ['d', 'a', pd.NA, 'c']),
    (pd.Series([True, None, False, True], dtype='boolean'), [True, True, pd.NA, False]),
    (pd.Series([2 ** 64 - 1, 1, 2, 3], dtype='uint64'), [3, 2 ** 64 - 1, 1, 2]),
])
def test_circular_shift_preserves_pandas_extension_values(data, expected):
    """Should preserve missing sentinels, categories, nullable dtypes, and exact large integers."""
    before = data.copy(deep=True)
    result = circular_shift(data, 1)

    pd.testing.assert_series_equal(result, pd.Series(expected, dtype=data.dtype, name=data.name))
    pd.testing.assert_series_equal(data, before)
    if isinstance(data.dtype, pd.CategoricalDtype):
        pd.testing.assert_index_equal(result.cat.categories, data.cat.categories)


@pytest.mark.parametrize('dtype, values, expected', [
    (pl.Int64, [1, None, 3, 4], [4, 1, None, 3]),
    (pl.Categorical, ['a', 'b', 'a', None], [None, 'a', 'b', 'a']),
    (pl.Boolean, [True, None, False, True], [True, True, None, False]),
    (pl.UInt64, [2 ** 64 - 1, 1, 2, 3], [3, 2 ** 64 - 1, 1, 2]),
    (pl.List(pl.Int64), [[1], None, [], [2, 3]], [[2, 3], [1], None, []]),
])
def test_circular_shift_preserves_polars_native_values(dtype, values, expected):
    """Should retain native missing values, categorical and nested schemas, and exact large integers."""
    data = pl.Series('feature', values, dtype=dtype)
    before = data.clone()
    result = circular_shift(data, 1)

    assert result.equals(pl.Series('feature', expected, dtype=dtype))
    assert result.dtype == data.dtype and data.equals(before)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_circular_shift_preserves_nan_infinity_and_all_missing_data(backend):
    """Should permute every observation without filtering or numerically coercing special values."""
    values = [np.nan, np.inf, -np.inf, 3.]
    data = pd.Series(values) if backend == 'pandas' else pl.Series(values)
    result = circular_shift(data, 2)

    np.testing.assert_array_equal(result.to_numpy(), [-np.inf, 3., np.nan, np.inf])
    missing = pd.Series([pd.NA] * 4, dtype='Float64') if backend == 'pandas' else pl.Series([None] * 4)
    shifted = circular_shift(missing, 2)
    assert shifted.dtype == missing.dtype
    assert (shifted.isna().all() if backend == 'pandas' else shifted.null_count() == 4)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('values', [[7, 7, 7, 7], [1, 2, 1, 2]])
def test_circular_shift_allows_identity_values_for_constant_or_periodic_data(backend, values):
    """Should validate offsets without falsely promising changed values for constant or periodic inputs."""
    data = pd.Series(values) if backend == 'pandas' else pl.Series(values)
    assert circular_shift(data, 2, min_shift=2).to_list() == values


def test_circular_shift_result_does_not_mutate_input(feature_series):
    """Should return an independently mutable result rather than overwrite the source feature."""
    shifted = circular_shift(feature_series, 2)
    if isinstance(shifted, pd.Series):
        shifted.iloc[0] = -100
    else:
        shifted[0] = -100
    assert feature_series.to_list() == [11, 23, 37, 41, 59, 61]


@pytest.mark.parametrize('data', [None, [1, 2], np.array([1, 2]),
                               pd.DataFrame({'feature': [1, 2]}), pl.DataFrame({'feature': [1, 2]})])
def test_circular_shift_rejects_unsupported_inputs(data):
    """Should reject frames and arrays so targets or temporal keys cannot be rotated accidentally."""
    with pytest.raises(TypeError, match='Pandas or Polars Series'):
        circular_shift(data, 1)


@pytest.mark.parametrize('data', [pd.Series([], dtype=float), pl.Series([], dtype=pl.Float64),
                               pd.Series([1]), pl.Series([1])])
def test_circular_shift_rejects_short_series(data):
    """Should reject empty and singleton inputs that have no nonzero circular offset."""
    with pytest.raises(ValueError, match='at least two observations'):
        circular_shift(data, 1)


@pytest.mark.parametrize('parameter', ['shift', 'min_shift'])
@pytest.mark.parametrize('value', [0, -1, 1.5, True, np.bool_(True), '2', None,
                                  np.timedelta64(2, 'D'), np.array(2), [2]])
def test_circular_shift_rejects_non_positive_or_non_integer_offsets(feature_series, parameter, value):
    """Should reject malformed offsets and exclusion distances without mutating the input."""
    arguments = {'shift': 2, 'min_shift': 1, parameter: value}
    with pytest.raises(ValueError, match=f'{parameter} must be a positive integer'):
        circular_shift(feature_series, **arguments)
    assert feature_series.to_list() == [11, 23, 37, 41, 59, 61]


@pytest.mark.parametrize('shift', [6, 7, 12, 13, 10 ** 100, np.uint64(2 ** 64 - 1)])
def test_circular_shift_rejects_full_cycles_and_oversized_offsets(feature_series, shift):
    """Should reject oversized shifts rather than wrap them to the identity or a nearby alignment."""
    with pytest.raises(ValueError, match='shift must be smaller'):
        circular_shift(feature_series, shift)


@pytest.mark.parametrize('shift', [1, 5])
def test_circular_shift_enforces_both_sides_of_exclusion(feature_series, shift):
    """Should reject near-original alignments from either circular direction."""
    with pytest.raises(ValueError, match='both circular directions'):
        circular_shift(feature_series, shift, min_shift=2)


@pytest.mark.parametrize('min_shift', [4, 6, 10 ** 100, np.uint64(2 ** 64 - 1)])
def test_circular_shift_rejects_impossible_exclusion(feature_series, min_shift):
    """Should report when the minimum distance excludes every possible rotation."""
    with pytest.raises(ValueError, match='excludes every possible'):
        circular_shift(feature_series, 2, min_shift=min_shift)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('shift', [np.int64(2), np.uint64(3)])
def test_circular_shift_accepts_numpy_offsets_at_odd_length_boundary(backend, shift):
    """Should accept both maximally distant offsets in an odd-length series without unsigned arithmetic errors."""
    values = [10, 20, 30, 40, 50]
    data = pd.Series(values) if backend == 'pandas' else pl.Series(values)
    expected = [40, 50, 10, 20, 30] if shift == 2 else [30, 40, 50, 10, 20]

    assert circular_shift(data, shift, min_shift=np.uint64(2)).to_list() == expected


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_circular_shift_supports_minimum_two_observations(backend):
    """Should exchange the only two observations at the sole admissible offset."""
    data = pd.Series([10, 20]) if backend == 'pandas' else pl.Series([10, 20])
    assert circular_shift(data, 1).to_list() == [20, 10]
