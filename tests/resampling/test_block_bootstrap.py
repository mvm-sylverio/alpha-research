from math import sqrt

import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.resampling.block_bootstrap import (
    BootstrapMetricsResults,
    _bootstrap_distribution_summary,
    _bootstrap_numeric_values,
    _concatenate_blocks,
    _validate_blocks,
    _validate_resampling_data,
    bootstrap_metrics,
    generate_moving_blocks,
    moving_block_bootstrap,
)


@pytest.mark.parametrize('backend', ['pandas_series', 'polars_series', 'pandas_frame', 'polars_frame'])
def test_block_progress_retains_continuity_and_order(backend):
    """Should report candidate filtering and retain only hand-verifiable contiguous blocks."""
    data = {
        'pandas_series': pd.Series([10, 20, 30, 40, 50]),
        'polars_series': pl.Series([10, 20, 30, 40, 50]),
        'pandas_frame': pd.DataFrame({'value': [10, 20, 30, 40, 50]}),
        'polars_frame': pl.DataFrame({'value': [10, 20, 30, 40, 50]}),
    }[backend]
    events = []
    blocks = generate_moving_blocks(data, 2, continuity=[0, 1, 4, 5, 6],
                                    on_progress=lambda *event: events.append(event))
    values = [block['value'] if 'frame' in backend else block for block in blocks]
    assert [value.to_list() for value in values] == [[10, 20], [30, 40], [40, 50]]
    assert ('candidate_blocks', 4, 4) in events
    assert events[-1] == ('moving_blocks', 3, 3)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_bootstrap_observation_preserves_draw_order_and_stops_future_draws(backend, monkeypatch):
    """Should keep predetermined RNG draws intact and never draw a sample after cancellation."""
    # Predetermined block choices form an independent, hand-verifiable oracle.
    blocks = [pd.Series([10, 20]), pd.Series([30, 40])]
    if backend == 'polars':
        blocks = [pl.Series(block) for block in blocks]
    draws = []
    seeds = []
    choices = [[1, 0], [0, 1], [1, 1]]

    class FixedGenerator:
        """Supply predetermined choices instead of reproducing the production RNG."""

        def integers(self, low, high, size):
            """Should record one original draw and return its predefined block indices."""
            assert (low, high, size) == (0, 2, 2)
            draws.append(choices[len(draws)])
            return np.array(draws[-1])

    monkeypatch.setattr('alpha_research.resampling.block_bootstrap.np.random.default_rng',
                        lambda seed: seeds.append(seed) or FixedGenerator())
    events = []
    samples = moving_block_bootstrap(blocks, 3, 3, 17, on_progress=lambda *event: events.append(event))
    assert [sample.to_list() for sample in samples] == [[30, 40, 10], [10, 20, 30], [30, 40, 30]]
    assert events[-1] == ('bootstrap_samples', 3, 3)
    assert seeds == [17]
    draws.clear()
    exception = TypeError('cancel sampling')
    callback = lambda phase, done, total: (_ for _ in ()).throw(exception) if done == 1 else None
    with pytest.raises(TypeError) as caught:
        moving_block_bootstrap(blocks, 3, 3, 17, on_progress=callback)
    assert caught.value is exception
    assert draws == [[1, 0]]


@pytest.mark.parametrize('function', [generate_moving_blocks, moving_block_bootstrap])
def test_resampling_rejects_invalid_observers_before_work(function):
    """Should validate observers before allocating blocks or bootstrap samples."""
    args = (pd.Series([1, 2]), 1) if function is generate_moving_blocks else ([pd.Series([1, 2])], 2, 2)
    with pytest.raises(TypeError, match='on_progress'):
        function(*args, on_progress=12)


# ------------------------------------------------------
# fixtures
# ------------------------------------------------------
@pytest.fixture
def series_pandas():
    return pd.Series([1, 2, 3, 4, 5, 6, 7])


@pytest.fixture
def series_polars(series_pandas):
    return pl.Series(series_pandas.to_list())

@pytest.fixture
def bootstrap_values():
    return [-2.0, -1.0, 0.0, 1.0, 2.0]


# ------------------------------------------------------
# _validate_resampling_data
# ------------------------------------------------------
@pytest.mark.parametrize('data', [
    pd.Series([1, 2]),
    pl.Series([1, 2]),
    pd.DataFrame({'value': [1, 2]}),
    pl.DataFrame({'value': [1, 2]}),
])
def test_validate_resampling_data_accepts_supported_data(data):
    """Should accept supported Pandas and Polars Series and DataFrames."""
    _validate_resampling_data(data)


def test_validate_resampling_data_rejects_unsupported_data():
    """Should reject data that is not a supported Series or DataFrame."""
    with pytest.raises(TypeError, match='Pandas or Polars Series or DataFrame'):
        _validate_resampling_data([1, 2])


# ------------------------------------------------------
# _validate_blocks
# ------------------------------------------------------
def test_validate_blocks_returns_common_length():
    """Should return the shared length of valid candidate blocks."""
    blocks = [pd.Series([1, 2]), pd.Series([2, 3])]

    assert _validate_blocks(blocks) == 2


def test_validate_blocks_rejects_empty_block_list():
    """Should reject an empty candidate block list."""
    with pytest.raises(ValueError, match='non-empty list'):
        _validate_blocks([])


def test_validate_blocks_rejects_empty_block():
    """Should reject a candidate block without observations."""
    with pytest.raises(ValueError, match='must not contain empty blocks'):
        _validate_blocks([pd.Series([], dtype=float)])


def test_validate_blocks_rejects_mixed_data_types():
    """Should reject blocks using different dataframe backends or types."""
    with pytest.raises(TypeError, match='same data type'):
        _validate_blocks([pd.Series([1, 2]), pl.Series([1, 2])])


def test_validate_blocks_rejects_different_lengths():
    """Should reject candidate blocks with different observation counts."""
    with pytest.raises(ValueError, match='same length'):
        _validate_blocks([pd.Series([1, 2]), pd.Series([2, 3, 4])])


# ------------------------------------------------------
# _concatenate_blocks
# ------------------------------------------------------
def test_concatenate_blocks_preserves_pandas_order_and_resets_index():
    """Should concatenate pandas blocks once in list order with a new index."""
    blocks = [pd.Series([1, 2], index=[10, 11]), pd.Series([3, 4], index=[20, 21])]

    result = _concatenate_blocks(blocks)

    assert result.to_list() == [1, 2, 3, 4]
    assert result.index.to_list() == [0, 1, 2, 3]


def test_concatenate_blocks_preserves_polars_order_and_backend():
    """Should concatenate Polars blocks once in list order."""
    result = _concatenate_blocks([pl.Series([1, 2]), pl.Series([3, 4])])

    assert isinstance(result, pl.Series)
    assert result.to_list() == [1, 2, 3, 4]


# ------------------------------------------------------
# generate_moving_blocks
# ------------------------------------------------------
def test_generate_moving_blocks_step_one(series_pandas):
    """Should generate canonical overlapping blocks when step is one."""
    blocks = generate_moving_blocks(series_pandas, block_length=3)

    assert [block.to_list() for block in blocks] == [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
    ]


def test_generate_moving_blocks_step_greater_than_one(series_pandas):
    """Should advance candidate block starts by the specified step."""
    blocks = generate_moving_blocks(series_pandas, block_length=3, step=2)

    assert [block.to_list() for block in blocks] == [
        [1, 2, 3],
        [3, 4, 5],
        [5, 6, 7],
    ]


def test_generate_moving_blocks_preserves_block_length(series_pandas):
    """Should return only blocks with the requested length."""
    blocks = generate_moving_blocks(series_pandas, block_length=4)

    assert all(len(block) == 4 for block in blocks)


def test_generate_moving_blocks_excludes_incomplete_last_block():
    """Should not include a final block when fewer than block_length values remain."""
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    blocks = generate_moving_blocks(data, block_length=3, step=3)

    assert [block.to_list() for block in blocks] == [[1, 2, 3], [4, 5, 6]]


def test_generate_moving_blocks_preserves_polars_backend_and_values(series_polars):
    """Should preserve Polars blocks and their original observation order."""
    blocks = generate_moving_blocks(series_polars, block_length=3, step=2)

    assert all(isinstance(block, pl.Series) for block in blocks)
    assert [block.to_list() for block in blocks] == [
        [1, 2, 3],
        [3, 4, 5],
        [5, 6, 7],
    ]


def test_generate_moving_blocks_preserves_input_order_without_sorting():
    """Should preserve the original order when input observations are unordered."""
    blocks = generate_moving_blocks(pd.Series([3, 1, 2, 4]), block_length=2)

    assert [block.to_list() for block in blocks] == [[3, 1], [1, 2], [2, 4]]


def test_generate_moving_blocks_is_deterministic_for_same_input(series_pandas):
    """Should generate identical blocks for repeated calls with the same input."""
    first = generate_moving_blocks(series_pandas, block_length=3, step=2)
    second = generate_moving_blocks(series_pandas, block_length=3, step=2)

    assert [block.to_list() for block in first] == [block.to_list() for block in second]


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_generate_moving_blocks_respects_original_continuity(backend):
    """Candidate blocks must not join observations separated before filtering."""
    values = [1, 2, 3, 4, 5, 6]
    data = pd.Series(values) if backend == 'pandas' else pl.Series(values)

    blocks = generate_moving_blocks(
        data, block_length=2, continuity=[0, 1, 4, 5, 6, 9],
    )

    assert [block.to_list() for block in blocks] == [[1, 2], [3, 4], [4, 5]]


@pytest.mark.parametrize(
    'continuity, error, message',
    [
        ([0, 1], ValueError, 'match'),
        ([0, 1, 1, 3, 4, 5, 6], ValueError, 'strictly increasing'),
        ([0, 2, 4, 6, 8, 10, 12], ValueError, 'no contiguous'),
        ([0, 1, 2, True, 4, 5, 6], TypeError, 'integer'),
        ('0,1,2,3,4,5,6', TypeError, 'sequence'),
    ],
)
def test_generate_moving_blocks_validates_continuity(
        series_pandas, continuity, error, message,
):
    """Malformed positions or too-short runs must fail explicitly."""
    with pytest.raises(error, match=message):
        generate_moving_blocks(
            series_pandas, block_length=2, continuity=continuity,
        )


@pytest.mark.parametrize('block_length', [0, -1, 8])
def test_generate_moving_blocks_rejects_invalid_block_length(series_pandas, block_length):
    """Should reject non-positive or oversized block lengths."""
    with pytest.raises(ValueError, match='block_length'):
        generate_moving_blocks(series_pandas, block_length=block_length)


@pytest.mark.parametrize('step', [0, -1])
def test_generate_moving_blocks_rejects_invalid_step(series_pandas, step):
    """Should reject non-positive steps."""
    with pytest.raises(ValueError, match='step'):
        generate_moving_blocks(series_pandas, block_length=3, step=step)


# ------------------------------------------------------
# moving_block_bootstrap
# ------------------------------------------------------
def test_moving_block_bootstrap_returns_requested_number_of_replicates(series_pandas):
    """Should return exactly n_bootstraps samples."""
    blocks = generate_moving_blocks(series_pandas, block_length=3)
    samples = moving_block_bootstrap(
        blocks,
        sample_size=7,
        n_bootstraps=4,
        random_state=42,
    )

    assert len(samples) == 4


def test_moving_block_bootstrap_returns_requested_sample_size(series_pandas):
    """Should truncate each bootstrap sample to exactly sample_size values."""
    blocks = generate_moving_blocks(series_pandas, block_length=3)
    samples = moving_block_bootstrap(
        blocks,
        sample_size=7,
        n_bootstraps=4,
        random_state=42,
    )

    assert all(len(sample) == 7 for sample in samples)


@pytest.mark.parametrize(
    ('sample_size', 'n_bootstraps'),
    [(0, 1), (-1, 1), (1, 0), (1, -1)],
)
def test_moving_block_bootstrap_rejects_invalid_sizes(
        series_pandas,
        sample_size,
        n_bootstraps,
):
    """Should reject non-positive sample sizes and replicate counts."""
    blocks = generate_moving_blocks(series_pandas, block_length=3)

    with pytest.raises(ValueError):
        moving_block_bootstrap(
            blocks,
            sample_size=sample_size,
            n_bootstraps=n_bootstraps,
        )


def test_moving_block_bootstrap_samples_blocks_with_replacement():
    """Should reuse the only candidate block when multiple draws are required."""
    blocks = generate_moving_blocks(pd.Series([1, 2, 3]), block_length=3)
    samples = moving_block_bootstrap(
        blocks,
        sample_size=6,
        n_bootstraps=1,
        random_state=42,
    )

    assert samples[0].to_list() == [1, 2, 3, 1, 2, 3]


def test_moving_block_bootstrap_is_reproducible_with_same_random_state(series_pandas):
    """Should generate identical samples from the same random state."""
    blocks = generate_moving_blocks(series_pandas, block_length=3)
    first = moving_block_bootstrap(blocks, sample_size=7, n_bootstraps=3, random_state=7)
    second = moving_block_bootstrap(blocks, sample_size=7, n_bootstraps=3, random_state=7)

    assert [sample.to_list() for sample in first] == [sample.to_list() for sample in second]


def test_moving_block_bootstrap_changes_with_different_random_states():
    """Should generate different samples from different random states."""
    blocks = generate_moving_blocks(pd.Series(range(30)), block_length=3)
    first = moving_block_bootstrap(blocks, sample_size=15, n_bootstraps=3, random_state=7)
    second = moving_block_bootstrap(blocks, sample_size=15, n_bootstraps=3, random_state=8)

    assert [sample.to_list() for sample in first] != [sample.to_list() for sample in second]


def test_moving_block_bootstrap_preserves_internal_block_order():
    """Should retain the candidate order within every sampled block."""
    blocks = generate_moving_blocks(pd.Series(range(12)), block_length=3, step=3)
    samples = moving_block_bootstrap(
        blocks,
        sample_size=9,
        n_bootstraps=1,
        random_state=42,
    )
    candidate_values = [block.to_list() for block in blocks]
    sample_values = samples[0].to_list()

    assert all(
        sample_values[start:start + 3] in candidate_values
        for start in range(0, 9, 3)
    )


def test_moving_block_bootstrap_preserves_dataframe_rows():
    """Should resample all DataFrame columns together without changing row order."""
    data = pd.DataFrame({
        'feature': [1, 2, 3],
        'target': [10, 20, 30],
    })
    blocks = generate_moving_blocks(data, block_length=3)
    samples = moving_block_bootstrap(
        blocks,
        sample_size=6,
        n_bootstraps=1,
        random_state=42,
    )

    assert samples[0].to_dict(orient='list') == {
        'feature': [1, 2, 3, 1, 2, 3],
        'target': [10, 20, 30, 10, 20, 30],
    }


def test_moving_block_bootstrap_preserves_polars_backend(series_polars):
    """Should return Polars samples when the candidate blocks are Polars."""
    blocks = generate_moving_blocks(series_polars, block_length=3)
    samples = moving_block_bootstrap(
        blocks,
        sample_size=7,
        n_bootstraps=2,
        random_state=42,
    )

    assert all(isinstance(sample, pl.Series) for sample in samples)


# ------------------------------------------------------
# bootstrap_metrics
# ------------------------------------------------------
def test_bootstrap_summary_returns_result_type(bootstrap_values):
    """Should return an immutable BootstrapSummary result."""
    result = bootstrap_metrics(bootstrap_values)

    assert isinstance(result, BootstrapMetricsResults)


def test_bootstrap_summary_calculates_mean(bootstrap_values):
    """Should calculate the mean of valid bootstrap estimates."""
    result = bootstrap_metrics(bootstrap_values)

    assert result.mean == pytest.approx(0.0)


def test_bootstrap_summary_calculates_sample_standard_deviation(bootstrap_values):
    """Should calculate the sample standard deviation of valid estimates."""
    result = bootstrap_metrics(bootstrap_values)

    assert result.std == pytest.approx(1.5811388300841898)


def test_bootstrap_summary_calculates_ninety_five_percent_interval(bootstrap_values):
    """Should calculate the 2.5th and 97.5th percentile interval bounds."""
    result = bootstrap_metrics(bootstrap_values, confidence_level=0.95)

    assert result.ci_lower == pytest.approx(-1.9)
    assert result.ci_upper == pytest.approx(1.9)


def test_bootstrap_summary_calculates_ninety_percent_interval(bootstrap_values):
    """Should calculate the 5th and 95th percentile interval bounds."""
    result = bootstrap_metrics(bootstrap_values, confidence_level=0.90)

    assert result.ci_lower == pytest.approx(-1.8)
    assert result.ci_upper == pytest.approx(1.8)


def test_bootstrap_summary_calculates_positive_proportion(bootstrap_values):
    """Should calculate the fraction of valid estimates strictly above zero."""
    result = bootstrap_metrics(bootstrap_values)

    assert result.pct_positive == pytest.approx(0.4)


def test_bootstrap_summary_calculates_non_positive_and_non_negative_counts(bootstrap_values):
    """Should count zero in both non-positive and non-negative totals."""
    result = bootstrap_metrics(bootstrap_values)

    assert result.n_non_positive == 3
    assert result.n_non_negative == 3


def test_bootstrap_summary_reports_valid_replicate_count(bootstrap_values):
    """Should report the number of finite bootstrap estimates used."""
    result = bootstrap_metrics(bootstrap_values)

    assert result.n_bootstraps == 5


@pytest.mark.parametrize('confidence_level', [0.0, 1.0, -0.1, 1.1])
def test_bootstrap_summary_rejects_invalid_confidence_level(confidence_level):
    """Should reject confidence levels outside the open interval from zero to one."""
    with pytest.raises(ValueError, match='confidence_level'):
        bootstrap_metrics([1.0, 2.0], confidence_level=confidence_level)


def test_bootstrap_summary_excludes_non_finite_values():
    """Should exclude NaN and infinite estimates from every summary statistic."""
    result = bootstrap_metrics([-1.0, np.nan, 0.0, np.inf, -np.inf, 1.0])

    assert result.mean == pytest.approx(0.0)
    assert result.pct_positive == pytest.approx(1 / 3)
    assert result.n_non_positive == 2
    assert result.n_non_negative == 2
    assert result.n_bootstraps == 3


def test_bootstrap_summary_rejects_values_without_finite_estimates():
    """Should reject bootstrap distributions without a valid finite estimate."""
    with pytest.raises(ValueError, match='at least one finite estimate'):
        bootstrap_metrics([np.nan, np.inf, -np.inf])


def test_bootstrap_metrics_accepts_all_supported_value_types(bootstrap_values):
    """Should return the same result for every supported values input type."""
    values_by_type = [
        bootstrap_values,
        tuple(bootstrap_values),
        np.asarray(bootstrap_values),
        pd.Series(bootstrap_values),
        pl.Series(bootstrap_values),
    ]

    results = [bootstrap_metrics(values) for values in values_by_type]

    assert results == [results[0]] * len(results)


@pytest.mark.parametrize('values', [
    [[-2.0, -1.0], [0.0, 1.0]],
    np.asarray([[-2.0, -1.0], [0.0, 1.0]]),
])
def test_bootstrap_metrics_rejects_multidimensional_values(values):
    """Should reject nested sequences and arrays with more than one dimension."""
    with pytest.raises(ValueError, match='one-dimensional'):
        bootstrap_metrics(values)


def test_bootstrap_metrics_rejects_non_numeric_values():
    """Should raise a clear error when estimates cannot be converted to floats."""
    with pytest.raises(TypeError, match='must contain numeric estimates'):
        bootstrap_metrics(['invalid', 'estimates'])


def test_bootstrap_metrics_rejects_unsupported_values_type():
    """Should reject objects that are not supported bootstrap value containers."""
    with pytest.raises(TypeError, match='one-dimensional sequence'):
        bootstrap_metrics({'estimate': [1.0, 2.0]})



# ------------------------------------------------------
# Schema and resampling boundary contracts
# ------------------------------------------------------
@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('change', ['names', 'order', 'dtype'])
def test_mbb_rejects_incompatible_schemas_before_rng(backend, change, monkeypatch):
    """Should reject every incompatible candidate before drawing any random block."""
    factory = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    first = factory({'a': [1, 2], 'b': [3, 4]})
    second = (factory({'c': [1, 2], 'b': [3, 4]}) if change == 'names' else
              factory({'b': [3, 4], 'a': [1, 2]}) if change == 'order' else
              factory({'a': [1., 2.], 'b': [3, 4]}))
    monkeypatch.setattr('alpha_research.resampling.block_bootstrap.np.random.default_rng',
                        lambda _: pytest.fail('Invalid schemas must fail before creating the RNG.'))
    with pytest.raises(ValueError, match='identical names'):
        _validate_blocks([first, second])
    with pytest.raises(ValueError, match='identical names'):
        moving_block_bootstrap([first, second], 4, 1, 42)


def test_blocks_rejects_duplicate_pandas_columns():
    """Should reject ambiguous dataframe schemas even with only one candidate."""
    with pytest.raises(ValueError, match='unique column names'):
        _validate_blocks([pd.DataFrame([[1, 2]], columns=['value', 'value'])])


@pytest.mark.parametrize('factory', [pd.Series, pl.Series])
@pytest.mark.parametrize('change', ['name', 'dtype'])
def test_blocks_requires_identical_series_names_and_dtypes(factory, change):
    """Should prevent silently changing a Series identity or dtype between selected blocks."""
    if factory is pd.Series:
        first = factory([1, 2], name='value')
        second = factory([3, 4], name='other') if change == 'name' else factory([3., 4.], name='value')
    else:
        first = factory('value', [1, 2])
        second = factory('other', [3, 4]) if change == 'name' else factory('value', [3., 4.])
    with pytest.raises(ValueError, match='identical names'):
        _validate_blocks([first, second])


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('sample_size', [1, 3, 4, 11])
def test_full_history_block_repeats_without_resampling_variation(backend, sample_size):
    """Should expose the full-history block's deterministic boundary for shorter and longer samples."""
    source = pd.Series([1, 2, 3], index=[10, 30, 90], name='value') if backend == 'pandas' else pl.Series('value', [1, 2, 3])
    samples = moving_block_bootstrap(generate_moving_blocks(source, 3), sample_size, 4, 42)
    expected = ([1, 2, 3] * 4)[:sample_size]
    assert all(sample.to_list() == expected for sample in samples)
    if backend == 'pandas':
        assert all(sample.index.to_list() == list(range(sample_size)) for sample in samples)
        assert source.index.to_list() == [10, 30, 90]


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_resamples_preserve_missing_rows_schema_and_duplicate_timestamps(backend):
    """Should repeat complete paired rows and timestamps without applying causal-key validation to resamples."""
    frame = pd.DataFrame({'time': pd.date_range('2024-01-01', periods=3, tz='UTC'),
                          'feature': [1., np.nan, 3.], 'target': [10., 20., np.inf]})
    if backend == 'polars':
        frame = pl.from_pandas(frame, nan_to_null=False)
    original = frame.copy(deep=True) if backend == 'pandas' else frame.clone()
    samples = moving_block_bootstrap(generate_moving_blocks(frame, 3), 7, 2, 42)
    for sample in samples:
        assert sample['time'].to_list() == frame['time'].to_list() * 2 + frame['time'].to_list()[:1]
        np.testing.assert_equal(sample['feature'].to_numpy(), [1., np.nan, 3., 1., np.nan, 3., 1.])
        np.testing.assert_equal(sample['target'].to_numpy(), [10., 20., np.inf, 10., 20., np.inf, 10.])
        if backend == 'pandas':
            assert sample.dtypes.equals(frame.dtypes)
        else:
            assert sample.schema == frame.schema
    assert frame.equals(original)


def test_moving_blocks_preserves_the_non_circular_endpoint_weights():
    """Should document MBB endpoint underrepresentation rather than changing the sampling method."""
    blocks = generate_moving_blocks(pd.Series(range(5)), 3)
    assert [block.to_list() for block in blocks] == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    assert [sum(value in block.to_list() for block in blocks) for value in range(5)] == [1, 2, 3, 2, 1]


@pytest.mark.parametrize('factory', [pd.Series, pl.Series])
def test_mbb_block_one_and_nondivisible_samples_preserve_seed_prefixes(factory):
    """Should preserve seeded replicate prefixes and backend parity without promising cross-version RNG stability."""
    source = factory([1, 2, 3, 4])
    for length in [1, 2, 3]:
        blocks = generate_moving_blocks(source, length)
        short = moving_block_bootstrap(blocks, 11, 3, 73)
        long = moving_block_bootstrap(blocks, 11, 9, 73)
        other = moving_block_bootstrap(generate_moving_blocks(
            pl.Series([1, 2, 3, 4]) if factory is pd.Series else pd.Series([1, 2, 3, 4]), length), 11, 3, 73)
        assert [sample.to_list() for sample in short] == [sample.to_list() for sample in long[:3]]
        assert [sample.to_list() for sample in short] == [sample.to_list() for sample in other]
        for sample in short:
            assert len(sample) == 11
            for start in range(0, 11, length):
                piece = sample.to_list()[start:start + length]
                assert piece in [block.to_list()[:len(piece)] for block in blocks]


# ------------------------------------------------------
# Strict real estimates, missingness parity, and independent numeric references
# ------------------------------------------------------
@pytest.mark.parametrize('values', [True, '1', 1j, np.datetime64('2024-01-01'), np.timedelta64(1, 'D')])
@pytest.mark.parametrize('container', ['list', 'tuple', 'numpy', 'pandas', 'polars'])
def test_bootstrap_metrics_rejects_invalid_estimate_types(values, container):
    """Should reject invalid values without coercion, imaginary-part loss, or silent row deletion."""
    entries = [values, values]
    data = (entries if container == 'list' else tuple(entries) if container == 'tuple' else
            np.array(entries) if container == 'numpy' else pd.Series(entries) if container == 'pandas' else
            pl.Series('value', entries, dtype=pl.Object))
    with pytest.raises(TypeError, match='numeric estimates'):
        _bootstrap_numeric_values(data)
    with pytest.raises(TypeError, match='numeric estimates'):
        bootstrap_metrics(data)


@pytest.mark.parametrize('container', ['list', 'tuple', 'numpy', 'pandas', 'polars'])
def test_bootstrap_numeric_values_has_missing_and_infinite_parity(container):
    """Should retain the same finite estimates across native missing values and all supported containers."""
    entries = [1., None, np.nan, np.inf, -np.inf, 3.]
    if container != 'polars':
        entries.insert(2, pd.NA)
    data = (entries if container == 'list' else tuple(entries) if container == 'tuple' else
            np.array(entries, dtype=object) if container == 'numpy' else
            pd.Series(entries, dtype='Float64') if container == 'pandas' else pl.Series(entries))
    np.testing.assert_array_equal(_bootstrap_numeric_values(data), [1., 3.])
    summary = bootstrap_metrics(data, confidence_level=.5)
    assert summary.mean == 2. and summary.std == pytest.approx(sqrt(2))
    assert (summary.ci_lower, summary.ci_upper) == (1.5, 2.5)
    assert summary.n_bootstraps == 2 and summary.pct_positive == 1.


@pytest.mark.parametrize('values', [[], [None, pd.NA, np.nan, np.inf], np.array([np.nan])])
def test_bootstrap_numeric_values_rejects_empty_effective_samples(values):
    """Should reject empty effective distributions rather than fabricate a summary."""
    with pytest.raises(ValueError, match='at least one finite'):
        _bootstrap_numeric_values(values)


def test_bootstrap_numeric_values_validates_container_dimensions_and_range():
    """Should reject unsupported containers, nested arrays, and finite integers outside Float64 range."""
    with pytest.raises(TypeError, match='one-dimensional sequence'):
        _bootstrap_numeric_values({'value': [1.]})
    with pytest.raises(ValueError, match='one-dimensional'):
        _bootstrap_numeric_values(np.ones((2, 2)))
    with pytest.raises(ValueError, match='representable'):
        _bootstrap_numeric_values([10 ** 400])


@pytest.mark.parametrize('values, expected_mean, expected_std, lower, upper', [
    ([1e308, 1e308], 1e308, 0., 1e308, 1e308),
    ([-1e308, 1e308], 0., sqrt(2) * 1e308, -.5e308, .5e308),
    ([1e200, 2e200], 1.5e200, 1e200 / sqrt(2), 1.25e200, 1.75e200),
    ([0., 0.], 0., 0., 0., 0.),
])
def test_bootstrap_summary_avoids_overflow_with_hand_calculated_statistics(values, expected_mean, expected_std, lower, upper):
    """Should preserve analytically known moments and linear percentiles for extreme finite estimates."""
    with np.errstate(over='raise', invalid='raise'):
        result = _bootstrap_distribution_summary(np.array(values), .25, .75)
        public = bootstrap_metrics(values, confidence_level=.5)
    assert result == pytest.approx((expected_mean, expected_std, lower, upper))
    assert (public.mean, public.std, public.ci_lower, public.ci_upper) == pytest.approx(result)
    assert public.ci_lower <= public.ci_upper


@pytest.mark.parametrize('values', [[1.], [0.], [None, 3., np.inf]])
def test_bootstrap_metrics_singleton_keeps_undefined_sample_deviation(values):
    """Should return a point interval and NaN deviation when exactly one finite estimate remains."""
    result = bootstrap_metrics(values)
    expected = next(value for value in values if value is not None and np.isfinite(value))
    assert result.mean == result.ci_lower == result.ci_upper == expected
    assert np.isnan(result.std) and result.n_bootstraps == 1


@pytest.mark.parametrize('values, lower, upper', [
    (np.array([]), .1, .9), (np.ones((2, 2)), .1, .9),
    (np.array([1., np.nan]), .1, .9), (np.array([1.]), -.1, .9),
    (np.array([1.]), .9, .1), (np.array([1.]), .1, 1.1),
    (np.array([1.]), None, .9), (np.array([1.]), True, .9),
])
def test_bootstrap_distribution_summary_validates_inputs(values, lower, upper):
    """Should reject invalid arrays and percentile probabilities directly."""
    with pytest.raises(ValueError, match='finite estimates and ordered'):
        _bootstrap_distribution_summary(values, lower, upper)


@pytest.mark.parametrize('values', [[1.], np.array(['1']), np.array([1j])])
def test_bootstrap_distribution_summary_rejects_non_real_arrays(values):
    """Should reject nonnumeric or unsupported array types before arithmetic."""
    with pytest.raises(TypeError, match='real numeric NumPy array'):
        _bootstrap_distribution_summary(values, .1, .9)


def test_bootstrap_metrics_rejects_unrepresentable_deviation():
    """Should raise explicitly when the actual sample deviation exceeds the float range."""
    with pytest.raises(ValueError, match='representable'):
        bootstrap_metrics([-1.7e308, 1.7e308])


@pytest.mark.parametrize('values', [pd.Series(['1', 'bad', '3']), pl.Series(['1', 'bad', '3']),
                                   np.array(['2024-01-01'], dtype='datetime64[ns]')])
def test_bootstrap_metrics_never_drops_bad_strings_or_converts_temporal_storage(values):
    """Should reject native nonnumeric containers instead of silently shortening the distribution."""
    with pytest.raises(TypeError, match='numeric estimates'):
        bootstrap_metrics(values)


def test_bootstrap_metrics_preserves_one_ulp_difference_at_large_magnitudes():
    """Should retain an analytically known deviation between adjacent representable extreme values."""
    first = 1e308
    last = float(np.nextafter(first, np.inf))
    result = bootstrap_metrics([first, last])
    assert result.std == pytest.approx((last - first) / sqrt(2), rel=1e-14)
    assert first <= result.ci_lower <= result.ci_upper <= last


def test_bootstrap_percentiles_preserve_small_neighbors_beside_an_extreme_outlier():
    """Should interpolate small order statistics without losing them to global rescaling."""
    result = _bootstrap_distribution_summary(np.array([1., 2., 1e308]), .25, .75)
    assert result[2] == 1.5
    assert result[3] == pytest.approx(5e307)


@pytest.mark.parametrize('confidence', [True, np.bool_(True), None, '.95', 1j, np.nan, np.inf])
def test_bootstrap_metrics_rejects_invalid_confidence_types_and_nonfinite_levels(confidence):
    """Should reject invalid confidence parameters consistently before normalizing estimates."""
    with pytest.raises(ValueError, match='confidence_level'):
        bootstrap_metrics([1., 2.], confidence_level=confidence)


# ------------------------------------------------------
# Bounded statistical properties of the existing non-circular MBB
# ------------------------------------------------------
@pytest.mark.parametrize('sample_size', [1, 2, 3, 11, 40])
def test_mbb_sample_size_has_exact_conditional_mean_and_variance(sample_size):
    """Should match independent moments of whole and truncated draws for arbitrary resample sizes."""
    # Candidate blocks [0,1] and [2,3] have sums 1/5 (mean 3,
    # variance 4). An odd final observation is 0/2 (mean 1, variance 1).
    # Blocks are drawn independently, so variances add before dividing by m**2.
    whole, tail = divmod(sample_size, 2)
    expected_mean = (3 * whole + tail) / sample_size
    expected_variance = (4 * whole + tail) / sample_size ** 2
    samples = moving_block_bootstrap(generate_moving_blocks(pd.Series([0., 1., 2., 3.]), 2, step=2),
                                     sample_size, 1024, 195)
    estimates = [sum(sample) / sample_size for sample in samples]
    average = sum(estimates) / len(estimates)
    variance = sum((value - average) ** 2 for value in estimates) / (len(estimates) - 1)
    assert abs(average - expected_mean) < 5 * sqrt(expected_variance / len(estimates))
    assert .8 * expected_variance < variance < 1.2 * expected_variance


def test_mbb_preserves_more_ar1_dependence_than_independent_resampling():
    """Should retain substantially more lag-one dependence than an independently sampled IID reference."""
    rng = np.random.default_rng(719)
    values = np.empty(400)
    values[0] = rng.normal() / sqrt(1 - .8 ** 2)
    for index in range(1, len(values)):
        values[index] = .8 * values[index - 1] + rng.normal()
    mbb = moving_block_bootstrap(generate_moving_blocks(pd.Series(values), 16), len(values), 160, 42)
    iid = np.random.default_rng(99).choice(values, size=(160, len(values)), replace=True)
    mbb_lag = sum(np.corrcoef(sample.to_numpy()[:-1], sample.to_numpy()[1:])[0, 1] for sample in mbb) / len(mbb)
    iid_lag = sum(np.corrcoef(sample[:-1], sample[1:])[0, 1] for sample in iid) / len(iid)
    assert abs(iid_lag) < .1
    assert .55 < mbb_lag < .9
    assert mbb_lag - iid_lag > .5


def test_mbb_mean_uncertainty_and_percentile_coverage_across_block_lengths():
    """Should characterize finite-sample MBB uncertainty against exact stationary AR(1) mean variance."""
    n = 200
    phi = .65
    repetitions = 60
    # Cov(X_t, X_{t+k}) = phi**k / (1-phi**2), innovation variance 1.
    # Sum every diagonal of the n-by-n covariance matrix, then divide by n**2.
    reference_variance = (n + 2 * sum((n - lag) * phi ** lag for lag in range(1, n))) / (n * n * (1 - phi * phi))
    covered = {length: 0 for length in (1, 8, 16)}
    variances = {length: [] for length in covered}
    for replication in range(repetitions):
        rng = np.random.default_rng(7000 + replication)
        values = np.empty(n)
        values[0] = rng.normal() / sqrt(1 - phi * phi)
        for index in range(1, n):
            values[index] = phi * values[index - 1] + rng.normal()
        for length in covered:
            samples = moving_block_bootstrap(generate_moving_blocks(pd.Series(values), length), n, 199, 8000 + replication)
            summary = bootstrap_metrics([sum(sample) / n for sample in samples])
            covered[length] += int(summary.ci_lower <= 0 <= summary.ci_upper)
            variances[length].append(summary.std ** 2)
    # Broad fixed-design bounds detect lost dependence, not a universal 95%
    # guarantee. Sixty trials also have material binomial Monte Carlo error.
    iid_ratio = sum(variances[1]) / repetitions / reference_variance
    assert .12 < iid_ratio < .4
    for length in (8, 16):
        ratio = sum(variances[length]) / repetitions / reference_variance
        assert .45 < ratio < 1.4
        assert ratio > 2 * iid_ratio
        assert covered[length] >= 45
        assert covered[length] >= covered[1] + 6
