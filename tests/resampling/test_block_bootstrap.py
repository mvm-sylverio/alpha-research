import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.resampling.block_bootstrap import (
    _concatenate_blocks,
    _validate_blocks,
    _validate_resampling_data,
    BootstrapMetricsResults,
    bootstrap_metrics,
    generate_moving_blocks,
    moving_block_bootstrap,
)


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
