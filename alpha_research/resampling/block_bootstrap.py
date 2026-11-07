from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

import numpy as np
import pandas as pd
import polars as pl

from alpha_research._utils import (
    _as_real_float,
    _validate_positive_integer,
    _validate_probability,
    _validate_random_state,
)
from alpha_research.progress import (
    ProgressCallback,
    _progress_iter,
    _validate_progress_callback,
)

__all__ = [
    'BootstrapMetricsResults',
    'bootstrap_metrics',
    'generate_moving_blocks',
    'moving_block_bootstrap',
]

ResamplingData: TypeAlias = pd.Series | pl.Series | pd.DataFrame | pl.DataFrame
BootstrapValues: TypeAlias = Sequence[float] | np.ndarray | pd.Series | pl.Series


def _validate_resampling_data(data: ResamplingData) -> None:
    """
    Validate data used to construct or concatenate resampling blocks.

    Parameters
    ----------
    data : pd.Series | pl.Series | pd.DataFrame | pl.DataFrame
        Series or DataFrame to validate.

    Returns
    -------
    None
        This function returns None when data uses a supported backend.

    Raises
    ------
    TypeError
        If data is not a Pandas or Polars Series or DataFrame.
    """
    if not isinstance(data, (pd.Series, pl.Series, pd.DataFrame, pl.DataFrame)):
        raise TypeError('data must be a Pandas or Polars Series or DataFrame.')


def generate_moving_blocks(
        data: ResamplingData,
        block_length: int,
        step: int = 1,
        continuity: Sequence[int] | None = None,
        *,
        on_progress: ProgressCallback | None = None,
) -> list[ResamplingData]:
    """
    Generate contiguous candidate blocks for Moving Block Bootstrap.

    Parameters
    ----------
    data : pd.Series | pl.Series | pd.DataFrame | pl.DataFrame
        Observations in their intended temporal order. The input is never
        reordered. Missing values and duplicate timestamps are preserved;
        temporal-key validation belongs to the caller.
    block_length : int
        Number of observations per contiguous block.
    step : int, default 1
        Distance between consecutive block starts. A value of 1 generates the
        canonical moving blocks with maximum overlap.
    continuity : Sequence[int] | None, default None
        Strictly increasing original observation positions. When supplied,
        candidate blocks cannot cross a gap in these positions.

    on_progress : callable | None, default None
        Optional synchronous observer called with (phase, completed, total).
        Counts describe processed local units, including skipped units;
        total may be None. Checkpoints may repeat a count. Exceptions
        propagate unchanged, allowing cooperative cancellation.
        Reports continuity filtering and block allocation separately.

    Returns
    -------
    list[pd.Series | pl.Series | pd.DataFrame | pl.DataFrame]
        Candidate blocks in the same backend as data. Only complete blocks are
        returned, preserving the original observation order within each block.

    Raises
    ------
    Exception
        Propagates observer exceptions unchanged.
    TypeError
        If on_progress is not a synchronous callable or None.
        If data is not a supported Series or DataFrame.
    ValueError
        If block_length or step is not a positive integer, or block_length is
        greater than the number of observations, or no contiguous block fits.
    TypeError
        If continuity does not contain integer positions.
    """
    _validate_progress_callback(on_progress)
    _validate_resampling_data(data)
    _validate_positive_integer(block_length, 'block_length')
    _validate_positive_integer(step, 'step')

    n_observations = len(data)

    if block_length > n_observations:
        raise ValueError(
            f'block_length ({block_length}) must not exceed the number of '
            f'observations ({n_observations}).'
        )

    if continuity is not None:
        if isinstance(continuity, (str, bytes)) or not isinstance(
                continuity, Sequence | np.ndarray | pd.Series | pl.Series
        ):
            raise TypeError('continuity must be a sequence of integer positions.')
        positions = list(continuity)
        if len(positions) != n_observations:
            raise ValueError('continuity must match the number of observations.')
        if any(
                not isinstance(value, (int, np.integer)) or isinstance(value, bool)
                for value in positions
        ):
            raise TypeError('continuity must contain only integer positions.')
        if any(
                positions[index] <= positions[index - 1]
                for index in range(1, len(positions))
        ):
            raise ValueError('continuity positions must be strictly increasing.')

    starts = range(0, n_observations - block_length + 1, step)
    if continuity is not None:
        starts = [
            start for start in _progress_iter(starts, on_progress, 'candidate_blocks')
            if positions[start + block_length - 1] - positions[start]
            == block_length - 1
        ]
        if not starts:
            raise ValueError('no contiguous candidate block fits block_length.')

    return [
        data.iloc[start:start + block_length]
        if isinstance(data, (pd.Series, pd.DataFrame))
        else data.slice(start, block_length)
        for start in _progress_iter(starts, on_progress, 'moving_blocks')
    ]


def _validate_blocks(blocks: list[ResamplingData]) -> int:
    """
    Validate candidate bootstrap blocks, concerning the type and lenght of each block,
    and return their common length.

    Parameters
    ----------
    blocks : list[pd.Series | pl.Series | pd.DataFrame | pl.DataFrame]
        Candidate blocks with one backend, equal lengths, and identical
        column order, names, and dtypes (or Series names and dtypes).

    Returns
    -------
    int
        Number of observations in each candidate block.

    Raises
    ------
    ValueError
        If blocks is empty, contains an empty block, or blocks have different
        lengths or incompatible schemas, including duplicate Pandas columns.
    TypeError
        If a block is unsupported or blocks use different data types.
    """
    if not isinstance(blocks, list) or not blocks:
        raise ValueError('blocks must be a non-empty list of candidate blocks.')

    first_block = blocks[0]
    _validate_resampling_data(first_block)
    block_type = type(first_block)
    block_length = len(first_block)

    if block_length == 0:
        raise ValueError('blocks must not contain empty blocks.')
    if isinstance(first_block, pd.DataFrame) and not first_block.columns.is_unique:
        raise ValueError('blocks must have unique column names.')

    for block in blocks[1:]:
        _validate_resampling_data(block)

        if type(block) is not block_type:
            raise TypeError('all blocks must use the same data type.')

        if len(block) != block_length:
            raise ValueError('all blocks must have the same length.')

        if isinstance(block, pd.DataFrame):
            same_schema = (block.columns.equals(first_block.columns)
                           and block.dtypes.equals(first_block.dtypes))
        elif isinstance(block, pl.DataFrame):
            same_schema = block.columns == first_block.columns and block.schema == first_block.schema
        elif isinstance(block, pd.Series):
            same_schema = (pd.Index([block.name]).equals(pd.Index([first_block.name]))
                           and block.dtype == first_block.dtype)
        else:
            same_schema = block.name == first_block.name and block.dtype == first_block.dtype
        if not same_schema:
            raise ValueError('all blocks must have identical names, column order, and dtypes.')

    return block_length


def _concatenate_blocks(blocks: list[ResamplingData]) -> ResamplingData:
    """
    Concatenate a validated block list once while preserving backend and order.

    Parameters
    ----------
    blocks : list[pd.Series | pl.Series | pd.DataFrame | pl.DataFrame]
        Non-empty, validated blocks of the same supported data type.

    Returns
    -------
    pd.Series | pl.Series | pd.DataFrame | pl.DataFrame
        One concatenated object in the same backend as the input blocks. Pandas
        indexes are reset because the library treats Pandas without an index.

    Raises
    ------
    TypeError
        If the backend cannot concatenate the supplied block schemas.

    Notes
    -----
    This helper is called after _validate_blocks(). It performs one bulk
    concatenation per bootstrap replicate, not one concatenation per block.
    """
    first_block = blocks[0]

    if isinstance(first_block, (pd.Series, pd.DataFrame)):
        return pd.concat(blocks, ignore_index=True)

    return pl.concat(blocks, rechunk=False)


def moving_block_bootstrap(
        blocks: list[ResamplingData],
        sample_size: int,
        n_bootstraps: int,
        random_state: int | None = None,
        *,
        on_progress: ProgressCallback | None = None,
) -> list[ResamplingData]:
    """
    Generate Moving Block Bootstrap samples from candidate blocks.

    For each replicate, blocks are sampled with replacement, concatenated in
    their sampled order, and truncated to exactly sample_size observations.

    Blocks do not wrap around the original history. Boundary observations can
    be underrepresented, and independently sampled block joins break temporal
    dependence. Repeated or unordered timestamps in resamples are expected.
    A block covering the entire history gives no resampling variation when
    sample_size equals that history length. Arbitrary sample sizes are supported
    mechanically; inferential validity for those sizes is the caller's concern.

    Parameters
    ----------
    blocks : list[pd.Series | pl.Series | pd.DataFrame | pl.DataFrame]
        Non-empty, equally sized candidate blocks, typically returned by
        generate_moving_blocks(). Names, column order, and dtypes must match.
    sample_size : int
        Required number of observations in each bootstrap sample.
    n_bootstraps : int
        Number of bootstrap replicates to generate.
    random_state : int | None, default None
        Seed for reproducible block sampling in the same NumPy environment.
        Increasing n_bootstraps preserves earlier samples for fixed arguments
        and seed. Bit-for-bit stability across NumPy versions is not promised.

    on_progress : callable | None, default None
        Optional synchronous observer called with (phase, completed, total).
        Counts describe processed local units, including skipped units;
        total may be None. Checkpoints may repeat a count. Exceptions
        propagate unchanged, allowing cooperative cancellation.
        Reports sample allocation; the return value remains a list.

    Returns
    -------
    list[pd.Series | pl.Series | pd.DataFrame | pl.DataFrame]
        Bootstrap samples in the same backend as the candidate blocks.

    Raises
    ------
    Exception
        Propagates observer exceptions unchanged.
    ValueError
        If blocks are invalid, or sample_size or n_bootstraps is not a
        positive integer.
    TypeError
        If on_progress is not a synchronous callable or None.
        If blocks contain unsupported or mixed data types.
    """
    _validate_progress_callback(on_progress)
    _validate_positive_integer(sample_size, 'sample_size')
    _validate_positive_integer(n_bootstraps, 'n_bootstraps')

    block_length = _validate_blocks(blocks)

    _validate_random_state(random_state)

    rng = np.random.default_rng(random_state)
    n_blocks = (int(sample_size) + block_length - 1) // block_length
    samples = []

    for _ in _progress_iter(range(n_bootstraps), on_progress, 'bootstrap_samples'):
        sampled_indices = rng.integers(0, len(blocks), size=n_blocks)
        sampled_blocks = [blocks[index] for index in sampled_indices]
        concatenated = _concatenate_blocks(sampled_blocks)

        # Cuts the blocks to keep the same size of original data
        if isinstance(concatenated, (pd.Series, pd.DataFrame)):
            samples.append(concatenated.iloc[:sample_size])
        else:
            samples.append(concatenated.slice(0, sample_size))

    return samples


@dataclass(frozen=True, slots=True)
class BootstrapMetricsResults:
    """
    Descriptive summary of a bootstrap estimate distribution.

    Attributes
    ----------
    mean : float
        Arithmetic mean of the finite bootstrap estimates.
    std : float
        Sample standard deviation of the finite bootstrap estimates (ddof=1).
        It is nan when only one finite estimate is available.
    ci_lower : float
        Lower bound of the percentile confidence interval.
    ci_upper : float
        Upper bound of the percentile confidence interval.
    pct_positive : float
        Fraction of finite estimates strictly greater than zero.
    n_non_positive : int
        Number of finite estimates less than or equal to zero.
    n_non_negative : int
        Number of finite estimates greater than or equal to zero.
    n_bootstraps : int
        Number of finite bootstrap estimates used in the summary.
    """
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    pct_positive: float
    n_non_positive: int
    n_non_negative: int
    n_bootstraps: int


def _bootstrap_numeric_values(values: BootstrapValues) -> np.ndarray:
    """Normalize real bootstrap estimates without turning invalid values into missing data.

    Parameters
    ----------
    values : Sequence | np.ndarray | pd.Series | pl.Series
        One-dimensional real estimates, optionally containing missing values.

    Returns
    -------
    np.ndarray
        Finite float estimates, preserving their original order.

    Raises
    ------
    TypeError
        If the container or an estimate is not supported real numeric data.
    ValueError
        If dimensions are invalid, no finite estimates remain, or a real
        estimate cannot be represented as a float.
    """
    if isinstance(values, (pd.Series, pl.Series)):
        values = values.to_list()
    elif not isinstance(values, np.ndarray) and (
            not isinstance(values, Sequence) or isinstance(values, (str, bytes))
    ):
        raise TypeError('values must be a one-dimensional sequence, NumPy array, Pandas Series, or Polars Series.')
    if isinstance(values, np.ndarray) and values.dtype.kind in 'Mm':
        raise TypeError('values must contain numeric estimates with real scalar types.')
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError('values must be one-dimensional.')
    numeric = []
    for value in array:
        if value is None or value is pd.NA:
            continue
        converted = _as_real_float(
            value,
            type_error='values must contain numeric estimates with real scalar types.',
            overflow_error='estimates must be representable as floats.',
        )
        if np.isfinite(converted):
            numeric.append(converted)
    if not numeric:
        raise ValueError('values must contain at least one finite estimate.')
    return np.asarray(numeric, dtype=float)


def _bootstrap_distribution_summary(values: np.ndarray, q_lower: float, q_upper: float) -> tuple:
    """Calculate mean, sample deviation, and linear percentiles with bounded intermediate values.

    Parameters
    ----------
    values : np.ndarray
        Nonempty one-dimensional finite real estimates.
    q_lower, q_upper : float
        Ordered percentile probabilities in [0, 1].

    Returns
    -------
    tuple[float, float, float, float]
        Mean, sample standard deviation, and percentile endpoints. A singleton
        has undefined sample deviation (NaN).

    Raises
    ------
    TypeError
        If values is not a real numeric NumPy array.
    ValueError
        If values or probabilities are invalid, or a summary is not representable.
    """
    if not isinstance(values, np.ndarray) or values.dtype.kind not in 'iuf':
        raise TypeError('summary values must be a real numeric NumPy array.')
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or any(not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
                   for value in (q_lower, q_upper))
            or not 0 <= q_lower <= q_upper <= 1):
        raise ValueError('summary requires finite estimates and ordered probabilities in [0, 1].')
    # A half-sum avoids overflowing the range for opposite-sign extremes.
    center = float(np.min(values)) / 2 + float(np.max(values)) / 2
    centered = values - center
    scale = float(np.max(np.abs(centered)))
    normalized = centered / scale if scale else centered
    mean = center + float(np.mean(normalized)) * scale
    std = float(np.std(normalized, ddof=1)) * scale if len(values) > 1 else np.nan
    # Interpolate the actual order statistics: globally scaling the data can
    # erase small neighboring estimates when a distant outlier is enormous.
    ordered = np.sort(values)
    quantiles = []
    for probability in (q_lower, q_upper):
        position = (len(ordered) - 1) * probability
        index = int(position)
        weight = position - index
        left = float(ordered[index])
        right = float(ordered[min(index + 1, len(ordered) - 1)])
        quantiles.append((1 - weight) * left + weight * right if left < 0 < right
                         else left + weight * (right - left))
    lower, upper = quantiles
    if (not all(np.isfinite(value) for value in (mean, lower, upper))
            or (len(values) > 1 and not np.isfinite(std)) or lower > upper):
        raise ValueError('bootstrap metrics must be representable as finite floats.')
    return mean, std, lower, upper


def bootstrap_metrics(
        values: BootstrapValues,
        confidence_level: float = 0.95,
) -> BootstrapMetricsResults:
    """
    Summarize a precomputed distribution of bootstrap estimate temporal associations.

    Missing and non-finite estimates are excluded from every reported statistic.
    Strings, booleans, complex values, and temporal values are rejected. This
    function describes the supplied distribution only; it does not perform a
    hypothesis test or infer statistical significance.

    Parameters
    ----------
    values : Sequence[float] | np.ndarray | pd.Series | pl.Series
        Precomputed bootstrap association values estimates. Values must form a one-dimensional
        numeric sequence.
    confidence_level : float, default 0.95
        Confidence level used for the percentile interval. It must be strictly
        between zero and one.

    Returns
    -------
    BootstrapMetricsResults
        Mean, sample standard deviation, percentile confidence interval, sign
        counts, and number of finite estimates.

    Raises
    ------
    TypeError
        If values is not a supported one-dimensional sequence.
    ValueError
        If confidence_level is outside (0, 1), values is not one-dimensional,
        or values contains no finite estimates, or a summary is not representable.
    """
    # checks on the inputs
    _validate_probability(confidence_level, 'confidence_level')

    finite_values = _bootstrap_numeric_values(values)

    # computations
    alpha = 1 - confidence_level
    q_lower = alpha / 2
    q_upper = 1 - q_lower
    n_bootstraps = len(finite_values)
    mean, std, lower, upper = _bootstrap_distribution_summary(finite_values, q_lower, q_upper)

    return BootstrapMetricsResults(
        mean=mean,
        std=std,
        ci_lower=lower,
        ci_upper=upper,
        pct_positive=float(np.mean(finite_values > 0)),
        n_non_positive=int(np.sum(finite_values <= 0)),
        n_non_negative=int(np.sum(finite_values >= 0)),
        n_bootstraps=n_bootstraps,
    )
