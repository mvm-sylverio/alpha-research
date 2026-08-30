from dataclasses import dataclass
from typing import Sequence, TypeAlias

import numpy as np
import pandas as pd
import polars as pl

from alpha_research._utils import _validate_positive_integer


__all__ = [
    'generate_moving_blocks',
    'moving_block_bootstrap',
    'BootstrapMetricsResults',
    'bootstrap_metrics',
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
) -> list[ResamplingData]:
    """
    Generate contiguous candidate blocks for Moving Block Bootstrap.

    Parameters
    ----------
    data : pd.Series | pl.Series | pd.DataFrame | pl.DataFrame
        Observations in their intended temporal order. The input is never
        reordered.
    block_length : int
        Number of observations per contiguous block.
    step : int, default 1
        Distance between consecutive block starts. A value of 1 generates the
        canonical moving blocks with maximum overlap.

    Returns
    -------
    list[pd.Series | pl.Series | pd.DataFrame | pl.DataFrame]
        Candidate blocks in the same backend as data. Only complete blocks are
        returned, preserving the original observation order within each block.

    Raises
    ------
    TypeError
        If data is not a supported Series or DataFrame.
    ValueError
        If block_length or step is not a positive integer, or block_length is
        greater than the number of observations.
    """
    _validate_resampling_data(data)
    _validate_positive_integer(block_length, 'block_length')
    _validate_positive_integer(step, 'step')

    n_observations = len(data)

    if block_length > n_observations:
        raise ValueError(
            f'block_length ({block_length}) must not exceed the number of '
            f'observations ({n_observations}).'
        )

    return [
        data.iloc[start:start + block_length]
        if isinstance(data, (pd.Series, pd.DataFrame))
        else data.slice(start, block_length)
        for start in range(0, n_observations - block_length + 1, step)
    ]


def _validate_blocks(blocks: list[ResamplingData]) -> int:
    """
    Validate candidate bootstrap blocks, concerning the type and lenght of each block,
    and return their common length.

    Parameters
    ----------
    blocks : list[pd.Series | pl.Series | pd.DataFrame | pl.DataFrame]
        Candidate blocks expected to use one backend and have equal lengths.

    Returns
    -------
    int
        Number of observations in each candidate block.

    Raises
    ------
    ValueError
        If blocks is empty, contains an empty block, or blocks have different
        lengths.
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

    for block in blocks[1:]:
        _validate_resampling_data(block)

        if type(block) is not block_type:
            raise TypeError('all blocks must use the same data type.')

        if len(block) != block_length:
            raise ValueError('all blocks must have the same length.')

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
) -> list[ResamplingData]:
    """
    Generate Moving Block Bootstrap samples from candidate blocks.

    For each replicate, blocks are sampled with replacement, concatenated in
    their sampled order, and truncated to exactly sample_size observations.

    Parameters
    ----------
    blocks : list[pd.Series | pl.Series | pd.DataFrame | pl.DataFrame]
        Non-empty, equally sized candidate blocks, typically returned by
        generate_moving_blocks().
    sample_size : int
        Required number of observations in each bootstrap sample.
    n_bootstraps : int
        Number of bootstrap replicates to generate.
    random_state : int | None, default None
        Seed for reproducible block sampling.

    Returns
    -------
    list[pd.Series | pl.Series | pd.DataFrame | pl.DataFrame]
        Bootstrap samples in the same backend as the candidate blocks.

    Raises
    ------
    ValueError
        If blocks are invalid, or sample_size or n_bootstraps is not a
        positive integer.
    TypeError
        If blocks contain unsupported or mixed data types.
    """
    _validate_positive_integer(sample_size, 'sample_size')
    _validate_positive_integer(n_bootstraps, 'n_bootstraps')

    block_length = _validate_blocks(blocks)

    if random_state is not None and (
            not isinstance(random_state, (int, np.integer))
            or isinstance(random_state, bool)
    ):
        raise TypeError('random_state must be an integer or None.')

    rng = np.random.default_rng(random_state)
    n_blocks = int(np.ceil(sample_size / block_length))
    samples = []

    for _ in range(n_bootstraps):
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


def bootstrap_metrics(
        values: BootstrapValues,
        confidence_level: float = 0.95,
) -> BootstrapMetricsResults:
    """
    Summarize a precomputed distribution of bootstrap estimate temporal associations.

    Non-finite estimates are excluded from every reported statistic. This
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
        or values contains no finite estimates.
    """
    # checks on the inputs
    if (
            not isinstance(confidence_level, (int, float, np.integer, np.floating))
            or isinstance(confidence_level, bool)
            or not 0 < confidence_level < 1
    ):
        raise ValueError('confidence_level must be strictly between 0 and 1.')

    if isinstance(values, pd.Series):
        values_array = values.to_numpy(dtype=float, na_value=np.nan)
    elif isinstance(values, pl.Series):
        values_array = (
            values
            .cast(pl.Float64, strict=False)
            .fill_null(np.nan)
            .to_numpy()
        )
    elif isinstance(values, np.ndarray):
        values_array = values
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        values_array = np.asarray(values)
    else:
        raise TypeError(
            'values must be a one-dimensional sequence, NumPy array, Pandas '
            'Series, or Polars Series.'
        )

    if values_array.ndim != 1:
        raise ValueError('values must be one-dimensional.')

    try:
        numeric_values = values_array.astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise TypeError('values must contain numeric estimates.') from error

    finite_values = numeric_values[np.isfinite(numeric_values)]

    if len(finite_values) == 0:
        raise ValueError('values must contain at least one finite estimate.')

    # computations
    alpha = 1 - confidence_level
    q_lower = alpha / 2
    q_upper = 1 - q_lower
    n_bootstraps = len(finite_values)

    return BootstrapMetricsResults(
        mean=float(np.mean(finite_values)),
        std=(
            float(np.std(finite_values, ddof=1))
            if n_bootstraps > 1
            else np.nan
        ),
        ci_lower=float(np.quantile(finite_values, q_lower)),
        ci_upper=float(np.quantile(finite_values, q_upper)),
        pct_positive=float(np.mean(finite_values > 0)),
        n_non_positive=int(np.sum(finite_values <= 0)),
        n_non_negative=int(np.sum(finite_values >= 0)),
        n_bootstraps=n_bootstraps,
    )
