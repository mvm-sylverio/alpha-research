"""Basic synthetic and shifted series for caller-defined null experiments."""

import numpy as np
import pandas as pd
import polars as pl

from alpha_research._utils import (
    _validate_finite_number,
    _validate_positive_integer,
    _validate_random_state,
)

__all__ = ['circular_shift', 'generate_random_walk']


def generate_random_walk(
        sample_size: int,
        random_state: int | None = None,
        *,
        initial_value: float = 0.0,
        step_std: float = 1.0,
) -> np.ndarray:
    """Generate one zero-drift random walk with independent Gaussian steps.

    Parameters
    ----------
    sample_size : int
        Positive number of observations, including the initial value.
    random_state : int | None, default None
        Non-negative seed for a local NumPy generator. The global RNG is not
        modified. Fixed arguments and seed reproduce the path in the same
        NumPy environment; a larger sample_size preserves its prefix.
    initial_value : float, default 0.0
        Finite starting value at position zero.
    step_std : float, default 1.0
        Finite, strictly positive standard deviation of the independent,
        zero-mean Gaussian increments.

    Returns
    -------
    np.ndarray
        One-dimensional Float64 path of length sample_size, with X[0] equal
        to initial_value and X[t] = X[t - 1] + epsilon[t] for t >= 1.
        It can be wrapped in a Pandas or Polars Series without changing values.

    Raises
    ------
    TypeError
        If initial_value or step_std is not a real numeric scalar, or
        random_state is neither an integer nor None.
    ValueError
        If sample_size is not a positive integer, the seed is negative,
        numeric parameters are non-finite or unrepresentable, step_std is
        not positive, or the generated path contains non-finite values.

    Notes
    -----
    A singleton contains only initial_value and draws no increments. No
    target is used, but independence by construction does not imply zero
    sample correlation. This function neither tests a hypothesis nor
    orchestrates replications. Bit-for-bit stability across NumPy versions
    is not promised.
    """
    _validate_positive_integer(sample_size, 'sample_size')
    _validate_random_state(random_state)
    initial_value = _validate_finite_number(initial_value, 'initial_value')
    step_std = _validate_finite_number(step_std, 'step_std')
    if step_std <= 0:
        raise ValueError('step_std must be strictly positive.')

    sample_size = int(sample_size)
    values = np.empty(sample_size, dtype=float)
    values[0] = initial_value
    with np.errstate(over='ignore', invalid='ignore'):
        if sample_size > 1:
            rng = np.random.default_rng(random_state)
            values[1:] = rng.normal(0.0, step_std, size=sample_size - 1)
        # Include the initial value in the recurrence to avoid an overflowing
        # intermediate sum when an offsetting initial value keeps the path finite.
        np.cumsum(values, out=values)
    if not np.isfinite(values).all():
        raise ValueError('the generated random walk must contain only finite values.')
    return values


def circular_shift(
        data: pd.Series | pl.Series,
        shift: int,
        min_shift: int = 1,
) -> pd.Series | pl.Series:
    """Rotate series values to the right without moving destination labels.

    Parameters
    ----------
    data : pd.Series | pl.Series
        At least two observations in caller-defined order. Values, including
        missing values, are permuted without filtering or sorting. Pass only
        the feature; timestamps, symbols, and the target remain with the caller.
    shift : int
        Positive rightward offset strictly smaller than len(data). It is not
        reduced modulo the length, preventing silent full-cycle realignment.
    min_shift : int, default 1
        Positive minimum circular distance from the original alignment.
        Both shift and len(data) - shift must be at least this value. The
        default excludes the identity only; the caller must choose a suitable
        exclusion distance for its horizon and temporal dependence.

    Returns
    -------
    pd.Series | pl.Series
        New series in the input backend, preserving length, name, and dtype.
        Pandas index labels stay in their original destination positions,
        so assignment to the original frame does not undo the rotation.
        The input is not modified.

    Raises
    ------
    TypeError
        If data is not a Pandas or Polars Series.
    ValueError
        If data has fewer than two observations, shift or min_shift is not a
        positive integer, shift reaches or exceeds the series length, or
        min_shift excludes the requested offset or every possible offset.

    Notes
    -----
    This is a positional transformation, not a temporal-key validator or a
    hypothesis test. The caller handles assets, gaps, missing-value masks,
    and the artificial join between the original end and beginning. A valid
    offset does not guarantee independence, and constant or periodic values
    can reproduce the original series even at an admissible offset.
    """
    if not isinstance(data, (pd.Series, pl.Series)):
        raise TypeError('data must be a Pandas or Polars Series.')
    _validate_positive_integer(shift, 'shift')
    _validate_positive_integer(min_shift, 'min_shift')
    n_observations = len(data)
    if n_observations < 2:
        raise ValueError('data must contain at least two observations.')

    shift, min_shift = int(shift), int(min_shift)
    if shift >= n_observations:
        raise ValueError('shift must be smaller than the number of observations.')
    if min_shift > n_observations // 2:
        raise ValueError('min_shift excludes every possible circular shift.')
    if min(shift, n_observations - shift) < min_shift:
        raise ValueError('shift must satisfy min_shift in both circular directions.')

    if isinstance(data, pd.Series):
        result = pd.concat([data.iloc[-shift:], data.iloc[:-shift]], ignore_index=True)
        result.index = data.index.copy()
        return result

    return pl.concat([data.slice(n_observations - shift), data.slice(0, n_observations - shift)])
