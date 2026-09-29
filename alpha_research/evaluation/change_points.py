from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl

from alpha_research._utils import (
    _validate_df,
    _validate_positive_integer,
    _validate_time_order,
)

__all__ = ['ChangePoint', 'ChangePointSegment', 'PeltResult', 'detect_change_points']


@dataclass(frozen=True, slots=True)
class ChangePoint:
    """One boundary at the first observation of a new segment."""

    position: int
    time: Any


@dataclass(frozen=True, slots=True)
class ChangePointSegment:
    """One half-open segment with descriptive statistics of its input values."""

    start: int
    stop: int
    first_time: Any
    last_time: Any
    n_obs: int
    mean: float
    variance: float


@dataclass(frozen=True, slots=True)
class PeltResult:
    """Offline PELT segmentation and its reproducibility parameters."""

    breakpoints: tuple[ChangePoint, ...]
    segments: tuple[ChangePointSegment, ...]
    value_col: str
    time_col: str
    model: str
    penalty: float
    min_size: int
    jump: int
    n_obs: int


def detect_change_points(
        df: pd.DataFrame | pl.DataFrame,
        value_col: str,
        model: Literal['l1', 'l2', 'normal', 'rbf'],
        penalty: float,
        time_col: str = 'time',
        min_size: int = 2,
        jump: int = 1,
) -> PeltResult:
    """
    Segment one ordered numeric series with offline PELT from ``ruptures``.

    PELT minimizes within-segment cost plus a penalty per change point. The
    selected cost defines what kind of change the segmentation can detect:
    ``l2`` targets mean shifts, while ``normal`` models changes in Gaussian
    mean and variance. Breakpoints are retrospective estimates, not p-values
    or evidence that a change was observable at its estimated time.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        One series in strictly increasing, unique time order. No symbol,
        feature, target, or market-data schema is required.
    value_col : str
        Numeric column to segment. All observations must be finite.
    model : {'l1', 'l2', 'normal', 'rbf'}
        Segment cost supported by ``ruptures``. Select it for the research
        question; the same series can yield different boundaries by cost.
    penalty : float
        Finite, positive cost added for each change point. Its scale depends
        on the cost model and input series.
    time_col : str, default 'time'
        Ordered observation key, independent of the DataFrame index.
    min_size : int, default 2
        Minimum requested segment length in observations. A cost may require
        a larger minimum internally.
    jump : int, default 1
        Candidate-boundary spacing. One considers every observation; larger
        values restrict the search grid.

    Returns
    -------
    PeltResult
        Boundaries, segments, and configuration. A breakpoint position is the
        zero-based index of the first observation in the following segment;
        segments use half-open ``[start, stop)`` positions. The terminal
        position returned by ``ruptures`` is excluded from breakpoints.
        Zero detected breaks produce an empty breakpoint tuple and one segment.
        Segment variance is descriptive sample variance (ddof=1), or NaN for
        a one-observation segment.

    Raises
    ------
    TypeError
        If df is not Pandas or Polars, or the value column is not numeric.
    KeyError
        If a requested column is absent.
    ValueError
        If columns or parameters are invalid, the series is unordered,
        values are non-finite, or there are too few observations.
    ImportError
        If the optional ``ruptures`` dependency is unavailable. Install the
        ``changepoint`` extra of Alpha Research.

    Notes
    -----
    This function delegates segmentation to ``ruptures.Pelt``; it adds the
    Alpha Research input contract and maps segment boundaries back to the
    original observation times. It does not calibrate a statistical test.

    References
    ----------
    Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of
    changepoints with a linear computational cost. Journal of the American
    Statistical Association, 107(500), 1590-1598.
    https://doi.org/10.1080/01621459.2012.737745

    Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline
    change point detection methods. Signal Processing, 167, 107299.
    https://doi.org/10.1016/j.sigpro.2019.107299
    """
    if not isinstance(value_col, str) or not value_col:
        raise ValueError('value_col must be a non-empty string.')
    if not isinstance(time_col, str) or not time_col:
        raise ValueError('time_col must be a non-empty string.')
    if value_col == time_col:
        raise ValueError('value_col and time_col must be different columns.')
    if not isinstance(model, str) or model not in {'l1', 'l2', 'normal', 'rbf'}:
        raise ValueError("model must be 'l1', 'l2', 'normal', or 'rbf'.")
    if not isinstance(penalty, Real) or isinstance(penalty, bool):
        raise TypeError('penalty must be a real number.')
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError('penalty must be a finite positive number.')
    _validate_positive_integer(min_size, 'min_size')
    _validate_positive_integer(jump, 'jump')
    _validate_df(df, [time_col, value_col])
    _validate_time_order(df, time_col)
    time_series = df[time_col]
    if isinstance(df, pd.DataFrame):
        numeric_time = pd.api.types.is_numeric_dtype(time_series.dtype)
    else:
        numeric_time = time_series.dtype.is_numeric()
    if numeric_time and not np.isfinite(
            np.asarray(time_series.to_numpy(), dtype=float)
    ).all():
        raise ValueError(f'{time_col} must contain only finite values.')
    if len(df) < min_size:
        raise ValueError('df must have at least min_size observations.')

    series = df[value_col]
    if isinstance(df, pd.DataFrame):
        is_numeric = pd.api.types.is_numeric_dtype(series.dtype)
        is_bool = pd.api.types.is_bool_dtype(series.dtype)
    else:
        is_numeric = series.dtype.is_numeric()
        is_bool = series.dtype == pl.Boolean
    if not is_numeric or is_bool:
        raise TypeError(f'{value_col} must contain numeric values.')

    if isinstance(df, pd.DataFrame):
        values = series.to_numpy(dtype=float, na_value=np.nan)
    else:
        values = np.asarray(series.to_numpy(), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f'{value_col} must contain only finite values.')

    try:
        import ruptures as rpt
    except ImportError as error:
        raise ImportError(
            'PELT requires ruptures. Install alpha-research[changepoint].'
        ) from error

    boundaries = rpt.Pelt(model=model, min_size=int(min_size), jump=int(jump)).fit_predict(
        values, pen=float(penalty),
    )
    positions = [int(position) for position in boundaries if position != len(values)]
    times = df[time_col].to_list() if isinstance(df, pl.DataFrame) else df[time_col].tolist()
    breakpoints = tuple(ChangePoint(position, times[position]) for position in positions)

    segments = []
    starts = [0, *positions]
    stops = [*positions, len(values)]
    for start, stop in zip(starts, stops):
        segment_values = values[start:stop]
        segments.append(ChangePointSegment(
            start=start,
            stop=stop,
            first_time=times[start],
            last_time=times[stop - 1],
            n_obs=stop - start,
            mean=float(np.mean(segment_values)),
            variance=(
                float(np.var(segment_values, ddof=1))
                if stop - start > 1 else float('nan')
            ),
        ))

    return PeltResult(
        breakpoints=breakpoints,
        segments=tuple(segments),
        value_col=value_col,
        time_col=time_col,
        model=model,
        penalty=float(penalty),
        min_size=int(min_size),
        jump=int(jump),
        n_obs=len(values),
    )
