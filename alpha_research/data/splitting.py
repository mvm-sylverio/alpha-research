import warnings

import numpy as np
import pandas as pd
import polars as pl
from dateutil.relativedelta import relativedelta

from alpha_research._utils import (
    _validate_df,
    _validate_finite_number,
    _validate_positive_integer,
)


def _split_dates(df, column):
    """Validate a split date column without coercing numeric or textual dates.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Observations containing the date column.
    column : str
        Date column to validate.

    Returns
    -------
    pd.DatetimeIndex
        Dates preserving timezone and timestamp precision.

    Raises
    ------
    TypeError
        If df is not a supported DataFrame.
    ValueError
        If dates are missing or not datetime typed.
    KeyError
        If the column is absent.
    """
    _validate_df(df, [column])
    series = df[column]
    if isinstance(df, pd.DataFrame):
        if not pd.api.types.is_datetime64_any_dtype(series.dtype):
            raise ValueError(f'{column} must have a datetime dtype.')
        dates = pd.DatetimeIndex(series)
    else:
        if series.dtype != pl.Date and not isinstance(series.dtype, pl.Datetime):
            raise ValueError(f'{column} must have a datetime dtype.')
        dates = pd.DatetimeIndex(series.to_pandas())
    if dates.hasnans:
        raise ValueError(f'{column} must not contain missing dates.')
    return dates


def research_test_split(
        df:pd.DataFrame | pl.DataFrame,
        purging_coefficient: float = 1.2,
        min_research_period=24,
        min_test_period=6,
        min_research_unique_pts=500,
        min_test_unique_pts=120,
        time_col: str = 'time',
        *,
        label_end_col: str | None = None,
) -> tuple[pd.DataFrame | pl.DataFrame, pd.DataFrame | pl.DataFrame]:
    """
    Split a dataset using chronological cutoffs, preserving input row order.

    The test block is always the most recent period.

    Both minimum size criteria (months and unique dates) must be satisfied
    and the more conservative cutoff is always used.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Input dataset, already sampled. Must contain time_col.
    purging_coefficient : float
        Multiplier applied to minimum test size to define the purge zone.
        A coefficient of 1.2 means research ends 1.2x the minimum test
        size before the last date, creating a gap of 0.2x between
        research end and test start. Default 1.2.
    min_research_period : int
        Recommended research block size in months; zero disables it. Default 24.
    min_test_period : int
        Minimum test block size in months; zero disables it. Default 6.
    min_research_unique_pts : int
        Minimum unique dates in research block. Default 500.
    min_test_unique_pts : int
        Minimum unique dates in test block. Default 120.
    time_col : str
        Name of the time column. Default 'time'.
    label_end_col : str | None, default None
        Optional datetime column giving the final information time of each
        label. Research labels ending at or after the first retained test
        date are excluded. Missing endpoints are rejected.

    Returns
    -------
    tuple[pd.DataFrame | pl.DataFrame, pd.DataFrame | pl.DataFrame]
        (df_research, df_test) — same type as input.

    Raises
    ------
    TypeError
        If df has an unsupported type or purging_coefficient is nonnumeric.
    ValueError
        If purging_coefficient < 1 or if total data is insufficient for the
        test block at minimum sizes or if resulting research block is
        smaller than half the minimum.

    Warns
    -----
    UserWarning
        If research block satisfies the absolute minimum (half) but not
        the recommended minimum defined by the user.

    Notes
    -----
    Test block requirements are prioritized because no valid methodological
    workaround exists for insufficient test data. Research block
    deficiencies may be addressed via bootstrapping or k-folds.

    The purge zone is the gap between research_end_date and
    test_start_date, controlled by purging_coefficient. This heuristic alone
    does not guarantee separation of forward-label information intervals;
    provide label_end_col for that check. Cutoffs use the global date grid,
    so incomplete assets may have different first/last retained observations.
    Repeated dates across assets and input row order are preserved. Monthly
    criteria can be disabled with zero; unique-date minima must be positive.
    Purge spans retain the established rounding down to whole months/points,
    so a coefficient slightly above one may not remove additional dates.
    """
    purging_coefficient = _validate_finite_number(purging_coefficient, 'purging_coefficient')
    for name, value in (('min_research_period', min_research_period), ('min_test_period', min_test_period)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f'{name} must be a non-negative integer.')
    _validate_positive_integer(min_research_unique_pts, 'min_research_unique_pts')
    _validate_positive_integer(min_test_unique_pts, 'min_test_unique_pts')
    if purging_coefficient < 1:
        raise ValueError(
            "purging_coefficient must be >= 1.0. "
            "A value < 1 would place research_end after test_start, causing severe leakage."
        )

    _validate_df(df, [time_col])

    # Keep timestamp precision and timezone when building the global date grid.
    dates = _split_dates(df, time_col)
    unique_dates = dates.unique().sort_values()
    label_ends = None
    if label_end_col is not None:
        label_ends = _split_dates(df, label_end_col)
        if label_ends.tz != dates.tz:
            raise ValueError('label end and observation dates must use the same timezone.')
        if (label_ends < dates).any():
            raise ValueError('label end dates must not precede observation dates.')

    n_unique = len(unique_dates)
    # Array already ordered
    first_date = unique_dates[0]
    last_date = unique_dates[-1]

    # total dataset size in months
    delta = relativedelta(last_date, first_date)
    full_months = 12 * delta.years + delta.months

    # 1. raise if total data cannot satisfy test block at minimum sizes
    if n_unique < min_test_unique_pts or full_months < min_test_period:
        raise ValueError(
            f"Insufficient data: {n_unique} unique dates ({full_months} months). "
            f"Minimum required: {min_test_unique_pts} dates ({min_test_period} months)."
        )

    # 2. test start — most conservative (earliest) of months and points criteria
    test_start_by_months = last_date - relativedelta(months=min_test_period)
    test_start_by_pts = unique_dates[-min_test_unique_pts]
    test_start_date = min(test_start_by_months, test_start_by_pts)

    # 3. research end — purging_coefficient extends the exclusion zone beyond test start
    #    creating a gap of (purging_coefficient - 1) * min_test size between blocks
    point_span = min_test_unique_pts * purging_coefficient
    month_span = min_test_period * purging_coefficient
    if not np.isfinite(point_span) or not np.isfinite(month_span):
        raise ValueError('Research block too small after purging: purge size is out of range.')
    purged_pts = int(point_span)
    purged_months = int(month_span)

    if purged_pts >= n_unique:
        raise ValueError('Research block too small after purging: purge consumes the available dates.')

    try:
        research_end_by_months = last_date - relativedelta(months=purged_months)
    except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime) as error:
        raise ValueError('Research block too small after purging: month cutoff is out of range.') from error
    research_end_by_pts = unique_dates[-purged_pts]
    research_end_date = min(research_end_by_months, research_end_by_pts)

    # 4. validate research block size
    research_mask = (dates <= research_end_date) & (dates < test_start_date)
    if label_ends is not None:
        first_test_date = unique_dates[unique_dates >= test_start_date][0]
        research_mask &= label_ends < first_test_date
    research_dates = dates[research_mask].unique().sort_values()
    research_n = len(research_dates)
    if not research_n:
        raise ValueError('Research block too small after purging: no observations remain.')
    research_delta = relativedelta(research_dates[-1], research_dates[0])
    research_months = 12 * research_delta.years + research_delta.months

    if (2 * research_n < min_research_unique_pts or
            2 * research_months < min_research_period):
        raise ValueError(
            f"Research block too small after purging: {research_n} dates "
            f"({research_months} months). "
            f"Minimum allowed: {(int(min_research_unique_pts) + 1) // 2} dates "
            f"({(int(min_research_period) + 1) // 2} months). "
            "Consider reducing purging_coefficient or min_test sizes."
        )

    if (research_n < min_research_unique_pts or
            research_months < min_research_period):
        warnings.warn(
            f"Research block ({research_n} dates, {research_months} months) is smaller "
            f"than recommended ({min_research_unique_pts} dates, {min_research_period} months). "
            "Consider increasing dataset size or adjusting split parameters.",
            UserWarning,
            stacklevel=2
        )

    # 5. split
    if isinstance(df, pd.DataFrame):
        df_research = df.loc[research_mask].copy()
        df_test = df.loc[dates >= test_start_date].copy()
    else:
        df_research = df.filter(pl.Series(research_mask))
        df_test = df.filter(pl.Series(dates >= test_start_date))

    return df_research, df_test
