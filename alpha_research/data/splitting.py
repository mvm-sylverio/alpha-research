import pandas as pd
import polars as pl
from dateutil.relativedelta import relativedelta
import warnings
import numpy as np


from alpha_research._utils import _validate_df


def research_test_split(
        df:pd.DataFrame | pl.DataFrame,
        purging_coefficient: float = 1.2,
        min_research_period=24,
        min_test_period=6,
        min_research_unique_pts=500,
        min_test_unique_pts=120,
        time_col: str = 'time',
) -> tuple[pd.DataFrame | pl.DataFrame, pd.DataFrame | pl.DataFrame]:
    """
    Split a dataset into research and test blocks, preserving temporal order.

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
        Minimum research block size in months. Default 24.
    min_test_period : int
        Minimum test block size in months. Default 6.
    min_research_unique_pts : int
        Minimum unique dates in research block. Default 500.
    min_test_unique_pts : int
        Minimum unique dates in test block. Default 120.
    time_col : str
        Name of the time column. Default 'time'.

    Returns
    -------
    tuple[pd.DataFrame | pl.DataFrame, pd.DataFrame | pl.DataFrame]
        (df_research, df_test) — same type as input.

    Raises
    ------
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
    test_start_date, controlled by purging_coefficient. This prevents
    leakage from overlapping forward return windows.
    """
    if purging_coefficient < 1:
        raise ValueError(
            f"purging_coefficient must be >= 1.0. "
            f"A value < 1 would place research_end after test_start, causing severe leakage."
        )

    _validate_df(df, [time_col])

    # sorted unique dates — to_pydatetime converts the array to a python native format
    # for usage in relativedelta
    if isinstance(df, pd.DataFrame):
        unique_dates = pd.DatetimeIndex(
            np.sort(df[time_col].unique())
        ).to_pydatetime()
    else:
        unique_dates = pd.DatetimeIndex(
            df[time_col].unique().sort().to_numpy()
        ).to_pydatetime()

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
    purged_pts = int(min_test_unique_pts * purging_coefficient)
    purged_months = int(min_test_period * purging_coefficient)

    research_end_by_months = last_date - relativedelta(months=purged_months)
    research_end_by_pts = unique_dates[-purged_pts]
    research_end_date = min(research_end_by_months, research_end_by_pts)

    # 4. validate research block size
    research_dates = unique_dates[unique_dates <= research_end_date]
    research_n = len(research_dates)
    research_delta = relativedelta(research_end_date, first_date)
    research_months = 12 * research_delta.years + research_delta.months

    if (research_n < min_research_unique_pts // 2 or
            research_months < min_research_period // 2):
        raise ValueError(
            f"Research block too small after purging: {research_n} dates "
            f"({research_months} months). "
            f"Minimum allowed: {min_research_unique_pts // 2} dates "
            f"({min_research_period // 2} months). "
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
        df_research = df.loc[df[time_col] <= research_end_date]
        df_test = df.loc[df[time_col] >= test_start_date]
    else:
        df_research = df.filter(pl.col(time_col) <= research_end_date)
        df_test = df.filter(pl.col(time_col) >= test_start_date)

    return df_research, df_test
