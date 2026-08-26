from typing import Literal

import pandas as pd
import polars as pl

from alpha_research._utils import (
    _validate_df,
    _validate_time_order,
)
from alpha_research.evaluation.ic import information_coefficient


__all__ = ['temporal_association']


def _validate_single_symbol(
        df: pd.DataFrame | pl.DataFrame,
        symbol_col: str,
) -> None:
    """Validate that a DataFrame represents exactly one non-missing asset."""
    symbols = df[symbol_col]

    if isinstance(df, pd.DataFrame):
        has_missing = symbols.isna().any()
        n_symbols = symbols.nunique(dropna=True)
    else:
        has_missing = symbols.is_null().any()
        n_symbols = symbols.drop_nulls().n_unique()

    if has_missing or n_symbols != 1:
        raise ValueError(
            f'{symbol_col} must contain exactly one non-missing symbol for '
            'temporal association.'
        )


def temporal_association(
        df: pd.DataFrame | pl.DataFrame,
        feature: str,
        target: str,
        corr_method: Literal['pearson', 'spearman'] = 'spearman',
        time_col: str = 'time',
        symbol_col: str = 'symbol',
) -> float:
    """
    Compute the association between a feature and a target along one asset's time axis.

    This is the time-series analogue of an Information Coefficient: it uses
    the same Pearson or Spearman correlation estimator as
    ``information_coefficient()``, but correlates paired observations through
    time instead of across assets at a single time.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Single-asset DataFrame containing time_col, symbol_col, feature, and
        target. Rows must be increasingly ordered by time_col and each time
        must identify at most one observation.
    feature : str
        Feature column known at each observation time.
    target : str
        Target column aligned with the feature observation time.
    corr_method : {'pearson', 'spearman'}
        Correlation estimator. Spearman is the default.
    time_col : str
        Time column name. Default is 'time'.
    symbol_col : str
        Single-asset identifier column. Default is 'symbol'.

    Returns
    -------
    float
        Correlation coefficient in [-1, 1], or nan when either aligned series
        is constant.

    Raises
    ------
    ValueError
        If df contains more than one symbol, missing symbols, duplicate or
        unordered times.
    KeyError
        If a required column is absent.
    TypeError
        If df is not a Pandas or Polars DataFrame.

    Notes
    -----
    Missing feature or target values are removed only after the temporal data
    contract has been validated. This function reports a point association;
    bootstrap confidence intervals and purged validation are separate steps.
    """
    _validate_df(df, [time_col, symbol_col, feature, target])
    _validate_single_symbol(df, symbol_col)
    _validate_time_order(df, time_col)

    if isinstance(df, pd.DataFrame):
        aligned = df[[feature, target]].dropna()
    else:
        aligned = (
            df
            .select([feature, target])
            .drop_nulls([feature, target])
            .drop_nans([feature, target])
        )

    return information_coefficient(
        aligned[feature],
        aligned[target],
        corr_method=corr_method,
    )
