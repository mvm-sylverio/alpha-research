from typing import Literal

import pandas as pd
import polars as pl

from alpha_research._utils import (
    _validate_df,
    _validate_time_order,
)
from alpha_research.evaluation.ic import information_coefficient
from alpha_research.evaluation.statistical_tests import wald_temporal_association_test
from alpha_research.resampling.block_bootstrap import (
    bootstrap_metrics,
    generate_moving_blocks,
    moving_block_bootstrap,
)


__all__ = ['temporal_association', 'temporal_association_summary_table']


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


def _select_valid_temporal_pairs(
        df: pd.DataFrame | pl.DataFrame,
        feature: str,
        target: str,
) -> pd.DataFrame | pl.DataFrame:
    """
    Select paired non-missing feature and target observations.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame containing feature and target columns. The caller is
        responsible for validating the temporal data contract first.
    feature : str
        Feature column name.
    target : str
        Target column name.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Feature and target columns only, excluding rows where either value is
        null or NaN. The input backend and original row order are preserved.

    Raises
    ------
    KeyError
        If feature or target is not a column of df.
    TypeError
        If df is not a Pandas or Polars DataFrame.
    """
    _validate_df(df, [feature, target])

    if isinstance(df, pd.DataFrame):
        return df[[feature, target]].dropna()

    return (
        df
        .select([feature, target])
        .drop_nulls([feature, target])
        .drop_nans([feature, target])
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

    aligned = _select_valid_temporal_pairs(df, feature, target)

    return information_coefficient(
        aligned[feature],
        aligned[target],
        corr_method=corr_method,
    )


def temporal_association_summary_table(
        df: pd.DataFrame | pl.DataFrame,
        feature_list: list[str],
        target: str,
        block_length: int,
        n_bootstraps: int,
        corr_method: Literal['pearson', 'spearman'] = 'spearman',
        step: int = 1,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        time_col: str = 'time',
        symbol_col: str = 'symbol',
        feature_groups: dict[str, str] | None = None,
) -> pd.DataFrame | pl.DataFrame:
    """
    Summarize bootstrap temporal-association diagnostics for one or more features.

    For each feature, this function computes the observed temporal association,
    applies Moving Block Bootstrap to its paired feature-target observations,
    summarizes the bootstrap estimates, and runs the Wald-style temporal
    association test. FDR correction is intentionally not applied.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Single-asset temporal dataset in increasing, unique time order.
    feature_list : list[str]
        Feature columns to evaluate. It must not be empty or contain duplicates.
    target : str
        Target column to associate with each feature.
    block_length : int
        Candidate block size passed to generate_moving_blocks().
    n_bootstraps : int
        Number of Moving Block Bootstrap samples per feature.
    corr_method : {'pearson', 'spearman'}, default 'spearman'
        Association estimator used for the observed and bootstrap estimates.
    step : int, default 1
        Candidate-block start increment passed to generate_moving_blocks().
    confidence_level : float, default 0.95
        Confidence level used by bootstrap_metrics() for its percentile
        interval and by the Wald test for its Wald interval.
    random_state : int | None, default None
        Reproducible seed passed to moving_block_bootstrap() for each feature.
    time_col : str, default 'time'
        Temporal key column.
    symbol_col : str, default 'symbol'
        Single-asset identifier column.
    feature_groups : dict[str, str] | None, default None
        Optional semantic group mapping. Missing features receive 'ungrouped'.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        One row per feature with observed association, bootstrap metrics,
        percentile bootstrap interval (bootstrap_ci_lower and
        bootstrap_ci_upper), individual Wald test results including the Wald
        interval (wald_ci_lower and wald_ci_upper), and feature group metadata.
        The output uses the same backend as df and contains no FDR correction.

    Raises
    ------
    ValueError
        If feature_list is empty or contains duplicates, or if a downstream
        temporal association, resampling, bootstrap metric, or test input is
        invalid.
    KeyError
        If required dataset columns are absent.
    TypeError
        If df or a downstream argument uses an unsupported type.

    Notes
    -----
    Candidate blocks are generated separately per feature because missing
    values can change the valid feature-target observations. Reusing blocks
    across features in that case would alter their estimands.

    MBB samples can repeat timestamps, so their estimates call the underlying
    information_coefficient() directly after resampling. The observed estimate
    still calls temporal_association(), which enforces the temporal contract.

    Every bootstrap sample has the same length as the valid feature-target
    pairs for its feature.

    The percentile bootstrap interval is reported as a distributional summary.
    reject_h0 and p_value are determined only by the Wald test.
    """
    if not feature_list:
        raise ValueError('feature_list must not be empty.')

    if len(set(feature_list)) != len(feature_list):
        raise ValueError('feature_list must not contain duplicates.')

    rows = []

    for feature in feature_list:
        observed_association = temporal_association(
            df=df,
            feature=feature,
            target=target,
            corr_method=corr_method,
            time_col=time_col,
            symbol_col=symbol_col,
        )
        valid_pairs = _select_valid_temporal_pairs(df, feature, target)
        blocks = generate_moving_blocks(valid_pairs, block_length, step)
        bootstrap_samples = moving_block_bootstrap(
            blocks,
            sample_size=len(valid_pairs),
            n_bootstraps=n_bootstraps,
            random_state=random_state,
        )
        bootstrap_estimates = [
            information_coefficient(
                sample[feature],
                sample[target],
                corr_method=corr_method,
            )
            for sample in bootstrap_samples
        ]
        metrics = bootstrap_metrics(bootstrap_estimates, confidence_level)
        test_result = wald_temporal_association_test(
            observed_association=observed_association,
            bootstrap_standard_error=metrics.std,
            confidence_level=confidence_level,
        )
        feature_group = (
            feature_groups.get(feature, 'ungrouped')
            if feature_groups is not None
            else 'ungrouped'
        )

        rows.append({
            'feature': feature,
            'association': observed_association,
            'corr_method': corr_method,
            'n_obs': len(valid_pairs),
            'bootstrap_mean': metrics.mean,
            'bootstrap_std': metrics.std,
            'bootstrap_pct_positive': metrics.pct_positive,
            'test_statistic': test_result.test_statistic,
            'p_value': test_result.p_value,
            'reject_h0': test_result.reject_h0,
            'wald_ci_lower': test_result.wald_ci_lower,
            'wald_ci_upper': test_result.wald_ci_upper,
            'confidence_level': test_result.confidence_level,
            'alpha': test_result.alpha,
            'n_bootstraps': metrics.n_bootstraps,
            'feature_group': feature_group,
        })

    if isinstance(df, pd.DataFrame):
        return pd.DataFrame(rows)

    return pl.DataFrame(rows)
