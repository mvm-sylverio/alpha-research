from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from alpha_research._utils import (
    _validate_positive_integer,
    _validate_df,
    _validate_time_order,
)
from alpha_research.evaluation.ic import (
    TargetFn,
    _generate_target_frames,
    information_coefficient,
)
from alpha_research.evaluation.statistical_tests import (
    fdr_correction,
    wald_temporal_association_test,
)
from alpha_research.features.schema import (
    get_feature_name,
    join_feature_target_frames,
)
from alpha_research.resampling.block_bootstrap import (
    bootstrap_metrics,
    generate_moving_blocks,
    moving_block_bootstrap,
)


__all__ = [
    'temporal_association',
    'temporal_association_summary_table',
    'TemporalAssociationDecayResult',
    'temporal_association_decay',
    'TemporalAssociationDecaySummaryResult',
    'temporal_association_decay_summary',
    'TemporalAssociationDecaySummaryTableResult',
    'temporal_association_decay_summary_table',
    'RollingTemporalAssociationResult',
    'rolling_temporal_association',
    'summarize_rolling_temporal_association',
]


@dataclass(frozen=True, slots=True)
class RollingTemporalAssociationResult:
    """
    Result of a rolling temporal-association analysis.

    Attributes
    ----------
    rolling_frame : pd.DataFrame | pl.DataFrame
        One row per requested rolling-window endpoint. It contains the
        observed association, percentile bootstrap interval, directional
        bootstrap stability, and a computation status.
    summary_table : pd.DataFrame | pl.DataFrame
        One-row descriptive summary of rolling_frame. Its statistics describe
        overlapping rolling windows and are not independent hypothesis tests.
    """
    rolling_frame: pd.DataFrame | pl.DataFrame
    summary_table: pd.DataFrame | pl.DataFrame


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


def _slice_rows(
        df: pd.DataFrame | pl.DataFrame,
        start: int,
        stop: int,
) -> pd.DataFrame | pl.DataFrame:
    """
    Select a positional half-open interval while preserving its DataFrame backend.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Ordered DataFrame from which to select rows. The caller has already
        validated the DataFrame and positional bounds.
    start : int
        Inclusive zero-based row position.
    stop : int
        Exclusive zero-based row position.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Rows in ``[start, stop)`` in their original order and in the same
        Pandas or Polars backend as df.

    Notes
    -----
    This helper deliberately performs no validation because it is used inside
    the rolling loop after the public function has constructed valid bounds.
    """
    if isinstance(df, pd.DataFrame):
        return df.iloc[start:stop]

    return df.slice(start, stop - start)


def _rolling_random_states(
        n_windows: int,
        random_state: int | None,
) -> list[int | None]:
    """
    Derive one bootstrap seed per rolling window from an optional root seed.

    Parameters
    ----------
    n_windows : int
        Number of rolling window endpoints that will run bootstrap sampling.
    random_state : int | None
        Root seed supplied to rolling_temporal_association(). ``None`` keeps
        bootstrap sampling non-deterministic.

    Returns
    -------
    list[int | None]
        ``n_windows`` deterministic child seeds when random_state is given,
        otherwise a list of ``None`` values. Child seeds prevent every rolling
        point from reusing the same random block-draw sequence.

    Raises
    ------
    TypeError
        If random_state is neither an integer nor None.
    """
    if random_state is None:
        return [None] * n_windows

    if (
            not isinstance(random_state, (int, np.integer))
            or isinstance(random_state, bool)
    ):
        raise TypeError('random_state must be an integer or None.')

    seed_sequence = np.random.SeedSequence(int(random_state))
    return [
        int(child.generate_state(1)[0])
        for child in seed_sequence.spawn(n_windows)
    ]


def _rolling_summary_pandas(
        rolling_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate descriptive diagnostics from one Pandas rolling-result frame.

    Parameters
    ----------
    rolling_frame : pd.DataFrame
        Validated non-empty rolling-result frame. It must contain the schema
        required by summarize_rolling_temporal_association().

    Returns
    -------
    pd.DataFrame
        One-row summary with counts of valid windows, association distribution
        metrics, bootstrap-directional stability, and proportions of windows
        whose percentile interval is positive, negative, or crosses zero.

    Notes
    -----
    Rows whose status is not ``'ok'`` are counted as invalid but excluded from
    numerical diagnostics. The public summary function validates schema and
    converts Polars input before calling this helper.
    """
    valid = rolling_frame.loc[
        (rolling_frame['status'] == 'ok')
        & np.isfinite(rolling_frame['association'])
    ].copy()

    association = valid['association'].to_numpy(dtype=float)
    pct_positive = valid['bootstrap_pct_positive'].to_numpy(dtype=float)
    finite_pct_positive = pct_positive[np.isfinite(pct_positive)]

    if len(valid):
        ci_lower = valid['bootstrap_ci_lower'].to_numpy(dtype=float)
        ci_upper = valid['bootstrap_ci_upper'].to_numpy(dtype=float)
        ci_strictly_positive = np.mean(ci_lower > 0.0)
        ci_strictly_negative = np.mean(ci_upper < 0.0)
        ci_contains_zero = np.mean((ci_lower <= 0.0) & (ci_upper >= 0.0))
        association_mean = float(np.mean(association))
        association_median = float(np.median(association))
        association_std = (
            float(np.std(association, ddof=1)) if len(association) > 1 else np.nan
        )
        association_min = float(np.min(association))
        association_max = float(np.max(association))
        association_q25 = float(np.quantile(association, 0.25))
        association_q75 = float(np.quantile(association, 0.75))
        association_pct_positive = float(np.mean(association > 0.0))
    else:
        ci_strictly_positive = np.nan
        ci_strictly_negative = np.nan
        ci_contains_zero = np.nan
        association_mean = np.nan
        association_median = np.nan
        association_std = np.nan
        association_min = np.nan
        association_max = np.nan
        association_q25 = np.nan
        association_q75 = np.nan
        association_pct_positive = np.nan

    first_row = rolling_frame.iloc[0]
    last_row = rolling_frame.iloc[-1]
    return pd.DataFrame([{
        'symbol': first_row['symbol'],
        'feature': first_row['feature'],
        'target': first_row['target'],
        'corr_method': first_row['corr_method'],
        'bootstrap_method': first_row['bootstrap_method'],
        'first_window_end': first_row['window_end'],
        'last_window_end': last_row['window_end'],
        'n_windows': len(rolling_frame),
        'n_valid_windows': len(valid),
        'n_invalid_windows': len(rolling_frame) - len(valid),
        'association_mean': association_mean,
        'association_median': association_median,
        'association_std': association_std,
        'association_min': association_min,
        'association_max': association_max,
        'association_quantile25': association_q25,
        'association_quantile75': association_q75,
        'association_pct_positive': association_pct_positive,
        'ci_pct_strictly_positive': float(ci_strictly_positive),
        'ci_pct_strictly_negative': float(ci_strictly_negative),
        'ci_pct_contains_zero': float(ci_contains_zero),
        'mean_bootstrap_pct_positive': (
            float(np.mean(finite_pct_positive))
            if len(finite_pct_positive)
            else np.nan
        ),
    }])


def summarize_rolling_temporal_association(
        rolling_frame: pd.DataFrame | pl.DataFrame,
) -> pd.DataFrame | pl.DataFrame:
    """
    Summarize one rolling temporal-association result frame.

    Parameters
    ----------
    rolling_frame : pd.DataFrame | pl.DataFrame
        Frame returned as ``rolling_frame`` by rolling_temporal_association().

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        One row with descriptive rolling diagnostics. The output backend
        matches rolling_frame.

    Notes
    -----
    Consecutive rolling windows overlap by construction. Therefore, the
    reported quantities are descriptive diagnostics, not independent tests of
    a temporal association hypothesis.
    """
    _validate_df(
        rolling_frame,
        [
            'symbol', 'feature', 'target', 'corr_method', 'bootstrap_method',
            'window_end', 'association', 'bootstrap_ci_lower',
            'bootstrap_ci_upper', 'bootstrap_pct_positive', 'status',
        ],
        check_all_missing=False,
    )

    pandas_frame = (
        rolling_frame
        if isinstance(rolling_frame, pd.DataFrame)
        else rolling_frame.to_pandas()
    )
    summary = _rolling_summary_pandas(pandas_frame)

    if isinstance(rolling_frame, pd.DataFrame):
        return summary

    return pl.from_pandas(summary)


def rolling_temporal_association(
        df: pd.DataFrame | pl.DataFrame,
        feature: str,
        target: str,
        window_size: int,
        block_length: int,
        n_bootstraps: int,
        corr_method: Literal['pearson', 'spearman'] = 'spearman',
        window_step: int = 1,
        bootstrap_method: Literal['moving_block'] = 'moving_block',
        bootstrap_step: int = 1,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        time_col: str = 'time',
        symbol_col: str = 'symbol',
) -> RollingTemporalAssociationResult:
    """
    Compute bootstrap temporal-association diagnostics in rolling time windows.

    Each full window estimates the temporal association of paired feature and
    target observations, then applies Moving Block Bootstrap (MBB) to the
    paired rows. Feature and target are therefore resampled together within
    every contiguous block.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Single-asset data with increasing, unique time values. It must contain
        time_col, symbol_col, feature, and target.
    feature : str
        Feature column known at each observation time.
    target : str
        Target column aligned with feature at the same observation time.
    window_size : int
        Number of consecutive input observations in each rolling window.
    block_length : int
        Number of observations in each MBB block. It must be smaller than
        window_size.
    n_bootstraps : int
        Number of MBB replicates per valid rolling window. It must be at least
        two so that bootstrap dispersion is defined.
    corr_method : {'pearson', 'spearman'}, default 'spearman'
        Correlation estimator for observed and bootstrap associations.
    window_step : int, default 1
        Number of input observations between consecutive window endpoints.
    bootstrap_method : {'moving_block'}, default 'moving_block'
        Resampling method. MBB is the only supported public method in this
        version; the explicit argument records the methodology and reserves a
        controlled extension point for future tested methods.
    bootstrap_step : int, default 1
        Candidate-block start increment for MBB. A value of one generates
        canonical overlapping candidate blocks.
    confidence_level : float, default 0.95
        Confidence level for the percentile bootstrap interval.
    random_state : int | None, default None
        Root seed. When provided, independent deterministic child seeds are
        used for consecutive windows.
    time_col : str, default 'time'
        Time column. Results are labelled by the feature-observation time at
        the end of each window.
    symbol_col : str, default 'symbol'
        Single-asset identifier column.

    Returns
    -------
    RollingTemporalAssociationResult
        ``rolling_frame`` contains one row per requested window endpoint with
        these main fields: window_start, window_end, association,
        bootstrap_ci_lower, bootstrap_ci_upper, bootstrap_pct_positive,
        n_obs, n_bootstraps, and status. ``summary_table`` describes the
        rolling output without treating overlapping windows as independent.

    Raises
    ------
    ValueError
        If temporal data are invalid, a sizing argument is invalid, the input
        is shorter than window_size, or bootstrap_method is unsupported.
    TypeError
        If df or random_state has an unsupported type.

    Notes
    -----
    Windows are strict: a window with any missing feature-target pair receives
    ``status='missing_pairs'`` and no association is calculated. This avoids
    deleting internal gaps and treating temporally separated observations as
    adjacent MBB data.

    The interval is a percentile bootstrap interval, suitable for visualizing
    local bootstrap uncertainty. The bootstrap sign proportion is a
    directional-stability diagnostic, not a p-value. No pointwise hypothesis
    tests are reported because rolling windows overlap strongly.
    """
    _validate_df(df, [time_col, symbol_col, feature, target])
    _validate_single_symbol(df, symbol_col)
    _validate_time_order(df, time_col)
    _validate_positive_integer(window_size, 'window_size')
    _validate_positive_integer(window_step, 'window_step')
    _validate_positive_integer(block_length, 'block_length')
    _validate_positive_integer(bootstrap_step, 'bootstrap_step')
    _validate_positive_integer(n_bootstraps, 'n_bootstraps')

    if bootstrap_method != 'moving_block':
        raise ValueError(
            "bootstrap_method must be 'moving_block' in this version.",
        )

    if window_size > len(df):
        raise ValueError(
            f'window_size ({window_size}) must not exceed the number of '
            f'observations ({len(df)}).',
        )

    if block_length >= window_size:
        raise ValueError('block_length must be smaller than window_size.')

    if n_bootstraps < 2:
        raise ValueError('n_bootstraps must be at least two.')

    if (
            not isinstance(confidence_level, (int, float, np.integer, np.floating))
            or isinstance(confidence_level, bool)
            or not 0 < confidence_level < 1
    ):
        raise ValueError('confidence_level must be strictly between 0 and 1.')

    window_end_positions = list(range(window_size - 1, len(df), window_step))
    window_random_states = _rolling_random_states(
        len(window_end_positions),
        random_state,
    )
    symbol = df[symbol_col].iloc[0] if isinstance(df, pd.DataFrame) else df[symbol_col][0]
    rows = []

    for end_position, window_random_state in zip(
            window_end_positions,
            window_random_states,
    ):
        start_position = end_position - window_size + 1
        window = _slice_rows(df, start_position, end_position + 1)
        valid_pairs = _select_valid_temporal_pairs(window, feature, target)
        row = {
            'symbol': symbol,
            'feature': feature,
            'target': target,
            'corr_method': corr_method,
            'bootstrap_method': bootstrap_method,
            'window_start': (
                window[time_col].iloc[0]
                if isinstance(window, pd.DataFrame)
                else window[time_col][0]
            ),
            'window_end': (
                window[time_col].iloc[-1]
                if isinstance(window, pd.DataFrame)
                else window[time_col][-1]
            ),
            'n_obs': len(valid_pairs),
            'association': np.nan,
            'bootstrap_ci_lower': np.nan,
            'bootstrap_ci_upper': np.nan,
            'bootstrap_pct_positive': np.nan,
            'n_bootstraps': 0,
            'status': 'missing_pairs',
        }

        if len(valid_pairs) != window_size:
            rows.append(row)
            continue

        observed_association = information_coefficient(
            valid_pairs[feature],
            valid_pairs[target],
            corr_method=corr_method,
        )
        row['association'] = observed_association

        if not np.isfinite(observed_association):
            row['status'] = 'undefined_association'
            rows.append(row)
            continue

        blocks = generate_moving_blocks(
            valid_pairs,
            block_length=block_length,
            step=bootstrap_step,
        )
        bootstrap_samples = moving_block_bootstrap(
            blocks,
            sample_size=window_size,
            n_bootstraps=n_bootstraps,
            random_state=window_random_state,
        )
        bootstrap_estimates = [
            information_coefficient(
                sample[feature],
                sample[target],
                corr_method=corr_method,
            )
            for sample in bootstrap_samples
        ]

        try:
            metrics = bootstrap_metrics(
                bootstrap_estimates,
                confidence_level=confidence_level,
            )
        except ValueError:
            row['status'] = 'undefined_bootstrap'
            rows.append(row)
            continue

        row.update({
            'bootstrap_ci_lower': metrics.ci_lower,
            'bootstrap_ci_upper': metrics.ci_upper,
            'bootstrap_pct_positive': metrics.pct_positive,
            'n_bootstraps': metrics.n_bootstraps,
            'status': 'ok',
        })
        rows.append(row)

    rolling_pandas = pd.DataFrame(rows)
    if isinstance(df, pd.DataFrame):
        rolling_frame = rolling_pandas
    else:
        rolling_frame = pl.from_pandas(rolling_pandas)

    return RollingTemporalAssociationResult(
        rolling_frame=rolling_frame,
        summary_table=summarize_rolling_temporal_association(rolling_frame),
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
        individual Wald test results including the Wald interval
        (wald_ci_lower and wald_ci_upper), and feature group metadata.
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


@dataclass(frozen=True, slots=True)
class TemporalAssociationDecayResult:
    """
    Result of a temporal-association decay computation.

    Attributes
    ----------
    table : pd.DataFrame | pl.DataFrame
        One row per target horizon. It contains the observed temporal
        association, Moving Block Bootstrap metrics, Wald-test results, and
        FDR-corrected significance for the evaluated feature.
    """
    table: pd.DataFrame | pl.DataFrame


def temporal_association_decay(
        df_feature: pd.DataFrame | pl.DataFrame,
        feature: str,
        target_data: pd.DataFrame | pl.DataFrame,
        horizons: list[int],
        target_fn: TargetFn,
        block_length: int,
        n_bootstraps: int,
        corr_method: Literal['pearson', 'spearman'] = 'spearman',
        step: int = 1,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        time_col: str = 'time',
        symbol_col: str = 'symbol',
        feature_groups: dict[str, str] | None = None,
        fdr: float = 0.05,
        fdr_method: Literal['bh', 'by'] = 'bh',
) -> TemporalAssociationDecayResult:
    """
    Compute the temporal-association decay curve of one feature.

    The feature is held fixed while target_fn generates one aligned forward
    target for every requested horizon. At each horizon, the function
    estimates the single-asset temporal association, obtains uncertainty with
    Moving Block Bootstrap (MBB), and applies a Wald test. FDR correction is
    then applied jointly across the horizons of the feature.

    Parameters
    ----------
    df_feature : pd.DataFrame | pl.DataFrame
        Wide single-asset DataFrame containing time_col, symbol_col, and
        feature. Times must be unique and strictly increasing.
    feature : str
        Feature column in df_feature to evaluate.
    target_data : pd.DataFrame | pl.DataFrame
        DataFrame supplied to target_fn to generate each horizon-specific
        target. It must use the same backend as df_feature.
    horizons : list[int]
        Unique positive forward horizons to evaluate.
    target_fn : Callable[[pd.DataFrame | pl.DataFrame, int], pd.DataFrame | pl.DataFrame]
        Function called as ``target_fn(target_data, horizon=horizon)``. Each
        returned frame must contain time_col, symbol_col, and exactly one
        target value column. Fixed target configuration may be supplied with
        functools.partial as long as target_data and horizon remain accepted.
    block_length : int
        Candidate MBB block size, in observations.
    n_bootstraps : int
        Number of MBB samples for each feature-horizon estimate.
    corr_method : {'pearson', 'spearman'}, default 'spearman'
        Correlation estimator for observed and bootstrap associations.
    step : int, default 1
        Candidate-block start increment passed to MBB.
    confidence_level : float, default 0.95
        Confidence level used in bootstrap metrics and Wald intervals.
    random_state : int | None, default None
        Root seed for MBB. When supplied, deterministic distinct child seeds
        are used for horizons so they do not reuse an identical block-draw
        sequence.
    time_col : str, default 'time'
        Temporal key column.
    symbol_col : str, default 'symbol'
        Single-asset identifier column.
    feature_groups : dict[str, str] | None, default None
        Optional semantic group mapping. It is metadata; the FDR family is
        always the horizons of this one feature.
    fdr : float, default 0.05
        False discovery rate for the horizon family.
    fdr_method : {'bh', 'by'}, default 'bh'
        Multiple-testing correction method.

    Returns
    -------
    TemporalAssociationDecayResult
        Detailed temporal-association results ordered by ascending horizon.

    Raises
    ------
    ValueError
        If the single-asset temporal contract, feature or target frame,
        horizons, MBB configuration, Wald test, or FDR configuration is
        invalid.
    KeyError
        If a required feature, key, or generated target column is absent.
    TypeError
        If DataFrame backends differ or target_fn returns an unsupported
        DataFrame type.

    Notes
    -----
    This is a full-sample horizon-persistence analysis. It is distinct from a
    rolling temporal-association diagnostic and does not construct windows.
    """
    _validate_df(df_feature, [time_col, symbol_col, feature])
    _validate_single_symbol(df_feature, symbol_col)
    _validate_time_order(df_feature, time_col)

    target_frames = _generate_target_frames(
        df_feature=df_feature,
        target_data=target_data,
        horizons=horizons,
        target_fn=target_fn,
    )
    return _temporal_association_decay_from_target_frames(
        df_feature=df_feature,
        feature=feature,
        target_frames=target_frames,
        block_length=block_length,
        n_bootstraps=n_bootstraps,
        corr_method=corr_method,
        step=step,
        confidence_level=confidence_level,
        random_state=random_state,
        time_col=time_col,
        symbol_col=symbol_col,
        feature_groups=feature_groups,
        fdr=fdr,
        fdr_method=fdr_method,
    )


def _temporal_association_decay_from_target_frames(
        df_feature: pd.DataFrame | pl.DataFrame,
        feature: str,
        target_frames: dict[int, pd.DataFrame | pl.DataFrame],
        block_length: int,
        n_bootstraps: int,
        corr_method: Literal['pearson', 'spearman'],
        step: int,
        confidence_level: float,
        random_state: int | None,
        time_col: str,
        symbol_col: str,
        feature_groups: dict[str, str] | None,
        fdr: float,
        fdr_method: Literal['bh', 'by'],
) -> TemporalAssociationDecayResult:
    """
    Compute one feature's temporal-association decay from generated targets.

    Parameters
    ----------
    df_feature : pd.DataFrame | pl.DataFrame
        Validated single-asset feature DataFrame.
    feature : str
        Feature column to evaluate.
    target_frames : dict[int, pd.DataFrame | pl.DataFrame]
        Generated target frames indexed by horizon. Each frame must contain
        time_col, symbol_col, and exactly one target value column.
    block_length : int
        Candidate MBB block size.
    n_bootstraps : int
        Number of MBB samples per horizon.
    corr_method : {'pearson', 'spearman'}
        Association estimator for observed and bootstrap estimates.
    step : int
        Candidate-block start increment passed to MBB.
    confidence_level : float
        Confidence level for bootstrap and Wald metrics.
    random_state : int | None
        Root seed used to derive one child seed per horizon.
    time_col : str
        Temporal key column.
    symbol_col : str
        Single-asset identifier column.
    feature_groups : dict[str, str] | None
        Optional feature-group mapping.
    fdr : float
        False discovery rate across horizons.
    fdr_method : {'bh', 'by'}
        FDR correction method.

    Returns
    -------
    TemporalAssociationDecayResult
        Horizon-level MBB, Wald, and FDR results sorted by horizon.

    Raises
    ------
    ValueError
        If a generated target or a downstream temporal-association calculation
        does not meet its required contract.
    TypeError
        If feature and target frames use different DataFrame backends.

    Notes
    -----
    This helper does not invoke target_fn. It is shared by the single-feature
    and multi-feature public functions so one generated target frame can be
    reused for every feature.
    """
    feature_group = (
        feature_groups.get(feature, 'ungrouped')
        if feature_groups is not None
        else 'ungrouped'
    )
    child_random_states = _rolling_random_states(
        n_windows=len(target_frames),
        random_state=random_state,
    )
    rows = []

    for (horizon, target_df), child_random_state in zip(
            target_frames.items(),
            child_random_states,
    ):
        target_col = get_feature_name(target_df, time_col, symbol_col)
        joined_df = join_feature_target_frames(
            feature_df=df_feature,
            target_df=target_df,
            feature_col=feature,
            target_col=target_col,
            time_col=time_col,
            symbol_col=symbol_col,
        )
        if isinstance(joined_df, pd.DataFrame):
            joined_df = joined_df.sort_values(time_col).reset_index(drop=True)
        else:
            joined_df = joined_df.sort(time_col)

        horizon_table = temporal_association_summary_table(
            df=joined_df,
            feature_list=[feature],
            target=target_col,
            block_length=block_length,
            n_bootstraps=n_bootstraps,
            corr_method=corr_method,
            step=step,
            confidence_level=confidence_level,
            random_state=child_random_state,
            time_col=time_col,
            symbol_col=symbol_col,
            feature_groups={feature: feature_group},
        )
        if isinstance(horizon_table, pd.DataFrame):
            row = horizon_table.iloc[0].to_dict()
        else:
            row = horizon_table.row(0, named=True)

        rows.append({
            **row,
            'target': target_col,
            'horizon': horizon,
        })

    if isinstance(df_feature, pd.DataFrame):
        result_table = pd.DataFrame(rows)
    else:
        result_table = pl.DataFrame(rows)

    result_table = fdr_correction(
        result_table,
        fdr=fdr,
        method=fdr_method,
    )
    if isinstance(result_table, pd.DataFrame):
        result_table = result_table.sort_values('horizon').reset_index(drop=True)
    else:
        result_table = result_table.sort('horizon')

    return TemporalAssociationDecayResult(table=result_table)


@dataclass(frozen=True, slots=True)
class TemporalAssociationDecaySummaryResult:
    """
    Scalar summary of a temporal-association decay curve.

    Attributes
    ----------
    peak_horizon : int
        Horizon with the largest absolute temporal association.
    peak_abs_association : float
        Absolute temporal association at peak_horizon.
    halflife_horizon : int | None
        First horizon at or after peak_horizon where absolute association is at
        most half of its peak magnitude.
    last_significant_horizon : int | None
        Largest horizon passing FDR correction.
    auc : float
        Trapezoidal area under absolute association versus horizon.
    """
    peak_horizon: int
    peak_abs_association: float
    halflife_horizon: int | None
    last_significant_horizon: int | None
    auc: float


def temporal_association_decay_summary(
        decay_curve: pd.DataFrame | pl.DataFrame,
) -> TemporalAssociationDecaySummaryResult:
    """
    Summarize a temporal-association decay curve into scalar diagnostics.

    Parameters
    ----------
    decay_curve : pd.DataFrame | pl.DataFrame
        Table returned by temporal_association_decay().table. It must contain
        horizon, association, and fdr_rejected columns.

    Returns
    -------
    TemporalAssociationDecaySummaryResult
        Peak magnitude, discrete half-life, FDR persistence, and area under
        the absolute association curve.

    Raises
    ------
    ValueError
        If decay_curve is empty, has duplicate horizons, or has no finite
        association values.
    KeyError
        If a required decay-curve column is absent.
    TypeError
        If decay_curve is not a Pandas or Polars DataFrame.

    Notes
    -----
    Half-life is a discrete grid diagnostic without interpolation. A zero peak
    has no half-life because a relative 50% threshold is not informative.
    AUC comparisons are meaningful only on the same horizon grid.
    """
    _validate_df(decay_curve, ['horizon', 'association', 'fdr_rejected'])

    if isinstance(decay_curve, pd.DataFrame):
        curve = decay_curve.sort_values('horizon')
        horizons = curve['horizon'].to_numpy(dtype=int)
        association = curve['association'].to_numpy(dtype=float)
        rejected = curve['fdr_rejected'].to_numpy(dtype=bool)
    else:
        curve = decay_curve.sort('horizon')
        horizons = curve['horizon'].to_numpy()
        association = curve['association'].to_numpy()
        rejected = curve['fdr_rejected'].to_numpy()

    if len(horizons) == 0:
        raise ValueError('decay_curve must not be empty.')

    if len(np.unique(horizons)) != len(horizons):
        raise ValueError('decay_curve must contain exactly one row per horizon.')

    finite_mask = np.isfinite(association)
    if not finite_mask.any():
        raise ValueError('decay_curve contains no finite association values.')

    valid_horizons = horizons[finite_mask]
    valid_abs_association = np.abs(association[finite_mask])
    peak_idx = int(np.argmax(valid_abs_association))
    peak_horizon = int(valid_horizons[peak_idx])
    peak_abs_association = float(valid_abs_association[peak_idx])

    halflife_horizon = None
    if not np.isclose(peak_abs_association, 0.0, atol=1e-12):
        post_peak_mask = valid_horizons >= peak_horizon
        half_mask = (
            valid_abs_association[post_peak_mask]
            <= peak_abs_association / 2
        )
        if half_mask.any():
            halflife_horizon = int(
                valid_horizons[post_peak_mask][np.flatnonzero(half_mask)[0]]
            )

    significant_horizons = horizons[rejected]
    last_significant_horizon = (
        int(np.max(significant_horizons))
        if len(significant_horizons)
        else None
    )
    auc = (
        float(np.trapezoid(y=valid_abs_association, x=valid_horizons))
        if len(valid_horizons) >= 2
        else 0.0
    )

    return TemporalAssociationDecaySummaryResult(
        peak_horizon=peak_horizon,
        peak_abs_association=peak_abs_association,
        halflife_horizon=halflife_horizon,
        last_significant_horizon=last_significant_horizon,
        auc=auc,
    )


@dataclass(frozen=True, slots=True)
class TemporalAssociationDecaySummaryTableResult:
    """
    Result of a multi-feature temporal-association decay computation.

    Attributes
    ----------
    table : pd.DataFrame | pl.DataFrame
        One scalar decay summary row per requested feature.
    decay_results : dict[str, TemporalAssociationDecayResult]
        Complete horizon-level decay results indexed by feature name.
    """
    table: pd.DataFrame | pl.DataFrame
    decay_results: dict[str, TemporalAssociationDecayResult]


def temporal_association_decay_summary_table(
        df_features: pd.DataFrame | pl.DataFrame,
        feature_list: list[str],
        target_data: pd.DataFrame | pl.DataFrame,
        horizons: list[int],
        target_fn: TargetFn,
        block_length: int,
        n_bootstraps: int,
        corr_method: Literal['pearson', 'spearman'] = 'spearman',
        step: int = 1,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        time_col: str = 'time',
        symbol_col: str = 'symbol',
        feature_groups: dict[str, str] | None = None,
        fdr: float = 0.05,
        fdr_method: Literal['bh', 'by'] = 'bh',
) -> TemporalAssociationDecaySummaryTableResult:
    """
    Compute temporal-association decay diagnostics for multiple features.

    The function generates each target once per requested horizon and reuses
    those target frames across all features. Each feature receives its own MBB
    estimates, Wald tests, and FDR correction across its horizon family.

    Parameters
    ----------
    df_features : pd.DataFrame | pl.DataFrame
        Wide single-asset DataFrame with time_col, symbol_col, and all columns
        in feature_list. Times must be unique and strictly increasing.
    feature_list : list[str]
        Non-empty, non-duplicated feature columns to evaluate.
    target_data : pd.DataFrame | pl.DataFrame
        DataFrame supplied to target_fn for each horizon.
    horizons : list[int]
        Unique positive target horizons shared by all features.
    target_fn : Callable[[pd.DataFrame | pl.DataFrame, int], pd.DataFrame | pl.DataFrame]
        Function called as ``target_fn(target_data, horizon=horizon)``.
    block_length : int
        Candidate MBB block size, in observations.
    n_bootstraps : int
        Number of MBB samples per feature and horizon.
    corr_method : {'pearson', 'spearman'}, default 'spearman'
        Association estimator.
    step : int, default 1
        Candidate-block start increment for MBB.
    confidence_level : float, default 0.95
        Confidence level for bootstrap and Wald metrics.
    random_state : int | None, default None
        Root MBB seed. Child seeds vary by horizon deterministically.
    time_col : str, default 'time'
        Temporal key column.
    symbol_col : str, default 'symbol'
        Single-asset identifier column.
    feature_groups : dict[str, str] | None, default None
        Optional semantic groups for result metadata.
    fdr : float, default 0.05
        False discovery rate per feature across its horizons.
    fdr_method : {'bh', 'by'}, default 'bh'
        Multiple-testing correction method.

    Returns
    -------
    TemporalAssociationDecaySummaryTableResult
        A summary table and complete detailed decay result for each feature.

    Raises
    ------
    ValueError
        If feature_list, horizons, temporal data, MBB configuration, or a
        downstream statistic is invalid.
    KeyError
        If required feature, key, or target columns are absent.
    TypeError
        If DataFrame backends differ.

    Notes
    -----
    This function does not perform rolling analysis. It evaluates one
    full-sample association per feature-horizon pair.
    """
    if not feature_list:
        raise ValueError('feature_list must not be empty.')

    if len(set(feature_list)) != len(feature_list):
        raise ValueError('feature_list must not contain duplicates.')

    _validate_df(df_features, [time_col, symbol_col] + feature_list)
    _validate_single_symbol(df_features, symbol_col)
    _validate_time_order(df_features, time_col)

    target_frames = _generate_target_frames(
        df_feature=df_features,
        target_data=target_data,
        horizons=horizons,
        target_fn=target_fn,
    )

    rows = []
    decay_results = {}
    for feature in feature_list:
        decay_result = _temporal_association_decay_from_target_frames(
            df_feature=df_features,
            feature=feature,
            target_frames=target_frames,
            block_length=block_length,
            n_bootstraps=n_bootstraps,
            corr_method=corr_method,
            step=step,
            confidence_level=confidence_level,
            random_state=random_state,
            time_col=time_col,
            symbol_col=symbol_col,
            feature_groups=feature_groups,
            fdr=fdr,
            fdr_method=fdr_method,
        )
        decay_table = decay_result.table
        if isinstance(decay_table, pd.DataFrame):
            feature_group = decay_table['feature_group'].iloc[0]
        else:
            feature_group = decay_table['feature_group'][0]

        summary = temporal_association_decay_summary(decay_table)
        decay_results[feature] = decay_result
        rows.append({
            'feature': feature,
            'peak_horizon': summary.peak_horizon,
            'peak_abs_association': summary.peak_abs_association,
            'halflife_horizon': summary.halflife_horizon,
            'last_significant_horizon': summary.last_significant_horizon,
            'auc': summary.auc,
            'feature_group': feature_group,
        })

    if isinstance(df_features, pd.DataFrame):
        table = pd.DataFrame(rows)
    else:
        table = pl.DataFrame(rows)

    return TemporalAssociationDecaySummaryTableResult(
        table=table,
        decay_results=decay_results,
    )
