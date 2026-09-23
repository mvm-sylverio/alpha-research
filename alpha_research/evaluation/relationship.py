from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from alpha_research._utils import (
    _validate_df,
    _validate_positive_integer,
    _validate_time_order,
    _validate_unique_keys,
)
from alpha_research.evaluation.timeseries import _validate_single_symbol
from alpha_research.resampling.block_bootstrap import (
    bootstrap_metrics,
    generate_moving_blocks,
    moving_block_bootstrap,
)

__all__ = [
    'FeatureTargetRelationshipResult',
    'FeatureTargetRelationshipUncertaintyResult',
    'feature_target_relationship',
    'temporal_feature_target_relationship_uncertainty',
]


@dataclass(frozen=True, slots=True)
class FeatureTargetRelationshipResult:
    """Store the raw and binned views of one feature-target relationship.

    Parameters
    ----------
    pairs : pd.DataFrame | pl.DataFrame
        Finite feature-target observations. The ``bin`` column is assigned
        from feature values only; target values never determine bins.
    bin_summary : pd.DataFrame | pl.DataFrame
        One row per available feature bin. It contains representative feature
        values, mean and median target values, and observation/group counts.
    group_summary : pd.DataFrame | pl.DataFrame | None
        Intermediate statistics for every group and feature bin when
        group_col was supplied. For example, with ``group_col='time'``, each
        row describes one date and one cross-sectional feature bin. Otherwise
        this field is None.
    feature, target : str
        Analyzed feature and target column names.
    binning : {'quantile', 'equal_width'}
        Binning rule used by the analysis.
    n_bins : int
        Requested number of bins.
    group_col : str | None
        Optional column within which bins were assigned independently.
    time_col, symbol_col : str
        Observation key-column names.
    n_input, n_valid, n_dropped, n_unassigned : int
        Row-accounting diagnostics for the analysis.

    Raises
    ------
    TypeError
        If instantiated with invalid values. Validation is normally performed
        by feature_target_relationship before construction.
    """

    pairs: pd.DataFrame | pl.DataFrame
    bin_summary: pd.DataFrame | pl.DataFrame
    group_summary: pd.DataFrame | pl.DataFrame | None
    feature: str
    target: str
    binning: Literal['quantile', 'equal_width']
    n_bins: int
    group_col: str | None
    time_col: str
    symbol_col: str
    n_input: int
    n_valid: int
    n_dropped: int
    n_unassigned: int


@dataclass(frozen=True, slots=True)
class FeatureTargetRelationshipUncertaintyResult:
    """Store temporal MBB uncertainty for one feature-target relationship.

    Parameters
    ----------
    relationship : FeatureTargetRelationshipResult
        Observed full-sample relationship summarized from the original data.
    bin_uncertainty : pd.DataFrame | pl.DataFrame
        One row per observed feature bin with percentile intervals and
        bootstrap standard errors for target means and medians.
    block_length, n_bootstraps, bootstrap_step : int
        Moving Block Bootstrap configuration.
    confidence_level : float
        Percentile interval confidence level.
    random_state : int | None
        Reproducibility seed used for block sampling.

    Returns
    -------
    FeatureTargetRelationshipUncertaintyResult
        Observed structure, bin-level uncertainty, and resampling metadata.

    Raises
    ------
    TypeError
        If instantiated with unsupported values. Validation is normally
        performed by temporal_feature_target_relationship_uncertainty.
    """

    relationship: FeatureTargetRelationshipResult
    bin_uncertainty: pd.DataFrame | pl.DataFrame
    block_length: int
    n_bootstraps: int
    bootstrap_step: int
    confidence_level: float
    random_state: int | None


def _summarize_relationship_pairs(
        pairs: pd.DataFrame,
        feature: str,
        target: str,
        n_bins: int,
        binning: Literal['quantile', 'equal_width'],
        group_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, int]:
    """Assign feature bins and summarize target values without key validation.

    Parameters
    ----------
    pairs : pd.DataFrame
        Finite feature-target observations already validated by the caller.
    feature, target : str
        Numeric feature and target columns.
    n_bins : int
        Positive requested number of feature bins.
    binning : {'quantile', 'equal_width'}
        Feature-only binning rule.
    group_col : str | None
        Optional column within which bins are rebuilt independently.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, int]
        Binned pairs, aggregate bin summary, optional group summary, and the
        number of pairs that could not receive a bin.

    Raises
    ------
    ValueError
        If no observation can receive a feature bin.

    Notes
    -----
    This internal estimator is shared by the observed relationship and each
    temporal bootstrap replicate. Bootstrap samples can repeat source keys,
    so observation-key validation deliberately remains in the public entry
    points rather than in this helper.
    """
    pairs = pairs.copy()
    pairs['bin'] = np.nan
    grouped_indices = (
        [(None, pairs.index)]
        if group_col is None
        else list(pairs.groupby(group_col, sort=False, dropna=False).groups.items())
    )
    for _, indices in grouped_indices:
        values = pairs.loc[indices, feature]
        if len(values) < n_bins:
            continue
        if binning == 'quantile':
            ranks = values.rank(method='average')
            assignments = np.ceil(ranks * n_bins / len(values))
        else:
            minimum = values.min()
            maximum = values.max()
            if minimum == maximum:
                continue
            assignments = np.floor(
                (values - minimum) / (maximum - minimum) * n_bins,
            ) + 1
            assignments = assignments.clip(lower=1, upper=n_bins)
        pairs.loc[indices, 'bin'] = assignments

    n_unassigned = int(pairs['bin'].isna().sum())
    assigned = pairs.dropna(subset=['bin']).copy()
    if assigned.empty:
        raise ValueError('no finite feature-target pairs could receive a bin.')
    pairs['bin'] = pairs['bin'].astype('Int64')
    assigned['bin'] = assigned['bin'].astype(int)

    aggregation = {
        feature: ['mean', 'median', 'min', 'max'],
        target: ['mean', 'median', 'size'],
    }
    if group_col is None:
        bin_summary = assigned.groupby('bin', sort=True).agg(aggregation)
        bin_summary.columns = [
            'feature_mean',
            'feature_median',
            'feature_min',
            'feature_max',
            'target_mean',
            'target_median',
            'n_obs',
        ]
        bin_summary = bin_summary.reset_index()
        bin_summary['n_groups'] = 1
        group_summary = None
    else:
        group_summary = assigned.groupby(
            [group_col, 'bin'],
            sort=True,
            dropna=False,
        ).agg(aggregation)
        group_summary.columns = [
            'feature_mean',
            'feature_median',
            'feature_min',
            'feature_max',
            'target_mean',
            'target_median',
            'n_obs',
        ]
        group_summary = group_summary.reset_index()
        bin_summary = group_summary.groupby('bin', sort=True).agg(
            feature_mean=('feature_mean', 'mean'),
            feature_median=('feature_median', 'median'),
            feature_min=('feature_min', 'min'),
            feature_max=('feature_max', 'max'),
            target_mean=('target_mean', 'mean'),
            target_median=('target_median', 'median'),
            n_obs=('n_obs', 'sum'),
            n_groups=(group_col, 'size'),
        ).reset_index()

    return pairs, bin_summary, group_summary, n_unassigned


def feature_target_relationship(
        df: pd.DataFrame | pl.DataFrame,
        feature: str,
        target: str,
        n_bins: int = 10,
        binning: Literal['quantile', 'equal_width'] = 'quantile',
        group_col: str | None = None,
        time_col: str = 'time',
        symbol_col: str = 'symbol',
) -> FeatureTargetRelationshipResult:
    """Summarize how a target behaves across low-to-high feature values.

    This function prepares a descriptive relationship diagnostic. It orders
    observations by the feature X, assigns X to bins, and summarizes target Y
    inside each X bin. It does not bin Y, calculate a correlation, select a
    transformation, or test statistical significance.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Aligned research frame with one unique row per time-symbol pair.
    feature : str
        Numeric X column used to order observations and define every bin.
    target : str
        Numeric Y column summarized after the X bins have been defined. Target
        values never influence bin assignment.
    n_bins : int, default 10
        Positive requested number of feature bins. With quantile binning and
        no ties, 10 bins represent successive tenths of the X distribution:
        bin 1 contains the lowest X values and bin 10 the highest.
    binning : {'quantile', 'equal_width'}, default 'quantile'
        How X values define bins. ``quantile`` aims for similar observation
        counts using X ranks. ``equal_width`` divides the numerical range from
        minimum X to maximum X into intervals of equal width; their counts can
        differ substantially.
    group_col : str | None, default None
        Optional column within which bins are assigned independently. Pass the
        time column for cross-sectional analysis; leave None for pooled or
        single-asset temporal analysis.
    time_col : str, default 'time'
        Observation-time key column.
    symbol_col : str, default 'symbol'
        Asset identifier key column.

    Returns
    -------
    FeatureTargetRelationshipResult
        Clean paired observations, aggregate bin statistics, optional
        per-group statistics, and row-accounting metadata. Output DataFrames
        preserve the input backend.

    Raises
    ------
    KeyError
        If a required column is absent.
    TypeError
        If df or a column-name argument has an unsupported type.
    ValueError
        If names, keys, numeric values, binning parameters, or sample size are
        invalid, or no observations can receive a bin.

    Notes
    -----
    Processing flow:

    1. Validate one unique observation per time-symbol pair.
    2. Remove rows where X or Y is missing or non-finite.
    3. Assign bins using X only.
    4. Compute representative X and mean/median Y inside every X bin.
    5. Return the cleaned pairs, summaries, and row-accounting diagnostics.

    When group_col is None, steps 3 and 4 use all supplied observations. This
    is appropriate for exploring one asset through time or a deliberately
    pooled sample. When group_col is ``time``, X bins are rebuilt independently
    inside every date. The function first creates date-bin summaries and then
    averages those summaries across dates with equal date weights. This is the
    cross-sectional mode.

    The main interpretation is an empirical response curve:
    ``representative X in bin -> mean or median Y in that bin``. A rising curve
    suggests a positive monotonic relationship; a falling curve suggests a
    negative one; U-shapes, thresholds, saturation, and asymmetry can motivate
    a separate researcher-defined transformation hypothesis.

    This is a descriptive diagnostic, not a transformation search or model
    selector. Quantile ties receive the same average-rank bin and can leave
    some labels unused. Groups with fewer than n_bins valid pairs do not
    receive bins. When grouped, bin_summary gives every available group equal
    weight for target statistics instead of letting large groups dominate.
    Pooled bins use the full supplied sample and must not be reused as causal
    out-of-sample feature transformations.

    Examples
    --------
    Single-asset or pooled temporal exploration::

        result = feature_target_relationship(
            df,
            feature='signal',
            target='fwd_return_5',
            n_bins=10,
        )

    Cross-sectional exploration, with independent X bins per date::

        result = feature_target_relationship(
            df,
            feature='signal',
            target='fwd_return_5',
            n_bins=10,
            group_col='time',
        )
    """
    for name, value in [
        ('feature', feature),
        ('target', target),
        ('time_col', time_col),
        ('symbol_col', symbol_col),
    ]:
        if not isinstance(value, str) or not value:
            raise TypeError(f'{name} must be a non-empty string.')
    if group_col is not None and (
            not isinstance(group_col, str) or not group_col
    ):
        raise TypeError('group_col must be a non-empty string or None.')
    if feature == target:
        raise ValueError('feature and target must refer to different columns.')
    if time_col == symbol_col:
        raise ValueError('time_col and symbol_col must refer to different columns.')
    if feature in {time_col, symbol_col} or target in {time_col, symbol_col}:
        raise ValueError('feature and target must differ from key columns.')
    if group_col in {feature, target}:
        raise ValueError('group_col must differ from feature and target.')

    _validate_positive_integer(n_bins, 'n_bins')
    if binning not in {'quantile', 'equal_width'}:
        raise ValueError("binning must be 'quantile' or 'equal_width'.")
    if isinstance(df, pd.DataFrame) and not df.columns.is_unique:
        raise ValueError('DataFrame must not contain duplicate column names.')

    required_cols = [time_col, symbol_col, feature, target]
    if group_col is not None and group_col not in required_cols:
        required_cols.append(group_col)
    _validate_df(df, required_cols, check_all_missing=False)
    _validate_unique_keys(df, [time_col, symbol_col], 'df')

    pandas_frame = df.copy() if isinstance(df, pd.DataFrame) else df.to_pandas()
    for key in [time_col, symbol_col] + ([] if group_col is None else [group_col]):
        if pandas_frame[key].isna().any():
            raise ValueError(f'{key} must not contain missing values.')
    for value_col in [feature, target]:
        if (
                not pd.api.types.is_numeric_dtype(pandas_frame[value_col])
                or pd.api.types.is_bool_dtype(pandas_frame[value_col])
                or pd.api.types.is_complex_dtype(pandas_frame[value_col])
        ):
            raise ValueError(f'{value_col} must be real numeric and non-boolean.')

    pair_columns = list(dict.fromkeys([
        time_col,
        symbol_col,
        *([] if group_col is None else [group_col]),
        feature,
        target,
    ]))
    pairs = pandas_frame[pair_columns].copy()
    finite_mask = (
        np.isfinite(pairs[feature].to_numpy(dtype=float, na_value=np.nan))
        & np.isfinite(pairs[target].to_numpy(dtype=float, na_value=np.nan))
    )
    n_input = len(pairs)
    pairs = pairs.loc[finite_mask].copy()
    n_valid = len(pairs)
    n_dropped = n_input - n_valid
    if n_valid < n_bins:
        raise ValueError('at least n_bins finite feature-target pairs are required.')

    pairs, bin_summary, group_summary, n_unassigned = (
        _summarize_relationship_pairs(
            pairs,
            feature,
            target,
            n_bins,
            binning,
            group_col,
        )
    )

    if isinstance(df, pl.DataFrame):
        output_pairs = pl.from_pandas(pairs)
        output_bin_summary = pl.from_pandas(bin_summary)
        output_group_summary = (
            None if group_summary is None else pl.from_pandas(group_summary)
        )
    else:
        output_pairs = pairs
        output_bin_summary = bin_summary
        output_group_summary = group_summary

    return FeatureTargetRelationshipResult(
        pairs=output_pairs,
        bin_summary=output_bin_summary,
        group_summary=output_group_summary,
        feature=feature,
        target=target,
        binning=binning,
        n_bins=n_bins,
        group_col=group_col,
        time_col=time_col,
        symbol_col=symbol_col,
        n_input=n_input,
        n_valid=n_valid,
        n_dropped=n_dropped,
        n_unassigned=n_unassigned,
    )


def temporal_feature_target_relationship_uncertainty(
        df: pd.DataFrame | pl.DataFrame,
        feature: str,
        target: str,
        block_length: int,
        n_bootstraps: int,
        n_bins: int = 10,
        binning: Literal['quantile', 'equal_width'] = 'quantile',
        bootstrap_step: int = 1,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        time_col: str = 'time',
        symbol_col: str = 'symbol',
) -> FeatureTargetRelationshipUncertaintyResult:
    """Estimate pointwise temporal MBB uncertainty for feature-target bins.

    Moving blocks are sampled from the complete ordered feature-target pairs.
    Every bootstrap replicate then rebuilds its feature bins before target
    means and medians are calculated. Resampling never occurs independently
    inside bins, so local temporal dependence and feature-target alignment are
    preserved by the bootstrap design.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Increasing, unique, single-asset temporal observations.
    feature, target : str
        Numeric feature and target columns analyzed jointly.
    block_length : int
        Positive number of consecutive observations in each moving block.
    n_bootstraps : int
        Number of bootstrap replicates, at least two.
    n_bins : int, default 10
        Positive requested number of feature bins.
    binning : {'quantile', 'equal_width'}, default 'quantile'
        Feature-only binning rule rebuilt inside every replicate.
    bootstrap_step : int, default 1
        Positive candidate-block start increment.
    confidence_level : float, default 0.95
        Pointwise percentile confidence level for each bin statistic.
    random_state : int | None, default None
        Optional deterministic block-sampling seed.
    time_col, symbol_col : str
        Temporal observation key columns.

    Returns
    -------
    FeatureTargetRelationshipUncertaintyResult
        Observed relationship plus pointwise bootstrap uncertainty by bin.

    Raises
    ------
    KeyError
        If a required column is absent.
    TypeError
        If df, names, or bootstrap arguments use unsupported types.
    ValueError
        If temporal keys, observations, bins, or bootstrap settings are
        invalid. Every supplied feature-target pair must be finite so that
        removing internal gaps cannot create artificial temporal adjacency.

    Notes
    -----
    The intervals are pointwise by bin, not a simultaneous confidence band or
    a test that the complete response curve differs from a flat relationship.
    Repeated post-selection interpretation still requires out-of-sample
    validation.
    """
    observed = feature_target_relationship(
        df=df,
        feature=feature,
        target=target,
        n_bins=n_bins,
        binning=binning,
        group_col=None,
        time_col=time_col,
        symbol_col=symbol_col,
    )
    _validate_single_symbol(df, symbol_col)
    _validate_time_order(df, time_col)
    if observed.n_dropped:
        raise ValueError(
            'temporal MBB uncertainty requires every feature-target pair to '
            'be finite.',
        )
    if not isinstance(n_bootstraps, (int, np.integer)) or isinstance(
            n_bootstraps,
            bool,
    ):
        raise TypeError('n_bootstraps must be an integer.')
    if n_bootstraps < 2:
        raise ValueError('n_bootstraps must be at least two.')
    if (
            not isinstance(
                confidence_level,
                (int, float, np.integer, np.floating),
            )
            or isinstance(confidence_level, bool)
            or not 0 < confidence_level < 1
    ):
        raise ValueError('confidence_level must be strictly between 0 and 1.')

    pairs = (
        observed.pairs.copy()
        if isinstance(observed.pairs, pd.DataFrame)
        else observed.pairs.to_pandas()
    )
    bootstrap_input = pairs[[feature, target]].copy()
    blocks = generate_moving_blocks(
        bootstrap_input,
        block_length=block_length,
        step=bootstrap_step,
    )
    samples = moving_block_bootstrap(
        blocks,
        sample_size=len(bootstrap_input),
        n_bootstraps=n_bootstraps,
        random_state=random_state,
    )

    bootstrap_summaries = []
    for bootstrap_id, sample in enumerate(samples, start=1):
        try:
            _, summary, _, _ = _summarize_relationship_pairs(
                sample,
                feature,
                target,
                n_bins,
                binning,
                None,
            )
        except ValueError:
            continue
        summary.insert(0, 'bootstrap_id', bootstrap_id)
        bootstrap_summaries.append(summary)

    bootstrap_frame = (
        pd.concat(bootstrap_summaries, ignore_index=True)
        if bootstrap_summaries
        else pd.DataFrame(columns=['bootstrap_id', 'bin'])
    )
    observed_summary = (
        observed.bin_summary
        if isinstance(observed.bin_summary, pd.DataFrame)
        else observed.bin_summary.to_pandas()
    )
    uncertainty_rows = []
    for bin_number in observed_summary['bin'].tolist():
        bin_replicates = bootstrap_frame.loc[
            bootstrap_frame['bin'] == bin_number
        ]
        row: dict[str, float | int | str] = {'bin': int(bin_number)}
        effective_counts = []
        for statistic in ('mean', 'median'):
            values = bin_replicates.get(
                f'target_{statistic}',
                pd.Series(dtype=float),
            )
            finite_values = values[np.isfinite(values.to_numpy(dtype=float))]
            effective_counts.append(len(finite_values))
            prefix = f'target_{statistic}'
            if finite_values.empty:
                row.update({
                    f'{prefix}_bootstrap_mean': np.nan,
                    f'{prefix}_bootstrap_standard_error': np.nan,
                    f'{prefix}_ci_lower': np.nan,
                    f'{prefix}_ci_upper': np.nan,
                })
                continue
            metrics = bootstrap_metrics(finite_values, confidence_level)
            row.update({
                f'{prefix}_bootstrap_mean': metrics.mean,
                f'{prefix}_bootstrap_standard_error': metrics.std,
                f'{prefix}_ci_lower': metrics.ci_lower,
                f'{prefix}_ci_upper': metrics.ci_upper,
            })
        effective = min(effective_counts)
        row['n_bootstraps_effective'] = effective
        row['status'] = 'ok' if effective >= 2 else 'insufficient_bootstraps'
        uncertainty_rows.append(row)

    bin_uncertainty = pd.DataFrame(uncertainty_rows)
    output_uncertainty = (
        pl.from_pandas(bin_uncertainty)
        if isinstance(df, pl.DataFrame)
        else bin_uncertainty
    )
    return FeatureTargetRelationshipUncertaintyResult(
        relationship=observed,
        bin_uncertainty=output_uncertainty,
        block_length=block_length,
        n_bootstraps=n_bootstraps,
        bootstrap_step=bootstrap_step,
        confidence_level=float(confidence_level),
        random_state=random_state,
    )
