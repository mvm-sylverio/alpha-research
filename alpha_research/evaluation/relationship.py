from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from alpha_research._utils import (
    _validate_df,
    _validate_positive_integer,
    _validate_unique_keys,
)

__all__ = [
    'FeatureTargetRelationshipResult',
    'feature_target_relationship',
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
