import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.evaluation import (
    FeatureTargetRelationshipResult,
    feature_target_relationship,
)


@pytest.fixture
def relationship_frame_pandas():
    """Create two four-asset cross-sections with an increasing relationship."""
    return pd.DataFrame({
        'time': [1] * 4 + [2] * 4,
        'symbol': ['A', 'B', 'C', 'D'] * 2,
        'feature': [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        'target': [1.0, 4.0, 9.0, 16.0, 2.0, 8.0, 18.0, 32.0],
    })


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_feature_target_relationship_returns_pooled_quantile_summary(
        relationship_frame_pandas,
        backend,
):
    """Should return structured paired data and deterministic pooled bins."""
    frame = (
        relationship_frame_pandas
        if backend == 'pandas'
        else pl.from_pandas(relationship_frame_pandas)
    )
    result = feature_target_relationship(
        frame,
        'feature',
        'target',
        n_bins=4,
    )
    summary = (
        result.bin_summary
        if backend == 'pandas'
        else result.bin_summary.to_pandas()
    )

    assert isinstance(result, FeatureTargetRelationshipResult)
    assert isinstance(result.pairs, type(frame))
    assert result.group_summary is None
    assert result.n_input == result.n_valid == 8
    assert result.n_dropped == result.n_unassigned == 0
    assert summary['bin'].tolist() == [1, 2, 3, 4]
    assert summary['n_obs'].tolist() == [2, 2, 2, 2]
    assert summary['n_groups'].tolist() == [1, 1, 1, 1]


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_feature_target_relationship_groups_cross_sectionally(
        relationship_frame_pandas,
        backend,
):
    """Should assign bins per date and aggregate date-level bin statistics."""
    frame = (
        relationship_frame_pandas
        if backend == 'pandas'
        else pl.from_pandas(relationship_frame_pandas)
    )
    result = feature_target_relationship(
        frame,
        'feature',
        'target',
        n_bins=2,
        group_col='time',
    )
    summary = (
        result.bin_summary
        if backend == 'pandas'
        else result.bin_summary.to_pandas()
    )
    group_summary = (
        result.group_summary
        if backend == 'pandas'
        else result.group_summary.to_pandas()
    )

    assert group_summary is not None
    assert len(group_summary) == 4
    assert summary['n_groups'].tolist() == [2, 2]
    assert summary['n_obs'].tolist() == [4, 4]
    np.testing.assert_allclose(summary['target_mean'], [3.75, 18.75])


def test_grouped_relationship_is_not_changed_by_future_cross_section_scale(
        relationship_frame_pandas,
):
    """Should assign an earlier cross-section without using later feature values."""
    original = feature_target_relationship(
        relationship_frame_pandas,
        'feature',
        'target',
        n_bins=2,
        group_col='time',
    )
    changed_frame = relationship_frame_pandas.copy()
    changed_frame.loc[changed_frame['time'] == 2, 'feature'] *= -10_000
    changed = feature_target_relationship(
        changed_frame,
        'feature',
        'target',
        n_bins=2,
        group_col='time',
    )

    original_bins = original.pairs.loc[original.pairs['time'] == 1, 'bin']
    changed_bins = changed.pairs.loc[changed.pairs['time'] == 1, 'bin']
    pd.testing.assert_series_equal(
        original_bins.reset_index(drop=True),
        changed_bins.reset_index(drop=True),
    )


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_feature_target_relationship_supports_equal_width_bins(
        relationship_frame_pandas,
        backend,
):
    """Should assign the minimum and maximum to the extreme equal-width bins."""
    frame = (
        relationship_frame_pandas
        if backend == 'pandas'
        else pl.from_pandas(relationship_frame_pandas)
    )
    result = feature_target_relationship(
        frame,
        'feature',
        'target',
        n_bins=4,
        binning='equal_width',
    )
    pairs = result.pairs if backend == 'pandas' else result.pairs.to_pandas()

    assert pairs.loc[pairs['feature'].idxmin(), 'bin'] == 1
    assert pairs.loc[pairs['feature'].idxmax(), 'bin'] == 4


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_feature_target_relationship_accounts_for_invalid_pairs(
        relationship_frame_pandas,
        backend,
):
    """Should remove non-finite feature-target pairs and report their count."""
    changed = relationship_frame_pandas.copy()
    changed.loc[0, 'feature'] = np.nan
    changed.loc[1, 'target'] = np.inf
    frame = changed if backend == 'pandas' else pl.from_pandas(changed)
    result = feature_target_relationship(frame, 'feature', 'target', n_bins=2)

    assert result.n_input == 8
    assert result.n_valid == 6
    assert result.n_dropped == 2
    assert len(result.pairs) == 6


def test_feature_target_relationship_reports_groups_too_small_for_bins():
    """Should retain valid scatter pairs while reporting unassigned small groups."""
    frame = pd.DataFrame({
        'time': [1] * 3 + [2] * 4,
        'symbol': ['A', 'B', 'C', 'A', 'B', 'C', 'D'],
        'feature': np.arange(7, dtype=float),
        'target': np.arange(7, dtype=float),
    })
    result = feature_target_relationship(
        frame,
        'feature',
        'target',
        n_bins=4,
        group_col='time',
    )

    assert result.n_valid == 7
    assert result.n_unassigned == 3
    assert result.pairs['bin'].isna().sum() == 3


@pytest.mark.parametrize(
    'kwargs, error_type, message',
    [
        ({'feature': '', 'target': 'target'}, TypeError, 'feature'),
        ({'feature': 'feature', 'target': 'feature'}, ValueError, 'different'),
        ({'feature': 'feature', 'target': 'target', 'n_bins': True}, ValueError, 'positive'),
        ({'feature': 'feature', 'target': 'target', 'binning': 'custom'}, ValueError, 'binning'),
        ({'feature': 'feature', 'target': 'target', 'group_col': 'feature'}, ValueError, 'differ'),
    ],
)
def test_feature_target_relationship_validates_configuration(
        relationship_frame_pandas,
        kwargs,
        error_type,
        message,
):
    """Should reject invalid names, bin counts, methods, and grouping columns."""
    with pytest.raises(error_type, match=message):
        feature_target_relationship(relationship_frame_pandas, **kwargs)


@pytest.mark.parametrize(
    'mutator, error_type, message',
    [
        (lambda frame: frame.drop(columns='target'), KeyError, 'missing required'),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), ValueError, 'unique'),
        (lambda frame: frame.assign(time=np.nan), ValueError, 'time'),
        (lambda frame: frame.assign(feature='bad'), ValueError, 'numeric'),
    ],
)
def test_feature_target_relationship_validates_input_frame(
        relationship_frame_pandas,
        mutator,
        error_type,
        message,
):
    """Should validate schema, unique keys, non-missing keys, and numeric values."""
    with pytest.raises(error_type, match=message):
        feature_target_relationship(
            mutator(relationship_frame_pandas),
            'feature',
            'target',
            n_bins=2,
        )


def test_feature_target_relationship_requires_enough_finite_pairs(
        relationship_frame_pandas,
):
    """Should reject a requested bin count above the finite pair count."""
    with pytest.raises(ValueError, match='at least n_bins'):
        feature_target_relationship(
            relationship_frame_pandas,
            'feature',
            'target',
            n_bins=9,
        )


def test_equal_width_relationship_rejects_constant_features(
        relationship_frame_pandas,
):
    """Should report when equal-width bins cannot be assigned."""
    constant = relationship_frame_pandas.assign(feature=1.0)
    with pytest.raises(ValueError, match='could receive a bin'):
        feature_target_relationship(
            constant,
            'feature',
            'target',
            n_bins=2,
            binning='equal_width',
        )


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_relationship_backend_results_are_numerically_equivalent(
        relationship_frame_pandas,
        backend,
):
    """Should preserve identical grouped summaries and bin assignments by backend."""
    pandas_result = feature_target_relationship(
        relationship_frame_pandas,
        'feature',
        'target',
        n_bins=2,
        group_col='time',
    )
    frame = (
        relationship_frame_pandas
        if backend == 'pandas'
        else pl.from_pandas(relationship_frame_pandas)
    )
    candidate = feature_target_relationship(
        frame,
        'feature',
        'target',
        n_bins=2,
        group_col='time',
    )
    candidate_pairs = (
        candidate.pairs
        if isinstance(candidate.pairs, pd.DataFrame)
        else candidate.pairs.to_pandas()
    )
    candidate_bins = (
        candidate.bin_summary
        if isinstance(candidate.bin_summary, pd.DataFrame)
        else candidate.bin_summary.to_pandas()
    )
    candidate_groups = (
        candidate.group_summary
        if isinstance(candidate.group_summary, pd.DataFrame)
        else candidate.group_summary.to_pandas()
    )

    pd.testing.assert_frame_equal(
        candidate_pairs,
        pandas_result.pairs,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        candidate_bins,
        pandas_result.bin_summary,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        candidate_groups,
        pandas_result.group_summary,
        check_dtype=False,
    )


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_relationship_handles_ties_without_splitting_them(backend):
    """Should assign equal feature values to one average-rank quantile bin."""
    pandas_frame = pd.DataFrame({
        'time': np.arange(6),
        'symbol': ['A'] * 6,
        'feature': [1.0, 1.0, 1.0, 2.0, 2.0, 3.0],
        'target': np.arange(6, dtype=float),
    })
    frame = pandas_frame if backend == 'pandas' else pl.from_pandas(pandas_frame)
    result = feature_target_relationship(frame, 'feature', 'target', n_bins=3)
    pairs = result.pairs if backend == 'pandas' else result.pairs.to_pandas()

    assert pairs.groupby('feature')['bin'].nunique().max() == 1


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_relationship_supports_one_bin_boundary(relationship_frame_pandas, backend):
    """Should summarize all finite observations when exactly one bin is requested."""
    frame = (
        relationship_frame_pandas
        if backend == 'pandas'
        else pl.from_pandas(relationship_frame_pandas)
    )
    result = feature_target_relationship(frame, 'feature', 'target', n_bins=1)
    summary = (
        result.bin_summary
        if backend == 'pandas'
        else result.bin_summary.to_pandas()
    )

    assert summary['bin'].tolist() == [1]
    assert summary.loc[0, 'n_obs'] == len(relationship_frame_pandas)
    assert summary.loc[0, 'target_mean'] == pytest.approx(
        relationship_frame_pandas['target'].mean(),
    )


def test_grouped_relationship_weights_unequal_groups_equally():
    """Should not let a larger cross-section dominate the aggregate target mean."""
    frame = pd.DataFrame({
        'time': [1, 1, 2, 2, 2, 2],
        'symbol': ['A', 'B', 'A', 'B', 'C', 'D'],
        'feature': [1.0, 2.0, 1.0, 2.0, 3.0, 4.0],
        'target': [0.0, 0.0, 10.0, 10.0, 20.0, 20.0],
    })
    result = feature_target_relationship(
        frame,
        'feature',
        'target',
        n_bins=2,
        group_col='time',
    )

    assert result.bin_summary.loc[0, 'target_mean'] == pytest.approx(5.0)
    assert result.bin_summary.loc[0, 'n_groups'] == 2
    assert result.bin_summary.loc[0, 'n_obs'] == 3


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_relationship_preserves_valid_pair_order_and_nullable_values(
        relationship_frame_pandas,
        backend,
):
    """Should accept nullable numeric data and preserve surviving observation order."""
    nullable = relationship_frame_pandas.iloc[[4, 0, 5, 1]].copy()
    nullable['feature'] = nullable['feature'].astype('Float64')
    nullable.loc[0, 'feature'] = pd.NA
    frame = nullable if backend == 'pandas' else pl.from_pandas(nullable)
    result = feature_target_relationship(frame, 'feature', 'target', n_bins=2)
    pairs = result.pairs if backend == 'pandas' else result.pairs.to_pandas()

    assert pairs['time'].tolist() == [2, 2, 1]
    assert pairs['symbol'].tolist() == ['A', 'B', 'B']
    assert result.n_dropped == 1


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_relationship_reports_unassigned_small_groups_in_both_backends(backend):
    """Should preserve nullable bin diagnostics across Pandas and Polars."""
    pandas_frame = pd.DataFrame({
        'time': [1] * 2 + [2] * 3,
        'symbol': ['A', 'B', 'A', 'B', 'C'],
        'feature': np.arange(5, dtype=float),
        'target': np.arange(5, dtype=float),
    })
    frame = pandas_frame if backend == 'pandas' else pl.from_pandas(pandas_frame)
    result = feature_target_relationship(
        frame,
        'feature',
        'target',
        n_bins=3,
        group_col='time',
    )
    pairs = result.pairs if backend == 'pandas' else result.pairs.to_pandas()

    assert result.n_unassigned == 2
    assert pairs['bin'].isna().sum() == 2


def test_relationship_summary_is_invariant_to_input_row_permutation(
        relationship_frame_pandas,
):
    """Should not change aggregate bins when observation rows are reordered."""
    original = feature_target_relationship(
        relationship_frame_pandas,
        'feature',
        'target',
        n_bins=2,
        group_col='time',
    )
    shuffled = feature_target_relationship(
        relationship_frame_pandas.sample(frac=1, random_state=12),
        'feature',
        'target',
        n_bins=2,
        group_col='time',
    )
    pd.testing.assert_frame_equal(original.bin_summary, shuffled.bin_summary)


@pytest.mark.parametrize(
    'mutator, message',
    [
        (lambda frame: frame.assign(feature=True), 'real numeric'),
        (lambda frame: frame.assign(feature=1 + 2j), 'real numeric'),
        (
            lambda frame: pd.concat([frame, frame['feature']], axis=1),
            'duplicate column',
        ),
    ],
)
def test_relationship_rejects_ambiguous_or_non_real_columns(
        relationship_frame_pandas,
        mutator,
        message,
):
    """Should reject values outside the unambiguous real-valued frame contract."""
    with pytest.raises(ValueError, match=message):
        feature_target_relationship(
            mutator(relationship_frame_pandas),
            'feature',
            'target',
            n_bins=2,
        )


@pytest.mark.parametrize(
    'kwargs, message',
    [
        ({'time_col': 'symbol'}, 'different'),
        ({'feature': 'time', 'target': 'target'}, 'key columns'),
        ({'feature': 'feature', 'target': 'symbol'}, 'key columns'),
    ],
)
def test_relationship_rejects_key_columns_as_analyzed_values(
        relationship_frame_pandas,
        kwargs,
        message,
):
    """Should keep observation identity separate from analyzed numeric values."""
    defaults = {'feature': 'feature', 'target': 'target'}
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        feature_target_relationship(
            relationship_frame_pandas,
            n_bins=2,
            **defaults,
        )


def test_relationship_validates_duplicate_polars_observation_keys(
        relationship_frame_pandas,
):
    """Should reject duplicate time-symbol observations natively in Polars."""
    duplicate = pd.concat([
        relationship_frame_pandas,
        relationship_frame_pandas.iloc[[0]],
    ])
    with pytest.raises(ValueError, match='unique'):
        feature_target_relationship(
            pl.from_pandas(duplicate),
            'feature',
            'target',
            n_bins=2,
        )


def test_relationship_validates_boolean_polars_values(
        relationship_frame_pandas,
):
    """Should reject Boolean analyzed values consistently in Polars."""
    frame = pl.from_pandas(relationship_frame_pandas).with_columns(
        pl.lit(True).alias('feature'),
    )
    with pytest.raises(ValueError, match='real numeric'):
        feature_target_relationship(frame, 'feature', 'target', n_bins=2)
