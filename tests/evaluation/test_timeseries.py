import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.evaluation.timeseries import (
    _select_valid_temporal_pairs,
    _validate_single_symbol,
    temporal_association,
    temporal_association_summary_table,
)
from alpha_research.evaluation.ic import information_coefficient
from alpha_research.evaluation.statistical_tests import wald_temporal_association_test
from alpha_research.resampling.block_bootstrap import (
    BootstrapMetricsResults,
    bootstrap_metrics,
    generate_moving_blocks,
    moving_block_bootstrap,
)


# ------------------------------------------------------
# fixtures
# ------------------------------------------------------
@pytest.fixture
def temporal_df_pandas():
    return pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=5, freq='D'),
        'symbol': ['AAPL'] * 5,
        'feature': [1.0, 2.0, 3.0, 4.0, 5.0],
        'target': [2.0, 4.0, 6.0, 8.0, 10.0],
    })


@pytest.fixture
def temporal_df_polars(temporal_df_pandas):
    return pl.from_pandas(temporal_df_pandas)


@pytest.fixture
def temporal_summary_df_pandas():
    return pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=12, freq='D'),
        'symbol': ['AAPL'] * 12,
        'feature_a': [0.2, -0.4, 0.1, -0.3, 0.6, -0.1, 0.4, -0.5, 0.0, 0.3, -0.2, 0.5],
        'feature_b': [-0.3, 0.2, -0.1, 0.5, -0.4, 0.1, -0.2, 0.4, -0.5, 0.3, 0.0, -0.1],
        'target': [0.4, -0.2, 0.3, -0.1, 0.5, -0.4, 0.2, -0.3, 0.1, 0.6, -0.5, 0.0],
    })


@pytest.fixture
def temporal_summary_df_polars(temporal_summary_df_pandas):
    return pl.from_pandas(temporal_summary_df_pandas)


# ------------------------------------------------------
# _validate_single_symbol
# ------------------------------------------------------
def test_validate_single_symbol_accepts_one_pandas_symbol(temporal_df_pandas):
    """Should accept a pandas DataFrame containing one non-missing symbol."""
    _validate_single_symbol(temporal_df_pandas, 'symbol')


def test_validate_single_symbol_accepts_one_polars_symbol(temporal_df_polars):
    """Should accept a Polars DataFrame containing one non-missing symbol."""
    _validate_single_symbol(temporal_df_polars, 'symbol')


def test_validate_single_symbol_rejects_multiple_pandas_symbols(temporal_df_pandas):
    """Should reject a pandas DataFrame containing multiple symbols."""
    df = temporal_df_pandas.copy()
    df.loc[4, 'symbol'] = 'MSFT'

    with pytest.raises(ValueError, match='exactly one non-missing symbol'):
        _validate_single_symbol(df, 'symbol')


def test_validate_single_symbol_rejects_multiple_polars_symbols(temporal_df_polars):
    """Should reject a Polars DataFrame containing multiple symbols."""
    df = temporal_df_polars.with_columns(
        pl.when(pl.int_range(pl.len()) == 4)
        .then(pl.lit('MSFT'))
        .otherwise(pl.col('symbol'))
        .alias('symbol')
    )

    with pytest.raises(ValueError, match='exactly one non-missing symbol'):
        _validate_single_symbol(df, 'symbol')


def test_validate_single_symbol_rejects_missing_pandas_symbols(temporal_df_pandas):
    """Should reject a pandas DataFrame containing missing symbols."""
    df = temporal_df_pandas.copy()
    df.loc[2, 'symbol'] = None

    with pytest.raises(ValueError, match='exactly one non-missing symbol'):
        _validate_single_symbol(df, 'symbol')


def test_validate_single_symbol_rejects_missing_polars_symbols(temporal_df_polars):
    """Should reject a Polars DataFrame containing missing symbols."""
    df = temporal_df_polars.with_columns(
        pl.when(pl.int_range(pl.len()) == 2)
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.col('symbol'))
        .alias('symbol')
    )

    with pytest.raises(ValueError, match='exactly one non-missing symbol'):
        _validate_single_symbol(df, 'symbol')


# ------------------------------------------------------
# _select_valid_temporal_pairs
# ------------------------------------------------------
def test_select_valid_temporal_pairs_excludes_missing_pandas_values(temporal_df_pandas):
    """Should keep only paired non-missing pandas feature and target values."""
    df = temporal_df_pandas.copy()
    df.loc[1, 'feature'] = np.nan
    df.loc[3, 'target'] = np.nan

    result = _select_valid_temporal_pairs(df, 'feature', 'target')

    assert list(result.columns) == ['feature', 'target']
    assert result.to_dict(orient='list') == {
        'feature': [1.0, 3.0, 5.0],
        'target': [2.0, 6.0, 10.0],
    }


def test_select_valid_temporal_pairs_excludes_missing_polars_values(temporal_df_polars):
    """Should keep only paired non-missing Polars feature and target values."""
    df = temporal_df_polars.with_columns([
        pl.when(pl.int_range(pl.len()) == 1)
        .then(pl.lit(float('nan')))
        .otherwise(pl.col('feature'))
        .alias('feature'),
        pl.when(pl.int_range(pl.len()) == 3)
        .then(pl.lit(float('nan')))
        .otherwise(pl.col('target'))
        .alias('target'),
    ])

    result = _select_valid_temporal_pairs(df, 'feature', 'target')

    assert isinstance(result, pl.DataFrame)
    assert result.to_dict(as_series=False) == {
        'feature': [1.0, 3.0, 5.0],
        'target': [2.0, 6.0, 10.0],
    }

# ------------------------------------------------------
# temporal_association
# ------------------------------------------------------
def test_temporal_association_returns_perfect_spearman_correlation(temporal_df_pandas):
    """Should return a perfect Spearman association for aligned monotonic series."""
    result = temporal_association(temporal_df_pandas, 'feature', 'target')

    assert result == pytest.approx(1.0)


def test_temporal_association_returns_perfect_pearson_correlation(temporal_df_pandas):
    """Should return a perfect Pearson association for linearly related series."""
    result = temporal_association(
        temporal_df_pandas,
        'feature',
        'target',
        corr_method='pearson',
    )

    assert result == pytest.approx(1.0)


def test_temporal_association_drops_missing_aligned_values(temporal_df_pandas):
    """Should calculate the association from aligned non-missing observations."""
    df = temporal_df_pandas.copy()
    df.loc[2, 'target'] = np.nan

    result = temporal_association(df, 'feature', 'target')

    assert result == pytest.approx(1.0)


def test_temporal_association_rejects_duplicate_times(temporal_df_pandas):
    """Should reject duplicate observation times through the temporal validator."""
    df = temporal_df_pandas.copy()
    df.loc[4, 'time'] = df.loc[3, 'time']

    with pytest.raises(ValueError, match='unique'):
        temporal_association(df, 'feature', 'target')


def test_temporal_association_rejects_unordered_times(temporal_df_pandas):
    """Should reject unordered observation times through the temporal validator."""
    df = temporal_df_pandas.iloc[[0, 2, 1, 3, 4]].reset_index(drop=True)

    with pytest.raises(ValueError, match='increasingly ordered'):
        temporal_association(df, 'feature', 'target')


def test_temporal_association_rejects_missing_time(temporal_df_pandas):
    """Should reject missing observation times through the temporal validator."""
    df = temporal_df_pandas.copy()
    df.loc[2, 'time'] = pd.NaT

    with pytest.raises(ValueError, match='time must not contain missing values'):
        temporal_association(df, 'feature', 'target')


def test_temporal_association_returns_nan_for_constant_series(temporal_df_pandas):
    """Should return nan when the feature series is constant."""
    df = temporal_df_pandas.copy()
    df['feature'] = 1.0

    result = temporal_association(df, 'feature', 'target')

    assert np.isnan(result)


def test_temporal_association_pandas_polars_consistency(
        temporal_df_pandas,
        temporal_df_polars,
):
    """Should return equivalent associations for pandas and Polars inputs."""
    pandas_result = temporal_association(temporal_df_pandas, 'feature', 'target')
    polars_result = temporal_association(temporal_df_polars, 'feature', 'target')

    assert pandas_result == pytest.approx(polars_result)


# ------------------------------------------------------
# temporal_association_summary_table
# ------------------------------------------------------
def test_temporal_association_summary_table_single_feature(temporal_summary_df_pandas):
    """Should return one result row with the expected temporal diagnostics."""
    result = temporal_association_summary_table(
        temporal_summary_df_pandas,
        feature_list=['feature_a'],
        target='target',
        block_length=3,
        n_bootstraps=20,
        random_state=42,
    )

    assert len(result) == 1
    assert result['feature'].iloc[0] == 'feature_a'
    assert set(result.columns) == {
        'feature', 'association', 'corr_method', 'n_obs', 'bootstrap_mean',
        'bootstrap_std',
        'bootstrap_pct_positive',
        'test_statistic', 'p_value', 'reject_h0', 'wald_ci_lower',
        'wald_ci_upper',
        'confidence_level', 'alpha', 'n_bootstraps', 'feature_group',
    }


def test_temporal_association_summary_table_multiple_features_and_groups(
        temporal_summary_df_pandas,
):
    """Should preserve feature order and group metadata for every feature."""
    result = temporal_association_summary_table(
        temporal_summary_df_pandas,
        feature_list=['feature_a', 'feature_b'],
        target='target',
        block_length=3,
        n_bootstraps=20,
        random_state=42,
        feature_groups={'feature_a': 'momentum'},
    )

    assert list(result['feature']) == ['feature_a', 'feature_b']
    assert list(result['feature_group']) == ['momentum', 'ungrouped']


@pytest.mark.parametrize('corr_method', ['pearson', 'spearman'])
def test_temporal_association_summary_table_matches_direct_pipeline(
        temporal_summary_df_pandas,
        corr_method,
):
    """Should report the same values produced by the underlying functions."""
    result = temporal_association_summary_table(
        temporal_summary_df_pandas,
        feature_list=['feature_a'],
        target='target',
        block_length=3,
        n_bootstraps=20,
        corr_method=corr_method,
        random_state=42,
    ).iloc[0]
    observed = temporal_association(
        temporal_summary_df_pandas,
        'feature_a',
        'target',
        corr_method=corr_method,
    )
    valid_pairs = _select_valid_temporal_pairs(
        temporal_summary_df_pandas,
        'feature_a',
        'target',
    )
    blocks = generate_moving_blocks(valid_pairs, block_length=3)
    samples = moving_block_bootstrap(
        blocks,
        sample_size=len(valid_pairs),
        n_bootstraps=20,
        random_state=42,
    )
    estimates = [
        information_coefficient(sample['feature_a'], sample['target'], corr_method)
        for sample in samples
    ]
    metrics = bootstrap_metrics(estimates)
    test_result = wald_temporal_association_test(observed, metrics.std)

    assert result['association'] == pytest.approx(observed)
    assert result['bootstrap_mean'] == pytest.approx(metrics.mean)
    assert result['bootstrap_std'] == pytest.approx(metrics.std)
    assert result['p_value'] == pytest.approx(test_result.p_value)
    assert result['test_statistic'] == pytest.approx(test_result.test_statistic)
    assert bool(result['reject_h0']) is test_result.reject_h0
    assert result['wald_ci_lower'] == pytest.approx(test_result.wald_ci_lower)
    assert result['wald_ci_upper'] == pytest.approx(test_result.wald_ci_upper)


def test_temporal_association_summary_table_is_reproducible(temporal_summary_df_pandas):
    """Should return the same table for repeated calls with the same seed."""
    kwargs = {
        'feature_list': ['feature_a', 'feature_b'],
        'target': 'target',
        'block_length': 3,
        'n_bootstraps': 20,
        'random_state': 42,
    }

    first = temporal_association_summary_table(temporal_summary_df_pandas, **kwargs)
    second = temporal_association_summary_table(temporal_summary_df_pandas, **kwargs)

    pd.testing.assert_frame_equal(first, second)


def test_temporal_association_summary_table_drops_invalid_feature_target_pairs(
        temporal_summary_df_pandas,
):
    """Should use the same paired missing-value treatment as temporal_association."""
    df = temporal_summary_df_pandas.copy()
    df.loc[1, 'feature_a'] = np.nan
    df.loc[3, 'target'] = np.nan

    result = temporal_association_summary_table(
        df,
        feature_list=['feature_a'],
        target='target',
        block_length=3,
        n_bootstraps=20,
        random_state=42,
    ).iloc[0]
    expected_association = temporal_association(df, 'feature_a', 'target')

    assert result['n_obs'] == 10
    assert result['association'] == pytest.approx(expected_association)


def test_temporal_association_summary_table_uses_valid_pair_count_as_sample_size(
        monkeypatch,
        temporal_summary_df_pandas,
):
    """Should match each bootstrap sample size to its valid pair count."""
    df = temporal_summary_df_pandas.copy()
    df.loc[1, 'feature_a'] = np.nan
    df.loc[3, 'target'] = np.nan
    observed_sample_sizes = []

    def capture_sample_size(blocks, sample_size, n_bootstraps, random_state):
        observed_sample_sizes.append(sample_size)
        return moving_block_bootstrap(
            blocks,
            sample_size=sample_size,
            n_bootstraps=n_bootstraps,
            random_state=random_state,
        )

    monkeypatch.setattr(
        'alpha_research.evaluation.timeseries.moving_block_bootstrap',
        capture_sample_size,
    )

    temporal_association_summary_table(
        df,
        feature_list=['feature_a', 'feature_b'],
        target='target',
        block_length=3,
        n_bootstraps=20,
        random_state=42,
    )

    assert observed_sample_sizes == [10, 11]


def test_temporal_association_summary_table_uses_wald_decision_not_percentile_ci(
        monkeypatch,
        temporal_summary_df_pandas,
):
    """
    Should propagate the Wald decision despite a percentile CI containing zero.

    The mocked bootstrap metrics use a percentile interval from -0.50 to 0.50,
    while their standard error produces a Wald interval excluding zero for the
    observed association. The percentile interval is intentionally not exposed
    by the summary table; this test verifies that it cannot determine reject_h0.
    """
    metrics = BootstrapMetricsResults(
        mean=0.20,
        std=0.05,
        ci_lower=-0.50,
        ci_upper=0.50,
        pct_positive=0.50,
        n_non_positive=10,
        n_non_negative=10,
        n_bootstraps=20,
    )
    monkeypatch.setattr(
        'alpha_research.evaluation.timeseries.temporal_association',
        lambda **_: 0.20,
    )
    monkeypatch.setattr(
        'alpha_research.evaluation.timeseries.bootstrap_metrics',
        lambda *_args, **_kwargs: metrics,
    )

    result = temporal_association_summary_table(
        temporal_summary_df_pandas,
        feature_list=['feature_a'],
        target='target',
        block_length=3,
        n_bootstraps=20,
        random_state=42,
    ).iloc[0]

    assert not result['wald_ci_lower'] <= 0.0 <= result['wald_ci_upper']
    assert bool(result['reject_h0']) is True


def test_temporal_association_summary_table_does_not_apply_fdr(temporal_summary_df_pandas):
    """Should expose individual p-values without FDR output columns."""
    result = temporal_association_summary_table(
        temporal_summary_df_pandas,
        feature_list=['feature_a', 'feature_b'],
        target='target',
        block_length=3,
        n_bootstraps=20,
        random_state=42,
    )

    assert 'p_value' in result.columns
    assert 'fdr_rejected' not in result.columns
    assert 'fdr_corrected_p_value' not in result.columns


def test_temporal_association_summary_table_pandas_polars_consistency(
        temporal_summary_df_pandas,
        temporal_summary_df_polars,
):
    """Should return equivalent diagnostics for Pandas and Polars inputs."""
    kwargs = {
        'feature_list': ['feature_a', 'feature_b'],
        'target': 'target',
        'block_length': 3,
        'n_bootstraps': 20,
        'random_state': 42,
    }

    pandas_result = temporal_association_summary_table(
        temporal_summary_df_pandas,
        **kwargs,
    )
    polars_result = temporal_association_summary_table(
        temporal_summary_df_polars,
        **kwargs,
    ).to_pandas()

    pd.testing.assert_frame_equal(pandas_result, polars_result, check_dtype=False)


def test_temporal_association_summary_table_rejects_empty_feature_list(
        temporal_summary_df_pandas,
):
    """Should reject an empty feature list before running the pipeline."""
    with pytest.raises(ValueError, match='feature_list must not be empty'):
        temporal_association_summary_table(
            temporal_summary_df_pandas,
            feature_list=[],
            target='target',
            block_length=3,
            n_bootstraps=20,
        )


def test_temporal_association_summary_table_rejects_duplicate_features(
        temporal_summary_df_pandas,
):
    """Should reject duplicate features instead of duplicating pipeline work."""
    with pytest.raises(ValueError, match='feature_list must not contain duplicates'):
        temporal_association_summary_table(
            temporal_summary_df_pandas,
            feature_list=['feature_a', 'feature_a'],
            target='target',
            block_length=3,
            n_bootstraps=20,
        )
