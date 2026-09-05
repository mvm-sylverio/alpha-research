import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.evaluation.timeseries import (
    _rolling_random_states,
    _rolling_summary_pandas,
    _select_valid_temporal_pairs,
    _slice_rows,
    _temporal_association_decay_from_target_frames,
    _validate_single_symbol,
    RollingTemporalAssociationResult,
    TemporalAssociationDecayResult,
    TemporalAssociationDecaySummaryTableResult,
    rolling_temporal_association,
    summarize_rolling_temporal_association,
    temporal_association,
    temporal_association_decay,
    temporal_association_decay_summary,
    temporal_association_decay_summary_table,
    temporal_association_summary_table,
)
from alpha_research.evaluation.ic import _generate_target_frames, information_coefficient
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


@pytest.fixture
def temporal_decay_df_pandas():
    rng = np.random.default_rng(42)
    target_seed = rng.normal(size=30)
    return pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=30, freq='D'),
        'symbol': ['AAPL'] * 30,
        'feature_a': target_seed + rng.normal(scale=0.4, size=30),
        'feature_b': -target_seed + rng.normal(scale=0.5, size=30),
        'target_seed': target_seed,
    })


@pytest.fixture
def temporal_decay_df_polars(temporal_decay_df_pandas):
    return pl.from_pandas(temporal_decay_df_pandas)


@pytest.fixture
def rolling_temporal_df_pandas():
    return pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=12, freq='D'),
        'symbol': ['AAPL'] * 12,
        'feature': [
            0.2, -0.4, 0.1, -0.3, 0.6, -0.1,
            0.4, -0.5, 0.0, 0.3, -0.2, 0.5,
        ],
        'target': [
            0.4, -0.2, 0.3, -0.1, 0.5, -0.4,
            0.2, -0.3, 0.1, 0.6, -0.5, 0.0,
        ],
    })


@pytest.fixture
def rolling_temporal_df_polars(rolling_temporal_df_pandas):
    return pl.from_pandas(rolling_temporal_df_pandas)


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
# _slice_rows
# ------------------------------------------------------
def test_slice_rows_preserves_pandas_backend_order_and_bounds(
        rolling_temporal_df_pandas,
):
    """Should select the requested half-open Pandas positional interval."""
    result = _slice_rows(rolling_temporal_df_pandas, start=2, stop=5)

    assert isinstance(result, pd.DataFrame)
    assert result.index.to_list() == [2, 3, 4]
    assert result['feature'].to_list() == [0.1, -0.3, 0.6]


def test_slice_rows_preserves_polars_backend_order_and_bounds(
        rolling_temporal_df_polars,
):
    """Should select the requested half-open Polars positional interval."""
    result = _slice_rows(rolling_temporal_df_polars, start=2, stop=5)

    assert isinstance(result, pl.DataFrame)
    assert result['feature'].to_list() == [0.1, -0.3, 0.6]

# ------------------------------------------------------
# _rolling_random_states
# ------------------------------------------------------
def test_rolling_random_states_returns_none_for_each_unseeded_window():
    """Should preserve non-deterministic bootstrap behavior without a root seed."""
    assert _rolling_random_states(n_windows=3, random_state=None) == [None, None, None]


def test_rolling_random_states_is_deterministic_and_distinct_per_window():
    """Should derive reproducible child seeds instead of reusing one seed."""
    first = _rolling_random_states(n_windows=3, random_state=42)
    second = _rolling_random_states(n_windows=3, random_state=42)

    assert first == second
    assert len(set(first)) == 3


@pytest.mark.parametrize('random_state', ['42', 1.5, True])
def test_rolling_random_states_rejects_invalid_root_seed(random_state):
    """Should reject values that NumPy could otherwise coerce ambiguously."""
    with pytest.raises(TypeError, match='random_state'):
        _rolling_random_states(n_windows=3, random_state=random_state)


# ------------------------------------------------------
# _rolling_summary_pandas
# ------------------------------------------------------
def test_rolling_summary_pandas_excludes_invalid_rows_from_metrics():
    """Should retain invalid-window counts but exclude them from numeric summaries."""
    frame = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL', 'AAPL'],
        'feature': ['feature'] * 3,
        'target': ['target'] * 3,
        'corr_method': ['spearman'] * 3,
        'bootstrap_method': ['moving_block'] * 3,
        'window_end': pd.date_range('2024-01-03', periods=3, freq='D'),
        'association': [0.2, -0.1, np.nan],
        'bootstrap_ci_lower': [0.1, -0.2, np.nan],
        'bootstrap_ci_upper': [0.3, -0.05, np.nan],
        'bootstrap_pct_positive': [0.95, 0.05, np.nan],
        'status': ['ok', 'ok', 'missing_pairs'],
    })

    summary = _rolling_summary_pandas(frame).iloc[0]

    assert summary['n_windows'] == 3
    assert summary['n_valid_windows'] == 2
    assert summary['n_invalid_windows'] == 1
    assert summary['association_mean'] == pytest.approx(0.05)
    assert summary['ci_pct_strictly_positive'] == pytest.approx(0.5)
    assert summary['ci_pct_strictly_negative'] == pytest.approx(0.5)
    assert summary['ci_pct_contains_zero'] == pytest.approx(0.0)
    assert summary['mean_bootstrap_pct_positive'] == pytest.approx(0.5)


# ------------------------------------------------------
# rolling_temporal_association
# ------------------------------------------------------
def test_rolling_temporal_association_returns_requested_windows_and_metrics(
        rolling_temporal_df_pandas,
):
    """Should calculate bootstrap diagnostics for every requested full window."""
    result = rolling_temporal_association(
        rolling_temporal_df_pandas,
        feature='feature',
        target='target',
        window_size=6,
        window_step=3,
        block_length=3,
        n_bootstraps=20,
        random_state=42,
    )

    frame = result.rolling_frame
    assert isinstance(result, RollingTemporalAssociationResult)
    assert list(frame['window_end']) == list(
        rolling_temporal_df_pandas['time'].iloc[[5, 8, 11]],
    )
    assert list(frame['window_start']) == list(
        rolling_temporal_df_pandas['time'].iloc[[0, 3, 6]],
    )
    assert set(frame['bootstrap_method']) == {'moving_block'}
    assert set(frame['status']) == {'ok'}
    assert frame['n_obs'].to_list() == [6, 6, 6]
    assert (frame['n_bootstraps'] == 20).all()
    assert np.isfinite(frame['association']).all()
    assert np.isfinite(frame['bootstrap_ci_lower']).all()
    assert np.isfinite(frame['bootstrap_ci_upper']).all()
    assert (frame['bootstrap_ci_lower'] <= frame['bootstrap_ci_upper']).all()


def test_rolling_temporal_association_matches_direct_window_estimate(
        rolling_temporal_df_pandas,
):
    """Should use the same association estimator as a direct temporal window."""
    result = rolling_temporal_association(
        rolling_temporal_df_pandas,
        feature='feature',
        target='target',
        window_size=6,
        block_length=3,
        n_bootstraps=20,
        random_state=42,
    )
    expected = temporal_association(
        rolling_temporal_df_pandas.iloc[:6],
        feature='feature',
        target='target',
    )

    assert result.rolling_frame['association'].iloc[0] == pytest.approx(expected)


@pytest.mark.parametrize('corr_method', ['pearson', 'spearman'])
def test_rolling_temporal_association_supports_established_correlation_methods(
        rolling_temporal_df_pandas,
        corr_method,
):
    """Should apply the requested estimator consistently to each window."""
    result = rolling_temporal_association(
        rolling_temporal_df_pandas,
        feature='feature',
        target='target',
        window_size=6,
        block_length=3,
        n_bootstraps=20,
        corr_method=corr_method,
        random_state=42,
    )
    expected = temporal_association(
        rolling_temporal_df_pandas.iloc[:6],
        feature='feature',
        target='target',
        corr_method=corr_method,
    )

    assert result.rolling_frame['association'].iloc[0] == pytest.approx(expected)
    assert set(result.rolling_frame['corr_method']) == {corr_method}


def test_rolling_temporal_association_marks_missing_pairs_without_compressing_time(
        rolling_temporal_df_pandas,
):
    """Should not bootstrap a window after a missing internal pair is dropped."""
    df = rolling_temporal_df_pandas.copy()
    df.loc[4, 'target'] = np.nan

    result = rolling_temporal_association(
        df,
        feature='feature',
        target='target',
        window_size=4,
        block_length=2,
        n_bootstraps=20,
        random_state=42,
    )
    frame = result.rolling_frame
    missing_rows = frame[frame['status'] == 'missing_pairs']

    assert len(missing_rows) == 4
    assert set(missing_rows['n_obs']) == {3}
    assert missing_rows['association'].isna().all()
    assert (frame['status'] == 'ok').any()


def test_rolling_temporal_association_is_reproducible_with_root_seed(
        rolling_temporal_df_pandas,
):
    """Should derive deterministic bootstrap samples for every window."""
    kwargs = {
        'feature': 'feature',
        'target': 'target',
        'window_size': 6,
        'block_length': 3,
        'n_bootstraps': 20,
        'random_state': 42,
    }

    first = rolling_temporal_association(rolling_temporal_df_pandas, **kwargs)
    second = rolling_temporal_association(rolling_temporal_df_pandas, **kwargs)

    pd.testing.assert_frame_equal(first.rolling_frame, second.rolling_frame)
    pd.testing.assert_frame_equal(first.summary_table, second.summary_table)


def test_rolling_temporal_association_passes_bootstrap_step_to_mbb(
        monkeypatch,
        rolling_temporal_df_pandas,
):
    """Should keep rolling-window cadence distinct from MBB candidate cadence."""
    observed_steps = []
    original = generate_moving_blocks

    def capture_bootstrap_step(data, block_length, step):
        observed_steps.append(step)
        return original(data, block_length=block_length, step=step)

    monkeypatch.setattr(
        'alpha_research.evaluation.timeseries.generate_moving_blocks',
        capture_bootstrap_step,
    )
    result = rolling_temporal_association(
        rolling_temporal_df_pandas,
        feature='feature',
        target='target',
        window_size=6,
        window_step=3,
        block_length=3,
        bootstrap_step=2,
        n_bootstraps=20,
        random_state=42,
    )

    assert observed_steps == [2] * len(result.rolling_frame)


def test_rolling_temporal_association_supports_custom_key_columns(
        rolling_temporal_df_pandas,
):
    """Should not require the default time and symbol column names."""
    df = rolling_temporal_df_pandas.rename(
        columns={'time': 'timestamp', 'symbol': 'asset'},
    )

    result = rolling_temporal_association(
        df,
        feature='feature',
        target='target',
        window_size=6,
        block_length=3,
        n_bootstraps=20,
        time_col='timestamp',
        symbol_col='asset',
        random_state=42,
    )

    assert set(result.rolling_frame['symbol']) == {'AAPL'}


def test_rolling_temporal_association_pandas_polars_consistency(
        rolling_temporal_df_pandas,
        rolling_temporal_df_polars,
):
    """Should preserve rolling diagnostics across supported DataFrame backends."""
    kwargs = {
        'feature': 'feature',
        'target': 'target',
        'window_size': 6,
        'block_length': 3,
        'n_bootstraps': 20,
        'random_state': 42,
    }
    pandas_result = rolling_temporal_association(rolling_temporal_df_pandas, **kwargs)
    polars_result = rolling_temporal_association(rolling_temporal_df_polars, **kwargs)

    pd.testing.assert_frame_equal(
        pandas_result.rolling_frame,
        polars_result.rolling_frame.to_pandas(),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        pandas_result.summary_table,
        polars_result.summary_table.to_pandas(),
        check_dtype=False,
    )


@pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            ({'bootstrap_method': 'iid'}, 'bootstrap_method'),
            ({'block_length': 6}, 'block_length'),
            ({'n_bootstraps': 1}, 'n_bootstraps'),
        ],
)
def test_rolling_temporal_association_rejects_invalid_bootstrap_configuration(
        rolling_temporal_df_pandas,
        kwargs,
        message,
):
    """Should require MBB with enough independent uncertainty diagnostics."""
    base_kwargs = {
        'feature': 'feature',
        'target': 'target',
        'window_size': 6,
        'block_length': 3,
        'n_bootstraps': 20,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        rolling_temporal_association(rolling_temporal_df_pandas, **base_kwargs)


@pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            ({'window_size': 0}, 'window_size'),
            ({'window_step': 0}, 'window_step'),
            ({'block_length': 0}, 'block_length'),
            ({'bootstrap_step': 0}, 'bootstrap_step'),
            ({'n_bootstraps': 0}, 'n_bootstraps'),
        ],
)
def test_rolling_temporal_association_rejects_non_positive_sizing_arguments(
        rolling_temporal_df_pandas,
        kwargs,
        message,
):
    """Should validate every rolling and MBB sizing argument before execution."""
    base_kwargs = {
        'feature': 'feature',
        'target': 'target',
        'window_size': 6,
        'block_length': 3,
        'n_bootstraps': 20,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        rolling_temporal_association(rolling_temporal_df_pandas, **base_kwargs)


def test_rolling_temporal_association_rejects_oversized_window(
        rolling_temporal_df_pandas,
):
    """Should reject a window that cannot be formed from the input history."""
    with pytest.raises(ValueError, match='window_size'):
        rolling_temporal_association(
            rolling_temporal_df_pandas,
            feature='feature',
            target='target',
            window_size=13,
            block_length=3,
            n_bootstraps=20,
        )


def test_rolling_temporal_association_rejects_non_dataframe_input():
    """Should use the shared DataFrame type validation at the public boundary."""
    with pytest.raises(TypeError, match='DataFrame'):
        rolling_temporal_association(
            [{'time': '2024-01-01'}],
            feature='feature',
            target='target',
            window_size=6,
            block_length=3,
            n_bootstraps=20,
        )


@pytest.mark.parametrize('random_state', ['42', 1.5, True])
def test_rolling_temporal_association_rejects_invalid_root_seed(
        rolling_temporal_df_pandas,
        random_state,
):
    """Should reject a root seed before attempting window-level sampling."""
    with pytest.raises(TypeError, match='random_state'):
        rolling_temporal_association(
            rolling_temporal_df_pandas,
            feature='feature',
            target='target',
            window_size=6,
            block_length=3,
            n_bootstraps=20,
            random_state=random_state,
        )


@pytest.mark.parametrize('confidence_level', [0.0, 1.0])
def test_rolling_temporal_association_propagates_invalid_confidence_level(
        rolling_temporal_df_pandas,
        confidence_level,
):
    """Should not silently accept an invalid percentile-bootstrap interval level."""
    with pytest.raises(ValueError, match='confidence_level'):
        rolling_temporal_association(
            rolling_temporal_df_pandas,
            feature='feature',
            target='target',
            window_size=6,
            block_length=3,
            n_bootstraps=20,
            confidence_level=confidence_level,
        )


def test_rolling_temporal_association_propagates_invalid_correlation_method(
        rolling_temporal_df_pandas,
):
    """Should use the established temporal-association estimator validation."""
    with pytest.raises(ValueError, match='corr_method'):
        rolling_temporal_association(
            rolling_temporal_df_pandas,
            feature='feature',
            target='target',
            window_size=6,
            block_length=3,
            n_bootstraps=20,
            corr_method='kendall',
        )


def test_rolling_temporal_association_rejects_invalid_temporal_contract(
        rolling_temporal_df_pandas,
):
    """Should enforce the same single-asset, ordered, unique-time contract."""
    df = rolling_temporal_df_pandas.copy()
    df.loc[1, 'time'] = df.loc[0, 'time']

    with pytest.raises(ValueError, match='unique'):
        rolling_temporal_association(
            df,
            feature='feature',
            target='target',
            window_size=6,
            block_length=3,
            n_bootstraps=20,
        )


def test_rolling_temporal_association_marks_undefined_observed_association(
        rolling_temporal_df_pandas,
):
    """Should expose constant-window correlations as a non-computable status."""
    df = rolling_temporal_df_pandas.copy()
    df['feature'] = 1.0

    result = rolling_temporal_association(
        df,
        feature='feature',
        target='target',
        window_size=6,
        block_length=3,
        n_bootstraps=20,
    )

    assert set(result.rolling_frame['status']) == {'undefined_association'}
    assert result.rolling_frame['association'].isna().all()


def test_rolling_temporal_association_marks_undefined_bootstrap(
        monkeypatch,
        rolling_temporal_df_pandas,
):
    """Should preserve observed estimates when bootstrap metrics are undefined."""
    monkeypatch.setattr(
        'alpha_research.evaluation.timeseries.bootstrap_metrics',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError('no finite bootstrap estimates'),
        ),
    )

    result = rolling_temporal_association(
        rolling_temporal_df_pandas,
        feature='feature',
        target='target',
        window_size=6,
        block_length=3,
        n_bootstraps=20,
    )

    assert set(result.rolling_frame['status']) == {'undefined_bootstrap'}
    assert np.isfinite(result.rolling_frame['association']).all()
    assert (result.rolling_frame['n_bootstraps'] == 0).all()


def test_summarize_rolling_temporal_association_reports_descriptive_counts(
        rolling_temporal_df_pandas,
):
    """Should summarize valid and invalid rolling windows separately."""
    df = rolling_temporal_df_pandas.copy()
    df.loc[4, 'target'] = np.nan
    result = rolling_temporal_association(
        df,
        feature='feature',
        target='target',
        window_size=4,
        block_length=2,
        n_bootstraps=20,
        random_state=42,
    )

    summary = summarize_rolling_temporal_association(result.rolling_frame).iloc[0]
    assert summary['n_windows'] == len(result.rolling_frame)
    assert summary['n_valid_windows'] == (result.rolling_frame['status'] == 'ok').sum()
    assert summary['n_invalid_windows'] == (result.rolling_frame['status'] != 'ok').sum()


def test_rolling_temporal_association_summarizes_all_invalid_windows(
        rolling_temporal_df_pandas,
):
    """Should return a descriptive summary even when no window is usable."""
    df = rolling_temporal_df_pandas.copy()
    df.loc[[0, 3, 6, 9], 'target'] = np.nan

    result = rolling_temporal_association(
        df,
        feature='feature',
        target='target',
        window_size=4,
        block_length=2,
        n_bootstraps=20,
        random_state=42,
    )
    summary = result.summary_table.iloc[0]

    assert summary['n_valid_windows'] == 0
    assert summary['n_invalid_windows'] == summary['n_windows']
    assert np.isnan(summary['association_mean'])


# ------------------------------------------------------
# summarize_rolling_temporal_association
# ------------------------------------------------------
def test_summarize_rolling_temporal_association_preserves_polars_backend(
        rolling_temporal_df_pandas,
):
    """Should return Polars summaries when the rolling frame is Polars."""
    result = rolling_temporal_association(
        pl.from_pandas(rolling_temporal_df_pandas),
        feature='feature',
        target='target',
        window_size=6,
        block_length=3,
        n_bootstraps=20,
        random_state=42,
    )

    summary = summarize_rolling_temporal_association(result.rolling_frame)
    assert isinstance(summary, pl.DataFrame)
    assert summary.height == 1


@pytest.mark.parametrize(
        ('rolling_frame', 'error_type', 'message'),
        [
            ([{'association': 0.1}], TypeError, 'DataFrame'),
            (pd.DataFrame(), ValueError, 'must not be empty'),
            (pd.DataFrame({'association': [0.1]}), KeyError, 'missing required columns'),
        ],
)
def test_summarize_rolling_temporal_association_validates_input_schema(
        rolling_frame,
        error_type,
        message,
):
    """Should use the shared DataFrame validator and require its result schema."""
    with pytest.raises(error_type, match=message):
        summarize_rolling_temporal_association(rolling_frame)


def test_summarize_rolling_temporal_association_allows_all_missing_diagnostics():
    """Should summarize an explicit all-invalid rolling result instead of rejecting it."""
    rolling_frame = pd.DataFrame({
        'symbol': ['AAPL'],
        'feature': ['feature'],
        'target': ['target'],
        'corr_method': ['spearman'],
        'bootstrap_method': ['moving_block'],
        'window_end': [pd.Timestamp('2024-01-01')],
        'association': [np.nan],
        'bootstrap_ci_lower': [np.nan],
        'bootstrap_ci_upper': [np.nan],
        'bootstrap_pct_positive': [np.nan],
        'status': ['missing_pairs'],
    })

    summary = summarize_rolling_temporal_association(rolling_frame).iloc[0]
    assert summary['n_valid_windows'] == 0
    assert np.isnan(summary['association_mean'])


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


# ------------------------------------------------------
# _temporal_association_decay_from_target_frames
# ------------------------------------------------------
def test_temporal_association_decay_from_target_frames_reuses_pre_generated_targets(
        temporal_decay_df_pandas,
):
    """Should calculate sorted decay results without regenerating targets."""
    calls = []

    def target_fn(target_data, horizon):
        calls.append(horizon)
        return target_data[['time', 'symbol']].assign(**{
            f'target_{horizon}': target_data['target_seed'].shift(-horizon),
        })

    target_frames = _generate_target_frames(
        df_feature=temporal_decay_df_pandas,
        target_data=temporal_decay_df_pandas,
        horizons=[3, 1],
        target_fn=target_fn,
    )
    result = _temporal_association_decay_from_target_frames(
        df_feature=temporal_decay_df_pandas,
        feature='feature_a',
        target_frames=target_frames,
        block_length=3,
        n_bootstraps=20,
        corr_method='spearman',
        step=1,
        confidence_level=0.95,
        random_state=42,
        time_col='time',
        symbol_col='symbol',
        feature_groups={'feature_a': 'momentum'},
        fdr=0.05,
        fdr_method='bh',
    )

    assert calls == [1, 3]
    assert result.table['horizon'].to_list() == [1, 3]
    assert result.table['feature_group'].to_list() == ['momentum', 'momentum']


def test_temporal_association_decay_from_target_frames_pandas_polars_consistency(
        temporal_decay_df_pandas,
        temporal_decay_df_polars,
):
    """Should preserve pre-generated-target decay results across backends."""
    def pandas_target_fn(target_data, horizon):
        return target_data[['time', 'symbol']].assign(**{
            f'target_{horizon}': target_data['target_seed'].shift(-horizon),
        })

    def polars_target_fn(target_data, horizon):
        return target_data.select(
            'time',
            'symbol',
            pl.col('target_seed').shift(-horizon).alias(f'target_{horizon}'),
        )

    pandas_target_frames = _generate_target_frames(
        df_feature=temporal_decay_df_pandas,
        target_data=temporal_decay_df_pandas,
        horizons=[1, 3],
        target_fn=pandas_target_fn,
    )
    polars_target_frames = _generate_target_frames(
        df_feature=temporal_decay_df_polars,
        target_data=temporal_decay_df_polars,
        horizons=[1, 3],
        target_fn=polars_target_fn,
    )
    kwargs = {
        'feature': 'feature_a',
        'block_length': 3,
        'n_bootstraps': 20,
        'corr_method': 'spearman',
        'step': 1,
        'confidence_level': 0.95,
        'random_state': 42,
        'time_col': 'time',
        'symbol_col': 'symbol',
        'feature_groups': None,
        'fdr': 0.05,
        'fdr_method': 'bh',
    }
    pandas_result = _temporal_association_decay_from_target_frames(
        df_feature=temporal_decay_df_pandas,
        target_frames=pandas_target_frames,
        **kwargs,
    ).table
    polars_result = _temporal_association_decay_from_target_frames(
        df_feature=temporal_decay_df_polars,
        target_frames=polars_target_frames,
        **kwargs,
    ).table.to_pandas()

    pd.testing.assert_frame_equal(pandas_result, polars_result, check_dtype=False)


# ------------------------------------------------------
# temporal_association_decay
# ------------------------------------------------------
def test_temporal_association_decay_returns_horizon_level_mbb_wald_and_fdr_results(
        temporal_decay_df_pandas,
):
    """Should calculate one complete temporal result per sorted horizon."""
    calls = []

    def target_fn(target_data, horizon):
        calls.append(horizon)
        return target_data[['time', 'symbol']].assign(**{
            f'target_{horizon}': target_data['target_seed'].shift(-horizon),
        })

    result = temporal_association_decay(
        df_feature=temporal_decay_df_pandas,
        feature='feature_a',
        target_data=temporal_decay_df_pandas,
        horizons=[3, 1],
        target_fn=target_fn,
        block_length=3,
        n_bootstraps=20,
        random_state=42,
        feature_groups={'feature_a': 'momentum'},
    )

    assert isinstance(result, TemporalAssociationDecayResult)
    assert calls == [1, 3]
    assert result.table['horizon'].to_list() == [1, 3]
    assert result.table['target'].to_list() == ['target_1', 'target_3']
    assert result.table['n_obs'].to_list() == [29, 27]
    assert result.table['feature_group'].to_list() == ['momentum', 'momentum']
    assert {
        'association', 'bootstrap_mean', 'bootstrap_std',
        'bootstrap_pct_positive', 'test_statistic', 'p_value', 'reject_h0',
        'wald_ci_lower', 'wald_ci_upper', 'fdr_rejected',
        'fdr_corrected_p_value',
    } <= set(result.table.columns)


def test_temporal_association_decay_pandas_polars_consistency(
        temporal_decay_df_pandas,
        temporal_decay_df_polars,
):
    """Should preserve temporal-decay results across DataFrame backends."""
    def pandas_target_fn(target_data, horizon):
        return target_data[['time', 'symbol']].assign(**{
            f'target_{horizon}': target_data['target_seed'].shift(-horizon),
        })

    def polars_target_fn(target_data, horizon):
        return target_data.select(
            'time',
            'symbol',
            pl.col('target_seed').shift(-horizon).alias(f'target_{horizon}'),
        )

    kwargs = {
        'feature': 'feature_a',
        'horizons': [1, 3],
        'block_length': 3,
        'n_bootstraps': 20,
        'random_state': 42,
    }
    pandas_result = temporal_association_decay(
        df_feature=temporal_decay_df_pandas,
        target_data=temporal_decay_df_pandas,
        target_fn=pandas_target_fn,
        **kwargs,
    ).table
    polars_result = temporal_association_decay(
        df_feature=temporal_decay_df_polars,
        target_data=temporal_decay_df_polars,
        target_fn=polars_target_fn,
        **kwargs,
    ).table.to_pandas()

    pd.testing.assert_frame_equal(pandas_result, polars_result, check_dtype=False)


# ------------------------------------------------------
# temporal_association_decay_summary
# ------------------------------------------------------
def test_temporal_association_decay_summary_calculates_known_curve_diagnostics():
    """Should derive peak, half-life, significance persistence, and area."""
    decay_curve = pd.DataFrame({
        'horizon': [20, 5, 10, 30],
        'association': [0.1, -0.5, 0.3, 0.2],
        'fdr_rejected': [False, True, False, True],
    })

    result = temporal_association_decay_summary(decay_curve)

    assert result.peak_horizon == 5
    assert result.peak_abs_association == pytest.approx(0.5)
    assert result.halflife_horizon == 20
    assert result.last_significant_horizon == 30
    assert result.auc == pytest.approx(5.5)


def test_temporal_association_decay_summary_handles_zero_non_significant_curve():
    """Should report no half-life or significance for a zero-magnitude curve."""
    result = temporal_association_decay_summary(pd.DataFrame({
        'horizon': [1, 5],
        'association': [0.0, 0.0],
        'fdr_rejected': [False, False],
    }))

    assert result.peak_horizon == 1
    assert result.peak_abs_association == 0.0
    assert result.halflife_horizon is None
    assert result.last_significant_horizon is None
    assert result.auc == 0.0


# ------------------------------------------------------
# temporal_association_decay_summary_table
# ------------------------------------------------------
def test_temporal_association_decay_summary_table_reuses_targets_for_all_features(
        temporal_decay_df_pandas,
):
    """Should generate each target once while retaining detailed feature results."""
    calls = []

    def target_fn(target_data, horizon):
        calls.append(horizon)
        return target_data[['time', 'symbol']].assign(**{
            f'target_{horizon}': target_data['target_seed'].shift(-horizon),
        })

    result = temporal_association_decay_summary_table(
        df_features=temporal_decay_df_pandas,
        feature_list=['feature_a', 'feature_b'],
        target_data=temporal_decay_df_pandas,
        horizons=[3, 1],
        target_fn=target_fn,
        block_length=3,
        n_bootstraps=20,
        random_state=42,
        feature_groups={
            'feature_a': 'momentum',
            'feature_b': 'reversal',
        },
    )

    assert isinstance(result, TemporalAssociationDecaySummaryTableResult)
    assert calls == [1, 3]
    assert result.table['feature'].to_list() == ['feature_a', 'feature_b']
    assert result.table['feature_group'].to_list() == ['momentum', 'reversal']
    assert set(result.decay_results) == {'feature_a', 'feature_b'}
    assert result.decay_results['feature_a'].table['horizon'].to_list() == [1, 3]
    assert {
        'feature', 'peak_horizon', 'peak_abs_association',
        'halflife_horizon', 'last_significant_horizon', 'auc',
        'feature_group',
    } == set(result.table.columns)


def test_temporal_association_decay_summary_table_rejects_empty_feature_list(
        temporal_decay_df_pandas,
):
    """Should reject an empty feature list before target generation."""
    with pytest.raises(ValueError, match='feature_list must not be empty'):
        temporal_association_decay_summary_table(
            df_features=temporal_decay_df_pandas,
            feature_list=[],
            target_data=temporal_decay_df_pandas,
            horizons=[1],
            target_fn=lambda data, horizon: data,
            block_length=3,
            n_bootstraps=20,
        )


def test_temporal_association_decay_summary_table_rejects_duplicate_features(
        temporal_decay_df_pandas,
):
    """Should reject duplicate features before generating targets."""
    with pytest.raises(ValueError, match='feature_list must not contain duplicates'):
        temporal_association_decay_summary_table(
            df_features=temporal_decay_df_pandas,
            feature_list=['feature_a', 'feature_a'],
            target_data=temporal_decay_df_pandas,
            horizons=[1],
            target_fn=lambda data, horizon: data,
            block_length=3,
            n_bootstraps=20,
        )


def test_temporal_association_decay_summary_table_pandas_polars_consistency(
        temporal_decay_df_pandas,
        temporal_decay_df_polars,
):
    """Should return equivalent batch temporal-decay results across backends."""
    def pandas_target_fn(target_data, horizon):
        return target_data[['time', 'symbol']].assign(**{
            f'target_{horizon}': target_data['target_seed'].shift(-horizon),
        })

    def polars_target_fn(target_data, horizon):
        return target_data.select(
            'time',
            'symbol',
            pl.col('target_seed').shift(-horizon).alias(f'target_{horizon}'),
        )

    kwargs = {
        'feature_list': ['feature_a', 'feature_b'],
        'horizons': [1, 3],
        'block_length': 3,
        'n_bootstraps': 20,
        'random_state': 42,
        'feature_groups': {
            'feature_a': 'momentum',
            'feature_b': 'reversal',
        },
    }
    pandas_result = temporal_association_decay_summary_table(
        df_features=temporal_decay_df_pandas,
        target_data=temporal_decay_df_pandas,
        target_fn=pandas_target_fn,
        **kwargs,
    )
    polars_result = temporal_association_decay_summary_table(
        df_features=temporal_decay_df_polars,
        target_data=temporal_decay_df_polars,
        target_fn=polars_target_fn,
        **kwargs,
    )

    pd.testing.assert_frame_equal(
        pandas_result.table,
        polars_result.table.to_pandas(),
        check_dtype=False,
    )
    for feature in kwargs['feature_list']:
        pd.testing.assert_frame_equal(
            pandas_result.decay_results[feature].table,
            polars_result.decay_results[feature].table.to_pandas(),
            check_dtype=False,
        )
