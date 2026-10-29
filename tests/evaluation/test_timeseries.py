import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.stats import norm, pearsonr, spearmanr

from alpha_research.evaluation.ic import _generate_target_frames
from alpha_research.evaluation.timeseries import (
    RollingTemporalAssociationResult,
    TemporalAssociationDecayResult,
    TemporalAssociationDecaySummaryTableResult,
    _rolling_random_states,
    _rolling_summary_pandas,
    _select_valid_temporal_pairs,
    _slice_rows,
    _temporal_association_decay_from_target_frames,
    _validate_continuity_column,
    _validate_feature_list,
    _validate_rolling_temporal_association_frame,
    _validate_single_symbol,
    _validate_summary_bootstrap_configuration,
    partial_temporal_association,
    partial_temporal_association_summary_table,
    rolling_temporal_association,
    summarize_rolling_temporal_association,
    temporal_association,
    temporal_association_decay,
    temporal_association_decay_summary,
    temporal_association_decay_summary_table,
    temporal_association_summary_table,
)
from alpha_research.resampling.block_bootstrap import (
    BootstrapMetricsResults,
    generate_moving_blocks,
    moving_block_bootstrap,
)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('partial', [False, True])
def test_temporal_summary_progress_preserves_seeded_output(temporal_summary_df_pandas, backend, partial):
    """Should retain seeded outputs and report sample generation separately from statistical estimation."""
    frame = temporal_summary_df_pandas
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    function = partial_temporal_association_summary_table if partial else temporal_association_summary_table
    args = {'df': frame, 'feature_list': ['feature_a'], 'target': 'target', 'block_length': 3,
                'n_bootstraps': 12, 'random_state': 71}
    if partial:
        args['covariates'] = ['feature_b']
    expected = function(**args)
    events = []
    result = function(**args, on_progress=lambda *event: events.append(event))
    # Existing tests cover numerical correctness; this checks instrumentation transparency.
    if backend == 'pandas':
        pd.testing.assert_frame_equal(result, expected)
    else:
        assert result.equals(expected)
    assert ('feature/feature_a/bootstrap_samples', 12, 12) in events
    assert ('feature/feature_a/bootstrap_estimates', 12, 12) in events
    assert events[-1] == ('features', 1, 1)


@pytest.mark.parametrize('partial', [False, True])
@pytest.mark.parametrize('phase_suffix', ['moving_blocks', 'bootstrap_samples', 'bootstrap_estimates'])
def test_temporal_summary_cancels_inside_each_work_loop(temporal_summary_df_pandas, partial, phase_suffix):
    """Should propagate a callback ValueError unchanged from each nested loop."""
    function = partial_temporal_association_summary_table if partial else temporal_association_summary_table
    args = {'df': temporal_summary_df_pandas, 'feature_list': ['feature_a'], 'target': 'target',
                'block_length': 3, 'n_bootstraps': 12, 'random_state': 71}
    if partial:
        args['covariates'] = ['feature_b']
    exception = ValueError('cancel')
    events = []
    def callback(phase, completed, total):
        """Should stop one chosen local loop after its first successful unit."""
        events.append((phase, completed, total))
        if phase.endswith('/' + phase_suffix) and completed == 1:
            raise exception
    with pytest.raises(ValueError) as caught:
        function(**args, on_progress=callback)
    assert caught.value is exception
    assert events[-1][1] == 1
    assert ('features', 1, 1) not in events


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_rolling_progress_counts_skipped_windows(rolling_temporal_df_pandas, backend):
    """Should count missing and undefined windows as processed without creating bootstrap estimates."""
    frame = rolling_temporal_df_pandas.copy()
    frame['feature'] = 1.0
    frame.loc[frame.index[0], 'target'] = np.nan
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    events = []
    result = rolling_temporal_association(frame, 'feature', 'target', window_size=5, block_length=2,
                                         n_bootstraps=8, random_state=2,
                                         on_progress=lambda *event: events.append(event))
    # Eight endpoints for 12 observations and a five-observation window.
    assert len(result.rolling_frame) == 8
    assert events[-1] == ('rolling_windows', 8, 8)
    assert all('bootstrap_estimates' not in event[0] for event in events)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_rolling_progress_preserves_seeded_windows(rolling_temporal_df_pandas, backend):
    """Should retain original child seeds and distinguish bootstrap work within each window."""
    frame = rolling_temporal_df_pandas if backend == 'pandas' else pl.from_pandas(rolling_temporal_df_pandas)
    args = {'df': frame, 'feature': 'feature', 'target': 'target', 'window_size': 6,
                'block_length': 2, 'n_bootstraps': 8, 'random_state': 17}
    expected = rolling_temporal_association(**args)
    events = []
    result = rolling_temporal_association(**args, on_progress=lambda *event: events.append(event))
    if backend == 'pandas':
        pd.testing.assert_frame_equal(result.rolling_frame, expected.rolling_frame)
        pd.testing.assert_frame_equal(result.summary_table, expected.summary_table)
    else:
        assert result.rolling_frame.equals(expected.rolling_frame)
        assert result.summary_table.equals(expected.summary_table)
    assert ('window/5/bootstrap_samples', 8, 8) in events
    assert ('window/11/bootstrap_estimates', 8, 8) in events
    assert events[-1] == ('rolling_windows', 7, 7)


@pytest.mark.parametrize('exception_type', [ValueError, TypeError, RuntimeError])
def test_rolling_cancellation_is_not_undefined_bootstrap(rolling_temporal_df_pandas, exception_type):
    """Should propagate observation failure instead of returning an undefined-bootstrap window."""
    exception = exception_type('cancel')
    callback = lambda phase, done, total: (_ for _ in ()).throw(exception) if phase.endswith('bootstrap_estimates') and done == 1 else None
    with pytest.raises(exception_type) as caught:
        rolling_temporal_association(rolling_temporal_df_pandas, 'feature', 'target',
                                     window_size=6, block_length=2, n_bootstraps=8,
                                     random_state=17, on_progress=callback)
    assert caught.value is exception


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_temporal_decay_progress_preserves_shared_targets_and_child_streams(temporal_decay_df_pandas, backend):
    """Should observe feature/horizon/bootstrap work while retaining target reuse and seeded output."""
    frame = temporal_decay_df_pandas if backend == 'pandas' else pl.from_pandas(temporal_decay_df_pandas)
    generated = []
    def target_fn(data, horizon):
        """Should generate exactly one shifted target per requested horizon."""
        generated.append(horizon)
        if backend == 'pandas':
            return data[['time', 'symbol']].assign(**{f'target_{horizon}': data['target_seed'].shift(-horizon)})
        return data.select('time', 'symbol', pl.col('target_seed').shift(-horizon).alias(f'target_{horizon}'))
    args = {'df_features': frame, 'feature_list': ['feature_a', 'feature_b'], 'target_data': frame,
                'horizons': [3, 1], 'target_fn': target_fn, 'block_length': 3, 'n_bootstraps': 8, 'random_state': 42}
    expected = temporal_association_decay_summary_table(**args)
    generated.clear()
    events = []
    result = temporal_association_decay_summary_table(**args, on_progress=lambda *event: events.append(event))
    assert generated == [1, 3]
    for feature in ['feature_a', 'feature_b']:
        if backend == 'pandas':
            pd.testing.assert_frame_equal(result.decay_results[feature].table, expected.decay_results[feature].table)
        else:
            assert result.decay_results[feature].table.equals(expected.decay_results[feature].table)
        for horizon in [1, 3]:
            assert (f'feature/{feature}/horizon/{horizon}/feature/{feature}/bootstrap_samples', 8, 8) in events
            assert (f'feature/{feature}/horizon/{horizon}/feature/{feature}/bootstrap_estimates', 8, 8) in events
    assert events[-1] == ('features', 2, 2)


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

@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_select_valid_temporal_pairs_all_missing_local_window(backend):
    """Should allow an empty selection for one validated rolling window."""
    values = {'feature': [None, float('nan')], 'target': [1., 2.]}
    frame = (
        pd.DataFrame(values)
        if backend == 'pandas'
        else pl.DataFrame(
            values,
            schema={'feature': pl.Float64, 'target': pl.Float64},
        )
    )

    result = _select_valid_temporal_pairs(
        frame,
        'feature',
        'target',
        check_all_missing=False,
    )

    assert len(result) == 0


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_select_valid_temporal_pairs_retains_non_missing_input_contract(backend):
    """Should retain schema and non-empty validation for local selections."""
    values = {'feature': [None, float('nan')], 'target': [1., 2.]}
    frame = (
        pd.DataFrame(values)
        if backend == 'pandas'
        else pl.DataFrame(
            values,
            schema={'feature': pl.Float64, 'target': pl.Float64},
        )
    )

    with pytest.raises(ValueError, match='missing all values'):
        _select_valid_temporal_pairs(frame, 'feature', 'target')
    with pytest.raises(KeyError):
        _select_valid_temporal_pairs(
            frame,
            'absent',
            'target',
            check_all_missing=False,
        )
    with pytest.raises(ValueError, match='empty'):
        _select_valid_temporal_pairs(
            frame.head(0),
            'feature',
            'target',
            check_all_missing=False,
        )


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('column', ['feature', 'target'])
@pytest.mark.parametrize('missing_window', [0, 1, 2])
@pytest.mark.parametrize('missing', ['null', 'nan', 'mixed'])
def test_rolling_temporal_association_skips_entirely_missing_local_windows(backend, column, missing_window, missing):
    """Should skip warmup, internal or trailing empty windows and preserve valid windows without compressing time."""
    values = {
        'time': list(pd.date_range('2024-01-01', periods=12)),
        'symbol': ['A'] * 12,
        'feature': [float(i) for i in range(12)],
        'target': [float(2 * i + 1) for i in range(12)],
    }
    for i in range(missing_window * 4, missing_window * 4 + 4):
        values[column][i] = (
            None
            if missing == 'null' or missing == 'mixed' and i % 2
            else float('nan')
        )
    frame = (
        pd.DataFrame(values)
        if backend == 'pandas'
        else pl.DataFrame(
            values,
            schema_overrides={'feature': pl.Float64, 'target': pl.Float64},
        )
    )
    result = rolling_temporal_association(
        frame,
        'feature',
        'target',
        window_size=4,
        window_step=4,
        block_length=2,
        n_bootstraps=4,
        random_state=42,
    ).rolling_frame
    rows = result.to_dict('records') if backend == 'pandas' else result.to_dicts()
    assert len(rows) == 3
    for index, row in enumerate(rows):
        assert pd.Timestamp(row['window_end']) == pd.Timestamp(
            values['time'][4 * index + 3],
        )
        if index == missing_window:
            assert row['status'] == 'missing_pairs'
            assert row['n_obs'] == 0
            assert row['n_bootstraps'] == 0
            assert pd.isna(row['association'])
        else:
            # Affine positive feature/target values have exact population correlation +1.
            assert row['status'] == 'ok'
            assert row['n_obs'] == 4
            assert row['association'] == pytest.approx(1.)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('column', ['feature', 'target'])
def test_rolling_temporal_association_rejects_globally_missing_columns(
        rolling_temporal_df_pandas,
        backend,
        column,
):
    """Should reject a required column with no value in the complete input."""
    entirely_missing = (
        rolling_temporal_df_pandas.copy()
        if backend == 'pandas'
        else pl.from_pandas(rolling_temporal_df_pandas)
    )
    if backend == 'pandas':
        entirely_missing[column] = np.nan
    else:
        entirely_missing = entirely_missing.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(column),
        )

    with pytest.raises(ValueError, match='missing all values'):
        rolling_temporal_association(
            entirely_missing,
            'feature',
            'target',
            window_size=4,
            block_length=2,
            n_bootstraps=4,
        )

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


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_select_valid_temporal_pairs_excludes_non_finite_values(
        temporal_df_pandas,
        backend,
):
    """Should jointly exclude positive and negative infinity across backends."""
    df = temporal_df_pandas.copy()
    df.loc[1, 'feature'] = np.inf
    df.loc[3, 'target'] = -np.inf
    if backend == 'polars':
        df = pl.from_pandas(df)

    result = _select_valid_temporal_pairs(df, 'feature', 'target')

    assert len(result) == 3
    assert np.isfinite(result.to_numpy().astype(float)).all()


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize(
    'values',
    [
        ['low', 'medium', 'high', 'higher', 'highest'],
        ['1', '2', '3', '4', '5'],
        [True, False, True, False, True],
    ],
)
def test_select_valid_temporal_pairs_rejects_non_numeric_values(
        temporal_df_pandas,
        backend,
        values,
):
    """Should reject strings and booleans instead of coercing analysis data."""
    df = temporal_df_pandas.copy()
    df['feature'] = values
    if backend == 'polars':
        df = pl.from_pandas(df)

    with pytest.raises(ValueError, match='must contain numeric values'):
        _select_valid_temporal_pairs(df, 'feature', 'target')

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


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_temporal_association_drops_non_finite_aligned_values(
        temporal_df_pandas,
        backend,
):
    """Should not interpret infinities as valid ranks or observations."""
    df = temporal_df_pandas.copy()
    df.loc[1, 'feature'] = np.inf
    df.loc[3, 'target'] = -np.inf
    expected = spearmanr(
        df.loc[[0, 2, 4], 'feature'],
        df.loc[[0, 2, 4], 'target'],
    ).statistic
    if backend == 'polars':
        df = pl.from_pandas(df)

    result = temporal_association(df, 'feature', 'target')

    assert result == pytest.approx(expected)


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


@pytest.mark.parametrize(
    ('corr_method', 'reference_function'),
    [('pearson', pearsonr), ('spearman', spearmanr)],
)
def test_temporal_association_matches_independent_scipy_reference(
        corr_method,
        reference_function,
):
    """Should match a trusted reference for non-perfect data with ties and NaN."""
    df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=8, freq='D'),
        'symbol': ['AAPL'] * 8,
        'feature': [1.0, 2.0, 2.0, np.nan, 4.0, 7.0, 6.0, 9.0],
        'target': [3.0, 1.0, 2.0, 8.0, 5.0, 4.0, 7.0, 6.0],
    })
    valid = df[['feature', 'target']].dropna()
    expected = reference_function(valid['feature'], valid['target']).statistic

    pandas_result = temporal_association(
        df,
        'feature',
        'target',
        corr_method=corr_method,
    )
    polars_result = temporal_association(
        pl.from_pandas(df),
        'feature',
        'target',
        corr_method=corr_method,
    )

    assert pandas_result == pytest.approx(expected)
    assert polars_result == pytest.approx(expected)


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
# temporal summary validation helpers
# ------------------------------------------------------
def test_validate_feature_list_accepts_unique_non_empty_names():
    """Should accept a non-empty list of unique feature names."""
    assert _validate_feature_list(['feature_a', 'feature_b']) is None


@pytest.mark.parametrize(
    ('feature_list', 'error_type', 'message'),
    [
        ('feature_a', TypeError, 'list of strings'),
        (['feature_a', 1], TypeError, 'list of strings'),
        ([], ValueError, 'must not be empty'),
        ([''], ValueError, 'empty names'),
        (['feature_a', 'feature_a'], ValueError, 'duplicates'),
    ],
)
def test_validate_feature_list_rejects_invalid_specifications(
        feature_list,
        error_type,
        message,
):
    """Should reject ambiguous feature collections before analysis begins."""
    with pytest.raises(error_type, match=message):
        _validate_feature_list(feature_list)


def test_validate_summary_bootstrap_configuration_accepts_valid_arguments():
    """Should accept a bootstrap configuration capable of Wald inference."""
    assert _validate_summary_bootstrap_configuration(
        block_length=3,
        n_bootstraps=20,
        step=1,
        confidence_level=0.95,
        random_state=42,
    ) is None


@pytest.mark.parametrize(
    ('kwargs', 'error_type', 'message'),
    [
        ({'block_length': 0}, ValueError, 'block_length'),
        ({'n_bootstraps': 1}, ValueError, 'at least two'),
        ({'n_bootstraps': True}, ValueError, 'n_bootstraps'),
        ({'step': 0}, ValueError, 'step'),
        ({'confidence_level': 0.0}, ValueError, 'confidence_level'),
        ({'confidence_level': True}, ValueError, 'confidence_level'),
        ({'random_state': '42'}, TypeError, 'random_state'),
        ({'random_state': True}, TypeError, 'random_state'),
    ],
)
def test_validate_summary_bootstrap_configuration_rejects_invalid_arguments(
        kwargs,
        error_type,
        message,
):
    """Should reject configurations that cannot support valid uncertainty."""
    arguments = {
        'block_length': 3,
        'n_bootstraps': 20,
        'step': 1,
        'confidence_level': 0.95,
        'random_state': 42,
    }
    arguments.update(kwargs)

    with pytest.raises(error_type, match=message):
        _validate_summary_bootstrap_configuration(**arguments)


@pytest.mark.parametrize(
    ('continuity_col', 'error_type', 'message'),
    [
        (1, TypeError, 'string or None'),
        ('', ValueError, 'must not be empty'),
        ('feature', ValueError, 'must be distinct'),
    ],
)
def test_validate_continuity_column_rejects_invalid_specifications(
        continuity_col,
        error_type,
        message,
):
    """Should reject ambiguous continuity column specifications."""
    with pytest.raises(error_type, match=message):
        _validate_continuity_column(
            continuity_col,
            ['time', 'symbol', 'feature', 'target'],
        )


@pytest.mark.parametrize('continuity_col', [None, 'original_position'])
def test_validate_continuity_column_accepts_distinct_optional_name(
        continuity_col,
):
    """Should accept an omitted or distinct continuity column name."""
    _validate_continuity_column(
        continuity_col,
        ['time', 'symbol', 'feature', 'target'],
    )


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


def test_validate_rolling_temporal_association_frame_normalizes_numeric_values():
    """Should normalize a consistent rolling result frame for summarization."""
    frame = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL'],
        'feature': ['feature', 'feature'],
        'target': ['target', 'target'],
        'corr_method': ['spearman', 'spearman'],
        'bootstrap_method': ['moving_block', 'moving_block'],
        'window_end': pd.date_range('2024-01-01', periods=2, freq='D'),
        'association': ['0.1', '0.2'],
        'bootstrap_ci_lower': ['-0.1', '0.0'],
        'bootstrap_ci_upper': ['0.3', '0.4'],
        'bootstrap_pct_positive': ['0.6', '0.7'],
        'status': ['ok', 'ok'],
    })

    result = _validate_rolling_temporal_association_frame(frame)

    assert result is not frame
    assert result['association'].to_list() == pytest.approx([0.1, 0.2])
    assert result['bootstrap_pct_positive'].to_list() == pytest.approx([0.6, 0.7])


@pytest.mark.parametrize(
    ('column', 'values', 'message'),
    [
        ('symbol', ['AAPL', 'MSFT'], 'exactly one non-missing'),
        ('corr_method', ['kendall', 'kendall'], 'corr_method'),
        ('bootstrap_method', ['iid', 'iid'], 'bootstrap_method'),
        ('window_end', [pd.Timestamp('2024-01-02')] * 2, 'unique'),
        (
            'window_end',
            [pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-01')],
            'increasingly ordered',
        ),
        ('status', ['ok', 'unsupported'], 'unsupported rolling'),
        ('status', ['missing_pairs', 'ok'], 'must not contain rolling diagnostics'),
        ('status', ['undefined_bootstrap', 'ok'], 'undefined_bootstrap rows require'),
        ('association', [0.1, np.nan], "status 'ok'"),
        ('association', [1.01, 0.2], r'within \[-1, 1\]'),
        ('bootstrap_ci_lower', [0.31, -0.1], 'must not exceed'),
        ('bootstrap_ci_upper', [1.01, 0.4], r'within \[-1, 1\]'),
        ('bootstrap_pct_positive', [1.1, 0.5], 'between zero and one'),
        ('association', ['invalid', 0.2], 'numeric values'),
    ],
)
def test_validate_rolling_temporal_association_frame_rejects_invalid_contract(
        column,
        values,
        message,
):
    """Should reject rolling frames that could produce misleading summaries."""
    frame = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL'],
        'feature': ['feature', 'feature'],
        'target': ['target', 'target'],
        'corr_method': ['spearman', 'spearman'],
        'bootstrap_method': ['moving_block', 'moving_block'],
        'window_end': pd.date_range('2024-01-01', periods=2, freq='D'),
        'association': [0.1, 0.2],
        'bootstrap_ci_lower': [-0.1, 0.0],
        'bootstrap_ci_upper': [0.3, 0.4],
        'bootstrap_pct_positive': [0.6, 0.7],
        'status': ['ok', 'ok'],
    })
    frame[column] = values

    with pytest.raises(ValueError, match=message):
        _validate_rolling_temporal_association_frame(frame)


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
    """Should match an independent Spearman calculation for the first window."""
    result = rolling_temporal_association(
        rolling_temporal_df_pandas,
        feature='feature',
        target='target',
        window_size=6,
        block_length=3,
        n_bootstraps=20,
        random_state=42,
    )
    first_window = rolling_temporal_df_pandas.iloc[:6]
    expected = spearmanr(
        first_window['feature'],
        first_window['target'],
    ).statistic

    assert result.rolling_frame['association'].iloc[0] == pytest.approx(expected)


def test_rolling_temporal_association_includes_one_exact_full_window(
        rolling_temporal_df_pandas,
):
    """Should include a window whose size exactly equals the input length."""
    result = rolling_temporal_association(
        rolling_temporal_df_pandas,
        feature='feature',
        target='target',
        window_size=len(rolling_temporal_df_pandas),
        block_length=3,
        n_bootstraps=20,
        random_state=42,
    ).rolling_frame

    assert len(result) == 1
    assert result['window_start'].iloc[0] == rolling_temporal_df_pandas['time'].iloc[0]
    assert result['window_end'].iloc[0] == rolling_temporal_df_pandas['time'].iloc[-1]


def test_rolling_temporal_association_does_not_look_beyond_window_end(
        rolling_temporal_df_pandas,
):
    """Should keep an early window unchanged when only future rows change."""
    changed_future = rolling_temporal_df_pandas.copy()
    changed_future.loc[6:, 'feature'] *= -100.0
    kwargs = {
        'feature': 'feature',
        'target': 'target',
        'window_size': 6,
        'block_length': 3,
        'n_bootstraps': 20,
        'random_state': 42,
    }

    baseline = rolling_temporal_association(
        rolling_temporal_df_pandas,
        **kwargs,
    ).rolling_frame
    changed = rolling_temporal_association(
        changed_future,
        **kwargs,
    ).rolling_frame

    pd.testing.assert_series_equal(
        baseline.iloc[0],
        changed.iloc[0],
        check_names=False,
    )
    assert not baseline.iloc[1:]['association'].equals(
        changed.iloc[1:]['association'],
    )


@pytest.mark.parametrize('corr_method', ['pearson', 'spearman'])
def test_rolling_temporal_association_supports_established_correlation_methods(
        rolling_temporal_df_pandas,
        corr_method,
):
    """Should match independent correlation references for each estimator."""
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
    first_window = rolling_temporal_df_pandas.iloc[:6]
    reference_function = pearsonr if corr_method == 'pearson' else spearmanr
    expected = reference_function(
        first_window['feature'],
        first_window['target'],
    ).statistic

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


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('invalid_value', [np.inf, -np.inf])
def test_rolling_temporal_association_marks_non_finite_pairs_as_missing(
        rolling_temporal_df_pandas,
        backend,
        invalid_value,
):
    """Should not compress a rolling window containing an infinite pair."""
    frame = rolling_temporal_df_pandas.copy()
    frame.loc[4, 'target'] = invalid_value
    data = frame if backend == 'pandas' else pl.from_pandas(frame)

    result = rolling_temporal_association(
        data,
        feature='feature',
        target='target',
        window_size=4,
        block_length=2,
        n_bootstraps=8,
        random_state=42,
    ).rolling_frame
    statuses = (
        result['status'].to_list()
        if backend == 'polars'
        else result['status'].tolist()
    )

    assert statuses.count('missing_pairs') == 4
    assert 'ok' in statuses


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


def test_summarize_rolling_temporal_association_rejects_mixed_metadata():
    """Should enforce rolling metadata consistency through the public API."""
    rolling_frame = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT'],
        'feature': ['feature', 'feature'],
        'target': ['target', 'target'],
        'corr_method': ['spearman', 'spearman'],
        'bootstrap_method': ['moving_block', 'moving_block'],
        'window_end': pd.date_range('2024-01-01', periods=2, freq='D'),
        'association': [0.1, 0.2],
        'bootstrap_ci_lower': [-0.1, 0.0],
        'bootstrap_ci_upper': [0.3, 0.4],
        'bootstrap_pct_positive': [0.6, 0.7],
        'status': ['ok', 'ok'],
    })

    with pytest.raises(ValueError, match='symbol must contain exactly one'):
        summarize_rolling_temporal_association(rolling_frame)


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


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_temporal_summary_does_not_bridge_filtered_time_gaps(
        temporal_summary_df_pandas, backend,
):
    """Should exclude every MBB block that straddles an original-position gap."""
    frame = temporal_summary_df_pandas.copy()
    frame['original_position'] = np.arange(len(frame)) * 2
    data = frame if backend == 'pandas' else pl.from_pandas(frame)

    with pytest.raises(ValueError, match='no contiguous candidate block'):
        temporal_association_summary_table(
            data, feature_list=['feature_a'], target='target',
            block_length=2, n_bootstraps=10,
            continuity_col='original_position',
        )


def test_temporal_summary_continuity_preserves_full_sequence_results(
        temporal_summary_df_pandas,
):
    """Should retain the established MBB estimates for a complete sequence."""
    frame = temporal_summary_df_pandas.copy()
    frame['original_position'] = np.arange(len(frame))
    kwargs = {
        'feature_list': ['feature_a'], 'target': 'target',
        'block_length': 3, 'n_bootstraps': 20, 'random_state': 42,
    }

    original = temporal_association_summary_table(frame, **kwargs)
    contiguous = temporal_association_summary_table(
        frame, continuity_col='original_position', **kwargs,
    )

    pd.testing.assert_frame_equal(original, contiguous)


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
def test_temporal_association_summary_matches_independent_fixed_resamples(
        monkeypatch,
        temporal_summary_df_pandas,
        corr_method,
):
    """Should match independent reference calculations for fixed resamples."""
    valid_pairs = temporal_summary_df_pandas[['feature_a', 'target']]
    sample_indices = [
        list(range(12)),
        [0, 1, 2, 3, 4, 5] * 2,
        [6, 7, 8, 9, 10, 11] * 2,
        [0, 2, 4, 6, 8, 10] * 2,
    ]
    samples = [
        valid_pairs.iloc[indices].reset_index(drop=True)
        for indices in sample_indices
    ]
    monkeypatch.setattr(
        'alpha_research.evaluation.timeseries.moving_block_bootstrap',
        lambda *args, **kwargs: samples,
    )

    result = temporal_association_summary_table(
        temporal_summary_df_pandas,
        feature_list=['feature_a'],
        target='target',
        block_length=3,
        n_bootstraps=4,
        corr_method=corr_method,
        random_state=42,
    ).iloc[0]
    reference_function = pearsonr if corr_method == 'pearson' else spearmanr
    observed = reference_function(
        valid_pairs['feature_a'],
        valid_pairs['target'],
    ).statistic
    estimates = np.asarray([
        reference_function(sample['feature_a'], sample['target']).statistic
        for sample in samples
    ])
    bootstrap_mean = np.mean(estimates)
    bootstrap_std = np.std(estimates, ddof=1)
    test_statistic = observed / bootstrap_std
    p_value = 2 * norm.sf(abs(test_statistic))
    z_critical = norm.ppf(0.975)

    assert result['association'] == pytest.approx(observed)
    assert result['bootstrap_mean'] == pytest.approx(bootstrap_mean)
    assert result['bootstrap_std'] == pytest.approx(bootstrap_std)
    assert result['bootstrap_pct_positive'] == pytest.approx(
        np.mean(estimates > 0),
    )
    assert result['test_statistic'] == pytest.approx(test_statistic)
    assert result['p_value'] == pytest.approx(p_value)
    assert bool(result['reject_h0']) is bool(p_value < 0.05)
    assert result['wald_ci_lower'] == pytest.approx(
        observed - z_critical * bootstrap_std,
    )
    assert result['wald_ci_upper'] == pytest.approx(
        observed + z_critical * bootstrap_std,
    )


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
    """Should exclude paired missing and non-finite observations jointly."""
    df = temporal_summary_df_pandas.copy()
    df.loc[1, 'feature_a'] = np.nan
    df.loc[3, 'target'] = np.inf

    result = temporal_association_summary_table(
        df,
        feature_list=['feature_a'],
        target='target',
        block_length=3,
        n_bootstraps=20,
        random_state=42,
    ).iloc[0]
    valid = df[['feature_a', 'target']].replace([np.inf, -np.inf], np.nan).dropna()
    expected_association = spearmanr(
        valid['feature_a'],
        valid['target'],
    ).statistic

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


@pytest.mark.parametrize(
    ('kwargs', 'error_type', 'message'),
    [
        ({'n_bootstraps': 1}, ValueError, 'at least two'),
        ({'block_length': 0}, ValueError, 'block_length'),
        ({'step': 0}, ValueError, 'step'),
        ({'confidence_level': np.inf}, ValueError, 'confidence_level'),
        ({'random_state': 1.5}, TypeError, 'random_state'),
    ],
)
def test_temporal_association_summary_table_validates_inference_configuration(
        temporal_summary_df_pandas,
        kwargs,
        error_type,
        message,
):
    """Should reject invalid uncertainty settings at the public boundary."""
    arguments = {
        'feature_list': ['feature_a'],
        'target': 'target',
        'block_length': 3,
        'n_bootstraps': 20,
    }
    arguments.update(kwargs)

    with pytest.raises(error_type, match=message):
        temporal_association_summary_table(
            temporal_summary_df_pandas,
            **arguments,
        )


def test_temporal_association_summary_table_rejects_undefined_observed_estimate(
        temporal_summary_df_pandas,
):
    """Should reject a constant feature before reporting invalid inference."""
    df = temporal_summary_df_pandas.copy()
    df['feature_a'] = 1.0

    with pytest.raises(ValueError, match="undefined for feature 'feature_a'"):
        temporal_association_summary_table(
            df,
            feature_list=['feature_a'],
            target='target',
            block_length=3,
            n_bootstraps=20,
        )


@pytest.mark.parametrize('standard_error', [0.0, np.nan, np.inf])
def test_temporal_association_summary_table_rejects_invalid_bootstrap_error(
        monkeypatch,
        temporal_summary_df_pandas,
        standard_error,
):
    """Should require a positive finite standard error for Wald inference."""
    metrics = BootstrapMetricsResults(
        mean=0.2,
        std=standard_error,
        ci_lower=0.2,
        ci_upper=0.2,
        pct_positive=1.0,
        n_non_positive=0,
        n_non_negative=20,
        n_bootstraps=20,
    )
    monkeypatch.setattr(
        'alpha_research.evaluation.timeseries.bootstrap_metrics',
        lambda *_args, **_kwargs: metrics,
    )

    with pytest.raises(ValueError, match='bootstrap standard error is undefined'):
        temporal_association_summary_table(
            temporal_summary_df_pandas,
            feature_list=['feature_a'],
            target='target',
            block_length=3,
            n_bootstraps=20,
        )


def test_temporal_association_summary_table_rejects_all_invalid_replicates(
        monkeypatch,
        temporal_summary_df_pandas,
):
    """Should reject a bootstrap distribution with no finite estimates."""
    constant_sample = pd.DataFrame({
        'feature_a': [1.0] * len(temporal_summary_df_pandas),
        'target': temporal_summary_df_pandas['target'],
    })
    monkeypatch.setattr(
        'alpha_research.evaluation.timeseries.moving_block_bootstrap',
        lambda *args, **kwargs: [constant_sample] * 4,
    )

    with pytest.raises(ValueError, match='at least one finite estimate'):
        temporal_association_summary_table(
            temporal_summary_df_pandas,
            feature_list=['feature_a'],
            target='target',
            block_length=3,
            n_bootstraps=4,
        )


def test_temporal_association_summary_table_validates_filtered_sample_length(
        temporal_summary_df_pandas,
):
    """Should compare block length with complete pairs after filtering."""
    df = temporal_summary_df_pandas.copy()
    df.loc[:9, 'feature_a'] = np.nan

    with pytest.raises(ValueError, match='must not exceed.*observations'):
        temporal_association_summary_table(
            df,
            feature_list=['feature_a'],
            target='target',
            block_length=3,
            n_bootstraps=4,
        )


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_temporal_summary_continuity_excludes_only_gap_crossing_blocks(
        monkeypatch,
        temporal_summary_df_pandas,
        backend,
):
    """Should retain valid blocks on both sides of an internal missing pair."""
    frame = temporal_summary_df_pandas.copy()
    frame['original_position'] = np.arange(len(frame))
    frame.loc[4, 'feature_a'] = np.nan
    data = frame if backend == 'pandas' else pl.from_pandas(frame)
    observed_blocks = []
    original = generate_moving_blocks

    def capture_blocks(data, block_length, step, continuity, **kwargs):
        blocks = original(
            data,
            block_length,
            step,
            continuity=continuity,
            **kwargs,
        )
        observed_blocks.extend(blocks)
        return blocks

    monkeypatch.setattr(
        'alpha_research.evaluation.timeseries.generate_moving_blocks',
        capture_blocks,
    )
    temporal_association_summary_table(
        data,
        feature_list=['feature_a'],
        target='target',
        block_length=3,
        n_bootstraps=4,
        random_state=3,
        continuity_col='original_position',
    )

    block_positions = [
        block['original_position'].to_list()
        for block in observed_blocks
    ]
    assert block_positions == [
        [0, 1, 2], [1, 2, 3],
        [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11],
    ]


@pytest.mark.parametrize('continuity_col', ['time', 'symbol', 'target', 'feature_a'])
def test_temporal_summary_rejects_continuity_role_collisions(
        temporal_summary_df_pandas,
        continuity_col,
):
    """Should reject a continuity column that also has an analytical role."""
    with pytest.raises(ValueError, match='must be distinct'):
        temporal_association_summary_table(
            temporal_summary_df_pandas,
            feature_list=['feature_a'],
            target='target',
            block_length=3,
            n_bootstraps=4,
            continuity_col=continuity_col,
        )


def test_temporal_summary_documents_raw_correlation_wald_boundary(
        monkeypatch,
        temporal_summary_df_pandas,
):
    """Should leave raw-correlation Wald bounds unclipped near one."""
    metrics = BootstrapMetricsResults(
        mean=0.98,
        std=0.10,
        ci_lower=0.80,
        ci_upper=1.00,
        pct_positive=1.0,
        n_non_positive=0,
        n_non_negative=20,
        n_bootstraps=20,
    )
    monkeypatch.setattr(
        'alpha_research.evaluation.timeseries.temporal_association',
        lambda **kwargs: 0.99,
    )
    monkeypatch.setattr(
        'alpha_research.evaluation.timeseries.bootstrap_metrics',
        lambda *args, **kwargs: metrics,
    )

    result = temporal_association_summary_table(
        temporal_summary_df_pandas,
        feature_list=['feature_a'],
        target='target',
        block_length=3,
        n_bootstraps=20,
    ).iloc[0]

    assert result['association'] == pytest.approx(0.99)
    assert result['wald_ci_upper'] > 1.0


def test_temporal_summary_mbb_inference_on_dependent_ar1_data():
    """Should control null rejections and detect a strong dependent signal."""
    n_replications = 30
    null_rejections = 0
    null_coverage = 0
    alternative_rejections = 0

    for replication in range(n_replications):
        rng = np.random.default_rng(10_000 + replication)
        n_obs = 120
        feature = np.empty(n_obs)
        null_target = np.empty(n_obs)
        alternative_target = np.empty(n_obs)
        feature[0] = rng.normal()
        null_target[0] = rng.normal()
        alternative_target[0] = 0.7 * feature[0] + rng.normal(scale=0.7)
        for index in range(1, n_obs):
            feature[index] = 0.65 * feature[index - 1] + rng.normal()
            null_target[index] = (
                0.65 * null_target[index - 1] + rng.normal()
            )
            alternative_target[index] = (
                0.45 * alternative_target[index - 1]
                + 0.7 * feature[index]
                + rng.normal(scale=0.7)
            )

        frame = pd.DataFrame({
            'time': pd.date_range('2020-01-01', periods=n_obs, freq='D'),
            'symbol': ['asset'] * n_obs,
            'feature': feature,
            'null_target': null_target,
            'alternative_target': alternative_target,
        })
        common = {
            'df': frame,
            'feature_list': ['feature'],
            'block_length': 8,
            'n_bootstraps': 79,
            'corr_method': 'pearson',
            'random_state': 20_000 + replication,
        }
        null_result = temporal_association_summary_table(
            target='null_target',
            **common,
        ).iloc[0]
        alternative_result = temporal_association_summary_table(
            target='alternative_target',
            **common,
        ).iloc[0]

        null_rejections += int(null_result['reject_h0'])
        null_coverage += int(
            null_result['wald_ci_lower'] <= 0.0 <= null_result['wald_ci_upper']
        )
        alternative_rejections += int(alternative_result['reject_h0'])

    # Fixed-seed Monte Carlo tolerances are deliberately wider than asymptotic
    # 95% targets, but still fail material loss of dependence-aware inference.
    assert null_rejections <= 5
    assert null_coverage >= 25
    assert alternative_rejections >= 24


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


def test_temporal_association_decay_applies_bh_across_the_horizon_family(
        monkeypatch,
        temporal_decay_df_pandas,
):
    """Should apply the BH definition jointly to one feature's horizons."""
    target_frames = {
        horizon: temporal_decay_df_pandas[['time', 'symbol']].assign(**{
            f'target_{horizon}': temporal_decay_df_pandas['target_seed'].shift(
                -horizon,
            ),
        })
        for horizon in [1, 2, 3]
    }
    p_values = {'target_1': 0.01, 'target_2': 0.03, 'target_3': 0.20}

    def fixed_horizon_summary(**kwargs):
        return pd.DataFrame([{
            'feature': kwargs['feature_list'][0],
            'association': 0.1,
            'p_value': p_values[kwargs['target']],
            'feature_group': 'ungrouped',
        }])

    monkeypatch.setattr(
        'alpha_research.evaluation.timeseries.temporal_association_summary_table',
        fixed_horizon_summary,
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
        feature_groups=None,
        fdr=0.05,
        fdr_method='bh',
    ).table

    assert result['fdr_corrected_p_value'].to_list() == pytest.approx(
        [0.03, 0.045, 0.20],
    )
    assert result['fdr_rejected'].to_list() == [True, True, False]


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_temporal_decay_propagates_original_positions_across_target_gaps(
        monkeypatch,
        temporal_decay_df_pandas,
        backend,
):
    """Should prevent decay MBB blocks from crossing a missing target time."""
    frame = temporal_decay_df_pandas.iloc[:12].copy()
    frame['original_position'] = np.arange(len(frame))
    target = frame[['time', 'symbol', 'target_seed']].drop(index=4)
    if backend == 'polars':
        frame = pl.from_pandas(frame)
        target = pl.from_pandas(target)
    observed_blocks = []
    original = generate_moving_blocks

    def capture_blocks(data, block_length, step, continuity, **kwargs):
        blocks = original(
            data,
            block_length,
            step,
            continuity=continuity,
            **kwargs,
        )
        observed_blocks.extend(blocks)
        return blocks

    monkeypatch.setattr(
        'alpha_research.evaluation.timeseries.generate_moving_blocks',
        capture_blocks,
    )
    _temporal_association_decay_from_target_frames(
        df_feature=frame,
        feature='feature_a',
        target_frames={1: target.rename(
            columns={'target_seed': 'target_1'},
        ) if backend == 'pandas' else target.rename({'target_seed': 'target_1'})},
        block_length=3,
        n_bootstraps=4,
        corr_method='spearman',
        step=1,
        confidence_level=0.95,
        random_state=42,
        time_col='time',
        symbol_col='symbol',
        feature_groups=None,
        fdr=0.05,
        fdr_method='bh',
        continuity_col='original_position',
    )

    positions = [block['original_position'].to_list() for block in observed_blocks]
    assert positions == [
        [0, 1, 2], [1, 2, 3],
        [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11],
    ]


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
    expected_associations = []
    for horizon in [1, 3]:
        aligned = pd.DataFrame({
            'feature': temporal_decay_df_pandas['feature_a'],
            'target': temporal_decay_df_pandas['target_seed'].shift(-horizon),
        }).dropna()
        expected_associations.append(
            spearmanr(aligned['feature'], aligned['target']).statistic,
        )
    assert result.table['association'].to_list() == pytest.approx(
        expected_associations,
    )
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


def test_temporal_association_decay_validates_before_target_generation(
        temporal_decay_df_pandas,
):
    """Should reject invalid inference settings before invoking target code."""
    calls = []

    def target_fn(target_data, horizon):
        calls.append(horizon)
        return target_data

    with pytest.raises(ValueError, match='at least two'):
        temporal_association_decay(
            df_feature=temporal_decay_df_pandas,
            feature='feature_a',
            target_data=temporal_decay_df_pandas,
            horizons=[1],
            target_fn=target_fn,
            block_length=3,
            n_bootstraps=1,
        )

    assert calls == []


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


def test_temporal_association_decay_summary_pandas_polars_consistency():
    """Should preserve scalar decay definitions across supported backends."""
    decay_curve = pd.DataFrame({
        'horizon': [1, 3, 7, 14],
        'association': [-0.4, 0.3, np.nan, -0.1],
        'fdr_rejected': [True, True, False, False],
    })

    pandas_result = temporal_association_decay_summary(decay_curve)
    polars_result = temporal_association_decay_summary(pl.from_pandas(decay_curve))

    assert polars_result == pandas_result
    assert pandas_result.peak_horizon == 1
    assert pandas_result.halflife_horizon == 14
    assert pandas_result.last_significant_horizon == 3
    assert pandas_result.auc == pytest.approx(
        np.trapezoid([0.4, 0.3, 0.1], [1, 3, 14]),
    )


@pytest.mark.parametrize(
    ('horizon', 'message'),
    [
        ([0, 2], 'positive integers'),
        ([-1, 2], 'positive integers'),
        ([1.5, 2], 'positive integers'),
        ([np.nan, 2], 'positive integers'),
        ([np.inf, 2], 'positive integers'),
        ([True, 2], 'numeric values'),
        (['1', '2'], 'numeric values'),
        (['one', 'two'], 'numeric values'),
        ([1, 'two'], 'numeric values'),
    ],
)
def test_temporal_association_decay_summary_rejects_invalid_horizons(
        horizon,
        message,
):
    """Should reject horizons that cannot identify a positive discrete grid."""
    decay_curve = pd.DataFrame({
        'horizon': horizon,
        'association': [0.2, 0.1],
        'fdr_rejected': [True, False],
    })

    with pytest.raises(ValueError, match=message):
        temporal_association_decay_summary(decay_curve)


@pytest.mark.parametrize('association', [[True, False], ['0.2', '0.1']])
def test_temporal_association_decay_summary_rejects_non_numeric_associations(
        association,
):
    """Should reject boolean and numeric-string association values."""
    decay_curve = pd.DataFrame({
        'horizon': [1, 2],
        'association': association,
        'fdr_rejected': [False, False],
    })

    with pytest.raises(ValueError, match='must contain numeric values'):
        temporal_association_decay_summary(decay_curve)


@pytest.mark.parametrize('invalid_flag', [1, 0.0, 'yes', None])
def test_temporal_association_decay_summary_rejects_non_boolean_fdr_flags(
        invalid_flag,
):
    """Should not coerce arbitrary truthy values into significance decisions."""
    decay_curve = pd.DataFrame({
        'horizon': [1, 2],
        'association': [0.2, 0.1],
        'fdr_rejected': [True, invalid_flag],
    })

    with pytest.raises(ValueError, match='must contain boolean values'):
        temporal_association_decay_summary(decay_curve)


def test_temporal_association_decay_summary_rejects_significant_non_finite_value():
    """Should not report FDR persistence for an undefined association."""
    decay_curve = pd.DataFrame({
        'horizon': [1, 2],
        'association': [0.2, np.nan],
        'fdr_rejected': [True, True],
    })

    with pytest.raises(ValueError, match='non-finite association'):
        temporal_association_decay_summary(decay_curve)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize(
    ('horizons', 'associations', 'message'),
    [
        ([1, 1], [0.2, 0.1], 'exactly one row per horizon'),
        ([1, 2], [np.inf, -np.inf], 'no finite association'),
        ([1, 2], ['strong', 'weak'], 'numeric values'),
    ],
)
def test_temporal_association_decay_summary_rejects_malformed_curve_values(
        horizons,
        associations,
        message,
        backend,
):
    """Should reject ambiguous or non-numeric decay curves across backends."""
    decay_curve = pd.DataFrame({
        'horizon': horizons,
        'association': associations,
        'fdr_rejected': [False, False],
    })
    if backend == 'polars':
        decay_curve = pl.from_pandas(decay_curve)

    with pytest.raises(ValueError, match=message):
        temporal_association_decay_summary(decay_curve)


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


def test_temporal_decay_summary_table_validates_before_target_generation(
        temporal_decay_df_pandas,
):
    """Should reject invalid uncertainty before generating shared targets."""
    calls = []

    def target_fn(target_data, horizon):
        calls.append(horizon)
        return target_data

    with pytest.raises(ValueError, match='at least two'):
        temporal_association_decay_summary_table(
            df_features=temporal_decay_df_pandas,
            feature_list=['feature_a'],
            target_data=temporal_decay_df_pandas,
            horizons=[1],
            target_fn=target_fn,
            block_length=3,
            n_bootstraps=1,
        )

    assert calls == []


def test_temporal_decay_uses_per_feature_fdr_families(
        monkeypatch,
        temporal_decay_df_pandas,
):
    """Should correct horizons per feature instead of as one global family."""
    p_values = {
        ('feature_a', 'target_1'): 0.01,
        ('feature_a', 'target_2'): 0.04,
        ('feature_b', 'target_1'): 0.03,
        ('feature_b', 'target_2'): 0.90,
    }

    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol']].assign(**{
            f'target_{horizon}': target_data['target_seed'].shift(-horizon),
        })

    def fixed_horizon_summary(**kwargs):
        feature = kwargs['feature_list'][0]
        return pd.DataFrame([{
            'feature': feature,
            'association': 0.4 if kwargs['target'] == 'target_1' else 0.2,
            'p_value': p_values[(feature, kwargs['target'])],
            'feature_group': 'ungrouped',
        }])

    monkeypatch.setattr(
        'alpha_research.evaluation.timeseries.temporal_association_summary_table',
        fixed_horizon_summary,
    )
    result = temporal_association_decay_summary_table(
        df_features=temporal_decay_df_pandas,
        feature_list=['feature_a', 'feature_b'],
        target_data=temporal_decay_df_pandas,
        horizons=[1, 2],
        target_fn=target_fn,
        block_length=3,
        n_bootstraps=4,
        random_state=42,
    )

    assert result.decay_results['feature_a'].table['fdr_rejected'].to_list() == [
        True, True,
    ]
    assert result.decay_results['feature_b'].table['fdr_rejected'].to_list() == [
        False, False,
    ]


def test_temporal_decay_aligns_horizon_random_streams_across_features(
        monkeypatch,
        temporal_decay_df_pandas,
):
    """Should deliberately reuse each horizon seed across feature analyses."""
    observed_seeds = {'feature_a': [], 'feature_b': []}

    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol']].assign(**{
            f'target_{horizon}': target_data['target_seed'].shift(-horizon),
        })

    def fixed_horizon_summary(**kwargs):
        feature = kwargs['feature_list'][0]
        observed_seeds[feature].append(kwargs['random_state'])
        return pd.DataFrame([{
            'feature': feature,
            'association': 0.2,
            'p_value': 0.5,
            'feature_group': 'ungrouped',
        }])

    monkeypatch.setattr(
        'alpha_research.evaluation.timeseries.temporal_association_summary_table',
        fixed_horizon_summary,
    )
    temporal_association_decay_summary_table(
        df_features=temporal_decay_df_pandas,
        feature_list=['feature_a', 'feature_b'],
        target_data=temporal_decay_df_pandas,
        horizons=[1, 3],
        target_fn=target_fn,
        block_length=3,
        n_bootstraps=4,
        random_state=42,
    )

    assert observed_seeds['feature_a'] == observed_seeds['feature_b']
    assert len(set(observed_seeds['feature_a'])) == 2


@pytest.mark.parametrize(
    'continuity_col',
    ['time', 'symbol', 'feature_a', 'feature_b'],
)
def test_temporal_decay_summary_rejects_continuity_role_collisions(
        temporal_decay_df_pandas,
        continuity_col,
):
    """Should reject continuity metadata that overlaps a batch input role."""
    with pytest.raises(ValueError, match='must be distinct'):
        temporal_association_decay_summary_table(
            df_features=temporal_decay_df_pandas,
            feature_list=['feature_a', 'feature_b'],
            target_data=temporal_decay_df_pandas,
            horizons=[1],
            target_fn=lambda data, horizon: data,
            block_length=3,
            n_bootstraps=4,
            continuity_col=continuity_col,
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


# ------------------------------------------------------
# partial_temporal_association
# ------------------------------------------------------
@pytest.fixture
def partial_temporal_data_pandas():
    """Create one ordered asset with two controls for partial association tests."""
    rng = np.random.default_rng(99)
    n_obs = 48
    covariate_a = rng.normal(size=n_obs)
    covariate_b = rng.normal(size=n_obs)
    feature = 0.8 * covariate_a - 0.4 * covariate_b
    feature += rng.normal(scale=0.7, size=n_obs)
    target = 0.6 * feature + 0.7 * covariate_a + 0.3 * covariate_b
    target += rng.normal(scale=0.7, size=n_obs)
    feature_b = -0.5 * covariate_a + 0.2 * covariate_b
    feature_b += rng.normal(scale=0.8, size=n_obs)
    return pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n_obs, freq='D'),
        'symbol': ['AAPL'] * n_obs,
        'feature': feature,
        'feature_b': feature_b,
        'target': target,
        'covariate_a': covariate_a,
        'covariate_b': covariate_b,
    })


@pytest.mark.parametrize('corr_method', ['pearson', 'spearman'])
def test_partial_temporal_association_matches_precision_matrix_definition(
        partial_temporal_data_pandas,
        corr_method,
):
    """Should match the independent inverse-correlation-matrix definition."""
    df = partial_temporal_data_pandas.copy()
    df.loc[2, 'feature'] = np.nan
    df.loc[7, 'covariate_a'] = np.inf
    columns = ['feature', 'target', 'covariate_a', 'covariate_b']
    values = df[columns].to_numpy(dtype=float)
    values = values[np.isfinite(values).all(axis=1)]
    if corr_method == 'spearman':
        values = pd.DataFrame(values).rank(method='average').to_numpy()
    precision = np.linalg.inv(np.corrcoef(values, rowvar=False))
    expected = -precision[0, 1] / np.sqrt(
        precision[0, 0] * precision[1, 1],
    )

    pandas_result = partial_temporal_association(
        df,
        feature='feature',
        target='target',
        covariates=['covariate_a', 'covariate_b'],
        corr_method=corr_method,
    )
    polars_result = partial_temporal_association(
        pl.from_pandas(df),
        feature='feature',
        target='target',
        covariates=['covariate_a', 'covariate_b'],
        corr_method=corr_method,
    )

    assert pandas_result == pytest.approx(expected)
    assert polars_result == pytest.approx(expected)


def test_partial_temporal_association_requires_one_ordered_symbol(
        partial_temporal_data_pandas,
):
    """Partial temporal association should retain the temporal data contract."""
    multiple_symbols = partial_temporal_data_pandas.copy()
    multiple_symbols.loc[0, 'symbol'] = 'MSFT'

    with pytest.raises(ValueError, match='exactly one'):
        partial_temporal_association(
            multiple_symbols,
            'feature',
            'target',
            covariates='covariate_a',
        )


# ------------------------------------------------------
# partial_temporal_association_summary_table
# ------------------------------------------------------
def test_partial_temporal_association_summary_table_returns_wald_results(
        partial_temporal_data_pandas,
):
    """The controlled temporal summary should retain MBB and Wald diagnostics."""
    result = partial_temporal_association_summary_table(
        partial_temporal_data_pandas,
        feature_list=['feature', 'feature_b'],
        target='target',
        covariates=['covariate_a', 'covariate_b'],
        block_length=4,
        n_bootstraps=30,
        random_state=7,
    )

    assert result['feature'].tolist() == ['feature', 'feature_b']
    assert result['covariates'].tolist() == [
        'covariate_a, covariate_b',
        'covariate_a, covariate_b',
    ]
    assert np.isfinite(result['association']).all()
    assert np.isfinite(result['wald_ci_lower']).all()
    assert np.isfinite(result['wald_ci_upper']).all()
    assert result['n_bootstraps'].tolist() == [30, 30]


def test_partial_temporal_summary_matches_independent_fixed_resample_inference(
        monkeypatch,
        partial_temporal_data_pandas,
):
    """Should match precision-matrix estimates and the Wald definition."""
    columns = ['feature', 'target', 'covariate_a', 'covariate_b']
    values = partial_temporal_data_pandas[columns]
    sample_indices = [
        list(range(48)),
        list(range(24)) * 2,
        list(range(24, 48)) * 2,
        list(range(0, 48, 2)) * 2,
    ]
    samples = [
        values.iloc[indices].reset_index(drop=True)
        for indices in sample_indices
    ]
    monkeypatch.setattr(
        'alpha_research.evaluation.timeseries.moving_block_bootstrap',
        lambda *args, **kwargs: samples,
    )

    def precision_partial(sample):
        correlation = np.corrcoef(sample[columns].to_numpy(), rowvar=False)
        precision = np.linalg.inv(correlation)
        return -precision[0, 1] / np.sqrt(
            precision[0, 0] * precision[1, 1],
        )

    observed = precision_partial(values)
    estimates = np.asarray([precision_partial(sample) for sample in samples])
    standard_error = np.std(estimates, ddof=1)
    statistic = observed / standard_error
    expected_p_value = 2 * norm.sf(abs(statistic))

    result = partial_temporal_association_summary_table(
        partial_temporal_data_pandas,
        feature_list=['feature'],
        target='target',
        covariates=['covariate_a', 'covariate_b'],
        block_length=4,
        n_bootstraps=4,
        corr_method='pearson',
        random_state=7,
    ).iloc[0]

    assert result['association'] == pytest.approx(observed)
    assert result['bootstrap_mean'] == pytest.approx(np.mean(estimates))
    assert result['bootstrap_std'] == pytest.approx(standard_error)
    assert result['test_statistic'] == pytest.approx(statistic)
    assert result['p_value'] == pytest.approx(expected_p_value)
    assert bool(result['reject_h0']) is bool(expected_p_value < 0.05)


def test_partial_temporal_summary_rejects_undefined_observed_estimate(
        partial_temporal_data_pandas,
):
    """Should not report inference when residual variation is absent."""
    df = partial_temporal_data_pandas.copy()
    df['feature'] = 1.0

    with pytest.raises(ValueError, match="undefined for feature 'feature'"):
        partial_temporal_association_summary_table(
            df,
            feature_list=['feature'],
            target='target',
            covariates='covariate_a',
            block_length=4,
            n_bootstraps=20,
        )


@pytest.mark.parametrize('standard_error', [0.0, np.nan, np.inf])
def test_partial_temporal_summary_rejects_invalid_bootstrap_error(
        monkeypatch,
        partial_temporal_data_pandas,
        standard_error,
):
    """Should require positive finite uncertainty for partial Wald inference."""
    metrics = BootstrapMetricsResults(
        mean=0.2,
        std=standard_error,
        ci_lower=0.2,
        ci_upper=0.2,
        pct_positive=1.0,
        n_non_positive=0,
        n_non_negative=20,
        n_bootstraps=20,
    )
    monkeypatch.setattr(
        'alpha_research.evaluation.timeseries.bootstrap_metrics',
        lambda *_args, **_kwargs: metrics,
    )

    with pytest.raises(ValueError, match='bootstrap standard error is undefined'):
        partial_temporal_association_summary_table(
            partial_temporal_data_pandas,
            feature_list=['feature'],
            target='target',
            covariates='covariate_a',
            block_length=4,
            n_bootstraps=20,
        )


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_partial_temporal_summary_does_not_bridge_filtered_time_gaps(
        partial_temporal_data_pandas, backend,
):
    """Should retain original observation adjacency in partial-MBB blocks."""
    frame = partial_temporal_data_pandas.copy()
    frame['original_position'] = np.arange(len(frame)) * 2
    data = frame if backend == 'pandas' else pl.from_pandas(frame)

    with pytest.raises(ValueError, match='no contiguous candidate block'):
        partial_temporal_association_summary_table(
            data, feature_list=['feature'], target='target',
            covariates='covariate_a', block_length=2, n_bootstraps=10,
            continuity_col='original_position',
        )


def test_partial_temporal_summary_continuity_preserves_full_sequence_results(
        partial_temporal_data_pandas,
):
    """Should retain existing partial-MBB estimates for a complete sequence."""
    frame = partial_temporal_data_pandas.copy()
    frame['original_position'] = np.arange(len(frame))
    kwargs = {
        'feature_list': ['feature'], 'target': 'target',
        'covariates': 'covariate_a', 'block_length': 4,
        'n_bootstraps': 20, 'random_state': 4,
    }

    original = partial_temporal_association_summary_table(frame, **kwargs)
    contiguous = partial_temporal_association_summary_table(
        frame, continuity_col='original_position', **kwargs,
    )

    pd.testing.assert_frame_equal(original, contiguous)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_partial_temporal_summary_excludes_only_gap_crossing_blocks(
        monkeypatch,
        partial_temporal_data_pandas,
        backend,
):
    """Should keep partial-MBB blocks on both sides of an internal gap."""
    frame = partial_temporal_data_pandas.iloc[:16].copy()
    frame['original_position'] = np.arange(len(frame))
    frame.loc[5, 'covariate_a'] = np.nan
    data = frame if backend == 'pandas' else pl.from_pandas(frame)
    observed_blocks = []
    original = generate_moving_blocks

    def capture_blocks(data, block_length, step, continuity, **kwargs):
        blocks = original(
            data,
            block_length,
            step,
            continuity=continuity,
            **kwargs,
        )
        observed_blocks.extend(blocks)
        return blocks

    monkeypatch.setattr(
        'alpha_research.evaluation.timeseries.generate_moving_blocks',
        capture_blocks,
    )
    partial_temporal_association_summary_table(
        data,
        feature_list=['feature'],
        target='target',
        covariates='covariate_a',
        block_length=3,
        n_bootstraps=4,
        random_state=4,
        continuity_col='original_position',
    )

    positions = [block['original_position'].to_list() for block in observed_blocks]
    assert positions == [
        [0, 1, 2], [1, 2, 3], [2, 3, 4],
        [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11],
        [10, 11, 12], [11, 12, 13], [12, 13, 14], [13, 14, 15],
    ]


@pytest.mark.parametrize(
    'continuity_col',
    ['time', 'symbol', 'target', 'feature', 'covariate_a'],
)
def test_partial_temporal_summary_rejects_continuity_role_collisions(
        partial_temporal_data_pandas,
        continuity_col,
):
    """Should reject continuity metadata that overlaps a controlled role."""
    with pytest.raises(ValueError, match='must be distinct'):
        partial_temporal_association_summary_table(
            partial_temporal_data_pandas,
            feature_list=['feature'],
            target='target',
            covariates='covariate_a',
            block_length=4,
            n_bootstraps=4,
            continuity_col=continuity_col,
        )


def test_partial_temporal_association_summary_table_preserves_polars_backend(
        partial_temporal_data_pandas,
):
    """Partial temporal summary should preserve Polars and fixed-seed results."""
    kwargs = {
        'feature_list': ['feature'],
        'target': 'target',
        'covariates': 'covariate_a',
        'block_length': 4,
        'n_bootstraps': 20,
        'random_state': 4,
    }
    pandas_result = partial_temporal_association_summary_table(
        partial_temporal_data_pandas,
        **kwargs,
    )
    polars_result = partial_temporal_association_summary_table(
        pl.from_pandas(partial_temporal_data_pandas),
        **kwargs,
    )

    assert isinstance(polars_result, pl.DataFrame)
    assert polars_result['association'].to_list() == pytest.approx(
        pandas_result['association'].tolist(),
    )
    assert polars_result['p_value'].to_list() == pytest.approx(
        pandas_result['p_value'].tolist(),
    )


def test_partial_temporal_summary_rejects_covariate_as_feature(
        partial_temporal_data_pandas,
):
    """A controlled temporal summary should reject a covariate feature role."""
    with pytest.raises(ValueError, match='must not contain covariate'):
        partial_temporal_association_summary_table(
            partial_temporal_data_pandas,
            feature_list=['covariate_a'],
            target='target',
            covariates='covariate_a',
            block_length=4,
            n_bootstraps=20,
        )


@pytest.mark.parametrize(
    ('kwargs', 'error_type', 'message'),
    [
        ({'n_bootstraps': 1}, ValueError, 'at least two'),
        ({'block_length': 0}, ValueError, 'block_length'),
        ({'step': 0}, ValueError, 'step'),
        ({'confidence_level': np.inf}, ValueError, 'confidence_level'),
        ({'random_state': 1.5}, TypeError, 'random_state'),
    ],
)
def test_partial_temporal_summary_validates_inference_configuration(
        partial_temporal_data_pandas,
        kwargs,
        error_type,
        message,
):
    """Should reject invalid partial-inference settings at the public boundary."""
    arguments = {
        'feature_list': ['feature'],
        'target': 'target',
        'covariates': 'covariate_a',
        'block_length': 4,
        'n_bootstraps': 20,
    }
    arguments.update(kwargs)

    with pytest.raises(error_type, match=message):
        partial_temporal_association_summary_table(
            partial_temporal_data_pandas,
            **arguments,
        )
