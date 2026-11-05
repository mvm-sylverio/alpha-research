import pytest
import numpy as np
import pandas as pd
import polars as pl
from math import erf, erfc, sqrt
from alpha_research.evaluation.ic import (
    compute_ic,
    compute_ic_metrics,
    _generate_target_frames,
    _ic_decay_from_target_frames,
    ic_decay,
    ic_decay_summary,
    ic_decay_summary_table,
    ic_summary_table,
    information_coefficient,
    PartialICSummaryResult,
    partial_ic_summary_table,
    partial_information_coefficient,
    _validate_ic_keys,
)
from alpha_research.evaluation.partial import partial_correlation


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('missing', [None, float('nan')], ids=['null', 'nan'])
@pytest.mark.parametrize('method', ['pearson', 'spearman'])
def test_missing_pairs_propagate_consistently_through_ic_and_decay(backend, missing, method):
    """Should use only the two complete pairs through the full IC and decay pipeline."""
    features = {'time': [1, 1, 1, 1], 'symbol': ['A', 'B', 'C', 'D'],
                'x': [1., missing, 3., 4.]}
    targets = {'time': [1, 1, 1, 1], 'symbol': ['A', 'B', 'C', 'D'],
               'y': [1., 2., 3., missing]}
    make = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    feature_frame, target_frame = make(features), make(targets)
    joined = make({'time': features['time'], 'symbol': features['symbol'],
                   'x': features['x'], 'y': targets['y']})
    direct = compute_ic(joined, 'x', 'y', method, symbol_column='symbol')
    summary = ic_summary_table(joined, ['x'], 'y', method, symbol_column='symbol')
    decay = ic_decay(feature_frame, 'x', target_frame, [1], lambda data, horizon: data,
                     corr_method=method)
    assert direct['ic'].to_list() == pytest.approx([1.])
    assert summary.ic_frames['x']['ic'].to_list() == pytest.approx([1.])
    assert decay.ic_frames[1]['ic'].to_list() == pytest.approx([1.])
    assert decay.table['n_obs'].to_list() == [1]


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('breadth,expected', [(0, []), (1, []), (2, [1.])])
def test_ic_cross_section_complete_pair_minimum(backend, breadth, expected):
    """Should require two complete nonconstant observations in each cross-section."""
    make = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    frame = make({'time': [1, 1, 1], 'symbol': ['A', 'B', 'C'],
                  'x': [float(i) if i < breadth else None for i in range(3)],
                  'y': [1., 2., 3.]})
    result = compute_ic(frame, 'x', 'y', symbol_column='symbol')
    assert result['ic'].to_list() == pytest.approx(expected)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('bad_horizon', [0, -1, 1.5, True, None, float('nan'), '2'])
def test_decay_summary_rejects_invalid_horizon_matrix(backend, bad_horizon):
    """Should reject nonpositive, noninteger, boolean, and missing horizons."""
    make = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    frame = make({'horizon': [bad_horizon], 'mean': [0.2], 'fdr_rejected': [False]})
    with pytest.raises((ValueError, TypeError)):
        ic_decay_summary(frame)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('case', ['missing_time', 'missing_symbol', 'missing_value', 'extra_value',
                                 'duplicate', 'null_key', 'other_backend'])
def test_decay_rejects_malformed_generated_targets(backend, case):
    """Should reject malformed target schemas, invalid keys, and backend mismatch."""
    make = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    features = make({'time': [1, 1], 'symbol': ['A', 'B'], 'x': [1., 2.]})
    data = {'time': [1, 1], 'symbol': ['A', 'B'], 'y': [1., 2.]}
    if case == 'missing_time':
        del data['time']
    elif case == 'missing_symbol':
        del data['symbol']
    elif case == 'missing_value':
        del data['y']
    elif case == 'extra_value':
        data['extra'] = [1., 2.]
    elif case == 'duplicate':
        data['symbol'] = ['A', 'A']
    elif case == 'null_key':
        data['symbol'] = ['A', None]
    target = (pl.DataFrame(data) if backend == 'pandas' else pd.DataFrame(data)) if case == 'other_backend' else make(data)
    with pytest.raises((KeyError, ValueError, TypeError)):
        ic_decay(features, 'x', features, [1], lambda frame, horizon: target)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('missing', [None, float('nan')], ids=['null', 'nan'])
def test_partial_ic_missing_values_and_minimum_breadth(backend, missing):
    """Should require four complete observations for a single covariate."""
    make = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    z, r, s = [-1., -1., 1., 1.], [-1., 1., -1., 1.], [-1., 1., 1., -1.]
    frame = make({'time': [1] * 5 + [2] * 4, 'symbol': list('ABCDE') + list('ABCD'),
                  'z': z + [missing] + z, 'x': [a + b for a, b in zip(z, r)] + [3.] + [1., 2., 3., missing],
                  'y': [2*a + b/2 + c for a, b, c in zip(z, r, s)] + [4.] + [1., 2., 3., 4.]})
    result = partial_information_coefficient(frame, 'x', 'y', 'z', corr_method='pearson', symbol_column='symbol')
    assert result['time'].to_list() == [1]
    assert result['partial_ic'].to_list() == pytest.approx([1 / sqrt(5)])


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_ic_public_outputs_schema_and_input_nonmutation(backend):
    """Should preserve input frames and return stable column order and date/numeric types."""
    z, r = [-1., -1., 1., 1.], [-1., 1., -1., 1.]
    rows = [{'time': day, 'symbol': str(i), 'x': z[i] + r[i], 'z': z[i],
             'y': 2*z[i] + (r[i] if day != 2 else -r[i])}
            for day in (1, 2, 3) for i in range(4)]
    make = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    frame = make(rows)
    before = frame.copy(deep=True) if backend == 'pandas' else frame.clone()
    target = frame[['time', 'symbol', 'y']]
    target_before = target.copy(deep=True) if backend == 'pandas' else target.clone()
    ordinary = compute_ic(frame, 'x', 'y', 'pearson', symbol_column='symbol')
    partial = partial_information_coefficient(frame, 'x', 'y', 'z', 'pearson', symbol_column='symbol')
    summary = ic_summary_table(frame, ['x'], 'y', 'pearson', symbol_column='symbol')
    partial_summary = partial_ic_summary_table(frame, ['x'], 'y', 'z', 'pearson', symbol_column='symbol')
    decay = ic_decay(frame, 'x', target, [1], lambda data, horizon: data, corr_method='pearson')
    decay_batch = ic_decay_summary_table(frame, ['x'], target, [1], lambda data, horizon: data,
                                         corr_method='pearson')
    curve_before = decay.table.copy(deep=True) if backend == 'pandas' else decay.table.clone()
    ic_decay_summary(decay.table)
    metrics_input = ordinary['ic'].copy() if backend == 'pandas' else ordinary['ic'].clone()
    compute_ic_metrics(ordinary['ic'])
    assert list(ordinary.columns) == ['time', 'ic']
    assert list(partial.columns) == ['time', 'partial_ic']
    assert list(summary.table.columns) == ['feature', 'mean', 'abs_mean', 'sign', 'std', 'stability',
                                           'pct_positive', 'quantile25', 'quantile50', 'quantile75',
                                           't_stat', 'p_value', 'nw_status', 'feature_group', 'n_obs']
    assert list(partial_summary.table.columns) == list(summary.table.columns) + ['covariates']
    assert list(decay_batch.table.columns) == ['feature', 'peak_horizon', 'peak_abs_ic', 'halflife_horizon',
                                               'last_significant_horizon', 'auc', 'feature_group']
    for result, name in [(ordinary, 'ic'), (partial, 'partial_ic'), (decay.ic_frames[1], 'ic')]:
        if backend == 'pandas':
            assert result['time'].dtype == frame['time'].dtype
            assert result[name].dtype == np.dtype('float64')
        else:
            assert result.schema == {'time': frame.schema['time'], name: pl.Float64}
    if backend == 'pandas':
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(target, target_before)
        pd.testing.assert_frame_equal(decay.table, curve_before)
        pd.testing.assert_series_equal(ordinary['ic'], metrics_input)
    else:
        assert frame.equals(before) and target.equals(target_before) and decay.table.equals(curve_before)
        assert ordinary['ic'].equals(metrics_input)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_all_undefined_decay_horizons_preserve_output_contract(backend):
    """Should retain every horizon and return undefined diagnostics when all ICs are undefined."""
    make = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    features = make({'time': [1, 1], 'symbol': ['A', 'B'], 'x': [1., 1.]})
    targets = make({'time': [1, 1], 'symbol': ['A', 'B'], 'y': [1., 2.]})
    result = ic_decay_summary_table(features, ['x'], targets, [1, 2], lambda data, horizon: data)
    table = result.decay_results['x'].table
    assert table['horizon'].to_list() == [1, 2]
    assert table['n_obs'].to_list() == [0, 0]
    assert table['nw_status'].to_list() == ['undefined', 'undefined']
    assert table['fdr_rejected'].to_list() == [False, False]
    assert all(len(frame) == 0 for frame in result.decay_results['x'].ic_frames.values())


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_empty_ic_output_preserves_date_and_float_schema(backend):
    """Should preserve timezone and float dtype when no paired IC can be computed."""
    frame = pd.DataFrame({'time': pd.to_datetime(['2024-01-01', '2024-01-02'], utc=True),
                          'x': [1., None], 'y': [None, 2.]})
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    result = compute_ic(frame, 'x', 'y')
    assert len(result) == 0 and list(result.columns) == ['time', 'ic']
    if backend == 'pandas':
        assert result['time'].dtype == frame['time'].dtype
        assert result['ic'].dtype == np.dtype('float64')
    else:
        assert result.schema == {'time': frame.schema['time'], 'ic': pl.Float64}


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('column', ['x', 'y', 'z'])
@pytest.mark.parametrize('invalid', ['text', float('inf')], ids=['nonnumeric', 'infinity'])
def test_partial_ic_rejects_invalid_selected_values(backend, column, invalid):
    """Should reject nonnumeric and infinite feature, target, and covariate values."""
    make = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    data = {'time': [1] * 4, 'x': [1., 2., 3., 4.], 'y': [1., 3., 2., 4.], 'z': [0., 1., 0., 1.]}
    data[column] = [invalid] * 4
    with pytest.raises(ValueError):
        partial_information_coefficient(make(data), 'x', 'y', 'z')


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_validate_ic_keys_enforces_numeric_values_dates_and_optional_identity(backend):
    """Should validate IC keys and finite values without requiring an asset column."""
    frame = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    valid = frame({'time': [1, 1], 'symbol': ['A', 'B'], 'x': [1., None]})
    assert _validate_ic_keys(valid, ['x'], 'time', 'symbol') is None
    for data, symbol, message in [
        ({'time': [1, None], 'x': [1., 2.]}, None, 'missing'),
        ({'time': [1, 1], 'symbol': ['A', 'A'], 'x': [1., 2.]}, 'symbol', 'unique'),
        ({'time': [1, 1], 'x': [1., float('inf')]}, None, 'finite'),
    ]:
        with pytest.raises(ValueError, match=message):
            _validate_ic_keys(frame(data), ['x'], 'time', symbol)
    with pytest.raises(KeyError):
        _validate_ic_keys(valid, ['absent'], 'time', None)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('undefined_date,expected_status', [(1, 'ok'), (3, 'internal_gap'), (5, 'ok')])
def test_ic_summary_keeps_reference_dates_for_newey_west(backend, undefined_date, expected_status):
    """Should suppress inference only when an undefined IC separates valid dates."""
    x = [-3., -1., 1., 3.]
    y = [x, [3., 1., -1., -3.], [1., -3., 3., -1.], x, [3., 1., -1., -3.]]
    rows = [{'time': date, 'symbol': str(i), 'x': value,
             'y': 1. if date == undefined_date else y[date - 1][i]}
            for date in range(1, 6) for i, value in enumerate(x)]
    frame = pd.DataFrame(rows) if backend == 'pandas' else pl.DataFrame(rows)
    result = ic_summary_table(frame, ['x'], 'y', 'pearson', symbol_column='symbol')
    row = result.table.to_dicts()[0] if backend == 'polars' else result.table.iloc[0].to_dict()
    assert row['n_obs'] == 4
    assert row['nw_status'] == expected_status
    if expected_status == 'internal_gap':
        assert np.isnan(row['t_stat']) and np.isnan(row['p_value'])
    else:
        assert np.isfinite(row['t_stat']) and np.isfinite(row['p_value'])


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_ic_decay_end_to_end_matches_hand_derived_hac_and_bh(backend):
    """Should match manually derived Pearson ICs, Bartlett HAC, and BH correction."""
    # Pearson ICs are [1, 0, -1] and [1, 1, -1]. For the latter,
    # lag-one Bartlett HAC gives t = 3/sqrt(20), and BH across both p-values
    # corrects the smaller p-value by a factor of two (capped at one).
    x = [-3., -1., 1., 3.]
    orthogonal = [1., -3., 3., -1.]
    inverse = [-v for v in x]
    first = [x, orthogonal, inverse]
    second = [x, x, inverse]
    features = [{'time': d + 1, 'symbol': str(i), 'feature': value}
                for d in range(3) for i, value in enumerate(x)]
    targets = [{'time': d + 1, 'symbol': str(i), 'target_1': first[d][i],
                'target_2': second[d][i]} for d in range(3) for i in range(4)]
    constructor = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    target_fn = lambda frame, horizon: frame[['time', 'symbol', f'target_{horizon}']]
    result = ic_decay(constructor(features), 'feature', constructor(targets), [2, 1], target_fn,
                      corr_method='pearson')
    rows = result.table.to_dicts() if backend == 'polars' else result.table.to_dict('records')
    assert [row['horizon'] for row in rows] == [1, 2]
    assert [row['mean'] for row in rows] == pytest.approx([0., 1 / 3])
    assert [row['t_stat'] for row in rows] == pytest.approx([0., 3 / sqrt(20)], abs=1e-12)
    p_value = 1 - erf((3 / sqrt(20)) / sqrt(2))
    assert [row['p_value'] for row in rows] == pytest.approx([1., p_value])
    assert [row['fdr_corrected_p_value'] for row in rows] == pytest.approx([1., min(1., 2 * p_value)])
    assert [row['nw_status'] for row in rows] == ['ok', 'ok']


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_partial_ic_gap_and_key_validation(backend):
    """Should suppress inference for an undefined interior date and validate partial IC inputs."""
    # z and r are orthogonal to each other and to the intercept, so the
    # residual feature is r and residual targets are r, zero, and -r.
    z = [-1., -1., 1., 1.]
    r = [-1., 1., -1., 1.]
    rows = [{'time': date, 'symbol': str(i), 'z': z[i], 'x': z[i] + r[i],
             'y': 2 * z[i] + (r[i] if date == 1 else -r[i]) if date != 2 else 1.}
            for date in range(1, 4) for i in range(4)]
    constructor = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    frame = constructor(rows)
    result = partial_ic_summary_table(frame, ['x'], 'y', 'z', corr_method='pearson', symbol_column='symbol')
    assert result.partial_ic_frames['x']['partial_ic'].to_list() == pytest.approx([1., -1.])
    row = result.table.to_dicts()[0] if backend == 'polars' else result.table.iloc[0].to_dict()
    assert row['nw_status'] == 'internal_gap'
    assert np.isnan(row['t_stat']) and np.isnan(row['p_value'])
    bad_rows = [dict(row) for row in rows]
    bad_rows[1]['symbol'] = '0'
    with pytest.raises(ValueError, match='unique'):
        partial_information_coefficient(constructor(bad_rows), 'x', 'y', 'z', symbol_column='symbol')
    bad_rows[1]['symbol'] = '1'
    bad_rows[0]['z'] = float('inf')
    with pytest.raises(ValueError, match='finite'):
        partial_information_coefficient(constructor(bad_rows), 'x', 'y', 'z')
    with pytest.raises(ValueError, match='corr_method'):
        partial_information_coefficient(frame, 'x', 'y', 'z', corr_method='invalid')


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_partial_ic_matches_nonperfect_orthogonal_residuals_with_missing(backend):
    """Should match an analytical partial correlation and exclude only incomplete pairs."""
    # z, r, and s are pairwise orthogonal to the intercept and each other.
    # Residual x is r; residual y is r/2+s, whose correlation is 1/sqrt(5).
    z = [-1., -1., 1., 1.]
    r = [-1., 1., -1., 1.]
    s = [-1., 1., 1., -1.]
    data = {'time': [1] * 5, 'z': z + [None],
            'x': [a + b for a, b in zip(z, r)] + [1.],
            'y': [2 * a + b / 2 + c for a, b, c in zip(z, r, s)] + [2.]}
    if backend == 'pandas':
        frame = pd.DataFrame(data).astype({'z': 'Float64', 'x': 'Float64', 'y': 'Float64'})
    else:
        frame = pl.DataFrame(data)
    result = partial_information_coefficient(frame, 'x', 'y', 'z', corr_method='pearson')
    assert result['partial_ic'].to_list() == pytest.approx([1 / sqrt(5)])


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_partial_ic_empty_output_preserves_date_type_and_input(backend):
    """Should preserve the timestamp schema and input when every partial IC is undefined."""
    frame = pd.DataFrame({'time': pd.to_datetime(['2024-01-01'] * 3, utc=True),
                          'x': [1., 2., 3.], 'y': [1., 2., 3.], 'z': [1., 1., 1.]})
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    before = frame.copy(deep=True) if backend == 'pandas' else frame.clone()
    result = partial_information_coefficient(frame, 'x', 'y', 'z')
    assert list(result.columns) == ['time', 'partial_ic']
    assert len(result) == 0
    if backend == 'pandas':
        assert result['time'].dtype == frame['time'].dtype
        assert result['partial_ic'].dtype == np.dtype('float64')
        pd.testing.assert_frame_equal(frame, before)
    else:
        assert result.schema == {'time': frame.schema['time'], 'partial_ic': pl.Float64}
        assert frame.equals(before)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_partial_ic_rejects_invalid_parameters_and_missing_keys(backend):
    """Should reject malformed partial IC specifications before processing dates."""
    constructor = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    data = {'time': [1, 1, 1, 1], 'symbol': ['A', 'B', 'C', None],
            'x': [1., 2., 3., 4.], 'y': [2., 3., 1., 4.], 'z': [0., 1., 0., 1.]}
    frame = constructor(data)
    with pytest.raises(ValueError, match='missing'):
        partial_information_coefficient(frame, 'x', 'y', 'z', symbol_column='symbol')
    with pytest.raises((ValueError, TypeError), match='min_n'):
        partial_information_coefficient(frame, 'x', 'y', 'z', min_n=1.5)
    with pytest.raises(ValueError, match='differ'):
        partial_information_coefficient(frame, 'x', 'y', 'z', ic_column='time')
    with pytest.raises(KeyError):
        partial_information_coefficient(frame, 'x', 'y', 'absent')
    data['time'][1] = None
    with pytest.raises(ValueError, match='missing'):
        partial_information_coefficient(constructor(data), 'x', 'y', 'z')


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_decay_detects_dates_lost_in_target_join(backend):
    """Should use feature dates as the reference even when a target date has no matches."""
    features = [{'time': date, 'symbol': str(i), 'x': float(i)}
                for date in range(1, 4) for i in range(4)]
    targets = [{'time': date, 'symbol': str(i), 'target': float(i if date == 1 else -i)}
               for date in (1, 3) for i in range(4)]
    constructor = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    result = ic_decay(constructor(features), 'x', constructor(targets), [1],
                      lambda frame, horizon: frame, corr_method='pearson')
    row = result.table.to_dicts()[0] if backend == 'polars' else result.table.iloc[0].to_dict()
    assert row['n_obs'] == 2
    assert row['nw_status'] == 'internal_gap'
    assert row['fdr_rejected'] is False or not row['fdr_rejected']
    assert row['p_value'] is None or np.isnan(row['p_value'])


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_decay_fdr_corrects_the_two_horizons_as_one_family(backend):
    """Should match hand-derived Bartlett HAC p-values and two-test BH correction."""
    # IC_1 alternates +1 and -1; IC_2 is +1 nineteen times, then -1.
    # With n=20 and two Bartlett lags, residual cross-products give
    # variance numerator 283/75 and t = 18/sqrt(283/75).
    x = [-3., -1., 1., 3.]
    features = [{'time': d, 'symbol': str(i), 'feature': value, 'other': value}
                for d in range(1, 21) for i, value in enumerate(x)]
    targets = [{'time': d, 'symbol': str(i),
                'target_1': value * (1 if d % 2 else -1),
                'target_2': value * (1 if d < 20 else -1)}
               for d in range(1, 21) for i, value in enumerate(x)]
    constructor = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    result = ic_decay(constructor(features), 'feature', constructor(targets), [1, 2],
                      lambda frame, horizon: frame[['time', 'symbol', f'target_{horizon}']],
                      corr_method='pearson')
    rows = result.table.to_dicts() if backend == 'polars' else result.table.to_dict('records')
    expected_t = 18 / sqrt(283 / 75)
    expected_p = erfc(expected_t / sqrt(2))
    assert rows[1]['t_stat'] == pytest.approx(expected_t)
    assert rows[1]['p_value'] == pytest.approx(expected_p, rel=1e-10, abs=0)
    assert rows[1]['fdr_corrected_p_value'] == pytest.approx(2 * expected_p, rel=1e-10, abs=0)
    assert rows[1]['fdr_rejected']
    assert not rows[0]['fdr_rejected']
    batch = ic_decay_summary_table(constructor(features), ['feature', 'other'], constructor(targets),
                                   [1, 2],
                                   lambda frame, horizon: frame[['time', 'symbol', f'target_{horizon}']],
                                   corr_method='pearson')
    for name in ('feature', 'other'):
        table = batch.decay_results[name].table
        corrected = table['fdr_corrected_p_value'].to_list()
        assert corrected[1] == pytest.approx(2 * expected_p, rel=1e-10, abs=0)


# ------------------------------------------------------
# Optional observation and cooperative cancellation
# ------------------------------------------------------
@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('method', ['pearson', 'spearman'])
def test_compute_ic_retains_known_native_values(cross_section_df_pandas, backend, method):
    """Should retain analytically known correlations of +1 and -1 using native grouped operations."""
    frame = cross_section_df_pandas if backend == 'pandas' else pl.from_pandas(cross_section_df_pandas)
    result = compute_ic(frame, 'feature', 'target', method)
    assert result['ic'].to_list() == pytest.approx([1.0, -1.0])


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_ic_preserves_undefined_date_semantics(backend):
    """Should omit undefined constant cross-sections on both backends."""
    frame = pd.DataFrame({'time': [1, 1, 2, 2], 'feature': [1., 2., 1., 1.], 'target': [1., 2., 3., 4.]})
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    result = compute_ic(frame, 'feature', 'target')
    assert result['ic'][0] == pytest.approx(1.0)
    assert len(result) == 1


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_partial_ic_reports_and_cancels_by_date(partial_ic_data_pandas, backend, monkeypatch):
    """Should stop partial IC between dates without entering the next residualization."""
    frame = partial_ic_data_pandas if backend == 'pandas' else pl.from_pandas(partial_ic_data_pandas)
    processed = []
    exception = ValueError('cancel')
    monkeypatch.setattr('alpha_research.evaluation.ic.partial_correlation',
                        lambda *args, **kwargs: processed.append(args[0]) or float('nan'))
    callback = lambda phase, done, total: (_ for _ in ()).throw(exception) if done == 1 else None
    with pytest.raises(ValueError) as caught:
        partial_information_coefficient(frame, 'feature', 'target', 'covariate_a', on_progress=callback)
    assert caught.value is exception
    assert len(processed) == 1


def test_summary_reports_ic_progress_by_feature(cross_section_df_pandas):
    """Should report existing feature-loop counts without adding date-loop events."""
    frame = cross_section_df_pandas.assign(second=cross_section_df_pandas['feature'] * -1)
    events = []
    result = ic_summary_table(frame, ['feature', 'second'], 'target', on_progress=lambda *event: events.append(event))
    assert result.ic_frames['feature']['ic'].tolist() == pytest.approx([1, -1])
    assert result.ic_frames['second']['ic'].tolist() == pytest.approx([-1, 1])
    assert {event[0] for event in events} == {'features'}
    assert {event[1] for event in events} == {0, 1, 2}
    assert events[-1] == ('features', 2, 2)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('exception_type', [ValueError, TypeError, RuntimeError])
def test_summary_cancels_between_atomic_ic_features(cross_section_df_pandas, backend, exception_type, monkeypatch):
    """Should finish the first grouped IC and stop before calculating a second feature."""
    frame = cross_section_df_pandas.assign(second=-cross_section_df_pandas['feature'])
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    original = compute_ic
    computed = []
    def grouped_ic(df, feature, target, corr_method, date_column):
        """Should observe native calls using only their original statistical arguments."""
        computed.append(feature)
        return original(df, feature, target, corr_method, date_column)
    monkeypatch.setattr('alpha_research.evaluation.ic.compute_ic', grouped_ic)
    exception = exception_type('cancel')
    callback = lambda phase, done, total: (_ for _ in ()).throw(exception) if done == 1 else None
    with pytest.raises(exception_type) as caught:
        ic_summary_table(frame, ['feature', 'second'], 'target', on_progress=callback)
    assert caught.value is exception
    assert computed == ['feature']


def test_target_generation_cancellation_preserves_sorted_horizons(monkeypatch):
    """Should generate only the first sorted horizon before cancellation without regenerating IDs or outputs."""
    frame = pd.DataFrame({'time': [1, 2], 'symbol': ['A', 'A'], 'value': [1., 2.]})
    generated = []
    exception = RuntimeError('cancel')
    target_fn = lambda data, horizon: generated.append(horizon) or data.copy()
    callback = lambda phase, done, total: (_ for _ in ()).throw(exception) if done == 1 else None
    with pytest.raises(RuntimeError) as caught:
        _generate_target_frames(frame, frame, [3, 1, 2], target_fn, on_progress=callback)
    assert caught.value is exception
    assert generated == [1]


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_ic_decay_progress_preserves_shared_targets_and_horizons(
        decay_features_df_pandas, decay_target_data_pandas, backend):
    """Should generate targets once and observe existing feature/horizon loops without changing output."""
    features = decay_features_df_pandas
    targets = decay_target_data_pandas
    if backend == 'polars':
        features, targets = pl.from_pandas(features), pl.from_pandas(targets)
    generated = []
    target_fn = lambda frame, horizon: generated.append(horizon) or frame[['time', 'symbol', f'target_{horizon}']]
    args = {'df_features': features, 'feature_list': ['feature_a', 'feature_b'], 'target_data': targets,
                'horizons': [2, 1], 'target_fn': target_fn}
    expected = ic_decay_summary_table(**args)
    generated.clear()
    events = []
    result = ic_decay_summary_table(**args, on_progress=lambda *event: events.append(event))
    assert generated == [1, 2]
    for feature in ['feature_a', 'feature_b']:
        if backend == 'pandas':
            pd.testing.assert_frame_equal(result.decay_results[feature].table, expected.decay_results[feature].table)
        else:
            assert result.decay_results[feature].table.equals(expected.decay_results[feature].table)
        assert (f'feature/{feature}/horizons', 2, 2) in events
    assert {phase for phase, _, _ in events} == {'target_horizons', 'features',
        'feature/feature_a/horizons', 'feature/feature_b/horizons'}
    assert events[-1] == ('features', 2, 2)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_partial_ic_observation_preserves_analytical_residuals(backend):
    """Should retain correlations of +1 and -1 after removing an orthogonal control."""
    # z is orthogonal to r and to the intercept. Residuals are exactly r and +/-r.
    z = [-1., -1., 1., 1.]
    r = [-1., 1., -1., 1.]
    frame = pd.DataFrame({'time': [1] * 4 + [2] * 4, 'control': z * 2,
                          'feature': [a + b for a, b in zip(z, r)] * 2,
                          'target': [2 * a + b for a, b in zip(z, r)] + [2 * a - b for a, b in zip(z, r)]})
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    events = []
    result = partial_information_coefficient(frame, 'feature', 'target', 'control', corr_method='pearson',
                                             on_progress=lambda *event: events.append(event))
    assert result['partial_ic'].to_list() == pytest.approx([1., -1.])
    assert events[-1] == ('partial_ic_dates', 2, 2)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('method,expected', [('spearman', 5 / 6), ('pearson', 2 / (5.5 ** 0.5))])
def test_native_ic_retains_ties_custom_columns_and_utc(backend, method, expected):
    """Should retain hand-derived tied correlations, UTC keys, custom output names, and untouched input."""
    # Centered average ranks have dot product 3.75 and squared norms 4.5;
    # centered raw values have dot product 2 and squared norms 2.75 and 2.
    frame = pd.DataFrame({'date': pd.to_datetime(['2024-01-02'] * 4 + ['2024-01-01'] * 4, utc=True),
                          'x': [1., 1., 2., 3.] * 2, 'y': [1., 2., 2., 3.] * 2})
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    before = frame.copy(deep=True) if backend == 'pandas' else frame.clone()
    result = compute_ic(frame, 'x', 'y', method, 'date', 'association')
    assert result['association'].to_list() == pytest.approx([expected, expected])
    if backend == 'pandas':
        pd.testing.assert_frame_equal(frame, before)
    else:
        assert frame.equals(before)


@pytest.mark.parametrize('method', ['pearson', 'spearman'])
@pytest.mark.parametrize('values', [[], [None, None], [float('nan'), 2.], [1., 1.], [1., 2.]])
def test_polars_ic_preserves_native_null_and_nan_semantics(method, values):
    """Should omit cross-sections without two nonconstant complete pairs."""
    frame = pl.DataFrame({'time': [1] * len(values), 'x': values, 'y': [1., 2.][:len(values)]},
                          schema={'time': pl.Int64, 'x': pl.Float64, 'y': pl.Float64})
    if not values:
        with pytest.raises(ValueError):
            compute_ic(frame, 'x', 'y', method)
        return
    result = compute_ic(frame, 'x', 'y', method)
    if values == [1., 2.]:
        assert result['ic'].to_list() == pytest.approx([1.0])
    else:
        assert result.is_empty()


def test_polars_ic_preserves_empty_joint_pairs_schema():
    """Should retain the native empty schema when individually valid columns have no joint pairs."""
    frame = pl.DataFrame({'time': [1, 2], 'x': [None, 2.], 'y': [1., None]})
    result = compute_ic(frame, 'x', 'y')
    assert result.schema == {'time': pl.Int64, 'ic': pl.Float64}
    assert result.is_empty()


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('method', ['pearson', 'spearman'])
def test_ic_ignores_missing_pairs_and_rejects_duplicate_keys(backend, method):
    """Should calculate the known correlation from complete pairs and validate asset identity."""
    data = {'time': [1, 1, 1], 'symbol': ['A', 'B', 'C'],
            'x': [1., float('nan'), 3.], 'y': [1., 2., 3.]}
    frame = pd.DataFrame(data) if backend == 'pandas' else pl.DataFrame(data)
    assert compute_ic(frame, 'x', 'y', method, symbol_column='symbol')['ic'].to_list() == pytest.approx([1.])
    data['symbol'][1] = 'A'
    duplicate = pd.DataFrame(data) if backend == 'pandas' else pl.DataFrame(data)
    with pytest.raises(ValueError, match='unique'):
        compute_ic(duplicate, 'x', 'y', method, symbol_column='symbol')


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_ic_summary_rejects_duplicate_asset_keys_when_requested(backend):
    """Should apply the asset identity contract to each summary feature."""
    data = {'time': [1, 1], 'symbol': ['A', 'A'], 'x': [1., 2.], 'y': [1., 2.]}
    frame = pd.DataFrame(data) if backend == 'pandas' else pl.DataFrame(data)
    with pytest.raises(ValueError, match='unique'):
        ic_summary_table(frame, ['x'], 'y', symbol_column='symbol')


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_ic_metrics_uses_only_valid_values(backend):
    """Should use two valid observations for every scalar and preserve missing positions."""
    series = pd.Series([0.1, np.nan, -0.2]) if backend == 'pandas' else pl.Series([0.1, None, -0.2])
    metrics = compute_ic_metrics(series)
    assert metrics.mean == pytest.approx(-0.05)
    assert metrics.abs_mean == pytest.approx(0.15)
    assert metrics.pct_positive == pytest.approx(0.5)
    assert len(metrics.adjusted_series) == 3
    with pytest.raises(ValueError, match='finite'):
        compute_ic_metrics(pd.Series([0.1, np.inf]))


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_decay_summary_rejects_fractional_horizons(backend):
    """Should reject fractional horizons before they can be truncated."""
    data = {'horizon': [1.2, 2.4], 'mean': [.2, .1], 'fdr_rejected': [False, True]}
    frame = pd.DataFrame(data) if backend == 'pandas' else pl.DataFrame(data)
    with pytest.raises(ValueError, match='positive integers'):
        ic_decay_summary(frame)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_decay_summary_rejects_missing_significance_flags(backend):
    """Should reject missing significance flags instead of interpreting them as true."""
    data = {'horizon': [1, 2], 'mean': [.2, .1], 'fdr_rejected': [False, None]}
    frame = pd.DataFrame(data) if backend == 'pandas' else pl.DataFrame(data)
    with pytest.raises(ValueError, match='boolean'):
        ic_decay_summary(frame)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_ic_rejects_infinite_values_and_missing_dates(backend):
    """Should reject nonfinite inputs and missing dates for either backend."""
    constructor = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    with pytest.raises(ValueError, match='finite'):
        compute_ic(constructor({'time': [1, 1], 'x': [1., np.inf], 'y': [1., 2.]}), 'x', 'y')
    with pytest.raises(ValueError, match='missing'):
        compute_ic(constructor({'time': [1, None], 'x': [1., 2.], 'y': [1., 2.]}), 'x', 'y')


def test_ic_invalid_method_on_constant_input():
    """Should reject an invalid correlation method even for constant inputs."""
    with pytest.raises(ValueError, match='corr_method'):
        information_coefficient(pd.Series([1., 1.]), pd.Series([1., 2.]), 'invalid')


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_ic_summary_all_missing_feature_does_not_block_others(backend):
    """Should retain an undefined feature and calculate the other feature independently."""
    data = {'time': [1, 1, 2, 2], 'missing': [None] * 4,
            'feature': [1., 2., 2., 1.], 'target': [1., 2., 1., 2.]}
    frame = pd.DataFrame(data) if backend == 'pandas' else pl.DataFrame(data)
    result = ic_summary_table(frame, ['missing', 'feature'], 'target')
    assert len(result.ic_frames['missing']) == 0
    assert result.ic_frames['feature']['ic'].to_list() == pytest.approx([1., -1.])


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_decay_summary_all_missing_feature_does_not_block_others(backend):
    """Should preserve undefined decay diagnostics while calculating the valid feature."""
    features = {'time': [1, 1, 2, 2], 'symbol': ['A', 'B', 'A', 'B'],
                'missing': [None] * 4, 'feature': [1., 2., 2., 1.]}
    targets = {'time': [1, 1, 2, 2], 'symbol': ['A', 'B', 'A', 'B'],
               'target': [1., 2., 1., 2.]}
    constructor = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    result = ic_decay_summary_table(constructor(features), ['missing', 'feature'],
                                    constructor(targets), [1], lambda frame, horizon: frame)
    assert len(result.decay_results['missing'].ic_frames[1]) == 0
    assert result.decay_results['feature'].ic_frames[1]['ic'].to_list() == pytest.approx([1., -1.])


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_decay_without_matching_keys_returns_undefined_results(backend):
    """Should return an empty IC series when feature and target keys do not overlap."""
    constructor = pd.DataFrame if backend == 'pandas' else pl.DataFrame
    features = constructor({'time': [1, 1], 'symbol': ['A', 'B'], 'feature': [1., 2.]})
    targets = constructor({'time': [2, 2], 'symbol': ['A', 'B'], 'target': [1., 2.]})
    result = ic_decay(features, 'feature', targets, [1], lambda frame, horizon: frame)
    assert len(result.ic_frames[1]) == 0
    mean = result.table['mean'][0]
    assert mean is None or np.isnan(mean)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_partial_ic_summary_all_missing_feature_does_not_block_others(partial_ic_data_pandas, backend):
    """Should retain an undefined partial IC feature and compute a valid feature."""
    frame = partial_ic_data_pandas.assign(missing=np.nan)
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    result = partial_ic_summary_table(frame, ['missing', 'feature'], 'target', 'covariate_a')
    assert len(result.partial_ic_frames['missing']) == 0
    assert len(result.partial_ic_frames['feature']) == 8


# ------------------------------------------------------
# fixtures
# ------------------------------------------------------
@pytest.fixture
def perfect_corr():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    return x, x.copy()

@pytest.fixture
def perfect_inverse():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
    return x, y

@pytest.fixture
def cross_section_df_pandas():
    """
    Two dates, with 5 assets each.
    date 1: perfect correlation
    date 2: perfect inverse correlation
    """
    return pd.DataFrame({
        'time': ['2024-01-01'] * 5 + ['2024-01-02'] * 5,
        'feature': [1.0, 2.0, 3.0, 4.0, 5.0,
                    1.0, 2.0, 3.0, 4.0, 5.0],
        'target':  [1.0, 2.0, 3.0, 4.0, 5.0,
                    5.0, 4.0, 3.0, 2.0, 1.0],
    })

@pytest.fixture
def cross_section_df_polars():
    return pl.DataFrame({
        'time': [
            '2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01',
            '2024-01-02', '2024-01-02', '2024-01-02', '2024-01-02', '2024-01-02',
        ],
        'feature': [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        'target':  [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    }).with_columns(pl.col('time').str.to_datetime())

# values were computed manually and checked
@pytest.fixture
def positive_ic_series():
    return pd.Series([0.04, 0.08, 0.06, 0.10, 0.02])
    # mean    = 0.060
    # abs_mean = mean(abs) = 0.060
    # std     = 0.03162  (ddof=1)
    # stability = abs(mean)/std = 0.060/0.03162 = approx 1.897
    # sign    = 1
    # pct_pos = 1.0

@pytest.fixture
def negative_ic_series():
    return pd.Series([-0.04, -0.08, -0.06, -0.10, -0.02])
    # mean     = -0.060
    # abs_mean = 0.060
    # sign     = -1
    # pct_pos  = 1.0  (should be all positive)

@pytest.fixture
def mixed_ic_series():
    return pd.Series([0.10, -0.10, 0.06, -0.06, 0.04])
    # mean     = 0.008  → sign = 1
    # abs_mean = mean([0.10, 0.10, 0.06, 0.06, 0.04]) = 0.072
    # pct_pos on adjusted series: [0.10, -0.10, 0.06, -0.06, 0.04] = 0.6

@pytest.fixture
def multi_feature_df_pandas():
    """
    DataFrame with 3 features and 2 dates, 5 assets each.
    feature_a: perfect positive correlation on both dates
    feature_b: perfect inverse correlation on both dates
    feature_c: constant values

    Precomputed:
        feature_a: mean IC ≈ 1.0
        feature_b: mean IC ≈ -1.0
        feature_c: mean IC = nan (constant)
    """
    return pd.DataFrame({
        'time':      ['2024-01-01'] * 5 + ['2024-01-02'] * 5,
        'feature_a': [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
        'feature_b': [5.0, 4.0, 3.0, 2.0, 1.0] * 2,
        'feature_c': [1.0, 1.0, 1.0, 1.0, 1.0] * 2,
        'target':    [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
    })


@pytest.fixture
def decay_features_df_pandas():
    return pd.DataFrame({
        'time': ['2024-01-01'] * 5 + ['2024-01-02'] * 5 + ['2024-01-03'] * 5,
        'symbol': ['A', 'B', 'C', 'D', 'E'] * 3,
        'feature_a': [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        'feature_b': [5.0, 4.0, 3.0, 2.0, 1.0] * 3,
    })


@pytest.fixture
def decay_target_data_pandas(decay_features_df_pandas):
    return pd.DataFrame({
        'time': decay_features_df_pandas['time'],
        'symbol': decay_features_df_pandas['symbol'],
        'target_1': [
            1.0, 2.0, 3.0, 4.0, 5.0,
            1.0, 2.0, 3.0, 5.0, 4.0,
            5.0, 4.0, 3.0, 2.0, 1.0,
        ],
        'target_2': [
            5.0, 4.0, 3.0, 2.0, 1.0,
            4.0, 5.0, 3.0, 2.0, 1.0,
            1.0, 2.0, 3.0, 4.0, 5.0,
        ],
    })


@pytest.fixture
def decay_features_df_polars(decay_features_df_pandas):
    return pl.from_pandas(decay_features_df_pandas)


@pytest.fixture
def decay_target_data_polars(decay_target_data_pandas):
    return pl.from_pandas(decay_target_data_pandas)


@pytest.fixture
def decay_summary_curve_pandas():
    return pd.DataFrame({
        'horizon': [4, 1, 3, 2],
        'mean': [0.04, 0.10, 0.08, 0.20],
        'fdr_rejected': [False, False, True, True],
    })


# ------------------------------------------------------
# information_coefficient
# ------------------------------------------------------
def test_ic_perfect_positive(perfect_corr):
    """Should return 1.0 for perfect positive correlation with spearman corr_method."""
    x, y = perfect_corr
    assert information_coefficient(x, y) == pytest.approx(1.0)

def test_ic_perfect_inverse(perfect_inverse):
    """Should return -1.0 for perfect inverse correlation with spearman corr_method."""
    x, y = perfect_inverse
    assert information_coefficient(x, y) == pytest.approx(-1.0)

def test_ic_pearson(perfect_corr):
    """Should return 1.0 for perfect positive correlation with pearson corr_method."""
    x, y = perfect_corr
    assert information_coefficient(x, y, corr_method='pearson') == pytest.approx(1.0)

def test_ic_length_mismatch_raises():
    """Should raise ValueError on length mismatch."""
    with pytest.raises(ValueError, match="len"):
        information_coefficient(pd.Series([1, 2]), pd.Series([1]))

def test_ic_invalid_method_raises():
    """Should raise ValueError for unsupported corr_method."""
    x = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="corr_method"):
        information_coefficient(x, x, corr_method='kendall')


def test_ic_constant_input_returns_nan():
    """Should return NaN directly when an input Series is constant."""
    x = pd.Series([1.0, 1.0, 1.0])
    y = pd.Series([1.0, 2.0, 3.0])

    result = information_coefficient(x, y)

    assert np.isnan(result)


# ------------------------------------------------------
# compute_ic
# ------------------------------------------------------
def test_compute_ic_pandas_shape(cross_section_df_pandas):
    """Should return one IC value per date and ic should be one of the two columns."""
    result = compute_ic(cross_section_df_pandas, 'feature', 'target')
    assert result.shape == (2, 2)
    assert 'ic' in result.columns

def test_compute_ic_pandas_values(cross_section_df_pandas):
    """Should return perfect positive correlation in first date and perfect inverse correlation in second date."""
    result = compute_ic(cross_section_df_pandas, 'feature', 'target')
    ic_values = result.set_index('time')['ic']
    assert ic_values['2024-01-01'] == pytest.approx(1.0)
    assert ic_values['2024-01-02'] == pytest.approx(-1.0)

def test_compute_ic_polars_shape(cross_section_df_polars):
    """Should return shape (2,2) because cross_section_df_polars has two dates."""
    result = compute_ic(cross_section_df_polars, 'feature', 'target')
    assert result.shape == (2, 2)

def test_compute_ic_polars_values(cross_section_df_polars):
    """Should return perfect positive correlation in first date and perfect inverse correlation in second date."""
    result = compute_ic(cross_section_df_polars, 'feature', 'target')
    result_sorted = result.sort('time')
    assert result_sorted['ic'][0] == pytest.approx(1.0)
    assert result_sorted['ic'][1] == pytest.approx(-1.0)

def test_compute_ic_pandas_polars_consistency(cross_section_df_pandas, cross_section_df_polars):
    """Backends should return numerically identical IC values."""
    res_pd = compute_ic(cross_section_df_pandas, 'feature', 'target').sort_values('time')
    res_pl = compute_ic(cross_section_df_polars, 'feature', 'target').sort('time')
    np.testing.assert_allclose(
        res_pd['ic'].values,
        res_pl['ic'].to_numpy(),
        rtol=1e-6
    )

def test_compute_ic_missing_column_raises(cross_section_df_pandas):
    """Should raise KeyError for missing columns."""
    with pytest.raises(KeyError):
        compute_ic(cross_section_df_pandas, 'nonexistent', 'target')

def test_compute_ic_invalid_df_raises():
    """Should raise ValueError with unsupported input types."""
    with pytest.raises(TypeError, match="Pandas or Polars"):
        compute_ic([[1, 2], [3, 4]], 'feature', 'target')

def test_compute_ic_custom_ic_column(cross_section_df_pandas):
    """Should return custom ic column name correctly."""
    result = compute_ic(cross_section_df_pandas, 'feature', 'target', ic_column='my_ic')
    assert 'my_ic' in result.columns


# ------------------------------------------------------
# ICMetrics and compute_ic_metrics
# ------------------------------------------------------
def test_ic_metrics_mean_preserves_sign(positive_ic_series, negative_ic_series):
    """Should preserve sign in mean."""
    assert compute_ic_metrics(positive_ic_series).mean > 0
    assert compute_ic_metrics(negative_ic_series).mean < 0

def test_ic_metrics_abs_mean_always_positive(negative_ic_series):
    """Should return positive abs_mean for negative IC series."""
    assert compute_ic_metrics(negative_ic_series).abs_mean > 0

def test_ic_metrics_sign_positive_series(positive_ic_series):
    """Should return sign=1 for positive IC series."""
    assert compute_ic_metrics(positive_ic_series).sign == 1

def test_ic_metrics_sign_negative_series(negative_ic_series):
    """Should return sign=-1 for negative IC series."""
    assert compute_ic_metrics(negative_ic_series).sign == -1

def test_ic_metrics_sign_zero_mean():
    """Should return sign=1 when mean IC is exactly zero."""
    series = pd.Series([0.1, -0.1, 0.1, -0.1])
    assert compute_ic_metrics(series).sign == 1

def test_ic_metrics_stability_is_nan_when_std_zero():
    """Should return nan stability when all IC values are identical."""
    series = pd.Series([0.05, 0.05, 0.05])
    assert np.isnan(compute_ic_metrics(series).stability)

def test_ic_metrics_pct_positive_on_adjusted_series(negative_ic_series):
    """pct_positive should be high for consistently negative IC (adjusted series)."""
    metrics = compute_ic_metrics(negative_ic_series)
    assert metrics.pct_positive == pytest.approx(1.0)

def test_ic_metrics_pct_positive_mixed(mixed_ic_series):
    """Should return 0.6 for mixed series."""
    assert compute_ic_metrics(mixed_ic_series).pct_positive == pytest.approx(0.6)

def test_ic_metrics_quantiles_ordered(negative_ic_series):
    """Should return q25 <= q50 <= q75 for negative IC (adjusted series)."""
    q = compute_ic_metrics(negative_ic_series).quantiles
    assert q['q25'] <= q['q50'] <= q['q75']

def test_ic_metrics_adjusted_series_positive_for_negative_input(negative_ic_series):
    """Adjusted series should be all positive for consistently negative IC."""
    metrics = compute_ic_metrics(negative_ic_series)
    assert (metrics.adjusted_series > 0).all()

def test_ic_metrics_original_series_unchanged(negative_ic_series):
    """Original series should stay negative for consistently negative IC."""
    metrics = compute_ic_metrics(negative_ic_series)
    assert (metrics.original_series < 0).all()

# Specific values tests as presented on fixtures
def test_ic_metrics_mean_value_on_positive_series(positive_ic_series):
    """Should compute mean correctly."""
    assert compute_ic_metrics(positive_ic_series).mean == pytest.approx(0.060)

def test_ic_metrics_mean_value_on_negative_series(negative_ic_series):
    """Should compute mean correctly."""
    assert compute_ic_metrics(negative_ic_series).mean == pytest.approx(-0.060)

def test_ic_metrics_abs_mean_is_mean_of_abs(mixed_ic_series):
    """abs_mean should be mean(abs(IC)), not abs(mean(IC))."""
    assert compute_ic_metrics(mixed_ic_series).abs_mean == pytest.approx(0.072)

def test_ic_metrics_abs_mean_differs_from_abs_of_mean(mixed_ic_series):
    """abs_mean and abs(mean) should differ for mixed series."""
    metrics = compute_ic_metrics(mixed_ic_series)
    assert metrics.abs_mean != pytest.approx(abs(metrics.mean))

def test_ic_metrics_std_value(positive_ic_series):
    """Should compute sample std correctly (ddof=1)."""
    assert compute_ic_metrics(positive_ic_series).std == pytest.approx(0.03162, rel=1e-3)

def test_ic_metrics_stability_formula(positive_ic_series):
    """Should compute stability metric correctly."""
    metrics = compute_ic_metrics(positive_ic_series)
    assert metrics.stability == pytest.approx(1.897, rel=1e-3)


# ------------------------------------------------------
# ic_summary_table
# ------------------------------------------------------
# ic_summary_table.table
def test_ic_summary_table_shape(cross_section_df_pandas):
    """table should return one row per feature."""
    result = ic_summary_table(cross_section_df_pandas, ['feature'], 'target').table
    assert result.shape[0] == 1

def test_ic_summary_table_columns(cross_section_df_pandas):
    """table should return expected columns."""
    result = ic_summary_table(cross_section_df_pandas, ['feature'], 'target').table
    expected = {'feature', 'mean', 'abs_mean', 'sign', 'std', 'stability',
                'pct_positive', 'quantile25', 'quantile50', 'quantile75',
                't_stat', 'p_value', 'nw_status', 'feature_group', 'n_obs'}
    assert set(result.columns) == expected


def test_ic_summary_table_n_obs_matches_number_of_ic_observations(
        cross_section_df_pandas,
):
    """n_obs should equal the number of cross-sectional IC observations."""
    result = ic_summary_table(
        cross_section_df_pandas,
        ['feature'],
        'target',
    )

    assert result.table['n_obs'].iloc[0] == 2
    assert result.table['n_obs'].iloc[0] == len(result.ic_frames['feature'])


def test_ic_summary_table_feature_group_default(cross_section_df_pandas):
    """table should label all features as ungrouped when feature_groups is None."""
    result = ic_summary_table(cross_section_df_pandas, ['feature'], 'target').table
    assert result['feature_group'].iloc[0] == 'ungrouped'

def test_ic_summary_table_feature_group_assigned(cross_section_df_pandas):
    """table should assign correct group when feature_groups is provided."""
    result = ic_summary_table(
        cross_section_df_pandas, ['feature'], 'target',
        feature_groups={'feature': 'momentum'}
    ).table
    assert result['feature_group'].iloc[0] == 'momentum'

def test_ic_summary_table_feature_group_unknown(cross_section_df_pandas):
    """table should label as ungrouped if feature not in feature_groups."""
    result = ic_summary_table(
        cross_section_df_pandas, ['feature'], 'target',
        feature_groups={'other_feature': 'momentum'}
    ).table
    assert result['feature_group'].iloc[0] == 'ungrouped'

def test_ic_summary_table_empty_list_raises(cross_section_df_pandas):
    """table should raise ValueError for empty feature_list."""
    with pytest.raises(ValueError, match="empty"):
        ic_summary_table(cross_section_df_pandas, [], 'target').table

def test_ic_summary_table_polars(cross_section_df_polars):
    """table should return pl.DataFrame for polars input."""
    result = ic_summary_table(cross_section_df_polars, ['feature'], 'target').table
    assert isinstance(result, pl.DataFrame)

def test_ic_summary_table_pandas_polars_consistency(cross_section_df_pandas, cross_section_df_polars):
    """table should return numerically identical results for pandas and polars input."""
    res_pd = ic_summary_table(cross_section_df_pandas, ['feature'], 'target').table
    res_pl = ic_summary_table(cross_section_df_polars, ['feature'], 'target').table
    assert res_pd['mean'].iloc[0] == pytest.approx(res_pl['mean'][0], rel=1e-6)
    assert res_pd['t_stat'].iloc[0] == pytest.approx(res_pl['t_stat'][0], rel=1e-6)

# Multiple features tests
def test_ic_summary_table_multiple_features_shape(multi_feature_df_pandas):
    """table should return one row per feature: 3 features = 3 rows."""
    result = ic_summary_table(
        multi_feature_df_pandas,
        ['feature_a', 'feature_b', 'feature_c'],
        'target'
    ).table
    assert result.shape[0] == 3

def test_ic_summary_table_multiple_features_values(multi_feature_df_pandas):
    """table should compute correct mean IC per feature."""
    result = ic_summary_table(
        multi_feature_df_pandas,
        ['feature_a', 'feature_b', 'feature_c'],
        'target'
    ).table

    means = dict(zip(result["feature"], result["mean"]))

    assert means["feature_a"] == pytest.approx(1.0)
    assert means["feature_b"] == pytest.approx(-1.0)
    assert np.isnan(means["feature_c"])

def test_ic_summary_table_multiple_feature_groups(multi_feature_df_pandas):
    """table should assign correct groups to multiple features."""
    groups = {
        'feature_a': 'momentum',
        'feature_b': 'momentum',
        'feature_c': 'volatility',
    }
    result = ic_summary_table(
        multi_feature_df_pandas,
        ['feature_a', 'feature_b', 'feature_c'],
        'target',
        feature_groups=groups
    ).table

    feature_groups_result = dict(zip(result["feature"], result["feature_group"]))

    assert feature_groups_result["feature_a"] == "momentum"
    assert feature_groups_result["feature_b"] == "momentum"
    assert feature_groups_result["feature_c"] == "volatility"


def test_ic_summary_table_feature_order_preserved(multi_feature_df_pandas):
    """table should preserve feature order from input list."""
    features = ['feature_c', 'feature_a', 'feature_b']
    result = ic_summary_table(multi_feature_df_pandas, features, 'target').table
    assert list(result['feature']) == features


# ic_summary_table.ic_frames
def test_ic_summary_table_ic_frames_keys(cross_section_df_pandas):
    """ic_frames should have one key per feature in feature_list."""
    result = ic_summary_table(cross_section_df_pandas, ['feature'], 'target')
    assert set(result.ic_frames.keys()) == {'feature'}

def test_ic_summary_table_ic_frames_columns(cross_section_df_pandas):
    """Each ic_frames DataFrame should have time and ic columns."""
    result = ic_summary_table(cross_section_df_pandas, ['feature'], 'target')
    assert 'time' in result.ic_frames['feature'].columns
    assert 'ic' in result.ic_frames['feature'].columns

def test_ic_summary_table_ic_frames_multiple_features(multi_feature_df_pandas):
    """ic_frames should have one key per feature."""
    features = ['feature_a', 'feature_b', 'feature_c']
    result = ic_summary_table(multi_feature_df_pandas, features, 'target')
    assert set(result.ic_frames.keys()) == set(features)

def test_ic_summary_table_ic_frames_mean_consistent_with_table(cross_section_df_pandas):
    """Mean of ic_frames series should match mean in summary table."""
    result = ic_summary_table(cross_section_df_pandas, ['feature'], 'target')
    ic_mean_from_series = result.ic_frames['feature']['ic'].mean()
    ic_mean_from_table = result.table['mean'].iloc[0]
    assert ic_mean_from_series == pytest.approx(ic_mean_from_table, rel=1e-6)

def test_ic_summary_table_ic_frames_polars(cross_section_df_polars):
    """ic_frames should contain pl.DataFrame for polars input."""
    result = ic_summary_table(cross_section_df_polars, ['feature'], 'target')
    assert isinstance(result.ic_frames['feature'], pl.DataFrame)


# ------------------------------------------------------
# _generate_target_frames
# ------------------------------------------------------
def test_generate_target_frames_calls_target_fn_once_per_sorted_horizon(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should generate one target DataFrame for each sorted horizon."""
    calls = []

    def target_fn(target_data, horizon):
        calls.append(horizon)
        return target_data[['time', 'symbol', f'target_{horizon}']]

    result = _generate_target_frames(
        df_feature=decay_features_df_pandas,
        target_data=decay_target_data_pandas,
        horizons=[2, 1],
        target_fn=target_fn,
    )

    assert calls == [1, 2]
    assert list(result) == [1, 2]
    assert list(result[1].columns) == ['time', 'symbol', 'target_1']
    assert list(result[2].columns) == ['time', 'symbol', 'target_2']


def test_generate_target_frames_rejects_duplicate_horizons(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject duplicate horizons before calling target_fn."""
    calls = []

    def target_fn(target_data, horizon):
        calls.append(horizon)
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='horizons must not contain duplicates'):
        _generate_target_frames(
            df_feature=decay_features_df_pandas,
            target_data=decay_target_data_pandas,
            horizons=[1, 1],
            target_fn=target_fn,
        )

    assert calls == []


def test_generate_target_frames_rejects_empty_horizons(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject an empty horizon list before calling target_fn."""
    calls = []

    def target_fn(target_data, horizon):
        calls.append(horizon)
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='horizons must not be empty'):
        _generate_target_frames(
            df_feature=decay_features_df_pandas,
            target_data=decay_target_data_pandas,
            horizons=[],
            target_fn=target_fn,
        )

    assert calls == []


def test_generate_target_frames_rejects_none_horizon(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject None as a horizon value."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='positive integers'):
        _generate_target_frames(
            df_feature=decay_features_df_pandas,
            target_data=decay_target_data_pandas,
            horizons=[None],
            target_fn=target_fn,
        )


def test_generate_target_frames_rejects_nan_horizon(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject NaN as a horizon value."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='positive integers'):
        _generate_target_frames(
            df_feature=decay_features_df_pandas,
            target_data=decay_target_data_pandas,
            horizons=[np.nan],
            target_fn=target_fn,
        )


def test_generate_target_frames_rejects_float_horizon(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject a floating-point horizon value."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='positive integers'):
        _generate_target_frames(
            df_feature=decay_features_df_pandas,
            target_data=decay_target_data_pandas,
            horizons=[1.0],
            target_fn=target_fn,
        )


def test_generate_target_frames_rejects_string_horizon(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject a string horizon value."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='positive integers'):
        _generate_target_frames(
            df_feature=decay_features_df_pandas,
            target_data=decay_target_data_pandas,
            horizons=['1'],
            target_fn=target_fn,
        )


def test_generate_target_frames_rejects_boolean_horizon(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject a boolean horizon value."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='positive integers'):
        _generate_target_frames(
            df_feature=decay_features_df_pandas,
            target_data=decay_target_data_pandas,
            horizons=[True],
            target_fn=target_fn,
        )


def test_generate_target_frames_rejects_non_positive_horizons(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject zero and negative horizon values."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='positive integers'):
        _generate_target_frames(
            df_feature=decay_features_df_pandas,
            target_data=decay_target_data_pandas,
            horizons=[0],
            target_fn=target_fn,
        )

    with pytest.raises(ValueError, match='positive integers'):
        _generate_target_frames(
            df_feature=decay_features_df_pandas,
            target_data=decay_target_data_pandas,
            horizons=[-1],
            target_fn=target_fn,
        )


def test_generate_target_frames_pandas_polars_consistency(
        decay_features_df_pandas,
        decay_target_data_pandas,
        decay_features_df_polars,
        decay_target_data_polars,
):
    """Should generate equivalent target mappings for both backends."""
    def pandas_target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    def polars_target_fn(target_data, horizon):
        return target_data.select(['time', 'symbol', f'target_{horizon}'])

    pandas_result = _generate_target_frames(
        df_feature=decay_features_df_pandas,
        target_data=decay_target_data_pandas,
        horizons=[2, 1],
        target_fn=pandas_target_fn,
    )
    polars_result = _generate_target_frames(
        df_feature=decay_features_df_polars,
        target_data=decay_target_data_polars,
        horizons=[2, 1],
        target_fn=polars_target_fn,
    )

    assert list(pandas_result) == list(polars_result) == [1, 2]

    for horizon in pandas_result:
        pd.testing.assert_frame_equal(
            pandas_result[horizon].reset_index(drop=True),
            polars_result[horizon].to_pandas().reset_index(drop=True),
            check_dtype=False,
        )


# ------------------------------------------------------
# _ic_decay_from_target_frames
# ------------------------------------------------------
def test_ic_decay_from_target_frames_uses_pre_generated_targets(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should compute decay directly from the supplied target mapping."""
    target_frames = {
        1: decay_target_data_pandas[['time', 'symbol', 'target_1']],
        2: decay_target_data_pandas[['time', 'symbol', 'target_2']],
    }

    result = _ic_decay_from_target_frames(
        df_feature=decay_features_df_pandas,
        feature='feature_a',
        target_frames=target_frames,
        corr_method='spearman',
        date_column='time',
        symbol_column='symbol',
        feature_groups={'feature_a': 'momentum'},
        fdr=0.05,
        fdr_method='bh',
    )

    assert list(result.table['horizon']) == [1, 2]
    assert list(result.table['feature_group']) == ['momentum', 'momentum']
    assert set(result.ic_frames) == {1, 2}


def test_ic_decay_from_target_frames_sorts_horizons_in_result_table(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should sort the result table even when target frames are unordered."""
    target_frames = {
        2: decay_target_data_pandas[['time', 'symbol', 'target_2']],
        1: decay_target_data_pandas[['time', 'symbol', 'target_1']],
    }

    result = _ic_decay_from_target_frames(
        df_feature=decay_features_df_pandas,
        feature='feature_a',
        target_frames=target_frames,
        corr_method='spearman',
        date_column='time',
        symbol_column='symbol',
        feature_groups=None,
        fdr=0.05,
        fdr_method='bh',
    )

    assert list(result.table['horizon']) == [1, 2]


def test_ic_decay_from_target_frames_pandas_polars_consistency(
        decay_features_df_pandas,
        decay_target_data_pandas,
        decay_features_df_polars,
        decay_target_data_polars,
):
    """Should return equivalent decay results for Pandas and Polars inputs."""
    pandas_target_frames = {
        1: decay_target_data_pandas[['time', 'symbol', 'target_1']],
        2: decay_target_data_pandas[['time', 'symbol', 'target_2']],
    }
    polars_target_frames = {
        1: decay_target_data_polars.select(['time', 'symbol', 'target_1']),
        2: decay_target_data_polars.select(['time', 'symbol', 'target_2']),
    }

    pandas_result = _ic_decay_from_target_frames(
        df_feature=decay_features_df_pandas,
        feature='feature_a',
        target_frames=pandas_target_frames,
        corr_method='spearman',
        date_column='time',
        symbol_column='symbol',
        feature_groups=None,
        fdr=0.05,
        fdr_method='bh',
    )
    polars_result = _ic_decay_from_target_frames(
        df_feature=decay_features_df_polars,
        feature='feature_a',
        target_frames=polars_target_frames,
        corr_method='spearman',
        date_column='time',
        symbol_column='symbol',
        feature_groups=None,
        fdr=0.05,
        fdr_method='bh',
    )

    pd.testing.assert_frame_equal(
        pandas_result.table.reset_index(drop=True),
        polars_result.table.to_pandas().reset_index(drop=True),
        check_dtype=False,
    )
    for horizon in pandas_result.ic_frames:
        pd.testing.assert_frame_equal(
            pandas_result.ic_frames[horizon].reset_index(drop=True),
            polars_result.ic_frames[horizon].to_pandas().reset_index(drop=True),
            check_dtype=False,
        )


# ------------------------------------------------------
# ic_decay
# ------------------------------------------------------
def test_ic_decay_preserves_polars_backend(
        decay_features_df_polars,
        decay_target_data_polars,
):
    """Should return Polars results when features and targets use Polars."""
    def target_fn(target_data, horizon):
        return target_data.select(['time', 'symbol', f'target_{horizon}'])

    result = ic_decay(
        df_feature=decay_features_df_polars,
        feature='feature_a',
        target_data=decay_target_data_polars,
        horizons=[1, 2],
        target_fn=target_fn,
    )

    assert isinstance(result.table, pl.DataFrame)
    assert all(isinstance(frame, pl.DataFrame) for frame in result.ic_frames.values())


def test_ic_decay_pandas_polars_consistency(
        decay_features_df_pandas,
        decay_target_data_pandas,
        decay_features_df_polars,
        decay_target_data_polars,
):
    """Should return equivalent decay results for Pandas and Polars inputs."""
    def pandas_target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    def polars_target_fn(target_data, horizon):
        return target_data.select(['time', 'symbol', f'target_{horizon}'])

    pandas_result = ic_decay(
        df_feature=decay_features_df_pandas,
        feature='feature_a',
        target_data=decay_target_data_pandas,
        horizons=[1, 2],
        target_fn=pandas_target_fn,
    )
    polars_result = ic_decay(
        df_feature=decay_features_df_polars,
        feature='feature_a',
        target_data=decay_target_data_polars,
        horizons=[1, 2],
        target_fn=polars_target_fn,
    )

    pd.testing.assert_frame_equal(
        pandas_result.table.reset_index(drop=True),
        polars_result.table.to_pandas().reset_index(drop=True),
        check_dtype=False,
    )
    for horizon in pandas_result.ic_frames:
        pd.testing.assert_frame_equal(
            pandas_result.ic_frames[horizon].reset_index(drop=True),
            polars_result.ic_frames[horizon].to_pandas().reset_index(drop=True),
            check_dtype=False,
        )


# ------------------------------------------------------
# ic_decay_summary
# ------------------------------------------------------
def test_ic_decay_summary_identifies_peak_halflife_and_significance(
        decay_summary_curve_pandas,
):
    """Should derive diagnostics from a curve sorted by horizon."""
    result = ic_decay_summary(decay_summary_curve_pandas)

    assert result.peak_horizon == 2
    assert result.peak_abs_ic == pytest.approx(0.20)
    assert result.halflife_horizon == 3
    assert result.last_significant_horizon == 3


def test_ic_decay_summary_returns_none_when_no_horizon_is_significant(
        decay_summary_curve_pandas,
):
    """Should return None when no horizon passes FDR correction."""
    decay_curve = decay_summary_curve_pandas.assign(fdr_rejected=False)

    result = ic_decay_summary(decay_curve)

    assert result.last_significant_horizon is None


def test_ic_decay_summary_returns_none_when_signal_never_reaches_half_peak():
    """Should return None when no post-peak value reaches half the peak."""
    decay_curve = pd.DataFrame({
        'horizon': [1, 2, 3],
        'mean': [0.20, 0.16, 0.11],
        'fdr_rejected': [True, True, False],
    })

    result = ic_decay_summary(decay_curve)

    assert result.peak_horizon == 1
    assert result.halflife_horizon is None


def test_ic_decay_summary_uses_first_peak_when_peak_values_are_equal():
    """Should use the earliest horizon when absolute peak IC values tie."""
    decay_curve = pd.DataFrame({
        'horizon': [1, 2, 3],
        'mean': [0.10, -0.10, 0.00],
        'fdr_rejected': [True, True, False],
    })

    result = ic_decay_summary(decay_curve)

    assert result.peak_horizon == 1
    assert result.peak_abs_ic == pytest.approx(0.10)
    assert result.halflife_horizon == 3


def test_ic_decay_summary_calculates_trapezoidal_auc():
    """Should calculate AUC using the horizon values as trapezoid spacing."""
    decay_curve = pd.DataFrame({
        'horizon': [1, 2, 4],
        'mean': [0.40, -0.20, 0.10],
        'fdr_rejected': [True, True, False],
    })

    result = ic_decay_summary(decay_curve)

    assert result.auc == pytest.approx(0.60)


def test_ic_decay_summary_rejects_empty_curve():
    """Should reject an empty decay curve."""
    decay_curve = pd.DataFrame({
        'horizon': [],
        'mean': [],
        'fdr_rejected': [],
    })

    with pytest.raises(ValueError, match='must not be empty'):
        ic_decay_summary(decay_curve)


def test_ic_decay_summary_rejects_duplicate_horizons():
    """Should require exactly one row for each horizon."""
    decay_curve = pd.DataFrame({
        'horizon': [1, 1],
        'mean': [0.10, 0.05],
        'fdr_rejected': [True, False],
    })

    with pytest.raises(ValueError, match='exactly one row per horizon'):
        ic_decay_summary(decay_curve)


def test_ic_decay_summary_rejects_curve_without_finite_mean_ic_values():
    """Should reject a curve that contains no usable mean IC values."""
    decay_curve = pd.DataFrame({
        'horizon': [1, 2, 3],
        'mean': [np.nan, np.inf, -np.inf],
        'fdr_rejected': [False, False, False],
    })

    with pytest.raises(ValueError, match='no finite mean IC values'):
        ic_decay_summary(decay_curve)


# ------------------------------------------------------
# ic_decay_summary_table
# ------------------------------------------------------
def test_ic_decay_summary_table_reuses_targets_for_all_features(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should generate each target once and reuse it for every feature."""
    calls = []

    def target_fn(target_data, horizon):
        calls.append(horizon)
        return target_data[['time', 'symbol', f'target_{horizon}']]

    result = ic_decay_summary_table(
        df_features=decay_features_df_pandas,
        feature_list=['feature_a', 'feature_b'],
        target_data=decay_target_data_pandas,
        horizons=[1, 2],
        target_fn=target_fn,
    )

    assert calls == [1, 2]


def test_ic_decay_summary_table_rejects_empty_feature_list(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject an empty feature list."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='feature_list must not be empty'):
        ic_decay_summary_table(
            df_features=decay_features_df_pandas,
            feature_list=[],
            target_data=decay_target_data_pandas,
            horizons=[1, 2],
            target_fn=target_fn,
        )


def test_ic_decay_summary_table_returns_expected_columns_and_feature_groups(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should return one summary row per feature with its assigned group."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    result = ic_decay_summary_table(
        df_features=decay_features_df_pandas,
        feature_list=['feature_a', 'feature_b'],
        target_data=decay_target_data_pandas,
        horizons=[1, 2],
        target_fn=target_fn,
        feature_groups={
            'feature_a': 'momentum',
            'feature_b': 'reversal',
        },
    )

    expected_columns = {
        'feature',
        'peak_horizon',
        'peak_abs_ic',
        'halflife_horizon',
        'last_significant_horizon',
        'auc',
        'feature_group',
    }

    assert set(result.table.columns) == expected_columns
    assert list(result.table['feature']) == ['feature_a', 'feature_b']
    assert list(result.table['feature_group']) == ['momentum', 'reversal']


def test_ic_decay_summary_table_preserves_individual_decay_results(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Stored decay results should match individual ic_decay computations."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    summary_result = ic_decay_summary_table(
        df_features=decay_features_df_pandas,
        feature_list=['feature_a', 'feature_b'],
        target_data=decay_target_data_pandas,
        horizons=[1, 2],
        target_fn=target_fn,
    )

    for feature in ['feature_a', 'feature_b']:
        expected_result = ic_decay(
            df_feature=decay_features_df_pandas,
            feature=feature,
            target_data=decay_target_data_pandas,
            horizons=[1, 2],
            target_fn=target_fn,
        )
        actual_result = summary_result.decay_results[feature]

        pd.testing.assert_frame_equal(
            actual_result.table,
            expected_result.table,
        )
        for horizon in expected_result.ic_frames:
            pd.testing.assert_frame_equal(
                actual_result.ic_frames[horizon],
                expected_result.ic_frames[horizon],
            )


def test_ic_decay_summary_table_pandas_polars_consistency(
        decay_features_df_pandas,
        decay_target_data_pandas,
        decay_features_df_polars,
        decay_target_data_polars,
):
    """Should return equivalent summary tables for Pandas and Polars inputs."""
    def pandas_target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    def polars_target_fn(target_data, horizon):
        return target_data.select(['time', 'symbol', f'target_{horizon}'])

    pandas_result = ic_decay_summary_table(
        df_features=decay_features_df_pandas,
        feature_list=['feature_a', 'feature_b'],
        target_data=decay_target_data_pandas,
        horizons=[1, 2],
        target_fn=pandas_target_fn,
    )
    polars_result = ic_decay_summary_table(
        df_features=decay_features_df_polars,
        feature_list=['feature_a', 'feature_b'],
        target_data=decay_target_data_polars,
        horizons=[1, 2],
        target_fn=polars_target_fn,
    )

    pd.testing.assert_frame_equal(
        pandas_result.table.reset_index(drop=True),
        polars_result.table.to_pandas().reset_index(drop=True),
        check_dtype=False,
    )
    assert set(pandas_result.decay_results) == set(polars_result.decay_results)


def test_ic_decay_summary_table_duplicate_feature_list_raises(
        decay_features_df_pandas,
        decay_target_data_pandas,
):
    """Should reject duplicate feature names."""
    def target_fn(target_data, horizon):
        return target_data[['time', 'symbol', f'target_{horizon}']]

    with pytest.raises(ValueError, match='feature_list must not contain duplicates'):
        ic_decay_summary_table(
            df_features=decay_features_df_pandas,
            feature_list=['feature_a', 'feature_a'],
            target_data=decay_target_data_pandas,
            horizons=[1, 2],
            target_fn=target_fn,
        )


# ------------------------------------------------------
# partial_information_coefficient
# ------------------------------------------------------
@pytest.fixture
def partial_ic_data_pandas():
    """Create controlled cross-sections across dates for partial IC tests."""
    rng = np.random.default_rng(321)
    frames = []
    for day in pd.date_range('2024-01-01', periods=8, freq='D'):
        covariate_a = rng.normal(size=48)
        covariate_b = rng.normal(size=48)
        feature = 0.8 * covariate_a - 0.4 * covariate_b
        feature += rng.normal(scale=0.7, size=48)
        target = 0.6 * feature + 0.7 * covariate_a + 0.3 * covariate_b
        target += rng.normal(scale=0.7, size=48)
        frames.append(pd.DataFrame({
            'time': day,
            'feature': feature,
            'target': target,
            'covariate_a': covariate_a,
            'covariate_b': covariate_b,
        }))

    return pd.concat(frames, ignore_index=True)


def test_partial_information_coefficient_matches_each_cross_section(
        partial_ic_data_pandas,
):
    """Partial IC should equal the controlled correlation at every date."""
    result = partial_information_coefficient(
        partial_ic_data_pandas,
        feature='feature',
        target='target',
        covariates=['covariate_a', 'covariate_b'],
    )
    expected = [
        partial_correlation(
            group,
            'feature',
            'target',
            covariates=['covariate_a', 'covariate_b'],
        )
        for _, group in partial_ic_data_pandas.groupby('time')
    ]

    assert result.columns.tolist() == ['time', 'partial_ic']
    assert result['partial_ic'].to_numpy() == pytest.approx(expected)


def test_partial_information_coefficient_preserves_native_polars_backend(
        monkeypatch,
        partial_ic_data_pandas,
):
    """Partial IC should preserve Polars without converting through Pandas."""
    polars_df = pl.from_pandas(partial_ic_data_pandas)

    def fail_to_pandas(*args, **kwargs):
        raise AssertionError('Partial IC must not call to_pandas().')

    monkeypatch.setattr(pl.DataFrame, 'to_pandas', fail_to_pandas)
    result = partial_information_coefficient(
        polars_df,
        feature='feature',
        target='target',
        covariates='covariate_a',
    )

    assert isinstance(result, pl.DataFrame)
    assert result.columns == ['time', 'partial_ic']


# -----------------------------------------------
# partial_ic_summary_table
# -----------------------------------------------
def test_partial_ic_summary_table_returns_metrics_and_partial_frames(
        partial_ic_data_pandas,
):
    """The summary should preserve IC metrics and record the controls used."""
    result = partial_ic_summary_table(
        partial_ic_data_pandas,
        feature_list=['feature'],
        target='target',
        covariates=['covariate_a', 'covariate_b'],
        feature_groups={'feature': 'candidate'},
    )

    assert isinstance(result, PartialICSummaryResult)
    assert result.table.loc[0, 'covariates'] == 'covariate_a, covariate_b'
    assert result.table.loc[0, 'feature_group'] == 'candidate'
    assert result.table.loc[0, 'n_obs'] == len(result.partial_ic_frames['feature'])
    assert result.table.loc[0, 'mean'] == pytest.approx(
        result.partial_ic_frames['feature']['partial_ic'].mean(),
    )


@pytest.mark.parametrize(
    ('feature_list', 'covariates', 'message'),
    [
        ([], ['covariate_a'], 'must not be empty'),
        (['feature', 'feature'], ['covariate_a'], 'must not contain duplicates'),
        (['covariate_a'], ['covariate_a'], 'must not contain covariate'),
    ],
)
def test_partial_ic_summary_table_validates_feature_roles(
        partial_ic_data_pandas,
        feature_list,
        covariates,
        message,
):
    """A partial IC summary should reject invalid feature/covariate roles."""
    with pytest.raises(ValueError, match=message):
        partial_ic_summary_table(
            partial_ic_data_pandas,
            feature_list=feature_list,
            target='target',
            covariates=covariates,
        )


@pytest.mark.parametrize('backend', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('confounded', [False, True])
def test_partial_ic_inherits_scale_safe_residuals_and_undefined_dates(backend, confounded):
    """Should preserve analytically known tiny signals and omit perfectly confounded cross-sections."""
    z = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
    u = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=float)
    v = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
    frame = backend({'time': [1] * 8, 'symbol': [str(i) for i in range(8)],
                     'feature': z * 1e15 if confounded else (z + u) * 1e-15,
                     'target': z + u + v, 'z': z})
    result = partial_information_coefficient(frame, 'feature', 'target', 'z',
                                            corr_method='pearson', symbol_column='symbol')
    assert list(result.columns) == ['time', 'partial_ic']
    if confounded:
        assert len(result) == 0
    else:
        assert result['partial_ic'].to_list() == pytest.approx([2 ** -.5])
    assert result['partial_ic'].dtype == (np.dtype('float64') if backend is pd.DataFrame else pl.Float64)
    with pytest.raises(ValueError, match='distinct'):
        partial_information_coefficient(frame, 'feature', 'feature', 'z')
