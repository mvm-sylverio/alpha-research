import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.evaluation.partial import (
    _partial_complete_cases,
    _partial_residuals,
    _validate_min_n,
    _validate_partial_columns,
    partial_correlation,
)


def _expected_partial_correlation_from_precision_matrix(
        df,
        feature,
        target,
        covariates,
        corr_method,
):
    """Calculate partial correlation by the inverse-correlation-matrix identity."""
    values = df[[feature, target, *covariates]].to_numpy(dtype=float)
    values = values[np.isfinite(values).all(axis=1)]
    if corr_method == 'spearman':
        # Average rank = 1 + count(strictly smaller) + half the other ties.
        # This oracle does not share SciPy's ranking code with production.
        values = np.column_stack([
            [1 + sum(other < value for other in column)
             + (sum(other == value for other in column) - 1) / 2 for value in column]
            for column in values.T
        ])

    precision = np.linalg.inv(np.corrcoef(values, rowvar=False))
    return -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])


# ------------------------------------------------------
# fixtures
# ------------------------------------------------------
@pytest.fixture
def partial_data_pandas():
    """Create data with two confounders and residual feature-target association."""
    rng = np.random.default_rng(123)
    n_obs = 96
    covariate_a = rng.normal(size=n_obs)
    covariate_b = rng.normal(size=n_obs)
    feature = 0.8 * covariate_a - 0.4 * covariate_b + rng.normal(
        scale=0.7,
        size=n_obs,
    )
    target = (
        0.6 * feature
        + 0.7 * covariate_a
        + 0.3 * covariate_b
        + rng.normal(scale=0.7, size=n_obs)
    )
    return pd.DataFrame({
        'feature': feature,
        'target': target,
        'covariate_a': covariate_a,
        'covariate_b': covariate_b,
    })


# ------------------------------------------------------
# private helpers
# ------------------------------------------------------
@pytest.mark.parametrize(
    ('feature', 'target', 'covariates', 'message'),
    [
        ('feature', 'target', ['feature'], 'feature or target'),
        ('feature', 'target', ['target'], 'feature or target'),
    ],
)
def test_validate_partial_columns_rejects_overlapping_roles(
        feature,
        target,
        covariates,
        message,
):
    """The role validator should prevent conditioning on a tested variable."""
    with pytest.raises(ValueError, match=message):
        _validate_partial_columns(feature, target, covariates)


def test_validate_partial_columns_accepts_distinct_roles():
    """The role validator should accept distinct feature, target, and controls."""
    assert _validate_partial_columns(
        'feature',
        'target',
        ['covariate_a', 'covariate_b'],
    ) is None


@pytest.mark.parametrize('min_n', [None, 1, np.int64(10)])
def test_validate_min_n_accepts_optional_positive_integers(min_n):
    """The sample-size helper should accept valid optional thresholds."""
    assert _validate_min_n(min_n) is None


@pytest.mark.parametrize(
    ('min_n', 'error_type'),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
        ('10', TypeError),
    ],
)
def test_validate_min_n_rejects_invalid_thresholds(min_n, error_type):
    """The sample-size helper should reject non-positive and non-integer values."""
    with pytest.raises(error_type, match='min_n'):
        _validate_min_n(min_n)


# ------------------------------------------------------
# partial_correlation
# ------------------------------------------------------
@pytest.mark.parametrize(
    ('corr_method', 'covariates'),
    [
        ('pearson', ['covariate_a']),
        ('spearman', ['covariate_a', 'covariate_b']),
    ],
)
def test_partial_correlation_matches_precision_matrix_definition(
        partial_data_pandas,
        corr_method,
        covariates,
):
    """Partial correlation should match the inverse-correlation-matrix identity."""
    expected = _expected_partial_correlation_from_precision_matrix(
        partial_data_pandas,
        'feature',
        'target',
        covariates,
        corr_method,
    )

    result = partial_correlation(
        partial_data_pandas,
        feature='feature',
        target='target',
        covariates=covariates,
        corr_method=corr_method,
    )

    assert result == pytest.approx(expected)


def test_partial_correlation_accepts_one_covariate_as_a_string(partial_data_pandas):
    """A single covariate string should be equivalent to a one-item list."""
    result = partial_correlation(
        partial_data_pandas,
        'feature',
        'target',
        covariates='covariate_a',
    )
    expected = partial_correlation(
        partial_data_pandas,
        'feature',
        'target',
        covariates=['covariate_a'],
    )

    assert result == pytest.approx(expected)


def test_partial_correlation_pandas_polars_consistency(partial_data_pandas):
    """Both supported DataFrame backends should estimate the same value."""
    kwargs = {
        'feature': 'feature',
        'target': 'target',
        'covariates': ['covariate_a', 'covariate_b'],
    }

    pandas_result = partial_correlation(partial_data_pandas, **kwargs)
    polars_result = partial_correlation(pl.from_pandas(partial_data_pandas), **kwargs)

    assert polars_result == pytest.approx(pandas_result)


def test_polars_partial_correlation_does_not_convert_to_pandas(
        monkeypatch,
        partial_data_pandas,
):
    """Polars partial correlation should remain native Polars."""
    polars_df = pl.from_pandas(partial_data_pandas)

    def fail_to_pandas(*args, **kwargs):
        raise AssertionError('Polars partial APIs must not call to_pandas().')

    monkeypatch.setattr(pl.DataFrame, 'to_pandas', fail_to_pandas)

    partial_result = partial_correlation(
        polars_df,
        'feature',
        'target',
        covariates=['covariate_a', 'covariate_b'],
    )
    assert np.isfinite(partial_result)


def test_partial_correlation_uses_complete_cases_and_minimum_sample_size(
        partial_data_pandas,
):
    """Missing values are jointly removed and undersized data returns nan."""
    complete_result = partial_correlation(
        partial_data_pandas,
        'feature',
        'target',
        covariates=['covariate_a'],
    )
    with_missing = partial_data_pandas.copy()
    with_missing.loc[0, 'covariate_a'] = np.nan
    with_missing.loc[1, 'feature'] = np.nan

    missing_result = partial_correlation(
        with_missing,
        'feature',
        'target',
        covariates=['covariate_a'],
    )
    undersized_result = partial_correlation(
        partial_data_pandas.iloc[:3],
        'feature',
        'target',
        covariates=['covariate_a'],
    )

    assert np.isfinite(complete_result)
    assert np.isfinite(missing_result)
    assert np.isnan(undersized_result)


@pytest.mark.parametrize(
    ('covariates', 'min_n', 'corr_method', 'error_type', 'message'),
    [
        ([], None, 'spearman', ValueError, 'must not be empty'),
        (['covariate_a', 'covariate_a'], None, 'spearman', ValueError, 'duplicates'),
        ('feature', None, 'spearman', ValueError, 'feature or target'),
        (['covariate_a'], 0, 'spearman', ValueError, 'min_n'),
        (['covariate_a'], '10', 'spearman', TypeError, 'min_n'),
        (['covariate_a'], None, 'kendall', ValueError, 'corr_method'),
    ],
)
def test_partial_correlation_validates_specification(
        partial_data_pandas,
        covariates,
        min_n,
        corr_method,
        error_type,
        message,
):
    """Controlled correlation should reject an invalid analysis specification."""
    with pytest.raises(error_type, match=message):
        partial_correlation(
            partial_data_pandas,
            'feature',
            'target',
            covariates=covariates,
            min_n=min_n,
            corr_method=corr_method,
        )


# ------------------------------------------------------
# Analytic column spaces and numerical resolution
# ------------------------------------------------------
@pytest.fixture
def orthogonal_partial_values():
    """Should supply centered orthogonal controls and residual signals.

    Parameters
    ----------
    None
        This fixture has no inputs.

    Returns
    -------
    dict
        Feature z+u, target z+u+v, and orthogonal Walsh vectors z, u, v.

    Raises
    ------
    None
        All values are fixed finite constants.
    """
    z = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
    u = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=float)
    v = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
    return {'feature': z + u, 'target': z + u + v, 'z': z, 'u': u, 'v': v}


@pytest.mark.parametrize('backend', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('scale', [1e-200, 1e-10, 1., 1e10, 1e200])
def test_partial_correlation_is_independent_of_units(orthogonal_partial_values, backend, scale):
    """Should retain correlation 1/sqrt(2) under independent scaling of tested variables and controls."""
    values = orthogonal_partial_values
    frame = backend({'feature': values['feature'] * scale, 'target': values['target'] / scale,
                     'z': values['z'] * scale})
    with np.errstate(over='raise', invalid='raise'):
        result = partial_correlation(frame, 'feature', 'target', 'z', 'pearson')
    assert result == pytest.approx(2 ** -.5, abs=1e-13)


@pytest.mark.parametrize('backend', [pd.DataFrame, pl.DataFrame])
def test_partial_affine_redundancy_and_order_invariants(orthogonal_partial_values, backend):
    """Should preserve the residual estimand under shifts, redundant controls, and row or control permutations."""
    values = orthogonal_partial_values
    frame = backend({'feature': 17 + 3 * values['feature'], 'target': -12 + 2 * values['target'],
                     'z': values['z'] - 100, 'redundant': 7 * values['z'] + 10, 'constant': np.ones(8)})
    snapshot = frame.copy(deep=True) if backend is pd.DataFrame else frame.clone()
    for controls in [['z'], ['z', 'redundant'], ['redundant', 'z'], ['constant', 'redundant', 'z']]:
        assert partial_correlation(frame, 'feature', 'target', controls, 'pearson') == pytest.approx(2 ** -.5)
    reversed_frame = frame.iloc[::-1] if backend is pd.DataFrame else frame.reverse()
    assert partial_correlation(reversed_frame, 'target', 'feature', ['z'], 'pearson') == pytest.approx(2 ** -.5)
    assert frame.equals(snapshot)


@pytest.mark.parametrize('backend', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('delta', [1e-4, 1e-8, 1e-10])
def test_nearly_collinear_controls_preserve_known_space(orthogonal_partial_values, backend, delta):
    """Should resolve orthogonal residuals within the precision allowed by control conditioning."""
    values = orthogonal_partial_values
    frame = backend({**values, 'near': values['z'] + delta * values['v']})
    # Controls span z,v, leaving u for both variables: correlation is +1.
    tolerance = max(1e-12, 32 * np.finfo(float).eps / delta)
    assert partial_correlation(frame, 'feature', 'target', ['z', 'near'], 'pearson') == pytest.approx(1., abs=tolerance)


@pytest.mark.parametrize('backend', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('role', ['feature', 'target'])
@pytest.mark.parametrize('scale', [1e-150, 1., 1e15, 1e150])
def test_perfect_confounding_is_undefined_at_every_scale(orthogonal_partial_values, backend, role, scale):
    """Should reject roundoff residuals from perfect confounding regardless of the variable's units."""
    values = orthogonal_partial_values
    frame = backend({**values, role: scale * values['z'] + scale})
    assert np.isnan(partial_correlation(frame, 'feature', 'target', 'z', 'pearson'))


@pytest.mark.parametrize('delta', [1e-6, 1e-10, 1e-12])
def test_nearly_collinear_perfect_confounding_is_undefined(orthogonal_partial_values, delta):
    """Should not report numerical leakage from almost redundant controls as residual association."""
    values = orthogonal_partial_values
    frame = pd.DataFrame({**values, 'near': values['z'] + delta * values['u']})
    assert np.isnan(partial_correlation(frame, 'feature', 'target', ['z', 'near'], 'pearson'))


def test_small_genuine_residual_is_not_an_absolute_constant(orthogonal_partial_values):
    """Should retain a resolvable small residual instead of using a fixed 1e-8 threshold."""
    values = orthogonal_partial_values
    frame = pd.DataFrame({**values, 'feature': values['z'] + 1e-10 * values['u']})
    assert partial_correlation(frame, 'feature', 'target', 'z', 'pearson') == pytest.approx(2 ** -.5, abs=1e-5)


@pytest.mark.parametrize('backend', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('method', ['pearson', 'spearman'])
def test_constant_controls_and_identical_distinct_signals(backend, method):
    """Should reduce constant controls to the intercept and permit identical values under distinct names."""
    frame = backend({'feature': [0., 1., 2., 3., 4.], 'target': [0., 2., 1., 4., 3.],
                     'same': [0., 1., 2., 3., 4.], 'z': [1.] * 5})
    assert partial_correlation(frame, 'feature', 'target', 'z', method) == pytest.approx(.8)
    assert partial_correlation(frame, 'feature', 'same', 'z', method) == pytest.approx(1.)


@pytest.mark.parametrize('backend', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('method', ['pearson', 'spearman'])
@pytest.mark.parametrize('n', [4, 5])
@pytest.mark.parametrize('minimum', [None, 1, 4, 5, 6])
def test_partial_sample_boundaries_count_declared_controls(backend, method, n, minimum):
    """Should enforce k+3 and explicit min_n exactly even with redundant controls."""
    frame = backend({'feature': [0., 1., 2., 3., 4.][:n], 'target': [0., 2., 1., 4., 3.][:n],
                     'z': [0.] * n, 'duplicate': [0.] * n})
    result = partial_correlation(frame, 'feature', 'target', ['z', 'duplicate'], method, min_n=minimum)
    if n == 4 or minimum == 6:
        assert np.isnan(result)
    else:
        # Centered sums of squares are 10,10 and cross-product 8.
        assert result == pytest.approx(.8)


# ------------------------------------------------------
# Complete cases, manual tied ranks, and input identity
# ------------------------------------------------------
@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_complete_cases_have_an_exact_independent_value(orthogonal_partial_values, backend):
    """Should jointly exclude invalid selected rows and recover the analytic residual correlation."""
    values = orthogonal_partial_values
    data = {'feature': [*values['feature'], None, 999., 4., -np.inf],
            'target': [*values['target'], 7., np.inf, 6., 10.],
            'z': [*values['z'], -3., 2., np.nan, 5.], 'unused': [None] * 12}
    frame = pd.DataFrame(data) if backend == 'pandas' else pl.DataFrame(data)
    if backend == 'pandas':
        frame['feature'] = frame['feature'].astype('Float64')
    cases = _partial_complete_cases(frame, ['feature', 'target', 'z'])
    np.testing.assert_array_equal(cases, np.column_stack([values['feature'], values['target'], values['z']]))
    assert partial_correlation(frame, 'feature', 'target', 'z', 'pearson') == pytest.approx(2 ** -.5)


@pytest.mark.parametrize('backend', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('controls', [['z'], ['z', 'redundant'], ['constant', 'redundant', 'z']])
def test_tied_spearman_has_manually_ranked_complete_case_reference(backend, controls):
    """Should rank complete cases only and match a hand-derived tied-rank partial correlation."""
    frame = backend({'feature': [1., 1., 2., 3., 3., 4., 100.],
                     'target': [2., 1., 1., 4., 3., 3., None], 'z': [0., 0., 1., 1., 2., 2., -100.],
                     'redundant': [5., 5., 7., 7., 9., 9., -195.], 'constant': [0.] * 7})
    # Average ranks after dropping the final row:
    # x=(1.5,1.5,3,4.5,4.5,6), y=(3,1.5,1.5,6,4.5,4.5),
    # z=(1.5,1.5,3.5,3.5,5.5,5.5). Center each at 3.5.
    # Sxx=Syy=33/2, Szz=16, Sxy=12, Sxz=15, Syz=9.
    # Residual cross-product=57/16, squares=39/16 and 183/16.
    assert partial_correlation(frame, 'feature', 'target', controls, 'spearman') == pytest.approx(57 / np.sqrt(39 * 183))


@pytest.mark.parametrize('backend', [pd.DataFrame, pl.DataFrame])
def test_partial_rejects_equal_tested_roles(backend):
    """Should reject a repeated tested column uniformly before backend-specific selection behavior."""
    frame = backend({'feature': [1., 2., 3., 4.], 'z': [0., 1., 0., 1.]})
    with pytest.raises(ValueError, match='distinct'):
        _validate_partial_columns('feature', 'feature', ['z'])
    with pytest.raises(ValueError, match='distinct'):
        partial_correlation(frame, 'feature', 'feature', 'z')


@pytest.mark.parametrize('name', ['', '   ', None, 3, ['feature']])
def test_partial_roles_reject_invalid_names(name):
    """Should reject empty or nonscalar tested-column names before dataframe access."""
    with pytest.raises((TypeError, ValueError)):
        _validate_partial_columns(name, 'target', ['z'])
    with pytest.raises((TypeError, ValueError)):
        _validate_partial_columns('feature', name, ['z'])


def test_partial_rejects_ambiguous_selected_pandas_columns(orthogonal_partial_values):
    """Should reject duplicate selected labels while permitting duplicate unused columns."""
    frame = pd.DataFrame(orthogonal_partial_values)
    ambiguous = pd.concat([frame[['feature']], frame], axis=1)
    with pytest.raises(ValueError, match='unique'):
        _partial_complete_cases(ambiguous, ['feature', 'target', 'z'])
    with pytest.raises(ValueError, match='unique'):
        partial_correlation(ambiguous, 'feature', 'target', 'z', 'pearson')
    unused_duplicates = pd.concat([frame, frame[['v']]], axis=1)
    assert partial_correlation(unused_duplicates, 'feature', 'target', 'z', 'pearson') == pytest.approx(2 ** -.5)


@pytest.mark.parametrize('role', ['feature', 'target', 'z'])
@pytest.mark.parametrize('bad', ['1', True, 1j, np.datetime64('2024-01-01'), np.timedelta64(1, 'D')])
@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_partial_rejects_lossy_role_conversions(orthogonal_partial_values, role, bad, backend):
    """Should reject strings, booleans, complex and temporal values in every statistical role."""
    if backend == 'pandas':
        frame = pd.DataFrame({**orthogonal_partial_values, role: [bad] * 8})
    else:
        frame = pl.DataFrame(orthogonal_partial_values).with_columns(pl.Series(role, [bad] * 8, dtype=pl.Object))
    with pytest.raises(TypeError, match='numeric real'):
        _partial_complete_cases(frame, ['feature', 'target', 'z'])
    with pytest.raises(TypeError, match='numeric real'):
        partial_correlation(frame, 'feature', 'target', 'z')


def test_partial_complete_cases_validates_columns_frame_and_numeric_range():
    """Should expose helper validation errors without requiring a statistical calculation."""
    frame = pd.DataFrame({'feature': [1.], 'target': [2.], 'z': [3.]})
    for columns in [None, [], ['feature', 1]]:
        with pytest.raises(TypeError, match='columns'):
            _partial_complete_cases(frame, columns)
    with pytest.raises(ValueError, match='distinct'):
        _partial_complete_cases(frame, ['feature', 'feature'])
    with pytest.raises(TypeError, match='DataFrame'):
        _partial_complete_cases([], ['feature'])
    with pytest.raises(ValueError, match='empty'):
        _partial_complete_cases(frame.iloc[:0], ['feature'])
    with pytest.raises(KeyError):
        _partial_complete_cases(frame, ['missing'])
    with pytest.raises(ValueError, match='representable'):
        _partial_complete_cases(pd.DataFrame({'feature': [10 ** 400]}, dtype=object), ['feature'])


@pytest.mark.parametrize('values', [[], np.array(['1']), np.array([1j])])
def test_partial_residuals_rejects_unsupported_arrays(values):
    """Should reject non-real array types before normalization or decomposition."""
    with pytest.raises(TypeError, match='real numeric'):
        _partial_residuals(values)


@pytest.mark.parametrize('values', [np.empty((0, 3)), np.ones(3), np.ones((4, 2)), np.array([[1., 2., np.inf]])])
def test_partial_residuals_validates_shapes_and_finiteness(values):
    """Should require a nonempty finite matrix containing both tested variables and controls."""
    with pytest.raises(ValueError, match='nonempty finite matrix'):
        _partial_residuals(values)


def test_partial_residuals_reproduces_known_orthogonal_directions(orthogonal_partial_values):
    """Should retain independently known u and u+v directions without mutating inputs."""
    values = orthogonal_partial_values
    matrix = np.column_stack([values['feature'], values['target'], values['z']])
    snapshot = matrix.copy()
    residuals = _partial_residuals(matrix)
    # Tested columns are normalized to unit norm before projection.
    np.testing.assert_allclose(residuals[:, 0], values['u'] / 4, atol=1e-15)
    np.testing.assert_allclose(residuals[:, 1], (values['u'] + values['v']) / np.sqrt(24), atol=1e-15)
    np.testing.assert_array_equal(matrix, snapshot)


def test_partial_residuals_propagates_solver_failures(orthogonal_partial_values, monkeypatch):
    """Should propagate the exact decomposition failure rather than report a misleading estimate."""
    error = np.linalg.LinAlgError('controlled decomposition failure')
    monkeypatch.setattr(np.linalg, 'svd', lambda *args, **kwargs: (_ for _ in ()).throw(error))
    values = orthogonal_partial_values
    with pytest.raises(np.linalg.LinAlgError) as caught:
        _partial_residuals(np.column_stack([values['feature'], values['target'], values['z']]))
    assert caught.value is error


@pytest.mark.parametrize('backend', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('n', [4, 5])
def test_full_rank_controls_have_exact_sample_boundaries(partial_data_pandas, backend, n):
    """Should enforce the same declared-control threshold for full-rank designs and match the precision identity."""
    source = partial_data_pandas.iloc[:n]
    frame = source if backend is pd.DataFrame else pl.from_pandas(source)
    result = partial_correlation(frame, 'feature', 'target', ['covariate_a', 'covariate_b'], 'pearson')
    if n == 4:
        assert np.isnan(result)
    else:
        expected = _expected_partial_correlation_from_precision_matrix(
            source, 'feature', 'target', ['covariate_a', 'covariate_b'], 'pearson')
        assert result == pytest.approx(expected)


@pytest.mark.parametrize('role', ['feature', 'target'])
@pytest.mark.parametrize('method', ['pearson', 'spearman'])
def test_constant_tested_columns_remain_undefined(orthogonal_partial_values, role, method):
    """Should report undefined correlation when either tested column is constant before or after ranking."""
    frame = pd.DataFrame({**orthogonal_partial_values, role: np.ones(8)})
    assert np.isnan(partial_correlation(frame, 'feature', 'target', 'z', method))


@pytest.mark.parametrize('backend', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('missing', [None, np.nan, np.inf, -np.inf])
def test_all_invalid_complete_cases_are_undefined(orthogonal_partial_values, backend, missing):
    """Should return NaN without solving when joint deletion leaves no observations."""
    frame = backend({**orthogonal_partial_values, 'feature': [missing] * 8})
    cases = _partial_complete_cases(frame, ['feature', 'target', 'z'])
    assert cases.shape == (0, 3)
    assert np.isnan(partial_correlation(frame, 'feature', 'target', 'z'))
    with pytest.raises(ValueError, match='corr_method'):
        partial_correlation(frame, 'feature', 'target', 'z', corr_method='invalid')
