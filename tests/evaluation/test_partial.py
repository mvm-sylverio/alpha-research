import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.stats import rankdata

from alpha_research.evaluation.partial import (
    _normalize_covariates,
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
        values = rankdata(values, axis=0)

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
def test_normalize_covariates_accepts_one_or_many_covariates():
    """The covariate helper should normalize strings without changing order."""
    assert _normalize_covariates('covariate_a') == ['covariate_a']
    assert _normalize_covariates(['covariate_a', 'covariate_b']) == [
        'covariate_a',
        'covariate_b',
    ]


@pytest.mark.parametrize(
    ('covariates', 'error_type', 'message'),
    [
        ((), TypeError, 'string or a list'),
        ([], ValueError, 'must not be empty'),
        (['covariate_a', 1], TypeError, 'string or a list'),
        (['covariate_a', 'covariate_a'], ValueError, 'duplicates'),
    ],
)
def test_normalize_covariates_rejects_invalid_specifications(
        covariates,
        error_type,
        message,
):
    """The covariate helper should expose all input-validation failures."""
    with pytest.raises(error_type, match=message):
        _normalize_covariates(covariates)


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
