from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import pearsonr, rankdata

from alpha_research._utils import _validate_df

__all__ = ['partial_correlation']


def _normalize_covariates(covariates: str | list[str]) -> list[str]:
    """
    Normalize one or more covariate names into a validated list.

    Parameters
    ----------
    covariates : str | list[str]
        One covariate name or a non-empty list of unique covariate names.

    Returns
    -------
    list[str]
        Covariate names in the order supplied by the caller.

    Raises
    ------
    TypeError
        If covariates is neither a string nor a list of strings.
    ValueError
        If covariates is empty or contains duplicate names.
    """
    if isinstance(covariates, str):
        covariates = [covariates]

    if not isinstance(covariates, list):
        raise TypeError('covariates must be a string or a list of strings.')
    if not covariates:
        raise ValueError('covariates must not be empty.')
    if not all(isinstance(covariate, str) for covariate in covariates):
        raise TypeError('covariates must be a string or a list of strings.')
    if any(not covariate.strip() for covariate in covariates):
        raise ValueError('covariate names must not be empty.')
    if len(set(covariates)) != len(covariates):
        raise ValueError('covariates must not contain duplicates.')

    return covariates


def _validate_partial_columns(
        feature: str,
        target: str,
        covariates: list[str],
) -> None:
    """
    Validate that controlled variables differ from the tested pair.

    Parameters
    ----------
    feature : str
        Feature column selected for the partial correlation.
    target : str
        Target column selected for the partial correlation.
    covariates : list[str]
        Previously normalized covariate names.

    Returns
    -------
    None
        This helper returns None when the columns have distinct roles.

    Raises
    ------
    TypeError
        If feature or target is not a string.
    ValueError
        If names are empty, feature equals target, or a covariate is also a tested column.
    """
    if not isinstance(feature, str) or not isinstance(target, str):
        raise TypeError('feature and target must be column-name strings.')
    if not feature.strip() or not target.strip():
        raise ValueError('feature and target names must not be empty.')
    if feature == target:
        raise ValueError('feature and target must be distinct columns.')
    if feature in covariates or target in covariates:
        raise ValueError(
            'covariates must not contain the feature or target column.',
        )


def _validate_min_n(min_n: int | None) -> None:
    """
    Validate the optional partial-correlation sample-size threshold.

    Parameters
    ----------
    min_n : int | None
        Optional additional minimum number of complete observations.

    Returns
    -------
    None
        This helper returns None when min_n is valid.

    Raises
    ------
    TypeError
        If min_n is neither a positive integer nor None.
    ValueError
        If min_n is less than one.
    """
    if min_n is None:
        return

    if not isinstance(min_n, (int, np.integer)) or isinstance(min_n, bool):
        raise TypeError('min_n must be a positive integer or None.')
    if min_n < 1:
        raise ValueError('min_n must be a positive integer or None.')


def _partial_complete_cases(df: pd.DataFrame | pl.DataFrame, columns: list[str]) -> np.ndarray:
    """Extract jointly finite real observations without coercing invalid data types.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Original observations, which are never modified.
    columns : list[str]
        Distinct selected column names in feature, target, control order.

    Returns
    -------
    np.ndarray
        Float64 complete cases in their original row and column order.

    Raises
    ------
    TypeError
        If the frame, column specification, or an estimate is not supported.
    KeyError
        If a selected column is absent.
    ValueError
        If the frame is empty, selected columns are ambiguous, or a finite
        value cannot be represented as a float.
    """
    if not isinstance(columns, list) or not columns or not all(isinstance(name, str) for name in columns):
        raise TypeError('columns must be a nonempty list of column-name strings.')
    if len(set(columns)) != len(columns):
        raise ValueError('selected columns must be distinct.')
    _validate_df(df, columns, check_all_missing=False)
    if isinstance(df, pd.DataFrame) and any(df.columns.to_list().count(name) != 1 for name in columns):
        raise ValueError('selected column names must be unique in the DataFrame.')
    values = np.empty((len(df), len(columns)), dtype=float)
    for index, column in enumerate(columns):
        series = df[column]
        # Bootstrap calls commonly contain native numeric columns. Keep their
        # conversion vectorized; scalar validation is needed only for objects.
        if isinstance(series, pd.Series) and series.dtype.kind in 'iuf' and series.dtype.itemsize <= 8:
            values[:, index] = series.to_numpy(dtype=float, na_value=np.nan)
            continue
        if isinstance(series, pl.Series) and (series.dtype.is_integer() or series.dtype.is_float()):
            values[:, index] = series.cast(pl.Float64).to_numpy()
            continue
        for row, value in enumerate(series.to_list()):
            if value is None or value is pd.NA:
                values[row, index] = np.nan
                continue
            if not isinstance(value, Real) or isinstance(value, (bool, np.bool_, np.datetime64, np.timedelta64)):
                raise TypeError('feature, target, and covariates must be numeric real scalars.')
            try:
                numeric = float(value)
            except OverflowError as error:
                raise ValueError('selected values must be representable as floats.') from error
            if not np.isfinite(numeric) and np.isfinite(value):
                raise ValueError('selected values must be representable as floats.')
            values[row, index] = numeric
    return values[np.isfinite(values).all(axis=1)]


def _partial_residuals(values: np.ndarray) -> np.ndarray:
    """Project the tested pair off the numerically resolved control space.

    Parameters
    ----------
    values : np.ndarray
        Nonempty finite real matrix with feature, target, and at least one control.

    Returns
    -------
    np.ndarray
        Two centered residual columns on dimensionless scales. A column is
        NaN when its residual norm is below numerical resolution.

    Raises
    ------
    TypeError
        If values is not a real numeric NumPy array.
    ValueError
        If dimensions or finite-value requirements are invalid.
    numpy.linalg.LinAlgError
        If the singular-value decomposition does not converge.

    Notes
    -----
    Centering and scaling preserve the intercept-adjusted control span. The
    SVD projection is the same least-squares residualization, without forming
    large cancelling regression coefficients. Numerical rank uses the default
    lstsq cutoff: eps * max(design.shape) * largest singular value. Residuals
    must exceed eps * max(design.shape) * the retained condition number in
    unit-norm tested coordinates; unresolved residual variation is undefined.
    """
    if not isinstance(values, np.ndarray) or values.dtype.kind not in 'iuf':
        raise TypeError('values must be a real numeric NumPy array.')
    if values.ndim != 2 or not len(values) or values.shape[1] < 3 or not np.isfinite(values).all():
        raise ValueError('values must be a nonempty finite matrix with at least three columns.')
    values = values.astype(float, copy=False)
    center = values.min(axis=0) / 2 + values.max(axis=0) / 2
    centered = values - center
    scales = np.max(np.abs(centered), axis=0)
    normalized = np.divide(centered, scales, out=np.zeros_like(centered, dtype=float), where=scales > 0)
    normalized -= normalized.mean(axis=0)
    design = np.column_stack([np.ones(len(values)), normalized[:, 2:]])
    basis, singular_values, _ = np.linalg.svd(design, full_matrices=False)
    cutoff = np.finfo(float).eps * max(design.shape)
    retained = singular_values > cutoff * singular_values[0]
    basis = basis[:, retained]
    pair = normalized[:, :2]
    norms = np.linalg.norm(pair, axis=0)
    pair = np.divide(pair, norms, out=np.zeros_like(pair), where=norms > 0)
    residuals = pair - basis @ (basis.T @ pair)
    residuals -= residuals.mean(axis=0)
    resolution = cutoff * singular_values[0] / singular_values[retained][-1]
    residuals[:, np.linalg.norm(residuals, axis=0) <= resolution] = np.nan
    return residuals


def partial_correlation(
        df: pd.DataFrame | pl.DataFrame,
        feature: str,
        target: str,
        covariates: str | list[str],
        corr_method: Literal['pearson', 'spearman'] = 'spearman',
        min_n: int | None = None,
) -> float:
    """
    Compute the partial correlation between a feature and target.

    The feature and target are each residualized on the supplied covariates,
    including an intercept. Their Pearson correlation is then calculated. For
    Spearman partial correlation, every variable is ranked before this same
    residualization, matching the rank-based partial-correlation estimand.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame containing feature, target, and covariate columns.
    feature : str
        Feature column whose association with target is measured.
    target : str
        Target column whose association with feature is measured.
    covariates : str | list[str]
        One or more columns conditioned on by both feature and target.
    corr_method : {'pearson', 'spearman'}, default 'spearman'
        Partial-correlation estimator. Spearman ranks all variables before
        residualization.
    min_n : int | None, default None
        Optional additional minimum number of complete observations. Regardless
        of this setting, at least ``len(covariates) + 3`` observations are
        required for a defined partial correlation.

    Returns
    -------
    float
        Partial correlation in [-1, 1], or nan when there are too few complete
        observations or either residual series is constant.

    Raises
    ------
    KeyError
        If a selected column is absent.
    TypeError
        If df, column names, selected values, covariates, or min_n has an unsupported type.
    ValueError
        If covariates is invalid, overlaps the feature or target, corr_method
        is unsupported, selected column names are ambiguous, or finite selected
        values cannot be represented as floats.

    Notes
    -----
    Rows with missing or non-finite selected values are excluded jointly before
    estimation. Collinear covariates are handled through least squares because
    only their shared column space affects the residuals.
    Strings, booleans, complex values, and temporal values are rejected.
    Numerical rank and residual resolution are scale-adjusted as documented
    in _partial_residuals; numerically unresolved variation returns nan.

    Methodology
    -----------
    Let ``Z`` be the design matrix formed by an intercept and the covariates.
    The implementation projects both variables onto the same least-squares space:

        feature = Z * beta_feature + residual_feature
        target = Z * beta_target + residual_target

    The reported partial correlation is the Pearson correlation between
    ``residual_feature`` and ``residual_target``. With
    ``corr_method='spearman'``, average ranks are calculated for feature,
    target, and every covariate before constructing ``Z``; the same residual
    procedure then gives the rank-based partial correlation.

    For full-rank data, this is algebraically equivalent to the precision-
    matrix formulation often used by partial-correlation packages. Given the
    correlation matrix ``R`` of feature, target, and covariates, with
    ``Omega = inverse(R)``, the same estimate is:

        -Omega[feature, target] /
        sqrt(Omega[feature, feature] * Omega[target, target])

    This implementation deliberately uses least squares rather than directly
    inverting ``R``. It avoids an explicit matrix inverse and remains defined
    when covariates are redundant, because only their column space is required.
    Nearly collinear controls are resolved up to floating-point precision;
    a defined estimate is not promised beyond that numerical resolution.
    """
    normalized_covariates = _normalize_covariates(covariates)
    _validate_partial_columns(feature, target, normalized_covariates)
    _validate_min_n(min_n)

    if corr_method not in ('pearson', 'spearman'):
        raise ValueError("corr_method must be 'spearman' or 'pearson'.")

    columns = [feature, target, *normalized_covariates]
    values = _partial_complete_cases(df, columns)
    required_n = max(len(normalized_covariates) + 3, min_n or 0)
    if len(values) < required_n:
        return np.nan

    if corr_method == 'spearman':
        values = rankdata(values, axis=0)

    residuals = _partial_residuals(values)
    feature_residuals = residuals[:, 0]
    target_residuals = residuals[:, 1]

    if not np.isfinite(residuals).all():
        return np.nan

    return float(np.clip(pearsonr(feature_residuals, target_residuals).statistic, -1.0, 1.0))
