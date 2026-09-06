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
        return [covariates]

    if not isinstance(covariates, list):
        raise TypeError('covariates must be a string or a list of strings.')
    if not covariates:
        raise ValueError('covariates must not be empty.')
    if not all(isinstance(covariate, str) for covariate in covariates):
        raise TypeError('covariates must be a string or a list of strings.')
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
    ValueError
        If a covariate is also the feature or target column.
    """
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
        If df, covariates, or min_n has an unsupported type.
    ValueError
        If covariates is invalid, overlaps the feature or target, corr_method
        is unsupported, or selected values cannot be converted to floats.

    Notes
    -----
    Rows with missing or non-finite selected values are excluded jointly before
    estimation. Collinear covariates are handled through least squares because
    only their shared column space affects the residuals.

    Methodology
    -----------
    Let ``Z`` be the design matrix formed by an intercept and the covariates.
    The implementation estimates both regressions in one least-squares solve:

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
    when covariates are redundant or nearly collinear, because only the
    covariates' column space is required to form the residuals.
    """
    normalized_covariates = _normalize_covariates(covariates)
    _validate_partial_columns(feature, target, normalized_covariates)
    _validate_min_n(min_n)

    if corr_method not in ('pearson', 'spearman'):
        raise ValueError("corr_method must be 'spearman' or 'pearson'.")

    columns = [feature, target, *normalized_covariates]
    _validate_df(df, columns, check_all_missing=False)
    try:
        values = (
            df[columns].to_numpy(dtype=float)
            if isinstance(df, pd.DataFrame)
            else df.select(columns).to_numpy().astype(float, copy=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError('feature, target, and covariates must be numeric.') from error

    values = values[np.isfinite(values).all(axis=1)]
    required_n = max(len(normalized_covariates) + 3, min_n or 0)
    if len(values) < required_n:
        return np.nan

    if corr_method == 'spearman':
        values = rankdata(values, axis=0)

    design = np.column_stack([np.ones(len(values)), values[:, 2:]])
    residuals = values[:, :2] - design @ np.linalg.lstsq(
        design,
        values[:, :2],
        rcond=None,
    )[0]
    feature_residuals = residuals[:, 0]
    target_residuals = residuals[:, 1]

    if (
            np.isclose(np.std(feature_residuals, ddof=1), 0.0)
            or np.isclose(np.std(target_residuals, ddof=1), 0.0)
    ):
        return np.nan

    return float(np.clip(pearsonr(feature_residuals, target_residuals).statistic, -1.0, 1.0))
