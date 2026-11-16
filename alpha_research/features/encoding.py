"""Explicit category mappings that preserve original observations and labels."""

import math
from itertools import pairwise
from numbers import Real

import pandas as pd
import polars as pl


def validate_mapping(categories, codes, *, binary=False):
    """Validate finite, strictly increasing codes in declared category order.

    Parameters
    ----------
    categories, codes : list
        Distinct scalar labels and corresponding numeric codes.
    binary : bool
        Require exactly two categories mapped to zero and one.

    Returns
    -------
    None
        The mapping is valid.

    Raises
    ------
    ValueError
        If labels are missing/duplicated or codes are invalid or unordered.
    """
    if not isinstance(categories, list) or not categories or any(
            type(value) not in (str, int, float, bool) or pd.isna(value) for value in categories):
        raise ValueError('Categories must be nonmissing scalar labels.')
    if len(set(categories)) != len(categories):
        raise ValueError('Categories must be distinct.')
    if not isinstance(codes, list) or len(codes) != len(categories) or any(
            isinstance(value, bool) or not isinstance(value, Real) or isinstance(value, int) and abs(value) > 2**53 - 1
            or not math.isfinite(value) for value in codes):
        raise ValueError('Provide one finite numeric code per category.')
    if any(left >= right for left, right in pairwise(codes)):
        raise ValueError('Codes must be strictly increasing in category order.')
    if binary and codes != [0, 1]:
        raise ValueError('Binary mapping requires exactly the codes 0 and 1.')


def ordinal_mapping(df: pd.DataFrame | pl.DataFrame, feature_cols: str | list[str],
                    categories: list, codes: list | None = None) -> pd.DataFrame | pl.DataFrame:
    """Append numeric variables using explicit category order; reject unknown labels.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Original observations, retained unchanged.
    feature_cols : str | list[str]
        Columns to encode with the same mapping.
    categories : list
        Labels from lowest to highest, without inferred textual ordering.
    codes : list | None
        Strictly increasing finite codes; defaults to zero-based ranks.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Original columns plus one ``_ordinal`` column per input. Missing values
        remain missing. Codes represent ranks, not inferred physical distances.

    Raises
    ------
    TypeError
        If the frame or feature selection has an unsupported type.
    ValueError
        If mapping, columns, output names or observed categories are invalid.
    """
    return _map_frame(df, feature_cols, categories, codes, '_ordinal')


def binary_mapping(df: pd.DataFrame | pl.DataFrame, feature_cols: str | list[str],
                   categories: list) -> pd.DataFrame | pl.DataFrame:
    """Append binary codes for explicitly selected zero and one categories.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Original observations.
    feature_cols : str | list[str]
        Columns to encode.
    categories : list
        Two labels, first for zero and second for one.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Original columns plus new ``_binary`` columns; missing stays missing.

    Raises
    ------
    TypeError, ValueError
        If frame, labels, observations or output names are invalid.
    """
    validate_mapping(categories, [0, 1], binary=True)
    return _map_frame(df, feature_cols, categories, [0, 1], '_binary')


def _map_frame(df, feature_cols, categories, codes, suffix):
    """Map validated observations into new columns while preserving every input.

    Parameters
    ----------
    df, feature_cols, categories, codes, suffix
        Frame, selected columns, explicit mapping and output suffix.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Detached frame with mapped outputs.

    Raises
    ------
    TypeError, ValueError
        If the frame, mapping, observations or output names are invalid.
    """
    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        raise TypeError('Use a Pandas or Polars DataFrame.')
    columns = [feature_cols] if isinstance(feature_cols, str) else feature_cols
    if not isinstance(columns, list) or any(not isinstance(column, str) for column in columns):
        raise TypeError('feature_cols must contain column names.')
    if not columns or len(set(columns)) != len(columns) or any(column not in df.columns for column in columns):
        raise ValueError('Select distinct existing columns.')
    codes = list(range(len(categories))) if codes is None and isinstance(categories, list) else codes
    validate_mapping(categories, codes)
    if any(column + suffix in df.columns for column in columns):
        raise ValueError('An encoded output column already exists.')
    result = df.copy() if isinstance(df, pd.DataFrame) else df.clone()
    mapping = dict(zip(categories, codes, strict=True))
    for column in columns:
        values = df[column].dropna().unique() if isinstance(df, pd.DataFrame) else [value for value in df[column].drop_nulls().unique() if not isinstance(value, float) or not math.isnan(value)]
        if any(value not in mapping for value in values):
            raise ValueError(f'Unknown category in {column}; every nonmissing label requires an explicit mapping.')
        if isinstance(df, pd.DataFrame):
            result[column + suffix] = df[column].astype(object).map(mapping).astype(float)
        else:
            expression = pl.lit(None, dtype=pl.Float64) if df.schema[column] == pl.Null else (pl.col(column).fill_nan(None) if df.schema[column].is_float() else pl.col(column)).replace_strict(categories, codes, return_dtype=pl.Float64)
            result = result.with_columns(expression.alias(column + suffix))
    return result
