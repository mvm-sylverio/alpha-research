from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from alpha_research._utils import (
    _validate_df,
    _validate_finite_number,
    _validate_unique_keys,
)
from alpha_research.features.transformations import _normalize_feature_cols

__all__ = [
    'feature_difference',
    'feature_product',
    'feature_ratio',
    'feature_where',
]


WhereOperator = Literal[
    'greater',
    'greater_equal',
    'less',
    'less_equal',
    'equal',
    'not_equal',
]


def _validate_interaction_inputs(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
        other_feature: str,
        symbol_col: str,
        time_col: str,
) -> list[str]:
    """Validate the shared aligned-frame contract for feature interactions.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame containing one unique row per time and symbol combination.
    feature_cols : str | list[str]
        Left-hand feature columns transformed independently.
    other_feature : str
        Right-hand feature used by every interaction.
    symbol_col : str
        Asset identifier column.
    time_col : str
        Observation-time column.

    Returns
    -------
    list[str]
        Normalized left-hand feature names in caller order.

    Raises
    ------
    KeyError
        If a required column is absent.
    TypeError
        If an argument has an unsupported type.
    ValueError
        If names, keys, or numeric feature values violate the interaction
        contract.

    Notes
    -----
    The helper validates alignment, not feature provenance. Cross-sectional
    and time-series-normalized features may be combined intentionally when
    they already belong to the same unique observation row.
    """
    features = _normalize_feature_cols(feature_cols)
    if any(not isinstance(feature, str) or not feature for feature in features):
        raise TypeError('every feature in feature_cols must be a non-empty string.')
    if len(set(features)) != len(features):
        raise ValueError('feature_cols must not contain duplicate names.')
    if not isinstance(other_feature, str) or not other_feature:
        raise TypeError('other_feature must be a non-empty string.')
    if not isinstance(symbol_col, str) or not symbol_col:
        raise TypeError('symbol_col must be a non-empty string.')
    if not isinstance(time_col, str) or not time_col:
        raise TypeError('time_col must be a non-empty string.')
    if time_col == symbol_col:
        raise ValueError('time_col and symbol_col must refer to different columns.')
    if other_feature in features:
        raise ValueError('other_feature must differ from every feature in feature_cols.')
    key_columns = {time_col, symbol_col}
    if other_feature in key_columns or any(
            feature in key_columns for feature in features
    ):
        raise ValueError('interaction features must differ from key columns.')
    if isinstance(df, pd.DataFrame) and not df.columns.is_unique:
        raise ValueError('DataFrame must not contain duplicate column names.')

    required_cols = [time_col, symbol_col, *features, other_feature]
    _validate_df(df, required_cols)
    _validate_unique_keys(df, [time_col, symbol_col], 'df')

    for key in [time_col, symbol_col]:
        series = df[key]
        if isinstance(df, pd.DataFrame):
            has_missing = bool(series.isna().any())
        else:
            has_missing = bool(series.is_null().any())
            if series.dtype.is_float():
                has_missing = has_missing or bool(series.is_nan().any())
        if has_missing:
            raise ValueError(f'{key} must not contain missing values.')

    for feature in [*features, other_feature]:
        series = df[feature]
        if isinstance(df, pd.DataFrame):
            if (
                    not pd.api.types.is_numeric_dtype(series)
                    or pd.api.types.is_bool_dtype(series)
                    or pd.api.types.is_complex_dtype(series)
            ):
                raise ValueError(f'{feature} must be real numeric and non-boolean.')
            values = series.dropna().to_numpy(dtype=float)
            has_infinite = bool(np.isinf(values).any())
        else:
            if not series.dtype.is_numeric():
                raise ValueError(f'{feature} must be numeric.')
            values = series.drop_nulls()
            has_infinite = (
                bool(values.is_infinite().any()) if values.dtype.is_float() else False
            )
        if has_infinite:
            raise ValueError(f'{feature} must contain only finite or missing values.')

    return features


def feature_ratio(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
        denominator_feature: str,
        zero_policy: Literal['null', 'raise'] = 'null',
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """Divide each requested feature by one aligned denominator feature.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame containing unique time-symbol observations.
    feature_cols : str | list[str]
        Numerator feature columns.
    denominator_feature : str
        Denominator feature column.
    zero_policy : {'null', 'raise'}, default 'null'
        Whether zero denominators produce missing output or raise an error.
    symbol_col : str, default 'symbol'
        Asset identifier column.
    time_col : str, default 'time'
        Observation-time column.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Input columns plus ``{feature}_div_{denominator_feature}`` columns.

    Raises
    ------
    KeyError
        If a required column is absent.
    TypeError
        If an argument has an unsupported type.
    ValueError
        If inputs violate the aligned-frame contract, zero_policy is invalid,
        or a zero denominator is found under the raise policy.
    """
    features = _validate_interaction_inputs(
        df,
        feature_cols,
        denominator_feature,
        symbol_col,
        time_col,
    )
    if zero_policy not in {'null', 'raise'}:
        raise ValueError("zero_policy must be 'null' or 'raise'.")

    denominator = df[denominator_feature]
    if isinstance(df, pd.DataFrame):
        has_zero = bool(denominator.eq(0).fillna(False).any())
    else:
        has_zero = bool(denominator.eq(0).fill_null(False).any())
    if zero_policy == 'raise' and has_zero:
        raise ValueError('denominator_feature must not contain zero values.')

    if isinstance(df, pd.DataFrame):
        result = df.copy()
        for feature in features:
            values = result[feature] / denominator
            result[f'{feature}_div_{denominator_feature}'] = values.where(
                denominator.ne(0),
            )
        return result

    return df.with_columns([
        pl.when(pl.col(denominator_feature) != 0)
        .then(pl.col(feature) / pl.col(denominator_feature))
        .otherwise(None)
        .alias(f'{feature}_div_{denominator_feature}')
        for feature in features
    ])


def feature_product(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
        other_feature: str,
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """Multiply each requested feature by one aligned feature.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame containing unique time-symbol observations.
    feature_cols : str | list[str]
        Left-hand feature columns.
    other_feature : str
        Right-hand multiplier feature.
    symbol_col : str, default 'symbol'
        Asset identifier column.
    time_col : str, default 'time'
        Observation-time column.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Input columns plus ``{feature}_mul_{other_feature}`` columns.

    Raises
    ------
    KeyError
        If a required column is absent.
    TypeError
        If an argument has an unsupported type.
    ValueError
        If inputs violate the aligned-frame contract.
    """
    features = _validate_interaction_inputs(
        df,
        feature_cols,
        other_feature,
        symbol_col,
        time_col,
    )
    if isinstance(df, pd.DataFrame):
        result = df.copy()
        for feature in features:
            result[f'{feature}_mul_{other_feature}'] = (
                result[feature] * result[other_feature]
            )
        return result
    return df.with_columns([
        (pl.col(feature) * pl.col(other_feature)).alias(
            f'{feature}_mul_{other_feature}',
        )
        for feature in features
    ])


def feature_difference(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
        subtract_feature: str,
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """Subtract one aligned feature from each requested feature as ``X - Y``.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame containing unique time-symbol observations.
    feature_cols : str | list[str]
        Minuend feature columns represented by X.
    subtract_feature : str
        Subtrahend feature column represented by Y.
    symbol_col : str, default 'symbol'
        Asset identifier column.
    time_col : str, default 'time'
        Observation-time column.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Input columns plus ``{feature}_minus_{subtract_feature}`` columns.

    Raises
    ------
    KeyError
        If a required column is absent.
    TypeError
        If an argument has an unsupported type.
    ValueError
        If inputs violate the aligned-frame contract.

    Notes
    -----
    The operation is intentionally directional. Reversing feature_cols and
    subtract_feature changes both the output name and its sign.
    """
    features = _validate_interaction_inputs(
        df,
        feature_cols,
        subtract_feature,
        symbol_col,
        time_col,
    )
    if isinstance(df, pd.DataFrame):
        result = df.copy()
        for feature in features:
            result[f'{feature}_minus_{subtract_feature}'] = (
                result[feature] - result[subtract_feature]
            )
        return result
    return df.with_columns([
        (pl.col(feature) - pl.col(subtract_feature)).alias(
            f'{feature}_minus_{subtract_feature}',
        )
        for feature in features
    ])


def feature_where(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
        condition_feature: str,
        operator: WhereOperator,
        threshold: Real,
        otherwise: Real | None = 0.0,
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """Keep each feature where another feature satisfies a fixed condition.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        DataFrame containing unique time-symbol observations.
    feature_cols : str | list[str]
        Features retained when the condition is true.
    condition_feature : str
        Numeric feature compared with threshold.
    operator : {'greater', 'greater_equal', 'less', 'less_equal', 'equal', 'not_equal'}
        Comparison applied to condition_feature.
    threshold : numbers.Real
        Finite researcher-supplied comparison threshold.
    otherwise : numbers.Real | None, default 0.0
        Finite fallback value. None produces missing values.
    symbol_col : str, default 'symbol'
        Asset identifier column.
    time_col : str, default 'time'
        Observation-time column.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Input columns plus descriptive conditional-feature columns.

    Raises
    ------
    KeyError
        If a required column is absent.
    TypeError
        If an argument has an unsupported type.
    ValueError
        If inputs, operator, threshold, or otherwise are invalid.

    Notes
    -----
    Missing condition values do not satisfy the condition and therefore use
    otherwise. The operation is row-wise and does not estimate a threshold
    from future observations.
    """
    features = _validate_interaction_inputs(
        df,
        feature_cols,
        condition_feature,
        symbol_col,
        time_col,
    )
    operators = {
        'greater',
        'greater_equal',
        'less',
        'less_equal',
        'equal',
        'not_equal',
    }
    if operator not in operators:
        raise ValueError(f'operator must be one of {sorted(operators)}.')
    normalized_threshold = _validate_finite_number(threshold, 'threshold')
    normalized_otherwise = (
        None if otherwise is None else _validate_finite_number(otherwise, 'otherwise')
    )
    output_suffix = f'where_{condition_feature}_{operator}_{threshold}'

    if isinstance(df, pd.DataFrame):
        condition_values = df[condition_feature]
        comparisons = {
            'greater': condition_values.gt(normalized_threshold),
            'greater_equal': condition_values.ge(normalized_threshold),
            'less': condition_values.lt(normalized_threshold),
            'less_equal': condition_values.le(normalized_threshold),
            'equal': condition_values.eq(normalized_threshold),
            'not_equal': condition_values.ne(normalized_threshold),
        }
        condition = comparisons[operator] & condition_values.notna()
        result = df.copy()
        for feature in features:
            result[f'{feature}_{output_suffix}'] = result[feature].where(
                condition,
                normalized_otherwise,
            )
        return result

    condition_values = pl.col(condition_feature).fill_nan(None)
    comparisons = {
        'greater': condition_values > normalized_threshold,
        'greater_equal': condition_values >= normalized_threshold,
        'less': condition_values < normalized_threshold,
        'less_equal': condition_values <= normalized_threshold,
        'equal': condition_values == normalized_threshold,
        'not_equal': condition_values != normalized_threshold,
    }
    condition = comparisons[operator].fill_null(False)
    return df.with_columns([
        pl.when(condition)
        .then(pl.col(feature))
        .otherwise(normalized_otherwise)
        .alias(f'{feature}_{output_suffix}')
        for feature in features
    ])
