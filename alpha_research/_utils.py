import pandas as pd
import polars as pl


def _validate_df(
        df: pd.DataFrame | pl.DataFrame,
        required_cols: list[str],
) -> None:
    """
    Validate a Dataframe before usage. Includes variable type check
    and columns existence check.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        df which will be validated.
    required_cols : list[str]
        Columns required to exist in the df

    Raises
    ------
    TypeError
        If df is not pandas or polars type.
    KeyError
        If columns in required_cols are not columns of the df.
    """
    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        raise TypeError('df must be Pandas or Polars DataFrame.')

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise KeyError(f'missing required columns in df: {missing_cols}.')

def _is_all_missing(series) -> bool:
    """
    Check if a Series of a DataFrame has only NaN or Null values.

    Parameters
    ----------
    series : pd.Series | pl.Series
        Series which will be checked for NaN or Null values.

    Returns
    -------
    bool
        True if the series has only NaN or Null values, False if not.

    Raises
    ------
    TypeError
        if Series is not Polars or Pandas type.

    Notes
    -----
    Assumes non-empty series — empty check should be performed upstream
    in _validate_df before calling this function.
    Polars treats NaN and null as distinct missing values; both are
    considered missing here. NaN only applies to floating-point columns.
    """
    if isinstance(series, pl.Series):
        if series.dtype.is_float():
            return (series.is_null() | series.is_nan()).all()

        return bool(series.is_null().all())

    if isinstance(series, pd.Series):
        return bool(series.isna().all())

    raise TypeError("Unsupported Series type.")
