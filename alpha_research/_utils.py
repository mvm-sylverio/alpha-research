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
