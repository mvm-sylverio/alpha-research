import pandas as pd
import polars as pl

from alpha_research._utils import _validate_df

__all__ = ['cross_sectional_rank']


def cross_sectional_rank(
        df: pd.DataFrame | pl.DataFrame,
        feature_cols: str | list[str],
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Compute the cross-sectional assets rank for all the features in feature_cols.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Wide DataFrame with the time_col, symbol_col and all features contained in feature_cols.
    feature_cols : str | list[str]
        All the columns in df which will be cross-sectionally ranked.
    symbol_col : str
        Column which contains the asset names of the dataset.
    time_col : str
        Time column of the dataset.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        DataFrame with the time_col, symbol_col, all the original feature_cols and their respective
        cross-sectional ranks columns.

    Raises
    ------
    KeyError
        If time_col, symbol_col and all the columns in feature_cols are not columns of the df.
    TypeError
        If df is not pandas or polars type.
        If feature_cols is not str or list[str].
    ValueError
        If feature_cols is an empty list.

    Notes
    -----
    Designed to operate on a merged feature DataFrame - typically the output
    of joining multiple feature DataFrames on ['time', 'symbol'] before
    applying cross-sectional transformations.

    Example workflow:
        df = simple_return(ohlcv, horizon=5)
            .merge(fwd_return(ohlcv, horizon=10), on=['time', 'symbol'])
        df_ranked = cross_sectional_rank(df, feature_cols=['simple_ret_5'])

    Ties are broken using the average method.

    Cross-sectional operation: rank is computed per date across assets,
    not along the time axis. Not meaningful for single-asset datasets.
    """
    # initial checks
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]  # transforms to list for iteration later
    elif not isinstance(feature_cols, list):  # not list
        raise TypeError('feature_cols must be str or list[str].')
    elif not feature_cols:  # empty list
        raise ValueError('feature_cols must not be an empty list.')

    _validate_df(df, [symbol_col, time_col] + feature_cols)

    if isinstance(df, pd.DataFrame):
        result = df.copy()
        for feature in feature_cols:
            result[f'{feature}_rank'] = result.groupby(time_col)[feature].rank(method='average')
        return result

    else:  # polars DataFrame
        return df.with_columns([
            pl.col(feature)
            .rank(method='average')
            .over(time_col)
            .alias(f'{feature}_rank')
            for feature in feature_cols
        ])
