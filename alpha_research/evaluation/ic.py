import polars as pl
import numpy as np
from typing import Literal


def compute_IC(df: pl.DataFrame,
               feature_col: str,
               target_col: str,
               corr_method: Literal['pearson', 'spearman'] = 'pearson'):
    """
    Information Coefficient of a feature and a target.
    """

    # Check if columns exist in df
    assert feature_col in df.columns, f'{feature_col} does not exist in df'
    assert target_col in df.columns, f'{target_col} does not exist in df'

    # remove nulls
    df_ic = df.drop_nulls(subset=[feature_col, target_col])

    # group by time
    df_ic = (
        df_ic
        .group_by("time")
        .agg(
            pl.corr(feature_col,
                    target_col,
                    method=corr_method).alias(f"IC_{target_col}")
        )
        .sort('time')
    )

    return df_ic


def IC_metrics(df_ic: pl.DataFrame,
               target_col: str):
    """
    Computes mean, std and estability index of IC serie.
    """

    ic_serie = df_ic[f'IC_{target_col}'].to_numpy()

    ic_mean = np.mean(ic_serie)

    # ddof=1 -> sample
    ic_std = np.std(ic_serie, ddof=1)

    return {
        "IC_mean": float(ic_mean),
        "IC_std": float(ic_std),
        "IC_estability": float(ic_mean / ic_std) if ic_std > 0 else np.nan,
        'T': len(ic_serie)
    }


# --------------------------
# Plots
# --------------------------
