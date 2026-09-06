from numbers import Real
from typing import Literal

import polars as pl
import pandas as pd
import numpy as np

from alpha_research._utils import _validate_df, _validate_positive_integer

__all__ = ['fwd_returns', 'ohlc_triple_barrier_labels']


def fwd_returns(
        df: pd.DataFrame | pl.DataFrame,
        horizon: int,
        price_col: str = 'close',
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Compute the forward returns on all symbols.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Wide DataFrame (usually a full OHLCV) with at least the columns price_col,
        symbol_col and time_col.
    horizon : int
        Forward horizon in which the forward returns will be computed.
    price_col : str
        Price column in which the forward returns will be computed. Usually the close
        price column.
    symbol_col : str
        Column which contains the asset names of the dataset.
    time_col : str
        Time column of the dataset.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        DataFrame with the time_col, symbol_col and the fwd_ret_{horizon} with
        the computed forward cross-sectional returns.

    Raises
    ------
    KeyError
        If time_col, symbol_col, price_col are not columns of the df.
    TypeError
        If df is not pandas or polars type.
    """
    _validate_df(df, [time_col, symbol_col, price_col])

    if isinstance(df, pd.DataFrame):
        result = df[[time_col, symbol_col]].copy()
        fwd_price = df.groupby(symbol_col)[price_col].shift(-horizon)
        result[f'fwd_ret_{horizon}'] = fwd_price / df[price_col] - 1
        return result

    else:  # pl.Dataframe type
        return df.select([
            time_col,
            symbol_col,
            (pl.col(price_col).shift(-horizon) / pl.col(price_col) - 1).over(symbol_col)
            .alias(f'fwd_ret_{horizon}')])


def ohlc_triple_barrier_labels(
        df: pd.DataFrame | pl.DataFrame,
        horizon: int,
        upper_barrier: float | str,
        lower_barrier: float | str,
        vertical_label: Literal['return_sign', 'neutral'] = 'return_sign',
        tie_break: Literal['neutral', 'take_profit', 'stop_loss'] = 'neutral',
        entry_col: str = 'close',
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Label every OHLC observation with the Triple-Barrier Method.

    A position is entered at ``entry_col`` on each row. Starting at the following
    bar, the function checks whether the upper or lower horizontal barrier is
    touched first using ``high_col`` and ``low_col``. If neither horizontal
    barrier is touched within ``horizon`` bars, the vertical barrier determines
    the label.

    This is a bar-based implementation. OHLC data cannot establish the order of
    an upper and lower barrier touch within the same bar; ``tie_break`` defines
    the convention used for that case.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        OHLC data containing one or more chronologically ordered symbols.
    horizon : int
        Number of future bars until the vertical barrier.
    upper_barrier : float or str
        Positive relative take-profit distance. A scalar applies to every event.
        A column name uses the positive distance on each event row, allowing
        barriers calculated causally from, for example, volatility or a regime.
    lower_barrier : float or str
        Positive relative stop-loss distance. A scalar applies to every event.
        A column name uses the positive distance on each event row.
    vertical_label : {'return_sign', 'neutral'}, default 'return_sign'
        Label when the vertical barrier is touched first. ``'return_sign'`` uses
        the sign of the return from entry to ``close_col`` at the horizon;
        ``'neutral'`` always returns zero.
    tie_break : {'neutral', 'take_profit', 'stop_loss'}, default 'neutral'
        Label when both horizontal barriers are touched in the same OHLC bar.
    entry_col : str, default 'close'
        Entry-price column.
    high_col : str, default 'high'
        Bar-high column used to test the upper barrier.
    low_col : str, default 'low'
        Bar-low column used to test the lower barrier.
    close_col : str, default 'close'
        Close-price column used to evaluate a vertical-barrier return.
    symbol_col : str, default 'symbol'
        Asset identifier column.
    time_col : str, default 'time'
        Observation-time column. Times must be unique and increasingly ordered
        within each symbol.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        DataFrame containing ``time_col``, ``symbol_col``, and
        ``triple_barrier_{horizon}``. Rows without a complete horizon, or with
        missing data needed to determine the first touch, receive a missing
        label.

    Raises
    ------
    KeyError
        If a required input or barrier column is missing.
    TypeError
        If df is not pandas or polars, or a barrier is neither a scalar nor a
        column name.
    ValueError
        If parameters are invalid, barrier distances are not positive, or the
        time series is not uniquely and increasingly ordered per symbol.
    """
    _validate_positive_integer(horizon, 'horizon')
    _validate_barrier_spec(upper_barrier, 'upper_barrier')
    _validate_barrier_spec(lower_barrier, 'lower_barrier')

    if vertical_label not in {'return_sign', 'neutral'}:
        raise ValueError("vertical_label must be 'return_sign' or 'neutral'.")

    if tie_break not in {'neutral', 'take_profit', 'stop_loss'}:
        raise ValueError(
            "tie_break must be 'neutral', 'take_profit', or 'stop_loss'."
        )

    required_cols = [
        time_col, symbol_col, entry_col, high_col, low_col, close_col,
    ]
    required_cols.extend(
        barrier for barrier in (upper_barrier, lower_barrier)
        if isinstance(barrier, str)
    )
    _validate_df(df, list(dict.fromkeys(required_cols)))

    symbols = _as_numpy_values(df, symbol_col)
    times = _as_numpy_values(df, time_col)
    group_indices = _validate_and_group_times(symbols, times, symbol_col, time_col)

    entry = _as_float_values(df, entry_col)
    high = _as_float_values(df, high_col)
    low = _as_float_values(df, low_col)
    close = _as_float_values(df, close_col)
    upper = _barrier_values(df, upper_barrier, 'upper_barrier')
    lower = _barrier_values(df, lower_barrier, 'lower_barrier')

    labels = np.full(len(df), np.nan, dtype=float)
    for indices in group_indices:
        labels[indices] = _triple_barrier_group_labels(
            entry=entry[indices],
            high=high[indices],
            low=low[indices],
            close=close[indices],
            upper=upper[indices],
            lower=lower[indices],
            horizon=horizon,
            vertical_label=vertical_label,
            tie_break=tie_break,
        )

    label_col = f'triple_barrier_{horizon}'
    if isinstance(df, pd.DataFrame):
        result = df[[time_col, symbol_col]].copy()
        result[label_col] = labels
        return result

    return df.select([time_col, symbol_col]).with_columns(
        pl.Series(label_col, labels).fill_nan(None)
    )


def _validate_barrier_spec(value: float | str, name: str) -> None:
    """
    Validate a scalar or column-name barrier specification.

    Parameters
    ----------
    value : float or str
        Positive scalar barrier distance or the name of a barrier column.
    name : str
        Argument name used in validation error messages.

    Returns
    -------
    None
        This function returns None when the specification is valid.

    Raises
    ------
    TypeError
        If value is neither a scalar nor a column name.
    ValueError
        If a scalar value is not positive and finite.
    """
    if isinstance(value, str):
        return

    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f'{name} must be a positive scalar or a column name.')

    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite scalar.')


def _as_numpy_values(
        df: pd.DataFrame | pl.DataFrame,
        column: str,
) -> np.ndarray:
    """
    Return a column as a NumPy array without changing its logical values.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Source DataFrame containing column.
    column : str
        Name of the column to extract.

    Returns
    -------
    np.ndarray
        Values from column in input row order.

    Raises
    ------
    KeyError
        If column is not present in df.
    """
    if isinstance(df, pd.DataFrame):
        return df[column].to_numpy()

    return df.get_column(column).to_numpy()


def _as_float_values(
        df: pd.DataFrame | pl.DataFrame,
        column: str,
) -> np.ndarray:
    """
    Return a numeric column as floats with missing values represented by NaN.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Source DataFrame containing column.
    column : str
        Name of the numeric column to extract.

    Returns
    -------
    np.ndarray
        Floating-point column values in input row order.

    Raises
    ------
    KeyError
        If column is not present in df.
    TypeError
        If column cannot be converted to floating-point values.
    """
    values = _as_numpy_values(df, column)

    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f'{column} must contain numeric values.') from error


def _validate_and_group_times(
        symbols: np.ndarray,
        times: np.ndarray,
        symbol_col: str,
        time_col: str,
) -> list[np.ndarray]:
    """
    Validate the per-symbol time contract and return input-order indices.

    Parameters
    ----------
    symbols : np.ndarray
        Asset identifiers in input row order.
    times : np.ndarray
        Observation times corresponding to symbols.
    symbol_col : str
        Symbol-column name used in validation error messages.
    time_col : str
        Time-column name used in validation error messages.

    Returns
    -------
    list[np.ndarray]
        One positional-index array per symbol, preserving input symbol order.

    Raises
    ------
    ValueError
        If symbols or times are missing, keys are duplicated, or times are not
        increasingly ordered within a symbol.
    """
    if pd.isna(symbols).any():
        raise ValueError(f'{symbol_col} must not contain missing values.')

    if pd.isna(times).any():
        raise ValueError(f'{time_col} must not contain missing values.')

    keys = pd.DataFrame({symbol_col: symbols, time_col: times})
    if keys.duplicated().any():
        raise ValueError(f'df must contain unique [{symbol_col}, {time_col}] combinations.')

    codes, _ = pd.factorize(symbols, sort=False)
    groups = [np.flatnonzero(codes == code) for code in range(codes.max() + 1)]

    for indices in groups:
        if not pd.Index(times[indices]).is_monotonic_increasing:
            raise ValueError(f'{time_col} must be increasingly ordered within each symbol.')

    return groups


def _barrier_values(
        df: pd.DataFrame | pl.DataFrame,
        barrier: float | str,
        name: str,
) -> np.ndarray:
    """
    Resolve a scalar or per-event barrier column to positive distances.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Source DataFrame used when barrier is a column name.
    barrier : float or str
        Positive scalar distance or the name of a per-event distance column.
    name : str
        Argument name used in validation error messages.

    Returns
    -------
    np.ndarray
        Per-row barrier distances, with missing column values preserved as NaN.

    Raises
    ------
    KeyError
        If a barrier column is not present in df.
    TypeError
        If a barrier column cannot be converted to floating-point values.
    ValueError
        If a non-missing barrier-column value is not positive and finite.
    """
    if isinstance(barrier, str):
        values = _as_float_values(df, barrier)
        non_missing = ~np.isnan(values)

        if np.isinf(values).any() or (values[non_missing] <= 0).any():
            raise ValueError(f'{name} column values must be positive and finite.')

        return values

    return np.full(len(df), barrier, dtype=float)


def _triple_barrier_group_labels(
        entry: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        upper: np.ndarray,
        lower: np.ndarray,
        horizon: int,
        vertical_label: str,
        tie_break: str,
) -> np.ndarray:
    """
    Vectorize Triple-Barrier labels for one ordered symbol series.

    Parameters
    ----------
    entry : np.ndarray
        Per-event entry prices.
    high : np.ndarray
        Bar-high prices in chronological order.
    low : np.ndarray
        Bar-low prices in chronological order.
    close : np.ndarray
        Bar-close prices in chronological order.
    upper : np.ndarray
        Positive relative upper-barrier distances.
    lower : np.ndarray
        Positive relative lower-barrier distances.
    horizon : int
        Number of future bars until the vertical barrier.
    vertical_label : str
        Vertical-barrier labeling convention.
    tie_break : str
        Intrabar simultaneous-touch convention.

    Returns
    -------
    np.ndarray
        Triple-Barrier labels aligned to the input group. Rows without a complete
        horizon or enough valid data are NaN.

    Raises
    ------
    ValueError
        If the supplied arrays do not have compatible one-dimensional shapes.
    """
    labels = np.full(len(entry), np.nan, dtype=float)
    n_events = len(entry) - horizon
    if n_events <= 0:
        return labels

    upper_prices = entry[:n_events] * (1 + upper[:n_events])
    lower_prices = entry[:n_events] * (1 - lower[:n_events])
    valid_event = (
        np.isfinite(entry[:n_events])
        & (entry[:n_events] > 0)
        & np.isfinite(upper[:n_events])
        & np.isfinite(lower[:n_events])
        & np.isfinite(upper_prices)
        & np.isfinite(lower_prices)
    )

    high_windows = np.lib.stride_tricks.sliding_window_view(high, horizon + 1)[:, 1:]
    low_windows = np.lib.stride_tricks.sliding_window_view(low, horizon + 1)[:, 1:]
    close_at_horizon = close[horizon:]
    batch_size = max(1, 1_000_000 // horizon)

    for start in range(0, n_events, batch_size):
        stop = min(start + batch_size, n_events)
        labels[start:stop] = _triple_barrier_batch_labels(
            high=high_windows[start:stop],
            low=low_windows[start:stop],
            entry=entry[start:stop],
            upper_price=upper_prices[start:stop],
            lower_price=lower_prices[start:stop],
            close_at_horizon=close_at_horizon[start:stop],
            valid_event=valid_event[start:stop],
            vertical_label=vertical_label,
            tie_break=tie_break,
        )

    return labels


def _triple_barrier_batch_labels(
        high: np.ndarray,
        low: np.ndarray,
        entry: np.ndarray,
        upper_price: np.ndarray,
        lower_price: np.ndarray,
        close_at_horizon: np.ndarray,
        valid_event: np.ndarray,
        vertical_label: str,
        tie_break: str,
) -> np.ndarray:
    """
    Resolve first touches for a bounded batch of event paths.

    Parameters
    ----------
    high : np.ndarray
        Two-dimensional high-price paths, one row per event.
    low : np.ndarray
        Two-dimensional low-price paths, one row per event.
    entry : np.ndarray
        Entry prices for batch events.
    upper_price : np.ndarray
        Absolute upper-barrier prices for batch events.
    lower_price : np.ndarray
        Absolute lower-barrier prices for batch events.
    close_at_horizon : np.ndarray
        Close prices at the vertical barrier.
    valid_event : np.ndarray
        Boolean mask identifying events with valid entry and barrier values.
    vertical_label : str
        Vertical-barrier labeling convention.
    tie_break : str
        Intrabar simultaneous-touch convention.

    Returns
    -------
    np.ndarray
        Resolved labels for the batch, with undetermined events set to NaN.

    Raises
    ------
    ValueError
        If path arrays or event vectors have incompatible shapes.
    """
    labels = np.full(len(entry), np.nan, dtype=float)
    valid_bars = np.isfinite(high) & np.isfinite(low)
    upper_touch = high >= upper_price[:, None]
    lower_touch = low <= lower_price[:, None]
    touch = upper_touch | lower_touch
    has_touch = touch.any(axis=1)
    first_touch = touch.argmax(axis=1)
    rows = np.arange(len(entry))
    path_valid_through_touch = (
        np.cumsum(~valid_bars, axis=1)[rows, first_touch] == 0
    )

    resolved = valid_event & has_touch & path_valid_through_touch
    if resolved.any():
        first_upper = upper_touch[rows, first_touch]
        first_lower = lower_touch[rows, first_touch]
        labels[resolved & first_upper & ~first_lower] = 1.0
        labels[resolved & first_lower & ~first_upper] = -1.0

        simultaneous = resolved & first_upper & first_lower
        if tie_break == 'take_profit':
            labels[simultaneous] = 1.0
        elif tie_break == 'stop_loss':
            labels[simultaneous] = -1.0
        else:
            labels[simultaneous] = 0.0

    vertical = valid_event & ~has_touch & valid_bars.all(axis=1)
    if vertical_label == 'neutral':
        labels[vertical] = 0.0
    else:
        valid_vertical_close = vertical & np.isfinite(close_at_horizon)
        returns = close_at_horizon[valid_vertical_close] / entry[valid_vertical_close] - 1
        labels[valid_vertical_close] = np.sign(returns)

    return labels
