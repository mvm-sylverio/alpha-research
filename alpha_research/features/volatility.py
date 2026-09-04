import numpy as np
import pandas as pd
import polars as pl

from alpha_research._utils import _validate_df, _validate_positive_integer
from alpha_research.features.returns import log_returns
from alpha_research.features.schema import get_feature_name

__all__ = [
    'realized_volatility',
    'average_true_range',
    'atr',
]


def _validate_annualization_factor(
        annualization_factor: int | float | None,
) -> None:
    """
    Validate an optional positive annualization factor.

    Parameters
    ----------
    annualization_factor : int | float | None
        Number of observations per annual period, or None to keep the feature
        in its original bar-frequency units.

    Returns
    -------
    None
        Returns None when annualization_factor is valid.

    Raises
    ------
    ValueError
        If annualization_factor is neither None nor a finite positive number.
    """
    if annualization_factor is None:
        return

    if (
            not isinstance(annualization_factor, (int, float, np.integer, np.floating))
            or isinstance(annualization_factor, bool)
            or not np.isfinite(annualization_factor)
            or annualization_factor <= 0
    ):
        raise ValueError(
            'annualization_factor must be a finite positive number or None.',
        )


def _format_annualization_factor(annualization_factor: int | float) -> str:
    """
    Format an annualization factor for an unambiguous feature name to be
    returned in correct str format.

    Parameters
    ----------
    annualization_factor : int | float
        Previously validated positive annualization factor.

    Returns
    -------
    str
        String representation suitable for use in a feature column name.
    """
    factor = float(annualization_factor)
    if factor.is_integer():
        return str(int(factor))

    return str(factor).replace('.', '_')


def _asset_positions(symbols: list[object]) -> list[list[int]]:
    """
    Return input-row positions grouped by symbol while preserving row order.

    Parameters
    ----------
    symbols : list[object]
        Asset identifier values in input-row order.

    Returns
    -------
    list[list[int]]
        One list of row positions for each distinct symbol. Positions within
        every asset list retain their original temporal order.
    """
    positions_by_symbol: dict[object, list[int]] = {}

    for position, symbol in enumerate(symbols):
        positions_by_symbol.setdefault(symbol, []).append(position)

    return list(positions_by_symbol.values())


def _true_range_by_symbol(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        symbols: list[object],
) -> np.ndarray:
    """
    Compute per-symbol True Range with high-low used at each series start.

    Parameters
    ----------
    high, low, close : np.ndarray
        OHLC values in input-row order.
    symbols : list[object]
        Asset identifiers aligned to the OHLC arrays.

    Returns
    -------
    np.ndarray
        True Range values aligned to the input rows. The first valid bar of an
        asset uses high-low because no previous close is available.
    """
    true_range = np.full(len(close), np.nan, dtype=float)

    for positions in _asset_positions(symbols):
        previous_close = np.nan

        for position in positions:
            current_high = high[position]
            current_low = low[position]
            current_close = close[position]

            if not np.isfinite(current_high) or not np.isfinite(current_low):
                previous_close = current_close
                continue

            intrabar_range = current_high - current_low
            if np.isfinite(previous_close):
                # max of the 3 possibilities
                true_range[position] = max(
                    intrabar_range,
                    abs(current_high - previous_close),
                    abs(current_low - previous_close),
                )
            else:
                true_range[position] = intrabar_range

            previous_close = current_close

    return true_range


def _wilder_smoothing_by_symbol(
        values: np.ndarray,
        symbols: list[object],
        window: int,
) -> np.ndarray:
    """
    Apply Wilder smoothing independently to contiguous valid asset values.

    Parameters
    ----------
    values : np.ndarray
        Values to smooth in input-row order.
    symbols : list[object]
        Asset identifiers aligned to values.
    window : int
        Positive period used to seed the first arithmetic mean and subsequent
        recursive Wilder updates.

    Returns
    -------
    np.ndarray
        Smoothed values aligned to the input rows. Missing values reset the
        seed, requiring a new complete contiguous window before output resumes.

    Notes
    -----
    Wilder smoothing is initialized with the arithmetic mean of the first
    ``window`` contiguous finite values of each asset. Each subsequent value
    follows ``((window - 1) * previous + current) / window``. It is therefore
    not a simple rolling mean and not an exponentially weighted mean seeded at
    the first observation. A null or NaN clears the current state: output stays
    missing until another full contiguous seed window has been observed.
    """
    smoothed = np.full(len(values), np.nan, dtype=float)

    for positions in _asset_positions(symbols):
        seed_values: list[float] = []
        previous_smoothed = np.nan

        for position in positions:
            value = values[position]

            if not np.isfinite(value):
                seed_values = []
                previous_smoothed = np.nan
                continue

            if len(seed_values) < window:
                seed_values.append(value)
                if len(seed_values) == window:
                    previous_smoothed = float(np.mean(seed_values))
                    smoothed[position] = previous_smoothed
                continue

            previous_smoothed = (
                (window - 1) * previous_smoothed + value
            ) / window
            smoothed[position] = previous_smoothed

    return smoothed


def _average_true_range_values(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        symbols: list[object],
        window: int,
) -> np.ndarray:
    """
    Compute classic Wilder ATR values for each asset in input order.

    Parameters
    ----------
    high, low, close : np.ndarray
        OHLC values in input-row order.
    symbols : list[object]
        Asset identifiers aligned to the OHLC arrays.
    window : int
        Positive Wilder smoothing period.

    Returns
    -------
    np.ndarray
        Average True Range values aligned to the input rows.
    """
    true_range = _true_range_by_symbol(high, low, close, symbols)
    return _wilder_smoothing_by_symbol(true_range, symbols, window)


def _feature_result_frame(
        df: pd.DataFrame | pl.DataFrame,
        values: np.ndarray,
        feature_name: str,
        time_col: str,
        symbol_col: str,
) -> pd.DataFrame | pl.DataFrame:
    """
    Return keys and one computed feature while preserving the input backend.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Source DataFrame whose backend and key columns are retained.
    values : np.ndarray
        Computed feature values aligned one-to-one with df rows.
    feature_name : str
        Output value-column name.
    time_col, symbol_col : str
        Key columns selected into the result.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Feature frame containing the keys and exactly one feature column.
    """
    if isinstance(df, pd.DataFrame):
        result = df[[time_col, symbol_col]].copy()
        result[feature_name] = values
        return result

    return df.select([time_col, symbol_col]).with_columns(
        pl.Series(feature_name, values),
    )


def realized_volatility(
        df: pd.DataFrame | pl.DataFrame,
        window: int,
        price_col: str = 'close',
        annualization_factor: int | float | None = None,
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Compute rolling realized volatility from one-bar log returns per asset.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        OHLCV-like DataFrame containing the temporal key, asset identifier,
        and price column. Rows must be ordered chronologically within each
        asset; this function does not sort them.
    window : int
        Number of past one-bar log returns used in each sample standard
        deviation. It must be greater than one.
    price_col : str, default 'close'
        Positive price column used to construct one-bar log returns.
    annualization_factor : int | float | None, default None
        Optional positive factor applied as ``sqrt(annualization_factor)``.
        For daily observations, 252 is a common explicit choice. When None,
        volatility remains in units of the input bar frequency.
    symbol_col : str, default 'symbol'
        Asset identifier column used to isolate rolling calculations.
    time_col : str, default 'time'
        Temporal key preserved in the result.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        DataFrame with time_col, symbol_col, and ``realized_vol_{window}``.
        When annualization_factor is supplied, the value column is named
        ``realized_vol_{window}_annualized_{factor}`` to preserve its units.

    Raises
    ------
    KeyError
        If a required column is absent.
    TypeError
        If df is not a Pandas or Polars DataFrame.
    ValueError
        If window is not greater than one or annualization_factor is invalid.

    Notes
    -----
    At time t, the feature uses only log returns through t. A window of W
    returns requires W + 1 observed prices before the first non-missing value.
    The sample standard deviation uses ``ddof=1``.
    """
    _validate_df(df, [time_col, symbol_col, price_col])
    _validate_positive_integer(window, 'window')
    _validate_annualization_factor(annualization_factor)

    if window == 1:
        raise ValueError('window must be greater than one for sample volatility.')

    feature_name = f'realized_vol_{window}'
    multiplier = 1.0
    if annualization_factor is not None:
        multiplier = float(np.sqrt(annualization_factor))
        feature_name += (
            f'_annualized_{_format_annualization_factor(annualization_factor)}'
        )

    log_return_frame = log_returns(
        df=df,
        horizon=1,
        price_col=price_col,
        symbol_col=symbol_col,
        time_col=time_col,
    )
    log_return_col = get_feature_name(
        log_return_frame,
        time_col=time_col,
        symbol_col=symbol_col,
    )

    if isinstance(df, pd.DataFrame):
        volatility = log_return_frame[log_return_col].groupby(
            log_return_frame[symbol_col],
            sort=False,
        ).transform(
            lambda values: values.rolling(window, min_periods=window).std(ddof=1),
        ) * multiplier
        result = df[[time_col, symbol_col]].copy()
        result[feature_name] = volatility
        return result

    return log_return_frame.select([
        time_col,
        symbol_col,
        (
            pl.col(log_return_col)
            .rolling_std(window_size=window, min_samples=window, ddof=1)
            .over(symbol_col)
            * multiplier
        ).alias(feature_name),
    ])


def average_true_range(
        df: pd.DataFrame | pl.DataFrame,
        window: int,
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        normalize: bool = True,
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Compute Wilder's Average True Range independently for every asset.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        OHLCV-like DataFrame containing high_col, low_col, close_col, and the
        time and symbol keys. Rows must be ordered chronologically within each
        asset; this function does not sort them.
    window : int
        Number of True Range observations used to initialize Wilder smoothing.
    high_col, low_col, close_col : str
        OHLC column names. True Range considers the intrabar range and gaps
        relative to the preceding close of the same asset.
    normalize : bool, default True
        Whether to divide ATR by the contemporaneous close. True returns the
        scale-comparable fractional ``natr_{window}``; False returns the raw
        price-unit ``atr_{window}``.
    symbol_col : str, default 'symbol'
        Asset identifier column used to isolate calculations.
    time_col : str, default 'time'
        Temporal key preserved in the result.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        DataFrame with time_col, symbol_col, and ``natr_{window}`` when
        normalize is True or ``atr_{window}`` when normalize is False.

    Raises
    ------
    KeyError
        If a required column is absent.
    TypeError
        If df is not a Pandas or Polars DataFrame.
    ValueError
        If window is not a positive integer.
    TypeError
        If normalize is not a boolean.

    Notes
    -----
    True Range is ``max(high - low, abs(high - previous_close),
    abs(low - previous_close))``. The first bar of each asset uses ``high -
    low`` because no preceding close exists. ATR is initialized with the mean
    of the first full True Range window and subsequently follows Wilder's
    recursive smoothing. Normalized ATR is expressed as a fraction, not a
    percentage. Zero or missing close values produce missing normalized values.
    """
    _validate_df(df, [time_col, symbol_col, high_col, low_col, close_col])
    _validate_positive_integer(window, 'window')
    if not isinstance(normalize, bool):
        raise TypeError('normalize must be a boolean.')

    high = np.asarray(df[high_col].to_numpy(), dtype=float)
    low = np.asarray(df[low_col].to_numpy(), dtype=float)
    close = np.asarray(df[close_col].to_numpy(), dtype=float)
    symbols = df[symbol_col].to_list()
    atr = _average_true_range_values(high, low, close, symbols, window)

    values = atr
    feature_name = f'atr_{window}'
    if normalize:
        values = np.divide(
            atr,
            close,
            out=np.full(len(close), np.nan, dtype=float),
            where=np.isfinite(close) & (close != 0),
        )
        feature_name = f'natr_{window}'

    return _feature_result_frame(
        df,
        values,
        feature_name,
        time_col,
        symbol_col,
    )


def atr(
        df: pd.DataFrame | pl.DataFrame,
        window: int,
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        normalize: bool = True,
        symbol_col: str = 'symbol',
        time_col: str = 'time',
) -> pd.DataFrame | pl.DataFrame:
    """
    Alias for function average_true_range.
    See average_true_range() for full documentation.
    """
    return average_true_range(
        df=df,
        window=window,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        normalize=normalize,
        symbol_col=symbol_col,
        time_col=time_col,
    )
