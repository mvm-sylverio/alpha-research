import builtins

import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.visualization import (
    plot_cross_sectional_value_summary,
    plot_time_series_value,
)


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_cross_sectional_plot_uses_independent_sorted_quantiles(constructor, plot_axis):
    """Should draw per-date medians and linearly interpolated quartiles from finite assets."""
    frame = constructor({'time': [2, 1, 2, 1, 1, 1], 'symbol': ['A', 'A', 'B', 'B', 'C', 'D'],
                         'value': [8., 0., 12., 4., np.inf, None]})
    original = frame.copy(deep=True) if isinstance(frame, pd.DataFrame) else frame.clone()
    result = plot_cross_sectional_value_summary(frame, ax=plot_axis)
    # Two valid assets per date give quartiles one quarter along each pair.
    np.testing.assert_array_equal(result.lines[0].get_xdata(), [1, 2])
    np.testing.assert_allclose(result.lines[0].get_ydata(), [2., 10.])
    vertices = result.collections[0].get_paths()[0].vertices
    for x, low, high in [(1, 1., 3.), (2, 9., 11.)]:
        ys = vertices[vertices[:, 0] == x, 1]
        assert (ys.min(), ys.max()) == pytest.approx((low, high))
    if isinstance(frame, pd.DataFrame):
        pd.testing.assert_frame_equal(frame, original)
    else:
        assert frame.equals(original)


@pytest.mark.parametrize('function', [plot_time_series_value, plot_cross_sectional_value_summary])
@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('invalid', ['duplicate', 'missing_time', 'missing_symbol', 'infinite_time'])
def test_value_plots_reject_invalid_observation_identity(function, constructor, invalid, plot_axis):
    """Should reject invalid observation keys before plotting or cross-sectional aggregation."""
    data = {'time': [1., 2.], 'symbol': ['A', 'A'], 'value': [1., 2.]}
    if invalid == 'duplicate':
        data['time'] = [1., 1.]
    elif invalid == 'missing_time':
        data['time'][1] = np.nan
    elif invalid == 'missing_symbol':
        data['symbol'][1] = None
    else:
        data['time'][1] = np.inf
    with pytest.raises(ValueError):
        function(constructor(data), ax=plot_axis)
    assert not plot_axis.lines and not plot_axis.collections


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_time_series_plot_validates_order_and_preserves_finite_gaps(constructor, plot_axis):
    """Should preserve chronological coordinates and mark missing or infinite values as gaps."""
    frame = constructor({'time': [1, 2, 3, 4], 'symbol': ['A'] * 4, 'value': [1., np.inf, None, 4.]})
    result = plot_time_series_value(frame, ax=plot_axis)
    np.testing.assert_array_equal(result.lines[0].get_xdata(), [1, 2, 3, 4])
    np.testing.assert_allclose(result.lines[0].get_ydata(), [1., np.nan, np.nan, 4.], equal_nan=True)
    reversed_frame = frame.iloc[::-1] if isinstance(frame, pd.DataFrame) else frame.reverse()
    with pytest.raises(ValueError, match='increasingly ordered'):
        plot_time_series_value(reversed_frame, ax=plot_axis)


@pytest.mark.parametrize('function', [plot_time_series_value, plot_cross_sectional_value_summary])
def test_value_plots_reject_text_instead_of_dropping_it(function, plot_axis):
    """Should fail on invalid values even when another value is drawable."""
    frame = pd.DataFrame({'time': [1, 2], 'symbol': ['A', 'A'], 'value': ['1', 'invalid']})
    with pytest.raises(ValueError, match='numeric'):
        function(frame, ax=plot_axis)


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_time_series_plot_preserves_timezone_coordinates(constructor, plot_axis):
    """Should display the same instants when the selected time column carries a timezone."""
    dates = pd.date_range('1970-01-02', periods=2, freq='D', tz='UTC').tz_convert('America/New_York')
    result = plot_time_series_value(constructor({'time': dates, 'symbol': ['A', 'A'], 'value': [1., 2.]}), ax=plot_axis)
    assert list(result.lines[0].get_xdata()) == list(dates)
    # Both instants are exact day offsets from Matplotlib's default Unix epoch.
    np.testing.assert_allclose(result.lines[0].get_xdata(orig=False), [1., 2.])


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_cross_sectional_plot_keeps_empty_dates_as_gaps(constructor, plot_axis):
    """Should retain dates with no finite assets rather than connecting across an omitted date."""
    frame = constructor({'time': [1, 2, 3], 'symbol': ['A'] * 3, 'value': [1., np.inf, 3.]})
    result = plot_cross_sectional_value_summary(frame, ax=plot_axis)
    np.testing.assert_array_equal(result.lines[0].get_xdata(), [1, 2, 3])
    np.testing.assert_allclose(result.lines[0].get_ydata(), [1., np.nan, 3.], equal_nan=True)


# ------------------------------------------------------
# fixtures
# ------------------------------------------------------
@pytest.fixture
def value_frame_pandas():
    """Create interleaved feature-like observations for two assets and dates."""
    return pd.DataFrame({
        'time': pd.to_datetime([
            '2024-01-01', '2024-01-01', '2024-01-01',
            '2024-01-02', '2024-01-02', '2024-01-02',
        ]),
        'symbol': ['AAPL', 'MSFT', 'NVDA', 'AAPL', 'MSFT', 'NVDA'],
        'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })


# ------------------------------------------------------
# plot_time_series_value
# ------------------------------------------------------
def test_plot_time_series_value_requires_a_symbol_for_multiple_assets(
        value_frame_pandas,
):
    """Should not choose an arbitrary asset from a multi-asset value frame."""
    with pytest.raises(ValueError, match='symbol must be provided'):
        plot_time_series_value(value_frame_pandas)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_plot_time_series_value_draws_the_selected_asset(
        value_frame_pandas,
        backend,
):
    """Should draw only the requested feature or target values through time."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frame = value_frame_pandas if backend == 'pandas' else pl.from_pandas(value_frame_pandas)
    figure, axis = plt.subplots()
    returned_axis = plot_time_series_value(
        frame,
        symbol='MSFT',
        ax=axis,
        label='MSFT value',
        title='Selected value',
    )

    assert returned_axis is axis
    assert len(axis.lines) == 1
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [2.0, 5.0])
    assert axis.lines[0].get_label() == 'MSFT value'
    assert axis.get_xlabel() == 'time'
    assert axis.get_ylabel() == 'value'
    assert axis.get_title() == 'Selected value'
    plt.close(figure)


def test_plot_time_series_value_uses_the_only_available_symbol(value_frame_pandas):
    """Should infer symbol only when the input is already single-asset."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    single_asset = value_frame_pandas.loc[value_frame_pandas['symbol'] == 'AAPL']
    figure, axis = plt.subplots()
    plot_time_series_value(single_asset, ax=axis)

    np.testing.assert_allclose(axis.lines[0].get_ydata(), [1.0, 4.0])
    plt.close(figure)


@pytest.mark.parametrize(
    'frame, value_col, error_type, message',
    [
        (pd.DataFrame({'time': [1], 'symbol': ['A']}), None, ValueError, 'exactly one'),
        (pd.DataFrame({'time': [1], 'symbol': ['A'], 'a': [1], 'b': [2]}), None, ValueError, 'exactly one'),
        (pd.DataFrame({'time': [1], 'symbol': ['A'], 'value': [1]}), 'missing', KeyError, 'missing required columns'),
    ],
)
def test_plot_time_series_value_validates_value_schema(
        frame,
        value_col,
        error_type,
        message,
):
    """Should validate the generic single-value frame contract before plotting."""
    with pytest.raises(error_type, match=message):
        plot_time_series_value(frame, value_col=value_col)


def test_plot_time_series_value_rejects_unknown_or_all_missing_selection(
        value_frame_pandas,
):
    """Should report unavailable symbols and unplottable value series."""
    with pytest.raises(ValueError, match='not present'):
        plot_time_series_value(value_frame_pandas, symbol='GOOGL')

    all_missing = value_frame_pandas.loc[
        value_frame_pandas['symbol'] == 'AAPL',
    ].assign(value=np.nan)
    with pytest.raises(ValueError, match='finite value'):
        plot_time_series_value(all_missing)


def test_plot_time_series_value_accepts_an_explicit_value_column(
        value_frame_pandas,
):
    """Should support a research frame containing more than one value column."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    multi_value_frame = value_frame_pandas.assign(other_value=[6, 5, 4, 3, 2, 1])
    figure, axis = plt.subplots()
    plot_time_series_value(
        multi_value_frame,
        symbol='AAPL',
        value_col='other_value',
        ax=axis,
    )

    np.testing.assert_allclose(axis.lines[0].get_ydata(), [6.0, 3.0])
    assert axis.get_ylabel() == 'other_value'
    plt.close(figure)


# ------------------------------------------------------
# plot_cross_sectional_value_summary
# ------------------------------------------------------
@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_plot_cross_sectional_value_summary_draws_median_and_quantile_band(
        value_frame_pandas,
        backend,
):
    """Should summarize each date across assets for either supported backend."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frame = value_frame_pandas if backend == 'pandas' else pl.from_pandas(value_frame_pandas)
    figure, axis = plt.subplots()
    returned_axis = plot_cross_sectional_value_summary(
        frame,
        ax=axis,
        label='Cross-sectional median',
        title='Cross-sectional value',
    )

    assert returned_axis is axis
    assert len(axis.lines) == 1
    assert len(axis.collections) == 1
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [2.0, 5.0])
    assert axis.lines[0].get_label() == 'Cross-sectional median'
    assert axis.get_title() == 'Cross-sectional value'
    plt.close(figure)


@pytest.mark.parametrize(
    'lower_quantile, upper_quantile, band_alpha, error_type, message',
    [
        (0.75, 0.25, 0.2, ValueError, 'lower_quantile'),
        (-0.1, 0.75, 0.2, ValueError, 'lower_quantile'),
        (0.25, 1.1, 0.2, ValueError, 'upper_quantile'),
        (0.25, 0.75, -0.1, ValueError, 'band_alpha'),
        (0.25, 0.75, 1.1, ValueError, 'band_alpha'),
        ('0.25', 0.75, 0.2, TypeError, 'lower_quantile'),
        (0.25, 0.75, True, TypeError, 'band_alpha'),
    ],
)
def test_plot_cross_sectional_value_summary_validates_parameters(
        value_frame_pandas,
        lower_quantile,
        upper_quantile,
        band_alpha,
        error_type,
        message,
):
    """Should reject invalid band parameters before importing Matplotlib."""
    with pytest.raises(error_type, match=message):
        plot_cross_sectional_value_summary(
            value_frame_pandas,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            band_alpha=band_alpha,
        )


def test_plot_cross_sectional_value_summary_rejects_all_missing_values(
        value_frame_pandas,
):
    """Should not create a summary panel without any finite values."""
    with pytest.raises(ValueError, match='finite value'):
        plot_cross_sectional_value_summary(value_frame_pandas.assign(value=np.nan))


@pytest.mark.parametrize(
    'plot_function, kwargs',
    [
        (plot_time_series_value, {'symbol': 'AAPL'}),
        (plot_cross_sectional_value_summary, {}),
    ],
)
@pytest.mark.parametrize(
    'frame, value_col, error_type, message',
    [
        (pd.DataFrame({'time': [1], 'symbol': ['A']}), None, ValueError, 'exactly one'),
        (
            pd.DataFrame({'time': [1], 'symbol': ['A'], 'a': [1], 'b': [2]}),
            None,
            ValueError,
            'exactly one',
        ),
        (
            pd.DataFrame({'time': [1], 'symbol': ['A'], 'value': [1]}),
            'missing',
            KeyError,
            'missing required columns',
        ),
    ],
)
def test_value_plot_functions_validate_the_generic_value_schema(
        plot_function,
        kwargs,
        frame,
        value_col,
        error_type,
        message,
):
    """Should enforce the shared feature-or-target single-value contract."""
    with pytest.raises(error_type, match=message):
        plot_function(frame, value_col=value_col, **kwargs)


def test_plot_cross_sectional_value_summary_accepts_an_explicit_value_column(
        value_frame_pandas,
):
    """Should summarize a selected value column from a multi-value research frame."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    multi_value_frame = value_frame_pandas.assign(other_value=[6, 5, 4, 3, 2, 1])
    figure, axis = plt.subplots()
    plot_cross_sectional_value_summary(
        multi_value_frame,
        value_col='other_value',
        ax=axis,
    )

    np.testing.assert_allclose(axis.lines[0].get_ydata(), [5.0, 2.0])
    assert axis.get_ylabel() == 'other_value'
    plt.close(figure)


# ------------------------------------------------------
# Optional Matplotlib backend
# ------------------------------------------------------
@pytest.mark.parametrize(
    'plot_function, kwargs',
    [
        (plot_time_series_value, {'symbol': 'AAPL'}),
        (plot_cross_sectional_value_summary, {}),
    ],
)
def test_value_plot_functions_explain_missing_optional_backend(
        monkeypatch,
        value_frame_pandas,
        plot_function,
        kwargs,
):
    """Should explain how to install the optional plotting dependency."""
    original_import = builtins.__import__

    def raise_matplotlib_import_error(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'matplotlib.pyplot':
            raise ImportError('simulated missing matplotlib')

        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', raise_matplotlib_import_error)

    with pytest.raises(ImportError, match=r'alpha-research\[viz\]'):
        plot_function(value_frame_pandas, **kwargs)
