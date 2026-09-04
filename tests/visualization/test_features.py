import builtins

import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.visualization import (
    plot_cross_sectional_value_summary,
    plot_time_series_value,
)


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
