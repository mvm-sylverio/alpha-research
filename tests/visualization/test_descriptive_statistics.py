import builtins

import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.evaluation import (
    distribution_summary,
    rolling_distribution_summary,
)
from alpha_research.visualization import (
    plot_distribution_histogram,
    plot_distribution_qq,
    plot_distribution_quantiles,
)
from alpha_research.visualization.descriptive_statistics import _select_plot_values


@pytest.fixture
def distribution_frame_pandas():
    """Create two assets with a separate five-point distribution per asset."""
    return pd.DataFrame({
        'time': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        'symbol': ['A', 'B'] * 5,
        'feature': [0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0, 4.0, 14.0],
    })


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_select_plot_values_uses_only_selected_finite_asset(distribution_frame_pandas, backend):
    """Should select one asset without pooling assets or invalid values."""
    frame = distribution_frame_pandas.copy()
    frame.loc[frame['time'] == 1, 'feature'] = np.nan
    frame = frame if backend == 'pandas' else pl.from_pandas(frame)
    values, name = _select_plot_values(frame, 'A', None, 'time', 'symbol')

    assert name == 'feature'
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0, 4.0])


def test_select_plot_values_rejects_ambiguous_or_empty_asset(distribution_frame_pandas):
    """Should require a symbol for multi-asset input and data for the selection."""
    with pytest.raises(ValueError, match='symbol must be provided'):
        _select_plot_values(distribution_frame_pandas, None, None, 'time', 'symbol')
    with pytest.raises(ValueError, match='not present'):
        _select_plot_values(distribution_frame_pandas, 'C', None, 'time', 'symbol')
    missing = distribution_frame_pandas.copy()
    missing.loc[missing['symbol'] == 'A', 'feature'] = np.nan
    with pytest.raises(ValueError, match='finite value'):
        _select_plot_values(missing, 'A', None, 'time', 'symbol')


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_plot_distribution_histogram_returns_axis_with_density(distribution_frame_pandas, backend):
    """Should plot one asset and return the supplied composable axis."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frame = distribution_frame_pandas if backend == 'pandas' else pl.from_pandas(distribution_frame_pandas)
    figure, axis = plt.subplots()
    returned = plot_distribution_histogram(
        frame, symbol='A', ax=axis, bins=5, density_color='black',
    )

    assert returned is axis
    assert len(axis.patches) == 5
    assert len(axis.lines) == 4
    assert axis.lines[0].get_color() == 'black'
    assert [line.get_xdata()[0] for line in axis.lines[1:]] == pytest.approx([0.2, 2.0, 3.8])
    assert axis.get_xlabel() == 'feature'
    plt.close(figure)


def test_plot_distribution_histogram_handles_constant_and_invalid_density(distribution_frame_pandas):
    """Should plot constant values without an undefined KDE curve."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    single = distribution_frame_pandas.loc[distribution_frame_pandas['symbol'] == 'A'].copy()
    single['feature'] = 2.0
    figure, axis = plt.subplots()
    plot_distribution_histogram(single, ax=axis)
    assert len(axis.lines) == 1
    with pytest.raises(TypeError, match='show_density'):
        plot_distribution_histogram(single, show_density='yes')
    with pytest.raises(TypeError, match='show_quantiles'):
        plot_distribution_histogram(single, show_quantiles='yes')
    plt.close(figure)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_plot_distribution_qq_returns_axis_with_ordered_values(distribution_frame_pandas, backend):
    """Should plot the known ordered sample on the Q-Q observed axis."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frame = distribution_frame_pandas if backend == 'pandas' else pl.from_pandas(distribution_frame_pandas)
    figure, axis = plt.subplots()
    returned = plot_distribution_qq(
        frame, symbol='A', ax=axis, reference_color='black',
    )

    assert returned is axis
    assert len(axis.lines) == 2
    assert axis.lines[1].get_color() == 'black'
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), [0.0, 1.0, 2.0, 3.0, 4.0])
    assert axis.get_ylabel() == 'Observed feature quantiles'
    plt.close(figure)


def test_plot_distribution_qq_requires_two_values(distribution_frame_pandas):
    """Should require two observations to draw a Q-Q reference line."""
    single = distribution_frame_pandas.iloc[[0]]
    with pytest.raises(ValueError, match='at least two'):
        plot_distribution_qq(single)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('rolling', [False, True])
def test_plot_distribution_quantiles_supports_date_and_rolling_tables(
        distribution_frame_pandas,
        backend,
        rolling,
):
    """Should render cross-sectional and rolling summaries on the same axis."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frame = distribution_frame_pandas if backend == 'pandas' else pl.from_pandas(distribution_frame_pandas)
    if rolling:
        frame = frame.filter(pl.col('symbol') == 'A') if backend == 'polars' else frame.loc[frame['symbol'] == 'A']
        summary = rolling_distribution_summary(frame, window_size=3)
        time_col = 'window_end'
        expected_medians = [1.0, 2.0, 3.0]
    else:
        summary = distribution_summary(frame, group_by='time')
        time_col = 'time'
        expected_medians = [5.0, 6.0, 7.0, 8.0, 9.0]

    figure, axis = plt.subplots()
    returned = plot_distribution_quantiles(summary, time_col=time_col, ax=axis)
    assert returned is axis
    assert len(axis.collections) == 2
    assert [band.get_label() for band in axis.collections] == [
        'Q05–Q95', 'Q25–Q75',
    ]
    np.testing.assert_allclose(axis.lines[0].get_ydata(), expected_medians)
    assert axis.get_xlabel() == time_col
    plt.close(figure)


def test_plot_distribution_quantiles_accepts_outer_one_and_ninety_nine(distribution_frame_pandas):
    """Should use the requested extreme quantiles for the outer band."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summary = distribution_summary(distribution_frame_pandas, group_by='time')
    figure, axis = plt.subplots()
    returned = plot_distribution_quantiles(
        summary,
        outer_lower_col='q01',
        outer_upper_col='q99',
        ax=axis,
    )
    assert returned is axis
    assert len(axis.collections) == 2
    assert [band.get_label() for band in axis.collections] == [
        'Q01–Q99', 'Q25–Q75',
    ]
    plt.close(figure)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_distribution_plots_respect_custom_key_columns(distribution_frame_pandas, backend):
    """Should use configured keys for plot selection and time axes in both backends."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    renamed = distribution_frame_pandas.rename(columns={'time': 'date', 'symbol': 'ticker'})
    frame = renamed if backend == 'pandas' else pl.from_pandas(renamed)
    summary = distribution_summary(
        frame, group_by='date', time_col='date', symbol_col='ticker',
    )
    figure, axes = plt.subplots(1, 3)

    assert plot_distribution_histogram(
        frame, symbol='A', time_col='date', symbol_col='ticker', ax=axes[0],
    ) is axes[0]
    assert plot_distribution_qq(
        frame, symbol='A', time_col='date', symbol_col='ticker', ax=axes[1],
    ) is axes[1]
    assert plot_distribution_quantiles(summary, time_col='date', ax=axes[2]) is axes[2]
    assert axes[2].get_xlabel() == 'date'
    np.testing.assert_allclose(axes[2].lines[0].get_ydata(), [5.0, 6.0, 7.0, 8.0, 9.0])
    plt.close(figure)


def test_polars_quantile_plot_accepts_datetime_axis_without_pandas_conversion(
        distribution_frame_pandas,
        monkeypatch,
):
    """Should pass Polars-backed datetime arrays to Matplotlib without conversion."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dated = distribution_frame_pandas.copy()
    dated['time'] = pd.date_range('2024-01-01', periods=5).repeat(2)
    summary = distribution_summary(pl.from_pandas(dated), group_by='time')

    def reject_conversion(*args, **kwargs):
        """Reject an unnecessary full-frame conversion.

        Parameters
        ----------
        *args, **kwargs
            Arguments from a possible conversion call.

        Returns
        -------
        None
            This function never returns.

        Raises
        ------
        AssertionError
            Always, because the plot needs only native vectors.
        """
        raise AssertionError('Polars frame converted to Pandas.')

    monkeypatch.setattr(pl.DataFrame, 'to_pandas', reject_conversion)
    figure, axis = plt.subplots()
    assert plot_distribution_quantiles(summary, ax=axis) is axis
    assert len(axis.lines[0].get_xdata()) == 5
    plt.close(figure)


def test_plot_distribution_quantiles_rejects_invalid_schema_and_bands(distribution_frame_pandas):
    """Should reject malformed summaries instead of drawing misleading quantile bands."""
    summary = distribution_summary(distribution_frame_pandas, group_by='time')
    with pytest.raises(KeyError, match='missing required columns'):
        plot_distribution_quantiles(summary.drop(columns='q25'))
    with pytest.raises(ValueError, match='ordered'):
        plot_distribution_quantiles(summary.iloc[::-1])
    invalid = summary.copy()
    invalid.loc[0, 'q25'] = 100.0
    with pytest.raises(ValueError, match='quantile bands'):
        plot_distribution_quantiles(invalid)
    all_missing = summary.copy()
    all_missing['median'] = np.nan
    with pytest.raises(ValueError, match='finite median'):
        plot_distribution_quantiles(all_missing)
    partial = summary.copy()
    partial.loc[0, 'q05'] = np.nan
    with pytest.raises(ValueError, match='entirely finite or missing'):
        plot_distribution_quantiles(partial)
    nonnumeric = summary.copy()
    nonnumeric['q25'] = 'invalid'
    with pytest.raises(TypeError, match='numeric'):
        plot_distribution_quantiles(nonnumeric)


@pytest.mark.parametrize('plot_name', ['histogram', 'qq', 'quantiles'])
def test_distribution_plots_explain_missing_matplotlib(
        distribution_frame_pandas,
        monkeypatch,
        plot_name,
):
    """Should explain how to install the optional plotting dependency."""
    single = distribution_frame_pandas.loc[distribution_frame_pandas['symbol'] == 'A']
    summary = distribution_summary(distribution_frame_pandas, group_by='time')
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        """Simulate a missing optional Matplotlib installation.

        Parameters
        ----------
        name : str
            Module name being imported.
        *args, **kwargs
            Additional import arguments passed through to the real importer.

        Returns
        -------
        object
            Imported module when it is not Matplotlib.

        Raises
        ------
        ImportError
            If Matplotlib is requested.
        """
        if name.startswith('matplotlib'):
            raise ImportError('simulated missing matplotlib')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', blocked_import)
    with pytest.raises(ImportError, match=r'alpha-research\[viz\]'):
        if plot_name == 'histogram':
            plot_distribution_histogram(single)
        elif plot_name == 'qq':
            plot_distribution_qq(single)
        else:
            plot_distribution_quantiles(summary)
