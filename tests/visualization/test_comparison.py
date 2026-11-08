import builtins
from unittest.mock import Mock

import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.visualization import (
    plot_boxplot_comparison,
    plot_ecdf_comparison,
    plot_exceedance_comparison,
    plot_histogram_comparison,
)
from alpha_research.visualization.comparison import (
    _comparison_histogram_edges,
    _comparison_plot,
    _prepare_distributions,
)

PLOTTERS = [plot_histogram_comparison, plot_ecdf_comparison,
            plot_exceedance_comparison, plot_boxplot_comparison]


@pytest.fixture
def comparison_options():
    """Should provide the explicit shared options for direct context-helper tests."""
    return {'colors': None, 'value_label': 'Estimate', 'title': None, 'note': None,
            'reference_value': None, 'reference_label': 'Observed'}


# ------------------------------------------------------
# preparation and shared presentation
# ------------------------------------------------------
@pytest.mark.parametrize('container', ['list', 'tuple', 'numpy', 'pandas', 'polars'])
def test_prepare_distributions_preserves_frequency_order_and_source(container):
    """Should retain repetitions and sample order while filtering missing values independently."""
    entries = [3., None, 1., np.nan, np.inf, -np.inf, 1.]
    data = (entries if container == 'list' else tuple(entries) if container == 'tuple' else
            np.array(entries, dtype=object) if container == 'numpy' else
            pd.Series(entries, dtype='Float64') if container == 'pandas' else pl.Series(entries))
    result = _prepare_distributions({'second': data, 'first': [7.]})

    assert list(result) == ['second', 'first']
    np.testing.assert_array_equal(result['second'], [3., 1., 1.])
    np.testing.assert_array_equal(result['first'], [7.])
    result['second'][0] = -99.
    first_value = data.iloc[0] if isinstance(data, pd.Series) else data[0]
    assert first_value == 3.


def test_prepare_distributions_reuses_numeric_string_plotting_contract():
    """Should accept numeric strings consistently with existing plotting helpers."""
    values = _prepare_distributions({'A': pd.Series(['1', None, '2.5']),
                                     'B': pl.Series(['1', None, '2.5'])})
    np.testing.assert_array_equal(values['A'], [1., 2.5])
    np.testing.assert_array_equal(values['A'], values['B'])


@pytest.mark.parametrize('data, error', [
    (None, TypeError), ([], TypeError), ({}, ValueError),
    ({'': [1]}, ValueError), ({'  ': [1]}, ValueError), ({1: [1]}, ValueError),
    ({'A': '123'}, TypeError), ({'A': b'12'}, TypeError), ({'A': {1, 2}}, TypeError),
    ({'A': 3}, TypeError), ({'A': {'x': 1}}, TypeError),
    ({'A': pd.DataFrame({'x': [1]})}, TypeError),
    ({'A': pl.DataFrame({'x': [1]})}, TypeError),
    ({'A': np.array(1)}, ValueError), ({'A': np.ones((2, 2))}, ValueError),
    ({'A': []}, ValueError), ({'A': [None, np.nan, np.inf, -np.inf]}, ValueError),
    ({'A': [1], 'B': []}, ValueError), ({'A': [True, False]}, ValueError),
    ({'A': [1, True]}, ValueError), ({'A': [1j]}, ValueError),
    ({'A': ['invalid']}, ValueError), ({'A': [10 ** 400]}, ValueError),
    ({'A': [[1], [2]]}, ValueError),
    ({'A': pd.Series(pd.date_range('2024-01-01', periods=2))}, ValueError),
    ({'A': pl.Series([True, False])}, ValueError),
    ({'A': pd.Series(pd.to_timedelta([1, 2], unit='D'))}, ValueError),
])
def test_prepare_distributions_rejects_ambiguous_or_unusable_samples(data, error):
    """Should reject malformed containers, identities, dimensions, and non-real or empty effective samples."""
    with pytest.raises(error):
        _prepare_distributions(data)


@pytest.mark.parametrize('option, value, error', [
    ('colors', ['red'], TypeError), ('colors', {'unknown': 'red'}, ValueError),
    ('colors', {'A': 'not-a-color'}, ValueError), ('colors', {'A': np.nan}, ValueError),
    ('value_label', None, TypeError), ('title', 1, TypeError), ('note', [], TypeError),
    ('reference_label', None, TypeError), ('reference_label', ' ', ValueError),
    ('reference_value', True, TypeError), ('reference_value', '1', TypeError),
    ('reference_value', np.inf, ValueError), ('reference_value', np.nan, ValueError),
    ('reference_value', 10 ** 400, ValueError), ('value_on_y', 1, TypeError),
])
def test_comparison_plot_validates_shared_options_before_figure_creation(comparison_options, option, value, error, plot_axis):
    """Should reject malformed shared options without creating or changing figures."""
    import matplotlib.pyplot as plt

    comparison_options[option] = value
    before = set(plt.get_fignums())
    with pytest.raises(error), _comparison_plot(['A'], None, **comparison_options):
        pytest.fail('invalid options must not reach drawing')
    assert set(plt.get_fignums()) == before
    assert not plot_axis.lines and not plot_axis.patches


def test_comparison_plot_rejects_invalid_axis(comparison_options, plot_axis):
    """Should reject a non-Axes object before attempting to draw."""
    with pytest.raises(TypeError, match='Matplotlib Axes'), _comparison_plot(['A'], object(), **comparison_options):
        pytest.fail('invalid axes must not reach drawing')


@pytest.mark.parametrize('value_on_y', [False, True])
def test_comparison_plot_sets_note_palette_reference_and_value_axis(comparison_options, value_on_y, plot_axis):
    """Should preserve explicit sample colors and orient the reference and numeric label consistently."""
    comparison_options.update(colors={'A': 'red'}, title='Comparison', note='Different origins',
                              reference_value=.25, value_on_y=value_on_y)
    with _comparison_plot(['A', 'B'], plot_axis, **comparison_options) as (axis, palette):
        assert axis is plot_axis and palette == ['red', 'C1']
    reference = plot_axis.lines[-1]
    np.testing.assert_array_equal(reference.get_ydata() if value_on_y else reference.get_xdata(), [.25, .25])
    assert (plot_axis.get_ylabel() if value_on_y else plot_axis.get_xlabel()) == 'Estimate'
    assert plot_axis.get_title() == 'Comparison'
    assert plot_axis.texts[0].get_text() == 'Different origins'
    assert plot_axis.texts[0].get_transform() is plot_axis.transAxes
    assert reference.get_label() == 'Observed'


@pytest.mark.parametrize('supplied', [False, True])
def test_comparison_plot_context_error_preserves_figure_ownership(comparison_options, supplied, plot_axis):
    """Should release failed owned figures and propagate the original exception without closing caller axes."""
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    error = RuntimeError('drawing failed')
    with pytest.raises(RuntimeError) as caught, _comparison_plot(
            ['A'], plot_axis if supplied else None, **comparison_options):
        raise error
    assert caught.value is error
    assert set(plt.get_fignums()) == before


def test_comparison_plot_converts_numeric_overflow_and_restores_numpy_policy(comparison_options, plot_axis):
    """Should report unrepresentable plotting arithmetic without leaking local NumPy error settings."""
    before = np.geterr().copy()
    with pytest.raises(ValueError, match='representable plotting range'), _comparison_plot(
            ['A'], plot_axis, **comparison_options):
        np.multiply(np.float64(1e308), np.float64(1e308))
    assert np.geterr() == before


@pytest.mark.parametrize('plotter', PLOTTERS)
def test_comparison_labels_starting_with_underscores_are_visible(plotter, plot_axis):
    """Should preserve every supplied label instead of silently omitting underscore-prefixed legends."""
    axis = plotter({'_sample': [1., 2.]}, ax=plot_axis, reference_value=1.5, reference_label='_reference')
    expected = ['_reference'] if plotter is plot_boxplot_comparison else ['_sample', '_reference']
    assert [text.get_text() for text in axis.get_legend().texts] == expected


@pytest.mark.parametrize('plotter', PLOTTERS)
def test_comparison_plotters_explain_missing_optional_matplotlib(plotter, monkeypatch):
    """Should explain the optional viz dependency without importing Matplotlib at module load time."""
    original_import = builtins.__import__
    blocked = Mock(side_effect=ImportError('simulated missing Matplotlib'))

    def blocked_import(name, *args, **kwargs):
        """Should block Matplotlib imports and pass all other imports through.

        Parameters
        ----------
        name : str
            Requested import name.
        *args, **kwargs
            Arguments forwarded to the real importer.

        Returns
        -------
        object
            Imported non-Matplotlib module.

        Raises
        ------
        ImportError
            If Matplotlib is requested.
        """
        if name.startswith('matplotlib'):
            return blocked(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', blocked_import)
    with pytest.raises(ImportError, match=r'alpha-research\[viz\]'):
        plotter({'A': [1., 2.]})
    blocked.assert_called_once()


# ------------------------------------------------------
# common histogram bins and normalization
# ------------------------------------------------------
@pytest.mark.parametrize('bins, expected', [(2, [0., 2., 4.]), (np.uint64(2), [0., 2., 4.]),
                                          ([0., 1., 4.], [0., 1., 4.])])
def test_histogram_edges_use_pooled_range_and_explicit_geometry(bins, expected):
    """Should produce hand-verifiable common edges rather than choose separate ranges per sample."""
    arrays = _prepare_distributions({'A': [0., 1.], 'B': [3., 4.]})
    np.testing.assert_array_equal(_comparison_histogram_edges(arrays, bins), expected)


@pytest.mark.parametrize('bins, error', [
    (0, ValueError), (-1, ValueError), (True, ValueError), (np.bool_(True), TypeError),
    (2., TypeError), (None, TypeError), ({1, 2}, TypeError), ('unknown-estimator', ValueError),
    ([], ValueError), ([1.], ValueError), ([[0., 4.]], ValueError),
    ([0., 0., 4.], ValueError), ([4., 0.], ValueError),
    ([np.nan, 4.], ValueError), ([0., np.inf], ValueError),
    ([1., 4.], ValueError), ([0., 3.], ValueError),
    (['0', '4'], TypeError), ([False, True], TypeError), ([0j, 4j], TypeError),
])
def test_histogram_edges_reject_invalid_bins_and_truncated_ranges(bins, error):
    """Should reject ambiguous edges, invalid counts, and bins silently excluding observations."""
    with pytest.raises(error):
        _comparison_histogram_edges(_prepare_distributions({'A': [0., 4.]}), bins)


def test_histogram_edges_reject_unrepresentable_widths():
    """Should fail explicitly when finite endpoints imply an overflowing bin width."""
    with pytest.raises(ValueError, match='representable finite bin widths'):
        _comparison_histogram_edges({'A': np.array([-1e308, 1e308])}, [-1e308, 1e308])


def test_histogram_edges_reject_integer_edges_collapsing_in_float64():
    """Should reject distinct integer edges that collapse to one plotted Float64 coordinate."""
    edges = np.array([2 ** 63, 2 ** 63 + 1], dtype=np.uint64)
    with pytest.raises(ValueError, match='strictly increasing'):
        _comparison_histogram_edges({'A': np.array([float(2 ** 63)])}, edges)


@pytest.mark.parametrize('stat, expected', [
    ('count', [[3., 1.], [1., 1.]]),
    ('probability', [[.75, .25], [.5, .5]]),
    ('density', [[.75, .125], [.5, .25]]),
])
def test_histogram_exact_heights_with_unequal_widths_and_sample_sizes(stat, expected, plot_axis):
    """Should normalize each sample independently and include the final right edge in the last bin."""
    # Bins [0,1) and [1,3] have widths 1 and 2. Counts are read by hand.
    result = plot_histogram_comparison({'A': [0., 0., .5, 3.], 'B': [0., 3.]},
                                       ax=plot_axis, bins=[0., 1., 3.], stat=stat)
    assert result is plot_axis
    for polygon, heights in zip(result.patches, expected):
        np.testing.assert_array_equal(polygon.get_xy()[:, 0], [0., 0., 1., 1., 3., 3.])
        np.testing.assert_allclose(polygon.get_xy()[:, 1], [0., heights[0], heights[0], heights[1], heights[1], 0.])
    assert [text.get_text() for text in result.get_legend().texts] == ['A', 'B']


@pytest.mark.parametrize('stat', ['bad', None, True, 1])
def test_histogram_rejects_unknown_normalization(stat, plot_axis):
    """Should reject unknown normalization instead of defaulting to counts or density."""
    with pytest.raises(ValueError, match='stat must be'):
        plot_histogram_comparison({'A': [1.]}, ax=plot_axis, stat=stat)
    assert not plot_axis.patches


# ------------------------------------------------------
# empirical curves and boxes
# ------------------------------------------------------
def test_ecdf_exact_coordinates_count_ties_and_sort_without_mutation(plot_axis):
    """Should draw right-continuous CDF jumps with hand-counted tie frequencies."""
    values = np.array([5., 1., 3., 1.])
    result = plot_ecdf_comparison({'A': values, 'B': [2., 4.]}, ax=plot_axis)

    np.testing.assert_array_equal(result.lines[0].get_xdata(), [1., 1., 3., 5.])
    np.testing.assert_array_equal(result.lines[0].get_ydata(), [0., .5, .75, 1.])
    np.testing.assert_array_equal(result.lines[1].get_xdata(), [2., 2., 4.])
    np.testing.assert_array_equal(result.lines[1].get_ydata(), [0., .5, 1.])
    assert result.lines[0].get_drawstyle() == 'steps-post'
    np.testing.assert_array_equal(values, [5., 1., 3., 1.])
    assert result.get_ylim() == (0., 1.)


def test_exceedance_exact_coordinates_include_equal_values_and_maximum(plot_axis):
    """Should count equality in exceedance, rather than incorrectly using one minus the CDF."""
    result = plot_exceedance_comparison({'A': [5., 1., 5., 3., 1.]}, ax=plot_axis)

    # At thresholds 1, 3, and 5, respectively 5, 3, and 2 of 5 values qualify.
    np.testing.assert_array_equal(result.lines[0].get_xdata(), [1., 3., 5., 5.])
    np.testing.assert_array_equal(result.lines[0].get_ydata(), [1., .6, .4, 0.])
    assert result.lines[0].get_drawstyle() == 'steps-pre'
    assert result.get_ylim() == (0., 1.)


@pytest.mark.parametrize('plotter, expected_y', [(plot_ecdf_comparison, [0., 1.]),
                                               (plot_exceedance_comparison, [1., 0.])])
@pytest.mark.parametrize('values', [[2.], [2., 2., 2.]])
def test_empirical_curves_handle_singletons_and_constant_samples(plotter, expected_y, values, plot_axis):
    """Should represent point-mass distributions without fabricating spread or smoothing ties."""
    axis = plotter({'A': values}, ax=plot_axis)
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), [2., 2.])
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), expected_y)


def test_boxplot_exact_quartiles_whiskers_outlier_and_sample_order(plot_axis):
    """Should draw independently derived linear quartiles and Tukey whiskers in mapping order."""
    # Six sorted observations: Q1=1.25, median=2.5, Q3=3.75.
    # IQR=2.5; upper fence=7.5, so 4 is the upper whisker and 100 is a flier.
    axis = plot_boxplot_comparison({'second': [0., 1., 2., 3., 4., 100.], 'first': [7.]},
                                   ax=plot_axis, colors={'second': 'red'})
    np.testing.assert_allclose(axis.patches[0].get_path().vertices[:, 1], [1.25, 1.25, 3.75, 3.75, 1.25, 1.25])
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), [1.25, 0.])
    np.testing.assert_array_equal(axis.lines[1].get_ydata(), [3.75, 4.])
    np.testing.assert_array_equal(axis.lines[4].get_ydata(), [2.5, 2.5])
    np.testing.assert_array_equal(axis.lines[5].get_ydata(), [100.])
    np.testing.assert_array_equal(axis.patches[1].get_path().vertices[:, 1], [7.] * 6)
    assert [label.get_text() for label in axis.get_xticklabels()] == ['second', 'first']
    assert axis.patches[0].get_facecolor() == (1., 0., 0., .4)


def test_boxplot_can_hide_fliers_without_changing_quartiles(plot_axis):
    """Should suppress outlier markers while preserving the original sample's quartiles."""
    axis = plot_boxplot_comparison({'A': [0., 1., 2., 3., 4., 100.]}, ax=plot_axis, show_fliers=False)
    assert len(axis.lines) == 5
    np.testing.assert_allclose(axis.patches[0].get_path().vertices[:, 1], [1.25, 1.25, 3.75, 3.75, 1.25, 1.25])


@pytest.mark.parametrize('kwargs, error', [({'whis': 0}, ValueError), ({'whis': -1}, ValueError),
                                         ({'whis': np.nan}, ValueError), ({'whis': True}, TypeError),
                                         ({'show_fliers': 1}, TypeError), ({'show_fliers': np.bool_(True)}, TypeError)])
def test_boxplot_rejects_invalid_whisker_and_flag_options(kwargs, error, plot_axis):
    """Should reject invalid whisker geometry and nonboolean visibility flags before drawing."""
    with pytest.raises(error):
        plot_boxplot_comparison({'A': [1., 2.]}, ax=plot_axis, **kwargs)
    assert not plot_axis.lines and not plot_axis.patches


def test_boxplot_rejects_unrepresentable_quartile_arithmetic(plot_axis):
    """Should reject overflowing percentile arithmetic instead of drawing invalid box geometry."""
    with pytest.raises(ValueError, match='representable plotting range'):
        plot_boxplot_comparison({'A': [-1e308, 1e308]}, ax=plot_axis)


# ------------------------------------------------------
# backend equivalence, figure lifetime, and drawing
# ------------------------------------------------------
@pytest.mark.parametrize('plotter', PLOTTERS)
@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_comparison_plotters_accept_nullable_series_and_keep_sources(plotter, backend, plot_axis):
    """Should plot finite nullable observations identically without changing either backend's source."""
    sample = (pd.Series([3., None, 1., 1.], index=[8, 2, 5, 2], dtype='Float64')
              if backend == 'pandas' else pl.Series('values', [3., None, 1., 1.]))
    before = sample.copy(deep=True) if backend == 'pandas' else sample.clone()
    axis = plotter({'A': sample}, ax=plot_axis)
    axis.figure.canvas.draw()
    if backend == 'pandas':
        pd.testing.assert_series_equal(sample, before)
    else:
        assert sample.equals(before)


@pytest.mark.parametrize('plotter', PLOTTERS)
def test_comparison_plotters_keep_successful_owned_figures_open(plotter, plot_axis):
    """Should return a drawable new axis and leave the successful figure lifetime to the caller."""
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    axis = plotter({'A': [1., 2., 4.], 'B': [2., 3.]}, note='Different sample origins', reference_value=2.5)
    try:
        axis.figure.canvas.draw()
        assert set(plt.get_fignums()) - before == {axis.figure.number}
    finally:
        plt.close(axis.figure)


@pytest.mark.parametrize('plotter', PLOTTERS)
@pytest.mark.parametrize('supplied', [False, True])
def test_comparison_plotters_release_only_owned_figures_on_late_error(plotter, supplied, plot_axis, monkeypatch):
    """Should clean up failed internally created figures even when annotation fails after drawing."""
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    before = set(plt.get_fignums())
    error = RuntimeError('annotation failed')
    monkeypatch.setattr(Axes, 'text', Mock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        plotter({'A': [1., 2., 3.]}, ax=plot_axis if supplied else None, note='Fail here')
    assert caught.value is error
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize('plotter', PLOTTERS)
def test_comparison_plotters_reject_empty_samples_before_creating_figures(plotter, plot_axis):
    """Should reject an empty effective sample before allocating a figure or modifying caller axes."""
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match='at least one finite'):
        plotter({'A': [np.nan]})
    assert set(plt.get_fignums()) == before
    assert not plot_axis.lines and not plot_axis.patches
