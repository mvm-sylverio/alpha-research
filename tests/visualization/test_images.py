from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_research.visualization import (
    plot_boxplot_comparison,
    plot_cross_sectional_value_summary,
    plot_decay_curves,
    plot_ecdf_comparison,
    plot_exceedance_comparison,
    plot_histogram_comparison,
    plot_ic_summary,
    plot_partial_ic_summary,
    plot_partial_temporal_association_summary,
    plot_rolling_temporal_association,
    plot_temporal_association_summary,
)


@pytest.mark.parametrize('kind', ['rolling_association', 'ranked_summary'])
def test_representative_plot_image_regression(kind):
    """Should match small Agg references drawn from hand-specified coordinates and styles."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    # References were rendered with explicit Matplotlib primitives and known
    # coordinates, without calling the plotting functions under test.
    settings = {**matplotlib.rcParamsDefault, 'backend': 'Agg', 'font.family': 'DejaVu Sans',
                'font.size': 10, 'figure.dpi': 100, 'savefig.dpi': 100}
    with matplotlib.rc_context(settings):
        figure, axis = plt.subplots(figsize=(5, 3), dpi=100)
        try:
            figure.subplots_adjust(left=0.2, bottom=0.2, right=0.96, top=0.95)
            if kind == 'rolling_association':
                table = pd.DataFrame({'window_end': [1, 2, 3], 'association': [0.2, -0.1, 0.3],
                                      'bootstrap_ci_lower': [0., -0.3, 0.1], 'bootstrap_ci_upper': [0.4, 0.2, 0.5]})
                plot_rolling_temporal_association(table, ax=axis)
                axis.set(xlim=(0.5, 3.5), ylim=(-0.5, 0.7), xticks=[1, 2, 3], yticks=[-0.4, 0., 0.4])
            else:
                table = pd.DataFrame({'feature': ['positive', 'inverse'], 'mean': [0.2, -0.4],
                                      'fdr_rejected': [False, True]})
                plot_ic_summary(table, ax=axis)
                axis.set(xlim=(-0.6, 0.4), ylim=(1.5, -0.5), xticks=[-0.4, 0., 0.4])
            figure.canvas.draw()
            actual = np.asarray(figure.canvas.buffer_rgba(), dtype=float) / 255.
            expected = mpimg.imread(Path(__file__).parent / 'baseline_images' / f'{kind}.png')
            assert actual.shape == expected.shape
            rms = np.sqrt(np.mean((actual - expected) ** 2))
            # A small pixel tolerance accommodates renderer/platform details;
            # numerical coordinates are checked independently in artist tests.
            assert rms <= 0.01, f'Image RMS difference {rms:.6f} exceeds 0.01.'
        finally:
            plt.close(figure)


@pytest.mark.parametrize('kind', ['histogram', 'ecdf', 'exceedance', 'boxplot'])
def test_comparison_plot_image_regression(kind):
    """Should preserve shared bins, tied empirical steps, boxes, legends, references, and annotations."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    # References use explicit polygons, line segments, and rectangles, without
    # these plotters, histogram/percentile calculations, or empirical-curve tools.
    # For A=[0,1,1,2,3,10], B=[1,2,3,3,5] and edges [0,1,3,10], bin
    # proportions are A=[1/6,3/6,2/6], B=[0,2/5,3/5]. At unique observed
    # thresholds, CDFs are A=[1/6,3/6,4/6,5/6,1], B=[1/5,2/5,4/5,1];
    # inclusive exceedances are A=[1,5/6,3/6,2/6,1/6], B=[1,4/5,3/5,1/5].
    # Linear quartiles/medians are A=(1,1.5,2.75), B=(2,3,3), whiskers
    # A=(0,3), B=(1,3), and fliers A=[10], B=[5], calculated by hand.
    settings = {**matplotlib.rcParamsDefault, 'backend': 'Agg', 'font.family': 'DejaVu Sans',
                'font.size': 10, 'figure.dpi': 100, 'savefig.dpi': 100,
                'legend.loc': 'lower right' if kind == 'ecdf' else 'upper right'}
    with matplotlib.rc_context(settings):
        figure, axis = plt.subplots(figsize=(8, 4), dpi=100)
        try:
            figure.subplots_adjust(left=.17, bottom=.2, right=.96, top=.88)
            samples = {'Sample A': [0., 1., 1., 2., 3., 10.], 'Sample B': [1., 2., 3., 3., 5.]}
            plotter = {'histogram': plot_histogram_comparison, 'ecdf': plot_ecdf_comparison,
                       'exceedance': plot_exceedance_comparison, 'boxplot': plot_boxplot_comparison}[kind]
            options = {'bins': [0., 1., 3., 10.]} if kind == 'histogram' else {}
            plotter(samples, ax=axis, colors={'Sample A': 'C0', 'Sample B': 'C1'},
                    value_label='Estimate', title=f'{kind.capitalize()} comparison',
                    note='Different sample origins; boxes show quartiles.' if kind == 'boxplot' else None,
                    reference_value=2.5, reference_label='Observed', **options)
            if kind == 'boxplot':
                axis.set(xlim=(.5, 2.5), ylim=(-1., 11.), yticks=[0, 2, 4, 6, 8, 10])
            else:
                axis.set(xlim=(-.5, 10.5), ylim=(0., 1.), xticks=[0, 1, 3, 5, 10], yticks=[0., .25, .5, .75, 1.])
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            # Guard against clipped labels/notes independently of pixel tolerance.
            for artist in [axis.title, axis.xaxis.label, axis.yaxis.label, axis.get_legend(), *axis.texts]:
                bounds = artist.get_window_extent(renderer)
                assert bounds.x0 >= figure.bbox.x0 and bounds.x1 <= figure.bbox.x1
                assert bounds.y0 >= figure.bbox.y0 and bounds.y1 <= figure.bbox.y1
            actual = np.asarray(figure.canvas.buffer_rgba(), dtype=float) / 255.
            expected = mpimg.imread(Path(__file__).parent / 'baseline_images' / f'comparison_{kind}.png')
            assert actual.shape == expected.shape
            rms = np.sqrt(np.mean((actual - expected) ** 2))
            assert rms <= .01, f'Image RMS difference {rms:.6f} exceeds 0.01.'
        finally:
            plt.close(figure)


@pytest.mark.parametrize('kind', ['decay_intervals', 'cross_sectional_band', 'temporal_summary',
                                 'partial_ic_summary', 'partial_temporal_summary'])
def test_additional_plot_image_regression(kind):
    """Should preserve intervals, significance markers, quantile bands, and conditioning labels."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    settings = {**matplotlib.rcParamsDefault, 'backend': 'Agg', 'font.family': 'DejaVu Sans',
                'font.size': 10, 'figure.dpi': 100, 'savefig.dpi': 100}
    with matplotlib.rc_context(settings):
        figure, axis = plt.subplots(figsize=(8, 4), dpi=100)
        try:
            temporal = kind in ('temporal_summary', 'partial_temporal_summary')
            partial = kind.startswith('partial_')
            figure.subplots_adjust(left=0.18, bottom=0.18, right=0.53 if temporal else 0.95,
                                   top=0.78 if partial else 0.9)
            if kind == 'decay_intervals':
                table = pd.DataFrame({'horizon': [4, 1, 2], 'association': [-0.2, 0.4, 0.1],
                                      'lower': [-0.4, 0.2, -0.1], 'upper': [0., 0.6, 0.3],
                                      'fdr_rejected': [False, True, True]})
                plot_decay_curves({'Signal': table}, 'association', axis,
                                  ci_lower_col='lower', ci_upper_col='upper')
                axis.set(xlabel='Horizon', ylabel='Association', title='Decay',
                         xlim=(0.5, 4.5), ylim=(-0.5, 0.7), xticks=[1, 2, 4], yticks=[-0.4, 0., 0.4])
                axis.legend(frameon=False)
            elif kind == 'cross_sectional_band':
                table = pd.DataFrame({'time': [3, 1, 2, 3, 1, 2], 'symbol': ['A'] * 3 + ['B'] * 3,
                                      'value': [8., 0., 4., 12., 4., 8.]})
                plot_cross_sectional_value_summary(table, ax=axis, label='Cross-sectional median',
                                                   title='Cross-sectional value')
                axis.set(xlim=(0.5, 3.5), ylim=(0., 12.), xticks=[1, 2, 3], yticks=[0, 4, 8, 12])
                axis.legend(frameon=False)
            else:
                table = pd.DataFrame({'feature': ['positive', 'inverse'], 'mean': [0.2, -0.4],
                                      'association': [0.2, -0.4], 'wald_ci_lower': [0.1, -0.55],
                                      'wald_ci_upper': [0.35, -0.25], 'reject_h0': [False, True],
                                      'covariates': ['market, volatility'] * 2})
                if kind == 'temporal_summary':
                    plot_temporal_association_summary(table, ax=axis, title='Temporal ranking')
                elif kind == 'partial_ic_summary':
                    table['fdr_rejected'] = [False, True]
                    plot_partial_ic_summary(table, ax=axis, title='Partial IC')
                else:
                    plot_partial_temporal_association_summary(table, ax=axis, title='Partial temporal')
                axis.set(xlim=(-0.65, 0.45), ylim=(1.5, -0.5), xticks=[-0.4, 0., 0.4])
            figure.canvas.draw()
            legend = axis.get_legend()
            if legend is not None:
                bounds = legend.get_window_extent()
                assert bounds.x0 >= figure.bbox.x0 and bounds.x1 <= figure.bbox.x1
                assert bounds.y0 >= figure.bbox.y0 and bounds.y1 <= figure.bbox.y1
            actual = np.asarray(figure.canvas.buffer_rgba(), dtype=float) / 255.
            expected = mpimg.imread(Path(__file__).parent / 'baseline_images' / f'{kind}.png')
            assert actual.shape == expected.shape
            rms = np.sqrt(np.mean((actual - expected) ** 2))
            assert rms <= 0.01, f'Image RMS difference {rms:.6f} exceeds 0.01.'
        finally:
            plt.close(figure)
