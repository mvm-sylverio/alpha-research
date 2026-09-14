import builtins

import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.evaluation import feature_target_relationship
from alpha_research.visualization import (
    plot_feature_target_bins,
    plot_feature_target_scatter,
)


@pytest.fixture
def relationship_result_pandas():
    """Create one generic feature-target relationship result for plotting."""
    frame = pd.DataFrame({
        'time': np.arange(8),
        'symbol': ['A'] * 8,
        'feature': np.arange(1, 9, dtype=float),
        'target': np.arange(1, 9, dtype=float) ** 2,
    })
    return feature_target_relationship(frame, 'feature', 'target', n_bins=4)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_plot_feature_target_scatter_draws_raw_pairs(
        relationship_result_pandas,
        backend,
):
    """Should draw raw finite pairs from either supported result backend."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    result = relationship_result_pandas
    if backend == 'polars':
        result = feature_target_relationship(
            pl.from_pandas(result.pairs.drop(columns='bin')),
            'feature',
            'target',
            n_bins=4,
        )
    figure, axis = plt.subplots()
    returned = plot_feature_target_scatter(result, ax=axis, alpha=0.5)

    assert returned is axis
    assert len(axis.collections[0].get_offsets()) == 8
    assert axis.get_xlabel() == 'feature'
    assert axis.get_ylabel() == 'target'
    plt.close(figure)


def test_plot_feature_target_scatter_samples_only_the_displayed_points(
        relationship_result_pandas,
):
    """Should cap scatter points without changing the stored result."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    plot_feature_target_scatter(
        relationship_result_pandas,
        ax=axis,
        max_points=3,
        random_state=7,
    )

    assert len(axis.collections[0].get_offsets()) == 3
    assert len(relationship_result_pandas.pairs) == 8
    plt.close(figure)


@pytest.mark.parametrize('target_statistic', ['mean', 'median'])
@pytest.mark.parametrize('feature_statistic', ['mean', 'median'])
def test_plot_feature_target_bins_draws_requested_statistics(
        relationship_result_pandas,
        target_statistic,
        feature_statistic,
):
    """Should draw every supported representative feature and target statistic."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    returned = plot_feature_target_bins(
        relationship_result_pandas,
        target_statistic=target_statistic,
        feature_statistic=feature_statistic,
        ax=axis,
    )

    summary = relationship_result_pandas.bin_summary
    assert returned is axis
    np.testing.assert_allclose(
        axis.lines[0].get_xdata(),
        summary[f'feature_{feature_statistic}'],
    )
    np.testing.assert_allclose(
        axis.lines[0].get_ydata(),
        summary[f'target_{target_statistic}'],
    )
    plt.close(figure)


@pytest.mark.parametrize(
    'kwargs, error_type, message',
    [
        ({'max_points': 0}, ValueError, 'positive'),
        ({'max_points': True}, ValueError, 'positive'),
        ({'random_state': 1.5}, TypeError, 'integer'),
        ({'alpha': True}, TypeError, 'numeric'),
        ({'alpha': np.inf}, ValueError, 'finite'),
        ({'alpha': 2}, ValueError, 'between'),
    ],
)
def test_plot_feature_target_scatter_validates_parameters(
        relationship_result_pandas,
        kwargs,
        error_type,
        message,
):
    """Should reject invalid scatter sampling and opacity parameters."""
    with pytest.raises(error_type, match=message):
        plot_feature_target_scatter(relationship_result_pandas, **kwargs)


def test_relationship_plots_require_structured_results():
    """Should not accept arbitrary frames without relationship metadata."""
    frame = pd.DataFrame({'feature': [1.0], 'target': [2.0]})
    with pytest.raises(TypeError, match='FeatureTargetRelationshipResult'):
        plot_feature_target_scatter(frame)
    with pytest.raises(TypeError, match='FeatureTargetRelationshipResult'):
        plot_feature_target_bins(frame)


def test_plot_feature_target_bins_validates_statistics(
        relationship_result_pandas,
):
    """Should reject unsupported feature and target summaries."""
    with pytest.raises(ValueError, match='target_statistic'):
        plot_feature_target_bins(
            relationship_result_pandas,
            target_statistic='maximum',
        )
    with pytest.raises(ValueError, match='feature_statistic'):
        plot_feature_target_bins(
            relationship_result_pandas,
            feature_statistic='minimum',
        )


def test_relationship_plots_validate_before_importing_optional_backend(
        monkeypatch,
        relationship_result_pandas,
):
    """Should report caller errors even when the optional backend is unavailable."""
    original_import = builtins.__import__

    def raise_matplotlib_import_error(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'matplotlib.pyplot':
            raise ImportError('simulated missing matplotlib')
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', raise_matplotlib_import_error)
    with pytest.raises(ValueError, match='alpha'):
        plot_feature_target_scatter(relationship_result_pandas, alpha=-1)
    with pytest.raises(ValueError, match='target_statistic'):
        plot_feature_target_bins(
            relationship_result_pandas,
            target_statistic='maximum',
        )


def test_scatter_sampling_is_reproducible_and_does_not_mutate_result(
        relationship_result_pandas,
):
    """Should use its seed deterministically while keeping stored pairs unchanged."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    original_pairs = relationship_result_pandas.pairs.copy(deep=True)
    figure, axes = plt.subplots(1, 2)
    plot_feature_target_scatter(
        relationship_result_pandas,
        ax=axes[0],
        max_points=4,
        random_state=23,
    )
    plot_feature_target_scatter(
        relationship_result_pandas,
        ax=axes[1],
        max_points=4,
        random_state=23,
    )

    np.testing.assert_allclose(
        axes[0].collections[0].get_offsets(),
        axes[1].collections[0].get_offsets(),
    )
    pd.testing.assert_frame_equal(relationship_result_pandas.pairs, original_pairs)
    plt.close(figure)


def test_plot_feature_target_bins_supports_polars_summary(
        relationship_result_pandas,
):
    """Should render binned statistics stored in a Polars-backed result."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    polars_result = feature_target_relationship(
        pl.from_pandas(relationship_result_pandas.pairs.drop(columns='bin')),
        'feature',
        'target',
        n_bins=4,
    )
    figure, axis = plt.subplots()
    plot_feature_target_bins(polars_result, ax=axis)

    assert len(axis.lines[0].get_xdata()) == 4
    plt.close(figure)
