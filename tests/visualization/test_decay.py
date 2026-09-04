import builtins

import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.visualization import plot_decay_curves


# ------------------------------------------------------
# fixtures
# ------------------------------------------------------
@pytest.fixture
def ic_decay_tables():
    """Create two IC-decay tables using both supported DataFrame backends."""
    return {
        'momentum': pd.DataFrame({
            'horizon': [5, 1, 3],
            'mean': [0.02, 0.10, 0.05],
            'fdr_rejected': [False, True, True],
        }),
        'reversal': pl.DataFrame({
            'horizon': [1, 3, 5],
            'mean': [-0.08, -0.03, 0.01],
            'fdr_rejected': [True, False, False],
        }),
    }


@pytest.fixture
def temporal_decay_tables():
    """Create a temporal-association decay table with Wald limits."""
    return {
        'momentum': pd.DataFrame({
            'horizon': [1, 3, 5],
            'association': [0.12, 0.06, 0.01],
            'wald_ci_lower': [0.04, -0.01, -0.05],
            'wald_ci_upper': [0.20, 0.13, 0.07],
            'fdr_rejected': [True, False, False],
        }),
    }


# ------------------------------------------------------
# plot_decay_curves
# ------------------------------------------------------
def test_plot_decay_curves_requires_non_empty_mapping_and_complete_ci_columns():
    """Should validate the generic container and optional interval contract."""
    with pytest.raises(TypeError, match='mapping'):
        plot_decay_curves([], value_col='mean', ax=object())

    with pytest.raises(ValueError, match='must not be empty'):
        plot_decay_curves({}, value_col='mean', ax=object())

    with pytest.raises(ValueError, match='provided together'):
        plot_decay_curves(
            {'feature': pd.DataFrame({'horizon': [1], 'mean': [0.1]})},
            value_col='mean',
            ax=object(),
            ci_lower_col='ci_lower',
        )


@pytest.mark.parametrize(
        ('decay_tables', 'error_type', 'message'),
        [
            (
                {'feature': pd.DataFrame({'horizon': [1], 'mean': [0.1]})},
                KeyError,
                'fdr_rejected',
            ),
            (
                {'feature': pd.DataFrame({
                    'horizon': [1, 1],
                    'mean': [0.1, 0.2],
                    'fdr_rejected': [True, False],
                })},
                ValueError,
                'one row per horizon',
            ),
            (
                {'feature': pd.DataFrame({
                    'horizon': [1],
                    'mean': [np.nan],
                    'fdr_rejected': [False],
                })},
                ValueError,
                'finite mean',
            ),
        ],
)
def test_plot_decay_curves_validates_each_curve_before_importing_matplotlib(
        decay_tables,
        error_type,
        message,
):
    """Should reject invalid curve tables before touching the plotting backend."""
    with pytest.raises(error_type, match=message):
        plot_decay_curves(decay_tables, value_col='mean', ax=object())


def test_plot_decay_curves_explains_missing_optional_backend(ic_decay_tables, monkeypatch):
    """Should explain how to install Matplotlib after input validation succeeds."""
    original_import = builtins.__import__

    def raise_matplotlib_import_error(
            name,
            globals=None,
            locals=None,
            fromlist=(),
            level=0,
    ):
        if name == 'matplotlib.pyplot':
            raise ImportError('simulated missing matplotlib')

        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', raise_matplotlib_import_error)

    with pytest.raises(ImportError, match=r'alpha-research\[viz\]'):
        plot_decay_curves(ic_decay_tables, value_col='mean', ax=object())


def test_plot_decay_curves_draws_multiple_ic_curves_on_supplied_axis(ic_decay_tables):
    """Should draw generic IC curves and significant-point outlines on one axis."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    returned_axis = plot_decay_curves(
        ic_decay_tables,
        value_col='mean',
        ax=axis,
    )

    assert returned_axis is axis
    assert len(axis.lines) == 3
    assert len(axis.collections) == 2
    assert axis.get_xlabel() == ''
    assert axis.get_ylabel() == ''
    plt.close(figure)


def test_plot_decay_curves_draws_temporal_wald_error_bars(temporal_decay_tables):
    """Should draw discrete error bars when temporal Wald limits are supplied."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    plot_decay_curves(
        temporal_decay_tables,
        value_col='association',
        ax=axis,
        ci_lower_col='wald_ci_lower',
        ci_upper_col='wald_ci_upper',
    )

    assert len(axis.lines) == 4
    assert len(axis.collections) == 2
    assert len(axis.containers) == 1
    plt.close(figure)


def test_plot_decay_curves_allows_tables_without_significance_column():
    """Should allow an explicitly disabled FDR marker for generic curve tables."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    returned_axis = plot_decay_curves(
        {'feature': pd.DataFrame({
            'horizon': [1, 3],
            'value': [0.10, 0.02],
        })},
        value_col='value',
        significance_col=None,
        ax=axis,
    )

    assert returned_axis is axis
    assert len(axis.lines) == 2
    assert len(axis.collections) == 0
    plt.close(figure)
