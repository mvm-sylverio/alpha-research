import builtins

import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.evaluation.timeseries import rolling_temporal_association
from alpha_research.visualization import plot_rolling_temporal_association


# ------------------------------------------------------
# fixtures
# ------------------------------------------------------
@pytest.fixture
def rolling_temporal_df_pandas():
    """Create a single-asset frame suitable for a rolling association result."""
    return pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=12, freq='D'),
        'symbol': ['AAPL'] * 12,
        'feature': [
            0.2, -0.4, 0.1, -0.3, 0.6, -0.1,
            0.4, -0.5, 0.0, 0.3, -0.2, 0.5,
        ],
        'target': [
            0.4, -0.2, 0.3, -0.1, 0.5, -0.4,
            0.2, -0.3, 0.1, 0.6, -0.5, 0.0,
        ],
    })


@pytest.fixture
def rolling_frame(rolling_temporal_df_pandas):
    """Create a rolling temporal-association result for plotting tests."""
    return rolling_temporal_association(
        rolling_temporal_df_pandas,
        feature='feature',
        target='target',
        window_size=6,
        block_length=3,
        n_bootstraps=20,
        random_state=42,
    ).rolling_frame


# ------------------------------------------------------
# plot_rolling_temporal_association
# ------------------------------------------------------
@pytest.mark.parametrize('band_alpha', [-0.1, 1.1, '0.2', True, np.timedelta64(1, 'D')])
def test_plot_rolling_temporal_association_rejects_invalid_band_alpha(
        rolling_frame,
        band_alpha,
):
    """Should validate plot opacity before importing the optional backend."""
    with pytest.raises(ValueError, match='band_alpha'):
        plot_rolling_temporal_association(
            rolling_frame,
            band_alpha=band_alpha,
        )


@pytest.mark.parametrize(
        ('rolling_frame', 'error_type', 'message'),
        [
            ([{'association': 0.1}], TypeError, 'DataFrame'),
            (pd.DataFrame(), ValueError, 'must not be empty'),
            (pd.DataFrame({'association': [0.1]}), KeyError, 'missing required columns'),
        ],
)
def test_plot_rolling_temporal_association_validates_input_schema(
        rolling_frame,
        error_type,
        message,
):
    """Should reject invalid frames before attempting to import Matplotlib."""
    with pytest.raises(error_type, match=message):
        plot_rolling_temporal_association(rolling_frame)


def test_plot_rolling_temporal_association_rejects_all_invalid_windows():
    """Should reject a frame that cannot produce an association line."""
    rolling_frame = pd.DataFrame({
        'window_end': [pd.Timestamp('2024-01-01')],
        'association': [np.nan],
        'bootstrap_ci_lower': [np.nan],
        'bootstrap_ci_upper': [np.nan],
    })

    with pytest.raises(ValueError, match='finite association'):
        plot_rolling_temporal_association(rolling_frame)


def test_plot_rolling_temporal_association_explains_missing_optional_backend(
        monkeypatch,
        rolling_frame,
):
    """Should fail clearly when importing the optional Matplotlib backend fails."""
    original_import = builtins.__import__

    def raise_matplotlib_import_error(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'matplotlib.pyplot':
            raise ImportError('simulated missing matplotlib')

        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', raise_matplotlib_import_error)

    with pytest.raises(ImportError, match=r'alpha-research\[viz\]'):
        plot_rolling_temporal_association(rolling_frame)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_plot_rolling_temporal_association_composes_on_supplied_axis(
        rolling_frame,
        backend,
):
    """Should draw the association panel on a supplied axis for both backends."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frame = rolling_frame if backend == 'pandas' else pl.from_pandas(rolling_frame)
    figure, axes = plt.subplots(nrows=2, sharex=True)
    returned_axis = plot_rolling_temporal_association(
        frame,
        ax=axes[0],
    )

    assert returned_axis is axes[0]
    assert len(axes[0].lines) == 2
    assert len(axes[0].collections) == 1
    assert len(axes[1].lines) == 0
    plt.close(figure)


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_rolling_plot_exact_coordinates_and_percentile_band(constructor, plot_axis):
    """Should draw the supplied percentile bounds even when they exclude the estimate."""
    frame = constructor({'window_end': [1, 2, 3], 'association': [0.1, 0.3, 0.2],
                         'bootstrap_ci_lower': [0.2, 0.1, -0.1], 'bootstrap_ci_upper': [0.4, 0.5, 0.3]})
    returned = plot_rolling_temporal_association(frame, ax=plot_axis, label='Signal', title='Rolling')
    np.testing.assert_array_equal(returned.lines[0].get_xdata(), [1, 2, 3])
    np.testing.assert_allclose(returned.lines[0].get_ydata(), [0.1, 0.3, 0.2])
    vertices = returned.collections[0].get_paths()[0].vertices
    for x, low, high in [(1, 0.2, 0.4), (2, 0.1, 0.5), (3, -0.1, 0.3)]:
        ys = vertices[vertices[:, 0] == x, 1]
        assert (ys.min(), ys.max()) == pytest.approx((low, high))
    np.testing.assert_allclose(returned.lines[1].get_ydata(), [0., 0.])
    assert returned.lines[0].get_label() == 'Signal'
    assert returned.get_xlabel() == 'window_end'
    assert returned.get_ylabel() == 'Temporal association'
    assert returned.get_title() == 'Rolling'


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize('times', [[2, 1], [1, 1], [1., np.nan], [1., np.inf]])
def test_rolling_plot_rejects_invalid_temporal_keys(constructor, times, plot_axis):
    """Should reject unordered, duplicate, missing, or infinite window endpoints before drawing."""
    frame = constructor({'window_end': times, 'association': [0.1, 0.2],
                         'bootstrap_ci_lower': [0., 0.], 'bootstrap_ci_upper': [0.3, 0.3]})
    with pytest.raises(ValueError):
        plot_rolling_temporal_association(frame, ax=plot_axis)
    assert not plot_axis.lines and not plot_axis.collections


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_rolling_plot_rejects_reversed_intervals(constructor, plot_axis):
    """Should report invalid finite intervals before adding any artists."""
    frame = constructor({'window_end': [1], 'association': [0.2],
                         'bootstrap_ci_lower': [0.4], 'bootstrap_ci_upper': [0.1]})
    with pytest.raises(ValueError, match='must not exceed'):
        plot_rolling_temporal_association(frame, ax=plot_axis)
    assert not plot_axis.lines and not plot_axis.collections


@pytest.mark.parametrize('constructor', [pd.DataFrame, pl.DataFrame])
def test_rolling_plot_masks_undefined_and_incomplete_intervals(constructor, plot_axis):
    """Should retain line gaps and avoid drawing a band through an undefined estimate."""
    frame = constructor({'window_end': [1, 2, 3, 4, 5], 'association': [0.1, np.nan, 0.2, 0.3, np.inf],
                         'bootstrap_ci_lower': [0., 0., None, 0., 0.], 'bootstrap_ci_upper': [0.4] * 5})
    original = frame.copy(deep=True) if isinstance(frame, pd.DataFrame) else frame.clone()
    returned = plot_rolling_temporal_association(frame, ax=plot_axis)
    np.testing.assert_allclose(returned.lines[0].get_ydata(), [0.1, np.nan, 0.2, 0.3, np.nan], equal_nan=True)
    vertices = np.concatenate([path.vertices for path in returned.collections[0].get_paths()])
    assert set(vertices[:, 0]) == {1., 4.}
    if isinstance(frame, pd.DataFrame):
        pd.testing.assert_frame_equal(frame, original)
    else:
        assert frame.equals(original)


def test_rolling_plot_validates_interval_types_before_allocating_a_figure():
    """Should reject malformed confidence limits without allocating a figure."""
    pytest.importorskip('matplotlib')
    import matplotlib.pyplot as plt

    frame = pd.DataFrame({'window_end': [1], 'association': [0.2],
                          'bootstrap_ci_lower': ['bad'], 'bootstrap_ci_upper': [0.3]})
    before = plt.get_fignums()
    with pytest.raises(ValueError, match='numeric'):
        plot_rolling_temporal_association(frame)
    assert plt.get_fignums() == before
