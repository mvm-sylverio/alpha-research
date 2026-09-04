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
@pytest.mark.parametrize('band_alpha', [-0.1, 1.1, '0.2', True])
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
