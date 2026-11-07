import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.visualization._utils import (
    _close_on_plot_error,
    _plot_decisions,
    _plot_numeric_values,
    _valid_plot_intervals,
)


@pytest.mark.parametrize('constructor', [pd.Series, pl.Series])
def test_plot_numeric_values_preserves_gaps_and_input(constructor):
    """Should normalize non-finite data without changing the original Series."""
    series = constructor([1., None, np.nan, np.inf, -np.inf, 3.])
    before = series.copy() if isinstance(series, pd.Series) else series.clone()
    np.testing.assert_allclose(_plot_numeric_values(series, 'value'), [1., np.nan, np.nan, np.nan, np.nan, 3.], equal_nan=True)
    if isinstance(series, pd.Series):
        pd.testing.assert_series_equal(series, before)
    else:
        assert series.equals(before)


@pytest.mark.parametrize('series', [pd.Series(['1', '2']), pl.Series(['1', '2']), pd.Series([1, None], dtype='Int64')])
def test_plot_numeric_values_accepts_numeric_strings_and_nullable_numbers(series):
    """Should retain supported numeric conversions and nullable storage."""
    expected = [1., np.nan] if series.dtype == pd.Int64Dtype() else [1., 2.]
    np.testing.assert_allclose(_plot_numeric_values(series, 'value'), expected, equal_nan=True)


@pytest.mark.parametrize('series', [pd.Series([True]), pl.Series([True]), pd.Series([1 + 2j]),
                                   pd.Series([1 + 2j], dtype=object),
                                   pd.Series([10 ** 400], dtype=object),
                                   pd.Series(pd.date_range('2024-01-01', periods=1)),
                                   pd.Series(pd.to_timedelta([1], unit='D')), pd.Series(['invalid'])])
def test_plot_numeric_values_rejects_non_numeric_data(series):
    """Should report invalid values instead of silently converting them to missing data."""
    with pytest.raises(ValueError, match='value must contain real numeric values'):
        _plot_numeric_values(series, 'value')


@pytest.mark.parametrize('series', [pd.Series([None, None]), pl.Series([None, None])])
def test_plot_numeric_values_retains_wholly_missing_columns(series):
    """Should preserve undefined numeric diagnostics as gaps for the caller to evaluate."""
    np.testing.assert_array_equal(np.isnan(_plot_numeric_values(series, 'value')), [True, True])


@pytest.mark.parametrize('function', [_plot_numeric_values, _plot_decisions])
def test_plot_series_helpers_reject_unsupported_containers(function):
    """Should reject unsupported containers before accessing dtype information."""
    with pytest.raises(TypeError, match='Pandas or Polars Series'):
        function([1, 2], 'value')


@pytest.mark.parametrize('constructor', [pd.Series, pl.Series])
def test_plot_decisions_accepts_only_observed_boolean_values(constructor):
    """Should preserve true and false decisions and give missing decisions neutral styling."""
    np.testing.assert_array_equal(_plot_decisions(constructor([True, None, False]), 'decision'), [True, False, False])


@pytest.mark.parametrize('constructor', [pd.Series, pl.Series])
@pytest.mark.parametrize('values', [['False', 'True'], [0, 1], [0., 1.], ['yes', 'no']])
def test_plot_decisions_rejects_truth_value_coercion(constructor, values):
    """Should reject text and numeric flags that could misrepresent significance."""
    with pytest.raises(ValueError, match='boolean decisions'):
        _plot_decisions(constructor(values), 'decision')


def test_valid_plot_intervals_distinguishes_percentile_and_centered_intervals():
    """Should accept an off-center percentile interval and enforce Wald containment when requested."""
    values, lower, upper = np.array([0.1]), np.array([0.2]), np.array([0.4])
    np.testing.assert_array_equal(_valid_plot_intervals(values, lower, upper), [True])
    with pytest.raises(ValueError, match='contain the plotted values'):
        _valid_plot_intervals(values, lower, upper, contain_estimate=True)


def test_valid_plot_intervals_preserves_missing_rows_and_out_of_range_wald_bounds():
    """Should retain legitimate wide intervals and exclude incomplete or undefined rows."""
    values = np.array([0.9, 0.2, np.nan, 0.1])
    lower = np.array([0.8, np.nan, -0.1, 0.])
    upper = np.array([1.2, 0.3, 0.1, np.inf])
    np.testing.assert_array_equal(_valid_plot_intervals(values, lower, upper, contain_estimate=True), [True, False, False, False])


@pytest.mark.parametrize('values, lower, upper, exception', [
    ([1.], np.array([0.]), np.array([2.]), TypeError),
    (np.array([1.]), np.array(['0']), np.array([2.]), TypeError),
    (np.array([[1.]]), np.array([0.]), np.array([2.]), ValueError),
    (np.array([1., 2.]), np.array([0.]), np.array([2.]), ValueError),
    (np.array([np.nan]), np.array([0.4]), np.array([0.2]), ValueError),
])
def test_valid_plot_intervals_rejects_invalid_inputs(values, lower, upper, exception):
    """Should reject invalid shapes, types, and reversed finite interval geometry."""
    with pytest.raises(exception):
        _valid_plot_intervals(values, lower, upper)


def test_valid_plot_intervals_validates_containment_flag():
    """Should reject ambiguous values for the containment contract."""
    with pytest.raises(TypeError, match='boolean'):
        _valid_plot_intervals(np.array([1.]), np.array([0.]), np.array([2.]), contain_estimate=1)


@pytest.mark.parametrize('created', [True, False])
def test_close_on_plot_error_preserves_ownership(created):
    """Should close a failed owned figure and retain a supplied caller-owned figure."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    try:
        with pytest.raises(RuntimeError, match='drawing failed'), _close_on_plot_error(axis, created):
            raise RuntimeError('drawing failed')
        assert plt.fignum_exists(figure.number) is not created
    finally:
        plt.close(figure)


def test_close_on_plot_error_retains_success_and_validates_flag():
    """Should retain successful figures and reject a malformed ownership flag."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    try:
        with _close_on_plot_error(axis, True) as returned:
            assert returned is axis
        assert plt.fignum_exists(figure.number)
        with pytest.raises(TypeError, match='created must be boolean'), _close_on_plot_error(axis, 1):
            pass
    finally:
        plt.close(figure)


@pytest.mark.parametrize('kind', ['time', 'cross_sectional', 'rolling', 'ic', 'temporal', 'partial_ic', 'partial_temporal'])
@pytest.mark.parametrize('supplied', [False, True])
def test_plot_errors_release_only_internally_created_figures(kind, supplied, plot_axis):
    """Should release failed figures across all plot factories while retaining caller-owned axes."""
    import matplotlib.pyplot as plt

    from alpha_research.visualization import (
        plot_cross_sectional_value_summary,
        plot_ic_summary,
        plot_partial_ic_summary,
        plot_partial_temporal_association_summary,
        plot_rolling_temporal_association,
        plot_temporal_association_summary,
        plot_time_series_value,
    )

    values = pd.DataFrame({'time': [1, 2], 'symbol': ['A', 'A'], 'value': [1., 2.]})
    rolling = pd.DataFrame({'window_end': [1, 2], 'association': [0.1, 0.2],
                            'bootstrap_ci_lower': [0., 0.], 'bootstrap_ci_upper': [0.3, 0.3]})
    summary = pd.DataFrame({'feature': ['x'], 'mean': [0.2], 'association': [0.2],
                            'wald_ci_lower': [0.1], 'wald_ci_upper': [0.3], 'covariates': ['z']})
    function, frame = {
        'time': (plot_time_series_value, values),
        'cross_sectional': (plot_cross_sectional_value_summary, values),
        'rolling': (plot_rolling_temporal_association, rolling),
        'ic': (plot_ic_summary, summary),
        'temporal': (plot_temporal_association_summary, summary),
        'partial_ic': (plot_partial_ic_summary, summary),
        'partial_temporal': (plot_partial_temporal_association_summary, summary),
    }[kind]
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        function(frame, ax=plot_axis if supplied else None, color='invalid-color')
    assert plt.get_fignums() == before
    assert plt.fignum_exists(plot_axis.figure.number)


def test_successful_auto_created_plot_remains_available_until_closed():
    """Should return a live owned figure and let the caller release it explicitly."""
    pytest.importorskip('matplotlib')
    import matplotlib.pyplot as plt

    from alpha_research.visualization import plot_time_series_value

    before = set(plt.get_fignums())
    axis = plot_time_series_value(pd.DataFrame({'time': [1], 'symbol': ['A'], 'value': [2.]}))
    try:
        assert set(plt.get_fignums()) - before == {axis.figure.number}
        axis.figure.canvas.draw()
    finally:
        plt.close(axis.figure)
    assert set(plt.get_fignums()) == before
