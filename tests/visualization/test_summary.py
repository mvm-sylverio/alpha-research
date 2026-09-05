import builtins

import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.visualization import (
    plot_ic_summary,
    plot_temporal_association_summary,
)
from alpha_research.visualization.summary import (
    _create_axis,
    _plot_ranked_estimates,
    _prepare_ranked_summary,
    _resolve_significance_column,
    _significance_labels,
    _validate_ranking_arguments,
)


# ------------------------------------------------------
# fixtures
# ------------------------------------------------------
@pytest.fixture
def ic_summary_table_pandas():
    """Create a corrected IC table with positive and negative candidates."""
    return pd.DataFrame({
        'feature': ['weak', 'negative_signal', 'positive_signal'],
        'mean': [0.02, -0.06, 0.04],
        'fdr_rejected': [False, True, True],
    })


@pytest.fixture
def temporal_summary_table_pandas():
    """Create temporal-association results with Wald decisions and intervals."""
    return pd.DataFrame({
        'feature': ['weak', 'negative_signal', 'positive_signal'],
        'association': [0.02, -0.06, 0.04],
        'wald_ci_lower': [-0.01, -0.10, 0.01],
        'wald_ci_upper': [0.05, -0.02, 0.07],
        'reject_h0': [False, True, True],
    })


# ------------------------------------------------------
# private helpers
# ------------------------------------------------------
@pytest.mark.parametrize(
    ('top_n', 'sort_by', 'significance_col', 'error_type', 'message'),
    [
        ('2', 'absolute', None, TypeError, 'top_n'),
        (True, 'absolute', None, TypeError, 'top_n'),
        (0, 'absolute', None, ValueError, 'top_n'),
        (None, 'unsupported', None, ValueError, 'sort_by'),
        (None, 'absolute', 1, TypeError, 'significance_col'),
    ],
)
def test_validate_ranking_arguments_rejects_invalid_values(
        top_n,
        sort_by,
        significance_col,
        error_type,
        message,
):
    """The shared ranking validator should reject unsupported arguments."""
    with pytest.raises(error_type, match=message):
        _validate_ranking_arguments(top_n, sort_by, significance_col)


def test_resolve_significance_column_uses_the_documented_precedence():
    """Explicit, FDR, fallback, and absent decisions should resolve in order."""
    fdr_table = pd.DataFrame({
        'feature': ['a'],
        'fdr_rejected': [True],
        'reject_h0': [True],
    })
    fallback_table = fdr_table.drop(columns='fdr_rejected')
    no_decision_table = fallback_table.drop(columns='reject_h0')

    assert _resolve_significance_column(fdr_table, 'custom', 'reject_h0') == 'custom'
    assert _resolve_significance_column(fdr_table, None, 'reject_h0') == 'fdr_rejected'
    assert _resolve_significance_column(fallback_table, None, 'reject_h0') == 'reject_h0'
    assert _resolve_significance_column(no_decision_table, None, 'reject_h0') is None


@pytest.mark.parametrize(
    ('summary_table', 'significance_col', 'fallback_col', 'message'),
    [
        ([{'feature': 'a'}], None, None, 'DataFrame'),
        (pd.DataFrame({'feature': ['a']}), 1, None, 'significance_col'),
        (pd.DataFrame({'feature': ['a']}), None, 1, 'fallback_col'),
    ],
)
def test_resolve_significance_column_validates_inputs(
        summary_table,
        significance_col,
        fallback_col,
        message,
):
    """The significance resolver should expose clear type errors."""
    with pytest.raises(TypeError, match=message):
        _resolve_significance_column(
            summary_table,
            significance_col,
            fallback_col,
        )


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_prepare_ranked_summary_filters_and_orders_finite_estimates(
        ic_summary_table_pandas,
        backend,
):
    """The preparation helper should preserve only the requested ranked rows."""
    table = ic_summary_table_pandas.assign(mean=[0.02, -0.06, np.nan])
    table = table if backend == 'pandas' else pl.from_pandas(table)

    result = _prepare_ranked_summary(
        table,
        feature_col='feature',
        value_col='mean',
        significance_col='fdr_rejected',
        top_n=1,
        sort_by='absolute',
    )

    assert isinstance(result, pd.DataFrame)
    assert result['feature'].tolist() == ['negative_signal']
    assert result['fdr_rejected'].tolist() == [True]


@pytest.mark.parametrize(
    ('summary_table', 'top_n', 'sort_by', 'ci_lower_col', 'ci_upper_col', 'error_type', 'message'),
    [
        (pd.DataFrame({'feature': ['a'], 'mean': [0.1]}), 0, 'absolute', None, None, ValueError, 'top_n'),
        (pd.DataFrame({'feature': ['a'], 'mean': [0.1]}), None, 'other', None, None, ValueError, 'sort_by'),
        (pd.DataFrame({'feature': ['a'], 'mean': [0.1]}), None, 'absolute', 'lower', None, ValueError, 'provided together'),
        (pd.DataFrame({'feature': ['a'], 'mean': [np.nan]}), None, 'absolute', None, None, ValueError, 'finite'),
        (pd.DataFrame({'feature': ['a']}), None, 'absolute', None, None, KeyError, 'missing required columns'),
        ([{'feature': 'a', 'mean': 0.1}], None, 'absolute', None, None, TypeError, 'DataFrame'),
    ],
)
def test_prepare_ranked_summary_validates_input(
        summary_table,
        top_n,
        sort_by,
        ci_lower_col,
        ci_upper_col,
        error_type,
        message,
):
    """The preparation helper should reject invalid plotting data early."""
    with pytest.raises(error_type, match=message):
        _prepare_ranked_summary(
            summary_table,
            feature_col='feature',
            value_col='mean',
            significance_col=None,
            top_n=top_n,
            sort_by=sort_by,
            ci_lower_col=ci_lower_col,
            ci_upper_col=ci_upper_col,
        )


def test_significance_labels_cover_known_and_custom_decisions():
    """Known decisions should use statistical labels and custom ones stay explicit."""
    assert _significance_labels('fdr_rejected') == (
        'Passed FDR correction',
        'Did not pass FDR correction',
    )
    assert _significance_labels('custom_decision') == (
        'custom_decision=True',
        'custom_decision=False',
    )


def test_significance_labels_rejects_non_string_column_name():
    """Legend-label creation should validate the decision-column name."""
    with pytest.raises(TypeError, match='significance_col'):
        _significance_labels(None)


def test_plot_ranked_estimates_draws_points_and_intervals(
        temporal_summary_table_pandas,
):
    """The drawing helper should combine points, Wald intervals, and zero."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    _plot_ranked_estimates(
        temporal_summary_table_pandas,
        feature_col='feature',
        value_col='association',
        significance_col='reject_h0',
        ax=axis,
        color='C0',
        non_significant_color='0.65',
        ci_lower_col='wald_ci_lower',
        ci_upper_col='wald_ci_upper',
    )

    assert len(axis.containers) == 2
    assert any(line.get_linestyle() == '--' for line in axis.lines)
    plt.close(figure)


@pytest.mark.parametrize(
    ('ranked_table', 'ax', 'ci_lower_col', 'ci_upper_col', 'error_type', 'message'),
    [
        (pd.DataFrame({'feature': ['a'], 'mean': [0.1]}), None, None, None, ValueError, 'ax'),
        (pd.DataFrame({'feature': ['a']}), object(), None, None, KeyError, 'missing required columns'),
        (pd.DataFrame({'feature': ['a'], 'mean': [0.1]}), object(), 'lower', None, ValueError, 'provided together'),
        ([{'feature': 'a', 'mean': 0.1}], object(), None, None, TypeError, 'DataFrame'),
        (pl.DataFrame({'feature': ['a'], 'mean': [0.1]}), object(), None, None, TypeError, 'Pandas'),
    ],
)
def test_plot_ranked_estimates_validates_input(
        ranked_table,
        ax,
        ci_lower_col,
        ci_upper_col,
        error_type,
        message,
):
    """The drawing helper should validate its direct-use contract."""
    with pytest.raises(error_type, match=message):
        _plot_ranked_estimates(
            ranked_table,
            feature_col='feature',
            value_col='mean',
            significance_col=None,
            ax=ax,
            color='C0',
            non_significant_color='0.65',
            ci_lower_col=ci_lower_col,
            ci_upper_col=ci_upper_col,
        )


def test_create_axis_reuses_an_existing_axis():
    """The axis helper should preserve a supplied axis without creating one."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    returned_axis, created = _create_axis(axis, n_features=2)

    assert returned_axis is axis
    assert created is False
    plt.close(figure)


@pytest.mark.parametrize('n_features', [0, -1, True, '2'])
def test_create_axis_validates_feature_count(n_features):
    """The axis helper should require a positive integer feature count."""
    with pytest.raises(ValueError, match='n_features'):
        _create_axis(None, n_features=n_features)


# ------------------------------------------------------
# plot functions
# ------------------------------------------------------
@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_plot_ic_summary_ranks_candidates_by_absolute_mean(
        ic_summary_table_pandas,
        backend,
):
    """The IC plot should rank finite features and show FDR decisions."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    table = (
        ic_summary_table_pandas
        if backend == 'pandas'
        else pl.from_pandas(ic_summary_table_pandas)
    )
    figure, axis = plt.subplots()
    returned_axis = plot_ic_summary(table, top_n=2, ax=axis, title='IC ranking')

    assert returned_axis is axis
    assert [label.get_text() for label in axis.get_yticklabels()] == [
        'negative_signal',
        'positive_signal',
    ]
    assert axis.get_xlabel() == 'Mean information coefficient'
    assert axis.get_ylabel() == 'Feature'
    assert axis.get_title() == 'IC ranking'
    assert axis.get_legend().get_texts()[0].get_text() == 'Passed FDR correction'
    assert len(axis.collections) == 1
    assert len(axis.lines) == 1
    plt.close(figure)


def test_plot_ic_summary_accepts_uncorrected_results(ic_summary_table_pandas):
    """Uncorrected IC summaries should remain useful without a legend."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    plot_ic_summary(ic_summary_table_pandas.drop(columns='fdr_rejected'), ax=axis)

    assert axis.get_legend() is None
    assert len(axis.collections) == 1
    plt.close(figure)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_plot_temporal_association_summary_draws_wald_intervals(
        temporal_summary_table_pandas,
        backend,
):
    """The temporal plot should rank associations and display Wald intervals."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    table = (
        temporal_summary_table_pandas
        if backend == 'pandas'
        else pl.from_pandas(temporal_summary_table_pandas)
    )
    figure, axis = plt.subplots()
    returned_axis = plot_temporal_association_summary(
        table,
        ax=axis,
        title='Temporal ranking',
    )

    assert returned_axis is axis
    assert [label.get_text() for label in axis.get_yticklabels()] == [
        'negative_signal',
        'positive_signal',
        'weak',
    ]
    assert axis.get_xlabel() == 'Temporal association'
    assert axis.get_ylabel() == 'Feature'
    assert axis.get_title() == 'Temporal ranking'
    assert axis.get_legend().get_texts()[0].get_text() == 'Wald rejects null hypothesis'
    assert len(axis.containers) == 2
    assert any(line.get_linestyle() == '--' for line in axis.lines)
    plt.close(figure)


def test_plot_temporal_association_summary_prefers_fdr_decisions(
        temporal_summary_table_pandas,
):
    """Explicitly corrected temporal summaries should show the FDR decision."""
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    corrected_table = temporal_summary_table_pandas.assign(
        fdr_rejected=[False, False, True],
    )
    figure, axis = plt.subplots()
    plot_temporal_association_summary(corrected_table, ax=axis)

    assert axis.get_legend().get_texts()[0].get_text() == 'Passed FDR correction'
    plt.close(figure)


@pytest.mark.parametrize(
    ('plot_function', 'table'),
    [
        (plot_ic_summary, pd.DataFrame({'feature': ['a'], 'mean': [np.nan]})),
        (
            plot_temporal_association_summary,
            pd.DataFrame({
                'feature': ['a'],
                'association': [np.nan],
                'wald_ci_lower': [np.nan],
                'wald_ci_upper': [np.nan],
            }),
        ),
    ],
)
def test_summary_plots_require_a_finite_estimate(plot_function, table):
    """No chart should be produced when every estimate is non-finite."""
    with pytest.raises(ValueError, match='finite'):
        plot_function(table)


@pytest.mark.parametrize('top_n', [0, -1, True, '2'])
def test_plot_ic_summary_validates_top_n(ic_summary_table_pandas, top_n):
    """Ranking limits should be positive integers when provided."""
    with pytest.raises((TypeError, ValueError), match='top_n'):
        plot_ic_summary(ic_summary_table_pandas, top_n=top_n)


def test_plot_ic_summary_validates_required_columns():
    """The plot should require the IC summary schema before importing Matplotlib."""
    with pytest.raises(KeyError, match='missing required columns'):
        plot_ic_summary(pd.DataFrame({'feature': ['a']}))


def test_plot_temporal_association_summary_requires_wald_interval_columns():
    """The temporal forest plot should require its documented interval bounds."""
    with pytest.raises(KeyError, match='missing required columns'):
        plot_temporal_association_summary(
            pd.DataFrame({'feature': ['a'], 'association': [0.1]}),
        )


@pytest.mark.parametrize(
    'plot_function, table',
    [
        (plot_ic_summary, pd.DataFrame({'feature': ['a'], 'mean': [0.1]})),
        (
            plot_temporal_association_summary,
            pd.DataFrame({
                'feature': ['a'],
                'association': [0.1],
                'wald_ci_lower': [0.0],
                'wald_ci_upper': [0.2],
            }),
        ),
    ],
)
def test_summary_plots_explain_missing_optional_backend(
        monkeypatch,
        plot_function,
        table,
):
    """Both plots should explain how to install the optional Matplotlib extra."""
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
        plot_function(table)
