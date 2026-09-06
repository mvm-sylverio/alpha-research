from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl

from alpha_research._utils import _validate_df, _validate_df_type

__all__ = [
    'plot_ic_summary',
    'plot_partial_ic_summary',
    'plot_temporal_association_summary',
    'plot_partial_temporal_association_summary',
]


def _validate_ranking_arguments(
        top_n: int | None,
        sort_by: Literal['absolute', 'value'],
        significance_col: str | None,
) -> None:
    """
    Validate shared arguments that control ranked summary plots.

    Parameters
    ----------
    top_n : int | None
        Maximum number of ranked features to plot, or ``None`` to retain all
        finite estimates.
    sort_by : {'absolute', 'value'}
        Rule used to rank estimates before plotting.
    significance_col : str | None
        Optional name of a boolean decision column.

    Returns
    -------
    None
        This helper returns ``None`` when every argument is valid.

    Raises
    ------
    TypeError
        If ``top_n`` is not an integer or ``None``, or if
        ``significance_col`` is neither a string nor ``None``.
    ValueError
        If ``top_n`` is less than one or ``sort_by`` is unsupported.
    """
    if top_n is not None:
        if not isinstance(top_n, (int, np.integer)) or isinstance(top_n, bool):
            raise TypeError('top_n must be a positive integer or None.')
        if top_n < 1:
            raise ValueError('top_n must be a positive integer or None.')

    if sort_by not in ('absolute', 'value'):
        raise ValueError("sort_by must be either 'absolute' or 'value'.")

    if significance_col is not None and not isinstance(significance_col, str):
        raise TypeError('significance_col must be a string or None.')


def _resolve_significance_column(
        summary_table: pd.DataFrame | pl.DataFrame,
        significance_col: str | None,
        fallback_col: str | None,
) -> str | None:
    """
    Resolve the significance-decision column to display in a ranked plot.

    An explicit ``significance_col`` has priority. Otherwise, an available
    FDR decision is preferred, followed by the function-specific fallback.

    Parameters
    ----------
    summary_table : pd.DataFrame | pl.DataFrame
        Summary table whose columns are inspected.
    significance_col : str | None
        Explicit decision-column name, if chosen by the caller.
    fallback_col : str | None
        Decision column to use when no explicit or FDR column is available.

    Returns
    -------
    str | None
        The selected column name, or ``None`` when no decision column should
        be displayed.

    Raises
    ------
    TypeError
        If ``summary_table`` is not a Pandas or Polars DataFrame, or if a
        supplied column name is not a string or ``None``.
    """
    _validate_df_type(summary_table)
    for name, value in [
        ('significance_col', significance_col),
        ('fallback_col', fallback_col),
    ]:
        if value is not None and not isinstance(value, str):
            raise TypeError(f'{name} must be a string or None.')

    if significance_col is not None:
        return significance_col

    if 'fdr_rejected' in summary_table.columns:
        return 'fdr_rejected'

    if fallback_col is not None and fallback_col in summary_table.columns:
        return fallback_col

    return None


def _prepare_ranked_summary(
        summary_table: pd.DataFrame | pl.DataFrame,
        feature_col: str,
        value_col: str,
        significance_col: str | None,
        top_n: int | None,
        sort_by: Literal['absolute', 'value'],
        ci_lower_col: str | None = None,
        ci_upper_col: str | None = None,
) -> pd.DataFrame:
    """
    Validate, normalize, and rank summary estimates for plotting.

    Non-finite estimates are excluded. The returned Pandas DataFrame is a
    plotting-only copy, so neither the input table nor its backend is changed.

    Parameters
    ----------
    summary_table : pd.DataFrame | pl.DataFrame
        Source table containing one estimate per feature.
    feature_col, value_col : str
        Columns containing feature labels and the estimate to rank.
    significance_col : str | None
        Optional boolean decision column retained for marker styling.
    top_n : int | None
        Maximum number of highest-ranked rows to retain, or ``None`` for all
        rows with finite estimates.
    sort_by : {'absolute', 'value'}
        Ranking rule for the estimate column.
    ci_lower_col, ci_upper_col : str | None, default None
        Optional matching confidence-interval bound columns.

    Returns
    -------
    pd.DataFrame
        Finite, ordered plotting data with a reset positional index.

    Raises
    ------
    TypeError
        If the input table or ranking arguments have unsupported types.
    KeyError
        If a selected required column is absent.
    ValueError
        If the ranking arguments are invalid, only one interval-bound column
        is supplied, or no finite estimate is available.
    """
    _validate_ranking_arguments(top_n, sort_by, significance_col)
    if (ci_lower_col is None) != (ci_upper_col is None):
        raise ValueError(
            'ci_lower_col and ci_upper_col must be provided together.',
        )
    required_columns = [feature_col, value_col]
    if significance_col is not None:
        required_columns.append(significance_col)
    if ci_lower_col is not None:
        required_columns.append(ci_lower_col)
    if ci_upper_col is not None:
        required_columns.append(ci_upper_col)
    _validate_df(summary_table, required_columns, check_all_missing=False)

    selected_columns = list(dict.fromkeys(required_columns))
    pandas_table = (
        summary_table[selected_columns].copy()
        if isinstance(summary_table, pd.DataFrame)
        else summary_table.select(selected_columns).to_pandas()
    )
    pandas_table[value_col] = pd.to_numeric(pandas_table[value_col], errors='coerce')
    pandas_table = pandas_table.loc[np.isfinite(pandas_table[value_col])].copy()

    if pandas_table.empty:
        raise ValueError(
            f'summary_table must contain at least one finite {value_col} value.',
        )

    if ci_lower_col is not None:
        pandas_table[ci_lower_col] = pd.to_numeric(
            pandas_table[ci_lower_col],
            errors='coerce',
        )
    if ci_upper_col is not None:
        pandas_table[ci_upper_col] = pd.to_numeric(
            pandas_table[ci_upper_col],
            errors='coerce',
        )
    if significance_col is not None:
        pandas_table[significance_col] = (
            pandas_table[significance_col].fillna(False).astype(bool)
        )

    sort_values = (
        pandas_table[value_col].abs()
        if sort_by == 'absolute'
        else pandas_table[value_col]
    )
    pandas_table = (
        pandas_table.assign(_sort_value=sort_values)
        .sort_values('_sort_value', ascending=False, kind='stable')
        .drop(columns='_sort_value')
    )
    if top_n is not None:
        pandas_table = pandas_table.head(top_n)

    return pandas_table.reset_index(drop=True)


def _significance_labels(significance_col: str) -> tuple[str, str]:
    """
    Return human-readable legend labels for a significance decision column.

    Parameters
    ----------
    significance_col : str
        Name of the boolean decision column used in the plotted summary.

    Returns
    -------
    tuple[str, str]
        Labels for ``True`` and ``False`` decisions, in that order.

    Raises
    ------
    TypeError
        If ``significance_col`` is not a string.
    """
    if not isinstance(significance_col, str):
        raise TypeError('significance_col must be a string.')

    if significance_col == 'fdr_rejected':
        return 'Passed FDR correction', 'Did not pass FDR correction'
    if significance_col == 'reject_h0':
        return 'Wald rejects null hypothesis', 'Wald does not reject null hypothesis'

    return f'{significance_col}=True', f'{significance_col}=False'


def _plot_ranked_estimates(
        ranked_table: pd.DataFrame,
        feature_col: str,
        value_col: str,
        significance_col: str | None,
        ax: Any,
        color: str,
        non_significant_color: str,
        ci_lower_col: str | None = None,
        ci_upper_col: str | None = None,
) -> None:
    """
    Draw ranked point estimates or forest estimates on an existing axis.

    Rows with valid confidence-interval bounds receive horizontal error bars;
    rows without valid bounds remain visible as points. This helper does not
    calculate intervals or statistical decisions.

    Parameters
    ----------
    ranked_table : pd.DataFrame
        Preprocessed plotting data produced by ``_prepare_ranked_summary()``.
    feature_col, value_col : str
        Columns containing feature labels and finite estimates.
    significance_col : str | None
        Optional boolean column used to distinguish marker groups.
    ax : matplotlib.axes.Axes
        Existing axis receiving the plot.
    color, non_significant_color : str
        Matplotlib colors for highlighted and non-highlighted estimates.
    ci_lower_col, ci_upper_col : str | None, default None
        Optional matching confidence-interval bound columns.

    Returns
    -------
    None
        The supplied axis is modified in place.

    Raises
    ------
    TypeError
        If ``ranked_table`` is not a Pandas DataFrame.
    KeyError
        If selected plotting columns are absent.
    ValueError
        If no axis is supplied or only one interval-bound column is provided.
    """
    if ax is None:
        raise ValueError('ax must be a Matplotlib Axes instance.')
    if (ci_lower_col is None) != (ci_upper_col is None):
        raise ValueError(
            'ci_lower_col and ci_upper_col must be provided together.',
        )
    if not isinstance(ranked_table, pd.DataFrame):
        raise TypeError('ranked_table must be a Pandas DataFrame.')

    required_columns = [feature_col, value_col]
    if significance_col is not None:
        required_columns.append(significance_col)
    if ci_lower_col is not None:
        required_columns.extend([ci_lower_col, ci_upper_col])
    _validate_df(ranked_table, required_columns, check_all_missing=False)

    y_positions = np.arange(len(ranked_table))
    values = ranked_table[value_col].to_numpy(dtype=float)

    if significance_col is None:
        groups = [(None, color, None)]
    else:
        positive_label, negative_label = _significance_labels(significance_col)
        groups = [
            (True, color, positive_label),
            (False, non_significant_color, negative_label),
        ]

    for decision, group_color, label in groups:
        mask = (
            np.ones(len(ranked_table), dtype=bool)
            if decision is None
            else ranked_table[significance_col].to_numpy(dtype=bool) == decision
        )
        if not mask.any():
            continue

        if ci_lower_col is None or ci_upper_col is None:
            ax.scatter(
                values[mask],
                y_positions[mask],
                color=group_color,
                label=label,
                zorder=3,
            )
            continue

        lower = ranked_table[ci_lower_col].to_numpy(dtype=float)
        upper = ranked_table[ci_upper_col].to_numpy(dtype=float)
        has_interval = (
            mask
            & np.isfinite(lower)
            & np.isfinite(upper)
            & (lower <= values)
            & (values <= upper)
        )
        if has_interval.any():
            ax.errorbar(
                values[has_interval],
                y_positions[has_interval],
                xerr=[
                    values[has_interval] - lower[has_interval],
                    upper[has_interval] - values[has_interval],
                ],
                fmt='o',
                capsize=3,
                color=group_color,
                ecolor=group_color,
                label=label,
                zorder=3,
            )

        without_interval = mask & ~has_interval
        if without_interval.any():
            ax.scatter(
                values[without_interval],
                y_positions[without_interval],
                color=group_color,
                label=label if not has_interval.any() else None,
                zorder=3,
            )

    ax.axvline(0.0, color='black', linewidth=1.0, linestyle='--', zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ranked_table[feature_col].astype(str).to_list())
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.25)
    ax.set_axisbelow(True)

    if significance_col is not None:
        ax.legend(frameon=False)


def _create_axis(ax: Any, n_features: int) -> tuple[Any, bool]:
    """
    Return a supplied axis or create one sized for the number of features.

    Parameters
    ----------
    ax : matplotlib.axes.Axes | None
        Existing axis to reuse, or ``None`` to create a new Matplotlib axis.
    n_features : int
        Number of plotted feature labels used to determine a readable figure
        height when creating an axis.

    Returns
    -------
    tuple[matplotlib.axes.Axes, bool]
        The plotting axis and whether it was created by this helper.

    Raises
    ------
    ImportError
        If Matplotlib is not installed and a new axis is required.
    ValueError
        If ``n_features`` is not a positive integer.
    """
    if not isinstance(n_features, (int, np.integer)) or isinstance(n_features, bool):
        raise ValueError('n_features must be a positive integer.')
    if n_features < 1:
        raise ValueError('n_features must be a positive integer.')

    if ax is not None:
        return ax, False

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            'Matplotlib is required for plotting. Install alpha-research[viz].',
        ) from error

    figure_height = max(2.5, 0.35 * n_features + 1.5)
    _, ax = plt.subplots(figsize=(8.0, figure_height))
    return ax, True


def _resolve_covariates_label(
        summary_table: pd.DataFrame | pl.DataFrame,
        covariates_col: str,
        covariates_label: str | None,
) -> str:
    """
    Resolve a single visible covariate specification for a partial-summary plot.

    Parameters
    ----------
    summary_table : pd.DataFrame | pl.DataFrame
        Partial-summary table containing the covariate metadata when no label
        is supplied explicitly.
    covariates_col : str
        Column that records the covariates used in each table row.
    covariates_label : str | None
        Explicit label used instead of reading covariates_col.

    Returns
    -------
    str
        Human-readable covariate specification shown on the plot.

    Raises
    ------
    TypeError
        If the table or label arguments have unsupported types.
    KeyError
        If covariates_col is absent and no explicit label is supplied.
    ValueError
        If an inferred label is missing or inconsistent across rows.
    """
    _validate_df_type(summary_table)
    if not isinstance(covariates_col, str):
        raise TypeError('covariates_col must be a string.')
    if covariates_label is not None:
        if not isinstance(covariates_label, str):
            raise TypeError('covariates_label must be a string or None.')
        if not covariates_label.strip():
            raise ValueError('covariates_label must not be empty.')
        return covariates_label

    _validate_df(summary_table, [covariates_col], check_all_missing=False)
    values = (
        summary_table[covariates_col].dropna().astype(str).unique().tolist()
        if isinstance(summary_table, pd.DataFrame)
        else summary_table[covariates_col].drop_nulls().cast(pl.String).unique().to_list()
    )
    if not values:
        raise ValueError(f'{covariates_col} must contain a covariate specification.')
    if len(values) != 1:
        raise ValueError(
            f'{covariates_col} must contain one shared covariate specification.',
        )

    return values[0]


def _add_covariates_note(
        ax: Any,
        covariates_label: str,
        title: str | None,
) -> None:
    """
    Add a visible partial-correlation covariate note to an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis receiving the note.
    covariates_label : str
        Human-readable covariate specification.
    title : str | None
        Optional title whose padding is increased to avoid overlapping the
        covariate note.

    Returns
    -------
    None
        The supplied axis is modified in place.
    """
    if title is not None:
        ax.set_title(title, pad=24)
    ax.text(
        0.0,
        1.02,
        f'Conditioned on: {covariates_label}',
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        color='0.35',
        fontsize='small',
    )


def plot_ic_summary(
        summary_table: pd.DataFrame | pl.DataFrame,
        top_n: int | None = None,
        sort_by: Literal['absolute', 'value'] = 'absolute',
        significance_col: str | None = None,
        ax: Any = None,
        feature_col: str = 'feature',
        value_col: str = 'mean',
        color: str = 'C0',
        non_significant_color: str = '0.65',
        title: str | None = None,
) -> Any:
    """
    Plot ranked mean information coefficients from an IC summary table.

    Features are ordered by absolute mean IC by default, preserving the sign
    on the horizontal axis. This makes positively and negatively associated
    candidates directly comparable without inferring semantics from feature
    names.

    Parameters
    ----------
    summary_table : pd.DataFrame | pl.DataFrame
        Table returned by ``ic_summary_table().table``. A table previously
        passed through ``fdr_correction()`` is supported and its FDR decision
        is shown automatically.
    top_n : int | None, default None
        Number of highest-ranked features to show. ``None`` displays every
        feature with a finite mean IC.
    sort_by : {'absolute', 'value'}, default 'absolute'
        Ranking rule. ``'absolute'`` prioritizes predictive magnitude;
        ``'value'`` orders from the largest signed mean IC to the smallest.
    significance_col : str | None, default None
        Boolean decision column used to distinguish markers. When omitted,
        ``'fdr_rejected'`` is used if available; otherwise all markers share
        one style.
    ax : matplotlib.axes.Axes | None, default None
        Axis to draw into. A new, height-adjusted axis is created when omitted.
    feature_col, value_col : str
        Columns containing feature labels and mean IC values.
    color, non_significant_color : str
        Matplotlib colors for significant and non-significant decisions.
    title : str | None, default None
        Optional axis title.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the ranked mean IC plot and zero reference line.

    Raises
    ------
    ImportError
        If Matplotlib is not installed. Install the optional ``viz`` extra.
    KeyError
        If a selected required column is absent.
    TypeError
        If the table or a ranking argument has an unsupported type.
    ValueError
        If a ranking argument is invalid or no finite mean IC is available.

    Notes
    -----
    ``ic_summary_table()`` does not expose a confidence interval, so this
    plot intentionally shows points only. It does not derive a substitute
    interval or recompute any statistical decision.
    """
    _validate_ranking_arguments(top_n, sort_by, significance_col)
    _validate_df(summary_table, [], check_all_missing=False)
    resolved_significance_col = _resolve_significance_column(
        summary_table,
        significance_col,
        fallback_col=None,
    )
    ranked_table = _prepare_ranked_summary(
        summary_table,
        feature_col=feature_col,
        value_col=value_col,
        significance_col=resolved_significance_col,
        top_n=top_n,
        sort_by=sort_by,
    )
    ax, created_axis = _create_axis(ax, len(ranked_table))
    _plot_ranked_estimates(
        ranked_table,
        feature_col=feature_col,
        value_col=value_col,
        significance_col=resolved_significance_col,
        ax=ax,
        color=color,
        non_significant_color=non_significant_color,
    )
    ax.set_xlabel('Mean information coefficient')
    ax.set_ylabel('Feature')
    if title is not None:
        ax.set_title(title)
    if created_axis:
        ax.figure.tight_layout()

    return ax


def plot_temporal_association_summary(
        summary_table: pd.DataFrame | pl.DataFrame,
        top_n: int | None = None,
        sort_by: Literal['absolute', 'value'] = 'absolute',
        significance_col: str | None = None,
        ax: Any = None,
        feature_col: str = 'feature',
        value_col: str = 'association',
        ci_lower_col: str = 'wald_ci_lower',
        ci_upper_col: str = 'wald_ci_upper',
        color: str = 'C0',
        non_significant_color: str = '0.65',
        title: str | None = None,
) -> Any:
    """
    Plot ranked temporal associations with their existing Wald intervals.

    The function only visualizes intervals and decisions already present in
    the summary table. It does not apply FDR correction or alter the temporal
    association methodology.

    Parameters
    ----------
    summary_table : pd.DataFrame | pl.DataFrame
        Table returned by ``temporal_association_summary_table()``. A table
        explicitly passed through ``fdr_correction()`` is also supported.
    top_n, sort_by, ax, feature_col, value_col, color,
    non_significant_color, title
        Have the same meaning as in ``plot_ic_summary()``.
    significance_col : str | None, default None
        Boolean decision column used to distinguish markers. When omitted,
        ``'fdr_rejected'`` is preferred when present; otherwise the summary
        table's ``'reject_h0'`` Wald decision is used when available.
    ci_lower_col, ci_upper_col : str
        Columns containing the Wald confidence-interval bounds.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing ranked associations, Wald intervals, and a zero
        reference line.

    Raises
    ------
    ImportError
        If Matplotlib is not installed. Install the optional ``viz`` extra.
    KeyError
        If a selected required column is absent.
    TypeError
        If the table or a ranking argument has an unsupported type.
    ValueError
        If a ranking argument is invalid or no finite association is available.
    """
    _validate_ranking_arguments(top_n, sort_by, significance_col)
    _validate_df(summary_table, [], check_all_missing=False)
    resolved_significance_col = _resolve_significance_column(
        summary_table,
        significance_col,
        fallback_col='reject_h0',
    )
    ranked_table = _prepare_ranked_summary(
        summary_table,
        feature_col=feature_col,
        value_col=value_col,
        significance_col=resolved_significance_col,
        top_n=top_n,
        sort_by=sort_by,
        ci_lower_col=ci_lower_col,
        ci_upper_col=ci_upper_col,
    )
    ax, created_axis = _create_axis(ax, len(ranked_table))
    _plot_ranked_estimates(
        ranked_table,
        feature_col=feature_col,
        value_col=value_col,
        significance_col=resolved_significance_col,
        ax=ax,
        color=color,
        non_significant_color=non_significant_color,
        ci_lower_col=ci_lower_col,
        ci_upper_col=ci_upper_col,
    )
    ax.set_xlabel('Temporal association')
    ax.set_ylabel('Feature')
    if title is not None:
        ax.set_title(title)
    if created_axis:
        ax.figure.tight_layout()

    return ax


def plot_partial_ic_summary(
        summary_table: pd.DataFrame | pl.DataFrame,
        top_n: int | None = None,
        sort_by: Literal['absolute', 'value'] = 'absolute',
        significance_col: str | None = None,
        ax: Any = None,
        feature_col: str = 'feature',
        value_col: str = 'mean',
        covariates_col: str = 'covariates',
        covariates_label: str | None = None,
        color: str = 'C0',
        non_significant_color: str = '0.65',
        title: str | None = None,
) -> Any:
    """
    Plot ranked mean partial information coefficients.

    This is the partial-IC counterpart to ``plot_ic_summary()``. The plot
    always states the covariates conditioned on, using the summary table's
    metadata by default or a caller-provided label.

    Parameters
    ----------
    summary_table : pd.DataFrame | pl.DataFrame
        Table returned by ``partial_ic_summary_table().table``, optionally
        passed through ``fdr_correction()``.
    top_n, sort_by, significance_col, ax, feature_col, value_col, color,
    non_significant_color, title
        Have the same meaning as in ``plot_ic_summary()``.
    covariates_col : str, default 'covariates'
        Column containing the shared covariate specification.
    covariates_label : str | None, default None
        Explicit covariate text shown on the chart. When None, the label is
        inferred from covariates_col, which must contain one shared value.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the ranked partial IC plot and covariate note.

    Raises
    ------
    ImportError
        If Matplotlib is not installed. Install the optional ``viz`` extra.
    KeyError
        If a selected required column is absent.
    TypeError
        If the table or plotting arguments have unsupported types.
    ValueError
        If the ranking or covariate specification is invalid.
    """
    resolved_covariates_label = _resolve_covariates_label(
        summary_table,
        covariates_col,
        covariates_label,
    )
    created_axis = ax is None
    ax = plot_ic_summary(
        summary_table,
        top_n=top_n,
        sort_by=sort_by,
        significance_col=significance_col,
        ax=ax,
        feature_col=feature_col,
        value_col=value_col,
        color=color,
        non_significant_color=non_significant_color,
        title=title,
    )
    ax.set_xlabel('Mean partial information coefficient')
    _add_covariates_note(ax, resolved_covariates_label, title)
    if created_axis:
        ax.figure.tight_layout()

    return ax


def plot_partial_temporal_association_summary(
        summary_table: pd.DataFrame | pl.DataFrame,
        top_n: int | None = None,
        sort_by: Literal['absolute', 'value'] = 'absolute',
        significance_col: str | None = None,
        ax: Any = None,
        feature_col: str = 'feature',
        value_col: str = 'association',
        ci_lower_col: str = 'wald_ci_lower',
        ci_upper_col: str = 'wald_ci_upper',
        covariates_col: str = 'covariates',
        covariates_label: str | None = None,
        color: str = 'C0',
        non_significant_color: str = '0.65',
        title: str | None = None,
) -> Any:
    """
    Plot ranked partial temporal associations with Wald intervals.

    This is the partial-association counterpart to
    ``plot_temporal_association_summary()``. It visualizes the existing Wald
    intervals and decision columns without recalculating either, and always
    identifies the covariates that were controlled for.

    Parameters
    ----------
    summary_table : pd.DataFrame | pl.DataFrame
        Table returned by ``partial_temporal_association_summary_table()``,
        optionally passed through ``fdr_correction()``.
    top_n, sort_by, significance_col, ax, feature_col, value_col,
    ci_lower_col, ci_upper_col, color, non_significant_color, title
        Have the same meaning as in
        ``plot_temporal_association_summary()``.
    covariates_col : str, default 'covariates'
        Column containing the shared covariate specification.
    covariates_label : str | None, default None
        Explicit covariate text shown on the chart. When None, the label is
        inferred from covariates_col, which must contain one shared value.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the partial-association forest plot and covariate note.

    Raises
    ------
    ImportError
        If Matplotlib is not installed. Install the optional ``viz`` extra.
    KeyError
        If a selected required column is absent.
    TypeError
        If the table or plotting arguments have unsupported types.
    ValueError
        If the ranking or covariate specification is invalid.
    """
    resolved_covariates_label = _resolve_covariates_label(
        summary_table,
        covariates_col,
        covariates_label,
    )
    created_axis = ax is None
    ax = plot_temporal_association_summary(
        summary_table,
        top_n=top_n,
        sort_by=sort_by,
        significance_col=significance_col,
        ax=ax,
        feature_col=feature_col,
        value_col=value_col,
        ci_lower_col=ci_lower_col,
        ci_upper_col=ci_upper_col,
        color=color,
        non_significant_color=non_significant_color,
        title=title,
    )
    ax.set_xlabel('Partial temporal association')
    _add_covariates_note(ax, resolved_covariates_label, title)
    if created_axis:
        ax.figure.tight_layout()

    return ax
