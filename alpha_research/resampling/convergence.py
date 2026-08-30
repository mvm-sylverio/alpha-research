from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from alpha_research._utils import _validate_positive_integer


__all__ = [
    'MonteCarloMetricDiagnostics',
    'MonteCarloLevelDiagnostics',
    'MonteCarloErrorResult',
    'monte_carlo_error',
]


@dataclass(frozen=True, slots=True)
class MonteCarloMetricDiagnostics:
    """
    Between-run Monte Carlo stability diagnostics for one metric at one B.

    Attributes
    ----------
    mean : float
        Mean metric value across independent bootstrap runs.
    std : float
        Sample standard deviation across independent bootstrap runs.
    minimum : float
        Smallest metric value across independent bootstrap runs.
    maximum : float
        Largest metric value across independent bootstrap runs.
    absolute_range : float
        Difference between maximum and minimum metric values.
    relative_range : float | None
        Absolute range divided by abs(mean), or None when abs(mean) is too
        close to zero for a stable relative calculation.
    absolute_tolerance : float | None
        Configured absolute tolerance for this metric.
    relative_tolerance : float | None
        Configured relative tolerance for this metric.
    absolute_passed : bool | None
        Whether the absolute criterion passed, or None when not configured.
    relative_passed : bool | None
        Whether the relative criterion passed, or None when not configured.
    passed : bool
        Whether every configured criterion for this metric passed.
    """
    mean: float
    std: float
    minimum: float
    maximum: float
    absolute_range: float
    relative_range: float | None
    absolute_tolerance: float | None
    relative_tolerance: float | None
    absolute_passed: bool | None
    relative_passed: bool | None
    passed: bool


@dataclass(frozen=True, slots=True)
class MonteCarloLevelDiagnostics:
    """
    Monte Carlo stability diagnostics for one evaluated bootstrap count.

    Attributes
    ----------
    n_bootstraps : int
        Number of bootstrap replications evaluated at this level.
    metrics : dict[str, MonteCarloMetricDiagnostics]
        Per-metric between-run stability diagnostics.
    all_metrics_passed : bool
        Whether every monitored metric passed its configured criteria.
    convergence_confirmed : bool
        Whether this level completed two consecutive passing levels.
    """
    n_bootstraps: int
    metrics: dict[str, MonteCarloMetricDiagnostics]
    all_metrics_passed: bool
    convergence_confirmed: bool


@dataclass(frozen=True, slots=True)
class MonteCarloErrorResult:
    """
    Result of a bootstrap Monte Carlo convergence evaluation.

    Attributes
    ----------
    recommended_n_bootstraps : int | None
        Smallest evaluated bootstrap count that begins two consecutive levels
        satisfying every configured stability criterion, or None when no such
        pair exists.
    converged : bool
        Whether two consecutive evaluated levels satisfied all criteria.
    diagnostics : tuple[MonteCarloLevelDiagnostics, ...]
        Diagnostics for evaluated grid levels only. Levels skipped by early
        stopping are not included, although bootstrap_fn has already generated
        the maximum grid level for every run.
    run_random_states : tuple[int, ...]
        Deterministic child random states assigned to independent runs.
    reference_run_id : int | None
        The deterministic reference run identifier. Run 1 is selected when
        reference_bootstrap_results is retained.
    reference_bootstrap_results : Any | None
        Unmodified results from reference run 1 at recommended_n_bootstraps,
        retained only when return_bootstrap_results is True and convergence is
        confirmed. Results from separate runs are never combined.
    """
    recommended_n_bootstraps: int | None
    converged: bool
    diagnostics: tuple[MonteCarloLevelDiagnostics, ...]
    run_random_states: tuple[int, ...]
    reference_run_id: int | None
    reference_bootstrap_results: Any | None


def _normalize_n_bootstraps_grid(
        n_bootstraps_grid: Iterable[int],
) -> tuple[int, ...]:
    """
    Validate, deduplicate, and sort candidate bootstrap counts.

    Parameters
    ----------
    n_bootstraps_grid : Iterable[int]
        Candidate positive integer replication counts.

    Returns
    -------
    tuple[int, ...]
        Unique candidate counts sorted in increasing order.

    Raises
    ------
    TypeError
        If n_bootstraps_grid is not an iterable of candidate counts.
    ValueError
        If the grid is empty or any count is not a positive integer.
    """
    if isinstance(n_bootstraps_grid, (str, bytes)):
        raise TypeError('n_bootstraps_grid must be an iterable of positive integers.')

    try:
        grid = list(n_bootstraps_grid)
    except TypeError as error:
        raise TypeError(
            'n_bootstraps_grid must be an iterable of positive integers.'
        ) from error

    if not grid:
        raise ValueError('n_bootstraps_grid must not be empty.')

    for n_bootstraps in grid:
        _validate_positive_integer(n_bootstraps, 'n_bootstraps_grid values')

    return tuple(sorted(set(grid)))


def _validate_tolerances(
        tolerances: Mapping[str, float] | None,
        name: str,
) -> dict[str, float]:
    """
    Validate and normalize per-metric Monte Carlo tolerances.

    Parameters
    ----------
    tolerances : Mapping[str, float] | None
        Optional mapping from metric names to non-negative finite tolerances.
    name : str
        Argument name used in validation messages.

    Returns
    -------
    dict[str, float]
        Copy of the validated tolerance mapping, or an empty dictionary when
        tolerances is None.

    Raises
    ------
    TypeError
        If tolerances is not a mapping or a tolerance is not numeric.
    ValueError
        If a metric name is empty, or a tolerance is non-finite or negative.
    """
    if tolerances is None:
        return {}

    if not isinstance(tolerances, Mapping):
        raise TypeError(f'{name} must be a mapping from metric names to tolerances.')

    validated_tolerances = {}

    for metric_name, tolerance in tolerances.items():
        if not isinstance(metric_name, str) or not metric_name:
            raise ValueError(f'{name} metric names must be non-empty strings.')

        if isinstance(tolerance, bool):
            raise TypeError(f'{name} values must be numeric.')

        try:
            numeric_tolerance = float(tolerance)
        except (TypeError, ValueError) as error:
            raise TypeError(f'{name} values must be numeric.') from error

        if not np.isfinite(numeric_tolerance) or numeric_tolerance < 0:
            raise ValueError(f'{name} values must be finite and non-negative.')

        validated_tolerances[metric_name] = numeric_tolerance

    return validated_tolerances


def _generate_run_random_states(
        random_state: int | None,
        n_runs: int,
) -> tuple[int, ...]:
    """
    Generate reproducible independent child random states for bootstrap runs.

    Parameters
    ----------
    random_state : int | None
        Master random state. An integer makes all child streams reproducible.
    n_runs : int
        Number of independent bootstrap runs to seed. It must be at least two.

    Returns
    -------
    tuple[int, ...]
        One deterministic child random state for each independent run.

    Raises
    ------
    TypeError
        If random_state is neither an integer nor None.
    ValueError
        If n_runs is not an integer of at least two.
    """
    _validate_positive_integer(n_runs, 'n_runs')

    if n_runs < 2:
        raise ValueError('n_runs must be at least two to measure dispersion.')

    if random_state is not None and (
            not isinstance(random_state, (int, np.integer))
            or isinstance(random_state, bool)
    ):
        raise TypeError('random_state must be an integer or None.')

    seed_sequence = np.random.SeedSequence(random_state)
    return tuple(
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(n_runs)
    )


def _bootstrap_ids_if_present(bootstrap_results: Any) -> Any | None:
    """
    Extract bootstrap identifiers when a supported result format exposes them.

    Parameters
    ----------
    bootstrap_results : Any
        Sized bootstrap replicate collection returned by bootstrap_fn.

    Returns
    -------
    Any | None
        Identifier values when a bootstrap_id field is present, otherwise None.

    Raises
    ------
    ValueError
        If a sequence of mapping replicates exposes bootstrap_id only for some
        replicates.
    """
    if isinstance(bootstrap_results, pd.DataFrame):
        return (
            bootstrap_results['bootstrap_id'].to_numpy()
            if 'bootstrap_id' in bootstrap_results.columns
            else None
        )

    if isinstance(bootstrap_results, pl.DataFrame):
        return (
            bootstrap_results['bootstrap_id'].to_numpy()
            if 'bootstrap_id' in bootstrap_results.columns
            else None
        )

    if (
            isinstance(bootstrap_results, Sequence)
            and len(bootstrap_results) > 0
            and isinstance(bootstrap_results[0], Mapping)
            and 'bootstrap_id' in bootstrap_results[0]
    ):
        if any('bootstrap_id' not in replicate for replicate in bootstrap_results):
            raise ValueError(
                'bootstrap_id must be present for every mapping bootstrap replicate.'
            )

        return [replicate['bootstrap_id'] for replicate in bootstrap_results]

    return None


def _validate_bootstrap_results(
        bootstrap_results: Any,
        n_bootstraps: int,
) -> None:
    """
    Validate a bootstrap result collection without modifying its replicate order.

    Parameters
    ----------
    bootstrap_results : Any
        Result collection returned by bootstrap_fn.
    n_bootstraps : int
        Expected number of bootstrap replicates in the collection.

    Returns
    -------
    None
        This function returns None when the result count is correct and any
        exposed bootstrap_id values are the deterministic sequence 1 through B.

    Raises
    ------
    TypeError
        If bootstrap_results is not a sized replicate collection.
    ValueError
        If its replicate count differs from n_bootstraps, or exposed
        bootstrap_id values are invalid or reordered.
    """
    if isinstance(bootstrap_results, (str, bytes, Mapping)):
        raise TypeError('bootstrap_fn must return a sized collection of bootstrap replicates.')

    try:
        result_length = len(bootstrap_results)
    except TypeError as error:
        raise TypeError(
            'bootstrap_fn must return a sized collection of bootstrap replicates.'
        ) from error

    if result_length != n_bootstraps:
        raise ValueError(
            'bootstrap_fn must return exactly n_bootstraps bootstrap replicates.'
        )

    bootstrap_ids = _bootstrap_ids_if_present(bootstrap_results)

    if bootstrap_ids is None:
        return

    try:
        ids = np.asarray(bootstrap_ids, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('bootstrap_id values must be finite consecutive integers.') from error

    expected_ids = np.arange(1, n_bootstraps + 1)

    if (
            ids.ndim != 1
            or len(ids) != n_bootstraps
            or not np.isfinite(ids).all()
            or not np.equal(ids, np.floor(ids)).all()
            or not np.array_equal(ids.astype(int), expected_ids)
    ):
        raise ValueError(
            'bootstrap_id values must be the ordered sequence from 1 to n_bootstraps.'
        )


def _select_bootstrap_prefix(
        bootstrap_results: Any,
        n_bootstraps: int,
) -> Any:
    """
    Select an ordered bootstrap prefix without changing replicate identities.

    Parameters
    ----------
    bootstrap_results : Any
        Full bootstrap result collection previously validated at the maximum B.
    n_bootstraps : int
        Number of leading replicates to retain for one candidate grid level.

    Returns
    -------
    Any
        First n_bootstraps replicates in the original result format.

    Raises
    ------
    TypeError
        If bootstrap_results does not support prefix selection.
    ValueError
        If the selected prefix does not contain exactly n_bootstraps ordered
        replicates or exposes invalid bootstrap_id values.

    Notes
    -----
    Pandas and Polars DataFrames use row prefixes. Other supported result
    collections must support slice notation. This helper never sorts, samples,
    or otherwise reorganizes bootstrap replicates.
    """
    if isinstance(bootstrap_results, (pd.Series, pd.DataFrame)):
        prefix = bootstrap_results.iloc[:n_bootstraps]
    elif isinstance(bootstrap_results, (pl.Series, pl.DataFrame)):
        prefix = bootstrap_results.slice(0, n_bootstraps)
    else:
        try:
            prefix = bootstrap_results[:n_bootstraps]
        except (TypeError, KeyError, IndexError) as error:
            raise TypeError(
                'bootstrap_fn results must support ordered prefix selection.'
            ) from error

    _validate_bootstrap_results(prefix, n_bootstraps)
    return prefix


def _extract_monitored_metrics(
        metrics_fn: Callable[[Any], Mapping[str, float]],
        bootstrap_results: Any,
        metric_names: tuple[str, ...],
) -> dict[str, float]:
    """
    Extract finite monitored metrics from one bootstrap run.

    Parameters
    ----------
    metrics_fn : Callable[[Any], Mapping[str, float]]
        Callable that summarizes bootstrap results into named metrics.
    bootstrap_results : Any
        Unmodified results returned by a bootstrap run.
    metric_names : tuple[str, ...]
        Configured metric names that metrics_fn must provide.

    Returns
    -------
    dict[str, float]
        Finite numeric values for every configured metric.

    Raises
    ------
    TypeError
        If metrics_fn does not return a mapping or a metric value is not numeric.
    KeyError
        If a configured metric is absent from metrics_fn output.
    ValueError
        If a metric value is non-finite.
    """
    metrics = metrics_fn(bootstrap_results)

    if not isinstance(metrics, Mapping):
        raise TypeError('metrics_fn must return a mapping from metric names to values.')

    extracted_metrics = {}

    for metric_name in metric_names:
        if metric_name not in metrics:
            raise KeyError(f'metrics_fn output is missing configured metric: {metric_name}.')

        metric_value = metrics[metric_name]

        if isinstance(metric_value, bool):
            raise TypeError('metrics_fn values must be numeric.')

        try:
            numeric_value = float(metric_value)
        except (TypeError, ValueError) as error:
            raise TypeError('metrics_fn values must be numeric.') from error

        if not np.isfinite(numeric_value):
            raise ValueError('metrics_fn values must be finite.')

        extracted_metrics[metric_name] = numeric_value

    return extracted_metrics


def _summarize_metric_values(
        values: Sequence[float],
        absolute_tolerance: float | None,
        relative_tolerance: float | None,
) -> MonteCarloMetricDiagnostics:
    """
    Summarize between-run variation and evaluate metric stability criteria.

    Parameters
    ----------
    values : Sequence[float]
        Finite values of one metric across independent bootstrap runs.
    absolute_tolerance : float | None
        Maximum permitted between-run range, if configured.
    relative_tolerance : float | None
        Maximum permitted range divided by abs(mean), if configured.

    Returns
    -------
    MonteCarloMetricDiagnostics
        Between-run summary statistics and criterion outcomes.

    Raises
    ------
    ValueError
        If values does not contain at least two finite numeric observations.

    Notes
    -----
    Relative range is not calculated when abs(mean) is at most machine epsilon.
    A configured relative criterion then fails rather than using an unstable
    near-zero denominator. When both criteria are configured, both must pass.
    """
    try:
        values_array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('metric values must be finite numeric values.') from error

    if values_array.ndim != 1 or len(values_array) < 2 or not np.isfinite(values_array).all():
        raise ValueError('metric values must contain at least two finite numeric values.')

    mean = float(np.mean(values_array))
    minimum = float(np.min(values_array))
    maximum = float(np.max(values_array))
    absolute_range = float(maximum - minimum)

    if abs(mean) > np.finfo(float).eps:
        relative_range = float(absolute_range / abs(mean))
    else:
        relative_range = None

    absolute_passed = (
        absolute_range <= absolute_tolerance
        if absolute_tolerance is not None
        else None
    )
    relative_passed = (
        relative_range is not None and relative_range <= relative_tolerance
        if relative_tolerance is not None
        else None
    )
    passed = bool(
        (absolute_passed is None or absolute_passed)
        and (relative_passed is None or relative_passed)
    )

    return MonteCarloMetricDiagnostics(
        mean=mean,
        std=float(np.std(values_array, ddof=1)),
        minimum=minimum,
        maximum=maximum,
        absolute_range=absolute_range,
        relative_range=relative_range,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_passed=absolute_passed,
        relative_passed=relative_passed,
        passed=passed,
    )


def monte_carlo_error(
        bootstrap_fn: Callable[[int, int], Any],
        metrics_fn: Callable[[Any], Mapping[str, float]],
        n_bootstraps_grid: Iterable[int],
        absolute_tolerances: Mapping[str, float] | None = None,
        relative_tolerances: Mapping[str, float] | None = None,
        n_runs: int = 5,
        random_state: int | None = None,
        return_bootstrap_results: bool = False,
) -> MonteCarloErrorResult:
    """
    Evaluate bootstrap Monte Carlo stability across candidate replication counts.

    This function measures Monte Carlo error: variation in derived metrics
    caused by a finite number of bootstrap replications. It does not estimate
    sampling uncertainty and is independent of the bootstrap algorithm,
    research statistic, or data domain.

    bootstrap_fn must accept n_bootstraps and random_state keyword arguments,
    return exactly that many replicates in deterministic order, and be prefix
    consistent. For a fixed random state, requesting a larger B must preserve
    every earlier replicate in the same order. If bootstrap results expose a
    bootstrap_id field, this function validates that it is the ordered sequence
    1 through B, but it never adds, sorts, or otherwise reorganizes results.

    The function generates the largest candidate B once for every independent
    random-number stream, then evaluates ordered prefixes at lower B levels.
    This avoids rerunning bootstrap generation as the grid advances.

    Workflow
    --------
    Consider n_bootstraps_grid=[1000, 2000, 4000, 8000] and n_runs=5.

    1. _normalize_n_bootstraps_grid() validates, deduplicates, and orders the
       candidate counts. The largest value, 8000 in this example, is the only
       count passed to bootstrap_fn.

    2. _validate_tolerances() validates absolute_tolerances and
       relative_tolerances. Their combined metric names define the values that
       metrics_fn must provide for every run.

    3. _generate_run_random_states() derives five reproducible independent
       child states from random_state. bootstrap_fn is called once for each
       state with n_bootstraps=8000.

    4. _validate_bootstrap_results() validates each full result collection. If
       the collection exposes bootstrap_id, it calls _bootstrap_ids_if_present()
       and verifies that identifiers are the ordered sequence 1 through 8000.

    5. For B=1000, then B=2000, and so on, _select_bootstrap_prefix() selects
       replicates 1:B from each full run without sorting or resampling. It
       validates the selected prefix again, preserving deterministic replicate
       identities across all candidate levels.

    6. _extract_monitored_metrics() calls metrics_fn on every run prefix and
       validates the configured finite metric values. For one metric, its five
       run values are passed to _summarize_metric_values(), which creates a
       MonteCarloMetricDiagnostics with between-run dispersion and tolerance
       outcomes.

    7. Those per-metric diagnostics form a MonteCarloLevelDiagnostics for the
       current B. When all monitored metrics pass for two consecutive levels,
       the second level is marked convergence_confirmed and this function
       returns a MonteCarloErrorResult. Its recommended_n_bootstraps is the
       first level of the passing pair. If requested, the result retains only
       reference run 1's prefix at that recommended count; it never combines
       different random-number streams.

    Each candidate B is evaluated across independent random-number streams.
    Metrics are summarized between runs, and every monitored metric must pass
    its configured criterion at two consecutive B levels. The recommended
    number is the first level in that confirmed pair: the smallest evaluated
    number of bootstrap replications satisfying the configured Monte Carlo
    stability criteria. It is not an "optimal" bootstrap count.

    Parameters
    ----------
    bootstrap_fn : Callable[[int, int], Any]
        Bootstrap generator called once per run as
        bootstrap_fn(n_bootstraps=max(n_bootstraps_grid),
        random_state=run_random_state). Bootstrap-specific arguments should be
        bound beforehand with functools.partial, a closure, or an equivalent.
    metrics_fn : Callable[[Any], Mapping[str, float]]
        Callable converting one run's bootstrap results into named finite
        metrics. The function does not calculate bootstrap metrics itself.
    n_bootstraps_grid : Iterable[int]
        Candidate bootstrap counts. Values are deduplicated and evaluated as
        increasing ordered prefixes until convergence is confirmed or the grid
        is exhausted. Doubling grids are a typical use case.
    absolute_tolerances : Mapping[str, float] | None, default None
        Per-metric maximum permitted between-run range.
    relative_tolerances : Mapping[str, float] | None, default None
        Per-metric maximum permitted range divided by abs(metric mean). When
        the mean is at most machine epsilon in magnitude, relative range is
        undefined and a configured relative criterion fails.
    n_runs : int, default 5
        Number of independent bootstrap random-number streams. At least two
        runs are required to measure between-run dispersion.
    random_state : int | None, default None
        Master random state from which deterministic independent child run
        states are generated.
    return_bootstrap_results : bool, default False
        If True and convergence is confirmed, retain the ordered prefix from
        deterministic reference run 1 at the recommended B.
        Results from different runs are never concatenated.

    Returns
    -------
    MonteCarloErrorResult
        Recommended bootstrap count, convergence flag, diagnostics for
        evaluated levels, child run states, and optional reference-run results.

    Raises
    ------
    TypeError
        If a callable, tolerance mapping, random state, bootstrap result, or
        metrics output uses an unsupported type.
    ValueError
        If the grid, run count, tolerances, bootstrap results, or metric values
        are invalid.
    KeyError
        If metrics_fn omits a configured metric.
    """
    if not callable(bootstrap_fn):
        raise TypeError('bootstrap_fn must be callable.')

    if not callable(metrics_fn):
        raise TypeError('metrics_fn must be callable.')

    if not isinstance(return_bootstrap_results, bool):
        raise TypeError('return_bootstrap_results must be a boolean.')

    grid = _normalize_n_bootstraps_grid(n_bootstraps_grid)
    validated_absolute_tolerances = _validate_tolerances(
        absolute_tolerances,
        'absolute_tolerances',
    )
    validated_relative_tolerances = _validate_tolerances(
        relative_tolerances,
        'relative_tolerances',
    )
    metric_names = tuple(dict.fromkeys(
        [*validated_absolute_tolerances, *validated_relative_tolerances]
    ))

    if not metric_names:
        raise ValueError('at least one metric tolerance must be configured.')

    run_random_states = _generate_run_random_states(random_state, n_runs)
    maximum_n_bootstraps = grid[-1]
    bootstrap_results_by_run = []

    for run_random_state in run_random_states:
        bootstrap_results = bootstrap_fn(
            n_bootstraps=maximum_n_bootstraps,
            random_state=run_random_state,
        )
        _validate_bootstrap_results(bootstrap_results, maximum_n_bootstraps)
        bootstrap_results_by_run.append(bootstrap_results)

    diagnostics = []
    previous_level_passed = False
    pending_n_bootstraps = None

    for n_bootstraps in grid:
        metric_values = {metric_name: [] for metric_name in metric_names}

        for bootstrap_results in bootstrap_results_by_run:
            bootstrap_prefix = _select_bootstrap_prefix(
                bootstrap_results,
                n_bootstraps,
            )
            run_metrics = _extract_monitored_metrics(
                metrics_fn,
                bootstrap_prefix,
                metric_names,
            )

            for metric_name, metric_value in run_metrics.items():
                metric_values[metric_name].append(metric_value)

        metric_diagnostics = {
            metric_name: _summarize_metric_values(
                metric_values[metric_name],
                validated_absolute_tolerances.get(metric_name),
                validated_relative_tolerances.get(metric_name),
            )
            for metric_name in metric_names
        }
        all_metrics_passed = all(
            metric_diagnostic.passed
            for metric_diagnostic in metric_diagnostics.values()
        )
        level_diagnostic = MonteCarloLevelDiagnostics(
            n_bootstraps=n_bootstraps,
            metrics=metric_diagnostics,
            all_metrics_passed=all_metrics_passed,
            convergence_confirmed=False,
        )
        diagnostics.append(level_diagnostic)

        if previous_level_passed and all_metrics_passed:
            diagnostics[-1] = replace(level_diagnostic, convergence_confirmed=True)

            return MonteCarloErrorResult(
                recommended_n_bootstraps=pending_n_bootstraps,
                converged=True,
                diagnostics=tuple(diagnostics),
                run_random_states=run_random_states,
                reference_run_id=1 if return_bootstrap_results else None,
                reference_bootstrap_results=(
                    _select_bootstrap_prefix(
                        bootstrap_results_by_run[0],
                        pending_n_bootstraps,
                    )
                    if return_bootstrap_results
                    else None
                ),
            )

        previous_level_passed = all_metrics_passed
        pending_n_bootstraps = n_bootstraps if all_metrics_passed else None

    return MonteCarloErrorResult(
        recommended_n_bootstraps=None,
        converged=False,
        diagnostics=tuple(diagnostics),
        run_random_states=run_random_states,
        reference_run_id=None,
        reference_bootstrap_results=None,
    )
