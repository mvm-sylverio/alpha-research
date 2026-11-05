from dataclasses import asdict
from functools import partial
from itertools import product
from math import fsum, sqrt
from statistics import mean, stdev

import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.resampling.block_bootstrap import (
    bootstrap_metrics,
    generate_moving_blocks,
    moving_block_bootstrap,
)
from alpha_research.resampling.convergence import (
    _bootstrap_ids_if_present,
    _extract_monitored_metrics,
    _generate_run_random_states,
    _normalize_n_bootstraps_grid,
    _select_bootstrap_prefix,
    _summarize_metric_values,
    _validate_bootstrap_results,
    _validate_tolerances,
    monte_carlo_error,
)


def test_convergence_progress_respects_early_confirmation():
    """Should count only two evaluated levels and preserve the original maximum-B generation contract."""
    calls = []
    prefixes = []
    events = []
    bootstrap = lambda n_bootstraps, random_state: calls.append((n_bootstraps, random_state)) or list(range(n_bootstraps))
    metrics = lambda values: prefixes.append(len(values)) or {'value': 1.0}
    result = monte_carlo_error(bootstrap, metrics, [2, 4, 8], {'value': 0.0}, n_runs=2,
                              random_state=17, on_progress=lambda *event: events.append(event))
    assert result.converged
    assert result.recommended_n_bootstraps == 2
    assert [count for count, _ in calls] == [8, 8]
    assert tuple(seed for _, seed in calls) == result.run_random_states
    assert prefixes == [2, 2, 4, 4]
    assert ('convergence_runs', 2, 2) in events
    assert events[-1] == ('convergence_levels', 2, 3)
    assert ('convergence_levels', 3, 3) not in events


@pytest.mark.parametrize('phase', ['convergence_runs', 'convergence_levels', 'convergence_metrics'])
@pytest.mark.parametrize('exception_type', [ValueError, TypeError, RuntimeError])
def test_convergence_propagates_callback_exceptions(phase, exception_type):
    """Should stop each coordination loop without retrying bootstrap or swallowing metric observation errors."""
    generated = []
    measured = []
    exception = exception_type('cancel')
    bootstrap = lambda n_bootstraps, random_state: generated.append(n_bootstraps) or list(range(n_bootstraps))
    metrics = lambda values: measured.append(len(values)) or {'value': 1.0}
    callback = lambda current, done, total: (_ for _ in ()).throw(exception) if current == phase and done == 1 else None
    with pytest.raises(exception_type) as caught:
        monte_carlo_error(bootstrap, metrics, [2, 4, 8], {'value': 0.0}, n_runs=2,
                          random_state=17, on_progress=callback)
    assert caught.value is exception
    if phase == 'convergence_runs':
        assert generated == [8]
        assert measured == []
    elif phase == 'convergence_levels':
        assert measured == [2, 2]
    else:
        assert measured == [2]


def _prefix_consistent_bootstrap(
        n_bootstraps: int,
        random_state: int,
) -> list[dict[str, float | int]]:
    """Generate simple deterministic bootstrap-like replicates for testing."""
    rng = np.random.default_rng(random_state)

    return [
        {
            'bootstrap_id': bootstrap_id,
            'statistic': float(rng.normal()),
            'run_random_state': random_state,
        }
        for bootstrap_id in range(1, n_bootstraps + 1)
    ]


def _mean_statistic_metric(
        bootstrap_results: list[dict[str, float | int]],
) -> dict[str, float]:
    """Return a simple metric derived from synthetic bootstrap replicates."""
    return {
        'bootstrap_se': float(np.mean([
            replicate['statistic'] for replicate in bootstrap_results
        ])),
    }


# ------------------------------------------------------
# _normalize_n_bootstraps_grid
# ------------------------------------------------------
def test_normalize_n_bootstraps_grid_sorts_and_deduplicates_values():
    """Should return unique candidate counts in increasing order."""
    result = _normalize_n_bootstraps_grid([4000, 1000, 2000, 2000])

    assert result == (1000, 2000, 4000)


def test_normalize_n_bootstraps_grid_rejects_empty_grid():
    """Should reject an empty candidate grid."""
    with pytest.raises(ValueError, match='must not be empty'):
        _normalize_n_bootstraps_grid([])


def test_normalize_n_bootstraps_grid_rejects_non_iterable_grid():
    """Should reject a non-iterable candidate grid."""
    with pytest.raises(TypeError, match='iterable'):
        _normalize_n_bootstraps_grid(1000)


@pytest.mark.parametrize('grid', [[0], [-1], [1.5], [True]])
def test_normalize_n_bootstraps_grid_rejects_invalid_counts(grid):
    """Should reject non-positive and noninteger bootstrap counts."""
    with pytest.raises(ValueError, match='positive integer'):
        _normalize_n_bootstraps_grid(grid)


# ------------------------------------------------------
# _validate_tolerances
# ------------------------------------------------------
def test_validate_tolerances_returns_validated_copy():
    """Should preserve configured finite non-negative tolerances."""
    tolerances = {'bootstrap_se': 0.001, 'p_value': 0.005}

    result = _validate_tolerances(tolerances, 'absolute_tolerances')

    assert result == tolerances
    assert result is not tolerances


def test_validate_tolerances_accepts_none_as_empty_mapping():
    """Should normalize an omitted tolerance mapping to an empty dictionary."""
    assert _validate_tolerances(None, 'relative_tolerances') == {}


def test_validate_tolerances_rejects_non_mapping():
    """Should reject tolerance inputs that are not mappings."""
    with pytest.raises(TypeError, match='must be a mapping'):
        _validate_tolerances([('bootstrap_se', 0.001)], 'absolute_tolerances')


@pytest.mark.parametrize('metric_name', ['', 1])
def test_validate_tolerances_rejects_invalid_metric_names(metric_name):
    """Should require non-empty string metric names."""
    with pytest.raises(ValueError, match='non-empty strings'):
        _validate_tolerances({metric_name: 0.001}, 'absolute_tolerances')


@pytest.mark.parametrize('tolerance', ['invalid', True])
def test_validate_tolerances_rejects_non_numeric_values(tolerance):
    """Should reject non-numeric tolerance values."""
    with pytest.raises(TypeError, match='must be numeric'):
        _validate_tolerances({'bootstrap_se': tolerance}, 'absolute_tolerances')


@pytest.mark.parametrize('tolerance', [-0.001, np.nan, np.inf])
def test_validate_tolerances_rejects_invalid_numeric_values(tolerance):
    """Should reject negative and non-finite tolerance values."""
    with pytest.raises(ValueError, match='finite and non-negative'):
        _validate_tolerances({'bootstrap_se': tolerance}, 'absolute_tolerances')


# ------------------------------------------------------
# _generate_run_random_states
# ------------------------------------------------------
def test_generate_run_random_states_is_reproducible_and_independent():
    """Should reproduce the same distinct child states from one master seed."""
    first = _generate_run_random_states(42, 5)
    second = _generate_run_random_states(42, 5)

    assert first == second
    assert len(set(first)) == 5


def test_generate_run_random_states_changes_with_master_seed():
    """Should derive different streams from different master seeds."""
    assert _generate_run_random_states(42, 3) != _generate_run_random_states(43, 3)


@pytest.mark.parametrize('n_runs', [0, -1, 1, 1.5, True])
def test_generate_run_random_states_rejects_insufficient_or_invalid_runs(n_runs):
    """Should require at least two independent integer runs."""
    with pytest.raises(ValueError):
        _generate_run_random_states(42, n_runs)


@pytest.mark.parametrize('random_state', ['42', 1.5, True])
def test_generate_run_random_states_rejects_invalid_random_state(random_state):
    """Should reject unsupported master random-state values."""
    with pytest.raises(TypeError, match='integer or None'):
        _generate_run_random_states(random_state, 2)


# ------------------------------------------------------
# _bootstrap_ids_if_present
# ------------------------------------------------------
def test_bootstrap_ids_if_present_extracts_dataframe_identifiers():
    """Should extract bootstrap identifiers from a supported dataframe result."""
    results = pd.DataFrame({'bootstrap_id': [1, 2], 'statistic': [0.1, 0.2]})

    assert _bootstrap_ids_if_present(results).tolist() == [1, 2]


def test_bootstrap_ids_if_present_returns_none_without_identifiers():
    """Should return None when results do not expose bootstrap identifiers."""
    assert _bootstrap_ids_if_present([0.1, 0.2]) is None


def test_bootstrap_ids_if_present_rejects_partial_mapping_identifiers():
    """Should reject mapping replicates when only some expose bootstrap_id."""
    results = [{'bootstrap_id': 1, 'statistic': 0.1}, {'statistic': 0.2}]

    with pytest.raises(ValueError, match='every mapping bootstrap replicate'):
        _bootstrap_ids_if_present(results)


# ------------------------------------------------------
# _validate_bootstrap_results
# ------------------------------------------------------
def test_validate_bootstrap_results_accepts_ordered_bootstrap_identifiers():
    """Should accept a correctly sized replicate list with ordered identifiers."""
    _validate_bootstrap_results(_prefix_consistent_bootstrap(3, 42), 3)


def test_validate_bootstrap_results_rejects_unsupported_collection_type():
    """Should reject results that are not sized replicate collections."""
    with pytest.raises(TypeError, match='sized collection'):
        _validate_bootstrap_results({'bootstrap_id': [1, 2]}, 2)


def test_validate_bootstrap_results_rejects_wrong_replicate_count():
    """Should require exactly the requested number of bootstrap replicates."""
    with pytest.raises(ValueError, match='exactly n_bootstraps'):
        _validate_bootstrap_results([0.1, 0.2], 3)


@pytest.mark.parametrize('bootstrap_ids', [[2, 1], [1, 3], [1.5, 2.0]])
def test_validate_bootstrap_results_rejects_invalid_bootstrap_identifiers(bootstrap_ids):
    """Should reject bootstrap identifiers that are reordered or non-consecutive."""
    results = [
        {'bootstrap_id': bootstrap_id, 'statistic': 0.0}
        for bootstrap_id in bootstrap_ids
    ]

    with pytest.raises(ValueError, match='ordered sequence'):
        _validate_bootstrap_results(results, 2)


# ------------------------------------------------------
# _select_bootstrap_prefix
# ------------------------------------------------------
def test_select_bootstrap_prefix_preserves_order_and_bootstrap_identifiers():
    """Should return the first requested replicates without reordering them."""
    results = _prefix_consistent_bootstrap(5, 42)

    prefix = _select_bootstrap_prefix(results, 3)

    assert prefix == results[:3]
    assert [replicate['bootstrap_id'] for replicate in prefix] == [1, 2, 3]


def test_select_bootstrap_prefix_preserves_dataframe_row_order():
    """Should use the leading dataframe rows as the bootstrap prefix."""
    results = pd.DataFrame({
        'bootstrap_id': [1, 2, 3],
        'statistic': [0.1, 0.2, 0.3],
    })

    prefix = _select_bootstrap_prefix(results, 2)

    assert prefix['bootstrap_id'].to_list() == [1, 2]


def test_select_bootstrap_prefix_rejects_non_prefixable_results():
    """Should reject a result collection that cannot produce an ordered prefix."""
    with pytest.raises(TypeError, match='ordered prefix selection'):
        _select_bootstrap_prefix({1, 2, 3}, 2)


# ------------------------------------------------------
# _extract_monitored_metrics
# ------------------------------------------------------
def test_extract_monitored_metrics_returns_configured_finite_values():
    """Should select configured finite numeric metrics from metrics_fn output."""
    result = _extract_monitored_metrics(
        lambda _: {'bootstrap_se': 0.01, 'unused': 1.0},
        [0.1, 0.2],
        ('bootstrap_se',),
    )

    assert result == {'bootstrap_se': 0.01}


def test_extract_monitored_metrics_rejects_non_mapping_output():
    """Should require metrics_fn to return a mapping."""
    with pytest.raises(TypeError, match='must return a mapping'):
        _extract_monitored_metrics(lambda _: [0.01], [0.1], ('bootstrap_se',))


def test_extract_monitored_metrics_rejects_missing_metric():
    """Should reject metrics_fn output without a configured metric."""
    with pytest.raises(KeyError, match='missing configured metric'):
        _extract_monitored_metrics(lambda _: {}, [0.1], ('bootstrap_se',))


@pytest.mark.parametrize('metric_value', ['invalid', True])
def test_extract_monitored_metrics_rejects_non_numeric_values(metric_value):
    """Should reject metrics that cannot represent numeric estimates."""
    with pytest.raises(TypeError, match='values must be numeric'):
        _extract_monitored_metrics(
            lambda _: {'bootstrap_se': metric_value},
            [0.1],
            ('bootstrap_se',),
        )


@pytest.mark.parametrize('metric_value', [np.nan, np.inf, -np.inf])
def test_extract_monitored_metrics_rejects_non_finite_values(metric_value):
    """Should reject non-finite derived bootstrap metrics."""
    with pytest.raises(ValueError, match='values must be finite'):
        _extract_monitored_metrics(
            lambda _: {'bootstrap_se': metric_value},
            [0.1],
            ('bootstrap_se',),
        )


# ------------------------------------------------------
# _summarize_metric_values
# ------------------------------------------------------
def test_summarize_metric_values_calculates_between_run_statistics():
    """Should calculate mean, sample standard deviation, extrema, and range."""
    result = _summarize_metric_values([1.0, 2.0, 3.0], 2.0, None)

    assert result.mean == pytest.approx(2.0)
    assert result.std == pytest.approx(1.0)
    assert result.minimum == pytest.approx(1.0)
    assert result.maximum == pytest.approx(3.0)
    assert result.absolute_range == pytest.approx(2.0)
    assert result.absolute_passed is True
    assert result.relative_passed is None
    assert result.passed is True


def test_summarize_metric_values_applies_relative_tolerance():
    """Should compare relative range against the configured relative tolerance."""
    result = _summarize_metric_values([100.0, 102.0], None, 0.02)

    assert result.relative_range == pytest.approx(2 / 101)
    assert result.relative_passed is True
    assert result.passed is True


def test_summarize_metric_values_requires_both_configured_criteria():
    """Should fail when either absolute or relative criterion fails."""
    result = _summarize_metric_values([100.0, 102.0], 1.0, 0.03)

    assert result.absolute_passed is False
    assert result.relative_passed is True
    assert result.passed is False


def test_summarize_metric_values_handles_near_zero_relative_reference_safely():
    """Should not divide by a zero or machine-near-zero metric mean."""
    result = _summarize_metric_values([0.0, 0.0], None, 0.02)

    assert result.relative_range is None
    assert result.relative_passed is False
    assert result.passed is False


@pytest.mark.parametrize('values', [[1.0], [1.0, np.nan], [[1.0], [2.0]]])
def test_summarize_metric_values_rejects_invalid_values(values):
    """Should require at least two finite one-dimensional metric values."""
    with pytest.raises(ValueError, match='at least two finite'):
        _summarize_metric_values(values, 0.01, None)


# ------------------------------------------------------
# monte_carlo_error
# ------------------------------------------------------
def test_monte_carlo_error_is_reproducible_for_same_random_state():
    """Should return exactly the same diagnostics and reference run for one seed."""
    kwargs = {
        'bootstrap_fn': _prefix_consistent_bootstrap,
        'metrics_fn': _mean_statistic_metric,
        'n_bootstraps_grid': [10, 20],
        'absolute_tolerances': {'bootstrap_se': 10.0},
        'random_state': 42,
        'return_bootstrap_results': True,
    }

    first = monte_carlo_error(**kwargs)
    second = monte_carlo_error(**kwargs)

    assert first == second


def test_monte_carlo_error_changes_replicates_with_different_random_states():
    """Should produce different reference-run replicates for a different master seed."""
    kwargs = {
        'bootstrap_fn': _prefix_consistent_bootstrap,
        'metrics_fn': _mean_statistic_metric,
        'n_bootstraps_grid': [10, 20],
        'absolute_tolerances': {'bootstrap_se': 10.0},
        'return_bootstrap_results': True,
    }

    first = monte_carlo_error(random_state=42, **kwargs)
    second = monte_carlo_error(random_state=43, **kwargs)

    assert first.reference_bootstrap_results != second.reference_bootstrap_results


def test_monte_carlo_error_generates_maximum_b_once_per_run_and_evaluates_prefixes():
    """Should generate B_max once per run before evaluating every needed prefix."""
    bootstrap_calls = []
    metric_calls = []

    def bootstrap_fn(n_bootstraps, random_state):
        bootstrap_calls.append((n_bootstraps, random_state))
        return _prefix_consistent_bootstrap(n_bootstraps, random_state)

    def metrics_fn(results):
        metric_calls.append(len(results))
        return {'metric': 1.0}

    result = monte_carlo_error(
        bootstrap_fn,
        metrics_fn,
        [10, 20],
        absolute_tolerances={'metric': 0.0},
        n_runs=3,
        random_state=42,
    )

    assert result.converged is True
    assert [n_bootstraps for n_bootstraps, _ in bootstrap_calls] == [20, 20, 20]
    assert len({seed for _, seed in bootstrap_calls}) == 3
    assert metric_calls == [10, 10, 10, 20, 20, 20]


def test_monte_carlo_error_evaluates_grid_progressively_and_deduplicates_values():
    """Should evaluate failing prefixes once in increasing deduplicated order."""
    bootstrap_calls = []
    metric_calls = []

    def bootstrap_fn(n_bootstraps, random_state):
        bootstrap_calls.append((n_bootstraps, random_state))
        return _prefix_consistent_bootstrap(n_bootstraps, random_state)

    def metrics_fn(results):
        metric_calls.append(len(results))
        return {'metric': float(results[0]['run_random_state'])}

    result = monte_carlo_error(
        bootstrap_fn,
        metrics_fn,
        [40, 10, 20, 20],
        absolute_tolerances={'metric': 0.0},
        n_runs=2,
        random_state=42,
    )

    assert result.converged is False
    assert [n_bootstraps for n_bootstraps, _ in bootstrap_calls] == [40, 40]
    assert metric_calls == [10, 10, 20, 20, 40, 40]


def test_monte_carlo_error_stops_after_two_consecutive_passing_levels():
    """Should stop diagnostics after confirmation while B_max was generated once."""
    bootstrap_calls = []
    metric_calls = []

    def bootstrap_fn(n_bootstraps, random_state):
        bootstrap_calls.append(n_bootstraps)
        return _prefix_consistent_bootstrap(n_bootstraps, random_state)

    def metrics_fn(results):
        metric_calls.append(len(results))
        if len(results) == 1000:
            return {'metric': float(results[0]['run_random_state'])}

        return {'metric': 1.0}

    result = monte_carlo_error(
        bootstrap_fn,
        metrics_fn,
        [1000, 2000, 4000, 8000],
        absolute_tolerances={'metric': 0.0},
        n_runs=2,
        random_state=42,
    )

    assert result.recommended_n_bootstraps == 2000
    assert result.converged is True
    assert [level.n_bootstraps for level in result.diagnostics] == [1000, 2000, 4000]
    assert bootstrap_calls == [8000, 8000]
    assert 8000 not in metric_calls
    assert result.diagnostics[-1].convergence_confirmed is True


def test_monte_carlo_error_requires_every_metric_to_pass():
    """Should fail a level when one configured metric remains unstable."""
    result = monte_carlo_error(
        _prefix_consistent_bootstrap,
        lambda results: {
            'stable_metric': 1.0,
            'unstable_metric': float(results[0]['run_random_state']),
        },
        [10, 20],
        absolute_tolerances={
            'stable_metric': 0.0,
            'unstable_metric': 0.0,
        },
        n_runs=2,
        random_state=42,
    )

    assert result.converged is False
    assert result.diagnostics[0].metrics['stable_metric'].passed is True
    assert result.diagnostics[0].metrics['unstable_metric'].passed is False
    assert result.diagnostics[0].all_metrics_passed is False


def test_monte_carlo_error_does_not_confirm_an_isolated_passing_level():
    """Should not converge when passing levels are separated by a failure."""
    def metrics_fn(results):
        if len(results) in {20, 80}:
            return {'metric': 1.0}

        return {'metric': float(results[0]['run_random_state'])}

    result = monte_carlo_error(
        _prefix_consistent_bootstrap,
        metrics_fn,
        [10, 20, 40, 80],
        absolute_tolerances={'metric': 0.0},
        n_runs=2,
        random_state=42,
    )

    assert result.converged is False
    assert result.recommended_n_bootstraps is None


def test_monte_carlo_error_retains_uncombined_reference_run_at_recommended_level():
    """Should retain only run 1 results at the recommended bootstrap count."""
    result = monte_carlo_error(
        _prefix_consistent_bootstrap,
        lambda _: {'metric': 1.0},
        [10, 20, 40],
        absolute_tolerances={'metric': 0.0},
        n_runs=3,
        random_state=42,
        return_bootstrap_results=True,
    )

    assert result.recommended_n_bootstraps == 10
    assert result.reference_run_id == 1
    assert len(result.reference_bootstrap_results) == 10
    assert [
        replicate['bootstrap_id']
        for replicate in result.reference_bootstrap_results
    ] == list(range(1, 11))


def test_monte_carlo_error_returns_no_reference_results_when_not_requested():
    """Should not return reference bootstrap results unless explicitly requested."""
    result = monte_carlo_error(
        _prefix_consistent_bootstrap,
        lambda _: {'metric': 1.0},
        [10, 20],
        absolute_tolerances={'metric': 0.0},
        random_state=42,
    )

    assert result.reference_run_id is None
    assert result.reference_bootstrap_results is None


def test_monte_carlo_error_returns_not_converged_when_grid_is_exhausted():
    """Should not recommend a count when no consecutive levels pass."""
    result = monte_carlo_error(
        _prefix_consistent_bootstrap,
        lambda results: {'metric': float(results[0]['run_random_state'])},
        [10, 20],
        absolute_tolerances={'metric': 0.0},
        n_runs=2,
        random_state=42,
    )

    assert result.recommended_n_bootstraps is None
    assert result.converged is False
    assert result.reference_bootstrap_results is None


def test_monte_carlo_error_rejects_non_callable_arguments():
    """Should reject non-callable bootstrap and metric callables."""
    with pytest.raises(TypeError, match='bootstrap_fn must be callable'):
        monte_carlo_error(None, lambda _: {'metric': 1.0}, [10], {'metric': 0.0})

    with pytest.raises(TypeError, match='metrics_fn must be callable'):
        monte_carlo_error(_prefix_consistent_bootstrap, None, [10], {'metric': 0.0})


def test_monte_carlo_error_rejects_non_boolean_result_retention_flag():
    """Should require an explicit boolean for bootstrap-result retention."""
    with pytest.raises(TypeError, match='must be a boolean'):
        monte_carlo_error(
            _prefix_consistent_bootstrap,
            lambda _: {'metric': 1.0},
            [10],
            absolute_tolerances={'metric': 0.0},
            n_runs=2,
            random_state=42,
            return_bootstrap_results=1,
        )


def test_monte_carlo_error_rejects_missing_metric_configuration():
    """Should require at least one absolute or relative metric tolerance."""
    with pytest.raises(ValueError, match='at least one metric tolerance'):
        monte_carlo_error(
            _prefix_consistent_bootstrap,
            _mean_statistic_metric,
            [10],
            n_runs=2,
            random_state=42,
        )


def test_monte_carlo_error_rejects_missing_metrics_fn_value():
    """Should reject a configured metric absent from metrics_fn output."""
    with pytest.raises(KeyError, match='missing configured metric'):
        monte_carlo_error(
            _prefix_consistent_bootstrap,
            lambda _: {},
            [10],
            absolute_tolerances={'bootstrap_se': 0.01},
            n_runs=2,
            random_state=42,
        )


@pytest.mark.parametrize('metric_value', [np.nan, np.inf])
def test_monte_carlo_error_rejects_non_finite_metrics(metric_value):
    """Should reject non-finite values returned by metrics_fn."""
    with pytest.raises(ValueError, match='metrics_fn values must be finite'):
        monte_carlo_error(
            _prefix_consistent_bootstrap,
            lambda _: {'bootstrap_se': metric_value},
            [10],
            absolute_tolerances={'bootstrap_se': 0.01},
            n_runs=2,
            random_state=42,
        )


def test_monte_carlo_error_rejects_incompatible_bootstrap_results():
    """Should reject bootstrap functions returning the wrong replicate count."""
    with pytest.raises(ValueError, match='exactly n_bootstraps'):
        monte_carlo_error(
            lambda n_bootstraps, random_state: [0.0] * (n_bootstraps - 1),
            lambda _: {'metric': 1.0},
            [10],
            absolute_tolerances={'metric': 0.0},
            n_runs=2,
            random_state=42,
        )


def test_monte_carlo_error_rejects_non_prefixable_bootstrap_results():
    """Should require B_max results to support ordered prefix selection."""
    with pytest.raises(TypeError, match='ordered prefix selection'):
        monte_carlo_error(
            lambda n_bootstraps, random_state: set(range(n_bootstraps)),
            lambda _: {'metric': 1.0},
            [10],
            absolute_tolerances={'metric': 0.0},
            n_runs=2,
            random_state=42,
        )


@pytest.mark.parametrize(
    ('n_bootstraps_grid', 'absolute_tolerances', 'n_runs', 'random_state'),
    [
        ([], {'metric': 0.0}, 2, 42),
        ([0], {'metric': 0.0}, 2, 42),
        ([10], {'metric': -0.01}, 2, 42),
        ([10], {'metric': 0.0}, 1, 42),
        ([10], {'metric': 0.0}, 2, 'invalid'),
    ],
)
def test_monte_carlo_error_propagates_core_argument_validation(
        n_bootstraps_grid,
        absolute_tolerances,
        n_runs,
        random_state,
):
    """Should reject invalid grid, tolerance, run-count, and seed arguments."""
    with pytest.raises((TypeError, ValueError)):
        monte_carlo_error(
            _prefix_consistent_bootstrap,
            lambda _: {'metric': 1.0},
            n_bootstraps_grid,
            absolute_tolerances=absolute_tolerances,
            n_runs=n_runs,
            random_state=random_state,
        )


def test_prefix_consistent_bootstrap_preserves_prefix_and_bootstrap_identifiers():
    """Should preserve every replicate and bootstrap_id when B increases for one seed."""
    small = _prefix_consistent_bootstrap(n_bootstraps=2000, random_state=42)
    large = _prefix_consistent_bootstrap(n_bootstraps=10000, random_state=42)

    assert small == large[:2000]
    assert [replicate['bootstrap_id'] for replicate in small] == list(range(1, 2001))
    assert large[1749] == small[1749]
    assert [replicate['bootstrap_id'] for replicate in large] == list(range(1, 10001))


# ------------------------------------------------------
# Scalar, identity, and numerical boundary contracts
# ------------------------------------------------------
@pytest.mark.parametrize('value', ['0.1', np.bool_(True), np.bool_(False), 1j,
                                   [0.1], np.array([0.1]), np.array(0.1)])
def test_convergence_rejects_non_real_scalar_tolerances_and_metrics(value):
    """Should reject coercible strings, booleans, complex values, and nonscalars."""
    with pytest.raises(TypeError, match='must be numeric'):
        _validate_tolerances({'metric': value}, 'absolute_tolerances')
    with pytest.raises(TypeError, match='must be numeric'):
        _extract_monitored_metrics(lambda _: {'metric': value}, [], ('metric',))


@pytest.mark.parametrize('value', [np.float32(.5), np.float64(.5), np.int32(1), np.int64(1)])
def test_convergence_accepts_numpy_real_scalars(value):
    """Should preserve valid NumPy real scalar metrics and tolerances."""
    assert _validate_tolerances({'metric': value}, 'tolerances') == {'metric': float(value)}
    assert _extract_monitored_metrics(lambda _: {'metric': value}, [], ('metric',)) == {'metric': float(value)}


def test_convergence_rejects_unrepresentable_integer_scalars():
    """Should report unrepresentable integers as invalid finite values without leaking overflow."""
    with pytest.raises(ValueError, match='finite'):
        _validate_tolerances({'metric': 10 ** 400}, 'tolerances')
    with pytest.raises(ValueError, match='finite'):
        _extract_monitored_metrics(lambda _: {'metric': 10 ** 400}, [], ('metric',))


@pytest.mark.parametrize('results', [
    [{'value': .1}, {'bootstrap_id': 2}],
    [0., {'bootstrap_id': 2}],
    [{'bootstrap_id': 1}, 0.],
    [{'value': .1}, {'value': .2}, {'bootstrap_id': 3}],
])
def test_bootstrap_ids_rejects_partial_presence_at_any_position(results):
    """Should reject partial or heterogeneous identities even when the first row has no ID."""
    with pytest.raises(ValueError, match='every mapping bootstrap replicate'):
        _bootstrap_ids_if_present(results)
    with pytest.raises(ValueError, match='every mapping bootstrap replicate'):
        _validate_bootstrap_results(results, len(results))


@pytest.mark.parametrize('identifier', [True, np.bool_(True), '1', 1., None, np.nan,
                                        np.inf, 1j, [1], 10 ** 400])
def test_bootstrap_results_requires_actual_integer_identities(identifier):
    """Should reject noninteger, boolean, missing, and out-of-order IDs without coercion."""
    with pytest.raises(ValueError, match='ordered sequence'):
        _validate_bootstrap_results([{'bootstrap_id': identifier}, {'bootstrap_id': 2}], 2)


@pytest.mark.parametrize('factory', [pd.DataFrame, pl.DataFrame])
def test_bootstrap_results_validates_dataframe_id_types(factory):
    """Should enforce integer IDs in both dataframe backends and accept NumPy integers."""
    _validate_bootstrap_results(factory({'bootstrap_id': [np.int64(1), np.int64(2)]}), 2)
    for ids in [[1., 2.], ['1', '2'], [None, None], [True, False]]:
        with pytest.raises(ValueError, match='ordered sequence'):
            _validate_bootstrap_results(factory({'bootstrap_id': ids}), 2)


@pytest.mark.parametrize('relative', [False, True])
@pytest.mark.parametrize('direction, passed', [(-1, False), (0, True), (1, True)])
def test_metric_tolerance_uses_inclusive_float_boundary(relative, direction, passed):
    """Should distinguish equality from adjacent floating-point tolerances for both criteria."""
    boundary = 1. if relative else 2.
    tolerance = boundary if direction == 0 else np.nextafter(boundary, -np.inf if direction < 0 else np.inf)
    result = _summarize_metric_values([1., 3.], None if relative else tolerance,
                                     tolerance if relative else None)
    assert result.passed is passed


@pytest.mark.parametrize('reference', [0., np.finfo(float).eps,
                                       np.nextafter(np.finfo(float).eps, np.inf)])
@pytest.mark.parametrize('sign', [-1, 1])
def test_relative_metric_denominator_boundary(reference, sign):
    """Should use the absolute mean and exclude equality at the machine-epsilon cutoff."""
    result = _summarize_metric_values([sign * reference] * 2, None, 0.)
    defined = bool(reference > np.finfo(float).eps)
    assert result.relative_range == (0. if defined else None)
    assert result.passed is defined


@pytest.mark.parametrize('values, expected_mean, expected_std, expected_range', [
    ([1e308, 1e308], 1e308, 0., 0.),
    ([1e200, 2e200], 1.5e200, 1e200 / sqrt(2), 1e200),
    ([-1e200, 1e200], 0., sqrt(2) * 1e200, 2e200),
    ([0., 0.], 0., 0., 0.),
])
def test_metric_summary_avoids_intermediate_overflow(values, expected_mean, expected_std, expected_range):
    """Should preserve analytically known finite diagnostics without overflow warnings."""
    with np.errstate(over='raise', invalid='raise'):
        result = _summarize_metric_values(values, expected_range, None)
    assert result.mean == pytest.approx(expected_mean)
    assert result.std == pytest.approx(expected_std)
    assert result.absolute_range == pytest.approx(expected_range)
    assert result.passed


@pytest.mark.parametrize('values', [[-1e308, 1e308], [-1e308, 1e308, 1.],
                                    [-5e307, 5e307, 1e-9]])
def test_metric_summary_rejects_unrepresentable_diagnostics(values):
    """Should reject an overflowing range or deviation instead of approving invalid diagnostics."""
    with pytest.raises(ValueError, match='representable as finite'):
        _summarize_metric_values(values, 1., None)


def test_convergence_preserves_finite_large_constant_diagnostics():
    """Should converge with finite mean and zero spread for a representable huge constant."""
    result = monte_carlo_error(lambda n_bootstraps, random_state: [1e308] * n_bootstraps,
                              lambda values: {'metric': values[0]}, [2, 4],
                              {'metric': 0.}, {'metric': 0.}, n_runs=2, random_state=42)
    assert result.converged
    for level in result.diagnostics:
        assert level.metrics['metric'].mean == 1e308
        assert level.metrics['metric'].std == 0.


def test_metric_summary_preserves_small_differences_between_large_values():
    """Should preserve a one-ULP difference instead of losing precision while scaling large values."""
    first = 1e308
    second = float(np.nextafter(first, np.inf))
    difference = second - first
    with np.errstate(over='raise', invalid='raise'):
        result = _summarize_metric_values([first, second], difference, None)
    assert result.std == pytest.approx(difference / sqrt(2), rel=1e-14)
    assert result.absolute_range == difference
    assert result.passed


# ------------------------------------------------------
# Real bootstrap integration and independent Monte Carlo references
# ------------------------------------------------------
@pytest.mark.parametrize('backend', ['pandas_series', 'pandas_frame', 'polars_series', 'polars_frame'])
@pytest.mark.parametrize('sample_size', [3, 4])
@pytest.mark.parametrize('continuity', [None, [0, 1, 4, 5]])
def test_convergence_integrates_real_mbb_and_metrics(backend, sample_size, continuity):
    """Should validate real MBB prefixes and all metric diagnostics against scalar reference calculations."""
    # This backend matrix can also be applied to additional bootstrap samplers.
    values = [0., 1., 4., 9.]
    source = (pd.Series(values) if backend == 'pandas_series' else
              pd.DataFrame({'value': values}) if backend == 'pandas_frame' else
              pl.Series('value', values) if backend == 'polars_series' else
              pl.DataFrame({'value': values}))
    blocks = generate_moving_blocks(source, block_length=2, continuity=continuity)
    bootstrap = partial(moving_block_bootstrap, blocks=blocks, sample_size=sample_size)
    estimates_by_call = []

    def metrics(samples):
        """Should summarize real samples while recording estimates for an independent oracle.

        Parameters
        ----------
        samples : list
            Real bootstrap samples in their original backend.

        Returns
        -------
        dict
            Production bootstrap metrics.

        Raises
        ------
        ValueError
            If bootstrap metrics cannot summarize the estimates.
        """
        estimates = [fsum(sample['value'].to_list() if 'frame' in backend else sample.to_list())
                     / sample_size for sample in samples]
        estimates_by_call.append(estimates)
        return asdict(bootstrap_metrics(estimates, confidence_level=.5))

    names = ['mean', 'std', 'ci_lower', 'ci_upper', 'pct_positive']
    result = monte_carlo_error(bootstrap, metrics, [8, 16, 32],
                              {name: 100. for name in names}, n_runs=3,
                              random_state=42, return_bootstrap_results=True)
    assert result.converged and result.recommended_n_bootstraps == 8
    assert [level.n_bootstraps for level in result.diagnostics] == [8, 16]
    # Explicitly enumerate possible block pairs, including truncation of the last block.
    candidates = [[0., 1.], [1., 4.], [4., 9.]] if continuity is None else [[0., 1.], [4., 9.]]
    support = {fsum((a + b)[:sample_size]) / sample_size for a, b in product(candidates, repeat=2)}
    for level_index, level in enumerate(result.diagnostics):
        references = []
        for estimates in estimates_by_call[level_index * 3:(level_index + 1) * 3]:
            assert set(estimates) <= support
            ordered = sorted(estimates)
            # Linear percentile interpolation, calculated without NumPy/production quantiles.
            bounds = []
            for probability in [.25, .75]:
                position = (len(ordered) - 1) * probability
                lower = int(position)
                weight = position - lower
                bounds.append(ordered[lower] * (1 - weight) + ordered[lower + 1] * weight)
            references.append({'mean': mean(estimates), 'std': stdev(estimates),
                               'ci_lower': bounds[0], 'ci_upper': bounds[1],
                               'pct_positive': sum(value > 0 for value in estimates) / len(estimates)})
        for name in names:
            expected = [row[name] for row in references]
            diagnostic = level.metrics[name]
            assert diagnostic.mean == pytest.approx(mean(expected))
            assert diagnostic.std == pytest.approx(stdev(expected))
            assert diagnostic.minimum == pytest.approx(min(expected))
            assert diagnostic.maximum == pytest.approx(max(expected))
            assert diagnostic.absolute_range == pytest.approx(max(expected) - min(expected))
        assert level.all_metrics_passed
    short = bootstrap(n_bootstraps=8, random_state=result.run_random_states[0])
    long = bootstrap(n_bootstraps=32, random_state=result.run_random_states[0])
    for retained, small, large in zip(result.reference_bootstrap_results, short, long[:8]):
        assert type(retained) is type(source)
        assert retained.equals(small) and small.equals(large)
    assert (source['value'].to_list() if 'frame' in backend else source.to_list()) == values


def test_real_mbb_monte_carlo_mean_has_inverse_root_b_scaling():
    """Should match the enumerated bootstrap mean variance and its inverse-root-B Monte Carlo scaling."""
    # Block means are 1/2, 5/2, 13/2. Two independent draws give
    # E[replicate mean] = 19/6 and Var[replicate mean] = 28/9.
    bootstrap = partial(moving_block_bootstrap,
                        blocks=generate_moving_blocks(pd.Series([0., 1., 4., 9.]), 2), sample_size=4)
    result = monte_carlo_error(bootstrap,
                              lambda samples: {'mean': bootstrap_metrics([sample.mean() for sample in samples]).mean},
                              [16, 256], {'mean': 0.}, n_runs=192, random_state=29)
    assert not result.converged
    for level in result.diagnostics:
        diagnostic = level.metrics['mean']
        expected_sd = sqrt(28 / (9 * level.n_bootstraps))
        # With 192 independent bounded runs, these deliberately broad bounds
        # characterize dispersion without demanding monotonicity for every seed.
        assert .7 * expected_sd < diagnostic.std < 1.3 * expected_sd
        assert abs(diagnostic.mean - 19 / 6) < 5 * expected_sd / sqrt(192)
    ratio = result.diagnostics[0].metrics['mean'].std / result.diagnostics[1].metrics['mean'].std
    assert 2.5 < ratio < 5.5


@pytest.mark.parametrize('relative', [False, True])
def test_real_mbb_zero_variance_respects_relative_zero_mean_policy(relative):
    """Should accept deterministic zero spread absolutely while leaving its relative criterion undefined."""
    bootstrap = partial(moving_block_bootstrap,
                        blocks=generate_moving_blocks(pd.Series([1., 1., 1.]), 2), sample_size=3)
    result = monte_carlo_error(bootstrap,
                              lambda samples: {'std': bootstrap_metrics([sample.mean() for sample in samples]).std},
                              [2, 4], absolute_tolerances=None if relative else {'std': 0.},
                              relative_tolerances={'std': 0.} if relative else None,
                              n_runs=2, random_state=42)
    assert result.converged is (not relative)
    for level in result.diagnostics:
        assert level.metrics['std'].mean == 0.
        assert level.metrics['std'].std == 0.
        assert level.metrics['std'].relative_range is None


def test_more_runs_can_increase_range_and_reverse_convergence():
    """Should retain existing streams while exposing range sensitivity to an additional run."""
    seeds = _generate_run_random_states(42, 3)
    bootstrap = lambda n_bootstraps, random_state: [float(seeds.index(random_state))] * n_bootstraps
    kwargs = {'bootstrap_fn': bootstrap, 'metrics_fn': lambda values: {'metric': values[0]},
              'n_bootstraps_grid': [2, 4], 'absolute_tolerances': {'metric': 1.}, 'random_state': 42}
    fewer = monte_carlo_error(**kwargs, n_runs=2)
    more = monte_carlo_error(**kwargs, n_runs=3)
    assert fewer.run_random_states == more.run_random_states[:2]
    assert fewer.converged and not more.converged
    for first, second in zip(fewer.diagnostics, more.diagnostics):
        assert first.metrics['metric'].absolute_range == 1.
        assert second.metrics['metric'].absolute_range == 2.


@pytest.mark.parametrize('container', ['list', 'tuple', 'numpy', 'pandas_series',
                                      'pandas_frame', 'polars_series', 'polars_frame'])
def test_convergence_preserves_supported_result_containers(container):
    """Should evaluate row prefixes and retain the original container without mutating generated results."""
    generated = []

    def bootstrap(n_bootstraps, random_state):
        """Should construct a deterministic container with nondefault Pandas indexes.

        Parameters
        ----------
        n_bootstraps, random_state : int
            Requested size and unused deterministic run seed.

        Returns
        -------
        object
            Sized ordered replicate collection.

        Raises
        ------
        ValueError
            If a dataframe cannot be constructed.
        """
        values = list(range(n_bootstraps))
        result = (values if container == 'list' else tuple(values) if container == 'tuple' else
                  np.array(values) if container == 'numpy' else
                  pd.Series(values, index=range(10, 10 + n_bootstraps)) if container == 'pandas_series' else
                  pd.DataFrame({'value': values}, index=range(10, 10 + n_bootstraps)) if container == 'pandas_frame' else
                  pl.Series('value', values) if container == 'polars_series' else pl.DataFrame({'value': values}))
        generated.append(result)
        return result

    metrics = lambda values: {'mean': mean(float(value) for value in
                                          (values['value'] if 'frame' in container else values))}
    result = monte_carlo_error(bootstrap, metrics, [2, 4, 8], {'mean': 0.}, n_runs=2,
                              random_state=42, return_bootstrap_results=True)
    assert result.converged and result.recommended_n_bootstraps == 2
    assert type(result.reference_bootstrap_results) is type(generated[0])
    retained = result.reference_bootstrap_results
    if container.startswith('pandas'):
        assert retained.index.equals(generated[0].index[:2])
        if container == 'pandas_frame':
            assert retained.columns.equals(generated[0].columns)
            assert retained.dtypes.equals(generated[0].dtypes)
        else:
            assert retained.dtype == generated[0].dtype
    elif container == 'polars_frame':
        assert retained.schema == generated[0].schema
    elif container in {'polars_series', 'numpy'}:
        assert retained.dtype == generated[0].dtype
    assert [level.metrics['mean'].mean for level in result.diagnostics] == [.5, 1.5]
    assert list(result.reference_bootstrap_results['value'] if 'frame' in container else result.reference_bootstrap_results) == [0, 1]
    for original in generated:
        assert list(original['value'] if 'frame' in container else original) == list(range(8))


def test_single_passing_level_cannot_confirm_convergence():
    """Should retain one passing diagnostic without recommending B or returning reference results."""
    result = monte_carlo_error(lambda n_bootstraps, random_state: [1.] * n_bootstraps,
                              lambda _: {'metric': 1.}, [2], {'metric': 0.}, n_runs=2,
                              random_state=42, return_bootstrap_results=True)
    assert result.diagnostics[0].all_metrics_passed
    assert not result.diagnostics[0].convergence_confirmed
    assert not result.converged
    assert result.recommended_n_bootstraps is None
    assert result.reference_bootstrap_results is None and result.reference_run_id is None


@pytest.mark.parametrize('stage', ['bootstrap', 'metrics'])
def test_convergence_propagates_user_callable_failures(stage):
    """Should propagate the original bootstrap or metrics exception without retries or later calls."""
    exception = RuntimeError('user calculation failed')
    calls = []
    bootstrap = lambda n_bootstraps, random_state: calls.append('bootstrap') or (
        (_ for _ in ()).throw(exception) if stage == 'bootstrap' else [1.] * n_bootstraps)
    metrics = lambda _: calls.append('metrics') or (_ for _ in ()).throw(exception)
    with pytest.raises(RuntimeError) as caught:
        monte_carlo_error(bootstrap, metrics, [2, 4], {'metric': 0.}, n_runs=2, random_state=42)
    assert caught.value is exception
    assert calls == (['bootstrap'] if stage == 'bootstrap' else ['bootstrap', 'bootstrap', 'metrics'])
