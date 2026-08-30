import numpy as np
import pandas as pd
import pytest

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
