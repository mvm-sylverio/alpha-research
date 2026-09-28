import numpy as np
import pandas as pd
import polars as pl
import pytest

from alpha_research.evaluation import (
    FeatureGroupSpec,
    economic_return_contrast,
    feature_target_relationship,
    preliminary_cost_margin,
)
from alpha_research.evaluation.economic_significance import _select_feature_group


@pytest.fixture
def return_relationship_pandas():
    """Should provide ten hand-verifiable feature and return pairs."""
    frame = pd.DataFrame({
        'time': range(10),
        'symbol': ['A'] * 10,
        'feature': range(1, 11),
        'target': [value / 100 for value in range(1, 11)],
    })
    return feature_target_relationship(frame, 'feature', 'target', n_bins=10)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('unit, expected_extreme, expected_wide', [
    ('fraction', (0.10, 0.01, 0.09), (0.095, 0.015, 0.08)),
    ('percent', (10.0, 1.0, 9.0), (9.5, 1.5, 8.0)),
    ('bps', (1000.0, 100.0, 900.0), (950.0, 150.0, 800.0)),
])
def test_economic_return_contrast_uses_existing_bins_and_units(
        backend, unit, expected_extreme, expected_wide,
):
    """Should compute Q10-Q1 and pooled two-bin means from simple returns."""
    frame = pd.DataFrame({
        'time': range(10),
        'symbol': ['A'] * 10,
        'feature': range(1, 11),
        'target': [value / 100 for value in range(1, 11)],
    })
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    relationship = feature_target_relationship(frame, 'feature', 'target', n_bins=10)
    extreme = economic_return_contrast(
        relationship, FeatureGroupSpec('Q10', bins=(10,)),
        FeatureGroupSpec('Q1', bins=(1,)), unit=unit,
    )
    wide = economic_return_contrast(
        relationship, FeatureGroupSpec('Q9-Q10', bins=(9, 10)),
        FeatureGroupSpec('Q1-Q2', bins=(1, 2)), unit=unit,
    )

    # The source returns are 1% through 10%; the two wider groups average
    # 9.5% and 1.5%, independently of the implementation's bin summaries.
    assert extreme.mean_a == pytest.approx(expected_extreme[0])
    assert extreme.mean_b == pytest.approx(expected_extreme[1])
    assert extreme.effect == pytest.approx(expected_extreme[2])
    assert extreme.n_a == extreme.n_b == 1
    assert wide.mean_a == pytest.approx(expected_wide[0])
    assert wide.mean_b == pytest.approx(expected_wide[1])
    assert wide.effect == pytest.approx(expected_wide[2])
    assert wide.n_a == wide.n_b == 2
    assert wide.n_valid == 10
    assert wide.n_dropped == wide.n_unassigned == 0
    assert wide.binning == 'quantile'
    assert wide.n_bins == 10
    assert wide.unit == unit


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_economic_return_contrast_selects_discrete_values_and_bounds(backend):
    """Should support sign-like values and explicit numeric thresholds."""
    sign_frame = pd.DataFrame({
        'time': range(6),
        'symbol': ['A'] * 6,
        'feature': [-1, -1, 0, 0, 1, 1],
        'target': [-0.02, -0.01, 0.0, 0.01, 0.03, 0.05],
    })
    if backend == 'polars':
        sign_frame = pl.from_pandas(sign_frame)
    sign_relation = feature_target_relationship(
        sign_frame, 'feature', 'target', n_bins=3,
    )
    signed = economic_return_contrast(
        sign_relation, FeatureGroupSpec('positive', values=(1,)),
        FeatureGroupSpec('negative', values=(-1,)),
    )
    assert signed.mean_a == pytest.approx(0.04)
    assert signed.mean_b == pytest.approx(-0.015)
    assert signed.effect == pytest.approx(0.055)
    assert signed.n_a == signed.n_b == 2

    threshold_frame = pd.DataFrame({
        'time': range(10),
        'symbol': ['A'] * 10,
        'feature': range(1, 11),
        'target': [value / 100 for value in range(1, 11)],
    })
    if backend == 'polars':
        threshold_frame = pl.from_pandas(threshold_frame)
    threshold_relation = feature_target_relationship(
        threshold_frame, 'feature', 'target', n_bins=5,
    )
    threshold = economic_return_contrast(
        threshold_relation, FeatureGroupSpec('at least six', lower=6),
        FeatureGroupSpec('below six', upper=6),
    )
    assert threshold.mean_a == pytest.approx(0.08)
    assert threshold.mean_b == pytest.approx(0.03)
    assert threshold.effect == pytest.approx(0.05)
    assert threshold.n_a == threshold.n_b == 5


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_economic_return_contrast_keeps_deciles_distinct_and_supports_reverse_direction(
        backend,
):
    """Should calculate an inner contrast and retain a negative reverse effect."""
    frame = pd.DataFrame({
        'time': range(10),
        'symbol': ['A'] * 10,
        'feature': range(1, 11),
        'target': [value / 100 for value in range(1, 11)],
    })
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    relationship = feature_target_relationship(frame, 'feature', 'target', n_bins=10)
    inner = economic_return_contrast(
        relationship, FeatureGroupSpec('Q8', bins=(8,)),
        FeatureGroupSpec('Q3', bins=(3,)),
    )
    reversed_contrast = economic_return_contrast(
        relationship, FeatureGroupSpec('Q1', bins=(1,)),
        FeatureGroupSpec('Q10', bins=(10,)),
    )

    assert inner.mean_a == pytest.approx(0.08)
    assert inner.mean_b == pytest.approx(0.03)
    assert inner.effect == pytest.approx(0.05)
    assert reversed_contrast.effect == pytest.approx(-0.09)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_economic_return_contrast_excludes_invalid_pairs_without_mutating_input(
        backend,
):
    """Should account for invalid targets while retaining the original pairs."""
    frame = pd.DataFrame({
        'time': range(8),
        'symbol': ['A'] * 8,
        'feature': range(1, 9),
        'target': [0.01, np.nan, 0.03, 0.04, np.inf, 0.06, 0.07, 0.08],
    })
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    relationship = feature_target_relationship(frame, 'feature', 'target', n_bins=3)
    original_pairs = relationship.pairs.clone() if backend == 'polars' else relationship.pairs.copy(deep=True)
    result = economic_return_contrast(
        relationship, FeatureGroupSpec('high', bins=(3,)),
        FeatureGroupSpec('low', bins=(1,)),
    )

    # Valid targets are 1%, 3%, 4%, 6%, 7%, and 8%. The lowest and highest
    # thirds therefore average 2% and 7.5%, respectively.
    assert result.mean_a == pytest.approx(0.075)
    assert result.mean_b == pytest.approx(0.02)
    assert result.effect == pytest.approx(0.055)
    assert result.n_a == result.n_b == 2
    assert result.n_valid == 6
    assert result.n_dropped == 2
    if backend == 'polars':
        assert relationship.pairs.equals(original_pairs)
    else:
        pd.testing.assert_frame_equal(relationship.pairs, original_pairs)


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
def test_economic_return_contrast_handles_ties_without_fabricating_extreme_bins(
        backend,
):
    """Should use actual sign values when ties leave requested deciles empty."""
    frame = pd.DataFrame({
        'time': range(10),
        'symbol': ['A'] * 10,
        'feature': [-1] * 5 + [1] * 5,
        'target': [value / 100 for value in range(1, 11)],
    })
    if backend == 'polars':
        frame = pl.from_pandas(frame)
    relationship = feature_target_relationship(frame, 'feature', 'target', n_bins=10)
    with pytest.raises(ValueError, match='min_group_size'):
        economic_return_contrast(
            relationship, FeatureGroupSpec('Q10', bins=(10,)),
            FeatureGroupSpec('Q1', bins=(1,)),
        )
    signed = economic_return_contrast(
        relationship, FeatureGroupSpec('positive', values=(1,)),
        FeatureGroupSpec('negative', values=(-1,)),
    )
    assert signed.mean_a == pytest.approx(0.08)
    assert signed.mean_b == pytest.approx(0.03)
    assert signed.effect == pytest.approx(0.05)
    assert signed.n_a == signed.n_b == 5


@pytest.mark.parametrize('kwargs, error', [
    ({}, ValueError),
    ({'bins': (1,), 'values': (1,)}, ValueError),
    ({'bins': ()}, ValueError),
    ({'bins': (1, 1)}, ValueError),
    ({'bins': (0,)}, ValueError),
    ({'bins': (True,)}, TypeError),
    ({'bins': (1.0,)}, TypeError),
    ({'bins': '1'}, TypeError),
    ({'values': ()}, ValueError),
    ({'values': (1, 1)}, ValueError),
    ({'values': (np.inf,)}, ValueError),
    ({'values': (10**1000,)}, ValueError),
    ({'values': (True,)}, TypeError),
    ({'values': ('one',)}, TypeError),
    ({'lower': 3, 'upper': 2}, ValueError),
    ({'lower': 2, 'upper': 2}, ValueError),
    ({'lower': np.nan}, ValueError),
    ({'lower': 10**1000}, ValueError),
    ({'upper': -np.inf}, ValueError),
    ({'upper': True}, TypeError),
    ({'lower': 0, 'include_lower': 1}, TypeError),
    ({'upper': 0, 'include_upper': 1}, TypeError),
])
def test_feature_group_spec_rejects_invalid_selectors(kwargs, error):
    """Should reject ambiguous or invalid feature group definitions."""
    with pytest.raises(error):
        FeatureGroupSpec('group', **kwargs)


def test_feature_group_spec_rejects_invalid_name():
    """Should require a nonempty caller-defined group name."""
    with pytest.raises(TypeError, match='name'):
        FeatureGroupSpec('', bins=(1,))


def test_feature_group_spec_normalizes_mutable_selector_inputs():
    """Should keep copied immutable selectors when caller lists change."""
    bins = [1, 2]
    values = [-1, 1]
    bin_group = FeatureGroupSpec('low bins', bins=bins)
    value_group = FeatureGroupSpec('signs', values=values)
    bins.append(3)
    values.append(0)

    assert bin_group.bins == (1, 2)
    assert value_group.values == (-1.0, 1.0)


def test_select_feature_group_handles_bins_values_and_boundary_rules(
        return_relationship_pandas,
):
    """Should select each group mode and reject bins outside the relationship."""
    pairs = return_relationship_pandas.pairs.copy()
    pairs.loc[0, 'bin'] = pd.NA
    by_bin = _select_feature_group(
        return_relationship_pandas, pairs, FeatureGroupSpec('Q1-Q2', bins=(1, 2)),
    )
    by_value = _select_feature_group(
        return_relationship_pandas, pairs, FeatureGroupSpec('one', values=(1,)),
    )
    by_bounds = _select_feature_group(
        return_relationship_pandas, pairs,
        FeatureGroupSpec('three through five', lower=3, upper=5, include_upper=True),
    )

    assert np.flatnonzero(by_bin).tolist() == [1]
    assert np.flatnonzero(by_value).tolist() == [0]
    assert np.flatnonzero(by_bounds).tolist() == [2, 3, 4]
    with pytest.raises(ValueError, match='n_bins'):
        _select_feature_group(
            return_relationship_pandas, pairs, FeatureGroupSpec('Q11', bins=(11,)),
        )


def test_select_feature_group_applies_exclusive_lower_and_inclusive_upper(
        return_relationship_pandas,
):
    """Should place a threshold value in exactly the requested group."""
    pairs = return_relationship_pandas.pairs
    above = _select_feature_group(
        return_relationship_pandas, pairs,
        FeatureGroupSpec('above five', lower=5, include_lower=False),
    )
    at_or_below = _select_feature_group(
        return_relationship_pandas, pairs,
        FeatureGroupSpec('at or below five', upper=5, include_upper=True),
    )
    assert np.flatnonzero(above).tolist() == [5, 6, 7, 8, 9]
    assert np.flatnonzero(at_or_below).tolist() == [0, 1, 2, 3, 4]
    assert not np.any(above & at_or_below)


def test_economic_return_contrast_validates_groups(return_relationship_pandas):
    """Should reject overlaps, unavailable bins, and insufficient groups."""
    q1 = FeatureGroupSpec('Q1', bins=(1,))
    q10 = FeatureGroupSpec('Q10', bins=(10,))
    with pytest.raises(ValueError, match='overlap'):
        economic_return_contrast(
            return_relationship_pandas,
            FeatureGroupSpec('left', bins=(1, 2)),
            FeatureGroupSpec('right', bins=(2, 3)),
        )
    with pytest.raises(ValueError, match='overlap'):
        economic_return_contrast(
            return_relationship_pandas, q1,
            FeatureGroupSpec('value one', values=(1,)),
        )
    with pytest.raises(ValueError, match='n_bins'):
        economic_return_contrast(
            return_relationship_pandas, FeatureGroupSpec('Q11', bins=(11,)), q1,
        )
    with pytest.raises(ValueError, match='min_group_size'):
        economic_return_contrast(return_relationship_pandas, q10, q1, min_group_size=2)
    with pytest.raises(ValueError, match='min_group_size'):
        economic_return_contrast(
            return_relationship_pandas,
            FeatureGroupSpec('absent', values=(100,)), q1,
        )
    with pytest.raises(ValueError, match='names'):
        economic_return_contrast(
            return_relationship_pandas, q1, FeatureGroupSpec('Q1', bins=(10,)),
        )


def test_economic_return_contrast_rejects_grouped_relationship():
    """Should not silently pool date-specific bins with unequal date weights."""
    frame = pd.DataFrame({
        'time': [1, 1, 2, 2],
        'symbol': ['A', 'B', 'A', 'B'],
        'feature': [1, 2, 3, 4],
        'target': [0.01, 0.02, 0.03, 0.04],
    })
    relationship = feature_target_relationship(
        frame, 'feature', 'target', n_bins=2, group_col='time',
    )
    with pytest.raises(ValueError, match='must not be grouped'):
        economic_return_contrast(
            relationship, FeatureGroupSpec('high', bins=(2,)),
            FeatureGroupSpec('low', bins=(1,)),
        )


@pytest.mark.parametrize('relationship, group_a, group_b, unit, minimum, error', [
    (None, FeatureGroupSpec('a', bins=(1,)), FeatureGroupSpec('b', bins=(2,)), 'fraction', 1, TypeError),
    ('bad', FeatureGroupSpec('a', bins=(1,)), FeatureGroupSpec('b', bins=(2,)), 'fraction', 1, TypeError),
    ('valid', None, FeatureGroupSpec('b', bins=(2,)), 'fraction', 1, TypeError),
    ('valid', FeatureGroupSpec('a', bins=(1,)), FeatureGroupSpec('b', bins=(2,)), 'points', 1, ValueError),
    ('valid', FeatureGroupSpec('a', bins=(1,)), FeatureGroupSpec('b', bins=(2,)), None, 1, TypeError),
    ('valid', FeatureGroupSpec('a', bins=(1,)), FeatureGroupSpec('b', bins=(2,)), 'fraction', True, TypeError),
    ('valid', FeatureGroupSpec('a', bins=(1,)), FeatureGroupSpec('b', bins=(2,)), 'fraction', 0, ValueError),
])
def test_economic_return_contrast_validates_configuration(
        return_relationship_pandas, relationship, group_a, group_b, unit, minimum, error,
):
    """Should reject invalid result, selectors, unit, and minimum sample size."""
    selected = return_relationship_pandas if relationship == 'valid' else relationship
    with pytest.raises(error):
        economic_return_contrast(selected, group_a, group_b, unit, minimum)


def test_economic_return_contrast_rejects_nonfinite_output():
    """Should reject a finite input pair whose signed difference overflows."""
    frame = pd.DataFrame({
        'time': [1, 2],
        'symbol': ['A', 'A'],
        'feature': [1.0, 2.0],
        'target': [-1e308, 1e308],
    })
    relationship = feature_target_relationship(frame, 'feature', 'target', n_bins=2)
    with pytest.raises(ValueError, match='finite'):
        economic_return_contrast(
            relationship, FeatureGroupSpec('high', bins=(2,)),
            FeatureGroupSpec('low', bins=(1,)),
        )


@pytest.mark.parametrize('unit, expected_components, expected_total, expected_margin', [
    (
        'fraction',
        {'spread': 0.001, 'fees': 0.0005, 'impact': 0.0002},
        0.0017, 0.0883,
    ),
    (
        'percent',
        {'spread': 0.1, 'fees': 0.05, 'impact': 0.02},
        0.17, 8.83,
    ),
    (
        'bps',
        {'spread': 10.0, 'fees': 5.0, 'impact': 2.0},
        17.0, 883.0,
    ),
])
def test_preliminary_cost_margin_preserves_component_breakdown(
        return_relationship_pandas, unit, expected_components, expected_total,
        expected_margin,
):
    """Should subtract explicit costs exactly once after unit conversion."""
    contrast = economic_return_contrast(
        return_relationship_pandas,
        FeatureGroupSpec('Q10', bins=(10,)),
        FeatureGroupSpec('Q1', bins=(1,)), unit=unit,
    )
    costs = {'spread': 0.001, 'fees': 0.0005, 'impact': 0.0002}
    result = preliminary_cost_margin(contrast, costs, 'base')

    assert result.scenario == 'base'
    assert result.contrast is contrast
    # Hand conversion: the 9% gross spread is 900 bps; the three costs total
    # 0.17% or 17 bps, leaving 8.83% or 883 bps.
    assert result.components == expected_components
    assert result.total_cost == pytest.approx(expected_total)
    assert result.margin == pytest.approx(expected_margin)
    assert costs['spread'] == 0.001


def test_preliminary_cost_margin_orders_explicit_scenarios(return_relationship_pandas):
    """Should keep gross effect fixed and reduce margin as supplied costs rise."""
    contrast = economic_return_contrast(
        return_relationship_pandas,
        FeatureGroupSpec('Q10', bins=(10,)),
        FeatureGroupSpec('Q1', bins=(1,)),
    )
    low = preliminary_cost_margin(contrast, {'spread': 0.001}, 'low')
    base = preliminary_cost_margin(contrast, {'spread': 0.001, 'fees': 0.001}, 'base')
    stress = preliminary_cost_margin(
        contrast, {'spread': 0.001, 'fees': 0.001, 'impact': 0.002}, 'stress',
    )
    zero = preliminary_cost_margin(contrast, {'explicit_zero': 0.0}, 'zero')

    assert low.margin == pytest.approx(0.089)
    assert base.margin == pytest.approx(0.088)
    assert stress.margin == pytest.approx(0.086)
    assert zero.margin == pytest.approx(0.09)
    assert low.margin > base.margin > stress.margin


def test_preliminary_cost_margin_preserves_negative_effect(return_relationship_pandas):
    """Should reduce a negative gross contrast instead of replacing its sign."""
    contrast = economic_return_contrast(
        return_relationship_pandas,
        FeatureGroupSpec('Q1', bins=(1,)),
        FeatureGroupSpec('Q10', bins=(10,)),
    )
    result = preliminary_cost_margin(contrast, {'spread': 0.002}, 'base')
    assert contrast.effect == pytest.approx(-0.09)
    assert result.margin == pytest.approx(-0.092)


def test_preliminary_cost_margin_rejects_nonfinite_arithmetic(return_relationship_pandas):
    """Should reject overflow during component conversion or cost summation."""
    contrast_bps = economic_return_contrast(
        return_relationship_pandas,
        FeatureGroupSpec('Q10', bins=(10,)),
        FeatureGroupSpec('Q1', bins=(1,)), unit='bps',
    )
    with pytest.raises(ValueError, match='selected unit'):
        preliminary_cost_margin(contrast_bps, {'spread': 1e308}, 'stress')

    contrast_fraction = economic_return_contrast(
        return_relationship_pandas,
        FeatureGroupSpec('Q10', bins=(10,)),
        FeatureGroupSpec('Q1', bins=(1,)),
    )
    with pytest.raises(ValueError, match='total cost'):
        preliminary_cost_margin(
            contrast_fraction, {'spread': 1e308, 'impact': 1e308}, 'stress',
        )


@pytest.mark.parametrize('components, scenario, error', [
    ({}, 'base', ValueError),
    (None, 'base', TypeError),
    ({'spread': -0.01}, 'base', ValueError),
    ({'spread': np.inf}, 'base', ValueError),
    ({'spread': np.nan}, 'base', ValueError),
    ({'spread': 10**1000}, 'base', ValueError),
    ({'spread': True}, 'base', TypeError),
    ({'': 0.01}, 'base', TypeError),
    ({'spread': 0.01}, '', TypeError),
])
def test_preliminary_cost_margin_validates_inputs(
        return_relationship_pandas, components, scenario, error,
):
    """Should require explicit nonnegative finite costs and a scenario name."""
    contrast = economic_return_contrast(
        return_relationship_pandas,
        FeatureGroupSpec('Q10', bins=(10,)),
        FeatureGroupSpec('Q1', bins=(1,)),
    )
    with pytest.raises(error):
        preliminary_cost_margin(contrast, components, scenario)
    with pytest.raises(TypeError, match='contrast'):
        preliminary_cost_margin(None, {'spread': 0.01}, 'base')
