"""Descriptive return contrasts and preliminary cost margins."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd

from alpha_research.evaluation.relationship import FeatureTargetRelationshipResult

__all__ = [
    'EconomicContrastResult',
    'FeatureGroupSpec',
    'PreliminaryMarginResult',
    'economic_return_contrast',
    'preliminary_cost_margin',
]

_UNIT_SCALES = {'fraction': 1.0, 'percent': 100.0, 'bps': 10_000.0}


@dataclass(frozen=True, slots=True)
class FeatureGroupSpec:
    """Select feature observations by bins, exact values, or numeric bounds.

    Parameters
    ----------
    name : str
        Caller-defined label for the selected observations.
    bins : tuple[int, ...] | None, default None
        Feature-only bin numbers from an existing relationship.
    values : tuple[float, ...] | None, default None
        Exact feature values, suitable for discrete transformations.
    lower, upper : float | None, default None
        Numeric bounds. At least one is required when bins and values are absent.
    include_lower, include_upper : bool
        Whether an available bound belongs to the selected group.

    Returns
    -------
    FeatureGroupSpec
        Validated selector independent of economic interpretation.

    Raises
    ------
    TypeError
        If a label, selector, bound, or inclusion flag has an invalid type.
    ValueError
        If selectors conflict or contain empty, repeated, or invalid values.
    """

    name: str
    bins: tuple[int, ...] | None = None
    values: tuple[float, ...] | None = None
    lower: float | None = None
    upper: float | None = None
    include_lower: bool = True
    include_upper: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError('name must be a non-empty string.')
        if not isinstance(self.include_lower, bool) or not isinstance(
            self.include_upper, bool,
        ):
            raise TypeError('bound inclusion flags must be booleans.')
        modes = sum((
            self.bins is not None,
            self.values is not None,
            self.lower is not None or self.upper is not None,
        ))
        if modes != 1:
            raise ValueError('specify exactly one of bins, values, or bounds.')
        if self.bins is not None:
            if not isinstance(self.bins, (tuple, list)) or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in self.bins
            ):
                raise TypeError('bins must be a tuple or list of integers.')
            if not self.bins or min(self.bins) < 1 or len(set(self.bins)) != len(self.bins):
                raise ValueError('bins must contain distinct positive integers.')
            object.__setattr__(self, 'bins', tuple(self.bins))
        if self.values is not None:
            if not isinstance(self.values, (tuple, list)) or any(
                not isinstance(value, Real) or isinstance(value, bool)
                for value in self.values
            ):
                raise TypeError('values must be a tuple or list of real numbers.')
            try:
                normalized = tuple(float(value) for value in self.values)
            except OverflowError as error:
                raise ValueError('values must contain finite numbers.') from error
            if not normalized or not all(np.isfinite(value) for value in normalized):
                raise ValueError('values must contain finite numbers.')
            if len(set(normalized)) != len(normalized):
                raise ValueError('values must not repeat.')
            object.__setattr__(self, 'values', normalized)
        for name in ('lower', 'upper'):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, Real) or isinstance(value, bool):
                    raise TypeError(f'{name} must be a real number or None.')
                try:
                    normalized_bound = float(value)
                except OverflowError as error:
                    raise ValueError(f'{name} must be finite.') from error
                if not np.isfinite(normalized_bound):
                    raise ValueError(f'{name} must be finite.')
                object.__setattr__(self, name, normalized_bound)
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise ValueError('lower must be less than upper.')


@dataclass(frozen=True, slots=True)
class EconomicContrastResult:
    """Store a signed difference between two conditional mean returns.

    Parameters
    ----------
    feature, target : str
        Analyzed column names.
    group_a, group_b : FeatureGroupSpec
        Ordered groups; effect is mean_a minus mean_b.
    mean_a, mean_b, effect : float
        Conditional means and their difference in the selected output unit.
    n_a, n_b, n_valid, n_dropped, n_unassigned : int
        Observation counts from the selected groups and relationship.
    binning : {'quantile', 'equal_width'}
        Feature-only binning used by the relationship.
    n_bins : int
        Requested number of feature bins.
    unit : {'fraction', 'percent', 'bps'}
        Display unit. The input target is always a fractional return.

    Returns
    -------
    EconomicContrastResult
        Descriptive contrast with its group and unit provenance.

    Raises
    ------
    TypeError
        If required constructor fields are omitted.
    """

    feature: str
    target: str
    group_a: FeatureGroupSpec
    group_b: FeatureGroupSpec
    mean_a: float
    mean_b: float
    effect: float
    n_a: int
    n_b: int
    n_valid: int
    n_dropped: int
    n_unassigned: int
    binning: Literal['quantile', 'equal_width']
    n_bins: int
    unit: Literal['fraction', 'percent', 'bps']


@dataclass(frozen=True, slots=True)
class PreliminaryMarginResult:
    """Store a contrast after caller-supplied preliminary costs.

    Parameters
    ----------
    contrast : EconomicContrastResult
        Gross conditional-return contrast.
    scenario : str
        Caller-defined cost scenario label.
    components : dict[str, float]
        Nonnegative cost components in the contrast display unit.
    total_cost, margin : float
        Sum of costs and gross effect minus total cost.

    Returns
    -------
    PreliminaryMarginResult
        Cost decomposition and preliminary margin.

    Raises
    ------
    TypeError
        If required constructor fields are omitted.
    """

    contrast: EconomicContrastResult
    scenario: str
    components: dict[str, float]
    total_cost: float
    margin: float


def _select_feature_group(
        relationship: FeatureTargetRelationshipResult,
        pairs: pd.DataFrame,
        group: FeatureGroupSpec,
) -> np.ndarray:
    """Select relationship pairs using one validated feature group.

    Parameters
    ----------
    relationship : FeatureTargetRelationshipResult
        Source of the feature name and requested bin count.
    pairs : pd.DataFrame
        Finite relationship pairs with their assigned feature bins.
    group : FeatureGroupSpec
        Selector expressed as bins, exact values, or bounds.

    Returns
    -------
    np.ndarray
        Boolean mask aligned to pairs, including False for unassigned bins.

    Raises
    ------
    ValueError
        If the selector refers to a bin beyond the requested bin count.
    """
    if group.bins is not None:
        if max(group.bins) > relationship.n_bins:
            raise ValueError('selected bins must not exceed relationship.n_bins.')
        return pairs['bin'].isin(group.bins).to_numpy(dtype=bool, na_value=False)
    feature_values = pairs[relationship.feature].to_numpy(dtype=float)
    if group.values is not None:
        return np.isin(feature_values, group.values)
    selected = np.ones(len(pairs), dtype=bool)
    if group.lower is not None:
        selected &= (
            feature_values >= group.lower if group.include_lower
            else feature_values > group.lower
        )
    if group.upper is not None:
        selected &= (
            feature_values <= group.upper if group.include_upper
            else feature_values < group.upper
        )
    return selected


def economic_return_contrast(
        relationship: FeatureTargetRelationshipResult,
        group_a: FeatureGroupSpec,
        group_b: FeatureGroupSpec,
        unit: Literal['fraction', 'percent', 'bps'] = 'fraction',
        min_group_size: int = 1,
) -> EconomicContrastResult:
    """Compare two caller-selected groups of fractional forward returns.

    Parameters
    ----------
    relationship : FeatureTargetRelationshipResult
        Existing feature-only binning and finite aligned feature-target pairs.
        Bins must be ungrouped; the target must represent a fractional return,
        not a categorical label.
    group_a, group_b : FeatureGroupSpec
        Disjoint groups selected by bin, feature value, or feature bounds.
    unit : {'fraction', 'percent', 'bps'}, default 'fraction'
        Unit of the returned means and effect; inputs remain fractional returns.
    min_group_size : int, default 1
        Minimum observations required in each group.

    Returns
    -------
    EconomicContrastResult
        Mean return of group A minus group B, group counts, and provenance.

    Raises
    ------
    TypeError
        If an argument has an unsupported type.
    ValueError
        If groups overlap, refer to unavailable bins, have insufficient rows,
        or the output unit is unsupported.

    Notes
    -----
    Means pool observations within each selected group. This is descriptive
    research, not a trading simulation or an uncertainty estimate.
    """
    if not isinstance(relationship, FeatureTargetRelationshipResult):
        raise TypeError('relationship must be a FeatureTargetRelationshipResult.')
    if relationship.group_col is not None:
        raise ValueError('relationship bins must not be grouped for this contrast.')
    if not isinstance(group_a, FeatureGroupSpec) or not isinstance(group_b, FeatureGroupSpec):
        raise TypeError('groups must be FeatureGroupSpec instances.')
    if group_a.name == group_b.name:
        raise ValueError('group names must differ.')
    if not isinstance(unit, str):
        raise TypeError('unit must be a string.')
    if unit not in _UNIT_SCALES:
        raise ValueError("unit must be 'fraction', 'percent', or 'bps'.")
    if not isinstance(min_group_size, int) or isinstance(min_group_size, bool):
        raise TypeError('min_group_size must be an integer.')
    if min_group_size < 1:
        raise ValueError('min_group_size must be positive.')

    pairs = (
        relationship.pairs
        if isinstance(relationship.pairs, pd.DataFrame)
        else relationship.pairs.to_pandas()
    )
    selected_a = _select_feature_group(relationship, pairs, group_a)
    selected_b = _select_feature_group(relationship, pairs, group_b)
    if np.any(selected_a & selected_b):
        raise ValueError('selected groups must not overlap.')
    n_a = int(selected_a.sum())
    n_b = int(selected_b.sum())
    if min(n_a, n_b) < min_group_size:
        raise ValueError('each group must contain at least min_group_size observations.')
    returns = pairs[relationship.target].to_numpy(dtype=float)
    scale = _UNIT_SCALES[unit]
    mean_a = float(returns[selected_a].mean()) * scale
    mean_b = float(returns[selected_b].mean()) * scale
    effect = mean_a - mean_b
    if not all(np.isfinite(value) for value in (mean_a, mean_b, effect)):
        raise ValueError('group means and effect must be finite in the selected unit.')
    return EconomicContrastResult(
        feature=relationship.feature,
        target=relationship.target,
        group_a=group_a,
        group_b=group_b,
        mean_a=mean_a,
        mean_b=mean_b,
        effect=effect,
        n_a=n_a,
        n_b=n_b,
        n_valid=relationship.n_valid,
        n_dropped=relationship.n_dropped,
        n_unassigned=relationship.n_unassigned,
        binning=relationship.binning,
        n_bins=relationship.n_bins,
        unit=unit,
    )


def preliminary_cost_margin(
        contrast: EconomicContrastResult,
        components: Mapping[str, Real],
        scenario: str,
) -> PreliminaryMarginResult:
    """Subtract explicit fractional-return costs from one gross contrast.

    Parameters
    ----------
    contrast : EconomicContrastResult
        Gross contrast, possibly displayed in percent or basis points.
    components : Mapping[str, Real]
        Nonnegative fractional-return costs already converted to the same
        exposure basis as the contrast. No cost or trade count is inferred.
    scenario : str
        Caller-defined cost scenario label.

    Returns
    -------
    PreliminaryMarginResult
        Components and margin in the contrast's display unit.

    Raises
    ------
    TypeError
        If inputs or a component value have unsupported types.
    ValueError
        If a label is empty or a cost is negative or non-finite.
    """
    if not isinstance(contrast, EconomicContrastResult):
        raise TypeError('contrast must be an EconomicContrastResult.')
    if not isinstance(components, Mapping):
        raise TypeError('components must be a mapping.')
    if not components:
        raise ValueError('components must not be empty.')
    if not isinstance(scenario, str) or not scenario.strip():
        raise TypeError('scenario must be a non-empty string.')
    converted = {}
    for name, value in components.items():
        if not isinstance(name, str) or not name.strip():
            raise TypeError('component names must be non-empty strings.')
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError('component costs must be real numbers.')
        try:
            fractional_cost = float(value)
        except OverflowError as error:
            raise ValueError('component costs must be nonnegative and finite.') from error
        if not np.isfinite(fractional_cost) or fractional_cost < 0:
            raise ValueError('component costs must be nonnegative and finite.')
        converted_cost = fractional_cost * _UNIT_SCALES[contrast.unit]
        if not np.isfinite(converted_cost):
            raise ValueError('component costs must be finite in the selected unit.')
        converted[name] = converted_cost
    total_cost = float(sum(converted.values()))
    if not np.isfinite(total_cost):
        raise ValueError('total cost must be finite in the selected unit.')
    margin = contrast.effect - total_cost
    if not np.isfinite(margin):
        raise ValueError('margin must be finite in the selected unit.')
    return PreliminaryMarginResult(
        contrast=contrast,
        scenario=scenario,
        components=converted,
        total_cost=total_cost,
        margin=margin,
    )
