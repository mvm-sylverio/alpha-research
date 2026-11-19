"""Domain-neutral input requirements for evaluation functions."""

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

__all__ = ['EVALUATION_REQUIREMENTS', 'EvaluationRequirements']


@dataclass(frozen=True, slots=True)
class EvaluationRequirements:
    """Describe calculation inputs, observation layout, and inference separately.

    Parameters
    ----------
    layout : str
        How the calculation organizes its input:

        - ``pooled``: Uses selected observations together. For a correlation,
          each aligned pair counts once, even when entities have different
          numbers of observations.
        - ``cross_sectional``: Compares observations within a shared coordinate
          or group. Each cross-section contains observations with the same
          coordinate or group key.
        - ``single_sequence``: Uses the order of observations within one
          sequence. An explicit axis column may be required, or the order may
          come from the supplied vector.
        - ``multiple_sequences``: Uses several sequences while retaining their
          boundaries. Order requirements apply within each sequence.
        - ``derived_results``: Consumes existing estimates or summaries, such as
          a vector of p-values or a completed rolling result.
    target : bool, default False
        True if the calculation requires an outcome aligned to the feature
        observations. The outcome may be directly observed or generated.
    covariates : bool, default False
        True if the calculation requires control columns in addition to the
        feature and target. Controls must align to the same observations.
    axis_order : str, default 'unused'
        Ordering requirement for an explicit coordinate column:

        - ``unused``: No ordering requirement on a coordinate column. Keys may
          still be required for identity, and techniques over vectors may use
          their supplied order.
        - ``sorted_by_calculation``: The function establishes coordinate order.
          For example, an IC can group unsorted input by exact coordinate and
          return its estimates in coordinate order.
        - ``strictly_increasing``: Each coordinate must be greater than the
          preceding one within the sequence. Repeated coordinates within the
          same sequence fail this requirement. Equal spacing is a separate
          constraint.
    entity_key : str, default 'unused'
        How the identifier of an observational unit or sequence is used:

        - ``unused``: The calculation does not require an entity key.
        - ``optional_unique_pair``: If an entity key is supplied, each
          entity-coordinate pair must occur at most once.
        - ``unique_pair``: Both keys are required and each entity-coordinate
          pair must occur at most once. Different entities may share the
          same coordinate.
        - ``single_entity``: The input must identify exactly one entity.

        An entity key identifies observations. It does not establish
        statistical independence.
    missing_policy : str, default ''
        Describes which measurements are excluded or rejected. For example,
        complete-pair filtering excludes a row if either measurement is
        missing, whereas a sequence test may reject internal gaps. An empty
        string means no additional policy is declared here. See the callable
        for its exact handling of nulls, NaN, and infinities.
    inference : str, default 'none'
        Procedure used in addition to the point calculation. Common values
        are ``none`` for descriptive output, ``newey_west`` for inference on an
        ordered coefficient series, ``moving_block_bootstrap`` for resampling
        consecutive observations, and ``wald`` for a test using the supplied
        estimate and standard error. ``augmented_dickey_fuller`` tests for a
        unit root, and ``stationarity_test`` identifies the stationarity
        wrapper. ``false_discovery_rate`` identifies a procedure with a
        configurable correction method. ``benjamini_hochberg`` and
        ``benjamini_yekutieli`` identify the specific corrections. Check the
        callable for the method's assumptions and conditions for defined output.
    value_type : {'numeric'}, default 'numeric'
        Representation of the measurements or statistics consumed. This
        applies to data values, not entity identifiers. A category label
        needs a meaningful numeric representation before numeric analysis.
    minimum_complete_pairs : int or None, default None
        Fixed lower bound on aligned feature-target pairs after filtering.
        It applies to the unit being evaluated, such as one section or
        sequence. Meeting it does not guarantee a defined estimate, for
        example when either variable is constant. None means there is no
        fixed minimum declared here. The callable may base its minimum on
        parameters such as the number of controls.
    constraints : tuple of str, default ()
        Additional requirements that depend on the technique or its
        parameters. Examples include a block length that fits the sample,
        unique positive horizons, and compatible units for cost comparisons.
        These descriptions accompany the callable's own validation.

    Returns
    -------
    EvaluationRequirements
        Immutable declaration accompanying the function's existing validators.

    Raises
    ------
    None
        This declaration does not validate input data.
    """

    layout: Literal['pooled', 'cross_sectional', 'single_sequence', 'multiple_sequences', 'derived_results']
    target: bool = False
    covariates: bool = False
    axis_order: Literal['unused', 'sorted_by_calculation', 'strictly_increasing'] = 'unused'
    entity_key: Literal['unused', 'optional_unique_pair', 'unique_pair', 'single_entity'] = 'unused'
    missing_policy: str = ''
    inference: str = 'none'
    value_type: Literal['numeric'] = 'numeric'
    minimum_complete_pairs: int | None = None
    constraints: tuple[str, ...] = ()


_sequence = EvaluationRequirements(
    'single_sequence', target=True, axis_order='strictly_increasing', entity_key='single_entity',
    missing_policy='Use jointly finite pairs after validating the observation axis.',
    minimum_complete_pairs=2)
_ic_summary = EvaluationRequirements(
    'cross_sectional', target=True, axis_order='sorted_by_calculation',
    entity_key='optional_unique_pair', missing_policy='Use complete pairs within each axis value.',
    inference='newey_west', minimum_complete_pairs=2)
_temporal_summary = replace(_sequence, inference='moving_block_bootstrap',
                            constraints=('Use consecutive eligible observations and a block length that fits the sample.',))
_decay = ('Use distinct positive horizons and aligned targets for each requested horizon.',)
_derived = EvaluationRequirements('derived_results')

EVALUATION_REQUIREMENTS = MappingProxyType({
    'information_coefficient': EvaluationRequirements(
        'pooled', target=True, missing_policy='Use complete pairs; reject infinite values.',
        minimum_complete_pairs=2),
    'compute_ic': EvaluationRequirements(
        'cross_sectional', target=True, axis_order='sorted_by_calculation',
        entity_key='optional_unique_pair', missing_policy='Use complete pairs within each axis value.',
        minimum_complete_pairs=2),
    'partial_information_coefficient': EvaluationRequirements(
        'cross_sectional', target=True, covariates=True, axis_order='sorted_by_calculation',
        entity_key='optional_unique_pair', missing_policy='Use complete rows across signal, target, and controls.'),
    'ic_summary_table': _ic_summary,
    'partial_ic_summary_table': EvaluationRequirements(
        'cross_sectional', target=True, covariates=True, axis_order='sorted_by_calculation',
        entity_key='optional_unique_pair', missing_policy='Use complete rows across signal, target, and controls.',
        inference='newey_west'),
    'temporal_association': _sequence,
    'distribution_summary': EvaluationRequirements(
        'pooled', entity_key='unique_pair', missing_policy='Summarize finite numeric values.'),
    'rolling_distribution_summary': EvaluationRequirements(
        'single_sequence', axis_order='strictly_increasing', entity_key='single_entity',
        missing_policy='Summarize finite numeric values within each observation window.'),
    'partial_correlation': EvaluationRequirements(
        'pooled', target=True, covariates=True, missing_policy='Use jointly finite rows.',
        constraints=('Complete rows must support the control model and the requested minimum sample size.',)),
    'compute_ic_metrics': replace(_derived, constraints=('Supply an IC series to summarize.',)),
    'ic_decay': replace(_ic_summary, constraints=_decay),
    'ic_decay_summary': replace(_derived, constraints=('Supply IC series indexed by horizon.',)),
    'ic_decay_summary_table': replace(_ic_summary, constraints=_decay),
    'partial_temporal_association': replace(
        _sequence, covariates=True, minimum_complete_pairs=None, missing_policy='Use jointly finite rows including controls.',
        constraints=('Complete rows must support the control model and the requested minimum sample size.',)),
    'temporal_association_summary_table': _temporal_summary,
    'partial_temporal_association_summary_table': replace(
        _temporal_summary, covariates=True, minimum_complete_pairs=None,
        missing_policy='Use jointly finite rows including controls.'),
    'summarize_rolling_temporal_association': replace(
        _derived, constraints=('Supply a completed rolling temporal association result.',)),
    'rolling_temporal_association': replace(
        _temporal_summary, constraints=('Use complete observation windows and a block length that fits each window.',)),
    'temporal_association_decay': replace(_temporal_summary, constraints=_temporal_summary.constraints + _decay),
    'temporal_association_decay_summary': replace(
        _derived, constraints=('Supply temporal association estimates indexed by horizon.',)),
    'temporal_association_decay_summary_table': replace(
        _temporal_summary, constraints=_temporal_summary.constraints + _decay),
    'feature_target_relationship': EvaluationRequirements(
        'pooled', target=True, missing_policy='Use finite pairs.',
        constraints=('At least the requested number of bins in finite pairs are required.',)),
    'temporal_feature_target_relationship_uncertainty': replace(
        _temporal_summary, missing_policy='Every supplied pair must be finite.',
        constraints=('Use the declared feature bins and a block length that fits the selected sequence.',)),
    'detect_change_points': EvaluationRequirements(
        'single_sequence', axis_order='strictly_increasing', missing_policy='All observations must be finite.',
        constraints=('Observation order defines the sequence. Minimum segment size and jump are measured in observations.',)),
    'economic_return_contrast': replace(
        _derived, constraints=('Use comparable target return units and the selected bin or group contrast.',)),
    'preliminary_cost_margin': replace(
        _derived, constraints=('Use a return contrast and compatible cost estimates in the same units.',)),
    'newey_west_tstat': EvaluationRequirements(
        'single_sequence', inference='newey_west',
        missing_policy='Trim missing endpoints; reject internal gaps and infinities.',
        constraints=('Supply an ordered coefficient series; observation order defines lag.',)),
    'wald_temporal_association_test': replace(
        _derived, inference='wald', constraints=('Supply a point estimate and valid bootstrap standard error.',)),
    'adf_test': EvaluationRequirements(
        'single_sequence', inference='augmented_dickey_fuller',
        missing_policy='Trim missing endpoints; reject internal gaps and infinities.',
        constraints=('Supply observations in sequence order and meet the minimum sample size.',)),
    'stationarity_test': EvaluationRequirements(
        'single_sequence', inference='stationarity_test', constraints=('Supply observations in sequence order.',)),
    'fdr_correction': replace(
        _derived, inference='false_discovery_rate', constraints=('Supply p-values and any declared testing-group keys.',)),
    'benjamini_hochberg': replace(
        _derived, inference='benjamini_hochberg', constraints=('Supply a vector of p-values.',)),
    'benjamini_yekutieli': replace(
        _derived, inference='benjamini_yekutieli', constraints=('Supply a vector of p-values.',)),
    'generate_moving_blocks': EvaluationRequirements(
        'single_sequence', constraints=('Supply ordered observations and positive block length and step.',)),
    'moving_block_bootstrap': EvaluationRequirements(
        'single_sequence', inference='moving_block_bootstrap',
        constraints=('Use ordered observations, valid contiguous blocks and an explicit replication count.',)),
    'bootstrap_metrics': replace(
        _derived, constraints=('Supply bootstrap statistics for one estimate and a valid confidence level.',)),
    'monte_carlo_error': replace(
        _derived, constraints=('Use reproducible bootstrap prefixes, candidate counts, and metric tolerances.',)),
    'generate_random_walk': EvaluationRequirements(
        'single_sequence', constraints=('Supply a positive sample size, finite initial value and positive step deviation.',)),
    'circular_shift': EvaluationRequirements(
        'single_sequence', constraints=('Supply an observation sequence and a valid circular shift.',)),
})
