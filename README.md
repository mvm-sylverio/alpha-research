# alpha-research

[![PyPI version](https://img.shields.io/pypi/v/alpha-research)](https://pypi.org/project/alpha-research/)
[![Python versions](https://img.shields.io/pypi/pyversions/alpha-research)](https://pypi.org/project/alpha-research/)
[![License](https://img.shields.io/pypi/l/alpha-research)](https://github.com/mvm-sylverio/alpha-research/blob/main/LICENSE)

`alpha-research` is a reusable Python library for quantitative signal research.
It provides feature and target construction, cross-sectional IC analysis,
single-asset temporal association diagnostics, and dependency-aware bootstrap
utilities. It is intentionally independent of databases, orchestration, and
backtesting applications.

The library supports both Pandas and Polars. Public functions preserve the
input DataFrame backend whenever practical.

## What it provides

- Feature and target utilities: simple/log returns, forward returns, OHLC
  Triple-Barrier labels, trend features, volatility measures including realized
  volatility and ATR, and ADX.
- Feature transformations: cross-sectional rank, z-score, and rank-based
  quantile groups; trailing per-asset rank, z-score, and rank-based quantile
  groups.
- Research data utilities: feature/target schema helpers and purged
  train/test splitting.
- Cross-sectional information coefficient analysis with Pearson or Spearman
  estimators, partial IC, Newey-West statistics, FDR correction, and IC decay.
- Single-asset temporal association with Moving Block Bootstrap (MBB), Wald
  diagnostics, partial association, decay analysis, bootstrap directional
  stability and FDR correction.
- Rolling temporal association with percentile bootstrap bands.
- Visualization utilities for ranked IC and temporal-association summaries,
  partial summaries with visible covariates, decay curves, rolling temporal
  association, time-series feature values, and cross-sectional feature summaries.
- Resampling utilities for moving-block bootstrap summaries and bootstrap
  Monte Carlo convergence diagnostics.
- ADF stationarity testing.

## Installation

Install the library and its core dependencies:

```bash
pip install alpha-research
```

Requires Python 3.11 or later.

Install Matplotlib support when using plotting functions:

```bash
pip install "alpha-research[viz]"
```

Install MetaTrader 5 support for OHLCV data ingestion:

```bash
pip install "alpha-research[mt5]"
```

## Cross-sectional IC

Cross-sectional IC measures the association between a feature and a forward
target across assets at each timestamp. The example below creates one feature,
joins it to a forward-return target, calculates IC diagnostics, and applies FDR
correction to the resulting feature family.

```python
from alpha_research.evaluation.ic import ic_summary_table
from alpha_research.evaluation.statistical_tests import fdr_correction
from alpha_research.features.schema import join_feature_target_frames
from alpha_research.features.targets import fwd_returns
from alpha_research.features.trend import price_to_sma_ratio

# ohlcv contains: time, symbol, close
feature_frame = price_to_sma_ratio(ohlcv, window=20)
target_frame = fwd_returns(ohlcv, horizon=10)

research_frame = join_feature_target_frames(
    feature_df=feature_frame,
    target_df=target_frame,
    feature_col='price_to_sma_ratio_20',
    target_col='fwd_ret_10',
    time_col='time',
    symbol_col='symbol',
)

ic_result = ic_summary_table(
    research_frame,
    feature_list=['price_to_sma_ratio_20'],
    target='fwd_ret_10',
    feature_groups={'price_to_sma_ratio_20': 'trend'},
)
ic_table = fdr_correction(ic_result.table, method='bh')
```

`ic_result.ic_frames` retains the per-date IC series used by the summary.

## Partial cross-sectional IC

Partial IC answers a narrower question: whether a feature retains a
cross-sectional association with the target after controlling for one or more
covariates at every timestamp. For Spearman partial IC, feature, target, and
covariates are ranked within each cross-section before the controls are removed.

```python
from alpha_research.evaluation.ic import partial_ic_summary_table
from alpha_research.evaluation.statistical_tests import fdr_correction
from alpha_research.visualization import plot_partial_ic_summary

# research_frame also contains market_return and realized_volatility.
partial_ic_result = partial_ic_summary_table(
    research_frame,
    feature_list=['price_to_sma_ratio_20'],
    target='fwd_ret_10',
    covariates=['market_return', 'realized_volatility'],
    feature_groups={'price_to_sma_ratio_20': 'trend'},
    corr_method='spearman',
)
partial_ic_table = fdr_correction(partial_ic_result.table, method='bh')
ax = plot_partial_ic_summary(partial_ic_table, top_n=20)
```

The returned axis identifies the covariates used by the analysis. As with
ordinary IC, `partial_ic_result.partial_ic_frames` retains the per-date series,
and its mean is tested with the existing Newey-West procedure.

## Ranked summary plots

Use the ranked plots to inspect many candidates without printing a wide table.
They return a Matplotlib `Axes`, so applications and notebooks can supply
`ax=...` to compose their own figures.

```python
from alpha_research.visualization import (
    plot_ic_summary,
    plot_temporal_association_summary,
)

ax = plot_ic_summary(ic_table, top_n=20)
ax = plot_temporal_association_summary(summary, top_n=20)
```

The IC plot shows point estimates because IC summaries do not provide a
confidence interval. The temporal-association plot is a forest plot using the
existing Wald intervals. Both prefer `fdr_rejected` when FDR has been applied;
the temporal plot otherwise uses its individual `reject_h0` Wald decision.

## Single-asset temporal association

Temporal association measures feature-target correlation through time for one
asset. It is distinct from cross-sectional IC: each observation is a paired
feature and aligned target at one point in time.

The global summary uses MBB, so feature and target are resampled together in
contiguous blocks rather than as IID observations.

```python
from alpha_research.evaluation.timeseries import temporal_association_summary_table

# single_asset_frame contains one symbol and: time, symbol, feature, fwd_ret_20
summary = temporal_association_summary_table(
    single_asset_frame,
    feature_list=['feature'],
    target='fwd_ret_20',
    block_length=20,
    n_bootstraps=2_000,
    corr_method='spearman',
    random_state=42,
)
```

The bootstrap sign proportion is a directional-stability diagnostic. It is not
a p-value. The Wald statistics in this global summary are separate from the
percentile bootstrap intervals used by the rolling analysis below.

## Partial temporal association

Partial temporal association measures the relationship through time after
conditioning both the feature and target on one or more covariates. It retains
the single-symbol, strictly ordered-time contract of temporal association.

```python
from alpha_research.evaluation.statistical_tests import fdr_correction
from alpha_research.evaluation.timeseries import (
    partial_temporal_association_summary_table,
)
from alpha_research.visualization import (
    plot_partial_temporal_association_summary,
)

# single_asset_frame also contains market_return and realized_volatility.
partial_summary = partial_temporal_association_summary_table(
    single_asset_frame,
    feature_list=['feature_a', 'feature_b'],
    target='fwd_ret_20',
    covariates=['market_return', 'realized_volatility'],
    block_length=20,
    n_bootstraps=2_000,
    corr_method='spearman',
    random_state=42,
)
partial_summary = fdr_correction(partial_summary, method='bh')
ax = plot_partial_temporal_association_summary(partial_summary, top_n=20)
```

MBB resamples complete feature-target-covariate rows in contiguous blocks. The
plot shows the existing Wald intervals and explicitly records the covariates
conditioned on.

## Rolling temporal association

`rolling_temporal_association()` evaluates local temporal association in full,
strict windows. A window with a missing feature-target pair is recorded with a
status instead of compressing the time axis before block bootstrap.

```python
from alpha_research.evaluation.timeseries import rolling_temporal_association
from alpha_research.visualization import plot_rolling_temporal_association

rolling_result = rolling_temporal_association(
    single_asset_frame,
    feature='feature',
    target='fwd_ret_20',
    window_size=252,
    window_step=20,
    block_length=20,
    n_bootstraps=2_000,
    bootstrap_method='moving_block',
    confidence_level=0.95,
    random_state=42,
)

rolling_frame = rolling_result.rolling_frame
rolling_summary = rolling_result.summary_table
ax = plot_rolling_temporal_association(rolling_frame)
```

Each rolling row contains the observed association, percentile bootstrap bounds,
bootstrap directional stability, effective bootstrap count, and a computation
status. The plot function accepts `ax=...`, allowing an application to compose
the association panel with its own shared-time context panels.

## Association and IC decay

Decay evaluates one fixed feature over several forward-target horizons. Both
IC decay and temporal-association decay apply FDR jointly across the requested
horizons for that feature. Temporal-association decay additionally uses MBB and
Wald diagnostics at each horizon.

```python
import matplotlib.pyplot as plt

from alpha_research.evaluation.timeseries import (
    temporal_association_decay,
    temporal_association_decay_summary,
)
from alpha_research.features.targets import fwd_returns
from alpha_research.visualization import plot_decay_curves

# single_asset_feature_frame contains time, symbol, and feature.
# single_asset_ohlcv contains time, symbol, and close.
temporal_decay = temporal_association_decay(
    df_feature=single_asset_feature_frame,
    feature='feature',
    target_data=single_asset_ohlcv,
    horizons=[1, 5, 10, 20, 60],
    target_fn=fwd_returns,
    block_length=20,
    n_bootstraps=2_000,
    corr_method='spearman',
    random_state=42,
)
temporal_decay_metrics = temporal_association_decay_summary(
    temporal_decay.table,
)

_, ax = plt.subplots()
plot_decay_curves(
    {'feature': temporal_decay.table},
    value_col='association',
    ci_lower_col='wald_ci_lower',
    ci_upper_col='wald_ci_upper',
    ax=ax,
)
ax.set_xlabel('Forward horizon')
ax.set_ylabel('Temporal association')
ax.legend()
```

`temporal_decay_metrics` reports peak magnitude, discrete half-life, the last
FDR-significant horizon, and area under the absolute curve. For IC decay, pass
the tables returned by `ic_decay()` to `plot_decay_curves()` with
`value_col='mean'`; IC-decay standard deviation is not treated as an interval.
For multiple features, `temporal_association_decay_summary_table()` returns the
scalar summary table together with each feature's complete horizon-level result.

## Scope

This project documents implemented research capabilities rather than a fixed
roadmap. New methods are added when they fit the library's statistical and
architectural boundaries.

## References

- *Active Portfolio Management* — Grinold & Kahn
- *Advances in Financial Machine Learning* — Marcos López de Prado
- *Machine Learning for Asset Managers* — Marcos López de Prado

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE).
