# alpha-research
Systematic alpha research framework focused on signal discovery, validation, and robustness.

Inspired by concepts from:

- *Active Portfolio Management* — Grinold & Kahn
- *Advances in Financial Machine Learning* — Marcos López de Prado
- *Machine Learning for Asset Managers* — Marcos López de Prado

## Stack

Python - Pandas - Polars - NumPy - Scipy - Statsmodels

Both Pandas and Polars are supported as backends. Pandas is supported because of its widespread usage in the community and Polars for its performance for bigger tasks. Pandas is treated without index, like polars.

## Pipeline
Completed modules are marked; remaining items reflect the planned roadmap.

### Pre-Research
- [ ] Data ingestion
- [ ] Data cleaning
- [ ] Volume bars / dollar bars
- [ ] Exploratory data analysis (EDA)
- [ ] Feature engineering and preprocessing - continuous implementation - basic are tracked here
  - **Features** (partial — expanding)
    - [x] Simple and Log return
    - [x] Forward return (target)
    - [x] Price-to-SMA ratio and SMA crossover
    - [ ] RSI, ATR, realized volatility *(planned)*
    - [ ] Composite and interaction features *(planned)*
  - **Preprocessing transformations** (planned)
    - [x] Rank
    - [ ] Z-score, winsorization, volatility scaling, volume scaling *(planned)*
- [x] Stationarity testing (ADF) - for time-series models and fractional differentiation
- [ ] Triple Barrier Method (labeling) - planned in targets module
- [ ] Event-based sampling (CUSUM filter)

### Research — "Does the signal exist?"
- [x] Research-test df split function with purging.

**Foundation — IC Analysis**
- [x] Cross-sectional Information Coefficient (IC) — Spearman and Pearson
- [x] IC metrics: mean, stability (IC IR), pct positive, quantiles, signal alignment
- [x] Newey-West HAC t-statistic for IC significance
- [x] IC summary table across feature universe with feature group classification
- [ ] Rolling IC stability
- [x] IC decay analysis
- [ ] Plots

**Multiple Testing Correction**
- [x] Benjamini-Hochberg correction per signal family
- [x] Benjamini-Yekutieli correction for arbitrary dependence

**Causality — "Is the signal real or spurious? Why does it work?"**

- [ ] Granger causality — linear filter
- [ ] PCMCI — causal discovery with multiple confounders
- [ ] Transfer entropy — non-linear relationships

**Purged K-Fold construction**
- [ ] K-fold split with purging and embargo (López de Prado)

**Structure — "Are features redundant?"**
- [ ] Feature correlation matrix
- [ ] Feature clustering (hierarchical, distance-based)
- [ ] PCA / orthogonalization (optional, for multicollinearity)

**Importance — "Which features really matter?"**
- [ ] Single Feature Importance (SFI)
- [ ] Mean Decrease Impurity (MDI)
- [ ] Mean Decrease Accuracy / Permutation Importance (MDA)

**Interpretability — "Why do they matter and when?"**
- [ ] SHAP values
- [ ] Rolling SHAP for regime detection

**Regime Detection**
- [ ] Hidden Markov Model (HMM) — latent volatility/return states
- [ ] Changepoint detection — structural distribution shifts
- [ ] Regime clustering (K-means on volatility and correlation features)
- [ ] Regime as meta-feature

**Advanced**
- [ ] Signature features (Rough Path Theory)
- [ ] Topological Data Analysis (TDA) — persistent homology
- [ ] Hawkes processes — temporal event clustering

### Out-of-Sample Validation
Final validation on the held-out period defined at the start of research.
- [ ] Expanding window walk-forward
- [ ] IC stability check out-of-sample
- [ ] Factor regression — residual alpha with t-stat > 2

### Backtesting
- [ ] Cross-sectional portfolio construction
- [ ] Metalabeling
- [ ] Position sizing
- [ ] Simulated P&L
- [ ] Turnover analysis
- [ ] Alpha evaluation
- [ ] Sharpe ratio
- [ ] Deflated Sharpe Ratio (DSR)

## Usage

```python
import pandas as pd
from alpha_research.features.returns import simple_returns, log_returns
from alpha_research.features.targets import fwd_returns
from alpha_research.features.trend import price_to_sma_ratio, sma_crossover
from alpha_research.features.transformations import cross_sectional_rank
from alpha_research.evaluation.ic import compute_ic, ic_summary_table
from alpha_research.evaluation.statistical_tests import benjamini_hochberg

# 1. Compute raw features from OHLCV
df_ohlcv = pd.DataFrame()  # your OHLCV data

ret_5    = simple_returns(df_ohlcv, horizon=5)
sma_ratio = price_to_sma_ratio(df_ohlcv, window=20)
target   = fwd_returns(df_ohlcv, horizon=10)

# 2. Merge features and target into research DataFrame
df = (ret_5
    .merge(sma_ratio, on=['time', 'symbol'])
    .merge(target,    on=['time', 'symbol'])
    .dropna()
)

# 3. Apply cross-sectional rank transformation
df = cross_sectional_rank(df, feature_cols=['simple_ret_5', 'price_to_sma_ratio_20'])

# 4. Compute IC summary table with feature group classification
feature_groups = {
    'simple_ret_5_rank': 'momentum',
    'price_to_sma_ratio_20_rank': 'trend',
}

table = ic_summary_table(
    df,
    feature_list=['simple_ret_5_rank', 'price_to_sma_ratio_20_rank'],
    target="fwd_ret_10",
    feature_groups=feature_groups,
)

# 5. Apply FDR correction per signal family with Benjamini-Hochberg correction
fdr_corrected_table = benjamini_hochberg(table)

significant_features = fdr_corrected_table[fdr_corrected_table['fdr_rejected']]
```