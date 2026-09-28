# Cross-Sectional Equity Alpha Modeling

## Overview

This project studies whether a small set of daily price, momentum, volatility, market-beta, and trading-activity features can predict **relative next-day returns** across a cross-section of large U.S. equities.

The models considered are:

- Ordinary Least Squares (OLS)
- Ridge regression
- Elastic Net
- XGBoost

The project is designed as a disciplined empirical research workflow rather than an exercise in maximizing backtest performance. Features are fixed before the final test, hyperparameters are selected only on a separate validation period, and all model specifications are frozen before a final expanding-window walk-forward evaluation.

The main result is negative but informative:

> **The models produced small positive validation-period Rank IC point estimates, but those estimates were statistically weak. In the untouched 2022–2026 walk-forward sample, no model produced statistically detectable cross-sectional predictive ability. The corresponding daily long-short portfolios also exhibited high turnover, making the signals economically unattractive after transaction costs.**

This project therefore illustrates a central difficulty in quantitative research: an apparently promising validation result may be too noisy to represent persistent alpha.

---

# Research Question

The central research question is:

> **Can simple price, momentum, volatility, and trading-activity features predict relative next-day equity returns out of sample, and does nonlinear machine learning provide economically meaningful improvements over regularized linear models after accounting for statistical uncertainty, turnover, and transaction costs?**

The workflow is:

\[
\text{market data}
\rightarrow
\text{features}
\rightarrow
\text{return scores}
\rightarrow
\text{cross-sectional ranking}
\rightarrow
\text{walk-forward test}
\rightarrow
\text{portfolio construction}
\rightarrow
\text{transaction costs}
\rightarrow
\text{economic evaluation}.
\]

---

# Cross-Sectional Alpha Modeling

An alpha model attempts to identify information associated with future returns.

This project is **cross-sectional** rather than directional. The goal is not primarily to predict whether the entire market will rise tomorrow. Instead, the model asks:

> Which stocks are more likely to outperform or underperform the other stocks in the universe tomorrow?

For stock \(i\) at date \(t\), let

\[
X_{i,t}
=
(x_{1,i,t},x_{2,i,t},...,x_{p,i,t})
\]

denote the observed feature vector.

A model produces

\[
\hat y_{i,t+1}=f(X_{i,t}),
\]

which can be interpreted as a relative-return score.

The stocks are then ranked by \(\hat y_{i,t+1}\). Because the economic application depends primarily on relative ordering, **Spearman Rank Information Coefficient** is the main statistical metric.

---

# Data

Daily OHLCV data are downloaded from Yahoo Finance using `yfinance`.

The frozen download window is:

```text
2010-01-01 to 2026-08-27
```

The final modeling dataset ends on August 26, 2026 because each observation requires a next-trading-day return target.

## Equity universe

The trading universe consists of 30 large U.S. equities.

### Technology

- AAPL
- MSFT
- IBM
- INTC
- CSCO
- ORCL
- QCOM
- TXN

### Financials

- JPM
- BAC
- GS
- MS
- C
- WFC

### Healthcare

- JNJ
- PFE
- MRK
- AMGN
- UNH

### Consumer

- WMT
- HD
- MCD
- KO
- PEP
- PG
- NKE

### Energy / Industrials

- XOM
- CVX
- CAT
- BA

`SPY` is included separately as a broad-market reference for the trading calendar and rolling beta calculation.

The frozen dataset was verified to contain complete common trading-date coverage across all 31 symbols over the sample window.

---

# Chronological Research Design

The sample is divided chronologically:

| Period | Dates | Purpose |
|---|---|---|
| Training | 2010–2018 | Estimate model parameters |
| Validation | 2019–2021 | Select hyperparameters |
| Final test | 2022–Aug. 2026 | Frozen walk-forward evaluation |

After rolling-feature warm-up, the usable modeling sample contains:

| Split | Rows | Trading dates |
|---|---:|---:|
| Training | 66,120 | 2,204 |
| Validation | 22,710 | 757 |
| Test | 34,980 | 1,166 |

Random train/test splitting is deliberately avoided because the problem is temporal and random splitting could mix future and past information.

---

# Return Target

Daily simple returns are

\[
r_{i,t}
=
\frac{P_{i,t}}{P_{i,t-1}}-1.
\]

The next-trading-day return is

\[
r_{i,t+1}.
\]

The prediction target is the next-day return relative to the cross-sectional average:

\[
y_{i,t+1}
=
r_{i,t+1}
-
\bar r_{t+1},
\]

where

\[
\bar r_{t+1}
=
\frac{1}{N}
\sum_{j=1}^{N}
r_{j,t+1}.
\]

Thus,

\[
\frac{1}{N}
\sum_i y_{i,t+1}
\approx 0
\]

each day.

This target removes the common daily market component and focuses the models on **relative stock performance**.

The economic backtest is evaluated using actual forward stock returns rather than the demeaned target.

---

# Features

The project intentionally uses a small and interpretable set of nine features.

| Feature | Interpretation |
|---|---|
| `ret_1d` | Most recent one-day return |
| `mom_5` | 5-day price momentum |
| `mom_20` | 20-day price momentum |
| `mom_60` | 60-day price momentum |
| `vol_20` | 20-day trailing return volatility |
| `vol_60` | 60-day trailing return volatility |
| `price_ma20_gap` | Price relative to its 20-day moving average |
| `dollar_volume_z20` | Abnormal dollar volume relative to the previous 20 trading days |
| `beta_60` | 60-day rolling market beta relative to SPY |

## Momentum

For horizon \(k\),

\[
mom^{(k)}_{i,t}
=
\frac{P_{i,t}}{P_{i,t-k}}-1.
\]

## Rolling volatility

\[
vol^{(k)}_{i,t}
=
Std(r_{i,t-k+1},...,r_{i,t}).
\]

## Price-to-moving-average gap

\[
gap_{i,t}
=
\frac{P_{i,t}}{MA_{20,i,t}}-1.
\]

## Abnormal dollar volume

Dollar trading volume is approximated by

\[
DV_{i,t}
=
Close_{i,t}\times Volume_{i,t}.
\]

The log dollar volume is compared with the stock's **previous** 20 observations:

\[
z^{DV}_{i,t}
=
\frac{
\log DV_{i,t}
-
\mu_{i,t-20:t-1}
}{
\sigma_{i,t-20:t-1}
}.
\]

The historical window is explicitly lagged so that today's observation is not included in its own baseline.

## Rolling beta

\[
\beta_{i,t}
=
\frac{
Cov(r_i,r_{SPY})
}{
Var(r_{SPY})
}
\]

using a 60-day rolling window.

---

# Cross-Sectional Standardization

Features are standardized across stocks independently on each date:

\[
z_{i,t}
=
\frac{x_{i,t}-\bar x_t}{s_t}.
\]

This transformation makes each feature represent the stock's position relative to the rest of the universe on the same trading day.

No clipping is applied to the standardized features.

---

# Exploratory Findings

Target-related exploratory analysis uses the **training period only**.

The individual-feature mean Rank IC values are small:

| Feature | Mean Rank IC |
|---|---:|
| `mom_60_z` | 0.0026 |
| `dollar_volume_z20_z` | 0.0023 |
| `mom_20_z` | -0.0025 |
| `vol_60_z` | -0.0057 |
| `vol_20_z` | -0.0069 |
| `beta_60_z` | -0.0101 |
| `price_ma20_gap_z` | -0.0101 |
| `mom_5_z` | -0.0151 |
| `ret_1d_z` | -0.0178 |

The most visible standalone pattern is therefore weak **short-horizon reversal**:

\[
IC(ret_{1d},r_{t+1})<0.
\]

However, daily IC volatility is much larger than these means, so the individual signals are extremely noisy.

---

# Multicollinearity

Several predictors contain overlapping information.

Variance Inflation Factors on the training period are:

| Feature | VIF |
|---|---:|
| `price_ma20_gap_z` | 7.28 |
| `mom_20_z` | 4.73 |
| `vol_60_z` | 3.42 |
| `mom_5_z` | 2.74 |
| `vol_20_z` | 2.50 |
| `beta_60_z` | 2.21 |
| `mom_60_z` | 1.46 |
| `ret_1d_z` | 1.23 |
| `dollar_volume_z20_z` | 1.01 |

The strongest redundancy occurs between 20-day momentum and distance from the 20-day moving average.

This motivates the use of regularized linear models.

---

# Models

## Ordinary Least Squares

OLS estimates

\[
\hat y
=
\beta_0+\sum_{j=1}^{p}\beta_j x_j
\]

by minimizing

\[
\sum_i(y_i-X_i\beta)^2.
\]

It serves as the unregularized linear benchmark.

---

## Ridge Regression

Ridge adds an \(L_2\) penalty:

\[
\min_\beta
\left[
\sum_i(y_i-X_i\beta)^2
+
\alpha\sum_j\beta_j^2
\right].
\]

The validation-selected value is

\[
\boxed{\alpha=300{,}000}.
\]

Because scikit-learn's Ridge squared-error term is not normalized by sample size, keeping the same numerical \(\alpha\) as the expanding training set grows corresponds to weaker effective regularization later in the walk-forward test. The primary experiment keeps the originally selected numerical value fixed rather than changing the methodology after viewing test results.

---

## Elastic Net

Elastic Net combines \(L_1\) and \(L_2\) regularization.

The selected validation specification is

\[
\boxed{
\alpha=3\times10^{-5},
\qquad
l1\_ratio=0.9.
}
\]

The selected model retains non-zero coefficients primarily for:

- recent return;
- 5-day momentum;
- 20-day momentum;
- 60-day momentum.

The chosen `l1_ratio = 0.9` lies at the upper edge of the pre-specified search grid, so it is interpreted as evidence that the validation criterion favored relatively strong \(L_1\) regularization—not as evidence that 0.9 is a uniquely optimal value.

---

## XGBoost

XGBoost provides a nonlinear challenger to the linear models.

The validation grid varies:

```text
max_depth      ∈ {1, 2, 3}
learning_rate  ∈ {0.02, 0.05}
n_estimators   ∈ {100, 300}
```

with fixed regularization and subsampling controls.

The selected configuration is:

```text
max_depth         = 1
learning_rate     = 0.02
n_estimators      = 300
subsample         = 0.8
colsample_bytree  = 0.8
min_child_weight  = 10
reg_lambda        = 10
reg_alpha         = 0
```

A depth-1 tree is a decision stump. The fact that the selected configuration uses depth one shows that complex interactions were **not required for the highest observed validation score**. It does not establish that deeper interactions are unhelpful; some deeper specifications produced comparable validation results.

---

# Evaluation Metrics

## Mean Squared Error

\[
MSE
=
\frac{1}{N}
\sum_i
(y_i-\hat y_i)^2.
\]

## \(R^2\) relative to the zero forecast

Because the cross-sectional target is demeaned each day, the natural benchmark forecast is

\[
\hat y=0.
\]

The zero-benchmark statistic is

\[
R^2_{0}
=
1-
\frac{
\sum_i(y_i-\hat y_i)^2
}{
\sum_i y_i^2
}.
\]

For validation and test observations this is an out-of-sample \(R^2\).

Predictions are not forcibly demeaned before computing this metric, so it penalizes both relative-return forecast errors and common prediction-level errors.

## Information Coefficient

For each date,

\[
IC_t
=
Corr_i(
\hat y_{i,t+1},
y_{i,t+1}
).
\]

Both Pearson IC and Spearman Rank IC are reported.

Rank IC is the primary ranking metric.

---

# Statistical Inference

Mean IC values are noisy time-series estimates.

For the validation and test periods, the project therefore reports **Newey-West / HAC standard errors** for the daily Rank IC series.

For a model,

\[
H_0:E[IC_t]=0.
\]

Pairwise model comparisons use

\[
d_t=IC_{A,t}-IC_{B,t}
\]

and test

\[
H_0:E[d_t]=0.
\]

The validation-period inference is considered **descriptive rather than confirmatory**, because the same validation period is used to select hyperparameters from multiple candidate configurations.

The untouched final test provides the more credible assessment of generalization.

---

# Validation Results: 2019–2021

Observed validation metrics are:

| Model | \(R^2_0\) | Pearson IC | Rank IC | Positive Rank-IC days |
|---|---:|---:|---:|---:|
| XGBoost | 0.000260 | 0.02021 | **0.01552** | 52.05% |
| Ridge | -0.000018 | 0.00013 | 0.00932 | 52.05% |
| Elastic Net | -0.000101 | -0.00277 | 0.00863 | 51.78% |
| OLS | -0.000175 | -0.00123 | 0.00214 | 51.39% |

The raw ranking is therefore

\[
XGBoost > Ridge \approx ElasticNet > OLS
\]

in terms of observed mean validation Rank IC.

However, statistical uncertainty is large.

## Newey-West inference

| Model | Mean Rank IC | NW SE | NW t-stat | p-value |
|---|---:|---:|---:|---:|
| OLS | 0.00214 | 0.01016 | 0.21 | 0.833 |
| Ridge | 0.00932 | 0.01082 | 0.86 | 0.389 |
| Elastic Net | 0.00863 | 0.01071 | 0.81 | 0.421 |
| XGBoost | 0.01552 | 0.01052 | 1.48 | 0.140 |

None of the validation-period mean Rank IC estimates is statistically distinguishable from zero at conventional significance levels.

## Pairwise validation comparisons

| Comparison | Mean IC difference | NW t-stat | p-value |
|---|---:|---:|---:|
| Ridge − OLS | 0.00718 | 1.25 | 0.211 |
| Elastic Net − OLS | 0.00649 | 1.16 | 0.244 |
| XGBoost − Ridge | 0.00620 | 0.72 | 0.470 |

Thus, the observed validation ordering should be treated as a **model-selection point estimate**, not proof that regularization or nonlinear modeling establishes superior predictive ability.

---

# XGBoost Seed Sensitivity

Because XGBoost uses row and feature subsampling, the selected hyperparameters were refit using 10 different random seeds.

Validation mean Rank IC across seeds:

```text
mean = 0.01366
std  = 0.00152
min  = 0.01197
max  = 0.01591
```

The positive validation point estimate is therefore not solely the result of a favorable `random_state=42`.

This seed stability does **not** eliminate the much larger sampling uncertainty in the daily IC series or the optimism created by hyperparameter selection.

---

# XGBoost Feature Importance

Gain-based importance is broadly distributed across the nine predictors:

| Feature | Gain share |
|---|---:|
| `dollar_volume_z20_z` | 13.45% |
| `ret_1d_z` | 13.03% |
| `vol_60_z` | 12.11% |
| `vol_20_z` | 11.36% |
| `mom_20_z` | 11.32% |
| `mom_5_z` | 10.15% |
| `price_ma20_gap_z` | 9.58% |
| `beta_60_z` | 9.52% |
| `mom_60_z` | 9.48% |

No single feature dominates.

These are model-specific gain importances and should not be interpreted as causal effects or variance-explained shares.

---

# Prediction Correlation

Validation prediction correlations are:

| | OLS | Ridge | Elastic Net | XGBoost |
|---|---:|---:|---:|---:|
| OLS | 1.000 | 0.828 | 0.823 | 0.252 |
| Ridge | 0.828 | 1.000 | 0.926 | 0.315 |
| Elastic Net | 0.823 | 0.926 | 1.000 | 0.307 |
| XGBoost | 0.252 | 0.315 | 0.307 | 1.000 |

The three linear models produce highly related rankings.

XGBoost produces a materially different score structure, but the statistical tests above show that this difference does not establish incremental predictive ability.

---

# Final Walk-Forward Test

After validation, all model choices and hyperparameters are frozen.

The final 2022–2026 period uses an annual expanding-window procedure.

For each prediction year:

1. use only training observations whose targets were known before the first prediction date;
2. refit fresh copies of the frozen models;
3. predict all signal dates in that year;
4. add the completed historical period to the next expanding training sample.

For example, the 2022 model is trained using data through December 30, 2021; the 2023 model uses data through December 29, 2022; and so on.

No hyperparameter is selected using final-test performance.

---

# Final Test Results: 2022–August 2026

| Model | \(R^2_0\) | Pearson IC | Rank IC | Positive Rank-IC days |
|---|---:|---:|---:|---:|
| XGBoost | -0.000524 | -0.00777 | -0.00121 | 50.43% |
| Ridge | -0.000022 | -0.00371 | -0.00433 | 50.60% |
| Elastic Net | -0.000080 | -0.00578 | -0.00573 | 48.63% |
| OLS | -0.000223 | -0.01127 | -0.01307 | 48.63% |

All four models have negative mean test Rank IC and negative \(R^2_0\).

More importantly, the Rank IC estimates are not statistically distinguishable from zero.

## Newey-West inference

| Model | Mean Rank IC | NW SE | NW t-stat | p-value |
|---|---:|---:|---:|---:|
| OLS | -0.01307 | 0.00792 | -1.65 | 0.099 |
| Ridge | -0.00433 | 0.00877 | -0.49 | 0.622 |
| Elastic Net | -0.00573 | 0.00806 | -0.71 | 0.477 |
| XGBoost | -0.00121 | 0.00767 | -0.16 | 0.874 |

The correct conclusion is therefore:

\[
\boxed{
\text{No frozen model shows statistically detectable test-period Rank IC.}
}
\]

The project does **not** claim that a statistically established alpha existed in validation and later decayed. Rather, the positive validation point estimates were noisy and failed to translate into detectable future predictability.

## Pairwise final-test comparisons

| Comparison | Mean IC difference | NW t-stat | p-value |
|---|---:|---:|---:|
| Ridge − OLS | 0.00874 | 1.58 | 0.114 |
| Elastic Net − OLS | 0.00734 | 1.42 | 0.155 |
| XGBoost − Ridge | 0.00312 | 0.45 | 0.651 |

The data also do not establish statistically significant differences between the frozen model classes.

---

# Test Rank IC by Year

| Year | OLS | Ridge | Elastic Net | XGBoost |
|---|---:|---:|---:|---:|
| 2022 | -0.0187 | -0.0327 | -0.0236 | -0.0398 |
| 2023 | -0.0278 | -0.0156 | -0.0095 | -0.0145 |
| 2024 | -0.0133 | -0.0006 | -0.0019 | 0.0143 |
| 2025 | -0.0015 | 0.0186 | 0.0069 | 0.0365 |
| 2026* | 0.0008 | 0.0156 | 0.0025 | -0.0032 |

`2026*` is a partial year through August.

The sign and magnitude of the point estimates vary considerably through time, reinforcing the instability of short-horizon return relationships.

---

# Long-Short Portfolio

Predicted scores are converted into a simple daily ranking portfolio.

Each trading day:

- rank all 30 stocks by predicted score;
- long the top 20%;
- short the bottom 20%;
- equal-weight within each side.

With 30 stocks, this gives:

```text
6 long stocks
6 short stocks
```

Long exposure is

\[
+0.5
\]

and short exposure is

\[
-0.5.
\]

Therefore,

\[
\sum_i w_i=0
\]

and

\[
\sum_i|w_i|=1.
\]

The portfolio is dollar neutral with 100% gross risky exposure.

Gross next-day portfolio return is

\[
r_{p,t+1}
=
\sum_i w_{i,t}r_{i,t+1}.
\]

---

# Turnover and Transaction Costs

Turnover is approximated as the change in target portfolio weights:

\[
Turnover_t
=
\sum_i
|w_{i,t}-w_{i,t-1}|.
\]

Under this convention, a value of 1.0 means trading an amount equal to portfolio capital during the rebalance.

Transaction costs are

\[
Cost_t
=
c\times Turnover_t
\]

and net returns are

\[
r^{net}_{p,t+1}
=
r^{gross}_{p,t+1}
-
Cost_t.
\]

The backtest evaluates:

- 0 bps
- 5 bps
- 10 bps

per dollar traded.

Average daily turnover is high:

| Model | Avg. daily turnover | Annualized turnover |
|---|---:|---:|
| Ridge | 75.81% | 191.1x |
| OLS | 76.80% | 193.5x |
| XGBoost | 84.28% | 212.4x |
| Elastic Net | 90.72% | 228.6x |

---

# Portfolio Results

## Before transaction costs

| Model | Annualized return | Annualized vol | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| OLS | -5.37% | 10.46% | -0.475 | -23.54% |
| Ridge | -0.90% | 12.24% | -0.013 | -15.33% |
| Elastic Net | -4.58% | 11.14% | -0.365 | -21.33% |
| XGBoost | **0.09%** | 9.82% | **0.058** | -22.33% |

The best gross result is economically close to flat.

## At 5 bps per dollar traded

| Model | Annualized return | Annualized vol | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| OLS | -14.10% | 10.47% | -1.399 | -51.01% |
| Ridge | -9.93% | 12.24% | -0.793 | -42.02% |
| Elastic Net | -14.89% | 11.13% | -1.392 | -53.14% |
| XGBoost | -10.00% | 9.84% | -1.022 | -41.32% |

## At 10 bps per dollar traded

| Model | Annualized return | Annualized vol | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| OLS | -22.03% | 10.49% | -2.319 | -68.62% |
| Ridge | -18.14% | 12.25% | -1.573 | -61.53% |
| Elastic Net | -24.09% | 11.14% | -2.418 | -72.37% |
| XGBoost | -19.07% | 9.86% | -2.096 | -63.46% |

High turnover therefore overwhelms the already weak gross performance.

---

# Long and Short Contributions

Before transaction costs, arithmetic annualized contributions are approximately:

| Model | Long contribution | Short contribution | Gross contribution |
|---|---:|---:|---:|
| OLS | 4.48% | -9.45% | -4.97% |
| Ridge | 5.05% | -5.21% | -0.15% |
| Elastic Net | 1.66% | -5.73% | -4.07% |
| XGBoost | 6.66% | -6.09% | 0.57% |

The XGBoost portfolio's long side contributes positively, but the short side loses almost as much, leaving little gross spread.

---

# Backtest Audit

The main backtest mechanics are independently reconstructed inside the final notebook.

The checks verify:

- forward-return alignment from adjusted prices;
- zero net exposure;
- 100% gross exposure;
- stock-level portfolio-return aggregation;
- target-weight turnover;
- transaction-cost arithmetic.

All assertions pass to floating-point precision.

This reduces the likelihood that the reported results arise from an obvious alignment or portfolio-accounting bug.

---

# Execution-Timing Robustness

The primary backtest uses end-of-day features through date \(t\), including the closing price, while evaluating the close-to-close return from \(t\) to \(t+1\).

This effectively assumes implementation at or near the same close used to finish constructing the signal. That is optimistic because the official closing price is not known before that execution price is available.

The main backtest should therefore be interpreted as a **research backtest**, not a production execution simulation.

A post-hoc robustness check delays all model scores by one trading day.

| Model | Lagged Rank IC | NW t-stat | p-value |
|---|---:|---:|---:|
| OLS | -0.01022 | -1.28 | 0.202 |
| Ridge | -0.00341 | -0.39 | 0.699 |
| Elastic Net | -0.00309 | -0.38 | 0.706 |
| XGBoost | 0.00712 | 0.99 | 0.321 |

The statistical conclusion remains unchanged.

The XGBoost point estimate becomes positive after delaying the score, but remains statistically indistinguishable from zero.

Because this robustness check was added after inspecting the final test results, it is not treated as a new primary specification.

---

# Main Findings

### 1. Individual daily alpha signals are weak

Training-period feature ICs are small compared with their day-to-day variability.

Recent-return and short-horizon momentum features show a weak reversal tendency, but the signal-to-noise ratio is low.

### 2. The feature set contains meaningful multicollinearity

Momentum and moving-average features overlap substantially, motivating shrinkage methods.

### 3. Regularized and nonlinear models have higher validation point estimates, but the differences are not statistically established

Ridge, Elastic Net, and XGBoost all have higher observed validation Rank IC than OLS.

However, Newey-West inference does not establish that those differences are statistically different from zero or from one another.

### 4. XGBoost seed randomness is not the main source of the validation result

The selected XGBoost specification produces positive validation Rank IC across all ten tested random seeds.

The much larger uncertainty comes from sampling variation through time and validation model selection.

### 5. The final test does not detect persistent alpha

All four frozen models have test-period Rank IC estimates statistically indistinguishable from zero.

This is the central empirical result.

### 6. Economic performance is also weak

The best gross portfolio result is approximately flat.

### 7. Turnover is very high

Average daily turnover ranges from approximately 76% to 91%, causing even modest per-dollar trading costs to create a large performance drag.

---

# Interpretation

The project does **not** claim to discover a profitable trading strategy.

It instead demonstrates a disciplined process for deciding whether an apparently promising alpha result is sufficiently robust to believe.

The research sequence is:

```text
define a hypothesis
        ↓
build leakage-controlled features and targets
        ↓
fit interpretable linear baselines
        ↓
study multicollinearity and regularization
        ↓
test a nonlinear challenger
        ↓
select hyperparameters only on validation data
        ↓
quantify validation uncertainty
        ↓
freeze all model choices
        ↓
run a future walk-forward test
        ↓
test statistical significance
        ↓
construct an economic portfolio
        ↓
apply turnover and transaction costs
        ↓
audit the implementation
        ↓
accept or reject the evidence
```

For this experiment, the final conclusion is:

\[
\boxed{
\text{no statistically detectable persistent cross-sectional alpha}
}
\]

and

\[
\boxed{
\text{high turnover further reduces economic viability}.
}
\]

---

# Project Structure

```text
Equity-Alpha-Modeling/
│
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_eda_and_features.ipynb
│   ├── 03_linear_models.ipynb
│   ├── 04_xgboost.ipynb
│   └── 05_walkforward_backtest.ipynb
│
├── src/
│   ├── config.py
│   ├── metrics.py
│   ├── models.py
│   └── backtest.py
│
├── data/
│   ├── raw_prices.csv
│   ├── alpha_dataset_base.csv
│   ├── alpha_dataset.csv
│   ├── linear_model_params.json
│   ├── xgboost_params.json
│   ├── model_validation_predictions.csv
│   ├── oos_predictions.csv
│   ├── portfolio_returns.csv
│   └── portfolio_summary.csv
│
└── README.md
```

Large generated CSV files may be excluded from Git and recreated by running the notebooks sequentially.

---

# Notebook Guide

## `01_data_pipeline.ipynb`

- downloads the frozen OHLCV sample;
- checks ticker/date coverage;
- computes simple returns;
- builds the market trading calendar;
- constructs leakage-controlled next-day targets;
- constructs the cross-sectionally demeaned target;
- creates chronological train/validation/test labels.

## `02_eda_and_features.ipynb`

- constructs the nine predictors;
- standardizes them cross-sectionally by date;
- evaluates missingness and feature distributions;
- studies feature correlation;
- evaluates individual-feature IC using training data only.

## `03_linear_models.ipynb`

- establishes the zero-return benchmark;
- fits OLS;
- studies coefficient interpretation and VIF;
- tunes Ridge using validation Rank IC;
- tunes Elastic Net using validation Rank IC;
- freezes the selected linear hyperparameters.

## `04_xgboost.ipynb`

- introduces a nonlinear gradient-boosting challenger;
- performs a small validation-only hyperparameter search;
- evaluates seed sensitivity;
- compares linear and nonlinear predictions;
- applies HAC inference to validation Rank IC;
- evaluates feature gain importance;
- saves the frozen XGBoost specification.

## `05_walkforward_backtest.ipynb`

- performs annual expanding-window refits;
- generates frozen-model predictions for the final test;
- evaluates overall and yearly statistical performance;
- applies Newey-West inference;
- constructs long-short portfolios;
- measures turnover and transaction-cost sensitivity;
- independently audits backtest mechanics;
- compares validation and test results;
- performs a post-hoc one-day signal-delay robustness check.

---

# Reproducibility

The project uses a frozen Yahoo Finance endpoint:

```python
START_DATE = "2010-01-01"
END_DATE   = "2026-08-28"
```

Yahoo Finance treats `end` as exclusive, so the data end on August 27, 2026.

Run the notebooks in order:

```text
01_data_pipeline.ipynb
02_eda_and_features.ipynb
03_linear_models.ipynb
04_xgboost.ipynb
05_walkforward_backtest.ipynb
```

Validation-selected parameters are saved to JSON and reloaded by the final notebook rather than retuned on the test period.

---

# Key Python Libraries

- `numpy`
- `pandas`
- `yfinance`
- `scikit-learn`
- `xgboost`
- `statsmodels`
- `plotly`

---

# Limitations

## Fixed survivor universe

The 30-stock universe is a fixed set of large companies with long histories.

It is not a point-in-time historical index universe and therefore contains survivorship and selection bias.

## Small universe

Thirty stocks are sufficient for a compact research project but much smaller than a typical institutional equity universe.

## Price/volume-focused features

The model excludes many potentially relevant information sources, including fundamentals, earnings revisions, analyst data, options, news, order-book information, and alternative data.

## Validation model-selection optimism

Ridge, Elastic Net, and XGBoost hyperparameters are chosen using the validation period.

Therefore the best validation scores are upward-biased relative to a genuinely untouched sample. Validation HAC tests are reported descriptively rather than treated as confirmatory inference.

## Ridge penalty scaling

The numerical Ridge penalty is frozen while the expanding training sample grows.

Because scikit-learn's Ridge loss is not normalized by sample size in the same way as Elastic Net, effective shrinkage becomes weaker as the walk-forward training sample expands.

This was not altered after observing the final test.

## Extreme observations and squared-error objectives

Features are not winsorized, and the return target is not clipped or rank-transformed.

Squared-error-based models can therefore be influenced by extreme return observations, including periods such as March 2020.

Alternative robust targets or losses would constitute a separate research specification.

## Same-close execution assumption

The primary backtest uses end-of-day information and a close-to-close forward return, effectively assuming implementation near the same close used to finish constructing the feature vector.

The one-day-delay diagnostic addresses this concern only as a post-hoc robustness check.

## No explicit sector or beta neutrality

The portfolio is dollar neutral but not sector neutral or beta neutral.

Residual factor exposures may remain.

## Simplified turnover

Turnover is based on changes in target weights rather than drift-adjusted pre-trade weights.

## Simplified transaction costs

Costs are modeled as fixed basis points per dollar traded.

The backtest does not explicitly model:

- bid-ask spreads;
- nonlinear market impact;
- commissions;
- short-borrow fees;
- financing costs;
- interest earned on cash/collateral;
- liquidity constraints.

The reported Sharpe ratio therefore uses the modeled long-short returns without a detailed financing/cash-return model.

---

# Possible Extensions

Natural extensions include:

- a larger point-in-time equity universe;
- sector-neutral and beta-neutral ranking;
- longer-horizon or slower-turnover signals;
- turnover-aware portfolio optimization;
- robust/winsorized targets;
- rank-based prediction targets;
- fundamentals and earnings features;
- explicit next-open execution;
- cross-sectional factor residualization;
- richer transaction-cost modeling;
- risk scaling using a separate volatility forecast.

A natural connection to a volatility-forecasting project is:

\[
\text{return ranking}
+
\text{conditional risk forecast}
\rightarrow
\text{risk-scaled alpha portfolio}.
\]

---

# Conclusion

This project examines the full lifecycle of a cross-sectional alpha hypothesis.

Regularized linear models and XGBoost produce higher **observed validation Rank IC** than OLS, with XGBoost reaching approximately

\[
0.0155.
\]

However, Newey-West inference shows that the validation estimates are too noisy to establish statistically detectable predictive ability:

\[
t_{NW,XGB}\approx1.48,
\qquad
p\approx0.14.
\]

The final frozen walk-forward test is more decisive.

XGBoost's mean test Rank IC is approximately

\[
-0.0012
\]

with

\[
p\approx0.87,
\]

while Ridge, Elastic Net, and OLS are also statistically indistinguishable from zero.

The economic results agree with the statistical evidence. XGBoost is approximately flat before transaction costs, while average daily turnover of roughly 84% causes performance to deteriorate sharply under even modest cost assumptions.

The final conclusion is therefore:

\[
\boxed{
\text{the experiment does not find statistically detectable persistent alpha}
}
\]

and

\[
\boxed{
\text{the tested daily ranking strategies are not economically attractive under the modeled transaction-cost assumptions}.
}
\]

The most important result is methodological rather than profitable: validation rankings, model complexity, and attractive point estimates are not enough. A credible alpha process must survive statistical uncertainty, future walk-forward testing, implementation checks, and modeled trading costs.
