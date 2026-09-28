# Cross-Sectional Equity Alpha Modeling

## Overview

This project studies whether a small set of daily price, momentum, volatility, market-beta, and trading-activity features can predict **relative next-day equity returns** across a cross-section of large U.S. stocks.

The analysis compares four models:

- Ordinary Least Squares (OLS)
- Ridge regression
- Elastic Net
- XGBoost

The project is designed as a research workflow rather than an exercise in maximizing a backtest. Model choices are made using a training/validation framework and then frozen before a final expanding-window out-of-sample test.

The central empirical result is that the models show **weak but positive validation-period ranking ability**, with shallow XGBoost performing best, but this apparent predictability does **not persist reliably in the untouched 2022-2026 test period**. The daily long-short portfolios also exhibit high turnover, so transaction costs eliminate the small gross performance that remains.

> A signal can look statistically promising during development and still fail to generalize or survive trading frictions.

---

## Research Question

> **Can simple price, momentum, volatility, and trading-activity features predict relative next-day equity returns out of sample, and does nonlinear machine learning provide economically meaningful improvements over regularized linear models after accounting for turnover and transaction costs?**

The workflow is:

\[
\text{market data}
\rightarrow
\text{features}
\rightarrow
\text{return forecast}
\rightarrow
\text{cross-sectional ranking}
\rightarrow
\text{portfolio construction}
\rightarrow
\text{transaction costs}
\rightarrow
\text{economic outcome}.
\]

---

# Cross-Sectional Alpha Modeling

An alpha model attempts to identify information associated with future returns. This project is specifically **cross-sectional**: rather than forecasting whether the entire market rises tomorrow, it attempts to rank stocks by their expected performance relative to one another.

For stock \(i\) on date \(t\), the feature vector is

\[
X_{i,t}=(x_{1,i,t},x_{2,i,t},...,x_{p,i,t}),
\]

and a model produces

\[
\hat y_{i,t+1}=f(X_{i,t}).
\]

The exact numerical return forecast need not be highly accurate for the signal to be useful. If the model consistently ranks future winners above future losers, the ranking may still have economic value. This motivates the use of **Information Coefficient**, especially Spearman Rank IC, alongside conventional squared-error metrics.

---

# Data

Daily OHLCV data are downloaded with `yfinance`.

## Trading universe

The stock universe contains 30 large U.S. equities:

**Technology:** AAPL, MSFT, IBM, INTC, CSCO, ORCL, QCOM, TXN  
**Financials:** JPM, BAC, GS, MS, C, WFC  
**Healthcare:** JNJ, PFE, MRK, AMGN, UNH  
**Consumer:** WMT, HD, MCD, KO, PEP, PG, NKE  
**Energy / Industrials:** XOM, CVX, CAT, BA

`SPY` is downloaded separately as a market reference for the trading calendar and rolling beta feature.

The frozen download runs from January 2010 through August 27, 2026. The configured Yahoo Finance end date is `2026-08-28`, which is exclusive. The modeling dataset ends one trading day earlier because a valid next-day target is required.

Adjusted close prices are used for return calculations. Raw close prices are used for dollar-volume calculations.

---

# Chronological Research Design

| Period | Dates | Purpose |
|---|---|---|
| Training | 2010-2018 | Fit model parameters |
| Validation | 2019-2021 | Select hyperparameters |
| Final test | 2022-Aug 2026 | Untouched walk-forward evaluation |

Random train-test splitting is deliberately avoided because this is a time-ordered forecasting problem. The final test period is not used for feature selection or hyperparameter tuning.

---

# Return Target

Daily simple returns are

\[
r_{i,t}=\frac{P_{i,t}}{P_{i,t-1}}-1.
\]

The one-day forward return is \(r_{i,t+1}\). The modeling target is the stock's next-day return relative to the cross-sectional average:

\[
y_{i,t+1}=r_{i,t+1}-\bar r_{t+1},
\]

where

\[
\bar r_{t+1}=\frac{1}{N}\sum_{j=1}^{N}r_{j,t+1}.
\]

This focuses the model on **relative stock performance** rather than the direction of the whole market. Portfolio returns are still evaluated using the actual raw forward stock returns.

A market-calendar check is used so that one-day returns and one-day targets correspond to consecutive market trading dates.

---

# Feature Engineering

The project intentionally uses a compact, interpretable set of nine predictors.

| Feature | Interpretation |
|---|---|
| `ret_1d` | Most recent one-day return |
| `mom_5` | 5-day price momentum |
| `mom_20` | 20-day price momentum |
| `mom_60` | 60-day price momentum |
| `vol_20` | 20-day rolling return volatility |
| `vol_60` | 60-day rolling return volatility |
| `price_ma20_gap` | Price relative to its 20-day moving average |
| `dollar_volume_z20` | Abnormal dollar trading activity versus the previous 20 days |
| `beta_60` | 60-day rolling beta relative to SPY |

### Momentum

\[
mom^{(k)}_{i,t}=\frac{P_{i,t}}{P_{i,t-k}}-1.
\]

### Rolling volatility

\[
vol^{(k)}_{i,t}=Std(r_{i,t-k+1},...,r_{i,t}).
\]

### Price-to-moving-average gap

\[
gap_{i,t}=\frac{P_{i,t}}{MA_{20,i,t}}-1.
\]

### Abnormal dollar volume

Dollar trading volume is approximated as

\[
DV_{i,t}=Close_{i,t}\times Volume_{i,t}.
\]

Today's log dollar volume is standardized relative to the **previous** 20 trading days:

\[
z^{DV}_{i,t}=\frac{\log DV_{i,t}-\mu_{i,t-20:t-1}}{\sigma_{i,t-20:t-1}}.
\]

### Rolling beta

\[
\beta_{i,t}=\frac{Cov(r_i,r_{SPY})}{Var(r_{SPY})}
\]

using a 60-day rolling window.

---

# Cross-Sectional Standardization

Each feature is standardized across stocks separately on each date:

\[
z_{i,t}=\frac{x_{i,t}-\bar x_t}{s_t}.
\]

This makes the inputs relative: for example, a high `mom_20_z` means a stock has stronger 20-day momentum than most stocks in the universe that day.

---

# Exploratory Findings

## Individual-feature Rank IC

Using the training period only:

| Feature | Mean Rank IC |
|---|---:|
| `mom_60_z` | 0.00257 |
| `dollar_volume_z20_z` | 0.00231 |
| `mom_20_z` | -0.00250 |
| `vol_60_z` | -0.00573 |
| `vol_20_z` | -0.00693 |
| `beta_60_z` | -0.01005 |
| `price_ma20_gap_z` | -0.01013 |
| `mom_5_z` | -0.01512 |
| `ret_1d_z` | -0.01785 |

The clearest standalone pattern is weak **short-horizon reversal**: recent relative winners tend, on average, to underperform slightly on the following day. However, daily IC dispersion is far larger than the mean effects, so the signal-to-noise ratio is very low.

---

# Multicollinearity

Several features contain overlapping information, especially momentum and price-position variables.

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

This motivates testing regularized linear models rather than relying only on OLS.

---

# Models

## OLS

\[
\hat y=\beta_0+\sum_{j=1}^{p}\beta_jx_j.
\]

OLS is the simplest multivariate benchmark and provides interpretable coefficients, but correlated predictors can make individual coefficients unstable.

## Ridge

Ridge adds an \(L_2\) penalty:

\[
\min_\beta\left[\sum_i(y_i-X_i\beta)^2+\alpha\sum_j\beta_j^2\right].
\]

The validation-selected penalty is

\[
\boxed{\alpha=300{,}000}.
\]

## Elastic Net

Elastic Net combines \(L_1\) and \(L_2\) regularization. The selected hyperparameters are

\[
\boxed{\alpha=3\times10^{-5}},\qquad \boxed{l1\_ratio=0.9}.
\]

The selected specification retains `ret_1d_z`, `mom_5_z`, `mom_20_z`, and `mom_60_z`, while the other coefficients are shrunk to zero. The largest retained coefficient is negative on `mom_5_z`, consistent with short-horizon reversal.

## XGBoost

XGBoost is used as a nonlinear challenger. The validation-selected specification is:

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

The selected depth is only one, so the model is best interpreted as learning **simple nonlinear thresholds in individual predictors**, not complex within-tree interactions.

---

# Evaluation Metrics

## Mean Squared Error

\[
MSE=\frac{1}{N}\sum_i(y_i-\hat y_i)^2.
\]

## Out-of-Sample \(R^2\)

Because the target is cross-sectionally demeaned, the natural benchmark is \(\hat y=0\):

\[
R^2_{OOS}=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i y_i^2}.
\]

A positive value beats the zero forecast; a negative value does not.

## Information Coefficient

For each date,

\[
IC_t=Corr_i(\hat y_{i,t+1},y_{i,t+1}).
\]

Both Pearson IC and Spearman Rank IC are reported. Rank IC is particularly relevant because the strategy trades on relative ordering.

---

# Validation Results: 2019-2021

| Model | OOS \(R^2\) | Pearson IC | Rank IC | Positive-IC Days |
|---|---:|---:|---:|---:|
| XGBoost | **0.000260** | **0.02021** | **0.01552** | 52.05% |
| Ridge | -0.000018 | 0.00013 | 0.00932 | 52.05% |
| Elastic Net | -0.000101 | -0.00277 | 0.00863 | 51.78% |
| OLS | -0.000175 | -0.00123 | 0.00214 | 51.39% |

XGBoost produces the strongest validation Rank IC and is the only model with a positive validation OOS \(R^2\), although the magnitude is extremely small.

For XGBoost:

| Period | OOS \(R^2\) | Pearson IC | Rank IC |
|---|---:|---:|---:|
| Training | 0.001437 | 0.03778 | 0.02984 |
| Validation | 0.000260 | 0.02021 | 0.01552 |

There is clear signal decay from training to validation, but the validation ranking remains positive.

---

# XGBoost Diagnostics

Gain importance is broadly distributed:

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

No single predictor dominates.

Validation prediction correlations are:

| | OLS | Ridge | Elastic Net | XGBoost |
|---|---:|---:|---:|---:|
| OLS | 1.00 | 0.83 | 0.82 | 0.25 |
| Ridge | 0.83 | 1.00 | 0.93 | 0.32 |
| Elastic Net | 0.82 | 0.93 | 1.00 | 0.31 |
| XGBoost | 0.25 | 0.32 | 0.31 | 1.00 |

The three linear models produce relatively similar predictions, while XGBoost extracts a materially different nonlinear signal.

---

# Final Expanding-Window Test

All feature definitions, model classes, and hyperparameters are frozen after validation.

The test period uses an annual expanding-window procedure. At the beginning of each year, the models are refit using all targets that would already have been observable, then used to predict the coming year.

## Overall test results: 2022-August 2026

| Model | OOS \(R^2\) | Pearson IC | Rank IC | Positive-IC Days |
|---|---:|---:|---:|---:|
| XGBoost | -0.000524 | -0.00777 | **-0.00121** | 50.43% |
| Ridge | -0.000022 | -0.00371 | -0.00433 | 50.60% |
| Elastic Net | -0.000080 | -0.00578 | -0.00573 | 48.63% |
| OLS | -0.000223 | -0.01127 | -0.01307 | 48.63% |

Every model has negative test-period Rank IC and negative OOS \(R^2\).

The strongest validation model, XGBoost, falls from

\[
IC^{rank}_{validation}=0.01552
\]

to

\[
IC^{rank}_{test}=-0.00121.
\]

This is the central empirical result of the project.

### Validation vs test Rank IC

| Model | Validation | Test |
|---|---:|---:|
| OLS | 0.00214 | -0.01307 |
| Ridge | 0.00932 | -0.00433 |
| Elastic Net | 0.00863 | -0.00573 |
| XGBoost | **0.01552** | **-0.00121** |

### Test Rank IC by year

| Year | OLS | Ridge | Elastic Net | XGBoost |
|---|---:|---:|---:|---:|
| 2022 | -0.01866 | -0.03268 | -0.02365 | -0.03979 |
| 2023 | -0.02781 | -0.01555 | -0.00955 | -0.01454 |
| 2024 | -0.01333 | -0.00060 | -0.00190 | 0.01427 |
| 2025 | -0.00149 | 0.01860 | 0.00687 | **0.03653** |
| 2026* | 0.00077 | 0.01559 | 0.00247 | -0.00321 |

`2026*` is a partial year through August.

The year-by-year results show substantial nonstationarity rather than a stable forecasting relationship.

---

# Long-Short Portfolio

Each day the stocks are ranked by predicted relative return.

- Long top 20%
- Short bottom 20%
- Equal weight within each side
- +50% total long exposure
- -50% total short exposure
- 100% gross exposure
- 0% net dollar exposure

With 30 stocks, the portfolio typically holds six names long and six names short.

Daily gross return is

\[
r_{p,t+1}=\sum_i w_{i,t}r_{i,t+1}.
\]

---

# Turnover and Transaction Costs

Turnover is approximated as

\[
Turnover_t=\sum_i|w_{i,t}-w_{i,t-1}|.
\]

Net return under a cost assumption \(c\) is

\[
r^{net}_{p,t+1}=r^{gross}_{p,t+1}-c\times Turnover_t.
\]

The analysis evaluates 0, 5, and 10 bps per dollar traded.

Average turnover is high:

| Model | Avg. daily turnover | Annualized turnover |
|---|---:|---:|
| Ridge | 75.81% | 191.1x |
| OLS | 76.80% | 193.5x |
| XGBoost | 84.28% | 212.4x |
| Elastic Net | 90.72% | 228.6x |

This makes small transaction costs economically important.

---

# Portfolio Results

## Zero transaction costs

| Model | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| OLS | -5.37% | 10.46% | -0.475 | -23.54% |
| Ridge | -0.90% | 12.24% | -0.013 | -15.33% |
| Elastic Net | -4.58% | 11.14% | -0.365 | -21.33% |
| XGBoost | **0.09%** | **9.82%** | **0.058** | -22.33% |

XGBoost is essentially flat before costs rather than strongly profitable.

Its annualized arithmetic leg contributions are approximately +6.66% from the long side and -6.09% from the short side, leaving only about +0.57% before compounding effects.

## 5 bps transaction costs

| Model | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| OLS | -14.10% | 10.47% | -1.399 | -51.01% |
| Ridge | **-9.93%** | 12.24% | **-0.793** | -42.02% |
| Elastic Net | -14.89% | 11.13% | -1.392 | -53.14% |
| XGBoost | -10.00% | **9.84%** | -1.022 | **-41.32%** |

## 10 bps transaction costs

| Model | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| OLS | -22.03% | 10.49% | -2.319 | -68.62% |
| Ridge | -18.14% | 12.25% | -1.573 | -61.53% |
| Elastic Net | -24.09% | 11.14% | -2.418 | -72.37% |
| XGBoost | -19.07% | 9.86% | -2.096 | -63.46% |

### Yearly 5 bps Sharpe

| Year | OLS | Ridge | Elastic Net | XGBoost |
|---|---:|---:|---:|---:|
| 2022 | -1.955 | -1.870 | -2.508 | -3.306 |
| 2023 | -1.104 | -0.940 | -1.216 | -0.695 |
| 2024 | -1.949 | -0.989 | -1.360 | -0.298 |
| 2025 | -0.875 | -0.441 | -0.899 | **0.728** |
| 2026* | -1.318 | 0.002 | -1.199 | -1.889 |

The positive XGBoost year in 2025 is not persistent enough to support the strategy over the full test period.

---

# Backtest Audits

Before interpreting the portfolio results, the implementation was independently checked for:

- forward-return reconstruction from adjusted-close prices;
- signal-date / return-date alignment;
- zero net exposure and 100% gross exposure;
- stock-level portfolio-return aggregation;
- turnover reconstruction;
- transaction-cost arithmetic.

All checks passed to floating-point precision.

---

# Main Findings

1. **Daily cross-sectional return predictability is extremely weak.** The strongest standalone feature behavior is short-horizon reversal, but IC variation is much larger than mean IC.

2. **Multicollinearity is meaningful.** Momentum and price-position variables overlap substantially, motivating Ridge and Elastic Net.

3. **Regularization improves validation ranking.** Validation Rank IC rises from about 0.002 for OLS to about 0.009 for Ridge and Elastic Net.

4. **Simple nonlinearities improve validation performance.** Depth-1 XGBoost reaches a validation Rank IC of 0.0155 and positive, though tiny, OOS \(R^2\).

5. **The validation signal does not survive the final test.** All models have negative mean Rank IC and negative OOS \(R^2\) in 2022-2026.

6. **Turnover is economically destructive.** Average daily turnover is roughly 76%-91%. XGBoost is essentially flat before costs but falls to roughly -10% annualized return under a 5 bps cost assumption.

The final evidence therefore leads to **rejection of the alpha signal in its current form** rather than post-test retuning.

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

The generated `data/` files do not all need to be committed to Git.

---

# Notebook Guide

### `01_data_pipeline.ipynb`

Downloads and cleans OHLCV data, checks market-calendar alignment, constructs simple returns and next-day targets, and assigns chronological splits.

### `02_eda_and_features.ipynb`

Constructs the nine model features, performs cross-sectional standardization, studies missingness/correlation/multicollinearity, and computes training-period single-feature IC.

### `03_linear_models.ipynb`

Fits OLS, analyzes VIF and coefficients, tunes Ridge and Elastic Net using validation Rank IC, and freezes linear-model hyperparameters.

### `04_xgboost.ipynb`

Runs a deliberately small validation-only XGBoost search, compares nonlinear and linear models, inspects feature importance and model-prediction correlations, and freezes the XGBoost specification.

### `05_walkforward_backtest.ipynb`

Runs annual expanding-window out-of-sample prediction, constructs the daily long-short portfolio, applies transaction costs, audits the backtest mechanics, and analyzes signal decay and turnover.

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

# Reproducibility

The project freezes the data endpoint at:

```text
END_DATE = 2026-08-28
```

Run the notebooks in order:

```text
01_data_pipeline.ipynb
02_eda_and_features.ipynb
03_linear_models.ipynb
04_xgboost.ipynb
05_walkforward_backtest.ipynb
```

Validation-selected model parameters are saved to JSON and reloaded in the final walk-forward notebook rather than retuned using the test period.

---

# Limitations

### Fixed survivor universe

The 30-stock universe contains large companies with long histories and current relevance. This introduces survivorship and selection bias. A production-quality study would use a point-in-time historical universe.

### Small cross-section

Thirty stocks are sufficient for an interview-scale research project but much smaller than a realistic institutional equity universe.

### Price/volume-only features

The project does not use fundamentals, earnings, analyst revisions, options, news, alternative data, or order-book information.

### No sector or beta neutrality

The portfolio is dollar neutral but not explicitly sector neutral or beta neutral, so residual factor exposures may remain.

### Simplified execution

Features use end-of-day information on date \(t\), while the backtest measures adjusted-close returns from \(t\) to \(t+1\). This implicitly assumes implementation at or near the close after observing the signal. A production study should specify an executable signal cutoff and execution price.

### Simplified costs

Transaction costs are modeled as fixed basis points per dollar traded. Bid-ask spread, market impact, short-borrow cost, commissions, financing, and liquidity constraints are not modeled separately.

### Target-weight turnover

Turnover is calculated from changes in target weights rather than drift-adjusted pre-trade weights.

---

# Possible Extensions

Natural extensions include:

- a larger point-in-time equity universe;
- sector-neutral and beta-neutral portfolio construction;
- slower-moving weekly signals;
- turnover-aware model objectives;
- cross-sectional factor residualization;
- fundamental and earnings features;
- transaction-cost-aware ranking thresholds;
- more realistic next-open execution;
- walk-forward feature-stability analysis;
- risk scaling using a separate volatility forecast.

A natural combination with a separate volatility project is:

\[
\text{return forecast}+\text{risk forecast}
\rightarrow
\text{risk-scaled alpha portfolio}.
\]

---

# Conclusion

The project studies the full lifecycle of a cross-sectional alpha hypothesis rather than stopping at in-sample or validation performance.

The development sequence appears encouraging:

\[
\text{OLS}
\rightarrow
\text{regularization}
\rightarrow
\text{shallow XGBoost},
\]

with validation Rank IC improving from roughly 0.002 to 0.0155.

However, the final expanding-window test rejects the apparent signal:

\[
\boxed{IC^{rank}_{XGB,validation}=0.0155}
\]

but

\[
\boxed{IC^{rank}_{XGB,test}=-0.0012}.
\]

The linear models also have negative final-test Rank IC. Economically, XGBoost is approximately flat before transaction costs and has about 84% average daily turnover. At a 5 bps cost assumption, annualized return falls to roughly -10%.

The final conclusion is therefore:

\[
\boxed{\text{weak validation predictability did not survive a strict future test}}
\]

and

\[
\boxed{\text{high turnover further reduced economic viability}}.
\]

That negative result is informative. It demonstrates why leakage control, chronological validation
, frozen hyperparameters, walk-forward testing, and transaction-cost analysis are essential when evaluating quantitative trading signals.
