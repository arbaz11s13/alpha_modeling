import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm


def oos_r2(y_true, y_pred):
    """
    OOS R² relative to predicting zero cross-sectional return.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    model_sse = np.sum(
        (y_true - y_pred) ** 2
    )

    zero_sse = np.sum(
        y_true ** 2
    )

    return 1 - model_sse / zero_sse

# --------------------------------------------------------------------------------------------------

def daily_ic(
    df,
    prediction_col,
    target_col="target_cs_1d",
    method="spearman",
):
    """
    Calculate cross-sectional prediction/return correlation
    separately for each date.
    """

    dates = []
    ic_values = []

    for date, d in df.groupby("date"):

        valid = d[
            [prediction_col, target_col]
        ].dropna()

        if len(valid) < 5:
            continue

        if (
            valid[prediction_col].nunique() < 2
            or valid[target_col].nunique() < 2
        ):
            ic = np.nan

        else:
            ic = valid[prediction_col].corr(
                valid[target_col],
                method=method,
            )

        dates.append(date)
        ic_values.append(ic)

    return pd.Series(
        ic_values,
        index=pd.to_datetime(dates),
        name=f"{method}_ic",
    )


# --------------------------------------------------------------------------------------------------

def newey_west_mean_test(
    series,
    maxlags=None,
):
    """
    Test whether the mean of a time series differs from zero
    using a Newey-West / HAC standard error.

    If maxlags is not supplied, use a common rule-of-thumb:
        floor(4 * (T / 100) ** (2 / 9))
    """

    s = (
        pd.Series(series)
        .dropna()
        .astype(float)
    )

    n = len(s)

    if n < 2:
        return {
            "n": n,
            "mean": np.nan,
            "nw_se": np.nan,
            "nw_t": np.nan,
            "nw_pvalue": np.nan,
            "maxlags": np.nan,
        }

    if maxlags is None:
        maxlags = max(
            1,
            int(
                np.floor(
                    4 * (n / 100) ** (2 / 9)
                )
            ),
        )

    # Regression on a constant:
    # intercept = sample mean
    X = np.ones(
        (n, 1)
    )

    fit = sm.OLS(
        s.to_numpy(),
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": maxlags
        },
    )

    mean = float(
        fit.params[0]
    )

    se = float(
        fit.bse[0]
    )

    t_stat = (
        mean / se
        if se > 0
        else np.nan
    )

    return {
        "n": n,
        "mean": mean,
        "nw_se": se,
        "nw_t": t_stat,
        "nw_pvalue": float(
            fit.pvalues[0]
        ),
        "maxlags": maxlags,
    }

# --------------------------------------------------------------------------------------------------------

def newey_west_paired_test(
    series_a,
    series_b,
    maxlags=None,
):
    """
    Test whether the mean difference between two aligned
    time series differs from zero using Newey-West SE.

        H0: E[series_a - series_b] = 0
    """

    aligned = pd.concat(
        [
            pd.Series(
                series_a
            ).rename("a"),

            pd.Series(
                series_b
            ).rename("b"),
        ],
        axis=1,
    ).dropna()

    difference = (
        aligned["a"]
        - aligned["b"]
    )

    result = (
        newey_west_mean_test(
            difference,
            maxlags=maxlags,
        )
    )

    result[
        "mean_a"
    ] = aligned["a"].mean()

    result[
        "mean_b"
    ] = aligned["b"].mean()

    result[
        "mean_difference"
    ] = difference.mean()

    return result


# -----------------------------------------------------------------------------------------------------------------

# General evaluation
def evaluate_predictions(
    data,
    predictions,
    model_name,
):
    """
    Evaluate prediction accuracy and cross-sectional ranking.
    """

    temp = data[
        ["date", "ticker", "target_cs_1d"]
    ].copy()

    temp["prediction"] = predictions

    mse = mean_squared_error(
        temp["target_cs_1d"],
        temp["prediction"],
    )

    r2 = oos_r2(
        temp["target_cs_1d"],
        temp["prediction"],
    )

    pearson_ic = daily_ic(
        temp,
        prediction_col="prediction",
        method="pearson",
    )

    rank_ic = daily_ic(
        temp,
        prediction_col="prediction",
        method="spearman",
    )

    return {
        "model": model_name,
        "mse": mse,
        "oos_r2": r2,
        "mean_pearson_ic":
            pearson_ic.mean(),
        "mean_rank_ic":
            rank_ic.mean(),
        "rank_ic_std":
            rank_ic.std(),
        "pct_rank_ic_positive":
            (rank_ic > 0).mean(),
    }

def performance_stats(
    returns,
    turnover=None,
):
    """
    Calculate annualized portfolio statistics.

    Assumes approximately 252 trading days/year.
    """

    returns = (
        pd.Series(returns)
        .dropna()
    )

    n = len(returns)

    if n == 0:
        return {}

    # Geometric annualized return
    annual_return = (
        (1 + returns).prod()
        ** (252 / n)
        - 1
    )

    annual_vol = (
        returns.std()
        * np.sqrt(252)
    )

    if returns.std() > 0:

        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(252)
        )

    else:
        sharpe = np.nan

    result = {
        "annual_return":
            annual_return,

        "annual_vol":
            annual_vol,

        "sharpe":
            sharpe,

        "max_drawdown":
            max_drawdown(
                returns
            ),
    }

    if turnover is not None:

        result[
            "avg_daily_turnover"
        ] = np.mean(
            turnover
        )

        result[
            "annualized_turnover"
        ] = (
            np.mean(turnover)
            * 252
        )

    return result

def max_drawdown(returns):
    """
    Maximum peak-to-trough drawdown of
    compounded portfolio wealth.
    """

    wealth = (
        1 + returns
    ).cumprod()

    running_peak = (
        wealth.cummax()
    )

    drawdown = (
        wealth / running_peak
        - 1
    )

    return drawdown.min()