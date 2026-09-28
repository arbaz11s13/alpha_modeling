import numpy as np
import pandas as pd

def build_long_short_portfolio(
    predictions,
    prediction_col,
    long_fraction=0.20,
):
    """
    Construct a daily dollar-neutral long-short portfolio.

    Long:
        top long_fraction of predictions

    Short:
        bottom long_fraction of predictions

    Gross exposure:
        100%

    Long exposure:
        +50%

    Short exposure:
        -50%
    """

    records = []

    # Store yesterday's target weights so
    # we can approximate today's turnover.
    previous_weights = {}

    for date, d in (
        predictions
        .sort_values(
            ["date", "ticker"]
        )
        .groupby("date")
    ):

        d = d.dropna(
            subset=[
                prediction_col,
                "fwd_ret_1d",
            ]
        ).copy()

        n_stocks = len(d)

        if n_stocks < 2:
            continue

        n_side = max(
            1,
            int(
                np.floor(
                    n_stocks
                    * long_fraction
                )
            )
        )

        # Deterministic tie breaking using ticker.
        d = d.sort_values(
            [
                prediction_col,
                "ticker",
            ]
        )

        shorts = d.head(
            n_side
        )

        longs = d.tail(
            n_side
        )

        # All other stocks receive zero weight.
        current_weights = {
            ticker: 0.0
            for ticker in d["ticker"]
        }

        # Total long exposure = +0.5
        long_weight = (
            0.5 / n_side
        )

        # Total short exposure = -0.5
        short_weight = (
            -0.5 / n_side
        )

        for ticker in longs["ticker"]:
            current_weights[ticker] = (
                long_weight
            )

        for ticker in shorts["ticker"]:
            current_weights[ticker] = (
                short_weight
            )

        # Map weights back onto rows.
        d["weight"] = (
            d["ticker"]
            .map(current_weights)
        )

        # Portfolio return over the next
        # market trading interval.
        gross_return = (
            d["weight"]
            * d["fwd_ret_1d"]
        ).sum()

        # Separate contributions for diagnostics.
        long_return = (
            d.loc[
                d["weight"] > 0,
                "weight"
            ]
            * d.loc[
                d["weight"] > 0,
                "fwd_ret_1d"
            ]
        ).sum()

        short_return = (
            d.loc[
                d["weight"] < 0,
                "weight"
            ]
            * d.loc[
                d["weight"] < 0,
                "fwd_ret_1d"
            ]
        ).sum()

        # ----------------------------------
        # Approximate turnover
        #
        # Sum of dollars traded as a
        # fraction of portfolio capital.
        # ----------------------------------

        all_tickers = (
            set(previous_weights)
            | set(current_weights)
        )

        turnover = sum(
            abs(
                current_weights.get(
                    ticker,
                    0.0
                )
                -
                previous_weights.get(
                    ticker,
                    0.0
                )
            )
            for ticker in all_tickers
        )

        records.append({
            "signal_date":
                date,

            "return_date":
                d["next_market_date"].iloc[0],

            "gross_return":
                gross_return,

            "long_return":
                long_return,

            "short_return":
                short_return,

            "turnover":
                turnover,

            "n_long":
                n_side,

            "n_short":
                n_side,
        })

        previous_weights = (
            current_weights.copy()
        )

    return pd.DataFrame(
        records
    )