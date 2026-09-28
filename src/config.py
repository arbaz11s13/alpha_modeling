# Setting up the intial configuraton with a universe of 30 stocks, a general market ticker

# 2010–2018    TRAIN
# 2019–2021    VALIDATION
# 2022–latest  FINAL OUT-OF-SAMPLE TEST

UNIVERSE = [
    # Technology
    # Apple, Microsoft, IBM, Intel, Cisco, Oracle, Qualcomm, Texas Instruments
    "AAPL", "MSFT", "IBM", "INTC", "CSCO", "ORCL", "QCOM", "TXN",

    # Financials
    # JP Morgan, Bank of America, Goldman Sachs, Morgan Stanley, Citigroup, Wells Fargo
    "JPM", "BAC", "GS", "MS", "C", "WFC",

    # Healthcare
    # Johnson & Johnson, Pfizer, Merck, Amgen, United Health
    "JNJ", "PFE", "MRK", "AMGN", "UNH",

    # Consumer
    # Walmart, Home Depot, McDonald, Coca Cola, Pepsi, Procter & Gamble, Nike
    "WMT", "HD", "MCD", "KO", "PEP", "PG", "NKE",

    # Energy / Industrials
    # Exxon Mobil, Chevron, Caterpillar, Boeing
    "XOM", "CVX", "CAT", "BA",
]

MARKET_TICKER = "SPY"

START_DATE = "2010-01-01"
END_DATE = "2026-08-28"

TRAIN_END = "2018-12-31"
VAL_END = "2021-12-31"