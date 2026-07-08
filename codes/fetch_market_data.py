"""
MARKET DATA FETCHER FOR FX BARRIER OPTION PROJECT
-----------------------------------------------------------------
What this script does:
1. Downloads real historical FX price data (e.g., USD/INR) using yfinance.
2. Computes the CURRENT spot rate (S0) - just the latest closing price.
3. Computes historical VOLATILITY (sigma) from daily log returns.
4. Fetches a proxy for the risk-free rate (r) using a short-term
   government bond yield ticker.
5. Writes all these numbers into a simple text file "inputs.txt" which
   the C++ program (barrier_pricer.cpp) reads at the start of main().

WHY THIS MATTERS:
Instead of hardcoding S0=83.0, sigma=0.10 etc, I now derive
these numbers directly from real market data.
"""

import yfinance as yf
import numpy as np

# ---------------------------------------------------------------
# STEP 1: Choose which FX pair and rate proxy to use
# ---------------------------------------------------------------
FX_TICKER = "INR=X"        # USD/INR exchange rate on Yahoo Finance
RATE_TICKER = "^IRX"       # 13-week US Treasury Bill yield (risk-free rate proxy)
LOOKBACK_PERIOD = "6mo"    # how much history to use for volatility estimate

# Fallback values in case internet/data fetch fails for any reason
FALLBACK_S0 = 83.0
FALLBACK_SIGMA = 0.10
FALLBACK_R = 0.05

# ---------------------------------------------------------------
# STEP 2: Fetch FX historical prices and compute spot + volatility
# ---------------------------------------------------------------
def get_spot_and_volatility(ticker, period):
    print(f"Fetching FX data for {ticker} ...")
    data = yf.download(ticker, period=period, progress=False)

    if data.empty:
        print("WARNING: No data returned. Using fallback values.")
        return FALLBACK_S0, FALLBACK_SIGMA

    close_prices = data["Close"].dropna()

    # current spot rate = most recent closing price
    S0 = float(close_prices.iloc[-1])

    # daily log returns: log(price_today / price_yesterday)
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()

    # annualize the daily volatility (standard assumption: 252 trading days/year)
    daily_std = log_returns.std()
    annualized_sigma = float(daily_std * np.sqrt(252))

    return S0, annualized_sigma


# ---------------------------------------------------------------
# STEP 3: Fetch risk-free rate proxy
# ---------------------------------------------------------------
def get_risk_free_rate(ticker):
    print(f"Fetching risk-free rate proxy from {ticker} ...")
    data = yf.download(ticker, period="5d", progress=False)

    if data.empty:
        print("WARNING: No rate data returned. Using fallback value.")
        return FALLBACK_R

    latest_yield_percent = float(data["Close"].dropna().iloc[-1])
    # ^IRX is quoted in percentage points (e.g., 5.2 means 5.2%), convert to decimal
    r = latest_yield_percent / 100.0
    return r


# ---------------------------------------------------------------
# STEP 4: Main driver - fetch everything and write inputs.txt
# ---------------------------------------------------------------
def main():
    S0, sigma = get_spot_and_volatility(FX_TICKER, LOOKBACK_PERIOD)
    r = get_risk_free_rate(RATE_TICKER)

    # Simple, transparent rule to set strike and barrier relative to spot.
    # Strike is set slightly above spot (5% out-of-the-money call).
    # Barrier is set further above spot (15% away), a typical corporate
    # hedging structure ("cheap" up-and-out call).
    K = round(S0 * 1.03, 2)
    barrier = round(S0 * 1.12, 2)
    T = 1.0  # 1-year maturity, kept fixed for simplicity

    print("\n----- MARKET-DERIVED OPTION INPUTS -----")
    print(f"Spot (S0)          : {S0:.4f}")
    print(f"Strike (K)         : {K:.4f}")
    print(f"Barrier (H)        : {barrier:.4f}")
    print(f"Risk-free rate (r) : {r:.4f}")
    print(f"Volatility (sigma) : {sigma:.4f}")
    print(f"Maturity (T)       : {T:.2f} years")

    # Write to a plain text file, one "key=value" pair per line.
    # Kept deliberately simple so the C++ side only needs basic string parsing.
    with open("inputs.txt", "w") as f:
        f.write(f"S0={S0}\n")
        f.write(f"K={K}\n")
        f.write(f"barrier={barrier}\n")
        f.write(f"r={r}\n")
        f.write(f"sigma={sigma}\n")
        f.write(f"T={T}\n")

    print("\nSaved to inputs.txt - now run the compiled C++ program.")


if __name__ == "__main__":
    main()