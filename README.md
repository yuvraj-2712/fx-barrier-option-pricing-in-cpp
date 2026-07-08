# FX Barrier Option Pricer — Monte Carlo with Variance Reduction (C++/Python)

An end-to-end FX barrier option pricing engine: market data pulled live in Python, priced
via a Monte Carlo engine written in C++, validated against a closed-form analytical price,
and visualized back in Python.

## Pipeline

```
fetch_market_data.py  →  inputs.txt  →  barrier_pricer.cpp  →  results.csv, *.csv  →  visualize_results.py
   (live market data)                  (C++ MC engine)                                (plots + diagnostics)
```

1. **`fetch_market_data.py`** — pulls the live USD/INR spot rate and 6-month historical
   volatility via `yfinance`, plus a 13-week T-Bill yield as a risk-free rate proxy. Strike
   and barrier are set at fixed offsets from spot (3% OTM strike, 12% barrier). Falls back
   to hardcoded defaults if the data fetch fails. Writes everything to `inputs.txt`.

2. **`barrier_pricer.cpp`** — reads `inputs.txt` (or uses defaults standalone), then prices
   an **Up-and-Out Call** via Geometric Brownian Motion (Euler-Maruyama discretization),
   three ways:
   - **Standard Monte Carlo**
   - **Antithetic Variates** (reuses the same random draws with flipped sign)
   - **Control Variates** (uses the Black-Scholes vanilla price as the control)

   Includes the **Broadie-Glasserman-Kou (1997) continuity correction** to compensate for
   discrete (vs. continuous) barrier monitoring, and validates the MC price against the
   **Reiner-Rubinstein (1991) closed-form** analytical formula for barrier options.

3. **`visualize_results.py`** — reads the C++ engine's CSV outputs and produces:
   - Simulated FX price paths (spaghetti plot) with the barrier level overlaid
   - Price convergence vs. number of simulations, across all three methods
   - Standard error decay (log-log) — the direct evidence of variance reduction working
   - Computation time comparison across methods
   - Monte Carlo convergence against the closed-form reference price
   - A printed variance-reduction efficiency summary (how many fewer simulations
     Antithetic/Control Variates need vs. Standard MC for the same accuracy)

4. **`code.ipynb`** — a Colab driver notebook that runs the whole pipeline end to end:
   installs dependencies, runs the market data fetch, compiles and runs the C++ engine,
   then runs the visualization script.

## How to Run

**Locally:**
```bash
pip install -r requirements.txt
python fetch_market_data.py
g++ -O2 -std=c++17 barrier_pricer.cpp -o barrier_pricer
./barrier_pricer
python visualize_results.py
```

**In Colab:** just open `code.ipynb` and run all cells — it handles installs, compilation,
and execution in sequence.

## Key Result

The engine demonstrates that Antithetic and Control Variates achieve materially lower
standard error than naive Monte Carlo at the same simulation count — the efficiency gain
(computed automatically by `visualize_results.py`) is expressed as how many times fewer
simulations the variance-reduced methods need to match Standard MC's accuracy.

The Monte Carlo estimates are also checked against the Reiner-Rubinstein closed-form price.
A genuine, expected discrepancy shows up for barriers sitting close to spot/strike, since
the closed-form assumes continuous barrier monitoring while the simulation checks
discretely (daily steps) — even with the continuity correction applied, this gap doesn't
fully close for options this sensitive to the barrier. This is a well-documented effect in
the barrier-options literature, not a bug, and reflects exactly the kind of discrepancy
trading desks watch for when relying on closed-form pricing for discretely-monitored
contracts.

## Known Simplification

The FX spot process uses a single risk-free rate `r` for drift (equity-style GBM) rather
than the domestic-minus-foreign rate differential that FX options technically require
(Garman-Kohlhagen). This is a deliberate simplification for this project's scope — a
natural next extension would be pulling a foreign (INR) short-rate proxy and using
`r_domestic - r_foreign` as the drift.

## Author

Yuvraj Singh
