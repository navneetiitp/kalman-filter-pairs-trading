# Dynamic Statistical Arbitrage — Kalman Filter Hedge Ratio

**Quant / Statistical-Arbitrage Research Project**

A causal, execution-aware research framework for testing whether a **time-varying hedge ratio** improves pairs trading relative to a static OLS hedge.

> **Important scope:** this repository uses daily equity data. It is intentionally **not marketed as an HFT trading system** because it has no order-book, queue-position, tick, latency or exchange-matching data. Instead, it demonstrates quantitative/stat-arb concepts that are directly relevant to HFT interviews: recursive estimation, state-space modeling, market-neutral exposure, execution timing, turnover costs, risk controls, and rigorous out-of-sample testing.

## Why this project matters for a Quant/HFT interview

A static pairs model assumes

\[
y_t = \alpha + \beta x_t + \epsilon_t.
\]

That assumption can become stale when the relative economics, volatility, or market regime of two securities changes.

This project replaces the fixed coefficients with latent states:

\[
\theta_t =
\begin{bmatrix}
\alpha_t\\
\beta_t
\end{bmatrix}
\]

and recursively updates them with a Kalman filter.

The interview question is therefore not **"Does Kalman make the Sharpe higher?"**. It is:

> **Can a recursively estimated hedge ratio adapt to a changing relationship without leaking future information, and does that adaptation survive realistic execution costs and risk controls?**

That framing is much more defensible in a quantitative interview.

---

## 1. Model

### Observation equation

\[
\log Y_t = \alpha_t + \beta_t\log X_t + \epsilon_t,
\qquad \epsilon_t\sim N(0,R)
\]

or

\[
y_t = H_t\theta_t + \epsilon_t,
\qquad H_t=[1\quad x_t].
\]

### State transition

\[
\theta_t = \theta_{t-1}+\eta_t,
\qquad \eta_t\sim N(0,Q).
\]

The hidden state is therefore the **intercept and hedge ratio**.

- `Q` controls how quickly the coefficients are allowed to evolve.
- `R` controls observation noise.
- The Kalman gain determines how strongly the new observation changes the previous state estimate.

### Dynamic spread

\[
s_t = \log Y_t-\alpha_t-\beta_t\log X_t.
\]

Trading is based on a causal rolling z-score of this residual.

---

## 2. Research pipeline

```text
Raw prices
    ↓
Training-only pair selection
    ↓
Training-only OLS initialization
    ↓
Training-only Q/R calibration
    ↓
Recursive Kalman filtering
    ↓
Dynamic α_t, β_t
    ↓
Dynamic residual spread
    ↓
Trailing z-score
    ↓
Signal at t
    ↓
Execute at t+1
    ↓
Dynamic hedge weights
    ↓
Turnover + transaction cost + slippage
    ↓
Risk / P&L / drawdown
    ↓
Untouched out-of-sample evaluation
```

The signal/execution boundary is explicit. A signal observed at `t` cannot trade using the return that produced that signal.

---

## 3. Pair selection

The candidate universe comes from the original research project. Pair selection is performed **only on the training sample** using Engle-Granger cointegration and residual half-life.

The selected pair and complete selection table are saved to:

`outputs/tables/pair_selection_train_only.csv`

No test-period observation is used to choose the pair.

---

## 4. Trading model

The portfolio is normalized by gross capital:

\[
w_Y=\frac{p_t}{1+|\beta_t|},
\qquad
w_X=-\frac{p_t\beta_t}{1+|\beta_t|}.
\]

where `p_t ∈ {-1,0,+1}`.

The implementation additionally applies a configurable absolute-beta cap to prevent pathological hedge ratios from creating excessive leg exposure.

Baseline signal configuration:

- Entry: `|z| >= 2.0`
- Exit: `|z| <= 0.5`
- Stop: `|z| >= 3.0`
- Maximum holding period: 60 observations
- Execution lag: 1 bar
- Transaction cost: 5 bps per unit of leg turnover
- Slippage: 5 bps per unit of leg turnover

These are configuration parameters, not values selected by maximizing test-set performance.

---

## 5. Execution model

The execution layer is deliberately separated from the alpha model.

`src/execution.py` handles:

- signal-to-execution latency
- executed weights
- two-leg turnover
- transaction-cost drag
- implementation-shortfall accounting

This separation is important because a research signal and an executable strategy are not the same object.

For actual HFT research, the execution model would be extended with:

- tick-by-tick trades/quotes
- bid/ask spread
- queue position
- fill probability
- order latency
- exchange fees/rebates
- market impact
- partial fills
- cancel/replace behavior

Those components are intentionally **not fabricated** here because the supplied dataset does not contain the required information.

---

## 6. Risk controls

The backtest includes:

- market-neutral pair construction
- absolute-beta cap
- gross exposure tracking
- stop-z threshold
- maximum holding period
- turnover-based execution costs
- drawdown measurement

The architecture makes it straightforward to add portfolio-level limits, volatility targeting, borrow constraints and kill switches when richer data are available.

---

## 7. Look-ahead-bias controls

This project treats information timing as a first-class research constraint.

1. Pair selection uses training observations only.
2. Initial OLS parameters use training observations only.
3. Kalman process-noise calibration uses training observations only.
4. The filter is recursive; no backward smoothing is used for trading.
5. The z-score uses a trailing window.
6. A signal at `t` is executed at `t+1`.
7. The static OLS benchmark is estimated on training data only.
8. Test-period performance is never used to select parameters.

A unit test explicitly verifies that modifying a future observation does not change earlier Kalman states.

---

## 8. Results

The repository reports the actual results generated by the supplied data. It does **not** optimize parameters to create an attractive Sharpe ratio.

See:

- `outputs/tables/performance_comparison.csv`
- `outputs/tables/research_summary.csv`
- `outputs/tables/robustness_sensitivity.csv`
- `outputs/tables/dynamic_trades.csv`

The main comparison is dynamic Kalman versus static OLS under the **same signal and execution framework**.

For the supplied run, the dynamic strategy was not profitable out of sample. That result is retained. A negative result is preferable to an impressive result produced by leakage or post-hoc parameter tuning.

---

## 9. Diagnostics

Generated research outputs include:

- `dynamic_hedge_ratio.png`
- `dynamic_spread.png`
- `zscore.png`
- `equity_curve_comparison.png`
- `drawdown_comparison.png`
- `kalman_gain_beta.png`
- `innovation.png`

The Kalman gain and innovation diagnostics are particularly useful in an interview because they show **how the estimator responds to new information**, rather than treating the filter as a black box.

---

## 10. Robustness

The research includes sensitivity analysis over entry thresholds and execution costs.

The purpose is not to select the most attractive parameter combination. It is to determine whether the conclusion is fragile to reasonable assumptions.

A stronger production study would extend this to:

- rolling walk-forward recalibration
- multiple independent test periods
- regime-conditioned analysis
- bootstrap confidence intervals
- deflated Sharpe / multiple-testing controls
- larger pair universes
- realistic borrow and liquidity constraints

---

## 11. HFT extension path

If this project is later upgraded with intraday data, the architecture is ready for a much more HFT-specific version:

### Alpha layer
- quote/trade features
- short-horizon spread innovations
- order-flow imbalance
- microprice
- lead-lag features
- online state estimation

### Execution layer
- bid/ask-aware fills
- latency model
- queue-position model
- partial fills
- maker/taker fees
- market-impact model

### Risk layer
- gross/net exposure limits
- per-symbol limits
- intraday drawdown limits
- inventory limits
- stale-data checks
- kill switch

### Evaluation
- fill-adjusted P&L
- implementation shortfall
- turnover
- inventory duration
- adverse selection
- capacity
- P&L attribution

That would turn the research framework into a genuine intraday/HFT research project once the required market data are available.

---

## 12. Repository structure

```text
kalman-filter-dynamic-hedge-ratio/
├── data/
│   └── raw/
│       └── stock_data.csv
├── src/
│   ├── backtest.py
│   ├── config.py
│   ├── data.py
│   ├── execution.py
│   ├── kalman_filter.py
│   ├── metrics.py
│   ├── pair_selection.py
│   ├── plots.py
│   ├── robustness.py
│   ├── risk.py
│   ├── run_research.py
│   └── strategy.py
├── tests/
│   ├── conftest.py
│   ├── test_execution.py
│   └── test_kalman_filter.py
├── outputs/
│   ├── plots/
│   └── tables/
├── docs/
│   ├── INTERVIEW.md
│   └── RESUME.md
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 13. Reproduce

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
python src/run_research.py
pytest -q
```

No internet connection is required for the supplied research snapshot.

---

## 14. Resume positioning

**Recommended title:**

> Dynamic Statistical Arbitrage — Kalman Filter Hedge-Ratio Model

Recommended resume bullets are in [`docs/RESUME.md`](docs/RESUME.md).

The strongest positioning is to emphasize **state-space modeling + causal research + execution/risk discipline**, not to claim that a daily-bar strategy is itself HFT.

---

## 15. Limitations

The dataset is daily equity data. Therefore this repository cannot empirically establish:

- sub-second alpha
- order-book alpha
- queue dynamics
- exchange latency
- fill probability
- intraday market impact
- high-frequency capacity

Those are data limitations, not assumptions that should be hidden.

The project is intended as a rigorous statistical-arbitrage research project that demonstrates foundations relevant to quantitative and HFT interviews.

---

## 16. What an HFT interviewer should see in this project

| Area | Evidence in repository |
|---|---|
| Quant modeling | State-space model + recursive Kalman filter |
| Statistical arbitrage | Cointegration + residual mean reversion |
| Online estimation | Filtered `alpha_t`, `beta_t` and Kalman gain |
| Research discipline | Training-only calibration + untouched test period |
| Execution | Explicit signal lag + two-leg turnover costs |
| Risk | Beta cap, stops, holding-period limit, drawdown |
| Diagnostics | Innovations, gain, spread, z-score, P&L |
| Robustness | Threshold/cost sensitivity |
| Software quality | Modular source + automated tests |
| HFT honesty | Explicit separation between daily research and future tick-data extension |

The strongest interview narrative is:

> **I started with static pairs trading, then built a state-space model where the hedge ratio is a latent time-varying state. I made the research causal, separated alpha from execution and risk, and compared it against a static OLS baseline without hiding a weak out-of-sample result. The next step is to replace daily bars with quote/trade data and test whether the signal survives spread, latency and market impact.**
