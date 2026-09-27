# Resume-ready project entry

## Recommended project title
**Dynamic Statistical Arbitrage — Kalman Filter Hedge-Ratio Model**

## Resume bullets

- Built a **state-space statistical-arbitrage engine** that recursively estimates time-varying intercept and hedge ratio with a two-state Kalman filter, replacing stale full-sample OLS assumptions.
- Designed a **causal out-of-sample backtest** with train-only pair selection/calibration, one-bar execution lag, dynamic hedge-weight turnover, transaction costs, slippage, beta caps, drawdown and trade-level diagnostics.
- Compared dynamic Kalman and static OLS hedges under identical execution assumptions; added **innovation/Kalman-gain diagnostics, sensitivity analysis and automated unit tests** to validate model stability and prevent look-ahead bias.

## HFT interview positioning

Use the project to demonstrate quantitative foundations relevant to HFT/stat-arb interviews:

- recursive estimation and Bayesian filtering
- state-space models and Gaussian noise assumptions
- market-neutral exposure construction
- execution timing and implementation shortfall
- turnover-aware transaction-cost modeling
- risk limits and beta/exposure controls
- out-of-sample research discipline
- failure modes and model risk

**Do not describe this repository as an HFT trading system.** The supplied data are daily bars and do not contain order-book, trade, queue, or latency information. The defensible description is a quantitative/statistical-arbitrage research project with HFT-relevant modeling and execution discipline.
