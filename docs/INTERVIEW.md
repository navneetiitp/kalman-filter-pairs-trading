# Interview Guide

## 1. Why use a Kalman filter instead of a single OLS hedge ratio?

OLS estimates one fixed coefficient for the whole estimation sample. If the relationship between two assets changes, that coefficient can become stale. The Kalman filter treats the intercept and hedge ratio as latent states that evolve over time and updates them recursively as new observations arrive.

The important point is not that Kalman is automatically better. It is a modeling choice for a potentially time-varying relationship, and it must be validated out of sample.

## 2. What are the hidden states?

The state vector is

\[
\theta_t = [\alpha_t, \beta_t]^T,
\]

where `alpha_t` is the dynamic intercept and `beta_t` is the dynamic hedge ratio.

## 3. What is the observation equation?

For log prices `y_t` and `x_t`:

\[
y_t = \alpha_t + \beta_t x_t + \epsilon_t,
\qquad \epsilon_t \sim N(0,R).
\]

Equivalently,

\[
y_t = [1\quad x_t]\theta_t + \epsilon_t.
\]

The residual

\[
e_t = y_t - \alpha_t - \beta_t x_t
\]

is the trading spread used by the strategy.

## 4. What is the state transition equation?

This project uses a random-walk state model:

\[
\theta_t = \theta_{t-1} + \eta_t,
\qquad \eta_t \sim N(0,Q).
\]

`Q` controls how quickly the latent parameters are allowed to move.

## 5. What do Q and R mean?

`Q` is process-noise covariance. Larger `Q` allows the hedge ratio and intercept to adapt more quickly. Smaller `Q` makes the state smoother.

`R` is observation-noise variance. It represents noise in the regression relationship that the filter cannot explain through the latent state.

In this implementation, `R` is estimated from training-sample OLS residual variance and `Q` is selected from a small grid using training-only one-step innovation likelihood.

## 6. What does the Kalman gain mean?

The Kalman gain determines how much the new observation changes the prior state estimate. A higher gain means the filter puts more weight on the new observation; a lower gain means it trusts the prior state more.

## 7. How is the hedge ratio updated?

At each timestamp the filter performs:

1. Predict the next state and covariance.
2. Form the one-step prediction error (innovation).
3. Compute innovation variance.
4. Compute Kalman gain.
5. Update `alpha_t` and `beta_t`.
6. Update state covariance.

The implementation uses only information available through timestamp `t` for the state at `t`.

## 8. Why should the spread mean revert?

The strategy assumes the selected pair has a sufficiently stable long-run relationship. If the residual is stationary or approximately mean reverting, unusually positive or negative residuals may tend to move back toward their historical center.

This is an empirical hypothesis, not a guarantee. A pair can stop being cointegrated or undergo a structural break.

## 9. How are entry and exit thresholds selected?

The baseline uses a z-score entry threshold of 2.0, exit threshold of 0.5, and stop threshold of 3.0. These are fixed research parameters rather than optimized on the test set. The repository also includes a sensitivity table for alternative entry thresholds and cost assumptions.

## 10. How do you prevent look-ahead bias?

There are several controls:

- Pair selection uses only the training sample.
- Initial OLS parameters use only training data.
- Kalman process-noise parameters are selected using training data only.
- The filter updates recursively with current/past observations.
- The z-score uses a trailing window rather than full-sample statistics.
- Signals generated at `t` are executed at `t+1`.
- Transaction costs are based on changes in executed leg weights.
- The static OLS benchmark is estimated once on training data.

## 11. Why compare against static OLS?

The static benchmark isolates the question the project is trying to study: does allowing the hedge ratio to evolve change the behavior of the strategy relative to a fixed hedge ratio under the same signal and execution framework?

## 12. What happens if the pair stops being cointegrated?

The residual may stop mean reverting, z-score excursions can persist, and the strategy can accumulate losses. A live system should monitor stationarity, residual behavior, drawdown, turnover, and structural-break indicators and have explicit deactivation rules.

## 13. How do transaction costs affect the strategy?

Dynamic hedge ratios can create additional turnover because the hedge weights change even when the directional signal does not. Costs therefore matter twice: opening/closing a position and rebalancing the hedge ratio. This is why the backtest charges costs on changes in both leg weights.

## 14. What are the major risks?

- Structural breaks in the asset relationship.
- Loss of cointegration or mean reversion.
- Model misspecification.
- Parameter sensitivity to `Q` and `R`.
- Transaction costs and slippage.
- Gap risk and execution risk.
- Crowded trades and liquidity deterioration.
- Data-quality problems.
- Multiple-testing / pair-selection bias.

## 15. Why might the backtest fail in live trading?

Historical prices do not reproduce live execution. Real trading introduces bid/ask spreads, market impact, latency, partial fills, corporate actions, borrow constraints, changing liquidity, and model drift. A backtest can also be too optimistic if its execution convention is unrealistic.

## 16. Why use log prices?

The regression is scale-aware in relative-price terms and the residual can be interpreted as a log-price relationship. The actual P&L, however, is calculated from percentage returns of the two traded assets rather than from changes in the residual itself.

## 17. Why not use the smoothed Kalman state?

A smoother generally uses future observations relative to timestamp `t`. That is useful for historical estimation and diagnostics, but using it for live signals would introduce look-ahead bias. This project uses the filtered state only.

## 18. What would you improve next?

A production research extension could add:

- Multiple independent pairs with portfolio-level risk budgeting.
- More rigorous walk-forward hyperparameter validation.
- Explicit market-impact models.
- Intraday data and execution simulation.
- Structural-break detection.
- Borrow availability and shorting constraints.
- Portfolio-level correlation and concentration limits.
- Online model monitoring and automatic strategy shutdown rules.

## HFT-oriented follow-up questions

### 19. Is this actually an HFT strategy?

No. The current dataset is daily and contains no quote-level or order-book information. I would describe it as a statistical-arbitrage research project with HFT-relevant quantitative and execution concepts. To claim HFT performance, I would need intraday/tick data and a fill/latency/market-impact simulator.

### 20. What would change with tick data?

The alpha horizon would become much shorter, the state would need to update online at event time, and the execution layer would need bid/ask-aware fills. I would also test whether the signal survives the bid/ask spread and adverse selection before claiming economic value.

### 21. Why separate alpha, execution and risk?

They answer different questions. Alpha estimates the expected opportunity, execution determines how much of it can actually be captured, and risk limits determine how much capital can be exposed. Mixing them makes it difficult to diagnose where P&L comes from or why a live system fails.

### 22. What is the main latency issue in this model?

The state estimate itself is recursive, so the computational update is naturally online. The more important practical issue is information-to-order latency: after observing a new quote/trade and updating the state, the order may reach the market after the opportunity has changed. With tick data I would measure signal decay as a function of latency rather than assuming zero latency.

### 23. How would you model market impact?

I would start with an empirical model estimated from historical executions or quote/trade data. A simple research baseline could relate implementation shortfall to participation rate, spread, volatility and traded notional. I would validate the model out of sample rather than choose an impact coefficient that makes the backtest attractive.

### 24. What would make this project genuinely stronger for an HFT role?

The next upgrade is not adding more indicators. It is replacing daily bars with event-level data and adding a realistic execution simulator: bid/ask spread, queue position, latency, partial fills, fees/rebates, market impact and inventory constraints. The existing alpha/risk/execution separation is designed to support that extension.
