from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from config import Config
from data import load_prices, split_index, ensure_output_dirs
from pair_selection import evaluate_pairs, select_pair
from kalman_filter import DynamicHedgeKalman, fit_process_noise, fit_initial_ols
from strategy import generate_signals
from backtest import run_backtest, extract_trades
from execution import ExecutionModel
from metrics import performance_metrics, compare_metrics
from plots import save_research_plots
from robustness import sensitivity


def main():
    cfg = Config()
    ensure_output_dirs()
    prices = load_prices(cfg.data_path)
    split = split_index(prices, cfg.train_fraction)

    selection = evaluate_pairs(prices, cfg.candidate_pairs, split)
    y_name, x_name = select_pair(selection)
    selection.to_csv("outputs/tables/pair_selection_train_only.csv", index=False)

    log_y = np.log(prices[y_name])
    log_x = np.log(prices[x_name])
    train_y = log_y.iloc[:split].values
    train_x = log_x.iloc[:split].values

    theta0, r = fit_initial_ols(train_y, train_x)
    p0 = np.diag([cfg.initial_state_variance_alpha, cfg.initial_state_variance_beta])
    hp = fit_process_noise(train_y, train_x, cfg.q_alpha_grid, cfg.q_beta_grid, p0)
    kalman = DynamicHedgeKalman(hp["q_alpha"], hp["q_beta"], hp["r"], theta0, p0)
    filtered = kalman.filter(log_y.values, log_x.values)

    model = pd.DataFrame(filtered, index=prices.index)
    model["spread"] = log_y - model["alpha"] - model["beta"] * log_x
    model["zscore"] = generate_signals(
        model["spread"], cfg.z_window, cfg.entry_z, cfg.exit_z, cfg.stop_z, cfg.max_holding_days
    )["zscore"]
    signals = generate_signals(
        model["spread"], cfg.z_window, cfg.entry_z, cfg.exit_z, cfg.stop_z, cfg.max_holding_days
    )

    execution = ExecutionModel(
        signal_lag=cfg.signal_lag,
        transaction_cost_bps=cfg.transaction_cost_bps,
        slippage_bps=cfg.slippage_bps,
    )
    dynamic_bt = run_backtest(
        prices[[y_name, x_name]], model["beta"], signals, split, execution,
        max_abs_beta=cfg.max_abs_beta
    )

    # Static baseline: OLS parameters estimated on training data only.
    static_alpha, static_beta = theta0
    static_spread = log_y - static_alpha - static_beta * log_x
    static_signals = generate_signals(
        static_spread, cfg.z_window, cfg.entry_z, cfg.exit_z, cfg.stop_z, cfg.max_holding_days
    )
    static_beta_series = pd.Series(static_beta, index=prices.index)
    static_bt = run_backtest(
        prices[[y_name, x_name]], static_beta_series, static_signals, split, execution,
        max_abs_beta=cfg.max_abs_beta
    )

    dynamic_test = dynamic_bt.loc[dynamic_bt["in_test"], "net_return"]
    static_test = static_bt.loc[static_bt["in_test"], "net_return"]
    metrics = compare_metrics(dynamic_test, static_test, cfg.annualization)
    metrics["average_daily_turnover"] = [dynamic_bt.loc[dynamic_bt.in_test, "turnover"].mean(), static_bt.loc[static_bt.in_test, "turnover"].mean()]
    metrics.to_csv("outputs/tables/performance_comparison.csv")

    dynamic_trades = extract_trades(dynamic_bt)
    if not dynamic_trades.empty:
        dynamic_trades = dynamic_trades[dynamic_trades.entry_date >= prices.index[split]]
    dynamic_trades.to_csv("outputs/tables/dynamic_trades.csv", index=False)

    model.loc[:, ["alpha", "beta", "cov_alpha", "cov_beta", "innovation", "innovation_variance", "gain_alpha", "gain_beta", "spread", "zscore"]].to_csv("outputs/tables/kalman_diagnostics.csv")

    adf_stat, adf_p, *_ = adfuller(model.loc[prices.index[:split], "spread"].dropna())
    test_adf_stat, test_adf_p, *_ = adfuller(model.loc[prices.index[split:], "spread"].dropna())
    summary = pd.DataFrame([{
        "selected_pair_y": y_name,
        "selected_pair_x": x_name,
        "train_start": prices.index[0].date(),
        "train_end": prices.index[split - 1].date(),
        "test_start": prices.index[split].date(),
        "test_end": prices.index[-1].date(),
        "train_observations": split,
        "test_observations": len(prices) - split,
        "train_spread_adf_pvalue": adf_p,
        "test_spread_adf_pvalue_descriptive": test_adf_p,
        "q_alpha": hp["q_alpha"],
        "q_beta": hp["q_beta"],
        "r": hp["r"],
        "initial_alpha": theta0[0],
        "initial_beta": theta0[1],
        "entry_z": cfg.entry_z,
        "exit_z": cfg.exit_z,
        "stop_z": cfg.stop_z,
        "max_holding_days": cfg.max_holding_days,
        "transaction_cost_bps_per_leg": cfg.transaction_cost_bps,
        "slippage_bps_per_leg": cfg.slippage_bps,
        "signal_execution_lag_bars": cfg.signal_lag,
        "max_abs_beta": cfg.max_abs_beta,
    }])
    summary.to_csv("outputs/tables/research_summary.csv", index=False)

    robust = sensitivity(prices[[y_name, x_name]], model["spread"], model["beta"], split, cfg)
    robust.to_csv("outputs/tables/robustness_sensitivity.csv", index=False)

    save_research_plots(prices[[y_name, x_name]], model, dynamic_bt, static_bt, split)

    print("Selected pair:", y_name, "vs", x_name)
    print("Train/Test split:", prices.index[split - 1].date(), "->", prices.index[split].date())
    print(metrics.to_string())
    print("Outputs written to outputs/tables and outputs/plots")


if __name__ == "__main__":
    main()
