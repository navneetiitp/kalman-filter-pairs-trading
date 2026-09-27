import pandas as pd

from backtest import run_backtest
from execution import ExecutionModel
from strategy import generate_signals
from metrics import performance_metrics


def sensitivity(prices, spread, beta, train_end, base_config):
    rows = []
    for entry in [1.5, 2.0, 2.5]:
        for cost in [5.0, 10.0, 20.0]:
            signals = generate_signals(
                spread,
                base_config.z_window,
                entry,
                base_config.exit_z,
                base_config.stop_z,
                base_config.max_holding_days,
            )
            execution = ExecutionModel(
                signal_lag=base_config.signal_lag,
                transaction_cost_bps=cost,
                slippage_bps=cost,
            )
            bt = run_backtest(
                prices,
                beta,
                signals,
                train_end,
                execution,
                max_abs_beta=base_config.max_abs_beta,
            )
            test = bt.loc[bt["in_test"], "net_return"]
            m = performance_metrics(test, base_config.annualization)
            rows.append({
                "entry_z": entry,
                "round_trip_cost_assumption_bps_per_leg": cost,
                "sharpe": m.get("sharpe"),
                "cagr": m.get("cagr"),
                "max_drawdown": m.get("max_drawdown"),
                "total_return": m.get("total_return"),
            })
    return pd.DataFrame(rows)
