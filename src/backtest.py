import numpy as np
import pandas as pd
from execution import ExecutionModel, execute_weights, transaction_costs, turnover


def run_backtest(
    prices: pd.DataFrame,
    beta: pd.Series,
    signals: pd.DataFrame,
    train_end: int,
    execution: ExecutionModel,
    max_abs_beta: float = 5.0,
):
    from risk import normalize_pair_weights

    y_name, x_name = prices.columns[:2]
    ret_y = prices[y_name].pct_change().fillna(0.0)
    ret_x = prices[x_name].pct_change().fillna(0.0)

    signal_t = signals["position_signal"]
    wy_t, wx_t = normalize_pair_weights(beta, signal_t, max_abs_beta=max_abs_beta)
    wy, wx = execute_weights(wy_t, wx_t, execution)

    gross_return = wy * ret_y + wx * ret_x
    trade_turnover = turnover(wy, wx)
    costs = transaction_costs(wy, wx, execution)
    net_return = gross_return - costs

    out = pd.DataFrame({
        "beta": beta,
        "zscore": signals["zscore"],
        "position_signal": signal_t,
        "weight_y": wy,
        "weight_x": wx,
        "gross_return": gross_return,
        "turnover": trade_turnover,
        "transaction_and_slippage_cost": costs,
        "net_return": net_return,
    }, index=prices.index)
    out["equity_curve"] = (1.0 + out["net_return"]).cumprod()
    out["drawdown"] = out["equity_curve"] / out["equity_curve"].cummax() - 1.0
    out["gross_exposure"] = out["weight_y"].abs() + out["weight_x"].abs()
    out["in_test"] = False
    out.iloc[train_end:, out.columns.get_loc("in_test")] = True
    return out


def extract_trades(backtest: pd.DataFrame):
    s = backtest["position_signal"]
    rows = []
    active = None
    for i, value in enumerate(s):
        date = s.index[i]
        if active is None and value != 0:
            active = {"entry_date": date, "side": "long_spread" if value > 0 else "short_spread", "entry_i": i}
        elif active is not None and value == 0:
            segment = backtest.iloc[active["entry_i"]:i]
            active.update({
                "exit_date": date,
                "holding_days": len(segment),
                "trade_return": float((1.0 + segment["net_return"]).prod() - 1.0),
            })
            rows.append(active)
            active = None
    if active is not None:
        segment = backtest.iloc[active["entry_i"]:]
        active.update({
            "exit_date": segment.index[-1],
            "holding_days": len(segment),
            "trade_return": float((1.0 + segment["net_return"]).prod() - 1.0),
        })
        rows.append(active)
    return pd.DataFrame(rows)
