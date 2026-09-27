import numpy as np
import pandas as pd


def performance_metrics(returns: pd.Series, annualization=252):
    r = returns.dropna()
    if len(r) == 0 or r.std(ddof=1) == 0:
        return {}
    equity = (1.0 + r).cumprod()
    years = len(r) / annualization
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and equity.iloc[-1] > 0 else np.nan
    dd = equity / equity.cummax() - 1.0
    downside = r[r < 0].std(ddof=1)
    return {
        "observations": int(len(r)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": float(cagr),
        "annualized_volatility": float(r.std(ddof=1) * np.sqrt(annualization)),
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(annualization)),
        "sortino": float(r.mean() / downside * np.sqrt(annualization)) if downside and np.isfinite(downside) else np.nan,
        "max_drawdown": float(dd.min()),
        "average_daily_turnover": np.nan,
    }


def compare_metrics(dynamic, static, annualization=252):
    rows = []
    for name, series in [("dynamic_kalman", dynamic), ("static_ols", static)]:
        m = performance_metrics(series, annualization)
        m["strategy"] = name
        rows.append(m)
    return pd.DataFrame(rows).set_index("strategy")
