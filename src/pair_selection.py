import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def half_life(residual: pd.Series) -> float:
    lag = residual.shift(1).dropna()
    delta = residual.diff().dropna()
    aligned = pd.concat([lag, delta], axis=1).dropna()
    if len(aligned) < 20:
        return np.inf
    beta = np.polyfit(aligned.iloc[:, 0], aligned.iloc[:, 1], 1)[0]
    if beta >= 0:
        return np.inf
    return float(-np.log(2.0) / beta)


def evaluate_pairs(prices: pd.DataFrame, pairs, train_end: int) -> pd.DataFrame:
    rows = []
    train = prices.iloc[:train_end]
    for y_name, x_name in pairs:
        y = np.log(train[y_name])
        x = np.log(train[x_name])
        X = np.column_stack([np.ones(len(x)), x])
        alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residual = y - (alpha + beta * x)
        stat, p_value, _ = coint(y, x)
        rows.append({
            "asset_y": y_name,
            "asset_x": x_name,
            "cointegration_stat": stat,
            "cointegration_pvalue": p_value,
            "ols_alpha": alpha,
            "ols_beta": beta,
            "residual_std": residual.std(ddof=1),
            "half_life_days": half_life(residual),
        })
    result = pd.DataFrame(rows).sort_values(
        ["cointegration_pvalue", "half_life_days"], ascending=[True, True]
    ).reset_index(drop=True)
    return result


def select_pair(selection: pd.DataFrame) -> tuple[str, str]:
    eligible = selection[
        (selection["cointegration_pvalue"] < 0.05)
        & (selection["half_life_days"] < 120)
    ]
    chosen = eligible.iloc[0] if not eligible.empty else selection.iloc[0]
    return str(chosen.asset_y), str(chosen.asset_x)
