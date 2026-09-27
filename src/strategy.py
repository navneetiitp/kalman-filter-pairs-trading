import numpy as np
import pandas as pd


def causal_zscore(series: pd.Series, window: int):
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=1)
    return (series - mean) / std


def generate_signals(
    spread: pd.Series,
    z_window: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_holding_days: int,
):
    z = causal_zscore(spread, z_window)
    position = pd.Series(0, index=spread.index, dtype=float)
    holding = 0

    for i in range(len(spread)):
        zi = z.iloc[i]
        prev = position.iloc[i - 1] if i > 0 else 0.0
        if not np.isfinite(zi):
            position.iloc[i] = 0.0
            holding = 0
            continue

        if prev == 0:
            if zi <= -entry_z:
                position.iloc[i] = 1.0
                holding = 1
            elif zi >= entry_z:
                position.iloc[i] = -1.0
                holding = 1
            else:
                position.iloc[i] = 0.0
        else:
            holding += 1
            position.iloc[i] = prev
            if abs(zi) <= exit_z or abs(zi) >= stop_z or holding >= max_holding_days:
                position.iloc[i] = 0.0
                holding = 0

    return pd.DataFrame({"spread": spread, "zscore": z, "position_signal": position})
