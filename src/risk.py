import numpy as np
import pandas as pd


def cap_beta(beta: pd.Series, max_abs_beta: float) -> pd.Series:
    """Limit hedge-ratio magnitude to avoid pathological notional exposure."""
    if max_abs_beta <= 0:
        raise ValueError("max_abs_beta must be positive")
    return beta.clip(-max_abs_beta, max_abs_beta)


def normalize_pair_weights(beta: pd.Series, position: pd.Series, max_abs_beta: float = 5.0):
    beta = cap_beta(beta, max_abs_beta)
    gross = 1.0 + beta.abs()
    weight_y = position / gross
    weight_x = -position * beta / gross
    return weight_y, weight_x


def realized_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252.0)
