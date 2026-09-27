import pandas as pd
from src.execution import ExecutionModel, execute_weights, turnover


def test_execution_is_lagged():
    idx = pd.date_range("2025-01-01", periods=4, freq="D")
    w = pd.Series([1.0, 0.0, -1.0, 0.0], index=idx)
    y, _ = execute_weights(w, w, ExecutionModel(signal_lag=1))
    assert y.iloc[0] == 0.0
    assert y.iloc[1] == 1.0


def test_turnover_counts_both_legs():
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    y = pd.Series([0.0, 0.5, 0.0], index=idx)
    x = pd.Series([0.0, -0.5, 0.0], index=idx)
    assert turnover(y, x).iloc[1] == 1.0
