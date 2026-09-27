from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutionModel:
    """Bar-level execution model.

    This is deliberately explicit about the information boundary: a signal
    observed at t is executed after `signal_lag` bars. For daily data the
    default is one bar, i.e. next-session execution.
    """
    signal_lag: int = 1
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0

    @property
    def cost_rate(self) -> float:
        return (self.transaction_cost_bps + self.slippage_bps) / 10_000.0


def execute_weights(weight_y: pd.Series, weight_x: pd.Series, model: ExecutionModel):
    if model.signal_lag < 1:
        raise ValueError("signal_lag must be >= 1 to preserve causal execution.")
    return weight_y.shift(model.signal_lag).fillna(0.0), weight_x.shift(model.signal_lag).fillna(0.0)


def turnover(weight_y: pd.Series, weight_x: pd.Series) -> pd.Series:
    return weight_y.diff().abs().fillna(weight_y.abs()) + weight_x.diff().abs().fillna(weight_x.abs())


def transaction_costs(weight_y: pd.Series, weight_x: pd.Series, model: ExecutionModel) -> pd.Series:
    return turnover(weight_y, weight_x) * model.cost_rate


def implementation_shortfall_bps(weight_y: pd.Series, weight_x: pd.Series, model: ExecutionModel) -> pd.Series:
    """Return the configured all-in execution drag in bps on turnover."""
    return turnover(weight_y, weight_x) * (model.transaction_cost_bps + model.slippage_bps)
