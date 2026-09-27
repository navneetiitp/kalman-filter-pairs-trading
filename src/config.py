from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    data_path: str = "data/raw/stock_data.csv"
    train_fraction: float = 0.70
    z_window: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.0
    max_holding_days: int = 60
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    signal_lag: int = 1
    max_abs_beta: float = 5.0
    annualization: int = 252
    initial_state_variance_alpha: float = 0.01
    initial_state_variance_beta: float = 0.01
    candidate_pairs: tuple = (
        ("HDFCBANK.NS", "ICICIBANK.NS"),
        ("INFY.NS", "TCS.NS"),
        ("HINDALCO.NS", "TATASTEEL.NS"),
        ("SBIN.NS", "AXISBANK.NS"),
    )
    q_alpha_grid: tuple = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
    q_beta_grid: tuple = (1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
