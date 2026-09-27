from pathlib import Path
import pandas as pd


def load_prices(path: str) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices = prices.sort_index().ffill().dropna(how="any")
    if prices.empty:
        raise ValueError("No usable price observations found.")
    return prices


def split_index(prices: pd.DataFrame, train_fraction: float):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    split = int(len(prices) * train_fraction)
    return split


def ensure_output_dirs():
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
