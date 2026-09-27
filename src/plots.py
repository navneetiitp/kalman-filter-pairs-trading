from pathlib import Path
import matplotlib.pyplot as plt


def save_research_plots(prices, model, dynamic_bt, static_bt, train_end, out_dir="outputs/plots"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    test_start = prices.index[train_end]

    plt.figure(figsize=(12, 5))
    plt.plot(model.index, model["beta"], label="Kalman beta")
    plt.axvline(test_start, linestyle="--", label="Train/Test split")
    plt.title("Dynamic Hedge Ratio")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{out_dir}/dynamic_hedge_ratio.png", dpi=160); plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(model.index, model["spread"], label="Kalman residual spread")
    plt.axvline(test_start, linestyle="--", label="Train/Test split")
    plt.title("Dynamic Spread / Residual")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{out_dir}/dynamic_spread.png", dpi=160); plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(model.index, model["zscore"], label="Z-score")
    plt.axhline(2, linestyle="--"); plt.axhline(-2, linestyle="--"); plt.axhline(0, linestyle="-")
    plt.axvline(test_start, linestyle="--", label="Train/Test split")
    plt.title("Causal Spread Z-score")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{out_dir}/zscore.png", dpi=160); plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(dynamic_bt.index, dynamic_bt["equity_curve"], label="Dynamic Kalman")
    plt.plot(static_bt.index, static_bt["equity_curve"], label="Static OLS")
    plt.axvline(test_start, linestyle="--", label="Train/Test split")
    plt.title("Equity Curves")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{out_dir}/equity_curve_comparison.png", dpi=160); plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(dynamic_bt.index, dynamic_bt["drawdown"], label="Dynamic drawdown")
    plt.plot(static_bt.index, static_bt["drawdown"], label="Static drawdown")
    plt.axhline(0, linestyle="-")
    plt.title("Drawdown Comparison")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{out_dir}/drawdown_comparison.png", dpi=160); plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(model.index, model["gain_beta"], label="Kalman gain for beta")
    plt.axvline(test_start, linestyle="--", label="Train/Test split")
    plt.title("Kalman Gain — Hedge Ratio Adaptation")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{out_dir}/kalman_gain_beta.png", dpi=160); plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(model.index, model["innovation"], label="Innovation")
    plt.axhline(0, linestyle="-")
    plt.axvline(test_start, linestyle="--", label="Train/Test split")
    plt.title("One-step-ahead Kalman Innovation")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{out_dir}/innovation.png", dpi=160); plt.close()
