import numpy as np
from src.kalman_filter import DynamicHedgeKalman


def test_filter_tracks_constant_beta():
    rng = np.random.default_rng(7)
    x = rng.normal(size=300)
    y = 0.5 + 1.2 * x + rng.normal(scale=0.05, size=300)
    model = DynamicHedgeKalman(
        q_alpha=1e-5,
        q_beta=1e-6,
        r=0.05**2,
        theta0=[0.0, 1.0],
        p0=np.eye(2),
    )
    result = model.filter(y, x)
    assert abs(result["beta"][-1] - 1.2) < 0.05


def test_filter_is_causal():
    x = np.linspace(1, 2, 50)
    y = 0.2 + 0.8 * x
    model = DynamicHedgeKalman(1e-5, 1e-6, 1e-4, [0.0, 0.5], np.eye(2))
    a = model.filter(y, x)
    y2 = y.copy(); y2[-1] += 100
    b = model.filter(y2, x)
    assert np.allclose(a["beta"][:-1], b["beta"][:-1])
