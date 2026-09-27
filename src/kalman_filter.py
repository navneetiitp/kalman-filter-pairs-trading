import numpy as np


class DynamicHedgeKalman:
    """Two-state random-walk Kalman filter for log-price regression.

    Observation: y_t = [1, x_t] theta_t + epsilon_t
    State:       theta_t = theta_{t-1} + eta_t
    where theta_t = [alpha_t, beta_t].
    """

    def __init__(self, q_alpha, q_beta, r, theta0, p0):
        self.q = np.diag([q_alpha, q_beta]).astype(float)
        self.r = float(r)
        self.theta0 = np.asarray(theta0, dtype=float)
        self.p0 = np.asarray(p0, dtype=float)

    def filter(self, y, x):
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        n = len(y)
        if len(x) != n:
            raise ValueError("y and x must have the same length.")

        states = np.zeros((n, 2))
        covariances = np.zeros((n, 2, 2))
        innovations = np.zeros(n)
        innovation_variances = np.zeros(n)
        gains = np.zeros((n, 2))

        theta = self.theta0.copy()
        P = self.p0.copy()
        identity = np.eye(2)

        for t in range(n):
            theta_pred = theta
            P_pred = P + self.q
            H = np.array([1.0, x[t]])
            innovation = y[t] - H @ theta_pred
            S = float(H @ P_pred @ H + self.r)
            if S <= 0:
                raise FloatingPointError("Innovation variance became non-positive.")
            K = (P_pred @ H) / S
            theta = theta_pred + K * innovation
            P = (identity - np.outer(K, H)) @ P_pred
            P = 0.5 * (P + P.T)

            states[t] = theta
            covariances[t] = P
            innovations[t] = innovation
            innovation_variances[t] = S
            gains[t] = K

        return {
            "alpha": states[:, 0],
            "beta": states[:, 1],
            "cov_alpha": covariances[:, 0, 0],
            "cov_beta": covariances[:, 1, 1],
            "innovation": innovations,
            "innovation_variance": innovation_variances,
            "gain_alpha": gains[:, 0],
            "gain_beta": gains[:, 1],
        }


def fit_initial_ols(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    theta = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ theta
    r = float(np.var(residual, ddof=1))
    return theta, r


def fit_process_noise(y, x, q_alpha_grid, q_beta_grid, p0):
    theta0, r = fit_initial_ols(y, x)
    best = None
    burn = min(50, max(5, len(y) // 10))
    for q_alpha in q_alpha_grid:
        for q_beta in q_beta_grid:
            model = DynamicHedgeKalman(q_alpha, q_beta, r, theta0, p0)
            result = model.filter(y, x)
            S = result["innovation_variance"][burn:]
            v = result["innovation"][burn:]
            nll = 0.5 * float(np.sum(np.log(S) + (v * v) / S))
            if best is None or nll < best["nll"]:
                best = {
                    "q_alpha": float(q_alpha),
                    "q_beta": float(q_beta),
                    "r": r,
                    "theta0_alpha": float(theta0[0]),
                    "theta0_beta": float(theta0[1]),
                    "nll": nll,
                }
    return best
