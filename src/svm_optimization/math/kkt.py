"""Kiểm tra các điều kiện KKT của Soft-Margin SVM."""
from __future__ import annotations
import numpy as np


def kkt_report(X, y, w, b, xi, alpha, C: float, tol: float = 1e-5) -> dict:
    """Residual primal/dual feasibility, stationarity và complementary slackness.

    Áp dụng cho Linear SVM có w tường minh. Với kernel phi tuyến, kiểm tra
    KKT qua kernel-space margins hoặc các đại lượng dual trực tiếp.
    """
    X, y = np.asarray(X, float), np.asarray(y, float).reshape(-1)
    w, xi, alpha = np.asarray(w, float).reshape(-1), np.asarray(xi, float).reshape(-1), np.asarray(alpha, float).reshape(-1)
    if not (len(X) == len(y) == len(xi) == len(alpha)):
        raise ValueError("X, y, xi, alpha phải có cùng số mẫu.")
    if C <= 0:
        raise ValueError("C phải dương.")
    margins = y * (X @ w + float(b))
    mu = C - alpha
    vals = {
        "primal_margin_violation": float(np.maximum(0, 1-xi-margins).max(initial=0)),
        "primal_slack_violation": float(np.maximum(0, -xi).max(initial=0)),
        "dual_alpha_lower_violation": float(np.maximum(0, -alpha).max(initial=0)),
        "dual_alpha_upper_violation": float(np.maximum(0, alpha-C).max(initial=0)),
        "dual_mu_lower_violation": float(np.maximum(0, -mu).max(initial=0)),
        "stationarity_w": float(np.linalg.norm(w - X.T @ (alpha*y))),
        "stationarity_b": float(abs(np.dot(alpha, y))),
        "stationarity_xi": float(np.max(np.abs(C-alpha-mu), initial=0)),
        "complementarity_margin": float(np.max(np.abs(alpha*(margins-1+xi)), initial=0)),
        "complementarity_slack": float(np.max(np.abs(mu*xi), initial=0)),
    }
    return {**vals, "kkt_satisfied": max(vals.values()) <= tol}


def alpha_roles(alpha: np.ndarray, C: float, eps: float = 1e-6) -> np.ndarray:
    """Phân nhóm alpha thành lower/free/upper-bound."""
    alpha = np.asarray(alpha, dtype=float)
    roles = np.full(alpha.shape, "free", dtype=object)
    roles[alpha <= eps] = "lower_bound"
    roles[alpha >= C-eps] = "upper_bound"
    return roles
