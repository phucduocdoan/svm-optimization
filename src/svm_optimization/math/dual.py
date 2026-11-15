"""Các hàm toán học của bài toán đối ngẫu SVM."""
from __future__ import annotations
import numpy as np


def dual_objective(alpha: np.ndarray, y: np.ndarray, K: np.ndarray) -> float:
    """W(alpha)=sum(alpha)-1/2 (alpha*y)^T K (alpha*y), cần MAXIMIZE."""
    alpha, y, K = np.asarray(alpha, float).reshape(-1), np.asarray(y, float).reshape(-1), np.asarray(K, float)
    if K.shape != (len(alpha), len(alpha)) or y.shape != alpha.shape:
        raise ValueError("K, alpha và y không tương thích kích thước.")
    weighted = alpha * y
    return float(alpha.sum() - 0.5 * weighted @ K @ weighted)


def dual_gradient(alpha: np.ndarray, y: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Gradient của W theo alpha: 1-y*(K@(alpha*y))."""
    alpha, y, K = np.asarray(alpha, float).reshape(-1), np.asarray(y, float).reshape(-1), np.asarray(K, float)
    return 1.0 - y * (K @ (alpha * y))


def check_dual_feasibility(alpha: np.ndarray, y: np.ndarray, C: float, tol: float = 1e-8) -> dict:
    """Kiểm tra 0<=alpha_i<=C và sum(alpha_i*y_i)=0."""
    alpha, y = np.asarray(alpha, float).reshape(-1), np.asarray(y, float).reshape(-1)
    if alpha.shape != y.shape:
        raise ValueError("alpha và y phải có cùng số phần tử.")
    lower = float(np.maximum(0.0, -alpha).max(initial=0.0))
    upper = float(np.maximum(0.0, alpha - C).max(initial=0.0))
    equality = float(abs(np.dot(alpha, y)))
    return {"feasible": max(lower, upper, equality) <= tol, "lower_bound_violation": lower,
            "upper_bound_violation": upper, "equality_residual": equality}
