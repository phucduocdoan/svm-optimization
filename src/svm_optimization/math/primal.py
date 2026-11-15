"""Các hàm cho bài toán Primal Soft-Margin SVM."""
from __future__ import annotations
import numpy as np


def hinge_losses(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Hinge loss từng mẫu: max(0, 1-y_i f(x_i)); y thuộc {-1,+1}."""
    y = np.asarray(y, dtype=float).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if y.shape != scores.shape:
        raise ValueError("y và scores phải có cùng số phần tử.")
    return np.maximum(0.0, 1.0 - y * scores)


def soft_margin_objective(w: np.ndarray, y: np.ndarray, scores: np.ndarray, C: float) -> float:
    """Tính 1/2 ||w||^2 + C * sum_i hinge_loss_i."""
    if C <= 0:
        raise ValueError("C phải dương.")
    w = np.asarray(w, dtype=float)
    return float(0.5 * np.dot(w, w) + C * hinge_losses(y, scores).sum())


def primal_constraints(X, y, w, b, xi, tol: float = 1e-8) -> dict:
    """Kiểm tra y_i(w^T x_i+b)>=1-xi_i và xi_i>=0."""
    X, y = np.asarray(X, float), np.asarray(y, float).reshape(-1)
    xi = np.asarray(xi, float).reshape(-1)
    scores = X @ np.asarray(w, float) + float(b)
    margin_violation = np.maximum(0.0, 1.0 - xi - y * scores)
    slack_violation = np.maximum(0.0, -xi)
    m1 = float(margin_violation.max(initial=0.0))
    m2 = float(slack_violation.max(initial=0.0))
    return {"feasible": max(m1, m2) <= tol, "max_margin_violation": m1, "max_slack_violation": m2}
