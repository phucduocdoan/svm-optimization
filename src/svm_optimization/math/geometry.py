"""Hình học của SVM: siêu phẳng, khoảng cách và margin.

Quy ước: w có shape (n_features,), b là bias, hyperplane là w^T x+b=0.
"""
from __future__ import annotations
import numpy as np


def decision_score(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Tính điểm f(x)=w^T x+b, chưa lấy sign để phân lớp."""
    return np.asarray(X, dtype=float) @ np.asarray(w, dtype=float) + float(b)


def predict_linear(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Dự đoán {-1,+1}; score bằng 0 được gán về +1."""
    return np.where(decision_score(X, w, b) >= 0.0, 1, -1)


def vector_norm(w: np.ndarray) -> float:
    """Tính chuẩn Euclid ||w||_2."""
    return float(np.linalg.norm(np.asarray(w, dtype=float)))


def margin_width(w: np.ndarray) -> float:
    """Độ rộng giữa hai biên hỗ trợ: 2/||w||."""
    norm = vector_norm(w)
    if norm <= 1e-15:
        raise ValueError("Margin không xác định khi ||w|| = 0.")
    return 2.0 / norm


def point_to_hyperplane_distance(x: np.ndarray, w: np.ndarray, b: float) -> float:
    """Khoảng cách hình học từ một điểm x tới hyperplane."""
    norm = vector_norm(w)
    if norm <= 1e-15:
        raise ValueError("Khoảng cách không xác định khi ||w|| = 0.")
    return float(abs(np.dot(w, np.asarray(x, dtype=float)) + b) / norm)


def signed_distance(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Khoảng cách có dấu tới hyperplane; dấu cho biết phía của điểm."""
    norm = vector_norm(w)
    if norm <= 1e-15:
        raise ValueError("Khoảng cách không xác định khi ||w|| = 0.")
    return decision_score(X, w, b) / norm
