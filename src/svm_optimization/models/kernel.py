"""Kernel functions và cách tạo Gram matrix."""

from __future__ import annotations
import numpy as np


def linear_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """K(x,z)=x^T z."""
    return np.asarray(X, dtype=float) @ np.asarray(Y, dtype=float).T


def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """K(x,z)=exp(-gamma*||x-z||^2)."""
    if gamma <= 0:
        raise ValueError("gamma phải dương.")
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    x2 = np.sum(X*X, axis=1)[:, None]
    y2 = np.sum(Y*Y, axis=1)[None, :]
    d2 = np.maximum(x2 + y2 - 2.0*(X @ Y.T), 0.0)
    return np.exp(-gamma*d2)


def make_kernel(name: str, gamma: float | None = None):
    """Trả về callable kernel(X,Y) theo tên linear hoặc rbf."""
    name = name.lower()
    if name == "linear":
        return linear_kernel
    if name == "rbf":
        if gamma is None:
            raise ValueError("Cần truyền gamma cho RBF kernel.")
        return lambda X, Y: rbf_kernel(X, Y, gamma=gamma)
    raise ValueError("kernel chỉ hỗ trợ 'linear' hoặc 'rbf'.")
