"""Kernel SVM binary classifier sử dụng SMOOptimizer."""
from __future__ import annotations
import numpy as np
from ..optimization.smo import SMOOptimizer
from .kernel import linear_kernel, rbf_kernel


class KernelSVM:
    """SVM nhị phân với Linear hoặc RBF kernel.

    Nên chuẩn hóa dữ liệu bằng StandardScaler trước khi fit.
    """
    def __init__(self, C=1.0, kernel="linear", gamma="scale", tol=1e-3,
                 eps=1e-8, max_passes=10, max_iter=100_000, random_state=42):
        if kernel not in {"linear", "rbf"}:
            raise ValueError("kernel chỉ hỗ trợ 'linear' hoặc 'rbf'.")
        self.C, self.kernel, self.gamma = float(C), kernel, gamma
        self.tol, self.eps = float(tol), float(eps)
        self.max_passes, self.max_iter = int(max_passes), int(max_iter)
        self.random_state = int(random_state)

    def _resolve_gamma(self, X):
        if self.kernel != "rbf":
            return 0.0
        if isinstance(self.gamma, (int, float)):
            if float(self.gamma) <= 0:
                raise ValueError("gamma phải dương.")
            return float(self.gamma)
        if self.gamma == "scale":
            variance = float(np.var(X))
            return 1.0/(X.shape[1]*variance) if variance > 0 else 1.0
        if self.gamma == "auto":
            return 1.0/X.shape[1]
        raise ValueError("gamma phải là số dương, 'scale' hoặc 'auto'.")

    def _kernel(self, X, Y):
        return linear_kernel(X, Y) if self.kernel == "linear" else rbf_kernel(X, Y, self.gamma_)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float).reshape(-1)
        if X.ndim != 2 or X.shape[0] != len(y):
            raise ValueError("X phải là ma trận 2D và có cùng số mẫu với y.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X chứa NaN hoặc infinity; hãy xử lý trước khi fit.")
        if set(np.unique(y)) != {-1.0, 1.0}:
            raise ValueError("KernelSVM yêu cầu nhãn -1 và +1.")
        self.X_train_, self.y_train_ = X.copy(), y.copy()
        self.gamma_ = self._resolve_gamma(X)
        self.optimizer_ = SMOOptimizer(
            C=self.C, tol=self.tol, eps=self.eps, max_passes=self.max_passes,
            max_iter=self.max_iter, random_state=self.random_state,
        ).fit(self._kernel(X, X), y)
        self.alpha_ = self.optimizer_.alpha_.copy()
        self.support_indices_ = np.flatnonzero(self.alpha_ > self.eps)
        self.support_vectors_ = X[self.support_indices_]
        self.support_labels_ = y[self.support_indices_]
        self.dual_coef_ = self.alpha_[self.support_indices_] * self.support_labels_
        self.intercept_ = float(self.optimizer_.b_)
        self.n_support_ = np.array([
            np.sum(self.support_labels_ == -1), np.sum(self.support_labels_ == 1)
        ], dtype=int)
        self.w_ = (self.alpha_ * y) @ X if self.kernel == "linear" else None
        self.is_fitted_ = True
        return self

    def decision_function(self, X):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Model chưa fit. Hãy gọi fit() trước.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._kernel(X, self.X_train_) @ (self.alpha_ * self.y_train_) + self.intercept_

    def predict(self, X):
        """Dự đoán nhãn -1/+1; score=0 được gán +1."""
        return np.where(self.decision_function(X) >= 0.0, 1, -1)

    def get_support_vectors(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Model chưa fit.")
        return self.support_vectors_.copy()
