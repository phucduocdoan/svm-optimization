"""Kernel SVM binary classifier sử dụng SMOOptimizer."""
from __future__ import annotations
from numbers import Real
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
        if isinstance(self.gamma, Real):
            if not np.isfinite(self.gamma) or self.gamma <= 0:
                raise ValueError("gamma phải hữu hạn và dương.")
            return float(self.gamma)
        if self.gamma == "scale":
            variance = float(np.var(X))
            if not np.isfinite(variance):
                raise ValueError("Không thể tính gamma từ phương sai không hữu hạn.")
            return 1.0/(X.shape[1]*variance) if variance > 0 else 1.0
        if self.gamma == "auto":
            return 1.0/X.shape[1]
        raise ValueError("gamma phải là số dương, 'scale' hoặc 'auto'.")

    def _kernel(self, X, Y):
        return linear_kernel(X, Y) if self.kernel_ == "linear" else rbf_kernel(X, Y, self.gamma_)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float).reshape(-1)
        if X.ndim != 2 or X.shape[0] != len(y) or X.shape[1] == 0:
            raise ValueError("X phải là ma trận 2D và có cùng số mẫu với y.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X chứa NaN hoặc infinity; hãy xử lý trước khi fit.")
        if set(np.unique(y)) != {-1.0, 1.0}:
            raise ValueError("KernelSVM yêu cầu nhãn -1 và +1.")
        gamma = self._resolve_gamma(X)
        gram = linear_kernel(X, X) if self.kernel == "linear" else rbf_kernel(X, X, gamma)
        optimizer = SMOOptimizer(
            C=self.C, tol=self.tol, eps=self.eps, max_passes=self.max_passes,
            max_iter=self.max_iter, random_state=self.random_state,
        ).fit(gram, y)
        alpha = optimizer.alpha_.copy()
        indices = np.flatnonzero(alpha > self.eps)
        support_labels = y[indices]
        # Commit fitted attributes together only after optimization succeeds.
        self.X_train_, self.y_train_ = X.copy(), y.copy()
        self.kernel_, self.gamma_, self.optimizer_ = self.kernel, gamma, optimizer
        self.alpha_, self.support_indices_ = alpha, indices
        self.support_vectors_, self.support_labels_ = X[indices], support_labels
        self.dual_coef_ = alpha[indices] * support_labels
        self.intercept_ = float(optimizer.b_)
        self.n_support_ = np.array([np.sum(support_labels == -1), np.sum(support_labels == 1)], dtype=int)
        self.w_ = (alpha * y) @ X if self.kernel_ == "linear" else None
        self.is_fitted_ = True
        return self

    def decision_function(self, X):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Model chưa fit. Hãy gọi fit() trước.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.X_train_.shape[1]:
            raise ValueError("X phải là ma trận 2D với cùng số đặc trưng như dữ liệu train.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X chứa NaN hoặc infinity; hãy xử lý trước khi dự đoán.")
        return self._kernel(X, self.X_train_) @ (self.alpha_ * self.y_train_) + self.intercept_

    def predict(self, X):
        """Dự đoán nhãn -1/+1; score=0 được gán +1."""
        return np.where(self.decision_function(X) >= 0.0, 1, -1)

    def get_support_vectors(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Model chưa fit.")
        return self.support_vectors_.copy()
