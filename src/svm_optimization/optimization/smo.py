"""SMO (Sequential Minimal Optimization) cho SVM nhị phân.

Bản SMO giáo dục, ưu tiên sự rõ ràng và dễ đọc hơn tốc độ của LIBSVM.
Mỗi bước cập nhật hai alpha để giữ sum(alpha_i*y_i)=0.
"""
from __future__ import annotations
import numpy as np
from ..math.dual import dual_objective


class SMOOptimizer:
    """Giải bài toán đối ngẫu Soft-Margin SVM bằng Sequential Minimal Optimization."""
    def __init__(self, C=1.0, tol=1e-3, eps=1e-8, max_passes=10,
                 max_iter=100_000, random_state=42):
        if C <= 0 or tol <= 0 or eps <= 0:
            raise ValueError("C, tol và eps phải dương.")
        if max_passes < 1 or max_iter < 1:
            raise ValueError("max_passes và max_iter phải >= 1.")
        self.C, self.tol, self.eps = float(C), float(tol), float(eps)
        self.max_passes, self.max_iter = int(max_passes), int(max_iter)
        self.random_state = int(random_state)

    def fit(self, K, y):
        K, y = np.asarray(K, float), np.asarray(y, float).reshape(-1)
        n = len(y)
        if K.shape != (n, n):
            raise ValueError(f"K phải có shape ({n}, {n}), nhận {K.shape}.")
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(y)):
            raise ValueError("K và y phải chứa giá trị hữu hạn.")
        if set(np.unique(y)) != {-1.0, 1.0}:
            raise ValueError("SMO yêu cầu nhãn -1 và +1, mỗi lớp phải có mẫu.")
        if not np.allclose(K, K.T, atol=1e-8):
            raise ValueError("Gram matrix K phải đối xứng.")
        self.K_, self.y_, self.n_samples_ = K, y, n
        self.alpha_, self.b_ = np.zeros(n), 0.0
        errors = -y.copy()  # alpha=b=0 => f(x_i)=0 => E_i=-y_i
        rng = np.random.default_rng(self.random_state)
        no_change, iterations, updates = 0, 0, 0
        self.objective_history_ = [dual_objective(self.alpha_, y, K)]
        while no_change < self.max_passes and iterations < self.max_iter:
            changed = 0
            for i in rng.permutation(n):
                if iterations >= self.max_iter:
                    break
                iterations += 1
                ai, Ei, yi = self.alpha_[i], errors[i], y[i]
                violates = ((yi*Ei < -self.tol and ai < self.C-self.eps)
                            or (yi*Ei > self.tol and ai > self.eps))
                if not violates:
                    continue
                order = np.argsort(-np.abs(Ei-errors))
                non_bound = [int(j) for j in order if j != i and self.eps < self.alpha_[j] < self.C-self.eps]
                nb_set = set(non_bound)
                remaining = [int(j) for j in order if j != i and j not in nb_set]
                for j in non_bound + remaining:
                    aj, yj, Ej = self.alpha_[j], y[j], errors[j]
                    if yi != yj:
                        L, H = max(0.0, aj-ai), min(self.C, self.C+aj-ai)
                    else:
                        L, H = max(0.0, ai+aj-self.C), min(self.C, ai+aj)
                    if H-L <= self.eps:
                        continue
                    Kii, Kjj, Kij = K[i,i], K[j,j], K[i,j]
                    eta = 2.0*Kij-Kii-Kjj
                    if eta >= -self.eps:  # suy biến: bỏ qua trong bản học tập
                        continue
                    aj_new = float(np.clip(aj-yj*(Ei-Ej)/eta, L, H))
                    if abs(aj_new-aj) < self.eps*(aj_new+aj+self.eps):
                        continue
                    ai_new = ai + yi*yj*(aj-aj_new)
                    if ai_new < -self.eps or ai_new > self.C+self.eps:
                        continue
                    ai_new = float(np.clip(ai_new, 0.0, self.C))
                    # Enforce equality constraint again to reduce round-off error.
                    aj_new = aj + yi*yj*(ai-ai_new)
                    if aj_new < L-self.eps or aj_new > H+self.eps:
                        continue
                    aj_new = float(np.clip(aj_new, L, H))
                    b_old = self.b_
                    b1 = b_old-Ei-yi*(ai_new-ai)*Kii-yj*(aj_new-aj)*Kij
                    b2 = b_old-Ej-yi*(ai_new-ai)*Kij-yj*(aj_new-aj)*Kjj
                    if self.eps < ai_new < self.C-self.eps:
                        b_new = b1
                    elif self.eps < aj_new < self.C-self.eps:
                        b_new = b2
                    else:
                        b_new = 0.5*(b1+b2)
                    dai, daj, db = ai_new-ai, aj_new-aj, b_new-b_old
                    self.alpha_[i], self.alpha_[j], self.b_ = ai_new, aj_new, float(b_new)
                    errors += yi*dai*K[:,i] + yj*daj*K[:,j] + db
                    changed += 1
                    updates += 1
                    break
            self.objective_history_.append(dual_objective(self.alpha_, y, K))
            no_change = no_change+1 if changed == 0 else 0
        self.n_iter_, self.n_updates_ = iterations, updates
        self.n_passes_without_change_ = no_change
        self.converged_ = no_change >= self.max_passes
        self.errors_ = errors
        self.objective_value_ = dual_objective(self.alpha_, y, K)
        self.support_indices_ = np.flatnonzero(self.alpha_ > self.eps)
        return self
