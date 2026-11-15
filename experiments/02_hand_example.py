"""Chặng 2: ví dụ giải tay có nghiệm chính xác và đối chiếu bằng SMO."""
import numpy as np
from svm_optimization.math.dual import dual_objective, check_dual_feasibility
from svm_optimization.math.geometry import margin_width
from svm_optimization.math.kkt import kkt_report
from svm_optimization.math.primal import soft_margin_objective
from svm_optimization.models.kernel import linear_kernel
from svm_optimization.models.svm import KernelSVM


def main():
    X = np.array([[-1., -1.], [1., 1.], [-2., -1.], [2., 1.]])
    y = np.array([-1., 1., -1., 1.])
    C = 1.0
    # Nghiệm tính tay: w=(0.5,0.5), b=0, alpha=(0.25,0.25,0,0).
    w, b = np.array([0.5, 0.5]), 0.0
    alpha = np.array([0.25, 0.25, 0.0, 0.0])
    scores = X @ w + b
    xi = np.maximum(0.0, 1.0-y*scores)
    K = linear_kernel(X, X)
    primal_value = soft_margin_objective(w, y, scores, C)
    dual_value = dual_objective(alpha, y, K)
    print("=== Analytic / hand-worked solution ===")
    print("w:", w, "b:", b, "scores:", scores)
    print("margins:", y*scores, "slack xi:", xi, "alpha:", alpha)
    print("support vector indices:", np.flatnonzero(alpha > 1e-8))
    print("margin width:", margin_width(w))
    print("primal objective:", primal_value, "dual objective:", dual_value)
    print("duality gap:", abs(primal_value-dual_value))
    print("dual feasibility:", check_dual_feasibility(alpha, y, C))
    print("KKT report:", kkt_report(X, y, w, b, xi, alpha, C))
    print("\n=== Same toy data solved by custom SMO ===")
    model = KernelSVM(C=C, kernel="linear", tol=1e-6, max_passes=20).fit(X, y)
    print("SMO w:", model.w_, "b:", model.intercept_)
    print("SMO alpha:", np.round(model.alpha_, 6))
    print("SMO predictions:", model.predict(X))
    print("SMO objective:", model.optimizer_.objective_value_)
    print("SMO converged:", model.optimizer_.converged_)


if __name__ == "__main__":
    main()
