"""Chặng 3: tự cài SMO và đánh giá trên Banknote Authentication."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from svm_optimization.data import load_banknote
from svm_optimization.evaluation import evaluate_binary, print_metrics
from svm_optimization.math.dual import check_dual_feasibility, dual_objective
from svm_optimization.math.kkt import kkt_report
from svm_optimization.math.primal import soft_margin_objective
from svm_optimization.models.kernel import linear_kernel
from svm_optimization.models.svm import KernelSVM
from svm_optimization.visualization import plot_objective_history


def main():
    X, y = load_banknote()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = KernelSVM(C=1.0, kernel="linear", tol=1e-3, max_passes=10,
                      max_iter=100_000, random_state=42)
    model.fit(X_train_scaled, y_train.to_numpy())
    predictions = model.predict(X_test_scaled)
    print_metrics(evaluate_binary(y_test, predictions), "Custom SMO - Linear Kernel")
    print("Support vectors per class:", model.n_support_)
    print("Iterations:", model.optimizer_.n_iter_)
    print("Pair updates:", model.optimizer_.n_updates_)
    print("Converged by no-change passes:", model.optimizer_.converged_)
    ytr = y_train.to_numpy()
    dual_value = dual_objective(model.alpha_, ytr, linear_kernel(X_train_scaled, X_train_scaled))
    print("Dual objective:", dual_value)
    print("Dual feasibility:", check_dual_feasibility(model.alpha_, ytr, C=1.0))
    if model.w_ is not None:
        train_scores = model.decision_function(X_train_scaled)
        primal_value = soft_margin_objective(model.w_, ytr, train_scores, C=1.0)
        print("Linear primal w:", model.w_)
        print("Margin width:", 2.0/np.linalg.norm(model.w_))
        print("Primal objective:", primal_value)
        print("Primal-dual gap:", abs(primal_value-dual_value))
        print("KKT report:", kkt_report(X_train_scaled, ytr, model.w_, model.intercept_,
              np.maximum(0.0, 1.0-ytr*train_scores), model.alpha_, C=1.0))
    plot_objective_history(model.optimizer_.objective_history_,
                           "reports/figures/custom_smo_objective.png")
    print("Saved: reports/figures/custom_smo_objective.png")


if __name__ == "__main__":
    main()
