"""Ví dụ nhỏ: Gram matrix và từng phần của score RBF."""
import numpy as np

from svm_optimization.models.kernel import linear_kernel, rbf_kernel
from svm_optimization.models.svm import KernelSVM


def main():
    X = np.array([[-1.0], [1.0]])
    y = np.array([-1, 1])
    x_new = np.array([[0.5]])
    gamma = 0.5

    model = KernelSVM(C=1, kernel="rbf", gamma=gamma, tol=1e-6).fit(X, y)
    similarities = rbf_kernel(X, x_new, gamma).ravel()
    contributions = model.alpha_ * y * similarities
    score = contributions.sum() + model.intercept_

    print("Mẫu train:", X.ravel(), "Nhãn:", y)
    print("Gram Linear:\n", linear_kernel(X, X))
    print("Gram RBF:\n", np.round(rbf_kernel(X, X, gamma), 4))
    print("K(x_i, x mới):", np.round(similarities, 4))
    print("alpha_i * y_i:", np.round(model.alpha_ * y, 4))
    print("Từng đóng góp:", np.round(contributions, 4))
    print(f"Bias: {model.intercept_:.4f}; score tự tính: {score:.4f}")
    print(f"Score từ model: {model.decision_function(x_new)[0]:.4f}")
    print("Nhãn dự đoán:", model.predict(x_new)[0])


if __name__ == "__main__":
    main()
