"""Chặng 1: chạy SVM thư viện trên Banknote Authentication.

Run from project root: python -m experiments.01_baseline_sklearn
"""
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from svm_optimization.data import load_banknote
from svm_optimization.evaluation import evaluate_binary, print_metrics
from svm_optimization.visualization import plot_confusion_matrix


def main():
    X, y = load_banknote()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Train: {len(X_train)}, test: {len(X_test)}")
    print_metrics(evaluate_binary(y_test, predictions), "scikit-learn Linear SVC")
    svc = model.named_steps["svc"]
    print("Support vectors per class:", svc.n_support_)
    plot_confusion_matrix(y_test, predictions, "reports/figures/baseline_confusion_matrix.png",
                          title="Baseline SVC - confusion matrix")
    print("Saved: reports/figures/baseline_confusion_matrix.png")


if __name__ == "__main__":
    main()
