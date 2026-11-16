"""Chặng 4: khảo sát ảnh hưởng của C và kernel.

Mặc định dùng SVC để tạo bảng baseline đáng tin cậy. Đặt USE_CUSTOM=True
sau khi custom SMO đã chạy ổn để khảo sát bản tự cài.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from svm_optimization.data import load_banknote
from svm_optimization.evaluation import evaluate_binary
from svm_optimization.models.svm import KernelSVM

USE_CUSTOM = False


def main():
    X, y = load_banknote()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rows = []
    for kernel in ("linear", "rbf"):
        for C in (0.1, 1.0, 10.0):
            if USE_CUSTOM:
                scaler = StandardScaler()
                Xtr, Xte = scaler.fit_transform(X_train), scaler.transform(X_test)
                model = KernelSVM(C=C, kernel=kernel, gamma="scale", max_passes=10)
                model.fit(Xtr, y_train.to_numpy())
                pred, n_support = model.predict(Xte), len(model.support_indices_)
            else:
                model = make_pipeline(StandardScaler(), SVC(C=C, kernel=kernel, gamma="scale"))
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                n_support = int(model.named_steps["svc"].support_.size)
            metrics = evaluate_binary(y_test, pred)
            rows.append({"implementation": "custom SMO" if USE_CUSTOM else "sklearn SVC",
                         "kernel": kernel, "C": C, "accuracy": metrics["accuracy"],
                         "precision": metrics["precision"], "recall": metrics["recall"],
                         "f1": metrics["f1"], "n_support_vectors": n_support})
    result = pd.DataFrame(rows)
    print(result.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    result.to_csv("reports/data/C_kernel_comparison.csv", index=False)
    print("Saved: reports/data/C_kernel_comparison.csv")


if __name__ == "__main__":
    main()
