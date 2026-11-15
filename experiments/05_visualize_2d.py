"""Vẽ boundary 2D trên hai feature đầu của Banknote.

Chỉ phục vụ trực quan: mô hình ở script này train trên 2 features, không phải
mô hình đủ 4 features dùng cho thực nghiệm phân loại chính.
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from svm_optimization.data import load_banknote
from svm_optimization.models.svm import KernelSVM
from svm_optimization.visualization import plot_2d_boundary


def main():
    X, y = load_banknote()
    X2 = X.iloc[:, :2]
    X_train, X_test, y_train, y_test = train_test_split(
        X2, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = KernelSVM(C=1.0, kernel="linear", max_passes=10, random_state=42)
    model.fit(X_train_scaled, y_train.to_numpy())
    pred = model.predict(X_test_scaled)
    print(f"2-feature visualization accuracy: {(pred == y_test.to_numpy()).mean():.4f}")
    print("Support vectors:", len(model.support_indices_))
    plot_2d_boundary(model, X_train_scaled, y_train.to_numpy(),
        feature_names=[f"standardized {name}" for name in X_train.columns],
        output_path="reports/figures/banknote_2d_boundary.png",
        title="Banknote: Linear SVM boundary on first two features")
    print("Saved: reports/figures/banknote_2d_boundary.png")


if __name__ == "__main__":
    main()
