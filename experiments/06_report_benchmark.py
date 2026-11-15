"""Benchmark phục vụ báo cáo theo template của cô Xuân.

Các nguyên tắc của benchmark:

* chỉ chia train/test một lần, có ``stratify`` và ``random_state`` cố định;
* chuẩn hóa trong pipeline hoặc fit riêng trên train, tránh rò rỉ test;
* chọn C/gamma bằng 5-fold CV trên train, chỉ đánh giá test sau khi chọn;
* ghi cả metrics phân loại, thời gian, support vectors và artifact biểu đồ.

Chạy từ thư mục gốc:

    python -m experiments.06_report_benchmark
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from svm_optimization.data import load_banknote
from svm_optimization.evaluation import evaluate_binary
from svm_optimization.math.primal import soft_margin_objective
from svm_optimization.models.svm import KernelSVM


RANDOM_STATE = 42
TEST_SIZE = 0.2
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DATA_OUT = REPORTS / "data"


def _savefig(fig: plt.Figure, name: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(ROOT))


def _evaluate(name, model, X_train, y_train, X_test, y_test, support_vectors=None):
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    predictions = model.predict(X_test)
    metrics = evaluate_binary(y_test, predictions)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    else:
        scores = None
    row = {
        "model": name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": float(roc_auc_score(y_test, scores)) if scores is not None else np.nan,
        "fit_seconds": fit_seconds,
        "n_support_vectors": support_vectors if support_vectors is not None else np.nan,
    }
    return row, predictions, scores


def _plot_boundaries(X, y, models, names, output_name):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[:, :2])
    x0 = np.linspace(X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5, 220)
    x1 = np.linspace(X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5, 220)
    xx, yy = np.meshgrid(x0, x1)
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.8), squeeze=False)
    for ax, model, name in zip(axes[0], models, names):
        model.fit(X_scaled, y)
        score = model.decision_function(grid).reshape(xx.shape)
        ax.contourf(xx, yy, score, levels=[-1e9, 0, 1e9], alpha=0.14, cmap="coolwarm")
        ax.contour(xx, yy, score, levels=[-1, 0, 1], colors=["#356AE6", "#222", "#C77822"],
                   linestyles=["--", "-", "--"], linewidths=1.0)
        ax.scatter(X_scaled[y == -1, 0], X_scaled[y == -1, 1], s=12, label="class -1")
        ax.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], s=12, marker="^", label="class +1")
        ax.set_title(name)
        ax.set_xlabel("standardized feature 1")
        ax.set_ylabel("standardized feature 2")
        ax.grid(alpha=0.2)
    axes[0][0].legend(loc="best", fontsize=8)
    fig.suptitle("Decision boundary trên hai đặc trưng đầu (minh họa)")
    fig.tight_layout()
    return _savefig(fig, output_name)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    X_frame, y_series = load_banknote()
    X = X_frame.to_numpy(dtype=float)
    y = y_series.to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    dataset_summary = {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "feature_names": list(X_frame.columns),
        "class_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "positive_class": 1,
    }
    (DATA_OUT / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    models = [
        ("Logistic Regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))),
        ("Linear SVC", make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))),
        ("Decision Tree", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
        ("Random Forest", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
    ]
    rows, predictions_by_model, scores_by_model = [], {}, {}
    for name, model in models:
        row, predictions, scores = _evaluate(name, model, X_train, y_train, X_test, y_test)
        if name == "Linear SVC":
            row["n_support_vectors"] = int(model.named_steps["svc"].support_.size)
        rows.append(row)
        predictions_by_model[name] = predictions
        scores_by_model[name] = scores

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tuned_rbf = GridSearchCV(
        make_pipeline(StandardScaler(), SVC(kernel="rbf")),
        param_grid={"svc__C": [0.1, 1.0, 10.0, 100.0], "svc__gamma": [0.001, 0.01, 0.1, 1.0]},
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    rbf_row, rbf_pred, rbf_scores = _evaluate("Tuned RBF SVC", tuned_rbf, X_train, y_train, X_test, y_test)
    rbf_row["n_support_vectors"] = int(tuned_rbf.best_estimator_.named_steps["svc"].support_.size)
    rbf_row["best_params"] = json.dumps(tuned_rbf.best_params_, ensure_ascii=False, sort_keys=True)
    rows.append(rbf_row)
    predictions_by_model["Tuned RBF SVC"] = rbf_pred
    scores_by_model["Tuned RBF SVC"] = rbf_scores

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    custom = KernelSVM(C=1.0, kernel="linear", tol=1e-3, max_passes=10,
                       max_iter=100_000, random_state=RANDOM_STATE)
    custom_row, custom_pred, custom_scores = _evaluate(
        "Custom SMO Linear", custom, X_train_scaled, y_train, X_test_scaled, y_test,
        support_vectors=None,
    )
    custom_row["n_support_vectors"] = int(len(custom.support_indices_))
    custom_row["n_iter"] = int(custom.optimizer_.n_iter_)
    custom_row["n_pair_updates"] = int(custom.optimizer_.n_updates_)
    custom_row["converged"] = bool(custom.optimizer_.converged_)
    train_scores = custom.decision_function(X_train_scaled)
    primal_value = soft_margin_objective(custom.w_, y_train, train_scores, C=1.0)
    custom_row["primal_dual_gap"] = float(abs(primal_value - custom.optimizer_.objective_value_))
    rows.append(custom_row)
    predictions_by_model["Custom SMO Linear"] = custom_pred
    scores_by_model["Custom SMO Linear"] = custom_scores

    results = pd.DataFrame(rows)
    results.to_csv(DATA_OUT / "model_metrics.csv", index=False, encoding="utf-8-sig")
    (DATA_OUT / "best_rbf_params.json").write_text(
        json.dumps(tuned_rbf.best_params_, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cv_results = pd.DataFrame(tuned_rbf.cv_results_)
    cv_results[["param_svc__C", "param_svc__gamma", "mean_test_score", "std_test_score", "rank_test_score"]].to_csv(
        DATA_OUT / "rbf_grid_search.csv", index=False, encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = results.set_index("model")[["accuracy", "precision", "recall", "f1"]]
    plot_df.plot.bar(ax=ax, color=["#356AE6", "#138A83", "#7757C7", "#C77822"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("So sánh metrics trên tập test")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    metrics_figure = _savefig(fig, "benchmark_metrics.png")

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (name, predictions) in zip(axes.flat, predictions_by_model.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions, labels=[-1, 1], display_labels=["-1", "+1"],
            colorbar=False, ax=ax, cmap="Blues"
        )
        ax.set_title(name, fontsize=9)
    for ax in axes.flat[len(predictions_by_model):]:
        ax.axis("off")
    fig.suptitle("Ma trận nhầm lẫn của các mô hình")
    fig.tight_layout()
    confusion_figure = _savefig(fig, "benchmark_confusion_matrices.png")

    heat = cv_results.pivot(index="param_svc__gamma", columns="param_svc__C", values="mean_test_score")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(heat.to_numpy(), cmap="YlGnBu", vmin=heat.to_numpy().min(), vmax=heat.to_numpy().max())
    ax.set_xticks(range(len(heat.columns)), [str(x) for x in heat.columns])
    ax.set_yticks(range(len(heat.index)), [str(x) for x in heat.index])
    ax.set_xlabel("C")
    ax.set_ylabel("gamma")
    ax.set_title("5-fold CV: F1 của RBF SVC")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.iloc[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean F1")
    fig.tight_layout()
    grid_figure = _savefig(fig, "rbf_grid_search_heatmap.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(custom.optimizer_.objective_history_, color="#7757C7", marker=".", markersize=2)
    ax.set_xlabel("pass")
    ax.set_ylabel("dual objective")
    ax.set_title("Hội tụ SMO Custom Linear")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    objective_figure = _savefig(fig, "custom_smo_convergence.png")

    boundary_figure = _plot_boundaries(
        X, y,
        [make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)),
         make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
         tuned_rbf.best_estimator_],
        ["Linear SVC", "RBF SVC mặc định", "RBF SVC tuned"],
        "decision_boundaries_2d.png",
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in ("Logistic Regression", "Linear SVC", "Tuned RBF SVC", "Custom SMO Linear"):
        scores = scores_by_model[name]
        RocCurveDisplay.from_predictions(y_test, scores, pos_label=1, name=name, ax=ax)
    ax.set_title("ROC trên tập test")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    roc_figure = _savefig(fig, "benchmark_roc.png")

    artifacts = {
        "dataset_summary": "reports/data/dataset_summary.json",
        "model_metrics": "reports/data/model_metrics.csv",
        "rbf_grid_search": "reports/data/rbf_grid_search.csv",
        "figures": {
            "metrics": metrics_figure,
            "confusion_matrices": confusion_figure,
            "grid_search": grid_figure,
            "custom_convergence": objective_figure,
            "decision_boundaries": boundary_figure,
            "roc": roc_figure,
        },
        "best_rbf_params": tuned_rbf.best_params_,
        "custom_smo": {
            "iterations": custom.optimizer_.n_iter_,
            "pair_updates": custom.optimizer_.n_updates_,
            "converged": custom.optimizer_.converged_,
        },
    }
    (REPORTS / "benchmark_artifacts.json").write_text(
        json.dumps(artifacts, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Best RBF params:", tuned_rbf.best_params_)
    print("Saved benchmark artifacts to reports/")


if __name__ == "__main__":
    main()
