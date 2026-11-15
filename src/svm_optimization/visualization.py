"""Biểu đồ hỗ trợ hiểu SVM và phân tích kết quả."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_2d_boundary(model, X: np.ndarray, y: np.ndarray, feature_names=None,
                     output_path: str | Path | None = None,
                     title: str = "SVM decision boundary", resolution: int = 250):
    """Vẽ decision boundary khi model được train trên đúng 2 features."""
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("plot_2d_boundary yêu cầu X có đúng 2 features.")
    padding = 0.08*(X.max(axis=0)-X.min(axis=0)+1e-9)
    x_min, x_max = X[:,0].min()-padding[0], X[:,0].max()+padding[0]
    y_min, y_max = X[:,1].min()-padding[1], X[:,1].max()+padding[1]
    xx, yy = np.meshgrid(np.linspace(x_min,x_max,resolution), np.linspace(y_min,y_max,resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    scores = model.decision_function(grid).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.contourf(xx, yy, scores, levels=[-1e9,0,1e9], alpha=0.12)
    ax.contour(xx, yy, scores, levels=[-1,0,1], linestyles=["--","-","--"])
    ax.scatter(X[y==-1,0], X[y==-1,1], label="Class -1", marker="o", edgecolor="black")
    ax.scatter(X[y==1,0], X[y==1,1], label="Class +1", marker="^", edgecolor="black")
    if hasattr(model, "support_vectors_"):
        sv = model.support_vectors_
        ax.scatter(sv[:,0], sv[:,1], s=150, facecolors="none", edgecolors="black",
                   linewidths=1.5, label="Support vectors")
    if feature_names is not None and len(feature_names)==2:
        ax.set_xlabel(str(feature_names[0])); ax.set_ylabel(str(feature_names[1]))
    else:
        ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.2); fig.tight_layout()
    if output_path is not None:
        path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig, ax


def plot_confusion_matrix(y_true, y_pred, output_path=None, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[-1,1],
        display_labels=["-1","+1"], ax=ax, colorbar=False)
    ax.set_title(title); fig.tight_layout()
    if output_path is not None:
        path=Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig, ax


def plot_objective_history(history, output_path=None):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(np.arange(len(history)), history, marker="o", markersize=3)
    ax.set_xlabel("Pass"); ax.set_ylabel("Dual objective W(alpha)")
    ax.set_title("SMO optimization progress (objective should not decrease materially)")
    ax.grid(alpha=0.25); fig.tight_layout()
    if output_path is not None:
        path=Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig, ax
