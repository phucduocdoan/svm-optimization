"""Các tiện ích đánh giá mô hình phân loại nhị phân."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)


def evaluate_binary(y_true, y_pred) -> dict:
    """Tổng hợp metrics; positive class là +1."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[-1, 1]),
        "classification_report": classification_report(
            y_true, y_pred, labels=[-1, 1], target_names=["class -1", "class +1"], zero_division=0
        ),
    }


def print_metrics(metrics: dict, title: str = "Evaluation") -> None:
    print(f"\n=== {title} ===")
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"{key:>10}: {metrics[key]:.4f}")
    print("Confusion matrix (rows=true [-1,+1], columns=pred [-1,+1]):")
    print(metrics["confusion_matrix"])
    print(metrics["classification_report"])
