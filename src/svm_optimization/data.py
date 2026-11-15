"""Tải và chuẩn bị Banknote Authentication từ UCI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

UCI_BANKNOTE_ID = 267


def load_banknote(cache_dir: str | Path = "data/banknote", refresh: bool = False):
    """Trả về (X, y) của Banknote Authentication.

    Dataset được tải lần đầu bằng package ucimlrepo và cache thành CSV để
    những lần chạy sau không cần tải lại. Nhãn được ánh xạ thành -1/+1.
    File dữ liệu không được đóng gói sẵn trong project.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "banknote_authentication.csv"

    if cache_file.exists() and not refresh:
        frame = pd.read_csv(cache_file)
        if "target" not in frame.columns:
            raise ValueError(f"Cache {cache_file} không có cột target.")
        X = frame.drop(columns=["target"])
        y_raw = frame["target"]
    else:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise ImportError("Thiếu ucimlrepo. Chạy: pip install ucimlrepo") from exc
        dataset = fetch_ucirepo(id=UCI_BANKNOTE_ID)
        X = dataset.data.features.copy()
        target = dataset.data.targets
        if target is None or target.shape[1] != 1:
            raise ValueError("Không tìm thấy đúng một cột target trong dữ liệu UCI.")
        y_raw = target.iloc[:, 0].copy()
        frame = X.copy()
        frame["target"] = y_raw.to_numpy()
        frame.to_csv(cache_file, index=False)

    if X.isna().any().any() or pd.isna(y_raw).any():
        raise ValueError("Dataset có missing values; cần xử lý trước khi train.")
    classes = sorted(pd.unique(y_raw).tolist())
    if len(classes) != 2:
        raise ValueError(f"Cần đúng 2 lớp, tìm thấy: {classes}")
    mapping = {classes[0]: -1, classes[1]: 1}
    y = y_raw.map(mapping).astype(int)
    return X.astype(float), y
