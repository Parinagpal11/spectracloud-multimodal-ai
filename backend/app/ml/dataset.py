# backend/app/ml/dataset.py

from __future__ import annotations
import os
import numpy as np

from app.pipeline.raman_preprocess_array import preprocess_spectrum_array
from app.ml.features import extract_raman_features


DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "raman-dataset",
    "data",
)
DATA_DIR = os.path.abspath(DATA_DIR)


def _load_npy(name: str) -> np.ndarray:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return np.load(path, allow_pickle=True)


def load_split_arrays(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    split: reference | finetune | test | 2018clinical | 2019clinical
    Returns: (wavenumbers, X, y)
    """
    w = _load_npy("wavenumbers.npy")
    X = _load_npy(f"X_{split}.npy")
    y = _load_npy(f"y_{split}.npy")

    w = np.asarray(w, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)

    # y can be int/float/str; keep as-is
    y = np.asarray(y).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"X_{split}.npy must be 2D, got shape {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y sample mismatch for split={split}: {X.shape[0]} vs {y.shape[0]}")
    if X.shape[1] != w.shape[0]:
        raise ValueError(f"X columns must match wavenumbers: {X.shape[1]} vs {w.shape[0]}")

    return w, X, y


def load_split_features(
    split: str = "reference",
    *,
    max_samples: int | None = None,
    preprocess_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts spectra -> preprocessing -> feature extraction.
    Returns:
      X_feat: (N, D)
      y:      (N,)
    """
    preprocess_kwargs = preprocess_kwargs or {}
    w, X, y = load_split_arrays(split)

    N = X.shape[0]
    if max_samples is not None:
        N = min(N, int(max_samples))

    feats = []
    ys = []

    for i in range(N):
        spec = X[i, :]
        processed = preprocess_spectrum_array(w, spec, **preprocess_kwargs)

        f = extract_raman_features(
            processed["x"],
            processed["y_processed"]
        )

        f = np.asarray(f, dtype=float).reshape(-1)
        feats.append(f)
        ys.append(y[i])

    X_feat = np.vstack(feats) if len(feats) else np.empty((0, 0), dtype=float)
    y_out = np.asarray(ys)

    return X_feat, y_out
