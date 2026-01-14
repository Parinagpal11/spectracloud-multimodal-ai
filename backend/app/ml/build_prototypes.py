# backend/app/ml/build_prototypes.py

from __future__ import annotations
import os
import json
import numpy as np

from app.ml.dataset import load_split_features


OUT_PATH = os.path.join(os.path.dirname(__file__), "prototypes.json")


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / (n + eps)


def build_prototypes(
    split: str = "reference",
    *,
    max_samples: int | None = None,
) -> dict:
    X_feat, y = load_split_features(split=split, max_samples=max_samples)

    if X_feat.size == 0:
        raise RuntimeError("No features were produced. Check dataset + feature extractor.")

    labels = np.unique(y)
    protos = {}

    for lab in labels:
        idx = np.where(y == lab)[0]
        class_mean = np.mean(X_feat[idx], axis=0)
        protos[str(lab)] = _l2_normalize(class_mean).tolist()

    payload = {
        "split": split,
        "feature_dim": int(X_feat.shape[1]),
        "num_samples": int(X_feat.shape[0]),
        "labels": [str(x) for x in labels.tolist()],
        "prototypes": protos,
    }
    return payload


def main():
    payload = build_prototypes(split="reference")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"✅ Saved prototypes to: {OUT_PATH}")
    print(f"labels={payload['labels']}, feature_dim={payload['feature_dim']}, N={payload['num_samples']}")


if __name__ == "__main__":
    main()
